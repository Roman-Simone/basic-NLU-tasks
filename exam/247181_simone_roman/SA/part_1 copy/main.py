from functions import *
from utils import *
from model import *


import copy
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from transformers import BertTokenizer
from torch.utils.data import DataLoader


if __name__ == "__main__":
    config={
        "lr": 0.0001,
        "batch_train_size": 128,
        "batch_dev_size": 128,
        "batch_test_size": 128,
        "clip": 5, # Clip the gradient
        "n_epochs": 100,
        "hid_size": 768,
    }


    print("TAKE DATASET")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tmp_train_raw = load_data(os.path.join(current_dir, "dataset/laptop14_train.txt"))
    test_raw = load_data(os.path.join(current_dir, "dataset/laptop14_test.txt"))

    #create the dev set
    train_raw, dev_raw = create_dev(tmp_train_raw)
    

    train_raw_split = split_data(train_raw)
    dev_raw_split = split_data(dev_raw)
    test_raw_split = split_data(test_raw)

    

    corpus = train_raw_split + dev_raw_split + test_raw_split # We do not wat unk labels, 

    slots = set(sum([line['slots'].split() for line in corpus],[]))
    

    
    print("CREATE LANG")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    lang = Lang(slots, cutoff=0)


    train_dataset = IntentsAndSlots(train_raw_split, lang, tokenizer)
    dev_dataset = IntentsAndSlots(dev_raw_split, lang, tokenizer)
    test_dataset = IntentsAndSlots(test_raw_split, lang, tokenizer)

   
    
    print("CREATE DATALOADERS")
    train_loader = DataLoader(train_dataset, batch_size=config["batch_train_size"], collate_fn=collate_fn,  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config["batch_dev_size"], collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_test_size"], collate_fn=collate_fn)

    

    patience= 3
    
    out_slot = len(lang.slot2id)

    model = ModelBert(config["hid_size"], out_slot).to(device)
    # model.apply(init_weights)
    
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss() # Because we do not have the pad token

    
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    results_dev = []
    best_f1 = 0

    pbar = tqdm(range(1, config["n_epochs"]))
    for x in pbar:
        loss = train_loop(train_loader, optimizer, criterion_slots, 
                        criterion_intents, model, clip=config["clip"])

        if x % 1 == 0: # We check the performance every 5 epochs
            sampled_epochs.append(x)
            losses_train.append(np.asarray(loss).mean())

            result_dev, loss_dev = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang, tokenizer)
            losses_dev.append(np.asarray(loss_dev).mean())
            results_dev.append(result_dev)
            precision = result_dev['Precision']
            recall = result_dev['Recall']
            f1 = result_dev['F1']

            # print(f"Epoch {x}: Precision {precision:.2f}, Recall {recall:.2f}, F1 {f1:.2f}")
            # For decreasing the patience you can also use the average between slot f1 and intent accuracy
            if f1 > best_f1:
                best_f1 = f1
                best_model = copy.deepcopy(model).to('cpu')
                patience = 3
            else:
                patience -= 1
            if patience <= 0: # Early stopping with patience
                break # Not nice but it keeps the code clean

        pbar.set_description(f"Loss: {np.asarray(loss).mean():.2f}")

    results_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang, tokenizer)    
    precision = results_test['Precision']
    recall = results_test['Recall']
    f1 = results_test['F1']
    
    print(f"Precision {precision:.2f}, Recall {recall:.2f}, F1 {f1:.2f}")

    name_exercise = "SA"
    save_result(name_exercise, sampled_epochs, losses_train, losses_dev, config, results_dev, results_test, best_model)


        
