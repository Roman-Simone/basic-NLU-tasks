from functions import *
from utils import *
from model import *

import copy
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer

if __name__ == "__main__":
    #Parameters
    config = {
        "lr": 5e-5,
        "batch_train_size": 32,
        "batch_dev_size": 64,
        "batch_test_size": 64,
        "hid_size": 768,
        "n_epochs": 100,
        "clip": 5,
    }
   

    #load data
    print("TAKE DATASET")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tmp_train_raw = load_data(os.path.join(current_dir, "dataset/ATIS/train.json"))
    test_raw = load_data(os.path.join(current_dir, "dataset/ATIS/test.json"))

    #create the dev set 
    print("CREATE DEV SET")
    train_raw, dev_raw = create_dev(tmp_train_raw)
    

    # All the words in the train
    corpus = train_raw + dev_raw + test_raw # We do not wat unk labels, 
    slots = set(sum([line['slots'].split() for line in corpus],[]))
    intents = set([line['intent'] for line in corpus])

    print("CREATE LANG")
    # Create an id for each word, intent and slot for word use bert tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    lang = Lang(intents, slots, cutoff=0)
    
    train_dataset = IntentsAndSlots(train_raw, lang, tokenizer)
    dev_dataset = IntentsAndSlots(dev_raw, lang, tokenizer)
    test_dataset = IntentsAndSlots(test_raw, lang, tokenizer)

    print("CREATE DATALOADERS")
    train_loader = DataLoader(train_dataset, batch_size=config["batch_train_size"], collate_fn=collate_fn,  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config["batch_dev_size"], collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_test_size"], collate_fn=collate_fn)

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)

    model = ModelBert(config["hid_size"], out_slot, out_int).to(device)
    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss() # Because we do not have the pad token

        
    patience = 3
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_score = 0
    for x in tqdm(range(1, config["n_epochs"])):
        loss = train_loop(train_loader, optimizer, criterion_slots, 
                        criterion_intents, model, clip=config["clip"])
        if x % 3 == 0: # We check the performance every n epochs
            sampled_epochs.append(x)
            losses_train.append(np.asarray(loss).mean())

            results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang, tokenizer)
            losses_dev.append(np.asarray(loss_dev).mean())
            
            f1 = results_dev['total']['f']
            intent_acc = intent_res["accuracy"]
            actual_score = (f1 + intent_acc) / 2
            # For decreasing the patience you can also use the average between slot f1 and intent accuracy
            if actual_score > best_score:
                best_score = actual_score
                best_model = copy.deepcopy(model).to('cpu')
                patience = 3
            else:
                patience -= 1
            if patience <= 0: # Early stopping with patience
                break 
    results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang, tokenizer)    
    test_f1 = results_test['total']['f']
    test_acc = intent_test['accuracy']
    
    print('Slot F1: ', test_f1)
    print('Intent Accuracy:', test_acc)


    name_exercise = "PART_2"
    save_result(name_exercise, sampled_epochs, losses_train, losses_dev, optimizer, model, config, test_f1, test_acc, best_model)


