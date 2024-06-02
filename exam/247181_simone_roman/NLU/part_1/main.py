from utils import *
from model import *
from functions import *

import copy
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader


if __name__ == "__main__":

    # PARAMETERS
    config = {
        "lr": 5e-10,
        "batch_train_size": 32,
        "batch_dev_size": 64,
        "batch_test_size": 64,
        "hid_size": 300,
        "emb_size": 400,
        "n_epochs": 200,
        "runs": 1,
        "flag_bidirectional": False,
        "flag_dropout": False,
    }

    # Load the data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tmp_train_raw = load_data(os.path.join(current_dir, "dataset/ATIS/train.json"))  
    test_raw = load_data(os.path.join(current_dir, "dataset/ATIS/test.json"))

    # Create the dev set
    train_raw, dev_raw = create_dev(tmp_train_raw)

    # All the words in the train
    words = sum([x['utterance'].split() for x in train_raw], [])

    corpus = train_raw + dev_raw + test_raw
    slots = set(sum([line['slots'].split() for line in corpus],[]))
    intents = set([line['intent'] for line in corpus])

    # Create an id for each word, intent and slot
    lang = Lang(words, intents, slots, cutoff=0)

    # Create our datasets
    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)

    # Dataloader instantiations
    train_loader = DataLoader(train_dataset, batch_size=config["batch_test_size"], collate_fn=collate_fn,  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config["batch_dev_size"], collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_test_size"], collate_fn=collate_fn)    

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)

    model = ModelIAS(config["hid_size"], out_slot, out_int, config["emb_size"], 
                    vocab_len, pad_index=PAD_TOKEN, flag_bidirectional=config["flag_bidirectional"], flag_dropout=config["flag_dropout"]).to(device)
    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss() # Because we do not have the pad token

    slot_f1s, intents_acc = [], []
    for x in tqdm(range(0, config["runs"])):
        model = ModelIAS(config["hid_size"], out_slot, out_int, config["emb_size"], vocab_len, 
                         pad_index=PAD_TOKEN, flag_bidirectional=config["flag_bidirectional"], flag_dropout=config["flag_dropout"]).to(device)
    
        model.apply(init_weights)

        optimizer = optim.Adam(model.parameters(), lr=config["lr"])
        criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
        criterion_intents = nn.CrossEntropyLoss()
        
        patience = 3
        losses_train = []
        losses_dev = []
        sampled_epochs = []
        best_score = 0
        for x in range(1, config["n_epochs"]):
            loss = train_loop(train_loader, optimizer, criterion_slots, 
                            criterion_intents, model)
            if x % 5 == 0:
                sampled_epochs.append(x)
                losses_train.append(np.asarray(loss).mean())
                results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, 
                                                            criterion_intents, model, lang)
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

        results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, 
                                                criterion_intents, model, lang)
        test_f1 = results_test['total']['f']
        test_acc = intent_test['accuracy']
        intents_acc.append(intent_test['accuracy'])
        slot_f1s.append(results_test['total']['f'])

        name_exercise = "PART_11"
        save_result(name_exercise, sampled_epochs, losses_train, losses_dev, optimizer, model, config, test_f1, test_acc, best_model)


    slot_f1s = np.asarray(slot_f1s)
    intents_acc = np.asarray(intents_acc)
    print('Slot F1', round(slot_f1s.mean(),3), '+-', round(slot_f1s.std(),3))
    print('Intent Acc', round(intents_acc.mean(), 3), '+-', round(slot_f1s.std(), 3))

    
