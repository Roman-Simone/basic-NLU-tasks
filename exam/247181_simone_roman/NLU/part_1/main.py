# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from utils import *
from model import *
import os
import numpy as np

from collections import Counter
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from pprint import pprint


if __name__ == "__main__":

    # Load the data
    # exampleod element -> {'utterance': 'what is the cost for these flights from baltimore to philadelphia', 
    #                       'slots': 'O O O O O O O O B-fromloc.city_name O B-toloc.city_name', 
    #                       'intent': 'airfare'}
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tmp_train_raw = load_data(os.path.join(current_dir, "dataset/ATIS/train.json"))  
    test_raw = load_data(os.path.join(current_dir, "dataset/ATIS/test.json"))

    # Create the dev set
    train_raw, dev_raw = create_dev(tmp_train_raw)

    # All the words in the train
    words = sum([x['utterance'].split() for x in train_raw], []) # No set() since we want to compute All dataset
    corpus = train_raw + dev_raw + test_raw # We do not wat unk labels, 
    slots = set(sum([line['slots'].split() for line in corpus],[]))
    intents = set([line['intent'] for line in corpus])

    # Create an id for each word, intent and slot ex -> {0: 'pad', 1: 'unk', 2: 'what', 3: 'is', 4: 'the', 5: 'cost', 6: 'for'...}
    lang = Lang(words, intents, slots, cutoff=0)

    # Create our datasets
    #Create the tensoer ecc ex->{'utterance': tensor([ 2.,  3.,  4., 35., 22.,  9., 36., 11., 37., 38., 24., 39.]), 
    #                            'slots': tensor([ 20.,  20.,  20.,  86.,  20.,  20.,  87.,  20., 118.,  37.,  20., 127.]), 
    #                            'intent': 17}
    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)

    # Dataloader instantiations
    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn,  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)    


    hid_size = 200
    emb_size = 300
    lr = 0.0001 # learning rate
    clip = 5 # Clip the gradient


    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)

    model = ModelIAS(hid_size, out_slot, out_int, emb_size, vocab_len, pad_index=PAD_TOKEN, bidirectional=True).to(device)
    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss() # Because we do not have the pad token


    n_epochs = 200
    runs = 5

    slot_f1s, intent_acc = [], []
    for x in tqdm(range(0, runs)):
        model = ModelIAS(hid_size, out_slot, out_int, emb_size, 
                        vocab_len, pad_index=PAD_TOKEN, bidirectional=True).to(device)
        model.apply(init_weights)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
        criterion_intents = nn.CrossEntropyLoss()
        

        patience = 3
        losses_train = []
        losses_dev = []
        sampled_epochs = []
        best_f1 = 0
        for x in range(1,n_epochs):
            loss = train_loop(train_loader, optimizer, criterion_slots, 
                            criterion_intents, model)
            if x % 5 == 0:
                sampled_epochs.append(x)
                losses_train.append(np.asarray(loss).mean())
                results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, 
                                                            criterion_intents, model, lang)
                losses_dev.append(np.asarray(loss_dev).mean())
                f1 = results_dev['total']['f']

                if f1 > best_f1:
                    best_f1 = f1
                else:
                    patience -= 1
                if patience <= 0: # Early stopping with patient
                    break # Not nice but it keeps the code clean

        results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, 
                                                criterion_intents, model, lang)
        intent_acc.append(intent_test['accuracy'])
        slot_f1s.append(results_test['total']['f'])


    slot_f1s = np.asarray(slot_f1s)
    intent_acc = np.asarray(intent_acc)
    print('Slot F1', round(slot_f1s.mean(),3), '+-', round(slot_f1s.std(),3))
    print('Intent Acc', round(intent_acc.mean(), 3), '+-', round(slot_f1s.std(), 3))
