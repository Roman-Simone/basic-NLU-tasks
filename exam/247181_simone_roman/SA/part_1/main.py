# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from utils import *
from model import *

import os
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from transformers import BertTokenizer
from torch.utils.data import DataLoader


if __name__ == "__main__":
    print("TAKE DATASET")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tmp_train_raw = load_data(os.path.join(current_dir, "dataset/twitter1_train.txt"))
    test_raw = load_data(os.path.join(current_dir, "dataset/twitter1_test.txt"))

    #create the dev set
    train_raw, dev_raw = create_dev(tmp_train_raw)
    

    train_raw_split = split_data(train_raw)
    dev_raw_split = split_data(dev_raw)
    test_raw_split = split_data(test_raw)
    

    corpus = train_raw_split + dev_raw_split + test_raw_split # We do not wat unk labels, 
    slots = set()
    for slot in corpus:
        for elem in slot['slots'].split():
            slots.add(elem)

    
    print("CREATE LANG")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    lang = Lang(slots, cutoff=0)


    train_dataset = IntentsAndSlots(train_raw_split, lang, tokenizer)
    dev_dataset = IntentsAndSlots(dev_raw_split, lang, tokenizer)
    test_dataset = IntentsAndSlots(test_raw_split, lang, tokenizer)
    
    print("CREATE DATALOADERS")
    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn,  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)




    hid_size = 768
    emb_size = 300
    lr = 0.0001 # learning rate
    clip = 5 # Clip the gradient

    out_slot = len(lang.slot2id)
   

    model = ModelBert(hid_size, out_slot).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss() # Because we do not have the pad token

        
    n_epochs = 200
    patience = 3
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_f1 = 0
    for x in tqdm(range(1,n_epochs)):
        loss = train_loop(train_loader, optimizer, criterion_slots, 
                        criterion_intents, model, clip=clip)
        
