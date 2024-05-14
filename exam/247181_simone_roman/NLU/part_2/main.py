from functions import *
from utils import *
from model import *

import os
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel






if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tmp_train_raw = load_data(os.path.join(current_dir, "dataset/ATIS/train.json"))
    test_raw = load_data(os.path.join(current_dir, "dataset/ATIS/test.json"))

    #create the dev set 
    train_raw, dev_raw = create_dev(tmp_train_raw)


    # All the words in the train
    words = sum([x['utterance'].split() for x in train_raw], []) # No set() since we want to compute All dataset
    corpus = train_raw + dev_raw + test_raw # We do not wat unk labels, 
    slots = set(sum([line['slots'].split() for line in corpus],[]))
    intents = set([line['intent'] for line in corpus])

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    lang = Lang(words, intents, slots, cutoff=0, tokenizer=tokenizer)



    train_dataset = IntentsAndSlots(train_raw, lang, tokenizer)
    dev_dataset = IntentsAndSlots(dev_raw, lang, tokenizer)
    test_dataset = IntentsAndSlots(test_raw, lang, tokenizer)

    

    # train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn,  shuffle=True)
    # dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
    # test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)



    # hid_size = 200
    # emb_size = 300
    # lr = 0.0001 # learning rate
    # clip = 5 # Clip the gradient


    # out_slot = len(lang.slot2id)
    # out_int = len(lang.intent2id)


    # model = ModelBert(hid_size, out_slot, out_int).to(device)
    
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    # criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    # criterion_intents = nn.CrossEntropyLoss() # Because we do not have the pad token

    # n_epochs = 2
    # runs = 1

    # slot_f1s, intent_acc = [], []

    # for x in tqdm(range(0, runs)):
    #     model = ModelBert(hid_size, out_slot, out_int).to(device)
    #     optimizer = optim.Adam(model.parameters(), lr=lr)
    #     criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    #     criterion_intents = nn.CrossEntropyLoss() # Because we do not have the pad token

    #     patience = 3
    #     losses_train = []
    #     losses_dev = []
    #     sampled_ephocs = []
    #     best_f1 = 0

    #     for epoch in range(1, n_epochs):
    #         loss = train_loop(train_loader, optimizer, criterion_slots, 
    #                         criterion_intents, model)

