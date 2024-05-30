# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from utils import *
from model import *

from functools import partial
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import copy
import math
import os


def average_weights(weights_list):
    avg_weights = copy.deepcopy(weights_list[0])
    for key in avg_weights.keys():
        for weights in weights_list[1:]:
            avg_weights[key] += weights[key]
        avg_weights[key] /= len(weights_list)
    return avg_weights


if __name__ == "__main__":
    
    #!PARAMETERS

    batch_size_train = 32
    batch_size_dev = 128
    batch_size_test = 128
    
    hid_size = 500
    emb_size = 500

    lr = 2.5 # This is definitely not good for SGD
    clip = 5 # Clip the gradient
    
    #*###############################################################################################

    #!LOADING DATA
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_raw = read_file(os.path.join(current_dir, "dataset/PennTreeBank/ptb.train.txt"))
    dev_raw = read_file(os.path.join(current_dir, "dataset/PennTreeBank/ptb.valid.txt"))
    test_raw = read_file(os.path.join(current_dir, "dataset/PennTreeBank/ptb.test.txt"))

    vocab = get_vocab(train_raw, ["<pad>", "<eos>"])
    lang = Lang(train_raw, ["<pad>", "<eos>"])

    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size_dev, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))

    vocab_len = len(lang.word2id)

    #*###############################################################################################

    #!TRAINING
    model = LM_LSTM_DROP(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
    model.apply(init_weights)

    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

    
    n_epochs = 100
    patience = 3
    losses_train = []
    losses_dev = []
    ppl_dev_list = []
    ppl_train_list = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    pbar = tqdm(range(1,n_epochs))
    final_epoch = 0
    switch_ASGD = False
    window = 3
    list_weights = []
   

    #If the PPL is too high try to change the learning rate
    for epoch in pbar:
        final_epoch = epoch
        ppl_train, loss_train = train_loop(train_loader, optimizer, criterion_train, model, clip)
        ppl_train_list.append(ppl_train)
        sampled_epochs.append(epoch)
        losses_train.append(np.asarray(loss_train).mean())


        #if switched in ASGD
        if switch_ASGD:

            # Save model weights
            weights = {}
            for prm in model.parameters():
                weights[prm] = prm.data.clone()
              

            if len(list_weights) < window:
                list_weights.append(weights)
            else:
                list_weights.pop(0)
                list_weights.append(weights)
            
            #media dei pesi

            avg_weights = {}

            if len(list_weights)==1:
                for prm in model.parameters():
                    avg_weights[prm] = prm.data.clone()
            else:
                for elem in list_weights:
                    for key in elem.keys():
                        if key not in avg_weights:
                            avg_weights[key] = elem[key].clone()
                        else:
                            avg_weights[key] += elem[key]
                for key in avg_weights.keys():
                    avg_weights[key] /= len(list_weights)

            
             # Load average weights into the model
            for prm in model.parameters():
                prm.data = avg_weights[prm].clone()

            optimizer = optim.SGD(model.parameters(), lr=lr)

            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            ppl_dev_list.append(ppl_dev)
            losses_dev.append(np.asarray(loss_dev).mean())
            pbar.set_description(f"lr= {lr} ASGD= {switch_ASGD} PPL: {ppl_dev}")       

            # Load model weights
            for prm in model.parameters():
                prm.data = weights[prm].clone()

        else:


            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            ppl_dev_list.append(ppl_dev)
            losses_dev.append(np.asarray(loss_dev).mean())
            pbar.set_description(f"lr= {lr} ASGD= {switch_ASGD} PPL: {ppl_dev}")

            if (len(ppl_dev_list) > window and ppl_dev > min(ppl_dev_list[:-window])):
                # lr *= 4/10
                # optimizer.param_groups[0]['lr'] = lr
                optimizer = optim.SGD(best_weights, lr=lr)
                print(epoch)
                switch_ASGD = True
            


        best_weights = model.parameters()

        if  ppl_dev < best_ppl: # the lower, the better
            best_ppl = ppl_dev
            best_model = copy.deepcopy(model).to('cpu')
            best_weights = model.parameters()
            patience = 5
        elif ppl_dev > best_ppl and switch_ASGD==False:
            lr *= 4/10
            optimizer.param_groups[0]['lr'] = lr
        elif ppl_dev > best_ppl and switch_ASGD==True:
            lr *= 4/10
            optimizer.param_groups[0]['lr'] = lr
            patience -= 1

        if patience <= 0: # Early stopping with patience
            break # Not nice but it keeps the code clean


    best_model.to(DEVICE)
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)
    print('Test ppl: ', final_ppl)

    name_exercise = "PART_22"
    save_result(name_exercise, sampled_epochs, losses_train, losses_dev,ppl_train_list, ppl_dev_list, hid_size, 
                emb_size, lr, clip, vocab_len, final_epoch,best_ppl, final_ppl, batch_size_train, batch_size_dev, batch_size_test, optimizer, model, best_model)

