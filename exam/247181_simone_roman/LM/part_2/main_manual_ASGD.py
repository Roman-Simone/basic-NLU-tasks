from utils import *
from model import *
from functions import *

import copy
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from functools import partial
from torch.utils.data import DataLoader


# Main function
if __name__ == "__main__":
    #PARAMETERS
    config = {
        "batch_size_train": 32,
        "batch_size_dev": 128,
        "batch_size_test": 128,
        "hid_size": 500,
        "emb_size": 500,
        "lr": 2.5 ,
        "clip": 5,
        "n_epochs": 100,
        "patience": 5
    }

    #LOADING DATA
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_raw = read_file(os.path.join(current_dir, "dataset/PennTreeBank/ptb.train.txt"))
    dev_raw = read_file(os.path.join(current_dir, "dataset/PennTreeBank/ptb.valid.txt"))
    test_raw = read_file(os.path.join(current_dir, "dataset/PennTreeBank/ptb.test.txt"))

    #get vocab and trasform in id
    vocab = get_vocab(train_raw, ["<pad>", "<eos>"])
    lang = Lang(train_raw, ["<pad>", "<eos>"])

    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size_train"], collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config["batch_size_dev"], collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size_test"], collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))

    vocab_len = len(lang.word2id)

    #TRAINING
    model = LM_LSTM_DROP(config["emb_size"], config["hid_size"], vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
    model.apply(init_weights)

    optimizer = optim.SGD(model.parameters(), lr=config["lr"])
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')


    losses_train = []
    losses_dev = []
    ppl_dev_list = []
    ppl_train_list = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    pbar = tqdm(range(1, config["n_epochs"]))
    final_epoch = 0
    switch_ASGD = False
    window = 3
    list_weights = []
   

    for epoch in pbar:
        final_epoch = epoch
        ppl_train, loss_train = train_loop(train_loader, optimizer, criterion_train, model, config["clip"])
        ppl_train_list.append(ppl_train)
        sampled_epochs.append(epoch)
        losses_train.append(np.asarray(loss_train).mean())


        #if switched in ASGD
        if switch_ASGD:

            # Save model weights
            weights = {}
            for prm in model.parameters():
                weights[prm] = prm.data.clone()
              
            
            # Save the last n_window weights
            if len(list_weights) < window:
                list_weights.append(weights)
                print("non dovrei essere qui")
            else:
                list_weights.pop(0)
                list_weights.append(weights)
            
            #media dei pesi
            avg_weights = average_weights(list_weights, model)
            
             # Load average weights into the model
            for prm in model.parameters():
                prm.data = avg_weights[prm].clone()

            optimizer = optim.SGD(model.parameters(), lr=config["lr"])

            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            ppl_dev_list.append(ppl_dev)
            losses_dev.append(np.asarray(loss_dev).mean())
            pbar.set_description(f"lr= {config["lr"]} ASGD= {switch_ASGD} PPL: {ppl_dev}")       

            # Load model weights
            for prm in model.parameters():
                prm.data = weights[prm].clone()

        else:


            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            ppl_dev_list.append(ppl_dev)
            losses_dev.append(np.asarray(loss_dev).mean())
            pbar.set_description(f"lr= {config["lr"]} ASGD= {switch_ASGD} PPL: {ppl_dev}")

            if (len(losses_dev) > window and loss_dev > min(losses_dev[:-window])):
                # lr *= 4/10
                # optimizer.param_groups[0]['lr'] = lr
                optimizer = optim.SGD(best_weights, lr=config["lr"])
                print(epoch)
                switch_ASGD = True
            

        if  ppl_dev < best_ppl: # the lower, the better
            best_ppl = ppl_dev
            best_model = copy.deepcopy(model).to('cpu')
            best_weights = model.parameters()
            patience = 5
        elif ppl_dev > best_ppl and switch_ASGD==False:
            config["lr"] *= 4/10
            optimizer.param_groups[0]['lr'] = config["lr"]
            # Save model weights
            weights = {}
            for prm in model.parameters():
                weights[prm] = prm.data.clone()
              
            
            # Save the last n_window weights
            if len(list_weights) < window:
                list_weights.append(weights)
            else:
                list_weights.pop(0)
                list_weights.append(weights)
        elif ppl_dev > best_ppl and switch_ASGD==True:
            config["lr"] *= 4/10
            optimizer.param_groups[0]['lr'] = config["lr"]
            patience -= 1

        if patience <= 0: # Early stopping with patience
            break # Not nice but it keeps the code clean


    best_model.to(DEVICE)
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)
    print('Test ppl: ', final_ppl)

    name_exercise = "PART_22"
    save_result(name_exercise, sampled_epochs, losses_train, losses_dev, ppl_train_list, ppl_dev_list, 
                final_epoch, best_ppl, final_ppl, optimizer, model, best_model, config)
