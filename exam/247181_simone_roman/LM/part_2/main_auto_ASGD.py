# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from utils import *
from model import *
import os
import copy
from collections import deque

def copy_model(model):
    copied_model = type(model)(*model.init_parameters())
    copied_model.load_state_dict(model.state_dict())
    return copied_model

if __name__ == "__main__":
    
    #!PARAMETERS

    batch_size_train = 32
    batch_size_dev = 64
    batch_size_test = 64
    
    hid_size = 650
    emb_size = 650

    lr = 10 # This is definitely not good for SGD
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
   

    #If the PPL is too high try to change the learning rate
    for epoch in pbar:
        final_epoch = epoch
        ppl_train, loss_train = train_loop(train_loader, optimizer, criterion_train, model, clip)
        ppl_train_list.append(ppl_train)
        sampled_epochs.append(epoch)
        losses_train.append(np.asarray(loss_train).mean())

        if 't0' in optimizer.param_groups[0]:
            tmp = {}
            for prm in model.parameters():
                tmp[prm] = prm.data.clone()
                prm.data = optimizer.state[prm]['ax'].clone()
                

            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            ppl_dev_list.append(ppl_dev)
            losses_dev.append(np.asarray(loss_dev).mean())
            pbar.set_description("PPL: %f" % ppl_dev)

            for prm in model.parameters():
                prm.data = tmp[prm].clone()

        else:


            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            ppl_dev_list.append(ppl_dev)
            losses_dev.append(np.asarray(loss_dev).mean())
            pbar.set_description("PPL: %f" % ppl_dev)

            if len(losses_dev)>window and loss_dev > min(losses_dev[:-window]):
                print('Switching to ASGD')
                optimizer = torch.optim.ASGD(best_weights, lr=2, t0=0, lambd=0., weight_decay=1.2e-06)
            



        if  ppl_dev < best_ppl: # the lower, the better
            best_ppl = ppl_dev
            best_model = copy.deepcopy(model).to('cpu')
            best_weights = model.parameters()
            patience = 3
        elif ppl_dev > best_ppl and 't0' in optimizer.param_groups[0]:
            patience -= 1
            lr=lr/2

        if epoch % 5 == 0:
            lr = lr - 0.75
            print('Learning rate changed to: ', lr)
            optimizer.param_groups[0]['lr'] = lr

        if patience <= 0 and switch_ASGD: # Early stopping with patience
            break # Not nice but it keeps the code clean


    best_model.to(DEVICE)
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)
    print('Test ppl: ', final_ppl)

    name_exercise = "PART_22"
    save_result(name_exercise, sampled_epochs, losses_train, losses_dev,ppl_train_list, ppl_dev_list, hid_size, 
                emb_size, lr, clip, vocab_len, final_epoch,best_ppl, final_ppl, batch_size_train, batch_size_dev, batch_size_test, optimizer, model, best_model)

