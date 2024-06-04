from utils import *
from model import *
from functions import *  # Import everything from functions.py file

from functools import partial
from torch.utils.data import DataLoader


# Main function
if __name__ == "__main__":
    # write path of the model
    path_model_saved = "/home/disi/NLU/exam/247181_simone_roman/LM/part_2/results/PART_22_MANUAL_test_1_PPL_90/model.pt"

    # PARAMETERS
    # For model_21.pt  setting --> "hid_size": 500, "emb_size": 500
    # For model_22.pt  setting --> "hid_size": 300, "emb_size": 300
    # For model_23.pt  setting --> "hid_size": 400, "emb_size": 400 
    config = {
        "batch_size_test": 128,
        "hid_size": 500,
        "emb_size": 500,
    }
    
    # LOADING DATA
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_raw = read_file(os.path.join(current_dir, "dataset/PennTreeBank/ptb.train.txt"))
    test_raw = read_file(os.path.join(current_dir, "dataset/PennTreeBank/ptb.test.txt"))

    #get vocab and trasform in id
    vocab = get_vocab(train_raw, ["<pad>", "<eos>"])
    lang = Lang(train_raw, ["<pad>", "<eos>"])

    test_dataset = PennTreeBank(test_raw, lang)

    test_loader = DataLoader(test_dataset, batch_size=config["batch_size_test"], 
                             collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))

    # Get vocab length
    vocab_len = len(lang.word2id)

    # TRAINING
    model = LM_LSTM(config["emb_size"], config["hid_size"], vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
    model.load_state_dict(torch.load(path_model_saved))

    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')


    final_ppl, _ = eval_loop(test_loader, criterion_eval, model)
    print('Test ppl: ', final_ppl)
