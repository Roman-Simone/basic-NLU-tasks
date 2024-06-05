import sys
import pathlib
from functools import partial
from torch.utils.data import DataLoader

# directory reach
directory = pathlib.Path(__file__).parent.resolve()
sys.path.append(str(directory.parent))

from utils import *
from model import *
from functions import *  # Import everything from functions.py file


# Main function FOR TESTING
if __name__ == "__main__":

    # NAME OF THE MODEL to load
    # name_model = "model_11.pt" -> LSTM with hid_size=500, emb_size=500
    # name_model = "model_12.pt" -> Dropout with hid_size=300, emb_size=300
    # name_model = "model_13.pt" -> AdamW with hid_size=400, emb_size=400
    name_model = "model_13.pt"
    path_model_saved = f"{directory.parent}/bin/{name_model}"


    # IPERPARAMETERS
    # For model_11.pt  setting --> "hid_size": 500, "emb_size": 500
    # For model_12.pt  setting --> "hid_size": 300, "emb_size": 300
    # For model_13.pt  setting --> "hid_size": 400, "emb_size": 400 
    config = {
        "batch_size_test": 128,
        "hid_size": 400,
        "emb_size": 400,
    }
    
    # LOADING DATA
    path_dataset = directory.parent
    train_raw = read_file(os.path.join(path_dataset, "dataset/PennTreeBank/ptb.train.txt"))
    test_raw = read_file(os.path.join(path_dataset, "dataset/PennTreeBank/ptb.test.txt"))

    #get vocab and trasform in id
    vocab = get_vocab(train_raw, ["<pad>", "<eos>"])
    lang = Lang(train_raw, ["<pad>", "<eos>"])

    test_dataset = PennTreeBank(test_raw, lang)

    test_loader = DataLoader(test_dataset, batch_size=config["batch_size_test"], 
                             collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))

    # Get vocab length
    vocab_len = len(lang.word2id)

    model = LM_LSTM(config["emb_size"], config["hid_size"], vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
    model.load_state_dict(torch.load(path_model_saved, map_location=DEVICE))
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')


    # TESTING
    final_ppl, _ = eval_loop(test_loader, criterion_eval, model)
    print('Test ppl: ', final_ppl)
