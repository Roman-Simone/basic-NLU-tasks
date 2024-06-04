from utils import *
from model import *
from functions import *

from torch.utils.data import DataLoader


if __name__ == "__main__":
    # write path of the model
    path_model_saved = "/home/disi/NLU/exam/247181_simone_roman/NLU/part_1/results/PART_13_test_2_f1_94.5_acc_94.51/model.pt"
    loaded_object = torch.load(path_model_saved)
    
    # PARAMETERS
    # For model_11.pt  setting --> "flag_bidirectional": False, "flag_dropout": False
    # For model_12.pt  setting --> "flag_bidirectional": True, "flag_dropout": False
    # For model_13.pt  setting --> "flag_bidirectional": True, "flag_dropout": True 
    config = {
        "batch_train_size": 64,
        "batch_dev_size": 64,
        "batch_test_size": 64,
        "hid_size": 300,
        "emb_size": 400,
        "flag_bidirectional": True,
        "flag_dropout": True,
    }

    # Load the data
    current_dir = os.path.dirname(os.path.abspath(__file__)) 
    test_raw = load_data(os.path.join(current_dir, "dataset/ATIS/test.json"))

    # Load lang
    lang = Lang([], [], [], cutoff=0)
    lang.load(loaded_object["word2id"], loaded_object["slot2id"], loaded_object["intent2id"])

    # Create our datasets
    test_dataset = IntentsAndSlots(test_raw, lang)

    # Dataloader instantiations
    test_loader = DataLoader(test_dataset, batch_size=config["batch_test_size"], collate_fn=collate_fn)    

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)

    # load model
    model = ModelIAS(config["hid_size"], out_slot, out_int, config["emb_size"], 
                    vocab_len, pad_index=PAD_TOKEN, flag_bidirectional=config["flag_bidirectional"], flag_dropout=config["flag_dropout"]).to(device)
    model.load_state_dict(loaded_object["model"])

    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss() # Because we do not have the pad token
  

    #Test model
    results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, 
                                                criterion_intents, model, lang)
    test_f1 = results_test['total']['f']
    test_acc = intent_test['accuracy']

    print(test_f1, test_acc)

    
