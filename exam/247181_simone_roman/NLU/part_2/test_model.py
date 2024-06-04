from functions import *
from utils import *
from model import *

import copy
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer

if __name__ == "__main__":
    # write path of the model
    path_model_saved = "/Users/simoneroman/Desktop/NLU/NLU/exam/247181_simone_roman/NLU/part_2/results/PART_2_test_4_f1_95.48_acc_96.98/model.pt"
    loaded_object = torch.load(path_model_saved, map_location=device)

    #Parameters
    config = {
        "batch_test_size": 64,
        "hid_size": 768,
    }
   
    #load data
    print("TAKE DATASET")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_raw = load_data(os.path.join(current_dir, "dataset/ATIS/test.json"))

    print("CREATE LANG")
    # Create an id for each word, intent and slot for word use bert tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # Load lang
    lang = Lang([], [], cutoff=0)
    lang.load(loaded_object["slot2id"], loaded_object["intent2id"])

    test_dataset = IntentsAndSlots(test_raw, lang, tokenizer)

    print("CREATE DATALOADERS")
    test_loader = DataLoader(test_dataset, batch_size=config["batch_test_size"], collate_fn=collate_fn)

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)

    model = ModelBert(config["hid_size"], out_slot, out_int).to(device)
    model.load_state_dict(loaded_object["model"])

    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss() # Because we do not have the pad token


    results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang, tokenizer)    
    test_f1 = results_test['total']['f']
    test_acc = intent_test['accuracy']
    
    print(f"Slot F1: {test_f1}  Intent Accuracy: {test_acc}")
