import sys
import pathlib
from transformers import BertTokenizer
from torch.utils.data import DataLoader


# directory reach
directory = pathlib.Path(__file__).parent.resolve()
sys.path.append(str(directory.parent))

from utils import *
from model import *
from functions import *  # Import everything from functions.py file


if __name__ == "__main__":
    # NAME OF THE MODEL to load
    # name_model = "model_21.pt"
    name_model = "model_21.pt"
    path_model_saved = f"{directory.parent}/bin/{name_model}"
    loaded_object = torch.load(path_model_saved, map_location=device)

    # HYPERPARAMETERS
    config = {
        "batch_test_size": 64,
        "hid_size": 768,
    }
   
    #load data
    path_dataset = directory.parent
    test_raw = load_data(os.path.join(path_dataset, "dataset/ATIS/test.json"))


    # Create an id for each word, intent and slot for word use bert tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # Load lang
    lang = Lang([], [], cutoff=0)
    lang.load(loaded_object["slot2id"], loaded_object["intent2id"])

    test_dataset = IntentsAndSlots(test_raw, lang, tokenizer)

    test_loader = DataLoader(test_dataset, batch_size=config["batch_test_size"], collate_fn=collate_fn)

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)

    model = ModelBert(config["hid_size"], out_slot, out_int).to(device)
    model.load_state_dict(loaded_object["model"])

    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss() # Because we do not have the pad token

    # Evaluate the model
    results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang, tokenizer)    
    test_f1 = results_test['total']['f']
    test_acc = intent_test['accuracy']
    
    # Print the results
    print(f"Slot F1: {test_f1}  Intent Accuracy: {test_acc}")
