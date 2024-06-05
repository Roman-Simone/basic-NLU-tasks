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

# Main function
if __name__ == "__main__":
    # NAME OF THE MODEL to load
    # name_model = "model_11.pt"
    name_model = "model_11.pt"
    path_model_saved = f"{directory.parent}/bin/{name_model}"
    loaded_object = torch.load(path_model_saved, map_location=device)

    # Configuration parameters
    config = {
        "batch_test_size": 128,       # Batch size for testing
        "hid_size": 768,              # Hidden state size
    }

    path_dataset = directory.parent
    # Load test data
    test_raw = load_data(os.path.join(path_dataset, "dataset/laptop14_test.txt"))
    
    # Split the data into sub-sequences
    test_raw_split = split_data(test_raw)

    # Initialize the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # Load lang
    lang = Lang([], cutoff=0)
    lang.load(loaded_object["aspect2id"])

    # Create datasets for training, validation, and testing
    test_dataset = map_aspect(test_raw_split, lang, tokenizer)

    # Create dataloaders for training, validation, and testing
    test_loader = DataLoader(test_dataset, batch_size=config["batch_test_size"], collate_fn=collate_fn)

    out_aspect = len(lang.aspect2id)

    # Initialize the BERT model
    model = ModelBert(config["hid_size"], out_aspect).to(device)
    model.load_state_dict(loaded_object["model"])
    
    # Configure the optimizer and loss function
    criterion_aspects = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    # Test the best model on the test set
    results_test, _ = eval_loop(test_loader, criterion_aspects, model, lang, tokenizer)    
    precision = results_test['Precision']
    recall = results_test['Recall']
    f1 = results_test['F1']
    
    print(f"Precision {precision:.2f}, Recall {recall:.2f}, F1 {f1:.2f}")
