from utils import *      # Import all functions from the utils module
from model import *      # Import all functions and classes from the model module
from functions import *  # Import all functions from the functions module

from transformers import BertTokenizer
from torch.utils.data import DataLoader

# Main function
if __name__ == "__main__":
    # write path of the model
    path_model_saved = "/home/disi/NLU/exam/247181_simone_roman/SA/part_1/results/SA_test_7_f1_86.37_Prec_91.13_recall_82.1/model.pt"
    loaded_object = torch.load(path_model_saved)

    # Configuration parameters
    config = {
        "batch_test_size": 128,       # Batch size for testing
        "hid_size": 768,              # Hidden state size
    }
    patience = 3

    print("TAKE DATASET")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Load test data
    test_raw = load_data(os.path.join(current_dir, "dataset/laptop14_test.txt"))
    
    # Split the data into sub-sequences
    test_raw_split = split_data(test_raw)
    
    print("CREATE LANG")
    # Initialize the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # Load lang
    lang = Lang([], cutoff=0)
    lang.load(loaded_object["aspect2id"])


    # Create datasets for training, validation, and testing
    test_dataset = map_aspect(test_raw_split, lang, tokenizer)

    print("CREATE DATALOADERS")
    # Create dataloaders for training, validation, and testing
    test_loader = DataLoader(test_dataset, batch_size=config["batch_test_size"], collate_fn=collate_fn)

    out_aspect = len(lang.aspect2id)

    # Initialize the BERT model
    model = ModelBert(config["hid_size"], out_aspect).to(device)
    model.load_state_dict(loaded_object["model"])
    
    # Configure the optimizer and loss function
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    # Test the best model on the test set
    results_test, _ = eval_loop(test_loader, criterion_slots, model, lang, tokenizer)    
    precision = results_test['Precision']
    recall = results_test['Recall']
    f1 = results_test['F1']
    
    print(f"Precision {precision:.2f}, Recall {recall:.2f}, F1 {f1:.2f}")
