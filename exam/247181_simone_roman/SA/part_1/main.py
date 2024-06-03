from utils import *      # Import all functions from the utils module
from model import *      # Import all functions and classes from the model module
from functions import *  # Import all functions from the functions module

import copy
import numpy as np
from tqdm import tqdm    # Import tqdm for progress bar visualization
import torch.optim as optim
from transformers import BertTokenizer
from torch.utils.data import DataLoader

# Main function
if __name__ == "__main__":
    # Configuration parameters
    config = {
        "lr": 0.0001,                 # Learning rate
        "batch_train_size": 128,      # Batch size for training
        "batch_dev_size": 128,        # Batch size for validation
        "batch_test_size": 128,       # Batch size for testing
        "clip": 5,                    # Gradient clipping
        "n_epochs": 100,              # Number of epochs
        "hid_size": 768,              # Hidden state size
    }
    patience = 3

    print("TAKE DATASET")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Load training and test data
    tmp_train_raw = load_data(os.path.join(current_dir, "dataset/laptop14_train.txt"))
    test_raw = load_data(os.path.join(current_dir, "dataset/laptop14_test.txt"))

    # Create the validation set
    train_raw, dev_raw = create_dev(tmp_train_raw)
    
    # Split the data into sub-sequences
    train_raw_split = split_data(train_raw)
    dev_raw_split = split_data(dev_raw)
    test_raw_split = split_data(test_raw)

    # Create the combined corpus
    corpus = train_raw_split + dev_raw_split + test_raw_split
    aspects = set(sum([line['aspect'].split() for line in corpus],[]))
    
    print("CREATE LANG")
    # Initialize the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # Create the Lang object
    lang = Lang(aspects, cutoff=0)

    # Create datasets for training, validation, and testing
    train_dataset = map_aspect(train_raw_split, lang, tokenizer)
    dev_dataset = map_aspect(dev_raw_split, lang, tokenizer)
    test_dataset = map_aspect(test_raw_split, lang, tokenizer)

    print("CREATE DATALOADERS")
    # Create dataloaders for training, validation, and testing
    train_loader = DataLoader(train_dataset, batch_size=config["batch_train_size"], collate_fn=collate_fn, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config["batch_dev_size"], collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_test_size"], collate_fn=collate_fn)

    out_aspect = len(lang.aspect2id)

    # Initialize the BERT model
    model = ModelBert(config["hid_size"], out_aspect).to(device)
    model.apply(init_weights)
    
    # Configure the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    losses_train = []
    losses_dev = []
    sampled_epochs = []
    results_dev = []
    best_score = 0

    # Progress bar for epochs
    pbar = tqdm(range(1, config["n_epochs"]))
    for x in pbar:
        # Execute the training loop
        loss = train_loop(train_loader, optimizer, criterion_slots, model, clip=config["clip"])

        # Validate the model every epoch
        if x % 1 == 0:  # Check performance every epoch
            sampled_epochs.append(x)
            losses_train.append(np.asarray(loss).mean())

            result_dev, loss_dev = eval_loop(dev_loader, criterion_slots, model, lang, tokenizer)
            losses_dev.append(np.asarray(loss_dev).mean())
            results_dev.append(result_dev)
            precision = result_dev['Precision']
            recall = result_dev['Recall']
            f1 = result_dev['F1']

            score = (precision + recall + f1) / 3

            # Save the best model
            if score > best_score:
                best_score = score
                best_model = copy.deepcopy(model).to('cpu')
                patience = 3
            else:
                patience -= 1
            if patience <= 0:  # Early stopping
                break  

        pbar.set_description(f"Loss: {np.asarray(loss).mean():.2f}, score-> {result_dev}")

    # Test the best model on the test set
    results_test, _ = eval_loop(test_loader, criterion_slots, model, lang, tokenizer)    
    precision = results_test['Precision']
    recall = results_test['Recall']
    f1 = results_test['F1']
    
    print(f"Precision {precision:.2f}, Recall {recall:.2f}, F1 {f1:.2f}")

    # Save the results
    name_exercise = "SA"
    save_result(name_exercise, sampled_epochs, losses_train, losses_dev, config, results_dev, results_test, best_model)
