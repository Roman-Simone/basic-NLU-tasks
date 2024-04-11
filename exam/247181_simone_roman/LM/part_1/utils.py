# Add functions or classes used for data loading and preprocessing
# Loading the corpus
import os
import matplotlib.pyplot as plt

def read_file(path, eos_token="<eos>"):
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line.strip() + " " + eos_token)
    return output

# Vocab with tokens to ids
def get_vocab(corpus, special_tokens=[]):
    output = {}
    i = 0
    for st in special_tokens:
        output[st] = i
        i += 1
    for sentence in corpus:
        for w in sentence.split():
            if w not in output:
                output[w] = i
                i += 1
    return output


# This class computes and stores our vocab
# Word to ids and ids to word
class Lang():
    def __init__(self, corpus, special_tokens=[]):
        self.word2id = self.get_vocab(corpus, special_tokens)
        self.id2word = {v:k for k, v in self.word2id.items()}
    def get_vocab(self, corpus, special_tokens=[]):
        output = {}
        i = 0
        for st in special_tokens:
            output[st] = i
            i += 1
        for sentence in corpus:
            for w in sentence.split():
                if w not in output:
                    output[w] = i
                    i += 1
        return output
    

def save_result(name_exercise, sampled_epochs, losses_train, losses_dev,ppl_train_list, ppl_dev_list, hid_size, emb_size, lr, clip, vocab_len, epoch, final_ppl, batch_size_train, batch_size_dev, batch_size_test, optimizer, model):
    # Create a folder
    folder_path = "results"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    num_folders = len([name for name in os.listdir(folder_path) if name.startswith(name_exercise)])
    title = f"{name_exercise}_test_{num_folders + 1}"
    folder_path = os.path.join(folder_path, title)
    os.makedirs(folder_path, exist_ok=True)

    plt.figure()
    plt.plot(sampled_epochs, losses_train, 'o-', label='Train')
    plt.plot(sampled_epochs, losses_dev, 'o-', label='Dev')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(folder_path, "LOSS_TRAIN_vs_DEV.pdf"))

    plt.figure()
    plt.plot(sampled_epochs, ppl_train_list, 'o-', label='Train')
    plt.plot(sampled_epochs, ppl_dev_list, 'o-', label='Dev')
    plt.xlabel('Epochs')
    plt.ylabel('PPL')
    plt.legend()
    plt.savefig(os.path.join(folder_path, "PPL_TRAIN_vs_DEV.pdf"))

    # Create a text file and save it in the folder_path with the training parameters
    file_path = os.path.join(folder_path, "training_parameters.txt")
    with open(file_path, "w") as file:
        file.write(f"{name_exercise}\n\n")
        file.write(f"Hidden Size: {hid_size}\n")
        file.write(f"Embedding Size: {emb_size}\n")
        file.write(f"Learning Rate: {lr}\n")
        file.write(f"Clip: {clip}\n")
        file.write(f"Vocabulary Length: {vocab_len}\n")
        file.write(f"Number of Epochs: {epoch}\n")
        file.write(f"Best Test PPL: {final_ppl}\n")
        file.write(f"Batch Size Train: {batch_size_train}\n")
        file.write(f"Batch Size Dev: {batch_size_dev}\n")
        file.write(f"Batch Size Test: {batch_size_test}\n")
        file.write(f"Optimizer: {optimizer}\n")
        file.write(f"Model: {model}\n")

