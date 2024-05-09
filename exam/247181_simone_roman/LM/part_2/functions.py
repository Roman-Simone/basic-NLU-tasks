# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import math
import os

DEVICE = 'cuda' # it can be changed with 'cpu' if you do not have a gpu


def train_loop(data, optimizer, criterion, model, clip=5):
    model.train()
    loss_array = []
    number_of_tokens = []

    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        output = model(sample['source'])
        loss = criterion(output, sample['target'])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid explosioning gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step() # Update the weights


    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))

    return ppl, loss_to_return

def eval_loop(data, eval_criterion, model):
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []
    # softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            output = model(sample['source'])
            loss = eval_criterion(output, sample['target'])
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])

    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return

def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)



def save_result(name_exercise, sampled_epochs, losses_train, losses_dev,ppl_train_list, 
                ppl_dev_list, hid_size, emb_size, lr, clip, vocab_len, epoch,best_ppl, final_ppl, 
                batch_size_train, batch_size_dev, batch_size_test, optimizer, model, best_model):
    # Create a folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(current_dir, "results")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    num_folders = len([name for name in os.listdir(folder_path) if name.startswith(name_exercise)])
    title = f"{name_exercise}_test_{num_folders + 1}_PPL_{int(final_ppl)}"
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
        file.write(f"Best Dev PPL: {best_ppl}\n")
        file.write(f"Best Test PPL: {final_ppl}\n")
        file.write(f"Batch Size Train: {batch_size_train}\n")
        file.write(f"Batch Size Dev: {batch_size_dev}\n")
        file.write(f"Batch Size Test: {batch_size_test}\n")
        file.write(f"Optimizer: {optimizer}\n")
        file.write(f"Model: {model}\n")

    # To save the model
    torch.save(best_model.state_dict(), os.path.join(folder_path, "model.pt"))


class MovingAverageWeights:
    def __init__(self, window_size):
        self.window_size = window_size
        self.sum_weights = {}
        self.iteration_weight = 0

    def set_sum_weights(self, sum_weights):
        self.sum_weights = sum_weights

    def add_weights(self, model):
        if self.iteration_weight == 0:
            for name, prm in model.named_parameters():
                self.sum_weights[name] = prm.data.clone()
        else:
            for name, prm in model.named_parameters():
                self.sum_weights[name] += prm.data

        self.iteration_weight += 1

        if self.iteration_weight > self.window_size:
            for prm in model.named_parameters():
                self.sum_weights[prm] -= prm.data.clone()

    def get_average_weights(self, model):
        avg_weights = {}
        for prm in model.named_parameters():
            avg_weights[prm] = self.sum_weights[prm] / min(self.window_size, self.iteration_weight)
        return avg_weights