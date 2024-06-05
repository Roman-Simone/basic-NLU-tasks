import os
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


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


def save_result(name_exercise, sampled_epochs, losses_train, losses_dev, ppl_train_list, ppl_dev_list, 
                final_epoch, best_ppl, final_ppl, optimizer, model, best_model, config):
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
    plt.plot(sampled_epochs, losses_train, '-', label='Train')
    plt.plot(sampled_epochs, losses_dev, '-', label='Dev')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(folder_path, "LOSS_TRAIN_vs_DEV.pdf"))

    plt.figure()
    plt.plot(sampled_epochs, ppl_train_list, '-', label='Train')
    plt.plot(sampled_epochs, ppl_dev_list, '-', label='Dev')
    plt.xlabel('Epochs')
    plt.ylabel('PPL')
    plt.legend()
    plt.savefig(os.path.join(folder_path, "PPL_TRAIN_vs_DEV.pdf"))

    # Create a text file and save it in the folder_path with the training parameters
    file_path = os.path.join(folder_path, "training_parameters.txt")
    with open(file_path, "w") as file:
        file.write(f"{name_exercise}\n\n")
        for key, value in config.items():
            file.write(f"{key}: {value}\n")
        file.write(f"Best Dev PPL: {best_ppl}\n")
        file.write(f"Best Test PPL: {final_ppl}\n")
        file.write(f"Optimizer: {optimizer}\n")
        file.write(f"Model: {model}\n")

    # To save the model
    torch.save(best_model.state_dict(), os.path.join(folder_path, "model.pt"))


# for manual ASGD it is necessary to compute the average of the weights
def average_weights(list_weights, model):
    avg_weights = {}

    if len(list_weights)==1:
        for prm in model.parameters():
            avg_weights[prm] = prm.data.clone()
    else:
        for elem in list_weights:
            for key in elem.keys():
                if key not in avg_weights:
                    avg_weights[key] = elem[key].clone()
                else:
                    avg_weights[key] += elem[key]
        for key in avg_weights.keys():
            avg_weights[key] /= len(list_weights)
    
    return avg_weights