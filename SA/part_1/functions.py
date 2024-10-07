import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Constants
SMALL_POSITIVE_CONST = 1e-4

# Training loop function
def train_loop(data, optimizer, criterion_aspects, model, clip=5):
    model.train()
    loss_array = []

    for sample in data:
        optimizer.zero_grad()  # Zeroing the gradient
        aspects = model(sample['sentences'], sample["attentions"], sample["token_type_ids"])
        loss_slot = criterion_aspects(aspects, sample['y_aspect'])

        loss_array.append(loss_slot.item())
        loss_slot.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  # Gradient clipping
        optimizer.step()

    return loss_array

# Evaluation loop function
def eval_loop(data, criterion_aspects, model, lang, tokenizer):
    model.eval()
    
    loss_array = []
    ref_aspects = []
    hyp_aspects = []

    id_pad = lang.aspect2id['pad']

    with torch.no_grad():
        for sample in data:
            aspects = model(sample['sentences'], sample["attentions"], sample["token_type_ids"])
            loss_slot = criterion_aspects(aspects, sample['y_aspect'])
            loss_array.append(loss_slot.item())

            # Slot inference
            output_aspects = torch.argmax(aspects, dim=1)
            for id_seq, seq in enumerate(output_aspects):
                
                #take ground truth
                gt_ids = sample['y_aspect'][id_seq].tolist()

                #take slot output 
                to_decode = seq.tolist()

                #remove pad from gt_ids and from to__decode
                gt_ids_no_pad = []
                to_decode_no_pad = []
                for elem_gt, elem_hyp in zip(gt_ids, to_decode):
                    if elem_gt != id_pad:
                        gt_ids_no_pad.append(elem_gt)
                        to_decode_no_pad.append(elem_hyp)

                ref_aspects.append(gt_ids_no_pad)
                hyp_aspects.append(to_decode_no_pad)

                if len(gt_ids_no_pad) != len(to_decode_no_pad):
                    print("Length mismatch between reference and hypothesis slots")

    try:
        results = evaluate_ote(ref_aspects, hyp_aspects, lang)
        results = {"Precision": results[0], "Recall": results[1], "F1": results[2]}
    except Exception as ex:
        # Handle cases where the model predicts a class not in reference
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_aspects])
        hyp_s = set([x[1] for x in hyp_aspects])
        print(hyp_s.difference(ref_s))
        results = {"total": {"f": 0}}

    return results, loss_array

# Initialize weights
def init_weights(mat):
    for n, m in mat.named_modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        torch.nn.init.xavier_uniform_(param[idx * mul:(idx + 1) * mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        torch.nn.init.orthogonal_(param[idx * mul:(idx + 1) * mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        elif type(m) == nn.Linear:
            if 'slot_out' in n or 'intent_out' in n:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

# Evaluate the model performance 
def evaluate_ote(gold_ot, pred_ot, lang):
    id_pad = lang.aspect2id['pad']
    id_T = lang.aspect2id['T']
    id_O = lang.aspect2id['O']
    n_samples = len(gold_ot)

    n_tp_ot, n_gold_ot, n_pred_ot = 0, 0, 0
    for i in range(n_samples):
        g_ot = gold_ot[i]
        p_ot = pred_ot[i]
        
        # hit if the target is T and the prediction is T
        n_hit_ot = sum(1 for ref, pred in zip(g_ot, p_ot) if pred == id_T and ref == id_T)

        n_tp_ot += n_hit_ot
        n_gold_ot += sum(1 for ot in g_ot if ot == id_T) # count the number of T in the reference
        n_pred_ot += sum(1 for ot in p_ot if ot == id_T) # count the number of T in the prediction

    ot_precision = float(n_tp_ot) / float(n_pred_ot + SMALL_POSITIVE_CONST)
    ot_recall = float(n_tp_ot) / float(n_gold_ot + SMALL_POSITIVE_CONST)
    ot_f1 = (2 * ot_precision * ot_recall) / (ot_precision + ot_recall + SMALL_POSITIVE_CONST)
    
    return ot_precision, ot_recall, ot_f1

# Save the training results
def save_result(name_exercise, sampled_epochs, losses_train, losses_dev, config, results_dev, results_test, best_model, lang, optimizer):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(current_dir, "results")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    num_folders = len([name for name in os.listdir(folder_path) if name.startswith(name_exercise)])
    precision = round(results_test['Precision']*100,2)
    recall = round(results_test['Recall']*100,2)
    f1 = round(results_test['F1']*100,2)
    title = f"{name_exercise}_test_{num_folders + 1}_f1_{f1}_Prec_{precision}_recall_{recall}"
    folder_path = os.path.join(folder_path, title)
    os.makedirs(folder_path, exist_ok=True)

    plt.figure()
    plt.plot(sampled_epochs, losses_train, '-', label='Train')
    plt.plot(sampled_epochs, losses_dev, '-', label='Dev')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(folder_path, "LOSS_TRAIN_vs_DEV.pdf"))

    file_path = os.path.join(folder_path, "training_parameters.txt")
    with open(file_path, "w") as file:
        file.write(f"{name_exercise}\n\n")
        for key, value in config.items():
            file.write(f"{key}: {value}\n")
        file.write("Results:\n")
        for key, value in results_test.items():
            file.write(f"{key}: {value}\n")

    # Save the best model
    path_model = os.path.join(os.path.join(folder_path, "model.pt"))
    saving_object = {"epoch": len(sampled_epochs), 
                     "model": best_model.state_dict(), 
                     "optimizer": optimizer.state_dict(), 
                     "aspect2id": lang.aspect2id}
    torch.save(saving_object, path_model)
