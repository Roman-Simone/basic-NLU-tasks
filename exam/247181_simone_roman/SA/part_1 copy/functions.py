# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Constants
SMALL_POSITIVE_CONST = 1e-4


def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    model.train()
    loss_array = []

    for sample in data:

        optimizer.zero_grad() # Zeroing the gradient
        # print(sample['utterances'])
        slots = model(sample['utterances'], sample["attentions"], sample["token_type_ids"])
        loss_slot = criterion_slots(slots, sample['y_slots'])

        loss_array.append(loss_slot.item())
        loss_slot.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

    return loss_array


def eval_loop(data, criterion_slots, criterion_intents, model, lang, tokenizer):
    model.eval()
    loss_array = []

    ref_slots = []
    hyp_slots = []

    with torch.no_grad():
        for sample in data:
            slots = model(sample["utterances"], sample["attentions"], sample["token_type_ids"])
            loss_slot = criterion_slots(slots, sample['y_slots'])
            loss_array.append(loss_slot.item())

            # slot inference
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):

                # attention_mask = sample["attention"].tolist()[id_seq]
                # length = sample['slots_len'].tolist()[id_seq]

                # utt_ids = sample['utterance'][id_seq].tolist()
                # utterance = tokenizer.convert_ids_to_tokens(utt_ids)

                gt_ids = sample['y_slots'][id_seq].tolist()
                # gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]

                # pad_positions = [i for i, slot in enumerate(gt_slots) if slot == 'pad']
                # ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots) if elem != 'pad'])
                # ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                ref_slots.append(gt_ids)
                
                to_decode = seq.tolist()  
                hyp_slots.append(to_decode)
                
                # print(to_decode)
                # tmp_seq = []
                # for id_el, elem in enumerate(to_decode):
                    # tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                # hyp_slots.append(tmp_seq)
                # hyp_slots.append([tmp_seq[id_el] for id_el, elem in enumerate(to_decode) if id_el not in pad_positions])
                # print("sium")
                # print(gt_ids, to_decode)
                # print(len(gt_ids), len(to_decode))
                # gt_slots = [lang.id2slot[elem] for elem in gt_ids]
                # hyp_slots = [lang.id2slot[elem] for elem in to_decode]
                # print(gt_slots, hyp_slots)
                # print()
              

    try:           
        results = evaluate_ote(ref_slots, hyp_slots, lang)
        results = {"Precision":results[0], "Recall":results[1], "F1":results[2]}
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total":{"f":0}}

    return results, loss_array


def init_weights(mat):
    for n, m in mat.named_modules():
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
                if 'slot_out' in n or 'intent_out' in n:
                    torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                    if m.bias != None:
                        m.bias.data.fill_(0.01)


def evaluate_ote(gold_ot, pred_ot, lang):
    """
    evaluate the model performce for the ote task
    :param gold_ot: gold standard ote tags
    :param pred_ot: predicted ote tags
    :return:
    """
    id_pad = lang.slot2id['pad']
    id_T = lang.slot2id['T']
    id_O = lang.slot2id['O']
    n_samples = len(gold_ot)
    # number of true positive, gold standard, predicted opinion targets
    n_tp_ot, n_gold_ot, n_pred_ot = 0, 0, 0
    for i in range(n_samples):
        g_ot = gold_ot[i]
        p_ot = pred_ot[i]
        
        # hit number
        n_hit_ot = 0
        for ref, pred in zip(g_ot, p_ot):
            if ref == pred and ref == id_T:
                n_hit_ot += 1

        n_tp_ot += n_hit_ot
        n_gold_ot += sum([1 for ot in g_ot if ot == id_T])
        n_pred_ot += sum([1 for ot in p_ot if ot == id_T])
    # add 0.001 for smoothing
    # calculate precision, recall and f1 for ote task
    ot_precision = float(n_tp_ot) / float(n_pred_ot + SMALL_POSITIVE_CONST)
    ot_recall = float(n_tp_ot) / float(n_gold_ot + SMALL_POSITIVE_CONST)
    ot_f1 = 2 * ot_precision * ot_recall / (ot_precision + ot_recall + SMALL_POSITIVE_CONST)
    ote_scores = (ot_precision, ot_recall, ot_f1)
    return ote_scores


def save_result(name_exercise, sampled_epochs, losses_train, losses_dev, config, results_dev, results_test, best_model):
    
    # Create a folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(current_dir, "results")
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

    # Create a text file and save it in the folder_path with the training parameters
    file_path = os.path.join(folder_path, "training_parameters.txt")
    with open(file_path, "w") as file:
        file.write(f"{name_exercise}\n\n")
        file.write(f"lr: {config['lr']}\n")
        file.write(f"clip: {config['clip']}\n")
        file.write(f"n_epochs: {config['n_epochs']}\n")
        file.write(f"hid_size: {config['hid_size']}\n")
        file.write(f"Results:\n")
        for key, value in results_test.items():
            file.write(f"{key}: {value}\n")
        

    # To save the model
    # torch.save(best_model.state_dict(), os.path.join(folder_path, "model.pt"))