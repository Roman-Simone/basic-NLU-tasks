# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
import torch
import torch.nn as nn
from conll import evaluate_ote


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
        results = {"total":{"f":results}}
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