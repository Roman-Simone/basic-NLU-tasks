# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
import torch
from semeval_base import category_detection

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
            output_slots = torch.argmax(slots,dim=1)
            for id_seq, seq in enumerate(output_slots):

                # attention_mask = sample["attention"].tolist()[id_seq]
                length = sample['slots_len'].tolist()[id_seq]

                utt_ids = sample['utterance'][id_seq][:length].tolist()
                utterance = tokenizer.convert_ids_to_tokens(utt_ids)

                gt_ids = sample['y_slots'][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]

                pad_positions = [i for i, slot in enumerate(gt_slots) if slot == 'pad']
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots) if elem != 'pad'])
                # ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                
                to_decode = seq[:length].tolist()  
                # print(to_decode)
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                # hyp_slots.append(tmp_seq)
                hyp_slots.append([tmp_seq[id_el] for id_el, elem in enumerate(to_decode) if id_el not in pad_positions])
              

    try:           
        results = category_detection(length, ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total":{"f":0}}

    return results, loss_array
