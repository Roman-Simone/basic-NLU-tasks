# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
import torch
from transformers import BertTokenizer
from conll import evaluate
from sklearn.metrics import classification_report

def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    model.train()
    loss_array = []

    for sample in data:

        optimizer.zero_grad() # Zeroing the gradient
        # print(sample['utterances'])
        slots, intent = model(sample['utterances'], sample["attentions"], sample["token_type_ids"])
        loss_intent = criterion_intents(intent, sample['intents'])
        loss_slot = criterion_slots(slots, sample['y_slots'])
        loss = loss_intent + loss_slot

        loss_array.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

    return loss_array

def eval_loop(data, criterion_slots, criterion_intents, model, lang, tokenizer):
    model.eval()
    loss_array = []

    ref_intents = []
    hyp_intents = []

    ref_slots = []
    hyp_slots = []

    with torch.no_grad():
        for sample in data:
            slots, intents = model(sample["utterances"], sample["attentions"], sample["token_type_ids"])
            loss_intent = criterion_intents(intents, sample['intents'])
            loss_slot = criterion_slots(slots, sample['y_slots'])
            loss = loss_intent + loss_slot 
            loss_array.append(loss.item())

            #intent inference
            #get the highest probable class
            out_intent = [lang.id2intent[x] for x in torch.argmax(intents, dim=1).tolist()] 
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intent)

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
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total":{"f":0}}
        
    report_intent = classification_report(ref_intents, hyp_intents, 
                                          zero_division=False, output_dict=True)
    return results, report_intent, loss_array




