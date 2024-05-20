# Add functions or classes used for data loading and preprocessing
import torch
import torch.utils.data as data

PAD_TOKEN = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")

def load_data(path):
    data = []
    with open(path, 'r') as file:
        for line in file:
            data.append(line.strip())
    return data


def create_dev(tmp_train_raw):
    dim_dev = 0.1
    dev_data = tmp_train_raw[:int(len(tmp_train_raw)*dim_dev)]
    train_data = tmp_train_raw[int(len(tmp_train_raw)*dim_dev):]
    return train_data, dev_data


def split_data(data):
    data_ret = []
    for elem in data:

        split_utt_slot = elem.split('####')
        
        tmp_slots = ""
        for slot in split_utt_slot[1].split(' '):
            value_slot = slot.split('=')
            if value_slot[-1] != 'O':
                tmp_slots += "T "
            else:
                tmp_slots += value_slot[-1] + " "
        
        tmp_slots = tmp_slots[:len(tmp_slots)-1]

        tmp_elem = {"utterance": split_utt_slot[0], "slots": tmp_slots}
        # if(len(tmp_elem["utterance"].split(" ")) != len(tmp_elem["slots"].split(" "))):
        #     print("Error in splitting")
        #     print(tmp_elem)
        #     print(split_utt_slot[0])
        #     print(split_utt_slot[1])


        data_ret.append(tmp_elem)

    return data_ret


class Lang():
    def __init__(self, slots, cutoff=0):
        self.slot2id = self.lab2id(slots)
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        
    
    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for elem in elements:
                vocab[elem] = int(len(vocab))
        return vocab


class IntentsAndSlots (data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, lang, tokenizer, unk='unk'):
        self.utterances = []
        self.slots = []
        self.unk = unk
        
        for x in dataset:
            self.utterances.append("[CLS] " + x['utterance'] + " [SEP]")
            self.slots.append("O " + x['slots'] + " O")

        self.utt_ids, self.slots_ids, self.attention_mask, self.token_type_id = self.mapping_seq(self.utterances, self.slots, tokenizer, lang.slot2id) 
        # self.check_len(self.utt_ids, self.slots_ids)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.Tensor(self.utt_ids[idx])
        slots = torch.Tensor(self.slots_ids[idx])
        attention = torch.Tensor(self.attention_mask[idx])
        token_type_id = torch.Tensor(self.token_type_id[idx])
        sample = {'utterance': utt, 'slots': slots, 'attention': attention, 'token_type_id': token_type_id}
        return sample
    
    # Auxiliary methods
    def mapping_seq(self, utterance, slots, tokenizer, mapper_slot): # Map sequences to number
        res_utterance = []
        res_slots = []
        res_attention = []
        res_token_type_id = []

        for sequence, slot in zip(utterance, slots):

            tmp_seq = []
            tmp_slot = []
            tmp_attention = []
            tmp_token_type_id = []

            for word, element in zip(sequence.split(), slot.split(' ')):
                tmp_attention.append(1)
                tmp_token_type_id.append(0)

                word_tokens = tokenizer(word)
                #remove CLS and SEP tokens
                word_tokens = word_tokens[1:-1]
                tmp_seq.extend(word_tokens["input_ids"])

                tmp_slot.extend([mapper_slot[element]] + [mapper_slot['pad']] * (len(word_tokens["input_ids"]) - 1))

                for i in range(len(word_tokens["input_ids"])-1):
                    tmp_attention.append(1)
                    tmp_token_type_id.append(0)
            
            # if(self.check_len(tmp_seq, tmp_slot)):
            #     print("Error in mapping")

            if(len(tmp_seq) != len(tmp_slot)):
                print("Error in mapping")
                print(tmp_seq)
                print(tmp_slot)

            res_utterance.append(tmp_seq)
            res_slots.append(tmp_slot)
            res_attention.append(tmp_attention)
            res_token_type_id.append(tmp_token_type_id)

        
        return res_utterance, res_slots, res_attention, res_token_type_id
    
    def check_len(self, utt_ids, slots_ids):
        for utt, slot in zip(utt_ids, slots_ids):
            if len(utt) != len(slot):
                print("Error: Lengths do not match")
        return True
    
def collate_fn(data):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape 
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        # print(padded_seqs)
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths
    

    # Sort data by seq lengths
    data.sort(key=lambda x: len(x['utterance']), reverse=True) 
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
        
    # We just need one length for packed pad seq, since len(utt) == len(slots)
    src_utt, _ = merge(new_item['utterance'])
    y_slots, y_lengths = merge(new_item["slots"])
    attention, _ = merge(new_item["attention"])
    token_type_id, _ = merge(new_item["token_type_id"])
    
    src_utt = src_utt.to(device) # We load the Tensor on our selected device
    y_slots = y_slots.to(device)
    y_lengths = torch.LongTensor(y_lengths).to(device)
    attention = attention.to(device)
    token_type_id = token_type_id.to(device)
    
    new_item["utterances"] = src_utt
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    new_item["attentions"] = attention
    new_item["token_type_ids"] = token_type_id

    return new_item

