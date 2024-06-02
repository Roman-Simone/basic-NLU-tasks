import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split

PAD_TOKEN = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")

# Function to load data from a file
def load_data(path):
    data = []
    with open(path, 'r') as file:
        for line in file:
            data.append(line.strip())
    return data

# Function to split the training data into training and validation sets
def create_dev(tmp_train_raw):
    train_data, dev_data = train_test_split(tmp_train_raw, test_size=0.1, random_state=42)
    return train_data, dev_data

# Function to split data into utterance and slot pairs
def split_data(data):
    data_ret = []
    for elem in data:
        split_row = elem.split('####')
        tmp_sentence = []
        tmp_aspects = []

        for element in split_row[1].split(" "):
            split_element = element.rsplit('=', 1)
            tmp_sentence.append(split_element[0])

            if split_element[1] == "O":
                tmp_aspects.append('O')
            elif split_element[1][0] == 'T':
                tmp_aspects.append('T')
            else:
                print("error")
        
        tmp_sentence = " ".join(tmp_sentence)
        tmp_aspects = " ".join(tmp_aspects)

        tmp_elem = {"sentence": tmp_sentence, "aspect": tmp_aspects}
        data_ret.append(tmp_elem)

    return data_ret

# Class to handle language-specific mappings
class Lang():
    def __init__(self, slots, cutoff=0):
        self.aspect2id = self.lab2id(slots)
        self.id2aspect = {v: k for k, v in self.aspect2id.items()}
    
    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for elem in elements:
            vocab[elem] = int(len(vocab))
        return vocab

# Custom dataset class for intents and slots
class map_aspect(data.Dataset):
    def __init__(self, dataset, lang, tokenizer, unk='unk'):
        self.sentences = []
        self.aspects = []
        self.unk = unk
        
        for x in dataset:
            self.sentences.append("[CLS] " + x['sentence'] + " [SEP]")
            self.aspects.append("pad " + x['aspect'] + " pad")

        self.sent_ids, self.aspect_ids, self.attention_mask, self.token_type_id = self.mapping_seq(self.sentences, self.aspects, tokenizer, lang.aspect2id)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sent = torch.Tensor(self.sent_ids[idx])
        aspects = torch.Tensor(self.aspect_ids[idx])
        attention = torch.Tensor(self.attention_mask[idx])
        token_type_id = torch.Tensor(self.token_type_id[idx])
        sample = {'sentence': sent, 'aspects': aspects, 'attention': attention, 'token_type_id': token_type_id}
        return sample
    
    # Map sequences to IDs
    def mapping_seq(self, sentences, aspects, tokenizer, mapper_aspect):
        res_sentences = []
        res_aspects = []
        res_attention = []
        res_token_type_id = []

        for sequence, aspect in zip(sentences, aspects):
            tmp_seq = []
            tmp_aspect = []
            tmp_attention = []
            tmp_token_type_id = []

            for word, element in zip(sequence.split(' '), aspect.split(' ')):
                tmp_attention.append(1)
                tmp_token_type_id.append(0)

                word_tokens = tokenizer(word)
                word_tokens = word_tokens[1:-1]  # Remove CLS and SEP tokens
                tmp_seq.extend(word_tokens["input_ids"])

                tmp_aspect.extend([mapper_aspect[element]] + [mapper_aspect['pad']] * (len(word_tokens["input_ids"]) - 1))

                for i in range(len(word_tokens["input_ids"]) - 1):
                    tmp_attention.append(1)
                    tmp_token_type_id.append(0)

            if len(tmp_seq) != len(tmp_aspect):
                print("Error in mapping")
                print(tmp_seq)
                print(tmp_aspect)

            res_sentences.append(tmp_seq)
            res_aspects.append(tmp_aspect)
            res_attention.append(tmp_attention)
            res_token_type_id.append(tmp_token_type_id)

        return res_sentences, res_aspects, res_attention, res_token_type_id


# Collate function to merge sequences
def collate_fn(data):
    def merge(sequences):
        '''
        Merge from batch * sent_len to batch * max_len 
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths) == 0 else max(lengths)
        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq  # Copy each sequence into the matrix
        padded_seqs = padded_seqs.detach()  # Remove tensors from the computational graph
        return padded_seqs, lengths
    
    # Sort data by sequence lengths
    data.sort(key=lambda x: len(x['sentence']), reverse=True)
    new_item = {key: [d[key] for d in data] for key in data[0].keys()}
    
    # Merge sequences
    src_sent, _ = merge(new_item['sentence'])
    y_aspect, y_lengths = merge(new_item["aspects"])
    attention, _ = merge(new_item["attention"])
    token_type_id, _ = merge(new_item["token_type_id"])
    
    # Move tensors to the appropriate device
    src_sent = src_sent.to(device)
    y_aspect = y_aspect.to(device)
    y_lengths = torch.LongTensor(y_lengths).to(device)
    attention = attention.to(device)
    token_type_id = token_type_id.to(device)
    
    new_item["sentences"] = src_sent
    new_item["y_aspect"] = y_aspect
    new_item["slots_len"] = y_lengths
    new_item["attentions"] = attention
    new_item["token_type_ids"] = token_type_id

    return new_item
