import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertConfig


import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ModelBert(nn.Module):

    def __init__(self, hid_size, out_slot, out_int):
        super(ModelBert, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
    
        self.slot_out = nn.Linear(hid_size, out_slot)
        self.intent_out = nn.Linear(hid_size, out_int)

        # Dropout layer How/Where do we apply it?
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, utterances):
        
        # Get the BERT output
        outputs = self.bert(utterances)

        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]
        
        # Compute slot logits
        slots = self.slot_out(sequence_output)
        # Compute intent logits
        intent = self.intent_out(pooled_output)
        
        # Slot size: batch_size, seq_len, classes 
        slots = slots.permute(0,2,1) # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len
        return slots, intent
    


