import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertConfig


class ModelBert(nn.Module):

    def __init__(self, hid_size, out_slot):
        super(ModelBert, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
    
        self.slot_out = nn.Linear(hid_size, out_slot)
        
    def forward(self, utterances, attentions=None, token_type_ids=None):
        
        # Get the BERT output
        outputs = self.bert(utterances, attention_mask=attentions, token_type_ids=token_type_ids)

        sequence_output = outputs[0]
        
        # Compute slot logits
        slots = self.slot_out(sequence_output)
        
        # Slot size: batch_size, seq_len, classes 
        slots = slots.permute(0,2,1) # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len
        return slots
    


