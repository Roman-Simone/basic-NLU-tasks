import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertConfig

class ModelBert(nn.Module):

    def __init__(self, hid_size, out_slot, out_int, dropout_rate=0.1):
        super(ModelBert, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        self.dropout = nn.Dropout(dropout_rate)  # Dropout layer with specified dropout rate
        
        self.slot_out = nn.Linear(hid_size, out_slot)
        self.intent_out = nn.Linear(hid_size, out_int)
        
    def forward(self, utterances, attentions=None, token_type_ids=None):
        
        # Get the BERT output
        outputs = self.bert(utterances, attention_mask=attentions, token_type_ids=token_type_ids)

        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        # Apply dropout
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)
        
        # Compute slot logits
        slots = self.slot_out(sequence_output)
        # Compute intent logits
        intent = self.intent_out(pooled_output)
        
        # Slot size: batch_size, seq_len, classes 
        slots = slots.permute(0, 2, 1)  # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len
        
        return slots, intent