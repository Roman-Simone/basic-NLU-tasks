import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertConfig

class ModelBert(nn.Module):

    def __init__(self, hid_size, out_slot, out_int):
        super(ModelBert, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # hid_size = self.bert.config.hidden_size

        self.slot_out = nn.Linear(hid_size, out_slot)
        self.intent_out = nn.Linear(hid_size, out_int)

    def forward(self, input_ids):

        outputs = self.bert(input_ids)
        
        
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        intent_logits = self.intent_out(pooled_output)
        slot_logits = self.slot_out(sequence_output)
        return slot_logits, intent_logits

