import torch
import torch.nn as nn

# Device
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

#PART 1.1
class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        super(LM_LSTM, self).__init__()
        # Token ids to vectors, we will better see this in the next lab
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # Pytorch's LSTM layer: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index
        # Linear layer to project the hidden layer to our output space
        self.output = nn.Linear(hidden_size, output_size)
        # Weight tying
        self.output.weight = self.embedding.weight

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        lstm_out, _  = self.lstm(emb)
        output = self.output(lstm_out).permute(0,2,1)
        return output


class VariationalDropout(nn.Module):
    def __init__(self, prob=0.5):
        super(VariationalDropout, self).__init__()
        self.prob = prob

    def forward(self, input):
        if not self.training:
            return input
        
        batch_size = input.shape[0]    #1=batch_size 2= lunghezza max sentence 3=embedding
        emb_size = input.shape[2]

        # create a mask with Bernoulli distribution
        benoulli = torch.distributions.bernoulli.Bernoulli(probs= 1 - self.prob)
        
        # sample the mask
        maskk = benoulli.sample((batch_size , 1, emb_size)).to(DEVICE)

        # expand the mask to the same shape as the input
        mask_expanded = maskk.expand_as(input)
        
        # scale the input
        output = input * mask_expanded / (1 - self.prob)
        output.to(DEVICE)
        
        return output


#PART 1.2
class LM_LSTM_DROP(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        super(LM_LSTM_DROP, self).__init__()
        # Token ids to vectors
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)

        self.emb_dropout = VariationalDropout(0.7)
        # Pytorch's LSTM layer
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.lstm.flatten_parameters()
        self.pad_token = pad_index
        # Linear layer to project the hidden layer to our output space
        self.output = nn.Linear(hidden_size, output_size)

        self.out_dropout = VariationalDropout(0.7)
        # Weight tying
        self.output.weight = self.embedding.weight

    def forward(self, input_sequence):
        self.lstm.flatten_parameters()
        
        emb = self.embedding(input_sequence)
        drop_emb = self.emb_dropout(emb)
        lstm_out, _ = self.lstm(drop_emb)
        drop_lstm = self.out_dropout(lstm_out)

        output = self.output(drop_lstm).permute(0,2,1)

        return output
    
    