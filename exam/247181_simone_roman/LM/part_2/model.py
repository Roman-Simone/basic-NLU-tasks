import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np

# RNN Elman version
# We are not going to use this since for efficiently purposes it's better to use the RNN layer provided by pytorch
DEVICE = 'cuda:0'
class RNN_cell(nn.Module):
    def __init__(self,  hidden_size, input_size, output_size, vocab_size, dropout=0.1):
        super(RNN_cell, self).__init__()

        self.W = nn.Linear(input_size, hidden_size, bias=False)
        self.U = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, vocab_size)
        self.vocab_size = vocab_size
        self.sigmoid = nn.Sigmoid()

    def forward(self, prev_hidden, word):
        input_emb = self.W(word)
        prev_hidden_rep = self.U(prev_hidden)
        # ht = σ(Wx + Uht-1 + b)
        hidden_state = self.sigmoid(input_emb + prev_hidden_rep)
        # yt = σ(Vht + b)
        output = self.output(hidden_state)
        return hidden_state, output
    


class LM_RNN(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        super(LM_RNN, self).__init__()
        # Token ids to vectors, we will better see this in the next lab
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # Pytorch's RNN layer: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        self.rnn = nn.RNN(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index
        # Linear layer to project the hidden layer to our output space
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        rnn_out, _  = self.rnn(emb)
        output = self.output(rnn_out).permute(0,2,1)
        return output

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

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        lstm_out, _  = self.lstm(emb)
        output = self.output(lstm_out).permute(0,2,1)
        return output



# class LockedDropout(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x, dropout=0.5):
#         if not self.training or not dropout:
#             return x
#         m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
#         mask = Variable(m, requires_grad=False) / (1 - dropout)
#         mask = mask.expand_as(x)
#         return mask * x



class VariationalDropout(nn.Module):
    def __init__(self, prob=0.5):
        super(VariationalDropout, self).__init__()
        self.prob = prob
        

    def forward(self, input):
        if not self.training:
            return input
        

        batch_size = input.shape[0]    #1=batch_size 2= lunghezza max sentence 3=embedding
        emb_size = input.shape[2]
        # dropout_mask = torch.bernoulli(torch.full((batch_size, emb_size)), 1 - self.prob)
        # dropout_mask = dropout_mask.unsqueeze(0).unsqueeze(2)
        benoulli = torch.distributions.bernoulli.Bernoulli(probs= 1 - self.prob)
        mask = benoulli.sample((batch_size,1,emb_size)).to(DEVICE)
        mask_expanded = mask.expand_as(input)
        #output
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

        self.emb_dropout = VariationalDropout(0.8)
        # Pytorch's LSTM layer
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index
        # Linear layer to project the hidden layer to our output space
        self.output = nn.Linear(hidden_size, output_size)

        self.out_dropout = VariationalDropout(0.5)
        # Weight tying
        self.output.weight = self.embedding.weight

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        drop_emb = self.emb_dropout(emb)
    
        lstm_out, _ = self.lstm(drop_emb)
        drop_lstm = self.out_dropout(lstm_out)

        output = self.output(drop_lstm).permute(0,2,1)
        return output
    
    