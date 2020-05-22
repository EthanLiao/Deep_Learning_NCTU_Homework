import torch
import torch.nn as nn
import numpy as np
from torch.nn import init

# ,seqAmt=2, featureNum=5,
class LSTM(nn.Module):
    def __init__(self,in_dim=6, hid_dim=10, layers=20, out_state = 2):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(
            input_size = in_dim,
            hidden_size = hid_dim,
            num_layers = layers,
            batch_first = True,
        )
        self.relu = nn.ReLU()
        self.out = nn.Linear(hid_dim, out_state)
        self.sgmd = nn.Sigmoid()
        self.hid_dim = hid_dim

    def forward(self,input):
        rout, (h_n, h_c) = self.rnn(input, None)
        re_out = self.relu(rout)
        d_out = self.out(re_out[:,-1,:]) # extract the last output
        out = self.sgmd(d_out)
        return out

    def weightInit(self, gain=1.0):
        for name, param in self.named_parameters():
            if 'rnn.weight' in name:
                init.orthogonal_(param, gain/self.hid_dim)
