import torch
from torch import nn
import numpy as np
from torch.nn import init

class GRU(nn.Module):
    def __init__(self,in_dim=6, hid_dim=10, layers=20, out_state = 2):
        super(GRU,self).__init__()
        self.rnn = nn.GRU(
            input_size = in_dim,
            hidden_size = hid_dim,
            num_layers = layers,
            batch_first = True,
        )
        self.out = nn.Linear(hid_dim, out_state)
        self.relu = nn.ReLU()
        self.sgmd = nn.Sigmoid()

    def forward(self,input):
        rout, _ = self.rnn(input, None)
        # print(rout.shape)
        dout = self.out(rout[:,-1,:])
        # print("success")
        r_out = self.relu(dout)
        out = self.sgmd(r_out)
        return out
