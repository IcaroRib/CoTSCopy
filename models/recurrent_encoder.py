import torch
from torch import nn


class DilatedRecurrentEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.LSTM(64, 320, num_layers=2)

    def forward(self, x):
        out, states = self.net(x)
        return out