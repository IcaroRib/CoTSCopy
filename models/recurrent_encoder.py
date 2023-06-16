import math
from typing import List

import torch
from torch import nn
import torch.nn.functional as F
import torch.fft as fft
from einops import reduce, rearrange, repeat

import numpy as np

from .encoder import BandedFourierLayer, generate_binomial_mask, generate_continuous_mask


class DilatedRecurrentEncoder(nn.Module):
    def __init__(self, input_size, output_size, architecture):
        super().__init__()
        self.net = architecture(input_size, output_size, num_layers=2)

    def forward(self, x):
        out, states = self.net(x)
        return out


class CoSTRecurrentEncoder(nn.Module):
    ARCHITECTURES = {
        "rnn": nn.RNN,
        "lstm": nn.LSTM,
        "gru": nn.GRU
    }

    def __init__(self, input_dims, output_dims,
                 kernels: List[int],
                 length: int,
                 hidden_dims=64, depth=10,
                 mask_mode='binomial',
                 architecture="rnn"):
        super().__init__()

        component_dims = output_dims // 2

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.component_dims = component_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.architecture = CoSTRecurrentEncoder.ARCHITECTURES[architecture]

        self.feature_extractor = DilatedRecurrentEncoder(hidden_dims, output_dims, self.architecture)

        self.repr_dropout = nn.Dropout(p=0.1)

        self.kernels = kernels

        self.tfd = nn.ModuleList(
            [self.architecture(output_dims, component_dims, 1, False) for k in kernels]
        )

        self.sfd = nn.ModuleList(
            [BandedFourierLayer(output_dims, component_dims, b, 1, length=length) for b in range(1)]
        )

    def forward(self, x, tcn_output=False, mask='all_true'):  # x: B x T x input_dims
        #print(x.shape) #B x T x input_dims
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x = self.input_fc(x)  # B x T x Ch

        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'

        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False

        mask &= nan_mask
        x[~mask] = 0

        # conv encoder
        # x = x.transpose(1, 2)  # B x Ch x T
        #print('X before feature_extractor')
        #print(x.shape)
        x = self.feature_extractor(x)  # B x Co x T

        if tcn_output:
            return x.transpose(1, 2)

        trend = []
        #print('X before LSTM')
        #print(x.shape)
        for mod in self.tfd:
            out, _ = mod(x)  # b t d
            #print(f"Out shape: {out.shape}")
            trend.append(out)
        trend = reduce(
            rearrange(trend, 'list b t d -> list b t d'),
            'list b t d -> b t d', 'mean'
        )

        season = []
        #print(f"X shape before season desintangler: {x.shape}" )
        for mod in self.sfd:
            out = mod(x)  # b t d
            season.append(out)
        season = season[0]

        print("trend shape = ")
        print(trend.shape)
        return trend, self.repr_dropout(season)
