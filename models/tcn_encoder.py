import math
from typing import List

import torch
from torch import nn
import torch.nn.functional as F
import torch.fft as fft
from einops import reduce, rearrange, repeat

import numpy as np

from .dilated_conv import DilatedConvEncoder

from .tcn2 import TemporalConvNet


def generate_continuous_mask(B, T, n=5, l=0.1):
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)
    
    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)
    
    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T-l+1)
            res[i, t:t+l] = False
    return res


def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)




class BandedFourierLayer(nn.Module):
    def __init__(self, in_channels, out_channels, band, num_bands, length=201):
        super().__init__()

        self.length = length
        self.total_freqs = (self.length // 2) + 1

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.band = band  # zero indexed
        self.num_bands = num_bands

        self.num_freqs = self.total_freqs // self.num_bands + (self.total_freqs % self.num_bands if self.band == self.num_bands - 1 else 0)

        self.start = self.band * (self.total_freqs // self.num_bands)
        self.end = self.start + self.num_freqs


        # case: from other frequencies
        self.weight = nn.Parameter(torch.empty((self.num_freqs, in_channels, out_channels), dtype=torch.cfloat))
        self.bias = nn.Parameter(torch.empty((self.num_freqs, out_channels), dtype=torch.cfloat))
        self.reset_parameters()

    def forward(self, input):
        # input - b t d
        b, t, _ = input.shape
        input_fft = fft.rfft(input, dim=1)
        output_fft = torch.zeros(b, t // 2 + 1, self.out_channels, device=input.device, dtype=torch.cfloat)
        output_fft[:, self.start:self.end] = self._forward(input_fft)
        return fft.irfft(output_fft, n=input.size(1), dim=1)

    def _forward(self, input):
        output = torch.einsum('bti,tio->bto', input[:, self.start:self.end], self.weight)
        return output + self.bias

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)


class CoSTTCNEncoder(nn.Module):
    def __init__(self, input_dims, output_dims,
                 kernels: List[int],
                 length: int,
                 hidden_dims=64, depth=10,
                 mask_mode='binomial'):
        super().__init__()

        component_dims = output_dims // 2

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.component_dims = component_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)

        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3
        )

        self.repr_dropout = nn.Dropout(p=0.1)

        self.kernels = kernels

        # # #tfd tendencia
        # self.tfd = nn.ModuleList(
        #     [nn.Conv1d(output_dims, component_dims, k, padding=k-1) for k in kernels]
        # )

        # #breakpoint()
        # #tfd tendencia
        # self.tfd = nn.ModuleList(
        #     [TCN(hidden_dims, component_dims, [hidden_dims] * depth + [output_dims], k) for k in kernels] 
        # )

        # #breakpoint()
        # #tfd tendencia
        # self.tfd = nn.ModuleList(
        #     [TemporalConvNet(input_dims, [hidden_dims] * depth + [output_dims], kernel_size=k) for k in kernels]
        #     ) #output_dims


        # #tfd tendencia
        # self.tfd = nn.ModuleList(
        #     [TemporalConvNet(hidden_dims, [hidden_dims] * depth + [output_dims], k) for k in kernels] 
        # )


        # #breakpoint()
        # #tfd tendencia
        # self.tfd = nn.ModuleList(
        #     [TemporalConvNet(hidden_dims, [hidden_dims] * depth + [int(output_dims/2)], k) for k in kernels] #component_dims
        # )


        #tfd tendencia
        self.tfd = nn.ModuleList(
            [TemporalConvNet(hidden_dims, [hidden_dims] * depth + [component_dims], 3)] 
        ) 


        #sfd sazonalidade
        self.sfd = nn.ModuleList(
            [BandedFourierLayer(output_dims, component_dims, b, 1, length=length) for b in range(1)]
        )

    def forward(self, x, tcn_output=False, mask='all_true'):  # x: B x T x input_dims
        #breakpoint() #torch.Size([8, 3000, 14])
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
        
        x = x.transpose(1, 2)  # B x Ch x T torch.Size([8, 64, 3000])
        dados_tcn = x
        x = self.feature_extractor(x)  # B x Co x T  torch.Size([8, 320, 3000])

        if tcn_output:
            return x.transpose(1, 2)

        #breakpoint()
        trend = []
        for idx, mod in enumerate(self.tfd): ######RuntimeError: Given groups=1, weight of size [64, 14, 1], expected input[8, 320, 3000] to have 14 channels, but got 320 channels instead
            #out = mod(x)  # b d t torch.Size([8, 160, 3000])
            out = mod(dados_tcn)
            # if self.kernels[idx] != 1:
            #     out = out[..., :-(self.kernels[idx] - 1)]
            #print(out.size())
            trend.append(out.transpose(1, 2))  # b t d
            #trend.append(out) #mz
        #breakpoint()
        trend = reduce(
            rearrange(trend, 'list b t d -> list b t d'),
            'list b t d -> b t d', 'mean'
        )

        # b: representa a dimensão do lote (batch size). É o número de exemplos de treinamento ou amostras processadas em paralelo.
        # t: representa a dimensão temporal. É a dimensão que indica o tempo ou a sequência ao longo da qual os dados estão organizados.
        # d: representa a dimensão dos recursos (features). É a dimensão que codifica as características ou atributos dos dados.

        x = x.transpose(1, 2)  # B x T x Co

        season = []
        for mod in self.sfd:
            out = mod(x)  # b t d
            season.append(out)
        season = season[0]

        return trend, self.repr_dropout(season)





# from torch-tcn import TemporalConvNet

# self.feature_extractor = TemporalConvNet(
#     num_inputs=hidden_dims,
#     num_channels=[hidden_dims] * depth + [output_dims],
#     kernel_size=3
# )

# class TemporalConvNet(nn.Module):
#     def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0.2):
#         super(TemporalConvNet, self).__init__()
#         import torch
#         import torch.nn as nn

#         layers = []
#         num_levels = len(num_channels)
#         for i in range(num_levels):
#             dilation_size = 2 ** i
#             in_channels = input_size if i == 0 else num_channels[i-1]
#             out_channels = num_channels[i]
#             layers += [nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation_size),
#                        nn.ReLU(),
#                        nn.Dropout(dropout)]

#         self.network = nn.Sequential(*layers)
#         self.fc = nn.Linear(num_channels[-1], output_size)

#     def forward(self, x):
#         x = self.network(x)
#         x = x.mean(dim=2)
#         x = self.fc(x)
#         return x



# class TemporalBlock(nn.Module):
#     def __init__(self, input_size, output_size, kernel_size, dilation=1, padding=1, dropout=0.2):
#         super(TemporalBlock, self).__init__()
#         self.conv1 = nn.Conv1d(input_size, output_size, kernel_size, dilation=dilation, padding=padding)
#         self.relu1 = nn.ReLU()
#         self.dropout1 = nn.Dropout(dropout)
#         self.conv2 = nn.Conv1d(output_size, output_size, kernel_size, dilation=dilation, padding=padding)
#         self.relu2 = nn.ReLU()
#         self.dropout2 = nn.Dropout(dropout)
#         self.downsample = nn.Conv1d(input_size, output_size, kernel_size=1) if input_size != output_size else None
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
        
#         residual = x
#         out = self.conv1(x)
#         out = self.relu1(out)
#         out = self.dropout1(out)
#         out = self.conv2(out)
#         out = self.relu2(out)
#         out = self.dropout2(out)
#         if self.downsample is not None:
#             residual = self.downsample(x)
#         out += residual
#         out = self.relu(out)
#         out = self.dropout(out)
#         return out

# class TCN(nn.Module):
#     def __init__(self, input_size, output_size, num_channels, kernel_size, dropout = 0.2):
#         super(TCN, self).__init__()
#         layers = []
#         num_levels = len(num_channels)
#         for i in range(num_levels):
#             dilation_size = 2 ** i
#             in_channels = input_size if i == 0 else num_channels[i - 1]
#             out_channels = num_channels[i]
#             layers.append(TemporalBlock(in_channels, out_channels, kernel_size, dilation=dilation_size, padding=(kernel_size-1) * dilation_size, dropout=dropout))
#         self.layers = nn.Sequential(*layers)
#         self.pooling = nn.AdaptiveAvgPool1d(1)
#         self.linear = nn.Linear(num_channels[-1], output_size)

#     def forward(self, x):
#         out = self.layers(x)
#         out = self.pooling(out)
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return out





# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class TemporalBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, dilation):
#         super(TemporalBlock, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=(kernel_size - 1) * dilation, dilation=dilation)
#         self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=(kernel_size - 1) * dilation, dilation=dilation)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.2)

#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.relu(out)
#         out = self.dropout(out)
#         out = self.conv2(out)
#         out = self.relu(out)
#         out = self.dropout(out)
#         return out

# class TemporalConvNet(nn.Module):
#     def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
#         super(TemporalConvNet, self).__init__()
#         layers = []
#         num_levels = len(num_channels)
#         for i in range(num_levels):
#             dilation_size = 2 ** i
#             in_channels = num_inputs if i == 0 else num_channels[i-1]
#             out_channels = num_channels[i]
#             layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size))
#         self.network = nn.Sequential(*layers)
#         self.fc = nn.Linear(num_channels[-1], 1)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         out = x.transpose(1, 2)  # Convert to (batch_size, num_inputs, sequence_length)
#         out = self.network(out)
#         out = out.transpose(1, 2)  # Convert back to (batch_size, sequence_length, num_channels[-1])
#         out = self.dropout(out)
#         out = self.fc(out[:, -1, :])  # Use only the last time step for prediction
#         return out