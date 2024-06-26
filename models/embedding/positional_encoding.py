import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, d_embed, max_len=256, dr_rate=0):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dr_rate)
        encoding = torch.zeros(max_len, d_embed)
        encoding.requires_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embed, 2) * (-math.log(10000.0) / d_embed))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = encoding.unsqueeze(0)
        self.register_buffer('pe', self.encoding)

    def forward(self, x):
        _, seq_len, _ = x.size()
        pos_embed = self.encoding[:, :seq_len, :]
        out = x + pos_embed
        out = self.dropout(out)
        return out
