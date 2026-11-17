import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)]


class AnomalyTransformer(nn.Module):
    def __init__(self, dim_in, horizon=6, d_model=64, nhead=8, enc_layers=2, dec_layers=2, dim_ff=256, dropout=0.2,
                 positional_encoding=True):
        super().__init__()
        self.horizon = horizon
        self.enc_in = nn.Linear(dim_in, d_model)
        self.dec_in = nn.Linear(1, d_model)
        self.start = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = PositionalEncoding(d_model) if positional_encoding else nn.Identity()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True), enc_layers)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True), dec_layers)
        self.out = nn.Linear(d_model, 1)
        mask = torch.triu(torch.full((horizon, horizon), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", mask)

    def encode(self, x):
        return self.encoder(self.drop(self.pos(self.enc_in(x))))

    def forward(self, x, y=None, teacher_forcing=0.0):
        memory = self.encode(x)
        tokens = [self.start.expand(x.size(0), 1, -1)]
        logits = []
        for k in range(self.horizon):
            h = self.drop(self.pos(torch.cat(tokens, dim=1)))
            step = self.decoder(h, memory, tgt_mask=self.causal_mask[: k + 1, : k + 1])
            logit = self.out(step[:, -1])
            logits.append(logit)
            if k + 1 < self.horizon:
                if y is not None and self.training and torch.rand(()) < teacher_forcing:
                    prev = y[:, k : k + 1]
                else:
                    prev = torch.sigmoid(logit)
                tokens.append(self.dec_in(prev).unsqueeze(1))
        return torch.cat(logits, dim=1)
