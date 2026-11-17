import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .metrics import best_threshold, prf
from .model import AnomalyTransformer


@dataclass
class TrainConfig:
    d_model: int = 64
    nhead: int = 8
    enc_layers: int = 2
    dec_layers: int = 2
    dim_ff: int = 256
    dropout: float = 0.2
    positional_encoding: bool = True
    lr: float = 2.5e-4
    weight_decay: float = 0.01
    batch_size: int = 256
    epochs: int = 40
    teacher_forcing: float = 0.5
    seed: int = 1


@torch.no_grad()
def predict(model, windows, batch_size=8192):
    model.eval()
    out = [torch.sigmoid(model(windows.x[i:i + batch_size])) for i in range(0, len(windows), batch_size)]
    return torch.cat(out).cpu().numpy()


def fit(train, tune, val, cfg, log=print):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = train.x.device
    model = AnomalyTransformer(train.x.shape[-1], train.y.shape[1], cfg.d_model, cfg.nhead, cfg.enc_layers,
                               cfg.dec_layers, cfg.dim_ff, cfg.dropout, cfg.positional_encoding).to(device)
    pos = train.y.sum().item()
    pos_weight = torch.tensor((train.y.numel() - pos) / max(pos, 1.0), device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
    y_tune, y_val = tune.y.cpu().numpy(), val.y.cpu().numpy()

    history, best = [], dict(val_f1=-1.0)
    for epoch in range(cfg.epochs):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(train), generator=gen).to(device)
        total = 0.0
        for i in range(0, len(train), cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            logits = model(train.x[idx], train.y[idx], cfg.teacher_forcing)
            loss = loss_fn(logits, train.y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        tau, tune_f1 = best_threshold(predict(model, tune), y_tune)
        val_f1 = prf(predict(model, val), y_val, tau)[2]
        rec = dict(epoch=epoch, train_loss=total / len(train), tune_f1=tune_f1, val_f1=val_f1, threshold=tau,
                   seconds=time.time() - t0)
        history.append(rec)
        if val_f1 > best["val_f1"]:
            best = dict(val_f1=val_f1, threshold=tau, epoch=epoch, state=copy.deepcopy(model.state_dict()))
        log(f"epoch {epoch:3d}  loss {rec['train_loss']:.4f}  tune F1 {tune_f1:.3f}  val F1 {val_f1:.3f}  tau {tau:.2f}  {rec['seconds']:.1f}s")
    model.load_state_dict(best["state"])
    return model, best, history, float(pos_weight)
