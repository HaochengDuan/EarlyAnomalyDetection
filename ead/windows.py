from dataclasses import dataclass

import numpy as np
import torch

FEATURES = ("speed", "slowdown", "tti", "srs", "report")


def split_days(n_days, train_ratio=0.7, val_ratio=0.2):
    n_train = int(np.ceil(train_ratio * n_days))
    n_val = int(np.ceil(val_ratio * n_days))
    days = np.arange(n_days)
    train, val, test = days[:n_train], days[n_train:n_train + n_val], days[n_train + n_val:]
    half = len(val) // 2
    return dict(train=train, tune=val[:half], val=val[half:], test=test)


def input_tensor(feats, link_idx, train_days, names=FEATURES):
    blocks = []
    for name in names:
        x = feats[name][..., link_idx].astype(np.float32)
        if name != "report":
            lo = x[train_days].min(axis=(0, 1), keepdims=True)
            hi = x[train_days].max(axis=(0, 1), keepdims=True)
            x = (x - lo) / np.maximum(hi - lo, 1e-6)
        blocks.append(x)
    return np.concatenate(blocks, axis=-1)


@dataclass
class Windows:
    x: torch.Tensor
    y: torch.Tensor
    day: np.ndarray
    slot: np.ndarray

    def __len__(self):
        return len(self.y)


def make_windows(inputs, target, days, seq_in, horizon, device):
    n_slots = inputs.shape[1]
    slots = np.arange(seq_in - 1, n_slots - horizon)
    day = np.repeat(days, len(slots))
    slot = np.tile(slots, len(days))
    x_idx = slot[:, None] + np.arange(-seq_in + 1, 1)[None, :]
    y_idx = slot[:, None] + np.arange(1, horizon + 1)[None, :]
    x = inputs[day[:, None], x_idx]
    y = target[day[:, None], y_idx]
    return Windows(torch.from_numpy(x).to(device), torch.from_numpy(y.astype(np.float32)).to(device), day, slot)
