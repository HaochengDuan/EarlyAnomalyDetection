import lightgbm as lgb
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn


def oversample(x, y, seed):
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    small, big = (pos, neg) if len(pos) < len(neg) else (neg, pos)
    idx = np.concatenate([big, small, rng.choice(small, len(big) - len(small), replace=True)])
    return x[idx], y[idx]


def undersample(x, y, seed, ratio=1.0):
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    neg = rng.choice(neg, min(len(neg), int(ratio * len(pos))), replace=False)
    idx = np.concatenate([pos, neg])
    return x[idx], y[idx]


def rf(xtr, ytr, xte, seed):
    xr, yr = oversample(xtr, ytr, seed)
    return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=8).fit(xr, yr).predict(xte)


def svm(xtr, ytr, xte, seed=0, downsample=True):
    if downsample:
        xtr, ytr = undersample(xtr, ytr, seed)
    scaler = StandardScaler().fit(xtr)
    return SVC(kernel="rbf", class_weight="balanced", cache_size=4000).fit(scaler.transform(xtr), ytr).predict(scaler.transform(xte))


def lightgbm(xtr, ytr, xte, seed):
    return lgb.LGBMClassifier(class_weight="balanced", random_state=seed, verbose=-1).fit(xtr, ytr).predict(xte)


class Generator(nn.Sequential):
    def __init__(self, nz, dim):
        super().__init__(nn.Linear(nz, 1024), nn.ReLU(True), nn.Linear(1024, 512), nn.ReLU(True),
                         nn.Linear(512, 256), nn.ReLU(True), nn.Linear(256, dim), nn.Tanh())


class Discriminator(nn.Sequential):
    def __init__(self, dim, dropout):
        super().__init__(nn.Linear(dim, 512), nn.LeakyReLU(0.2, True), nn.BatchNorm1d(512), nn.Dropout(dropout),
                         nn.Linear(512, 64), nn.LeakyReLU(0.2, True), nn.BatchNorm1d(64), nn.Dropout(dropout),
                         nn.Linear(64, 1), nn.Sigmoid())


def gan_samples(x_pos, seed, n=None, nz=100, epochs=200, batch_size=64, lr=2.5e-5, betas=(0.9, 0.99), dropout=0.2, device="cuda", n_val=0):
    torch.manual_seed(seed)
    dim = x_pos.shape[1]
    g, d = Generator(nz, dim).to(device), Discriminator(dim, dropout).to(device)
    og, od = torch.optim.Adam(g.parameters(), lr, betas=betas), torch.optim.Adam(d.parameters(), lr, betas=betas)
    bce = nn.BCELoss()
    real_all = torch.tensor(x_pos, dtype=torch.float32, device=device)
    fixed = torch.randn(n or len(x_pos), nz, device=device)
    best, out = float("inf"), None
    for _ in range(epochs):
        g.train()
        d.train()
        for idx in torch.randperm(len(real_all), device=device).split(batch_size):
            if len(idx) < 2:
                continue
            real = real_all[idx]
            ones, zeros = torch.ones(len(idx), 1, device=device), torch.zeros(len(idx), 1, device=device)
            fake = g(torch.randn(len(idx), nz, device=device))
            loss_d = bce(d(real), ones) + bce(d(fake.detach()), zeros)
            od.zero_grad()
            loss_d.backward()
            od.step()
            loss_g = bce(d(fake), ones)
            og.zero_grad()
            loss_g.backward()
            og.step()
        if n_val:
            g.eval()
            d.eval()
            with torch.no_grad():
                val = -torch.log(d(g(torch.randn(n_val, nz, device=device))).clamp_min(1e-12)).mean().item()
                if val < best:
                    best, out = val, g(fixed).cpu().numpy()
    if out is None:
        g.eval()
        with torch.no_grad():
            out = g(fixed).cpu().numpy()
    return out


def gan_svm(xtr, ytr, xte, seed, device="cuda", n_val=0):
    fake = gan_samples(xtr[ytr == 1], seed, device=device, n_val=n_val)
    x = np.concatenate([xtr, fake])
    y = np.concatenate([ytr, np.ones(len(fake), dtype=ytr.dtype)])
    return svm(x, y, xte, seed)


class MLP(nn.Sequential):
    def __init__(self, dims):
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.LeakyReLU(0.2)]
        super().__init__(*layers[:-1])


class TSSAE(nn.Module):
    def __init__(self, link_cols):
        super().__init__()
        self.link_cols = link_cols
        k = len(link_cols[0])
        self.enc = nn.ModuleList(MLP([k, 24, 12]) for _ in link_cols)
        self.dec = nn.ModuleList(MLP([12, 24, k]) for _ in link_cols)
        self.head = MLP([12 * len(link_cols), 64, 1])

    def codes(self, x):
        return [e(x[:, c]) for e, c in zip(self.enc, self.link_cols)]

    def forward(self, x):
        return self.head(torch.cat(self.codes(x), 1)).squeeze(1)


def tssae(x, y, link_cols, seed, device, epochs_pre=15, epochs_ft=15):
    torch.manual_seed(seed)
    m = TSSAE(link_cols).to(device)
    xt, yt = torch.tensor(x, dtype=torch.float32, device=device), torch.tensor(y, dtype=torch.float32, device=device)
    opt = torch.optim.Adam(list(m.enc.parameters()) + list(m.dec.parameters()), 1e-3)
    for _ in range(epochs_pre):
        for idx in torch.randperm(len(xt), device=device).split(512):
            loss = sum(((dec(c) - xt[idx][:, cols]) ** 2).mean() for dec, c, cols in zip(m.dec, m.codes(xt[idx]), link_cols))
            opt.zero_grad()
            loss.backward()
            opt.step()
    opt = torch.optim.Adam(m.parameters(), 3e-4)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(epochs_ft):
        for idx in torch.randperm(len(xt), device=device).split(512):
            loss = lossf(m(xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    m.eval()
    return lambda z: m(torch.tensor(z, dtype=torch.float32, device=device)).detach().cpu().numpy()


class Seq2SeqAttn(nn.Module):
    def __init__(self, d_in, h=64):
        super().__init__()
        self.enc = nn.GRU(d_in, h, batch_first=True)
        self.att = nn.Linear(2 * h, 1)
        self.out = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x):
        o, hn = self.enc(x)
        q = hn[-1].unsqueeze(1).expand_as(o)
        w = torch.softmax(self.att(torch.cat([o, q], -1)), 1)
        return self.out(torch.cat([(w * o).sum(1), hn[-1]], -1)).squeeze(1)


def chebyshev_distance(speed, train_days, seq):
    prof = np.full_like(speed, np.nan)
    for d in range(speed.shape[0]):
        past = [d - 5 * k for k in range(1, 5) if d - 5 * k >= 0]
        if past:
            prof[d] = speed[past].mean(0)
    prof = np.where(np.isnan(prof), np.nanmean(speed[train_days], 0, keepdims=True), prof)
    diff = np.abs(speed - prof)
    return np.stack([diff[:, max(s - seq + 1, 0):s + 1].max(1) for s in range(speed.shape[1])], 1)


def cfod_state(znorm, deriv, days, h):
    out = np.zeros((len(days), znorm.shape[1]), dtype=np.int8)
    for i, d in enumerate(days):
        on = False
        for s in range(1, znorm.shape[1]):
            if not on and deriv[d, s] >= h:
                on = True
            elif on and (deriv[d, s] <= -h or znorm[d, s] <= 0):
                on = False
            out[i, s] = on
    return out
