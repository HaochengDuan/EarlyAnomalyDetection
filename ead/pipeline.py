from dataclasses import dataclass

import numpy as np

from . import dataset, features, labels, windows
from .metrics import alert_matrix, event_metrics, per_horizon


@dataclass
class LabelConfig:
    top_percent: float = 5.0
    removal_max: float = 0.6
    addition_max: float = 1.6
    min_steps: int = 3
    ahead_steps: int = 3


@dataclass
class Prepared:
    data: dataset.LinkData
    link: str
    subgraph: list
    inputs: np.ndarray
    raw: np.ndarray
    denoised: labels.DenoiseResult
    ahead: np.ndarray
    split: dict

    def target(self, kind):
        return {"raw": self.raw, "denoised": self.denoised.labels, "ahead": self.ahead}[kind]

    def windows(self, kind, part, seq_in, horizon, device):
        return windows.make_windows(self.inputs, self.target(kind), self.split[part], seq_in, horizon, device)

    def target_reports(self):
        return self.data.reports[self.data.reports.link_id == self.link]


def prepare(data_dir, link, label_cfg, upstream_mi=2.0, downstream_mi=1.0, slowdown_mi=0.3, feature_names=windows.FEATURES):
    data = dataset.load(data_dir)
    feats = features.compute(data, slowdown_mi)
    i = data.corridor.index(link)
    raw = feats["report"][..., i].astype(np.int8)
    den = labels.denoise(raw, feats["slowdown"][..., i], label_cfg.removal_max, label_cfg.addition_max,
                         label_cfg.min_steps, label_cfg.top_percent)
    ahead = labels.ahead_label(den.labels, label_cfg.ahead_steps)
    sub = data.corridor.subgraph(link, upstream_mi, downstream_mi)
    split = windows.split_days(len(data.dates))
    inputs = windows.input_tensor(feats, [data.corridor.index(l) for l in sub], split["train"], feature_names)
    return Prepared(data, link, sub, inputs, raw, den, ahead, split)


def evaluate(prep, prob, test_windows, tau, rule="min"):
    y = test_windows.y.cpu().numpy()
    days = prep.split["test"]
    alerts, first = alert_matrix(prob, tau, test_windows.day, test_windows.slot, days, rule)
    events = event_metrics(alerts, first, prep.data.dates[days], prep.target_reports(), prep.denoised.labels[days])
    return dict(threshold=tau, horizon=per_horizon(prob, y, tau), events=events)
