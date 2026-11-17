from dataclasses import dataclass, field

import numpy as np


def runs(row):
    padded = np.concatenate([[0], row.astype(np.int8), [0]])
    diff = np.diff(padded)
    return list(zip(np.flatnonzero(diff == 1), np.flatnonzero(diff == -1)))


def significant_reports(inc, asd):
    out = np.zeros_like(inc)
    for d in range(inc.shape[0]):
        for a, b in runs(inc[d]):
            if asd[d, a:b].any():
                out[d, a:b] = 1
    return out


def prolonged_anomalies(asd, min_steps):
    out = np.zeros_like(asd)
    for d in range(asd.shape[0]):
        for a, b in runs(asd[d]):
            if b - a >= min_steps:
                out[d, a:b] = 1
    return out


@dataclass
class DenoiseResult:
    labels: np.ndarray
    significant: np.ndarray
    added: np.ndarray
    top_percent: float
    threshold: float
    removal: float
    addition: float
    history: list = field(default_factory=list)


def denoise(inc, sd, removal_max=0.6, addition_max=1.0, min_steps=3, top_percent=3.0, step=0.5, max_iter=400, min_step=0.01):
    inc = inc.astype(np.int8)
    history = []
    last_move = 0
    for _ in range(max_iter):
        threshold = np.percentile(sd, 100 - top_percent)
        asd = (sd >= threshold).astype(np.int8)
        sir = significant_reports(inc, asd)
        psa = prolonged_anomalies(asd, min_steps)
        added = ((psa == 1) & (sir == 0)).astype(np.int8)
        removal = 1 - sir.sum() / inc.sum()
        addition = added.sum() / inc.sum()
        history.append((top_percent, float(threshold), float(removal), float(addition)))
        if removal <= removal_max and addition <= addition_max:
            return DenoiseResult((sir | added).astype(np.int8), sir, added, top_percent, float(threshold), float(removal), float(addition), history)
        if removal > removal_max and addition <= addition_max:
            move = 1
        elif removal <= removal_max and addition > addition_max:
            move = -1
        else:
            break
        if last_move and move != last_move:
            step /= 2
        if step < min_step:
            break
        top_percent += move * step
        last_move = move
        if not 0 < top_percent < 50:
            break
    raise ValueError(f"no feasible top-n% for removal<={removal_max}, addition<={addition_max}; history={history[-5:]}")


def ahead_label(labels, steps):
    out = labels.copy()
    for d in range(labels.shape[0]):
        for a, _ in runs(labels[d]):
            out[d, max(a - steps, 0):a] = 1
    return out


def report_slowdown_overlap(inc, sd, top_percent):
    asd = sd >= np.percentile(sd, 100 - top_percent)
    both = (inc == 1) & asd
    connected = significant_reports(inc.astype(np.int8), asd.astype(np.int8))
    return dict(
        time_reported_pct=100 * inc.mean(),
        reports_in_top_slowdown_pct=100 * both.sum() / asd.sum(),
        report_time_connected_pct=100 * connected.sum() / inc.sum(),
    )
