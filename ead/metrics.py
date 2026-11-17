import numpy as np

from .labels import runs
from .timegrid import SLOT_MINUTES, SLOT_START_MINUTE

THRESHOLDS = np.round(np.arange(1, 1000) * 0.001, 3)
REPORT_WINDOW = (6 * 60, 20 * 60 + 25)


def prf(prob, y, tau):
    pred = prob >= tau
    tp = np.sum(pred & (y == 1))
    fp = np.sum(pred & (y == 0))
    fn = np.sum(~pred & (y == 1))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(precision), float(recall), float(f1)


def best_threshold(prob, y, grid=THRESHOLDS):
    p, t = np.ravel(prob), np.ravel(y) == 1
    pos, neg = np.sort(p[t]), np.sort(p[~t])
    tp = len(pos) - np.searchsorted(pos, grid, side="left")
    fp = len(neg) - np.searchsorted(neg, grid, side="left")
    f1 = 2 * tp / np.maximum(2 * tp + fp + (len(pos) - tp), 1)
    k = int(np.argmax(f1))
    return float(grid[k]), float(f1[k])


def per_horizon(prob, y, tau):
    return [dict(zip(("precision", "recall", "f1"), prf(prob[:, h], y[:, h], tau))) for h in range(prob.shape[1])]


def alert_matrix(prob, tau, day, slot, days, rule="min"):
    score = {"min": prob.min(axis=1), "max": prob.max(axis=1), "mean": prob.mean(axis=1)}[rule]
    first = int(slot.min())
    out = np.zeros((len(days), int(slot.max()) - first + 1), dtype=np.int8)
    row = {d: i for i, d in enumerate(days)}
    on = score >= tau
    out[[row[d] for d in day[on]], slot[on] - first] = 1
    return out, first


def alert_periods(alerts_row, first_slot):
    return [(SLOT_START_MINUTE + SLOT_MINUTES * (first_slot + a), SLOT_START_MINUTE + SLOT_MINUTES * (first_slot + b))
            for a, b in runs(alerts_row)]


def label_periods(label_row):
    return [(SLOT_START_MINUTE + SLOT_MINUTES * (a - 1), SLOT_START_MINUTE + SLOT_MINUTES * (b - 1)) for a, b in runs(label_row)]


def report_periods(reports, date):
    day = reports[(reports.start_time.dt.normalize() == date) | (reports.end_time.dt.normalize() == date)]
    periods = []
    for r in day.itertuples(index=False):
        s = max((r.start_time - date).total_seconds() / 60, REPORT_WINDOW[0])
        e = min((r.end_time - date).total_seconds() / 60, REPORT_WINDOW[1])
        if s < e:
            periods.append((s, e))
    periods.sort()
    merged = []
    for s, e in periods:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _overlap(a, b):
    return a[0] < b[1] and a[1] > b[0]


def _detection(reports, alerts):
    ttd = []
    for r in reports:
        hits = [a[0] - r[0] for a in alerts if _overlap(a, r)]
        if hits:
            ttd.append(min(hits))
    return len(reports), ttd


def event_metrics(alerts, first_slot, dates, reports, anomaly):
    n_rep = n_sig = n_alert = n_false = 0
    ttd, ttd_sig = [], []
    for i, date in enumerate(dates):
        al = alert_periods(alerts[i], first_slot)
        rp = report_periods(reports, date)
        an = label_periods(anomaly[i])
        sig = [r for r in rp if any(_overlap(r, a) for a in an)]
        n, t = _detection(rp, al)
        n_rep += n
        ttd += t
        n, t = _detection(sig, al)
        n_sig += n
        ttd_sig += t
        n_alert += len(al)
        n_false += sum(not any(_overlap(a, x) for x in rp + an) for a in al)
    return dict(
        DR=len(ttd) / n_rep if n_rep else 0.0,
        MTTD=float(np.mean(ttd)) if ttd else float("nan"),
        FAR=n_false / n_alert if n_alert else 0.0,
        DR_S=len(ttd_sig) / n_sig if n_sig else 0.0,
        MTTD_S=float(np.mean(ttd_sig)) if ttd_sig else float("nan"),
        n_reports=n_rep, n_significant_reports=n_sig, n_alerts=n_alert, n_false_alerts=n_false,
    )
