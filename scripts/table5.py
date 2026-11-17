import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ead import baselines as B
from ead import features
from ead.metrics import alert_matrix, event_metrics, prf
from ead.pipeline import LabelConfig, prepare
from ead.windows import make_windows

SEQ = 7


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/synthetic_link")
    p.add_argument("--link", default="SYN-T0")
    p.add_argument("--features", default="speed,slowdown,tti,srs")
    p.add_argument("--runs", default="runs")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="results")
    a = p.parse_args()
    seed, dev = a.seed, a.device

    prep = prepare(a.data, a.link, LabelConfig(), feature_names=tuple(a.features.split(",")))
    speed = features.compute(prep.data)["speed"][..., prep.data.corridor.index(a.link)]
    split, raw = prep.split, prep.raw
    test = split["test"]
    dates, reports, anomaly = prep.data.dates[test], prep.target_reports(), prep.denoised.labels[test]
    last_slot = prep.inputs.shape[1] - SEQ

    def windows(part, target, device="cpu"):
        return make_windows(prep.inputs, target, split[part], SEQ, 1, device)

    def flat(part, target):
        w = windows(part, target)
        return w.x.reshape(len(w), -1).numpy(), w.y.numpy()[:, 0].astype(int), w.day, w.slot

    def events(flags, day, slot, shift=0):
        keep = slot + shift <= last_slot
        alerts, first = alert_matrix(flags[keep, None].astype(float), 0.5, day[keep], slot[keep] + shift, test)
        return event_metrics(alerts, first, dates, reports, anomaly)

    onset = ((raw == 1) & (np.concatenate([np.zeros_like(raw[:, :1]), raw[:, :-1]], axis=1) == 0)).astype(raw.dtype)
    xtr, ytr, _, _ = flat("train", raw)
    xtu, ytu, _, _ = flat("tune", raw)
    xte, _, dte, ste = flat("test", raw)
    _, otr, _, _ = flat("train", onset)
    _, otu, _, _ = flat("tune", onset)
    ktr, ktu = (otr == 1) | (ytr == 0), (otu == 1) | (ytu == 0)
    xtr, ytr, xtu, ytu = xtr[ktr], otr[ktr], xtu[ktu], otu[ktu]
    n_val = int(ytu.sum())
    results = {}

    n_links = len(prep.subgraph)
    n_feat = prep.inputs.shape[-1] // n_links
    link_cols = [np.array([t * prep.inputs.shape[-1] + f * n_links + l for t in range(SEQ) for f in range(n_feat)]) for l in range(n_links)]
    fake = B.gan_samples(xtr[ytr == 1], seed, n=int((ytr == 0).sum() - ytr.sum()), device=dev, n_val=n_val)
    best = None
    for x, y in ((xtr, ytr), (np.concatenate([xtr, fake]), np.concatenate([ytr, np.ones(len(fake), int)]))):
        f = B.tssae(x, y, link_cols, seed, dev)
        f1 = prf((f(xtu) >= 0).astype(float), ytu, 0.5)[2]
        if best is None or f1 > best[0]:
            best = (f1, f(xte) >= 0)
    results["TSSAE"] = events(best[1], dte, ste)

    torch.manual_seed(seed)
    s2s = B.Seq2SeqAttn(prep.inputs.shape[-1]).to(dev)
    opt = torch.optim.Adam(s2s.parameters(), 1e-3)
    w = windows("train", speed, dev)
    for _ in range(30):
        for idx in torch.randperm(len(w), device=dev).split(512):
            loss = ((s2s(w.x[idx]) - w.y[idx, 0]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    s2s.eval()

    def residual(part):
        w = windows(part, speed, dev)
        with torch.no_grad():
            return (s2s(w.x) - w.y[:, 0]).cpu().numpy(), w.day, w.slot

    sigma = residual("train")[0].std()
    r_tu = residual("tune")[0]
    r_te, d_te, s_te = residual("test")
    thr = np.quantile(r_tu / sigma, 1 - raw[split["tune"]].mean())
    results["OOD"] = events(r_te / sigma >= thr, d_te, s_te, shift=1)

    dist = B.chebyshev_distance(speed, split["train"], SEQ)
    znorm = (dist - dist[split["tune"]].mean()) / dist[split["tune"]].std()
    deriv = np.diff(znorm, axis=1, prepend=np.nan)
    slots = np.arange(SEQ - 1, speed.shape[1] - 6)
    h = max(np.linspace(0.2, 4.0, 39), key=lambda h: prf(B.cfod_state(znorm, deriv, split["tune"], h)[:, slots].ravel().astype(float),
                                                        raw[split["tune"]][:, slots].ravel(), 0.5)[2])
    results["CFOD"] = event_metrics(B.cfod_state(znorm, deriv, test, h)[:, slots], int(slots[0]), dates, reports, anomaly)

    results["SVM"] = events(B.svm(xtr, ytr, xte, seed), dte, ste)
    results["GAN"] = events(B.gan_svm(xtr, ytr, xte, seed, dev, n_val=n_val), dte, ste)
    for name, run in (("Ours", "denoised"), ("Ours (a)", "ahead")):
        results[name] = json.loads((Path(a.runs) / run / "metrics.json").read_text())["events"]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "table5.json").write_text(json.dumps(results, indent=2))
    fmt = lambda x, k: "-" if x != x else f"{x:.{k}f}"
    lines = ["| Model | DR | MTTD (min) | FAR | DR(S) | MTTD(S) (min) |", "|---|---|---|---|---|---|"]
    for name in ("OOD", "CFOD", "SVM", "GAN", "TSSAE", "Ours", "Ours (a)"):
        e = results[name]
        row = f"{fmt(e['DR'], 2)} | {fmt(e['MTTD'], 0)} | {fmt(e['FAR'], 2)} | {fmt(e['DR_S'], 2)} | {fmt(e['MTTD_S'], 0)}"
        if e["FAR"] > 0.8 or (e["DR"] == 0 and e["DR_S"] == 0):
            row = " | ".join(f"~~{c}~~" for c in row.split(" | "))
        lines.append(f"| {name} | {row} |")
    (out / "table5.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
