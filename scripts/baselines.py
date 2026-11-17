import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ead import baselines as B
from ead.metrics import prf
from ead.pipeline import LabelConfig, prepare


def flat(w):
    return w.x.reshape(len(w), -1).cpu().numpy(), w.y.cpu().numpy().astype(int)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/synthetic_link")
    p.add_argument("--link", default="SYN-T0")
    p.add_argument("--labels", choices=["denoised", "ahead", "raw"], default="denoised")
    p.add_argument("--models", nargs="+", default=["RF", "SVM", "GAN", "LightGBM"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--seq_in", type=int, default=7)
    p.add_argument("--horizon", type=int, default=6)
    p.add_argument("--features", default="speed,slowdown,tti,srs")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="runs/table3")
    a = p.parse_args()

    prep = prepare(a.data, a.link, LabelConfig(), feature_names=tuple(a.features.split(",")))
    xtr, ytr = flat(prep.windows(a.labels, "train", a.seq_in, a.horizon, "cpu"))
    xte, yte = flat(prep.windows(a.labels, "test", a.seq_in, a.horizon, "cpu"))
    _, ytu = flat(prep.windows(a.labels, "tune", a.seq_in, a.horizon, "cpu"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in a.models:
        rows = []
        for h in range(a.horizon):
            y = ytr[:, h]
            if name == "RF":
                pred = B.rf(xtr, y, xte, a.seed)
            elif name == "SVM":
                pred = B.svm(xtr, y, xte, a.seed)
            elif name == "GAN":
                pred = B.gan_svm(xtr, y, xte, a.seed, a.device, n_val=int(ytu[:, h].sum()))
            else:
                pred = B.lightgbm(xtr, y, xte, a.seed)
            pr, rc, f1 = prf(pred.astype(float), yte[:, h], 0.5)
            rows.append(dict(precision=pr, recall=rc, f1=f1))
            print(f"{name} +{5 * (h + 1)}min  P {pr:.2f}  R {rc:.2f}  F1 {f1:.2f}", flush=True)
        (out / f"{name}_{a.labels}_s{a.seed}.json").write_text(json.dumps(dict(model=name, labels=a.labels, seed=a.seed, horizon=rows), indent=2))


if __name__ == "__main__":
    main()
