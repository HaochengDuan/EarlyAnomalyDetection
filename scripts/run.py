import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ead.pipeline import LabelConfig, evaluate, prepare
from ead.train import TrainConfig, fit, predict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/synthetic_link")
    p.add_argument("--link", default="SYN-T0")
    p.add_argument("--labels", choices=["denoised", "ahead", "raw"], default="denoised")
    p.add_argument("--out", required=True)
    p.add_argument("--seq_in", type=int, default=7)
    p.add_argument("--horizon", type=int, default=6)
    p.add_argument("--upstream_mi", type=float, default=2.0)
    p.add_argument("--downstream_mi", type=float, default=1.0)
    p.add_argument("--alert_rule", choices=["min", "max", "mean"], default="min")
    p.add_argument("--features", default="speed,slowdown,tti,srs,report")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    for name, value in list(asdict(LabelConfig()).items()) + list(asdict(TrainConfig()).items()):
        if isinstance(value, bool):
            p.add_argument(f"--{name}", type=lambda s: s.lower() in ("1", "true", "yes"), default=value)
        else:
            p.add_argument(f"--{name}", type=type(value), default=value)
    a = p.parse_args()
    label_cfg = LabelConfig(**{k: getattr(a, k) for k in asdict(LabelConfig())})
    train_cfg = TrainConfig(**{k: getattr(a, k) for k in asdict(TrainConfig())})

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log_file = open(out / "train.log", "w")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    prep = prepare(a.data, a.link, label_cfg, a.upstream_mi, a.downstream_mi, feature_names=tuple(a.features.split(",")))
    log(f"subgraph {prep.subgraph}  input dim {prep.inputs.shape[-1]}  "
        f"slowdown threshold {prep.denoised.threshold:.1f} mph (top {prep.denoised.top_percent}%)")
    parts = {k: prep.windows(a.labels, k, a.seq_in, a.horizon, a.device) for k in ("train", "tune", "val", "test")}
    model, best, history, pos_weight = fit(parts["train"], parts["tune"], parts["val"], train_cfg, log)
    prob = predict(model, parts["test"])
    result = evaluate(prep, prob, parts["test"], best["threshold"], a.alert_rule)
    result.update(labels=a.labels, best_epoch=best["epoch"], val_f1=best["val_f1"], pos_weight=pos_weight,
                  label_config=asdict(label_cfg), train_config=asdict(train_cfg), subgraph=prep.subgraph,
                  features=a.features.split(","))
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    (out / "history.json").write_text(json.dumps(history))
    torch.save(dict(state=model.state_dict(), threshold=best["threshold"]), out / "model.pt")
    np.savez_compressed(out / "test_predictions.npz", prob=prob, day=parts["test"].day, slot=parts["test"].slot)
    e = result["events"]
    log(f"TEST  DR {e['DR']:.2f}  MTTD {e['MTTD']:.1f}  FAR {e['FAR']:.2f}  DR(S) {e['DR_S']:.2f}  MTTD(S) {e['MTTD_S']:.1f}  "
        f"F1@5min {result['horizon'][0]['f1']:.2f}")


if __name__ == "__main__":
    main()
