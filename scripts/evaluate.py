import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ead.model import AnomalyTransformer
from ead.pipeline import LabelConfig, evaluate, prepare
from ead.train import TrainConfig, predict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", default="data/synthetic_link")
    p.add_argument("--link", default="SYN-T0")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    ck = torch.load(a.checkpoint, map_location=a.device, weights_only=True)
    cfg = TrainConfig(**ck["train_config"])
    prep = prepare(a.data, a.link, LabelConfig(**ck["label_config"]), ck["upstream_mi"], ck["downstream_mi"], feature_names=tuple(ck["features"]))
    test = prep.windows(ck["labels"], "test", ck["seq_in"], ck["horizon"], a.device)
    model = AnomalyTransformer(test.x.shape[-1], test.y.shape[1], cfg.d_model, cfg.nhead, cfg.enc_layers, cfg.dec_layers,
                               cfg.dim_ff, cfg.dropout, cfg.positional_encoding).to(a.device)
    model.load_state_dict(ck["state"])
    prob = predict(model, test)
    result = evaluate(prep, prob, test, ck["threshold"], ck["alert_rule"])
    result.update(labels=ck["labels"], label_config=ck["label_config"], train_config=ck["train_config"], subgraph=prep.subgraph, features=ck["features"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    (out / "history.json").write_text(json.dumps(ck["history"]))
    np.savez_compressed(out / "test_predictions.npz", prob=prob, day=test.day, slot=test.slot)
    e = result["events"]
    print(f"{ck['labels']}  DR {e['DR']:.2f}  MTTD {e['MTTD']:.1f}  FAR {e['FAR']:.2f}  DR(S) {e['DR_S']:.2f}  MTTD(S) {e['MTTD_S']:.1f}")


if __name__ == "__main__":
    main()
