import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAMES = {"denoised": "Ours", "ahead": "Ours (a)", "raw": "Incident report"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="results")
    a = p.parse_args()
    runs, out = Path(a.runs), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    load = lambda f: json.loads(f.read_text())

    rows = {"Transformer": load(runs / "denoised" / "metrics.json")["horizon"]}
    for name in ("LightGBM", "RF", "GAN", "SVM"):
        for f in sorted((runs / "table3").glob(f"{name}_denoised_s*.json")):
            rows[name] = load(f)["horizon"]
    n = len(rows["Transformer"])
    lines = ["| Model | Metric | " + " | ".join(f"{5 * (h + 1)} min" for h in range(n)) + " |", "|---|---|" + "---|" * n]
    for name, r in rows.items():
        for key in ("precision", "recall", "f1"):
            lines.append(f"| {name} | {key} | " + " | ".join(f"{x[key]:.2f}" for x in r) + " |")
    (out / "table3.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines) + "\n")

    fmt = lambda x, k: "-" if x != x else f"{x:.{k}f}"
    lines = ["| Labels | DR | MTTD (min) | FAR | final train loss |", "|---|---|---|---|---|"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for k in ("denoised", "ahead", "raw"):
        e = load(runs / k / "metrics.json")["events"]
        loss = [h["train_loss"] for h in load(runs / k / "history.json")]
        lines.append(f"| {NAMES[k]} | {fmt(e['DR'], 2)} | {fmt(e['MTTD'], 0)} | {fmt(e['FAR'], 2)} | {loss[-1]:.2f} |")
        ax.plot(loss, label=NAMES[k])
    (out / "table6.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    ax.set_xlabel("epoch")
    ax.set_ylabel("train loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "training_loss.png", dpi=150)


if __name__ == "__main__":
    main()
