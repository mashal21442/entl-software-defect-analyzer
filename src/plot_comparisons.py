from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.published_baselines import PUBLISHED


METRICS = ["PD", "PF", "F1-score", "G-Mean", "AUC"]
METHODS = ["ENTL", "EGW", "HDP_KS", "CTKCCA", "EMKCA", "WPDP"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / "results" / "graphs" / "paper_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric in METRICS:
        values = [PUBLISHED[metric]["Average"][m] for m in METHODS]

        plt.figure(figsize=(8, 5))
        bars = plt.bar(METHODS, values)
        plt.ylim(0, 1)
        plt.ylabel(metric)
        plt.title(f"Published average {metric}: ENTL vs baselines")
        plt.xticks(rotation=25)

        for bar, value in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{value:.3f}",
                ha="center",
                fontsize=8,
            )

        plt.tight_layout()
        plt.savefig(out_dir / f"{metric.replace('-', '_')}_published.png", dpi=200)
        plt.close()

    print("Wrote comparison graphs to:", out_dir)


if __name__ == "__main__":
    main()
