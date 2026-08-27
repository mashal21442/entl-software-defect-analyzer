from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = ["PD", "PF", "F1-score", "G-Mean", "AUC"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    input_path = root / "results" / "metrics" / "all_target_results.csv"
    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} does not exist. Run experiment_runner.py first."
        )

    df = pd.read_csv(input_path)

    numeric_rows = []
    formatted_rows = []

    for family, group in df.groupby("target_family", sort=False):
        numeric = {"Target": family}
        formatted = {"Target": family}

        for metric in METRICS:
            mean = group[metric].mean()
            std = group[metric].std(ddof=1)
            numeric[f"{metric}_mean"] = mean
            numeric[f"{metric}_std"] = std
            formatted[metric] = f"{mean:.3f} ± {std:.3f}"

        numeric_rows.append(numeric)
        formatted_rows.append(formatted)

    # Paper-style last row: average across project-family means.
    overall_numeric = {"Target": "Average"}
    overall_formatted = {"Target": "Average"}

    family_numeric = pd.DataFrame(numeric_rows)
    for metric in METRICS:
        means = family_numeric[f"{metric}_mean"]
        overall_mean = means.mean()
        overall_std = means.std(ddof=1)
        overall_numeric[f"{metric}_mean"] = overall_mean
        overall_numeric[f"{metric}_std"] = overall_std
        overall_formatted[metric] = f"{overall_mean:.3f} ± {overall_std:.3f}"

    numeric_rows.append(overall_numeric)
    formatted_rows.append(overall_formatted)

    out_dir = root / "results" / "metrics"
    pd.DataFrame(numeric_rows).to_csv(
        out_dir / "project_summary_numeric.csv",
        index=False,
    )
    pd.DataFrame(formatted_rows).to_csv(
        out_dir / "project_summary_mean_std.csv",
        index=False,
    )

    print(pd.DataFrame(formatted_rows).to_string(index=False))


if __name__ == "__main__":
    main()
