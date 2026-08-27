from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.published_baselines import PUBLISHED, to_long_dataframe


METRICS = ["PD", "PF", "F1-score", "G-Mean", "AUC"]
HCPDP_BASELINES = ["EGW", "HDP_KS", "CTKCCA", "EMKCA"]


def percent_improvement(entl: float, baseline: float) -> float:
    return ((entl - baseline) / baseline) * 100.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / "results" / "paper_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    published_long = to_long_dataframe()
    published_long.to_csv(out_dir / "published_baseline_means.csv", index=False)

    # RQ1: published ENTL against the four HCPDP baselines.
    rq1_rows = []
    for metric in METRICS:
        entl = PUBLISHED[metric]["Average"]["ENTL"]
        for baseline in HCPDP_BASELINES:
            value = PUBLISHED[metric]["Average"][baseline]
            if metric == "PF":
                entl_better = entl < value
            else:
                entl_better = entl > value

            rq1_rows.append(
                {
                    "Metric": metric,
                    "ENTL_published_average": entl,
                    "Baseline": baseline,
                    "Baseline_published_average": value,
                    "ENTL_better": entl_better,
                }
            )

    pd.DataFrame(rq1_rows).to_csv(
        out_dir / "RQ1_published_comparison.csv",
        index=False,
    )

    # RQ2: published ENTL against WPDP.
    rq2_rows = []
    for metric in METRICS:
        entl = PUBLISHED[metric]["Average"]["ENTL"]
        wpdp = PUBLISHED[metric]["Average"]["WPDP"]

        row = {
            "Metric": metric,
            "ENTL_published_average": entl,
            "WPDP_published_average": wpdp,
        }

        if metric != "PF":
            row["Improvement_percent"] = percent_improvement(entl, wpdp)
        else:
            row["Improvement_percent"] = None
            row["Note"] = "Lower PF is better; WPDP has lower published average PF."

        rq2_rows.append(row)

    pd.DataFrame(rq2_rows).to_csv(
        out_dir / "RQ2_published_comparison.csv",
        index=False,
    )

    # Compare the locally reproduced ENTL means (if present) to the paper's ENTL.
    local_summary = root / "results" / "metrics" / "project_summary_numeric.csv"
    if local_summary.exists():
        local = pd.read_csv(local_summary)
        rows = []

        for _, record in local.iterrows():
            target = record["Target"]
            if target not in PUBLISHED["PD"]:
                continue

            for metric in METRICS:
                rows.append(
                    {
                        "Target": target,
                        "Metric": metric,
                        "Local_ENTL_mean": record[f"{metric}_mean"],
                        "Local_ENTL_std": record[f"{metric}_std"],
                        "Published_ENTL_mean": PUBLISHED[metric][target]["ENTL"],
                        "Difference": (
                            record[f"{metric}_mean"]
                            - PUBLISHED[metric][target]["ENTL"]
                        ),
                    }
                )

        pd.DataFrame(rows).to_csv(
            out_dir / "local_ENTL_vs_published_ENTL.csv",
            index=False,
        )

    print("Wrote RQ1/RQ2 comparison files to:", out_dir)


if __name__ == "__main__":
    main()
