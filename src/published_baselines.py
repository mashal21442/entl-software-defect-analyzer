from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# Published project-level means corresponding to the ENTL comparison tables.
# The original paper reports mean ± standard deviation in Tables 2-6.
# The accessible machine-readable text exposes the means but not the cell-level
# standard deviations, so this file intentionally stores means only.
#
# Methods:
# ENTL, EGW, HDP_KS, CTKCCA, EMKCA, WPDP

PUBLISHED = {
    "PD": {
        "AEEM":    {"ENTL": .751, "EGW": .598, "HDP_KS": .488, "CTKCCA": .271, "EMKCA": .080, "WPDP": .375},
        "JIRA":    {"ENTL": .675, "EGW": .662, "HDP_KS": .515, "CTKCCA": .182, "EMKCA": .028, "WPDP": .395},
        "NASA":    {"ENTL": .751, "EGW": .612, "HDP_KS": .581, "CTKCCA": .298, "EMKCA": .105, "WPDP": .272},
        "PROMISE": {"ENTL": .663, "EGW": .505, "HDP_KS": .416, "CTKCCA": .455, "EMKCA": .151, "WPDP": .590},
        "Average": {"ENTL": .711, "EGW": .593, "HDP_KS": .551, "CTKCCA": .283, "EMKCA": .095, "WPDP": .411},
    },
    "PF": {
        "AEEM":    {"ENTL": .278, "EGW": .265, "HDP_KS": .251, "CTKCCA": .115, "EMKCA": .105, "WPDP": .169},
        "JIRA":    {"ENTL": .325, "EGW": .278, "HDP_KS": .296, "CTKCCA": .022, "EMKCA": .025, "WPDP": .105},
        "NASA":    {"ENTL": .250, "EGW": .281, "HDP_KS": .284, "CTKCCA": .121, "EMKCA": .037, "WPDP": .061},
        "PROMISE": {"ENTL": .338, "EGW": .255, "HDP_KS": .282, "CTKCCA": .151, "EMKCA": .142, "WPDP": .412},
        "Average": {"ENTL": .297, "EGW": .271, "HDP_KS": .282, "CTKCCA": .101, "EMKCA": .091, "WPDP": .193},
    },
    "F1-score": {
        "AEEM":    {"ENTL": .325, "EGW": .430, "HDP_KS": .372, "CTKCCA": .311, "EMKCA": .111, "WPDP": .353},
        "JIRA":    {"ENTL": .449, "EGW": .457, "HDP_KS": .363, "CTKCCA": .282, "EMKCA": .027, "WPDP": .421},
        "NASA":    {"ENTL": .470, "EGW": .337, "HDP_KS": .322, "CTKCCA": .191, "EMKCA": .145, "WPDP": .311},
        "PROMISE": {"ENTL": .607, "EGW": .548, "HDP_KS": .463, "CTKCCA": .525, "EMKCA": .215, "WPDP": .586},
        "Average": {"ENTL": .475, "EGW": .453, "HDP_KS": .412, "CTKCCA": .335, "EMKCA": .120, "WPDP": .413},
    },
    "G-Mean": {
        "AEEM":    {"ENTL": .712, "EGW": .657, "HDP_KS": .579, "CTKCCA": .400, "EMKCA": .145, "WPDP": .547},
        "JIRA":    {"ENTL": .780, "EGW": .691, "HDP_KS": .583, "CTKCCA": .305, "EMKCA": .032, "WPDP": .592},
        "NASA":    {"ENTL": .831, "EGW": .662, "HDP_KS": .633, "CTKCCA": .317, "EMKCA": .185, "WPDP": .500},
        "PROMISE": {"ENTL": .758, "EGW": .600, "HDP_KS": .507, "CTKCCA": .571, "EMKCA": .249, "WPDP": .581},
        "Average": {"ENTL": .770, "EGW": .653, "HDP_KS": .601, "CTKCCA": .402, "EMKCA": .154, "WPDP": .553},
    },
    "AUC": {
        "AEEM":    {"ENTL": .622, "EGW": .665, "HDP_KS": .656, "CTKCCA": .542, "EMKCA": .515, "WPDP": .573},
        "JIRA":    {"ENTL": .657, "EGW": .690, "HDP_KS": .642, "CTKCCA": .581, "EMKCA": .551, "WPDP": .685},
        "NASA":    {"ENTL": .727, "EGW": .665, "HDP_KS": .698, "CTKCCA": .497, "EMKCA": .625, "WPDP": .713},
        "PROMISE": {"ENTL": .647, "EGW": .621, "HDP_KS": .606, "CTKCCA": .662, "EMKCA": .525, "WPDP": .597},
        "Average": {"ENTL": .663, "EGW": .661, "HDP_KS": .631, "CTKCCA": .611, "EMKCA": .593, "WPDP": .641},
    },
}


def to_long_dataframe() -> pd.DataFrame:
    rows = []
    for metric, targets in PUBLISHED.items():
        for target, methods in targets.items():
            for method, value in methods.items():
                rows.append(
                    {
                        "Metric": metric,
                        "Target": target,
                        "Method": method,
                        "Published_mean": value,
                    }
                )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / "results" / "paper_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = to_long_dataframe()
    df.to_csv(out_dir / "published_baseline_means.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
