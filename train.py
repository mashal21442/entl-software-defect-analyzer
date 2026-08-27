from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str], cwd: Path) -> None:
    print()
    print("=" * 80)
    print("RUN:", " ".join(command))
    print("=" * 80)

    subprocess.run(
        command,
        cwd=cwd,
        check=True
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run the ENTL paper-reproduction experiment."
        )
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Fast smoke test: 2 ENTL iterations "
            "and fewer epochs."
        ),
    )

    parser.add_argument(
        "--only-target",
        default=None,
        help=(
            "Run one target only, "
            "for example AEEM/EQ"
        ),
    )

    parser.add_argument(
        "--ensemble-mode",
        choices=[
            "algorithm1",
            "prose"
        ],
        default="algorithm1",
    )

    args = parser.parse_args()

    root = Path(
        __file__
    ).resolve().parent

    python = sys.executable

    # ========================================================
    # DATASET VALIDATION
    # ========================================================
    #
    # Full experiment requires all 16 datasets.
    # When --only-target is supplied, validation is skipped
    # here because the experiment runner checks the required
    # source and target files itself.
    # ========================================================

    if not args.only_target:

        run(
            [
                python,
                "-m",
                "src.validate_datasets"
            ],
            root
        )

    # ========================================================
    # QUICK / FULL SETTINGS
    # ========================================================

    if args.quick:

        encoder_epochs = "20"
        nn_epochs = "20"
        iterations = "2"

    else:

        # Paper setting:
        # 20 ENTL iterations.
        #
        # Epoch values are reproduction assumptions because
        # the paper does not publish exact epoch counts.
        encoder_epochs = "100"
        nn_epochs = "100"
        iterations = "20"

    # ========================================================
    # MAIN EXPERIMENT
    # ========================================================

    experiment_command = [
        python,
        "-m",
        "src.experiment_runner",

        "--latent-dim",
        "16",

        "--encoder-epochs",
        encoder_epochs,

        "--nn-epochs",
        nn_epochs,

        "--iterations",
        iterations,

        "--threshold",
        "0.5",

        "--ensemble-mode",
        args.ensemble_mode,
    ]

    if args.only_target:

        experiment_command.extend(
            [
                "--only-target",
                args.only_target
            ]
        )

    run(
        experiment_command,
        root
    )

    # ========================================================
    # AGGREGATE LOCAL RESULTS
    # ========================================================

    run(
        [
            python,
            "-m",
            "src.aggregate_results"
        ],
        root
    )

    # ========================================================
    # PUBLISHED BASELINES
    # ========================================================

    run(
        [
            python,
            "-m",
            "src.published_baselines"
        ],
        root
    )

    # ========================================================
    # RQ1 / RQ2 COMPARISON
    # ========================================================

    run(
        [
            python,
            "-m",
            "src.compare_rq"
        ],
        root
    )

    # ========================================================
    # COMPARISON GRAPHS
    # ========================================================

    run(
        [
            python,
            "-m",
            "src.plot_comparisons"
        ],
        root
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 80)
    print("ENTL PAPER-REPRODUCTION PIPELINE COMPLETE")
    print("=" * 80)

    print()
    print("Main outputs:")

    print(
        " - results/metrics/"
        "all_target_results.csv"
    )

    print(
        " - results/metrics/"
        "project_summary_mean_std.csv"
    )

    print(
        " - results/paper_comparison/"
        "RQ1_published_comparison.csv"
    )

    print(
        " - results/paper_comparison/"
        "RQ2_published_comparison.csv"
    )

    print(
        " - results/paper_comparison/"
        "local_ENTL_vs_published_ENTL.csv"
    )

    print(
        " - results/graphs/"
        "paper_comparison/"
    )


if __name__ == "__main__":
    main()
