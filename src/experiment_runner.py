from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATH SETUP
# ============================================================
#
# This allows this file to be executed directly:
#
# python src/experiment_runner.py
#
# while still supporting imports such as:
#
# from src.data_io import prepare_dataset
#
# Without this, Python may raise:
#
# ModuleNotFoundError: No module named 'src'
# ============================================================

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data_io import prepare_dataset
from src.dataset_registry import get_spec, raw_path
from src.encoder_networks import encode_pair
from src.paper_entl import run_entl
from src.metrics import calculate_metrics


# ============================================================
# SAFE OUTPUT NAME
# ============================================================

def safe_name(key: str) -> str:
    """
    Convert dataset keys into filesystem-safe names.

    Example:
        JIRA/activemq5.0.0
    becomes:
        JIRA__activemq5_0_0
    """

    return (
        key
        .replace("/", "__")
        .replace(".", "_")
        .replace("\\", "__")
    )


# ============================================================
# CHECK DATASET FILE
# ============================================================

def check_dataset_exists(
    dataset_path: Path,
    dataset_key: str,
) -> None:
    """
    Raise a clear error if a required dataset file
    does not exist.
    """

    if not dataset_path.exists():
        raise FileNotFoundError(
            "\n"
            + "=" * 70
            + "\nDATASET FILE NOT FOUND\n"
            + "=" * 70
            + f"\nDataset: {dataset_key}"
            + f"\nExpected path:\n{dataset_path}"
            + "\n\nRun dataset validation with:\n"
            + "python -m src.validate_datasets\n"
            + "=" * 70
        )


# ============================================================
# RUN ONE ENTL EXPERIMENT
# ============================================================

def run_experiment(
    project_root: Path,
    source_key: str,
    target_key: str,
    *,
    latent_dim: int,
    encoder_epochs: int,
    nn_epochs: int,
    iterations: int,
    threshold: float,
    seed: int,
    ensemble_mode: str,
    mapping_status: str,
) -> dict:

    # --------------------------------------------------------
    # Dataset specifications
    # --------------------------------------------------------

    source_spec = get_spec(source_key)
    target_spec = get_spec(target_key)

    # --------------------------------------------------------
    # HCPDP requires different project families
    # --------------------------------------------------------

    if source_spec.family == target_spec.family:
        raise ValueError(
            f"Source {source_key} and target {target_key} "
            "are from the same project family.\n"
            "The paper requires source and target datasets "
            "from different project families."
        )

    # --------------------------------------------------------
    # Resolve raw dataset paths
    # --------------------------------------------------------

    source_path = raw_path(
        project_root,
        source_key,
    )

    target_path = raw_path(
        project_root,
        target_key,
    )

    # --------------------------------------------------------
    # Verify dataset files exist
    # --------------------------------------------------------

    check_dataset_exists(
        source_path,
        source_key,
    )

    check_dataset_exists(
        target_path,
        target_key,
    )

    # --------------------------------------------------------
    # Experiment information
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LOADING EXPERIMENT DATA")
    print("=" * 70)

    print(f"Source : {source_key}")
    print(f"Target : {target_key}")

    print()
    print(f"Source file: {source_path}")
    print(f"Target file: {target_path}")

    # --------------------------------------------------------
    # Prepare datasets
    # --------------------------------------------------------

    source = prepare_dataset(
        source_path,
        source_spec,
    )

    target = prepare_dataset(
        target_path,
        target_spec,
    )

    print()
    print("Prepared dataset shapes:")
    print(
        f"Source: {source.X.shape[0]} samples, "
        f"{source.X.shape[1]} features"
    )

    print(
        f"Target: {target.X.shape[0]} samples, "
        f"{target.X.shape[1]} features"
    )

    # --------------------------------------------------------
    # Determine encoder latent dimension
    # --------------------------------------------------------

    min_features = min(
        source.X.shape[1],
        target.X.shape[1],
    )

    if min_features < 2:
        raise ValueError(
            "Source or target dataset has fewer than "
            "2 usable features after preprocessing."
        )

    if latent_dim >= min_features:
        effective_latent = max(
            2,
            min_features - 1,
        )
    else:
        effective_latent = latent_dim

    print()
    print("=" * 70)
    print("ENCODER CONFIGURATION")
    print("=" * 70)

    print(f"Requested latent dimension: {latent_dim}")
    print(f"Effective latent dimension: {effective_latent}")
    print(f"Encoder epochs: {encoder_epochs}")

    # --------------------------------------------------------
    # Separate source and target encoders
    # --------------------------------------------------------

    H_source, H_target = encode_pair(
        source.X,
        target.X,
        latent_dim=effective_latent,
        epochs=encoder_epochs,
        seed=seed,
    )

    # --------------------------------------------------------
    # Verify encoded dimensions
    # --------------------------------------------------------

    if H_source.shape[1] != H_target.shape[1]:
        raise ValueError(
            "Encoder dimension mismatch.\n"
            f"Source encoded dimension: {H_source.shape[1]}\n"
            f"Target encoded dimension: {H_target.shape[1]}"
        )

    print()
    print("Encoded dimensions match:")
    print(H_source.shape[1])

    # --------------------------------------------------------
    # Run ENTL Algorithm 1
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RUNNING ENTL")
    print("=" * 70)

    print(f"Iterations: {iterations}")
    print(f"NN epochs: {nn_epochs}")
    print(f"Pseudo-label threshold: {threshold}")
    print(f"Ensemble mode: {ensemble_mode}")
    print(f"Seed: {seed}")

    start = time.time()

    result = run_entl(
        H_source,
        source.y.values,
        H_target,
        target.y.values,
        iterations=iterations,
        nn_epochs=nn_epochs,
        threshold=threshold,
        seed=seed,
        ensemble_mode=ensemble_mode,
    )

    seconds = time.time() - start

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    out_dir = (
        project_root
        / "results"
        / "individual"
        / (
            f"{safe_name(target_key)}"
            f"__from__"
            f"{safe_name(source_key)}"
        )
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    prediction_df = pd.DataFrame(
        {
            "ENTL_probability":
                result["final_probability"],

            "ENTL_prediction":
                result["final_prediction"],

            "Algorithm1_probability":
                result["algorithm1_probability"],

            "Equation3_probability":
                result["equation3_probability"],

            "XGB_probability_average":
                result["probability_average_probability"],

            "Prose_probability":
                result["prose_probability"],

            "MT_mean_probability":
                result["mt_mean_probability"],

            "MT_last_probability":
                result["mt_last_probability"],

            "MT_last_prediction":
                result["mt_last_prediction"],

            "true_label":
                target.y.values,
        }
    )

    predictions_path = (
        out_dir
        / "predictions.csv"
    )

    prediction_df.to_csv(
        predictions_path,
        index=False,
    )

    # --------------------------------------------------------
    # Save metrics for every ensemble interpretation
    # --------------------------------------------------------

    mode_scores = {
        "algorithm1":
            result["algorithm1_probability"],

        "equation3":
            result["equation3_probability"],

        "xgb_probability_average":
            result["probability_average_probability"],

        "prose":
            result["prose_probability"],

        "mt_mean":
            result["mt_mean_probability"],

        "mt_last":
            result["mt_last_probability"],
    }

    mode_metric_rows = []

    for mode_name, mode_probability in mode_scores.items():

        mode_metrics = calculate_metrics(
            target.y.values,
            mode_probability,
            threshold=threshold,
        )

        mode_metric_rows.append(
            {
                "mode": mode_name,
                "threshold": threshold,
                **mode_metrics,
            }
        )

    modes_metrics_path = (
        out_dir
        / "metrics_by_mode.csv"
    )

    pd.DataFrame(
        mode_metric_rows
    ).to_csv(
        modes_metrics_path,
        index=False,
    )

    # --------------------------------------------------------
    # Save iteration results
    # --------------------------------------------------------

    iterations_path = (
        out_dir
        / "iterations.csv"
    )

    pd.DataFrame(
        result["iteration_rows"]
    ).to_csv(
        iterations_path,
        index=False,
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "source": source_key,
        "target": target_key,

        "source_family":
            source_spec.family,

        "target_family":
            target_spec.family,

        "mapping_status":
            mapping_status,

        "latent_dim":
            effective_latent,

        "encoder_epochs":
            encoder_epochs,

        "nn_epochs":
            nn_epochs,

        "iterations":
            iterations,

        "threshold":
            threshold,

        "ensemble_mode":
            ensemble_mode,

        "seed":
            seed,

        "source_samples":
            int(source.X.shape[0]),

        "target_samples":
            int(target.X.shape[0]),

        "source_features":
            int(source.X.shape[1]),

        "target_features":
            int(target.X.shape[1]),

        "source_warnings":
            source.warnings,

        "target_warnings":
            target.warnings,

        "source_dropped_columns":
            source.dropped_columns,

        "target_dropped_columns":
            target.dropped_columns,

        "runtime_seconds":
            seconds,
    }

    metadata_path = (
        out_dir
        / "metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = dict(
        result["metrics"]
    )

    metrics.update(
        {
            "source":
                source_key,

            "target":
                target_key,

            "source_family":
                source_spec.family,

            "target_family":
                target_spec.family,

            "mapping_status":
                mapping_status,

            "latent_dim":
                effective_latent,

            "runtime_seconds":
                seconds,
        }
    )

    metrics_path = (
        out_dir
        / "metrics.csv"
    )

    pd.DataFrame(
        [metrics]
    ).to_csv(
        metrics_path,
        index=False,
    )

    # --------------------------------------------------------
    # Completion information
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    print(
        f"Runtime: {seconds:.2f} seconds"
    )

    print()
    print(
        f"Predictions saved: {predictions_path}"
    )

    print(
        f"Iterations saved: {iterations_path}"
    )

    print(
        f"Metrics saved: {metrics_path}"
    )

    print(
        f"All-mode metrics saved: {modes_metrics_path}"
    )

    print(
        f"Metadata saved: {metadata_path}"
    )

    return metrics


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run ENTL heterogeneous cross-project "
            "software defect prediction experiments."
        )
    )

    parser.add_argument(
        "--root",
        default=".",
        help="Project root directory.",
    )

    parser.add_argument(
        "--experiments",
        default="config/experiments.csv",
        help="Experiment mapping CSV file.",
    )

    parser.add_argument(
        "--latent-dim",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--encoder-epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--nn-epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--ensemble-mode",
        choices=[
            "algorithm1",
            "equation3",
            "prose",
        ],
        default="algorithm1",
    )

    parser.add_argument(
        "--only-target",
        default=None,
        help=(
            "Optional dataset key. "
            "Example: AEEM/EQ"
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Project root
    # --------------------------------------------------------

    project_root = Path(
        args.root
    ).resolve()

    print()
    print("=" * 70)
    print("ENTL EXPERIMENT RUNNER")
    print("=" * 70)

    print(
        f"Project root:\n{project_root}"
    )

    # --------------------------------------------------------
    # Load experiment plan
    # --------------------------------------------------------

    experiment_file = (
        project_root
        / args.experiments
    )

    if not experiment_file.exists():
        raise FileNotFoundError(
            "\nExperiment configuration file not found:\n"
            f"{experiment_file}"
        )

    plan = pd.read_csv(
        experiment_file
    )

    required_columns = {
        "source",
        "target",
    }

    missing_columns = (
        required_columns
        - set(plan.columns)
    )

    if missing_columns:
        raise ValueError(
            "Experiment CSV is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # --------------------------------------------------------
    # Filter by target
    # --------------------------------------------------------

    if args.only_target:

        plan = plan.loc[
            plan["target"]
            == args.only_target
        ].copy()

        if plan.empty:
            raise ValueError(
                "No experiment found for target "
                f"{args.only_target!r}"
            )

    # Reset index after filtering.
    #
    # This ensures output such as:
    #
    # [1/1]
    #
    # instead of using the original row index.
    # --------------------------------------------------------

    plan = plan.reset_index(
        drop=True
    )

    print()
    print(
        f"Experiments to run: {len(plan)}"
    )

    if args.only_target:
        print(
            f"Target filter: {args.only_target}"
        )

    # --------------------------------------------------------
    # Run experiments
    # --------------------------------------------------------

    rows = []

    for index, row in plan.iterrows():

        source_key = str(
            row["source"]
        ).strip()

        target_key = str(
            row["target"]
        ).strip()

        print()
        print("=" * 70)

        print(
            f"[{index + 1}/{len(plan)}] "
            f"{source_key} -> {target_key}"
        )

        print("=" * 70)

        mapping_status = row.get(
            "mapping_status",
            "unknown",
        )

        if pd.isna(mapping_status):
            mapping_status = "unknown"

        metrics = run_experiment(
            project_root,
            source_key,
            target_key,

            latent_dim=
                args.latent_dim,

            encoder_epochs=
                args.encoder_epochs,

            nn_epochs=
                args.nn_epochs,

            iterations=
                args.iterations,

            threshold=
                args.threshold,

            seed=
                args.seed
                + index * 1000,

            ensemble_mode=
                args.ensemble_mode,

            mapping_status=
                str(mapping_status),
        )

        rows.append(
            metrics
        )

        print()
        print("Paper metrics:")

        print(
            " | ".join(
                f"{metric}="
                f"{metrics[metric]:.4f}"

                for metric in [
                    "PD",
                    "PF",
                    "F1-score",
                    "G-Mean",
                    "AUC",
                ]
            )
        )

    # --------------------------------------------------------
    # Save combined results
    # --------------------------------------------------------

    metrics_dir = (
        project_root
        / "results"
        / "metrics"
    )

    metrics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results_path = (
        metrics_dir
        / "all_target_results.csv"
    )

    pd.DataFrame(
        rows
    ).to_csv(
        all_results_path,
        index=False,
    )

    print()
    print("=" * 70)
    print("ALL REQUESTED EXPERIMENTS COMPLETE")
    print("=" * 70)

    print(
        f"Combined results saved to:\n"
        f"{all_results_path}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()