from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from tensorflow.keras import backend as K

from src.data_io import prepare_dataset
from src.dataset_registry import get_spec, raw_path
from src.encoder_networks import encode_pair
from src.paper_entl import build_xgboost, fit_nn
from src.project_analyzer.universal_metrics import numeric_feature_names


DEFAULT_SOURCE_DATASET = "JIRA/activemq5.0.0"


def _prepare_target_metrics(metrics_rows: list[dict]) -> pd.DataFrame:
    """
    Convert universal per-file metrics into the numeric target matrix
    expected by the target encoder.

    No target defect labels are required.
    """
    if not metrics_rows:
        raise ValueError("No source-code metrics were generated for this project.")

    feature_names = numeric_feature_names()

    frame = pd.DataFrame(metrics_rows)

    missing = [
        feature
        for feature in feature_names
        if feature not in frame.columns
    ]

    if missing:
        raise ValueError(
            "Target project metrics are missing required features: "
            + ", ".join(missing)
        )

    X = frame[feature_names].copy()

    for column in X.columns:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    for column in X.columns:
        if X[column].isna().all():
            X[column] = 0.0
        else:
            X[column] = X[column].fillna(
                X[column].median()
            )

    scaler = StandardScaler()

    scaled = scaler.fit_transform(X)

    return pd.DataFrame(
        scaled,
        columns=X.columns,
        index=X.index,
    )


def run_entl_unlabeled(
    H_source,
    y_source,
    H_target,
    *,
    iterations: int = 20,
    nn_epochs: int = 20,
    threshold: float = 0.5,
    seed: int = 42,
) -> dict:
    """
    Deployment/inference form of ENTL.

    Unlike src.paper_entl.run_entl(), this function DOES NOT accept
    true target labels and DOES NOT calculate PD/PF/F1/G-Mean/AUC.

    It follows the same main learning flow:
      1. Train source NN Ms.
      2. Pseudo-label the target representation.
      3. Augment source + pseudo-labelled target data.
      4. Train secondary NN MT.
      5. Train XGBoost.
      6. Repeat and aggregate target predictions.

    Returned model scores are predictions/risk scores, not proof that
    a file definitely contains a defect.
    """
    H_source = np.asarray(
        H_source,
        dtype=np.float32,
    )

    H_target = np.asarray(
        H_target,
        dtype=np.float32,
    )

    y_source = np.asarray(
        y_source,
        dtype=np.int32,
    )

    if H_source.ndim != 2 or H_target.ndim != 2:
        raise ValueError(
            "Encoded source and target features must be 2-D matrices."
        )

    if len(H_target) == 0:
        raise ValueError("Target project has no analyzable source files.")

    if H_source.shape[1] != H_target.shape[1]:
        raise ValueError(
            "Encoded source and target dimensions must match."
        )

    if len(np.unique(y_source)) != 2:
        raise ValueError(
            "Historical source dataset must contain both clean and defective modules."
        )

    if iterations < 1:
        raise ValueError("iterations must be at least 1.")

    if nn_epochs < 1:
        raise ValueError("nn_epochs must be at least 1.")

    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    xgb_probabilities = []
    xgb_predictions = []
    mt_probabilities = []
    pseudo_probabilities = []
    iteration_rows = []

    for i in range(iterations):
        iteration_seed = seed + i

        K.clear_session()

        source_model = fit_nn(
            H_source,
            y_source,
            seed=iteration_seed,
            epochs=nn_epochs,
        )

        pseudo_probability = (
            source_model.predict(
                H_target,
                verbose=0,
            )
            .ravel()
        )

        pseudo_labels = (
            pseudo_probability >= threshold
        ).astype(int)

        H_augmented = np.vstack(
            [
                H_source,
                H_target,
            ]
        )

        y_augmented = np.concatenate(
            [
                y_source,
                pseudo_labels,
            ]
        )

        target_model = fit_nn(
            H_augmented,
            y_augmented,
            seed=10_000 + iteration_seed,
            epochs=nn_epochs,
        )

        mt_probability = (
            target_model.predict(
                H_target,
                verbose=0,
            )
            .ravel()
        )

        xgb_model = build_xgboost(
            20_000 + iteration_seed
        )

        sample_weights = compute_sample_weight(
            class_weight="balanced",
            y=y_augmented,
        )

        xgb_model.fit(
            H_augmented,
            y_augmented,
            sample_weight=sample_weights,
        )

        xgb_probability = (
            xgb_model.predict_proba(
                H_target
            )[:, 1]
        )

        xgb_prediction = (
            xgb_model.predict(
                H_target
            )
            .astype(int)
        )

        pseudo_probabilities.append(
            pseudo_probability
        )

        mt_probabilities.append(
            mt_probability
        )

        xgb_probabilities.append(
            xgb_probability
        )

        xgb_predictions.append(
            xgb_prediction
        )

        iteration_rows.append(
            {
                "iteration": i + 1,
                "pseudo_defective": int(
                    (pseudo_labels == 1).sum()
                ),
                "pseudo_clean": int(
                    (pseudo_labels == 0).sum()
                ),
                "pseudo_probability_mean": float(
                    pseudo_probability.mean()
                ),
                "xgb_predicted_defective": int(
                    (xgb_prediction == 1).sum()
                ),
                "xgb_predicted_clean": int(
                    (xgb_prediction == 0).sum()
                ),
                "xgb_probability_mean": float(
                    xgb_probability.mean()
                ),
            }
        )

    xgb_probability_matrix = np.vstack(
        xgb_probabilities
    )

    xgb_prediction_matrix = np.vstack(
        xgb_predictions
    )

    mt_probability_matrix = np.vstack(
        mt_probabilities
    )

    pseudo_probability_matrix = np.vstack(
        pseudo_probabilities
    )

    # Literal Algorithm-1 hard vote fraction.
    algorithm1_vote_fraction = (
        xgb_prediction_matrix.mean(axis=0)
    )

    # Continuous score is more useful for ranking files in a UI.
    # It is kept separate from the literal Algorithm-1 vote.
    xgb_probability_average = (
        xgb_probability_matrix.mean(axis=0)
    )

    mt_probability_average = (
        mt_probability_matrix.mean(axis=0)
    )

    pseudo_probability_average = (
        pseudo_probability_matrix.mean(axis=0)
    )

    final_prediction = (
        algorithm1_vote_fraction >= threshold
    ).astype(int)

    return {
        "algorithm1_vote_fraction":
            algorithm1_vote_fraction,
        "xgb_probability_average":
            xgb_probability_average,
        "mt_probability_average":
            mt_probability_average,
        "pseudo_probability_average":
            pseudo_probability_average,
        "final_prediction":
            final_prediction,
        "iteration_rows":
            iteration_rows,
    }


def predict_project_defect_risk(
    project_root: str | Path,
    metrics_rows: list[dict],
    *,
    source_key: str = DEFAULT_SOURCE_DATASET,
    latent_dim: int = 16,
    encoder_epochs: int = 20,
    nn_epochs: int = 20,
    iterations: int = 20,
    threshold: float = 0.5,
    seed: int = 42,
) -> dict:
    """
    Run ENTL-style defect-risk inference for every analyzed file in an
    uploaded project ZIP.

    The historical source dataset is labelled.
    The uploaded target project is intentionally UNLABELLED.
    """
    project_root = Path(project_root).resolve()

    source_spec = get_spec(source_key)

    source_path = raw_path(
        project_root,
        source_key,
    )

    if not source_path.exists():
        raise FileNotFoundError(
            "Historical source dataset was not found. "
            f"Expected: {source_path}"
        )

    source = prepare_dataset(
        source_path,
        source_spec,
    )

    target_X = _prepare_target_metrics(
        metrics_rows
    )

    min_features = min(
        source.X.shape[1],
        target_X.shape[1],
    )

    if min_features < 2:
        raise ValueError(
            "Source or target requires at least 2 usable numeric features."
        )

    effective_latent = (
        max(2, min_features - 1)
        if latent_dim >= min_features
        else latent_dim
    )

    H_source, H_target = encode_pair(
        source.X,
        target_X,
        latent_dim=effective_latent,
        epochs=encoder_epochs,
        seed=seed,
    )

    result = run_entl_unlabeled(
        H_source,
        source.y.values,
        H_target,
        iterations=iterations,
        nn_epochs=nn_epochs,
        threshold=threshold,
        seed=seed,
    )

    risk_scores = (
        np.asarray(
            result["xgb_probability_average"],
            dtype=float,
        )
        * 100.0
    )

    vote_scores = (
        np.asarray(
            result["algorithm1_vote_fraction"],
            dtype=float,
        )
        * 100.0
    )

    # Rank is descriptive only. It tells which files are riskier
    # relative to other files in this same upload.
    rank_series = pd.Series(
        risk_scores
    ).rank(
        method="average",
        pct=True,
    )

    relative_percentiles = (
        rank_series.to_numpy(dtype=float)
        * 100.0
    )

    predictions = []

    for index, metrics in enumerate(metrics_rows):
        model_prediction = int(
            result["final_prediction"][index]
        )

        raw_score = float(
            risk_scores[index]
        )

        vote_score = float(
            vote_scores[index]
        )

        percentile = float(
            relative_percentiles[index]
        )

        if model_prediction == 1:
            risk_level = "high"
        elif raw_score >= 35.0:
            risk_level = "medium"
        else:
            risk_level = "low"

        predictions.append(
            {
                "path": metrics["path"],
                "language": metrics["language"],
                "entl_risk_score": round(
                    raw_score,
                    2,
                ),
                "algorithm1_vote_percent": round(
                    vote_score,
                    2,
                ),
                "relative_risk_percentile": round(
                    percentile,
                    2,
                ),
                "prediction": (
                    "likely_defect_prone"
                    if model_prediction == 1
                    else "not_flagged_by_entl"
                ),
                "risk_level": risk_level,
            }
        )

    predictions.sort(
        key=lambda item: (
            item["entl_risk_score"],
            item["algorithm1_vote_percent"],
        ),
        reverse=True,
    )

    return {
        "source_dataset": source_key,
        "source_samples": int(
            source.X.shape[0]
        ),
        "source_features": int(
            source.X.shape[1]
        ),
        "target_files": len(
            metrics_rows
        ),
        "target_features": int(
            target_X.shape[1]
        ),
        "latent_dim": int(
            effective_latent
        ),
        "threshold": float(
            threshold
        ),
        "predictions": predictions,
        "iteration_rows":
            result["iteration_rows"],
        "important_note": (
            "ENTL values are defect-risk predictions. "
            "They do not prove that a particular line contains a bug."
        ),
    }
