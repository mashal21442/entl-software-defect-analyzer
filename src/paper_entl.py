from __future__ import annotations

import random

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import (
    compute_class_weight,
    compute_sample_weight,
)
from tensorflow.keras import Sequential, backend as K
from tensorflow.keras.layers import Dense, Input
from xgboost import XGBClassifier

from src.metrics import calculate_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def class_weights(y) -> dict[int, float]:
    y = np.asarray(y).astype(int)
    classes = np.unique(y)

    if len(classes) != 2:
        raise ValueError(
            "Training labels must contain both 0 and 1."
        )

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y,
    )

    return {
        int(c): float(w)
        for c, w in zip(classes, weights)
    }


def build_nn(
    input_dim: int,
    seed: int,
) -> Sequential:
    """
    Reproduction assumption:
    The paper states neural-network classifiers with binary
    cross-entropy and cost-sensitive class weights, but does
    not publish exact hidden layer sizes, optimizer settings,
    or epoch counts.

    These fixed layers are therefore implementation choices.
    """

    set_seed(seed)

    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
    )

    return model


def fit_nn(
    X,
    y,
    seed: int,
    epochs: int,
) -> Sequential:
    """
    Train the ENTL neural-network classifier.

    Important:
    We train directly with balanced class weights and BCE.

    Validation-based early stopping was removed because it was
    an additional reproduction assumption and could select a
    checkpoint biased toward the majority clean class.
    """

    X = np.asarray(
        X,
        dtype=np.float32,
    )

    y = np.asarray(
        y,
        dtype=np.int32,
    )

    model = build_nn(
        X.shape[1],
        seed,
    )

    weights = class_weights(y)

    model.fit(
        X,
        y,
        epochs=epochs,
        batch_size=min(32, len(X)),
        class_weight=weights,
        shuffle=True,
        verbose=0,
    )

    return model


def build_xgboost(
    seed: int,
) -> XGBClassifier:
    """
    Reproduction assumption:
    The paper names XGBoost but does not publish the exact
    XGBoost hyperparameters.

    Class imbalance is handled at fit time using balanced
    per-sample weights.
    """

    return XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.80,
        colsample_bytree=0.80,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=1,
    )


def run_entl(
    H_source,
    y_source,
    H_target,
    y_target,
    *,
    iterations: int = 20,
    nn_epochs: int = 100,
    threshold: float = 0.5,
    seed: int = 42,
    ensemble_mode: str = "algorithm1",
) -> dict:
    """
    Implements Algorithm 1 from the ENTL paper.

    Every iteration:
      1) train Ms on Hs, Ys with class weighting
      2) pseudo-label Ht using threshold 0.5
      3) create HA, YA
      4) train MT on HA, YA with class weighting
      5) train one XGBoost model on HA, YA using balanced
         sample weights
      6) store that XGBoost target hard prediction (0/1)

    Algorithm 1 then averages those stored target predictions.
    The resulting value is a vote fraction, and threshold 0.5
    corresponds to majority voting.

    Paper inconsistency:
    The surrounding prose/Equation 3 says MT also contributes
    to the ensemble, while Algorithm 1's prediction list P
    contains only XGBoost outputs.

    Default ensemble_mode="algorithm1" uses hard XGBoost votes,
    which is the literal reading of Algorithm 1's "predict Y_T"
    step. The previous predict_proba average is retained as a
    diagnostic return value.

    ensemble_mode="prose" uses MT plus XGBoost probability
    outputs as a separate diagnostic interpretation.
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

    y_target = np.asarray(
        y_target,
        dtype=np.int32,
    )

    if H_source.shape[1] != H_target.shape[1]:
        raise ValueError(
            "Encoded source and target dimensions must match."
        )

    if iterations < 1:
        raise ValueError(
            "iterations must be at least 1."
        )

    if nn_epochs < 1:
        raise ValueError(
            "nn_epochs must be at least 1."
        )

    if not 0.0 < threshold < 1.0:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    xgb_probabilities = []
    xgb_predictions = []
    mt_probabilities = []
    iteration_rows = []

    # ========================================================
    # ENTL ITERATIONS
    # ========================================================

    for i in range(iterations):

        iteration_seed = seed + i

        # Clear old TensorFlow/Keras graph state between
        # repeated neural-network training runs.
        K.clear_session()

        # ----------------------------------------------------
        # 4.1 TRAIN SOURCE MODEL Ms
        # ----------------------------------------------------

        Ms = fit_nn(
            H_source,
            y_source,
            seed=iteration_seed,
            epochs=nn_epochs,
        )

        # ----------------------------------------------------
        # SOURCE-MODEL DIAGNOSTICS
        # ----------------------------------------------------
        # These diagnostics do NOT change training. They only
        # measure whether Ms learned the labelled source data
        # before it is transferred to the target representation.

        source_probability = (
            Ms.predict(
                H_source,
                verbose=0,
            )
            .ravel()
        )

        source_prediction = (
            source_probability >= threshold
        ).astype(int)

        source_tp = int(
            ((y_source == 1) & (source_prediction == 1)).sum()
        )
        source_fn = int(
            ((y_source == 1) & (source_prediction == 0)).sum()
        )
        source_fp = int(
            ((y_source == 0) & (source_prediction == 1)).sum()
        )
        source_tn = int(
            ((y_source == 0) & (source_prediction == 0)).sum()
        )

        source_pd = (
            source_tp / (source_tp + source_fn)
            if (source_tp + source_fn) > 0
            else 0.0
        )

        source_pf = (
            source_fp / (source_fp + source_tn)
            if (source_fp + source_tn) > 0
            else 0.0
        )

        # ----------------------------------------------------
        # 4.2 GENERATE TARGET PSEUDO-LABELS
        # ----------------------------------------------------

        pseudo_probability = (
            Ms.predict(
                H_target,
                verbose=0,
            )
            .ravel()
        )

        pseudo_labels = (
            pseudo_probability >= threshold
        ).astype(int)

        # ----------------------------------------------------
        # 4.3 CREATE AUGMENTED DATASET
        # ----------------------------------------------------

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

        augmented_clean = int(
            (y_augmented == 0).sum()
        )

        augmented_defective = int(
            (y_augmented == 1).sum()
        )

        # ----------------------------------------------------
        # 4.4 TRAIN SECONDARY MODEL MT
        # ----------------------------------------------------

        MT = fit_nn(
            H_augmented,
            y_augmented,
            seed=10_000 + iteration_seed,
            epochs=nn_epochs,
        )

        mt_probability = (
            MT.predict(
                H_target,
                verbose=0,
            )
            .ravel()
        )

        mt_probabilities.append(
            mt_probability
        )

        # ----------------------------------------------------
        # 4.5 / 4.6 TRAIN XGBOOST WITH BALANCED SAMPLE WEIGHTS
        # ----------------------------------------------------

        XG = build_xgboost(
            20_000 + iteration_seed
        )

        xgb_weights = compute_sample_weight(
            class_weight="balanced",
            y=y_augmented,
        )

        XG.fit(
            H_augmented,
            y_augmented,
            sample_weight=xgb_weights,
        )

        # Keep probability output for diagnostics/prose mode.
        xgb_probability = (
            XG.predict_proba(
                H_target
            )[:, 1]
        )

        # Literal Algorithm 1 interpretation:
        # predict Y_T with each stored XGBoost model.
        xgb_prediction = (
            XG.predict(
                H_target
            )
            .astype(int)
        )

        xgb_probabilities.append(
            xgb_probability
        )

        xgb_predictions.append(
            xgb_prediction
        )

        # ----------------------------------------------------
        # ITERATION DIAGNOSTICS
        # ----------------------------------------------------

        iteration_rows.append(
            {
                "iteration": i + 1,

                "source_predicted_clean": int(
                    (source_prediction == 0).sum()
                ),

                "source_predicted_defective": int(
                    (source_prediction == 1).sum()
                ),

                "source_predicted_defect_rate": float(
                    source_prediction.mean()
                ),

                "source_PD": float(
                    source_pd
                ),

                "source_PF": float(
                    source_pf
                ),

                "source_probability_mean": float(
                    source_probability.mean()
                ),

                "source_probability_max": float(
                    source_probability.max()
                ),

                "pseudo_clean": int(
                    (pseudo_labels == 0).sum()
                ),

                "pseudo_defective": int(
                    (pseudo_labels == 1).sum()
                ),

                "pseudo_defect_rate": float(
                    pseudo_labels.mean()
                ),

                "pseudo_probability_min": float(
                    pseudo_probability.min()
                ),

                "pseudo_probability_mean": float(
                    pseudo_probability.mean()
                ),

                "pseudo_probability_max": float(
                    pseudo_probability.max()
                ),

                "augmented_clean": (
                    augmented_clean
                ),

                "augmented_defective": (
                    augmented_defective
                ),

                "augmented_defect_rate": float(
                    y_augmented.mean()
                ),

                "mt_probability_mean": float(
                    mt_probability.mean()
                ),

                "mt_probability_max": float(
                    mt_probability.max()
                ),

                "xgb_probability_mean": float(
                    xgb_probability.mean()
                ),

                "xgb_probability_max": float(
                    xgb_probability.max()
                ),

                "xgb_predicted_clean": int(
                    (xgb_prediction == 0).sum()
                ),

                "xgb_predicted_defective": int(
                    (xgb_prediction == 1).sum()
                ),

                "xgb_predicted_defect_rate": float(
                    xgb_prediction.mean()
                ),
            }
        )

    # ========================================================
    # COMBINE ITERATION PREDICTIONS
    # ========================================================

    # Probability matrix retained for diagnostics and prose mode.
    xgb_probability_matrix = np.vstack(
        xgb_probabilities
    )

    # Hard 0/1 predictions from each XGBoost iteration.
    xgb_prediction_matrix = np.vstack(
        xgb_predictions
    )

    mt_matrix = np.vstack(
        mt_probabilities
    )

    # Literal Algorithm 1 interpretation:
    # each XGBoost predicts Y_T (0/1), then P is averaged.
    algorithm1_probability = (
        xgb_prediction_matrix.mean(axis=0)
    )

    # Previous probability-average interpretation retained
    # separately as a diagnostic.
    probability_average_probability = (
        xgb_probability_matrix.mean(axis=0)
    )

    # Mean MT probability across all ENTL iterations.
    mt_mean_probability = (
        mt_matrix.mean(axis=0)
    )

    # ========================================================
    # LITERAL EQUATION 3 INTERPRETATION
    # ========================================================

    # Equation 3 refers to prediction S from MT.
    #
    # Because MT is retrained inside every iteration but the
    # paper does not explicitly state which MT prediction is
    # used in Equation 3, this implementation uses the final
    # trained MT model as S and records that as a reproduction
    # assumption.

    mt_last_probability = mt_matrix[-1]

    mt_last_prediction = (
        mt_last_probability >= threshold
    ).astype(int)

    # Published Equation 3:
    #
    #             S + sum(X_i)
    #       E = ----------------
    #                  n
    #
    # The published denominator is n, not n + 1.
    equation3_probability = (
        mt_last_prediction
        + xgb_prediction_matrix.sum(axis=0)
    ) / iterations

    # ========================================================
    # PROSE / PROBABILITY DIAGNOSTIC INTERPRETATION
    # ========================================================

    # Diagnostic interpretation that combines mean MT
    # probability with all XGBoost probability outputs.
    prose_probability = np.vstack(
        [
            mt_mean_probability.reshape(
                1,
                -1,
            ),
            xgb_probability_matrix,
        ]
    ).mean(axis=0)

    # ========================================================
    # SELECT ENSEMBLE MODE
    # ========================================================

    if ensemble_mode == "algorithm1":

        # Literal Algorithm 1:
        # average only XGBoost hard predictions.
        final_probability = (
            algorithm1_probability
        )

    elif ensemble_mode == "equation3":

        # Literal published Equation 3 interpretation.
        final_probability = (
            equation3_probability
        )

    elif ensemble_mode == "prose":

        # Diagnostic probability-based interpretation.
        final_probability = (
            prose_probability
        )

    else:

        raise ValueError(
            "ensemble_mode must be "
            "'algorithm1', 'equation3', or 'prose'."
        )

    # ========================================================
    # FINAL PREDICTIONS
    # ========================================================

    final_prediction = (
        final_probability >= threshold
    ).astype(int)

    metrics = calculate_metrics(
        y_target,
        final_probability,
        threshold=threshold,
    )

    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {
        "final_probability": (
            final_probability
        ),

        "final_prediction": (
            final_prediction
        ),

        "algorithm1_probability": (
            algorithm1_probability
        ),

        "equation3_probability": (
            equation3_probability
        ),

        "prose_probability": (
            prose_probability
        ),

        "mt_mean_probability": (
            mt_mean_probability
        ),

        "mt_last_probability": (
            mt_last_probability
        ),

        "mt_last_prediction": (
            mt_last_prediction
        ),

        "probability_average_probability": (
            probability_average_probability
        ),

        "xgb_prediction_matrix": (
            xgb_prediction_matrix
        ),

        "xgb_probability_matrix": (
            xgb_probability_matrix
        ),

        "iteration_rows": (
            iteration_rows
        ),

        "metrics": metrics,

        "ensemble_mode": (
            ensemble_mode
        ),
    }
