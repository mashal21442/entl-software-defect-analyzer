from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.dataset_registry import DatasetSpec


# ============================================================
# GENERIC LABEL NAMES
# ============================================================

GENERIC_LABEL_CANDIDATES = (
    "RealBug",
    "realbug",
    "bug",
    "bugs",
    "buggy",
    "defect",
    "defects",
    "defective",
    "class",
    "label",
    "target",
)


# ============================================================
# PREPARED DATASET
# ============================================================

@dataclass
class PreparedDataset:
    X: pd.DataFrame
    y: pd.Series
    scaler: StandardScaler
    label_column: str
    dropped_columns: list[str]
    warnings: list[str]
    raw_rows: int
    raw_columns: int


# ============================================================
# NORMALIZE COLUMN NAME
# ============================================================

def _normalise_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace(":", "")
    )


# ============================================================
# DETECT LABEL COLUMN
# ============================================================

def detect_label_column(
    df: pd.DataFrame,
    candidates: Iterable[str]
) -> str:

    by_normalised = {
        _normalise_name(column): column
        for column in df.columns
    }

    all_candidates = (
        tuple(candidates)
        + GENERIC_LABEL_CANDIDATES
    )

    for candidate in all_candidates:

        key = _normalise_name(candidate)

        if key in by_normalised:
            return by_normalised[key]

    raise ValueError(
        "Could not detect the defect label column.\n"
        f"Available columns: {list(df.columns)}"
    )


# ============================================================
# CONVERT LABEL TO 0 / 1
# ============================================================

def convert_binary_labels(
    series: pd.Series
) -> pd.Series:

    # --------------------------------------------------------
    # Boolean
    # --------------------------------------------------------

    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)

    # --------------------------------------------------------
    # Numeric labels
    #
    # 0   = clean
    # > 0 = defective
    # --------------------------------------------------------

    if pd.api.types.is_numeric_dtype(series):

        numeric = pd.to_numeric(
            series,
            errors="coerce"
        )

        return (
            numeric > 0
        ).astype("Int64")

    # --------------------------------------------------------
    # Text labels
    # --------------------------------------------------------

    clean = {
        "0",
        "false",
        "f",
        "no",
        "n",
        "clean",
        "nondefective",
        "non-defective",
        "notbuggy",
    }

    defective = {
        "1",
        "true",
        "t",
        "yes",
        "y",
        "buggy",
        "defective",
        "bug",
        "bugs",
    }

    output = []

    for value in series:

        if pd.isna(value):
            output.append(pd.NA)
            continue

        text = (
            str(value)
            .strip()
            .lower()
            .replace(" ", "")
        )

        # Handle AEEEM byte-style labels such as b'1' and b'0'
        if text.startswith("b'") and text.endswith("'"):
            text = text[2:-1]

        elif text.startswith('b"') and text.endswith('"'):
            text = text[2:-1]

        if text in clean:

            output.append(0)

        elif text in defective:

            output.append(1)

        else:

            try:

                numeric_value = float(text)

                output.append(
                    1 if numeric_value > 0 else 0
                )

            except ValueError as exc:

                raise ValueError(
                    f"Unknown label value: {value!r}"
                ) from exc

    return pd.Series(
        output,
        index=series.index,
        dtype="Int64"
    )


# ============================================================
# REMOVE METADATA + NON-NUMERIC FEATURES
# ============================================================

def _drop_metadata_and_make_numeric(
    X: pd.DataFrame,
    explicit_drop: Iterable[str],
) -> tuple[pd.DataFrame, list[str]]:

    X = X.copy()

    dropped: list[str] = []

    # --------------------------------------------------------
    # Explicit columns to remove
    # --------------------------------------------------------

    explicit_by_normalised = {
        _normalise_name(column)
        for column in explicit_drop
    }

    for column in list(X.columns):

        if (
            _normalise_name(column)
            in explicit_by_normalised
        ):

            X = X.drop(
                columns=[column]
            )

            dropped.append(column)

    # --------------------------------------------------------
    # Convert usable columns to numeric
    # --------------------------------------------------------

    for column in list(X.columns):

        if pd.api.types.is_numeric_dtype(
            X[column]
        ):
            continue

        converted = pd.to_numeric(
            X[column],
            errors="coerce"
        )

        numeric_ratio = (
            converted.notna().mean()
        )

        # Keep column if at least 80% is numeric
        if numeric_ratio >= 0.80:

            X[column] = converted

        else:

            X = X.drop(
                columns=[column]
            )

            dropped.append(column)

    return X, dropped


# ============================================================
# LOAD CSV
# ============================================================

def load_csv_auto_separator(
    path: Path
) -> pd.DataFrame:
    """
    Automatically detect CSV separator.

    Supports:
    ,
    ;
    tab
    """

    try:

        df = pd.read_csv(
            path,
            sep=None,
            engine="python",
            skipinitialspace=True
        )

    except Exception as exc:

        raise ValueError(
            f"Could not read dataset:\n{path}\n\n"
            f"Reason: {exc}"
        ) from exc

    # --------------------------------------------------------
    # Clean column names
    # --------------------------------------------------------

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    # --------------------------------------------------------
    # Remove empty columns caused by trailing separators
    # --------------------------------------------------------

    unnamed_columns = [
        column
        for column in df.columns
        if str(column).startswith(
            "Unnamed"
        )
    ]

    if unnamed_columns:

        df = df.drop(
            columns=unnamed_columns
        )

    # --------------------------------------------------------
    # Remove completely empty columns
    # --------------------------------------------------------

    df = df.dropna(
        axis=1,
        how="all"
    )

    return df


# ============================================================
# PREPARE DATASET
# ============================================================

def prepare_dataset(
    path: Path,
    spec: DatasetSpec
) -> PreparedDataset:

    # --------------------------------------------------------
    # CHECK FILE
    # --------------------------------------------------------

    if not path.exists():

        raise FileNotFoundError(
            f"Dataset file not found:\n{path}"
        )

    # --------------------------------------------------------
    # LOAD CSV
    # --------------------------------------------------------

    df = load_csv_auto_separator(
        path
    )

    raw_rows, raw_columns = (
        df.shape
    )

    warnings: list[str] = []

    print()
    print(
        f"Loaded: {spec.key}"
    )

    print(
        f"Rows: {raw_rows}"
    )

    print(
        f"Columns: {raw_columns}"
    )

    # --------------------------------------------------------
    # DETECT LABEL
    # --------------------------------------------------------

    label_column = (
        detect_label_column(
            df,
            spec.label_candidates
        )
    )

    print(
        f"Detected label: "
        f"{label_column}"
    )

    # --------------------------------------------------------
    # CONVERT LABELS
    # --------------------------------------------------------

    labels = convert_binary_labels(
        df[label_column]
    )

    # --------------------------------------------------------
    # REMOVE ROWS WITH MISSING LABELS
    # --------------------------------------------------------

    valid = labels.notna()

    if (~valid).any():

        missing_count = int(
            (~valid).sum()
        )

        warnings.append(
            f"Dropped {missing_count} rows "
            f"with missing labels."
        )

        df = df.loc[
            valid
        ].copy()

        labels = labels.loc[
            valid
        ]

    # --------------------------------------------------------
    # FINAL LABEL VECTOR
    # --------------------------------------------------------

    y = (
        labels
        .astype(int)
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # FEATURE MATRIX
    # --------------------------------------------------------

    X = (
        df
        .drop(
            columns=[label_column]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # REMOVE METADATA / STRING COLUMNS
    # --------------------------------------------------------

    X, dropped = (
        _drop_metadata_and_make_numeric(
            X,
            spec.drop_if_present
        )
    )

    # --------------------------------------------------------
    # REPLACE INFINITY
    # --------------------------------------------------------

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # --------------------------------------------------------
    # REMOVE COMPLETELY EMPTY FEATURES
    # --------------------------------------------------------

    all_nan = [
        column
        for column in X.columns
        if X[column].isna().all()
    ]

    if all_nan:

        X = X.drop(
            columns=all_nan
        )

        dropped.extend(
            all_nan
        )

    # --------------------------------------------------------
    # FILL MISSING VALUES WITH MEDIAN
    # --------------------------------------------------------

    for column in X.columns:

        if X[column].isna().any():

            median = (
                X[column]
                .median()
            )

            X[column] = (
                X[column]
                .fillna(median)
            )

   

    # --------------------------------------------------------
    # VERIFY FEATURES EXIST
    # --------------------------------------------------------

    if X.empty:

        raise ValueError(
            f"No numeric feature columns "
            f"remain for {spec.key}."
        )

    # --------------------------------------------------------
    # Z-SCORE NORMALIZATION
    # --------------------------------------------------------

    scaler = StandardScaler()

    scaled = scaler.fit_transform(
        X
    )

    X_scaled = pd.DataFrame(
        scaled,
        columns=X.columns,
        index=X.index
    )

    # --------------------------------------------------------
    # DATASET VALIDATION
    # --------------------------------------------------------

    actual_bugs = int(
        y.sum()
    )

    actual_features = (
        X_scaled.shape[1]
    )

    # --------------------------------------------------------
    # ROW COUNT
    # --------------------------------------------------------

    if (
        raw_rows
        != spec.expected_entries
    ):

        warnings.append(
            f"Row count is {raw_rows}, "
            f"while the paper lists "
            f"{spec.expected_entries}. "
            f"You may have a different "
            f"dataset version."
        )

    # --------------------------------------------------------
    # DEFECT COUNT
    # --------------------------------------------------------

    if (
        actual_bugs
        != spec.expected_bugs
    ):

        warnings.append(
            f"Defect count is {actual_bugs}, "
            f"while the paper lists "
            f"{spec.expected_bugs}. "
            f"You may have a different "
            f"dataset version."
        )

    # --------------------------------------------------------
    # FEATURE COUNT
    # --------------------------------------------------------

    if (
        actual_features
        != spec.expected_metrics
    ):

        warnings.append(
            f"Usable numeric feature count is "
            f"{actual_features}, while the "
            f"paper lists "
            f"{spec.expected_metrics} metrics. "
            f"This suggests a different dataset "
            f"version or representation."
        )

    # --------------------------------------------------------
    # DISPLAY INFORMATION
    # --------------------------------------------------------

    print(
        f"Numeric features: "
        f"{actual_features}"
    )

    print(
        f"Clean modules: "
        f"{int((y == 0).sum())}"
    )

    print(
        f"Defective modules: "
        f"{int((y == 1).sum())}"
    )

    if dropped:

        print(
            f"Dropped columns: "
            f"{dropped}"
        )

    if warnings:

        print()
        print(
            "DATASET WARNINGS"
        )

        print(
            "-" * 60
        )

        for warning in warnings:

            print(
                f"WARNING: {warning}"
            )

    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    return PreparedDataset(
        X=X_scaled,
        y=y,
        scaler=scaler,
        label_column=label_column,
        dropped_columns=dropped,
        warnings=warnings,
        raw_rows=raw_rows,
        raw_columns=raw_columns,
    )