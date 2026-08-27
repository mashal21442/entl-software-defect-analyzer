from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data_io import prepare_dataset
from src.dataset_registry import DATASETS, raw_path


def validate(project_root: Path) -> int:
    missing = []
    rows = []

    for key, spec in DATASETS.items():
        path = raw_path(project_root, key)

        if not path.exists():
            missing.append(str(path))
            rows.append(
                {
                    "dataset": key,
                    "status": "MISSING",
                    "path": str(path),
                    "rows": "",
                    "bugs": "",
                    "features": "",
                    "warnings": "",
                }
            )
            continue

        try:
            prepared = prepare_dataset(path, spec)
            rows.append(
                {
                    "dataset": key,
                    "status": "READY",
                    "path": str(path),
                    "rows": prepared.raw_rows,
                    "bugs": int(prepared.y.sum()),
                    "features": prepared.X.shape[1],
                    "warnings": " | ".join(prepared.warnings),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "dataset": key,
                    "status": "ERROR",
                    "path": str(path),
                    "rows": "",
                    "bugs": "",
                    "features": "",
                    "warnings": str(exc),
                }
            )

    output = pd.DataFrame(rows)
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(results_dir / "dataset_validation.csv", index=False)

    print(output.to_string(index=False))

    if missing:
        print("\nMissing datasets:")
        for item in missing:
            print(" -", item)
        return 1

    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    raise SystemExit(validate(Path(args.root).resolve()))


if __name__ == "__main__":
    main()
