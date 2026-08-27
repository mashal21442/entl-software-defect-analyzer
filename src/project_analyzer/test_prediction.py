from __future__ import annotations

import json
import sys
from pathlib import Path

from src.project_analyzer.defect_predictor import (
    predict_project_defect_risk,
)
from src.project_analyzer.language_detector import (
    scan_project_languages,
)
from src.project_analyzer.universal_metrics import (
    analyze_project_metrics,
)
from src.project_analyzer.zip_handler import (
    extract_project_zip,
)


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage:")
        print(
            "python -m src.project_analyzer.test_prediction "
            "<project.zip>"
        )
        raise SystemExit(1)

    zip_path = Path(sys.argv[1])

    repo_root = Path.cwd()

    project_dir, _ = extract_project_zip(
        zip_path,
        repo_root
        / "analysis_results"
        / "extracted",
    )

    language_info = scan_project_languages(
        project_dir
    )

    metrics_rows = analyze_project_metrics(
        project_dir,
        language_info["files"],
    )

    print()
    print("=" * 78)
    print("ENTL PROJECT DEFECT-RISK PREDICTION")
    print("=" * 78)
    print(f"Detected files: {len(metrics_rows)}")
    print()
    print(
        "Training/running ENTL inference. "
        "This can take a little while..."
    )

    # Fast test settings.
    # The web app can later use 20/20/20 for the fuller analysis.
    result = predict_project_defect_risk(
        repo_root,
        metrics_rows,
        encoder_epochs=5,
        nn_epochs=5,
        iterations=2,
    )

    print()
    print("=" * 78)
    print("RISK RESULTS")
    print("=" * 78)

    for item in result["predictions"]:
        print("-" * 78)
        print(f"File:       {item['path']}")
        print(f"Language:   {item['language']}")
        print(
            f"ENTL score: {item['entl_risk_score']:.2f}%"
        )
        print(
            "Vote score: "
            f"{item['algorithm1_vote_percent']:.2f}%"
        )
        print(
            "Relative:   "
            f"{item['relative_risk_percentile']:.2f} percentile"
        )
        print(f"Risk level: {item['risk_level']}")
        print(f"Prediction: {item['prediction']}")

    output_dir = (
        repo_root
        / "analysis_results"
        / "predictions"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / f"{project_dir.name}_prediction.json"
    )

    output_path.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("JSON OUTPUT")
    print("=" * 78)
    print(output_path)


if __name__ == "__main__":
    main()
