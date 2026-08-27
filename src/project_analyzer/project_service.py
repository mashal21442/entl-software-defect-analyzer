from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from src.project_analyzer.defect_predictor import predict_project_defect_risk
from src.project_analyzer.language_detector import scan_project_languages
from src.project_analyzer.static_analyzer import analyze_project_with_optional_semgrep
from src.project_analyzer.universal_metrics import analyze_project_metrics
from src.project_analyzer.zip_handler import extract_project_zip


def analyze_uploaded_project(
    repo_root: str | Path,
    zip_path: str | Path,
    *,
    analysis_mode: str = "quick",
) -> dict:
    """
    Complete uploaded-project analysis pipeline.

    quick:
        encoder epochs = 5
        NN epochs      = 5
        ENTL iterations= 2

    full:
        encoder epochs = 20
        NN epochs      = 20
        ENTL iterations= 20
    """
    repo_root = Path(repo_root).resolve()
    zip_path = Path(zip_path).resolve()

    if analysis_mode not in {"quick", "full"}:
        raise ValueError("analysis_mode must be 'quick' or 'full'.")

    if analysis_mode == "full":
        encoder_epochs = 20
        nn_epochs = 20
        iterations = 20
    else:
        encoder_epochs = 5
        nn_epochs = 5
        iterations = 2

    extracted_root = (
        repo_root
        / "analysis_results"
        / "extracted"
    )

    project_dir, zip_info = extract_project_zip(
        zip_path,
        extracted_root,
    )

    language_info = scan_project_languages(
        project_dir
    )

    if language_info["detected_file_count"] == 0:
        raise ValueError(
            "No supported source-code files were detected inside the ZIP."
        )

    metrics_rows = analyze_project_metrics(
        project_dir,
        language_info["files"],
    )

    static_result = analyze_project_with_optional_semgrep(
        project_dir,
        language_info["files"],
    )

    prediction_result = predict_project_defect_risk(
        repo_root,
        metrics_rows,
        encoder_epochs=encoder_epochs,
        nn_epochs=nn_epochs,
        iterations=iterations,
    )

    metrics_by_path = {
        row["path"]: row
        for row in metrics_rows
    }

    issues_by_path = defaultdict(list)

    for issue in static_result["issues"]:
        issues_by_path[issue["path"]].append(issue)

    predictions_by_path = {
        row["path"]: row
        for row in prediction_result["predictions"]
    }

    file_results = []

    for detected in language_info["files"]:
        path = detected["path"]

        metrics = metrics_by_path.get(
            path,
            {}
        )

        prediction = predictions_by_path.get(
            path,
            {
                "entl_risk_score": 0.0,
                "algorithm1_vote_percent": 0.0,
                "relative_risk_percentile": 0.0,
                "prediction": "not_available",
                "risk_level": "low",
            },
        )

        issues = issues_by_path.get(
            path,
            [],
        )

        high_count = sum(
            1 for item in issues
            if item.get("severity") == "high"
        )

        medium_count = sum(
            1 for item in issues
            if item.get("severity") == "medium"
        )

        low_count = sum(
            1 for item in issues
            if item.get("severity") == "low"
        )

        # Combined UI severity.
        # Static high-severity findings always force high risk.
        if high_count > 0:
            combined_level = "high"
        elif prediction["risk_level"] == "high":
            combined_level = "high"
        elif medium_count > 0:
            combined_level = "medium"
        elif prediction["risk_level"] == "medium":
            combined_level = "medium"
        else:
            combined_level = "low"

        file_results.append(
            {
                "path": path,
                "language": detected["language"],
                "risk_level": combined_level,
                "entl_risk_score": prediction[
                    "entl_risk_score"
                ],
                "algorithm1_vote_percent": prediction[
                    "algorithm1_vote_percent"
                ],
                "relative_risk_percentile": prediction[
                    "relative_risk_percentile"
                ],
                "entl_prediction": prediction[
                    "prediction"
                ],
                "issues": issues,
                "issue_count": len(issues),
                "high_issue_count": high_count,
                "medium_issue_count": medium_count,
                "low_issue_count": low_count,
                "metrics": metrics,
            }
        )

    risk_order = {
        "high": 0,
        "medium": 1,
        "low": 2,
    }

    file_results.sort(
        key=lambda row: (
            risk_order.get(
                row["risk_level"],
                9,
            ),
            -float(
                row["entl_risk_score"]
            ),
            row["path"].lower(),
        )
    )

    high_risk_files = sum(
        1 for row in file_results
        if row["risk_level"] == "high"
    )

    medium_risk_files = sum(
        1 for row in file_results
        if row["risk_level"] == "medium"
    )

    low_risk_files = sum(
        1 for row in file_results
        if row["risk_level"] == "low"
    )

    return {
        "analysis_mode": analysis_mode,
        "project_directory": str(
            project_dir
        ),
        "zip_info": zip_info,
        "languages": language_info[
            "languages"
        ],
        "detected_file_count": language_info[
            "detected_file_count"
        ],
        "unknown_file_count": language_info[
            "unknown_file_count"
        ],
        "total_issues": static_result[
            "issue_count"
        ],
        "high_issues": static_result[
            "high_count"
        ],
        "medium_issues": static_result[
            "medium_count"
        ],
        "low_issues": static_result[
            "low_count"
        ],
        "semgrep_used": static_result[
            "semgrep_used"
        ],
        "source_dataset": prediction_result[
            "source_dataset"
        ],
        "high_risk_files": high_risk_files,
        "medium_risk_files": medium_risk_files,
        "low_risk_files": low_risk_files,
        "files": file_results,
        "important_note": (
            "ENTL scores estimate defect-proneness. "
            "Static-analysis findings identify concrete suspicious code patterns. "
            "Neither should be treated as proof that a defect definitely exists."
        ),
    }
