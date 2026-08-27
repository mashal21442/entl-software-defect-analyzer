from pathlib import Path
import csv
import sys

from src.project_analyzer.zip_handler import extract_project_zip
from src.project_analyzer.language_detector import scan_project_languages
from src.project_analyzer.universal_metrics import analyze_project_metrics


def main():
    if len(sys.argv) != 2:
        print("Usage:")
        print("python -m src.project_analyzer.test_metrics <project.zip>")
        raise SystemExit(1)

    zip_path = Path(sys.argv[1])

    project_dir, zip_info = extract_project_zip(
        zip_path,
        Path("analysis_results") / "extracted"
    )

    language_info = scan_project_languages(project_dir)

    metrics = analyze_project_metrics(
        project_dir,
        language_info["files"]
    )

    output_dir = Path("analysis_results") / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / f"{project_dir.name}_metrics.csv"

    if metrics:
        fieldnames = list(metrics[0].keys())

        with output_csv.open(
            "w",
            newline="",
            encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames
            )

            writer.writeheader()

            for row in metrics:
                row = dict(row)

                if isinstance(row.get("risk_flags"), list):
                    row["risk_flags"] = ";".join(
                        row["risk_flags"]
                    )

                writer.writerow(row)

    print()
    print("=" * 75)
    print("PROJECT METRICS ANALYSIS")
    print("=" * 75)

    print(f"ZIP files:             {zip_info['file_count']}")
    print(
        f"Detected source files: "
        f"{language_info['detected_file_count']}"
    )
    print(f"Metric rows generated: {len(metrics)}")

    print()

    for row in metrics:

        flags = ", ".join(
            row.get("risk_flags", [])
        ) or "none"

        print("-" * 75)

        print(f"File:        {row['path']}")
        print(f"Language:    {row['language']}")
        print(f"LOC:         {row['loc']}")
        print(f"Functions:   {row['function_count']}")
        print(f"Classes:     {row['class_count']}")
        print(f"Imports:     {row['import_count']}")
        print(f"Branches:    {row['branch_count']}")
        print(f"Loops:       {row['loop_count']}")
        print(
            f"Complexity:  "
            f"{row['complexity_proxy']}"
        )
        print(
            f"Max nesting: "
            f"{row['max_nesting']}"
        )
        print(
            f"TODO/FIXME:  "
            f"{row['todo_fixme_count']}"
        )
        print(f"Risk flags:  {flags}")

    print()
    print("=" * 75)
    print("CSV OUTPUT")
    print("=" * 75)
    print(output_csv)


if __name__ == "__main__":
    main()
