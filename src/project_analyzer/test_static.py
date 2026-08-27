from __future__ import annotations

import json
import sys
from pathlib import Path

from src.project_analyzer.language_detector import scan_project_languages
from src.project_analyzer.static_analyzer import (
    analyze_project_with_optional_semgrep,
)
from src.project_analyzer.zip_handler import extract_project_zip


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage:")
        print("python -m src.project_analyzer.test_static <project.zip>")
        raise SystemExit(1)

    zip_path = Path(sys.argv[1])

    project_dir, _ = extract_project_zip(
        zip_path,
        Path("analysis_results") / "extracted",
    )

    language_info = scan_project_languages(project_dir)

    result = analyze_project_with_optional_semgrep(
        project_dir,
        language_info["files"],
    )

    print()
    print("=" * 78)
    print("STATIC DEFECT / ISSUE ANALYSIS")
    print("=" * 78)

    print(f"Source files analyzed: {language_info['detected_file_count']}")
    print(f"Total issues:          {result['issue_count']}")
    print(f"High severity:         {result['high_count']}")
    print(f"Medium severity:       {result['medium_count']}")
    print(f"Low severity:          {result['low_count']}")
    print(f"Semgrep available:     {result['semgrep_used']}")
    print()

    if not result["issues"]:
        print("No issues detected in this test project.")
    else:
        for issue in result["issues"]:
            print("-" * 78)
            print(
                f"[{issue['severity'].upper()}] "
                f"{issue['path']} : line {issue['line']}"
            )
            print(f"Rule:    {issue['rule']}")
            print(f"Problem: {issue['message']}")
            if issue.get("code"):
                print(f"Code:    {issue['code']}")
            print(f"Source:  {issue['source']}")

    output_dir = Path("analysis_results") / "static"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{project_dir.name}_issues.json"
    output_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("JSON OUTPUT")
    print("=" * 78)
    print(output_path)


if __name__ == "__main__":
    main()
