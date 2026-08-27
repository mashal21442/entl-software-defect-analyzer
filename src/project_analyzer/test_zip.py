from pathlib import Path
import json
import sys

from src.project_analyzer.zip_handler import extract_project_zip
from src.project_analyzer.language_detector import scan_project_languages


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage:")
        print("python -m src.project_analyzer.test_zip <project.zip>")
        raise SystemExit(1)

    zip_path = Path(sys.argv[1])

    project_dir, zip_info = extract_project_zip(
        zip_path,
        Path("analysis_results") / "extracted",
    )

    language_info = scan_project_languages(project_dir)

    print()
    print("=" * 70)
    print("ZIP VALIDATION")
    print("=" * 70)
    print(json.dumps(zip_info, indent=2))

    print()
    print("=" * 70)
    print("LANGUAGES DETECTED")
    print("=" * 70)

    if not language_info["languages"]:
        print("No recognized source-code languages found.")
    else:
        for language, count in language_info["languages"].items():
            print(f"{language:20} {count:6} files")

    print()
    print(f"Detected source files: {language_info['detected_file_count']}")
    print(f"Unknown text files:    {language_info['unknown_file_count']}")
    print()
    print(f"Extracted project: {project_dir}")


if __name__ == "__main__":
    main()
