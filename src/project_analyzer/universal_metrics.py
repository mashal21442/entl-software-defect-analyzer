from __future__ import annotations

import re
from pathlib import Path
from statistics import mean
from typing import Iterable


COMMENT_MARKERS = {
    "python": ("#",),
    "ruby": ("#",),
    "r": ("#",),
    "bash": ("#",),
    "powershell": ("#",),
    "perl": ("#",),
    "javascript": ("//",),
    "typescript": ("//",),
    "tsx": ("//",),
    "java": ("//",),
    "kotlin": ("//",),
    "c": ("//",),
    "cpp": ("//",),
    "c_sharp": ("//",),
    "go": ("//",),
    "rust": ("//",),
    "swift": ("//",),
    "scala": ("//",),
    "dart": ("//",),
    "php": ("//", "#"),
    "solidity": ("//",),
    "groovy": ("//",),
    "sql": ("--",),
    "lua": ("--",),
    "html": ("<!--",),
    "xml": ("<!--",),
    "css": ("/*",),
    "scss": ("//", "/*"),
    "less": ("//", "/*"),
}

FUNCTION_PATTERNS = {
    "python": [
        r"^\s*(?:async\s+)?def\s+[A-Za-z_]\w*\s*\(",
    ],
    "javascript": [
        r"^\s*(?:async\s+)?function\s+[A-Za-z_$][\w$]*\s*\(",
        r"^\s*(?:const|let|var)\s+[A-Za-z_$][\w$]*\s*=\s*(?:async\s*)?\([^)]*\)\s*=>",
        r"^\s*[A-Za-z_$][\w$]*\s*\([^)]*\)\s*\{",
    ],
    "typescript": [
        r"^\s*(?:export\s+)?(?:async\s+)?function\s+[A-Za-z_$][\w$]*\s*\(",
        r"^\s*(?:const|let|var)\s+[A-Za-z_$][\w$]*\s*=\s*(?:async\s*)?\([^)]*\)\s*=>",
        r"^\s*(?:public|private|protected|static|async|\s)*[A-Za-z_$][\w$]*\s*\([^;)]*\)\s*(?::[^{]+)?\{",
    ],
    "tsx": [
        r"^\s*(?:export\s+)?(?:async\s+)?function\s+[A-Za-z_$][\w$]*\s*\(",
        r"^\s*(?:const|let|var)\s+[A-Za-z_$][\w$]*\s*=\s*(?:async\s*)?\([^)]*\)\s*=>",
    ],
    "java": [
        r"^\s*(?:public|private|protected|static|final|synchronized|abstract|native|\s)+[\w<>\[\],.?]+\s+[A-Za-z_]\w*\s*\([^;]*\)\s*(?:throws\s+[^{]+)?\{",
    ],
    "kotlin": [
        r"^\s*(?:public|private|protected|internal|open|override|suspend|\s)*fun\s+[A-Za-z_]\w*\s*\(",
    ],
    "c": [
        r"^\s*(?!if\b|for\b|while\b|switch\b)[A-Za-z_][\w\s\*]*\s+[A-Za-z_]\w*\s*\([^;]*\)\s*\{",
    ],
    "cpp": [
        r"^\s*(?!if\b|for\b|while\b|switch\b)[A-Za-z_:~][\w:<>,~\s\*&]*\s+[A-Za-z_:~]\w*(?:::\w+)*\s*\([^;]*\)\s*(?:const\s*)?\{",
    ],
    "c_sharp": [
        r"^\s*(?:public|private|protected|internal|static|virtual|override|async|sealed|partial|\s)+[\w<>\[\],.?]+\s+[A-Za-z_]\w*\s*\([^;]*\)\s*\{",
    ],
    "go": [
        r"^\s*func\s+(?:\([^)]*\)\s*)?[A-Za-z_]\w*\s*\(",
    ],
    "rust": [
        r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+[A-Za-z_]\w*\s*\(",
    ],
    "php": [
        r"^\s*(?:public|private|protected|static|final|abstract|\s)*function\s+[A-Za-z_]\w*\s*\(",
    ],
    "ruby": [
        r"^\s*def\s+[A-Za-z_]\w*[!?=]?",
    ],
    "swift": [
        r"^\s*(?:public|private|internal|fileprivate|open|static|class|mutating|\s)*func\s+[A-Za-z_]\w*\s*\(",
    ],
    "dart": [
        r"^\s*(?:static\s+)?(?:Future<[^>]+>|[\w<>?]+)\s+[A-Za-z_]\w*\s*\([^;]*\)\s*(?:async\s*)?\{",
    ],
    "scala": [
        r"^\s*(?:private|protected|override|final|implicit|\s)*def\s+[A-Za-z_]\w*\s*\(",
    ],
    "lua": [
        r"^\s*(?:local\s+)?function\s+[A-Za-z_][\w.:]*\s*\(",
    ],
}

CLASS_PATTERNS = {
    "python": [r"^\s*class\s+[A-Za-z_]\w*"],
    "javascript": [r"^\s*(?:export\s+)?class\s+[A-Za-z_$][\w$]*"],
    "typescript": [
        r"^\s*(?:export\s+)?(?:abstract\s+)?class\s+[A-Za-z_$][\w$]*",
        r"^\s*(?:export\s+)?interface\s+[A-Za-z_$][\w$]*",
    ],
    "tsx": [
        r"^\s*(?:export\s+)?(?:abstract\s+)?class\s+[A-Za-z_$][\w$]*",
        r"^\s*(?:export\s+)?interface\s+[A-Za-z_$][\w$]*",
    ],
    "java": [r"^\s*(?:public\s+|private\s+|protected\s+|abstract\s+|final\s+)*class\s+[A-Za-z_]\w*"],
    "kotlin": [r"^\s*(?:data\s+|sealed\s+|open\s+|abstract\s+)?class\s+[A-Za-z_]\w*"],
    "c_sharp": [r"^\s*(?:public|private|protected|internal|abstract|sealed|partial|\s)*class\s+[A-Za-z_]\w*"],
    "cpp": [
        r"^\s*class\s+[A-Za-z_]\w*",
        r"^\s*struct\s+[A-Za-z_]\w*",
    ],
    "swift": [
        r"^\s*(?:public|private|internal|open|\s)*(?:class|struct|protocol)\s+[A-Za-z_]\w*",
    ],
    "php": [
        r"^\s*(?:abstract\s+|final\s+)?class\s+[A-Za-z_]\w*",
        r"^\s*interface\s+[A-Za-z_]\w*",
    ],
    "ruby": [r"^\s*class\s+[A-Za-z_:]\w*"],
    "scala": [
        r"^\s*(?:case\s+)?class\s+[A-Za-z_]\w*",
        r"^\s*(?:sealed\s+)?trait\s+[A-Za-z_]\w*",
    ],
}

IMPORT_PATTERNS = {
    "python": [r"^\s*import\s+", r"^\s*from\s+\S+\s+import\s+"],
    "javascript": [r"^\s*import\s+", r"\brequire\s*\("],
    "typescript": [r"^\s*import\s+", r"\brequire\s*\("],
    "tsx": [r"^\s*import\s+", r"\brequire\s*\("],
    "java": [r"^\s*import\s+"],
    "kotlin": [r"^\s*import\s+"],
    "c": [r"^\s*#\s*include\b"],
    "cpp": [r"^\s*#\s*include\b"],
    "c_sharp": [r"^\s*using\s+"],
    "go": [r"^\s*import\b"],
    "rust": [r"^\s*use\s+"],
    "swift": [r"^\s*import\s+"],
    "php": [r"^\s*(?:use|require|require_once|include|include_once)\b"],
    "ruby": [r"^\s*(?:require|require_relative)\b"],
    "dart": [r"^\s*import\s+"],
    "scala": [r"^\s*import\s+"],
}

BRANCH_RE = re.compile(r"\b(if|elif|else\s+if|elseif|switch|case|match|when|guard)\b")
LOOP_RE = re.compile(r"\b(for|while|foreach|forEach|do)\b")
EXCEPTION_RE = re.compile(r"\b(try|catch|except|finally|throw|throws|raise)\b")
RETURN_RE = re.compile(r"\b(return|yield)\b")
BOOL_OPERATOR_RE = re.compile(r"(&&|\|\||\band\b|\bor\b)")
TODO_RE = re.compile(r"\b(TODO|FIXME|HACK|XXX|BUG)\b", re.IGNORECASE)


def _read_text(path: Path) -> str:
    encodings = ("utf-8", "utf-8-sig", "latin-1")
    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        except OSError:
            return ""
    return ""


def _count_matches(lines: Iterable[str], patterns: list[str]) -> int:
    compiled = [re.compile(pattern) for pattern in patterns]
    count = 0
    for line in lines:
        if any(pattern.search(line) for pattern in compiled):
            count += 1
    return count


def _estimate_comment_lines(lines: list[str], language: str) -> int:
    markers = COMMENT_MARKERS.get(language, ("//", "#", "--"))
    count = 0
    in_block_comment = False

    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            continue

        if in_block_comment:
            count += 1
            if "*/" in stripped or "-->" in stripped:
                in_block_comment = False
            continue

        if stripped.startswith("/*") or stripped.startswith("<!--"):
            count += 1
            if "*/" not in stripped and "-->" not in stripped:
                in_block_comment = True
            continue

        if any(stripped.startswith(marker) for marker in markers):
            count += 1

    return count


def _estimate_python_nesting(lines: list[str]) -> int:
    max_level = 0

    for line in lines:
        if not line.strip():
            continue

        expanded = line.expandtabs(4)
        spaces = len(expanded) - len(expanded.lstrip(" "))
        level = spaces // 4
        max_level = max(max_level, level)

    return max_level


def _estimate_brace_nesting(lines: list[str]) -> int:
    current = 0
    maximum = 0

    for line in lines:
        # This is deliberately a lightweight universal estimate.
        opens = line.count("{")
        closes = line.count("}")

        current += opens
        maximum = max(maximum, current)
        current = max(0, current - closes)

    return maximum


def _risk_flags(metrics: dict) -> list[str]:
    flags = []

    if metrics["loc"] >= 500:
        flags.append("very_large_file")
    elif metrics["loc"] >= 250:
        flags.append("large_file")

    if metrics["complexity_proxy"] >= 40:
        flags.append("very_high_complexity")
    elif metrics["complexity_proxy"] >= 20:
        flags.append("high_complexity")

    if metrics["max_nesting"] >= 6:
        flags.append("deep_nesting")

    if metrics["import_count"] >= 20:
        flags.append("many_dependencies")

    if metrics["todo_fixme_count"] >= 3:
        flags.append("many_todo_fixme_markers")

    if metrics["long_line_count"] >= 10:
        flags.append("many_long_lines")

    if metrics["comment_ratio"] < 0.02 and metrics["loc"] >= 100:
        flags.append("very_low_comment_ratio")

    return flags


def analyze_source_file(
    file_path: str | Path,
    language: str,
    *,
    project_root: str | Path | None = None,
) -> dict:
    """
    Calculate common source-code metrics for one file.

    These metrics are intentionally language-agnostic so heterogeneous
    projects can be represented in one common target feature space.
    """
    path = Path(file_path)
    text = _read_text(path)
    lines = text.splitlines()

    total_lines = len(lines)
    blank_lines = sum(1 for line in lines if not line.strip())
    comment_lines = _estimate_comment_lines(lines, language)
    loc = max(0, total_lines - blank_lines - comment_lines)

    function_count = _count_matches(
        lines,
        FUNCTION_PATTERNS.get(language, []),
    )

    class_count = _count_matches(
        lines,
        CLASS_PATTERNS.get(language, []),
    )

    import_count = _count_matches(
        lines,
        IMPORT_PATTERNS.get(language, []),
    )

    branch_count = sum(len(BRANCH_RE.findall(line)) for line in lines)
    loop_count = sum(len(LOOP_RE.findall(line)) for line in lines)
    exception_count = sum(len(EXCEPTION_RE.findall(line)) for line in lines)
    return_count = sum(len(RETURN_RE.findall(line)) for line in lines)
    boolean_operator_count = sum(
        len(BOOL_OPERATOR_RE.findall(line))
        for line in lines
    )
    todo_fixme_count = len(TODO_RE.findall(text))

    complexity_proxy = (
        1
        + branch_count
        + loop_count
        + exception_count
        + boolean_operator_count
    )

    if language == "python":
        max_nesting = _estimate_python_nesting(lines)
    else:
        max_nesting = _estimate_brace_nesting(lines)

    line_lengths = [len(line) for line in lines] if lines else [0]
    average_line_length = mean(line_lengths) if line_lengths else 0.0
    max_line_length = max(line_lengths) if line_lengths else 0
    long_line_count = sum(1 for length in line_lengths if length > 120)

    comment_ratio = (
        comment_lines / max(1, loc + comment_lines)
    )

    function_density = function_count / max(1, loc)
    branch_density = branch_count / max(1, loc)
    dependency_density = import_count / max(1, loc)

    if project_root is not None:
        try:
            relative_path = str(
                path.resolve().relative_to(Path(project_root).resolve())
            ).replace("\\", "/")
        except ValueError:
            relative_path = path.name
    else:
        relative_path = str(path)

    result = {
        "path": relative_path,
        "language": language,
        "file_size_bytes": path.stat().st_size if path.exists() else 0,
        "total_lines": total_lines,
        "loc": loc,
        "blank_lines": blank_lines,
        "comment_lines": comment_lines,
        "comment_ratio": round(comment_ratio, 6),
        "function_count": function_count,
        "class_count": class_count,
        "import_count": import_count,
        "branch_count": branch_count,
        "loop_count": loop_count,
        "exception_count": exception_count,
        "return_count": return_count,
        "boolean_operator_count": boolean_operator_count,
        "complexity_proxy": complexity_proxy,
        "max_nesting": max_nesting,
        "average_line_length": round(float(average_line_length), 3),
        "max_line_length": max_line_length,
        "long_line_count": long_line_count,
        "todo_fixme_count": todo_fixme_count,
        "function_density": round(function_density, 6),
        "branch_density": round(branch_density, 6),
        "dependency_density": round(dependency_density, 6),
    }

    result["risk_flags"] = _risk_flags(result)
    return result


def analyze_project_metrics(
    project_root: str | Path,
    detected_files: list[dict],
) -> list[dict]:
    """
    Calculate metrics for all files returned by language_detector.py.
    """
    project_root = Path(project_root)

    rows = []

    for item in detected_files:
        relative_path = item["path"]
        language = item["language"]

        absolute_path = project_root / relative_path

        metrics = analyze_source_file(
            absolute_path,
            language,
            project_root=project_root,
        )
        rows.append(metrics)

    return rows


def numeric_feature_names() -> list[str]:
    """
    Numeric columns suitable for later ML/ENTL target encoding.
    """
    return [
        "file_size_bytes",
        "total_lines",
        "loc",
        "blank_lines",
        "comment_lines",
        "comment_ratio",
        "function_count",
        "class_count",
        "import_count",
        "branch_count",
        "loop_count",
        "exception_count",
        "return_count",
        "boolean_operator_count",
        "complexity_proxy",
        "max_nesting",
        "average_line_length",
        "max_line_length",
        "long_line_count",
        "todo_fixme_count",
        "function_density",
        "branch_density",
        "dependency_density",
    ]
