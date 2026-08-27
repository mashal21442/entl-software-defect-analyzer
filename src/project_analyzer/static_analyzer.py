from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


SECRET_PATTERNS = [
    re.compile(
        r'''(?ix)
        \b(password|passwd|pwd|secret|api[_-]?key|access[_-]?key|token)
        \b
        \s*[:=]\s*
        ["'][^"']{4,}["']
        '''
    ),
]

TODO_PATTERN = re.compile(r"\b(TODO|FIXME|HACK|XXX|BUG)\b", re.IGNORECASE)

DANGEROUS_PATTERNS = [
    (
        re.compile(r"\beval\s*\("),
        "dangerous_eval",
        "Use of eval() can execute untrusted code.",
        "high",
    ),
    (
        re.compile(r"\bexec\s*\("),
        "dangerous_exec",
        "Use of exec() can execute arbitrary code.",
        "high",
    ),
    (
        re.compile(r"\bos\.system\s*\("),
        "shell_execution",
        "os.system() executes shell commands and should be reviewed carefully.",
        "high",
    ),
    (
        re.compile(r"\bsubprocess\.(?:Popen|run|call|check_output|check_call)\s*\("),
        "subprocess_execution",
        "Subprocess execution found. Validate any user-controlled command input.",
        "medium",
    ),
]

DEBUG_PATTERNS = {
    "python": [
        re.compile(r"^\s*print\s*\("),
        re.compile(r"\bbreakpoint\s*\("),
    ],
    "javascript": [
        re.compile(r"\bconsole\.(?:log|debug)\s*\("),
        re.compile(r"\bdebugger\s*;?"),
    ],
    "typescript": [
        re.compile(r"\bconsole\.(?:log|debug)\s*\("),
        re.compile(r"\bdebugger\s*;?"),
    ],
    "tsx": [
        re.compile(r"\bconsole\.(?:log|debug)\s*\("),
        re.compile(r"\bdebugger\s*;?"),
    ],
    "java": [
        re.compile(r"\bSystem\.out\.print(?:ln)?\s*\("),
    ],
    "c_sharp": [
        re.compile(r"\bConsole\.Write(?:Line)?\s*\("),
    ],
    "php": [
        re.compile(r"\bvar_dump\s*\("),
        re.compile(r"\bprint_r\s*\("),
    ],
}

EXCEPTION_START_PATTERNS = {
    "python": re.compile(r"^\s*except\b.*:\s*$"),
    "javascript": re.compile(r"^\s*catch\s*\([^)]*\)\s*\{\s*$"),
    "typescript": re.compile(r"^\s*catch\s*\([^)]*\)\s*\{\s*$"),
    "tsx": re.compile(r"^\s*catch\s*\([^)]*\)\s*\{\s*$"),
    "java": re.compile(r"^\s*catch\s*\([^)]*\)\s*\{\s*$"),
    "kotlin": re.compile(r"^\s*catch\s*\([^)]*\)\s*\{\s*$"),
    "c_sharp": re.compile(r"^\s*catch\s*(?:\([^)]*\))?\s*\{\s*$"),
    "cpp": re.compile(r"^\s*catch\s*\([^)]*\)\s*\{\s*$"),
    "php": re.compile(r"^\s*catch\s*\([^)]*\)\s*\{\s*$"),
}

SQL_CONCAT_PATTERN = re.compile(
    r'''(?ix)
    \b(SELECT|INSERT|UPDATE|DELETE)\b
    .*
    (\+|\.concat\s*\(|\$\{)
    '''
)


def _read_lines(path: Path) -> list[str]:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=encoding).splitlines()
        except UnicodeDecodeError:
            continue
        except OSError:
            return []
    return []


def _issue(
    *,
    path: str,
    line: int,
    rule: str,
    severity: str,
    message: str,
    code: str = "",
) -> dict[str, Any]:
    return {
        "path": path,
        "line": int(line),
        "rule": rule,
        "severity": severity,
        "message": message,
        "code": code.strip()[:300],
        "source": "builtin",
    }


def _detect_empty_exception(
    lines: list[str],
    language: str,
    relative_path: str,
) -> list[dict]:
    issues: list[dict] = []
    pattern = EXCEPTION_START_PATTERNS.get(language)

    if pattern is None:
        return issues

    for index, line in enumerate(lines):
        if not pattern.search(line):
            continue

        start_line = index + 1

        if language == "python":
            base_indent = len(line) - len(line.lstrip())
            body_found = False
            meaningful_found = False

            for following in lines[index + 1:index + 8]:
                if not following.strip():
                    continue

                indent = len(following) - len(following.lstrip())

                if indent <= base_indent:
                    break

                body_found = True
                stripped = following.strip()

                if stripped not in {"pass", "..."} and not stripped.startswith("#"):
                    meaningful_found = True
                    break

            if body_found and not meaningful_found:
                issues.append(
                    _issue(
                        path=relative_path,
                        line=start_line,
                        rule="empty_exception_handler",
                        severity="medium",
                        message="Exception handler appears empty or only contains pass.",
                        code=line,
                    )
                )
        else:
            block_text = "\n".join(lines[index:index + 6])
            compact = re.sub(r"\s+", " ", block_text)
            match = re.search(
                r"catch\s*\([^)]*\)\s*\{\s*\}",
                compact,
                re.IGNORECASE,
            )

            if match:
                issues.append(
                    _issue(
                        path=relative_path,
                        line=start_line,
                        rule="empty_exception_handler",
                        severity="medium",
                        message="Empty catch block can hide failures.",
                        code=line,
                    )
                )

    return issues


def analyze_file_static(
    file_path: str | Path,
    language: str,
    *,
    project_root: str | Path | None = None,
) -> list[dict]:
    path = Path(file_path)

    if project_root is not None:
        try:
            relative_path = str(
                path.resolve().relative_to(Path(project_root).resolve())
            ).replace("\\", "/")
        except ValueError:
            relative_path = path.name
    else:
        relative_path = path.name

    lines = _read_lines(path)
    issues: list[dict] = []

    issues.extend(
        _detect_empty_exception(
            lines,
            language,
            relative_path,
        )
    )

    debug_patterns = DEBUG_PATTERNS.get(language, [])

    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()

        if not stripped:
            continue

        if len(line) > 160:
            issues.append(
                _issue(
                    path=relative_path,
                    line=line_number,
                    rule="very_long_line",
                    severity="low",
                    message="Line is longer than 160 characters.",
                    code=line,
                )
            )

        if TODO_PATTERN.search(line):
            issues.append(
                _issue(
                    path=relative_path,
                    line=line_number,
                    rule="todo_fixme",
                    severity="low",
                    message="Unresolved TODO/FIXME-style marker found.",
                    code=line,
                )
            )

        if any(pattern.search(line) for pattern in SECRET_PATTERNS):
            issues.append(
                _issue(
                    path=relative_path,
                    line=line_number,
                    rule="possible_hardcoded_secret",
                    severity="high",
                    message="Possible hard-coded credential or secret.",
                    code=line,
                )
            )

        for pattern, rule, message, severity in DANGEROUS_PATTERNS:
            if pattern.search(line):
                issues.append(
                    _issue(
                        path=relative_path,
                        line=line_number,
                        rule=rule,
                        severity=severity,
                        message=message,
                        code=line,
                    )
                )

        if any(pattern.search(line) for pattern in debug_patterns):
            issues.append(
                _issue(
                    path=relative_path,
                    line=line_number,
                    rule="debug_statement",
                    severity="low",
                    message="Debug/output statement found in source code.",
                    code=line,
                )
            )

        if language in {
            "javascript",
            "typescript",
            "tsx",
            "java",
            "c_sharp",
            "php",
            "python",
        } and SQL_CONCAT_PATTERN.search(line):
            issues.append(
                _issue(
                    path=relative_path,
                    line=line_number,
                    rule="possible_sql_injection",
                    severity="high",
                    message=(
                        "SQL statement appears to be built using string concatenation. "
                        "Use parameterized queries."
                    ),
                    code=line,
                )
            )

    unique = []
    seen = set()

    for item in issues:
        key = (
            item["path"],
            item["line"],
            item["rule"],
            item["message"],
        )
        if key not in seen:
            seen.add(key)
            unique.append(item)

    return unique


def analyze_project_static(
    project_root: str | Path,
    detected_files: list[dict],
) -> list[dict]:
    project_root = Path(project_root)
    issues: list[dict] = []

    for item in detected_files:
        relative_path = item["path"]
        language = item["language"]

        issues.extend(
            analyze_file_static(
                project_root / relative_path,
                language,
                project_root=project_root,
            )
        )

    severity_order = {"high": 0, "medium": 1, "low": 2}

    issues.sort(
        key=lambda item: (
            severity_order.get(item["severity"], 9),
            item["path"].lower(),
            item["line"],
        )
    )

    return issues


def semgrep_available() -> bool:
    return shutil.which("semgrep") is not None


def run_semgrep(
    project_root: str | Path,
    *,
    timeout_seconds: int = 180,
) -> list[dict]:
    if not semgrep_available():
        return []

    project_root = Path(project_root)

    command = [
        "semgrep",
        "scan",
        "--config",
        "auto",
        "--json",
        "--quiet",
        str(project_root),
    ]

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []

    if not completed.stdout.strip():
        return []

    try:
        import json
        payload = json.loads(completed.stdout)
    except Exception:
        return []

    issues = []

    for result in payload.get("results", []):
        extra = result.get("extra", {})
        start = result.get("start", {})
        raw_path = Path(result.get("path", ""))

        try:
            relative_path = str(
                raw_path.resolve().relative_to(project_root.resolve())
            ).replace("\\", "/")
        except Exception:
            relative_path = raw_path.name or str(raw_path)

        severity_raw = str(extra.get("severity", "WARNING")).upper()
        severity_map = {
            "ERROR": "high",
            "WARNING": "medium",
            "INFO": "low",
        }

        issues.append(
            {
                "path": relative_path,
                "line": int(start.get("line", 0) or 0),
                "rule": result.get("check_id", "semgrep"),
                "severity": severity_map.get(severity_raw, "medium"),
                "message": extra.get("message", "Semgrep issue"),
                "code": str(extra.get("lines", ""))[:300],
                "source": "semgrep",
            }
        )

    return issues


def analyze_project_with_optional_semgrep(
    project_root: str | Path,
    detected_files: list[dict],
) -> dict:
    builtin_issues = analyze_project_static(
        project_root,
        detected_files,
    )

    semgrep_issues = run_semgrep(project_root)
    all_issues = builtin_issues + semgrep_issues

    counts = {"high": 0, "medium": 0, "low": 0}

    for item in all_issues:
        severity = item.get("severity", "medium")
        if severity in counts:
            counts[severity] += 1

    return {
        "issues": all_issues,
        "issue_count": len(all_issues),
        "high_count": counts["high"],
        "medium_count": counts["medium"],
        "low_count": counts["low"],
        "semgrep_used": semgrep_available(),
    }
