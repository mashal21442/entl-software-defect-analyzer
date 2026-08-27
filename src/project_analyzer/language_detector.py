from __future__ import annotations

from collections import Counter
from pathlib import Path

try:
    from tree_sitter_language_pack import detect_language
except ImportError:
    detect_language = None


IGNORED_DIRECTORIES = {
    ".git",
    ".hg",
    ".svn",
    ".idea",
    ".vscode",
    "__pycache__",
    "node_modules",
    "vendor",
    "vendors",
    "dist",
    "build",
    "target",
    "coverage",
    ".next",
    ".nuxt",
    ".venv",
    "venv",
    "env",
    "site-packages",
    "Pods",
    "DerivedData",
}

IGNORED_FILENAMES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "Pipfile.lock",
}

BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".ico",
    ".pdf", ".zip", ".rar", ".7z", ".tar", ".gz",
    ".exe", ".dll", ".so", ".dylib", ".class", ".jar", ".war",
    ".mp3", ".wav", ".mp4", ".mov", ".avi",
    ".woff", ".woff2", ".ttf", ".otf",
    ".db", ".sqlite", ".sqlite3",
    ".pyc", ".pyo",
}

# Fallback mapping used when tree-sitter-language-pack does not
# identify a file or is not installed.
FALLBACK_LANGUAGE_MAP = {
    ".py": "python",
    ".pyw": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".cs": "c_sharp",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".scala": "scala",
    ".dart": "dart",
    ".lua": "lua",
    ".r": "r",
    ".R": "r",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".ps1": "powershell",
    ".sql": "sql",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "scss",
    ".sass": "scss",
    ".less": "less",
    ".vue": "vue",
    ".svelte": "svelte",
    ".xml": "xml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".toml": "toml",
    ".md": "markdown",
    ".gradle": "groovy",
    ".groovy": "groovy",
    ".sol": "solidity",
    ".ex": "elixir",
    ".exs": "elixir",
    ".erl": "erlang",
    ".hrl": "erlang",
    ".fs": "fsharp",
    ".fsx": "fsharp",
    ".vb": "visual_basic",
    ".pl": "perl",
    ".pm": "perl",
    ".clj": "clojure",
    ".cljs": "clojure",
}


def should_ignore_path(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True

    if any(part in IGNORED_DIRECTORIES for part in relative.parts):
        return True

    if path.name in IGNORED_FILENAMES:
        return True

    if path.suffix.lower() in BINARY_EXTENSIONS:
        return True

    return False


def looks_binary(path: Path, sample_size: int = 4096) -> bool:
    """
    Lightweight binary-file check.
    """
    try:
        with path.open("rb") as handle:
            sample = handle.read(sample_size)
    except OSError:
        return True

    if not sample:
        return False

    if b"\x00" in sample:
        return True

    return False


def detect_file_language(path: str | Path) -> str | None:
    path = Path(path)

    if detect_language is not None:
        try:
            detected = detect_language(str(path))
            if detected:
                return str(detected)
        except Exception:
            pass

    suffix = path.suffix

    if suffix in FALLBACK_LANGUAGE_MAP:
        return FALLBACK_LANGUAGE_MAP[suffix]

    lower_suffix = suffix.lower()
    return FALLBACK_LANGUAGE_MAP.get(lower_suffix)


def scan_project_languages(
    project_root: str | Path,
) -> dict:
    """
    Scan an extracted project and detect programming / markup languages.

    Returns a JSON-serializable dictionary.
    """
    project_root = Path(project_root).resolve()

    if not project_root.exists():
        raise FileNotFoundError(
            f"Project directory not found: {project_root}"
        )

    language_counts: Counter[str] = Counter()
    detected_files = []
    unknown_files = []

    for path in project_root.rglob("*"):
        if not path.is_file():
            continue

        if should_ignore_path(path, project_root):
            continue

        if looks_binary(path):
            continue

        language = detect_file_language(path)

        relative_path = str(
            path.relative_to(project_root)
        ).replace("\\", "/")

        if language:
            language_counts[language] += 1
            detected_files.append(
                {
                    "path": relative_path,
                    "language": language,
                    "extension": path.suffix.lower(),
                }
            )
        else:
            unknown_files.append(relative_path)

    detected_files.sort(
        key=lambda item: item["path"].lower()
    )

    return {
        "languages": dict(
            language_counts.most_common()
        ),
        "detected_file_count": len(detected_files),
        "unknown_file_count": len(unknown_files),
        "files": detected_files,
        "unknown_files": unknown_files,
    }
