
from __future__ import annotations

import shutil
import stat
import uuid
import zipfile
from pathlib import Path


class ZipSecurityError(ValueError):
    """Raised when an uploaded ZIP is unsafe or exceeds configured limits."""


# ==========================================================
# ZIP LIMITS
# ==========================================================

DEFAULT_MAX_FILES = 20_000

# Maximum total extracted/uncompressed project size = 500 MB
DEFAULT_MAX_TOTAL_UNCOMPRESSED_BYTES = (
    500 * 1024 * 1024
)

# Maximum size of any single file inside the ZIP = 100 MB
DEFAULT_MAX_SINGLE_FILE_BYTES = (
    100 * 1024 * 1024
)


# ==========================================================
# CHECK SYMBOLIC LINK
# ==========================================================

def _is_symlink(
    info: zipfile.ZipInfo,
) -> bool:

    mode = info.external_attr >> 16

    return stat.S_ISLNK(mode)


# ==========================================================
# SAFE DESTINATION
# ==========================================================

def _safe_destination(
    root: Path,
    member_name: str,
) -> Path:
    """
    Resolve a ZIP member path and make sure it stays
    inside the extraction root.

    Prevents:
    - ../ path traversal
    - absolute paths
    """

    normalized = member_name.replace(
        "\\",
        "/",
    )

    # Reject Unix absolute paths
    if normalized.startswith("/"):

        raise ZipSecurityError(
            f"ZIP contains an absolute path: {member_name}"
        )

    # Reject Windows absolute paths such as C:/...
    if (
        len(normalized) >= 2
        and normalized[1] == ":"
    ):

        raise ZipSecurityError(
            f"ZIP contains an absolute Windows path: {member_name}"
        )

    destination = (
        root
        / normalized
    ).resolve()

    root_resolved = (
        root.resolve()
    )

    try:

        destination.relative_to(
            root_resolved
        )

    except ValueError as exc:

        raise ZipSecurityError(
            f"Unsafe ZIP path detected: {member_name}"
        ) from exc

    return destination


# ==========================================================
# VALIDATE ZIP
# ==========================================================

def validate_zip(
    zip_path: str | Path,
    *,
    max_files: int = DEFAULT_MAX_FILES,
    max_total_uncompressed_bytes: int = (
        DEFAULT_MAX_TOTAL_UNCOMPRESSED_BYTES
    ),
    max_single_file_bytes: int = (
        DEFAULT_MAX_SINGLE_FILE_BYTES
    ),
) -> dict:
    """
    Validate an uploaded project ZIP before extraction.

    Checks:
    - valid ZIP format
    - maximum number of files
    - maximum expanded project size
    - maximum single-file size
    - path traversal attempts
    - symbolic links
    """

    zip_path = Path(
        zip_path
    )

    # ------------------------------------------------------
    # File must exist
    # ------------------------------------------------------

    if not zip_path.exists():

        raise FileNotFoundError(
            f"ZIP file not found: {zip_path}"
        )

    # ------------------------------------------------------
    # Only ZIP extension allowed
    # ------------------------------------------------------

    if zip_path.suffix.lower() != ".zip":

        raise ZipSecurityError(
            "Only .zip project uploads are allowed."
        )

    file_count = 0
    total_size = 0
    skipped_symlinks = 0

    try:

        with zipfile.ZipFile(
            zip_path,
            "r",
        ) as archive:

            # --------------------------------------------------
            # Check ZIP corruption
            # --------------------------------------------------

            bad_member = archive.testzip()

            if bad_member is not None:

                raise ZipSecurityError(
                    "Corrupt ZIP member detected: "
                    f"{bad_member}"
                )

            # --------------------------------------------------
            # Inspect every member
            # --------------------------------------------------

            for info in archive.infolist():

                if info.is_dir():
                    continue

                # Validate path before extraction
                _safe_destination(
                    Path("."),
                    info.filename,
                )

                # Skip symbolic links
                if _is_symlink(info):

                    skipped_symlinks += 1

                    continue

                file_count += 1

                file_size = int(
                    info.file_size
                )

                total_size += (
                    file_size
                )

                # ----------------------------------------------
                # Maximum file count
                # ----------------------------------------------

                if file_count > max_files:

                    raise ZipSecurityError(
                        "Project contains more than "
                        f"{max_files:,} files."
                    )

                # ----------------------------------------------
                # Maximum single-file size
                # ----------------------------------------------

                if (
                    file_size
                    > max_single_file_bytes
                ):

                    limit_mb = (
                        max_single_file_bytes
                        / (1024 * 1024)
                    )

                    raise ZipSecurityError(
                        "A file inside the project exceeds "
                        f"the {limit_mb:.0f} MB limit: "
                        f"{info.filename}"
                    )

                # ----------------------------------------------
                # Maximum total expanded size
                # ----------------------------------------------

                if (
                    total_size
                    > max_total_uncompressed_bytes
                ):

                    limit_mb = (
                        max_total_uncompressed_bytes
                        / (1024 * 1024)
                    )

                    raise ZipSecurityError(
                        "Expanded project exceeds the "
                        f"{limit_mb:.0f} MB limit."
                    )

    except zipfile.BadZipFile as exc:

        raise ZipSecurityError(
            "The uploaded file is not a valid ZIP archive."
        ) from exc

    # ------------------------------------------------------
    # Validation information
    # ------------------------------------------------------

    return {

        "file_count":
            file_count,

        "total_uncompressed_bytes":
            total_size,

        "total_uncompressed_mb":
            round(
                total_size
                / (1024 * 1024),
                2,
            ),

        "max_single_file_mb":
            round(
                max_single_file_bytes
                / (1024 * 1024),
                2,
            ),

        "max_total_project_mb":
            round(
                max_total_uncompressed_bytes
                / (1024 * 1024),
                2,
            ),

        "skipped_symlinks":
            skipped_symlinks,

    }


# ==========================================================
# EXTRACT PROJECT ZIP
# ==========================================================

def extract_project_zip(
    zip_path: str | Path,
    extraction_root: str | Path,
) -> tuple[Path, dict]:
    """
    Safely extract one uploaded project into a unique
    working directory.

    Returns:
        (
            project_directory,
            validation_summary
        )
    """

    zip_path = Path(
        zip_path
    )

    extraction_root = Path(
        extraction_root
    )

    # ------------------------------------------------------
    # Validate before extracting
    # ------------------------------------------------------

    validation = validate_zip(
        zip_path
    )

    # ------------------------------------------------------
    # Make extraction root
    # ------------------------------------------------------

    extraction_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------
    # Unique extracted project folder
    # ------------------------------------------------------

    project_dir = (

        extraction_root
        / (
            "project_"
            f"{uuid.uuid4().hex}"
        )

    )

    project_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    try:

        with zipfile.ZipFile(
            zip_path,
            "r",
        ) as archive:

            for info in archive.infolist():

                # ------------------------------------------
                # Skip directories
                # ------------------------------------------

                if info.is_dir():
                    continue

                # ------------------------------------------
                # Skip symlinks
                # ------------------------------------------

                if _is_symlink(
                    info
                ):
                    continue

                # ------------------------------------------
                # Safe output path
                # ------------------------------------------

                destination = (
                    _safe_destination(
                        project_dir,
                        info.filename,
                    )
                )

                # ------------------------------------------
                # Create parent directories
                # ------------------------------------------

                destination.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                # ------------------------------------------
                # Extract file safely
                # ------------------------------------------

                with archive.open(
                    info,
                    "r",
                ) as source:

                    with destination.open(
                        "wb"
                    ) as target:

                        shutil.copyfileobj(
                            source,
                            target,
                        )

    except Exception:

        # If anything fails, remove incomplete extraction
        shutil.rmtree(
            project_dir,
            ignore_errors=True,
        )

        raise

    return (
        project_dir,
        validation,
    )


# ==========================================================
# DELETE EXTRACTED PROJECT
# ==========================================================

def delete_extracted_project(
    project_dir: str | Path,
) -> None:
    """
    Delete a temporary extracted project directory.
    """

    shutil.rmtree(
        Path(project_dir),
        ignore_errors=True,
    )

