from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path

from dotenv import load_dotenv

from flask import (
    Flask,
    abort,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)

from werkzeug.utils import secure_filename

from src.project_analyzer.project_service import (
    analyze_uploaded_project,
)

from src.project_analyzer.zip_handler import (
    ZipSecurityError,
)


# ==========================================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================================

load_dotenv()


# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

UPLOAD_FOLDER = (
    PROJECT_ROOT
    / "uploads"
)

REPORT_FOLDER = (
    PROJECT_ROOT
    / "analysis_results"
    / "reports"
)


UPLOAD_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)


# ==========================================================
# FLASK APP
# ==========================================================

app = Flask(__name__)


# ==========================================================
# SECRET KEY
# ==========================================================

app.secret_key = os.getenv(
    "FLASK_SECRET_KEY"
)

if not app.secret_key:

    raise RuntimeError(
        "FLASK_SECRET_KEY is missing. "
        "Create a .env file and add "
        "FLASK_SECRET_KEY=your-secret-key"
    )


# ==========================================================
# UPLOAD SETTINGS
# ==========================================================

# Maximum uploaded ZIP size = 100 MB
app.config["MAX_CONTENT_LENGTH"] = (
    100
    * 1024
    * 1024
)


# ==========================================================
# HOME PAGE
# ==========================================================

@app.get("/")
def home():

    return render_template(
        "index.html"
    )


# ==========================================================
# ANALYZE PROJECT
# ==========================================================

@app.post("/analyze")
def analyze():

    uploaded_file = request.files.get(
        "project_zip"
    )


    # ------------------------------------------------------
    # Check file exists
    # ------------------------------------------------------

    if (
        uploaded_file is None
        or not uploaded_file.filename
    ):

        flash(
            "Please choose a project ZIP file."
        )

        return redirect(
            url_for("home")
        )


    # ------------------------------------------------------
    # Secure uploaded filename
    # ------------------------------------------------------

    original_name = secure_filename(
        uploaded_file.filename
    )


    # ------------------------------------------------------
    # Only allow ZIP files
    # ------------------------------------------------------

    if not original_name.lower().endswith(
        ".zip"
    ):

        flash(
            "Only .zip project files are supported."
        )

        return redirect(
            url_for("home")
        )


    # ------------------------------------------------------
    # Analysis mode
    # ------------------------------------------------------

    analysis_mode = request.form.get(
        "analysis_mode",
        "quick",
    )

    if analysis_mode not in {
        "quick",
        "full",
    }:

        analysis_mode = "quick"


    # ------------------------------------------------------
    # Generate unique upload filename
    # ------------------------------------------------------

    unique_name = (
        f"{uuid.uuid4().hex}_"
        f"{original_name}"
    )


    saved_path = (
        UPLOAD_FOLDER
        / unique_name
    )


    # ------------------------------------------------------
    # Save uploaded ZIP temporarily
    # ------------------------------------------------------

    uploaded_file.save(
        saved_path
    )


    try:

        # ==================================================
        # RUN COMPLETE PROJECT ANALYSIS
        # ==================================================

        result = analyze_uploaded_project(
            PROJECT_ROOT,
            saved_path,
            analysis_mode=analysis_mode,
        )


        # ==================================================
        # CREATE DOWNLOADABLE JSON REPORT
        # ==================================================

        report_id = (
            uuid.uuid4().hex
        )


        report_path = (
            REPORT_FOLDER
            / f"{report_id}.json"
        )


        report_payload = {

            "project_name":
                original_name,

            "result":
                result,

        }


        report_path.write_text(

            json.dumps(
                report_payload,
                indent=2,
            ),

            encoding="utf-8",

        )


        # ==================================================
        # SHOW RESULTS PAGE
        # ==================================================

        return render_template(

            "results.html",

            project_name=
                original_name,

            result=
                result,

            report_id=
                report_id,

        )


    # ======================================================
    # EXPECTED ERRORS
    # ======================================================

    except (
        ZipSecurityError,
        ValueError,
        FileNotFoundError,
    ) as exc:

        flash(
            str(exc)
        )

        return redirect(
            url_for("home")
        )


    # ======================================================
    # UNEXPECTED ERRORS
    # ======================================================

    except Exception as exc:

        print()

        print(
            "=" * 70
        )

        print(
            "PROJECT ANALYSIS ERROR"
        )

        print(
            "=" * 70
        )

        print(
            repr(exc)
        )

        print(
            "=" * 70
        )

        print()


        flash(
            "Project analysis failed. "
            "Check the terminal for the detailed error."
        )

        return redirect(
            url_for("home")
        )


    # ======================================================
    # DELETE TEMPORARY UPLOAD
    # ======================================================

    finally:

        try:

            saved_path.unlink(
                missing_ok=True
            )

        except OSError:

            pass


# ==========================================================
# DOWNLOAD JSON REPORT
# ==========================================================

@app.get(
    "/download-report/<report_id>"
)
def download_report(
    report_id: str,
):


    # ------------------------------------------------------
    # Validate report ID
    # ------------------------------------------------------

    if not re.fullmatch(
        r"[0-9a-f]{32}",
        report_id,
    ):

        abort(404)


    report_path = (
        REPORT_FOLDER
        / f"{report_id}.json"
    )


    # ------------------------------------------------------
    # Check report exists
    # ------------------------------------------------------

    if not report_path.is_file():

        abort(404)


    # ------------------------------------------------------
    # Download report
    # ------------------------------------------------------

    return send_file(

        report_path,

        as_attachment=True,

        download_name=(
            "project_defect_analysis.json"
        ),

        mimetype=(
            "application/json"
        ),

    )


# ==========================================================
# FILE TOO LARGE ERROR
# ==========================================================

@app.errorhandler(413)
def upload_too_large(
    _error,
):

    flash(
        "The ZIP is too large. "
        "Maximum upload size is 100 MB."
    )

    return redirect(
        url_for("home")
    )


# ==========================================================
# RUN SERVER
# ==========================================================

if __name__ == "__main__":

    app.run(

        host="127.0.0.1",

        port=5000,

        debug=False,

        use_reloader=False,

    )