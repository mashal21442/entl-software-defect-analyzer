from __future__ import annotations

import json
import uuid
from pathlib import Path

import streamlit as st

from src.project_analyzer.project_service import (
    analyze_uploaded_project,
)

from src.project_analyzer.zip_handler import (
    ZipSecurityError,
)


PROJECT_ROOT = Path(__file__).resolve().parent

UPLOAD_FOLDER = PROJECT_ROOT / "uploads"

UPLOAD_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="ENTL Software Defect Analyzer",
    page_icon="🔍",
    layout="wide",
)


# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at top left,
                rgba(124, 140, 255, 0.18),
                transparent 30%
            ),
            #0b1020;

        color: #f6f8ff;
    }

    .main-title {
        font-size: 48px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .subtitle {
        color: #98a3ba;
        font-size: 17px;
        margin-bottom: 28px;
    }

    .health-card {
        padding: 20px;
        border-radius: 18px;
        background: #131a2d;
        border: 1px solid rgba(255,255,255,.1);
        margin-bottom: 20px;
    }

    .risk-high {
        color: #ff647c;
        font-weight: 700;
    }

    .risk-medium {
        color: #ffc857;
        font-weight: 700;
    }

    .risk-low {
        color: #48d597;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# HEADER
# ==========================================================

st.markdown(
    '<div class="main-title">'
    'ENTL Software Defect Analyzer'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Upload a complete software project as a ZIP file.
    The system detects languages, analyzes code,
    identifies static-analysis issues and ranks files
    by experimental ENTL defect risk.
    </div>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# FILE UPLOAD
# ==========================================================

uploaded_file = st.file_uploader(
    "Upload project ZIP",
    type=["zip"],
    max_upload_size=100,
)


# ==========================================================
# ANALYSIS MODE
# ==========================================================

analysis_mode = st.radio(
    "Analysis mode",
    options=[
        "quick",
        "full",
    ],
    format_func=lambda value: (
        "Quick Analysis"
        if value == "quick"
        else "Full Analysis"
    ),
    horizontal=True,
)


# ==========================================================
# ANALYZE
# ==========================================================

if uploaded_file is not None:

    if st.button(
        "Analyze Project",
        type="primary",
        use_container_width=True,
    ):

        unique_name = (
            f"{uuid.uuid4().hex}_"
            f"{uploaded_file.name}"
        )

        saved_path = (
            UPLOAD_FOLDER
            / unique_name
        )

        try:

            saved_path.write_bytes(
                uploaded_file.getbuffer()
            )

            with st.spinner(
                "Analyzing project..."
            ):

                result = analyze_uploaded_project(
                    PROJECT_ROOT,
                    saved_path,
                    analysis_mode=analysis_mode,
                )


            # ==================================================
            # RESULTS
            # ==================================================

            st.success(
                "Project analysis completed."
            )


            # --------------------------------------------------
            # Health score
            # --------------------------------------------------

            health_score = result.get(
                "project_health_score",
                0,
            )

            health_label = result.get(
                "project_health_label",
                "Unknown",
            )

            st.markdown(
                f"""
                <div class="health-card">
                    <h2>
                        Project Health:
                        {health_score}/100
                    </h2>

                    <p>
                        {health_label}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )


            # --------------------------------------------------
            # Summary
            # --------------------------------------------------

            col1, col2, col3, col4, col5 = (
                st.columns(5)
            )

            col1.metric(
                "Files analyzed",
                result.get(
                    "detected_file_count",
                    0,
                ),
            )

            col2.metric(
                "High risk",
                result.get(
                    "high_risk_files",
                    0,
                ),
            )

            col3.metric(
                "Medium risk",
                result.get(
                    "medium_risk_files",
                    0,
                ),
            )

            col4.metric(
                "Low risk",
                result.get(
                    "low_risk_files",
                    0,
                ),
            )

            col5.metric(
                "Static issues",
                result.get(
                    "total_issues",
                    0,
                ),
            )


            # --------------------------------------------------
            # Languages
            # --------------------------------------------------

            st.subheader(
                "Languages detected"
            )

            languages = result.get(
                "languages",
                {},
            )

            if languages:

                st.write(
                    languages
                )

            else:

                st.info(
                    "No supported source languages detected."
                )


            # ==================================================
            # FILTERS
            # ==================================================

            st.subheader(
                "Files ranked by risk"
            )

            search_text = st.text_input(
                "Search filename or path"
            )

            filter_col1, filter_col2 = (
                st.columns(2)
            )

            risk_filter = (
                filter_col1.selectbox(
                    "Risk level",
                    [
                        "All",
                        "High",
                        "Medium",
                        "Low",
                    ],
                )
            )

            available_languages = sorted(
                languages.keys()
            )

            language_filter = (
                filter_col2.selectbox(
                    "Language",
                    [
                        "All",
                        *available_languages,
                    ],
                )
            )


            # ==================================================
            # FILE RESULTS
            # ==================================================

            files = result.get(
                "files",
                [],
            )

            visible_files = []

            for file_result in files:

                path = file_result.get(
                    "path",
                    "",
                )

                language = file_result.get(
                    "language",
                    "unknown",
                )

                risk_level = file_result.get(
                    "risk_level",
                    "low",
                )


                if search_text:

                    if (
                        search_text.lower()
                        not in path.lower()
                    ):

                        continue


                if (
                    risk_filter != "All"
                    and risk_level.lower()
                    != risk_filter.lower()
                ):

                    continue


                if (
                    language_filter != "All"
                    and language
                    != language_filter
                ):

                    continue


                visible_files.append(
                    file_result
                )


            st.caption(
                f"Showing "
                f"{len(visible_files)} "
                f"of {len(files)} files"
            )


            # --------------------------------------------------
            # Display each file
            # --------------------------------------------------

            for file_result in visible_files:

                path = file_result.get(
                    "path",
                    "Unknown file",
                )

                language = file_result.get(
                    "language",
                    "unknown",
                )

                risk_level = file_result.get(
                    "risk_level",
                    "low",
                )

                entl_score = file_result.get(
                    "entl_risk_score",
                    0,
                )

                vote = file_result.get(
                    "algorithm1_vote_percent",
                    0,
                )

                issues = file_result.get(
                    "issues",
                    [],
                )

                metrics = file_result.get(
                    "metrics",
                    {},
                )


                with st.expander(
                    f"{risk_level.upper()} "
                    f"• {path} "
                    f"• ENTL {entl_score:.1f}%"
                ):

                    col1, col2, col3 = (
                        st.columns(3)
                    )

                    col1.metric(
                        "ENTL score",
                        f"{entl_score:.1f}%",
                    )

                    col2.metric(
                        "Algorithm-1 vote",
                        f"{vote:.1f}%",
                    )

                    col3.metric(
                        "Static issues",
                        len(issues),
                    )


                    st.write(
                        f"**Language:** "
                        f"{language}"
                    )


                    metric_col1, metric_col2, metric_col3 = (
                        st.columns(3)
                    )

                    metric_col1.metric(
                        "LOC",
                        metrics.get(
                            "loc",
                            0,
                        ),
                    )

                    metric_col2.metric(
                        "Complexity",
                        metrics.get(
                            "complexity_proxy",
                            0,
                        ),
                    )

                    metric_col3.metric(
                        "Max nesting",
                        metrics.get(
                            "max_nesting",
                            0,
                        ),
                    )


                    if issues:

                        st.markdown(
                            "### Problems detected"
                        )

                        for issue in issues:

                            severity = issue.get(
                                "severity",
                                "low",
                            )

                            line = issue.get(
                                "line",
                                "?",
                            )

                            rule = issue.get(
                                "rule",
                                "Issue",
                            )

                            message = issue.get(
                                "message",
                                "",
                            )


                            st.markdown(
                                f"**{severity.upper()} "
                                f"— Line {line} "
                                f"— {rule}**"
                            )

                            st.write(
                                message
                            )


                            context = issue.get(
                                "context",
                                [],
                            )

                            if context:

                                code_text = "\n".join(
                                    f"{item.get('line_number', '')}: "
                                    f"{item.get('text', '')}"
                                    for item in context
                                )

                                st.code(
                                    code_text
                                )

                            elif issue.get(
                                "code"
                            ):

                                st.code(
                                    issue["code"]
                                )


                    else:

                        st.success(
                            "No concrete static-analysis "
                            "issue detected in this file."
                        )


            # ==================================================
            # JSON DOWNLOAD
            # ==================================================

            report_payload = {

                "project_name":
                    uploaded_file.name,

                "result":
                    result,

            }

            report_json = json.dumps(
                report_payload,
                indent=2,
            )

            st.download_button(

                label=
                    "Download JSON Report",

                data=
                    report_json,

                file_name=
                    "project_defect_analysis.json",

                mime=
                    "application/json",

                use_container_width=True,

            )


            # --------------------------------------------------
            # Important note
            # --------------------------------------------------

            st.warning(
                result.get(
                    "important_note",
                    (
                        "ENTL scores are experimental "
                        "defect-risk prioritization scores "
                        "and are not proof that a file "
                        "contains a defect."
                    ),
                )
            )


        except (
            ZipSecurityError,
            ValueError,
            FileNotFoundError,
        ) as exc:

            st.error(
                str(exc)
            )


        except Exception as exc:

            st.error(
                "Project analysis failed."
            )

            st.exception(
                exc
            )


        finally:

            try:

                saved_path.unlink(
                    missing_ok=True
                )

            except OSError:

                pass