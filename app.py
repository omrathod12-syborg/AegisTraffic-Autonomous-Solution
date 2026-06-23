#!/usr/bin/env python3
"""
Traffic Violation Detection — Streamlit Demo Dashboard
======================================================
Commercial-grade frontend for the CV backend pipeline. Presents a control-room
view for image ingestion, live violation citations, and a session audit ledger.
"""

from __future__ import annotations

import textwrap
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from traffic_violation_pipeline import PipelineConfig, TrafficViolationPipeline

# ---------------------------------------------------------------------------
# Page configuration — wide layout; dark theme reinforced via custom CSS below
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Traffic Violation Command Center",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Design system — high-contrast dark theme
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    /* Hide default Streamlit chrome for a cleaner product feel */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 100%; }

    /* App shell */
    .app-header {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.25rem 1.75rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45);
    }
    .app-title {
        font-size: 1.65rem;
        font-weight: 700;
        color: #f0f6fc;
        letter-spacing: -0.02em;
        margin: 0;
    }
    .app-subtitle {
        font-size: 0.9rem;
        color: #8b949e;
        margin-top: 0.35rem;
    }
    .badge-live {
        display: inline-block;
        background: #238636;
        color: #fff;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 0.2rem 0.55rem;
        border-radius: 999px;
        margin-left: 0.5rem;
        vertical-align: middle;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.65; }
    }

    /* Section panels */
    .panel {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1rem 1.15rem;
        margin-bottom: 0.85rem;
    }
    .panel-title {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #58a6ff;
        margin-bottom: 0.65rem;
    }

    /* Citation metric cards */
    .citation-card {
        background: linear-gradient(145deg, #1c2128 0%, #161b22 100%);
        border: 1px solid #30363d;
        border-left: 4px solid #f85149;
        border-radius: 10px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.85rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
    }
    .citation-card.safe {
        border-left-color: #3fb950;
    }
    .citation-label {
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #8b949e;
        margin-bottom: 0.2rem;
    }
    .citation-value {
        font-size: 1.05rem;
        font-weight: 600;
        color: #f0f6fc;
        word-break: break-all;
    }
    .citation-value.mono {
        font-family: 'SF Mono', 'Consolas', monospace;
        font-size: 0.88rem;
        color: #79c0ff;
    }
    .citation-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.75rem;
        margin-top: 0.5rem;
    }

    /* Status pills */
    .status-pill {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .pill-red { background: #3d1214; color: #ff7b72; border: 1px solid #f85149; }
    .pill-yellow { background: #3d2e00; color: #d29922; border: 1px solid #d29922; }
    .pill-green { background: #0f2d17; color: #3fb950; border: 1px solid #3fb950; }
    .pill-neutral { background: #21262d; color: #8b949e; border: 1px solid #30363d; }

    /* Image frame labels */
    .frame-caption {
        font-size: 0.75rem;
        font-weight: 600;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.4rem;
    }

    /* Audit ledger header */
    .ledger-header {
        font-size: 1rem;
        font-weight: 700;
        color: #f0f6fc;
        border-bottom: 2px solid #30363d;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 0.75rem 0;
    }

    /* Streamlit widget tuning for dark UI */
    .stFileUploader > div { background: #161b22; border-color: #30363d; border-radius: 8px; }
    div[data-testid="stMetric"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 0.65rem 0.85rem;
    }
    div[data-testid="stMetric"] label { color: #8b949e !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #f0f6fc !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached backend — YOLO weights load once per session
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing CV pipeline & YOLO model…")
def load_pipeline(model_path: str = "yolov8n.pt") -> TrafficViolationPipeline:
    """Load and cache the violation detection pipeline."""
    config = PipelineConfig(model_path=model_path)
    return TrafficViolationPipeline(config)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def init_session_state() -> None:
    """Initialize persistent session containers."""
    if "audit_ledger" not in st.session_state:
        st.session_state.audit_ledger: List[Dict[str, Any]] = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "last_process_key" not in st.session_state:
        st.session_state.last_process_key = None


def build_process_key(
    uploaded_file,
    traffic_light: str,
    model_path: str,
) -> Optional[str]:
    """Unique key for upload + signal + model — triggers re-inference when changed."""
    if uploaded_file is None:
        return None
    return f"{uploaded_file.name}:{uploaded_file.size}:{traffic_light}:{model_path}"


def run_pipeline(
    image_bgr: np.ndarray,
    traffic_light: str,
    model_path: str,
) -> Dict[str, Any]:
    """Execute backend pipeline with user-facing error wrapping."""
    try:
        pipeline = load_pipeline(model_path)
        return pipeline.process_image(image_bgr, traffic_light=traffic_light)
    except ModuleNotFoundError as exc:
        return {
            "success": False,
            "error": f"Missing dependency: {exc}. Run: pip install -r requirements.txt",
            "violations_detected": False,
            "tickets": [],
            "annotated_frame": None,
        }
    except OSError as exc:
        return {
            "success": False,
            "error": (
                f"Could not load model '{model_path}': {exc}. "
                "Ensure you have internet access for the first download."
            ),
            "violations_detected": False,
            "tickets": [],
            "annotated_frame": None,
        }
    except Exception as exc:
        return {
            "success": False,
            "error": f"Pipeline error: {type(exc).__name__}: {exc}",
            "violations_detected": False,
            "tickets": [],
            "annotated_frame": None,
        }


def append_tickets_to_ledger(result: Dict[str, Any]) -> None:
    """Append newly issued tickets to the session audit ledger (deduplicated)."""
    if not result or not result.get("success"):
        return
    tickets = result.get("tickets", [])
    det_count = result.get("detection_count", 0)
    existing_ids = {row["ticket_id"] for row in st.session_state.audit_ledger}
    for ticket in tickets:
        tid = ticket.get("ticket_id")
        if tid and tid not in existing_ids:
            st.session_state.audit_ledger.append(ticket_to_ledger_row(ticket, det_count))
            existing_ids.add(tid)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert OpenCV BGR array to RGB for Streamlit/PIL display."""
    if image is None or image.size == 0:
        return image
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def uploaded_file_to_bgr(uploaded_file) -> Optional[np.ndarray]:
    """Decode a Streamlit UploadedFile into a BGR NumPy array."""
    if uploaded_file is None:
        return None
    try:
        raw = uploaded_file.getvalue()
        arr = np.frombuffer(raw, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return image
    except Exception:
        return None


def json_safe_preview(result: Dict[str, Any]) -> Dict[str, Any]:
    """Return a JSON-serializable copy of the pipeline result for st.json."""
    preview = {k: v for k, v in result.items() if k != "annotated_frame"}

    def _sanitize(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k): _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    return _sanitize(preview)


def render_system_status(model_path: str) -> None:
    """Show model readiness banner so first-run delays are not mistaken for crashes."""
    if st.session_state.get("model_error"):
        st.error(
            f"**Detection engine failed to start:** {st.session_state.model_error}  \n"
            "Check your internet connection (needed once to download YOLO weights), "
            "then refresh the page."
        )
        return

    if st.session_state.get("model_ready"):
        st.success(f"Detection engine online · `{model_path}` loaded and ready.")
    else:
        st.info(
            "Starting detection engine… First launch downloads YOLO weights (~6 MB) "
            "and may take up to a minute."
        )


def format_violation_type(raw: str) -> str:
    """Human-readable violation label."""
    return raw.replace("_", " ").title()


def traffic_light_pill(state: str) -> str:
    """Return HTML badge for the current signal state."""
    key = state.upper()
    css = {"RED": "pill-red", "YELLOW": "pill-yellow", "GREEN": "pill-green"}.get(
        key, "pill-neutral"
    )
    return f'<span class="status-pill {css}">SIGNAL · {key}</span>'


def ticket_to_ledger_row(ticket: Dict[str, Any], detection_count: int) -> Dict[str, Any]:
    """Flatten a ticket dict into a row suitable for the audit dataframe."""
    conf = ticket.get("confidence", {})
    return {
        "ticket_id": ticket.get("ticket_id", "—"),
        "timestamp": ticket.get("timestamp", datetime.now(timezone.utc).isoformat()),
        "violation_type": format_violation_type(ticket.get("violation_type", "unknown")),
        "license_plate": ticket.get("license_plate", "UNKNOWN"),
        "system_confidence": conf.get("violation", 0.0),
        "detection_confidence": conf.get("detection", 0.0),
        "plate_confidence": conf.get("license_plate", 0.0),
        "vehicle_class": ticket.get("subject", {}).get("class_name", "—"),
        "objects_in_frame": detection_count,
    }


def render_citation_card(ticket: Dict[str, Any]) -> None:
    """Render a styled metric card for a single violation ticket."""
    conf = ticket.get("confidence", {})
    vtype = format_violation_type(ticket.get("violation_type", "Unknown"))
    ticket_id = ticket.get("ticket_id", "N/A")
    plate = ticket.get("license_plate", "UNKNOWN")
    sys_conf = conf.get("violation", 0.0)

    card_html = textwrap.dedent(
        f"""
        <div class="citation-card">
            <div class="citation-label">Violation Type</div>
            <div class="citation-value">{vtype}</div>
            <div class="citation-grid">
                <div>
                    <div class="citation-label">Ticket ID</div>
                    <div class="citation-value mono">{ticket_id[:13]}…</div>
                </div>
                <div>
                    <div class="citation-label">Plate Number</div>
                    <div class="citation-value mono">{plate}</div>
                </div>
                <div>
                    <div class="citation-label">System Confidence</div>
                    <div class="citation-value">{sys_conf:.1%}</div>
                </div>
                <div>
                    <div class="citation-label">Detection Confidence</div>
                    <div class="citation-value">{conf.get("detection", 0.0):.1%}</div>
                </div>
            </div>
        </div>
        """
    ).strip()
    st.markdown(card_html, unsafe_allow_html=True)


def render_clear_card(message: str) -> None:
    """Render a green-bordered card when no violations are found."""
    st.markdown(
        f"""
        <div class="citation-card safe">
            <div class="citation-label">Status</div>
            <div class="citation-value">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
def render_header() -> None:
    st.markdown(
        """
        <div class="app-header">
            <p class="app-title">
                Traffic Violation Command Center
                <span class="badge-live">LIVE</span>
            </p>
            <p class="app-subtitle">
                AI-powered enforcement demo · YOLOv8/v10 · OpenCV preprocessing · Mock ALPR
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------------
def main() -> None:
    init_session_state()
    render_header()

    # Optional model selector in a slim top bar
    model_col, stats_col = st.columns([3, 1])
    with model_col:
        model_path = st.selectbox(
            "Detection Model",
            options=["yolov8n.pt", "yolov8s.pt", "yolov10n.pt"],
            index=0,
            help="Ultralytics weights file. First run downloads weights automatically.",
        )
    with stats_col:
        st.metric("Session Citations", len(st.session_state.audit_ledger))

    # Pre-warm YOLO so the first upload does not appear to hang or fail.
    if st.session_state.get("loaded_model_path") != model_path:
        st.session_state.model_ready = False
        st.session_state.model_error = None

    if not st.session_state.get("model_ready"):
        try:
            with st.spinner(f"Loading {model_path} — please wait…"):
                load_pipeline(model_path)
            st.session_state.model_ready = True
            st.session_state.model_error = None
            st.session_state.loaded_model_path = model_path
        except Exception as exc:
            st.session_state.model_ready = False
            st.session_state.model_error = f"{type(exc).__name__}: {exc}"

    render_system_status(model_path)

    # 60 / 40 two-column layout
    col_control, col_citation = st.columns([6, 4], gap="large")

    # ------------------------------------------------------------------
    # LEFT — Control Room
    # ------------------------------------------------------------------
    with col_control:
        st.markdown('<div class="panel-title">Control Room · Image Ingestion</div>', unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload traffic camera frame",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            help="Supported formats: JPG, PNG, WEBP, BMP",
        )

        traffic_light = st.selectbox(
            "Simulated Traffic Signal",
            options=["RED", "YELLOW", "GREEN"],
            index=0,
            help="Stop-line violations are evaluated only when signal is RED.",
        )

        st.markdown(traffic_light_pill(traffic_light), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        if uploaded is None:
            st.info("Upload a traffic image to begin analysis.")
        else:
            original_bgr = uploaded_file_to_bgr(uploaded)
            if original_bgr is None:
                st.error("Could not decode the uploaded image. Please try another file.")
            else:
                process_key = build_process_key(uploaded, traffic_light, model_path)
                if process_key != st.session_state.last_process_key:
                    with st.status("Analyzing frame…", expanded=True) as status:
                        st.write("Preprocessing image…")
                        result = run_pipeline(original_bgr, traffic_light, model_path)
                        if result.get("success"):
                            st.write(f"Detected {result.get('detection_count', 0)} object(s).")
                            status.update(
                                label="Analysis complete",
                                state="complete",
                                expanded=False,
                            )
                        else:
                            status.update(label="Analysis failed", state="error", expanded=True)
                        st.session_state.last_result = result
                        st.session_state.last_process_key = process_key
                        append_tickets_to_ledger(result)

                result = st.session_state.last_result

                img_col1, img_col2 = st.columns(2)
                with img_col1:
                    st.markdown('<p class="frame-caption">Original Frame</p>', unsafe_allow_html=True)
                    st.image(
                        bgr_to_rgb(original_bgr),
                        width="stretch",
                        caption=f"{uploaded.name} · {original_bgr.shape[1]}×{original_bgr.shape[0]}",
                    )

                with img_col2:
                    st.markdown('<p class="frame-caption">Annotated Output</p>', unsafe_allow_html=True)
                    if result is None:
                        st.markdown(
                            '<div class="panel" style="min-height:200px;display:flex;align-items:center;'
                            'justify-content:center;color:#8b949e;">Awaiting analysis…</div>',
                            unsafe_allow_html=True,
                        )
                    elif not result.get("success"):
                        st.error(result.get("error", "Unknown pipeline failure."))
                    else:
                        annotated = result.get("annotated_frame")
                        if annotated is not None:
                            st.image(
                                bgr_to_rgb(annotated),
                                width="stretch",
                                caption="Detections & violation overlays",
                            )
                        else:
                            st.warning("No annotated frame returned.")

                # Post-analysis status strip (left column)
                if result and result.get("success"):
                    det_count = result.get("detection_count", 0)
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Objects Detected", det_count)
                    m2.metric("Violations", result.get("ticket_count", 0))
                    low_light = result.get("preprocessing", {}).get("low_light_detected", False)
                    m3.metric("Low-Light Mode", "ON" if low_light else "OFF")

                    if det_count == 0:
                        st.info(
                            "Analysis finished successfully, but no vehicles or persons were detected. "
                            "Try a clearer photo, switch to **yolov8s.pt**, or use an image with visible traffic."
                        )
                    elif not result.get("violations_detected"):
                        st.success("Analysis complete — no violations flagged for current signal state.")

    # ------------------------------------------------------------------
    # RIGHT — Citation Engine
    # ------------------------------------------------------------------
    with col_citation:
        st.markdown('<div class="panel-title">Citation Engine · Violation Tickets</div>', unsafe_allow_html=True)

        result = st.session_state.last_result

        if result is None:
            render_clear_card("No analysis run yet. Upload a traffic camera frame to begin.")
        elif not result.get("success"):
            st.error(result.get("error", "Pipeline failed."))
        else:
            tickets: List[Dict[str, Any]] = result.get("tickets", [])

            if not tickets:
                det = result.get("detection_count", 0)
                if det == 0:
                    render_clear_card("Frame processed — no vehicles or persons detected.")
                else:
                    render_clear_card(
                        f"Frame clear — {det} object(s) detected, zero violations for **{traffic_light}** signal."
                    )
            else:
                st.markdown(
                    f'<p style="color:#f85149;font-weight:600;margin-bottom:0.75rem;">'
                    f"⚠ {len(tickets)} violation(s) issued</p>",
                    unsafe_allow_html=True,
                )
                for ticket in tickets:
                    render_citation_card(ticket)

            # Raw JSON expander for judges / debugging
            with st.expander("View raw pipeline JSON"):
                st.json(json_safe_preview(result))

    # ------------------------------------------------------------------
    # BOTTOM — Audit Ledger
    # ------------------------------------------------------------------
    st.markdown('<div class="ledger-header">📋 Audit Ledger · Session Violation Database</div>', unsafe_allow_html=True)

    if not st.session_state.audit_ledger:
        st.markdown(
            '<div class="panel" style="color:#8b949e;text-align:center;padding:1.5rem;">'
            "No citations recorded this session. Violations will appear here automatically."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        df = pd.DataFrame(st.session_state.audit_ledger)
        display_cols = [
            "timestamp",
            "ticket_id",
            "violation_type",
            "license_plate",
            "system_confidence",
            "vehicle_class",
            "objects_in_frame",
        ]
        df_display = df[display_cols].copy()
        df_display["system_confidence"] = df_display["system_confidence"].apply(
            lambda x: f"{float(x):.1%}"
        )
        df_display.columns = [
            "Timestamp (UTC)",
            "Ticket ID",
            "Violation Type",
            "Plate Number",
            "System Confidence",
            "Vehicle Class",
            "Objects in Frame",
        ]

        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            height=min(42 + 35 * len(df_display), 320),
        )

        btn_col1, btn_col2, _ = st.columns([1, 1, 4])
        with btn_col1:
            if st.button("Export CSV", use_container_width=True):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download ledger.csv",
                    data=csv,
                    file_name=f"violation_ledger_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        with btn_col2:
            if st.button("Clear Ledger", use_container_width=True):
                st.session_state.audit_ledger = []
                st.session_state.last_result = None
                st.session_state.last_process_key = None
                st.rerun()


if __name__ == "__main__":
    main()
