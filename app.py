"""IntelliWatch — Privacy-aware physical asset monitoring (local Streamlit MVP)."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from database import (
    STATUS_NOT_DETECTED,
    STATUS_POSSIBLE,
    STATUS_UNCHANGED,
    asset_exists,
    get_asset,
    get_assets,
    get_dashboard_stats,
    get_events,
    init_db,
    insert_asset,
    insert_event,
)
from demo import load_demo_records
from paths import (
    ALLOWED_IMAGE_TYPES,
    BASELINES_DIR,
    CURRENT_DIR,
    EVIDENCE_DIR,
    ensure_directories,
)
from vision import (
    ILLUSTRATIVE_CATEGORIES,
    AnalysisResult,
    analyze_images,
    decode_uploaded_bytes,
    detections_to_json,
    load_yolo_model,
    save_bgr_image,
)

st.set_page_config(
    page_title="IntelliWatch",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .block-container { padding-top: 1.4rem; padding-bottom: 2.5rem; max-width: 1280px; }
    [data-testid="stSidebar"] { background: #0B1F33; }
    [data-testid="stSidebar"] [data-testid="stMarkdown"] p,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] h3 { color: #E8EEF4 !important; }
    [data-testid="stSidebar"] .stRadio label { font-size: 0.95rem; }
    [data-testid="stSidebar"] button,
    [data-testid="stSidebar"] button * { color: #0B1F33 !important; background: #E8EEF4 !important;
                                       border: 0; font-weight: 600; }
    [data-testid="stSidebar"] [data-testid="stAlert"] p { color: #0B1F33 !important; }
    .iw-kicker { letter-spacing: 0.12em; text-transform: uppercase; font-size: 0.72rem;
                 color: #5B6B7C; font-weight: 600; margin-bottom: 0.2rem; }
    .iw-title { font-size: 1.85rem; font-weight: 700; color: #0B1F33; margin: 0 0 0.35rem 0; }
    .iw-sub { color: #4A5A6A; font-size: 0.98rem; margin-bottom: 1.1rem; }
    .iw-card { background: #fff; border: 1px solid #E2E8F0; border-radius: 12px;
               padding: 1rem 1.1rem; box-shadow: 0 1px 2px rgba(11,31,51,0.04); }
    .iw-metric-label { font-size: 0.78rem; color: #5B6B7C; text-transform: uppercase;
                       letter-spacing: 0.08em; font-weight: 600; }
    .iw-metric-value { font-size: 1.9rem; font-weight: 700; color: #0B1F33; line-height: 1.2; }
    .iw-pill { display: inline-block; padding: 0.2rem 0.65rem; border-radius: 999px;
               font-size: 0.78rem; font-weight: 650; }
    .ok { background: #E7F6EC; color: #0F7B3A; }
    .warn { background: #FFF4E5; color: #C47B00; }
    .crit { background: #FDECEC; color: #B42318; }
    .info { background: #E8F1F8; color: #0B4F6C; }
    .iw-banner { border-radius: 10px; padding: 0.85rem 1rem; margin: 0.6rem 0 1rem 0; font-size: 0.92rem; }
    .iw-banner.privacy { background: #EEF4F8; border: 1px solid #C9D7E3; color: #1B2430; }
    .iw-banner.limit { background: #FFF8EB; border: 1px solid #E8D3A2; color: #5C4300; }
    .iw-banner.human { background: #F4F0FF; border: 1px solid #D4CBEE; color: #2F2460; }
    .iw-empty { text-align: center; padding: 2rem 1rem; color: #5B6B7C; }
    footer { visibility: hidden; }
</style>
"""


def bootstrap() -> None:
    ensure_directories()
    init_db()


@st.cache_resource(show_spinner="Loading YOLOv8n (first run may download the nano weights)...")
def get_model_state():
    return load_yolo_model()


def status_class(status: str) -> str:
    if status == STATUS_UNCHANGED:
        return "ok"
    if status == STATUS_POSSIBLE:
        return "warn"
    return "crit"


def pill(status: str) -> str:
    return f'<span class="iw-pill {status_class(status)}">{status}</span>'


def save_upload(data: bytes, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return dest


def header(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="iw-kicker">{kicker}</div>'
        f'<p class="iw-title">{title}</p>'
        f'<p class="iw-sub">{subtitle}</p>',
        unsafe_allow_html=True,
    )


def privacy_notes() -> None:
    st.markdown(
        """
        <div class="iw-banner privacy">
            <strong>Privacy:</strong> No facial recognition. Images stay on this computer
            (local folders + SQLite). Nothing is uploaded to a cloud vision API.
        </div>
        <div class="iw-banner limit">
            <strong>Detection limit:</strong> YOLOv8 identifies COCO object <em>categories</em>
            (for example a laptop-shaped object), not the unique identity of a specific printer
            or monitor. Categories such as printer/monitor/laptop/keyboard are illustrative and
            are not confirmed L&amp;T Finance inventory.
        </div>
        <div class="iw-banner human">
            <strong>Human verification required:</strong> A missing detection must never be
            treated as confirmed theft. Review evidence before any operational action.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    st.sidebar.markdown("### IntelliWatch")
    st.sidebar.caption("Privacy-aware physical asset monitoring · local MVP")
    page = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Register asset", "Image monitoring", "Event history"],
        label_visibility="collapsed",
    )
    st.sidebar.divider()
    model_state = get_model_state()
    if model_state.available:
        st.sidebar.markdown("**Model:** YOLOv8n · local · CPU")
        st.sidebar.success("Detector ready")
    else:
        st.sidebar.error("Detector unavailable")
        st.sidebar.caption(model_state.message)

    st.sidebar.divider()
    if st.sidebar.button("Load demo records", width="stretch"):
        with st.sidebar:
            with st.spinner("Creating labelled synthetic records..."):
                message = load_demo_records()
            st.session_state["flash"] = message
            st.rerun()
    st.sidebar.caption(
        "Demo records are synthetic and labelled. They are not YOLO output and not real branch events."
    )
    return page, model_state


def page_dashboard(model_state) -> None:
    header(
        "Overview",
        "Branch asset monitoring dashboard",
        "Local prototype for finance-branch desks. Status is a review cue, not a confirmed incident.",
    )
    privacy_notes()
    stats = get_dashboard_stats()
    recent = get_events(limit=8)

    if model_state.available:
        sys_label, sys_cls, sys_text = (
            "Operational (local)",
            "ok",
            "YOLO nano model is loaded. Analysis stays on this laptop.",
        )
    else:
        sys_label, sys_cls, sys_text = (
            "Detection offline",
            "crit",
            model_state.message + " Use demo records to walk through the UI, or fix the model setup and restart.",
        )

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "Total registered assets", stats["total_assets"], None),
        (c2, "Assets currently detected", stats["assets_detected"], "Latest event = UNCHANGED / DETECTED"),
        (c3, "Open warnings", stats["open_warnings"], "Latest event needs review"),
        (c4, "System status", sys_label, sys_text),
    ]
    for col, label, value, hint in cards:
        with col:
            extra = ""
            if label == "System status":
                extra = (
                    f'<div style="margin-top:0.35rem">'
                    f'<span class="iw-pill {sys_cls}">{sys_label}</span></div>'
                )
                st.markdown(
                    f'<div class="iw-card"><div class="iw-metric-label">{label}</div>'
                    f'{extra}<div style="font-size:0.82rem;color:#4A5A6A;margin-top:0.45rem;">{hint}</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                hint_html = f'<div style="font-size:0.78rem;color:#5B6B7C;margin-top:0.25rem;">{hint}</div>' if hint else ""
                st.markdown(
                    f'<div class="iw-card"><div class="iw-metric-label">{label}</div>'
                    f'<div class="iw-metric-value">{value}</div>{hint_html}</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("#### Recent events")
    live, demo = stats["live_events"], stats["demo_events"]
    st.caption(f"{live} image-analysis event(s) · {demo} labelled demo event(s). Demo rows are not model results.")

    if not recent:
        st.markdown(
            '<div class="iw-card iw-empty">No events yet. Register an asset, run image monitoring, '
            "or load demo records from the sidebar.</div>",
            unsafe_allow_html=True,
        )
        return

    for event in recent:
        source_tag = "DEMO" if event["source"] == "demo" else "LIVE"
        source_cls = "info" if event["source"] == "demo" else "ok"
        cols = st.columns([1.4, 1.1, 1.6, 2.8, 0.9])
        cols[0].markdown(f"**{event['timestamp']}**")
        cols[1].markdown(f"`{event['asset_id']}`")
        cols[2].markdown(pill(event["status"]), unsafe_allow_html=True)
        cols[3].caption((event.get("zone") or "—") + " · " + (event["explanation"][:140] + ("…" if len(event["explanation"]) > 140 else "")))
        cols[4].markdown(f'<span class="iw-pill {source_cls}">{source_tag}</span>', unsafe_allow_html=True)


def page_register() -> None:
    header(
        "Assets",
        "Register an asset",
        "Store a baseline/reference photo for a named zone. These categories are examples, not an official inventory list.",
    )
    privacy_notes()

    with st.form("register_asset", clear_on_submit=False):
        c1, c2 = st.columns(2)
        asset_id = c1.text_input("Asset ID", placeholder="e.g. BR12-PRN-004").strip()
        name = c2.text_input("Asset name", placeholder="e.g. Operations desk printer").strip()
        c3, c4 = st.columns(2)
        category = c3.selectbox("Asset category (illustrative)", ILLUSTRATIVE_CATEGORIES)
        zone = c4.text_input("Branch / zone name", placeholder="e.g. Pune branch · Operations desk").strip()
        baseline_file = st.file_uploader(
            "Baseline / reference image (same predefined camera view you will monitor)",
            type=list(ALLOWED_IMAGE_TYPES),
        )
        submitted = st.form_submit_button("Register asset", type="primary")

    if not submitted:
        assets = get_assets()
        st.markdown("#### Registered assets")
        if not assets:
            st.info("No assets yet. Register one above, or load demo records.")
            return
        st.dataframe(
            [
                {
                    "Asset ID": a["asset_id"],
                    "Name": a["name"],
                    "Category": a["category"],
                    "Zone": a["zone"],
                    "Registered": a["created_at"],
                }
                for a in assets
            ],
            width="stretch",
            hide_index=True,
        )
        return

    errors: list[str] = []
    if not asset_id:
        errors.append("Asset ID is required.")
    elif asset_exists(asset_id):
        errors.append(f"Asset ID '{asset_id}' is already registered.")
    if not name:
        errors.append("Asset name is required.")
    if not zone:
        errors.append("Branch / zone name is required.")
    if baseline_file is None:
        errors.append("A baseline image is required.")
    if errors:
        for err in errors:
            st.error(err)
        return

    try:
        data = baseline_file.getvalue()
        decode_uploaded_bytes(data, baseline_file.name)
        suffix = Path(baseline_file.name).suffix.lower() or ".jpg"
        dest = BASELINES_DIR / f"{asset_id}{suffix}"
        save_upload(data, dest)
        insert_asset(asset_id, name, category, zone, str(dest))
    except ValueError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"Could not save the asset. {exc}")
        return

    st.success(f"Registered {asset_id} — {name} in {zone}.")
    st.image(dest, caption="Baseline (stored locally)", width="stretch")


def show_analysis(result: AnalysisResult, baseline_path: Path, current_path: Path) -> None:
    st.markdown("#### Comparison")
    left, right = st.columns(2)
    with left:
        st.caption("Baseline view")
        if baseline_path.exists():
            st.image(str(baseline_path), width="stretch")
        else:
            st.error("Baseline file is missing from disk.")
    with right:
        st.caption("Current view (annotated if YOLO ran)")
        if result.annotated_bgr is not None:
            rgb = result.annotated_bgr[:, :, ::-1]
            st.image(rgb, width="stretch")
        elif current_path.exists():
            st.image(str(current_path), width="stretch")

    st.markdown(
        f"**Status:** {pill(result.status)} &nbsp; · &nbsp; similarity {result.similarity:.0%} "
        f"· pixel-change {result.change_ratio:.0%}",
        unsafe_allow_html=True,
    )
    st.write(result.explanation)
    if result.detections:
        st.caption("YOLO category detections (not unique IDs)")
        st.dataframe(
            [
                {"Class": d["name"], "Confidence": d["confidence"], "Box": d["box"]}
                for d in result.detections
            ],
            hide_index=True,
            width="stretch",
        )
    else:
        st.caption("No YOLO boxes on this frame, or the detector was unavailable.")
    if result.error:
        st.warning(result.error)


def page_monitor(model_state) -> None:
    header(
        "Monitoring",
        "Image monitoring",
        "Upload a new photo of the same predefined camera view and compare it with the baseline.",
    )
    privacy_notes()

    assets = get_assets()
    if not assets:
        st.info("Register an asset first, or load demo records, then return here to analyse a new image.")
        return

    labels = {f"{a['asset_id']} — {a['name']} ({a['zone']})": a["asset_id"] for a in assets}
    choice = st.selectbox("Registered asset / zone", list(labels.keys()))
    asset = get_asset(labels[choice])
    if not asset:
        st.error("Selected asset could not be loaded.")
        return

    st.markdown(
        f"**Category:** `{asset['category']}` · **Zone:** {asset['zone']}  \n"
        "Use the same framing as the baseline. Lighting and clutter will affect comparison."
    )

    if not model_state.available:
        st.error(
            "YOLO is not available, so this run cannot claim object detection succeeded. "
            "You can still compare images visually, or use demo mode. " + model_state.message
        )

    upload = st.file_uploader(
        "Current zone image",
        type=list(ALLOWED_IMAGE_TYPES),
        key="current_image",
    )
    run = st.button("Run analysis", type="primary", disabled=upload is None)

    if not run:
        return
    if upload is None:
        st.error("Upload a current image first.")
        return

    try:
        data = upload.getvalue()
        current_bgr = decode_uploaded_bytes(data, upload.name)
    except ValueError as exc:
        st.error(str(exc))
        return

    suffix = Path(upload.name).suffix.lower() or ".jpg"
    current_path = CURRENT_DIR / f"{asset['asset_id']}_{Path(upload.name).stem}{suffix}"
    try:
        save_upload(data, current_path)
    except Exception as exc:
        st.error(f"Could not store the uploaded image locally. {exc}")
        return

    with st.spinner("Analysing locally..."):
        result = analyze_images(
            model_state,
            asset["baseline_path"],
            current_bgr,
            asset["category"],
        )

    evidence_path = None
    if result.annotated_bgr is not None:
        evidence_path = str(EVIDENCE_DIR / f"{asset['asset_id']}_{Path(current_path).stem}_evidence.jpg")
        try:
            save_bgr_image(result.annotated_bgr, Path(evidence_path))
        except Exception as exc:
            st.warning(f"Evidence image could not be written: {exc}")
            evidence_path = str(current_path)
    else:
        evidence_path = str(current_path)

    event_id = insert_event(
        asset_id=asset["asset_id"],
        status=result.status,
        explanation=result.explanation,
        evidence_path=evidence_path,
        source="live",
        similarity=result.similarity,
        detections_json=detections_to_json(result.detections),
    )
    st.success(f"Event #{event_id} saved locally and will appear on the dashboard and history.")
    show_analysis(result, Path(asset["baseline_path"]), Path(current_path))


def page_history() -> None:
    header(
        "Audit",
        "Event history",
        "Search and filter stored events. Evidence files remain in the local evidence folder.",
    )
    c1, c2, c3 = st.columns([2, 1, 1])
    search = c1.text_input("Search (asset ID, name, zone, explanation)", "")
    status = c2.selectbox(
        "Status",
        ["All", STATUS_UNCHANGED, STATUS_POSSIBLE, STATUS_NOT_DETECTED],
    )
    source = c3.selectbox("Source", ["All", "live", "demo"])
    events = get_events(search=search, status=status, source=source, limit=300)

    if not events:
        st.markdown(
            '<div class="iw-card iw-empty">No matching events. Adjust filters, run monitoring, or load demo records.</div>',
            unsafe_allow_html=True,
        )
        return

    st.caption(f"{len(events)} row(s). LIVE = image analysis on this machine. DEMO = labelled synthetic records.")
    table = []
    for event in events:
        table.append(
            {
                "ID": event["id"],
                "Timestamp": event["timestamp"],
                "Asset ID": event["asset_id"],
                "Zone": event.get("zone") or "—",
                "Status": event["status"],
                "Source": event["source"].upper(),
                "Explanation": event["explanation"],
            }
        )
    st.dataframe(table, width="stretch", hide_index=True)

    ids = [e["id"] for e in events]
    selected = st.selectbox("Open evidence for event", ids, format_func=lambda i: f"#{i}")
    chosen = next(e for e in events if e["id"] == selected)
    st.markdown(
        f"{pill(chosen['status'])} &nbsp; `{chosen['asset_id']}` &nbsp; · &nbsp; {chosen['timestamp']}",
        unsafe_allow_html=True,
    )
    st.write(chosen["explanation"])
    path = chosen.get("evidence_path")
    if path and Path(path).exists():
        st.image(path, caption=f"Local evidence · {path}", width="stretch")
        st.caption("File is stored only on this computer.")
    elif path:
        st.error(f"Evidence file is missing: {path}")
    else:
        st.info("This event has no evidence image path.")


def main() -> None:
    bootstrap()
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    page, model_state = render_sidebar()
    if st.session_state.get("flash"):
        st.sidebar.success(st.session_state.pop("flash"))

    if page == "Dashboard":
        page_dashboard(model_state)
    elif page == "Register asset":
        page_register()
    elif page == "Image monitoring":
        page_monitor(model_state)
    else:
        page_history()


main()
