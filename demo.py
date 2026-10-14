"""Clearly labelled synthetic demo records — not real monitoring data."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

from database import (
    STATUS_NOT_DETECTED,
    STATUS_POSSIBLE,
    STATUS_UNCHANGED,
    asset_exists,
    demo_already_loaded,
    insert_asset,
    insert_event,
)
from paths import DEMO_DIR, ensure_directories
from vision import save_bgr_image


def _banner(image: np.ndarray, text: str, color: tuple[int, int, int]) -> None:
    h, w = image.shape[:2]
    cv2.rectangle(image, (0, 0), (w, 46), color, -1)
    cv2.putText(
        image,
        text,
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "SYNTHETIC / NOT REAL BRANCH IMAGERY",
        (16, h - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _box(image: np.ndarray, xyxy: tuple[int, int, int, int], label: str, color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = xyxy
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
    cv2.putText(
        image,
        label,
        (x1 + 8, y1 + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


def _make_scene(
    filename: str,
    bg: tuple[int, int, int],
    objects: list[tuple[tuple[int, int, int, int], str, tuple[int, int, int]]],
    banner: str,
    banner_color: tuple[int, int, int],
) -> Path:
    ensure_directories()
    image = np.zeros((480, 720, 3), dtype=np.uint8)
    image[:] = bg
    # desk surface
    cv2.rectangle(image, (40, 300), (680, 450), (70, 78, 86), -1)
    for xyxy, label, color in objects:
        _box(image, xyxy, label, color)
    _banner(image, banner, banner_color)
    path = DEMO_DIR / filename
    save_bgr_image(image, path)
    return path


def load_demo_records() -> str:
    """Insert labelled sample assets and events if they are not already present."""
    ensure_directories()
    if demo_already_loaded():
        return (
            "Demo records are already in the local database. "
            "They remain labelled as DEMO and are not YOLO results."
        )

    samples = [
        {
            "asset_id": "DEMO-PRN-001",
            "name": "Sample floor printer (illustrative)",
            "category": "printer",
            "zone": "Demo Branch — Operations Desk",
            "status": STATUS_UNCHANGED,
            "explanation": (
                "DEMO RECORD — not produced by YOLO. Synthetic scene labelled as a printer "
                "in a stable camera view. This is for dashboard walkthrough only."
            ),
            "baseline": ("demo_prn_baseline.jpg", (90, 96, 104), [((430, 160, 650, 340), "DEMO printer", (40, 170, 80))]),
            "evidence": ("demo_prn_evidence.jpg", (90, 96, 104), [((430, 160, 650, 340), "DEMO printer", (40, 170, 80))]),
            "hours_ago": 3,
        },
        {
            "asset_id": "DEMO-MON-014",
            "name": "Sample teller monitor (illustrative)",
            "category": "monitor",
            "zone": "Demo Branch — Teller Window 2",
            "status": STATUS_POSSIBLE,
            "explanation": (
                "DEMO RECORD — not produced by YOLO. Synthetic scene where the labelled monitor "
                "region has moved. Flagged as POSSIBLE CHANGE for UI demonstration only."
            ),
            "baseline": ("demo_mon_baseline.jpg", (82, 88, 98), [((80, 90, 340, 280), "DEMO monitor", (200, 160, 40))]),
            "evidence": ("demo_mon_evidence.jpg", (70, 74, 84), [((300, 110, 560, 300), "DEMO monitor (moved)", (30, 160, 210))]),
            "hours_ago": 2,
        },
        {
            "asset_id": "DEMO-LAP-007",
            "name": "Sample relationship-manager laptop (illustrative)",
            "category": "laptop",
            "zone": "Demo Branch — RM Cubicle A",
            "status": STATUS_NOT_DETECTED,
            "explanation": (
                "DEMO RECORD — not produced by YOLO. Synthetic empty desk used to show the "
                "NOT DETECTED — VERIFY workflow. This is not a theft finding and has no model confidence."
            ),
            "baseline": ("demo_lap_baseline.jpg", (88, 92, 100), [((180, 140, 520, 320), "DEMO laptop", (40, 90, 200))]),
            "evidence": ("demo_lap_evidence.jpg", (60, 64, 72), []),
            "hours_ago": 1,
        },
        {
            "asset_id": "DEMO-KBD-022",
            "name": "Sample shared keyboard (illustrative)",
            "category": "keyboard",
            "zone": "Demo Branch — Back-office Bench",
            "status": STATUS_UNCHANGED,
            "explanation": (
                "DEMO RECORD — not produced by YOLO. Synthetic labelled keyboard scene for "
                "history-table filtering. Categories are examples, not confirmed L&T Finance inventory."
            ),
            "baseline": ("demo_kbd_baseline.jpg", (96, 100, 108), [((160, 310, 500, 390), "DEMO keyboard", (90, 90, 200))]),
            "evidence": ("demo_kbd_evidence.jpg", (96, 100, 108), [((160, 310, 500, 390), "DEMO keyboard", (90, 90, 200))]),
            "hours_ago": 5,
        },
    ]

    created_assets = 0
    created_events = 0
    now = datetime.now(timezone.utc)

    for sample in samples:
        b_name, b_bg, b_objs = sample["baseline"]
        e_name, e_bg, e_objs = sample["evidence"]
        baseline_path = _make_scene(
            b_name,
            b_bg,
            b_objs,
            "DEMO BASELINE — NOT A REAL BRANCH PHOTO",
            (11, 79, 108),
        )
        evidence_path = _make_scene(
            e_name,
            e_bg,
            e_objs,
            f"DEMO EVIDENCE — {sample['status']}",
            (180, 80, 20) if sample["status"] != STATUS_UNCHANGED else (20, 120, 70),
        )

        if not asset_exists(sample["asset_id"]):
            insert_asset(
                asset_id=sample["asset_id"],
                name=sample["name"],
                category=sample["category"],
                zone=sample["zone"],
                baseline_path=str(baseline_path),
                created_at=(now - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S UTC"),
            )
            created_assets += 1

        stamp = (now - timedelta(hours=sample["hours_ago"])).strftime("%Y-%m-%d %H:%M:%S UTC")
        insert_event(
            asset_id=sample["asset_id"],
            status=sample["status"],
            explanation=sample["explanation"],
            evidence_path=str(evidence_path),
            source="demo",
            similarity=None,
            detections_json=None,
            timestamp=stamp,
        )
        created_events += 1

    return (
        f"Loaded {created_events} labelled demo events"
        f"{f' and {created_assets} demo assets' if created_assets else ''} "
        "(synthetic data, clearly marked, not model output)."
    )
