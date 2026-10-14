"""SQLite persistence for assets and monitoring events."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from paths import DB_PATH, ensure_directories

STATUS_UNCHANGED = "UNCHANGED / DETECTED"
STATUS_POSSIBLE = "POSSIBLE CHANGE"
STATUS_NOT_DETECTED = "NOT DETECTED — VERIFY"

OPEN_WARNING_STATUSES = (STATUS_POSSIBLE, STATUS_NOT_DETECTED)


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    ensure_directories()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assets (
                asset_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                zone TEXT NOT NULL,
                baseline_path TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                asset_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                status TEXT NOT NULL,
                explanation TEXT NOT NULL,
                evidence_path TEXT,
                source TEXT NOT NULL DEFAULT 'live',
                similarity REAL,
                detections_json TEXT,
                FOREIGN KEY (asset_id) REFERENCES assets(asset_id)
            )
            """
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def insert_asset(
    asset_id: str,
    name: str,
    category: str,
    zone: str,
    baseline_path: str,
    created_at: str | None = None,
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO assets (asset_id, name, category, zone, baseline_path, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (asset_id, name, category, zone, baseline_path, created_at or utc_now()),
        )


def get_assets() -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM assets ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_asset(asset_id: str) -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM assets WHERE asset_id = ?", (asset_id,)
        ).fetchone()
    return dict(row) if row else None


def asset_exists(asset_id: str) -> bool:
    return get_asset(asset_id) is not None


def insert_event(
    asset_id: str,
    status: str,
    explanation: str,
    evidence_path: str | None,
    source: str = "live",
    similarity: float | None = None,
    detections_json: str | None = None,
    timestamp: str | None = None,
) -> int:
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO events (
                asset_id, timestamp, status, explanation, evidence_path,
                source, similarity, detections_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                asset_id,
                timestamp or utc_now(),
                status,
                explanation,
                evidence_path,
                source,
                similarity,
                detections_json,
            ),
        )
        return int(cursor.lastrowid)


def get_events(
    search: str = "",
    status: str = "All",
    source: str = "All",
    limit: int = 200,
) -> list[dict[str, Any]]:
    query = """
        SELECT e.*, a.name AS asset_name, a.zone, a.category
        FROM events e
        LEFT JOIN assets a ON a.asset_id = e.asset_id
        WHERE 1=1
    """
    params: list[Any] = []
    if search.strip():
        query += """
            AND (
                e.asset_id LIKE ?
                OR IFNULL(a.name, '') LIKE ?
                OR IFNULL(a.zone, '') LIKE ?
                OR e.explanation LIKE ?
            )
        """
        like = f"%{search.strip()}%"
        params.extend([like, like, like, like])
    if status != "All":
        query += " AND e.status = ?"
        params.append(status)
    if source != "All":
        query += " AND e.source = ?"
        params.append(source)
    query += " ORDER BY e.timestamp DESC, e.id DESC LIMIT ?"
    params.append(limit)

    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def get_dashboard_stats() -> dict[str, Any]:
    with get_connection() as conn:
        total_assets = conn.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
        open_warnings = conn.execute(
            f"""
            SELECT COUNT(*) FROM (
                SELECT e.asset_id, e.status
                FROM events e
                INNER JOIN (
                    SELECT asset_id, MAX(id) AS max_id
                    FROM events
                    GROUP BY asset_id
                ) latest ON latest.max_id = e.id
                WHERE e.status IN ({",".join("?" for _ in OPEN_WARNING_STATUSES)})
            )
            """,
            OPEN_WARNING_STATUSES,
        ).fetchone()[0]
        detected_ok = conn.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT e.asset_id
                FROM events e
                INNER JOIN (
                    SELECT asset_id, MAX(id) AS max_id
                    FROM events
                    GROUP BY asset_id
                ) latest ON latest.max_id = e.id
                WHERE e.status = ?
            )
            """,
            (STATUS_UNCHANGED,),
        ).fetchone()[0]
        live_events = conn.execute(
            "SELECT COUNT(*) FROM events WHERE source = 'live'"
        ).fetchone()[0]
        demo_events = conn.execute(
            "SELECT COUNT(*) FROM events WHERE source = 'demo'"
        ).fetchone()[0]

    return {
        "total_assets": total_assets,
        "assets_detected": detected_ok,
        "open_warnings": open_warnings,
        "live_events": live_events,
        "demo_events": demo_events,
    }


def demo_already_loaded() -> bool:
    with get_connection() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM events WHERE source = 'demo'"
        ).fetchone()[0]
    return count > 0
