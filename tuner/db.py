from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    campaign TEXT NOT NULL,
    candidate TEXT NOT NULL,
    stage TEXT NOT NULL,
    prompt_id TEXT NOT NULL,
    run_index INTEGER NOT NULL,
    target_path TEXT NOT NULL,
    draft_path TEXT NOT NULL,
    tokens INTEGER NOT NULL,
    tps REAL NOT NULL,
    wall_seconds REAL NOT NULL,
    rounds INTEGER,
    peak_gb REAL,
    text_sha256 TEXT NOT NULL,
    modules_json TEXT,
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_campaign_candidate
ON runs(campaign, candidate);

CREATE INDEX IF NOT EXISTS idx_runs_stage_prompt
ON runs(stage, prompt_id);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p)
    conn.executescript(SCHEMA)
    return conn


def insert_run(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    cols = [
        "campaign", "candidate", "stage", "prompt_id", "run_index",
        "target_path", "draft_path", "tokens", "tps", "wall_seconds",
        "rounds", "peak_gb", "text_sha256", "modules_json", "notes",
    ]
    values = [row.get(c) for c in cols]
    placeholders = ",".join("?" for _ in cols)
    conn.execute(
        f"INSERT INTO runs ({','.join(cols)}) VALUES ({placeholders})",
        values,
    )
    conn.commit()


def candidate_summary(conn: sqlite3.Connection, campaign: str, candidate: str) -> list[tuple]:
    return conn.execute(
        """
        SELECT stage, prompt_id,
               COUNT(*) AS n,
               AVG(tps) AS mean_tps,
               MIN(tps) AS min_tps,
               MAX(tps) AS max_tps,
               AVG(rounds) AS mean_rounds
        FROM runs
        WHERE campaign = ? AND candidate = ?
        GROUP BY stage, prompt_id
        ORDER BY stage, prompt_id
        """,
        (campaign, candidate),
    ).fetchall()
