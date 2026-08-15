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
    tokens_per_round REAL,
    peak_gb REAL,
    text_sha256 TEXT NOT NULL,
    modules_json TEXT,
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_campaign_candidate
ON runs(campaign, candidate);

CREATE INDEX IF NOT EXISTS idx_runs_stage_prompt
ON runs(stage, prompt_id);

CREATE TABLE IF NOT EXISTS microbench_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    campaign TEXT NOT NULL,
    candidate TEXT NOT NULL,
    component TEXT NOT NULL,
    run_index INTEGER NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    unit TEXT NOT NULL,
    direction TEXT NOT NULL,
    command_json TEXT NOT NULL,
    stdout_sha256 TEXT NOT NULL,
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_microbench_campaign_candidate
ON microbench_runs(campaign, candidate, component);
"""


def _migrate(conn: sqlite3.Connection) -> None:
    """Apply additive migrations to DBs created by earlier tuner versions."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(runs)")}
    if "tokens_per_round" not in cols:
        conn.execute("ALTER TABLE runs ADD COLUMN tokens_per_round REAL")
        conn.commit()


def connect(path: str | Path) -> sqlite3.Connection:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p)
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


def insert_run(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    cols = [
        "campaign", "candidate", "stage", "prompt_id", "run_index",
        "target_path", "draft_path", "tokens", "tps", "wall_seconds",
        "rounds", "tokens_per_round", "peak_gb", "text_sha256",
        "modules_json", "notes",
    ]
    values = [row.get(c) for c in cols]
    placeholders = ",".join("?" for _ in cols)
    conn.execute(
        f"INSERT INTO runs ({','.join(cols)}) VALUES ({placeholders})",
        values,
    )
    conn.commit()


def insert_microbench(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    cols = [
        "campaign", "candidate", "component", "run_index", "metric_name",
        "metric_value", "unit", "direction", "command_json", "stdout_sha256",
        "notes",
    ]
    values = [row.get(c) for c in cols]
    placeholders = ",".join("?" for _ in cols)
    conn.execute(
        f"INSERT INTO microbench_runs ({','.join(cols)}) VALUES ({placeholders})",
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
               AVG(rounds) AS mean_rounds,
               AVG(tokens_per_round) AS mean_tokens_per_round
        FROM runs
        WHERE campaign = ? AND candidate = ?
        GROUP BY stage, prompt_id
        ORDER BY stage, prompt_id
        """,
        (campaign, candidate),
    ).fetchall()
