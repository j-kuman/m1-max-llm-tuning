from __future__ import annotations

import argparse
import sqlite3


def main() -> None:
    ap = argparse.ArgumentParser(description="Show candidate leaderboard from the tuning SQLite database.")
    ap.add_argument("--db", default="results/tuning.sqlite")
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--stage", default="production")
    ap.add_argument("--prompt-id", default="canonical-lru-python")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    rows = conn.execute(
        """
        SELECT candidate,
               COUNT(*) AS n,
               AVG(tps) AS mean_tps,
               MIN(tps) AS min_tps,
               MAX(tps) AS max_tps,
               AVG(rounds) AS mean_rounds,
               MAX(peak_gb) AS peak_gb
        FROM runs
        WHERE campaign = ? AND stage = ? AND prompt_id = ?
        GROUP BY candidate
        ORDER BY mean_tps DESC
        LIMIT ?
        """,
        (args.campaign, args.stage, args.prompt_id, args.limit),
    ).fetchall()

    if not rows:
        print("no matching results")
        return

    print(f"{'candidate':18s} {'n':>3s} {'mean':>8s} {'min':>8s} {'max':>8s} {'rounds':>8s} {'peakGB':>8s}")
    for candidate, n, mean_tps, min_tps, max_tps, rounds, peak in rows:
        print(
            f"{candidate:18s} {n:3d} {mean_tps:8.3f} {min_tps:8.3f} "
            f"{max_tps:8.3f} {rounds:8.2f} {peak:8.3f}"
        )


if __name__ == "__main__":
    main()
