from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

from .promote import classify
from .search import generate_neighbors
from .spec import expand_path, load_campaign


def run(cmd: list[str], env: dict[str, str]) -> None:
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def screen_summary(db_path: Path, campaign: str, candidate: str, prompt_id: str) -> tuple[float, float, int] | None:
    if not db_path.exists():
        return None
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        """
        SELECT AVG(tps), AVG(rounds), COUNT(*)
        FROM runs
        WHERE campaign = ? AND candidate = ? AND stage = 'screen' AND prompt_id = ?
        """,
        (campaign, candidate, prompt_id),
    ).fetchone()
    if not row or not row[2]:
        return None
    return float(row[0]), float(row[1]), int(row[2])


def main() -> None:
    ap = argparse.ArgumentParser(description="Budgeted v0 campaign runner: neighbors -> build -> validate -> screen -> optional DEV/production.")
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--screen-runs", type=int, default=1)
    ap.add_argument("--db", default="results/tuning.sqlite")
    ap.add_argument("--candidate-root", default="candidates/generated")
    ap.add_argument("--run-dev", action="store_true")
    ap.add_argument("--run-production", action="store_true")
    ap.add_argument("--production-runs", type=int, default=10)
    ap.add_argument("--prompt-id", default="canonical-lru-python")
    args = ap.parse_args()

    cfg = load_campaign(args.campaign)
    name = cfg["campaign"]["name"]
    output_root = expand_path(cfg["campaign"]["output_dir"])
    db_path = expand_path(args.db)
    candidate_root = Path(args.candidate_root) / name
    candidate_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["MLX_QMV_FAST_M4"] = "1"
    env.pop("MLX_QMV_FAST_M3", None)

    neighbors = generate_neighbors(cfg)[: args.budget]
    print(f"Campaign {name}: evaluating {len(neighbors)} of {len(generate_neighbors(cfg))} local neighbors")

    advanced: list[str] = []
    for index, cand in enumerate(neighbors, 1):
        cid = cand["id"]
        cand_file = candidate_root / f"{cid}.json"
        cand_file.write_text(json.dumps(cand, indent=2) + "\n")
        draft_dir = output_root / cid

        print(f"\n{'=' * 72}\n[{index}/{len(neighbors)}] {cid} mutation={cand['mutation']}\n{'=' * 72}")

        if not (draft_dir / "autotune-manifest.json").exists():
            run([
                sys.executable, "-m", "tuner.build",
                "--campaign", args.campaign,
                "--candidate", str(cand_file),
            ], env)
        else:
            print("build exists; reusing", draft_dir)

        run([
            sys.executable, "-m", "tuner.validate",
            "--draft", str(draft_dir),
            "--candidate", str(cand_file),
        ], env)

        existing = screen_summary(db_path, name, cid, args.prompt_id)
        if not existing or existing[2] < args.screen_runs:
            needed = args.screen_runs - (existing[2] if existing else 0)
            run([
                sys.executable, "-m", "tuner.benchmark",
                "--campaign", args.campaign,
                "--draft", str(draft_dir),
                "--candidate-id", cid,
                "--stage", "screen",
                "--runs", str(needed),
                "--warmups", "1",
                "--db", str(db_path),
                "--prompt-id", args.prompt_id,
            ], env)

        summary = screen_summary(db_path, name, cid, args.prompt_id)
        assert summary is not None
        mean_tps, mean_rounds, n = summary
        decision, reason = classify(cfg, mean_tps, mean_rounds)
        print(f"SCREEN {cid}: n={n} mean={mean_tps:.3f} rounds={mean_rounds:.2f} => {decision.upper()} ({reason})")

        if decision != "advance":
            continue
        advanced.append(cid)

        if args.run_dev:
            run([
                sys.executable, "-m", "tuner.suite",
                "--campaign", args.campaign,
                "--draft", str(draft_dir),
                "--candidate-id", cid,
                "--stage", "dev",
                "--db", str(db_path),
            ], env)

        if args.run_production:
            run([
                sys.executable, "-m", "tuner.benchmark",
                "--campaign", args.campaign,
                "--draft", str(draft_dir),
                "--candidate-id", cid,
                "--stage", "production",
                "--runs", str(args.production_runs),
                "--warmups", "1",
                "--db", str(db_path),
                "--prompt-id", args.prompt_id,
            ], env)

    print("\n========== CAMPAIGN RUN COMPLETE ==========")
    print("advanced:", advanced if advanced else "none")
    print("database:", db_path)
    print("Champion is NOT updated automatically in v0; promotion remains explicit.")


if __name__ == "__main__":
    main()
