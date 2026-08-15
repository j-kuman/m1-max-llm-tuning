from __future__ import annotations

import argparse
import sqlite3

from .spec import load_campaign


def classify(cfg: dict, mean_tps: float, mean_rounds: float) -> tuple[str, str]:
    champion = cfg["champion"]
    rules = cfg.get("promotion", {})
    ref_tps = float(champion["mean_tps"])
    ref_rounds = int(champion["rounds"])
    delta = round(mean_rounds) - ref_rounds
    ratio = mean_tps / ref_tps

    if delta < 0:
        return "advance", f"verification rounds improved by {-delta}; always advance"
    if delta == 0:
        need = float(rules.get("equal_round_min_tps_ratio", 0.995))
        return ("advance", f"same rounds and TPS ratio {ratio:.4f} >= {need:.4f}") if ratio >= need else ("reject", f"same rounds but TPS ratio {ratio:.4f} < {need:.4f}")
    if delta == 1:
        need = float(rules.get("one_extra_round_min_tps_ratio", 1.005))
        return ("advance", f"one extra round but TPS ratio {ratio:.4f} >= {need:.4f}") if ratio >= need else ("reject", f"one extra round and TPS ratio {ratio:.4f} < {need:.4f}")
    reject_at = int(rules.get("reject_round_delta_at_or_above", 2))
    if delta >= reject_at:
        return "reject", f"round delta +{delta} meets automatic reject threshold +{reject_at}"
    return "review", f"round delta +{delta}; no rule matched"


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply relative screen rules to a candidate already stored in SQLite.")
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--candidate-id", required=True)
    ap.add_argument("--db", default="results/tuning.sqlite")
    ap.add_argument("--stage", default="screen")
    ap.add_argument("--prompt-id", default="canonical-lru-python")
    args = ap.parse_args()

    cfg = load_campaign(args.campaign)
    conn = sqlite3.connect(args.db)
    row = conn.execute(
        """
        SELECT AVG(tps), AVG(rounds), COUNT(*)
        FROM runs
        WHERE campaign = ? AND candidate = ? AND stage = ? AND prompt_id = ?
        """,
        (cfg["campaign"]["name"], args.candidate_id, args.stage, args.prompt_id),
    ).fetchone()
    if not row or not row[2]:
        raise SystemExit("no matching runs found")

    mean_tps, mean_rounds, count = row
    decision, reason = classify(cfg, float(mean_tps), float(mean_rounds))
    print("candidate:", args.candidate_id)
    print("runs:", count)
    print("mean tps:", f"{mean_tps:.4f}")
    print("mean rounds:", f"{mean_rounds:.2f}")
    print("decision:", decision.upper())
    print("reason:", reason)


if __name__ == "__main__":
    main()
