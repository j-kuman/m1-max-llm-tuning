from __future__ import annotations

import argparse
from dataclasses import dataclass
import sqlite3

from .db import connect
from .gates import champion_reference_id, load_reference
from .spec import load_campaign


@dataclass(frozen=True)
class ScreenStats:
    mean_tps: float
    mean_rounds: float
    mean_tokens: float
    mean_tokens_per_round: float
    count: int
    min_tokens: int
    max_tokens: int
    min_rounds: int
    max_rounds: int
    distinct_hashes: int
    text_sha256: str


def load_screen_stats(
    conn: sqlite3.Connection,
    campaign: str,
    candidate: str,
    prompt_id: str,
    stage: str = "screen",
    *,
    latest_n: int | None = None,
) -> ScreenStats | None:
    """Load aggregate screen stats, optionally restricted to the newest N rows.

    The latest-N mode is important when a candidate has been measured in multiple
    macOS performance states. A fresh live champion control must only be compared
    with candidate rows from the same tuning session, not an average containing
    stale historical screens.
    """
    if latest_n is not None:
        if latest_n < 1:
            raise ValueError("latest_n must be >= 1")
        row = conn.execute(
            """
            SELECT AVG(tps), AVG(rounds), AVG(tokens),
                   AVG(COALESCE(tokens_per_round, CASE WHEN rounds > 0 THEN 1.0 * tokens / rounds END)),
                   COUNT(*), MIN(tokens), MAX(tokens), MIN(rounds), MAX(rounds),
                   COUNT(DISTINCT text_sha256), MIN(text_sha256)
            FROM (
                SELECT tps, rounds, tokens, tokens_per_round, text_sha256
                FROM runs
                WHERE campaign = ? AND candidate = ? AND stage = ? AND prompt_id = ?
                ORDER BY id DESC
                LIMIT ?
            )
            """,
            (campaign, candidate, stage, prompt_id, latest_n),
        ).fetchone()
    else:
        row = conn.execute(
            """
            SELECT AVG(tps), AVG(rounds), AVG(tokens),
                   AVG(COALESCE(tokens_per_round, CASE WHEN rounds > 0 THEN 1.0 * tokens / rounds END)),
                   COUNT(*), MIN(tokens), MAX(tokens), MIN(rounds), MAX(rounds),
                   COUNT(DISTINCT text_sha256), MIN(text_sha256)
            FROM runs
            WHERE campaign = ? AND candidate = ? AND stage = ? AND prompt_id = ?
            """,
            (campaign, candidate, stage, prompt_id),
        ).fetchone()
    if not row or not row[4]:
        return None
    if row[3] is None:
        raise RuntimeError("candidate has no valid positive-round efficiency data")
    return ScreenStats(
        mean_tps=float(row[0]),
        mean_rounds=float(row[1]),
        mean_tokens=float(row[2]),
        mean_tokens_per_round=float(row[3]),
        count=int(row[4]),
        min_tokens=int(row[5]),
        max_tokens=int(row[6]),
        min_rounds=int(row[7]),
        max_rounds=int(row[8]),
        distinct_hashes=int(row[9]),
        text_sha256=str(row[10]),
    )


def classify(
    cfg: dict,
    stats: ScreenStats,
    *,
    reference_tokens: int,
    reference_hash: str,
) -> tuple[str, str]:
    """Classify a canonical screen only after correctness invariants pass."""
    champion = cfg["champion"]
    rules = cfg.get("promotion", {})
    ref_tps = float(champion["mean_tps"])
    ref_rounds = int(champion["rounds"])
    ref_tokens = int(champion.get("tokens", cfg["campaign"].get("max_tokens", reference_tokens)))

    if stats.distinct_hashes != 1 or stats.text_sha256 != reference_hash:
        return "reject", "exact output hash does not match the frozen champion reference"
    if stats.min_tokens != stats.max_tokens or stats.min_tokens != reference_tokens:
        return "reject", (
            f"token count is not reference-exact: range={stats.min_tokens}-{stats.max_tokens}, "
            f"reference={reference_tokens}"
        )
    if stats.min_tokens != ref_tokens:
        return "reject", f"candidate tokens={stats.min_tokens} != champion canonical tokens={ref_tokens}"
    if stats.min_rounds <= 0:
        return "reject", f"invalid verification round count {stats.min_rounds}; telemetry failed closed"
    if stats.min_rounds != stats.max_rounds:
        return "reject", (
            f"verification rounds are nondeterministic across identical greedy runs: "
            f"{stats.min_rounds}-{stats.max_rounds}"
        )

    rounds = stats.min_rounds
    delta = rounds - ref_rounds
    ratio = stats.mean_tps / ref_tps
    ref_tpr = ref_tokens / ref_rounds
    tpr_ratio = stats.mean_tokens_per_round / ref_tpr

    if delta < 0:
        return "advance", (
            f"verification rounds improved by {-delta}; tok/round ratio={tpr_ratio:.4f}; exactness PASS"
        )
    if delta == 0:
        need = float(rules.get("equal_round_min_tps_ratio", 0.995))
        return (
            ("advance", f"same rounds, TPS ratio {ratio:.4f} >= {need:.4f}; exactness PASS")
            if ratio >= need
            else ("reject", f"same rounds but TPS ratio {ratio:.4f} < {need:.4f}")
        )
    if delta == 1:
        need = float(rules.get("one_extra_round_min_tps_ratio", 1.005))
        return (
            ("advance", f"one extra round but TPS ratio {ratio:.4f} >= {need:.4f}; exactness PASS")
            if ratio >= need
            else ("reject", f"one extra round and TPS ratio {ratio:.4f} < {need:.4f}")
        )
    reject_at = int(rules.get("reject_round_delta_at_or_above", 2))
    if delta >= reject_at:
        return "reject", f"round delta +{delta} meets automatic reject threshold +{reject_at}"
    return "review", f"round delta +{delta}; no rule matched"


def _latest_dev_rows(
    conn: sqlite3.Connection,
    campaign: str,
    candidate: str,
) -> dict[str, tuple]:
    """Return the newest DEV row for each prompt for one candidate/control."""
    rows = conn.execute(
        """
        SELECT r.prompt_id, r.tps,
               COALESCE(r.tokens_per_round, CASE WHEN r.rounds > 0 THEN 1.0 * r.tokens / r.rounds END),
               r.tokens, r.text_sha256
        FROM runs AS r
        JOIN (
            SELECT prompt_id, MAX(id) AS max_id
            FROM runs
            WHERE campaign = ? AND candidate = ? AND stage = 'dev'
            GROUP BY prompt_id
        ) AS newest ON newest.max_id = r.id
        ORDER BY r.prompt_id
        """,
        (campaign, candidate),
    ).fetchall()
    return {str(r[0]): r for r in rows}


def _frozen_dev_reference_rows(
    conn: sqlite3.Connection,
    campaign: str,
    reference_candidate: str,
) -> dict[str, tuple]:
    rows = conn.execute(
        """
        SELECT prompt_id, AVG(tps),
               AVG(COALESCE(tokens_per_round, CASE WHEN rounds > 0 THEN 1.0 * tokens / rounds END)),
               MIN(tokens), MAX(tokens), COUNT(DISTINCT text_sha256), MIN(text_sha256)
        FROM runs
        WHERE campaign = ? AND candidate = ? AND stage = 'reference'
        GROUP BY prompt_id
        """,
        (campaign, reference_candidate),
    ).fetchall()
    return {str(r[0]): r for r in rows}


def evaluate_dev_gate(
    cfg: dict,
    conn: sqlite3.Connection,
    candidate: str,
    reference_candidate: str,
    performance_reference_candidate: str | None = None,
) -> tuple[str, str, dict]:
    """Compare DEV against frozen correctness + optional live performance control.

    Token count, output hash and speculative efficiency are compared with the
    frozen champion reference. TPS is compared with a same-session live champion
    DEV control when supplied, preventing stale machine-state measurements from
    biasing promotion decisions.
    """
    campaign = cfg["campaign"]["name"]
    rules = cfg.get("promotion", {})

    cand = _latest_dev_rows(conn, campaign, candidate)
    refs = _frozen_dev_reference_rows(conn, campaign, reference_candidate)

    if performance_reference_candidate:
        perf_refs = _latest_dev_rows(conn, campaign, performance_reference_candidate)
    else:
        perf_refs = {
            prompt_id: (prompt_id, row[1], row[2], row[3], row[6])
            for prompt_id, row in refs.items()
        }

    required = int(rules.get("dev_min_prompts", 8))
    if len(cand) < required:
        return "reject", f"DEV has only {len(cand)} prompts; need at least {required}", {}

    prompt_ids = set(cand)
    missing_frozen = sorted(prompt_ids - set(refs))
    if missing_frozen:
        return "reject", f"DEV frozen reference coverage missing prompts: {missing_frozen}", {}
    missing_perf = sorted(prompt_ids - set(perf_refs))
    if missing_perf:
        return "reject", f"DEV live performance control missing prompts: {missing_perf}", {}

    tpr_ratios: list[float] = []
    tps_ratios: list[float] = []
    strict_eff_wins = 0
    exact = 0

    for prompt_id in sorted(cand):
        c = cand[prompt_id]
        r = refs[prompt_id]
        p = perf_refs[prompt_id]

        c_tps = float(c[1])
        c_tpr = c[2]
        r_tpr = r[2]
        p_tps = float(p[1])
        if c_tpr is None or r_tpr is None:
            return "reject", f"DEV invalid round telemetry on {prompt_id}", {}
        c_tpr = float(c_tpr)
        r_tpr = float(r_tpr)
        if c_tpr <= 0 or r_tpr <= 0 or p_tps <= 0:
            return "reject", f"DEV non-positive performance/efficiency value on {prompt_id}", {}

        c_tokens, c_hash = int(c[3]), str(c[4])
        r_min_tok, r_max_tok, r_hashes, r_hash = int(r[3]), int(r[4]), int(r[5]), str(r[6])
        if r_hashes != 1 or c_hash != r_hash:
            return "reject", f"DEV exactness failure on {prompt_id}", {}
        if r_min_tok != r_max_tok or c_tokens != r_min_tok:
            return "reject", f"DEV token-count exactness failure on {prompt_id}", {}

        exact += 1
        tpr_ratios.append(c_tpr / r_tpr)
        tps_ratios.append(c_tps / p_tps)
        if c_tpr > r_tpr:
            strict_eff_wins += 1

    mean_tpr_ratio = sum(tpr_ratios) / len(tpr_ratios)
    mean_tps_ratio = sum(tps_ratios) / len(tps_ratios)
    per_prompt_floor = float(rules.get("dev_min_prompt_tpr_ratio", 0.995))
    non_regress = sum(r >= per_prompt_floor for r in tpr_ratios)
    min_non_regress = int(rules.get("dev_min_non_regress_prompts", max(1, len(tpr_ratios) - 2)))
    min_mean_tpr = float(rules.get("dev_min_mean_tpr_ratio", 0.995))
    min_mean_tps = float(rules.get("dev_min_mean_tps_ratio", 0.995))

    screen_rounds = load_screen_stats(
        conn, campaign, candidate, "canonical-lru-python", "screen", latest_n=1
    )
    screen_improved = bool(screen_rounds and screen_rounds.min_rounds < int(cfg["champion"]["rounds"]))
    min_wins = int(rules.get("dev_min_efficiency_wins_if_screen_improves", 5))

    details = {
        "prompts": len(tpr_ratios),
        "exact_prompts": exact,
        "mean_tpr_ratio": mean_tpr_ratio,
        "mean_tps_ratio": mean_tps_ratio,
        "non_regress_prompts": non_regress,
        "strict_efficiency_wins": strict_eff_wins,
        "screen_rounds_improved": screen_improved,
        "performance_reference_candidate": performance_reference_candidate or reference_candidate,
    }

    if non_regress < min_non_regress:
        return "reject", f"DEV tok/round non-regression held on {non_regress}/{len(tpr_ratios)} prompts; need {min_non_regress}", details
    if mean_tpr_ratio < min_mean_tpr:
        return "reject", f"DEV mean tok/round ratio {mean_tpr_ratio:.4f} < {min_mean_tpr:.4f}", details
    if mean_tps_ratio < min_mean_tps:
        return "reject", f"DEV mean TPS ratio {mean_tps_ratio:.4f} < {min_mean_tps:.4f}", details
    if screen_improved and strict_eff_wins < min_wins:
        return "reject", f"canonical round win generalized to only {strict_eff_wins} DEV prompts; need {min_wins}", details

    return "advance", (
        f"DEV PASS: exact={exact}/{len(tpr_ratios)}, tok/round ratio={mean_tpr_ratio:.4f}, "
        f"live TPS ratio={mean_tps_ratio:.4f}, efficiency wins={strict_eff_wins}"
    ), details


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply fail-closed screen or DEV promotion gates.")
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--candidate-id", required=True)
    ap.add_argument("--db", default="results/tuning.sqlite")
    ap.add_argument("--stage", choices=("screen", "dev"), default="screen")
    ap.add_argument("--prompt-id", default="canonical-lru-python")
    ap.add_argument("--reference-candidate")
    ap.add_argument("--performance-reference-candidate")
    ap.add_argument("--latest-n", type=int)
    args = ap.parse_args()

    cfg = load_campaign(args.campaign)
    conn = connect(args.db)
    reference_candidate = args.reference_candidate or champion_reference_id(cfg)

    if args.stage == "dev":
        decision, reason, details = evaluate_dev_gate(
            cfg,
            conn,
            args.candidate_id,
            reference_candidate,
            performance_reference_candidate=args.performance_reference_candidate,
        )
        print("candidate:", args.candidate_id)
        print("decision:", decision.upper())
        print("reason:", reason)
        if details:
            print("details:", details)
        return

    stats = load_screen_stats(
        conn,
        cfg["campaign"]["name"],
        args.candidate_id,
        args.prompt_id,
        latest_n=args.latest_n,
    )
    if stats is None:
        raise SystemExit("no matching runs found")
    reference = load_reference(conn, cfg["campaign"]["name"], reference_candidate, args.prompt_id)
    if reference is None:
        raise SystemExit("no frozen champion reference found")

    decision, reason = classify(
        cfg,
        stats,
        reference_tokens=reference.tokens,
        reference_hash=reference.text_sha256,
    )
    print("candidate:", args.candidate_id)
    print("runs:", stats.count)
    print("mean tps:", f"{stats.mean_tps:.4f}")
    print("rounds:", stats.min_rounds)
    print("mean tokens/round:", f"{stats.mean_tokens_per_round:.4f}")
    print("decision:", decision.upper())
    print("reason:", reason)


if __name__ == "__main__":
    main()
