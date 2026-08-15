from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .db import connect
from .gates import champion_reference_id, load_reference
from .promote import classify, evaluate_dev_gate, load_screen_stats
from .search import generate_neighbors
from .spec import expand_path, load_campaign


def run(cmd: list[str], env: dict[str, str]) -> None:
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def campaign_env(cfg: dict) -> dict[str, str]:
    env = os.environ.copy()
    env_cfg = cfg.get("environment", {})
    for key, value in env_cfg.get("set", {}).items():
        env[str(key)] = str(value)
    for key in env_cfg.get("unset", []):
        env.pop(str(key), None)
    return env


def ensure_canonical_reference(
    cfg: dict,
    campaign_path: str,
    db_path: Path,
    reference_candidate: str,
    prompt_id: str,
    env: dict[str, str],
) -> None:
    conn = connect(db_path)
    ref = load_reference(conn, cfg["campaign"]["name"], reference_candidate, prompt_id)
    if ref is not None:
        return

    print("\nEstablishing frozen canonical champion reference...")
    run([
        sys.executable, "-m", "tuner.benchmark",
        "--campaign", campaign_path,
        "--draft", str(expand_path(cfg["champion"]["draft"])),
        "--candidate-id", reference_candidate,
        "--stage", "reference",
        "--runs", "1",
        "--warmups", "1",
        "--db", str(db_path),
        "--prompt-id", prompt_id,
    ], env)


def ensure_dev_reference(
    cfg: dict,
    campaign_path: str,
    db_path: Path,
    reference_candidate: str,
    prompt_file: str,
    env: dict[str, str],
) -> None:
    prompts = json.loads(expand_path(prompt_file).read_text())
    conn = connect(db_path)
    missing = [
        p["id"] for p in prompts
        if load_reference(conn, cfg["campaign"]["name"], reference_candidate, p["id"]) is None
    ]
    if not missing:
        return

    print(f"\nEstablishing frozen champion DEV reference ({len(missing)} prompts missing)...")
    run([
        sys.executable, "-m", "tuner.suite",
        "--campaign", campaign_path,
        "--draft", str(expand_path(cfg["champion"]["draft"])),
        "--candidate-id", reference_candidate,
        "--stage", "reference",
        "--prompt-file", prompt_file,
        "--runs-per-prompt", "1",
        "--db", str(db_path),
    ], env)


def main() -> None:
    ap = argparse.ArgumentParser(description="Budgeted campaign runner: neighbors -> build -> validate -> exact screen -> DEV gate -> optional production.")
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--screen-runs", type=int, default=1)
    ap.add_argument("--db", default="results/tuning.sqlite")
    ap.add_argument("--candidate-root", default="candidates/generated")
    ap.add_argument("--run-dev", action="store_true")
    ap.add_argument("--run-production", action="store_true")
    ap.add_argument("--production-runs", type=int, default=10)
    ap.add_argument("--prompt-id", default="canonical-lru-python")
    ap.add_argument("--dev-prompt-file", default="tuner/prompts/dev.json")
    args = ap.parse_args()

    cfg = load_campaign(args.campaign)
    name = cfg["campaign"]["name"]
    output_root = expand_path(cfg["campaign"]["output_dir"])
    db_path = expand_path(args.db)
    candidate_root = Path(args.candidate_root) / name
    candidate_root.mkdir(parents=True, exist_ok=True)

    env = campaign_env(cfg)

    reference_candidate = champion_reference_id(cfg)
    ensure_canonical_reference(
        cfg, args.campaign, db_path, reference_candidate, args.prompt_id, env
    )
    canonical_reference = load_reference(
        connect(db_path), name, reference_candidate, args.prompt_id
    )
    assert canonical_reference is not None

    need_dev = args.run_dev or args.run_production
    if need_dev:
        ensure_dev_reference(
            cfg,
            args.campaign,
            db_path,
            reference_candidate,
            args.dev_prompt_file,
            env,
        )

    neighbors = generate_neighbors(cfg)[: args.budget]
    print(f"Campaign {name}: evaluating {len(neighbors)} of {len(generate_neighbors(cfg))} local neighbors")
    print("reference candidate:", reference_candidate)

    advanced_screen: list[str] = []
    advanced_dev: list[str] = []

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

        conn = connect(db_path)
        existing = load_screen_stats(conn, name, cid, args.prompt_id, "screen")
        if not existing or existing.count < args.screen_runs:
            needed = args.screen_runs - (existing.count if existing else 0)
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
                "--reference-candidate", reference_candidate,
            ], env)

        stats = load_screen_stats(connect(db_path), name, cid, args.prompt_id, "screen")
        assert stats is not None
        decision, reason = classify(
            cfg,
            stats,
            reference_tokens=canonical_reference.tokens,
            reference_hash=canonical_reference.text_sha256,
        )
        print(
            f"SCREEN {cid}: n={stats.count} mean={stats.mean_tps:.3f} "
            f"rounds={stats.min_rounds} tok/round={stats.mean_tokens_per_round:.4f} "
            f"=> {decision.upper()} ({reason})"
        )

        if decision != "advance":
            continue
        advanced_screen.append(cid)

        if need_dev:
            run([
                sys.executable, "-m", "tuner.suite",
                "--campaign", args.campaign,
                "--draft", str(draft_dir),
                "--candidate-id", cid,
                "--stage", "dev",
                "--prompt-file", args.dev_prompt_file,
                "--db", str(db_path),
                "--reference-candidate", reference_candidate,
            ], env)

            dev_decision, dev_reason, details = evaluate_dev_gate(
                cfg, connect(db_path), cid, reference_candidate
            )
            print(f"DEV {cid}: {dev_decision.upper()} ({dev_reason})")
            if details:
                print("DEV details:", details)
            if dev_decision != "advance":
                continue
            advanced_dev.append(cid)

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
                "--reference-candidate", reference_candidate,
            ], env)

    print("\n========== CAMPAIGN RUN COMPLETE ==========")
    print("screen advanced:", advanced_screen if advanced_screen else "none")
    if need_dev:
        print("DEV advanced:", advanced_dev if advanced_dev else "none")
    print("database:", db_path)
    print("Champion is NOT updated automatically; final holdout/promotion remains explicit.")


if __name__ == "__main__":
    main()
