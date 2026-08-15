from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
import mlx_vlm
from mlx_vlm.speculative import load_drafter

from .db import connect, insert_run
from .spec import expand_path, load_campaign


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a multi-prompt DEV/holdout suite with one isolated drafter loaded once.")
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--candidate-id", required=True)
    ap.add_argument("--prompt-file", default="tuner/prompts/dev.json")
    ap.add_argument("--stage", choices=("dev", "holdout"), default="dev")
    ap.add_argument("--runs-per-prompt", type=int, default=1)
    ap.add_argument("--pause", type=float, default=2.0)
    ap.add_argument("--db", default="results/tuning.sqlite")
    args = ap.parse_args()

    campaign = load_campaign(args.campaign)
    c = campaign["campaign"]
    target_path = expand_path(c["target"])
    draft_path = expand_path(args.draft)
    prompts = json.loads(expand_path(args.prompt_file).read_text())

    print("Loading target ONCE...")
    model, processor = mlx_vlm.load(str(target_path))
    print("Loading drafter ONCE...")
    draft, kind = load_drafter(str(draft_path), kind=c.get("draft_kind", "mtp"))
    mx.eval(model.parameters())
    mx.eval(draft.parameters())
    mx.synchronize()

    conn = connect(args.db)
    all_tps: list[float] = []
    total_tokens = 0
    total_wall = 0.0
    total_rounds = 0

    for prompt in prompts:
        prompt_tps: list[float] = []
        print(f"\n========== {prompt['id']} ==========")
        for i in range(1, args.runs_per_prompt + 1):
            draft.reset(model)
            mx.synchronize()
            t0 = time.perf_counter()
            result = mlx_vlm.generate(
                model,
                processor,
                prompt["text"],
                max_tokens=int(prompt.get("max_tokens", c.get("max_tokens", 512))),
                temperature=float(prompt.get("temperature", c.get("temperature", 0.0))),
                draft_model=draft,
                draft_kind=kind,
                draft_block_size=int(c.get("block_size", 4)),
                verbose=False,
            )
            mx.synchronize()
            wall = time.perf_counter() - t0
            rounds = len(list(getattr(draft, "accept_lens", [])))
            text_sha = hashlib.sha256(result.text.encode()).hexdigest()
            tps = float(result.generation_tps)
            tokens = int(result.generation_tokens)

            insert_run(conn, {
                "campaign": c["name"],
                "candidate": args.candidate_id,
                "stage": args.stage,
                "prompt_id": prompt["id"],
                "run_index": i,
                "target_path": str(target_path),
                "draft_path": str(draft_path),
                "tokens": tokens,
                "tps": tps,
                "wall_seconds": wall,
                "rounds": rounds,
                "peak_gb": float(result.peak_memory),
                "text_sha256": text_sha,
                "modules_json": None,
                "notes": prompt.get("category"),
            })

            print(f"run {i}: {tokens} tok | {tps:.3f} tok/s | rounds={rounds} | wall={wall:.3f}s")
            prompt_tps.append(tps)
            all_tps.append(tps)
            total_tokens += tokens
            total_wall += wall
            total_rounds += rounds
            if args.pause:
                time.sleep(args.pause)

        print(f"prompt mean: {statistics.mean(prompt_tps):.3f} tok/s")

    print("\n========== SUITE SUMMARY ==========")
    print("prompts:", len(prompts))
    print("runs:", len(all_tps))
    print("mean per-run TPS:", f"{statistics.mean(all_tps):.3f}")
    print("median per-run TPS:", f"{statistics.median(all_tps):.3f}")
    print("aggregate tokens/wall:", f"{total_tokens / total_wall:.3f} tok/s")
    print("aggregate tokens/round:", f"{total_tokens / total_rounds:.3f}" if total_rounds else "n/a")
    print("stored in:", args.db)


if __name__ == "__main__":
    main()
