from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time

import mlx.core as mx
import mlx_vlm
from mlx_vlm.speculative import load_drafter

from .db import connect, insert_run
from .gates import (
    champion_reference_id,
    extract_round_stats,
    load_reference,
    require_exact_match,
)
from .spec import expand_path, load_campaign


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a multi-prompt DEV/holdout suite with one isolated drafter loaded once.")
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--candidate-id", required=True)
    ap.add_argument("--prompt-file", default="tuner/prompts/dev.json")
    ap.add_argument("--stage", choices=("reference", "dev", "holdout"), default="dev")
    ap.add_argument("--runs-per-prompt", type=int, default=1)
    ap.add_argument("--pause", type=float, default=2.0)
    ap.add_argument("--db", default="results/tuning.sqlite")
    ap.add_argument("--reference-candidate")
    args = ap.parse_args()

    campaign = load_campaign(args.campaign)
    c = campaign["campaign"]
    target_path = expand_path(c["target"])
    draft_path = expand_path(args.draft)
    prompts = json.loads(expand_path(args.prompt_file).read_text())
    reference_candidate = args.reference_candidate or champion_reference_id(campaign)

    conn = connect(args.db)
    references = {}
    if args.stage != "reference":
        for prompt in prompts:
            if float(prompt.get("temperature", c.get("temperature", 0.0))) != 0.0:
                raise RuntimeError("autotuner exactness gates currently require temperature=0")
            ref = load_reference(conn, c["name"], reference_candidate, prompt["id"])
            if ref is None:
                raise RuntimeError(
                    f"missing exact reference for DEV prompt {prompt['id']!r}; "
                    "run the champion suite with --stage reference first"
                )
            references[prompt["id"]] = ref

    print("Loading target ONCE...")
    model, processor = mlx_vlm.load(str(target_path))
    print("Loading drafter ONCE...")
    draft, kind = load_drafter(str(draft_path), kind=c.get("draft_kind", "mtp"))
    mx.eval(model.parameters())
    mx.eval(draft.parameters())
    mx.synchronize()

    all_tps: list[float] = []
    total_tokens = 0
    total_wall = 0.0
    total_rounds = 0

    for prompt in prompts:
        prompt_tps: list[float] = []
        prompt_eff: list[float] = []
        print(f"\n========== {prompt['id']} ==========")
        for i in range(1, args.runs_per_prompt + 1):
            temperature = float(prompt.get("temperature", c.get("temperature", 0.0)))
            if temperature != 0.0:
                raise RuntimeError("autotuner exactness gates currently require temperature=0")

            draft.reset(model)
            mx.synchronize()
            t0 = time.perf_counter()
            result = mlx_vlm.generate(
                model,
                processor,
                prompt["text"],
                max_tokens=int(prompt.get("max_tokens", c.get("max_tokens", 512))),
                temperature=temperature,
                draft_model=draft,
                draft_kind=kind,
                draft_block_size=int(c.get("block_size", 4)),
                verbose=False,
            )
            mx.synchronize()
            wall = time.perf_counter() - t0
            tokens = int(result.generation_tokens)
            rounds, tokens_per_round = extract_round_stats(draft, tokens)
            text_sha = hashlib.sha256(result.text.encode("utf-8")).hexdigest()

            if args.stage != "reference":
                require_exact_match(
                    references[prompt["id"]],
                    tokens=tokens,
                    text_sha256=text_sha,
                    candidate=args.candidate_id,
                )

            tps = float(result.generation_tps)

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
                "tokens_per_round": tokens_per_round,
                "peak_gb": float(result.peak_memory),
                "text_sha256": text_sha,
                "modules_json": None,
                "notes": prompt.get("category"),
            })

            print(
                f"run {i}: {tokens} tok | {tps:.3f} tok/s | rounds={rounds} | "
                f"tok/round={tokens_per_round:.4f} | wall={wall:.3f}s"
            )
            prompt_tps.append(tps)
            prompt_eff.append(tokens_per_round)
            all_tps.append(tps)
            total_tokens += tokens
            total_wall += wall
            total_rounds += rounds
            if args.pause:
                time.sleep(args.pause)

        print(f"prompt mean TPS: {statistics.mean(prompt_tps):.3f}")
        print(f"prompt mean tok/round: {statistics.mean(prompt_eff):.4f}")

    print("\n========== SUITE SUMMARY ==========")
    print("prompts:", len(prompts))
    print("runs:", len(all_tps))
    print("mean per-run TPS:", f"{statistics.mean(all_tps):.3f}")
    print("median per-run TPS:", f"{statistics.median(all_tps):.3f}")
    print("aggregate tokens/wall:", f"{total_tokens / total_wall:.3f} tok/s")
    print("aggregate tokens/round:", f"{total_tokens / total_rounds:.3f}")
    print("exact reference match:", "established" if args.stage == "reference" else "PASS")
    print("stored in:", args.db)


if __name__ == "__main__":
    main()
