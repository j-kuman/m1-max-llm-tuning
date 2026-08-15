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


def load_prompt(path: Path, prompt_id: str) -> dict:
    items = json.loads(path.read_text())
    for item in items:
        if item["id"] == prompt_id:
            return item
    raise KeyError(f"Prompt {prompt_id!r} not found in {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Isolated persistent-process benchmark for one target/drafter pair.")
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--candidate-id", required=True)
    ap.add_argument("--prompt-file", default="tuner/prompts/canonical.json")
    ap.add_argument("--prompt-id", default="canonical-lru-python")
    ap.add_argument("--stage", choices=("screen", "dev", "production", "holdout"), default="screen")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--pause", type=float, default=5.0)
    ap.add_argument("--db", default="results/tuning.sqlite")
    args = ap.parse_args()

    campaign = load_campaign(args.campaign)
    c = campaign["campaign"]
    target_path = expand_path(c["target"])
    draft_path = expand_path(args.draft)
    prompt = load_prompt(expand_path(args.prompt_file), args.prompt_id)
    max_tokens = int(prompt.get("max_tokens", c.get("max_tokens", 512)))
    temperature = float(prompt.get("temperature", c.get("temperature", 0.0)))
    block_size = int(c.get("block_size", 4))
    draft_kind_requested = c.get("draft_kind", "mtp")

    print("Loading target ONCE...")
    model, processor = mlx_vlm.load(str(target_path))
    print("Loading drafter ONCE...")
    draft, kind = load_drafter(str(draft_path), kind=draft_kind_requested)
    print("draft kind:", kind)
    print("draft class:", type(draft))

    mx.eval(model.parameters())
    mx.eval(draft.parameters())
    mx.synchronize()

    conn = connect(args.db)
    rows: list[dict] = []

    def one_run(i: int, measured: bool) -> dict:
        draft.reset(model)
        mx.synchronize()
        t0 = time.perf_counter()
        result = mlx_vlm.generate(
            model,
            processor,
            prompt["text"],
            max_tokens=max_tokens,
            temperature=temperature,
            draft_model=draft,
            draft_kind=kind,
            draft_block_size=block_size,
            verbose=False,
        )
        mx.synchronize()
        wall = time.perf_counter() - t0
        accepts = list(getattr(draft, "accept_lens", []))
        rounds = len(accepts)
        text_sha = hashlib.sha256(result.text.encode()).hexdigest()

        row = {
            "campaign": c["name"],
            "candidate": args.candidate_id,
            "stage": args.stage if measured else "warmup",
            "prompt_id": args.prompt_id,
            "run_index": i,
            "target_path": str(target_path),
            "draft_path": str(draft_path),
            "tokens": int(result.generation_tokens),
            "tps": float(result.generation_tps),
            "wall_seconds": wall,
            "rounds": rounds,
            "peak_gb": float(result.peak_memory),
            "text_sha256": text_sha,
            "modules_json": None,
            "notes": None,
        }
        print(
            f"{'MEASURED' if measured else 'WARMUP':8s} {i:2d}: "
            f"{row['tokens']:3d} tok | {row['tps']:7.3f} tok/s | "
            f"wall={wall:7.3f}s | rounds={rounds:3d} | peak={row['peak_gb']:.3f} GB"
        )
        if measured:
            insert_run(conn, row)
            rows.append(row)
        return row

    for i in range(args.warmups):
        one_run(i, measured=False)
        if args.pause:
            time.sleep(args.pause)

    for i in range(1, args.runs + 1):
        one_run(i, measured=True)
        if i != args.runs and args.pause:
            time.sleep(args.pause)

    if rows:
        speeds = [r["tps"] for r in rows]
        rounds = [r["rounds"] for r in rows]
        hashes = [r["text_sha256"] for r in rows]
        print("\n========== SUMMARY ==========")
        print("runs:  ", " ".join(f"{x:.3f}" for x in speeds))
        print(f"median: {statistics.median(speeds):.3f} tok/s")
        print(f"mean:   {statistics.mean(speeds):.3f} tok/s")
        print(f"min:    {min(speeds):.3f} tok/s")
        print(f"max:    {max(speeds):.3f} tok/s")
        print(f"spread: {max(speeds)-min(speeds):.3f} tok/s")
        print("round counts:", rounds)
        print("all generated text identical:", len(set(hashes)) == 1)
        print("stored in:", args.db)


if __name__ == "__main__":
    main()
