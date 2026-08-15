import statistics
import time

import mlx.core as mx
import mlx_vlm
from mlx_vlm.speculative import load_drafter

TARGET = "/Users/skylinej17/models/Qwen3.8-27B-MLX-6bit-FP16-Q8HEAD"
DRAFT  = "/Users/skylinej17/models/Qwen3.8-27B-MTP-oQ6-FP16"

# Match the effective chat-formatted prompt we've been benchmarking.
PROMPT = """<|im_start|>user
Implement an LRU cache in Python using a dictionary and doubly linked list. Include type hints and a usage example.<|im_end|>
<|im_start|>assistant
<think>

</think>

"""

print("Loading target ONCE...")
model, processor = mlx_vlm.load(TARGET)

print("Loading MTP drafter ONCE...")
draft, kind = load_drafter(DRAFT, kind="mtp")

print("draft kind:", kind)
print("draft class:", type(draft))

# Force all lazy weights resident before benchmarking.
mx.eval(model.parameters())
mx.eval(draft.parameters())
mx.synchronize()

def one_run(i, measured=True):
    # Explicitly clear all request-specific MTP state while preserving weights.
    draft.reset(model)

    # Ensure previous GPU work is complete before the timer inside generation begins.
    mx.synchronize()

    t0 = time.perf_counter()

    result = mlx_vlm.generate(
        model,
        processor,
        PROMPT,
        max_tokens=512,
        temperature=0,
        draft_model=draft,
        draft_kind=kind,
        draft_block_size=4,
        verbose=False,
    )

    mx.synchronize()
    wall = time.perf_counter() - t0

    accepts = list(getattr(draft, "accept_lens", []))
    drafts  = list(getattr(draft, "draft_lens", []))

    rounds = len(accepts)

    print(
        f"{'MEASURED' if measured else 'WARMUP':8s} {i:2d}: "
        f"{result.generation_tokens:3d} tok | "
        f"{result.generation_tps:7.3f} tok/s | "
        f"wall={wall:7.3f}s | "
        f"rounds={rounds:3d} | "
        f"peak={result.peak_memory:.3f} GB"
    )

    return result.generation_tps, result.text, rounds


print("\n========== WARMUP ==========")
one_run(0, measured=False)

# Short pause, but we remain in exactly the same Python/Metal process.
time.sleep(10)

print("\n========== PERSISTENT-PROCESS RUNS ==========")

speeds = []
texts = []
rounds_all = []

for i in range(1, 11):
    tps, text, rounds = one_run(i)
    speeds.append(tps)
    texts.append(text)
    rounds_all.append(rounds)

    # Enough separation to avoid one generation literally running into the next,
    # but not enough to restart Python / reload Metal / reload weights.
    time.sleep(5)

print("\n========== SUMMARY ==========")
print("runs:  ", " ".join(f"{x:.3f}" for x in speeds))
print(f"median: {statistics.median(speeds):.3f} tok/s")
print(f"mean:   {statistics.mean(speeds):.3f} tok/s")
print(f"min:    {min(speeds):.3f} tok/s")
print(f"max:    {max(speeds):.3f} tok/s")
print(f"spread: {max(speeds)-min(speeds):.3f} tok/s")

same_text = all(x == texts[0] for x in texts)
same_rounds = len(set(rounds_all)) == 1

print("all generated text identical:", same_text)
print("round counts:", rounds_all)
print("all round counts identical:", same_rounds)
