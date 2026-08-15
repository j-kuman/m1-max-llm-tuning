import statistics
import time

import mlx.core as mx
import mlx_vlm
from mlx_vlm.speculative import load_drafter

TARGET = "/Users/skylinej17/models/Qwen3.8-27B-MLX-6bit-FP16-Q8HEAD"
DRAFT  = "/Users/skylinej17/models/Qwen3.8-27B-MTP-oQ6-FP16"

PROMPT = """<|im_start|>user
Implement an LRU cache in Python using a dictionary and doubly linked list. Include type hints and a usage example.<|im_end|>
<|im_start|>assistant
<think>

</think>

"""

BLOCKS = [2, 3, 4, 5, 6, 7, 8]

print("Loading target ONCE...")
model, processor = mlx_vlm.load(TARGET)

print("Loading drafter ONCE...")
draft, kind = load_drafter(DRAFT, kind="mtp")

mx.eval(model.parameters())
mx.eval(draft.parameters())
mx.synchronize()

reference_text = None
results = []

def run(bs, max_tokens):
    draft.reset(model)
    mx.synchronize()

    r = mlx_vlm.generate(
        model,
        processor,
        PROMPT,
        max_tokens=max_tokens,
        temperature=0,
        draft_model=draft,
        draft_kind=kind,
        draft_block_size=bs,
        verbose=False,
    )

    mx.synchronize()

    accepts = list(getattr(draft, "accept_lens", []))
    drafts = list(getattr(draft, "draft_lens", []))

    accepted = sum(accepts)
    proposed = sum(drafts)
    ratio = accepted / proposed if proposed else 0.0

    return r, len(accepts), accepted, proposed, ratio


print("\n========== BLOCK-SIZE SWEEP ==========")

for bs in BLOCKS:
    print(f"\n----- BLOCK {bs} -----")

    # Warm the block-size-specific path/JIT.
    try:
        rw, _, _, _, _ = run(bs, 128)
        print(f"warmup: {rw.generation_tps:.3f} tok/s")
    except Exception as e:
        print(f"WARMUP ERROR: {e!r}")
        continue

    time.sleep(2)

    speeds = []

    for rep in range(2):
        try:
            r, rounds, accepted, proposed, ratio = run(bs, 512)
        except Exception as e:
            print(f"MEASURE ERROR: {e!r}")
            speeds = []
            break

        if reference_text is None:
            reference_text = r.text

        same = (r.text == reference_text)

        speeds.append(r.generation_tps)

        print(
            f"run {rep+1}: "
            f"{r.generation_tps:7.3f} tok/s | "
            f"rounds={rounds:3d} | "
            f"accepted={accepted:4d} | "
            f"proposed={proposed:4d} | "
            f"ratio={ratio:.4f} | "
            f"same_text={same}"
        )

        time.sleep(3)

    if speeds:
        results.append((
            bs,
            statistics.mean(speeds),
            statistics.median(speeds),
            min(speeds),
            max(speeds),
        ))


print("\n========== RANKING ==========")

for bs, mean, med, lo, hi in sorted(
    results,
    key=lambda x: x[1],
    reverse=True
):
    print(
        f"block={bs}: "
        f"mean={mean:.3f} "
        f"median={med:.3f} "
        f"range={lo:.3f}-{hi:.3f}"
    )
