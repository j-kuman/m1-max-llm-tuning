import mlx_vlm
from mlx_vlm.speculative import load_drafter

TARGET = "/Users/skylinej17/models/Qwen3.8-27B-MLX-6bit-FP16-Q8HEAD"

DRAFTS = {
    "Q6": "/Users/skylinej17/models/Qwen3.8-27B-MTP-oQ6-FP16",
    "Q3MLP_Q6ATTN": (
        "/Users/skylinej17/models/"
        "Qwen3.8-27B-MTP-Q3MLP-Q6ATTN-FP16-27.305"
    ),
}

PROMPT = (
    "Implement an LRU cache in Python using a dictionary and doubly linked "
    "list. Include type hints and a usage example."
)

print("Loading target ONCE...")
model, processor = mlx_vlm.load(TARGET)

outputs = {}

for label, path in DRAFTS.items():
    print(f"\n========== {label} ==========")

    draft, kind = load_drafter(path, kind="mtp")

    # Bind/reset drafter against this target exactly as our persistent setup does.
    if hasattr(draft, "reset"):
        draft.reset(model)

    result = mlx_vlm.generate(
        model,
        processor,
        PROMPT,
        draft_model=draft,
        draft_kind=kind,
        draft_block_size=4,
        max_tokens=512,
        temperature=0,
        verbose=False,
    )

    outputs[label] = result.text

    print("tokens:", result.generation_tokens)
    print("tps:", result.generation_tps)
    print("rounds:", getattr(result, "speculative_rounds", "n/a"))

open("/tmp/q38-output-q6.txt", "w").write(outputs["Q6"])
open("/tmp/q38-output-q3mlp.txt", "w").write(outputs["Q3MLP_Q6ATTN"])

print("\n========== COMPARISON ==========")
print("character lengths:",
      len(outputs["Q6"]),
      len(outputs["Q3MLP_Q6ATTN"]))
print("exact text equality:",
      outputs["Q6"] == outputs["Q3MLP_Q6ATTN"])

if outputs["Q6"] != outputs["Q3MLP_Q6ATTN"]:
    import difflib
    print("\n".join(difflib.unified_diff(
        outputs["Q6"].splitlines(),
        outputs["Q3MLP_Q6ATTN"].splitlines(),
        fromfile="Q6",
        tofile="Q3MLP-Q6ATTN",
        lineterm="",
    )))
