from pathlib import Path
import mlx.core as mx
from mlx_vlm.speculative import load_drafter

SRC = Path(
    "/Users/skylinej17/.cache/huggingface/hub/"
    "models--Qwen--Qwen3.8-27B/"
    "snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/"
    "model-00018-of-00018.safetensors"
)

Q6 = "/Users/skylinej17/models/Qwen3.8-27B-MTP-oQ6-FP16"
Q8 = "/Users/skylinej17/models/Qwen3.8-27B-MTP-oQ8-FP16"

src_all = mx.load(str(SRC))
src = {
    k.removeprefix("mtp."): v
    for k, v in src_all.items()
    if k.startswith("mtp.")
}

d6, _ = load_drafter(Q6, kind="mtp")
d8, _ = load_drafter(Q8, kind="mtp")

mods6 = dict(d6.named_modules())
mods8 = dict(d8.named_modules())

names = [
    "layers.0.mlp.gate_proj",
    "layers.0.mlp.up_proj",
    "layers.0.mlp.down_proj",
    "layers.0.self_attn.q_proj",
    "layers.0.self_attn.k_proj",
    "layers.0.self_attn.v_proj",
    "layers.0.self_attn.o_proj",
]

def deq(m):
    x = mx.dequantize(
        m.weight,
        m.scales,
        m.biases,
        group_size=m.group_size,
        bits=m.bits,
        mode="affine",
    )
    mx.eval(x)
    return x

print("\n========== SOURCE vs KNOWN-GOOD Q6 vs NEW Q8 ==========\n")

for name in names:
    sw = src[name + ".weight"].astype(mx.float32)

    w6 = deq(mods6[name]).astype(mx.float32)
    w8 = deq(mods8[name]).astype(mx.float32)

    mx.eval(sw, w6, w8)

    print(name)
    print("  source:", sw.shape)
    print("  Q6:    ", w6.shape)
    print("  Q8:    ", w8.shape)

    if sw.shape == w6.shape:
        d6err = mx.abs(sw - w6)
        d8err = mx.abs(sw - w8)

        q6_q8 = mx.abs(w6 - w8)

        vals = [
            mx.mean(d6err),
            mx.max(d6err),
            mx.mean(d8err),
            mx.max(d8err),
            mx.mean(q6_q8),
            mx.max(q6_q8),
        ]
        mx.eval(*vals)

        print(
            "  source↔Q6: "
            f"mean={float(vals[0]):.8f} "
            f"max={float(vals[1]):.8f}"
        )
        print(
            "  source↔Q8: "
            f"mean={float(vals[2]):.8f} "
            f"max={float(vals[3]):.8f}"
        )
        print(
            "  Q6↔Q8:     "
            f"mean={float(vals[4]):.8f} "
            f"max={float(vals[5]):.8f}"
        )
    else:
        print("  *** SHAPE MISMATCH ***")

    print()

# Also check every unquantized tensor. These should be close to source
# after BF16 -> FP16 conversion.
print("\n========== NON-QUANTIZED TENSORS ==========\n")

params6 = dict(d6.parameters())
params8 = dict(d8.parameters())

for name in sorted(src):
    if name.endswith(".weight") and name[:-7] in names:
        continue

    if name not in params6 or name not in params8:
        print(name, "not directly present in flattened parameters")
        continue

    a = src[name].astype(mx.float32)
    b = params6[name].astype(mx.float32)
    c = params8[name].astype(mx.float32)

    mx.eval(a, b, c)

    e6 = mx.max(mx.abs(a-b))
    e8 = mx.max(mx.abs(a-c))
    mx.eval(e6, e8)

    print(
        f"{name:60s} "
        f"src-Q6 max={float(e6):.8f} "
        f"src-Q8 max={float(e8):.8f}"
    )
