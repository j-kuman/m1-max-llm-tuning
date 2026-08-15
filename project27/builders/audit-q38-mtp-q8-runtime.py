from pathlib import Path
import mlx.core as mx
from mlx_vlm.speculative import load_drafter

SRC = Path(
    "/Users/skylinej17/.cache/huggingface/hub/"
    "models--Qwen--Qwen3.8-27B/"
    "snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/"
    "model-00018-of-00018.safetensors"
)

Q6 = Path.home() / "models/Qwen3.8-27B-MTP-oQ6-FP16"
Q8 = Path.home() / "models/Qwen3.8-27B-MTP-oQ8-FP16"

src_all = mx.load(str(SRC))
src = {
    k.removeprefix("mtp."): v
    for k, v in src_all.items()
    if k.startswith("mtp.")
}

f6 = mx.load(str(Q6 / "model.safetensors"))
f8 = mx.load(str(Q8 / "model.safetensors"))

d6, _ = load_drafter(str(Q6), kind="mtp")
d8, _ = load_drafter(str(Q8), kind="mtp")

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

quant_weight_names = {n + ".weight" for n in names}


def deq(m):
    w = mx.dequantize(
        m.weight,
        m.scales,
        m.biases,
        group_size=m.group_size,
        bits=m.bits,
        mode="affine",
    )
    mx.eval(w)
    return w


def stats(a, b):
    a = a.astype(mx.float32)
    b = b.astype(mx.float32)

    err = mx.abs(a - b)
    mean_err = mx.mean(err)
    max_err = mx.max(err)
    denom = mx.mean(mx.abs(b))
    rel = mean_err / (denom + 1e-12)

    mx.eval(mean_err, max_err, denom, rel)

    return (
        float(mean_err),
        float(max_err),
        float(rel),
    )


print("\n========== NON-QUANTIZED FILE TENSORS ==========\n")

for name in sorted(src):
    if name in quant_weight_names:
        continue

    a = src[name].astype(mx.float32)
    b = f6[name].astype(mx.float32)
    c = f8[name].astype(mx.float32)

    mx.eval(a, b, c)

    e6 = stats(b, a)
    e8 = stats(c, a)
    e68 = stats(c, b)

    print(name)
    print(
        f"  source-Q6 mean={e6[0]:.8f} "
        f"max={e6[1]:.8f}"
    )
    print(
        f"  source-Q8 mean={e8[0]:.8f} "
        f"max={e8[1]:.8f}"
    )
    print(
        f"  Q6-Q8     mean={e68[0]:.8f} "
        f"max={e68[1]:.8f}"
    )


print("\n========== QUANTIZED LINEAR RUNTIME AUDIT ==========\n")

for name in names:
    m6 = mods6[name]
    m8 = mods8[name]

    wsrc = src[name + ".weight"].astype(mx.float16)

    w6 = deq(m6).astype(mx.float16)
    w8 = deq(m8).astype(mx.float16)

    in_dim = wsrc.shape[1]

    print("\n" + name)
    print(
        f"  source={wsrc.shape} "
        f"Q6 bits={m6.bits} "
        f"Q8 bits={m8.bits}"
    )

    for M in (1, 4):
        # Fixed deterministic input.
        mx.random.seed(12345 + M)
        x = (
            mx.random.normal((M, in_dim))
            * 0.1
        ).astype(mx.float16)

        # Actual QuantizedLinear runtime.
        y6 = m6(x)
        y8 = m8(x)

        # Explicit matmul against dequantized weights.
        r6 = mx.matmul(x, w6.T)
        r8 = mx.matmul(x, w8.T)

        # Pristine source reference.
        rs = mx.matmul(x, wsrc.T)

        mx.eval(y6, y8, r6, r8, rs)

        q6_runtime = stats(y6, r6)
        q8_runtime = stats(y8, r8)

        q6_source = stats(y6, rs)
        q8_source = stats(y8, rs)

        print(f"\n  M={M}")
        print(
            "    Q6 runtime vs dequant: "
            f"mean={q6_runtime[0]:.8f} "
            f"max={q6_runtime[1]:.8f} "
            f"rel={q6_runtime[2]:.6e}"
        )
        print(
            "    Q8 runtime vs dequant: "
            f"mean={q8_runtime[0]:.8f} "
            f"max={q8_runtime[1]:.8f} "
            f"rel={q8_runtime[2]:.6e}"
        )
        print(
            "    Q6 runtime vs source:  "
            f"mean={q6_source[0]:.8f} "
            f"max={q6_source[1]:.8f} "
            f"rel={q6_source[2]:.6e}"
        )
        print(
            "    Q8 runtime vs source:  "
            f"mean={q8_source[0]:.8f} "
            f"max={q8_source[1]:.8f} "
            f"rel={q8_source[2]:.6e}"
        )

print("\nDONE")
