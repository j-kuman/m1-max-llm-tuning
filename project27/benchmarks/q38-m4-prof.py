import time
import statistics
from pathlib import Path

import mlx.core as mx

from mlx_vlm.utils import load_model

MODEL = Path.home() / "models/Qwen3.8-27B-MLX-6bit-FP16-Q8HEAD"

print("Loading M4 production target...")
model = load_model(MODEL)

lm = getattr(model, "language_model", model)
layers = lm.model.layers

print("layers:", len(layers))

mx.random.seed(240824)
x = mx.random.normal((1, 4, 5120)).astype(mx.float16)
mx.eval(x)


def flatten_arrays(obj):
    out = []

    if isinstance(obj, mx.array):
        return [obj]

    if isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(flatten_arrays(v))

    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(flatten_arrays(v))

    return out


def timed(fn, warm=2, reps=5):
    for _ in range(warm):
        vals = fn()
        vals = flatten_arrays(vals)
        if vals:
            mx.eval(*vals)

    times = []

    for _ in range(reps):
        t0 = time.perf_counter()

        vals = fn()
        vals = flatten_arrays(vals)

        if vals:
            mx.eval(*vals)

        times.append((time.perf_counter() - t0) * 1000)

    return statistics.median(times)


# ------------------------------------------------------------
# MLP: all 64
# ------------------------------------------------------------
mlp_times = []

for i, layer in enumerate(layers):
    def run_mlp(layer=layer):
        return layer.mlp(x, target_verify=True)

    ms = timed(run_mlp)
    mlp_times.append(ms)

print()
print("========== MLP ==========")
print(
    f"n={len(mlp_times)}  "
    f"median={statistics.median(mlp_times):.3f} ms  "
    f"mean={statistics.mean(mlp_times):.3f} ms  "
    f"sum-medians={sum(mlp_times):.3f} ms"
)


# ------------------------------------------------------------
# GDN: 48 linear-attention layers
#
# IMPORTANT:
# gdn_sink=[] forces the actual speculative-verification path,
# including verifier state/intermediate-state generation.
# ------------------------------------------------------------
gdn_times = []

for i, layer in enumerate(layers):
    if not layer.is_linear:
        continue

    def run_gdn(i=i, layer=layer):
        cache = lm.make_cache()[i]
        sink = []

        y = layer.linear_attn(
            x,
            cache=cache,
            gdn_sink=sink,
            target_verify=True,
        )

        vals = [y]
        vals.extend(flatten_arrays(sink))

        # ArraysCache holds conv + delta state.
        try:
            vals.extend(flatten_arrays(cache[0]))
            vals.extend(flatten_arrays(cache[1]))
        except Exception:
            pass

        return vals

    ms = timed(run_gdn)
    gdn_times.append(ms)

print()
print("========== GDN ==========")
print(
    f"n={len(gdn_times)}  "
    f"median={statistics.median(gdn_times):.3f} ms  "
    f"mean={statistics.mean(gdn_times):.3f} ms  "
    f"sum-medians={sum(gdn_times):.3f} ms"
)


# ------------------------------------------------------------
# Full attention: 16 layers
# ------------------------------------------------------------
attn_times = []

for i, layer in enumerate(layers):
    if layer.is_linear:
        continue

    def run_attn(i=i, layer=layer):
        cache = lm.make_cache()[i]

        y = layer.self_attn(
            x,
            cache=cache,
            target_verify=True,
        )

        return y

    ms = timed(run_attn)
    attn_times.append(ms)

print()
print("========== FULL ATTENTION ==========")
print(
    f"n={len(attn_times)}  "
    f"median={statistics.median(attn_times):.3f} ms  "
    f"mean={statistics.mean(attn_times):.3f} ms  "
    f"sum-medians={sum(attn_times):.3f} ms"
)


print()
print("========== ROUGH COMPONENT SUM ==========")

mlp_sum = sum(mlp_times)
gdn_sum = sum(gdn_times)
attn_sum = sum(attn_times)

print(f"64 MLPs:       {mlp_sum:8.3f} ms")
print(f"48 GDNs:       {gdn_sum:8.3f} ms")
print(f"16 attentions: {attn_sum:8.3f} ms")
print(f"sum:           {mlp_sum + gdn_sum + attn_sum:8.3f} ms")
