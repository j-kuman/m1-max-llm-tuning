import statistics
import time

import mlx.core as mx
import mlx_vlm
from mlx_vlm.models.qwen3_5 import language as q

MODEL = "/Users/skylinej17/models/Qwen3.8-27B-MLX-6bit-FP16-Q8HEAD"

print("Loading...")
model, processor = mlx_vlm.load(MODEL)

lm = model.language_model
head = lm.lm_head

print("head bits:", head.bits)
print("group:", head.group_size)
print("weight shape:", head.weight.shape)

K = head.weight.shape[1] * 32 // head.bits
N = head.weight.shape[0]

# Production speculative verify shape.
T = 4
B = 1

# Deterministic-ish test input.
mx.random.seed(1234)
x = mx.random.normal((B, T, K)).astype(head.scales.dtype)

mx.eval(x, head.weight, head.scales, head.biases)
mx.synchronize()

num_tiles = N // 8

kernel = q._target_verify_qargmax_kernel(
    head.bits,
    head.group_size,
    x.dtype,
    T,
    K,
    N,
)

def stage1():
    vals, idx = kernel(
        inputs=[
            mx.contiguous(x),
            head.weight,
            head.scales,
            head.biases,
        ],
        template=[
            ("T", x.dtype),
            ("VERIFY_T", T),
            ("K_SIZE", K),
            ("N_SIZE", N),
            ("NUM_TILES", num_tiles),
        ],
        grid=(32, 2 * num_tiles, B),
        threadgroup=(32, 2, 1),
        output_shapes=[
            (B, T, num_tiles),
            (B, T, num_tiles),
        ],
        output_dtypes=[x.dtype, mx.int32],
    )
    return vals, idx

def reduce(vals, idx):
    best = mx.argmax(vals, axis=-1)
    out = mx.take_along_axis(
        idx, best[..., None], axis=-1
    ).squeeze(-1)
    return out

# JIT/warm everything.
vals, idx = stage1()
mx.eval(vals, idx)
out = reduce(vals, idx)
mx.eval(out)

ref = q._target_verify_quantized_argmax(head, x)
mx.eval(ref)

print("exact output:", bool(mx.all(out == ref).item()))
print("tokens:", ref.tolist())

def bench(fn, n=20):
    xs = []
    for _ in range(n):
        mx.synchronize()
        t0 = time.perf_counter()
        y = fn()
        if isinstance(y, tuple):
            mx.eval(*y)
        else:
            mx.eval(y)
        mx.synchronize()
        xs.append((time.perf_counter() - t0) * 1000)
    return statistics.median(xs), statistics.mean(xs)

s1_med, s1_mean = bench(stage1)

# Materialize one stage1 result, then benchmark reduction alone.
vals, idx = stage1()
mx.eval(vals, idx)

r_med, r_mean = bench(lambda: reduce(vals, idx))

full_med, full_mean = bench(
    lambda: q._target_verify_quantized_argmax(head, x)
)

print("\n========== RESULT ==========")
print(f"stage1 Q8 scan:      median={s1_med:.3f} ms mean={s1_mean:.3f} ms")
print(f"final tile reduction: median={r_med:.3f} ms mean={r_mean:.3f} ms")
print(f"full helper:          median={full_med:.3f} ms mean={full_mean:.3f} ms")
print(f"stage1+reduce:         {s1_med+r_med:.3f} ms")
print(f"tiles/token:           {num_tiles}")
