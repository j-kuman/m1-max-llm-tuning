import statistics
import time

import mlx.core as mx
import mlx_vlm
from mlx_vlm.models.qwen3_5 import language as q

MODEL = "/Users/skylinej17/models/Qwen3.8-27B-MLX-6bit-FP16-Q8HEAD"

model, _ = mlx_vlm.load(MODEL)
head = model.language_model.lm_head

K = head.weight.shape[1] * 32 // head.bits

mx.random.seed(1234)
x = mx.random.normal((1, 4, K)).astype(head.scales.dtype)

mx.eval(x, head.weight, head.scales, head.biases)

# Warm/JIT.
ref = q._target_verify_quantized_argmax(head, x)
mx.eval(ref)

times = []

for _ in range(30):
    mx.synchronize()
    t0 = time.perf_counter()

    y = q._target_verify_quantized_argmax(head, x)
    mx.eval(y)

    mx.synchronize()
    times.append((time.perf_counter() - t0) * 1000)

print("tokens:", ref.tolist())
print("median:", f"{statistics.median(times):.3f} ms")
print("mean:  ", f"{statistics.mean(times):.3f} ms")
print("min:   ", f"{min(times):.3f} ms")
print("max:   ", f"{max(times):.3f} ms")
