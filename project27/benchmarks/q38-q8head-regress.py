import json
import mlx.core as mx
import mlx_vlm
from mlx_vlm.models.qwen3_5 import language as q

MODEL = "/Users/skylinej17/models/Qwen3.8-27B-MLX-6bit-FP16-Q8HEAD"

model, _ = mlx_vlm.load(MODEL)
head = model.language_model.lm_head

K = head.weight.shape[1] * 32 // head.bits

results = {}

for seed in range(100):
    mx.random.seed(seed)

    # Production shape: B=1, T=4
    x = mx.random.normal((1, 4, K)).astype(head.scales.dtype)

    out = q._target_verify_quantized_argmax(head, x)
    mx.eval(out)

    results[str(seed)] = out.tolist()

print(json.dumps(results, sort_keys=True))
