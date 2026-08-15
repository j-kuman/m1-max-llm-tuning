import os
import json
import shutil
from pathlib import Path

import mlx.core as mx

BITS = int(os.environ["BITS"])
assert BITS in (2, 3, 4, 5, 6, 8)

SRC = Path(
    "/Users/skylinej17/.cache/huggingface/hub/"
    "models--Qwen--Qwen3.8-27B/"
    "snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/"
    "model-00018-of-00018.safetensors"
)

GOOD = Path.home() / "models/Qwen3.8-27B-MTP-oQ6-FP16"
OUT = Path.home() / f"models/Qwen3.8-27B-MTP-oQ{BITS}-FP16"

if OUT.exists():
    raise RuntimeError(f"Refusing to overwrite {OUT}")

OUT.mkdir(parents=True)

# Copy tokenizer/support files from known-good standalone MTP.
for p in GOOD.iterdir():
    if p.name in {"model.safetensors", "config.json"}:
        continue
    if p.is_file():
        shutil.copy2(p, OUT / p.name)

# Loader-compatible config.
cfg = json.loads((GOOD / "config.json").read_text())
qcfg = {
    "bits": BITS,
    "group_size": 64,
    "mode": "affine",
}
cfg["quantization"] = dict(qcfg)
cfg["quantization_config"] = dict(qcfg)

(OUT / "config.json").write_text(
    json.dumps(cfg, indent=2) + "\n"
)

src_all = mx.load(str(SRC))
src = {
    k.removeprefix("mtp."): v
    for k, v in src_all.items()
    if k.startswith("mtp.")
}

good = mx.load(str(GOOD / "model.safetensors"))

QUANT = {
    "layers.0.mlp.down_proj.weight",
    "layers.0.mlp.gate_proj.weight",
    "layers.0.mlp.up_proj.weight",
    "layers.0.self_attn.k_proj.weight",
    "layers.0.self_attn.o_proj.weight",
    "layers.0.self_attn.q_proj.weight",
    "layers.0.self_attn.v_proj.weight",
}

# Everything that isn't one of the seven projection weights must come
# verbatim from the known-good converted sidecar, preserving norm sanitation.
out = {}

for name in sorted(src):
    if name in QUANT:
        x = src[name].astype(mx.float16)

        wq, scales, biases = mx.quantize(
            x,
            group_size=64,
            bits=BITS,
            mode="affine",
        )
        mx.eval(wq, scales, biases)

        prefix = name[:-len(".weight")]
        out[name] = wq
        out[prefix + ".scales"] = scales
        out[prefix + ".biases"] = biases

        print(
            f"Q{BITS:<2} {name:60s} "
            f"{str(x.shape):24s} -> {wq.shape}"
        )
    else:
        assert name in good, f"Missing known-good tensor: {name}"
        out[name] = good[name]
        mx.eval(out[name])
        print(f"GOOD {name:60s} {out[name].shape}")

mx.save_safetensors(
    str(OUT / "model.safetensors"),
    out,
    metadata={"format": "mlx"},
)

print("\nDONE:", OUT)
print(
    "model.safetensors:",
    f"{(OUT / 'model.safetensors').stat().st_size / 1024**2:.3f} MiB"
)
