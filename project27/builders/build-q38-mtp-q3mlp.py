from pathlib import Path
import json
import shutil
import mlx.core as mx

Q3 = Path.home() / "models/Qwen3.8-27B-MTP-oQ3-FP16"
Q6 = Path.home() / "models/Qwen3.8-27B-MTP-oQ6-FP16"

OUT = Path.home() / "models/Qwen3.8-27B-MTP-Q3MLP-Q6ATTN-FP16"

if OUT.exists():
    raise RuntimeError(f"Refusing to overwrite {OUT}")

OUT.mkdir(parents=True)

w3 = mx.load(str(Q3 / "model.safetensors"))
w6 = mx.load(str(Q6 / "model.safetensors"))

MLP = [
    "layers.0.mlp.gate_proj",
    "layers.0.mlp.up_proj",
    "layers.0.mlp.down_proj",
]

ATTN = [
    "layers.0.self_attn.q_proj",
    "layers.0.self_attn.k_proj",
    "layers.0.self_attn.v_proj",
    "layers.0.self_attn.o_proj",
]

NONQUANT = [
    "fc.weight",
    "layers.0.input_layernorm.weight",
    "layers.0.post_attention_layernorm.weight",
    "layers.0.self_attn.k_norm.weight",
    "layers.0.self_attn.q_norm.weight",
    "norm.weight",
    "pre_fc_norm_embedding.weight",
    "pre_fc_norm_hidden.weight",
]

def put(dst, src, base):
    for suffix in ("weight", "scales", "biases"):
        key = f"{base}.{suffix}"
        assert key in src
        dst[key] = src[key]

for p in Q6.iterdir():
    if p.name not in {"model.safetensors", "config.json"} and p.is_file():
        shutil.copy2(p, OUT / p.name)

out = {}

for key in NONQUANT:
    out[key] = w6[key]

for base in MLP:
    put(out, w3, base)

for base in ATTN:
    put(out, w6, base)

mx.save_safetensors(
    str(OUT / "model.safetensors"),
    out,
    metadata={"format": "mlx"},
)

cfg = json.loads((Q6 / "config.json").read_text())

quant = {
    "group_size": 64,
    "bits": 6,
    "mode": "affine",
}

for base in MLP:
    quant[base] = {
        "group_size": 64,
        "bits": 3,
        "mode": "affine",
    }

for base in ATTN:
    quant[base] = {
        "group_size": 64,
        "bits": 6,
        "mode": "affine",
    }

cfg["quantization"] = quant
cfg["quantization_config"] = quant

(OUT / "config.json").write_text(
    json.dumps(cfg, indent=2) + "\n"
)

print("DONE:", OUT)
print(
    "size:",
    f"{(OUT / 'model.safetensors').stat().st_size / 1024**2:.3f} MiB"
)
