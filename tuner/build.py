from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import mlx.core as mx

from .spec import expand_path, load_campaign, load_candidate


BUILD_VERSION = 2


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def normalized_qspec(qspec: dict) -> tuple[int, int, str]:
    return (
        int(qspec["bits"]),
        int(qspec["group_size"]),
        str(qspec.get("mode", "affine")),
    )


def modules_to_rebuild(campaign: dict, candidate: dict) -> list[str]:
    """Return only modules whose quantization spec differs from the champion.

    Search candidates are defined as coordinate mutations around the frozen
    champion. Unchanged quantized tensors must therefore be copied byte-for-byte
    from the champion sidecar rather than being re-quantized from source BF16.
    """
    champion_modules = campaign["champion"]["modules"]
    changed: list[str] = []
    for module, qspec in candidate["modules"].items():
        base_qspec = champion_modules.get(module)
        if base_qspec is None or normalized_qspec(qspec) != normalized_qspec(base_qspec):
            changed.append(module)
    return changed


def build_candidate(campaign: dict, candidate: dict, output: Path, force: bool = False) -> Path:
    c = campaign["campaign"]
    source_shard = expand_path(c["pristine_mtp_shard"])
    base_draft = expand_path(candidate.get("base_draft", campaign["champion"]["draft"]))

    if output.exists():
        complete = (output / "autotune-manifest.json").exists()
        if complete and not force:
            raise RuntimeError(f"Refusing to overwrite completed build {output}; pass --force to replace it")
        if not complete and not force:
            print("Removing incomplete candidate build:", output)
        shutil.rmtree(output)
    output.mkdir(parents=True)

    print("Loading base drafter:", base_draft)
    base_weights = mx.load(str(base_draft / "model.safetensors"))

    modules = candidate["modules"]
    rebuilt_modules = modules_to_rebuild(campaign, candidate)
    preserved_modules = [m for m in modules if m not in rebuilt_modules]

    if not rebuilt_modules:
        raise RuntimeError("candidate does not differ from champion quantization map")

    print("Preserving champion quantized tensors for:", ", ".join(preserved_modules))
    print("Rebuilding from pristine BF16:", ", ".join(rebuilt_modules))

    print("Loading pristine MTP shard:", source_shard)
    source_all = mx.load(str(source_shard))

    out = dict(base_weights)

    for module in rebuilt_modules:
        qspec = modules[module]
        bits = int(qspec["bits"])
        group = int(qspec["group_size"])
        mode = qspec.get("mode", "affine")
        source_key = f"mtp.{module}.weight"
        if source_key not in source_all:
            raise KeyError(f"Missing pristine tensor: {source_key}")

        for suffix in ("weight", "scales", "biases"):
            out.pop(f"{module}.{suffix}", None)

        x = source_all[source_key].astype(mx.float16)
        wq, scales, biases = mx.quantize(
            x,
            group_size=group,
            bits=bits,
            mode=mode,
        )
        mx.eval(wq, scales, biases)

        out[f"{module}.weight"] = wq
        out[f"{module}.scales"] = scales
        out[f"{module}.biases"] = biases
        print(f"{module:42s} q{bits} g{group} {tuple(x.shape)} -> {tuple(wq.shape)}")

    model_path = output / "model.safetensors"
    mx.save_safetensors(str(model_path), out, metadata={"format": "mlx"})

    for p in base_draft.iterdir():
        if p.is_file() and p.name not in {"model.safetensors", "config.json"}:
            shutil.copy2(p, output / p.name)

    cfg = json.loads((base_draft / "config.json").read_text())
    default_q = campaign.get("quantization", {})
    qcfg = {
        "bits": int(default_q.get("bits", 6)),
        "group_size": int(default_q.get("group_size", 64)),
        "mode": default_q.get("mode", "affine"),
    }
    for module, qspec in modules.items():
        qcfg[module] = {
            "bits": int(qspec["bits"]),
            "group_size": int(qspec["group_size"]),
            "mode": qspec.get("mode", "affine"),
        }

    cfg["quantization"] = qcfg
    cfg["quantization_config"] = dict(qcfg)
    (output / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    manifest = {
        "build_version": BUILD_VERSION,
        "candidate": candidate,
        "campaign": c["name"],
        "base_draft": str(base_draft),
        "pristine_mtp_shard": str(source_shard),
        "rebuilt_modules": rebuilt_modules,
        "preserved_modules": preserved_modules,
        "model_sha256": sha256_file(model_path),
        "config_sha256": sha256_file(output / "config.json"),
    }
    (output / "autotune-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("BUILT:", output)
    print("model size MiB:", f"{model_path.stat().st_size / 1024**2:.3f}")
    print("model sha256:", manifest["model_sha256"])
    return output


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a mixed-precision MTP candidate while preserving unchanged champion tensors exactly.")
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--output")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    campaign = load_campaign(args.campaign)
    candidate = load_candidate(args.candidate)
    root = expand_path(campaign["campaign"]["output_dir"])
    output = expand_path(args.output) if args.output else root / candidate["id"]
    build_candidate(campaign, candidate, output, force=args.force)


if __name__ == "__main__":
    main()
