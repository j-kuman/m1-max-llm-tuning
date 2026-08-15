# M1 Max LLM Tuning

Performance engineering notes, benchmark records, source patches, benchmark harnesses, environment snapshots, and reproducible tuning experiments for running modern local LLMs on Apple M1 Max hardware.

## Current milestone: Project27

**Qwen3.8-27B Q6 on one 64GB M1 Max**

- 27.305 tok/s sustained mean
- 27.315 tok/s median
- 27.272 tok/s minimum across 10 measured 512-token runs
- 139 speculative verification rounds on every measured run
- 24.858 GB peak memory
- deterministic cross-drafter output matched the prior exact Q6 drafter character-for-character

Final stack:

- Q6 affine group-64 target
- custom RAWX M4 target kernel
- Q8 affine group-64 lm_head with exact SHARED4 argmax path
- native MTP block size 4
- mixed MTP precision: Q3 MLP + Q6 attention

## Full tuning log

See [`PROJECT27_QWEN38_27B_M1_MAX_TUNING_LOG.md`](PROJECT27_QWEN38_27B_M1_MAX_TUNING_LOG.md) for the complete tuning history, benchmark methodology, failed experiments, artifact hashes, restoration notes, and portability ideas for other Qwen3.8 quants and the two-M1-Max DeepSeek V4 Flash TP project.

## Actual implementation artifacts

This repo is intended to preserve more than notes. [`project27/MANIFEST.md`](project27/MANIFEST.md) defines the reproducibility bundle: the actual benchmark scripts, MTP builders, MLX RAWX source snapshots/patches, mlx-vlm Q8 SHARED4 patch and exact installed source snapshot, negative-result patches, environment metadata, configs, and SHA-256 manifests.

The source-of-truth versions of several of those files currently live on the tuning Mac under `~/src/mlx-m1-qmv`, `~/project24-patches`, and `/tmp`. The repo contains a one-command capture script so those exact local files—not reconstructed approximations—can be checked in:

```bash
cd ~/src/m1-max-llm-tuning
bash scripts/snapshot_local_project27.sh
```

That script records the local MLX git history/tag/base state, exports the final RAWX source files and patch, copies the exact Q8 SHARED4 `language.py`/patch, captures the benchmark and MTP-builder scripts, stores target/drafter configs plus hashes, snapshots the Python/Metal environment, generates `project27/SHA256SUMS.txt`, commits, and pushes the result.

## Scope

Model weight files are not committed here; the repository records configs, quantization maps, hashes, code, patches, exact modified-source snapshots, and benchmark harnesses needed to reconstruct the tuned stack.

Future work will include Project28 experiments, Qwen3.8 Q4/Q5/Q8 target tuning, and DeepSeek V4 Flash 0731 tensor-parallel optimization across two 64GB M1 Max machines.
