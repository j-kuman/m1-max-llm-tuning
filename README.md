# M1 Max LLM Tuning

Performance engineering notes, benchmark records, and reproducible tuning experiments for running modern local LLMs on Apple M1 Max hardware.

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

## Scope

This repository is intended to preserve the engineering process and reproducible artifacts around M1 Max inference tuning. Model weight files are not committed here; the log records paths, quantization maps, and hashes for locally frozen artifacts.

Future work will include Project28 experiments, Qwen3.8 Q4/Q5/Q8 target tuning, and DeepSeek V4 Flash 0731 tensor-parallel optimization across two 64GB M1 Max machines.
