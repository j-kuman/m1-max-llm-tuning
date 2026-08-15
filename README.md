# M1 Max LLM Tuning

Performance engineering notes, benchmark records, source patches, benchmark harnesses, environment snapshots, reproducible tuning experiments, and an emerging config-driven autotuner for modern local LLMs on Apple M1 Max hardware.

## Current Qwen3.8-27B Q6 state

The frozen Project27 milestone remains preserved at **27.305 tok/s sustained mean** with Q3 MLP + Q6 attention MTP and 139 verification rounds.

The current experimental champion has moved beyond that baseline:

- **27.556 tok/s sustained mean**
- **27.561 tok/s median**
- 27.518 tok/s minimum across 10 measured 512-token runs
- 27.579 tok/s maximum
- 138 speculative verification rounds on every measured run
- exact deterministic text across the measured batch

Current experimental stack:

- Q6 affine group-64 target
- custom RAWX M4 target kernel
- Q8 affine group-64 lm_head with exact SHARED4 argmax path
- native MTP block size 4
- mixed MTP MLP precision: gate Q4 g64 / up Q3 g64 / down Q4 g64
- MTP attention Q6 g64
- MTP `fc` Q6 g64

## Autotuner

[`tuner/README.md`](tuner/README.md) documents the new config-driven tuning framework. The initial implementation includes campaign definitions, local-neighborhood search, a generic mixed-precision MTP builder, loader validation, SQLite result storage, isolated production benchmarking, DEV/holdout suite execution, promotion rules, and a leaderboard.

The first campaign is [`campaigns/qwen38-q6.toml`](campaigns/qwen38-q6.toml). The goal is to make Q4/Q5/Q6/Q8 targets and new model variants separate campaigns rather than repeating the entire tuning process manually.

## Full Project27 tuning log

See [`PROJECT27_QWEN38_27B_M1_MAX_TUNING_LOG.md`](PROJECT27_QWEN38_27B_M1_MAX_TUNING_LOG.md) for the original tuning history, benchmark methodology, failed experiments, artifact hashes, restoration notes, and portability ideas.

## Actual implementation artifacts

[`project27/MANIFEST.md`](project27/MANIFEST.md) defines the reproducibility bundle: benchmark scripts, MTP builders, MLX RAWX source snapshots/patches, the mlx-vlm Q8 SHARED4 patch and exact installed source snapshot, negative-result patches, environment metadata, configs, and SHA-256 manifests.

The snapshot already checked into this repository preserves the exact Project27 implementation state. Model weight files themselves are intentionally not committed; configs, precision maps, hashes, code, patches, and modified-source snapshots are.

## Scope

Near-term work includes finishing the unattended autotune campaign runner, validating numerical tuning gains against a multi-prompt DEV suite and sealed local holdout, continuing the Qwen3.8 Q6 chase toward 28 tok/s, then applying the same machinery to Q4/Q5/Q8 target quants, alternate model variants, and the two-M1-Max DeepSeek tensor-parallel project.
