# M1 Max LLM Tuning

Performance engineering notes, benchmark records, source patches, benchmark harnesses, environment snapshots, reproducible tuning experiments, and an emerging config-driven autotuner for modern local LLMs on Apple M1 Max hardware.

## Current Qwen3.8-27B Q6 state

The original Project27 milestone remains preserved at **27.305 tok/s sustained mean** with Q3 MLP + Q6 attention MTP and 139 verification rounds. That historical state is documented in [`PROJECT27_QWEN38_27B_M1_MAX_TUNING_LOG.md`](PROJECT27_QWEN38_27B_M1_MAX_TUNING_LOG.md).

The current certified experimental line has moved substantially beyond Project27. The latest champion is **P44B1**, measured in a same-process alternating 10-pair benchmark:

- **28.953139 tok/s mean** for P44B1 Q5/G64
- **28.824832 tok/s mean** for the certified P43C Q5/G32 control
- **+0.4452% mean paired speedup**
- **+0.5223% median paired speedup**
- **9/10 pair wins**
- 138 speculative verification rounds on every measured run
- exact deterministic text and exact draft/accept trajectory on every run
- 414 exact Q6 jury calls per canonical 512-token generation

Canonical determinism gates:

```text
text hash:       e39b478ae4a8
trajectory hash: f7801569fbdd
rounds:          138
```

P44B1 also passed the full distributional robustness battery:

- 30 prompt/context cases
- five prompt families: code, reasoning, prose, structured, dialogue
- context targets from 64 through 16,384 tokens
- **3,886 real MTP head decisions**
- Q6/G32 winner present in Q5/G64 top-4: **3886/3886**
- exact top-4 -> Q6/G32 jury decision: **3886/3886**
- failures: **0**
- maximum observed Q6-winner rank under Q5/G64: **3**

Current post-Project27 stack:

- Q6 affine group-64 target
- fixed-shape / compiled target-verifier work retained from the P34-P36 line
- native Qwen MTP, block size 4 (three actual drafted tokens)
- exact Q6/G32 draft-side target-head oracle/jury
- Q5/G64 full-vocabulary search head
- deterministic hierarchical Metal top-4 reducer, 64 groups x 256 threads
- exact Q6/G32 x4 gathered jury with full-vocabulary tie semantics
- Q8 affine group-64 target lm_head with fused multi-token quantized argmax on the verifier path

The continuation from the original Project27 milestone through P44B1 is documented in [`QWEN38_27B_POST_PROJECT27_TUNING_LOG.md`](QWEN38_27B_POST_PROJECT27_TUNING_LOG.md).

## Autotuner

[`tuner/README.md`](tuner/README.md) documents the config-driven tuning framework. The initial implementation includes campaign definitions, local-neighborhood search, a generic mixed-precision MTP builder, loader validation, SQLite result storage, isolated production benchmarking, DEV/holdout suite execution, promotion rules, and a leaderboard.

The first campaign is [`campaigns/qwen38-q6.toml`](campaigns/qwen38-q6.toml). The goal is to make Q4/Q5/Q6/Q8 targets and new model variants separate campaigns rather than repeating the entire tuning process manually.

## Tuning logs

- [`PROJECT27_QWEN38_27B_M1_MAX_TUNING_LOG.md`](PROJECT27_QWEN38_27B_M1_MAX_TUNING_LOG.md) preserves the original tuning campaign through the 27.305 tok/s Project27 milestone.
- [`QWEN38_27B_POST_PROJECT27_TUNING_LOG.md`](QWEN38_27B_POST_PROJECT27_TUNING_LOG.md) records the later fixed-shape verifier, draft-head quantization, exact-jury, robustness, and bandwidth-optimization work through P44B1.

## Actual implementation artifacts

[`project27/MANIFEST.md`](project27/MANIFEST.md) defines the original Project27 reproducibility bundle: benchmark scripts, MTP builders, MLX RAWX source snapshots/patches, the mlx-vlm Q8 SHARED4 patch and exact installed source snapshot, negative-result patches, environment metadata, configs, and SHA-256 manifests.

The Project27 snapshot remains intentionally immutable as a historical milestone. Later P34-P44 work is documented separately rather than rewriting the original Project27 artifact bundle. Model weight files themselves are intentionally not committed; configs, precision maps, hashes, code, patches, and modified-source snapshots are.

## Scope

Near-term work is now focused on the remaining draft-head bandwidth ceiling: Q4/G32 and Q4/G64 search heads with exact Q6/G32 shortlist jury recovery, plus continued robustness-first optimization of the existing MTP architecture. Larger architecture experiments remain separate challengers and do not replace the frozen champion unless they beat it under the same benchmark discipline.
