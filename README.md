# M1 Max LLM Tuning

Performance engineering notes, benchmark records, source patches, benchmark harnesses, environment snapshots, reproducible tuning experiments, and an emerging config-driven autotuner for modern local LLMs on Apple M1 Max hardware.

## Current Qwen3.8-27B Q6 state

The original Project27 milestone remains preserved at **27.305 tok/s sustained mean** with Q3 MLP + Q6 attention MTP and 139 verification rounds. That historical state is documented in [`PROJECT27_QWEN38_27B_M1_MAX_TUNING_LOG.md`](PROJECT27_QWEN38_27B_M1_MAX_TUNING_LOG.md).

The current certified experimental line has moved substantially beyond Project27. The latest champion is **P44B3**, which changes only the approximate search-head grouping from P44B2 Q4/G32 to lower-metadata **Q4/G64** while keeping the deterministic top-4 reducer and exact Q6/G32 x4 jury unchanged.

Direct same-process alternating 10-pair benchmark:

- **29.201393 tok/s mean** for P44B3 Q4/G64
- **29.057401 tok/s mean** for the certified P44B2 Q4/G32 control in the same session
- **+0.4956% mean paired speedup**
- **+0.4830% median paired speedup**
- **10/10 pair wins**
- 138 speculative verification rounds on every measured run
- exact deterministic text and exact draft/accept trajectory on every run
- 414 exact Q6 jury calls per canonical 512-token generation

Canonical determinism gates:

```text
text hash:       e39b478ae4a8
trajectory hash: f7801569fbdd
rounds:          138
```

P44B3 also passed the full distributional robustness battery:

- 30 prompt/context cases
- five prompt families: code, reasoning, prose, structured, dialogue
- context targets from 64 through 16,384 tokens
- **3,886 real MTP head decisions**
- Q6/G32 winner present in Q4/G64 top-4: **3886/3886**
- exact top-4 -> Q6/G32 jury decision: **3886/3886**
- failures: **0**
- maximum observed Q6-winner rank under Q4/G64: **4**

Observed Q6-winner rank distribution under the Q4/G64 search head:

```text
rank1: 3782
rank2:   94
rank3:    9
rank4:    1
```

The single observed rank-4 state means the current top-4 repair envelope is now fully exercised; future lower-metadata or lower-bit search heads must be robustness-tested before any live speed promotion.

Current post-Project27 stack:

- Q6 affine group-64 target
- fixed-shape / compiled target-verifier work retained from the P34-P36 line
- native Qwen MTP, block size 4 (three actual drafted tokens)
- exact Q6/G32 draft-side target-head oracle/jury
- **Q4/G64 full-vocabulary approximate search head**
- deterministic hierarchical Metal top-4 reducer, 64 groups x 256 threads
- exact Q6/G32 x4 gathered jury with full-vocabulary tie semantics
- Q8 affine group-64 target lm_head with fused multi-token quantized argmax on the verifier path

The detailed continuation from the original Project27 milestone through P44B1 is documented in [`QWEN38_27B_POST_PROJECT27_TUNING_LOG.md`](QWEN38_27B_POST_PROJECT27_TUNING_LOG.md). The later search-head milestones are preserved separately in [`post-project27/P44B2_CERTIFICATION.md`](post-project27/P44B2_CERTIFICATION.md) and [`post-project27/P44B3_CERTIFICATION.md`](post-project27/P44B3_CERTIFICATION.md).

## Autotuner

[`tuner/README.md`](tuner/README.md) documents the config-driven tuning framework. The initial implementation includes campaign definitions, local-neighborhood search, a generic mixed-precision MTP builder, loader validation, SQLite result storage, isolated production benchmarking, DEV/holdout suite execution, promotion rules, and a leaderboard.

The first campaign is [`campaigns/qwen38-q6.toml`](campaigns/qwen38-q6.toml). The goal is to make Q4/Q5/Q6/Q8 targets and new model variants separate campaigns rather than repeating the entire tuning process manually.

## Tuning logs

- [`PROJECT27_QWEN38_27B_M1_MAX_TUNING_LOG.md`](PROJECT27_QWEN38_27B_M1_MAX_TUNING_LOG.md) preserves the original tuning campaign through the 27.305 tok/s Project27 milestone.
- [`QWEN38_27B_POST_PROJECT27_TUNING_LOG.md`](QWEN38_27B_POST_PROJECT27_TUNING_LOG.md) records the later fixed-shape verifier, draft-head quantization, exact-jury, robustness, and bandwidth-optimization work through P44B1.
- [`post-project27/P44B2_CERTIFICATION.md`](post-project27/P44B2_CERTIFICATION.md) records the first certified 29 tok/s-class P44B2 milestone.
- [`post-project27/P44B3_CERTIFICATION.md`](post-project27/P44B3_CERTIFICATION.md) records the Q4/G64 29.20 tok/s milestone, including the full robustness result and direct paired benchmark.

## Actual implementation artifacts

[`project27/MANIFEST.md`](project27/MANIFEST.md) defines the original Project27 reproducibility bundle: benchmark scripts, MTP builders, MLX RAWX source snapshots/patches, the mlx-vlm Q8 SHARED4 patch and exact installed source snapshot, negative-result patches, environment metadata, configs, and SHA-256 manifests.

The Project27 snapshot remains intentionally immutable as a historical milestone. Later P34-P44 work is documented separately rather than rewriting the original Project27 artifact bundle. Model weight files themselves are intentionally not committed; configs, precision maps, hashes, code, patches, and modified-source snapshots are.

## Scope

P44B3 establishes a certified **29.20 tok/s** robust configuration on this tuning line. The next controlled bandwidth experiment is **Q4/G128**, keeping the same packed 4-bit weight payload while halving scale/bias metadata again. Because P44B3 already contains one real rank-4 state, Q4/G128 must clear the full shortlist-recall battery before any speed result can be promoted. Lower-bit search heads remain later challengers rather than the immediate next step.
