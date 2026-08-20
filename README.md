# M1 Max LLM Tuning

Performance engineering notes, benchmark records, source patches, benchmark harnesses, environment snapshots, reproducible tuning experiments, and an emerging config-driven autotuner for modern local LLMs on Apple M1 Max hardware.

> **Cold-start / new-chat handoff:** read [`CURRENT_STATUS.md`](CURRENT_STATUS.md) first. It records the certified controls, current local development state, closed P47-P50 experiments, benchmark discipline, and the exact next experiment.

## Current Qwen3.8-27B Q6 state

The original Project27 milestone remains preserved at **27.305 tok/s sustained mean** with Q3 MLP + Q6 attention MTP and 139 verification rounds. That historical state is documented in [`PROJECT27_QWEN38_27B_M1_MAX_TUNING_LOG.md`](PROJECT27_QWEN38_27B_M1_MAX_TUNING_LOG.md).

The current certified overall champion is **P45B2B D4**, a GPU-resident conditional-jury design built on the certified P44B3 Q4/G64 search head.

P45B2B keeps the Q4/G64 full-vocabulary approximate search, deterministic hierarchical Metal top-4 reduction, and exact Q6/G32 fallback semantics. It adds a GPU-resident confidence test using the Q4 top1-top2 score gap. When the gap is strictly greater than `0.671875`, the approximate top1 is returned immediately; otherwise a specialized exact Q6/G32 x4 Metal kernel resolves the four candidates. No scalar is transferred to the CPU and no per-decision host synchronization is introduced.

Normal-performance direct 10-pair rerun:

- **29.316537 tok/s mean** for P45B2B
- **29.157123 tok/s mean** for P44B3 in the same alternating session
- **+0.5469% mean paired speedup**
- **+0.5201% median paired speedup**
- **10/10 pair wins**

An earlier independent 10-pair session also produced **10/10 wins**, for **20/20 cumulative paired wins** across the two promotion sessions.

Canonical determinism remained unchanged:

```text
text hash:       e39b478ae4a8
trajectory hash: f7801569fbdd
rounds:          138
jury calls:      414
```

The actual P45B2B GPU policy passed the full distributional robustness battery:

- 30 prompt/context cases
- five prompt families: code, reasoning, prose, structured, dialogue
- context targets from 64 through 16,384 tokens
- **3,886 real MTP head decisions**
- actual conditional GPU-policy decisions: **3886/3886 exact**
- policy mismatches: **0**

The specialized Q6/G32 x4 fallback was separately verified at score level:

```text
FP16 score equality:    3886/3886
maximum absolute error: 0.0
token mismatches:       0
```

Current P45B2B D4 stack:

- Q6 affine group-64 target
- fixed-shape / compiled target-verifier work retained from the P34-P36 line
- native Qwen MTP, block size 4 (three actual drafted tokens)
- **Q4/G64 full-vocabulary approximate draft-side search head**
- deterministic hierarchical Metal top-4 reducer, 64 groups x 256 threads
- GPU-resident Q4 confidence test with strict `gap > 0.671875`
- specialized exact Q6/G32 x4 fallback kernel
- exact minimum-vocabulary-ID tie semantics
- Q8 affine group-64 target lm_head with fused multi-token quantized argmax on the verifier path

## P46 D5 checkpoint

P46 explores requested speculative width 5, which means **four actual drafted tokens per round** in the current wrapper. D5 improves commits per target round and reduces the canonical target-round count from 138 to 128, but it also performs 512 draft/jury decisions instead of 414. The initial D5 implementation was therefore much slower than D4 until the M=5 body and T=5 verifier paths were specialized directly.

The current certified P46 D5 checkpoint is **26.550819 tok/s**.

Final 10-pair certification of the last selective fixed-shape step:

```text
baseline mean:      26.524079 tok/s
candidate mean:     26.550819 tok/s
paired mean:        +0.1008%
paired median:      +0.1230%
wins:                8/10
round latency:       150.654 ms
```

Every run preserved the canonical D5 behavior:

```text
text hash:       e39b478ae4a8
trajectory hash: 183cd3043746
rounds:          128
jury calls:      512
```

Major P46 steps include:

- native shared-five Q6/G64 `FAST_M5`, replacing the old split M5 construction
- N=48 generic fallback
- fixed M5 specialization for the major model geometries
- a Q8/G64 T=5 target-head **2+2+1** decomposition that removed a sharp verifier resource cliff
- selective extra fixed shapes for `5120 -> 6144` and `6144 -> 5120`
- explicit rejection of the noisy `5120 -> 1024` fixed specialization after the three-shape bundle failed end-to-end certification

The Q8 T5 2+2+1 verifier-head change was the largest late P46 win:

```text
native-head mean: 26.050966 tok/s
2+2+1 mean:       26.514193 tok/s
paired mean:      +1.7782%
median:           +1.7866%
wins:             10/10
saved/round:       2.683 ms
```

P46 also preserves negative results. A lower-occupancy P37-8 attention geometry regressed versus P36, and a Q6 M5 CHUNK2 experiment was bitwise exact but lost end-to-end because repeated decode work erased the reduced-live-state benefit.

D5 is **not** the overall champion yet. At 26.550819 tok/s it remains about **9.43% below** the certified P45B2B D4 champion. The D5 round-latency break-even target is approximately **136.442 ms/round**, versus the current 150.654 ms/round.

The detailed P46 record is preserved in [`post-project27/P46_D5_CHECKPOINT.md`](post-project27/P46_D5_CHECKPOINT.md).

The detailed continuation from the original Project27 milestone through P44B1 is documented in [`QWEN38_27B_POST_PROJECT27_TUNING_LOG.md`](QWEN38_27B_POST_PROJECT27_TUNING_LOG.md). Certified later milestones are preserved in [`post-project27/P44B2_CERTIFICATION.md`](post-project27/P44B2_CERTIFICATION.md), [`post-project27/P44B3_CERTIFICATION.md`](post-project27/P44B3_CERTIFICATION.md), [`post-project27/P45B2B_CERTIFICATION.md`](post-project27/P45B2B_CERTIFICATION.md), and [`post-project27/P46_D5_CHECKPOINT.md`](post-project27/P46_D5_CHECKPOINT.md).

## Autotuner

[`tuner/README.md`](tuner/README.md) documents the config-driven tuning framework. The initial implementation includes campaign definitions, local-neighborhood search, a generic mixed-precision MTP builder, loader validation, SQLite result storage, isolated production benchmarking, DEV/holdout suite execution, promotion rules, and a leaderboard.

The first campaign is [`campaigns/qwen38-q6.toml`](campaigns/qwen38-q6.toml). The goal is to make Q4/Q5/Q6/Q8 targets and new model variants separate campaigns rather than repeating the entire tuning process manually.

## Tuning logs

- [`CURRENT_STATUS.md`](CURRENT_STATUS.md) is the cold-start handoff and current resume point.
- [`PROJECT27_QWEN38_27B_M1_MAX_TUNING_LOG.md`](PROJECT27_QWEN38_27B_M1_MAX_TUNING_LOG.md) preserves the original tuning campaign through the 27.305 tok/s Project27 milestone.
- [`QWEN38_27B_POST_PROJECT27_TUNING_LOG.md`](QWEN38_27B_POST_PROJECT27_TUNING_LOG.md) records the later fixed-shape verifier, draft-head quantization, exact-jury, robustness, and bandwidth-optimization work through P44B1.
- [`post-project27/P44B2_CERTIFICATION.md`](post-project27/P44B2_CERTIFICATION.md) records the first certified 29 tok/s-class P44B2 milestone.
- [`post-project27/P44B3_CERTIFICATION.md`](post-project27/P44B3_CERTIFICATION.md) records the Q4/G64 29.20 tok/s milestone, including the full robustness result and direct paired benchmark.
- [`post-project27/P45B2B_CERTIFICATION.md`](post-project27/P45B2B_CERTIFICATION.md) records the GPU-resident conditional-jury champion, including exact 3,886-state policy certification and two independent 10-pair promotion sessions.
- [`post-project27/P46_D5_CHECKPOINT.md`](post-project27/P46_D5_CHECKPOINT.md) records the D5 width-5 campaign through native M5, the Q8 T5 2+2+1 verifier head, rejected CHUNK2/P37-8 paths, and the certified selective two-shape fixed-M5 checkpoint.
- [`post-project27/P47_P50_SEARCH_HEAD_CHECKPOINT.md`](post-project27/P47_P50_SEARCH_HEAD_CHECKPOINT.md) records the post-P46 search-head/reducer campaign and its pause point.

## Actual implementation artifacts

[`project27/MANIFEST.md`](project27/MANIFEST.md) defines the original Project27 reproducibility bundle: benchmark scripts, MTP builders, MLX RAWX source snapshots/patches, the mlx-vlm Q8 SHARED4 patch and exact installed source snapshot, negative-result patches, environment metadata, configs, and SHA-256 manifests.

The Project27 snapshot remains intentionally immutable as a historical milestone. Later P34-P46 work is documented separately rather than rewriting the original Project27 artifact bundle. Model weight files themselves are intentionally not committed; configs, precision maps, hashes, code, patches, and modified-source snapshots are.

## Scope

P45B2B establishes the current certified **29.32 tok/s-class overall champion** on this tuning line. P46 establishes a separate certified **26.55 tok/s-class D5 checkpoint** whose value is architectural: it narrows the width-5 round-cost gap while preserving a higher commits-per-round regime.

Post-P46 search-head/reducer experiments through P50B are documented in `CURRENT_STATUS.md` and `post-project27/P47_P50_SEARCH_HEAD_CHECKPOINT.md`. The exact resume point is **P50C fixed-shape Q4/G64 specialization** preserving native 2-SIMDgroup x 4-row geometry while baking the hot M=1, K=5120, N=248320 shape into the kernel.

A later context-length matrix should compare D4 and D5 directly. D5 saves target forwards, so the optimal runtime may eventually dispatch different speculative widths by context length rather than use one global width.
