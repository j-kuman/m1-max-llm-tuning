# M1 Max LLM Tuning

Performance engineering notes, benchmark records, source patches, benchmark harnesses, environment snapshots, reproducible tuning experiments, and an emerging config-driven autotuner for modern local LLMs on Apple M1 Max hardware.

## Current Qwen3.8-27B Q6 state

The original Project27 milestone remains preserved at **27.305 tok/s sustained mean** with Q3 MLP + Q6 attention MTP and 139 verification rounds. That historical state is documented in [`PROJECT27_QWEN38_27B_M1_MAX_TUNING_LOG.md`](PROJECT27_QWEN38_27B_M1_MAX_TUNING_LOG.md).

The current certified champion is **P45B2B**, a GPU-resident conditional-jury design built on the certified P44B3 Q4/G64 search head.

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

Current post-Project27 stack:

- Q6 affine group-64 target
- fixed-shape / compiled target-verifier work retained from the P34-P36 line
- native Qwen MTP, block size 4 (three actual drafted tokens)
- **Q4/G64 full-vocabulary approximate draft-side search head**
- deterministic hierarchical Metal top-4 reducer, 64 groups x 256 threads
- GPU-resident Q4 confidence test with strict `gap > 0.671875`
- specialized exact Q6/G32 x4 fallback kernel
- exact minimum-vocabulary-ID tie semantics
- Q8 affine group-64 target lm_head with fused multi-token quantized argmax on the verifier path

The detailed continuation from the original Project27 milestone through P44B1 is documented in [`QWEN38_27B_POST_PROJECT27_TUNING_LOG.md`](QWEN38_27B_POST_PROJECT27_TUNING_LOG.md). Certified later milestones are preserved in [`post-project27/P44B2_CERTIFICATION.md`](post-project27/P44B2_CERTIFICATION.md), [`post-project27/P44B3_CERTIFICATION.md`](post-project27/P44B3_CERTIFICATION.md), and [`post-project27/P45B2B_CERTIFICATION.md`](post-project27/P45B2B_CERTIFICATION.md).

## Autotuner

[`tuner/README.md`](tuner/README.md) documents the config-driven tuning framework. The initial implementation includes campaign definitions, local-neighborhood search, a generic mixed-precision MTP builder, loader validation, SQLite result storage, isolated production benchmarking, DEV/holdout suite execution, promotion rules, and a leaderboard.

The first campaign is [`campaigns/qwen38-q6.toml`](campaigns/qwen38-q6.toml). The goal is to make Q4/Q5/Q6/Q8 targets and new model variants separate campaigns rather than repeating the entire tuning process manually.

## Tuning logs

- [`PROJECT27_QWEN38_27B_M1_MAX_TUNING_LOG.md`](PROJECT27_QWEN38_27B_M1_MAX_TUNING_LOG.md) preserves the original tuning campaign through the 27.305 tok/s Project27 milestone.
- [`QWEN38_27B_POST_PROJECT27_TUNING_LOG.md`](QWEN38_27B_POST_PROJECT27_TUNING_LOG.md) records the later fixed-shape verifier, draft-head quantization, exact-jury, robustness, and bandwidth-optimization work through P44B1.
- [`post-project27/P44B2_CERTIFICATION.md`](post-project27/P44B2_CERTIFICATION.md) records the first certified 29 tok/s-class P44B2 milestone.
- [`post-project27/P44B3_CERTIFICATION.md`](post-project27/P44B3_CERTIFICATION.md) records the Q4/G64 29.20 tok/s milestone, including the full robustness result and direct paired benchmark.
- [`post-project27/P45B2B_CERTIFICATION.md`](post-project27/P45B2B_CERTIFICATION.md) records the GPU-resident conditional-jury champion, including exact 3,886-state policy certification and two independent 10-pair promotion sessions.

## Actual implementation artifacts

[`project27/MANIFEST.md`](project27/MANIFEST.md) defines the original Project27 reproducibility bundle: benchmark scripts, MTP builders, MLX RAWX source snapshots/patches, the mlx-vlm Q8 SHARED4 patch and exact installed source snapshot, negative-result patches, environment metadata, configs, and SHA-256 manifests.

The Project27 snapshot remains intentionally immutable as a historical milestone. Later P34-P45 work is documented separately rather than rewriting the original Project27 artifact bundle. Model weight files themselves are intentionally not committed; configs, precision maps, hashes, code, patches, and modified-source snapshots are.

## Scope

P45B2B establishes the current certified **29.32 tok/s-class** robust configuration on this tuning line.

The key change is no longer lower-bit search-head bandwidth alone: P45B2B uses the already-certified Q4/G64 search head as a confidence oracle and conditionally skips the exact Q6/G32 x4 jury on sufficiently separated states, while preserving exact observed behavior across the full 3,886-state battery.

The next controlled optimization is reducer/jury fusion: combine the final hierarchical top-4 reduction, confidence test, and conditional Q6/G32 fallback into a single Metal kernel to remove the remaining dispatch boundary. P45B2B remains the frozen control.
