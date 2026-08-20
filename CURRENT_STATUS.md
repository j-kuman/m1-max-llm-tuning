# Current status / cold-start handoff

**Purpose:** this is the first file to read when resuming the Qwen3.8-27B M1 Max tuning project from a fresh chat or a fresh session.

This file is intentionally concise enough to scan quickly but complete enough to reconstruct the active state, the certified controls, the closed branches, the benchmark discipline, and the exact next experiment.

---

## 1. What project is this?

We are tuning **Qwen3.8-27B** inference on an **Apple M1 Max, 64 GB unified memory, 32-core GPU**, using a custom MLX fork and a native Qwen MTP speculative drafter.

The project has two separate notions of "best":

- the **overall throughput champion** is the D4 configuration, P45B2B;
- the active engineering line is a **D5 width-5 campaign**, whose current certified checkpoint is P46.

Do not overwrite or reinterpret either certified state when exploring new candidates.

---

## 2. Certified controls

### Overall champion: P45B2B D4

```text
throughput:      29.316537 tok/s
requested width: 4
actual drafts:   3 tokens/round
rounds:          138
jury decisions:  414
text hash:       e39b478ae4a8
trajectory hash: f7801569fbdd
```

Architecture:

```text
Q4/G64 full-vocabulary approximate draft head
  -> deterministic hierarchical Metal top4
  -> Q4 top1-top2 confidence gap
  -> if gap > 0.671875: return Q4 top1
  -> otherwise exact Q6/G32 x4 Metal jury
  -> lowest-vocabulary-ID tie semantics
```

Robustness:

```text
3,886 real MTP head decisions
conditional policy exact: 3886/3886
Q6/G32 x4 score parity:    3886/3886
maximum score error:       0.0
```

Frozen local source milestone:

```text
commit: 4efc3b4
tag:    project45b2b-gpu-bypass-certified-29.32
```

Detailed record:

- `post-project27/P45B2B_CERTIFICATION.md`

### Certified D5 checkpoint: P46

```text
throughput:      26.550819 tok/s
requested width: 5
actual drafts:   4 tokens/round
rounds:          128
jury decisions:  512
round latency:   150.654 ms
text hash:       e39b478ae4a8
trajectory hash: 183cd3043746
```

The D5 break-even target versus the D4 champion is approximately:

```text
136.442 ms/round
```

Frozen public MLX source:

```text
repository: skylinej3o1/mlx
commit:     22b52dfa9192ccb76a2c0e84d6e2f1383505023f
tag:        project46-d5-certified-26.55
```

Certified P46 implementation includes:

```text
body:
  Q6/G64
  native FAST_M5
  N=48 generic fallback
  fixed major M5 shapes
  extra fixed 5120 -> 6144
  extra fixed 6144 -> 5120
  5120 -> 1024 explicitly rejected/skipped

target verifier:
  VERIFY_LEN=5
  P36 T5 attention
  Q8/G64 target head staged 2+2+1

draft token policy:
  inherited P45B2B Q4/G64 search
  deterministic top4
  GPU confidence bypass
  exact Q6/G32 x4 fallback
```

Detailed records:

- `post-project27/P46_D5_CHECKPOINT.md`
- `post-project27/P46_SOURCE_FREEZE.md`

---

## 3. Current local development state

The active local custom-MLX checkout is:

```text
~/src/mlx-m1-qmv
```

Current experimental local branch at the pause point:

```text
project48-search-head-bandwidth
```

The post-P46 experiments described below were done with standalone `/tmp` harnesses. **No P47-P50 candidate was integrated into the certified MLX source.** The certified P46 freeze remains the source control.

Before any future source mutation, confirm:

```bash
git status -sb
git rev-parse --short=12 HEAD
```

Do not mutate the P46 champion snapshot or certified tag.

---

## 4. What was tested after P46?

The full pause record is:

- `post-project27/P47_P50_SEARCH_HEAD_CHECKPOINT.md`

Summary follows.

### P47: reducer / fusion / threadgroup experiments — CLOSED

P47A fused reducer/confidence/conditional-Q6 work regressed. Stage-2 threadgroup sweeps at 32, 64, and 128 did not produce a repeatable win.

The apparently promising TG64 3-pair result failed its 10-pair certification:

```text
paired mean:   -0.0109%
paired median: +0.0030%
wins:           6/10
saved/round:   -0.017 ms
```

Conclusion: reducer/dispatch geometry is exhausted for now.

### P48: Q4/G128 search head — CLOSED

Robustness was excellent:

```text
rank1: 3781/3886
rank2: 3881/3886
rank4: 3885/3886
rank5: 3886/3886
max rank: 5
safe strict gap threshold: > 0.671875
safe bypass: 3385/3886 = 87.108%
```

But the live G128/top5/x5 path lost:

```text
paired mean: -0.3892%
wins:         0/3
round delta: -0.588 ms saved/round
```

A head-only G64-vs-G128 top4 diagnostic was effectively flat, proving that the lost time was mainly repair/top5 tax rather than a hidden G128 projection win.

Conclusion: G128 closed.

### P49 Q3/G64 — CLOSED

Robustness was surprisingly good:

```text
rank1: 3719/3886
max rank: 6
top8: 3886/3886
safe strict gap threshold: > 1.390625
safe bypass: 2925/3886 = 75.270%
```

But native MLX Q3/G64 was slower than Q4/G64:

```text
Q4 mean:       2.895722 ms
Q3 mean:       3.103388 ms
paired speed: -6.0632%
median:       -5.1672%
wins:          3/10
```

Reason: the 3-bit packed decode/unpack path is more expensive and erases the memory-traffic advantage.

Conclusion: Q3 closed.

### P49 Q2/G64 — CLOSED, but important result

Q2 had a real raw projection win:

```text
Q4 mean:        3.121500 ms
Q2 mean:        2.649028 ms
paired speedup: +18.8004%
median:         +17.2529%
wins:            8/10
gross saving:    +0.472473 ms/head
                 +1.889890 ms/D5 round
```

Full 3,886-state Q2 robustness:

```text
rank histogram:
  1: 3467
  2: 309
  3: 68
  4: 25
  5: 7
  6: 6
  7: 1
  8: 1
  9: 2

max rank: 9
top9:    3886/3886
```

Safe confidence policy:

```text
max wrong-top1 gap: 2.750000
safe rule:          bypass iff gap > 2.750000
safe bypass:        2181/3886 = 56.125%
repair:             1705/3886 = 43.875%
wrong bypass:       0
```

A staged exact repair architecture was explored:

```text
Q2 full vocab
  -> shortlist
  -> Q4/G64 shortlist rerank
  -> Q4 top4
  -> exact Q6/G32 x4
```

It was mathematically viable, but repair frequency and launch/scoring cost consumed the raw Q2 win.

Representative repair micros:

```text
P49E generic Q4x16 + generic Q6x4:
  repair sum: 1.309561 ms
  rough net:  +0.036311 ms/D5 round before reducer cost

P49F1 generic Q4x16 + certified direct Q6x4:
  repair sum: 1.168095 ms
  rough net:  +0.212343 ms/D5 round before reducer cost

P49F2 custom direct Q4x16 + direct Q6x4:
  Q4 parity: bitwise exact
  repair sum: 1.274245 ms
  rough net: +0.011242 ms/D5 round

P49F2B two-64-thread-group Q4x16:
  Q4 parity: bitwise exact
  repair sum: 1.282810 ms
  rough net: -0.008853 ms/D5 round

P49G generic Q4x12 + direct Q6x4:
  repair sum: 1.258372 ms
  rough net: +0.063452 ms/D5 round before reducer cost
```

Conclusion: Q2 itself is genuinely faster and surprisingly accurate, but a 43.875% repair rate makes exact recovery too expensive on this implementation. Close P49 unless a fundamentally cheaper repair primitive appears later.

### P50A/B: Q4/G64 full-head SIMD geometry — CLOSED

All tested full-head custom geometries were bitwise exact against native MLX:

```text
S1R8  bitwise exact
S2R4  bitwise exact
S4R2  bitwise exact
S8R1  bitwise exact
```

P50A sweep:

```text
native mean: 3.481347 ms
S1R8:       3.785594 ms  (-8.037%)
S2R4:       3.470414 ms  (+0.315%, noisy)
S4R2:       3.581755 ms  (-2.803%)
S8R1:       4.246861 ms  (-18.025%)
```

Only S2R4 was close enough to isolate. P50B then ran a 10-pair native-vs-fixed-S2R4 test with 40 calls/sample:

```text
native mean:       2.914103 ms
fixed S2R4 mean:   2.981191 ms
paired mean:      -3.5885%
paired median:    +6.4824%
wins:              6/10
mean latency save: -0.067088 ms/head
D5 gross save:     -0.268352 ms/round
```

Conclusion: native MLX's **2 SIMDgroups x 4 output rows** geometry is already the right geometry. Do not continue permuting SIMDgroup/result layouts.

---

## 5. Exact next experiment after the detour

### P50C: fixed-shape Q4/G64 specialization, native geometry preserved

This is the immediate resume point.

Keep the native Q4/G64 row/SIMD mapping unchanged:

```text
M = 1
K = 5120
N = 248320
bits = 4
group size = 64
num SIMDgroups = 2
results per SIMDgroup = 4
K block size = 512
K blocks = 10 exactly
```

The hypothesis is **not** a new geometry. The hypothesis is that this one extremely hot fixed shape may benefit from compile-time specialization while preserving native arithmetic:

```text
- bake K=5120
- bake N=248320
- bake Q4/G64
- preserve 2x4 geometry
- simplify runtime dimension arithmetic
- simplify pointer/stride setup
- allow the compiler to unroll or optimize the exact 10-block loop
```

Frozen MLX `qmv_fast_impl` still carries runtime `in_vec_size` and `out_vec_size`, computes dynamic row/weight/group strides, and loops using the runtime K bound. P50C should remove only that generality.

### P50C required order

Do not jump straight into end-to-end integration.

```text
1. standalone fixed-shape Metal prototype
2. bitwise parity against native mx.quantized_matmul
3. isolated same-process native-vs-fixed microbenchmark
4. if convincingly positive: 3-pair D5 E2E scout
5. only if scout survives: 10-pair promotion test
6. integrate source only after the performance gate passes
```

If P50C is flat/negative, close fixed-shape Q4 projection specialization and choose a genuinely different bottleneck rather than another geometry permutation.

---

## 6. Benchmark and promotion discipline

These rules matter. Small local wins have repeatedly disappeared end-to-end.

### Exactness

Unless an experiment is explicitly approximate, require bitwise or exact deterministic behavior.

Canonical D5:

```text
text hash:       e39b478ae4a8
trajectory hash: 183cd3043746
rounds:          128
jury decisions:  512
```

Canonical D4 champion:

```text
text hash:       e39b478ae4a8
trajectory hash: f7801569fbdd
rounds:          138
jury decisions:  414
```

### Approximate-head candidates

Before promotion:

```text
1. full 30-case / 3,886-state rank census
2. exact shortlist envelope
3. maximum wrong-top1 confidence gap
4. strict safe threshold
5. full conditional-policy exactness 3886/3886
6. E2E scout
7. full paired promotion test
```

### Small performance deltas

Use same-process alternating pairs.

Three pairs are a **scout only**.

Promotion gate for a 10-pair run:

```text
paired mean   > +0.03%
paired median > 0
wins          >= 6/10
exact         10/10
```

Do not promote from an isolated microbenchmark alone.

---

## 7. Working conventions for a fresh chat

When helping with this project:

- Treat pasted terminal output and local source inspection as authoritative.
- Do not blindly mutate source. Use exact/structural old->new replacements with assertions and abort if anchors differ.
- Never mutate a certified champion directly; work on a branch/copy, validate, then promote.
- Do not use `git add -A`, `git add .`, or `git add --all`; stage only explicit paths.
- Prefer `git --no-pager diff` when asking for diffs.
- On macOS zsh, avoid standalone shell comment lines beginning with `#` in copy/paste command blocks.
- Put `env -u NAME` options before environment assignments.
- If a benchmark labels variants "dynamic" or "fixed", inspect the actual wrapper logic rather than trusting stale labels.
- Keep negative results. They are part of the search-space map.
- Do not claim a new champion unless it passes the exactness and paired promotion gates.

---

## 8. Useful local harness names at the pause point

Recent standalone harnesses include:

```text
/tmp/p49b-q3-vs-q4-head-micro.py
/tmp/p49c-q2-vs-q4-head-micro.py
/tmp/p49d-q2g64-full-battery.py
/tmp/p49e-shortlist-repair-micro.py
/tmp/p49f1-shortlist-custom-q6-micro.py
/tmp/p49f2-direct-q4x16-micro.py
/tmp/p49f2b-direct-q4x16-2x64-micro.py
/tmp/p49g-q4x12-custom-q6-micro.py
/tmp/p50a-q4-fullhead-geometry.py
/tmp/p50b-q4-native-vs-s2r4.py
```

These are local `/tmp` artifacts, not guaranteed to exist after reboot or on another machine. The GitHub checkpoint documents their results so the project state does not depend on them surviving.

---

## 9. File-reading order for a brand-new session

For a cold start, read in this order:

```text
1. CURRENT_STATUS.md                         <- start here
2. post-project27/P47_P50_SEARCH_HEAD_CHECKPOINT.md
3. post-project27/P46_SOURCE_FREEZE.md
4. post-project27/P46_D5_CHECKPOINT.md
5. post-project27/P45B2B_CERTIFICATION.md
```

Only go back into older Project27 history if an experiment depends on it.

---

## 10. One-paragraph handoff

The project currently has a frozen **P45B2B D4 overall champion at 29.316537 tok/s** and a frozen **P46 D5 checkpoint at 26.550819 tok/s**. Post-P46 work exhaustively tested reducer fusion, Q4/G128, Q3/G64, Q2/G64 with staged exact repair, and alternative Q4 full-head SIMD geometries. None earned source integration. The most interesting negative result was Q2/G64: about **+18.8% raw head speed**, exact-winner rank <= 9 across all 3,886 robustness states, but only **56.125% safe bypass**, making exact repair too frequent and expensive. P50A/B then established that native MLX's **2 SIMDgroup x 4-row Q4 geometry is already the correct geometry**. The immediate next experiment is therefore **P50C: a fixed-shape Q4/G64 M=1, K=5120, N=248320 specialization that preserves native 2x4 arithmetic/geometry and tests only whether compile-time constants, simpler addressing, and a fixed 10-block K loop can beat generic `qmv_fast_impl`.**
