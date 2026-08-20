# P47-P50 search-head checkpoint — reducer, bandwidth, lower-bit, and Q4 geometry experiments

**Status:** pause / detour checkpoint  
**Date:** 2026-08-20  
**Active local MLX branch during experiments:** `project48-search-head-bandwidth`  
**Certified D5 control:** P46 D5 at **26.550819 tok/s**  
**Overall certified champion:** P45B2B D4 at **29.316537 tok/s**

This document records the post-P46 search-head campaign through P50B before a planned detour. None of the P47-P50 candidates below was promoted into the certified MLX source tree. The P46 source freeze remains the control:

```text
repository: skylinej3o1/mlx
commit:     22b52dfa9192ccb76a2c0e84d6e2f1383505023f
tag:        project46-d5-certified-26.55
snapshot:   champion_snapshots/p46-d5-fixed2-certified/
```

The local source tree remained clean during these `/tmp` harness experiments.

---

## 1. Control and benchmark discipline

P46 D5 canonical behavior remains:

```text
text hash:       e39b478ae4a8
trajectory hash: 183cd3043746
rounds:          128
jury decisions:  512
round latency:   150.654 ms
throughput:      26.550819 tok/s
```

P45B2B D4 remains the overall throughput champion:

```text
throughput:      29.316537 tok/s
rounds:          138
jury decisions:  414
```

Small effects continue to require same-process alternating benchmarks. Three pairs are scout-only; promotion normally requires at least 10 paired runs, positive paired mean and median, at least 6/10 wins, and exact behavior.

---

## 2. P47 — reducer / dispatch experiments closed

### P47A stage2 + conditional Q6 fusion

A fused stage2 reducer and conditional Q6 jury was exact but slower end-to-end:

```text
3-pair paired mean:   -0.1535%
paired median:        -0.1352%
wins:                  0/3
saved/round:          -0.231 ms
exactness:             PASS
```

Interpretation: extra code footprint / register pressure inflated the stage2 path enough to erase the dispatch reduction.

### Stage2 threadgroup sweep

```text
TG32:  paired mean -0.0116%, median -0.0125%, wins 1/3
TG128: paired mean +0.0058%, median -0.0143%, wins 1/3
```

TG64 initially looked promising in a 3-pair scout:

```text
paired mean:   +0.2508%
median:        +0.4286%
wins:           2/3
saved/round:   +0.376 ms
```

But the required 10-pair certification rejected it:

```text
dynamic mean: 26.562181 tok/s
fixed mean:   26.559247 tok/s
paired mean:  -0.0109%
median:       +0.0030%
wins:          6/10
saved/round:  -0.017 ms
exactness:     PASS
```

**P47 verdict:** closed. Reducer/dispatch geometry is considered exhausted for now.

---

## 3. P48 — Q4/G128 bandwidth experiment closed

### P48A Q4/G128 full robustness

Across the same 30-case / 3,886-state corpus:

```text
rank1: 3781/3886 = 97.297993%
top2:  3881/3886 = 99.871333%
top4:  3885/3886 = 99.974267%
top5:  3886/3886 = 100.000000%
max rank: 5
```

The sole rank-5 state was `reasoning-ctx1024`, call 98.

Safe confidence policy remained possible:

```text
max wrong-top1 gap: 0.671875
strict bypass rule: gap > 0.671875
safe bypass:        3385/3886 = 87.108%
wrong bypass:       0
```

Approximate parameter stream fell from roughly 715 MB at Q4/G64 to roughly 675 MB at Q4/G128, a total reduction of only about 5.5% because the packed weight payload is unchanged.

### P48B actual top5 + exact Q6x5

The deterministic Metal top5 reducer plus exact gathered Q6/G32 x5 recovered all 3,886 states exactly.

### P48C live G64/top4/x4 vs G128/top5/x5

The live candidate regressed clearly:

```text
G64/top4/x4 mean:  26.579383 tok/s
G128/top5/x5 mean: 26.475915 tok/s
paired mean:       -0.3892%
median:            -0.4338%
wins:               0/3
round latency:      150.493 -> 151.081 ms
saved/round:        -0.588 ms
exactness:          PASS
```

### P48D head-only G64 vs G128, both top4/x4

This isolated the G128 projection itself:

```text
G64 mean:     26.576811 tok/s
G128 mean:    26.571179 tok/s
paired mean:  -0.0213%
median:       +0.0547%
wins:          2/3
saved/round:  -0.032 ms
```

**P48 verdict:** closed. G128 itself is effectively flat; most of P48C's loss came from the wider top5/x5 repair tax.

---

## 4. P49 — lower-bit search heads closed

### P49A Q3/G64 robustness

Q3/G64 was unexpectedly robust:

```text
rank1: 3719/3886 = 95.702522%
top2:  3865/3886 = 99.459599%
top4:  3885/3886 = 99.974267%
top8:  3886/3886 = 100.000000%
max rank: 6
```

Safe confidence required a much larger threshold:

```text
max wrong-top1 gap: 1.390625
strict bypass:      gap > 1.390625
safe bypass:        2925/3886 = 75.270%
```

### P49B Q3/G64 head microbenchmark

Despite about 22.22% lower approximate parameter traffic, native MLX Q3 was slower than Q4:

```text
Q4 mean:        2.895722 ms
Q3 mean:        3.103388 ms
paired speedup: -6.0632%
median:         -5.1672%
wins:            3/10
gross D5 effect: -0.830666 ms/round
```

The frozen MLX source explains the result: 3-bit packing crosses byte boundaries and requires substantially more mask / reconstruction work than the simple power-of-two Q4 path.

**Q3 verdict:** closed.

### P49C Q2/G64 head microbenchmark

Q2 was the first lower-bit candidate with a real speed win:

```text
Q4 mean:        3.121500 ms
Q2 mean:        2.649028 ms
paired speedup: +18.8004%
median:         +17.2529%
wins:            8/10
saving/head:    +0.472473 ms
gross D5 save:  +1.889890 ms/round
```

Approximate parameter stream:

```text
Q4/G64: 715.2 MB
Q2/G64: 397.3 MB
reduction: 44.44%
```

### P49D Q2/G64 full robustness

Q2 remained surprisingly close to the exact Q6 winner:

```text
rank histogram:
  1: 3467
  2:  309
  3:   68
  4:   25
  5:    7
  6:    6
  7:    1
  8:    1
  9:    2

top4: 3869/3886 = 99.562532%
top5: 3876/3886 = 99.742666%
top8: 3884/3886 = 99.948533%
top9: 3886/3886 = 100.000000%
max rank: 9
```

However, the safe confidence threshold became expensive:

```text
max wrong-top1 gap: 2.750000
strict bypass:      gap > 2.750000
safe bypass:        2181/3886 = 56.125%
fallback:           1705/3886 = 43.875%
wrong bypass:       0
```

This motivated a staged repair concept:

```text
Q2 full-vocabulary search
 -> confidence bypass when safe
 -> Q2 shortlist
 -> Q4 shortlist rerank
 -> Q4 top4
 -> exact Q6/G32 x4
```

### Repair-cost experiments

P49E generic Q4x16 + generic Q6x4:

```text
Q4x16: 0.475493 ms
Q6x4:  0.834068 ms
sum:   1.309561 ms
rough net before shortlist reducer: +0.036311 ms/D5 round
```

P49F1 generic Q4x16 + certified direct-row Q6x4:

```text
Q4x16: 0.469519 ms
Q6x4:  0.698576 ms
sum:   1.168095 ms
rough net before shortlist reducer: +0.212343 ms/D5 round
```

P49F2 direct-row custom Q4x16 + direct Q6x4 was bitwise exact but slower:

```text
Q4x16: 0.603436 ms
Q6x4:  0.670810 ms
sum:   1.274245 ms
rough net: +0.011242 ms/D5 round
```

P49F2B two 64-thread groups was also exact and no better:

```text
Q4x16: 0.621497 ms
Q6x4:  0.661313 ms
rough net: -0.008853 ms/D5 round
```

P49G reduced the shortlist to 12 candidates, still safely above the observed rank-9 envelope:

```text
Q4x12: 0.541889 ms
Q6x4:  0.716483 ms
sum:   1.258372 ms
rough net before shortlist reducer: +0.063452 ms/D5 round
```

That remaining margin is below benchmark noise and does not include the shortlist reducer / dispatch overhead.

**P49 verdict:** closed. Q2 proves that the full-vocabulary search head is bandwidth-sensitive, but its 43.875% exact-repair frequency consumes the raw 18.8% projection win.

---

## 5. P50 — Q4/G64 full-head geometry closed

After lower-bit search heads failed economically, P50 returned to the surviving Q4/G64 head itself.

### P50A full-head geometry sweep

All tested custom geometries were bitwise exact against native `mx.quantized_matmul`:

```text
S1R8: bitwise exact
S2R4: bitwise exact
S4R2: bitwise exact
S8R1: bitwise exact
```

Five-pass scout means:

```text
native: 3.481347 ms
S1R8:   3.785594 ms   -8.037%
S2R4:   3.470414 ms   +0.315%
S4R2:   3.581755 ms   -2.803%
S8R1:   4.246861 ms  -18.025%
```

Only S2R4 was remotely competitive, which matches native MLX's existing geometry: two SIMDgroups with four rows per SIMDgroup.

### P50B isolated native vs fixed S2R4

The required 10-pair same-process test rejected the apparent scout win:

```text
native mean:       2.914103 ms
fixed S2R4 mean:   2.981191 ms
paired mean:       -3.5885%
paired median:     +6.4824%
wins:               6/10
mean latency save: -0.067088 ms/head
D5 gross save:     -0.268352 ms/round
```

The pair distribution was very noisy, but there was no repeatable benefit and the raw mean was slower.

**P50A/B verdict:** close SIMD geometry permutations. Native 2x4 is already the correct row/SIMD mapping.

---

## 6. What the campaign established

The post-P46 experiments narrow the remaining draft-head problem substantially:

1. Reducer / conditional-jury fusion is not automatically free; P47 showed code footprint and register pressure can erase dispatch savings.
2. Q4/G128 reduces metadata but not enough total traffic to matter; the extra repair width is more expensive than the bandwidth saved.
3. Q3 loses because unpack / bit-manipulation cost dominates its lower memory traffic.
4. Q2 proves that the Q4 search head is materially bandwidth-bound, but exactness repair is too frequent to monetize the raw projection gain.
5. The native Q4/G64 2-SIMD x 4-row geometry is already well chosen.

No P47-P50 candidate beats or replaces the certified P46 checkpoint.

---

## 7. Immediate resume point after the detour

The next experiment is **P50C: fixed-shape specialization of the surviving Q4/G64 full-vocabulary search head while preserving native 2x4 geometry exactly**.

Target shape:

```text
M = 1
K = 5120
N = 248320
bits = 4
group size = 64
SIMD geometry = 2 x 4
K blocks = 10 exactly
```

The frozen native `qmv_fast_impl` still accepts runtime `in_vec_size` / `out_vec_size`, computes row and stride expressions from those values, advances generic pointers through the K loop, and uses the generic templated Q4 path. P50C should test whether compile-time specialization of this one hot shape can reduce address arithmetic / loop overhead and allow better unrolling **without changing arithmetic order or row/SIMD geometry**.

Required order on resume:

```text
1. standalone fixed-shape Q4/G64 kernel
2. bitwise full-vector parity against native Q4/G64
3. isolated same-process native-vs-fixed microbenchmark
4. only if clearly positive: live P46 D5 3-pair scout
5. only if scout survives: 10-pair promotion test
```

Do not resume by trying more SIMD geometry permutations. P50A/B already answered that question.

---

## 8. Pause-state summary

```text
certified P46 D5:        26.550819 tok/s
certified P45B2B D4:     29.316537 tok/s

P47 reducer fusion:      CLOSED
P48 Q4/G128:             CLOSED
P49 Q3/G64:              CLOSED
P49 Q2/G64 staged repair:CLOSED
P50 geometry sweep:      CLOSED

NEXT:
P50C fixed-shape Q4/G64
K=5120, N=248320, M=1
preserve native S2R4 geometry
```

This is the intended resume point after the detour.
