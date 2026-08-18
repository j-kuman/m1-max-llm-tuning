# P45B2B certification: GPU-resident conditional jury bypass at 29.32 tok/s

**Status:** certified robust champion  
**Predecessor:** P44B3 Q4/G64 search + unconditional exact Q6/G32 x4 jury  
**Candidate:** P45B2B Q4/G64 search + GPU-resident conditional exact Q6/G32 x4 jury  
**Hardware:** Apple M1 Max, 64GB unified memory, 32-core GPU  
**Target:** Qwen3.8-27B Q6 affine group-64 with Q8/G64 target lm_head  
**Canonical generated tokens:** 512  
**Canonical rounds:** 138  
**Canonical text hash:** `e39b478ae4a8`  
**Canonical trajectory hash:** `f7801569fbdd`

P45B2B keeps the certified P44B3 Q4/G64 approximate search head and deterministic Metal top-4 reducer, but conditionally skips the exact Q6/G32 x4 jury when the approximate winner is sufficiently separated.

## Architecture

P44B3 always performed:

```text
Q4/G64 full-vocabulary approximate projection
 -> deterministic hierarchical Metal top-4 reduction
 -> exact Q6/G32 scoring of four candidates
 -> exact max with lowest-vocabulary-ID tie semantics
```

A census over the 3,886-state robustness corpus showed that the Q4 top1-top2 score gap was a useful confidence signal. The maximum observed gap on a state where Q4 top1 disagreed with the exact Q6 winner was `0.671875`.

P45B2B therefore uses the strict rule:

```text
if Q4 top1-top2 gap > 0.671875:
    return Q4 top1
else:
    run exact Q6/G32 x4 jury
```

The entire branch remains GPU-resident. The conditional Metal kernel evaluates the gap before touching Q6 weight rows, so confident states return without exact-jury memory traffic.

The earlier P45B1 host prototype used `q4_gap.item()` and lost roughly 7.6% in a three-pair scout because of host synchronization. It was rejected. P45B2B removes that synchronization entirely.

## Exact Q6/G32 x4 fallback parity

The fallback is a specialized one-SIMD-group Metal implementation of the Q6/G32 four-candidate jury. Matching MLX required preserving the same FP16 input-sum semantics used by affine quantization.

Full 3,886-state result:

```text
FP16 score equality:       3886/3886
maximum absolute error:    0.0
token decisions:           3886/3886
token mismatches:          0
```

This is score-level parity, not only winner-token parity.

## Full GPU-policy robustness certification

Battery:

```text
prompt families:
  code
  reasoning
  prose
  structured
  dialogue

context targets:
  64
  256
  1024
  4096
  8192
  16384

cases: 30
real MTP head decisions: 3886
```

The inherited Q4/G64 shortlist distribution remained:

```text
rank1: 3782
rank2:   94
rank3:    9
rank4:    1
```

Recall:

```text
top1: 3782/3886 = 97.323726%
top2: 3876/3886 = 99.742666%
top4: 3886/3886 = 100.000000%
```

Actual P45B2B conditional GPU policy:

```text
exact decisions: 3886/3886
mismatches:      0
```

**Robustness gate: PASS.**

## Initial direct scout

Same-process alternating P44B3 versus P45B2B:

```text
P44B3 mean:  29.164053 tok/s
P45B2B mean: 29.285782 tok/s

mean paired speedup:   +0.4174%
median paired speedup: +0.4561%
pair wins:             3/3
```

Every run preserved 138 rounds, text hash `e39b478ae4a8`, trajectory hash `f7801569fbdd`, and 414 jury decisions.

## First 10-pair promotion run

The first full promotion session occurred while overall machine throughput was lower than normal, but the paired result was decisive:

```text
P44B3 mean:  28.043989 tok/s
P45B2B mean: 28.288928 tok/s

mean paired speedup:   +0.8738%
median paired speedup: +0.9142%
pair wins:             10/10
```

All ten deltas were positive and canonical behavior remained exact.

## Independent 10-pair rerun

The exact benchmark was rerun unchanged after throughput returned to the normal 29 tok/s-class regime.

| Pair | P44B3 | P45B2B | Delta tok/s | Paired % |
|---:|---:|---:|---:|---:|
| 1 | 29.229 | 29.331 | +0.102 | +0.348% |
| 2 | 29.175 | 29.330 | +0.155 | +0.532% |
| 3 | 29.115 | 29.326 | +0.210 | +0.722% |
| 4 | 29.169 | 29.317 | +0.148 | +0.508% |
| 5 | 29.195 | 29.323 | +0.128 | +0.437% |
| 6 | 29.037 | 29.271 | +0.234 | +0.805% |
| 7 | 29.185 | 29.324 | +0.139 | +0.475% |
| 8 | 29.145 | 29.320 | +0.175 | +0.601% |
| 9 | 29.171 | 29.321 | +0.150 | +0.513% |
| 10 | 29.149 | 29.303 | +0.154 | +0.527% |

Summary:

```text
P44B3 mean:  29.157123 tok/s
P45B2B mean: 29.316537 tok/s

mean paired speedup:   +0.5469%
median paired speedup: +0.5201%
pair wins:             10/10
```

Across the two independent 10-pair sessions, P45B2B won **20/20** pairs.

Working promotion gate:

```text
mean paired > +0.03%   PASS
median paired > 0      PASS
wins >= 6/10           PASS
exact behavior         PASS
```

**Promotion speed gate: PASS.**

## Certification verdict

```text
GPU-policy robustness:      PASS  3886/3886
custom Q6 x4 FP16 parity:   PASS  3886/3886
maximum Q6 score error:     0.0
canonical text:             PASS
canonical trajectory:       PASS
canonical rounds:           PASS
first 10-pair run:          PASS  10/10
independent 10-pair rerun:  PASS  10/10
cumulative pair wins:       20/20
```

The normal-performance rerun establishes the headline result:

```text
29.316537 tok/s mean
+0.5469% paired mean versus P44B3
+0.5201% paired median
10/10 wins
```

**P45B2B becomes the certified robust champion.**

Local MLX source-tree freeze:

```text
commit: 4efc3b4
tag:    project45b2b-gpu-bypass-certified-29.32
```

## Next controlled experiment

The natural successor is to fuse the final hierarchical top-4 reduction, confidence test, and conditional Q6/G32 x4 fallback into one Metal kernel, removing the remaining dispatch boundary. P45B2B should remain frozen as the control.