# Qwen3.8-27B Q6 on M1 Max: Post-Project27 tuning log

**Scope:** continuation of the original Project27 campaign after the 27.305 tok/s milestone  
**Hardware:** Apple M1 Max, 64GB unified memory, 32-core GPU  
**Target:** Qwen3.8-27B Q6 affine group-64 with Q8/G64 lm_head  
**Current certified experimental champion:** **P44B1 Q5/G64 search + exact Q6/G32 jury**  
**Direct 10-pair candidate mean:** **28.953139 tok/s**  
**Canonical rounds:** **138**  
**Canonical text hash:** `e39b478ae4a8`  
**Canonical trajectory hash:** `f7801569fbdd`

This document continues [`PROJECT27_QWEN38_27B_M1_MAX_TUNING_LOG.md`](PROJECT27_QWEN38_27B_M1_MAX_TUNING_LOG.md). The original Project27 artifact bundle is preserved as a historical milestone rather than rewritten in place.

---

## 1. Benchmark and promotion discipline

Post-Project27 work tightened correctness requirements beyond matching final text.

A candidate is not promoted unless it preserves the canonical speculative behavior:

- 512 generated tokens
- 138 speculative rounds
- exact generated text hash `e39b478ae4a8`
- exact draft/accept trajectory hash `f7801569fbdd`
- exact accepted/drafted length sequence where directly compared

For small performance deltas, direct same-process alternating paired benchmarks are preferred over separate runs. The working promotion gate is:

```text
mean paired speedup   > +0.03%
median paired speedup > 0
pair wins             >= 6/10
canonical behavior    exact on every measured run
```

Candidates that change the final token decision are rejected even if final text happens to remain the same.

For shortlist-based approximate heads, broad robustness is a separate gate. The current battery contains 30 prompt/context cases:

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
```

The battery contains **3,886 real MTP draft-head decisions**.

---

## 2. Current architecture

The current line keeps native Qwen MTP with block size 4. The implementation therefore drafts three actual tokens per speculative round.

The post-Project27 stack combines:

- fixed-shape target-verifier optimizations
- exact custom target attention/cache work
- an exact Q6/G32 draft-side target-head oracle
- lower-bit full-vocabulary search heads
- deterministic Metal shortlist reduction
- exact gathered Q6/G32 jury recovery

The central draft-head pattern is:

```text
approximate full-vocabulary search head
        |
        v
deterministic top-k shortlist
        |
        v
exact Q6/G32 gathered scores
        |
        v
exact token decision
```

This allows aggressive output-head approximation without allowing the approximation to enter the hidden-state trajectory.

---

## 3. Major milestones after Project27

### P34 - fixed-shape clean baseline

Certified tag in the custom source tree:

```text
project34-fixed-shape-clean-28.056
```

Representative certification:

```text
mean:   28.0613518531 tok/s
median: 28.0721417108 tok/s
stdev:  0.0298346421
runs:   10/10 at exactly 138 rounds
```

This established the first stable ~28 tok/s post-Project27 baseline.

### P36 - whole M=4 target verifier compile / fixed cache

The target verifier was compiled across all 64 layers with fixed physical cache storage and tensor-backed logical length. A custom M4 attention path preserved exact standalone behavior.

Direct paired result versus the P34 reference:

```text
P34 mean: 27.9943
P36 mean: 28.0999
mean paired improvement: +0.377%
wins: 3/3
```

This became the verifier base for later work.

### P38E - Q6/G32 draft-side target head

The target Q8/G64 lm_head was dequantized and requantized to Q6/G32 for the MTP draft-side target-head decisions.

Geometry:

```text
Q8/G64 original:
  weight: (248320, 1280)
  scales: (248320, 80)
  biases: (248320, 80)

Q6/G32:
  weight: (248320, 960)
  scales: (248320, 160)
  biases: (248320, 160)
```

Direct 10-pair result:

```text
Q8 mean:     28.095826
Q6/G32 mean: 28.527238
mean paired: +1.5358%
median:      +1.5270%
wins:        10/10
```

This was a major draft-head win and established Q6/G32 as the exact oracle for subsequent output-head search experiments.

### P39 - internal low-bit approximations rejected

Q4/G32 and Q5/G32 experiments inside the MTP hidden path changed hidden-state trajectories. Even when final text remained unchanged, the trajectory gate rejected them.

Key lesson:

> Internal hidden approximation compounds. Output-head approximation is much safer because it terminates at an argmax/token decision that can be repaired exactly.

### P40 - compile attempts closed

Pre-FC and whole-M1 compile experiments were exact but did not produce a robust end-to-end win. Fixed-capacity and M1 scheduling taxes erased the isolated gains.

### P41 - Q5/G32 search + exact Q6 jury

A Q5/G32 full-vocabulary search head was built from the target Q8/G64 head.

Geometry:

```text
Q5/G32:
  weight: (248320, 800)
  scales: (248320, 160)
  biases: (248320, 160)
```

On the canonical 414 draft-head decisions, the Q6 winner ranked:

```text
rank 1 under Q5: 413
rank 2 under Q5:   1
```

This motivated a two-stage algorithm:

```text
Q5 full-vocab search
 -> shortlist
 -> exact gathered Q6 scores
 -> Q6 winner
```

P41E live 10-pair result:

```text
Q6/G32 mean: 28.546963
Q5+jury mean: 28.628993
mean paired:  +0.2874%
median:       +0.3235%
wins:         9/10
```

### P42 - custom hierarchical top-k and robustness work

A GPU-wide hierarchical Metal reducer replaced generic `mx.argpartition` shortlist selection.

The initial top-2 implementation was faster but failed the broad robustness requirement:

```text
3886 real decisions
Q6 winner rank under Q5:
  rank1: 3824
  rank2:   61
  rank3:    1
```

Top-2 therefore covered only 3885/3886 states.

P42R5 widened the shortlist to top-4 and added exact full-vocabulary tie semantics. It achieved:

```text
top4 recall: 3886/3886
exact jury:   3886/3886
failures:     0
max rank:     3
```

This became the robust line even though the raw top-2 implementation was slightly faster.

### P43C - reducer geometry tuning

P43C kept the exact P42R5 algorithm but reduced stage-1 reducer geometry from 256 groups to 64 groups at 256 threads/group.

Geometry:

```text
P42R5:
  groups:        256
  threads/group: 256
  TOTAL_THREADS: 65536
  stage2 NCAND:  1024

P43C:
  groups:        64
  threads/group: 256
  TOTAL_THREADS: 16384
  stage2 NCAND:  256
```

Direct 10-pair result:

```text
P42R5 mean: 28.784395
P43C mean:  28.802659
mean paired:   +0.0635%
median paired: +0.0749%
wins:          7/10
```

Broad robustness remained exact:

```text
3886/3886 exact Q6 decisions
failures: 0
```

Certified custom-source tag:

```text
project43c-g64-certified-28.820
```

### P44A - synchronization audit, closed

A source audit initially suggested that the fallback deferred greedy verifier could be projecting target head rows one at a time with repeated synchronization.

An instrumented runtime probe showed:

```text
_speculative_walk_deferred_greedy calls: 0
```

The active greedy Qwen3.8 path already uses the efficient fast path:

```text
verify hidden block
 -> speculative_argmax_from_hidden(hidden)
 -> custom multi-token quantized argmax
 -> asynchronous evaluation
 -> speculative acceptance walk
```

The apparent large verifier batching opportunity was therefore a dead end for the active configuration. No patch was promoted.

This is preserved as a useful negative result: inspect actual runtime dispatch before optimizing fallback code.

### P44B1 - Q5/G64 search head

P44B1 tests the bandwidth hypothesis by changing only the approximate search-head group size:

```text
P43C Q5/G32:
  weight: (248320, 800)
  scales: (248320, 160)
  biases: (248320, 160)

P44B1 Q5/G64:
  weight: (248320, 800)
  scales: (248320, 80)
  biases: (248320, 80)
```

The weight payload is unchanged; scale/bias metadata is halved.

Everything after the Q5 projection remains identical:

```text
P43C deterministic G64 top-4 reducer
 -> exact Q6/G32 x4 gathered jury
 -> full-vocabulary tie semantics
```

#### Robustness

Full 30-case battery:

```text
total real MTP head decisions: 3886

rank1: 3817
rank2:   66
rank3:    3

Q5/G64 top4 contains Q6/G32 winner: 3886/3886
exact Q6/G32 jury decision:          3886/3886
failures:                            0
max observed rank:                   3
```

#### Direct 10-pair benchmark

Measured candidate/control means:

```text
P43C Q5/G32 mean:  28.824832 tok/s
P44B1 Q5/G64 mean: 28.953139 tok/s
```

Pair-by-pair percentage deltas:

```text
+0.501%
+0.558%
+0.246%
+0.402%
+0.550%
+0.631%
+0.538%
+0.507%
-0.047%
+0.566%
```

Summary:

```text
mean paired speedup:   +0.4452%
median paired speedup: +0.5223%
pair wins:              9/10
behavioral gate:         PASS
promotion speed gate:    PASS
```

All measured candidate and control runs preserved:

```text
rounds:          138
text hash:       e39b478ae4a8
trajectory hash: f7801569fbdd
jury calls:      414
```

P44B1 is therefore the current certified experimental champion by benchmark and robustness criteria.

---

## 4. Current champion summary

```text
Target:
  Qwen3.8-27B Q6 affine G64
  Q8/G64 target lm_head

Drafter:
  native Qwen MTP
  block size 4
  3 drafted tokens per round

Draft-side token head:
  Q5/G64 full-vocab search
  deterministic hierarchical Metal top4
  exact gathered Q6/G32 x4 jury
  exact lowest-vocab-ID tie semantics

Canonical:
  512 generated tokens
  138 rounds
  text hash e39b478ae4a8
  trajectory hash f7801569fbdd
  414 draft-side jury calls

Performance:
  direct candidate mean 28.953139 tok/s
  vs P43C control +0.4452% paired mean
```

The plain target autoregressive baseline remains roughly 14.24 tok/s, so the current line is slightly above a 2x throughput improvement on the same M1 Max while preserving deterministic speculative behavior.

---

## 5. Key lessons from P34-P44

1. **Determinism must include the speculative trajectory.** Final text equality alone is not a sufficient promotion criterion.
2. **Broad robustness matters for shortlist methods.** A single rank-3 state invalidated top-2 despite excellent canonical behavior.
3. **Output-head approximation is unusually safe when followed by an exact jury.** Internal hidden approximations were not.
4. **Small isolated kernel wins frequently disappear end-to-end.** Serial and paired benchmarks are mandatory.
5. **Memory traffic remains a real lever.** P44B1 halved Q5 quant metadata and produced a clear ~0.45% E2E gain.
6. **Runtime dispatch must be measured, not inferred from source presence.** P44A's apparent serial verifier path was inactive.
7. **Preserve negative results.** They narrow the search space and prevent repeating expensive dead ends.

---

## 6. Next experiments

The P44B1 result validates continuing down the search-head bandwidth curve.

Priority candidates:

```text
P44B2: Q4/G32 search -> shortlist -> exact Q6/G32 jury
P44B3: Q4/G64 search -> shortlist -> exact Q6/G32 jury
```

For each candidate, measure Q6-winner recall at least through:

```text
top1
top2
top4
top8
top16
top32
```

No live-generation benchmark should be promoted until the final shortlist/jury path restores **3886/3886** exact Q6 decisions on the broad battery.

Architecture-changing challengers such as DSpark remain separate branches. The existing MTP champion is not replaced unless a challenger beats it under the same end-to-end and correctness discipline.

---

## 7. Source-tree and artifact notes

The active custom MLX/mlx-vlm development checkout during this work is:

```text
~/src/mlx-m1-qmv
```

Confirmed P43C source-tree milestone:

```text
commit: 4bcda12
 tag:   project43c-g64-certified-28.820
```

P44B1 benchmark and robustness records were produced from local harnesses named:

```text
/tmp/p44b1-direct-g32-vs-g64.py
/tmp/p44b1-direct-g32-vs-g64.log
/tmp/p44b1-q5g64-full-battery.py
/tmp/p44b1-q5g64-full-battery.log
/tmp/p44b1-q5g64-robustness-results.json
```

The Project27 artifact bundle in this repository remains historical and unchanged. Later implementation snapshots should be preserved separately rather than silently replacing the original Project27 files.
