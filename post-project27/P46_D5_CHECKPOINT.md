# P46 D5 checkpoint — native M5, Q8 T5 2+2+1, and selective fixed shapes

**Hardware:** Apple M1 Max, 64GB unified memory, 32-core GPU  
**Target:** Qwen3.8-27B Q6 affine G64 body with Q8/G64 target lm_head  
**Drafter:** canonical Qwen3.8-27B MTP build  
**Requested speculative block size:** 5  
**Actual drafted tokens per round:** 4  
**Generated tokens:** 512  
**Canonical D5 rounds:** 128  
**Canonical D5 jury decisions:** 512  
**Canonical text hash:** `e39b478ae4a8`  
**Canonical D5 trajectory hash:** `183cd3043746`

P46 is a width-5 optimization campaign. It does **not** replace the overall P45B2B D4 champion, which remains the fastest certified configuration at **29.316537 tok/s**. The purpose of P46 is to reduce the much higher per-round cost of D5 enough to exploit D5's higher commits-per-round.

The certified P46 D5 checkpoint reaches **26.550819 tok/s** in the final 10-pair certification below.

---

## 1. Width economics

With this wrapper, requested block size and actual drafted-token count differ by one:

```text
D4 requested width -> 3 actual drafted tokens
D5 requested width -> 4 actual drafted tokens
D6 requested width -> 5 actual drafted tokens
```

Canonical behavior:

```text
D4:
  rounds:          138
  jury decisions:  414
  commits/round:   3.7101
  draft acceptance: 373/414 = 90.0966%

D5:
  rounds:          128
  jury decisions:  512
  commits/round:   4.0000
```

D5 therefore improves commits per target round by about 7.8% and saves 10 target forwards over 512 generated tokens, but performs about 23.7% more draft/jury work.

Using the certified D4 champion throughput of 29.316537 tok/s, the D5 round-latency break-even point is approximately:

```text
136.442 ms/round
```

The final P46 checkpoint is still above that threshold, so D4 remains the short-context champion.

---

## 2. D5 progression

The D5 line advanced approximately as follows:

```text
naive D5                              ~16.65 tok/s
compiled VERIFY_LEN=5 + P36 T5 attn  ~20.82 tok/s
native shared-five Q6 M5             ~25.75 tok/s
selective N=48 generic fallback      ~25.81 tok/s
fixed M5 major shapes                ~26.00 tok/s
Q8/G64 T5 head 2+2+1                 26.514193 tok/s certified
selective extra fixed shapes         26.550819 tok/s certified
```

The large gains came from making M=5 a native first-class kernel shape and eliminating a verifier-head resource cliff. The final fixed-shape result is intentionally modest; by that point the campaign was already deep into diminishing returns.

---

## 3. Native FAST_M5

P46 added a native shared-five Q6/G64 affine path rather than constructing M=5 work from smaller launches.

Core additions included:

- `qdot_q6_8_m4_shared`
- `qdot_q6_8_m5_shared`
- `qmv_fast_m4_impl`
- `qmv_fast_m5_impl`
- `affine_qmv_fast_m4`
- `affine_qmv_fast_m5`
- Metal instantiation for FP16 / G64 / Q6 / batch 0
- host dispatch through `MLX_QMV_FAST_M5`

The native path was bitwise exact across the production Q6/G64 shapes used by the model.

Native M5 versus the older split construction produced a large scout win:

```text
split mean:   21.138749 tok/s
native mean:  25.746575 tok/s
paired mean: +21.798%
median:      +21.822%
wins:        3/3
```

This established native M5 as the required base for further D5 work.

---

## 4. N=48 generic fallback

A selective fallback keeps the small N=48 geometry on the generic path:

```text
MLX_QMV_FAST_M5_SKIP_N48=1
```

Scout result:

```text
native mean:  ~25.7722 tok/s
skip-N48:     ~25.8057 tok/s
paired mean:  +0.1299%
median:       +0.0968%
wins:          3/3
```

This remained part of the common P46 stack.

---

## 5. Fixed M5 major shapes

The first fixed-shape specialization covered four large production geometries:

```text
5120  -> 17408   gate
17408 -> 5120    down
5120  -> 10240   GDN qkv
5120  -> 12288   attention q
```

The fixed kernels preserve the exact Q6 RAWX FP32 load and arithmetic order while baking K and N into the Metal specialization.

Parity:

```text
8/8 bitwise exact
maximum absolute error: 0
```

Three-pair scout:

```text
dynamic mean: 25.746150 tok/s
fixed mean:   25.998312 tok/s
paired mean:  +0.9796%
median:       +0.9718%
wins:          3/3
```

This was a meaningful body-kernel gain and became the P46 fixed-M5 base.

---

## 6. Q8/G64 target verifier head: the T=5 resource cliff

The target verifier already used a custom quantized argmax path rather than a materialized full Q8 lm_head projection. Profiling showed a sharp width cliff:

```text
T=1  ~4.02 ms
T=2  ~3.98 ms
T=3  ~4.93 ms
T=4  ~4.76 ms
T=5  ~7.60 ms
T=6  ~8.11 ms
T=7  ~9.16 ms
T=8 ~10.25 ms
```

The T=5 kernel carried five simultaneous row bundles (`result[T][4]`, activation row state, and accumulators). The working hypothesis was a register/resource occupancy cliff at five live rows.

A chunk sweep tested equivalent exact decompositions:

```text
native
3+2
2+3
2+2+1
1+1+1+1+1
```

All candidates were bitwise exact in parity testing. The result was decisive:

```text
native median:  6.938291 ms
3+2:            6.608125 ms   -4.759%
2+3:            6.678500 ms   -3.744%
2+2+1:          4.421125 ms  -36.279%
1x5:            4.433500 ms  -36.101%
```

The 2+2+1 decomposition removed the resource cliff while avoiding the overhead of five single-row launches.

### 10-pair certification

```text
native-head mean: 26.050966 tok/s
2+2+1 mean:       26.514193 tok/s
paired mean:      +1.7782%
paired median:    +1.7866%
wins:             10/10
native round:      153.545 ms
2+2+1 round:       150.863 ms
saved/round:        2.683 ms
```

Every run preserved:

```text
text:   e39b478ae4a8
traj:   183cd3043746
rounds: 128
jury:   512
```

This is the largest certified P46 optimization after native M5 itself.

---

## 7. Negative experiment: P37-8 attention geometry

P36 T5 attention remained the control. A P37-8 variant reduced the physical threadgroup from 1024 threads / 32 SIMDgroups to 256 threads / 8 physical SIMDgroups while preserving virtual bookkeeping.

Corrected scout:

```text
P36 mean:   26.088398 tok/s
P37-8 mean: 26.052862 tok/s
paired:     -0.1361%
median:     -0.0968%
wins:        0/3
```

P37-8 was rejected. P36 remains the T5 attention implementation in the certified stack.

---

## 8. Negative experiment: Q6 M5 CHUNK2

The Q8 T=5 result suggested that reducing simultaneous live rows might help Q6 M5 as well. CHUNK2 therefore tried processing the five activation rows in smaller groups while reusing decoded Q6 pack/meta state.

Parity was perfect:

```text
24/24 bitwise exact
```

But the 10-pair per-shape microbench did not reproduce the Q8 benefit:

```text
5120 -> 17408  ~noise
17408 -> 5120   regression
5120 -> 10240  ~noise
5120 -> 12288   regression
```

Approximate weighted effect was about `-0.256 ms` per target forward.

Conclusion: the repeated Q6 decode cost erased the benefit from reducing live activation state. CHUNK2 was rejected and the source was restored before subsequent work.

---

## 9. Extra fixed-shape search

Three remaining production geometries were specialized:

```text
5120 -> 6144
6144 -> 5120
5120 -> 1024
```

Initial exactness battery:

```text
18/18 bitwise exact
maximum absolute error: 0
```

Direct same-process 10-pair microbench suggested:

```text
5120 -> 6144  clearly positive
6144 -> 5120  clearly positive
5120 -> 1024  small / noisy positive
```

However, enabling all three shapes failed end-to-end certification:

```text
paired mean:   -0.0570%
paired median: -0.0496%
wins:           4/10
saved/round:   -0.089 ms
```

All behavior remained exact, but the speed gate failed.

The `5120 -> 1024` specialization was then explicitly skipped, leaving only:

```text
5120 -> 6144
6144 -> 5120
```

A three-pair scout became clearly positive:

```text
paired mean:   +0.3590%
paired median: +0.3860%
wins:           2/3
saved/round:   +0.551 ms
```

### Final 10-pair certification

The two-shape candidate then passed the full promotion gate:

```text
baseline mean:      26.524079 tok/s
candidate mean:     26.550819 tok/s
paired mean:        +0.1008%
paired median:      +0.1230%
wins:                8/10
baseline round:      150.806 ms
candidate round:     150.654 ms
saved/round:          0.152 ms
```

Every measured run preserved:

```text
text:   e39b478ae4a8
traj:   183cd3043746
rounds: 128
jury:   512
```

The isolated `5120 -> 1024` microbench therefore did not justify enabling that specialization in production. This is another example of a small local kernel gain failing to survive end-to-end scheduling and launch effects.

---

## 10. Certified P46 D5 checkpoint

The certified D5 stack at this checkpoint is:

```text
body:
  Q6 affine G64
  native FAST_M5
  N=48 generic fallback
  fixed M5 major shapes
  extra fixed 5120 -> 6144
  extra fixed 6144 -> 5120
  fixed 5120 -> 1024 explicitly disabled

target verifier:
  VERIFY_LEN=5
  P36 T5 attention
  Q8/G64 target head 2+2+1

draft-side token policy:
  P45B2B Q4/G64 approximate full-vocab head
  deterministic top-4 reducer
  GPU confidence bypass
  exact Q6/G32 x4 fallback
```

Runtime environment for the final body selection includes:

```text
MLX_QMV_FAST_M5=1
MLX_QMV_FAST_M5_SKIP_N48=1
MLX_QMV_FAST_M5_FIXED=1
MLX_QMV_FAST_M5_FIXED_EXTRA=1
MLX_QMV_FAST_M5_FIXED_EXTRA_SKIP_1024=1
```

Headline result:

```text
P46 D5 certified mean: 26.550819 tok/s
round latency:          150.654 ms
D5 break-even:          136.442 ms
vs D4 champion:          -9.43%
throughput still needed: +10.42%
```

P45B2B D4 therefore remains the overall short-context champion at **29.316537 tok/s**.

---

## 11. Lessons and next target

P46 reinforced several recurring rules:

1. **Native width support matters.** Constructing M=5 from smaller kernels left a very large amount of performance on the table.
2. **Resource cliffs can dominate arithmetic cost.** The Q8 T=5 2+2+1 change produced a ~36% kernel-level improvement with identical math.
3. **A lesson from one kernel family does not automatically transfer to another.** Q8 row chunking was excellent; Q6 CHUNK2 was not.
4. **Small fixed-shape microbench wins still require end-to-end certification.** `5120 -> 1024` looked mildly positive alone and hurt the full candidate.
5. **Paired alternating tests remain essential.** Absolute machine speed drifted between sessions, while paired deltas remained interpretable.

The next high-leverage target is no longer another tiny M5 geometry specialization. D5 performs **512 jury decisions**, versus **414** for D4. The next structural experiment should therefore fuse more of the draft-side token-decision path:

```text
hierarchical reducer
 -> confidence decision
 -> conditional exact Q6/G32 fallback
```

Reducing dispatch and intermediate traffic in that path attacks a cost that scales directly with D5's extra draft/jury work.

Longer term, D4 and D5 should be benchmarked across context length. D5 saves target forwards and may become favorable once full-attention target-round cost rises sufficiently with context. A final runtime can therefore use different speculative widths and kernels by context regime rather than forcing one width globally.

---

## 12. Source-state note

The active custom MLX work was performed on the local `project46-fast-m5-scout` branch in `~/src/mlx-m1-qmv`.

The public repository records the benchmark logic and certified result. A separate local source-tree freeze/tag should be made before the P46 branch is mutated for the next structural experiment; this document does not claim a local P46 tag that has not been explicitly created and verified.
