# P44B3 certification: Q4/G64 search head at 29.20 tok/s

**Status:** certified robust champion  
**Predecessor:** P44B2 Q4/G32 search + exact Q6/G32 jury  
**Candidate:** P44B3 Q4/G64 search + exact Q6/G32 jury  
**Hardware:** Apple M1 Max, 64GB unified memory, 32-core GPU  
**Target:** Qwen3.8-27B Q6 affine group-64 with Q8/G64 target lm_head  
**Canonical generated tokens:** 512  
**Canonical rounds:** 138  
**Canonical text hash:** `e39b478ae4a8`  
**Canonical trajectory hash:** `f7801569fbdd`

P44B3 continues the output-head bandwidth line by changing only the approximate search-head group size from 32 to 64 at 4 bits.

---

## 1. Change under test

### P44B2 baseline

```text
Q4/G32
weight: (248320, 640)
scales: (248320, 160)
biases: (248320, 160)
```

### P44B3 candidate

```text
Q4/G64
weight: (248320, 640)
scales: (248320, 80)
biases: (248320, 80)
```

The packed 4-bit weight payload is unchanged. P44B3 halves the scale/bias metadata for the approximate full-vocabulary search head.

Everything after the approximate projection remains unchanged:

```text
Q4 approximate full-vocab search
 -> deterministic P43C hierarchical Metal top4
 -> exact gathered Q6/G32 x4 jury
 -> lowest-vocabulary-ID exact tie semantics
```

The exact Q6/G32 oracle remains:

```text
weight: (248320, 960)
scales: (248320, 160)
biases: (248320, 160)
group size: 32
bits: 6
```

This makes the direct P44B2/P44B3 benchmark a tightly controlled group-geometry / metadata-bandwidth experiment.

---

## 2. Distributional robustness gate

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

Observed Q6-winner rank under Q4/G64:

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
top8: 3886/3886 = 100.000000%
```

Exact gathered-jury result:

```text
exact Q6/G32 decisions: 3886/3886
actual jury failures:   0
maximum observed rank:  4
```

The single rank-4 state was:

```text
prose-ctx8192
call=73
rank=4
margin=0.296875
Q6 winner=314
jury winner=314
exact=True
```

Observed rank-3 states:

```text
reasoning-ctx1024   call=98   margin=0.062500  Q6=89653
structured-ctx8192  call=99   margin=0.109375  Q6=2972
reasoning-ctx256    call=91   margin=0.140625  Q6=31084
dialogue-ctx16384   call=131  margin=0.156250  Q6=5439
dialogue-ctx1024    call=35   margin=0.171875  Q6=303
reasoning-ctx8192   call=12   margin=0.265625  Q6=8286
dialogue-ctx8192    call=138  margin=0.281250  Q6=557
structured-ctx64    call=30   margin=0.296875  Q6=5939
reasoning-ctx8192   call=140  margin=0.359375  Q6=72
```

P44B3 therefore uses the full observed top-4 repair envelope. Top-4 remains sufficient, but there is no observed slack to reduce the shortlist below four candidates.

**Robustness gate: PASS.**

---

## 3. Canonical live exactness

The direct benchmark compared the certified P44B2 Q4/G32 search head against P44B3 Q4/G64 inside one persistent process with alternating run order.

Warm / post-JIT behavior:

```text
P44B2 warm:   29.030 tok/s
P44B3 warm:   29.217 tok/s
P44B2 trace:  29.064 tok/s
P44B3 trace:  29.212 tok/s
```

Every timed candidate and control run preserved:

```text
rounds:          138
text hash:       e39b478ae4a8
trajectory hash: f7801569fbdd
jury calls:      414
```

The accepted/drafted arrays also matched the canonical reference exactly.

**Behavioral certification: PASS.**

---

## 4. Direct 10-pair speed benchmark

Alternating AB/BA pair results:

| Pair | P44B2 Q4/G32 | P44B3 Q4/G64 | Delta tok/s | Paired % |
|---:|---:|---:|---:|---:|
| 1 | 29.078 | 29.256 | +0.177 | +0.610% |
| 2 | 29.055 | 29.190 | +0.135 | +0.465% |
| 3 | 29.070 | 29.188 | +0.118 | +0.405% |
| 4 | 29.056 | 29.171 | +0.115 | +0.396% |
| 5 | 29.036 | 29.213 | +0.177 | +0.609% |
| 6 | 29.086 | 29.186 | +0.100 | +0.343% |
| 7 | 29.071 | 29.242 | +0.171 | +0.589% |
| 8 | 29.053 | 29.170 | +0.118 | +0.406% |
| 9 | 29.029 | 29.212 | +0.183 | +0.631% |
| 10 | 29.041 | 29.186 | +0.146 | +0.501% |

Summary:

```text
P44B2 mean: 29.057401 tok/s
P44B3 mean: 29.201393 tok/s

mean paired speedup:   +0.4956%
median paired speedup: +0.4830%
pair wins:             10/10
```

Every pair favored P44B3.

Working promotion gate:

```text
mean paired > +0.03%   PASS
median paired > 0      PASS
wins >= 6/10           PASS
exact behavior         PASS
```

**Promotion speed gate: PASS.**

---

## 5. Bandwidth ladder

The controlled search-head bandwidth line now contains three consecutive paired wins:

```text
Q5/G32 -> Q5/G64   +0.4452%
Q5/G64 -> Q4/G32   +0.3942%
Q4/G32 -> Q4/G64   +0.4956%
```

The three changes are not identical in kernel behavior, but together they strongly support the working hypothesis that draft-side output-head memory traffic is a meaningful end-to-end bottleneck on this M1 Max configuration.

P44B3 is especially clean as a geometry experiment because both sides use:

```text
bits: 4
packed weight shape: (248320, 640)
top-k: 4
exact jury: Q6/G32 x4
```

The primary changed payload is quantization metadata:

```text
Q4/G32 scales/biases: (248320, 160)
Q4/G64 scales/biases: (248320, 80)
```

---

## 6. Certification verdict

P44B3 satisfies every current promotion requirement:

```text
broad robustness:      PASS
canonical text:        PASS
canonical trajectory:  PASS
accepted/drafted arrays: PASS
jury call count:       PASS
paired mean:           PASS
paired median:         PASS
pair wins:             PASS
```

**Certified direct paired-session mean: 29.201393 tok/s.**

P44B3 becomes the current robust champion.

---

## 7. Next experiment

The next controlled bandwidth candidate is Q4/G128.

Rationale:

- preserves the 4-bit packed weight payload
- halves scale/bias metadata again relative to Q4/G64
- isolates group geometry before attempting lower bit width

However, P44B3 already produced one real rank-4 state. Q4/G128 therefore has less robustness headroom than earlier candidates.

Required order:

```text
10-case rank scout
 -> full 3886-state robustness battery if top4 survives
 -> widen shortlist only if needed and still economically plausible
 -> direct alternating paired speed benchmark only after exact recovery
```

A lower-bit search head such as Q3 should remain a later challenger rather than the immediate next step.

---

## 8. Local artifacts

The benchmark and robustness results were produced from local harnesses named:

```text
/tmp/p44b3-q4g64-scout.py
/tmp/p44b3-q4g64-scout.log
/tmp/p44b3-q4g64-scout-results.json
/tmp/p44b3-q4g64-full-battery.py
/tmp/p44b3-q4g64-full-battery.log
/tmp/p44b3-q4g64-robustness-results.json
/tmp/p44b3-direct-q4g32-vs-q4g64.py
/tmp/p44b3-direct-q4g32-vs-q4g64.log
```

The local source-tree snapshot/tag should be recorded separately once the corresponding freeze is confirmed. This GitHub document does not claim an unverified local source-tree commit or tag.
