# P44B2 certification: Q4/G32 search head at 29 tok/s

**Status:** certified robust champion  
**Predecessor:** P44B1 Q5/G64 search + exact Q6/G32 jury  
**Candidate:** P44B2 Q4/G32 search + exact Q6/G32 jury  
**Hardware:** Apple M1 Max, 64GB unified memory, 32-core GPU  
**Target:** Qwen3.8-27B Q6 affine group-64 with Q8/G64 target lm_head  
**Canonical generated tokens:** 512  
**Canonical rounds:** 138  
**Canonical text hash:** `e39b478ae4a8`  
**Canonical trajectory hash:** `f7801569fbdd`

P44B2 is the first certified 29 tok/s-class configuration in the post-Project27 tuning line.

---

## 1. Change under test

P44B2 changes only the approximate full-vocabulary draft-side search head.

### P44B1 baseline

```text
Q5/G64
weight: (248320, 800)
scales: (248320, 80)
biases: (248320, 80)
```

### P44B2 candidate

```text
Q4/G32
weight: (248320, 640)
scales: (248320, 160)
biases: (248320, 160)
```

Everything after the approximate projection is unchanged:

```text
approximate full-vocab search
 -> deterministic P43C hierarchical Metal top4
 -> exact gathered Q6/G32 x4 jury
 -> lowest-vocabulary-ID exact tie semantics
```

The exact Q6/G32 oracle remains:

```text
weight: (248320, 960)
scales: (248320, 160)
biases: (248320, 160)
```

The candidate therefore reduces approximate search-head weight traffic while retaining an exact final token decision.

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

Observed Q6-winner rank under Q4/G32:

```text
rank1: 3800
rank2:   81
rank3:    5
```

Recall:

```text
top1: 3800/3886 = 97.786927%
top2: 3881/3886 = 99.871333%
top4: 3886/3886 = 100.000000%
top8: 3886/3886 = 100.000000%
```

Exact gathered-jury result:

```text
exact Q6/G32 decisions: 3886/3886
actual jury failures:   0
maximum observed rank:  3
```

The five observed rank-3 states were:

```text
reasoning-ctx256   call=12   margin=0.015625  Q6=21218
reasoning-ctx256   call=91   margin=0.046875  Q6=31084
dialogue-ctx1024   call=37   margin=0.015625  Q6=16
reasoning-ctx1024  call=147  margin=0.156250  Q6=23275
prose-ctx8192      call=73   margin=0.156250  Q6=314
```

Top-4 therefore remained sufficient across every observed real state; widening the shortlist was not required.

**Robustness gate: PASS.**

---

## 3. Canonical live exactness

The direct benchmark compared the certified P44B1 Q5/G64 search head against P44B2 Q4/G32 inside one persistent process with alternating run order.

Warm / post-JIT candidate behavior:

```text
P44B2 warm:   29.068 tok/s
P44B2 trace:  29.062 tok/s
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

| Pair | P44B1 Q5/G64 | P44B2 Q4/G32 | Delta | Paired % |
|---:|---:|---:|---:|---:|
| 1 | 28.962 | 29.044 | +0.082 | +0.282% |
| 2 | 28.949 | 29.057 | +0.109 | +0.375% |
| 3 | 28.896 | 29.059 | +0.163 | +0.566% |
| 4 | 28.946 | 29.050 | +0.104 | +0.360% |
| 5 | 28.911 | 29.037 | +0.126 | +0.436% |
| 6 | 28.950 | 29.077 | +0.127 | +0.439% |
| 7 | 28.914 | 29.051 | +0.137 | +0.473% |
| 8 | 28.960 | 28.986 | +0.026 | +0.089% |
| 9 | 28.918 | 29.037 | +0.119 | +0.410% |
| 10 | 28.899 | 29.047 | +0.148 | +0.512% |

Summary:

```text
P44B1 mean: 28.930412 tok/s
P44B2 mean: 29.044437 tok/s

mean paired speedup:   +0.3942%
median paired speedup: +0.4231%
pair wins:             10/10
```

The control ran slightly below its earlier P44B1 certification-session mean, which is why the same-process alternating paired comparison is the promotion metric. P44B2 won every pair.

Working promotion gate:

```text
mean paired > +0.03%   PASS
median paired > 0      PASS
wins >= 6/10           PASS
exact behavior         PASS
```

**Promotion speed gate: PASS.**

---

## 5. Certification verdict

P44B2 is promoted as the new robust champion.

```text
current direct paired-session mean: 29.044437 tok/s
robustness:                          3886/3886 exact
canonical rounds:                    138
canonical text hash:                 e39b478ae4a8
canonical trajectory hash:           f7801569fbdd
canonical jury calls:                414
```

The local custom-source freeze is intended to be tagged:

```text
project44b2-q4g32-certified-29.04
```

P44B1 remains preserved as the predecessor rather than overwritten.

---

## 6. Why P44B2 matters

P44B1 established that reducing quantization metadata in the approximate search head produced a real end-to-end gain. P44B2 extends the same bandwidth strategy into the packed weight payload itself.

The important result is not only that Q4/G32 is faster. It is that the lower-bit search remains close enough to the exact Q6/G32 ordering that a four-token shortlist still recovers every observed exact winner.

That keeps the architecture simple:

```text
Q4/G32 full-vocabulary search
 -> top4
 -> exact Q6/G32 x4 jury
```

No larger shortlist and no hidden-state approximation are required.

---

## 7. Next experiment

The next bandwidth candidate is:

```text
P44B3: Q4/G64 search
 -> deterministic top4
 -> exact Q6/G32 jury
```

P44B3 should first pass the same 30-case / 3,886-decision robustness battery before any speed promotion. The existing P44B2 champion remains frozen unless the candidate passes both exactness and direct paired speed gates.
