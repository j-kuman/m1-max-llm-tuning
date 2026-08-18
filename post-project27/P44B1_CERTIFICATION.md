# P44B1 certification: Q5/G64 search head

**Candidate:** P44B1 Q5/G64 full-vocabulary search head  
**Control:** P43C Q5/G32 full-vocabulary search head  
**Unchanged after search projection:** P43C G64 deterministic top-4 reducer + exact Q6/G32 x4 jury  
**Generated tokens:** 512  
**Canonical rounds:** 138  
**Text hash:** `e39b478ae4a8`  
**Trajectory hash:** `f7801569fbdd`  
**Jury calls:** 414 per canonical generation

## Quant geometry

```text
P43C Q5/G32
  W: (248320, 800)
  S: (248320, 160)
  B: (248320, 160)

P44B1 Q5/G64
  W: (248320, 800)
  S: (248320, 80)
  B: (248320, 80)
```

The P44B1 change halves scale/bias metadata while leaving the 5-bit weight payload unchanged.

## Full robustness battery

```text
cases: 30
real MTP head decisions: 3886

rank1: 3817 / 3886
rank2-or-better: 3883 / 3886
rank4-or-better: 3886 / 3886
rank8-or-better: 3886 / 3886

rank histogram:
  1: 3817
  2: 66
  3: 3

max Q6-winner rank under Q5/G64: 3
exact P43C top4 -> Q6 jury: 3886 / 3886
failures: 0
```

Prompt families:

```text
code
reasoning
prose
structured
dialogue
```

Context targets:

```text
64
256
1024
4096
8192
16384
```

## Direct same-process 10-pair benchmark

Warm / post-JIT observations:

```text
stock warm:        28.635
P43C G32 warm:     28.817
P44B1 G64 warm:    28.960
P43C G32 trace:    28.844
P44B1 G64 trace:   28.967
```

Measured pairs:

| Pair | P43C G32 | P44B1 G64 | Delta tok/s | Delta % |
|---:|---:|---:|---:|---:|
| 1 | 28.840 | 28.984 | +0.145 | +0.501% |
| 2 | 28.806 | 28.966 | +0.161 | +0.558% |
| 3 | 28.869 | 28.940 | +0.071 | +0.246% |
| 4 | 28.841 | 28.957 | +0.116 | +0.402% |
| 5 | 28.812 | 28.970 | +0.159 | +0.550% |
| 6 | 28.803 | 28.984 | +0.182 | +0.631% |
| 7 | 28.806 | 28.961 | +0.155 | +0.538% |
| 8 | 28.831 | 28.977 | +0.146 | +0.507% |
| 9 | 28.817 | 28.804 | -0.014 | -0.047% |
| 10 | 28.824 | 28.987 | +0.163 | +0.566% |

Summary:

```text
P43C Q5/G32 mean:  28.824832 tok/s
P44B1 Q5/G64 mean: 28.953139 tok/s

mean paired speedup:   +0.4452%
median paired speedup: +0.5223%
wins:                   9 / 10

behavioral certification: PASS
promotion speed gate:   PASS
```

Every measured candidate and control run preserved the exact 138-round canonical text and speculative trajectory.

## Promotion verdict

**PASS.** P44B1 becomes the current certified experimental champion by the project's speed, determinism, and broad-robustness criteria.

## Local source artifacts

```text
/tmp/p44b1-direct-g32-vs-g64.py
/tmp/p44b1-direct-g32-vs-g64.log
/tmp/p44b1-q5g64-full-battery.py
/tmp/p44b1-q5g64-full-battery.log
/tmp/p44b1-q5g64-robustness-results.json
```

The local `mlx-m1-qmv` source tree should preserve these under a dedicated P44B1 champion snapshot before further mutation.
