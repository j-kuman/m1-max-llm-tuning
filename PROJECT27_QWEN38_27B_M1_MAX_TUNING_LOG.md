# Project27: Qwen3.8-27B Q6 on a 64GB M1 Max

**Status:** Project27 achieved  
**Best sustained result:** **27.305 tok/s mean**, **27.315 tok/s median**, **27.272 tok/s minimum**, **27.329 tok/s maximum** over 10 measured 512-token runs  
**Hardware:** 2021 MacBook Pro, Apple M1 Max, 64GB unified memory  
**Target model:** Qwen3.8-27B, Q6 affine group-64, FP16/BF16 activations, Q8 lm_head  
**Speculative drafter:** native Qwen MTP, **Q3 MLP + Q6 attention**, block size 4  
**Target kernel stack:** custom MLX RAWX Q6 M4 path + exact Q8 SHARED4 argmax head  
**Correctness:** deterministic 512-token canonical output matched the prior exact Q6 drafter character-for-character

This document preserves the tuning process, measurements, failed experiments, exact artifact identifiers, and the reasoning that led from roughly 14–15 tok/s plain Q6 inference to a sustained 27.3 tok/s exact speculative-decoding configuration on a single M1 Max.

---

## 1. Final production configuration

### Hardware

- Apple M1 Max
- 64GB unified memory
- macOS
- `iogpu.wired_limit_mb = 60416`
- UI/GPU contention matters: avoid Mission Control, Space switching, window animations, scrolling, or moving windows during serious timing runs.

### Python / MLX environment

Virtual environment:

```text
/Users/skylinej17/.venvs/mlx-dspark
```

Custom MLX source tree:

```text
~/src/mlx-m1-qmv
```

Important environment details recorded during tuning:

```text
setuptools 84.0
wheel 0.48
CMake 4.4.2
/opt/homebrew/bin/cmake
```

Known package-version warnings were intentionally tolerated because the custom setup depended on patched MLX / mlx-vlm behavior:

- `mlx-vlm 0.6.10` formally wanted final `mlx>=0.32.0`
- `mlx-dspark 0.10.0` wanted `mlx>=0.32.0` and `mlx-vlm>=0.6.12`
- `mlx-vlm 0.6.10` was intentionally retained because the Q8 target-head patch lived in its installed `language.py`

**Do not casually upgrade mlx-vlm without first preserving and reapplying the custom Q8 target-head work.**

### Target model

```text
~/models/Qwen3.8-27B-MLX-6bit-FP16-Q8HEAD
```

Base/original Q6 model:

```text
~/models/Qwen3.8-27B-MLX-6bit-FP16
```

Approximate geometry:

```text
hidden size:       5120
intermediate size: 17408
layers:            64
GDN/linear-attn:   48
full-attention:    16
target quant:      Q6 affine, group 64
lm_head:           Q8 affine, group 64
```

### Final MTP drafter

Frozen artifact:

```text
~/models/Qwen3.8-27B-MTP-Q3MLP-Q6ATTN-FP16-27.305
```

Precision map:

```text
MLP:
  gate_proj  Q3
  up_proj    Q3
  down_proj  Q3

Attention:
  q_proj     Q6
  k_proj     Q6
  v_proj     Q6
  o_proj     Q6

Non-quantized:
  fc.weight and norms preserved from the known-good converted FP16 sidecar
```

Draft block size:

```text
4
```

Frozen SHA-256 hashes:

```text
model.safetensors
204ba1fa67abdf1b1f167afbb9f6a9279b9d73b71c677fde5452ffa1af7fd77f

config.json
dd458a3174355b1673ccee3c9980f62692f89ede6bf7841a986762ca4b3c7336
```

### Canonical benchmark launch

```bash
MLX_QMV_FAST_M4=1 \
env -u MLX_QMV_FAST_M3 \
python /tmp/q38-q3mlp-full.py
```

Equivalent production generation shape:

```bash
MLX_QMV_FAST_M4=1 \
env -u MLX_QMV_FAST_M3 \
python -m mlx_vlm.generate \
  --model ~/models/Qwen3.8-27B-MLX-6bit-FP16-Q8HEAD \
  --draft-model ~/models/Qwen3.8-27B-MTP-Q3MLP-Q6ATTN-FP16-27.305 \
  --draft-kind mtp \
  --draft-block-size 4 \
  --prompt "Implement an LRU cache in Python using a dictionary and doubly linked list. Include type hints and a usage example." \
  --max-tokens 512 \
  --temperature 0
```

---

## 2. Final sustained benchmark

Persistent process: load target once, load MTP once, reset drafter each request, one warmup, 10 measured 512-token runs.

```text
WARMUP
27.273 tok/s
139 rounds
24.858 GB peak

MEASURED
1   27.281
2   27.290
3   27.315
4   27.276
5   27.272
6   27.329
7   27.316
8   27.325
9   27.324
10  27.325

median: 27.315 tok/s
mean:   27.305 tok/s
min:    27.272 tok/s
max:    27.329 tok/s
spread: 0.057 tok/s

rounds: 139 on every run
all generated text identical: True
peak memory: 24.858 GB
```

This was the first configuration where **every measured run exceeded 27 tok/s**.

---

## 3. Correctness check

The final Q3-MLP/Q6-attention drafter was compared directly against the prior exact Q6 drafter using:

- same target model
- same deterministic prompt
- temperature 0
- 512 generated tokens
- draft block size 4

Result:

```text
Q6 character length:              1808
Q3MLP/Q6ATTN character length:    1808
exact text equality:              True

CROSS-DRAFTER OUTPUT: EXACT
```

The timing numbers from this correctness harness (~22.9 tok/s) are **not production benchmark numbers** because the harness loads and compares drafters sequentially. It was used only to establish output identity.

---

## 4. Benchmark discipline

The tuning campaign became much more reliable after standardizing measurement.

### Persistent-process protocol

1. Load the target once.
2. Load the MTP drafter once.
3. Explicitly reset/rebind the drafter between requests.
4. One warmup request.
5. Ten measured 512-token requests.
6. Five-second separation where useful.
7. Use `result.generation_tps` as the primary metric.
8. Record speculative-round count and output equality.

### macOS UI interference

Window animations and macOS Spaces can measurably contend with the GPU.

For clean runs:

- remain on one desktop
- do not invoke Mission Control
- do not swipe Spaces
- do not resize/move windows
- avoid scrolling while the benchmark is running
- a static browser window is generally fine

### Measured vs estimated

Throughout this work:

- **Measured** = produced by an actual benchmark.
- **Estimated** = analytical projection only.
- Never upgrade an estimate into a measured result.

---

## 5. Performance progression

Approximate major milestones:

| Stage | Measured throughput |
|---|---:|
| Plain Qwen3.8-27B Q6 | ~14.2–14.6 tok/s |
| Stock/native MTP | 17.398 tok/s |
| M3 work | ~22.0 tok/s |
| M4 pre-K2 | ~22.673 tok/s median |
| K2 | 23.394 tok/s |
| K4 | 23.613 tok/s |
| exact QDOT4 | 23.893 tok/s |
| exact SHIFT256 | 24.917 tok/s |
| RAWX exact, persistent clean | ~26.57 tok/s |
| RAWX + Q8 SHARED4 | **26.694 tok/s mean** |
| Full-Q4 MTP | **26.770 tok/s mean** |
| Q4 MLP + Q6 attention | **26.952 tok/s mean** |
| **Q3 MLP + Q6 attention** | **27.305 tok/s mean** |

The total practical gain from plain-ish Q6 to the final exact configuration was close to a 2× throughput improvement on the same M1 Max.

---

## 6. Git / kernel milestones already preserved

Recorded milestones from the custom MLX work:

```text
M3
commit: 13b14b1
tag: project24-q6-m3-4x2-22.0

M4 pre-K2
commit: 95767f5
tag: project24-q6-m4-4x2-22.7

K2
tag: project24-q6-m4-k2-23.4

K4
commit around: ae2676b
tag: project24-q6-m4-k4-23.6

QDOT4 exact
commit: 58c2497
tag: project24-q6-m4-k4-qdot4-23.9

SHIFT256 exact
commit: 194a432
tag: project24-q6-m4-k4-qdot4-shift256-exact-24.9

RAWX exact
commit observed around: 679d196
tag: project24-q6-m4-k4-qdot4-rawx-exact-26.57
```

A final Project27 git tag was proposed but was **not yet confirmed as created at the time this document was written**. Before relying on one, verify with:

```bash
cd ~/src/mlx-m1-qmv
git tag --list 'project27*'
git rev-parse HEAD
git status --short
```

---

## 7. RAWX target kernel

RAWX was the major target-body breakthrough.

### Host dispatch shape

Relevant custom path in `quantized.cpp`:

- block N: 8
- block K: 32
- fast path requires roughly:
  - `N % 8 == 0`
  - `K % 512 == 0`
  - bits = 6
  - group size = 64
  - M = 4
  - batch = 1
  - FP16/BF16 activation path as configured

Launch geometry:

```text
threadgroup/grid concept:
4 SIMDgroups × 2 rows = 8 output rows per TG
VALUES_PER_THREAD = 8
BLOCK_SIZE = 256
PACKS_PER_THREAD = 2
```

### Core ideas

- Directly stage raw X as FP32.
- Share Q6 weight decode across the four M=4 verification rows.
- Preserve the exact FP accumulation order needed for deterministic equality.
- Avoid manual over-unrolling that causes register/code-size blowups.
- Optimize the real speculative verification shape rather than generic GEMM.

### Exactness

RAWX produced zero observed max difference on the key gate/down/MLP/GDN checks used during development.

### RAWX profiling

Representative target-body profile:

```text
64 MLPs:        ~91.2 ms sum of medians
48 GDNs:        ~49.1–49.6 ms
16 attentions:  ~15.0 ms
rough component sum: ~155 ms
```

Important: this component profiler is useful for **relative hotspot weighting**, not as an additive generation-wall-clock model.

---

## 8. RAWX experiments that did not win

Preserved negative results matter because they prevent repeating dead-end work.

Experiments that were flat or worse in the RAWX era included:

- mask cleanup
- float4
- 8x1
- 4x1
- 4x4
- 8x2
- HALFX
- K2
- earlier shared unpack variants
- `packs_per_thread=1`
- old raw-half X
- metadata broadcast experiments
- shared-X attempts
- q+k RMS fusion
- convolution specialization attempts
- gate/up fusion
- qdot2x2
- manual K4 over-unroll, which was catastrophic (~12.2 tok/s class behavior due to register/code pressure)

Some failed patches were preserved under paths such as:

```text
/tmp/project24-rawx-mask-cleanup-flat.patch
/tmp/project24-rawx-float4-flat.patch
/tmp/project24-rawx-8x1-dead.patch
/tmp/project24-rawx-4x1-dead.patch
/tmp/project24-rawx-4x4-dead.patch
/tmp/project24-rawx-8x2-tie.patch
/tmp/project24-rawx-halfx-tie-production-loss.patch
/tmp/project24-rawx-k2-loss.patch
```

---

## 9. MTP block-size sweep

A dedicated sweep tested draft block sizes 2 through 8.

Representative measured result:

| Block | Throughput | Acceptance |
|---:|---:|---:|
| 2 | ~18.370 | ~0.9692 |
| 3 | ~18.363 | ~0.9248 |
| **4** | **~26.504** | **~0.8747** |
| 5 | ~15.993 | ~0.7500 |
| 6 | ~14.641 | ~0.6764 |
| 7 | ~15.006 | ~0.6039 |
| 8 | ~15.489 | ~0.5635 |

The enormous block-4 advantage was not purely an intrinsic acceptance optimum: **M=4 was the shape that hit the custom FAST_M4 target kernel**.

Therefore:

```text
draft block size 4
```

became a structural part of the production configuration.

---

## 10. Q8 target-head optimization

The target `lm_head` was converted to:

```text
Q8 affine
group size 64
logical shape 5120 -> 248320
```

Custom code was placed in:

```text
~/.venvs/mlx-dspark/lib/python3.14/site-packages/mlx_vlm/models/qwen3_5/language.py
```

Preserved files:

```text
~/project24-patches/language.py.q8shared4-26.69
~/project24-patches/language.py.q8shared4-26.69-exact
~/project24-patches/project24-q8head-shared4-26.69.patch
```

### Baseline Q8 head

Initial helper timing for T=4:

```text
full helper median: ~5.846 ms
```

### SHARED4 breakthrough

The original Q8 kernel independently performed the same Q8 weight decode for each of the four verify-token rows.

The winning change introduced a T=4 shared helper conceptually like:

```cpp
qdot_q8_t4_shared(
    w,
    x0, x1, x2, x3,
    scale, bias,
    sum0, sum1, sum2, sum3,
    out0, out1, out2, out3
);
```

Weight byte decode was shared across the four X rows while preserving the intended arithmetic ordering.

Measured helper performance:

```text
baseline full helper:  ~5.846 ms
SHARED4 full helper:   ~4.754 ms

head microbenchmark improvement:
~18.7%
```

### Exactness

A 100-seed deterministic T=4 regression compared the pre-SHARED4 implementation against SHARED4:

```text
Q8 HEAD: EXACT ACROSS 100 RANDOM T=4 CASES
```

### Production effect

Persistent RAWX baseline:

```text
~26.574 tok/s mean-ish clean baseline
```

RAWX + Q8 SHARED4:

```text
warmup: 26.842

10 measured:
26.688
26.705
26.705
26.691
26.730
26.727
26.682
26.647
26.689
26.680

mean:   26.694
median: 26.690
min:    26.647
max:    26.730
spread: 0.083
rounds: 141 every run
```

This became the pre-MTP-precision champion.

---

## 11. Q8 head dead ends

After SHARED4, geometry and load experiments mapped the local optimum.

| Variant | Helper median | Result |
|---|---:|---|
| **SHARED4 R4** | **~4.754 ms** | champion |
| SG4 / 16-row TG | ~4.770 ms | flat/slight loss |
| K2 | ~4.930 ms | loss |
| PACK2 32-bit explicit byte extraction | ~5.334 ms | hard loss |
| R2 | ~5.515 ms | loss |
| R8 | ~5.659 ms | loss |

Important lesson:

**R4 + 2 SIMDgroups + scalar byte loads was a sharp local optimum.**

Explicit packed 32-bit loads looked attractive but added enough shift/extract machinery to lose badly; the compiler/native byte path was already efficient.

---

## 12. MTP precision campaign

The most surprising part of Project27 was discovering that MTP precision is not monotonic with speculative usefulness.

### Original known-good MTP sidecar

```text
~/models/Qwen3.8-27B-MTP-oQ6-FP16
```

File size:

```text
388.5 MiB
```

Quantized modules:

```text
layers.0.mlp.up_proj
layers.0.mlp.down_proj
layers.0.mlp.gate_proj
layers.0.self_attn.o_proj
layers.0.self_attn.v_proj
layers.0.self_attn.k_proj
layers.0.self_attn.q_proj
```

All were originally Q6 affine, group 64.

The sidecar also contained roughly:

```text
float16 payload: 122.237 MiB
uint32 packed Q6 payload: 266.250 MiB
```

### Pristine source MTP

The official Qwen3.8-27B Hugging Face snapshot was already indexed locally, but the final source shard had to be downloaded:

```text
Qwen/Qwen3.8-27B
revision:
1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0

source shard:
model-00018-of-00018.safetensors
```

All 15 `mtp.*` tensors lived in that shard.

The pristine MTP source payload was:

```text
810.050 MiB BF16
```

Key BF16 source matrices included:

```text
mtp.fc.weight                              (5120, 10240)
mtp.layers.0.mlp.down_proj.weight          (5120, 17408)
mtp.layers.0.mlp.gate_proj.weight          (17408, 5120)
mtp.layers.0.mlp.up_proj.weight            (17408, 5120)
mtp.layers.0.self_attn.q_proj.weight       (12288, 5120)
mtp.layers.0.self_attn.k_proj.weight       (1024, 5120)
mtp.layers.0.self_attn.v_proj.weight       (1024, 5120)
mtp.layers.0.self_attn.o_proj.weight       (5120, 6144)
```

---

## 13. The broken first Q8 MTP build and the norm-sanitization discovery

The first homemade BF16 -> FP16 -> Q8 sidecar catastrophically failed:

```text
warmup:
~7.345 tok/s
511 speculative rounds
```

This was not ordinary Q8 quality loss.

### Projection audit

Dequantized Q8 projection weights were approximately 4× closer to the pristine BF16 source than Q6.

Examples:

```text
gate_proj mean source error
Q6: 0.00022621
Q8: 0.00005690

q_proj
Q6: 0.00027002
Q8: 0.00006773

o_proj
Q6: 0.00039802
Q8: 0.00009489
```

Runtime `QuantizedLinear` tests at both M=1 and M=4 also showed that the Q8 execution path was healthy and closer to source than Q6.

### Actual bug: norms

The non-quantized norm vectors in the known-good Q6 sidecar differed from pristine source by almost exactly 1.0, while the homemade Q8 sidecar had copied pristine values directly.

Representative pattern:

```text
input_layernorm:
source -> Q6 difference ~1.0
source -> homemade Q8 difference 0

post_attention_layernorm:
source -> Q6 difference ~1.0
source -> homemade Q8 difference 0

q_norm / k_norm / final norm / pre_fc norms:
same pattern
```

This revealed that the proper conversion path applies a normalization-weight sanitization/transformation.

### Safe fix

Do not reproduce the transform by guessing.

Instead:

- build the seven quantized projection matrices from pristine BF16 source
- copy `fc.weight` and all non-quantized norm tensors **verbatim from the known-good Q6 sidecar**

The fixed Q8 sidecar was:

```text
477.240 MiB
```

and structurally valid.

---

## 14. Full-precision sweep results

### Q8 MTP

Fixed Q8 behaved normally but lost acceptance:

```text
144 rounds
~26.09 tok/s
```

### Q5 MTP

Built correctly from pristine source plus known-good sanitized non-quantized tensors:

```text
344.115 MiB
warmup: 26.293 tok/s
rounds: 143
```

Not competitive.

### Q4 MTP

Full Q4 was the first precision-sweep winner:

```text
warmup: 26.827

10 measured:
26.759
26.736
26.753
26.795
26.733
26.818
26.787
26.781
26.772
26.767

median: 26.769
mean:   26.770
rounds: 142
peak:   24.865 GB
```

Despite requiring one extra target verification round relative to Q6, Q4 drafting was cheap enough to win end-to-end.

### Key lesson

Speculative acceptance is **not monotonic with quantization precision**.

Closer-to-source weights do not necessarily produce better alignment with the quantized target's greedy path.

---

## 15. Mixed-precision MTP breakthrough

Two broad hybrids were tested.

### Q4 MLP + Q6 attention

```text
10 measured:
26.898
26.953
26.949
27.049
26.993
26.954
26.929
26.926
26.899
26.973

median: 26.951
mean:   26.952
rounds: 141
peak:   24.891 GB
```

This became the first build to produce an individual measured run above 27 tok/s:

```text
27.049 tok/s
```

### Q6 MLP + Q4 attention

```text
median: 26.826
mean:   26.826
rounds: 141
peak:   ~24.932 GB
```

Conclusion:

**The MLP was the high-value place to reduce precision.**

---

## 16. Attention single-projection sweep

Starting from Q4 MLP + Q6 attention, each attention projection was independently switched to Q4.

Warmup results:

```text
Q4 MLP + Q4 q_proj + Q6 K/V/O
26.370 tok/s
143 rounds
DEAD

Q4 MLP + Q4 k_proj + Q6 Q/V/O
26.983 tok/s
141 rounds
LIVE

Q4 MLP + Q4 v_proj + Q6 Q/K/O
26.836 tok/s
141 rounds
LIVE

Q4 MLP + Q4 o_proj + Q6 Q/K/V
26.830 tok/s
141 rounds
LIVE
```

This strongly implicated **q_proj Q4** as the main attention-side acceptance problem.

### K/V/O together

A combined:

```text
Q4 MLP
Q6 q_proj
Q4 k/v/o
```

configuration retained 141 rounds but was basically flat:

```text
mean:   26.944
median: 26.941
rounds: 141
```

Versus Q4 MLP + all-Q6 attention:

```text
26.952
```

So the cleaner all-Q6 attention map remained preferable.

---

## 17. Project27 jackpot: Q3 MLP + Q6 attention

The next experiment dropped only the three MLP projections from Q4 to Q3 while keeping all attention Q6.

Warmup:

```text
27.446 tok/s
139 rounds
```

A fresh full run confirmed it was real:

```text
warmup: 27.273
rounds: 139

mean:   27.305
median: 27.315
min:    27.272
max:    27.329
spread: 0.057
rounds: 139 every measured run
```

This was a double win:

1. Q3 MLP computation was cheaper.
2. The quantization perturbation **improved speculative agreement enough to remove two whole target verification rounds** compared with the Q6 baseline.

This is the central Project27 result.

---

## 18. MTP acceptance landscape

Observed full/hybrid configurations:

| Drafter map | Rounds | Throughput |
|---|---:|---:|
| Full Q8 | 144 | ~26.09 |
| Full Q5 | 143 | ~26.29 warmup |
| Full Q4 | 142 | 26.770 mean |
| Full Q6 | 141 | 26.694 mean |
| Q6 MLP + Q4 attention | 141 | 26.826 mean |
| Q4 MLP + Q6 attention | 141 | 26.952 mean |
| Q4 MLP + Q6Q + Q4KVO | 141 | 26.944 mean |
| **Q3 MLP + Q6 attention** | **139** | **27.305 mean** |

The optimization target for a speculative drafter is therefore not simply:

> minimize quantization error.

It is closer to:

> minimize total end-to-end generation time, which depends on both draft cost and agreement with the target's discrete argmax trajectory.

A noisier drafter can be better if its errors happen to align it with the target more often.

---

## 19. Q6 vs Q8 MTP projection timing

A simple M=1 per-projection microbenchmark found:

```text
Q6 summed projection medians: 3.1442 ms
Q8 summed projection medians: 3.0694 ms
```

This benchmark had noticeable per-module noise, so the exact 2.4% difference should not be over-interpreted.

But it was strong enough to show that the fixed Q8 MTP production loss was **not caused by Q8 compute being dramatically slower**. The primary loss came from worse speculative acceptance and extra target rounds.

---

## 20. Why Project27 worked

The performance improvement was not one trick. It was stacked:

### Target body

Custom M4 Q6 path / RAWX reduced the dominant dense target work.

### Target head

Q8 SHARED4 reused weight decoding across four verification positions.

### Speculative block size

Block 4 matched the custom M=4 target kernel.

### Drafter quantization

Mixed precision reduced draft cost while **improving target agreement**:

```text
Q3 MLP
Q6 attention
```

### Measurement discipline

Persistent-process timing eliminated a large amount of process/JIT/noise confusion.

The final configuration aligned the hardware, quantization, speculative shape, and kernel geometry around the actual production workload.

---

## 21. Things not to repeat without a changed hypothesis

Dead or unproductive ideas included:

- MTP block sizes 2, 3, 5, 6, 7, 8 under the current target specialization
- Q8 head SG4
- Q8 head R2 / R8
- Q8 head K2
- Q8 head PACK2 explicit 32-bit load/extract
- RAWX mask cleanup
- RAWX float4
- RAWX 8x1 / 4x1 / 4x4
- RAWX K2
- RAWX HALFX
- over-aggressive manual unrolling
- full Q8 MTP
- full Q5 MTP
- full Q4 as the final answer
- Q4 `q_proj` in the MTP attention path
- Q4 K/V/O together as a meaningful speed improvement

These may become interesting again only if the compiler, model geometry, target quant, speculative shape, or hardware changes.

---

## 22. Reconstructing the target-side Q8 patch

The Q8 target-head customizations live in the installed mlx-vlm package, not in the MLX git tree:

```text
~/.venvs/mlx-dspark/lib/python3.14/site-packages/
mlx_vlm/models/qwen3_5/language.py
```

Preserved exact champion copy:

```text
~/project24-patches/language.py.q8shared4-26.69-exact
```

To restore:

```bash
LANGFILE=~/.venvs/mlx-dspark/lib/python3.14/site-packages/mlx_vlm/models/qwen3_5/language.py

cp \
  ~/project24-patches/language.py.q8shared4-26.69-exact \
  "$LANGFILE"

cmp -s \
  "$LANGFILE" \
  ~/project24-patches/language.py.q8shared4-26.69-exact \
  && echo "Q8 SHARED4 RESTORED"
```

**Do not use the shell variable `LANG` for this path.** Earlier tuning commands temporarily did so, causing locale warnings. Use `LANGFILE`.

If locale was accidentally overwritten:

```bash
unset LANG
unset LC_ALL
export LANG=en_US.UTF-8
locale
```

---

## 23. Useful restoration command for RAWX MLX files

Before starting a new MLX kernel experiment, restore the RAWX exact state for the principal modified files:

```bash
git restore \
  --source project24-q6-m4-k4-qdot4-rawx-exact-26.57 \
  --staged --worktree \
  mlx/backend/metal/kernels/quantized.h \
  mlx/backend/metal/quantized.cpp
```

Be careful: Q8-head changes are in mlx-vlm `language.py`, not this MLX repository.

---

## 24. Preservation checklist

Already frozen:

- [x] Final MTP directory copied to `...-27.305`
- [x] MTP `model.safetensors` SHA-256 recorded
- [x] MTP `config.json` SHA-256 recorded
- [x] Cross-drafter deterministic exact output confirmed
- [x] Q8 SHARED4 exact `language.py` copy preserved
- [x] Q8 SHARED4 patch preserved
- [x] RAWX git tag previously preserved

Still worth doing:

- [ ] Copy current final benchmark script into a permanent project directory
- [ ] Copy cross-drafter correctness harness into the same directory
- [ ] Hash the preserved `language.py`
- [ ] Record current MLX `git rev-parse HEAD`
- [ ] Confirm repository `git status --short`
- [ ] Create a dedicated final Project27 annotated git tag if desired
- [ ] Preserve this Markdown document in GitHub

Suggested final target-side snapshot:

```bash
mkdir -p ~/project24-patches/project27-27.305

cp \
  ~/.venvs/mlx-dspark/lib/python3.14/site-packages/mlx_vlm/models/qwen3_5/language.py \
  ~/project24-patches/project27-27.305/language.py

cp \
  /tmp/q38-q3mlp-full.py \
  ~/project24-patches/project27-27.305/benchmark.py

cp \
  /tmp/q38-cross-drafter-exact.py \
  ~/project24-patches/project27-27.305/cross-drafter-exact.py

shasum -a 256 \
  ~/project24-patches/project27-27.305/language.py \
  ~/project24-patches/project27-27.305/benchmark.py \
  ~/project24-patches/project27-27.305/cross-drafter-exact.py
```

---

## 25. Portability to other Qwen3.8-27B target quants

### High-transfer lessons

These should transfer strongly to Q4/Q5/Q8 target tuning:

- persistent-process measurement
- MTP block-size sweep
- optimize the exact speculative verification shape
- treat drafter precision as an end-to-end acceptance/speed parameter
- mixed precision by submodule
- preserve sensitive attention components independently of MLP
- measure rounds before spending time on long timing runs
- target-head specialization
- exactness regression before promoting a kernel

### Medium-transfer pieces

- Q8 SHARED4 head logic should transfer well where the same Q8 head shape remains.
- RAWX design principles transfer, but exact code depends on packed target representation.

### Low-transfer assumption

Do **not** assume:

```text
Q3 MLP + Q6 attention
```

will be the winner against every target quant.

The final drafter map is target-trajectory-specific. A Q4 target might prefer Q2/Q3 MLP with Q4/Q5 attention; a Q5 or Q8 target could have a different acceptance optimum.

The process transfers more strongly than the literal bit map.

---

## 26. Related two-M1-Max DeepSeek project

Separate from this single-Mac Qwen work, two 64GB M1 Max machines were tested with **real tensor parallel inference** on DeepSeek V4 Flash 0731 Q2/Q4.

Current remembered/measured anchor from that branch:

```text
~17–17.6 tok/s decode
2 × M1 Max 64GB
TP across both machines
no DSpark/speculative decoding in that baseline
```

The Project27 methodology suggests a strong plan for revisiting it:

1. reproduce the ~17.5 tok/s TP baseline under the same strict persistent measurement discipline
2. split decode time into local expert compute, non-expert compute, TP communication, synchronization/idle, and CPU overhead
3. specialize only the true hot local shapes
4. revisit mixed expert precision instead of assuming higher precision is monotonically better
5. tune layer/expert distribution for communication balance
6. add speculative decoding only after the target TP baseline is well characterized

A move from ~17.5 to 20 tok/s is about a 14% gain. This is **not yet measured**, but Project27 demonstrated that substantial hidden performance can exist in execution shape, kernel specialization, and mixed quantization policy.

---

## 27. Broader conclusion

The M1 Max inference ceiling is not well described by a simplistic statement such as “only feasible up to 35B.”

The 64GB M1 Max combines:

- high unified-memory capacity
- high memory bandwidth
- no discrete CPU/GPU VRAM boundary
- capable Metal compute
- enough flexibility for model-specific specialization

With generic software, Qwen3.8-27B Q6 was roughly a 14–15 tok/s experience.

With a tuned stack, the same hardware sustained:

```text
27.305 tok/s mean
27.315 tok/s median
27.272 tok/s floor
```

on the Project27 canonical 512-token workload.

The central lesson is not simply that the M1 Max is fast. It is that **architecture-aware software tuning can change the practical class of the hardware**.

---

## 28. Next obvious research directions

Do not modify the frozen Project27 artifacts. Branch experiments from copies.

Potential next projects:

### Project28

The final Project27 result is ~27.305 tok/s. Reaching 28 requires roughly another 2.55% improvement.

Candidate directions:

- Q2/Q3 submodule-level MTP sweeps rather than global changes
- identify whether only one or two Q3 MLP projections produce the 139-round trajectory
- test Q2 selectively while preserving 139 rounds
- profile the final 27.305 build again to identify the post-MTP bottleneck
- explore target-body improvements only after the drafter map is fully characterized
- revisit attention/norm/elementwise fusion if profiling justifies it

### Other Qwen3.8-27B target quants

Run the same precision-map / acceptance search independently for Q4, Q5, and Q8 targets.

### DeepSeek V4 Flash 0731 TP

Use the ~17.5 tok/s two-M1 TP baseline as the immutable starting point and attack communication-aware MoE execution.

---

## 29. Final Project27 identity

```text
PROJECT:
Project27

MODEL:
Qwen3.8-27B

HARDWARE:
1 × Apple M1 Max 64GB

TARGET:
Q6 affine group 64
RAWX exact M4 target body
Q8 affine group 64 lm_head
Q8 SHARED4 exact argmax

MTP:
block size 4
Q3 MLP
Q6 attention
known-good sanitized FP16 non-quantized tensors

SUSTAINED RESULT:
27.305 tok/s mean
27.315 tok/s median
27.272 tok/s minimum
27.329 tok/s maximum
0.057 tok/s spread
139 speculative rounds
24.858 GB peak
10/10 outputs identical

CROSS-DRAFTER CORRECTNESS:
Q6 drafter output == Q3MLP/Q6ATTN output
512 tokens
temperature 0
character-for-character exact

FROZEN MTP MODEL:
~/models/Qwen3.8-27B-MTP-Q3MLP-Q6ATTN-FP16-27.305

SHA256 model.safetensors:
204ba1fa67abdf1b1f167afbb9f6a9279b9d73b71c677fde5452ffa1af7fd77f

SHA256 config.json:
dd458a3174355b1673ccee3c9980f62692f89ede6bf7841a986762ca4b3c7336
```

---

*Preservation note: this document intentionally records both wins and failed experiments. Negative results are part of the asset because they prevent repeating already-explored local optima.*
