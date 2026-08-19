# P46 D5 source freeze

This record was added after the P46 D5 checkpoint documentation landed in PR #5. It supersedes the provisional source-state note in `P46_D5_CHECKPOINT.md`, which said that a local source-tree freeze still needed to be made.

## Certified source state

The exact P46 D5 source state is now frozen in the public MLX fork:

```text
repository: skylinej3o1/mlx
branch:     project46-fast-m5-scout
commit:     22b52dfa9192ccb76a2c0e84d6e2f1383505023f
tag:        project46-d5-certified-26.55
```

Commit message:

```text
P46 D5 native M5 and T5 verifier checkpoint
```

Annotated tag message:

```text
P46 D5 certified 26.550819 tok/s; two-shape fixed M5; Q8 T5 2+2+1
```

## Frozen artifacts

The commit includes the active MLX source changes and the certified snapshot directory:

```text
mlx/backend/metal/kernels/quantized.h
mlx/backend/metal/kernels/quantized.metal
mlx/backend/metal/quantized.cpp
champion_snapshots/p46-d5-fixed2-certified/
```

The snapshot contains:

```text
CERTIFICATION.txt
p46-d5-fixed2-vs-baseline-3pair.py
p46-d5-fixed2-vs-baseline-cert.py
quantized.cpp
quantized.h
quantized.metal
```

## Certified result represented by this freeze

```text
P46 D5 candidate mean: 26.550819 tok/s
baseline mean:         26.524079 tok/s
paired mean:           +0.1008%
paired median:         +0.1230%
wins:                  8/10
candidate round:       150.654 ms
```

Exact invariants:

```text
text hash:       e39b478ae4a8
trajectory hash: 183cd3043746
rounds:          128
jury decisions:  512
```

Selective extra fixed shapes:

```text
5120 -> 6144   enabled
6144 -> 5120   enabled
5120 -> 1024   rejected / explicitly skipped
```

The same checkpoint also includes the certified Q8/G64 T5 `2+2+1` verifier-head path, native M5 body path, N=48 generic fallback, fixed major M5 shapes, P36 T5 attention, and the P45B2B draft-side token policy.

P45B2B D4 remains the overall throughput champion at 29.316537 tok/s; this source freeze records the certified D5 checkpoint before the next structural P46 experiments begin.
