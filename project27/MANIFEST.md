# Project27 reproducibility manifest

This directory is intended to preserve the **actual implementation artifacts** behind the 27.305 tok/s Project27 result, not only the narrative tuning log.

## Final measured configuration

- Hardware: 1 × Apple M1 Max, 64GB unified memory
- Target: Qwen3.8-27B Q6 affine group-64
- Target body: custom exact RAWX M4 Q6 path
- lm_head: Q8 affine group-64 with exact SHARED4 target argmax
- Speculative drafter: native MTP, block size 4
- MTP precision: Q3 MLP + Q6 attention
- Sustained result: 27.305 tok/s mean, 27.315 median, 27.272 minimum, 27.329 maximum
- Speculative verification: 139 rounds on every measured 512-token run
- Cross-drafter deterministic output: exact character-for-character match against the prior Q6 drafter on the canonical prompt

## Frozen MTP artifact

Local path used for the frozen champion:

```text
~/models/Qwen3.8-27B-MTP-Q3MLP-Q6ATTN-FP16-27.305
```

Known SHA-256:

```text
model.safetensors
204ba1fa67abdf1b1f167afbb9f6a9279b9d73b71c677fde5452ffa1af7fd77f

config.json
dd458a3174355b1673ccee3c9980f62692f89ede6bf7841a986762ca4b3c7336
```

Model weights themselves are not intended to be committed to this repository. Configs, hashes, precision maps, builders, source patches, and benchmark harnesses are.

## MLX fork milestones

Recorded Project24/27 milestones include:

```text
13b14b1  project24-q6-m3-4x2-22.0
95767f5  project24-q6-m4-4x2-22.7
          project24-q6-m4-k2-23.4
ae2676b-ish / recorded tag project24-q6-m4-k4-23.6
58c2497  project24-q6-m4-k4-qdot4-23.9
194a432  project24-q6-m4-k4-qdot4-shift256-exact-24.9
679d196  project24-q6-m4-k4-qdot4-rawx-exact-26.57
```

The local snapshot script records the authoritative tag/branch/history state from the actual tuning repo rather than relying on this summary.

## Files captured by `scripts/snapshot_local_project27.sh`

The script is designed to copy the exact working artifacts from the tuning Mac into this repository:

```text
project27/
├── benchmarks/
│   ├── q38-persistent-rawx.py
│   ├── q38-q3mlp-full.py
│   ├── q38-persistent-warmup-only.py
│   ├── q38-cross-drafter-exact.py
│   ├── q38-mtp-block-sweep.py
│   ├── q38-m4-prof.py
│   ├── q38-q8head-regress.py
│   ├── q38-q8head-helper-only.py
│   └── q38-q8head-split.py
├── builders/
│   ├── build-q38-mtp-bits.py
│   ├── build-q38-mtp-q3mlp.py
│   ├── compare-q38-mtp-weights.py
│   └── audit-q38-mtp-q8-runtime.py
├── patches/
│   ├── mlx/
│   │   └── Project24 → RAWX exact source patch/history material
│   └── mlx-vlm/
│       └── project24-q8head-shared4-26.69.patch
├── source-snapshots/
│   ├── quantized.h.rawx-26.57
│   ├── quantized.cpp.rawx-26.57
│   ├── quantized.metal.rawx-26.57
│   ├── language.py.q8shared4-26.69-exact
│   └── language.py.installed-project27
├── negative-results/
│   └── preserved dead/flat RAWX experiment patches
├── environment/
│   ├── mlx-git-state.txt
│   ├── mlx-all-history.txt
│   ├── mlx-tags.txt
│   ├── mlx-branches.txt
│   ├── python-and-system.txt
│   └── metal-and-memory.txt
├── model-manifests/
│   ├── target-config.json
│   ├── target-sha256.txt
│   ├── draft-config.json
│   └── draft-sha256.txt
└── SHA256SUMS.txt
```

## Why snapshot source files as well as patches

A patch is only fully useful when its exact base is known. The snapshot therefore preserves both:

1. Git history/tag/base metadata.
2. A consolidated source patch when possible.
3. The exact final versions of the modified MLX files.
4. The exact installed mlx-vlm `language.py` that provided the Q8 SHARED4 path.

This makes the winning state recoverable even if upstream MLX or mlx-vlm changes later.

## Canonical benchmark discipline

Project27 benchmarks should be interpreted only under the persistent-process protocol:

- target loaded once
- drafter loaded once
- explicit drafter reset/rebind between requests
- one warmup
- ten measured 512-token generations
- temperature 0
- block size 4
- avoid Mission Control / Space switching / window animations during runs
- use `result.generation_tps` as the primary throughput metric
- always record speculative round counts and deterministic output equality

## Local capture

Run on the Project27 M1 Max after cloning this repository:

```bash
cd ~/src/m1-max-llm-tuning
bash scripts/snapshot_local_project27.sh
```

The script commits and pushes the captured implementation assets to `main` after generating SHA-256 checksums.
