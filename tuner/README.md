# Autotuner v0

This directory turns the manual Project27/28 workflow into a config-driven tuning funnel. The first implementation deliberately keeps **screening** and **production benchmarking** separate: screening can be aggressive, while final numbers still use one target + one drafter in an isolated persistent process.

## Current pieces

- `spec.py` — campaign/candidate loading and stable candidate IDs
- `search.py` — one-coordinate local-neighborhood generation around the current champion
- `build.py` — generic mixed-precision MTP builder using pristine BF16 projection tensors while preserving the known-good sidecar for nonquantized/sanitized tensors
- `validate.py` — verifies loaded `bits` / `group_size` against a candidate spec
- `benchmark.py` — isolated persistent-process canonical benchmark with SQLite logging
- `suite.py` — DEV/holdout multi-prompt runner with one target/drafter loaded once
- `promote.py` — relative round/TPS screen rules
- `results.py` — SQLite leaderboard
- `db.py` — result schema/storage
- `prompts/canonical.json` — the historical LRU optimization prompt
- `prompts/dev.json` — multi-domain development validation suite

The real holdout should be created locally as `tuner/prompts/holdout.local.json`; that path is gitignored so tuning decisions cannot quietly turn the public holdout into another training benchmark.

## Campaigns

`../campaigns/qwen38-q6.toml` describes the current Qwen3.8-27B Q6 target campaign. A new target quant or model variant should get its own TOML file rather than modifying this one in place.

The campaign separates:

- target path and MTP source shard
- current champion path + measured reference TPS/rounds
- exact module precision map
- per-module local search spaces
- early promotion thresholds

## First-pass workflow

From the repository root, in the Project27 MLX environment:

```bash
python -m tuner.search \
  --campaign campaigns/qwen38-q6.toml
```

This emits one-coordinate candidate specs under `candidates/generated/qwen38-q6/`.

Build one candidate:

```bash
python -m tuner.build \
  --campaign campaigns/qwen38-q6.toml \
  --candidate candidates/generated/qwen38-q6/CANDIDATE.json
```

Validate it:

```bash
python -m tuner.validate \
  --draft ~/models/autotune/qwen38-q6/CANDIDATE \
  --candidate candidates/generated/qwen38-q6/CANDIDATE.json
```

Canonical one-run screen:

```bash
MLX_QMV_FAST_M4=1 env -u MLX_QMV_FAST_M3 \
python -m tuner.benchmark \
  --campaign campaigns/qwen38-q6.toml \
  --draft ~/models/autotune/qwen38-q6/CANDIDATE \
  --candidate-id CANDIDATE \
  --stage screen \
  --runs 1
```

Apply the early screen rule:

```bash
python -m tuner.promote \
  --campaign campaigns/qwen38-q6.toml \
  --candidate-id CANDIDATE \
  --stage screen
```

For a serious candidate, run DEV before final promotion:

```bash
MLX_QMV_FAST_M4=1 env -u MLX_QMV_FAST_M3 \
python -m tuner.suite \
  --campaign campaigns/qwen38-q6.toml \
  --draft ~/models/autotune/qwen38-q6/CANDIDATE \
  --candidate-id CANDIDATE \
  --stage dev
```

Then run the isolated 10-run production benchmark:

```bash
MLX_QMV_FAST_M4=1 env -u MLX_QMV_FAST_M3 \
python -m tuner.benchmark \
  --campaign campaigns/qwen38-q6.toml \
  --draft ~/models/autotune/qwen38-q6/CANDIDATE \
  --candidate-id CANDIDATE \
  --stage production \
  --runs 10 \
  --warmups 1
```

Leaderboard:

```bash
python -m tuner.results \
  --campaign qwen38-q6 \
  --stage production
```

## Search strategy

v0 intentionally generates **one-coordinate mutations** rather than a full Cartesian product. The next layer will automate the funnel:

1. generate local neighbors
2. build/cache module variants
3. one-run canonical screen
4. reject candidates with bad round/TPS tradeoffs
5. component microbench when the mutation has a measurable mechanical path
6. short multi-run screen
7. DEV validation
8. isolated 10-run production benchmark
9. sealed holdout before final promotion
10. update champion and generate the next neighborhood

Pairwise interaction sweeps should only be opened around useful boundaries, such as the observed FC precision/group-size interaction, rather than blindly multiplying every bit and group-size combination.

## Repeating on other target quants

Create a separate campaign file, for example:

```text
campaigns/qwen38-q4.toml
campaigns/qwen38-q5.toml
campaigns/qwen38-q8.toml
```

Kernel work can be inherited when quantization geometry/packing is compatible. Drafter precision maps should be re-optimized per target because speculative acceptance follows the target's quantized trajectory.

## Status

This is a functional **v0 foundation**, not yet the unattended optimizer. The next implementation step is a campaign runner that executes the entire screen/build/validate/DEV/production funnel under a run budget and updates the SQLite leaderboard automatically.
