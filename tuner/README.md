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
- `run_campaign.py` — budgeted unattended v0 loop: generate -> build -> validate -> screen -> optional DEV/production
- `db.py` — result schema/storage
- `prompts/canonical.json` — the historical LRU optimization prompt
- `prompts/dev.json` — multi-domain development validation suite

The real holdout should be created locally as `tuner/prompts/holdout.local.json`; that path is gitignored so tuning decisions cannot quietly turn the public holdout into another training benchmark.

## Campaigns

`../campaigns/qwen38-q6.toml` describes the current Qwen3.8-27B Q6 target campaign. A new target quant or model variant should get its own TOML file rather than modifying this one in place.

The campaign separates target/source paths, the current champion and measured reference, the exact module precision map, per-module local search spaces, and early promotion thresholds.

## Fastest way to use v0

From the repository root, in the Project27 MLX environment, run a small unattended screen budget:

```bash
python -m tuner.run_campaign \
  --campaign campaigns/qwen38-q6.toml \
  --budget 8 \
  --screen-runs 1
```

The runner generates local one-coordinate neighbors, builds them under `~/models/autotune/qwen38-q6/`, validates the loader-visible quantization map, executes isolated canonical screens, stores results in `results/tuning.sqlite`, and applies relative round/TPS rules. It deliberately **does not update the champion automatically**.

Add DEV for candidates that pass the screen:

```bash
python -m tuner.run_campaign \
  --campaign campaigns/qwen38-q6.toml \
  --budget 8 \
  --screen-runs 1 \
  --run-dev
```

For a tightly controlled small budget, production runs can also be requested:

```bash
python -m tuner.run_campaign \
  --campaign campaigns/qwen38-q6.toml \
  --budget 3 \
  --screen-runs 1 \
  --run-dev \
  --run-production \
  --production-runs 10
```

Because production runs are expensive, the recommended default is to let the campaign runner screen candidates and explicitly choose finalists for the 10-run production test.

## Manual workflow

Generate candidate specs:

```bash
python -m tuner.search --campaign campaigns/qwen38-q6.toml
```

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

Run DEV:

```bash
MLX_QMV_FAST_M4=1 env -u MLX_QMV_FAST_M3 \
python -m tuner.suite \
  --campaign campaigns/qwen38-q6.toml \
  --draft ~/models/autotune/qwen38-q6/CANDIDATE \
  --candidate-id CANDIDATE \
  --stage dev
```

Run the isolated production benchmark:

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
python -m tuner.results --campaign qwen38-q6 --stage production
```

## Search strategy

v0 intentionally generates **one-coordinate mutations** rather than a full Cartesian product. The funnel is:

1. generate local neighbors
2. build candidates from pristine BF16 module tensors
3. validate the actual loader-visible precision map
4. one-run canonical screen
5. reject bad round/TPS tradeoffs
6. DEV validation for serious candidates
7. isolated 10-run production benchmark for finalists
8. sealed local holdout before a major promotion
9. explicitly update the campaign champion
10. generate the next neighborhood

Pairwise interaction sweeps should only be opened around useful boundaries, such as the observed FC precision/group-size interaction, rather than blindly multiplying every bit and group-size combination.

## Repeating on other target quants

Create a separate campaign file, for example:

```text
campaigns/qwen38-q4.toml
campaigns/qwen38-q5.toml
campaigns/qwen38-q8.toml
```

Kernel work can be inherited when quantization geometry/packing is compatible. Drafter precision maps should be re-optimized per target because speculative acceptance follows the target's quantized trajectory.

## Next implementation layer

The current runner removes most of the human copy/paste work but still rebuilds whole candidates and uses simple rule-based local search. The next layer should add a reusable per-module tensor cache, component microbench plugins, pairwise interaction generation around discovered boundaries, DEV-vs-reference comparison, and explicit champion promotion that rewrites the campaign only after all gates pass.
