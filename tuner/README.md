# Autotuner v0.1

This directory turns the manual Project27/28 workflow into a config-driven tuning funnel. Screening and production benchmarking stay separate: screening can be aggressive, while final numbers still use one target + one drafter in an isolated persistent process.

The critical rule is now **fail closed**. At temperature 0 with exact target verification, changing the drafter must not change target output. The tuner therefore freezes champion output hashes per prompt and treats any candidate hash/token mismatch as a correctness failure, not a quality tradeoff.

## Current pieces

- `spec.py` — campaign/candidate loading, portable path expansion, stable candidate IDs
- `search.py` — one-coordinate local-neighborhood generation around the current champion
- `build.py` — generic mixed-precision MTP builder using pristine BF16 projection tensors while preserving the known-good sidecar for nonquantized/sanitized tensors
- `validate.py` — verifies loaded `bits` / `group_size` against a candidate spec
- `gates.py` — exact-output reference handling and fail-closed speculative-round telemetry
- `benchmark.py` — isolated canonical/reference/production benchmark with exactness gates and SQLite logging
- `suite.py` — DEV/holdout multi-prompt runner with exactness gates and normalized `tokens/round`
- `promote.py` — canonical screen rules plus a real prompt-by-prompt DEV promotion gate
- `microbench.py` — command-backed component microbenchmark plugin runner for target-body, lm_head, drafter, transport, or kernel metrics
- `results.py` — SQLite leaderboard
- `run_campaign.py` — budgeted unattended loop: establish champion reference -> generate -> build -> validate -> exact screen -> DEV gate -> optional production
- `db.py` — result and component-microbenchmark storage
- `prompts/canonical.json` — the historical LRU optimization prompt
- `prompts/dev.json` — 10-prompt multi-domain development validation suite

The real holdout should be created locally as `tuner/prompts/holdout.local.json`; that path is gitignored so tuning decisions cannot quietly turn the public holdout into another training benchmark.

## Correctness invariants

Before a candidate can advance:

1. `draft.accept_lens` must exist and be non-empty for a non-empty generation. Missing/renamed telemetry raises immediately; it can never become `rounds=0`.
2. Candidate `generation_tokens` must equal the frozen champion reference for that prompt.
3. Candidate `text_sha256` must equal the frozen champion reference for that prompt.
4. Repeated greedy canonical runs must have deterministic round counts.
5. Raw canonical round counts are compared only after the token-count invariant passes.
6. DEV efficiency is compared as `tokens/round`, so different prompt lengths and legitimate early EOS do not create fake round wins.

The campaign runner automatically establishes the frozen champion reference in SQLite before screening the first candidate. A new champion precision map gets a new stable reference ID.

## DEV is a gate

DEV is no longer a report for a human to eyeball. A candidate that passes the canonical screen must also satisfy the campaign's `[promotion]` DEV thresholds before an automated production run is allowed.

The current Qwen3.8 Q6 campaign requires all 10 DEV prompts to be present and exact, at least 8/10 prompts to stay within the configured per-prompt `tokens/round` floor, aggregate normalized efficiency and TPS to remain within their configured floors, and a claimed canonical round improvement to reproduce as strict `tokens/round` wins on a minimum number of DEV prompts.

## Campaigns and environment

`../campaigns/qwen38-q6.toml` describes the current Qwen3.8-27B Q6 target campaign. A new target quant or model variant should get its own TOML file rather than modifying this one in place.

Campaign paths use `~` / environment expansion rather than hardcoded usernames. Runtime flags also belong in the campaign:

```toml
[environment]
set = { MLX_QMV_FAST_M4 = "1" }
unset = ["MLX_QMV_FAST_M3"]
```

That matters when Q4/Q5/Q8 targets or other model families use different kernel paths.

## Fastest way to use it

From the repository root, in the Project27 MLX environment:

```bash
python -m tuner.run_campaign \
  --campaign campaigns/qwen38-q6.toml \
  --budget 8 \
  --screen-runs 1
```

On the first run the runner benchmarks the current champion once to establish the canonical exact-output reference. Each candidate then gets built, loader-validated, exactness-checked, and screened.

Add DEV:

```bash
python -m tuner.run_campaign \
  --campaign campaigns/qwen38-q6.toml \
  --budget 8 \
  --screen-runs 1 \
  --run-dev
```

The first DEV invocation establishes a frozen champion reference for all DEV prompts. Candidate DEV runs are then compared prompt-by-prompt and `promote.evaluate_dev_gate()` decides whether the candidate advances.

Production is automatically DEV-gated:

```bash
python -m tuner.run_campaign \
  --campaign campaigns/qwen38-q6.toml \
  --budget 3 \
  --screen-runs 1 \
  --run-production \
  --production-runs 10
```

`--run-production` implicitly requires/executes the DEV gate even if `--run-dev` is not specified.

## Component microbench plugins

The precision-map search can only attack drafter/acceptance work. Target-body kernel work needs a second axis. `microbench.py` is the first piece of that layer: it runs a campaign-defined command, parses one scalar metric, and stores it separately from end-to-end TPS.

A campaign can define a component such as:

```toml
[microbench.target_body]
command = ["python", "benchmarks/q38-target-body.py", "--model", "{target}"]
metric_regex = "component_sum_ms=(?P<value>[0-9.]+)"
metric_name = "target_body_sum"
unit = "ms"
direction = "minimize"
runs = 3
```

Then:

```bash
python -m tuner.microbench \
  --campaign campaigns/qwen38-q6.toml \
  --component target_body \
  --candidate-id kernel-EXPERIMENT
```

The plugin inherits campaign environment flags and supports per-component overrides. This lets the same database distinguish **mechanical wins** (component latency falls) from **trajectory wins** (acceptance/tokens-per-round improves).

The next kernel-search layer should enumerate existing MLX git refs/patches, rebuild safely from a clean worktree, run the relevant component plugins, enforce exact-output canaries, and only pay for end-to-end generation when the kernel metric improves.

## Search strategy

One-coordinate precision search remains useful for local cleanup, especially the newly exposed `group_size` axis, but it should not be mistaken for the whole route to 30 tok/s. At 27.556 tok/s and 138 canonical verification rounds, holding per-round cost constant would require roughly 127 rounds to reach 30 tok/s. That is a much larger acceptance shift than the manual precision campaign has produced so far.

So the intended search is two-dimensional:

1. **trajectory/acceptance search** — MTP precision and group-size maps, scored on canonical + DEV `tokens/round`;
2. **mechanical/kernel search** — target MLP/GDN/attention/lm_head and other hot components, scored first with component microbenchmarks and exactness canaries.

Pairwise interaction sweeps should be opened around useful boundaries, such as the observed FC precision/group-size interaction, rather than blindly multiplying every bit and group-size combination.

## Repeating on other target quants

Create separate campaign files such as:

```text
campaigns/qwen38-q4.toml
campaigns/qwen38-q5.toml
campaigns/qwen38-q8.toml
```

Kernel work can be inherited when quantization geometry/packing is compatible. Drafter precision maps should be re-optimized per target because speculative acceptance follows the target's quantized trajectory.

## Tests

The pure correctness/promotion gates can be checked without loading a model:

```bash
python -m unittest tests.test_tuner_gates
```

## Next implementation layer

Highest priority now:

1. check the existing target-body profiler into the repo and expose its MLP/GDN/attention metrics through `microbench.py`;
2. add safe MLX git-ref/patch kernel variant orchestration;
3. add a reusable quantized-tensor cache so precision candidate builds become assembly rather than repeated quantization;
4. add pairwise/interacting search around discovered boundaries;
5. keep explicit final holdout/champion promotion even after the rest of the funnel is unattended.
