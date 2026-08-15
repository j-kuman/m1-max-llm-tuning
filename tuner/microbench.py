from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
from pathlib import Path
from typing import Any

from .db import connect, insert_microbench
from .spec import expand_path, load_campaign


def _format(value: str, ctx: dict[str, str]) -> str:
    return value.format(**ctx)


def run_component(
    cfg: dict[str, Any],
    component: str,
    *,
    candidate: str,
    draft: str | None,
    db_path: str,
    runs_override: int | None = None,
) -> list[float]:
    """Run one command-backed component microbenchmark.

    A component definition lives under [microbench.NAME] in a campaign TOML.
    The command's stdout is parsed with metric_regex, which must contain a
    named group `value`. This deliberately keeps the tuner independent of any
    particular Metal/MLX benchmark script: target MLP, GDN, lm_head, FC, TP
    transport, or future kernel experiments can all expose one scalar metric.
    """
    sections = cfg.get("microbench", {})
    if component not in sections:
        raise KeyError(f"microbench component {component!r} is not defined in campaign")
    spec = sections[component]

    command = spec.get("command")
    if not isinstance(command, list) or not command:
        raise ValueError(f"microbench.{component}.command must be a non-empty array")
    pattern = re.compile(str(spec["metric_regex"]), re.MULTILINE)
    if "value" not in pattern.groupindex:
        raise ValueError(f"microbench.{component}.metric_regex must define (?P<value>...)")

    campaign = cfg["campaign"]
    ctx = {
        "target": str(expand_path(campaign["target"])),
        "draft": str(expand_path(draft)) if draft else "",
        "candidate": candidate,
        "campaign": campaign["name"],
        "repo": str(Path.cwd()),
    }
    cmd = [_format(str(part), ctx) for part in command]
    cwd = expand_path(_format(str(spec.get("cwd", ".")), ctx))
    env = os.environ.copy()
    for key, value in spec.get("env", {}).items():
        env[str(key)] = _format(str(value), ctx)
    for key in spec.get("unset_env", []):
        env.pop(str(key), None)

    metric_name = str(spec.get("metric_name", component))
    unit = str(spec.get("unit", "ms"))
    direction = str(spec.get("direction", "minimize"))
    if direction not in {"minimize", "maximize"}:
        raise ValueError("microbench direction must be minimize or maximize")

    runs = int(runs_override or spec.get("runs", 3))
    if runs <= 0:
        raise ValueError("microbench runs must be positive")

    conn = connect(db_path)
    values: list[float] = []

    for i in range(1, runs + 1):
        print("$", " ".join(cmd), flush=True)
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
        matches = list(pattern.finditer(proc.stdout))
        if not matches:
            raise RuntimeError(
                f"microbench {component!r} could not parse metric from stdout with {pattern.pattern!r}"
            )
        value = float(matches[-1].group("value"))
        values.append(value)
        insert_microbench(conn, {
            "campaign": campaign["name"],
            "candidate": candidate,
            "component": component,
            "run_index": i,
            "metric_name": metric_name,
            "metric_value": value,
            "unit": unit,
            "direction": direction,
            "command_json": json.dumps(cmd),
            "stdout_sha256": hashlib.sha256(proc.stdout.encode("utf-8")).hexdigest(),
            "notes": spec.get("notes"),
        })
        print(f"MICROBENCH {component} run {i}: {metric_name}={value:g} {unit}")

    print("\n========== MICROBENCH SUMMARY ==========")
    print("component:", component)
    print("candidate:", candidate)
    print("runs:", runs)
    print("mean:", f"{statistics.mean(values):.6f} {unit}")
    print("median:", f"{statistics.median(values):.6f} {unit}")
    print("best:", f"{(min(values) if direction == 'minimize' else max(values)):.6f} {unit}")
    return values


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a campaign-defined component microbenchmark plugin.")
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--component", required=True)
    ap.add_argument("--candidate-id", required=True)
    ap.add_argument("--draft")
    ap.add_argument("--runs", type=int)
    ap.add_argument("--db", default="results/tuning.sqlite")
    args = ap.parse_args()

    cfg = load_campaign(args.campaign)
    run_component(
        cfg,
        args.component,
        candidate=args.candidate_id,
        draft=args.draft,
        db_path=args.db,
        runs_override=args.runs,
    )


if __name__ == "__main__":
    main()
