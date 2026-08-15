from __future__ import annotations

import hashlib
import json
import os
import tomllib
from pathlib import Path
from typing import Any


def expand_path(value: str | Path) -> Path:
    text = os.path.expandvars(str(value))
    return Path(text).expanduser().resolve()


def load_campaign(path: str | Path) -> dict[str, Any]:
    p = expand_path(path)
    with p.open("rb") as f:
        cfg = tomllib.load(f)
    cfg["_path"] = str(p)
    return cfg


def load_candidate(path: str | Path) -> dict[str, Any]:
    p = expand_path(path)
    data = json.loads(p.read_text())
    data["_path"] = str(p)
    return data


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def spec_id(modules: dict[str, dict[str, Any]]) -> str:
    digest = hashlib.sha256(canonical_json(modules).encode()).hexdigest()[:12]
    return f"cand-{digest}"


def module_alias(name: str) -> str:
    aliases = {
        "layers.0.mlp.gate_proj": "gate",
        "layers.0.mlp.up_proj": "up",
        "layers.0.mlp.down_proj": "down",
        "layers.0.self_attn.q_proj": "q",
        "layers.0.self_attn.k_proj": "k",
        "layers.0.self_attn.v_proj": "v",
        "layers.0.self_attn.o_proj": "o",
        "fc": "fc",
    }
    return aliases.get(name, name.replace("layers.0.", "").replace(".", "-"))


def describe_modules(modules: dict[str, dict[str, Any]]) -> str:
    parts: list[str] = []
    for name in sorted(modules):
        spec = modules[name]
        bits = spec.get("bits")
        group = spec.get("group_size")
        parts.append(f"{module_alias(name)}=q{bits}g{group}")
    return ",".join(parts)
