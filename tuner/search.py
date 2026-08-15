from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from .spec import describe_modules, load_campaign, spec_id


def generate_neighbors(cfg: dict) -> list[dict]:
    champion = cfg["champion"]
    base_modules = champion["modules"]
    search = cfg.get("search", {})
    out: list[dict] = []
    seen: set[str] = set()

    for module, space in search.items():
        if module not in base_modules:
            continue
        current = base_modules[module]

        for bits in space.get("bits", []):
            if bits == current["bits"]:
                continue
            modules = copy.deepcopy(base_modules)
            modules[module]["bits"] = bits
            cid = spec_id(modules)
            if cid not in seen:
                seen.add(cid)
                out.append({
                    "id": cid,
                    "base_draft": champion["draft"],
                    "modules": modules,
                    "mutation": {"module": module, "field": "bits", "value": bits},
                })

        for group in space.get("group_size", []):
            if group == current["group_size"]:
                continue
            modules = copy.deepcopy(base_modules)
            modules[module]["group_size"] = group
            cid = spec_id(modules)
            if cid not in seen:
                seen.add(cid)
                out.append({
                    "id": cid,
                    "base_draft": champion["draft"],
                    "modules": modules,
                    "mutation": {"module": module, "field": "group_size", "value": group},
                })

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate one-coordinate neighbors around a campaign champion.")
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--output", default="candidates/generated")
    args = ap.parse_args()

    cfg = load_campaign(args.campaign)
    name = cfg["campaign"]["name"]
    root = Path(args.output) / name
    root.mkdir(parents=True, exist_ok=True)

    neighbors = generate_neighbors(cfg)
    for cand in neighbors:
        path = root / f"{cand['id']}.json"
        path.write_text(json.dumps(cand, indent=2) + "\n")
        print(cand["id"], cand["mutation"], describe_modules(cand["modules"]))

    print(f"\nGenerated {len(neighbors)} candidates in {root}")


if __name__ == "__main__":
    main()
