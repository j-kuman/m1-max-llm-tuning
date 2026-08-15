from __future__ import annotations

import argparse

from mlx_vlm.speculative import load_drafter

from .spec import expand_path, load_candidate


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify that a built candidate loads with the requested module bit/group map.")
    ap.add_argument("--draft", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--kind", default="mtp")
    args = ap.parse_args()

    candidate = load_candidate(args.candidate)
    draft, kind = load_drafter(str(expand_path(args.draft)), kind=args.kind)
    modules = dict(draft.named_modules())
    failures = 0

    print("draft kind:", kind)
    for name, expected in candidate["modules"].items():
        if name not in modules:
            print("MISSING:", name)
            failures += 1
            continue
        mod = modules[name]
        bits = getattr(mod, "bits", None)
        group = getattr(mod, "group_size", None)
        ok = bits == int(expected["bits"]) and group == int(expected["group_size"])
        print(
            f"{'OK' if ok else 'FAIL':4s} {name:42s} "
            f"bits={bits} group={group} "
            f"expected=q{expected['bits']}g{expected['group_size']}"
        )
        failures += 0 if ok else 1

    if failures:
        raise SystemExit(f"validation failed: {failures} module(s)")
    print("validation passed")


if __name__ == "__main__":
    main()
