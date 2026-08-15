from __future__ import annotations

from dataclasses import dataclass
import sqlite3
from typing import Any

from .spec import spec_id


@dataclass(frozen=True)
class ReferenceRun:
    prompt_id: str
    tokens: int
    text_sha256: str


def champion_reference_id(cfg: dict[str, Any]) -> str:
    """Stable DB identity for the campaign's current champion reference."""
    return f"champion-{spec_id(cfg['champion']['modules']).removeprefix('cand-')}"


def extract_round_stats(draft: Any, tokens: int) -> tuple[int, float]:
    """Return (rounds, tokens_per_round), failing closed if telemetry disappeared.

    accept_lens is part of the MTP drafter instrumentation we rely on for tuning.
    A missing/renamed attribute must never silently look like zero rounds because
    that would make a broken telemetry path appear to be an enormous improvement.
    """
    if not hasattr(draft, "accept_lens"):
        raise RuntimeError(
            "drafter no longer exposes accept_lens; cannot score verification rounds safely"
        )

    accepts = getattr(draft, "accept_lens")
    if accepts is None:
        raise RuntimeError("drafter.accept_lens is None; refusing to score candidate")

    try:
        accepts = list(accepts)
    except TypeError as exc:
        raise RuntimeError("drafter.accept_lens is not iterable") from exc

    rounds = len(accepts)
    if tokens > 0 and rounds <= 0:
        raise RuntimeError(
            f"generated {tokens} tokens but observed {rounds} speculative rounds; telemetry is invalid"
        )

    return rounds, (tokens / rounds if rounds else 0.0)


def load_reference(
    conn: sqlite3.Connection,
    campaign: str,
    reference_candidate: str,
    prompt_id: str,
) -> ReferenceRun | None:
    rows = conn.execute(
        """
        SELECT tokens, text_sha256
        FROM runs
        WHERE campaign = ? AND candidate = ? AND stage = 'reference' AND prompt_id = ?
        ORDER BY id
        """,
        (campaign, reference_candidate, prompt_id),
    ).fetchall()

    if not rows:
        return None

    token_values = {int(r[0]) for r in rows}
    hash_values = {str(r[1]) for r in rows}
    if len(token_values) != 1 or len(hash_values) != 1:
        raise RuntimeError(
            f"reference for {prompt_id!r} is internally inconsistent: "
            f"tokens={sorted(token_values)} hashes={len(hash_values)}"
        )

    return ReferenceRun(
        prompt_id=prompt_id,
        tokens=next(iter(token_values)),
        text_sha256=next(iter(hash_values)),
    )


def require_exact_match(
    reference: ReferenceRun,
    *,
    tokens: int,
    text_sha256: str,
    candidate: str,
) -> None:
    """Hard correctness gate for exact greedy speculative decoding."""
    if tokens != reference.tokens:
        raise RuntimeError(
            f"EXACTNESS FAILURE for {candidate}/{reference.prompt_id}: "
            f"generated token count {tokens} != reference {reference.tokens}"
        )
    if text_sha256 != reference.text_sha256:
        raise RuntimeError(
            f"EXACTNESS FAILURE for {candidate}/{reference.prompt_id}: "
            f"text sha256 {text_sha256} != reference {reference.text_sha256}"
        )
