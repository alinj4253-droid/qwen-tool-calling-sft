"""Canonical de-duplication hashing over (messages, tools).

v1 hashed ``messages`` only, which wrongly collapsed two samples that share
the same conversation but expose different tool sets.  The fingerprint now
covers BOTH messages and tools, with canonicalization that is invariant to:

  * JSON object key order
  * Unicode normalization (NFC)
  * whitespace differences inside strings

Two samples are duplicates iff their canonical messages AND canonical tools
are equal.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata

_WS_RE = re.compile(r"\s+", flags=re.UNICODE)


def canonicalize(obj):
    """Return a deterministic, whitespace/key-order invariant representation."""
    if isinstance(obj, str):
        s = unicodedata.normalize("NFC", obj)
        s = _WS_RE.sub(" ", s).strip()
        return s
    if isinstance(obj, dict):
        return {k: canonicalize(obj[k]) for k in sorted(obj.keys())}
    if isinstance(obj, (list, tuple)):
        return [canonicalize(x) for x in obj]
    return obj


def canonical_json(obj) -> str:
    """Stable JSON serialization: sorted keys, no extra whitespace, NFC text."""
    return json.dumps(canonicalize(obj), sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))


def messages_tools_fingerprint(messages: list, tools: list | None) -> str:
    """md5 fingerprint over canonical messages AND canonical tools."""
    payload = canonical_json(messages) + "||" + canonical_json(tools or [])
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def sample_fingerprint(sample: dict) -> str:
    return messages_tools_fingerprint(sample.get("messages", []), sample.get("tools"))


def dedup_samples(samples: list[dict]) -> tuple[list[dict], int]:
    """Return (deduped, n_duplicates) preserving first-seen order."""
    seen: set[str] = set()
    out: list[dict] = []
    n_dup = 0
    for s in samples:
        h = sample_fingerprint(s)
        if h in seen:
            n_dup += 1
            continue
        seen.add(h)
        out.append(s)
    return out, n_dup
