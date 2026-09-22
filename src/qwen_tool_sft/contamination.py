"""Train / benchmark decontamination checks.

Three checks (task P0-3):

  1. Exact query match          - identical canonical user-turn text(s)
  2. Exact query + tool schema  - identical user text(s) AND identical tools
  3. Near duplicate             - char 4-gram cosine similarity >= threshold

Near-duplicate search uses an inverted index over shingles so it scales to a
~100k-row training file while still reporting inspectable top matches for
every benchmark case.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Iterable

from .dedup import canonical_json, canonicalize

_N = 4
_WS = re.compile(r"\s+", flags=re.UNICODE)


def user_query_text(sample: dict) -> str:
    """Concatenate the genuine user turns (tool responses excluded)."""
    parts = []
    for m in sample.get("messages", []):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                parts.append(c)
    return "\n".join(parts)


def query_key(sample: dict) -> str:
    """Canonical key over user-turn texts."""
    return canonical_json([canonicalize(m.get("content", ""))
                           for m in sample.get("messages", [])
                           if m.get("role") == "user"])


def tools_key(sample: dict) -> str:
    return canonical_json(sample.get("tools") or [])


def normalize_text(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


def char_shingles(text: str, n: int = _N) -> Counter:
    t = normalize_text(text)
    t = re.sub(r"\s+", "", t)  # char n-grams ignore whitespace entirely
    if len(t) <= n:
        return Counter({t: 1}) if t else Counter()
    return Counter(t[i:i + n] for i in range(len(t) - n + 1))


def _norm(counter: Counter) -> float:
    return math.sqrt(sum(v * v for v in counter.values()))


def cosine(a: Counter, b: Counter, na: float | None = None,
           nb: float | None = None) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = na or _norm(a)
    nb = nb or _norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class NearDuplicateIndex:
    """Inverted index of training shingles for fast cosine neighbours."""

    def __init__(self, train_samples: Iterable[dict]):
        self.queries = [normalize_text(user_query_text(s)) for s in train_samples]
        self.counters = [char_shingles(q) for q in self.queries]
        self.norms = [_norm(c) for c in self.counters]
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, c in enumerate(self.counters):
            for gram, cnt in c.items():
                self.postings[gram].append((i, cnt))

    def top_matches(self, bench_sample: dict, top_k: int = 5,
                    min_score: float = 0.0) -> list[tuple[int, float]]:
        c = char_shingles(user_query_text(bench_sample))
        nc = _norm(c)
        if nc == 0:
            return []
        dots: dict[int, float] = defaultdict(float)
        for gram, cnt in c.items():
            for ti, tcnt in self.postings.get(gram, ()):
                dots[ti] += cnt * tcnt
        scored = [(ti, d / (nc * self.norms[ti])) for ti, d in dots.items()
                  if self.norms[ti] > 0]
        scored = [x for x in scored if x[1] >= min_score]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


def run_contamination(train_samples: list[dict], bench_samples: list[dict],
                      threshold: float = 0.8, top_k: int = 5) -> dict:
    """Run all three checks. Returns a JSON-serializable report dict."""
    train_q = {}
    train_qt = defaultdict(list)
    for ti, s in enumerate(train_samples):
        qk, tk = query_key(s), tools_key(s)
        train_q.setdefault(qk, []).append(ti)
        train_qt[(qk, tk)].append(ti)

    index = NearDuplicateIndex(train_samples)

    exact_matches, near_duplicates, top_matches = [], [], []
    n_exact_query = n_exact_query_tools = n_near = 0

    for bi, b in enumerate(bench_samples):
        bid = b.get("id", f"bench-{bi}")
        qk, tk = query_key(b), tools_key(b)

        q_hits = train_q.get(qk, [])
        qt_hits = train_qt.get((qk, tk), [])
        if q_hits:
            n_exact_query += 1
            exact_matches.append({
                "bench_id": bid, "category": b.get("category", ""),
                "check": "exact_query",
                "bench_query": user_query_text(b)[:500],
                "train_indices": q_hits[:10],
            })
        if qt_hits:
            n_exact_query_tools += 1
            exact_matches.append({
                "bench_id": bid, "category": b.get("category", ""),
                "check": "exact_query_and_tools",
                "bench_query": user_query_text(b)[:500],
                "train_indices": qt_hits[:10],
            })

        tops = index.top_matches(b, top_k=top_k)
        for rank, (ti, score) in enumerate(tops):
            row = {
                "bench_id": bid, "category": b.get("category", ""),
                "rank": rank, "similarity": round(score, 4),
                "bench_query": user_query_text(b)[:300],
                "train_query": index.queries[ti][:300],
                "train_index": ti,
            }
            top_matches.append(row)
            if score >= threshold:
                near_duplicates.append(row)
        if tops and tops[0][1] >= threshold:
            n_near += 1

    by_category: dict[str, int] = defaultdict(int)
    for r in near_duplicates:
        by_category[r["category"]] += 1

    return {
        "summary": {
            "n_train": len(train_samples),
            "n_benchmark": len(bench_samples),
            "near_threshold": threshold,
            "n_exact_query_match": n_exact_query,
            "n_exact_query_and_tools_match": n_exact_query_tools,
            "n_near_duplicate_cases": n_near,
            "n_near_duplicate_pairs": len(near_duplicates),
            "near_by_category": dict(by_category),
            "exact_train_duplicate_required_zero": n_exact_query == 0,
        },
        "exact_matches": exact_matches,
        "near_duplicates": near_duplicates,
        "top_matches": top_matches,
    }
