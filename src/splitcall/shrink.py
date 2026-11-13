"""Delta-debugging shrinker for one failing input.

The search replays a single split family. Grain starts at 8 characters and
drops to 1 once the string is short. The test budget stops the search.
"""

from __future__ import annotations

from collections.abc import Callable


def shrink_text(text: str, predicate: Callable[[str], bool], max_tests: int = 400) -> str:
    """Return a smaller string that still satisfies `predicate`.

    `predicate` is true when the failure is still present. If `text` itself
    does not fail, it is returned unchanged.
    """
    budget = [max_tests]

    def check(candidate: str) -> bool:
        if budget[0] <= 0:
            return False
        budget[0] -= 1
        return predicate(candidate)

    if not text or not check(text):
        return text

    if len(text) > 24:
        grain = 8
        blocks = [text[i : i + grain] for i in range(0, len(text), grain)]
        blocks = _ddmin(blocks, lambda parts: check("".join(parts)), budget)
        text = "".join(blocks)
    if len(text) <= 160 and budget[0] > 0:
        chars = _ddmin(list(text), lambda parts: check("".join(parts)), budget)
        text = "".join(chars)
    return text


def _ddmin(parts: list, predicate: Callable[[list], bool], budget: list[int]) -> list:
    """Hildebrandt and Zeller's ddmin over a list of pieces."""
    if len(parts) < 2:
        return parts
    data = list(parts)
    n = 2
    while len(data) >= 2 and budget[0] > 0:
        subset_len = len(data) // n
        if subset_len < 1:
            break
        reduced = False
        for i in range(n):
            start = i * subset_len
            end = len(data) if i == n - 1 else (i + 1) * subset_len
            complement = data[:start] + data[end:]
            subset = data[start:end]
            if len(complement) < len(data) and predicate(complement):
                data = complement
                n = max(n - 1, 2)
                reduced = True
                break
            if subset and len(subset) < len(data) and predicate(subset):
                data = subset
                n = max(n - 1, 2)
                reduced = True
                break
        if reduced:
            continue
        if n >= len(data):
            break
        n = min(n * 2, len(data))
    return data
