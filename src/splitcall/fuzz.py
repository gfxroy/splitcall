"""Chunk-boundary fuzzer.

Every case is replayed as two-chunk cuts at every offset, as fixed chunk
sizes 1, 2, 3, 5 and 8, and as every truncation prefix. Splits are Unicode
code points, not UTF-8 bytes and not tokenizer pieces.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field

from splitcall.adapters.runtime import accepts, all_parsers, probe_mistral_prev11
from splitcall.corpus import corpus, formats
from splitcall.invariants import judge
from splitcall.shrink import shrink_text
from splitcall.types import Case, Split

SPLIT_INVARIANTS = {
    "no_crash",
    "stream_equals_batch",
    "count",
    "index_ordinal",
    "index_stable",
    "id_stable",
    "args_json",
    "no_markup_leak",
    "type_function",
    "args_are_strings",
    "finish_reason",
}


@dataclass
class Example:
    parser: str
    runtime: str
    format: str
    case_id: str
    invariant: str
    detail: str
    text: str
    split: str
    family: str
    batch_summary: str
    stream_summary: str
    shrunk: str = ""


@dataclass
class Matrix:
    cases: int
    parsers: list[str]
    formats: list[str]
    splits: int = 0
    elapsed_sec: float = 0.0
    cells: dict = field(default_factory=dict)
    invariants: dict = field(default_factory=dict)
    examples: list[Example] = field(default_factory=list)
    extra: list[dict] = field(default_factory=list)
    repair_failures: int = 0


def iter_splits(text: str, quick: bool):
    n = len(text)
    if quick:
        points = sorted({0, n // 2, n})
        for offset in points:
            yield Split("cut", at=offset), [text[:offset], text[offset:]], text, True
        yield Split("chunks", size=1), (list(text) or [""]), text, True
        for offset in points:
            yield Split("trunc", at=offset), [text[:offset]], text[:offset], False
        return
    for offset in range(n + 1):
        yield Split("cut", at=offset), [text[:offset], text[offset:]], text, True
    for size in (1, 2, 3, 5, 8):
        chunks = [text[i : i + size] for i in range(0, n, size)] or [""]
        yield Split("chunks", size=size), chunks, text, True
    for offset in range(n + 1):
        prefix = text[:offset]
        yield Split("trunc", at=offset), [prefix], prefix, False


def _summary(result) -> str:
    if result.error:
        return "error: " + result.error[:160]
    calls = []
    for call in result.calls:
        args = call.arguments if len(call.arguments) <= 60 else call.arguments[:57] + "..."
        calls.append(f"{call.index}:{call.name}:{args}")
    content = result.content if len(result.content) <= 40 else result.content[:37] + "..."
    return f"finish={result.finish_reason} content={content!r} calls=[{', '.join(calls)}]"


def _family(split: Split) -> str:
    if split.mode == "chunks":
        return f"chunks:{split.size}"
    return split.mode


def _cell():
    return {"pass": 0, "fail": 0, "gap": 0, "skip": 0}


def run_fuzz(
    *,
    quick: bool = False,
    parser_names: set[str] | None = None,
    shrink: bool = True,
    max_repros: int = 12,
) -> Matrix:
    cases = corpus()
    parsers = [
        parser for parser in all_parsers() if parser_names is None or parser.name in parser_names
    ]
    matrix = Matrix(
        cases=len(cases),
        parsers=[parser.name for parser in parsers],
        formats=formats(),
    )
    counts: dict[tuple[str, str], dict] = defaultdict(_cell)
    inv_counts: dict[tuple[str, str], dict] = defaultdict(_cell)
    seen: dict[tuple[str, str, str, str], Example] = {}
    started = time.perf_counter()
    # Upstream parsers log a traceback per rejected chunk. That is their
    # behavior, not a splitcall error, and it drowns the run.
    logging.disable(logging.ERROR)
    try:
        for parser in parsers:
            for case in cases:
                if not accepts(parser, case):
                    continue
                batch_cache: dict[str, object] = {}
                for split, chunks, batch_text, full_text in iter_splits(case.text, quick):
                    matrix.splits += 1
                    if full_text and case.text in batch_cache:
                        batch = batch_cache[case.text]
                    else:
                        batch = parser.parse_batch(batch_text, case.format)
                        if full_text:
                            batch_cache[case.text] = batch
                    stream = parser.parse_stream(chunks, case.format)
                    verdicts = judge(case, batch, stream, full_text=full_text)
                    for verdict in verdicts:
                        # The oracle describes the whole sample. Count it once.
                        if verdict.name.startswith("oracle_") and not (
                            split.mode == "cut" and split.at == 0
                        ):
                            continue
                        counts[(parser.name, case.format)][verdict.status] += 1
                        inv_counts[(parser.name, verdict.name)][verdict.status] += 1
                        if verdict.status != "fail":
                            continue
                        if parser.name == "splitcall-repair":
                            matrix.repair_failures += 1
                        key = (parser.name, case.format, case.id, verdict.name)
                        if key in seen:
                            continue
                        seen[key] = Example(
                            parser=parser.name,
                            runtime=parser.runtime,
                            format=case.format,
                            case_id=case.id,
                            invariant=verdict.name,
                            detail=verdict.detail,
                            text=case.text,
                            split=split.label(),
                            family=_family(split),
                            batch_summary=_summary(batch),
                            stream_summary=_summary(stream),
                        )
        if shrink and seen:
            matrix.examples = _shrink_examples(parsers, seen, max_repros)
        else:
            matrix.examples = list(seen.values())[:max_repros]
        matrix.cells = {name: {} for name in matrix.parsers}
        for (parser_name, fmt), bucket in counts.items():
            matrix.cells[parser_name][fmt] = dict(bucket)
        matrix.invariants = {name: {} for name in matrix.parsers}
        for (parser_name, inv_name), bucket in inv_counts.items():
            matrix.invariants[parser_name][inv_name] = dict(bucket)
        probe_text = '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Paris"}}]'
        try:
            matrix.extra.append(
                {"probe": "vllm-mistral-prev11-engine", **probe_mistral_prev11(probe_text)}
            )
        except Exception as exc:
            matrix.extra.append(
                {
                    "probe": "vllm-mistral-prev11-engine",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        matrix.elapsed_sec = round(time.perf_counter() - started, 3)
        return matrix
    finally:
        logging.disable(logging.NOTSET)


def _parser_map(parsers):
    return {parser.name: parser for parser in parsers}


def _shrink_examples(parsers, seen: dict, max_repros: int) -> list[Example]:
    by_parser = _parser_map(parsers)
    ranked = sorted(seen.values(), key=lambda item: (item.parser, item.invariant, item.case_id))
    chosen: list[Example] = []
    used: set[tuple[str, str]] = set()
    for item in ranked:
        if item.invariant not in SPLIT_INVARIANTS:
            continue
        sig = (item.parser, item.invariant)
        if sig in used:
            continue
        used.add(sig)
        chosen.append(item)
        if len(chosen) >= max_repros:
            break
    for item in chosen:
        parser = by_parser[item.parser]
        case = _case_for(item)
        item.shrunk = shrink_text(
            item.text,
            _predicate(parser, case, item.invariant, item.family),
            max_tests=400,
        )
        # The fields above were taken from the original case. Replace them
        # with one split of the shrunk string that still fails the same check.
        captured = _first_failure(parser, case, item.invariant, item.family, item.shrunk)
        if captured is not None:
            item.detail, item.split, item.batch_summary, item.stream_summary = captured
    return chosen


def _case_for(item: Example) -> Case:
    for case in corpus():
        if case.id == item.case_id:
            return case
    raise KeyError(item.case_id)


def _iter_family(text: str, family: str):
    """Replay the split family that first exposed the failure."""
    if family.startswith("chunks:"):
        size = int(family.split(":")[1])
        chunks = [text[i : i + size] for i in range(0, len(text), size)] or [""]
        yield Split("chunks", size=size), chunks, text, True
        return
    if family == "trunc":
        for offset in range(len(text) + 1):
            yield Split("trunc", at=offset), [text[:offset]], text[:offset], False
        return
    for offset in range(len(text) + 1):
        yield Split("cut", at=offset), [text[:offset], text[offset:]], text, True


def _first_failure(parser, case: Case, invariant: str, family: str, text: str):
    """One failing split of `text`, or None if the invariant no longer fails."""
    for split, chunks, batch_text, full in _iter_family(text, family):
        batch = parser.parse_batch(batch_text, case.format)
        stream = parser.parse_stream(chunks, case.format)
        for verdict in judge(case, batch, stream, full_text=full and text == case.text):
            if verdict.name == invariant and verdict.status == "fail":
                return verdict.detail, split.label(), _summary(batch), _summary(stream)
    return None


def _predicate(parser, case: Case, invariant: str, family: str):
    def pred(text: str) -> bool:
        return _first_failure(parser, case, invariant, family, text) is not None

    return pred
