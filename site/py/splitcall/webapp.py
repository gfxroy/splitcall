"""Browser entry point for the Pages checker.

Pyodide loads this module. It does not import the vendored vLLM or SGLang
parsers. The repaired read is ``parse_repair``. The split reader parses each
chunk on its own and concatenates the results, which is the same function as
the one-piece read when the text arrives as a single chunk.
"""

from __future__ import annotations

import json

from splitcall.corpus import corpus
from splitcall.invariants import judge
from splitcall.repair import parse_repair, parse_repair_chunks
from splitcall.shrink import shrink_text
from splitcall.types import Case, NormCall, Normalized

# Long pastes still get a repaired read. The exhaustive split walk stays
# bounded so the page remains responsive.
FUZZ_LIMIT = 360
REPAIR_LIMIT = 8000

PRESETS: list[tuple[str, str]] = [
    ("hermes-basic", "Weather, Hermes tags"),
    ("hermes-prose", "A sentence, then a Hermes call"),
    ("hermes-trunc-args", "Hermes call cut off inside the city"),
    ("hermes-prose-tag", "Hermes tag mentioned in a sentence"),
    ("qwen-basic", "Weather, Qwen XML"),
    ("mistral-array", "Weather, Mistral JSON list"),
    ("mistral-compact", "Weather, Mistral compact form"),
    ("mistral-prose", "Mistral marker mentioned in a sentence"),
    ("llama-tag", "Weather, Llama tag"),
    ("llama-python-dict", "Llama tag with a Python dict"),
    ("harmony-basic", "Weather, Harmony channel"),
    ("harmony-no-constrain", "Harmony channel without constrain"),
]

FORMAT_LABELS = {
    "hermes": "Hermes — JSON inside <tool_call>",
    "qwen": "Qwen — <function> and <parameter> tags",
    "mistral": "Mistral — [TOOL_CALLS]",
    "llama": "Llama — <|python_tag|> or a JSON object",
    "harmony": "Harmony — <|channel|> blocks",
}

_SPLIT_INVARIANTS = (
    "no_crash",
    "stream_equals_batch",
    "count",
    "args_json",
    "no_markup_leak",
    "finish_reason",
)


def presets_json() -> str:
    by_id = {case.id: case for case in corpus()}
    rows = []
    for case_id, label in PRESETS:
        case = by_id[case_id]
        rows.append(
            {
                "id": case.id,
                "label": label,
                "format": case.format,
                "text": case.text,
            }
        )
    return json.dumps(rows, ensure_ascii=False)


def analyze(fmt: str, text: str) -> str:
    """Return a JSON report for one pasted sample."""
    if fmt not in FORMAT_LABELS:
        return json.dumps({"error": "Choose a format."})
    if text is None:
        text = ""
    if len(text) > REPAIR_LIMIT:
        return json.dumps(
            {
                "error": f"That text is longer than {REPAIR_LIMIT} characters. Paste a shorter sample.",
            }
        )
    if not text.strip():
        return json.dumps({"error": "Paste a model output, or pick an example."})

    repaired = parse_repair(fmt, text)
    fuzz_text = text if len(text) <= FUZZ_LIMIT else text[:FUZZ_LIMIT]
    repair_fuzz = _fuzz(fmt, fuzz_text, parse_repair_chunks, parse_repair, shrink=False)
    split_fuzz = _fuzz(fmt, fuzz_text, _parse_pieces, _parse_one_piece, shrink=True)
    return json.dumps(
        {
            "format": fmt,
            "format_label": FORMAT_LABELS[fmt],
            "characters": len(text),
            "fuzz_characters": len(fuzz_text),
            "truncated_for_fuzz": len(text) > FUZZ_LIMIT,
            "repair": _repair_payload(repaired),
            "repair_fuzz": repair_fuzz,
            "split_reader": split_fuzz,
        },
        ensure_ascii=False,
    )


def _repair_payload(result: Normalized) -> dict:
    calls = []
    for call in result.calls:
        pretty = call.arguments
        try:
            pretty = json.dumps(json.loads(call.arguments), ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            pass
        calls.append(
            {
                "index": call.index,
                "id": call.id,
                "name": call.name,
                "arguments": call.arguments,
                "arguments_pretty": pretty,
                "type": call.type,
            }
        )
    reason = result.finish_reason or ""
    labels = {
        "tool_calls": "Tool call",
        "length": "A tool call started and was cut off",
        "stop": "No tool call",
    }
    return {
        "content": result.content,
        "reasoning": result.reasoning,
        "finish_reason": reason,
        "finish_label": labels.get(reason, reason or "Unknown"),
        "error": result.error,
        "calls": calls,
    }


def _parse_one_piece(fmt: str, text: str) -> Normalized:
    return _parse_pieces(fmt, [text])


def _parse_pieces(fmt: str, chunks: list[str]) -> Normalized:
    """Parse every chunk alone. A marker cut between chunks is invisible."""
    content: list[str] = []
    reasoning: list[str] = []
    calls: list[NormCall] = []
    finish = "stop"
    for chunk in chunks:
        one = parse_repair(fmt, chunk)
        content.append(one.content)
        if one.reasoning:
            reasoning.append(one.reasoning)
        for call in one.calls:
            calls.append(
                NormCall(
                    index=len(calls),
                    name=call.name,
                    arguments=call.arguments,
                    id=f"call_{len(calls):04d}",
                    type="function",
                )
            )
        if one.finish_reason == "tool_calls":
            finish = "tool_calls"
        elif one.finish_reason == "length" and finish != "tool_calls":
            finish = "length"
    return Normalized(
        content="".join(content),
        calls=calls,
        finish_reason=finish,
        reasoning="".join(reasoning),
        assigns_id=True,
        assigns_finish=True,
        assigns_type=True,
    )


def _iter_splits(text: str):
    n = len(text)
    for offset in range(n + 1):
        yield "cut", f"cut@{offset}", [text[:offset], text[offset:]], text
    for size in (1, 2, 3, 5, 8):
        chunks = [text[i : i + size] for i in range(0, n, size)] or [""]
        yield "chunks", f"chunks:{size}", chunks, text
    for offset in range(n + 1):
        prefix = text[:offset]
        yield "trunc", f"trunc@{offset}", [prefix], prefix


def _plain(result: Normalized) -> str:
    if result.error:
        return "error: " + result.error[:160]
    calls = []
    for call in result.calls:
        args = call.arguments if len(call.arguments) <= 72 else call.arguments[:69] + "..."
        calls.append(f"{call.name} {args}")
    content = result.content if len(result.content) <= 72 else result.content[:69] + "..."
    prose = f"prose {content!r}" if content else "no prose"
    found = "calls " + "; ".join(calls) if calls else "no tool call"
    return f"{prose}; {found}"


def _fuzz(fmt: str, text: str, parse_stream, parse_batch, *, shrink: bool) -> dict:
    case = Case(id="input", format=fmt, text=text, complete=False)
    counts = {"pass": 0, "fail": 0, "gap": 0, "skip": 0}
    splits = 0
    # First failing split per invariant. A character split is preferred
    # because the shrunk repro is cheaper to check and easier to read.
    found: dict[str, dict] = {}
    full_batch: dict[str, Normalized] = {}
    for family, label, chunks, batch_text in _iter_splits(text):
        splits += 1
        if batch_text in full_batch:
            batch = full_batch[batch_text]
        else:
            batch = parse_batch(fmt, batch_text)
            full_batch[batch_text] = batch
        stream = parse_stream(fmt, chunks)
        for verdict in judge(case, batch, stream, full_text=False):
            counts[verdict.status] = counts.get(verdict.status, 0) + 1
            if verdict.status != "fail" or verdict.name not in _SPLIT_INVARIANTS:
                continue
            prefer = label == "chunks:1"
            current = found.get(verdict.name)
            if current is not None and (current["prefer"] or not prefer):
                continue
            found[verdict.name] = {
                "prefer": prefer,
                "invariant": verdict.name,
                "family": "chunks:1" if prefer else family,
                "split": label,
                "detail": verdict.detail,
                "one_piece": _plain(batch),
                "split_up": _plain(stream),
            }
    repros: list[dict] = []
    if shrink and found:
        order = ["count", "stream_equals_batch", "args_json", "no_markup_leak", "no_crash", "finish_reason"]
        for name in order:
            item = found.get(name)
            if item is None:
                continue
            if len(repros) >= 3:
                break
            repros.append(_shrink_one(fmt, text, item, parse_stream, parse_batch))
    return {
        "splits": splits,
        "fail": counts["fail"],
        "pass": counts["pass"],
        "gap": counts["gap"],
        "repros": repros,
    }


def _shrink_one(fmt: str, text: str, item: dict, parse_stream, parse_batch) -> dict:
    family = item["family"]
    invariant = item["invariant"]

    def pred(candidate: str) -> bool:
        # The failure has to come from the split. A candidate that already
        # fails as one piece is a different bug, and shrinking toward it
        # would pair the short text with the wrong parse.
        alone = parse_batch(fmt, candidate)
        case = Case(id="input", format=fmt, text=candidate, complete=False)
        alone_bad = any(
            verdict.name == invariant and verdict.status == "fail"
            for verdict in judge(case, alone, alone, full_text=False)
        )
        if alone_bad:
            return False
        return _first_failure(fmt, candidate, family, invariant, parse_stream, parse_batch) is not None

    shrunk = shrink_text(text, pred, max_tests=160)
    captured = _first_failure(fmt, shrunk, family, invariant, parse_stream, parse_batch)
    if captured is None:
        captured = {
            "detail": item["detail"],
            "split": item["split"],
            "one_piece": item["one_piece"],
            "split_up": item["split_up"],
            "explain": "",
        }
    return {
        "invariant": invariant,
        "label": _invariant_label(invariant),
        "family": family,
        "split": captured["split"],
        "detail": captured["detail"],
        "shrunk": shrunk,
        "one_piece": captured["one_piece"],
        "split_up": captured["split_up"],
        "explain": captured["explain"],
    }


def _first_failure(fmt, text, family, invariant, parse_stream, parse_batch):
    case = Case(id="input", format=fmt, text=text, complete=False)
    for label, chunks, batch_text in _family_splits(text, family):
        batch = parse_batch(fmt, batch_text)
        stream = parse_stream(fmt, chunks)
        for verdict in judge(case, batch, stream, full_text=False):
            if verdict.name == invariant and verdict.status == "fail":
                return {
                    "detail": verdict.detail,
                    "split": label,
                    "one_piece": _plain(batch),
                    "split_up": _plain(stream),
                    "explain": _explain(invariant, batch, stream),
                }
    return None


def _family_splits(text: str, family: str):
    if family.startswith("chunks:"):
        size = int(family.split(":")[1])
        chunks = [text[i : i + size] for i in range(0, n, size)] if (n := len(text)) else [""]
        if not chunks:
            chunks = [""]
        yield f"chunks:{size}", chunks, text
        return
    if family == "trunc":
        for offset in range(len(text) + 1):
            prefix = text[:offset]
            yield f"trunc@{offset}", [prefix], prefix
        return
    for offset in range(len(text) + 1):
        yield f"cut@{offset}", [text[:offset], text[offset:]], text


def _invariant_label(name: str) -> str:
    return {
        "count": "The number of tool calls changed",
        "stream_equals_batch": "The split read does not match the one-piece read",
        "args_json": "Arguments were not valid JSON",
        "no_markup_leak": "Marker text leaked into the reply",
        "no_crash": "The reader crashed",
        "finish_reason": "The finish reason does not match the calls",
    }.get(name, name)


def _explain(invariant: str, batch: Normalized, stream: Normalized) -> str:
    if invariant == "count":
        return (
            f"Read as one piece: {len(batch.calls)} tool call(s). "
            f"Read in separate pieces: {len(stream.calls)}."
        )
    if invariant == "stream_equals_batch":
        return "The prose or the call itself changes when the text is cut into pieces."
    if invariant == "no_markup_leak":
        return "A tool-call marker was left in the reply text even though a call was found."
    if invariant == "args_json":
        return "A tool call was reported with arguments that are not valid JSON."
    return ""
