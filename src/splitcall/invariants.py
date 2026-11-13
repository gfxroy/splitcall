"""OpenAI-shaped checks for one batch parse and one streamed parse.

A missing id or finish_reason is a gap when the parser API does not assign
one. A wrong index, a stream that disagrees with the batch parse, invalid
final JSON, or markup left in content is a failure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from splitcall.types import Case, Normalized

LEAK_MARKERS = {
    "hermes": ("<tool_call>", "</tool_call>"),
    "qwen": (
        "<tool_call>",
        "</tool_call>",
        "<function=",
        "</function>",
        "<parameter=",
        "</parameter>",
    ),
    "mistral": ("[TOOL_CALLS]", "[ARGS]"),
    "llama": ("<|python_tag|>",),
    "harmony": (
        "<|channel|>",
        "<|constrain|>",
        "<|message|>",
        "<|call|>",
        "<|start|>",
        "to=functions.",
    ),
}


@dataclass
class Verdict:
    name: str
    status: str
    detail: str = ""


def _content(value: str | None) -> str:
    return (value or "").strip()


def _loads(text: str) -> tuple[bool, object]:
    try:
        return True, json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return False, None


def judge(case: Case, batch: Normalized, stream: Normalized, *, full_text: bool) -> list[Verdict]:
    if batch.error or stream.error:
        which = "batch" if batch.error else "stream"
        err = batch.error or stream.error or ""
        return [Verdict("no_crash", "fail", f"{which}: {err[:240]}")]

    out: list[Verdict] = [Verdict("no_crash", "pass")]
    out.append(_stream_equals(batch, stream))
    out.append(_count(batch, stream))
    out.append(_index_ordinal(batch, stream))
    out.append(_index_stable(stream))
    out.append(_id_present(batch, stream))
    out.append(_id_stable(stream))
    out.append(_args_json(batch, stream))
    out.append(_finish(batch, stream))
    out.append(_leak(case, batch, stream))
    out.append(_type_function(batch, stream))
    out.append(_args_are_strings(case, batch, stream))
    out.extend(_oracle(case, batch, full_text=full_text))
    return out


def _stream_equals(batch: Normalized, stream: Normalized) -> Verdict:
    if _content(batch.content) != _content(stream.content):
        return Verdict(
            "stream_equals_batch",
            "fail",
            f"content {batch.content!r} vs {stream.content!r}",
        )
    if len(batch.calls) != len(stream.calls):
        return Verdict(
            "stream_equals_batch",
            "fail",
            f"call count {len(batch.calls)} vs {len(stream.calls)}",
        )
    for index, (left, right) in enumerate(zip(batch.calls, stream.calls)):
        if left.name != right.name:
            return Verdict(
                "stream_equals_batch",
                "fail",
                f"call {index} name {left.name!r} vs {right.name!r}",
            )
        if left.index != right.index:
            return Verdict(
                "stream_equals_batch",
                "fail",
                f"call {index} index {left.index} vs {right.index}",
            )
        ok_l, val_l = _loads(left.arguments)
        ok_r, val_r = _loads(right.arguments)
        if ok_l and ok_r:
            if val_l != val_r:
                return Verdict(
                    "stream_equals_batch",
                    "fail",
                    f"call {index} arguments differ",
                )
        elif left.arguments != right.arguments:
            return Verdict(
                "stream_equals_batch",
                "fail",
                f"call {index} raw arguments differ",
            )
    return Verdict("stream_equals_batch", "pass")


def _count(batch: Normalized, stream: Normalized) -> Verdict:
    if len(batch.calls) != len(stream.calls):
        return Verdict(
            "count",
            "fail",
            f"{len(batch.calls)} batch vs {len(stream.calls)} stream",
        )
    return Verdict("count", "pass")


def _ordinal_one(label: str, result: Normalized) -> str | None:
    if not result.calls:
        return None
    got = [call.index for call in result.calls]
    want = list(range(len(result.calls)))
    if got != want:
        return f"{label} indices {got}"
    return None


def _index_ordinal(batch: Normalized, stream: Normalized) -> Verdict:
    if not batch.calls and not stream.calls:
        return Verdict("index_ordinal", "skip", "no calls")
    problems = [item for item in (_ordinal_one("batch", batch), _ordinal_one("stream", stream)) if item]
    if problems:
        return Verdict("index_ordinal", "fail", "; ".join(problems))
    return Verdict("index_ordinal", "pass")


def _index_stable(stream: Normalized) -> Verdict:
    saw = False
    by_id: dict[str, int] = {}
    for delta in stream.deltas:
        toolish = any(
            value is not None for value in (delta.name, delta.arguments, delta.id, delta.type)
        )
        if not toolish:
            continue
        saw = True
        if delta.index is None:
            return Verdict("index_stable", "fail", "tool delta missing index")
        if delta.id:
            previous = by_id.get(delta.id)
            if previous is not None and previous != delta.index:
                return Verdict(
                    "index_stable",
                    "fail",
                    f"id {delta.id} moved {previous} -> {delta.index}",
                )
            by_id[delta.id] = delta.index
    for call in stream.calls:
        if call.id_changes:
            return Verdict("index_stable", "fail", f"index {call.index} id changed with the call")
    if not saw and not stream.calls:
        return Verdict("index_stable", "skip", "no calls")
    return Verdict("index_stable", "pass")


def _id_present(batch: Normalized, stream: Normalized) -> Verdict:
    if not batch.assigns_id and not stream.assigns_id:
        return Verdict("id_present", "gap", "parser API does not assign ids")
    missing = []
    for label, result in (("batch", batch), ("stream", stream)):
        for call in result.calls:
            if not call.id:
                missing.append(f"{label}[{call.index}]")
    if missing:
        return Verdict("id_present", "fail", "missing " + ", ".join(missing))
    if not batch.calls and not stream.calls:
        return Verdict("id_present", "skip", "no calls")
    return Verdict("id_present", "pass")


def _id_stable(stream: Normalized) -> Verdict:
    if not stream.assigns_id:
        return Verdict("id_stable", "gap", "parser API does not assign ids")
    seen: dict[int, str] = {}
    for delta in stream.deltas:
        if delta.index is None or not delta.id:
            continue
        previous = seen.get(delta.index)
        if previous is None:
            seen[delta.index] = delta.id
        elif previous != delta.id:
            return Verdict(
                "id_stable",
                "fail",
                f"index {delta.index} id {previous} -> {delta.id}",
            )
    for call in stream.calls:
        if call.id_changes:
            return Verdict("id_stable", "fail", f"index {call.index} changed id")
    if not stream.calls:
        return Verdict("id_stable", "skip", "no calls")
    return Verdict("id_stable", "pass")


def _args_json(batch: Normalized, stream: Normalized) -> Verdict:
    if not batch.calls and not stream.calls:
        return Verdict("args_json", "skip", "no calls")
    bad = []
    for label, result in (("batch", batch), ("stream", stream)):
        for call in result.calls:
            ok, _ = _loads(call.arguments)
            if not ok:
                bad.append(f"{label}[{call.index}] {call.arguments!r}")
    if bad:
        return Verdict("args_json", "fail", "; ".join(bad)[:240])
    return Verdict("args_json", "pass")


def _finish(batch: Normalized, stream: Normalized) -> Verdict:
    if not batch.assigns_finish and not stream.assigns_finish:
        return Verdict("finish_reason", "gap", "parser API does not set finish_reason")
    problems = []
    for label, result in (("batch", batch), ("stream", stream)):
        reason = result.finish_reason
        if result.calls and reason != "tool_calls":
            problems.append(f"{label} finish {reason!r} with calls")
        elif not result.calls and reason == "tool_calls":
            problems.append(f"{label} finish tool_calls without calls")
        elif reason not in {"tool_calls", "stop", "length"}:
            problems.append(f"{label} finish {reason!r}")
    if problems:
        return Verdict("finish_reason", "fail", "; ".join(problems))
    return Verdict("finish_reason", "pass")


def _leak(case: Case, batch: Normalized, stream: Normalized) -> Verdict:
    if not batch.calls and not stream.calls:
        return Verdict("no_markup_leak", "skip", "no calls")
    markers = LEAK_MARKERS.get(case.format, ())
    problems = []
    for label, result in (("batch", batch), ("stream", stream)):
        if not result.calls:
            continue
        for marker in markers:
            if marker in (result.content or ""):
                problems.append(f"{label} contains {marker}")
                break
    if problems:
        return Verdict("no_markup_leak", "fail", "; ".join(problems))
    return Verdict("no_markup_leak", "pass")


def _type_function(batch: Normalized, stream: Normalized) -> Verdict:
    if not batch.assigns_type and not stream.assigns_type:
        return Verdict("type_function", "gap", "parser API does not set type")
    bad = []
    for label, result in (("batch", batch), ("stream", stream)):
        for call in result.calls:
            if call.type != "function":
                bad.append(f"{label}[{call.index}]={call.type!r}")
    if bad:
        return Verdict("type_function", "fail", "; ".join(bad))
    if not batch.calls and not stream.calls:
        return Verdict("type_function", "skip", "no calls")
    return Verdict("type_function", "pass")


def _args_are_strings(case: Case, batch: Normalized, stream: Normalized) -> Verdict:
    if case.format != "qwen":
        return Verdict("args_are_strings", "skip", "not qwen XML parameters")
    if not batch.calls and not stream.calls:
        return Verdict("args_are_strings", "skip", "no calls")
    bad = []
    for label, result in (("batch", batch), ("stream", stream)):
        for call in result.calls:
            ok, value = _loads(call.arguments)
            if not ok or not isinstance(value, dict):
                continue
            for key, item in value.items():
                if not isinstance(item, str):
                    bad.append(f"{label}.{key}={type(item).__name__}")
    if bad:
        return Verdict("args_are_strings", "fail", "; ".join(bad)[:240])
    return Verdict("args_are_strings", "pass")


def _oracle(case: Case, batch: Normalized, *, full_text: bool) -> list[Verdict]:
    if not full_text or not case.complete:
        return [
            Verdict("oracle_names", "skip", "truncation or prefix"),
            Verdict("oracle_args", "skip", "truncation or prefix"),
            Verdict("oracle_content", "skip", "truncation or prefix"),
        ]
    got_names = [call.name for call in batch.calls]
    want_names = [call.name for call in case.expect_calls]
    names = Verdict("oracle_names", "pass")
    if got_names != want_names:
        names = Verdict("oracle_names", "fail", f"{got_names} vs {want_names}")
    args = Verdict("oracle_args", "pass")
    if not case.check_args:
        args = Verdict("oracle_args", "skip", "check_args is false")
    elif names.status == "fail":
        args = Verdict("oracle_args", "fail", "names differ")
    else:
        for index, expect in enumerate(case.expect_calls):
            ok, value = _loads(batch.calls[index].arguments)
            if not ok or value != expect.arguments:
                args = Verdict(
                    "oracle_args",
                    "fail",
                    f"call {index} {batch.calls[index].arguments!r}",
                )
                break
    content = Verdict("oracle_content", "pass")
    if case.expect_content is None and not case.expect_content_contains:
        content = Verdict("oracle_content", "skip", "no content oracle")
    else:
        got = _content(batch.content)
        if case.expect_content is not None and got != case.expect_content:
            content = Verdict("oracle_content", "fail", f"{batch.content!r}")
        elif case.expect_content_contains and case.expect_content_contains not in (
            batch.content or ""
        ):
            content = Verdict("oracle_content", "fail", f"missing {case.expect_content_contains!r}")
    return [names, args, content]
