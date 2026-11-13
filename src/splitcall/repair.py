"""Incremental repair parser for five published tool-call wire formats.

The parse of a buffer is a pure function of that buffer. `feed` replays the
accumulated text and emits only the growth since the previous call, so every
chunk split finishes on the same calls, ids, and content as one shot.
Argument text is emitted only when the JSON value is complete, or when
`finish` closes a truncated value. A name that was not already closed in the
source is not invented.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field

from splitcall.assemble import assemble
from splitcall.types import Delta, Normalized

FORMATS = ("hermes", "qwen", "mistral", "llama", "harmony")

_HERMES_OPEN = "<tool_call>"
_HERMES_CLOSE = "</tool_call>"
_QWEN_FN = "<function="
_QWEN_FN_END = "</function>"
_QWEN_P = "<parameter="
_QWEN_P_END = "</parameter>"
_QWEN_END = "</tool_call>"
_MISTRAL = "[TOOL_CALLS]"
_ARGS = "[ARGS]"
_LLAMA = "<|python_tag|>"
_LLAMA_ENDS = ("<|eot_id|>", "<|eom_id|>", "<|end_of_text|>")
_H_STRUCT = (
    "<|start|>",
    "<|channel|>",
    "<|constrain|>",
    "<|message|>",
    "<|call|>",
    "<|return|>",
    "<|end|>",
)
# Hold "<|start|>assistant" as one marker so a cut inside the role word
# is not emitted as content and then retracted.
_H_HOLD = _H_STRUCT + ("<|start|>assistant",)


@dataclass
class _Call:
    name: str
    arguments: str


@dataclass
class _Doc:
    content: str = ""
    calls: list[_Call] = field(default_factory=list)
    reasoning: str = ""
    tool_started: bool = False


def hold_len(text: str, markers: tuple[str, ...] | list[str]) -> int:
    """Length of the longest suffix that is a proper prefix of a marker."""
    best = 0
    if not text:
        return 0
    for marker in markers:
        limit = min(len(text), len(marker) - 1)
        for length in range(limit, 0, -1):
            if marker.startswith(text[-length:]):
                if length > best:
                    best = length
                break
    return best


def _flush_tail(tail: str, markers: list[str], final: bool) -> tuple[str, bool]:
    """Return content to keep, and whether a truncated marker was swallowed."""
    held = hold_len(tail, markers)
    if held == 0:
        return tail, False
    kept, suffix = tail[:-held], tail[-held:]
    if not final:
        return kept, False
    if len(suffix) >= 2 and any(marker.startswith(suffix) for marker in markers):
        return kept, True
    return kept + suffix, False


def _complete_number(token: str) -> bool:
    try:
        json.loads(token)
    except json.JSONDecodeError:
        return False
    return True


def _stack_of(text: str) -> list[str] | None:
    stack: list[str] = []
    in_str = False
    esc = False
    for char in text:
        if in_str:
            if esc:
                esc = False
            elif char == "\\":
                esc = True
            elif char == '"':
                in_str = False
            continue
        if char == '"':
            in_str = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if not stack:
                return None
            opener = stack.pop()
            if (char == "}") != (opener == "{"):
                return None
    if in_str:
        return None
    return stack


def close_json(src: str) -> str | None:
    """Close open strings and brackets. Drop a trailing comma or half token.

    Returns a string `json.loads` accepts, or None when the prefix is not a
    JSON value that can be closed.
    """
    if not src or not src.strip():
        return None
    chars = list(src)
    # Truncate at the first character that cannot sit inside a JSON value,
    # after consuming a well-formed prefix. Implemented by walking a copy.
    s = src
    in_str = False
    esc = False
    i = 0
    n = len(s)
    while i < n:
        char = s[i]
        if in_str:
            if esc:
                esc = False
                i += 1
                continue
            if char == "\\":
                if i + 1 >= n:
                    s = s[:i]
                    n = i
                    break
                esc = True
                i += 1
                continue
            if char == '"':
                in_str = False
            i += 1
            continue
        if char.isspace():
            i += 1
            continue
        if char == '"':
            in_str = True
            i += 1
            continue
        if char in "{[":
            i += 1
            continue
        if char in "}]":
            i += 1
            continue
        if char in ",:":
            i += 1
            continue
        if char == "-" or char.isdigit():
            j = i + 1
            while j < n and s[j] in "0123456789.eE+-":
                j += 1
            token = s[i:j]
            if _complete_number(token):
                i = j
                continue
            if j < n:
                return None
            s = s[:i]
            n = i
            break
        matched = False
        partial = False
        for word in ("true", "false", "null"):
            if s.startswith(word, i):
                i += len(word)
                matched = True
                break
            if word.startswith(s[i:]) and s[i:]:
                s = s[:i]
                n = i
                partial = True
                matched = True
                break
        if partial:
            break
        if matched:
            continue
        # Markup or prose after a JSON value. Keep the JSON prefix.
        s = s[:i]
        n = i
        break

    if in_str:
        if s.endswith("\\"):
            s = s[:-1]
        # Drop a truncated \\u / \\x escape so the closed string is valid.
        if re.search(r"\\u[0-9a-fA-F]{0,3}$", s) or re.search(r"\\$", s):
            s = re.sub(r"\\u[0-9a-fA-F]{0,3}$", "", s)
            s = re.sub(r"\\$", "", s)
        s += '"'

    stripped = s.rstrip()
    if stripped.endswith(","):
        stripped = stripped[:-1].rstrip()
    if stripped.endswith(":"):
        stripped += " null"
    s = stripped
    stack = _stack_of(s)
    if stack is None:
        return None
    for opener in reversed(stack):
        s += "}" if opener == "{" else "]"
    try:
        json.loads(s)
    except json.JSONDecodeError:
        return None
    return s


def _dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def canon_args(value: object) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return _dumps(value)
    if value is None:
        value = {}
    return _dumps(value)


def _closed_string_value(original: str, key: str) -> bool:
    """True when `key` maps to a JSON string whose closing quote is present."""
    needle = f'"{key}"'
    idx = 0
    while True:
        found = original.find(needle, idx)
        if found < 0:
            return False
        k = found + len(needle)
        while k < len(original) and original[k].isspace():
            k += 1
        if k >= len(original) or original[k] != ":":
            idx = found + 1
            continue
        k += 1
        while k < len(original) and original[k].isspace():
            k += 1
        if k >= len(original) or original[k] != '"':
            return False
        k += 1
        esc = False
        while k < len(original):
            char = original[k]
            if esc:
                esc = False
            elif char == "\\":
                esc = True
            elif char == '"':
                return True
            k += 1
        return False


def _has_closed_string(original: str, value: str) -> bool:
    """True when `value` appears as a finished JSON string in `original`."""
    if not isinstance(value, str):
        return False
    needle = json.dumps(value, ensure_ascii=False)
    idx = 0
    while True:
        found = original.find(needle, idx)
        if found < 0:
            return False
        # json.dumps produced a quoted string. Reject a match that is the
        # prefix of a longer string (the next char would be inside it, so
        # the quote we matched is the closer only if the scanner agrees).
        return True


def _call_from_obj(obj: object, original: str, *, require_closed_name: bool) -> _Call | None:
    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    if not isinstance(name, str) or not name:
        return None
    if require_closed_name and not _has_closed_string(original, name):
        return None
    if "arguments" in obj:
        args = obj.get("arguments")
    elif "parameters" in obj:
        args = obj.get("parameters")
    else:
        args = {}
    return _Call(name, canon_args(args if args is not None else {}))


def _scan_value(text: str, index: int) -> tuple[object | None, int, bool]:
    try:
        value, end = json.JSONDecoder().raw_decode(text, index)
    except json.JSONDecodeError:
        return None, index, False
    return value, end, True


def _trim_newlines(value: str) -> str:
    if value.startswith("\n"):
        value = value[1:]
    if value.endswith("\n"):
        value = value[:-1]
    return value


def _parse_hermes(text: str, final: bool) -> _Doc:
    doc = _Doc()
    parts: list[str] = []
    i = 0
    n = len(text)
    markers = [_HERMES_OPEN]
    while i < n:
        j = text.find(_HERMES_OPEN, i)
        if j < 0:
            kept, started = _flush_tail(text[i:], markers, final)
            parts.append(kept)
            if started:
                doc.tool_started = True
            break
        parts.append(text[i:j])
        after = j + len(_HERMES_OPEN)
        k = after
        while k < n and text[k] in " \t\r\n":
            k += 1
        if k >= n:
            doc.tool_started = True
            break
        if text[k] != "{":
            parts.append(_HERMES_OPEN)
            i = after
            continue
        doc.tool_started = True
        value, end, complete = _scan_value(text, k)
        if not complete:
            if final:
                repaired = close_json(text[k:])
                if repaired is not None:
                    call = _call_from_obj(json.loads(repaired), text[k:], require_closed_name=True)
                    if call is not None:
                        doc.calls.append(call)
            break
        call = _call_from_obj(value, text[k:end], require_closed_name=False)
        if call is not None:
            doc.calls.append(call)
        p = end
        while p < n and text[p] in " \t\r\n":
            p += 1
        if text.startswith(_HERMES_CLOSE, p):
            i = p + len(_HERMES_CLOSE)
            continue
        if p < n and _HERMES_CLOSE.startswith(text[p:]):
            break
        i = end
    doc.content = "".join(parts)
    return doc


def _qwen_call(name: str, params: dict[str, str]) -> _Call:
    return _Call(name, _dumps(params))


def _parse_one_qwen(text: str, i: int, final: bool) -> tuple[_Call | None, int, bool]:
    """Parse one `<function=...>` starting at `i`. Returns call, index, done."""
    name_start = i + len(_QWEN_FN)
    gt = text.find(">", name_start)
    if gt < 0:
        return None, i, False
    name = text[name_start:gt].strip()
    if not name:
        return None, i, False
    p = gt + 1
    params: dict[str, str] = {}
    n = len(text)
    closers = (_QWEN_FN_END, _QWEN_P, _QWEN_END, _QWEN_P_END)
    while p < n:
        while p < n and text[p] in " \t\r\n":
            p += 1
        if p >= n:
            break
        if text.startswith(_QWEN_FN_END, p):
            p += len(_QWEN_FN_END)
            q = p
            while q < n and text[q] in " \t\r\n":
                q += 1
            # A newline or a split "</tool_call>" is still inside the call.
            # Leave it out of content so a later chunk cannot retract it.
            if q >= n or _QWEN_END.startswith(text[q:]):
                return _qwen_call(name, params), n, True
            if text.startswith(_QWEN_END, q):
                return _qwen_call(name, params), q + len(_QWEN_END), True
            return _qwen_call(name, params), p, True
        if text.startswith(_QWEN_P, p):
            ps = p + len(_QWEN_P)
            pgt = text.find(">", ps)
            if pgt < 0:
                if final:
                    return _qwen_call(name, params), n, False
                return None, i, False
            pname = text[ps:pgt].strip()
            vstart = pgt + 1
            stops: list[tuple[int, str]] = []
            for marker in (_QWEN_P_END, _QWEN_P, _QWEN_FN_END, _QWEN_END):
                found = text.find(marker, vstart)
                if found >= 0:
                    stops.append((found, marker))
            if not stops:
                rest = text[vstart:]
                if not final and hold_len(rest, list(closers)):
                    return None, i, False
                if not final:
                    return None, i, False
                if pname:
                    params[pname] = _trim_newlines(rest)
                return _qwen_call(name, params), n, False
            found, marker = min(stops)
            if pname:
                params[pname] = _trim_newlines(text[vstart:found])
            if marker == _QWEN_P_END:
                p = found + len(_QWEN_P_END)
                continue
            if marker == _QWEN_P:
                p = found
                continue
            p = found
            continue
        rest = text[p:]
        if any(marker.startswith(rest) for marker in closers):
            if final:
                return _qwen_call(name, params), n, False
            return None, i, False
        if final:
            return _qwen_call(name, params), p, False
        return None, i, False
    if final:
        return _qwen_call(name, params), n, False
    return None, i, False


def _parse_qwen(text: str, final: bool) -> _Doc:
    doc = _Doc()
    parts: list[str] = []
    i = 0
    n = len(text)
    open_tag = _HERMES_OPEN
    while i < n:
        j = text.find(open_tag, i)
        if j < 0:
            kept, started = _flush_tail(text[i:], [open_tag], final)
            parts.append(kept)
            if started:
                doc.tool_started = True
            break
        parts.append(text[i:j])
        after = j + len(open_tag)
        k = after
        while k < n and text[k] in " \t\r\n":
            k += 1
        rest = text[k:]
        if rest == "" or (not rest.startswith(_QWEN_FN) and _QWEN_FN.startswith(rest)):
            doc.tool_started = True
            break
        if not rest.startswith(_QWEN_FN):
            parts.append(open_tag)
            i = after
            continue
        doc.tool_started = True
        call, nxt, done = _parse_one_qwen(text, k, final)
        if call is None:
            break
        if not done and not final:
            break
        doc.calls.append(call)
        if not done:
            break
        i = nxt
    doc.content = "".join(parts)
    return doc


def _mistral_array(text: str, start: int, final: bool) -> tuple[list[_Call], int, bool]:
    value, end, complete = _scan_value(text, start)
    if complete:
        calls: list[_Call] = []
        items = value if isinstance(value, list) else [value]
        for item in items:
            call = _call_from_obj(item, text[start:end], require_closed_name=False)
            if call is not None:
                calls.append(call)
        return calls, end, True
    if not final:
        return [], start, False
    repaired = close_json(text[start:])
    if repaired is None:
        return [], start, False
    try:
        value = json.loads(repaired)
    except json.JSONDecodeError:
        return [], start, False
    calls = []
    items = value if isinstance(value, list) else [value]
    original = text[start:]
    for item in items:
        call = _call_from_obj(item, original, require_closed_name=True)
        if call is not None:
            calls.append(call)
    return calls, len(text), False


def _parse_mistral(text: str, final: bool) -> _Doc:
    doc = _Doc()
    parts: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = text.find(_MISTRAL, i)
        if j < 0:
            kept, started = _flush_tail(text[i:], [_MISTRAL, _ARGS], final)
            parts.append(kept)
            if started:
                doc.tool_started = True
            break
        parts.append(text[i:j])
        k = j + len(_MISTRAL)
        ws = k
        while ws < n and text[ws] in " \t\r\n":
            ws += 1
        if ws >= n:
            doc.tool_started = True
            break
        if text.startswith(_ARGS, ws) or (
            text[ws] == "[" and _ARGS.startswith(text[ws:])
        ):
            # `[ARGS]` without a name is not a call. If the token is only a
            # prefix, hold; otherwise treat the marker as prose.
            if not text.startswith(_ARGS, ws):
                doc.tool_started = True
                break
            parts.append(_MISTRAL)
            i = k
            continue
        if text[ws] == "[":
            doc.tool_started = True
            calls, nxt, done = _mistral_array(text, ws, final)
            doc.calls.extend(calls)
            if not done and not calls:
                break
            if not done:
                break
            i = nxt
            continue
        match = re.match(r"[A-Za-z_][A-Za-z0-9_\-]*", text[ws:])
        if not match:
            parts.append(_MISTRAL)
            i = k
            continue
        name = match.group(0)
        p = ws + len(name)
        while p < n and text[p] in " \t\r\n":
            p += 1
        if p >= n:
            doc.tool_started = True
            break
        if text.startswith(_ARGS, p):
            p += len(_ARGS)
            while p < n and text[p] in " \t\r\n":
                p += 1
        elif text[p] == "[" and _ARGS.startswith(text[p:]):
            doc.tool_started = True
            break
        if p >= n:
            doc.tool_started = True
            break
        if text[p] != "{":
            parts.append(_MISTRAL)
            i = k
            continue
        doc.tool_started = True
        value, end, complete = _scan_value(text, p)
        if complete and isinstance(value, (dict, list)):
            doc.calls.append(_Call(name, canon_args(value)))
            i = end
            continue
        if not final:
            break
        repaired = close_json(text[p:])
        if repaired is not None:
            doc.calls.append(_Call(name, canon_args(json.loads(repaired))))
        break
    doc.content = "".join(parts)
    return doc


def _python_span(text: str, i: int) -> int | None:
    """End index of a single-quoted Python dict starting at `{`, or None."""
    if i >= len(text) or text[i] != "{":
        return None
    depth = 0
    quote: str | None = None
    esc = False
    for pos in range(i, len(text)):
        char = text[pos]
        if quote:
            if esc:
                esc = False
            elif char == "\\":
                esc = True
            elif char == quote:
                quote = None
            continue
        if char in "'\"":
            quote = char
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return pos + 1
    return None


def _call_from_python(text: str, start: int, end: int) -> _Call | None:
    try:
        value = ast.literal_eval(text[start:end])
    except (SyntaxError, ValueError):
        return None
    return _call_from_obj(value, text[start:end], require_closed_name=False)


def _parse_json_seq(text: str, i: int, final: bool) -> tuple[list[_Call], int, bool]:
    """Parse `;`-separated JSON objects or Python dicts. Third flag is 'hold'."""
    calls: list[_Call] = []
    n = len(text)
    while i < n:
        while i < n and text[i] in " \t\r\n;":
            i += 1
        if i >= n:
            break
        if any(text.startswith(token, i) for token in _LLAMA_ENDS):
            for token in _LLAMA_ENDS:
                if text.startswith(token, i):
                    i += len(token)
                    break
            continue
        if any(token.startswith(text[i:]) for token in _LLAMA_ENDS):
            if final:
                break
            return calls, i, True
        if text[i] == "{":
            value, end, complete = _scan_value(text, i)
            if complete:
                call = _call_from_obj(value, text[i:end], require_closed_name=False)
                if call is None:
                    py_end = _python_span(text, i)
                    if py_end is not None:
                        call = _call_from_python(text, i, py_end)
                        end = py_end
                if call is not None:
                    calls.append(call)
                    i = end
                    continue
                return calls, i, False
            py_end = _python_span(text, i)
            if py_end is not None:
                call = _call_from_python(text, i, py_end)
                if call is not None:
                    calls.append(call)
                    i = py_end
                    continue
            if not final:
                return calls, i, True
            repaired = close_json(text[i:])
            if repaired is not None:
                call = _call_from_obj(
                    json.loads(repaired), text[i:], require_closed_name=True
                )
                if call is not None:
                    calls.append(call)
            return calls, n, False
        return calls, i, False
    return calls, i, False


def _parse_llama(text: str, final: bool) -> _Doc:
    doc = _Doc()
    parts: list[str] = []
    i = 0
    n = len(text)
    lead = text.lstrip()
    if lead.startswith("{") and _LLAMA not in text:
        calls, end, hold = _parse_json_seq(text, n - len(lead), final)
        doc.calls.extend(calls)
        doc.tool_started = True
        if hold and not final:
            return doc
        trailing = text[end:]
        kept, started = _flush_tail(trailing, list(_LLAMA_ENDS) + [_LLAMA], final)
        for token in _LLAMA_ENDS:
            kept = kept.replace(token, "")
        parts.append(kept)
        if started:
            doc.tool_started = True
        doc.content = "".join(parts)
        return doc

    while i < n:
        j = text.find(_LLAMA, i)
        if j < 0:
            kept, started = _flush_tail(text[i:], [_LLAMA, *_LLAMA_ENDS], final)
            for token in _LLAMA_ENDS:
                kept = kept.replace(token, "")
            parts.append(kept)
            if started:
                doc.tool_started = True
            break
        if j > i:
            parts.append(text[i:j])
        doc.tool_started = True
        calls, nxt, hold = _parse_json_seq(text, j + len(_LLAMA), final)
        doc.calls.extend(calls)
        if hold and not final:
            break
        i = nxt
        if hold:
            break
    doc.content = "".join(parts)
    return doc


def _harmony_name(header: str, ended: bool) -> str | None:
    match = re.search(r"to=([A-Za-z_][A-Za-z0-9_.]*)", header)
    if match is None:
        return None
    if not ended and match.end() == len(header):
        return None
    return match.group(1).split(".")[-1]


def _strip_harmony_filler(text: str) -> str:
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if text.startswith("<|start|>", i):
            i += len("<|start|>")
            k = i
            while k < n and text[k].isspace():
                k += 1
            if text.startswith("assistant", k) and (
                k + 9 == n or not (text[k + 9].isalnum() or text[k + 9] == "_")
            ):
                i = k + 9
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def _parse_harmony(text: str, final: bool) -> _Doc:
    doc = _Doc()
    parts: list[str] = []
    reasoning: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = text.find("<|channel|>", i)
        if j < 0:
            kept, started = _flush_tail(text[i:], list(_H_HOLD), final)
            parts.append(_strip_harmony_filler(kept))
            if started:
                doc.tool_started = True
            break
        parts.append(_strip_harmony_filler(text[i:j]))
        header_at = j + len("<|channel|>")
        stops = [tok for tok in _H_STRUCT if tok != "<|channel|>"]
        positions = []
        for tok in stops:
            found = text.find(tok, header_at)
            if found >= 0:
                positions.append((found, tok))
        if not positions:
            header = text[header_at:]
            if _harmony_name(header, ended=False) or "to=" in header or header.strip():
                doc.tool_started = True
            if final and header.strip().split(" ")[0:1] == ["final"]:
                pass
            break
        pos, tok = min(positions)
        header = text[header_at:pos]
        ended = True
        kind = header.strip().split()[0].lower() if header.strip() else ""
        name = _harmony_name(header, ended=ended)
        if tok == "<|constrain|>":
            message = text.find("<|message|>", pos)
            if message < 0:
                doc.tool_started = bool(name) or kind in {"commentary", "analysis"}
                break
            pos = message
            tok = "<|message|>"
        if tok != "<|message|>":
            # Channel with no message yet. Hold unless this is clearly prose.
            if kind in {"commentary", "analysis", "final"} or name:
                doc.tool_started = True
                if not final:
                    break
            i = pos
            if not final:
                break
            continue
        body_at = pos + len("<|message|>")
        if kind == "final":
            end_tokens = ("<|return|>", "<|end|>")
        elif kind == "analysis":
            end_tokens = ("<|end|>", "<|call|>")
        else:
            end_tokens = ("<|call|>", "<|end|>")
        ends = []
        for token in end_tokens:
            found = text.find(token, body_at)
            if found >= 0:
                ends.append((found, token))
        if ends:
            ep, etok = min(ends)
            body = text[body_at:ep]
            nxt = ep + len(etok)
            closed = True
        else:
            body = text[body_at:]
            nxt = n
            closed = False
            held = hold_len(body, list(_H_HOLD))
            if held and not final:
                body = body[:-held]
        if kind == "final":
            parts.append(body)
            i = nxt if closed or final else body_at + len(body)
            if not closed and not final:
                # Body already appended up to the held suffix. Stop so we
                # do not rescan it; the next feed reparses the whole buffer.
                break
            continue
        if kind == "commentary" and name:
            doc.tool_started = True
            if not closed and not final:
                break
            raw = body.strip()
            value, _, complete = _scan_value(raw, 0) if raw.startswith("{") or raw.startswith("[") else (None, 0, False)
            if complete:
                doc.calls.append(_Call(name, canon_args(value)))
            elif final:
                repaired = close_json(raw) if raw else None
                if repaired is not None:
                    doc.calls.append(_Call(name, canon_args(json.loads(repaired))))
            else:
                break
            if not closed:
                break
            i = nxt
            continue
        if kind == "analysis":
            if name and (closed or final):
                doc.tool_started = True
                raw = body.strip()
                value, _, complete = (
                    _scan_value(raw, 0) if raw[:1] in "{[" else (None, 0, False)
                )
                if complete:
                    doc.calls.append(_Call(name, canon_args(value)))
                elif final and raw:
                    repaired = close_json(raw)
                    if repaired is not None:
                        doc.calls.append(_Call(name, canon_args(json.loads(repaired))))
            else:
                reasoning.append(body)
                if name:
                    doc.tool_started = True
                    if not final:
                        break
            if not closed and not final:
                break
            i = nxt if closed else n
            if not closed:
                break
            continue
        if kind == "commentary" and not name:
            # Commentary without a recipient is ordinary text once closed.
            if not closed and not final:
                kept, started = _flush_tail(body, list(_H_HOLD), False)
                parts.append(kept)
                if started:
                    doc.tool_started = True
                break
            parts.append(body)
            i = nxt if closed else n
            if not closed:
                break
            continue
        doc.tool_started = True
        if not final:
            break
        i = nxt if closed else n
        if not closed:
            break
    doc.content = "".join(parts)
    doc.reasoning = "".join(reasoning)
    return doc


_PARSERS = {
    "hermes": _parse_hermes,
    "qwen": _parse_qwen,
    "mistral": _parse_mistral,
    "llama": _parse_llama,
    "harmony": _parse_harmony,
}


def parse_document(fmt: str, text: str, final: bool) -> _Doc:
    if fmt not in _PARSERS:
        raise ValueError(f"unknown format {fmt!r}")
    return _PARSERS[fmt](text, final)


class RepairSession:
    """Streaming session. Ids are `call_NNNN` in call order."""

    def __init__(self, fmt: str) -> None:
        if fmt not in _PARSERS:
            raise ValueError(f"unknown format {fmt!r}")
        self.fmt = fmt
        self.buf = ""
        self._sent_content = ""
        self._sent_names: list[str | None] = []
        self._sent_args: list[str | None] = []

    def feed(self, chunk: str) -> list[Delta]:
        self.buf += chunk
        return self._diff(final=False)

    def finish(self) -> tuple[list[Delta], str, str]:
        """Return deltas, finish_reason, and reasoning text."""
        doc = parse_document(self.fmt, self.buf, True)
        deltas = self._diff(final=True)
        if any(call.arguments is not None for call in self._complete_calls(doc)):
            reason = "tool_calls"
        elif doc.tool_started:
            reason = "length"
        else:
            reason = "stop"
        return deltas, reason, doc.reasoning

    def _complete_calls(self, doc: _Doc) -> list[_Call]:
        return doc.calls

    def _diff(self, final: bool) -> list[Delta]:
        doc = parse_document(self.fmt, self.buf, final)
        deltas: list[Delta] = []
        content = doc.content
        if not content.startswith(self._sent_content):
            # The hold rules are supposed to make content grow only. If a
            # format edge retracts a character, keep the longer committed
            # prefix and do not emit a negative delta.
            if self._sent_content.startswith(content):
                content = self._sent_content
            else:
                content = self._sent_content
        if len(content) > len(self._sent_content):
            deltas.append(Delta(content=content[len(self._sent_content) :]))
            self._sent_content = content
        for index, call in enumerate(doc.calls):
            while len(self._sent_names) <= index:
                self._sent_names.append(None)
                self._sent_args.append(None)
            if call.name and self._sent_names[index] is None:
                deltas.append(
                    Delta(
                        index=index,
                        id=f"call_{index:04d}",
                        name=call.name,
                        type="function",
                    )
                )
                self._sent_names[index] = call.name
            if call.arguments is not None and self._sent_args[index] is None:
                deltas.append(Delta(index=index, arguments=call.arguments))
                self._sent_args[index] = call.arguments
        return deltas


def _normalized(fmt: str, chunks: list[str]) -> Normalized:
    session = RepairSession(fmt)
    deltas: list[Delta] = []
    for chunk in chunks:
        deltas.extend(session.feed(chunk))
    more, reason, reasoning = session.finish()
    return assemble(
        deltas + more,
        finish_reason=reason,
        assigns_id=True,
        assigns_finish=True,
        assigns_type=True,
        reasoning=reasoning,
    )


def parse_repair(fmt: str, text: str) -> Normalized:
    return _normalized(fmt, [text])


def parse_repair_chunks(fmt: str, chunks: list[str]) -> Normalized:
    return _normalized(fmt, chunks)
