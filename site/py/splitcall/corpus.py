"""Hand-written wire samples.

These strings follow the published tool-call layouts and the examples in the
vendored parsers' tests. They are not generations from a live model. Each
item is short so every character offset can be replayed.
"""

from __future__ import annotations

from splitcall.types import Case, ExpectCall

_W = ("get_weather", {"city": "Paris"})
_S = ("search", {"query": "umbrellas"})


def _case(
    case_id: str,
    fmt: str,
    text: str,
    calls: list[tuple[str, object]],
    **kwargs: object,
) -> Case:
    return Case(
        id=case_id,
        format=fmt,
        text=text,
        expect_calls=[ExpectCall(name, args) for name, args in calls],
        **kwargs,  # type: ignore[arg-type]
    )


def corpus() -> list[Case]:
    """Return the full corpus. Order is stable for the results matrix."""
    h = '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
    cases = [
        _case("hermes-basic", "hermes", h, [_W]),
        _case(
            "hermes-prose",
            "hermes",
            "Sure. " + h,
            [_W],
            expect_content="Sure.",
        ),
        _case(
            "hermes-parallel",
            "hermes",
            h + '<tool_call>{"name": "search", "arguments": {"query": "umbrellas"}}</tool_call>',
            [_W, _S],
        ),
        _case(
            "hermes-repeat",
            "hermes",
            h + h,
            [_W, _W],
        ),
        _case(
            "hermes-unicode",
            "hermes",
            '<tool_call>{"name": "get_weather", "arguments": {"city": "東京"}}</tool_call>',
            [("get_weather", {"city": "東京"})],
        ),
        _case(
            "hermes-braces",
            "hermes",
            '<tool_call>{"name": "echo", "arguments": {"text": "a{b}c"}}</tool_call>',
            [("echo", {"text": "a{b}c"})],
        ),
        _case(
            "hermes-nested",
            "hermes",
            '<tool_call>{"name": "search", "arguments": {"filter": {"q": "a{b}c", "n": 2}}}</tool_call>',
            [("search", {"filter": {"q": "a{b}c", "n": 2}})],
        ),
        _case(
            "hermes-escape",
            "hermes",
            '<tool_call>{"name": "echo", "arguments": {"text": "say \\"hi\\""}}</tool_call>',
            [("echo", {"text": 'say "hi"'})],
        ),
        _case(
            "hermes-trunc-args",
            "hermes",
            '<tool_call>{"name": "get_weather", "arguments": {"city": "Par"',
            [],
            complete=False,
            note="Name is closed. Arguments stop inside a string.",
        ),
        _case(
            "hermes-trunc-name",
            "hermes",
            '<tool_call>{"name": "get_weat',
            [],
            complete=False,
            note="Name string is not closed.",
        ),
        _case(
            "hermes-prose-tag",
            "hermes",
            "A <tool_call> is documentation.",
            [],
            expect_content="A <tool_call> is documentation.",
        ),
        _case(
            "qwen-basic",
            "qwen",
            "<tool_call><function=get_weather><parameter=city>Paris</parameter></function></tool_call>",
            [_W],
        ),
        _case(
            "qwen-multiline",
            "qwen",
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>",
            [_W],
        ),
        _case(
            "qwen-parallel",
            "qwen",
            "<tool_call><function=get_weather><parameter=city>Paris</parameter></function></tool_call>"
            "<tool_call><function=search><parameter=query>umbrellas</parameter></function></tool_call>",
            [_W, _S],
        ),
        _case(
            "qwen-unicode",
            "qwen",
            "<tool_call><function=get_weather><parameter=city>São Paulo</parameter></function></tool_call>",
            [("get_weather", {"city": "São Paulo"})],
        ),
        _case(
            "qwen-braces",
            "qwen",
            "<tool_call><function=echo><parameter=text>a{b}c</parameter></function></tool_call>",
            [("echo", {"text": "a{b}c"})],
        ),
        _case(
            "qwen-two-params",
            "qwen",
            "<tool_call><function=get_weather><parameter=city>Paris</parameter>"
            "<parameter=unit>c</parameter></function></tool_call>",
            [("get_weather", {"city": "Paris", "unit": "c"})],
        ),
        _case(
            "qwen-trunc-value",
            "qwen",
            "<tool_call><function=get_weather><parameter=city>Par",
            [],
            complete=False,
        ),
        _case(
            "qwen-trunc-name",
            "qwen",
            "<tool_call><function=get_weat",
            [],
            complete=False,
        ),
        _case(
            "qwen-prose-tag",
            "qwen",
            "See <tool_call> in the docs.",
            [],
            expect_content="See <tool_call> in the docs.",
        ),
        _case(
            "mistral-array",
            "mistral",
            '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Paris"}}]',
            [_W],
            subset="json_array",
        ),
        _case(
            "mistral-array-parallel",
            "mistral",
            '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Paris"}}, '
            '{"name": "search", "arguments": {"query": "umbrellas"}}]',
            [_W, _S],
            subset="json_array",
        ),
        _case(
            "mistral-array-unicode",
            "mistral",
            '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "東京"}}]',
            [("get_weather", {"city": "東京"})],
            subset="json_array",
        ),
        _case(
            "mistral-array-braces",
            "mistral",
            '[TOOL_CALLS] [{"name": "echo", "arguments": {"text": "a{b}c"}}]',
            [("echo", {"text": "a{b}c"})],
            subset="json_array",
        ),
        _case(
            "mistral-array-trunc",
            "mistral",
            '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Par"',
            [],
            complete=False,
            subset="json_array",
        ),
        _case(
            "mistral-compact",
            "mistral",
            '[TOOL_CALLS]get_weather[ARGS]{"city": "Paris"}',
            [_W],
            subset="compact",
        ),
        _case(
            "mistral-compact-space",
            "mistral",
            '[TOOL_CALLS] get_weather[ARGS]{"city": "Paris"}',
            [_W],
            subset="compact",
        ),
        _case(
            "mistral-compact-parallel",
            "mistral",
            '[TOOL_CALLS]get_weather[ARGS]{"city": "Paris"}[TOOL_CALLS]search[ARGS]{"query": "umbrellas"}',
            [_W, _S],
            subset="compact",
        ),
        _case(
            "mistral-compact-nested",
            "mistral",
            '[TOOL_CALLS]search[ARGS]{"filter": {"q": "a{b}"}}',
            [("search", {"filter": {"q": "a{b}"}})],
            subset="compact",
        ),
        _case(
            "mistral-compact-trunc",
            "mistral",
            '[TOOL_CALLS]get_weather[ARGS]{"city": "Par"',
            [],
            complete=False,
            subset="compact",
        ),
        _case(
            "mistral-prose",
            "mistral",
            "[TOOL_CALLS] is the marker name",
            [],
            expect_content="[TOOL_CALLS] is the marker name",
        ),
        _case(
            "llama-tag",
            "llama",
            '<|python_tag|>{"name": "get_weather", "arguments": {"city": "Paris"}}',
            [_W],
        ),
        _case(
            "llama-bare",
            "llama",
            '{"name": "get_weather", "arguments": {"city": "Paris"}}',
            [_W],
        ),
        _case(
            "llama-parallel",
            "llama",
            '<|python_tag|>{"name": "get_weather", "arguments": {"city": "Paris"}};'
            '{"name": "search", "arguments": {"query": "umbrellas"}}',
            [_W, _S],
        ),
        _case(
            "llama-unicode",
            "llama",
            '<|python_tag|>{"name": "get_weather", "arguments": {"city": "東京"}}',
            [("get_weather", {"city": "東京"})],
        ),
        _case(
            "llama-braces",
            "llama",
            '<|python_tag|>{"name": "echo", "arguments": {"text": "a{b}c"}}',
            [("echo", {"text": "a{b}c"})],
        ),
        _case(
            "llama-nested",
            "llama",
            '<|python_tag|>{"name": "search", "arguments": {"filter": {"q": "a{b}", "n": 1}}}',
            [("search", {"filter": {"q": "a{b}", "n": 1}})],
        ),
        _case(
            "llama-python-dict",
            "llama",
            "<|python_tag|>{'name': 'get_weather', 'arguments': {'city': 'Paris'}}",
            [_W],
            check_args=False,
            note="Single-quoted Python dict. Argument equality is not required.",
        ),
        _case(
            "llama-eot",
            "llama",
            '<|python_tag|>{"name": "get_weather", "arguments": {"city": "Paris"}}<|eot_id|>',
            [_W],
        ),
        _case(
            "llama-trunc",
            "llama",
            '<|python_tag|>{"name": "get_weather", "arguments": {"city": "Par"',
            [],
            complete=False,
        ),
        _case(
            "harmony-basic",
            "harmony",
            '<|start|>assistant<|channel|>commentary to=functions.get_weather'
            '<|constrain|>json<|message|>{"city": "Paris"}<|call|>',
            [_W],
        ),
        _case(
            "harmony-no-constrain",
            "harmony",
            '<|start|>assistant<|channel|>commentary to=functions.get_weather'
            '<|message|>{"city": "Paris"}<|call|>',
            [_W],
        ),
        _case(
            "harmony-no-start",
            "harmony",
            '<|channel|>commentary to=functions.get_weather<|constrain|>json'
            '<|message|>{"city": "Paris"}<|call|>',
            [_W],
            note="No <|start|>assistant prefix.",
        ),
        _case(
            "harmony-final",
            "harmony",
            "<|start|>assistant<|channel|>analysis<|message|>t<|end|>"
            "<|start|>assistant<|channel|>commentary to=functions.get_weather"
            '<|constrain|>json<|message|>{"city":"P"}<|call|>'
            "<|start|>assistant<|channel|>final<|message|>Done<|return|>",
            [("get_weather", {"city": "P"})],
            expect_content="Done",
        ),
        _case(
            "harmony-parallel",
            "harmony",
            "<|start|>assistant<|channel|>commentary to=functions.get_weather"
            '<|constrain|>json<|message|>{"city":"P"}<|call|>'
            "<|start|>assistant<|channel|>commentary to=functions.search"
            '<|constrain|>json<|message|>{"query":"u"}<|call|>',
            [("get_weather", {"city": "P"}), ("search", {"query": "u"})],
        ),
        _case(
            "harmony-unicode",
            "harmony",
            '<|start|>assistant<|channel|>commentary to=functions.get_weather'
            '<|constrain|>json<|message|>{"city": "東京"}<|call|>',
            [("get_weather", {"city": "東京"})],
        ),
        _case(
            "harmony-braces",
            "harmony",
            '<|start|>assistant<|channel|>commentary to=functions.echo'
            '<|constrain|>json<|message|>{"text": "a{b}c"}<|call|>',
            [("echo", {"text": "a{b}c"})],
        ),
        _case(
            "harmony-nested",
            "harmony",
            '<|start|>assistant<|channel|>commentary to=functions.search'
            '<|constrain|>json<|message|>{"filter": {"q": "a{b}"}}<|call|>',
            [("search", {"filter": {"q": "a{b}"}})],
        ),
        _case(
            "harmony-trunc",
            "harmony",
            '<|start|>assistant<|channel|>commentary to=functions.get_weather'
            '<|constrain|>json<|message|>{"city": "Par"',
            [],
            complete=False,
        ),
        _case(
            "harmony-prose",
            "harmony",
            "No tool in this note.",
            [],
            expect_content="No tool in this note.",
        ),
    ]
    for case in cases:
        if len(case.text) > 240:
            raise ValueError(f"{case.id} is {len(case.text)} chars")
    return cases


def formats() -> list[str]:
    return ["hermes", "qwen", "mistral", "llama", "harmony"]
