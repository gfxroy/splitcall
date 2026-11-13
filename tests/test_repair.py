import json

from splitcall.corpus import corpus
from splitcall.repair import parse_repair, parse_repair_chunks


def _sig(result):
    return (
        result.content,
        result.finish_reason,
        result.reasoning,
        result.error,
        [
            (call.index, call.id, call.name, call.arguments, call.type)
            for call in result.calls
        ],
    )


def test_hermes_complete_and_prose():
    done = parse_repair(
        "hermes",
        '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>',
    )
    assert done.finish_reason == "tool_calls"
    assert done.calls[0].name == "get_weather"
    assert json.loads(done.calls[0].arguments) == {"city": "Paris"}
    assert done.calls[0].id == "call_0000"
    assert done.calls[0].index == 0
    assert done.calls[0].type == "function"
    assert done.content == ""

    prose = parse_repair("hermes", "A <tool_call> is documentation.")
    assert prose.calls == []
    assert prose.finish_reason == "stop"
    assert "<tool_call>" in prose.content


def test_truncation_repairs_closed_name_only():
    repaired = parse_repair(
        "hermes",
        '<tool_call>{"name": "get_weather", "arguments": {"city": "Par"',
    )
    assert repaired.finish_reason == "tool_calls"
    assert repaired.calls[0].name == "get_weather"
    assert json.loads(repaired.calls[0].arguments)["city"] == "Par"
    assert "<tool_call>" not in repaired.content

    unnamed = parse_repair("hermes", '<tool_call>{"name": "get_weat')
    assert unnamed.calls == []
    assert unnamed.finish_reason == "length"
    assert "<tool_call>" not in unnamed.content


def test_qwen_keeps_parameter_text():
    result = parse_repair(
        "qwen",
        "<tool_call><function=echo><parameter=text>a{b}c</parameter></function></tool_call>",
    )
    assert json.loads(result.calls[0].arguments) == {"text": "a{b}c"}


def test_mistral_array_and_compact():
    array = parse_repair(
        "mistral",
        '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Paris"}}]',
    )
    compact = parse_repair("mistral", '[TOOL_CALLS]get_weather[ARGS]{"city": "Paris"}')
    assert json.loads(array.calls[0].arguments) == {"city": "Paris"}
    assert json.loads(compact.calls[0].arguments) == {"city": "Paris"}
    prose = parse_repair("mistral", "[TOOL_CALLS] is the marker name")
    assert prose.calls == []
    assert "[TOOL_CALLS]" in prose.content


def test_llama_parallel_python_dict_and_bare():
    parallel = parse_repair(
        "llama",
        '<|python_tag|>{"name": "get_weather", "arguments": {"city": "Paris"}};'
        '{"name": "search", "arguments": {"query": "umbrellas"}}',
    )
    assert [call.name for call in parallel.calls] == ["get_weather", "search"]
    assert [call.index for call in parallel.calls] == [0, 1]
    quoted = parse_repair(
        "llama",
        "<|python_tag|>{'name': 'get_weather', 'arguments': {'city': 'Paris'}}",
    )
    assert quoted.calls[0].name == "get_weather"
    assert json.loads(quoted.calls[0].arguments) == {"city": "Paris"}
    bare = parse_repair("llama", '{"name": "echo", "arguments": {"text": "a{b}"}}')
    assert bare.calls[0].name == "echo"


def test_harmony_channels():
    basic = parse_repair(
        "harmony",
        '<|start|>assistant<|channel|>commentary to=functions.get_weather'
        '<|constrain|>json<|message|>{"city":"Paris"}<|call|>',
    )
    assert basic.calls[0].name == "get_weather"
    assert "<|channel|>" not in basic.content
    final = parse_repair(
        "harmony",
        "<|start|>assistant<|channel|>analysis<|message|>t<|end|>"
        "<|start|>assistant<|channel|>commentary to=functions.echo"
        '<|constrain|>json<|message|>{"text":"x"}<|call|>'
        "<|start|>assistant<|channel|>final<|message|>Done<|return|>",
    )
    assert final.content == "Done"
    assert final.reasoning == "t"
    assert final.calls[0].name == "echo"
    no_start = parse_repair(
        "harmony",
        '<|channel|>commentary to=functions.get_weather<|constrain|>json'
        '<|message|>{"city":"Paris"}<|call|>',
    )
    assert no_start.calls[0].name == "get_weather"


def test_every_corpus_split_matches_batch():
    for case in corpus():
        batch = parse_repair(case.format, case.text)
        expected = _sig(batch)
        for offset in range(len(case.text) + 1):
            streamed = parse_repair_chunks(
                case.format, [case.text[:offset], case.text[offset:]]
            )
            assert _sig(streamed) == expected, (case.id, "cut", offset)
        for size in (1, 2, 3, 5, 8):
            chunks = [case.text[i : i + size] for i in range(0, len(case.text), size)] or [""]
            streamed = parse_repair_chunks(case.format, ["", *chunks, ""])
            assert _sig(streamed) == expected, (case.id, "chunks", size)
        for offset in range(len(case.text) + 1):
            prefix = case.text[:offset]
            assert _sig(parse_repair(case.format, prefix)) == _sig(
                parse_repair_chunks(case.format, [prefix])
            )
