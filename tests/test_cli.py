import json

from splitcall.cli import main


def test_list_corpus(capsys):
    assert main(["list-corpus"]) == 0
    out = capsys.readouterr().out
    assert "hermes-basic" in out
    assert "harmony-basic" in out


def test_parse_repair(capsys):
    text = '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
    assert main(["parse", "--format", "hermes", "--parser", "repair", text]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["finish_reason"] == "tool_calls"
    assert payload["calls"][0]["name"] == "get_weather"
    assert payload["calls"][0]["id"] == "call_0000"
