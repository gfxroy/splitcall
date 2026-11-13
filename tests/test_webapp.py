import json
from pathlib import Path

from splitcall.corpus import corpus
from splitcall.webapp import analyze, presets_json

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = (
    "__init__.py",
    "types.py",
    "assemble.py",
    "repair.py",
    "invariants.py",
    "shrink.py",
    "corpus.py",
    "webapp.py",
)


def test_pages_bundle_matches_the_library():
    for name in BUNDLE:
        source = (ROOT / "src" / "splitcall" / name).read_bytes()
        bundled = (ROOT / "site" / "py" / "splitcall" / name).read_bytes()
        assert source == bundled, name


def test_presets_come_from_the_corpus():
    rows = json.loads(presets_json())
    by_id = {case.id: case for case in corpus()}
    assert rows
    for row in rows:
        case = by_id[row["id"]]
        assert row["text"] == case.text
        assert row["format"] == case.format


def test_hermes_example_repairs_and_finds_a_split_failure():
    text = next(case.text for case in corpus() if case.id == "hermes-basic")
    report = json.loads(analyze("hermes", text))
    assert report["repair"]["calls"][0]["name"] == "get_weather"
    assert report["repair"]["calls"][0]["id"] == "call_0000"
    assert json.loads(report["repair"]["calls"][0]["arguments"]) == {"city": "Paris"}
    assert report["repair"]["finish_reason"] == "tool_calls"
    assert report["repair_fuzz"]["fail"] == 0
    assert report["split_reader"]["fail"] > 0
    repros = report["split_reader"]["repros"]
    assert repros
    count = next(item for item in repros if item["invariant"] == "count")
    assert "<tool_call>" in count["shrunk"]
    assert "1 tool call" in count["explain"]
    assert "0." in count["explain"]
    # The lines describe the shrunk string, not the original sample.
    assert "no tool call" in count["split_up"]
    assert "calls " in count["one_piece"]


def test_truncated_hermes_is_repaired():
    text = '<tool_call>{"name": "get_weather", "arguments": {"city": "Par"'
    report = json.loads(analyze("hermes", text))
    args = json.loads(report["repair"]["calls"][0]["arguments"])
    assert args == {"city": "Par"}
    assert report["repair"]["finish_reason"] == "tool_calls"


def test_every_preset_keeps_a_stable_repair():
    for row in json.loads(presets_json()):
        report = json.loads(analyze(row["format"], row["text"]))
        assert "error" not in report, row["id"]
        assert report["repair_fuzz"]["fail"] == 0, row["id"]
        assert report["repair"]["error"] is None


def test_empty_input_is_an_error():
    report = json.loads(analyze("hermes", "   "))
    assert "Paste" in report["error"]
