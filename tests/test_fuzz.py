import logging

from splitcall.adapters.runtime import all_parsers, probe_mistral_prev11
from splitcall.invariants import judge
from splitcall.report import render_html
from splitcall.shrink import shrink_text
from splitcall.types import Case, Normalized


def test_failure_capture_describes_the_shrunk_string():
    from splitcall.fuzz import _first_failure

    class Planted:
        def parse_batch(self, text, fmt):
            return Normalized(content=text)

        def parse_stream(self, chunks, fmt):
            text = "".join(chunks)
            if len(chunks) > 1:
                return Normalized(content="LEAKED")
            return Normalized(content=text)

    case = Case(id="planted", format="hermes", text="abcd", expect_calls=[])
    detail, split, batch, stream = _first_failure(
        Planted(), case, "stream_equals_batch", "cut", "ab"
    )
    assert split == "cut@0"
    assert "ab" in batch
    assert "LEAKED" in stream
    assert "content" in detail


def test_shrink_keeps_the_planted_token():
    shrunk = shrink_text("prefix BAD suffix and more", lambda text: "BAD" in text, max_tests=400)
    assert shrunk == "BAD"


def test_fuzzer_predicate_shrinks_a_planted_split_bug():
    class Planted:
        def parse_batch(self, text, fmt):
            return Normalized(content=text)

        def parse_stream(self, chunks, fmt):
            text = "".join(chunks)
            if "BAD" in text and len(chunks) > 1:
                return Normalized(content="LEAKED")
            return Normalized(content=text)

    case = Case(id="planted", format="hermes", text="xxBADyy", expect_calls=[])
    parser = Planted()
    batch = parser.parse_batch(case.text, "hermes")
    stream = parser.parse_stream(list(case.text), "hermes")
    failed = [item.name for item in judge(case, batch, stream, full_text=True) if item.status == "fail"]
    assert "stream_equals_batch" in failed

    def pred(text: str) -> bool:
        chunks = list(text) or [""]
        left = parser.parse_batch(text, "hermes")
        right = parser.parse_stream(chunks, "hermes")
        return any(
            item.name == "stream_equals_batch" and item.status == "fail"
            for item in judge(case, left, right, full_text=False)
        )

    assert shrink_text("zz BAD yy", pred, max_tests=400) == "BAD"


def test_upstream_parsers_import_and_parse_one_sample():
    logging.disable(logging.ERROR)
    try:
        parsers = {parser.name: parser for parser in all_parsers()}
        hermes = '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
        parsed = parsers["vllm-hermes"].parse_batch(hermes, "hermes")
        assert parsed.error is None
        assert parsed.calls[0].name == "get_weather"
        sglang = parsers["sglang-hermes"].parse_batch(hermes, "hermes")
        assert sglang.error is None
        assert sglang.calls[0].name == "get_weather"
        repair = parsers["splitcall-repair"].parse_batch(hermes, "hermes")
        assert repair.finish_reason == "tool_calls"
        assert repair.calls[0].id == "call_0000"
    finally:
        logging.disable(logging.NOTSET)


def test_prev11_engine_drops_the_call_the_legacy_path_keeps():
    logging.disable(logging.ERROR)
    try:
        probe = probe_mistral_prev11(
            '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Paris"}}]'
        )
    finally:
        logging.disable(logging.NOTSET)
    assert probe["engine_tools_called"] is False
    assert probe["engine_names"] == []
    assert probe["legacy_tools_called"] is True
    assert probe["legacy_names"] == ["get_weather"]


def test_report_renders_a_failed_cell_in_black():
    page = render_html(
        {
            "version": "0.1.0",
            "cases": 1,
            "splits": 3,
            "elapsed_sec": 0.1,
            "formats": ["hermes"],
            "parsers": ["example"],
            "cells": {"example": {"hermes": {"pass": 1, "fail": 2, "gap": 0, "skip": 0}}},
            "invariants": {"example": {}},
            "repros": [],
            "extra": [],
            "limits": ["code points"],
        }
    )
    assert 'class="fail"' in page
    assert "splitcall" in page
    assert "example" in page
