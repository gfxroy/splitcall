from splitcall.invariants import judge
from splitcall.types import Case, Delta, ExpectCall, NormCall, Normalized


def _case() -> Case:
    return Case(
        id="c",
        format="hermes",
        text="x",
        expect_calls=[ExpectCall("get_weather", {"city": "Paris"})],
    )


def _call(**kwargs) -> NormCall:
    base = dict(index=0, name="get_weather", arguments='{"city": "Paris"}', id="call_0000", type="function")
    base.update(kwargs)
    return NormCall(**base)


def test_matching_parse_passes():
    result = Normalized(
        calls=[_call()],
        finish_reason="tool_calls",
        assigns_id=True,
        assigns_finish=True,
        assigns_type=True,
        deltas=[Delta(index=0, id="call_0000", name="get_weather", type="function"), Delta(index=0, arguments='{"city": "Paris"}')],
    )
    verdicts = {item.name: item.status for item in judge(_case(), result, result, full_text=True)}
    assert verdicts["stream_equals_batch"] == "pass"
    assert verdicts["index_ordinal"] == "pass"
    assert verdicts["id_present"] == "pass"
    assert verdicts["finish_reason"] == "pass"
    assert verdicts["oracle_names"] == "pass"
    assert verdicts["oracle_args"] == "pass"


def test_schema_index_and_missing_id_are_not_the_same_status():
    batch = Normalized(calls=[_call(index=1, id=None, type=None)], assigns_id=False, assigns_type=False)
    stream = Normalized(
        calls=[_call(index=0, id=None, type=None)],
        assigns_id=False,
        assigns_type=False,
        deltas=[Delta(index=0, name="get_weather", arguments='{"city": "Paris"}')],
    )
    verdicts = {item.name: item for item in judge(_case(), batch, stream, full_text=True)}
    assert verdicts["stream_equals_batch"].status == "fail"
    assert verdicts["index_ordinal"].status == "fail"
    assert verdicts["id_present"].status == "gap"
    assert verdicts["finish_reason"].status == "gap"
    assert verdicts["type_function"].status == "gap"


def test_leaked_markup_only_counts_when_a_call_was_parsed():
    leaked = Normalized(
        content="see <tool_call>",
        calls=[_call()],
        finish_reason="tool_calls",
        assigns_id=True,
        assigns_finish=True,
        assigns_type=True,
    )
    prose = Normalized(content="A <tool_call> is documentation.", finish_reason="stop", assigns_id=True, assigns_finish=True, assigns_type=True)
    leaked_v = {item.name: item.status for item in judge(_case(), leaked, leaked, full_text=False)}
    prose_v = {item.name: item.status for item in judge(_case(), prose, prose, full_text=False)}
    assert leaked_v["no_markup_leak"] == "fail"
    assert prose_v["no_markup_leak"] == "skip"


def test_later_null_id_is_stable():
    stream = Normalized(
        calls=[_call()],
        finish_reason="tool_calls",
        assigns_id=True,
        assigns_finish=True,
        assigns_type=True,
        deltas=[
            Delta(index=0, id="chatcmpl-tool-abc", name="get_weather", type="function"),
            Delta(index=0, arguments='{"city": '),
            Delta(index=0, arguments='"Paris"}'),
        ],
    )
    verdicts = {item.name: item.status for item in judge(_case(), stream, stream, full_text=True)}
    assert verdicts["id_stable"] == "pass"
    assert verdicts["id_present"] == "pass"
    assert verdicts["args_json"] == "pass"
