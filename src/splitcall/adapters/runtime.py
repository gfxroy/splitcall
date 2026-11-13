"""Adapters over vendored vLLM parsers, SGLang detectors, and splitcall repair.

Each adapter returns a fresh parser per call. Streaming and batch parses are
never compared across runtimes. They are compared to themselves.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

from splitcall.assemble import assemble, assemble_calls
from splitcall.repair import parse_repair, parse_repair_chunks
from splitcall.types import Delta, NormCall, Normalized
from splitcall.vendoring import activate

activate()

# Vendored modules import vllm/sglang, so this stays after activate().
from sglang.srt.entrypoints.openai.protocol import Function, Tool  # noqa: E402
from vllm.parser.llama_json import LlamaJsonParser  # noqa: E402
from vllm.parser.mistral import MistralParser  # noqa: E402
from vllm.parser.qwen3 import Qwen3Parser  # noqa: E402
from vllm.tool_parsers.hermes_tool_parser import Hermes2ProToolParser  # noqa: E402
from sglang.srt.function_call.gpt_oss_detector import GptOssDetector  # noqa: E402
from sglang.srt.function_call.hermes_detector import HermesDetector  # noqa: E402
from sglang.srt.function_call.llama32_detector import Llama32Detector  # noqa: E402
from sglang.srt.function_call.mistral_detector import MistralDetector  # noqa: E402
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector  # noqa: E402

for _name in ("vllm", "sglang"):
    logging.getLogger(_name).setLevel(logging.CRITICAL)


def _request() -> SimpleNamespace:
    return SimpleNamespace(
        tools=None,
        tool_choice="auto",
        model="splitcall",
        structured_outputs=None,
        response_format=None,
        include_reasoning=True,
        skip_special_tokens=False,
        _grammar_from_parser=False,
    )


class _Tokenizer:
    def __init__(self, vocab: dict[str, int] | None = None) -> None:
        self._vocab = dict(vocab or {})
        self.is_mistral = False
        self.supports_grammar = False
        self.all_special_tokens: list[str] = []
        self.all_special_ids: list[int] = []

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)


def _schema(properties: dict[str, str]) -> dict:
    return {
        "type": "object",
        "properties": {key: {"type": kind} for key, kind in properties.items()},
    }


def _tool(name: str, properties: dict[str, str]) -> Tool:
    return Tool(
        type="function",
        function=Function(name=name, description=name, parameters=_schema(properties)),
    )


# ping is first on purpose: detectors that copy the schema index instead of
# the call ordinal land on 1 for get_weather.
SGLANG_TOOLS = [
    _tool("ping", {"host": "string"}),
    _tool("get_weather", {"city": "string", "unit": "string"}),
    _tool("search", {"query": "string"}),
    _tool("echo", {"text": "string"}),
    _tool("lookup", {"id": "string"}),
    _tool("sum", {"a": "string", "b": "string"}),
]


def _fn_parts(function) -> tuple[str | None, str | None]:
    if function is None:
        return None, None
    if isinstance(function, dict):
        return function.get("name"), function.get("arguments")
    return getattr(function, "name", None), getattr(function, "arguments", None)


def _as_text(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value if value else None
    return str(value)


def _deltas_from_message(message) -> list[Delta]:
    if message is None:
        return []
    deltas: list[Delta] = []
    content = getattr(message, "content", None)
    if content:
        deltas.append(Delta(content=content))
    for tool_call in getattr(message, "tool_calls", None) or []:
        name, arguments = _fn_parts(getattr(tool_call, "function", None))
        deltas.append(
            Delta(
                index=getattr(tool_call, "index", None),
                id=getattr(tool_call, "id", None),
                name=name,
                arguments=_as_text(arguments),
                type=getattr(tool_call, "type", None),
            )
        )
    return deltas


def _from_extracted(info, *, assigns_id: bool, assigns_type: bool) -> Normalized:
    calls: list[NormCall] = []
    for index, tool_call in enumerate(getattr(info, "tool_calls", None) or []):
        name, arguments = _fn_parts(getattr(tool_call, "function", None))
        calls.append(
            NormCall(
                index=index,
                name=name,
                arguments=arguments if isinstance(arguments, str) else _as_text(arguments) or "",
                id=getattr(tool_call, "id", None),
                type=getattr(tool_call, "type", None) or ("function" if assigns_type else None),
            )
        )
    return assemble_calls(
        calls,
        getattr(info, "content", None),
        finish_reason=None,
        assigns_id=assigns_id,
        assigns_finish=False,
        assigns_type=assigns_type,
    )


def _sglang_deltas(result) -> list[Delta]:
    if result is None:
        return []
    deltas: list[Delta] = []
    if getattr(result, "normal_text", None):
        deltas.append(Delta(content=result.normal_text))
    for item in getattr(result, "calls", None) or []:
        deltas.append(
            Delta(
                index=item.tool_index,
                name=item.name,
                arguments=item.parameters if item.parameters else None,
            )
        )
    return deltas


def _guard(fn, *, assigns_id: bool, assigns_finish: bool, assigns_type: bool) -> Normalized:
    try:
        return fn()
    except Exception as exc:
        return Normalized(
            error=f"{type(exc).__name__}: {exc}"[:300],
            assigns_id=assigns_id,
            assigns_finish=assigns_finish,
            assigns_type=assigns_type,
        )


class VLLMAdapter:
    def __init__(
        self,
        name: str,
        formats: set[str],
        factory,
        *,
        subsets: set[str] | None = None,
        legacy_batch: bool = False,
        finish: bool = False,
    ) -> None:
        self.name = name
        self.runtime = "vllm"
        self.formats = formats
        self.subsets = subsets
        self.assigns_id = True
        self.assigns_finish = False
        self.assigns_type = True
        self._factory = factory
        self._legacy_batch = legacy_batch
        self._finish = finish

    def parse_batch(self, text: str, fmt: str) -> Normalized:
        def run() -> Normalized:
            parser = self._factory()
            request = _request()
            if self._legacy_batch:
                info = parser.extract_tool_calls_from_content(text, request)
            else:
                info = parser.extract_tool_calls(text, request)
            return _from_extracted(info, assigns_id=True, assigns_type=True)

        return _guard(run, assigns_id=True, assigns_finish=False, assigns_type=True)

    def parse_stream(self, chunks: list[str], fmt: str) -> Normalized:
        def run() -> Normalized:
            parser = self._factory()
            request = _request()
            deltas: list[Delta] = []
            previous = ""
            for chunk in chunks:
                current = previous + chunk
                message = parser.extract_tool_calls_streaming(
                    previous,
                    current,
                    chunk,
                    [],
                    [],
                    [],
                    request,
                )
                deltas.extend(_deltas_from_message(message))
                previous = current
            if self._finish:
                deltas.extend(_deltas_from_message(parser.finish_streaming()))
            return assemble(
                deltas,
                finish_reason=None,
                assigns_id=True,
                assigns_finish=False,
                assigns_type=True,
            )

        return _guard(run, assigns_id=True, assigns_finish=False, assigns_type=True)


class SGLangAdapter:
    def __init__(self, name: str, formats: set[str], factory, subsets: set[str] | None = None) -> None:
        self.name = name
        self.runtime = "sglang"
        self.formats = formats
        self.subsets = subsets
        self.assigns_id = False
        self.assigns_finish = False
        self.assigns_type = False
        self._factory = factory

    def parse_batch(self, text: str, fmt: str) -> Normalized:
        def run() -> Normalized:
            detector = self._factory()
            result = detector.detect_and_parse(text, SGLANG_TOOLS)
            calls = [
                NormCall(
                    index=item.tool_index,
                    name=item.name,
                    arguments=item.parameters or "",
                    id=None,
                    type=None,
                )
                for item in result.calls
            ]
            return assemble_calls(
                calls,
                result.normal_text,
                finish_reason=None,
                assigns_id=False,
                assigns_finish=False,
                assigns_type=False,
            )

        return _guard(run, assigns_id=False, assigns_finish=False, assigns_type=False)

    def parse_stream(self, chunks: list[str], fmt: str) -> Normalized:
        def run() -> Normalized:
            detector = self._factory()
            deltas: list[Delta] = []
            for chunk in chunks:
                deltas.extend(_sglang_deltas(detector.parse_streaming_increment(chunk, SGLANG_TOOLS)))
            deltas.extend(_sglang_deltas(detector.finish(SGLANG_TOOLS)))
            return assemble(
                deltas,
                finish_reason=None,
                assigns_id=False,
                assigns_finish=False,
                assigns_type=False,
            )

        return _guard(run, assigns_id=False, assigns_finish=False, assigns_type=False)


class RepairAdapter:
    name = "splitcall-repair"
    runtime = "splitcall"
    formats = {"hermes", "qwen", "mistral", "llama", "harmony"}
    subsets = None
    assigns_id = True
    assigns_finish = True
    assigns_type = True

    def parse_batch(self, text: str, fmt: str) -> Normalized:
        return _guard(
            lambda: parse_repair(fmt, text),
            assigns_id=True,
            assigns_finish=True,
            assigns_type=True,
        )

    def parse_stream(self, chunks: list[str], fmt: str) -> Normalized:
        return _guard(
            lambda: parse_repair_chunks(fmt, chunks),
            assigns_id=True,
            assigns_finish=True,
            assigns_type=True,
        )


def _empty_tok() -> _Tokenizer:
    return _Tokenizer()


def all_parsers() -> list:
    return [
        VLLMAdapter("vllm-hermes", {"hermes"}, lambda: Hermes2ProToolParser(_empty_tok())),
        VLLMAdapter(
            "vllm-qwen3",
            {"qwen"},
            lambda: Qwen3Parser(
                _empty_tok(),
                chat_template_kwargs={"enable_thinking": False},
            ),
            finish=True,
        ),
        VLLMAdapter(
            "vllm-llama",
            {"llama"},
            lambda: LlamaJsonParser(_empty_tok()),
            finish=True,
        ),
        VLLMAdapter(
            "vllm-mistral-prev11",
            {"mistral"},
            lambda: MistralParser(_Tokenizer({"[TOOL_CALLS]": 1})),
            subsets={"json_array"},
            legacy_batch=True,
            finish=False,
        ),
        VLLMAdapter(
            "vllm-mistral-v11",
            {"mistral"},
            lambda: MistralParser(_Tokenizer({"[TOOL_CALLS]": 1, "[ARGS]": 2})),
            subsets={"compact"},
            finish=True,
        ),
        SGLangAdapter("sglang-hermes", {"hermes"}, HermesDetector),
        SGLangAdapter("sglang-qwen3", {"qwen"}, Qwen3CoderDetector),
        SGLangAdapter("sglang-mistral", {"mistral"}, MistralDetector),
        SGLangAdapter("sglang-llama32", {"llama"}, Llama32Detector),
        SGLangAdapter("sglang-harmony", {"harmony"}, GptOssDetector),
        RepairAdapter(),
    ]


def accepts(parser, case) -> bool:
    if case.format not in parser.formats:
        return False
    if case.subset is None or parser.subsets is None:
        return True
    return case.subset in parser.subsets


def probe_mistral_prev11(text: str) -> dict:
    """Compare the engine entry point with the legacy pre-v11 entry point.

    The public `extract_tool_calls` path rebuilds calls from engine slots.
    Pre-v11 streaming fills a DeltaMessage and leaves those slots empty.
    """
    parser = MistralParser(_Tokenizer({"[TOOL_CALLS]": 1}))
    engine = parser.extract_tool_calls(text, _request())
    legacy_parser = MistralParser(_Tokenizer({"[TOOL_CALLS]": 1}))
    legacy = legacy_parser.extract_tool_calls_from_content(text, _request())
    return {
        "text": text,
        "engine_tools_called": bool(engine.tools_called),
        "engine_names": [tc.function.name for tc in engine.tool_calls],
        "legacy_tools_called": bool(legacy.tools_called),
        "legacy_names": [tc.function.name for tc in legacy.tool_calls],
    }
