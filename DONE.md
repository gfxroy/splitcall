# splitcall

- Repo: https://github.com/gfxroy/splitcall
- Pages: https://gfxroy.github.io/splitcall/ — the front page checks a pasted sample in the browser. The published matrix is the second tab (`site/matrix.html`).
- CI on `main`: success (pytest). Pages workflow: success after Pages was enabled with `build_type=workflow` and the first deploy was re-run.

## What shipped

Python CLI and library `splitcall` 0.1.0, MIT. Hand-written corpus of Hermes, Qwen3-coder, Mistral (JSON-array and compact), Llama 3 JSON, and OpenAI Harmony wire samples, plus edges. Vendored parser classes from vLLM `f6cbdda2b6f54fee19823db25b068a08fd0f2313` and SGLang `23a45fd33921164fa440fd7c28cc63341259c28c` (Apache-2.0). Incremental repair parser. JSON and monochrome HTML matrix. pytest (19 tests locally). The tests do not require upstream cells to be clean.

## Run

`splitcall fuzz` on the full suite, not `--quick`: 50 cases, 24,237 splits, 3.895s, `repair_failures` 0, 12 shrunk repros. Numbers are in `results/matrix.json`.

Repair is 0 fail / 0 gap on all five formats (78,998 passes). Every upstream parser has failed checks. `finish_reason` is a gap on every upstream parser, because the detector classes do not set it. SGLang `id` and `type` are the same kind of gap. vLLM does assign ids.

## Findings

- Complete Hermes, Qwen3, Llama JSON, and Mistral v11 compact samples match the oracle on vLLM’s batch path and on a character stream (`get_weather`, `{"city": "Paris"}`, index 0). SGLang Qwen3-coder does too. SGLang Harmony with `<|constrain|>json` does on one chunk, and does not on character chunks.
- Partial markers: vLLM Hermes truncated to `<` puts that character in batch content and leaves the stream empty. vLLM Llama holds `<|python_tag|>` the same way at truncation 14. SGLang Qwen does it for `<`.
- Closed name, unfinished arguments: vLLM and SGLang Hermes at truncation 33 of `hermes-basic` extract nothing on the batch path and a `get_weather` call with empty arguments on the stream path. vLLM Qwen at truncation 49 returns `{}` from batch and `{"city": "` from stream. SGLang Qwen on `qwen-parallel` truncated at 117 returns one batch call and two stream calls.
- vLLM Mistral pre-v11: a one-character cut of `[TOOL_CALLS]` drops the call and dumps the whole sample into content. The public `extract_tool_calls` entry point drops the call even on an intact buffer (`tools_called` false) because pre-v11 streaming never fills the engine slots. The legacy parser on the same string returns `get_weather`. That probe is in the matrix `extra` field.
- vLLM Mistral v11 treats `[TOOL_CALLS] is the marker name` as a compact call named `is the marker name`. The oracle and the repair parser keep it as prose.
- SGLang batch `tool_index` for Hermes, Llama 3.2, and Mistral JSON-array is the schema slot (`get_weather` is 1, behind `ping`). The stream path uses the call ordinal 0. This fails on the unsplit sample, at `cut@0`. A single streamed chunk of those samples returns the name and empty arguments; character chunks return the JSON. On `mistral-compact-parallel`, one chunk returns only the first call.
- SGLang Mistral `mistral-compact-parallel` truncated at 58 leaves `[TOOL_CALLS]` in batch content while a call was extracted (`no_markup_leak`, 34 splits).
- SGLang Harmony character chunks of `harmony-basic` extract no call and leak the `<|start|>assistant<|channel|>...` header. A cut at offset 1 keeps the call and leaks `<`. `harmony-no-constrain` (no `<|constrain|>json`) extracts nothing on batch or stream. vLLM’s GptOss parser is an upstream stub and is not scored.
- `qwen-prose-tag` (`See <tool_call> in the docs.`) is parsed by vLLM and SGLang as content `See ` on both paths. Repair keeps the sentence. vLLM Llama does not parse the single-quoted `llama-python-dict`; repair does, via `ast.literal_eval`.

The 12 stored repros are all SGLang because the shrink slots are filled in parser-name order. Their batch and stream lines were re-parsed from the shrunk string. Several `stream_equals_batch` shrinks collapse to a held character (`<`, `{`, `<|call|>`). The index shrinks still show batch index 1. vLLM’s failures are in the matrix counts.

## Limits

Splits are Unicode code points, not bytes or tokenizer boundaries. The corpus is hand-written, not live model output. Serving code that would assign `finish_reason` and SGLang ids is not imported. llama.cpp and Ollama are not included. The repair parser is scored against itself and the hand oracle, not against every upstream quirk.
