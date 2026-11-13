# splitcall

Chunk-boundary fuzzer and repair parser for streaming tool calls.

Live suites such as tool-eval-bench, K2-Vendor-Verifier, and willitcall talk to a running server. splitcall does not. It replays hand-written wire samples through the parser classes themselves, once as a whole buffer and again at every Unicode-code-point split and truncation, and checks the two results against each other and against a few OpenAI-shaped rules.

The latest full run is committed in [`results/matrix.json`](results/matrix.json). The Pages site at [gfxroy.github.io/splitcall](https://gfxroy.github.io/splitcall/) checks a pasted sample in the browser. The matrix from this run is the second tab.

## Install

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
splitcall fuzz -o results --site site
```

Python 3.11 or newer. The system interpreter on this machine is PEP 668-locked, so use a virtualenv. `pip install -e .` is enough to import the package. Until then, `PYTHONPATH=src`.

```bash
splitcall list-corpus
splitcall parse --format hermes '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
```

`splitcall fuzz` writes `results/matrix.json`, `results/matrix.html`, and `site/matrix.html`. The process exits 0 when an upstream parser fails. `--strict` exits 1 only when `splitcall-repair` fails. `--quick` is a smoke subset. The numbers below are from a full run, not `--quick`.

## What was run

50 hand-written cases (11 Hermes, 9 Qwen, 11 Mistral, 9 Llama, 10 Harmony; 42 complete, 8 truncated). 24,237 splits. Elapsed 3.895s. `splitcall-repair` failures: 0.

Each case is cut between every pair of code points, streamed in chunk sizes 1, 2, 3, 5, and 8, and truncated at every prefix. A black cell in the HTML has at least one failed check. A gap is a field the parser class never emits. Skip means the check did not apply (no calls on that split, an oracle on a truncated prefix, or a rule that is not for that format).

Vendored commits, recorded in `src/splitcall/vendor_path/UPSTREAM_COMMITS.txt`:

| Tree | Commit | What is scored |
| --- | --- | --- |
| vLLM | `f6cbdda2b6f54fee19823db25b068a08fd0f2313` | Hermes, Qwen3 (`enable_thinking=False`), Llama JSON, Mistral |
| SGLang | `23a45fd33921164fa440fd7c28cc63341259c28c` | Hermes, Qwen3-coder, Mistral, Llama 3.2, gpt-oss / Harmony |

Parser method bodies are the upstream files. Shims exist so those files import without torch. See [VENDORED.md](VENDORED.md).

### Format matrix

Fail / pass / gap on that parser’s own format. vLLM Mistral is split in two because the two vocabularies take different samples: `vllm-mistral-prev11` is the JSON-array subset, `vllm-mistral-v11` is the compact `[TOOL_CALLS]name[ARGS]{...}` subset. SGLang Mistral and the repair parser take both.

| Parser | Fail | Pass | Gap |
| --- | ---: | ---: | ---: |
| vllm-hermes | 2043 | 13981 | 1893 |
| vllm-qwen3 | 409 | 15187 | 1591 |
| vllm-llama | 1342 | 9854 | 1403 |
| vllm-mistral-prev11 | 1110 | 5847 | 888 |
| vllm-mistral-v11 | 576 | 5511 | 658 |
| sglang-hermes | 4506 | 6909 | 7572 |
| sglang-qwen3 | 1020 | 9958 | 6364 |
| sglang-mistral | 2666 | 5600 | 5908 |
| sglang-llama32 | 3310 | 5242 | 5612 |
| sglang-harmony | 1092 | 11251 | 10152 |
| splitcall-repair (five formats) | 0 | 78998 | 0 |

### Checks that failed

Counts are failed splits. Oracle rows are counted once per case (`cut@0`), not on every prefix.

| Parser | What failed |
| --- | --- |
| vllm-hermes | `stream_equals_batch` 1000, `count` 601, `args_json` 442. `finish_reason` is a gap on all 1893 scored splits. |
| vllm-qwen3 | `stream_equals_batch` 204, `args_json` 204, `oracle_content` 1. `finish_reason` gap 1591. |
| vllm-llama | `stream_equals_batch` 702, `count` 408, `args_json` 231, `oracle_names` 1. `finish_reason` gap 1403. |
| vllm-mistral-prev11 | `stream_equals_batch` 518, `count` 339, `args_json` 252, `oracle_content` 1. `finish_reason` gap 888. |
| vllm-mistral-v11 | `stream_equals_batch` 243, `count` 7, `args_json` 323, and one each of `oracle_names`, `oracle_args`, `oracle_content`. `finish_reason` gap 658. |
| sglang-hermes | `stream_equals_batch` 1863 (30 passes), `count` 881, `index_ordinal` 876, `args_json` 886. `id_present`, `id_stable`, `finish_reason`, and `type_function` are gaps (1893 each). |
| sglang-qwen3 | `stream_equals_batch` 530, `count` 61, `args_json` 428, `oracle_content` 1. Id, finish, and type are gaps (1591). `index_ordinal` did not fail. |
| sglang-mistral | `stream_equals_batch` 1064, `count` 472, `index_ordinal` 663, `args_json` 432, `no_markup_leak` 34, `oracle_content` 1. Id, finish, and type are gaps (1477). |
| sglang-llama32 | `stream_equals_batch` 1386 (17 passes), `count` 482, `index_ordinal` 727, `args_json` 715. Id, finish, and type are gaps (1403). |
| sglang-harmony | `stream_equals_batch` 785, `count` 303, `oracle_names` 2, `oracle_args` 2. Id, finish, and type are gaps (2538). |
| splitcall-repair | No invariant failed. |

`index_ordinal` is not in the vLLM fail tallies. vLLM’s assembled index is the call ordinal. SGLang’s batch `tool_index` for Hermes, Llama 3.2, and Mistral JSON-array is the position of the function in the tool schema. The schema used here is `ping`, `get_weather`, `search`, `echo`, `lookup`, `sum`, so `get_weather` is index 1 on the batch path and index 0 on the stream path. Qwen3-coder and gpt-oss use their own ordinal, and they do not fail `index_ordinal`.

`finish_reason` is a gap on every upstream parser. The detector classes do not set it. Serving code that would set `finish_reason: "tool_calls"` is not imported. SGLang tool-call `id` and `type` are the same kind of gap. vLLM does assign an id (`chatcmpl-tool-` plus hex for Hermes, Qwen, and Llama; a 9-character id for Mistral). The id is random, so the fuzzer does not require the stream id to equal the batch id. A later delta with `id=None` is not counted as an id change.

pytest does not require the upstream cells to be clean. It checks the repair parser, a planted shrink bug, one imported sample per runtime, the pre-v11 probe below, and that a failed cell renders black.

## Findings from this run

Complete samples that agree with the hand oracle on both the batch path and a character stream:

- vLLM Hermes, Qwen3, Llama JSON, and Mistral v11 compact: `get_weather` / `{"city": "Paris"}`, index 0.
- SGLang Qwen3-coder: the same, index 0.
- SGLang Harmony with `<|constrain|>json`, fed as one chunk: the same call, index 0. Character chunks of that sample do not. See below.
- splitcall-repair on the Hermes, Qwen, Mistral, Llama, and Harmony examples in the corpus, including the cases upstream misses.

Places the two paths, or the path and the oracle, disagree. These are the first failing split of each invariant in this run.

**Partial markers are held on one path.** `vllm-hermes` on `hermes-basic` truncated to 1 character: batch content is `<`, stream content is empty. `vllm-llama` on `llama-tag` truncated to 14 characters: batch content is `<|python_tag|>`, stream content is empty. `sglang-qwen3` on `qwen-basic` truncated to 1 character is the same `<` versus empty split. A character split of a marker that production would keep as one special token is in scope here, and it is a real disagreement between the two parser entry points.

**A closed name and an unfinished argument object.** `vllm-hermes` and `sglang-hermes` on `hermes-basic` truncated at 33 (`<tool_call>{"name": "get_weather"`): batch extracts no call and leaves the prefix as content; stream extracts `get_weather` with empty arguments, which also fails `args_json`. `vllm-llama` on `llama-tag` truncated at 36: batch extracts nothing; stream extracts one call whose arguments are `{}`. `vllm-qwen3` on `qwen-basic` truncated at 49: batch arguments are `{}`, stream arguments are `{"city": "`. `sglang-qwen3` on `qwen-parallel` truncated at 117: batch has one call (`get_weather`); stream has two (`get_weather`, then `search` with empty arguments).

**vLLM Mistral pre-v11 loses the call when `[TOOL_CALLS]` is split.** On `mistral-array`, the legacy batch parser returns `get_weather`. A cut at offset 1 (and a character stream of the whole sample) returns no call and puts the sample, marker included, into content. One intact chunk still returns the call. The public engine entry point is worse even on an intact buffer: `extract_tool_calls` reports `tools_called=false` and no names, because pre-v11 streaming fills a delta and never the engine’s tool slots, and the engine rebuilds from the slots. The legacy parser on the same string returns `get_weather`. That probe is in `results/matrix.json` under `extra` and is locked by a test. The matrix scores the legacy batch path against the stream path so the comparison is the one that can see the call.

**vLLM Mistral v11 treats prose as a compact call.** `[TOOL_CALLS] is the marker name` is a prose case. The oracle expects no call and the sentence as content. v11’s batch path returns a call named `is the marker name` with arguments `{}`. The character stream names it ` is the marker name` (leading space) and leaves the arguments empty. Truncating the real compact sample `mistral-compact` at 12 characters yields no batch call and one stream call with name `None` and empty arguments.

**SGLang batch index is the schema slot.** On intact `hermes-basic`, `llama-tag`, and `mistral-array`, batch index is 1 and the arguments are `{"city": "Paris"}`. The character stream index is 0 with the same JSON. A single streamed chunk of those three samples returns the name and empty arguments. `index_ordinal` already fails on `cut@0`, before any truncation. On `mistral-compact-parallel`, the batch indices are 1 and 2 (schema slots of `get_weather` and `search`). The character stream uses 0 and 1. One streamed chunk of that sample returns only the first call.

**SGLang Mistral can leave the marker in content after extracting a call.** `mistral-compact-parallel` truncated at 58: batch content is `[TOOL_CALLS]` while a `get_weather` call was extracted. That is the `no_markup_leak` failure (34 splits). The leak check runs only when a call was extracted.

**Harmony.** Character chunks of `harmony-basic` (`<|start|>assistant<|channel|>commentary to=functions.get_weather<|constrain|>json<|message|>{"city": "Paris"}<|call|>`): batch extracts `get_weather`; stream extracts nothing and the channel header leaks into content. A cut at offset 1 keeps the call on both sides and puts `<` into the stream content. `harmony-no-constrain` drops `<|constrain|>json`. SGLang’s batch detector, a one-chunk stream, and a character stream all extract no call. The oracle wants `get_weather`, so `oracle_names` and `oracle_args` fail (2 cases in the harmony set). vLLM’s `GptOssToolParser` is an upstream `NotImplementedError` stub and is not in the matrix.

**Oracles the parsers miss on an intact sample.**

- `qwen-prose-tag` is `See <tool_call> in the docs.` vLLM Qwen3 and SGLang Qwen3 both return content `See ` and no call, on the batch path and on the stream path. The oracle wants the whole sentence. Repair keeps the sentence and extracts no call.
- `llama-python-dict` is a single-quoted dict after `<|python_tag|>`. vLLM Llama extracts no call. The batch path leaves the raw text as content. Repair accepts it via `ast.literal_eval` and returns `get_weather`.
- `mistral-prose` under vLLM pre-v11: the batch path consumes `[TOOL_CALLS]` and leaves `is the marker name`. The oracle wants the marker kept, because it is not a call. Repair keeps the sentence, finish reason `stop`.
- SGLang on `mistral-prose` returns empty content and no call. The oracle wants the sentence.

## Minimal repros

The run stores 12 shrunk examples, one per `(parser, invariant)` pair, parser name first, so all 12 are SGLang. vLLM’s failures are in the counts above. They are not missing from the matrix. They lost the 12 documentation slots to sort order.

After shrinking, the report re-parses the short string and stores the batch line, the stream line, the detail, and the split label from a split that still fails. The HTML lines match that short string. ddmin will delete characters inside a token when a shorter string still fails, so several repros are a different face of the same bug:

- `sglang-harmony` / `count`, family `chunks:1`. The short string is still a commentary channel plus a call. Batch returns `get_weather` / `{"city": "Paris"}`. The character stream returns no call and leaks the `<|start|>assistant<|channel|>...` header into content.
- `sglang-harmony` / `stream_equals_batch` shrinks to `<|call|>`. Batch content is that token. Stream content is empty.
- `sglang-hermes` / `stream_equals_batch` shrinks to `<`. Batch emits it. Stream holds it.
- `sglang-llama32` / `stream_equals_batch` shrinks to `{`, same hold.
- `sglang-hermes` and `sglang-llama32` / `index_ordinal` stay on a closed call. Batch index is 1 (and, after shrinking the argument object, the arguments are `{}`). Stream index is 0.
- `sglang-hermes`, `sglang-llama32`, and `sglang-mistral` / `args_json` and `count` shrink to an opened call with no closing brace. Batch extracts nothing. Stream emits `get_weather` with empty arguments. For the Mistral pair the failing split of the short string is a cut or a truncation at offset 35, not the one-chunk parse (that one-chunk parse extracts no call, so `args_json` does not apply).

## Repair parser

`splitcall.repair` is a separate incremental parser. It is compared to itself and to the hand oracle. It is not claimed to reproduce every upstream quirk.

On this corpus it holds the longest suffix that is a proper prefix of a format marker, so a cut inside `<tool_call>`, `<|python_tag|>`, `[TOOL_CALLS]`, or `<|start|>assistant` does not leak a piece into content and then take it back. A call is emitted when its region is committed: JSON complete, or `finish()` after a string-aware close of the object. A name that was not a closed string in the source is not invented. Truncated JSON is closed the same way on the batch path and the stream path (`{"city": "Par"}` stays a string, a dangling colon becomes null, a trailing comma is dropped). Qwen parameter values stay strings. Repair ids are `call_0000`, `call_0001`, … Finish reason is `tool_calls` when a call was committed, `length` when a tool start was seen and no name was closed, and `stop` otherwise.

`splitcall parse` only implements this parser.

## Corpus

The strings are written from published wire formats: Hermes JSON inside `<tool_call>`, Qwen3-coder `<function=>` / `<parameter=>`, Mistral JSON-array and compact `[ARGS]`, Llama 3 `<|python_tag|>` JSON (plus a single-quoted dict), and OpenAI Harmony commentary channels. Edges cover unicode, nested objects, braces inside strings, parallel calls, and truncation. They are not outputs of a live model. Nothing here was sampled from an API.

## Limits

- Splits are Unicode code points, not UTF-8 bytes and not tokenizer pieces. A marker that is one special token in production is still cut apart here. That shows up in the results, and it is not a claim about BPE boundaries.
- The corpus is hand-written. It is not a conformance certificate for a model.
- vLLM and SGLang are scored as parser classes. HTTP serving, which assigns `finish_reason` and, for SGLang, the OpenAI id and the streamed index remap, is not imported. Those fields are gaps, not fake passes.
- vLLM Harmony is not included. llama.cpp and Ollama are not included.
- The repair parser can pass while disagreeing with an upstream quirk on purpose. The Mistral prose sentence and the single-quoted Llama dict are two of those.
- Schema types passed to SGLang are strings, so `"Paris"` stays a string. vLLM is called with `tools=None` and does not coerce through that schema.

## Development

```bash
.venv/bin/python -m pytest -q
```

CI is `.github/workflows/ci.yml` (Python 3.12, `pip install -e ".[dev]"`, `pytest -q`). GitHub Pages is `.github/workflows/pages.yml`, deploying the `site/` directory. The front page runs the repair parser and the chunk-split check in Pyodide. The matrix from this run is the second tab, black and white.

## License

MIT. Copyright (c) 2026 gfxroy. See [LICENSE](LICENSE).

The vendored vLLM and SGLang parser sources are Apache-2.0. See [NOTICE](NOTICE) and `src/splitcall/vendor_path/LICENSE.Apache-2.0`.
