# Vendored parsers

splitcall imports the real streaming tool parsers. It does not reimplement them.

| Tree | Commit | License |
| --- | --- | --- |
| vLLM | `f6cbdda2b6f54fee19823db25b068a08fd0f2313` | Apache-2.0 |
| SGLang | `23a45fd33921164fa440fd7c28cc63341259c28c` | Apache-2.0 |

The commits are also in `src/splitcall/vendor_path/UPSTREAM_COMMITS.txt`.
`activate()` inserts `vendor_path` at the front of `sys.path`, so `import vllm`
and `import sglang` load this tree.

## What was copied unchanged

- vLLM Hermes parser, Qwen3 parser, Llama JSON parser, Mistral parser, and the
  declarative parser engine those three use.
- SGLang Hermes, Qwen3-coder, Mistral, Llama 3.2, and gpt-oss detectors, plus
  `HarmonyParser`.

## Shims

The full vLLM and SGLang packages import torch, xgrammar, and the HTTP server.
Those are not imported. Small modules under `vendor_path` stand in for the
leaves the parsers touch:

- `vllm/__init__.py` does not import torch.
- `vllm/tool_parsers/abstract_tool_parser.py` only keeps the buffers Hermes uses.
- `vllm/parser/abstract_parser.py` keeps the `StreamState` fields the engine
  constructs. It is not the xgrammar parser.
- Protocol modules are the pydantic models the parsers construct, not the
  HTTP schema.
- `sglang/srt/environ.py` is only `SGLANG_FORWARD_UNKNOWN_TOOLS=False`.
- `openai` is the real SDK, installed as a dependency, so response tool types
  resolve. There is no fake `openai` package on `vendor_path`.

Parser method bodies are the upstream files. A behavior in the matrix is theirs.

## Not included

- vLLM `GptOssToolParser`. Upstream raises `NotImplementedError`. Harmony in
  this matrix is SGLang's detector.
- Serving code that assigns `finish_reason` and, for SGLang, the OpenAI `id`
  and the final streamed `index`. The matrix scores those as gaps when the
  parser class does not emit them.
- llama.cpp and Ollama. This environment has no cmake, and Ollama's tool files
  are browser tools, not a model-output parser.
