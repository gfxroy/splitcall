"""JSON and monochrome HTML for one fuzz matrix."""

from __future__ import annotations

import html
import json
from pathlib import Path

from splitcall import __version__
from splitcall.fuzz import Matrix

INVARIANT_ORDER = [
    "no_crash",
    "stream_equals_batch",
    "count",
    "index_ordinal",
    "index_stable",
    "id_present",
    "id_stable",
    "args_json",
    "finish_reason",
    "no_markup_leak",
    "type_function",
    "args_are_strings",
    "oracle_names",
    "oracle_args",
    "oracle_content",
]


def matrix_payload(matrix: Matrix) -> dict:
    return {
        "tool": "splitcall",
        "version": __version__,
        "elapsed_sec": matrix.elapsed_sec,
        "cases": matrix.cases,
        "splits": matrix.splits,
        "parsers": matrix.parsers,
        "formats": matrix.formats,
        "cells": matrix.cells,
        "invariants": matrix.invariants,
        "repair_failures": matrix.repair_failures,
        "repros": [
            {
                "parser": item.parser,
                "runtime": item.runtime,
                "format": item.format,
                "case_id": item.case_id,
                "invariant": item.invariant,
                "detail": item.detail,
                "split": item.split,
                "family": item.family,
                "text": item.text,
                "shrunk": item.shrunk or item.text,
                "batch": item.batch_summary,
                "stream": item.stream_summary,
            }
            for item in matrix.examples
        ],
        "extra": matrix.extra,
        "limits": [
            "Splits are Unicode code points, not UTF-8 bytes.",
            "Splits are not real tokenizer boundaries. A marker that is one special token in production is still cut apart here.",
            "Corpus strings are hand-written from published wire formats, not live model samples.",
            "vLLM and SGLang are the parser classes, not their HTTP serving layers. finish_reason and, for SGLang, tool-call ids are filled in by serving code that is not imported.",
            "The vLLM GptOss tool parser is an upstream stub and is not in the matrix. Harmony is the SGLang detector.",
        ],
    }


def write_outputs(matrix: Matrix, json_path: Path, html_paths: list[Path]) -> None:
    payload = matrix_payload(matrix)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    page = render_html(payload)
    for path in html_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(page, encoding="utf-8")


def _cell_class(bucket: dict | None) -> tuple[str, str]:
    if not bucket:
        return "skip", "—"
    fails = bucket.get("fail", 0)
    gaps = bucket.get("gap", 0)
    if fails:
        return "fail", f"{fails} fail"
    if gaps:
        return "gap", f"ok · {gaps} gap"
    if bucket.get("pass", 0):
        return "pass", "ok"
    return "skip", "—"


def render_html(payload: dict) -> str:
    formats = payload["formats"]
    parsers = payload["parsers"]
    header = "".join(f"<th>{html.escape(fmt)}</th>" for fmt in formats)
    rows = []
    for parser in parsers:
        cells = payload["cells"].get(parser, {})
        bits = [f"<th>{html.escape(parser)}</th>"]
        for fmt in formats:
            kind, label = _cell_class(cells.get(fmt))
            bits.append(f'<td class="{kind}">{html.escape(label)}</td>')
        rows.append("<tr>" + "".join(bits) + "</tr>")

    inv_header = "".join(f"<th>{html.escape(name)}</th>" for name in INVARIANT_ORDER)
    inv_rows = []
    for parser in parsers:
        buckets = payload["invariants"].get(parser, {})
        bits = [f"<th>{html.escape(parser)}</th>"]
        for name in INVARIANT_ORDER:
            kind, label = _cell_class(buckets.get(name))
            bits.append(f'<td class="{kind}">{html.escape(label)}</td>')
        inv_rows.append("<tr>" + "".join(bits) + "</tr>")

    repros = []
    for item in payload["repros"]:
        shrunk = item["shrunk"]
        repros.append(
            "<article>"
            f"<h3>{html.escape(item['parser'])} · {html.escape(item['invariant'])}</h3>"
            f"<p>{html.escape(item['case_id'])} · {html.escape(item['format'])} · "
            f"family {html.escape(item['family'])} · {html.escape(item['split'])} "
            f"on the shrunk text</p>"
            f"<p>{html.escape(item['detail'])}</p>"
            f"<pre>{html.escape(shrunk)}</pre>"
            f"<p class=\"mono\">batch {html.escape(item['batch'])}</p>"
            f"<p class=\"mono\">stream {html.escape(item['stream'])}</p>"
            "</article>"
        )
    if not repros:
        repros.append("<p>No failing split in this run.</p>")

    extras = []
    for item in payload["extra"]:
        extras.append(f"<pre>{html.escape(json.dumps(item, ensure_ascii=False, indent=2))}</pre>")

    limits = "".join(f"<li>{html.escape(line)}</li>" for line in payload["limits"])
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>splitcall</title>
<style>
  :root {{ color-scheme: light; }}
  html {{ background: #fff; color: #111; }}
  body {{
    margin: 0 auto;
    max-width: 1120px;
    padding: 56px 24px 96px;
    font-family: "Iowan Old Style", Palatino, "Palatino Linotype", Georgia, serif;
    font-size: 18px;
    line-height: 1.45;
  }}
  h1 {{ font-weight: 500; font-size: 48px; letter-spacing: -0.03em; margin: 0 0 8px; }}
  h2 {{ font-weight: 500; font-size: 26px; margin: 48px 0 12px; }}
  h3 {{ font-weight: 500; font-size: 18px; margin: 0 0 4px; font-family: ui-monospace, Menlo, Consolas, monospace; }}
  p {{ margin: 0 0 12px; }}
  a {{ color: #111; }}
  .dek {{ max-width: 40em; }}
  .meta {{ font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 13px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 12px; margin: 12px 0 8px; }}
  th, td {{
    border: 1px solid #111;
    padding: 6px 8px;
    text-align: left;
    vertical-align: top;
    font-family: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
    font-weight: 400;
  }}
  td.fail {{ background: #111; color: #fff; }}
  td.gap {{ background: #fff; color: #333; }}
  td.pass {{ background: #fff; }}
  td.skip {{ background: #fff; color: #888; }}
  article {{ border-top: 1px solid #111; padding: 16px 0; }}
  pre {{
    font-family: ui-monospace, Menlo, Consolas, monospace;
    font-size: 13px;
    white-space: pre-wrap;
    word-break: break-word;
    background: #111;
    color: #fff;
    padding: 12px 14px;
    margin: 8px 0;
  }}
  .mono {{ font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 12px; }}
  ul {{ padding-left: 1.2em; }}
  footer {{ margin-top: 48px; font-size: 14px; }}
</style>
</head>
<body>
<header>
  <h1>splitcall</h1>
  <p class="dek">Chunk-boundary fuzzer for streaming tool-call parsers. Each cell is one runtime parser against one wire format. A black cell has at least one failed check. Gaps are fields the parser class never emits.</p>
  <p class="meta">{payload["cases"]} cases · {payload["splits"]} splits · {payload["elapsed_sec"]}s · splitcall {html.escape(str(payload["version"]))}</p>
</header>
<h2>Formats</h2>
<table>
  <thead><tr><th>parser</th>{header}</tr></thead>
  <tbody>
    {''.join(rows)}
  </tbody>
</table>
<h2>Invariants</h2>
<table>
  <thead><tr><th>parser</th>{inv_header}</tr></thead>
  <tbody>
    {''.join(inv_rows)}
  </tbody>
</table>
<h2>Minimal repros</h2>
{''.join(repros)}
<h2>Probe</h2>
{''.join(extras)}
<h2>Limits</h2>
<ul>{limits}</ul>
<footer>
  <p><a href="https://github.com/gfxroy/splitcall">github.com/gfxroy/splitcall</a></p>
</footer>
</body>
</html>
"""
