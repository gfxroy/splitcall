"""Command line for the fuzzer and the repair parser."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from splitcall.corpus import corpus
from splitcall.repair import parse_repair


def _cmd_fuzz(args: argparse.Namespace) -> int:
    from splitcall.fuzz import run_fuzz
    from splitcall.report import write_outputs

    names = {item.strip() for item in args.parsers.split(",") if item.strip()} or None
    matrix = run_fuzz(
        quick=args.quick,
        parser_names=names,
        shrink=not args.no_shrink,
    )
    out = Path(args.out)
    site = Path(args.site)
    write_outputs(
        matrix,
        out / "matrix.json",
        [out / "matrix.html", site / "index.html"],
    )
    print(
        json.dumps(
            {
                "splits": matrix.splits,
                "elapsed_sec": matrix.elapsed_sec,
                "repair_failures": matrix.repair_failures,
                "repros": len(matrix.examples),
                "json": str(out / "matrix.json"),
            }
        )
    )
    if args.strict and matrix.repair_failures:
        return 1
    return 0


def _cmd_parse(args: argparse.Namespace) -> int:
    text = args.text if args.text is not None else sys.stdin.read()
    if args.parser != "repair":
        print("parse only implements the repair parser", file=sys.stderr)
        return 2
    result = parse_repair(args.format, text)
    payload = {
        "content": result.content,
        "reasoning": result.reasoning,
        "finish_reason": result.finish_reason,
        "error": result.error,
        "calls": [
            {
                "index": call.index,
                "id": call.id,
                "name": call.name,
                "arguments": call.arguments,
                "type": call.type,
            }
            for call in result.calls
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if result.error is None else 1


def _cmd_list(args: argparse.Namespace) -> int:
    for case in corpus():
        calls = ",".join(call.name for call in case.expect_calls) or "-"
        print(f"{case.id}\t{case.format}\t{case.subset or '-'}\t{calls}\t{len(case.text)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="splitcall")
    sub = parser.add_subparsers(dest="cmd", required=True)

    fuzz = sub.add_parser("fuzz", help="run the chunk-boundary matrix")
    fuzz.add_argument("-o", "--out", default="results")
    fuzz.add_argument("--site", default="site")
    fuzz.add_argument("--no-shrink", action="store_true")
    fuzz.add_argument("--quick", action="store_true")
    fuzz.add_argument("--parsers", default="")
    fuzz.add_argument("--strict", action="store_true")
    fuzz.set_defaults(func=_cmd_fuzz)

    parse = sub.add_parser("parse", help="parse one buffer with the repair parser")
    parse.add_argument("--format", required=True, choices=["hermes", "qwen", "mistral", "llama", "harmony"])
    parse.add_argument("--parser", default="repair")
    parse.add_argument("text", nargs="?")
    parse.set_defaults(func=_cmd_parse)

    listed = sub.add_parser("list-corpus", help="print the corpus")
    listed.set_defaults(func=_cmd_list)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
