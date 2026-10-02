"""``devenv-decisions`` — inspect and evaluate the decision layer.

Commands:
  status                         show the configured provider/mode
  eval <domain> [--corpus PATH]  run an evaluation and print metrics
  replay [--corpus-dir DIR]      offline heuristic regression gate (CI)
  show [--tail N] [--db-path P]  read recent decision.result audit events
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from core.decisions.config import DecisionConfig
from core.decisions.eval import (
    compare_to_baseline,
    default_baseline_dir,
    default_corpus_dir,
    evaluate,
    load_corpus,
)


def _resolve_corpus(domain: str, corpus: str | None) -> Path:
    if corpus:
        return Path(corpus)
    return default_corpus_dir() / f"{domain}.jsonl"


def _resolve_baseline(domain: str, baseline: str | None) -> Path:
    if baseline:
        return Path(baseline)
    return default_baseline_dir() / f"{domain}.json"


def _cmd_status(args: argparse.Namespace) -> int:
    config = DecisionConfig.from_env()
    payload = config.summary()
    payload["api_key_present"] = bool(config.api_key)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    config = DecisionConfig.from_env()
    if args.provider:
        config = DecisionConfig(**{**config.__dict__, "provider": args.provider})
    corpus_path = _resolve_corpus(args.domain, args.corpus)
    if not corpus_path.exists():
        print(f"corpus not found: {corpus_path}", file=sys.stderr)
        return 2
    corpus = load_corpus(corpus_path)
    outcome = evaluate(args.domain, corpus, config=config)
    print(json.dumps(outcome.to_dict(), indent=2, sort_keys=True))
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    # Replay always uses the heuristic provider: it needs no key and never
    # touches the network, so it is safe to run on every commit.
    config = DecisionConfig.from_env()
    config = DecisionConfig(**{**config.__dict__, "provider": "heuristic"})
    corpus_dir = Path(args.corpus_dir) if args.corpus_dir else default_corpus_dir()
    baseline_dir = Path(args.baseline_dir) if args.baseline_dir else default_baseline_dir()
    tolerance = float(args.tolerance)

    failures: list[str] = []
    for domain in ("intent", "memory"):
        corpus_path = corpus_dir / f"{domain}.jsonl"
        if not corpus_path.exists():
            print(f"skip {domain}: corpus not found: {corpus_path}")
            continue
        outcome = evaluate(domain, load_corpus(corpus_path), config=config)
        metrics = outcome.metrics.to_dict()
        print(f"{domain}: {json.dumps(metrics, sort_keys=True)}")

        baseline_path = _resolve_baseline(domain, str(baseline_dir / f'{domain}.json'))
        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            ok, detail = compare_to_baseline(outcome, baseline, tolerance=tolerance)
            print(f"  baseline: {detail}")
            if not ok:
                failures.append(detail)
        else:
            print(f"  baseline not found: {baseline_path} (skipping comparison)")

    if failures:
        print("REGRESSION:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    from core.memory.storage import SQLiteMemoryStore

    store = SQLiteMemoryStore(args.db_path)
    events = store.list_runtime_events(event_type="decision.result", limit=args.tail)
    for event in reversed(events):
        payload = event.get("payload_json", "{}")
        try:
            parsed: Any = json.loads(payload)
        except (TypeError, ValueError):
            parsed = payload
        print(json.dumps({"ts": event.get("ts"), "payload": parsed}, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="devenv-decisions", description="Inspect and evaluate the decision layer.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status", help="show configured provider and gate modes")
    status.set_defaults(func=_cmd_status)

    evaluate_parser = subparsers.add_parser("eval", help="evaluate a gate against a corpus")
    evaluate_parser.add_argument("domain", choices=["intent", "memory"])
    evaluate_parser.add_argument("--corpus", default=None)
    evaluate_parser.add_argument("--provider", default=None, choices=["heuristic", "typesafe", "local", "auto"])
    evaluate_parser.set_defaults(func=_cmd_eval)

    replay = subparsers.add_parser("replay", help="offline heuristic regression gate")
    replay.add_argument("--corpus-dir", default=None)
    replay.add_argument("--baseline-dir", default=None)
    replay.add_argument("--tolerance", type=float, default=0.0)
    replay.set_defaults(func=_cmd_replay)

    show = subparsers.add_parser("show", help="read recent decision.result audit events")
    show.add_argument("--tail", type=int, default=20)
    show.add_argument("--db-path", default="memory.db")
    show.set_defaults(func=_cmd_show)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
