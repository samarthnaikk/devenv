#!/usr/bin/env python3
"""Run the retrieval engine against the evaluation questions and store the results.

This harness drives the *real* retrieval engine (`ContextBuilderService`) exactly the
way the runtime does:

    ContextBuilderService(workspace, memory=..., provider_configs=...,
                          performance_mode=...)
    service.set_runtime_allowed_providers({"codex", "opencode"})   # builds the index
    service.build_runtime_memory_context(external_query)           # returns memory

It reads ONLY the "Part A / Part B" question text from the questions file. The answer
key section is parsed separately, *after* retrieval, purely to score the engine. The
engine never receives the answer key.

Benchmark hygiene: sessions created while generating the questions (2026-09-15 onward,
plus the known meta/explore titles) are excluded from the corpus, otherwise they would
leak the answers. Everything else (all Codex + OpenCode sessions) is available to the
engine.

Outputs (under ``.devenv/retrieval_eval/``):
  * results.json  - raw per-question retrieval output
  * report.md     - scoring table + per-question detail
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.runtime.context_builder import (
    ContextBuilderService,
    _build_query_variants,
    _default_provider_configs,
)

try:  # exact runtime query composition; fall back to the raw prompt if unavailable
    from core.runtime.kernels.runtime import _compose_external_memory_query
except Exception:  # pragma: no cover - defensive

    def _compose_external_memory_query(
        user_prompt: str, _conversation: list[dict[str, Any]]
    ) -> str:
        return user_prompt


DEFAULT_QUESTIONS = "retrieval_eval_questions.md"
OPENCODE_DB = Path(os.path.expanduser("~/.local/share/opencode/opencode.db"))
META_TITLE_PATTERNS = (
    "Retrieval engine branch changes review",
    "Explore retrieval engine code",
    "Mine ",
    "sessions (@explore subagent)",
)
STOPWORDS = {
    "with",
    "that",
    "this",
    "from",
    "have",
    "were",
    "which",
    "their",
    "there",
    "about",
    "into",
    "then",
    "than",
    "them",
    "these",
    "those",
    "when",
    "what",
    "where",
    "would",
    "could",
    "should",
    "because",
    "before",
    "after",
    "while",
    "still",
    "instead",
    "using",
    "error",
    "value",
    "field",
    "added",
    "fixed",
    "line",
    "file",
    "code",
}


# --------------------------------------------------------------------------- parsing


QUESTION_RE = re.compile(
    r"\*\*Q(\d+)\.\*\*\s*(.*?)(?=\n\n\*\*Q\d+\.\*\*|\n\n---|\Z)", re.S
)
CODEX_ID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
)
OPENCODE_ID_RE = re.compile(r"\bses_[A-Za-z0-9]+\b")


def parse_questions(text: str) -> list[dict[str, str]]:
    head = text.split("# Answer key", 1)[0]
    questions: list[dict[str, str]] = []
    for match in QUESTION_RE.finditer(head):
        question = re.sub(r"\s+", " ", match.group(2)).strip()
        questions.append({"id": f"Q{match.group(1)}", "question": question})
    return questions


def parse_answer_key(text: str) -> dict[str, dict[str, Any]]:
    key: dict[str, dict[str, Any]] = {}
    if "# Answer key" not in text:
        return key
    body = text.split("# Answer key", 1)[1]
    parts = re.split(r"\n## (Q\d+) —", body)
    for index in range(1, len(parts), 2):
        qid = parts[index]
        block = parts[index + 1]
        codex_ids = CODEX_ID_RE.findall(block)
        opencode_ids = OPENCODE_ID_RE.findall(block)
        proof_match = re.search(r"\*\*Proof:\*\*\s*(.*)", block)
        proof = proof_match.group(1).strip() if proof_match else ""
        key[qid] = {
            "session_ids": list(dict.fromkeys(codex_ids + opencode_ids)),
            "codex_ids": codex_ids,
            "opencode_ids": opencode_ids,
            "proof": proof,
            "keywords": proof_keywords(proof),
        }
    return key


def proof_keywords(proof: str) -> list[str]:
    tokens: set[str] = set()
    for word in re.findall(r"[A-Za-z_][A-Za-z0-9_.\-/]{4,}", proof):
        lowered = word.lower().strip("'\"`")
        if lowered not in STOPWORDS and not lowered.startswith(("http", "www")):
            tokens.add(lowered)
    for number in re.findall(r"\d+\.\d+|\d{2,}", proof):
        tokens.add(number)
    return sorted(tokens)


# ------------------------------------------------------------------------- meta filter


def load_meta_session_ids(cutoff_iso: str) -> set[str]:
    if not OPENCODE_DB.exists():
        return set()
    cutoff = datetime.fromisoformat(cutoff_iso).replace(tzinfo=timezone.utc)
    meta: set[str] = set()
    connection = sqlite3.connect(OPENCODE_DB)
    connection.row_factory = sqlite3.Row
    try:
        for row in connection.execute("select id, title, time_updated from session"):
            title = str(row["title"] or "")
            updated = datetime.fromtimestamp(
                int(row["time_updated"]) / 1000, tz=timezone.utc
            )
            if updated >= cutoff or any(
                pattern in title for pattern in META_TITLE_PATTERNS
            ):
                meta.add(str(row["id"]))
    finally:
        connection.close()
    return meta


def install_meta_filter(service: ContextBuilderService, meta_ids: set[str]) -> None:
    if not meta_ids:
        return
    for provider in service.providers.values():
        original = provider.list_sessions

        def make_filtered(original_list_sessions=original):
            def filtered() -> list[Any]:
                return [
                    summary
                    for summary in original_list_sessions()
                    if summary.session_id not in meta_ids
                ]

            return filtered

        provider.list_sessions = make_filtered()  # type: ignore[method-assign]


# --------------------------------------------------------------------------- retrieval


def summarise_match(match: dict[str, Any]) -> dict[str, Any]:
    summary = match.get("summary")
    return {
        "session_id": getattr(summary, "session_id", ""),
        "title": getattr(summary, "title", ""),
        "workspace_path": getattr(summary, "workspace_path", None),
        "provider": getattr(summary, "provider", ""),
        "score": match.get("score"),
        "content_score": match.get("content_score"),
        "semantic_score": match.get("semantic_score"),
        "semantic_rank": match.get("semantic_rank"),
        "fused_score": match.get("fused_score"),
        "strong_match": match.get("strong_match"),
    }


def run_question(
    service: ContextBuilderService,
    question_id: str,
    question: str,
    providers: list[str],
    *,
    diagnostics: bool,
) -> dict[str, Any]:
    external_query = _compose_external_memory_query(question, [])
    record: dict[str, Any] = {
        "id": question_id,
        "question": question,
        "external_query": external_query,
    }

    started = time.time()
    try:
        context, session_ids, metadata = service.build_runtime_memory_context(
            external_query
        )
        record["runtime"] = {
            "session_ids": list(session_ids),
            "context": context,
            "metadata": dict(metadata),
        }
    except Exception as exc:  # pragma: no cover
        record["runtime"] = {"error": f"{type(exc).__name__}: {exc}"}
    record["runtime_seconds"] = round(time.time() - started, 2)

    per_provider: dict[str, Any] = {}
    candidates: dict[str, Any] = {}
    for provider_name in providers:
        try:
            context, session_ids, metadata = (
                service._build_runtime_memory_context_for_provider(
                    external_query, provider_name=provider_name, max_lines=6
                )
            )
            per_provider[provider_name] = {
                "session_ids": list(session_ids),
                "context": context,
                "metadata": dict(metadata),
            }
        except Exception as exc:
            per_provider[provider_name] = {"error": f"{type(exc).__name__}: {exc}"}

        if diagnostics:
            try:
                provider = service._get_provider(provider_name)
                matches = service._select_relevant_sessions(provider, external_query)
                candidates[provider_name] = [
                    summarise_match(match) for match in matches
                ]
            except Exception as exc:
                candidates[provider_name] = [{"error": f"{type(exc).__name__}: {exc}"}]

    record["per_provider"] = per_provider
    record["candidates"] = candidates
    return record


# --------------------------------------------------------------------------- scoring


def position(ground_truth: set[str], sequence: list[str]) -> int | None:
    for index, item in enumerate(sequence, start=1):
        if item in ground_truth:
            return index
    return None


def _normalize_for_match(text: str) -> str:
    cleaned = re.sub(r"[`*_#]+", "", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def score_question(
    record: dict[str, Any],
    truth: dict[str, Any],
    providers: list[str],
    meta_ids: set[str],
) -> dict[str, Any]:
    ground_truth = set(truth.get("session_ids", []))
    runtime = record.get("runtime", {})
    runtime_ids = list(runtime.get("session_ids", []))

    provider_ids: list[str] = []
    for provider_name in providers:
        provider_ids.extend(
            record.get("per_provider", {}).get(provider_name, {}).get("session_ids", [])
        )

    candidate_ids: list[str] = []
    for provider_name in providers:
        for candidate in record.get("candidates", {}).get(provider_name, []):
            session_id = candidate.get("session_id")
            if session_id and session_id not in candidate_ids:
                candidate_ids.append(session_id)

    context_text = str(runtime.get("context", "")).lower()
    keywords = truth.get("keywords", [])
    covered = [keyword for keyword in keywords if keyword in context_text]
    context_coverage = round(len(covered) / len(keywords), 2) if keywords else 0.0
    proof = _normalize_for_match(str(truth.get("proof", "")))
    proof_present = bool(proof) and proof in _normalize_for_match(context_text)

    return {
        "id": record["id"],
        "ground_truth": sorted(ground_truth),
        "runtime_ids": runtime_ids,
        "runtime_rank": position(ground_truth, runtime_ids),
        "provider_ids": provider_ids,
        "provider_rank": position(ground_truth, provider_ids),
        "candidate_rank": position(ground_truth, candidate_ids),
        "candidate_ids": candidate_ids,
        "runtime_used_meta": [sid for sid in runtime_ids if sid in meta_ids],
        "keywords": keywords,
        "keywords_covered": covered,
        "context_coverage": context_coverage,
        "proof_present": proof_present,
    }


# --------------------------------------------------------------------------- reporting


def write_report(
    path: Path,
    scores: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
    questions: list[dict[str, str]],
    *,
    refresh_seconds: float,
    index_seconds: float,
    per_question_seconds: float,
) -> None:
    lines: list[str] = []
    lines.append("# Retrieval Engine Evaluation — Results\n")
    lines.append(
        f"Index build: {index_seconds:.1f}s · per-question retrieval: ~{per_question_seconds:.1f}s avg · "
        f"embedding refresh: {refresh_seconds:.1f}s\n"
    )
    lines.append(
        "`rank` = position of the ground-truth session in that ranked list (blank = not present).\n"
    )
    lines.append(
        "| Q | Runtime rank | Per-provider rank | Candidate rank | Context fact coverage | Proof present | Notes |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for score in scores:
        notes = []
        if score["runtime_used_meta"]:
            notes.append("meta/leakage in runtime result")
        notes.append(", ".join(score["keywords_covered"][:3]))
        lines.append(
            "| {id} | {rr} | {pr} | {cr} | {cov} | {proof} | {notes} |".format(
                id=score["id"],
                rr=score["runtime_rank"] or "—",
                pr=score["provider_rank"] or "—",
                cr=score["candidate_rank"] or "—",
                cov=score["context_coverage"],
                proof="yes" if score.get("proof_present") else "no",
                notes="; ".join(notes),
            )
        )

    lines.append("\n## Per-question detail\n")
    by_id = {question["id"]: question for question in questions}
    for score in scores:
        record = records[score["id"]]
        lines.append(f"### {score['id']}")
        lines.append(
            f"**Question:** {by_id.get(score['id'], {}).get('question', '')}\n"
        )
        lines.append(f"**Ground truth:** `{'`, `'.join(score['ground_truth'])}`\n")
        lines.append(
            f"**Runtime session ids:** `{'`, `'.join(score['runtime_ids']) or '(none)'}`\n"
        )
        candidates = score["candidate_ids"]
        lines.append(
            f"**Ranked candidates (semantic+lexical):** `{'`, `'.join(candidates) or '(none)'}`\n"
        )
        runtime_context = record.get("runtime", {}).get("context", "")
        lines.append("**Retrieved memory:**\n")
        lines.append("```")
        lines.append(runtime_context[:1200] if runtime_context else "(empty)")
        lines.append("```\n")

    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS)
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--providers", default="codex,opencode")
    parser.add_argument(
        "--performance-mode", default="high", choices=("low", "medium", "high")
    )
    parser.add_argument("--output-dir", default=".devenv/retrieval_eval")
    parser.add_argument(
        "--refresh-embeddings",
        action="store_true",
        help="embed any sessions missing vectors",
    )
    parser.add_argument(
        "--no-index", action="store_true", help="skip building the lexical index"
    )
    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="skip per-provider candidate capture",
    )
    parser.add_argument(
        "--exclude-after",
        default="2026-09-15T00:00:00",
        help="treat sessions updated after this as benchmark meta",
    )
    parser.add_argument("--questions-limit", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1, help="run the question set N times to check determinism")
    parser.add_argument(
        "--elimination",
        choices=("default", "on", "off"),
        default="default",
        help="enable/disable bounded context elimination for this run",
    )
    args = parser.parse_args()

    questions_path = Path(args.questions)
    text = questions_path.read_text(encoding="utf-8")
    questions = parse_questions(text)
    answer_key = parse_answer_key(text)
    if args.questions_limit:
        questions = questions[: args.questions_limit]

    providers = [name.strip() for name in args.providers.split(",") if name.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.elimination == "on":
        os.environ["DEVENV_CONTEXT_ELIMINATION"] = "1"
    elif args.elimination == "off":
        os.environ["DEVENV_CONTEXT_ELIMINATION"] = "0"

    meta_ids = load_meta_session_ids(args.exclude_after)
    print(
        f"questions: {len(questions)} · providers: {providers} · excluded meta sessions: {len(meta_ids)}"
    )

    service = ContextBuilderService(
        str(Path(args.workspace).resolve()),
        provider_configs=_default_provider_configs(),
        performance_mode=args.performance_mode,
    )
    install_meta_filter(service, meta_ids)

    refresh_seconds = 0.0
    if args.refresh_embeddings:
        started = time.time()
        for provider_name in providers:
            try:
                service.list_sessions(provider_name)
            except Exception as exc:
                print(f"  embedding refresh failed for {provider_name}: {exc}")
        refresh_seconds = time.time() - started
        print(f"embedding refresh: {refresh_seconds:.1f}s")

    index_seconds = 0.0
    if not args.no_index:
        started = time.time()
        service.set_runtime_allowed_providers(set(providers))
        while True:
            status = service.indexing_status()
            if status.get("completed") or (
                not status.get("active") and status.get("finished_at")
            ):
                break
            if time.time() - started > 900:
                print("index build timed out; continuing without a complete index")
                break
            time.sleep(0.5)
        index_seconds = time.time() - started
        print(
            f"index build: {index_seconds:.1f}s ({service.indexing_status().get('message')})"
        )

    runs: list[dict[str, dict[str, Any]]] = []
    scores: list[dict[str, Any]] = []
    started = time.time()
    for repeat in range(max(args.repeat, 1)):
        run_records: dict[str, dict[str, Any]] = {}
        run_scores: list[dict[str, Any]] = []
        for question in questions:
            record = run_question(
                service,
                question["id"],
                question["question"],
                providers,
                diagnostics=not args.no_diagnostics,
            )
            run_records[question["id"]] = record
            truth = answer_key.get(question["id"])
            if truth:
                run_scores.append(score_question(record, truth, providers, meta_ids))
            if repeat == 0:
                print(
                    f"  {question['id']} runtime_rank={run_scores[-1]['runtime_rank'] if truth else '?'} "
                    f"candidate_rank={run_scores[-1]['candidate_rank'] if truth else '?'} "
                    f"({record.get('runtime_seconds', 0)}s)"
                )
        runs.append(run_records)
        if not scores:
            scores = run_scores
    per_question_seconds = (time.time() - started) / max(len(questions) * max(args.repeat, 1), 1)

    records = runs[-1]
    stability: dict[str, bool] = {}
    if len(runs) > 1:
        for question in questions:
            qid = question["id"]
            variants = {
                tuple(run.get(qid, {}).get("runtime", {}).get("session_ids", []))
                for run in runs
            }
            stability[qid] = len(variants) == 1

    (output_dir / "results.json").write_text(
        json.dumps({"records": records, "scores": scores, "stability": stability}, indent=2),
        encoding="utf-8",
    )
    write_report(
        output_dir / "report.md",
        scores,
        records,
        questions,
        refresh_seconds=refresh_seconds,
        index_seconds=index_seconds,
        per_question_seconds=per_question_seconds,
    )

    runtime_hits = sum(1 for score in scores if score["runtime_rank"])
    candidate_hits = sum(1 for score in scores if score["candidate_rank"])
    if stability:
        stable_count = sum(1 for value in stability.values() if value)
        print(
            f"Determinism: {stable_count}/{len(stability)} questions identical across {len(runs)} runs"
        )
        for qid, value in stability.items():
            if not value:
                print(f"  unstable: {qid}")
    print(
        f"\nRuntime-result recall: {runtime_hits}/{len(scores)} · "
        f"Candidate recall: {candidate_hits}/{len(scores)} · "
        f"saved {output_dir / 'results.json'} and {output_dir / 'report.md'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
