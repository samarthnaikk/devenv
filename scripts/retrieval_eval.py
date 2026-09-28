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
    "Devenv: devenv",
    "New session -",
    "OpenCode as reasoning layer",
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


def load_live_opencode_ids() -> set[str]:
    if not OPENCODE_DB.exists():
        return set()
    connection = sqlite3.connect(OPENCODE_DB)
    try:
        return {str(row[0]) for row in connection.execute("select id from session")}
    finally:
        connection.close()


def install_meta_filter(service: ContextBuilderService, meta_ids: set[str]) -> None:
    """Install the benchmark meta filter and drop stale/dead sessions.

    The engine indexes whatever ``provider.list_sessions()`` returns, so a session
    deleted from the source store keeps scoring until it is filtered here. Live
    OpenCode ids are read once and any indexed session not present is treated as
    stale (this is where deleted ``ses_...`` sessions and eval-only sessions leak).
    """
    live_opencode_ids = load_live_opencode_ids()
    for provider_name, provider in service.providers.items():
        original = provider.list_sessions

        def make_filtered(
            original_list_sessions=original,
            provider_name=provider_name,
            live_ids=live_opencode_ids,
        ):
            def filtered() -> list[Any]:
                kept: list[Any] = []
                for summary in original_list_sessions():
                    session_id = getattr(summary, "session_id", "")
                    if session_id in meta_ids:
                        continue
                    if provider_name == "opencode" and live_ids and session_id not in live_ids:
                        continue
                    source_path = getattr(summary, "source_path", "")
                    if source_path and not Path(source_path).exists():
                        continue
                    kept.append(summary)
                return kept

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
    orchestrator: Any | None = None,
    answer_core: Any | None = None,
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

    if orchestrator is not None and getattr(orchestrator, "uses_selector", False):
        selector_started = time.time()
        try:
            selector_context, selector_ids, selector_metadata = orchestrator.select(
                external_query, max_lines=12
            )
            record["selector"] = {
                "session_ids": list(selector_ids),
                "context": selector_context,
                "metadata": dict(selector_metadata),
                "seconds": round(time.time() - selector_started, 2),
            }
        except Exception as exc:  # pragma: no cover - defensive
            record["selector"] = {"error": f"{type(exc).__name__}: {exc}"}

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

    relevant_ids: set[str] = set(record.get("runtime", {}).get("session_ids", []))
    for provider_name in providers:
        for candidate in candidates.get(provider_name, []):
            session_id = candidate.get("session_id")
            if session_id:
                relevant_ids.add(session_id)
    workspace_by_session: dict[str, str | None] = {}
    try:
        for provider_name in providers:
            provider = service._get_provider(provider_name)
            for summary in provider.list_sessions():
                if summary.session_id in relevant_ids:
                    workspace_by_session[summary.session_id] = summary.workspace_path
    except Exception:  # pragma: no cover - workspace capture is best effort
        pass
    record["workspace_by_session"] = workspace_by_session
    if answer_core is not None:
        context_for_answer = (
            record.get("selector", {}).get("context")
            or record.get("runtime", {}).get("context", "")
        )
        record["answer"] = _answer_with_core(answer_core, question, context_for_answer)
    return record


# --------------------------------------------------------------------------- selector / answer helpers


def _build_opencode_core(workspace: str, model: str) -> Any:
    from core.ai.routing import OpenCodeAICore

    return OpenCodeAICore(workspace_path=workspace, model=(model or None))


def _build_orchestrator(
    service: ContextBuilderService,
    workspace: str,
    selector_model: str,
    answer_model: str,
) -> Any:
    from core.runtime.session_selection import build_session_orchestrator

    os.environ["DEVENV_SESSION_SELECTOR"] = "1"
    ai = _build_opencode_core(workspace, answer_model)
    return build_session_orchestrator(
        service, ai, service.workspace_path, selector_model=selector_model
    )


def _answer_with_core(answer_core: Any, question: str, context: str) -> str:
    system = (
        "You are a careful engineering assistant. Answer using ONLY the provided "
        "prior-session context. Quote exact identifiers, filenames, and values."
    )
    user = (
        f"QUESTION:\n{question}\n\n"
        f"PRIOR SESSION CONTEXT:\n{context or '(empty)'}\n\n"
        "Answer the question as fully as the context allows:"
    )
    try:
        response = answer_core.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}]
        )
        return str(getattr(response, "content", "") or "")
    except Exception as exc:  # pragma: no cover - backend failures
        return f"(answer model error: {type(exc).__name__}: {exc})"


def build_custom_score(
    record: dict[str, Any],
    truth_ids: list[str],
    expects: list[str],
) -> dict[str, Any]:
    ground_truth = set(truth_ids)
    runtime_ids = list(record.get("runtime", {}).get("session_ids", []))
    selector_ids = list(record.get("selector", {}).get("session_ids", []))
    candidate_ids: list[str] = []
    for candidates in record.get("candidates", {}).values():
        for candidate in candidates:
            session_id = candidate.get("session_id")
            if session_id and session_id not in candidate_ids:
                candidate_ids.append(session_id)
    engine_context = str(record.get("runtime", {}).get("context", "")).lower()
    selector_context = str(record.get("selector", {}).get("context", "")).lower()
    answer = str(record.get("answer", ""))
    answer_lower = answer.lower()

    def coverage(text: str) -> list[str]:
        return [token for token in expects if token.lower() in text]

    selector_metadata = record.get("selector", {}).get("metadata", {})
    return {
        "id": record["id"],
        "question": record["question"],
        "ground_truth": sorted(ground_truth),
        "candidate_ids": candidate_ids,
        "candidate_rank": position(ground_truth, candidate_ids),
        "runtime_ids": runtime_ids,
        "runtime_rank": position(ground_truth, runtime_ids),
        "selector_ids": selector_ids,
        "selector_rank": position(ground_truth, selector_ids) if selector_ids else None,
        "engine_context_coverage": coverage(engine_context),
        "selector_context_coverage": coverage(selector_context) if selector_ids else [],
        "answer_coverage": coverage(answer_lower) if answer else [],
        "selector_confidence": selector_metadata.get("selector_confidence"),
        "selector_attempts": selector_metadata.get("selector_attempts"),
        "selector_evidence_used": selector_metadata.get("selector_evidence_used"),
        "selector_abstained": selector_metadata.get("selector_abstained"),
        "selector_drill_line_count": selector_metadata.get("selector_drill_line_count"),
        "selector_engine_lines_used": selector_metadata.get("selector_engine_lines_used"),
        "selector_seconds": record.get("selector", {}).get("seconds"),
        "answer": answer,
    }


def run_custom_queries(
    service: ContextBuilderService,
    providers: list[str],
    queries: list[str],
    truth_ids: list[str],
    expects: list[str],
    *,
    orchestrator: Any | None,
    answer_core: Any | None,
    output_dir: Path,
) -> int:
    if not truth_ids:
        print("warning: no --truth provided; rank scoring will be blank")
    results: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for index, question in enumerate(queries, start=1):
        question_id = f"C{index}"
        print(f"\n=== {question_id}: {question}")
        record = run_question(
            service,
            question_id,
            question,
            providers,
            diagnostics=True,
            orchestrator=orchestrator,
            answer_core=answer_core,
        )
        score = build_custom_score(record, truth_ids, expects)
        results.append((record, score))
        print(f"  candidate_rank={score['candidate_rank']} runtime_rank={score['runtime_rank']} "
              f"selector_rank={score['selector_rank']} "
              f"confidence={score['selector_confidence']} attempts={score['selector_attempts']} "
              f"seconds={record.get('runtime_seconds', 0)}s")
        print(f"  candidates: {[c.get('session_id') for cs in record.get('candidates', {}).values() for c in cs][:12]}")
        print(f"  engine_context_covered={score['engine_context_coverage']}")
        print(f"  selector_context_covered={score['selector_context_coverage']}")
        print(
            f"  drill_lines={score.get('selector_drill_line_count')} "
            f"evidence_used={score.get('selector_evidence_used')} "
            f"engine_lines_used={score.get('selector_engine_lines_used')}"
        )
        if answer_core is not None:
            print(f"  answer_covered={score['answer_coverage']}")
            print("  ---- answer ----")
            print("  " + (score["answer"] or "(empty)").replace("\n", "\n  "))
            print("  ----------------")

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"scores": [score for _record, score in results],
               "records": {score["id"]: record for record, score in results}}
    (output_dir / "custom_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"\nsaved {output_dir / 'custom_results.json'}")
    return 0


# --------------------------------------------------------------------------- scoring


def position(ground_truth: set[str], sequence: list[str]) -> int | None:
    for index, item in enumerate(sequence, start=1):
        if item in ground_truth:
            return index
    return None


def precision_at_k(ground_truth: set[str], sequence: list[str], k: int) -> float:
    if not ground_truth or k <= 0:
        return 0.0
    window = sequence[:k]
    if not window:
        return 0.0
    hits = sum(1 for item in window if item in ground_truth)
    return round(hits / len(window), 3)


def ndcg_at_k(ground_truth: set[str], sequence: list[str], k: int) -> float:
    import math

    if not ground_truth or k <= 0:
        return 0.0
    dcg = 0.0
    for index, item in enumerate(sequence[:k], start=1):
        if item in ground_truth:
            dcg += 1.0 / math.log2(index + 1)
    ideal_hits = min(len(ground_truth), k)
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return round(dcg / idcg, 3)


def _normalize_workspace(path: str | None) -> str:
    if not path:
        return ""
    return os.path.normpath(str(path).strip()).replace("\\", "/").rstrip("/").lower()


def same_project_precision(
    sequence: list[str],
    workspace_by_session: dict[str, str | None],
    workspace_path: str | None,
    k: int,
) -> float:
    target = _normalize_workspace(workspace_path)
    if not target:
        return 0.0
    window = sequence[:k]
    if not window:
        return 0.0
    matches = sum(
        1
        for session_id in window
        if _normalize_workspace(workspace_by_session.get(session_id)) == target
    )
    return round(matches / len(window), 3)


def ground_truth_project(
    ground_truth: set[str],
    workspace_by_session: dict[str, str | None],
) -> str | None:
    for session_id in sorted(ground_truth):
        workspace = workspace_by_session.get(session_id)
        if workspace:
            return workspace
    return None


def _normalize_for_match(text: str) -> str:
    cleaned = re.sub(r"[`*_#]+", "", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def score_question(
    record: dict[str, Any],
    truth: dict[str, Any],
    providers: list[str],
    meta_ids: set[str],
    *,
    workspace_path: str | None = None,
) -> dict[str, Any]:
    ground_truth = set(truth.get("session_ids", []))
    runtime = record.get("runtime", {})
    runtime_ids = list(runtime.get("session_ids", []))
    workspace_by_session = dict(record.get("workspace_by_session", {}) or {})

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
    selector_record = record.get("selector", {}) or {}
    selector_ids = list(selector_record.get("session_ids", []) or [])
    truth_project = ground_truth_project(ground_truth, workspace_by_session)
    target_workspace = _normalize_workspace(workspace_path)

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
        "precision_at_1": precision_at_k(ground_truth, runtime_ids, 1),
        "ndcg_at_4": ndcg_at_k(ground_truth, runtime_ids, 4),
        "same_project_precision_at_3": same_project_precision(
            runtime_ids, workspace_by_session, workspace_path, 3
        ),
        "same_project_precision_at_12": same_project_precision(
            runtime_ids, workspace_by_session, workspace_path, 12
        ),
        "ground_truth_project": truth_project,
        "ground_truth_same_project": bool(
            target_workspace
            and _normalize_workspace(truth_project) == target_workspace
        ),
        "selector_ids": selector_ids,
        "selector_rank": position(ground_truth, selector_ids) if selector_ids else None,
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
        "| Q | Runtime rank | Per-provider rank | Candidate rank | Project P@3 | P@1 | nDCG@4 | Sel. rank | Context fact coverage | Proof present | Notes |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for score in scores:
        notes = []
        if score["runtime_used_meta"]:
            notes.append("meta/leakage in runtime result")
        if score.get("ground_truth_project") and not score.get("ground_truth_same_project"):
            notes.append("truth in another project")
        notes.append(", ".join(score["keywords_covered"][:3]))
        lines.append(
            "| {id} | {rr} | {pr} | {cr} | {sp3} | {p1} | {ndcg} | {sr} | {cov} | {proof} | {notes} |".format(
                id=score["id"],
                rr=score["runtime_rank"] or "—",
                pr=score["provider_rank"] or "—",
                cr=score["candidate_rank"] or "—",
                sp3=score.get("same_project_precision_at_3", 0.0),
                p1=score.get("precision_at_1", 0.0),
                ndcg=score.get("ndcg_at_4", 0.0),
                sr=score.get("selector_rank") or "—",
                cov=score["context_coverage"],
                proof="yes" if score.get("proof_present") else "no",
                notes="; ".join(notes),
            )
        )

    lines.append("\n## Aggregate\n")
    total = len(scores) or 1
    runtime_recall = sum(1 for score in scores if score["runtime_rank"])
    same_project_truth = sum(1 for score in scores if score.get("ground_truth_same_project"))
    selector_ranked = sum(1 for score in scores if score.get("selector_rank"))
    lines.append(
        f"- Runtime-result recall: {runtime_recall}/{len(scores)}\n"
        f"- Ground-truth same-project: {same_project_truth}/{len(scores)}"
        f" (other-project questions: {len(scores) - same_project_truth})\n"
        f"- Same-project precision@3 (mean): "
        f"{round(sum(s.get('same_project_precision_at_3', 0.0) for s in scores) / total, 3)}\n"
        f"- Same-project precision@12 (mean): "
        f"{round(sum(s.get('same_project_precision_at_12', 0.0) for s in scores) / total, 3)}\n"
        f"- P@1 (mean): {round(sum(s.get('precision_at_1', 0.0) for s in scores) / total, 3)}\n"
        f"- nDCG@4 (mean): {round(sum(s.get('ndcg_at_4', 0.0) for s in scores) / total, 3)}\n"
        f"- Selector rank available: {selector_ranked}/{len(scores)}\n"
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
    parser.add_argument(
        "--selector",
        action="store_true",
        help="also route retrieval through the session-selection orchestration layer",
    )
    parser.add_argument(
        "--selector-model",
        default=os.getenv("DEVENV_SESSION_SELECTOR_MODEL", ""),
        help="model id for the selection/reasoning role",
    )
    parser.add_argument(
        "--answer-model",
        default=os.getenv("OPENCODE_MODEL", ""),
        help="model id for the answering role",
    )
    parser.add_argument(
        "--answer",
        action="store_true",
        help="also run the answer model over the selected context",
    )
    parser.add_argument(
        "--query",
        action="append",
        default=[],
        help="ad-hoc multi-component question (repeatable); bypasses the questions file",
    )
    parser.add_argument(
        "--truth",
        action="append",
        default=[],
        help="ground-truth session id for --query scoring (repeatable)",
    )
    parser.add_argument(
        "--expect",
        action="append",
        default=[],
        help="expected substring in the context/answer for --query (repeatable)",
    )
    args = parser.parse_args()

    if args.query:
        os.makedirs(args.output_dir, exist_ok=True)
        providers = [name.strip() for name in args.providers.split(",") if name.strip()]
        meta_ids = load_meta_session_ids(args.exclude_after)
        service = ContextBuilderService(
            str(Path(args.workspace).resolve()),
            provider_configs=_default_provider_configs(),
            performance_mode=args.performance_mode,
        )
        install_meta_filter(service, meta_ids)
        service.set_runtime_allowed_providers(set(providers))
        while True:
            status = service.indexing_status()
            if status.get("completed") or (
                not status.get("active") and status.get("finished_at")
            ):
                break
            time.sleep(0.5)
        print(f"index build: {service.indexing_status().get('message')}")
        orchestrator = None
        if args.selector:
            orchestrator = _build_orchestrator(
                service, args.workspace, args.selector_model, args.answer_model
            )
        answer_core = _build_opencode_core(args.workspace, args.answer_model) if args.answer else None
        return run_custom_queries(
            service,
            providers,
            args.query,
            args.truth,
            args.expect,
            orchestrator=orchestrator,
            answer_core=answer_core,
            output_dir=Path(args.output_dir),
        )

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

    orchestrator = None
    if args.selector:
        orchestrator = _build_orchestrator(
            service, args.workspace, args.selector_model, args.answer_model
        )
    answer_core = (
        _build_opencode_core(args.workspace, args.answer_model) if args.answer else None
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
                orchestrator=orchestrator,
                answer_core=answer_core,
            )
            run_records[question["id"]] = record
            truth = answer_key.get(question["id"])
            if truth:
                run_scores.append(
                    score_question(
                        record,
                        truth,
                        providers,
                        meta_ids,
                        workspace_path=service.workspace_path,
                    )
                )
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
    if scores:
        total = len(scores)
        same_project_truth = sum(
            1 for score in scores if score.get("ground_truth_same_project")
        )
        print(
            f"Same-project precision@3: "
            f"{sum(s.get('same_project_precision_at_3', 0.0) for s in scores) / total:.3f} · "
            f"P@1: {sum(s.get('precision_at_1', 0.0) for s in scores) / total:.3f} · "
            f"nDCG@4: {sum(s.get('ndcg_at_4', 0.0) for s in scores) / total:.3f} · "
            f"truth-in-project: {same_project_truth}/{total}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
