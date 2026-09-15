from __future__ import annotations

import json
import sqlite3
import tempfile
import time
import unittest
from unittest import mock
from pathlib import Path

from core.memory.models import ExternalSessionChunkEmbedding
from core.memory.storage import SQLiteMemoryStore
from core.runtime.context_builder import ContextBuilderService
from core.runtime.models import (
    ExternalSessionProviderConfig,
    ExternalSessionSummary,
    PreparedPromptRequest,
)


class ContextBuilderServiceTest(unittest.TestCase):
    @staticmethod
    def _create_opencode_db(path: Path) -> None:
        connection = sqlite3.connect(path)
        try:
            connection.executescript(
                """
                create table session (
                    id text primary key,
                    project_id text not null,
                    parent_id text,
                    slug text not null,
                    directory text not null,
                    title text not null,
                    version text not null,
                    share_url text,
                    summary_additions integer,
                    summary_deletions integer,
                    summary_files integer,
                    summary_diffs text,
                    revert text,
                    permission text,
                    time_created integer not null,
                    time_updated integer not null,
                    time_compacting integer,
                    time_archived integer,
                    workspace_id text
                );
                create table message (
                    id text primary key,
                    session_id text not null,
                    time_created integer not null,
                    time_updated integer not null,
                    data text not null
                );
                create table part (
                    id text primary key,
                    message_id text not null,
                    session_id text not null,
                    time_created integer not null,
                    time_updated integer not null,
                    data text not null
                );
                """
            )
            connection.commit()
        finally:
            connection.close()
    def test_codex_provider_parses_session_index_and_detail(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            workspace = Path(tempdir) / "workspace"
            workspace.mkdir()
            (workspace / "README.md").write_text("Devenv workspace for prompt builder tests.", encoding="utf-8")

            codex_root = Path(tempdir) / ".codex"
            sessions_dir = codex_root / "sessions" / "2026" / "06" / "28"
            sessions_dir.mkdir(parents=True)
            session_id = "session-123"
            (codex_root / "session_index.jsonl").write_text(
                json.dumps(
                    {
                        "id": session_id,
                        "thread_name": "Integrate prompt builder",
                        "updated_at": "2026-06-28T10:10:10Z",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (codex_root / "history.jsonl").write_text(
                json.dumps({"session_id": session_id, "ts": 1, "text": "Please wire the context builder."}) + "\n",
                encoding="utf-8",
            )
            session_file = sessions_dir / f"rollout-2026-06-28T10-09-00-{session_id}.jsonl"
            session_file.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "timestamp": "2026-06-28T10:09:00Z",
                                "type": "session_meta",
                                "payload": {
                                    "id": session_id,
                                    "cwd": str(workspace),
                                    "source": "vscode",
                                    "model_provider": "openai",
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "timestamp": "2026-06-28T10:09:05Z",
                                "type": "event_msg",
                                "payload": {
                                    "type": "agent_message",
                                    "message": "I inspected the repo and found the web runtime entrypoint.",
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "timestamp": "2026-06-28T10:09:08Z",
                                "type": "response_item",
                                "payload": {
                                    "type": "message",
                                    "role": "assistant",
                                    "content": [
                                        {"type": "output_text", "text": "Next I will prepare the prompt preview UI."}
                                    ],
                                },
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            service = ContextBuilderService(
                str(workspace),
                provider_configs=(
                    ExternalSessionProviderConfig(provider="codex", root_path=str(codex_root), index_path="session_index.jsonl"),
                ),
            )

            sessions = service.list_sessions("codex")
            detail = service.get_session("codex", session_id)

        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0].title, "Integrate prompt builder")
        self.assertEqual(sessions[0].updated_at, "2026-06-28T10:10:10Z")
        self.assertEqual(sessions[0].unified_session_id, f"codex:{session_id}")
        self.assertTrue(len(sessions[0].embedding) > 0)
        self.assertEqual(detail.summary.workspace_path, str(workspace))
        self.assertEqual(sessions[0].workspace_path, str(workspace))
        self.assertEqual(detail.metadata["unified_session_id"], f"codex:{session_id}")
        self.assertEqual(detail.metadata["embedding"], list(sessions[0].embedding))
        self.assertTrue(any(message.role == "user" for message in detail.messages))
        self.assertTrue(any("prompt preview ui" in message.content.lower() for message in detail.messages))

    def test_context_builder_persists_unified_session_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            workspace = Path(tempdir) / "workspace"
            workspace.mkdir()

            codex_root = Path(tempdir) / ".codex"
            sessions_dir = codex_root / "sessions" / "2026" / "08" / "20"
            sessions_dir.mkdir(parents=True)
            session_id = "session-store-1"
            (codex_root / "session_index.jsonl").write_text(
                json.dumps({"id": session_id, "thread_name": "Store embeddings", "updated_at": "2026-08-20T10:10:10Z"}) + "\n",
                encoding="utf-8",
            )
            (sessions_dir / f"rollout-2026-08-20T10-09-00-{session_id}.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"timestamp": "2026-08-20T10:09:00Z", "type": "session_meta", "payload": {"id": session_id, "cwd": str(workspace)}}),
                        json.dumps({"timestamp": "2026-08-20T10:09:02Z", "type": "event_msg", "payload": {"type": "agent_message", "message": "Persist the whole-session embedding for later reuse."}}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            service = ContextBuilderService(
                str(workspace),
                provider_configs=(
                    ExternalSessionProviderConfig(provider="codex", root_path=str(codex_root), index_path="session_index.jsonl"),
                ),
            )

            sessions = service.list_sessions("codex")
            embeddings = service.list_session_embeddings("codex")

        self.assertEqual(len(sessions), 1)
        self.assertEqual(len(embeddings), 1)
        self.assertEqual(embeddings[0].unified_session_id, f"codex:{session_id}")
        self.assertEqual(embeddings[0].session_id, session_id)
        self.assertEqual(tuple(sessions[0].embedding), embeddings[0].embedding)

    def test_select_relevant_sessions_fuses_semantic_only_match(self) -> None:
        class _FakeEmbedder:
            dimension = 2

            def embed(self, text: str) -> list[float]:
                lowered = text.lower()
                if any(marker in lowered for marker in ("database", "schema", "migration", "deployment", "incident")):
                    return [1.0, 0.0]
                if any(marker in lowered for marker in ("css", "button", "styling")):
                    return [0.0, 1.0]
                return [0.0, 0.0]

        class _FakeMemory:
            def __init__(self, store: object, embedder: object) -> None:
                self.store = store
                self.embedder = embedder

        with tempfile.TemporaryDirectory() as tempdir:
            workspace = Path(tempdir) / "workspace"
            workspace.mkdir()
            codex_root = Path(tempdir) / ".codex"
            sessions_dir = codex_root / "sessions" / "2026" / "08" / "20"
            sessions_dir.mkdir(parents=True)
            (codex_root / "session_index.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"id": "session-db", "thread_name": "Database project", "updated_at": "2026-08-20T10:00:00Z"}),
                        json.dumps({"id": "session-css", "thread_name": "Button styling", "updated_at": "2026-08-19T10:00:00Z"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (sessions_dir / "rollout-2026-08-20T09-00-00-session-db.jsonl").write_text(
                json.dumps({"timestamp": "2026-08-20T09:00:00Z", "type": "session_meta", "payload": {"id": "session-db", "cwd": str(workspace)}})
                + "\n"
                + json.dumps({"timestamp": "2026-08-20T09:00:01Z", "type": "event_msg", "payload": {"type": "agent_message", "message": "Applied the database schema migration for the release."}})
                + "\n",
                encoding="utf-8",
            )
            (sessions_dir / "rollout-2026-08-19T09-00-00-session-css.jsonl").write_text(
                json.dumps({"timestamp": "2026-08-19T09:00:00Z", "type": "session_meta", "payload": {"id": "session-css", "cwd": str(workspace)}})
                + "\n"
                + json.dumps({"timestamp": "2026-08-19T09:00:01Z", "type": "event_msg", "payload": {"type": "agent_message", "message": "Tweaked the button styling tokens in the stylesheet."}})
                + "\n",
                encoding="utf-8",
            )

            store = SQLiteMemoryStore(str(workspace / "memory.db"))
            service = ContextBuilderService(
                str(workspace),
                memory=_FakeMemory(store=store, embedder=_FakeEmbedder()),
                provider_configs=(
                    ExternalSessionProviderConfig(provider="codex", root_path=str(codex_root), index_path="session_index.jsonl"),
                ),
            )
            service.list_sessions("codex")

            provider = service._get_provider("codex")
            matches = service._select_relevant_sessions(provider, "deployment incident")

        matched_ids = {match["summary"].session_id for match in matches}
        self.assertIn("session-db", matched_ids)
        self.assertNotIn("session-css", matched_ids)
        db_match = next(match for match in matches if match["summary"].session_id == "session-db")
        self.assertGreaterEqual(db_match["semantic_score"], 0.35)

    def test_session_embedding_document_cache_avoids_reparsing(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            workspace = Path(tempdir) / "workspace"
            workspace.mkdir()
            codex_root = Path(tempdir) / ".codex"
            sessions_dir = codex_root / "sessions" / "2026" / "08" / "20"
            sessions_dir.mkdir(parents=True)
            session_id = "session-cache"
            (codex_root / "session_index.jsonl").write_text(
                json.dumps({"id": session_id, "thread_name": "Cache session", "updated_at": "2026-08-20T10:00:00Z"}) + "\n",
                encoding="utf-8",
            )
            (sessions_dir / f"rollout-2026-08-20T09-00-00-{session_id}.jsonl").write_text(
                json.dumps({"timestamp": "2026-08-20T09:00:00Z", "type": "session_meta", "payload": {"id": session_id, "cwd": str(workspace)}})
                + "\n"
                + json.dumps({"timestamp": "2026-08-20T09:00:01Z", "type": "event_msg", "payload": {"type": "agent_message", "message": "Cache this document."}})
                + "\n",
                encoding="utf-8",
            )

            service = ContextBuilderService(
                str(workspace),
                provider_configs=(
                    ExternalSessionProviderConfig(provider="codex", root_path=str(codex_root), index_path="session_index.jsonl"),
                ),
            )
            provider = service._get_provider("codex")
            with mock.patch.object(provider, "build_index_chunks", wraps=provider.build_index_chunks) as spy:
                service.list_sessions("codex")
                service.list_sessions("codex")

        self.assertEqual(spy.call_count, 1)

    def test_codex_provider_ignores_developer_rows_and_reads_user_event_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            workspace = Path(tempdir) / "workspace"
            workspace.mkdir()

            codex_root = Path(tempdir) / ".codex"
            sessions_dir = codex_root / "sessions" / "2026" / "06" / "29"
            sessions_dir.mkdir(parents=True)
            session_id = "session-user-event"
            (codex_root / "session_index.jsonl").write_text(
                json.dumps(
                    {
                        "id": session_id,
                        "thread_name": "Review follow-up",
                        "updated_at": "2026-06-29T10:10:10Z",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (sessions_dir / f"rollout-2026-06-29T10-09-00-{session_id}.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "timestamp": "2026-06-29T10:09:00Z",
                                "type": "session_meta",
                                "payload": {"id": session_id, "cwd": str(workspace)},
                            }
                        ),
                        json.dumps(
                            {
                                "timestamp": "2026-06-29T10:09:01Z",
                                "type": "response_item",
                                "payload": {
                                    "type": "message",
                                    "role": "developer",
                                    "content": [{"type": "input_text", "text": "internal instructions"}],
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "timestamp": "2026-06-29T10:09:02Z",
                                "type": "event_msg",
                                "payload": {
                                    "type": "user_message",
                                    "message": "what were the review issues again?",
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "timestamp": "2026-06-29T10:09:03Z",
                                "type": "response_item",
                                "payload": {
                                    "type": "message",
                                    "role": "assistant",
                                    "content": [{"type": "output_text", "text": "Need to fix the remaining review items."}],
                                },
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            service = ContextBuilderService(
                str(workspace),
                provider_configs=(
                    ExternalSessionProviderConfig(provider="codex", root_path=str(codex_root), index_path="session_index.jsonl"),
                ),
            )
            detail = service.get_session("codex", session_id)

        roles = [message.role for message in detail.messages]
        contents = [message.content for message in detail.messages]
        self.assertEqual(roles, ["user", "assistant"])
        self.assertNotIn("internal instructions", " ".join(contents))
        self.assertIn("what were the review issues again?", contents[0].lower())

    def test_codex_provider_keeps_useful_function_call_output(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            workspace = Path(tempdir) / "workspace"
            workspace.mkdir()

            codex_root = Path(tempdir) / ".codex"
            sessions_dir = codex_root / "sessions" / "2026" / "06" / "29"
            sessions_dir.mkdir(parents=True)
            session_id = "session-tool-output"
            (codex_root / "session_index.jsonl").write_text(
                json.dumps({"id": session_id, "thread_name": "Reviewer notes", "updated_at": "2026-06-29T11:10:10Z"}) + "\n",
                encoding="utf-8",
            )
            (sessions_dir / f"rollout-2026-06-29T11-09-00-{session_id}.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"timestamp": "2026-06-29T11:09:00Z", "type": "session_meta", "payload": {"id": session_id, "cwd": str(workspace)}}),
                        json.dumps(
                            {
                                "timestamp": "2026-06-29T11:09:02Z",
                                "type": "response_item",
                                "payload": {
                                    "type": "function_call_output",
                                    "output": "Sharmil001 | File: reviews.md | Comment: move this logic to the backend action.",
                                },
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            service = ContextBuilderService(
                str(workspace),
                provider_configs=(
                    ExternalSessionProviderConfig(provider="codex", root_path=str(codex_root), index_path="session_index.jsonl"),
                ),
            )
            detail = service.get_session("codex", session_id)
            result = service.prepare_prompt(
                PreparedPromptRequest(
                    task="What did Sharmil say about the backend action?",
                    provider="codex",
                    include_workspace_scan=False,
                    include_prior_context=True,
                )
            )

        self.assertTrue(any(message.role == "tool" for message in detail.messages))
        self.assertIn("backend action", result.prompt.lower())
        self.assertNotIn("chunk id", result.prompt.lower())
        self.assertNotIn("operation not permitted", result.prompt.lower())

    def test_codex_provider_drops_noisy_function_call_output_from_runtime_context(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            workspace = Path(tempdir) / "workspace"
            workspace.mkdir()

            codex_root = Path(tempdir) / ".codex"
            sessions_dir = codex_root / "sessions" / "2026" / "08" / "12"
            sessions_dir.mkdir(parents=True)
            session_id = "session-noisy-tool-output"
            (codex_root / "session_index.jsonl").write_text(
                json.dumps({"id": session_id, "thread_name": "Improve retrieval", "updated_at": "2026-08-12T06:10:00Z"}) + "\n",
                encoding="utf-8",
            )
            (sessions_dir / f"rollout-2026-08-12T06-09-00-{session_id}.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"timestamp": "2026-08-12T06:09:00Z", "type": "session_meta", "payload": {"id": session_id, "cwd": str(workspace)}}),
                        json.dumps(
                            {
                                "timestamp": "2026-08-12T06:09:01Z",
                                "type": "response_item",
                                "payload": {
                                    "type": "function_call_output",
                                    "output": 'session_index_lines 49 {"id":"x","thread_name":"Fix search tool selection","updated_at":"2026-07-04T11:30:14Z"}',
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "timestamp": "2026-08-12T06:09:02Z",
                                "type": "event_msg",
                                "payload": {"type": "agent_message", "message": "We improved retrieval by splitting compound recall prompts and fusing the matches."},
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            service = ContextBuilderService(
                str(workspace),
                provider_configs=(
                    ExternalSessionProviderConfig(provider="codex", root_path=str(codex_root), index_path="session_index.jsonl"),
                ),
            )
            context, session_ids, _metadata = service.build_runtime_memory_context("improve retrieval")

        self.assertEqual(session_ids, (session_id,))
        self.assertIn("splitting compound recall prompts", context)
        self.assertNotIn("session_index_lines", context)
        self.assertNotIn('"thread_name"', context)

    def test_opencode_provider_reports_not_configured(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            workspace = Path(tempdir) / "workspace"
            workspace.mkdir()
            service = ContextBuilderService(
                str(workspace),
                provider_configs=(
                    ExternalSessionProviderConfig(provider="opencode", root_path=str(Path(tempdir) / ".opencode")),
                ),
            )

            health = service.list_sources()[0]
            sessions = service.list_sessions("opencode")

        self.assertFalse(health.available)
        self.assertEqual(sessions, [])

    def test_opencode_provider_reads_sqlite_sessions_and_chunks_runtime_context(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            workspace = Path(tempdir) / "workspace"
            workspace.mkdir()
            db_path = Path(tempdir) / "opencode.db"
            self._create_opencode_db(db_path)

            connection = sqlite3.connect(db_path)
            try:
                connection.execute(
                    "insert into session (id, project_id, parent_id, slug, directory, title, version, time_created, time_updated, workspace_id) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "ses_getdrip_1",
                        "proj_1",
                        None,
                        "get-drip-review",
                        "/Users/samarthnaik/Desktop/LoopedIn/get-drip",
                        "Review get-drip bugs",
                        "1",
                        1782809000000,
                        1782809100000,
                        None,
                    ),
                )
                connection.execute(
                    "insert into message (id, session_id, time_created, time_updated, data) values (?, ?, ?, ?, ?)",
                    (
                        "msg_user_1",
                        "ses_getdrip_1",
                        1782809001000,
                        1782809001000,
                        json.dumps({"role": "user"}),
                    ),
                )
                connection.execute(
                    "insert into part (id, message_id, session_id, time_created, time_updated, data) values (?, ?, ?, ?, ?, ?)",
                    (
                        "prt_user_1",
                        "msg_user_1",
                        "ses_getdrip_1",
                        1782809001001,
                        1782809001001,
                        json.dumps({"type": "text", "text": "What exact bugs did we fix in get-drip?"}),
                    ),
                )
                connection.execute(
                    "insert into message (id, session_id, time_created, time_updated, data) values (?, ?, ?, ?, ?)",
                    (
                        "msg_assistant_1",
                        "ses_getdrip_1",
                        1782809002000,
                        1782809002000,
                        json.dumps({"role": "assistant"}),
                    ),
                )
                connection.execute(
                    "insert into part (id, message_id, session_id, time_created, time_updated, data) values (?, ?, ?, ?, ?, ?)",
                    (
                        "prt_reason_1",
                        "msg_assistant_1",
                        "ses_getdrip_1",
                        1782809002001,
                        1782809002001,
                        json.dumps({"type": "reasoning", "text": "The PR review found an authentication bypass and an open email relay."}),
                    ),
                )
                connection.execute(
                    "insert into part (id, message_id, session_id, time_created, time_updated, data) values (?, ?, ?, ?, ?, ?)",
                    (
                        "prt_tool_1",
                        "msg_assistant_1",
                        "ses_getdrip_1",
                        1782809002002,
                        1782809002002,
                        json.dumps(
                            {
                                "type": "tool",
                                "tool": "bash",
                                "state": {
                                    "title": "Review findings",
                                    "output": "ISSUE-001 Critical Security Authentication bypass. ISSUE-002 Critical Security Open email relay.",
                                },
                            }
                        ),
                    ),
                )
                connection.commit()
            finally:
                connection.close()

            service = ContextBuilderService(
                str(workspace),
                provider_configs=(
                    ExternalSessionProviderConfig(provider="opencode", root_path=str(db_path)),
                ),
            )
            sessions = service.list_sessions("opencode")
            detail = service.get_session("opencode", "ses_getdrip_1")
            service.set_runtime_allowed_providers({"opencode"})
            for _ in range(40):
                if service.indexing_status()["completed"]:
                    break
                time.sleep(0.05)
            context, session_ids, metadata = service.build_runtime_memory_context(
                "what exact bugs did we fix in get-drip?",
                provider_name="opencode",
                max_lines=6,
            )

        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0].workspace_path, "/Users/samarthnaik/Desktop/LoopedIn/get-drip")
        self.assertTrue(any("authentication bypass" in message.content.lower() for message in detail.messages))
        self.assertEqual(session_ids, ("ses_getdrip_1",))
        self.assertTrue(metadata["index_ready"])
        self.assertIn("open email relay", context.lower())

    def test_opencode_provider_query_falls_back_to_immutable_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            db_path = Path(tempdir) / "opencode.db"
            self._create_opencode_db(db_path)
            provider = ContextBuilderService(
                str(Path(tempdir) / "workspace"),
                provider_configs=(
                    ExternalSessionProviderConfig(provider="opencode", root_path=str(db_path)),
                ),
            ).providers["opencode"]

            original_connect = sqlite3.connect

            def flaky_connect(target, *args, **kwargs):
                if target == str(db_path):
                    class FlakyConnection:
                        row_factory = None

                        def execute(self, *_args, **_kwargs):
                            raise sqlite3.OperationalError("unable to open database file")

                        def close(self):
                            return None

                    return FlakyConnection()
                return original_connect(target, *args, **kwargs)

            with mock.patch("core.runtime.context_builder.sqlite3.connect", side_effect=flaky_connect):
                rows = provider._query_all("select name from sqlite_master where type = 'table'")

        self.assertTrue(rows)

    def test_prepare_prompt_merges_session_and_workspace_context(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            workspace = Path(tempdir) / "workspace"
            workspace.mkdir()
            (workspace / "README.md").write_text("This repo ships a website runtime and context builder.", encoding="utf-8")
            (workspace / "interface").mkdir()

            codex_root = Path(tempdir) / ".codex"
            sessions_dir = codex_root / "sessions" / "2026" / "06" / "28"
            sessions_dir.mkdir(parents=True)
            session_id = "session-456"
            (codex_root / "session_index.jsonl").write_text(
                json.dumps(
                    {
                        "id": session_id,
                        "thread_name": "Website changes",
                        "updated_at": "2026-06-28T11:00:00Z",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (codex_root / "history.jsonl").write_text(
                json.dumps({"session_id": session_id, "ts": 2, "text": "Add a context builder panel with copy support."}) + "\n",
                encoding="utf-8",
            )
            (sessions_dir / f"rollout-2026-06-28T10-59-00-{session_id}.jsonl").write_text(
                json.dumps(
                    {
                        "timestamp": "2026-06-28T10:59:03Z",
                        "type": "event_msg",
                        "payload": {
                            "type": "agent_message",
                            "message": "The prompt panel should sit beside the chat flow and stay copy-paste only.",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            service = ContextBuilderService(
                str(workspace),
                provider_configs=(
                    ExternalSessionProviderConfig(provider="codex", root_path=str(codex_root), index_path="session_index.jsonl"),
                ),
            )
            result = service.prepare_prompt(
                PreparedPromptRequest(
                    task="Add the context builder with minimal changes and verify the UI.",
                    provider="codex",
                    session_ids=(session_id,),
                    include_workspace_scan=True,
                    include_prior_context=True,
                )
            )

        self.assertIn("Task:", result.prompt)
        self.assertIn("Relevant prior session context:", result.prompt)
        self.assertIn("Workspace context:", result.prompt)
        self.assertIn("Constraints:", result.prompt)
        self.assertTrue(any("minimal changes" in line.lower() for line in result.constraints))

    def test_prepare_prompt_auto_selects_relevant_sessions_when_none_are_provided(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            workspace = Path(tempdir) / "workspace"
            workspace.mkdir()
            (workspace / "README.md").write_text("This repo contains the devenv web runtime.", encoding="utf-8")

            codex_root = Path(tempdir) / ".codex"
            sessions_dir = codex_root / "sessions" / "2026" / "06" / "28"
            sessions_dir.mkdir(parents=True)
            relevant_session_id = "session-devenv"
            old_session_id = "session-other"
            (codex_root / "session_index.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": relevant_session_id,
                                "thread_name": "Integrate devenv context builder",
                                "updated_at": "2026-06-28T11:30:00Z",
                            }
                        ),
                        json.dumps(
                            {
                                "id": old_session_id,
                                "thread_name": "Unrelated notes",
                                "updated_at": "2026-06-20T11:30:00Z",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (codex_root / "history.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"session_id": relevant_session_id, "ts": 2, "text": "Prepare context for the devenv web runtime."}),
                        json.dumps({"session_id": old_session_id, "ts": 1, "text": "Something unrelated."}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (sessions_dir / f"rollout-2026-06-28T11-29-00-{relevant_session_id}.jsonl").write_text(
                json.dumps(
                    {
                        "timestamp": "2026-06-28T11:29:03Z",
                        "type": "session_meta",
                        "payload": {"id": relevant_session_id, "cwd": str(workspace)},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (sessions_dir / f"rollout-2026-06-20T11-29-00-{old_session_id}.jsonl").write_text(
                json.dumps(
                    {
                        "timestamp": "2026-06-20T11:29:03Z",
                        "type": "session_meta",
                        "payload": {"id": old_session_id, "cwd": "/tmp/other"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            service = ContextBuilderService(
                str(workspace),
                provider_configs=(
                    ExternalSessionProviderConfig(provider="codex", root_path=str(codex_root), index_path="session_index.jsonl"),
                ),
            )
            result = service.prepare_prompt(
                PreparedPromptRequest(
                    task="Update the devenv context builder UI.",
                    provider="codex",
                    include_workspace_scan=False,
                    include_prior_context=True,
                )
            )

        self.assertIn(relevant_session_id, result.session_ids)
        self.assertEqual(result.metadata["selection_mode"], "automatic")

    def test_prepare_prompt_auto_selects_session_when_match_exists_only_in_message_body(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            workspace = Path(tempdir) / "workspace"
            workspace.mkdir()

            codex_root = Path(tempdir) / ".codex"
            sessions_dir = codex_root / "sessions" / "2026" / "06" / "28"
            sessions_dir.mkdir(parents=True)
            message_match_session = "session-message-only"
            generic_session = "session-generic"
            (codex_root / "session_index.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"id": message_match_session, "thread_name": "Notes", "updated_at": "2026-06-28T12:00:00Z"}),
                        json.dumps({"id": generic_session, "thread_name": "Devenv work", "updated_at": "2026-06-28T13:00:00Z"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (sessions_dir / f"rollout-2026-06-28T12-00-00-{message_match_session}.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"timestamp": "2026-06-28T11:59:00Z", "type": "session_meta", "payload": {"id": message_match_session, "cwd": str(workspace)}}),
                        json.dumps({"timestamp": "2026-06-28T11:59:01Z", "type": "event_msg", "payload": {"type": "agent_message", "message": "Project Chimera needed a scraper retry path and TPM tuning."}}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (sessions_dir / f"rollout-2026-06-28T13-00-00-{generic_session}.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"timestamp": "2026-06-28T12:59:00Z", "type": "session_meta", "payload": {"id": generic_session, "cwd": str(workspace)}}),
                        json.dumps({"timestamp": "2026-06-28T12:59:01Z", "type": "event_msg", "payload": {"type": "agent_message", "message": "Worked on Devenv UI polish."}}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            service = ContextBuilderService(
                str(workspace),
                provider_configs=(
                    ExternalSessionProviderConfig(provider="codex", root_path=str(codex_root), index_path="session_index.jsonl"),
                ),
            )
            result = service.prepare_prompt(
                PreparedPromptRequest(
                    task="Do you know about Project Chimera?",
                    provider="codex",
                    include_workspace_scan=False,
                    include_prior_context=True,
                )
            )

        self.assertIn(message_match_session, result.session_ids)

    def test_prepare_prompt_prefers_exact_name_match_over_generic_same_workspace_session(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            workspace = Path(tempdir) / "workspace"
            workspace.mkdir()

            codex_root = Path(tempdir) / ".codex"
            sessions_dir = codex_root / "sessions" / "2026" / "06" / "28"
            sessions_dir.mkdir(parents=True)
            exact_match_session = "session-sharmil"
            generic_workspace_session = "session-devenv"
            (codex_root / "session_index.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"id": exact_match_session, "thread_name": "Reviewer follow-up", "updated_at": "2026-06-28T12:00:00Z"}),
                        json.dumps({"id": generic_workspace_session, "thread_name": "Devenv review work", "updated_at": "2026-06-28T13:00:00Z"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (sessions_dir / f"rollout-2026-06-28T12-00-00-{exact_match_session}.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"timestamp": "2026-06-28T11:59:00Z", "type": "session_meta", "payload": {"id": exact_match_session, "cwd": "/tmp/other"}}),
                        json.dumps({"timestamp": "2026-06-28T11:59:01Z", "type": "event_msg", "payload": {"type": "user_message", "message": "What did Sharmil say in review?"}}),
                        json.dumps({"timestamp": "2026-06-28T11:59:02Z", "type": "response_item", "payload": {"type": "function_call_output", "output": "Sharmil001 | Comment: use a convex action instead of the frontend route."}}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (sessions_dir / f"rollout-2026-06-28T13-00-00-{generic_workspace_session}.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"timestamp": "2026-06-28T12:59:00Z", "type": "session_meta", "payload": {"id": generic_workspace_session, "cwd": str(workspace)}}),
                        json.dumps({"timestamp": "2026-06-28T12:59:01Z", "type": "event_msg", "payload": {"type": "agent_message", "message": "Worked on generic devenv review fixes."}}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            service = ContextBuilderService(
                str(workspace),
                provider_configs=(
                    ExternalSessionProviderConfig(provider="codex", root_path=str(codex_root), index_path="session_index.jsonl"),
                ),
            )
            result = service.prepare_prompt(
                PreparedPromptRequest(
                    task="What were the issues Sharmil was talking about?",
                    provider="codex",
                    include_workspace_scan=False,
                    include_prior_context=True,
                )
            )

        self.assertEqual(result.session_ids[0], exact_match_session)

    def test_prepare_prompt_returns_no_sessions_when_query_has_no_real_match(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            workspace = Path(tempdir) / "workspace"
            workspace.mkdir()

            codex_root = Path(tempdir) / ".codex"
            sessions_dir = codex_root / "sessions" / "2026" / "06" / "28"
            sessions_dir.mkdir(parents=True)
            (codex_root / "session_index.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"id": "session-devenv", "thread_name": "Integrate Devenv", "updated_at": "2026-06-28T12:00:00Z"}),
                        json.dumps({"id": "session-review", "thread_name": "Review fixes", "updated_at": "2026-06-28T13:00:00Z"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (sessions_dir / "rollout-2026-06-28T12-00-00-session-devenv.jsonl").write_text(
                json.dumps({"timestamp": "2026-06-28T11:59:00Z", "type": "event_msg", "payload": {"type": "agent_message", "message": "Worked on the Devenv context builder."}})
                + "\n",
                encoding="utf-8",
            )
            (sessions_dir / "rollout-2026-06-28T13-00-00-session-review.jsonl").write_text(
                json.dumps({"timestamp": "2026-06-28T12:59:00Z", "type": "event_msg", "payload": {"type": "agent_message", "message": "Fixed review comments in another repo."}})
                + "\n",
                encoding="utf-8",
            )

            service = ContextBuilderService(
                str(workspace),
                provider_configs=(
                    ExternalSessionProviderConfig(provider="codex", root_path=str(codex_root), index_path="session_index.jsonl"),
                ),
            )
            result = service.prepare_prompt(
                PreparedPromptRequest(
                    task="Do you know about Project Atlas?",
                    provider="codex",
                    include_workspace_scan=False,
                    include_prior_context=True,
                )
            )

        self.assertEqual(result.session_ids, ())

    def test_prepare_prompt_treats_short_greeting_as_new_context(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            workspace = Path(tempdir) / "workspace"
            workspace.mkdir()

            codex_root = Path(tempdir) / ".codex"
            sessions_dir = codex_root / "sessions" / "2026" / "06" / "28"
            sessions_dir.mkdir(parents=True)
            session_id = "session-devenv"
            (codex_root / "session_index.jsonl").write_text(
                json.dumps({"id": session_id, "thread_name": "Integrate Devenv", "updated_at": "2026-06-28T12:00:00Z"}) + "\n",
                encoding="utf-8",
            )
            (sessions_dir / f"rollout-2026-06-28T12-00-00-{session_id}.jsonl").write_text(
                json.dumps(
                    {
                        "timestamp": "2026-06-28T11:59:00Z",
                        "type": "event_msg",
                        "payload": {"type": "agent_message", "message": "Worked on the Devenv context builder."},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            service = ContextBuilderService(
                str(workspace),
                provider_configs=(
                    ExternalSessionProviderConfig(provider="codex", root_path=str(codex_root), index_path="session_index.jsonl"),
                ),
            )
            result = service.prepare_prompt(
                PreparedPromptRequest(
                    task="hi",
                    provider="codex",
                    include_workspace_scan=False,
                    include_prior_context=True,
                )
            )

        self.assertEqual(result.session_ids, ())
        self.assertEqual(result.metadata["context_match_state"], "new_context")

    def test_runtime_memory_context_reports_reused_prior_context_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            workspace = Path(tempdir) / "workspace"
            workspace.mkdir()

            codex_root = Path(tempdir) / ".codex"
            sessions_dir = codex_root / "sessions" / "2026" / "06" / "28"
            sessions_dir.mkdir(parents=True)
            session_id = "session-memory"
            (codex_root / "session_index.jsonl").write_text(
                json.dumps({"id": session_id, "thread_name": "Infinite memory retrieval", "updated_at": "2026-06-28T12:00:00Z"}) + "\n",
                encoding="utf-8",
            )
            (sessions_dir / f"rollout-2026-06-28T12-00-00-{session_id}.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"timestamp": "2026-06-28T11:59:00Z", "type": "session_meta", "payload": {"id": session_id, "cwd": str(workspace)}}),
                        json.dumps({"timestamp": "2026-06-28T11:59:01Z", "type": "event_msg", "payload": {"type": "agent_message", "message": "Need better retrieval from prior Codex sessions and prompt generation."}}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            service = ContextBuilderService(
                str(workspace),
                provider_configs=(
                    ExternalSessionProviderConfig(provider="codex", root_path=str(codex_root), index_path="session_index.jsonl"),
                ),
            )
            context, session_ids, metadata = service.build_runtime_memory_context("Improve retrieval from prior Codex sessions.")

        self.assertIn("External Session Context", context)
        self.assertEqual(session_ids, (session_id,))
        self.assertEqual(metadata["context_match_state"], "reused_prior_context")

    def test_prepare_prompt_matches_hyphenated_project_from_workspace_path(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            workspace = Path(tempdir) / "workspace"
            workspace.mkdir()

            codex_root = Path(tempdir) / ".codex"
            sessions_dir = codex_root / "sessions" / "2026" / "06" / "28"
            sessions_dir.mkdir(parents=True)
            session_id = "session-get-drip"
            (codex_root / "session_index.jsonl").write_text(
                json.dumps({"id": session_id, "thread_name": "Unrelated title", "updated_at": "2026-06-28T12:00:00Z"}) + "\n",
                encoding="utf-8",
            )
            (sessions_dir / f"rollout-2026-06-28T12-00-00-{session_id}.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"timestamp": "2026-06-28T11:59:00Z", "type": "session_meta", "payload": {"id": session_id, "cwd": "/Users/samarthnaik/Desktop/LoopedIn/get-drip"}}),
                        json.dumps({"timestamp": "2026-06-28T11:59:01Z", "type": "event_msg", "payload": {"type": "agent_message", "message": "The project had Convex generation issues and onboarding routes."}}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            service = ContextBuilderService(
                str(workspace),
                provider_configs=(
                    ExternalSessionProviderConfig(provider="codex", root_path=str(codex_root), index_path="session_index.jsonl"),
                ),
            )
            result = service.prepare_prompt(
                PreparedPromptRequest(
                    task="Do you remember the get-drip project?",
                    provider="codex",
                    include_workspace_scan=False,
                    include_prior_context=True,
                )
            )

        self.assertEqual(result.session_ids, (session_id,))
        self.assertEqual(result.metadata["context_match_state"], "reused_prior_context")

    def test_list_sessions_includes_unindexed_session_files(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            workspace = Path(tempdir) / "workspace"
            workspace.mkdir()

            codex_root = Path(tempdir) / ".codex"
            sessions_dir = codex_root / "sessions" / "2026" / "06" / "28"
            sessions_dir.mkdir(parents=True)
            indexed_id = "session-indexed"
            unindexed_id = "session-unindexed"
            (codex_root / "session_index.jsonl").write_text(
                json.dumps({"id": indexed_id, "thread_name": "Indexed session", "updated_at": "2026-06-28T12:00:00Z"}) + "\n",
                encoding="utf-8",
            )
            (sessions_dir / f"rollout-2026-06-28T12-00-00-{indexed_id}.jsonl").write_text(
                json.dumps({"timestamp": "2026-06-28T11:59:00Z", "type": "session_meta", "payload": {"id": indexed_id, "cwd": str(workspace)}}) + "\n",
                encoding="utf-8",
            )
            (sessions_dir / f"rollout-2026-06-28T12-30-00-{unindexed_id}.jsonl").write_text(
                json.dumps({"timestamp": "2026-06-28T12:29:00Z", "type": "session_meta", "payload": {"id": unindexed_id, "cwd": "/tmp/other"}}) + "\n",
                encoding="utf-8",
            )

            service = ContextBuilderService(
                str(workspace),
                provider_configs=(
                    ExternalSessionProviderConfig(provider="codex", root_path=str(codex_root), index_path="session_index.jsonl"),
                ),
            )
            session_ids = {session.session_id for session in service.list_sessions("codex")}

        self.assertIn(indexed_id, session_ids)
        self.assertIn(unindexed_id, session_ids)

    def test_prepare_prompt_matches_project_from_session_title(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            workspace = Path(tempdir) / "workspace"
            workspace.mkdir()

            codex_root = Path(tempdir) / ".codex"
            sessions_dir = codex_root / "sessions" / "2026" / "04" / "24"
            sessions_dir.mkdir(parents=True)
            session_id = "session-codeguide"
            (codex_root / "session_index.jsonl").write_text(
                json.dumps({"id": session_id, "thread_name": "Integrate CodeGuide with GetGit", "updated_at": "2026-04-24T19:16:23Z"}) + "\n",
                encoding="utf-8",
            )
            (sessions_dir / f"rollout-2026-04-24T19-16-23-{session_id}.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"timestamp": "2026-04-24T19:16:23Z", "type": "session_meta", "payload": {"id": session_id, "cwd": "/Users/samarthnaik/Desktop/work/hirex-frontend"}}),
                        json.dumps({"timestamp": "2026-04-24T19:16:24Z", "type": "event_msg", "payload": {"type": "agent_message", "message": "CodeGuide was about integrating its flow with GetGit and ai_services without duplicating logic."}}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            service = ContextBuilderService(
                str(workspace),
                provider_configs=(
                    ExternalSessionProviderConfig(provider="codex", root_path=str(codex_root), index_path="session_index.jsonl"),
                ),
            )
            result = service.prepare_prompt(
                PreparedPromptRequest(
                    task="do you remember codeguide? what was it about?",
                    provider="codex",
                    include_workspace_scan=False,
                    include_prior_context=True,
                )
            )

        self.assertEqual(result.session_ids, (session_id,))
        self.assertEqual(result.metadata["context_match_state"], "reused_prior_context")
        self.assertIn("CodeGuide", result.prompt)

    def test_prepare_prompt_prefers_external_project_session_over_current_meta_session(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            workspace = Path(tempdir) / "devenv"
            workspace.mkdir()

            codex_root = Path(tempdir) / ".codex"
            sessions_dir = codex_root / "sessions" / "2026" / "07" / "03"
            sessions_dir.mkdir(parents=True)
            meta_id = "session-meta"
            project_id = "session-project"
            (codex_root / "session_index.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"id": meta_id, "thread_name": "Match Codex Mac UI", "updated_at": "2026-07-03T00:48:26Z"}),
                        json.dumps({"id": project_id, "thread_name": "Fix 7 bugs", "updated_at": "2026-07-02T11:52:11Z"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (sessions_dir / f"rollout-2026-07-03T00-48-26-{meta_id}.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"timestamp": "2026-07-03T00:48:26Z", "type": "session_meta", "payload": {"id": meta_id, "cwd": str(workspace)}}),
                        json.dumps({"timestamp": "2026-07-03T00:48:27Z", "type": "event_msg", "payload": {"type": "user_message", "message": "hey, do you remember about get-drip project?"}}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (sessions_dir / f"rollout-2026-07-02T13-12-12-{project_id}.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"timestamp": "2026-07-02T13:12:12Z", "type": "session_meta", "payload": {"id": project_id, "cwd": "/Users/samarthnaik/Desktop/LoopedIn/get-drip"}}),
                        json.dumps({"timestamp": "2026-07-02T13:12:13Z", "type": "event_msg", "payload": {"type": "agent_message", "message": "get-drip still had bugs around root URL redirects and Convex generated imports."}}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            service = ContextBuilderService(
                str(workspace),
                provider_configs=(
                    ExternalSessionProviderConfig(provider="codex", root_path=str(codex_root), index_path="session_index.jsonl"),
                ),
            )
            result = service.prepare_prompt(
                PreparedPromptRequest(
                    task="hey, do you remember about get-drip project?",
                    provider="codex",
                    include_workspace_scan=False,
                    include_prior_context=True,
                )
            )

        self.assertEqual(result.session_ids, (project_id,))
        self.assertNotIn(meta_id, result.session_ids)

    def test_runtime_memory_context_includes_issue_lines_for_project_recall_before_index_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            workspace = Path(tempdir) / "workspace"
            workspace.mkdir()

            codex_root = Path(tempdir) / ".codex"
            sessions_dir = codex_root / "sessions" / "2026" / "07" / "02"
            sessions_dir.mkdir(parents=True)
            session_id = "session-get-drip"
            (codex_root / "session_index.jsonl").write_text(
                json.dumps({"id": session_id, "thread_name": "Fix 7 bugs", "updated_at": "2026-07-02T11:52:11Z"}) + "\n",
                encoding="utf-8",
            )
            (sessions_dir / f"rollout-2026-07-02T13-12-12-{session_id}.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"timestamp": "2026-07-02T13:12:12Z", "type": "session_meta", "payload": {"id": session_id, "cwd": "/Users/samarthnaik/Desktop/LoopedIn/get-drip"}}),
                        json.dumps({"timestamp": "2026-07-02T13:12:13Z", "type": "event_msg", "payload": {"type": "user_message", "message": "Create Workspace should accept the https link and convert it internally."}}),
                        json.dumps({"timestamp": "2026-07-02T13:12:14Z", "type": "event_msg", "payload": {"type": "user_message", "message": "DRIP pipeline chat does not work and test/publish should be reachable after approvals."}}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            service = ContextBuilderService(
                str(workspace),
                provider_configs=(
                    ExternalSessionProviderConfig(provider="codex", root_path=str(codex_root), index_path="session_index.jsonl"),
                ),
            )
            context, session_ids, metadata = service.build_runtime_memory_context("hey, do you remember about get-drip project?")

        self.assertEqual(session_ids, (session_id,))
        self.assertFalse(metadata["index_ready"])
        self.assertIn("Create Workspace should accept the https link", context)
        self.assertIn("DRIP pipeline chat does not work", context)

    def test_runtime_memory_context_prefers_cleanup_session_over_unrelated_project_session(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            workspace = Path(tempdir) / "workspace"
            workspace.mkdir()

            codex_root = Path(tempdir) / ".codex"
            sessions_dir = codex_root / "sessions" / "2026" / "07" / "05"
            sessions_dir.mkdir(parents=True)
            cleanup_id = "session-cleanup"
            validators_id = "session-validators"
            (codex_root / "session_index.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"id": cleanup_id, "thread_name": "Clean up schema", "updated_at": "2026-07-05T12:00:00Z"}),
                        json.dumps({"id": validators_id, "thread_name": "Explain validators", "updated_at": "2026-07-05T12:05:00Z"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (sessions_dir / f"rollout-2026-07-05T12-00-00-{cleanup_id}.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"timestamp": "2026-07-05T12:00:00Z", "type": "session_meta", "payload": {"id": cleanup_id, "cwd": "/Users/samarthnaik/Desktop/LoopedIn/get-drip"}}),
                        json.dumps({"timestamp": "2026-07-05T12:00:01Z", "type": "event_msg", "payload": {"type": "agent_message", "message": "The schema cleanup was mainly about root URL redirects, Convex generated imports, and authentication bypass."}}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (sessions_dir / f"rollout-2026-07-05T12-05-00-{validators_id}.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"timestamp": "2026-07-05T12:05:00Z", "type": "session_meta", "payload": {"id": validators_id, "cwd": "/Users/samarthnaik/Desktop/LoopedIn/get-drip"}}),
                        json.dumps({"timestamp": "2026-07-05T12:05:01Z", "type": "event_msg", "payload": {"type": "agent_message", "message": "They are Convex argument validators defined in shared.ts for mock CRM sources."}}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            service = ContextBuilderService(
                str(workspace),
                provider_configs=(
                    ExternalSessionProviderConfig(provider="codex", root_path=str(codex_root), index_path="session_index.jsonl"),
                ),
            )
            context, session_ids, _metadata = service.build_runtime_memory_context("what do you know about clean up schrema og get-drip")

        self.assertIn(cleanup_id, session_ids)
        self.assertIn("root URL redirects", context)
        self.assertNotIn("validators", context.lower())

    def test_runtime_memory_context_prefers_bug_preview_session_for_last_time_get_drip_issue_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            workspace = Path(tempdir) / "workspace"
            workspace.mkdir()

            codex_root = Path(tempdir) / ".codex"
            sessions_dir = codex_root / "sessions" / "2026" / "07" / "03"
            sessions_dir.mkdir(parents=True)
            recent_commit_id = "session-commit"
            bug_preview_id = "session-bug-preview"
            (codex_root / "session_index.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"id": recent_commit_id, "thread_name": "Archive rollout", "updated_at": "2026-07-03T20:05:00Z"}),
                        json.dumps({"id": bug_preview_id, "thread_name": "Get-drip bug list", "updated_at": "2026-07-03T20:01:00Z"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (sessions_dir / f"rollout-2026-07-03T20-05-00-{recent_commit_id}.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"timestamp": "2026-07-03T20:05:00Z", "type": "session_meta", "payload": {"id": recent_commit_id, "cwd": "/Users/samarthnaik/Desktop/LoopedIn/get-drip"}}),
                        json.dumps({"timestamp": "2026-07-03T20:05:01Z", "type": "event_msg", "payload": {"type": "agent_message", "message": "[bug-fixes-sendinvite 8a0a7f8] fix(settings): use saved timezone and locale dropdowns"}}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (sessions_dir / f"rollout-2026-07-03T20-01-00-{bug_preview_id}.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"timestamp": "2026-07-03T20:01:00Z", "type": "session_meta", "payload": {"id": bug_preview_id, "cwd": str(workspace)}}),
                        json.dumps({"timestamp": "2026-07-03T20:01:01Z", "type": "event_msg", "payload": {"type": "agent_message", "message": "Based on memory from prior sessions, the get-drip bug list is root URL redirects, Convex generated imports, and DRIP pipeline chat flow not working."}}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            service = ContextBuilderService(
                str(workspace),
                provider_configs=(
                    ExternalSessionProviderConfig(provider="codex", root_path=str(codex_root), index_path="session_index.jsonl"),
                ),
            )
            context, session_ids, metadata = service.build_runtime_memory_context(
                "hey, do you remember what issue did we get while working with get-drip last time?",
                provider_name="codex",
            )

        self.assertIn(bug_preview_id, session_ids)
        self.assertIn("root URL redirects", context)
        self.assertNotIn("timezone and locale dropdowns", context.lower())
        self.assertEqual(metadata["context_match_state"], "reused_prior_context")

    def test_runtime_memory_context_can_match_compound_prompt_via_query_variants(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            workspace = Path(tempdir) / "workspace"
            workspace.mkdir()

            codex_root = Path(tempdir) / ".codex"
            sessions_dir = codex_root / "sessions" / "2026" / "07" / "06"
            sessions_dir.mkdir(parents=True)
            auth_id = "session-auth"
            react_id = "session-react"
            (codex_root / "session_index.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"id": auth_id, "thread_name": "Django auth fixes", "updated_at": "2026-07-06T12:00:00Z"}),
                        json.dumps({"id": react_id, "thread_name": "React preference notes", "updated_at": "2026-07-06T12:05:00Z"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (sessions_dir / f"rollout-2026-07-06T12-00-00-{auth_id}.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"timestamp": "2026-07-06T12:00:00Z", "type": "session_meta", "payload": {"id": auth_id, "cwd": "/tmp/rxgpt"}}),
                        json.dumps({"timestamp": "2026-07-06T12:00:01Z", "type": "event_msg", "payload": {"type": "agent_message", "message": "Django auth used custom middleware and session cookies."}}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (sessions_dir / f"rollout-2026-07-06T12-05-00-{react_id}.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"timestamp": "2026-07-06T12:05:00Z", "type": "session_meta", "payload": {"id": react_id, "cwd": "/tmp/rxgpt"}}),
                        json.dumps({"timestamp": "2026-07-06T12:05:01Z", "type": "event_msg", "payload": {"type": "agent_message", "message": "The user preferred functional React components over class components."}}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            service = ContextBuilderService(
                str(workspace),
                provider_configs=(
                    ExternalSessionProviderConfig(provider="codex", root_path=str(codex_root), index_path="session_index.jsonl"),
                ),
            )
            context, session_ids, metadata = service.build_runtime_memory_context(
                "What did we decide about django auth and React preferences?"
            )

        self.assertEqual(session_ids, (auth_id, react_id))
        self.assertEqual(metadata["context_match_state"], "reused_prior_context")
        self.assertIn("custom middleware and session cookies", context)
        self.assertIn("functional React components", context)

    def test_runtime_memory_context_skips_failing_provider_and_uses_successful_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            workspace = Path(tempdir) / "workspace"
            workspace.mkdir()
            service = ContextBuilderService(str(workspace), provider_configs=())
            service.providers = {"broken": mock.Mock(), "good": mock.Mock()}
            service.set_runtime_allowed_providers({"broken", "good"})
            good_provider = service.providers["good"]
            chunk = ExternalSessionChunkEmbedding(
                unified_session_id="good:session-good",
                provider="good",
                session_id="session-good",
                chunk_index=0,
                content_hash="hash",
                embedding=(1.0, 0.0),
                role="assistant",
                source="good",
                text="Based on memory from prior sessions, the get-drip bug list is root URL redirects and Convex generated imports.",
            )
            match = {
                "summary": ExternalSessionSummary(
                    provider="good",
                    session_id="session-good",
                    title="get-drip bug session",
                    workspace_path="/tmp/get-drip",
                    updated_at="",
                ),
                "chunks": [],
                "semantic_chunks": [(0.9, chunk)],
                "semantic_chunk": chunk,
                "semantic_score": 0.9,
                "identity_token_hits": 1,
                "token_hits": 2,
                "strong_match": True,
                "score": 99,
                "content_score": 99,
            }

            def fake_select(task: str, *, provider_name: str):
                if provider_name == "broken":
                    raise sqlite3.OperationalError("unable to open database file")
                return (
                    [match],
                    good_provider,
                    {
                        "context_match_state": "reused_prior_context",
                        "context_match_reason": "Matched prior get-drip bug session.",
                        "context_match_score": 99,
                    },
                )

            service._select_runtime_matches_for_provider = fake_select  # type: ignore[method-assign]
            context, session_ids, metadata = service.build_runtime_memory_context(
                "hey, do you remember what issue did we get while working with get-drip last time?"
            )

        self.assertEqual(session_ids, ("session-good",))
        self.assertIn("root URL redirects", context)
        self.assertEqual(metadata["context_match_state"], "reused_prior_context")

    def test_runtime_memory_context_fuses_sessions_across_providers(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            workspace = Path(tempdir) / "workspace"
            workspace.mkdir()
            service = ContextBuilderService(str(workspace), provider_configs=())
            service.providers = {"alpha": mock.Mock(), "beta": mock.Mock()}
            service.set_runtime_allowed_providers({"alpha", "beta"})

            def make_match(provider_name: str, text: str, score: float) -> dict:
                session_id = f"session-{provider_name}"
                chunk = ExternalSessionChunkEmbedding(
                    unified_session_id=f"{provider_name}:{session_id}",
                    provider=provider_name,
                    session_id=session_id,
                    chunk_index=0,
                    content_hash="hash",
                    embedding=(1.0, 0.0),
                    role="assistant",
                    source=provider_name,
                    text=text,
                )
                return {
                    "summary": ExternalSessionSummary(
                        provider=provider_name,
                        session_id=session_id,
                        title=f"{provider_name} session",
                        workspace_path="/tmp/proj",
                        updated_at="",
                    ),
                    "chunks": [],
                    "semantic_chunks": [(score, chunk)],
                    "semantic_chunk": chunk,
                    "semantic_score": score,
                    "identity_token_hits": 1,
                    "token_hits": 1,
                    "strong_match": True,
                    "score": int(score * 100),
                    "content_score": int(score * 100),
                }

            def fake_select(task: str, *, provider_name: str):
                if provider_name == "alpha":
                    return (
                        [make_match("alpha", "root URL redirects bug list", 0.99)],
                        service.providers["alpha"],
                        {
                            "context_match_state": "reused_prior_context",
                            "context_match_reason": "alpha match",
                            "context_match_score": 999,
                        },
                    )
                return (
                    [make_match("beta", "the actual fix", 0.5)],
                    service.providers["beta"],
                    {
                        "context_match_state": "reused_prior_context",
                        "context_match_reason": "beta match",
                        "context_match_score": 5,
                    },
                )

            service._select_runtime_matches_for_provider = fake_select  # type: ignore[method-assign]
            context, session_ids, metadata = service.build_runtime_memory_context(
                "hey, what was the bug we faced last time?"
            )

        self.assertIn("session-alpha", session_ids)
        self.assertIn("session-beta", session_ids)
        self.assertIn("the actual fix", context)
        self.assertEqual(metadata["context_match_providers"], ["alpha", "beta"])
    def test_combine_indexed_and_semantic_matches_keeps_index_rank_one(self) -> None:
        from core.runtime.context_builder import _combine_indexed_and_semantic_matches
        from core.runtime.context_builder import ExternalSessionSummary

        def make_match(session_id: str) -> dict:
            return {
                "summary": ExternalSessionSummary(
                    provider="codex", session_id=session_id, title=session_id, updated_at=""
                ),
                "chunks": [],
                "identity_token_hits": 1,
                "token_hits": 1,
            }

        indexed = [make_match("session-a"), make_match("session-b")]
        semantic = [make_match("session-b"), make_match("session-c")]
        combined = _combine_indexed_and_semantic_matches("what were the regex fixes?", indexed, semantic)
        ids = [match["summary"].session_id for match in combined]

        self.assertIn("session-a", ids)
        self.assertIn("session-b", ids)
        self.assertIn("session-c", ids)
        self.assertEqual(ids[0], "session-b")

    def test_semantic_session_scores_use_chunk_vectors(self) -> None:
        from core.memory.models import ExternalSessionChunkEmbedding, ExternalSessionEmbedding
        from core.runtime.models import ExternalSessionSummary

        class _QueryEmbedder:
            dimension = 2

            def embed(self, text: str) -> list[float]:
                return [1.0, 0.0] if "needle" in text.lower() else [0.0, 1.0]

        class _FakeMemory:
            def __init__(self, store: object, embedder: object) -> None:
                self.store = store
                self.embedder = embedder

        class _Provider:
            name = "codex"

        with tempfile.TemporaryDirectory() as tempdir:
            workspace = Path(tempdir) / "workspace"
            workspace.mkdir()
            store = SQLiteMemoryStore(str(workspace / "memory.db"))
            store.upsert_external_session_embedding(
                ExternalSessionEmbedding(
                    unified_session_id="codex:s1",
                    provider="codex",
                    session_id="s1",
                    content_hash="session-hash",
                    embedding=(0.0, 1.0),
                )
            )
            store.replace_external_session_chunk_embeddings(
                "codex:s1",
                [
                    ExternalSessionChunkEmbedding(
                        unified_session_id="codex:s1",
                        provider="codex",
                        session_id="s1",
                        chunk_index=0,
                        content_hash="chunk-hash",
                        embedding=(1.0, 0.0),
                    )
                ],
            )
            service = ContextBuilderService(
                str(workspace),
                memory=_FakeMemory(store=store, embedder=_QueryEmbedder()),
                provider_configs=(),
            )
            summary = ExternalSessionSummary(provider="codex", session_id="s1", title="S1", updated_at="")
            scores = service._semantic_session_scores(_Provider(), [summary], ("find the needle",))

        self.assertGreater(scores.get("s1", 0.0), 0.9)

    def test_codex_provider_dedupes_sessions_sharing_meta_id(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            workspace = Path(tempdir) / "workspace"
            workspace.mkdir()
            codex_root = Path(tempdir) / ".codex"
            sessions_dir = codex_root / "sessions" / "2026" / "08" / "20"
            sessions_dir.mkdir(parents=True)
            shared_id = "019f9f43-9c40-75b3-8b7f-64e83415b469"
            (codex_root / "session_index.jsonl").write_text(
                json.dumps({"id": shared_id, "thread_name": "Shared session", "updated_at": "2026-08-20T10:00:00Z"}) + "\n",
                encoding="utf-8",
            )
            for filename_id in ("019f9f52-13c7-7e42-8c9f-3086ab0c819c", "019f9f43-9c40-75b3-8b7f-64e83415b469"):
                (sessions_dir / f"rollout-2026-08-20T09-00-00-{filename_id}.jsonl").write_text(
                    json.dumps({"timestamp": "2026-08-20T09:00:00Z", "type": "session_meta", "payload": {"id": shared_id, "cwd": str(workspace)}})
                    + "\n"
                    + json.dumps({"timestamp": "2026-08-20T09:00:01Z", "type": "event_msg", "payload": {"type": "agent_message", "message": "hello"}})
                    + "\n",
                    encoding="utf-8",
                )
            service = ContextBuilderService(
                str(workspace),
                provider_configs=(
                    ExternalSessionProviderConfig(provider="codex", root_path=str(codex_root), index_path="session_index.jsonl"),
                ),
            )
            ids = [summary.session_id for summary in service._get_provider("codex").list_sessions()]

        self.assertEqual(ids.count(shared_id), 1)

    def test_context_lines_surface_semantic_chunk_from_lower_ranked_session(self) -> None:
        from core.memory.models import ExternalSessionChunkEmbedding
        from core.runtime.context_builder import (
            ExternalSessionSummary,
            _collect_per_session_context_lines,
        )

        def make_match(session_id: str, title: str, text: str, score: float) -> dict:
            chunk = ExternalSessionChunkEmbedding(
                unified_session_id=f"opencode:{session_id}",
                provider="opencode",
                session_id=session_id,
                chunk_index=0,
                content_hash="hash",
                embedding=(1.0, 0.0),
                role="assistant",
                source="opencode",
                text=text,
            )
            return {
                "summary": ExternalSessionSummary(
                    provider="opencode", session_id=session_id, title=title, updated_at=""
                ),
                "chunks": [],
                "semantic_chunk": chunk,
                "semantic_score": score,
            }

        matches = [
            make_match(
                "s1",
                "Noisy session",
                "the bug and the session and the fix and the dashboard and the bug list " * 3,
                0.9,
            ),
            make_match(
                "s2",
                "Real answer",
                "the recruiter dashboard went blank because the auth cache was not invalidated after saving details",
                0.8,
            ),
        ]

        lines = _collect_per_session_context_lines(
            "why did the recruiter dashboard go blank after saving details?",
            matches,
            {},
            4,
        )

        self.assertIn("auth cache was not invalidated", "\n".join(lines))

    def test_opencode_provider_excludes_derived_sessions_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            db_path = Path(tempdir) / "opencode.db"
            connection = sqlite3.connect(db_path)
            connection.executescript(
                """
                create table session (id text, title text, directory text, time_updated integer, time_archived integer, parent_id text);
                create table message (id text, session_id text, data text, time_created integer);
                create table part (id text, message_id text, session_id text, data text, time_created integer);
                """
            )
            connection.execute("insert into session values (?,?,?,?,?,?)", ("ses_human", "Human session", "/tmp/proj", 1000, None, None))
            connection.execute("insert into session values (?,?,?,?,?,?)", ("ses_derived", "Explore (@explore subagent)", "/tmp/proj", 2000, None, "ses_human"))
            connection.commit()
            connection.close()

            config = (ExternalSessionProviderConfig(provider="opencode", root_path=str(db_path)),)
            service = ContextBuilderService(tempdir, provider_configs=config)
            ids = [summary.session_id for summary in service.providers["opencode"].list_sessions()]
            service_with_derived = ContextBuilderService(
                tempdir, provider_configs=config, exclude_derived_sessions=False
            )
            ids_with_derived = [
                summary.session_id for summary in service_with_derived.providers["opencode"].list_sessions()
            ]

        self.assertEqual(ids, ["ses_human"])
        self.assertIn("ses_derived", ids_with_derived)

    def test_select_context_lines_keeps_lower_ranked_answer_when_over_budget(self) -> None:
        from core.runtime.context_builder import select_context_lines

        huge = "Assistant reported: " + ("noise " * 2000) + " alpha"
        answer = "Assistant reported: the real answer beta"
        per_session = [[(100, huge)], [(90, answer)]]

        selection = select_context_lines("tell me the real answer", per_session, max_lines=6, max_chars=500, eliminate=True)
        joined = "\n".join(selection.kept)

        self.assertIn("the real answer beta", joined)
        self.assertTrue(any(entry["reason"] for entry in selection.eliminated) or len(selection.kept) >= 2)

    def test_select_context_lines_unchanged_when_within_budget(self) -> None:
        from core.runtime.context_builder import select_context_lines

        per_session = [[(5, "Assistant reported: alpha")], [(4, "Assistant reported: beta")]]
        selection = select_context_lines("alpha beta", per_session, max_lines=6, max_chars=8000, eliminate=True)

        self.assertEqual(selection.kept, ("Assistant reported: alpha", "Assistant reported: beta"))
        self.assertEqual(selection.eliminated, ())

    def test_window_context_line_keeps_matched_region(self) -> None:
        from core.runtime.context_builder import _window_context_line

        line = ("x" * 5000) + " NEEDLE_TOKEN " + ("y" * 5000)
        windowed = _window_context_line(line, {"needle_token"}, max_chars=2000)

        self.assertIn("NEEDLE_TOKEN", windowed)
        self.assertLessEqual(len(windowed), 2010)


if __name__ == "__main__":
    unittest.main()
