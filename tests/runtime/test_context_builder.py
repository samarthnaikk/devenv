from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from core.runtime.context_builder import ContextBuilderService
from core.runtime.models import ExternalSessionProviderConfig, PreparedPromptRequest


class ContextBuilderServiceTest(unittest.TestCase):
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
        self.assertEqual(detail.summary.workspace_path, str(workspace))
        self.assertEqual(sessions[0].workspace_path, str(workspace))
        self.assertTrue(any(message.role == "user" for message in detail.messages))
        self.assertTrue(any("prompt preview ui" in message.content.lower() for message in detail.messages))

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


if __name__ == "__main__":
    unittest.main()
