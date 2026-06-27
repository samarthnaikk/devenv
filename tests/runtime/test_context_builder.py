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
        self.assertTrue(any(message.role == "user" for message in detail.messages))
        self.assertTrue(any("prompt preview ui" in message.content.lower() for message in detail.messages))

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


if __name__ == "__main__":
    unittest.main()
