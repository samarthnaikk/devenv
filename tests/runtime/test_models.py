from __future__ import annotations

import unittest

from core.runtime.models import (
    CheckpointTask,
    ExecutionMode,
    MemorySummary,
    RepairState,
    RuntimeTurnResult,
    ToolPolicyEvent,
    TurnOutcome,
)


class RuntimeModelsTest(unittest.TestCase):
    def test_runtime_turn_result_serializes_additive_agent_contract_fields(self) -> None:
        result = RuntimeTurnResult(
            final_response="done",
            execution_mode=ExecutionMode.REPAIR.value,
            turn_outcome=TurnOutcome.VERIFICATION_FAILURE.value,
            memory_summary=MemorySummary(
                used_working_memory=True,
                used_associative_memory=True,
                used_external_context=False,
                privacy_mode="no_memory",
                context_chars=128,
            ),
            tool_policy_events=[
                ToolPolicyEvent(
                    tool_name="read_file",
                    category="inspect",
                    decision="allow",
                    reason="Read-only tool allowed during planning.",
                    mode=ExecutionMode.PLAN_ONLY.value,
                    retry_safe=True,
                )
            ],
            repair_state=RepairState(
                active_checkpoint_id=3,
                repair_attempt_count=1,
                max_repair_attempts=2,
                last_failure_reason="Diagnostics failed.",
            ),
        )

        payload = result.to_dict()

        self.assertEqual(payload["execution_mode"], "repair")
        self.assertEqual(payload["turn_outcome"], "verification_failure")
        self.assertTrue(payload["memory_summary"]["used_working_memory"])
        self.assertEqual(payload["tool_policy_events"][0]["tool_name"], "read_file")
        self.assertEqual(payload["repair_state"]["active_checkpoint_id"], 3)

    def test_checkpoint_task_serializes_execution_contract_fields(self) -> None:
        task = CheckpointTask(
            task_id=1,
            description="Update backend route",
            allowed_tool_names=("read_file", "edit_file"),
            expects_mutation=True,
            requires_verification=True,
            repair_attempt_count=1,
            max_repair_attempts=2,
        )

        payload = task.to_dict()

        self.assertEqual(payload["allowed_tool_names"], ["read_file", "edit_file"])
        self.assertTrue(payload["expects_mutation"])
        self.assertTrue(payload["requires_verification"])
        self.assertEqual(payload["repair_attempt_count"], 1)
        self.assertEqual(payload["max_repair_attempts"], 2)


if __name__ == "__main__":
    unittest.main()
