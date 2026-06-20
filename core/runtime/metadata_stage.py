from __future__ import annotations

from dataclasses import dataclass

from .models import CheckpointTask, StageTrace, ToolExecutionStep


@dataclass(frozen=True)
class CheckpointMetadataRecord:
    original_objective: str
    checkpoint_id: int
    checkpoint_objective: str
    target_path_hint: str | None
    output_destination: str
    files_touched: tuple[str, ...]
    verification_target: str
    completion_summary: str

    def to_dict(self) -> dict[str, object]:
        return {
            "original_objective": self.original_objective,
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_objective": self.checkpoint_objective,
            "target_path_hint": self.target_path_hint,
            "output_destination": self.output_destination,
            "files_touched": list(self.files_touched),
            "verification_target": self.verification_target,
            "completion_summary": self.completion_summary,
        }


def build_checkpoint_metadata(
    *,
    original_objective: str,
    checkpoint: CheckpointTask,
    checkpoint_steps: list[ToolExecutionStep],
    completion_summary: str,
) -> tuple[CheckpointMetadataRecord, StageTrace]:
    files_touched = tuple(_extract_touched_files(checkpoint_steps))
    output_destination = checkpoint.output_destination or _infer_output_destination(checkpoint_steps, checkpoint)
    record = CheckpointMetadataRecord(
        original_objective=original_objective,
        checkpoint_id=checkpoint.task_id,
        checkpoint_objective=checkpoint.objective or checkpoint.description,
        target_path_hint=checkpoint.target_path_hint,
        output_destination=output_destination,
        files_touched=files_touched,
        verification_target=checkpoint.verification_mode,
        completion_summary=completion_summary,
    )
    trace = StageTrace(
        stage="metadata",
        checkpoint_id=checkpoint.task_id,
        success=True,
        summary="Recorded checkpoint metadata",
        logs=[f"Output destination: {output_destination}", f"Touched files: {len(files_touched)}"],
        payload=record.to_dict(),
    )
    return record, trace


def _extract_touched_files(steps: list[ToolExecutionStep]) -> list[str]:
    paths: list[str] = []
    for step in steps:
        for key in ("path", "file_path", "target_path"):
            value = step.arguments.get(key)
            if isinstance(value, str) and value.strip():
                paths.append(value)
    return list(dict.fromkeys(paths))


def _infer_output_destination(steps: list[ToolExecutionStep], checkpoint: CheckpointTask) -> str:
    tool_names = {step.tool_name for step in steps}
    if "write_file" in tool_names:
        return "file_write"
    if "edit_file" in tool_names:
        return "file_edit"
    if "remove_file" in tool_names:
        return "delete"
    if "run_diagnostics" in tool_names:
        return "diagnostic"
    return checkpoint.expected_artifact
