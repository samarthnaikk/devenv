"""Durable storage for execution plans (blueprints).

Plans are written as one JSON document per plan under
``<workspace>/.devenv/plans/``. Both representations are kept so the TUI
(markdown checklist) and the web planner (task/edge JSON) can round-trip the
same artifact: ``raw_plan_markdown`` plus ``tasks`` and ``edges``.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _slugify(value: str, *, limit: int = 48) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", str(value or "").strip().lower()).strip("-")
    return (cleaned[:limit] or "plan").strip("-")


@dataclass(frozen=True)
class PlanEntry:
    plan_id: str
    objective: str
    created_at: float
    raw_plan_markdown: str = ""
    tasks: tuple[dict[str, Any], ...] = ()
    edges: tuple[dict[str, Any], ...] = ()
    mode: str = "plan_only"
    source: str = "tui"
    path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "objective": self.objective,
            "created_at": self.created_at,
            "raw_plan_markdown": self.raw_plan_markdown,
            "tasks": list(self.tasks),
            "edges": list(self.edges),
            "mode": self.mode,
            "source": self.source,
            "path": self.path,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PlanEntry":
        return cls(
            plan_id=str(payload.get("plan_id") or ""),
            objective=str(payload.get("objective") or ""),
            created_at=float(payload.get("created_at") or 0.0),
            raw_plan_markdown=str(payload.get("raw_plan_markdown") or ""),
            tasks=tuple(payload.get("tasks") or ()),
            edges=tuple(payload.get("edges") or ()),
            mode=str(payload.get("mode") or "plan_only"),
            source=str(payload.get("source") or "tui"),
            path=str(payload.get("path") or ""),
        )


class PlanStore:
    def __init__(self, workspace_path: str) -> None:
        self.workspace_path = str(Path(workspace_path).expanduser().resolve())

    @property
    def directory(self) -> Path:
        return Path(self.workspace_path) / ".devenv" / "plans"

    def _path_for(self, plan_id: str) -> Path:
        return self.directory / f"{plan_id}.json"

    def save(
        self,
        *,
        objective: str,
        raw_plan_markdown: str = "",
        tasks: list[dict[str, Any]] | None = None,
        edges: list[dict[str, Any]] | None = None,
        mode: str = "plan_only",
        source: str = "tui",
        created_at: float | None = None,
    ) -> PlanEntry:
        """Persist a plan and return its entry.

        The plan id is ``<epoch>-<slug>`` so files sort chronologically and stay
        human-readable. Saving the same objective again creates a new revision.
        """

        created = float(created_at if created_at is not None else time.time())
        plan_id = f"{int(created)}-{_slugify(objective)}"
        entry = PlanEntry(
            plan_id=plan_id,
            objective=str(objective or ""),
            created_at=created,
            raw_plan_markdown=str(raw_plan_markdown or ""),
            tasks=tuple(tasks or ()),
            edges=tuple(edges or ()),
            mode=str(mode or "plan_only"),
            source=str(source or "tui"),
            path=str(self._path_for(plan_id)),
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        # Include a short random suffix only on collision so revisions never
        # silently overwrite a previous plan.
        if self._path_for(plan_id).exists():
            plan_id = f"{plan_id}-{uuid.uuid4().hex[:6]}"
            entry = PlanEntry(
                plan_id=plan_id,
                objective=entry.objective,
                created_at=entry.created_at,
                raw_plan_markdown=entry.raw_plan_markdown,
                tasks=entry.tasks,
                edges=entry.edges,
                mode=entry.mode,
                source=entry.source,
                path=str(self._path_for(plan_id)),
            )
        try:
            self._path_for(plan_id).write_text(
                json.dumps(entry.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:  # pragma: no cover - defensive
            logger.warning("Failed to save plan: plan_id=%s error=%s", plan_id, exc)
        return entry

    def list(self, *, limit: int | None = None) -> list[PlanEntry]:
        if not self.directory.exists():
            return []
        entries: list[PlanEntry] = []
        for path in sorted(self.directory.glob("*.json"), reverse=True):
            entry = self._read(path)
            if entry is not None:
                entries.append(entry)
        return entries[:limit] if limit else entries

    def load(self, plan_id: str) -> PlanEntry | None:
        return self._read(self._path_for(plan_id))

    def delete(self, plan_id: str) -> bool:
        path = self._path_for(plan_id)
        try:
            if path.exists():
                path.unlink()
                return True
        except OSError as exc:  # pragma: no cover - defensive
            logger.warning("Failed to delete plan: plan_id=%s error=%s", plan_id, exc)
        return False

    def export_to(self, plan_id: str, destination: Path) -> Path | None:
        entry = self.load(plan_id)
        if entry is None:
            return None
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.suffix.lower() in {".md", ".markdown"}:
            destination.write_text(entry.raw_plan_markdown or "", encoding="utf-8")
        else:
            destination.write_text(json.dumps(entry.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return destination

    def _read(self, path: Path) -> PlanEntry | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        payload.setdefault("path", str(path))
        return PlanEntry.from_dict(payload)


__all__ = ["PlanEntry", "PlanStore"]
