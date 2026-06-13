from __future__ import annotations

from typing import Any

from .interface import MemoryEngineInterface
from .models import ConsolidationResult, RetrievalResult, RetrievalTrace


class MemoryEngine(MemoryEngineInterface):
    def __init__(self, db_path: str = "memory.db", vector_dir: str = "vectors/"):
        self.db_path = db_path
        self.vector_dir = vector_dir

    def add_episodic_log(
        self,
        user_prompt: str,
        agent_response: str,
        node_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        raise NotImplementedError

    def retrieve_context(self, current_prompt: str, top_k: int = 5) -> RetrievalResult:
        raise NotImplementedError

    def update_associative_tree(self, node_data: dict[str, Any]) -> str:
        raise NotImplementedError

    def run_consolidation(self, since: float | None = None) -> ConsolidationResult:
        raise NotImplementedError

    def forget_node(self, node_id: str, strategy: str = "prune") -> bool:
        raise NotImplementedError

    def get_context_trace(self) -> RetrievalTrace:
        raise NotImplementedError

    def record_working_memory(self, messages: list[dict[str, Any]], active_state: dict[str, Any]) -> None:
        raise NotImplementedError

