from __future__ import annotations

import argparse
import inspect
import json
import logging
from pathlib import Path
from typing import Annotated, Any, Literal, get_args

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from core.logging_utils import configure_logging
from core.memory import MemoryEngine
from core.memory.embeddings import HashingEmbedder
from core.tools.base import BaseTool

from .state import resolve_memory_paths
from .tooling import build_runtime_tools

logger = logging.getLogger(__name__)


def create_mcp_server(*, workspace_path: str, db_path: str = "memory.db", vector_dir: str = "vectors") -> FastMCP:
    resolved_db_path, resolved_vector_dir = resolve_memory_paths(db_path, vector_dir)
    memory = MemoryEngine(
        db_path=resolved_db_path,
        vector_dir=resolved_vector_dir,
        embedder=HashingEmbedder(dimension=384),
    )
    mcp = FastMCP("Devenv Local Tool Deck")

    for tool in build_runtime_tools(memory):
        wrapper = _build_tool_wrapper(tool)
        mcp.add_tool(
            wrapper,
            name=tool.name,
            description=tool.description,
            structured_output=False,
        )

    return mcp


def _build_tool_wrapper(tool: BaseTool):
    schema = tool.input_schema()
    properties = schema.get("properties", {})
    required = set(schema.get("required", ()))
    parameters: list[inspect.Parameter] = []
    annotations: dict[str, Any] = {"return": str}

    for name, property_schema in properties.items():
        annotation = _annotation_for_property(name, property_schema, required=name in required)
        default = inspect.Parameter.empty if name in required else property_schema.get("default", None)
        parameters.append(
            inspect.Parameter(
                name=name,
                kind=inspect.Parameter.KEYWORD_ONLY,
                default=default,
                annotation=annotation,
            )
        )
        annotations[name] = annotation

    signature = inspect.Signature(parameters=parameters, return_annotation=str)

    def wrapper(**kwargs) -> str:
        result = tool.execute(**kwargs)
        payload = {
            "success": result.success,
            "output": result.output,
            "data": result.data,
        }
        return json.dumps(payload, sort_keys=True)

    wrapper.__name__ = f"{tool.name}_adapter"
    wrapper.__doc__ = tool.description
    wrapper.__signature__ = signature
    wrapper.__annotations__ = annotations
    return wrapper


def _annotation_for_property(name: str, property_schema: dict[str, Any], *, required: bool) -> Any:
    schema_type = property_schema.get("type")
    enum = property_schema.get("enum")
    description = str(property_schema.get("description", "")).strip()

    if enum:
        literal_values = tuple(str(value) for value in enum)
        annotation = Literal.__getitem__(literal_values)
    elif schema_type == "integer":
        annotation = int
    else:
        annotation = str

    if not required:
        annotation = annotation | None

    if description:
        return Annotated[annotation, Field(description=description)]
    return annotation


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the local Devenv MCP tool server.")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--db-path", default="memory.db")
    parser.add_argument("--vector-dir", default="vectors")
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args()

    configure_logging(args.log_level)
    server = create_mcp_server(
        workspace_path=str(Path(args.workspace).expanduser().resolve()),
        db_path=args.db_path,
        vector_dir=args.vector_dir,
    )
    logger.info("Starting Devenv MCP server for workspace=%s", args.workspace)
    server.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
