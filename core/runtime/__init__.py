__all__ = [
    "DevenvKernel",
    "PathSandbox",
    "RunConfig",
    "RuntimeTurnResult",
    "ToolExecutionStep",
]


def __getattr__(name: str):
    if name == "DevenvKernel":
        from .kernel import DevenvKernel

        return DevenvKernel
    if name == "PathSandbox":
        from .sandbox import PathSandbox

        return PathSandbox
    if name in {"RunConfig", "RuntimeTurnResult", "ToolExecutionStep"}:
        from .models import RunConfig, RuntimeTurnResult, ToolExecutionStep

        return {
            "RunConfig": RunConfig,
            "RuntimeTurnResult": RuntimeTurnResult,
            "ToolExecutionStep": ToolExecutionStep,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
