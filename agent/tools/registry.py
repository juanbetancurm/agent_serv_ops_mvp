"""
what: the single function through which every tool call in the agent passes.
why:  one choke point means one place to enforce policy. The graph never calls a
      tool directly -- it calls dispatch(), and dispatch() decides what runs.

      !!! UNGUARDED ON PURPOSE IN STAGE 0 !!!
      Right now this dispatcher does whatever the model asks. It would run
      restart_container(name="postgres") without hesitation. That is harmless
      today ONLY because every tool is fake. Stage 1 begins by writing the test
      `restart_container(name="postgres")` must raise ToolNotAllowed, watching
      it FAIL against this exact file, and then adding the allowlist here.
      Nothing in graph.py will change when that happens -- which is the reason
      the lookup lives in this file and not inside the graph.

how:  REGISTRY maps a tool name to a function, and dispatch() calls it with the
      model's args. The gate is deliberately NOT injected into the graph as a
      swappable dependency: metrics sources are adapters and should be
      swappable, but a safety boundary that run.py could replace with a more
      permissive one is not a safety boundary.
"""

from collections.abc import Callable

# Stage 1 changes this ONE import to `from agent.tools import container`.
from agent.tools import container_fake as container

REGISTRY: dict[str, Callable[..., str]] = {
    "get_container_stats": container.get_container_stats,
    "get_container_logs": container.get_container_logs,
    "restart_container": container.restart_container,
}


def dispatch(tool: str, args: dict) -> str:
    """Run the named tool with the model's arguments and return its observation.

    An unknown tool currently raises KeyError and would crash the run. MockLLM
    never asks for one, so Stage 0 never sees it. Stage 1 turns that into
    ToolNotAllowed, and the graph turns ToolNotAllowed into an observation the
    model can read -- a refusal, not a crash.
    """
    return REGISTRY[tool](**args)
