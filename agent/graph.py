"""
what: the agent itself -- five nodes, the edges between them, and the two
      routers that decide where each run goes next.
why:  this file replaces the `while True:` loop in 01_practice/mock_agent.py.
      A hand-written loop hides its control flow inside if / break / continue;
      a graph states it as nodes and edges you can list, draw and test one at a
      time. The payoff arrives in Stage 5: a graph can stop between two nodes,
      save its state, and resume in a different process. A while loop cannot.
how:  build_graph() receives the four adapters as arguments and the nodes close
      over them. This file never imports a concrete adapter -- that is rule 6,
      and it is why run.py can swap FakeMetrics for DockerMetrics without this
      file noticing. The dispatcher, by contrast, IS imported directly, because
      a safety gate must not be swappable (see agent/tools/registry.py).

      Each node returns ONLY the keys it changed; LangGraph merges them into the
      state, appending to `history` because of its reducer (see state.py).

          START
            |
          detect ----- no incident -----> END
            |
          recall
            |
          reason <---------+
            |              |
         route()           |
          |    |           |
          |    +--> act ---+     use_tool AND steps < MAX_STEPS
          |
          +--> record --> END    conclude, OR the step bound was hit
"""

import json

from langgraph.graph import END, START, StateGraph

from agent.detectors import detect
from agent.models import AgentDecision
from agent.ports import DocsPort, LLMPort, MemoryPort, MetricsPort
from agent.state import AgentState
from agent.tools.registry import REGISTRY, dispatch

# The hard bound on tool calls per run. 01_practice had the same idea as
# `if step > MAX_STEPS: break`; here it is a condition on an edge, checked in
# Python on every pass. LangGraph has its own recursion_limit too, but it is a
# last-resort crash set far too high to protect a budget: on the installed
# release, a runaway loop with no bound made over 5,000 model calls before it
# fired. That limit is the framework's; this one is ours, tested, and ends the
# run cleanly with a record of what happened.
MAX_STEPS = 5


def initial_state(container: str) -> AgentState:
    """The state before any node has run. Kept in one place so run.py and the
    tests cannot drift apart on what a fresh run looks like."""
    return {
        "container": container,
        "metrics": {},
        "incident": None,
        "prior_incidents": 0,
        "docs": [],
        "decision": None,
        "history": [],
        "steps": 0,
    }


def build_prompt(state: AgentState) -> str:
    """Everything the model is allowed to know, as plain text.

    MockLLM ignores this completely. It is still built and tested now so that
    Stage 3 changes exactly one thing: a model starts reading it.
    """
    incident = state["incident"]
    runbook = "\n\n".join(state["docs"]) if state["docs"] else "(none retrieved)"
    return (
        "You are diagnosing a container incident. Reply with ONE JSON object and nothing else.\n"
        # Generated from the Pydantic model, not typed by hand: the schema the
        # model is shown can never drift from the schema it is validated against.
        f"JSON schema: {json.dumps(AgentDecision.model_json_schema())}\n\n"
        f"INCIDENT {incident.kind} on {incident.container}: {incident.summary}\n"
        f"Evidence: {incident.evidence}\n"
        f"Prior {incident.kind} incidents on this container in the last 24h: {state['prior_incidents']}\n\n"
        f"Runbook excerpts:\n{runbook}\n\n"
        # The prompt DESCRIBES the tools; registry.py ENFORCES them. Listing a
        # tool here grants nothing -- the same lesson as ALLOWED_TOOLS in
        # 01_practice/llm_reasoner.py.
        f"Tools you may request: {', '.join(sorted(REGISTRY))}\n"
        # Informational only. route() enforces the bound whatever the model
        # makes of this line.
        f"Tool calls used: {state['steps']} of {MAX_STEPS}\n\n"
        "Investigation so far:\n" + "\n".join(state["history"]) + "\n"
    )


def route_after_detect(state: AgentState) -> str:
    """No incident, no model call. The cheapest LLM call is the one never made."""
    return "recall" if state["incident"] is not None else "end"


def route(state: AgentState) -> str:
    """The step bound (rule 3). Both conditions, every pass, in Python.

    A model that asks for a tool on its sixth pass is not refused politely in a
    prompt -- it is simply never routed to `act` again.
    """
    decision = state["decision"]
    if decision.action == "use_tool" and state["steps"] < MAX_STEPS:
        return "act"
    return "record"


def build_graph(metrics: MetricsPort, llm: LLMPort, docs: DocsPort, memory: MemoryPort):
    """Wire the five nodes to the four ports and compile the graph."""

    def detect_node(state: AgentState) -> dict:
        stats = metrics.container_stats(state["container"])
        incident = detect(stats)
        if incident is None:
            line = f"DETECT   {state['container']}: healthy, nothing to investigate"
        else:
            line = f"DETECT   {incident.kind} | {incident.summary}"
        return {"metrics": stats, "incident": incident, "history": [line]}

    def recall_node(state: AgentState) -> dict:
        incident = state["incident"]
        # The port filters by kind only, exactly as mvp_plan.md defines it;
        # narrowing to this container happens here rather than widening the port.
        prior = [
            i
            for i in memory.recent_incidents(incident.kind, hours=24)
            if i.get("container") == incident.container
        ]
        chunks = docs.search(incident.summary, k=3)
        line = (
            f"RECALL   {len(prior)} prior {incident.kind} incident(s) on "
            f"{incident.container} in 24h | {len(chunks)} runbook chunk(s)"
        )
        return {"prior_incidents": len(prior), "docs": chunks, "history": [line]}

    def reason_node(state: AgentState) -> dict:
        raw = llm.decide(build_prompt(state))
        # THE TRUST BOUNDARY. `raw` is text from outside our control; after this
        # line it is a validated AgentDecision or an exception. In Stage 0 the
        # exception would crash the run, which is acceptable only because MockLLM
        # never sends bad JSON. Stage 3 catches ValidationError here and feeds the
        # message back as an observation, bounded by the same step counter.
        decision = AgentDecision.model_validate_json(raw)
        if decision.action == "use_tool":
            what = f"use_tool {decision.tool} {decision.args}"
        else:
            what = f"conclude: {decision.diagnosis}"
        line = (
            f"REASON   [{state['steps']}/{MAX_STEPS}] {what} "
            f"(confidence {decision.confidence}) | {decision.reasoning}"
        )
        return {"decision": decision, "history": [line]}

    def act_node(state: AgentState) -> dict:
        decision = state["decision"]
        observation = dispatch(decision.tool, decision.args)
        return {
            # Incremented HERE, where a tool actually runs, so `steps` counts
            # tool calls made rather than thoughts had.
            "steps": state["steps"] + 1,
            # The observation goes into history, and history goes into the next
            # prompt. That is the whole feedback loop, in one line.
            "history": [f"ACT      {decision.tool} -> {observation}"],
        }

    def record_node(state: AgentState) -> dict:
        incident = state["incident"]
        decision = state["decision"]
        concluded = decision.action == "conclude"
        # No timestamp here: the graph stays off the clock, and the Stage 4
        # SQLite adapter stamps the time when it stores the row.
        memory.record(
            {
                "kind": incident.kind,
                "container": incident.container,
                # None when the bound stopped the run: an honest "no conclusion"
                # beats recording a diagnosis the model never gave.
                "diagnosis": decision.diagnosis if concluded else None,
                "confidence": decision.confidence,
                "action_taken": None,  # nothing gated has run yet; Stage 5 fills this
            }
        )
        if concluded:
            outcome = "concluded"
        else:
            outcome = f"STOPPED by step bound after {MAX_STEPS} tool calls, no conclusion"
        return {"history": [f"RECORD   {outcome} | saved to episodic memory"]}

    builder = StateGraph(AgentState)
    builder.add_node("detect", detect_node)
    builder.add_node("recall", recall_node)
    builder.add_node("reason", reason_node)
    builder.add_node("act", act_node)
    builder.add_node("record", record_node)

    builder.add_edge(START, "detect")
    builder.add_conditional_edges("detect", route_after_detect, {"recall": "recall", "end": END})
    builder.add_edge("recall", "reason")
    builder.add_conditional_edges("reason", route, {"act": "act", "record": "record"})
    builder.add_edge("act", "reason")  # the loop: 01's `continue`, as an edge
    builder.add_edge("record", END)

    return builder.compile()
