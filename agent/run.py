"""
what: the command-line entrypoint -- `python -m agent.run`.
why:  this is the ONLY file that names concrete adapters. Every other file is
      written against the ports, so something has to pick the real classes, and
      keeping that choice in one place is what makes each later stage a
      one-line change here and nowhere else (rule 6).

      In 01_practice, main() in mock_agent.py both chose the parts AND ran the
      loop -- which is why switching to a real model meant copying the whole file
      into llm_agent.py. Here the two jobs are split: run.py chooses, graph.py runs.

      This file deliberately has NO test. From Stage 1 it talks to real Docker,
      and from Stage 3 to a paid model; a test that imported it would drag both
      into pytest, which must stay offline and free (rule 5). The logic it wires
      together is already tested, with fakes, in tests/test_graph_mock.py.
how:  build four adapters, hand them to build_graph(), invoke once on
      lab-victim, then print the history as a trace and a short verdict.
"""

import time

from dotenv import load_dotenv

from agent.adapters.docs_tfidf import TfidfDocs
from agent.adapters.llm_openai import OpenAIAdapter
from agent.adapters.memory_sqlite import SqliteMemory
from agent.adapters.metrics_prometheus import PrometheusMetrics
from agent.graph import MAX_STEPS, build_graph, initial_state

CONTAINER = "lab-victim"


def main() -> None:
    # Load .env before any adapter is built, because an adapter reads its
    # configuration from the environment at construction. The composition root
    # is the right place for this: an adapter that quietly loaded a file would
    # behave differently depending on where it was imported from.
    load_dotenv()

    # ---- the four adapter choices: the only lines later stages edit ---------
    # THE STAGE 2 SWAP, and it is this one line again. FakeMetrics ->
    # DockerMetrics -> PrometheusMetrics: three sources, three technologies, a
    # scrape loop and a time-series database now in the path -- and graph.py,
    # detectors.py and every test have still never been touched.
    metrics = PrometheusMetrics()  # was DockerMetrics(), before that FakeMetrics()
    # THE STAGE 3 SWAP: the line that starts costing money. MockLLM is not gone
    # -- it is still the default in every test (rule 5), which is what keeps
    # `pytest` free while this file spends.
    llm = OpenAIAdapter()  # was MockLLM()
    # THE STAGE 4 SWAP (second half). The model can now read the runbook that
    # forbids the remediation it kept proposing. Costs ~800 prompt tokens a call.
    docs = TfidfDocs()  # was NullDocs(); reads runbooks/RB-002-...md
    # THE STAGE 4 SWAP (first half). Every earlier trace said "0 prior
    # incidents" because NullMemory died with the process. From here the count
    # is real, and the prompt carries it.
    memory = SqliteMemory()  # was NullMemory(); writes ./lab_agent.db

    graph = build_graph(metrics, llm, docs, memory)

    # Timed around invoke() only: Python and LangGraph start-up are not the
    # agent, and "under a second" in mvp_plan.md is about the agent.
    started = time.perf_counter()
    final = graph.invoke(initial_state(CONTAINER))
    elapsed_ms = (time.perf_counter() - started) * 1000

    adapters = ", ".join(type(a).__name__ for a in (metrics, llm, docs, memory))
    print(f"lab-agent | {CONTAINER} | adapters: {adapters}")
    print("-" * 80)
    for line in final["history"]:
        print(line)
    print("-" * 80)

    # Counted from the trace, not from MockLLM.calls: that attribute exists only
    # on the mock, and this file must keep working when the adapter is real.
    model_calls = sum(1 for line in final["history"] if line.startswith("REASON"))
    decision = final["decision"]
    if final["incident"] is None:
        print("VERDICT  healthy -- no incident, so the model was never consulted")
    elif decision.action == "conclude":
        print(f"VERDICT  {decision.diagnosis} (confidence {decision.confidence})")
    else:
        print(f"VERDICT  no conclusion -- the step bound ended the run after {MAX_STEPS} tool calls")
    # getattr, because only the real adapter counts tokens; MockLLM has none and
    # this file must keep working with either.
    tokens = getattr(llm, "tokens", 0)
    print(
        f"         {final['steps']} tool call(s) | {model_calls} model call(s) | "
        f"{tokens} tokens | {elapsed_ms:.0f} ms"
    )


if __name__ == "__main__":
    main()
