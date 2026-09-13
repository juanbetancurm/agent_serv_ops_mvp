"""
what: tests for the tool dispatcher, in its Stage 0 form.
why:  these only prove the plumbing -- that a tool name reaches a function and an
      observation comes back as text. They say nothing about SAFETY, because the
      Stage 0 dispatcher has none. That silence is deliberate and temporary.
      Stage 1 adds the two tests that matter to this file:
          test_rejects_unknown_tool
          test_rejects_write_on_non_lab_container
      and writes them BEFORE the allowlist exists, so you watch them fail first.
how:  call dispatch() directly with the args shape the model produces.
"""

from agent.tools.registry import REGISTRY, dispatch


def test_the_three_tools_are_registered():
    # Given:    the registry
    # Expected: exactly the three tools in scope
    # Why:      a fourth tool is scope creep -- and, from Stage 1, one more thing to gate
    assert set(REGISTRY) == {"get_container_stats", "get_container_logs", "restart_container"}


def test_dispatch_reaches_a_read_tool_and_returns_text():
    # Given:    get_container_stats for lab-victim
    # Expected: a string containing oom_killed=True
    # Why:      observations are text, because they get pasted into the next prompt
    observation = dispatch("get_container_stats", {"name": "lab-victim"})
    assert isinstance(observation, str)
    assert "oom_killed=True" in observation


def test_dispatch_passes_optional_args_through():
    # Given:    get_container_logs with lines=10
    # Expected: the observation mentions 10 lines
    # Why:      the model's args must reach the tool unchanged
    observation = dispatch("get_container_logs", {"name": "lab-victim", "lines": 10})
    assert "last 10 log lines" in observation


def test_the_fake_write_tool_changes_nothing():
    # Given:    restart_container on lab-victim
    # Expected: "nothing actually happened"
    # Why:      Stage 0 writes are fake; the gate that makes real ones safe arrives in Stage 1
    assert "nothing actually happened" in dispatch("restart_container", {"name": "lab-victim"})
