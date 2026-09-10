"""
what: tests for the two seams written in Step 2 -- Pydantic validation of a
      model response, and structural typing of a port.
why:  these two mechanisms are invisible until something violates them. The
      tests below are the violations, written down once, so that "Pydantic
      catches it" and "an adapter needs no import to satisfy a port" stop being
      claims and become facts you have watched pass.
how:  every test constructs the bad case on purpose and asserts the failure.
      No network, no API key, no cost -- and that is permanent: pytest must run
      offline and free for the entire life of this project.
"""

import json

import pytest
from pydantic import ValidationError

from agent.models import AgentDecision, Incident
from agent.ports import MetricsPort

# A well-formed response, as raw text, because raw text is what LLMPort returns.
GOOD_RESPONSE = json.dumps(
    {
        "action": "use_tool",
        "tool": "get_container_stats",
        "args": {"name": "lab-victim"},
        "confidence": 0.8,
        "reasoning": "Need current memory usage before diagnosing.",
    }
)


def test_well_formed_response_parses():
    """The happy path: text in, validated object out."""
    decision = AgentDecision.model_validate_json(GOOD_RESPONSE)
    assert decision.action == "use_tool"
    assert decision.tool == "get_container_stats"
    assert decision.args["name"] == "lab-victim"


def test_confidence_above_one_is_rejected():
    """The classic hallucinated number. ge/le on the field catches it."""
    bad = json.dumps(
        {"action": "conclude", "diagnosis": "OOM", "confidence": 1.7, "reasoning": "sure"}
    )
    with pytest.raises(ValidationError) as err:
        AgentDecision.model_validate_json(bad)
    # Printed so you can read the actual message pytest -s shows; in Stage 3
    # this exact text is fed back to the model as an observation to retry from.
    print("\nconfidence 1.7 rejected:\n", err.value)


def test_invented_field_is_rejected():
    """extra='forbid' turning a silent drop into a loud failure."""
    bad = json.dumps(
        {
            "action": "conclude",
            "diagnosis": "OOM loop",
            "confidence": 0.9,
            "reasoning": "restart count climbing",
            "priority": "high",  # no such field
        }
    )
    with pytest.raises(ValidationError):
        AgentDecision.model_validate_json(bad)


def test_use_tool_without_a_tool_name_is_rejected():
    """A cross-field rule: every field is individually fine, the whole is not."""
    bad = json.dumps({"action": "use_tool", "confidence": 0.5, "reasoning": "let me look"})
    with pytest.raises(ValidationError):
        AgentDecision.model_validate_json(bad)


def test_incident_is_immutable():
    """A finding is not a workspace: no node may edit the evidence after the fact."""
    incident = Incident(kind="RB-002", container="lab-victim", summary="OOM loop")
    with pytest.raises(Exception):  # dataclasses raise FrozenInstanceError
        incident.container = "postgres"  # type: ignore[misc]


class ThingWithStats:
    """Deliberately imports nothing from agent.ports and inherits from nothing."""

    def container_stats(self, name: str) -> dict:
        return {"name": name}


class ThingWithoutStats:
    def something_else(self) -> None:  # pragma: no cover - never called
        pass


def test_a_port_is_satisfied_by_shape_not_by_inheritance():
    """The ports-and-adapters lesson, as one assertion.

    ThingWithStats never heard of MetricsPort. It satisfies it anyway, because
    it has the method. This is why swapping the metrics source in Stage 2 is a
    one-line change in run.py and nothing else.
    """
    assert isinstance(ThingWithStats(), MetricsPort)


def test_missing_method_does_not_satisfy_the_port():
    assert not isinstance(ThingWithoutStats(), MetricsPort)
