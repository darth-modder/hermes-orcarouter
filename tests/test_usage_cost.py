"""get_usage_cost: CostResult from usage.raw_usage["cost_usd"].

CostResult is a frozen dataclass (amount_usd: Decimal | None, status:
CostStatus, source: CostSource, label: str, ...); core's
estimate_usage_cost() calls profile.get_usage_cost(model_name, usage) and
returns a non-None result as-is. Status is "estimated", not "actual" (see
orcarouter/__init__.py's get_usage_cost docstring); "estimated" and
source="provider_cost_api" are independent Literal types
(agent/usage_pricing.py) so this pairing is valid even though the amount
comes straight from the provider's own response.
"""

from __future__ import annotations

import json
import math
from decimal import Decimal
from pathlib import Path

import pytest

pytest.importorskip("providers", reason="hermes-agent not importable (set HERMES_AGENT_DIR to a checkout)")

_FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((_FIXTURES / name).read_text(encoding="utf-8"))


def test_real_fixture_cost_usd_becomes_estimated_cost_result(orcarouter_profile):
    from agent.usage_pricing import normalize_usage

    fx = _load("real_200_usage_with_cost.json")
    usage_dict = fx["body"]["usage"]
    usage = normalize_usage(usage_dict, provider="orcarouter", api_mode="chat_completions")
    assert usage.raw_usage is not None
    assert usage.raw_usage.get("cost_usd") == pytest.approx(1e-05)

    result = orcarouter_profile.get_usage_cost("openai/gpt-5.4-nano", usage)
    assert result is not None
    assert result.status == "estimated"
    assert result.source == "provider_cost_api"
    assert result.amount_usd == Decimal("1e-05")
    assert isinstance(result.label, str) and result.label


def test_real_fixture_through_sdk_model_dump_path(orcarouter_profile):
    """The alarm for a fail-open import breakage of agent.usage_pricing.CostResult
    (or of the openai SDK dependency itself): build a real openai
    CompletionUsage via model_validate on the fixture's usage dict, run it
    through core's own normalize_usage exactly as core does with a live SDK
    response object (not a plain dict), and confirm a non-None CostResult
    comes out the other end. CompletionUsage keeps extra="allow" fields (like
    cost_usd, which is not one of its declared fields) through model_dump(),
    which is how normalize_usage's raw_usage ends up with it when the SDK
    handed core a real usage object rather than a parsed dict.
    """
    openai = pytest.importorskip("openai", reason="openai not importable (see requirements-dev.txt)")
    from openai.types.completion_usage import CompletionUsage

    from agent.usage_pricing import normalize_usage

    fx = _load("real_200_usage_with_cost.json")
    usage_dict = fx["body"]["usage"]
    sdk_usage = CompletionUsage.model_validate(usage_dict)
    assert "cost_usd" not in CompletionUsage.model_fields  # confirms this rides in via extra="allow"

    usage = normalize_usage(sdk_usage, provider="orcarouter", api_mode="chat_completions")
    assert usage.raw_usage is not None
    assert usage.raw_usage.get("cost_usd") == pytest.approx(1e-05)

    result = orcarouter_profile.get_usage_cost("openai/gpt-5.4-nano", usage)
    assert result is not None
    assert result.status == "estimated"
    assert result.amount_usd == Decimal("1e-05")


@pytest.mark.parametrize(
    "raw_usage",
    [
        None,
        {},
        {"cost_usd": None},
        {"cost_usd": -0.01},
        {"cost_usd": "0.01"},
        {"cost_usd": True},
        {"cost_usd": float("inf")},
        {"cost_usd": float("-inf")},
        {"cost_usd": float("nan")},
    ],
)
def test_missing_or_invalid_cost_usd_returns_none(orcarouter_profile, raw_usage):
    from agent.usage_pricing import CanonicalUsage

    usage = CanonicalUsage(input_tokens=1, output_tokens=1, raw_usage=raw_usage)
    assert orcarouter_profile.get_usage_cost("openai/gpt-5.4-nano", usage) is None


def test_no_raw_usage_attribute_returns_none(orcarouter_profile):
    class _NoRawUsage:
        pass

    assert orcarouter_profile.get_usage_cost("openai/gpt-5.4-nano", _NoRawUsage()) is None


def test_zero_cost_is_estimated_not_none(orcarouter_profile):
    from agent.usage_pricing import CanonicalUsage

    usage = CanonicalUsage(raw_usage={"cost_usd": 0.0})
    result = orcarouter_profile.get_usage_cost("openai/gpt-5.4-nano", usage)
    assert result is not None
    assert result.amount_usd == Decimal("0")
    assert result.status == "estimated"


@pytest.mark.parametrize("bad_value", [float("inf"), float("-inf"), float("nan")])
def test_non_finite_cost_is_rejected_directly(orcarouter_profile, bad_value):
    """math.isfinite guard, checked directly against the sentinel values it
    exists for (nan in particular: `nan < 0` is False in Python, so the
    pre-existing negative check alone would silently let it through)."""
    from agent.usage_pricing import CanonicalUsage

    assert not math.isfinite(bad_value)
    usage = CanonicalUsage(raw_usage={"cost_usd": bad_value})
    assert orcarouter_profile.get_usage_cost("openai/gpt-5.4-nano", usage) is None
