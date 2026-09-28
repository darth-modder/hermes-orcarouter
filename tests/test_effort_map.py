"""build_api_kwargs_extras: the local reasoning-effort map.

Levels outside low-xhigh are mapped to the nearest supported level;
disabled/none omit the field. See the map in orcarouter/__init__.py.
Loaded through the public discovery entry points.
"""

from __future__ import annotations

import pytest

pytest.importorskip("providers", reason="hermes-agent not importable (set HERMES_AGENT_DIR to a checkout)")


@pytest.mark.parametrize(
    "effort,expected_wire_value",
    [
        ("low", "low"),
        ("medium", "medium"),
        ("high", "high"),
        ("xhigh", "xhigh"),
        ("minimal", "low"),  # outside low-xhigh: mapped to the nearest supported level
        ("max", "xhigh"),  # outside low-xhigh: mapped to the nearest supported level
    ],
)
def test_effort_levels_map_to_wire_value(orcarouter_profile, effort, expected_wire_value):
    extra_body, top_level = orcarouter_profile.build_api_kwargs_extras(
        reasoning_config={"enabled": True, "effort": effort}
    )
    assert extra_body == {}
    assert top_level == {"reasoning_effort": expected_wire_value}


def test_unset_reasoning_config_omits_field(orcarouter_profile):
    extra_body, top_level = orcarouter_profile.build_api_kwargs_extras(reasoning_config=None)
    assert extra_body == {}
    assert top_level == {}


def test_disabled_via_enabled_false_omits_field(orcarouter_profile):
    """Disabled reasoning omits the field rather than sending an explicit
    value (see the effort map in orcarouter/__init__.py)."""
    extra_body, top_level = orcarouter_profile.build_api_kwargs_extras(
        reasoning_config={"enabled": False, "effort": "high"}
    )
    assert extra_body == {}
    assert top_level == {}


def test_disabled_via_effort_none_omits_field(orcarouter_profile):
    extra_body, top_level = orcarouter_profile.build_api_kwargs_extras(
        reasoning_config={"enabled": True, "effort": "none"}
    )
    assert extra_body == {}
    assert top_level == {}


def test_unknown_effort_value_omits_field(orcarouter_profile):
    extra_body, top_level = orcarouter_profile.build_api_kwargs_extras(
        reasoning_config={"enabled": True, "effort": "ultra-mega"}
    )
    assert extra_body == {}
    assert top_level == {}
