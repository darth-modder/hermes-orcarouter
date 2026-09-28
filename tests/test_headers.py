"""Static default_headers: exactly the cost-shaping flag, nothing identifying.

Loaded through the public discovery entry points, same as test_discovery.py.
"""

from __future__ import annotations

import pytest

pytest.importorskip("providers", reason="hermes-agent not importable (set HERMES_AGENT_DIR to a checkout)")


def test_default_headers_is_exactly_the_cost_flag(orcarouter_profile):
    assert orcarouter_profile.default_headers == {"X-OrcaRouter-Include-Cost": "true"}
