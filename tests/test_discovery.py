"""Hermetic discovery tests for the orcarouter-provider plugin.

Load ONLY through the public discovery entry points
(``providers.get_provider_profile`` / ``providers.list_providers``)
against a temp ``HERMES_HOME`` holding the flat install layout
(``plugins/orcarouter-provider/__init__.py`` + ``plugin.yaml``) that
``hermes plugins install`` produces. Never call the private
``providers._scan_home_layer`` (or any other private scanning function)
directly from a test.

Registry/cache reset between tests DOES touch private module globals
(``_REGISTRY``, ``_ALIASES``, ``_PROVIDER_LIST_CACHE``, ``_discovered``,
``sys.modules``) -- this mirrors hermes-agent's own
``tests/providers/test_installed_plugin_discovery.py::_clear_provider_caches``
exactly. That is cache invalidation between test runs (so a prior test's
import doesn't leak into the next one), not a substitute for the tests
themselves going through the public lookup functions.

The whole suite is skipped cleanly (not failed) when hermes-agent isn't
importable -- see ``conftest.py`` for how ``HERMES_AGENT_DIR`` is resolved.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

providers = pytest.importorskip(
    "providers", reason="hermes-agent not importable (set HERMES_AGENT_DIR to a checkout)"
)

from hermes_cli.plugins_manifest import version_satisfies  # noqa: E402

# The plugin surface lives under orcarouter/ (repo root also holds
# README.md, dev tooling and tests/, none of which ship to a
# `hermes plugins install` target). Kept as a plain module-level constant
# (not imported from conftest.py) since tests/ has no __init__.py and is not
# a package -- see conftest.py's identically-named PLUGIN_DIR for the
# hermes_home / orcarouter_profile fixtures shared across the suite.
_PLUGIN_DIR = Path(__file__).resolve().parent.parent / "orcarouter"
_VERSION_LIKE = re.compile(r"\d+\.\d+")
_FORBIDDEN_HEADER_NAMES = {"user-agent", "http-referer", "x-title"}

# hermes_home fixture is shared via conftest.py (used by every test module
# that loads the profile through the public discovery entry points).


def test_registers_by_canonical_name(hermes_home):
    profile = providers.get_provider_profile("orcarouter")
    assert profile is not None
    assert profile.name == "orcarouter"


def test_registers_by_alias(hermes_home):
    profile = providers.get_provider_profile("orca")
    assert profile is not None
    assert profile.name == "orcarouter"


def test_base_url_env_vars_auth_type_api_mode(hermes_home):
    profile = providers.get_provider_profile("orcarouter")
    assert profile.base_url == "https://api.orcarouter.ai/v1"
    assert profile.env_vars == ("ORCAROUTER_API_KEY", "ORCAROUTER_BASE_URL")
    assert profile.auth_type == "api_key"
    assert profile.api_mode == "chat_completions"


def test_appears_in_list_providers(hermes_home):
    names = [p.name for p in providers.list_providers()]
    assert "orcarouter" in names


def test_no_identifying_default_headers(hermes_home):
    """No attribution headers ship without opt-in (hermes-agent AGENTS.md:
    "Outbound telemetry / usage attribution without opt-in gating" is
    rejected outright).

    The one static header this profile does send
    (``X-OrcaRouter-Include-Cost``) shapes the response, not attribution, so
    this test does not require ``default_headers`` to be empty; it asserts
    the narrower, durable invariant: no User-Agent/HTTP-Referer/X-Title
    override, and no header whose value leaks a version, install id, or the
    string "hermes" -- the shape an attribution tag would take.
    """
    profile = providers.get_provider_profile("orcarouter")
    headers = profile.default_headers
    assert not (set(h.lower() for h in headers) & _FORBIDDEN_HEADER_NAMES)
    for name, value in headers.items():
        assert "hermes" not in str(name).lower()
        assert "hermes" not in str(value).lower()
        assert not _VERSION_LIKE.search(str(value)), f"{name!r} looks like it carries a version: {value!r}"


def test_plugin_yaml_parses_and_declares_model_provider():
    manifest = yaml.safe_load((_PLUGIN_DIR / "plugin.yaml").read_text(encoding="utf-8"))
    assert manifest["kind"] == "model-provider"
    # `requires_hermes` must be a real Hermes release semver bound: Hermes'
    # own version resolution (`hermes_cli.plugins_manifest.running_hermes_version()`)
    # returns a base_version like "0.21.5", not a CalVer release-date tag
    # such as "2026.9.24". Assert this behaviourally against core's own
    # comparator rather than string-matching the prefix.
    req = manifest["requires_hermes"]
    assert version_satisfies(req, "0.21.5") is True
    assert version_satisfies(req, "0.19.0") is False
