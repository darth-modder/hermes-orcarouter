"""Pytest configuration for the orcarouter-provider test suite.

Locates a Hermes Agent checkout so tests can exercise real discovery
(hermetic tests go through the PUBLIC discovery entry points only, never a
private scanning function). Resolution order:

1. ``HERMES_AGENT_DIR`` env var -- path to a hermes-agent checkout; added to
   ``sys.path`` if not already importable.
2. An already-importable hermes-agent install (e.g. this environment's
   interpreter already has it installed, editable or not).

If neither resolves, ``providers`` stays unimportable. By default every test
in ``test_discovery.py`` is then skipped cleanly via
``pytest.importorskip("providers")`` -- never a hard failure for a
contributor who hasn't set up a hermes-agent checkout locally.

Set ``REQUIRE_HERMES=1`` (CI, nightly canary) to turn that skip into a hard
failure instead: a missing ``providers`` module then means the discovery
suite silently ran zero real assertions, which is worse than an explicit
failure that says why.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

_hermes_agent_dir = os.environ.get("HERMES_AGENT_DIR", "").strip()
if _hermes_agent_dir:
    _resolved = str(Path(_hermes_agent_dir).resolve())
    if _resolved not in sys.path:
        sys.path.insert(0, _resolved)

if os.environ.get("REQUIRE_HERMES", "").strip() == "1":
    try:
        import providers  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "REQUIRE_HERMES=1 but hermes-agent's `providers` module is not importable. "
            "Set HERMES_AGENT_DIR to a hermes-agent checkout (or install hermes-agent "
            "into this interpreter)."
        ) from exc

# The plugin surface lives under orcarouter/ (repo root also holds
# README.md, dev tooling and tests/, none of which ship to a
# `hermes plugins install` target).
PLUGIN_DIR = Path(__file__).resolve().parent.parent / "orcarouter"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def clear_provider_caches() -> None:
    """Reset process-wide provider discovery state between tests.

    Mirrors hermes-agent's tests/providers/test_installed_plugin_discovery.py
    ``_clear_provider_caches()`` verbatim in intent. Shared across every test
    module in this suite (not just test_discovery.py) because several other
    suites also load the profile through the public discovery entry points
    (the hermetic-discovery rule) and must not leak state between tests.
    """
    import providers as _providers

    _providers._REGISTRY.clear()
    _providers._ALIASES.clear()
    _providers._PROVIDER_LIST_CACHE = None
    _providers._discovered = False
    for mod in list(sys.modules):
        if mod.startswith(("plugins.model_providers", "_hermes_user_provider")):
            del sys.modules[mod]


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """A temp HERMES_HOME with the plugin installed in the flat layout
    ``hermes plugins install`` produces: ``$HERMES_HOME/plugins/<name>/``.

    Source files are copied from ``orcarouter/`` -- the real installer's own
    subdir handling is not covered by this test suite; this fixture only
    needs to reproduce the flat layout the installer produces on the other
    end.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    install_dir = tmp_path / "plugins" / "orcarouter-provider"
    install_dir.mkdir(parents=True)
    shutil.copy2(PLUGIN_DIR / "__init__.py", install_dir / "__init__.py")
    shutil.copy2(PLUGIN_DIR / "plugin.yaml", install_dir / "plugin.yaml")
    clear_provider_caches()
    yield tmp_path
    clear_provider_caches()


@pytest.fixture
def orcarouter_profile(hermes_home):
    """The registered ``orcarouter`` ``ProviderProfile``, loaded through the
    public discovery entry point rather than imported directly."""
    import providers as _providers

    profile = _providers.get_provider_profile("orcarouter")
    assert profile is not None
    return profile
