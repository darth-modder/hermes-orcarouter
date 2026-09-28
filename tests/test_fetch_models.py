"""fetch_models: process-level cache, dedupe, router ids first.

The network call itself (``ProviderProfile.fetch_models``, the base
implementation) is mocked throughout -- these tests never touch the network.
Loaded through the public discovery entry points; the ``hermes_home``
fixture reinstalls the plugin under a fresh per-test module name (keyed by a
digest of the temp HERMES_HOME path), so the process-level ``_MODELS_CACHE``
module global starts empty in every test.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

pytest.importorskip("providers", reason="hermes-agent not importable (set HERMES_AGENT_DIR to a checkout)")


def test_dedupe_and_routers_first(orcarouter_profile):
    with patch(
        "providers.base.ProviderProfile.fetch_models",
        return_value=["openai/gpt-5.4-nano", "orcarouter/auto", "openai/gpt-5.4-nano", "orcarouter/free"],
    ) as mocked:
        result = orcarouter_profile.fetch_models(api_key="k1", base_url=None)
    assert result == ["orcarouter/auto", "orcarouter/free", "openai/gpt-5.4-nano"]
    assert mocked.call_count == 1


def test_none_result_passes_through(orcarouter_profile):
    with patch("providers.base.ProviderProfile.fetch_models", return_value=None) as mocked:
        result = orcarouter_profile.fetch_models(api_key="k1", base_url=None)
    assert result is None
    assert mocked.call_count == 1


def test_cache_hit_avoids_second_network_call(orcarouter_profile):
    with patch("providers.base.ProviderProfile.fetch_models", return_value=["orcarouter/auto"]) as mocked:
        first = orcarouter_profile.fetch_models(api_key="k1", base_url=None)
        second = orcarouter_profile.fetch_models(api_key="k1", base_url=None)
    assert first == second == ["orcarouter/auto"]
    assert mocked.call_count == 1


def test_failed_fetch_is_not_cached(orcarouter_profile):
    """A None (failed) fetch must not poison the cache -- the next call retries."""
    with patch(
        "providers.base.ProviderProfile.fetch_models", side_effect=[None, ["orcarouter/auto"]]
    ) as mocked:
        first = orcarouter_profile.fetch_models(api_key="k1", base_url=None)
        second = orcarouter_profile.fetch_models(api_key="k1", base_url=None)
    assert first is None
    assert second == ["orcarouter/auto"]
    assert mocked.call_count == 2


def test_different_base_url_does_not_share_cache(orcarouter_profile):
    with patch(
        "providers.base.ProviderProfile.fetch_models", side_effect=[["orcarouter/auto"], ["auto"]]
    ) as mocked:
        hosted = orcarouter_profile.fetch_models(api_key="k1", base_url=None)
        lite = orcarouter_profile.fetch_models(api_key="k1", base_url="https://lite.example.com/v1")
    assert hosted == ["orcarouter/auto"]
    assert lite == ["auto"]
    assert mocked.call_count == 2


def test_different_api_key_does_not_share_cache(orcarouter_profile):
    with patch(
        "providers.base.ProviderProfile.fetch_models", side_effect=[["orcarouter/a"], ["orcarouter/b"]]
    ) as mocked:
        first = orcarouter_profile.fetch_models(api_key="key-one", base_url=None)
        second = orcarouter_profile.fetch_models(api_key="key-two", base_url=None)
    assert first == ["orcarouter/a"]
    assert second == ["orcarouter/b"]
    assert mocked.call_count == 2


def test_raw_api_key_absent_from_plugin_cache(orcarouter_profile):
    """The raw key string must appear in no key or value of the plugin's own
    module-level ``_MODELS_CACHE`` -- it is keyed by
    ``(base_url, sha256(api_key)[:16])``, never the key itself. This reads
    the plugin's own global (found via the profile class's ``__module__``,
    since the discovery layer registers the plugin under a per-test hashed
    module name), not a private hermes-agent internal, so it does not
    conflict with the hermetic-discovery rule (tests load the profile only
    through the public entry points; this only inspects state the profile's
    own fetch_models() call just populated)."""
    secret = "super-secret-key"  # noqa: S105 - not a real credential, just a marker string
    with patch("providers.base.ProviderProfile.fetch_models", return_value=["orcarouter/auto"]):
        result = orcarouter_profile.fetch_models(api_key=secret, base_url=None)
    assert result == ["orcarouter/auto"]

    module = sys.modules[type(orcarouter_profile).__module__]
    cache = module._MODELS_CACHE
    assert cache, "fetch_models() above must have populated the module-level cache"
    for cache_key, cached_ids in cache.items():
        assert all(secret not in str(part) for part in cache_key)
        assert all(secret not in str(model_id) for model_id in cached_ids)


def test_non_string_ids_are_skipped_fail_open(orcarouter_profile):
    """A malformed catalog entry (a non-string id) must not crash fetch_models
    out of a picker-refresh path -- it is skipped instead."""
    with patch(
        "providers.base.ProviderProfile.fetch_models",
        return_value=["orcarouter/auto", None, 42, {"id": "bad"}, "openai/gpt-5.4-nano"],
    ) as mocked:
        result = orcarouter_profile.fetch_models(api_key="k1", base_url=None)
    assert result == ["orcarouter/auto", "openai/gpt-5.4-nano"]
    assert mocked.call_count == 1


def test_empty_result_is_returned_but_not_cached(orcarouter_profile):
    """An empty catalog is returned to the caller but never cached -- the next
    call retries instead of being stuck with an empty picker forever."""
    with patch(
        "providers.base.ProviderProfile.fetch_models", side_effect=[[], ["orcarouter/auto"]]
    ) as mocked:
        first = orcarouter_profile.fetch_models(api_key="k1", base_url=None)
        second = orcarouter_profile.fetch_models(api_key="k1", base_url=None)
    assert first == []
    assert second == ["orcarouter/auto"]
    assert mocked.call_count == 2


def test_all_non_string_ids_is_treated_as_empty_and_not_cached(orcarouter_profile):
    with patch(
        "providers.base.ProviderProfile.fetch_models", side_effect=[[None, 42], ["orcarouter/auto"]]
    ) as mocked:
        first = orcarouter_profile.fetch_models(api_key="k1", base_url=None)
        second = orcarouter_profile.fetch_models(api_key="k1", base_url=None)
    assert first == []
    assert second == ["orcarouter/auto"]
    assert mocked.call_count == 2


def test_cached_result_is_a_copy_not_the_cached_object(orcarouter_profile):
    """A caller mutating its returned list must not corrupt the cache entry a
    later call reads back."""
    with patch("providers.base.ProviderProfile.fetch_models", return_value=["orcarouter/auto"]) as mocked:
        first = orcarouter_profile.fetch_models(api_key="k1", base_url=None)
        first.append("corrupted")
        second = orcarouter_profile.fetch_models(api_key="k1", base_url=None)
    assert first == ["orcarouter/auto", "corrupted"]
    assert second == ["orcarouter/auto"]
    assert first is not second
    assert mocked.call_count == 1


def test_kwargs_are_forwarded_to_base_fetch_models(orcarouter_profile):
    """Forward-compatibility: an unknown kwarg a future core release might pass
    is accepted here and forwarded to the base implementation rather than
    raising a TypeError at this profile's own signature boundary."""
    with patch("providers.base.ProviderProfile.fetch_models", return_value=["orcarouter/auto"]) as mocked:
        result = orcarouter_profile.fetch_models(api_key="k1", base_url=None, some_future_kwarg=True)
    assert result == ["orcarouter/auto"]
    mocked.assert_called_once_with(api_key="k1", base_url=None, timeout=8.0, some_future_kwarg=True)
