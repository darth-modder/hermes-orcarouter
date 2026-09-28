"""OrcaRouter provider profile for Hermes Agent.

Registers OrcaRouter (https://orcarouter.ai) as a model-provider plugin, the
same way the in-tree ``openrouter`` provider works, with no core edits.
OrcaRouter-Lite uses bare, non-vendor-prefixed model ids and the bare
literal ``auto`` for its auto router. This plugin is tested against hosted
OrcaRouter.

Hooks implemented here:

- ``default_headers`` — a single static, non-identifying response-shaping
  flag (see the field itself for why it is not attribution).
- ``fetch_models`` — calls the base implementation, then dedupes and orders
  router ids first; cached per process, keyed by (base_url, key fingerprint)
  so OrcaRouter-Lite and hosted OrcaRouter (or two different keys) never
  share a cache entry.
- ``build_api_kwargs_extras`` — a local effort map onto OrcaRouter's
  top-level ``reasoning_effort`` (see the map for why each level maps where
  it does).
- ``classify_api_error`` — overrides core's classification only for the
  free-access eligibility gate, keyed on ``metadata.reason`` (never a
  message-text match). The verdict is ``upstream_rate_limit``, not
  ``model_entitlement``: the free-tier block is a session-scoped throttle,
  not a permanent loss of entitlement, so it must not pin a long-lived
  persisted cooldown on the (key, model) pair the way ``model_entitlement``
  does. ``upstream_rate_limit`` skips credential rotation, falls back
  immediately, and only starts an in-memory session cooldown -- nothing
  persisted. Everything else returns ``None`` so core's own pipeline
  decides, including a 503 ``upstream_unavailable`` (core retries, then
  falls back once a fallback chain is configured) and a 400
  ``upstream_rejected_request`` for an invalid ``reasoning_effort`` (core
  classifies it as a format error and falls back).
  Note: even a verdict with ``should_fallback=False`` can still fall back
  through core's own retry/exhaustion paths when ``fallback_providers`` is
  configured (eager fallback for the rate-limit family and, after a couple
  of retries, for transport failures; exhaustion fallback for any reason
  once retries are exhausted) -- "no override" is not "no fallback ever".
- ``get_usage_cost`` — reads ``usage.cost_usd`` from the raw usage payload
  (present only because of the cost header above).
- ``default_aux_model`` — a fixed cheap model, not an OrcaRouter router id
  (see the field for why).

This module performs no direct process-environment reads: credentials and
the base-URL override flow through the declared ``env_vars`` and Hermes' own
secret scope instead.
"""

from __future__ import annotations

import hashlib
import math
from decimal import Decimal
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

# Reason string OrcaRouter uses on the free-tier eligibility block, matched
# on metadata.reason, independent of HTTP status. Backed by a recorded real
# 429 body from a free-model call (request id scrubbed), committed alongside
# this plugin's tests.
_FREE_ACCESS_DENIED_REASON = "err_free_access_denied"


def _free_access_verdict() -> dict[str, Any]:
    """The verdict for the free-access eligibility gate: ``upstream_rate_limit``,
    not ``model_entitlement``. The block is a session-scoped throttle, not a
    permanent loss of entitlement -- ``model_entitlement`` pins a long-lived
    persisted cooldown on the (key, model) pair, which is reserved for actual
    plan properties. ``upstream_rate_limit`` skips credential rotation and
    falls back immediately, with only an in-memory session cooldown, nothing
    persisted to disk. Returns a fresh dict each call so callers can't mutate
    shared state. Never includes ``buy_credits_url`` in ``error_context``.
    """
    return {
        "reason": "upstream_rate_limit",
        "should_fallback": True,
        "error_context": {"orcarouter_reason": _FREE_ACCESS_DENIED_REASON},
    }


# Effort map for OrcaRouter's top-level ``reasoning_effort``. Not every
# upstream model accepts every level in the wire vocabulary, so each level
# is mapped to the nearest level that is broadly accepted, rather than sent
# unmapped and left to fail:
#   - low/medium/high/xhigh: sent as-is.
#   - minimal: mapped down to "low".
#   - max: mapped up to "xhigh".
#   - unset / unknown / disabled (enabled False or effort "none"): the field
#     is omitted entirely rather than guessed. Unsupported levels are mapped
#     rather than sent, so a request is not rejected for them.
_EFFORT_MAP: dict[str, str] = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "minimal": "low",
    "max": "xhigh",
}

# Process-level fetch_models cache, keyed by (effective_base_url, key
# fingerprint) so a base-URL override (OrcaRouter-Lite) or a different key
# never shares an entry with hosted OrcaRouter. The key itself is never
# stored -- only a truncated SHA-256 fingerprint, and only to distinguish
# cache entries, never for any security purpose.
_MODELS_CACHE: dict[tuple[str, str], list[str]] = {}


def _models_cache_key(base_url: str | None, api_key: str | None, default_base_url: str) -> tuple[str, str]:
    effective_base = (base_url or default_base_url or "").rstrip("/")
    fingerprint = hashlib.sha256(api_key.encode()).hexdigest()[:16] if api_key else ""
    return (effective_base, fingerprint)


def _classify_api_error(
    error: Exception,
    *,
    status_code: int | None = None,
    error_code: str | None = None,
    message: str | None = None,
    body: dict | None = None,
    model: str | None = None,
) -> dict[str, Any] | None:
    """Override core's classification only for the free-access eligibility gate.

    Matched on ``error.metadata.reason == "err_free_access_denied"`` alone,
    independent of HTTP status. See ``_free_access_verdict`` for why the
    verdict is ``upstream_rate_limit`` rather than ``model_entitlement``.

    The openai SDK's own exception construction already unwraps the
    envelope: ``exc.body`` is the INNER ``error`` object, not
    ``{"error": {...}}`` (``client._make_status_error`` does
    ``body.get("error", body)`` before building the exception -- see
    ``openai/_client.py``). Core forwards whatever ``body`` it was given
    unchanged. A caller that builds a body directly from a raw wire capture
    (as the unit tests here do, and as a status-less streaming ``APIError``
    does) still passes the full ``{"error": {...}}`` envelope, so both
    shapes are accepted here.

    Everything else -- 401, 429 without this metadata, 503, 400, non-JSON
    bodies, garbage of any shape -- returns None so core's own pipeline
    classifies it; this function never raises.
    """
    try:
        err = body["error"] if isinstance(body, dict) and isinstance(body.get("error"), dict) else body
        if not isinstance(err, dict):
            return None
        metadata = err.get("metadata")
        if not isinstance(metadata, dict):
            return None
        if metadata.get("reason") != _FREE_ACCESS_DENIED_REASON:
            return None
    except Exception:
        return None
    return _free_access_verdict()


class OrcaRouterProfile(ProviderProfile):
    """OrcaRouter — unified inference gateway. See module docstring for scope."""

    def fetch_models(
        self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 8.0, **kwargs: Any
    ) -> list[str] | None:
        """Live catalog via the base implementation, deduped with router ids first.

        Cached per process (see ``_MODELS_CACHE``); a failed fetch is never
        cached so a transient failure can succeed on the next call, and an
        empty result is returned but not cached either (an empty catalog is
        more likely a transient hiccup than a real state worth remembering).
        Callers fall back to ``fallback_models`` on a ``None`` return,
        unchanged from the base contract. Every return is a fresh list copy,
        never the object stored in the cache, whether this call was a hit or
        the one that populated the entry -- so a caller mutating its result
        can't corrupt what a later call reads back.

        Accepts and forwards ``**kwargs`` to the base implementation for
        forward compatibility with core call sites this plugin doesn't know
        about yet. Known limit: core also calls ``fetch_models()`` with no
        arguments at all in some code paths (``hermes_cli/models.py``), where
        the per-(base_url, key) cache isolation below doesn't apply -- that
        call shares the no-key/default-base-url cache entry.
        """
        cache_key = _models_cache_key(base_url, api_key, self.base_url)
        cached = _MODELS_CACHE.get(cache_key)
        if cached is not None:
            return list(cached)
        try:
            result = super().fetch_models(api_key=api_key, base_url=base_url, timeout=timeout, **kwargs)
        except Exception:
            return None
        if result is None:
            return None
        # Fail open on a malformed catalog entry (a non-string id) instead of
        # raising out of a picker-refresh path -- skip it rather than crash.
        str_ids = [m for m in result if isinstance(m, str)]
        deduped = list(dict.fromkeys(str_ids))
        routers = [m for m in deduped if m.startswith("orcarouter/")]
        others = [m for m in deduped if not m.startswith("orcarouter/")]
        ordered = routers + others
        if ordered:
            _MODELS_CACHE[cache_key] = ordered
        return list(ordered)

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, **context: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Map ``reasoning_config`` onto OrcaRouter's top-level ``reasoning_effort``
        via ``_EFFORT_MAP``. Deliberately does not gate on ``supports_reasoning``
        (core's allowlist doesn't cover this provider) and does not branch on
        model id.
        """
        if not isinstance(reasoning_config, dict) or reasoning_config.get("enabled") is False:
            return {}, {}
        effort = str(reasoning_config.get("effort") or "").strip().lower()
        if not effort or effort == "none":
            return {}, {}
        mapped = _EFFORT_MAP.get(effort)
        if mapped is None:
            return {}, {}
        return {}, {"reasoning_effort": mapped}

    def get_usage_cost(self, model: str, usage: Any) -> Any | None:
        """CostResult from OrcaRouter's per-response ``usage.cost_usd`` (only present
        when the cost header in ``default_headers`` is sent).

        ``CostResult`` is a frozen dataclass (``amount_usd: Decimal | None``,
        ``status: CostStatus``, ``source: CostSource``, ``label: str``, ...);
        core's ``estimate_usage_cost()`` calls ``profile.get_usage_cost(model_name,
        usage)`` and uses a non-None result as-is, in preference to its own
        models.dev-derived estimate. ``usage.raw_usage`` holds the raw dict (or
        ``model_dump()``) of whatever usage object the SDK returned, so
        ``cost_usd`` -- kept by the openai SDK's ``extra="allow"`` model config --
        survives there even though the canonical usage type has no dedicated
        field for it.

        Status is ``"estimated"``, not ``"actual"``. ``"estimated"`` and
        ``source="provider_cost_api"`` are both valid independently
        (``CostStatus``/``CostSource`` are unrelated Literal types; see
        ``agent/usage_pricing.py``), so this pairing is legal even though the
        amount comes directly from the provider's own response rather than
        from a separate estimate.

        Fails open: missing/non-numeric/non-finite/negative ``cost_usd`` ->
        None, so core's own estimator takes over.
        """
        raw = getattr(usage, "raw_usage", None)
        if not isinstance(raw, dict):
            return None
        cost = raw.get("cost_usd")
        if (
            isinstance(cost, bool)
            or not isinstance(cost, (int, float))
            or not math.isfinite(cost)
            or cost < 0
        ):
            return None
        try:
            from agent.usage_pricing import CostResult, format_cost_label

            amount = Decimal(str(cost))
            return CostResult(
                amount_usd=amount, status="estimated", source="provider_cost_api", label=format_cost_label(amount)
            )
        except Exception:
            return None


orcarouter = OrcaRouterProfile(
    name="orcarouter",
    aliases=("orca",),
    display_name="OrcaRouter",
    description="OrcaRouter — unified API for multiple model backends",
    signup_url="https://docs.orcarouter.ai/getting-started/get-api-key",
    base_url="https://api.orcarouter.ai/v1",
    env_vars=("ORCAROUTER_API_KEY", "ORCAROUTER_BASE_URL"),
    auth_type="api_key",
    fallback_models=("orcarouter/auto",),
    # Response-shaping flag: makes OrcaRouter include usage.cost_usd in the
    # response body (confirmed: /v1/models ignores it either way). It carries
    # no client identity, version or install id, the same precedent as
    # OpenRouter's x-grok-conv-id.
    default_headers={"X-OrcaRouter-Include-Cost": "true"},
    # A cheap, concrete model confirmed working for auxiliary tasks.
    # Deliberately NOT an OrcaRouter router id (orcarouter/auto or
    # orcarouter/free): auto's underlying model varies per call and free
    # fails outright for accounts without a linked, established GitHub
    # account, either of which would make auxiliary tasks (compression,
    # titles, vision) unreliable.
    default_aux_model="openai/gpt-5.4-nano",
    classify_api_error=_classify_api_error,
)

register_provider(orcarouter)
