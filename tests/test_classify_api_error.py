"""classify_api_error: the free-access eligibility override, and everything
else declining to core's own classification.

The override is backed by a recorded real wire body
(``real_429_free_access_denied.json`` -- a live 429 from a free-model call,
request id scrubbed). ``synthetic_403_free_access.json`` is a clearly-labeled
synthetic replay of that same body at a different status, to prove the match
is keyed on ``error.metadata.reason`` alone. Every other fixture in this
file asserts the override declines (None) and core's own baseline verdict
is unaffected by the plugin being installed.

Two kinds of tests live here:

1. Pure-function parser tests -- calling the hook directly with a plain
   dict body. These test ``_classify_api_error``'s own parsing logic (body
   shape, ``metadata.reason`` matching) in isolation; they do not exercise
   a real SDK exception object and are not a substitute for #2.
2. SDK-driven tests through core's public ``agent.error_classifier.classify_api_error``.
   Every exception here is built through the real openai SDK's own
   construction path: a ``httpx.MockTransport``-backed client call that
   raises exactly as a live request against OrcaRouter would (matching
   ``OpenAI._make_status_error_from_response``), or, for the one case that
   path can't reach -- a mid-stream SSE error, which never carries a status
   code -- the same ``openai.APIError(message, request=..., body=...)`` call
   the SDK itself makes in its streaming decoder (see
   ``test_status_less_sse_mid_stream_error_matches`` for the citation). No
   hand-built exception stand-ins and no envelope-shaped fakes.

Following Hermes' plugin testing guidance (assert relations, not frozen values):
every "declines to core" test classifies the SAME SDK-built exception twice
-- once with the plugin's provider slug (``orcarouter``) and once with a
provider slug that has no registered profile -- and asserts the two verdicts
are equal, i.e. installing this plugin changed nothing about that case. This
avoids pinning the test suite to core's current classification for cases
this plugin has no opinion about.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("providers", reason="hermes-agent not importable (set HERMES_AGENT_DIR to a checkout)")

httpx = pytest.importorskip("httpx", reason="httpx not importable (see requirements-dev.txt)")
openai = pytest.importorskip("openai", reason="openai not importable (see requirements-dev.txt)")

_FIXTURES = Path(__file__).resolve().parent / "fixtures"

# A provider slug with no registered profile: the "no plugin hook" baseline
# for the equal-verdict relation below. Deliberately not a real provider name
# (avoids any provider-specific carve-outs in core's classifier that key off
# a real slug, e.g. the Nous welcome-tier path).
_NO_HOOK_PROVIDER = "orcarouter-test-no-profile"


def _load(name: str) -> dict:
    return json.loads((_FIXTURES / name).read_text(encoding="utf-8"))


def _sdk_exception(status: int, body: dict, *, model: str = "orcarouter/free") -> Exception:
    """Build a real openai SDK exception the way a live call against
    OrcaRouter would produce one: a MockTransport returns the recorded
    status/body, and the client's own ``_make_status_error_from_response``
    (invoked internally by the request path) raises the matching exception
    class with ``exc.body`` set to the INNER error object -- exactly the
    shape ``agent.error_classifier._body_of`` receives in production.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body, request=request)

    # max_retries=0 matches how Hermes itself constructs the client (see
    # agent's client builder) -- without it, the SDK's own retry/backoff
    # wrapper would transparently retry a 429/503 against the MockTransport
    # (sleeping between attempts) before finally raising, which is not what
    # a caller of classify_api_error ever observes in production: Hermes'
    # own retry loop lives above the SDK call, not inside it.
    client = openai.OpenAI(
        api_key="sk-test", base_url="https://api.orcarouter.ai/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )
    with pytest.raises(openai.APIStatusError) as exc_info:
        client.chat.completions.create(model=model, messages=[{"role": "user", "content": "hi"}])
    return exc_info.value


def _classify_with_and_without_plugin(exc: Exception, *, model: str):
    """Classify the same exception with and without the plugin's provider slug.

    ``ClassifiedError.provider`` just echoes back whichever slug the caller
    passed in (for logging/context) and is not part of the classification
    decision, so it is normalized out of the equality check below -- the two
    calls necessarily pass different slugs by construction.
    """
    import dataclasses

    from agent.error_classifier import classify_api_error

    with_plugin = classify_api_error(exc, provider="orcarouter", model=model)
    without_plugin = classify_api_error(exc, provider=_NO_HOOK_PROVIDER, model=model)
    return with_plugin, dataclasses.replace(without_plugin, provider=with_plugin.provider)


# ── 1. Pure-function parser tests ───────────────────────────────────────


def test_parser_matches_enveloped_body(orcarouter_profile):
    """A body shaped ``{"error": {...}}`` -- the shape a raw wire capture uses."""
    fx = _load("real_429_free_access_denied.json")
    result = orcarouter_profile.classify_api_error(
        Exception("boom"), status_code=fx["status"], error_code="free_rate_limited",
        message="", body=fx["body"], model="orcarouter/free",
    )
    assert result == {
        "reason": "upstream_rate_limit",
        "should_fallback": True,
        "error_context": {"orcarouter_reason": "err_free_access_denied"},
    }


def test_parser_matches_inner_body_shape(orcarouter_profile):
    """The INNER error object alone, with no ``{"error": ...}`` envelope -- the
    shape the openai SDK's own ``exc.body`` actually carries (see the hook's
    docstring for why both shapes must be accepted)."""
    fx = _load("real_429_free_access_denied.json")
    inner_body = fx["body"]["error"]
    result = orcarouter_profile.classify_api_error(
        Exception("boom"), status_code=fx["status"], error_code="free_rate_limited",
        message="", body=inner_body, model="orcarouter/free",
    )
    assert result == {
        "reason": "upstream_rate_limit",
        "should_fallback": True,
        "error_context": {"orcarouter_reason": "err_free_access_denied"},
    }


def test_parser_synthetic_403_same_body_still_matches(orcarouter_profile):
    """Matched on error.metadata.reason alone, independent of HTTP status."""
    fx = _load("synthetic_403_free_access.json")
    result = orcarouter_profile.classify_api_error(
        Exception("boom"), status_code=fx["status"], error_code="free_rate_limited",
        message="", body=fx["body"], model="orcarouter/free",
    )
    assert result is not None
    assert result["reason"] == "upstream_rate_limit"
    assert result["should_fallback"] is True
    assert "retryable" not in result
    assert "should_rotate_credential" not in result
    assert "buy_credits_url" not in result["error_context"]


def test_parser_other_reason_declines(orcarouter_profile):
    fx = _load("synthetic_429_other_reason.json")
    result = orcarouter_profile.classify_api_error(
        Exception("boom"), status_code=fx["status"], error_code="free_rate_limited",
        message="", body=fx["body"], model="orcarouter/free",
    )
    assert result is None


def test_parser_missing_metadata_declines(orcarouter_profile):
    fx = _load("synthetic_429_no_metadata.json")
    result = orcarouter_profile.classify_api_error(
        Exception("boom"), status_code=fx["status"], error_code="free_rate_limited",
        message="", body=fx["body"], model="orcarouter/free",
    )
    assert result is None


@pytest.mark.parametrize(
    "fixture_name",
    ["real_401_invalid_key.json", "real_400_unknown_model.json", "real_503_upstream_unavailable.json",
     "real_400_upstream_rejected_request.json"],
)
def test_parser_other_real_fixtures_decline(orcarouter_profile, fixture_name):
    fx = _load(fixture_name)
    result = orcarouter_profile.classify_api_error(
        Exception("boom"), status_code=fx["status"], error_code="",
        message="", body=fx["body"], model="openai/gpt-5.4-nano",
    )
    assert result is None


@pytest.mark.parametrize(
    "body",
    [None, {}, "not even a dict", "<html>502 Bad Gateway</html>", {"error": "a bare string"},
     {"error": {"metadata": "not a dict"}}, {"error": {"metadata": {"reason": None}}}, 12345, []],
)
def test_parser_garbage_bodies_never_raise(orcarouter_profile, body):
    result = orcarouter_profile.classify_api_error(
        Exception("boom"), status_code=429, error_code="", message="", body=body, model=None,
    )
    assert result is None


# ── 2. SDK-driven tests through core's public classifier ────────────────


def test_real_rate_limit_error_reaches_upstream_rate_limit(orcarouter_profile):
    """The plugin registration itself is enough for agent.error_classifier's
    _profile_verdict stage to consult our hook -- prove it is actually wired in,
    not just correct in isolation, using a REAL openai.RateLimitError built by
    the SDK from the recorded 429 body."""
    fx = _load("real_429_free_access_denied.json")
    exc = _sdk_exception(fx["status"], fx["body"])
    assert isinstance(exc, openai.RateLimitError)

    from agent.error_classifier import FailoverReason, classify_api_error

    result = classify_api_error(exc, provider="orcarouter", model="orcarouter/free")
    assert result.reason == FailoverReason.upstream_rate_limit
    assert result.should_fallback is True
    assert result.should_rotate_credential is False
    assert result.error_context.get("orcarouter_reason") == "err_free_access_denied"


def test_real_permission_denied_error_reaches_upstream_rate_limit(orcarouter_profile):
    """Same body, replayed as a real openai.PermissionDeniedError (403) -- the
    override is keyed on error.metadata.reason, not on which exception class
    the SDK picked for the status."""
    fx = _load("synthetic_403_free_access.json")
    exc = _sdk_exception(fx["status"], fx["body"])
    assert isinstance(exc, openai.PermissionDeniedError)

    from agent.error_classifier import FailoverReason, classify_api_error

    result = classify_api_error(exc, provider="orcarouter", model="orcarouter/free")
    assert result.reason == FailoverReason.upstream_rate_limit
    assert result.should_fallback is True


def test_status_less_sse_mid_stream_error_matches(orcarouter_profile):
    """A mid-stream SSE error never carries a status code, so it can't be
    driven through MockTransport/_make_status_error_from_response (that path
    only runs for a non-2xx HTTP response). Instead this replicates the SDK's
    OWN construction call for this exact case, verbatim: openai/_streaming.py
    (``Stream._iter_events``), ``raise APIError(message=message,
    request=self.response.request, body=data["error"])`` -- i.e. a status-less
    APIError whose body is already the inner error object, read out of a
    `{"error": {...}}` SSE payload the same way core's fixture body is
    shaped."""
    fx = _load("real_429_free_access_denied.json")
    exc = openai.APIError(
        fx["body"]["error"]["message"],
        request=httpx.Request("POST", "https://api.orcarouter.ai/v1/chat/completions"),
        body=fx["body"]["error"],
    )
    assert getattr(exc, "status_code", None) is None

    from agent.error_classifier import FailoverReason, classify_api_error

    result = classify_api_error(exc, provider="orcarouter", model="orcarouter/free")
    assert result.reason == FailoverReason.upstream_rate_limit
    assert result.should_fallback is True


@pytest.mark.parametrize(
    "fixture_name,model",
    [
        ("real_401_invalid_key.json", "openai/gpt-5.4-nano"),
        ("real_400_unknown_model.json", "this-model-does-not-exist-test-probe"),
        ("real_503_upstream_unavailable.json", "anthropic/claude-haiku-4.5"),
        ("real_400_upstream_rejected_request.json", "openai/gpt-5.4-nano"),
        ("synthetic_429_other_reason.json", "orcarouter/free"),
        ("synthetic_429_no_metadata.json", "orcarouter/free"),
    ],
)
def test_declines_leave_core_classification_unchanged(orcarouter_profile, fixture_name, model):
    """For every case this plugin has no opinion about, classifying the SAME
    SDK-built exception with the plugin's provider slug and with a slug that
    has no registered profile must give the identical verdict -- installing
    this plugin does not change core's behaviour here."""
    fx = _load(fixture_name)
    exc = _sdk_exception(fx["status"], fx["body"], model=model)
    with_plugin, without_plugin = _classify_with_and_without_plugin(exc, model=model)
    assert with_plugin == without_plugin


def test_reasoning_rejection_with_upstream_message_declines_and_core_repairs_session(orcarouter_profile):
    """A 400 whose message names ``reasoning_effort`` (real, from
    openai/gpt-4.1-nano, code null) is left to core; core recognises it as a
    reasoning rejection via ``is_reasoning_field_rejection``
    (agent/error_classifier.py) and repairs the session by dropping the
    field and retrying once. A future override must not intercept it before
    that repair runs."""
    fx = _load("real_400_reasoning_unsupported.json")
    exc = _sdk_exception(fx["status"], fx["body"], model="openai/gpt-4.1-nano")

    with_plugin, without_plugin = _classify_with_and_without_plugin(exc, model="openai/gpt-4.1-nano")
    assert with_plugin == without_plugin

    from agent.error_classifier import FailoverReason

    # Single documented value check (no relation available -- this IS the
    # thing under test): core's reasoning-rejection verdict.
    assert without_plugin.reason == FailoverReason.reasoning_mandatory


def test_reasoning_rejection_without_upstream_message_does_not_get_reasoning_verdict(orcarouter_profile):
    """The counterpart on a different model: an ``upstream_rejected_request``
    400 (real_400_upstream_rejected_request.json, openai/gpt-5.4-nano) is
    left to core; core's classification is not the reasoning_mandatory
    verdict. Guards against a future override producing the reasoning
    verdict for a request core does not recognize as one."""
    fx = _load("real_400_upstream_rejected_request.json")
    exc = _sdk_exception(fx["status"], fx["body"], model="openai/gpt-5.4-nano")

    with_plugin, without_plugin = _classify_with_and_without_plugin(exc, model="openai/gpt-5.4-nano")
    assert with_plugin == without_plugin

    from agent.error_classifier import FailoverReason

    assert without_plugin.reason != FailoverReason.reasoning_mandatory


# ── 3. Side-effect test: the verdict must not persist a model bench ─────


@pytest.mark.core_internal
def test_upstream_rate_limit_verdict_does_not_bench_credential_or_model(tmp_path, monkeypatch, orcarouter_profile):
    """Regression for the reason this override is upstream_rate_limit and not
    model_entitlement: driven through a REAL credential pool (one OrcaRouter
    key, on disk) and core's real recovery entry point,
    ``agent.agent_runtime_helpers.recover_with_credential_pool``, the
    free-access verdict must not rotate the credential, must not record a
    model cooldown, and the same key must stay selectable afterwards -- for
    the SAME model and for a different one.

    ``recover_with_credential_pool`` lazily imports the whole ``run_agent``
    module tree just to fetch a logger (``_ra().logger``, module-level helper
    at the top of agent/agent_runtime_helpers.py) -- that import chain pulls
    in Hermes' full CLI/tool runtime (terminal sandboxes, Modal, browser
    tools) which has nothing to do with credential-pool recovery and is not
    a reasonable dependency for this plugin's own test suite. ``_ra`` is
    monkeypatched to a stub logger so this test drives the REAL decision
    logic in ``recover_with_credential_pool`` end-to-end without installing
    hermes-agent's entire runtime dependency tree.
    """
    import json as _json

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("ORCAROUTER_API_KEY", raising=False)
    (hermes_home / "auth.json").write_text(
        _json.dumps(
            {
                "version": 1,
                "credential_pool": {
                    "orcarouter": [
                        {
                            "id": "cred-1",
                            "label": "cred-1",
                            "auth_type": "api_key",
                            "priority": 0,
                            "source": "manual",
                            "access_token": "sk-test-key",
                            "base_url": "https://api.orcarouter.ai/v1",
                        }
                    ]
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    fx = _load("real_429_free_access_denied.json")
    exc = _sdk_exception(fx["status"], fx["body"])

    from agent.error_classifier import classify_api_error

    result = classify_api_error(exc, provider="orcarouter", model="orcarouter/free")

    from agent.credential_pool import load_pool

    pool = load_pool("orcarouter")
    before = pool.entries()[0]
    assert before.model_cooldowns is None
    assert before.last_status is None

    import agent.agent_runtime_helpers as arh

    class _StubLogger:
        @staticmethod
        def info(*a, **k):
            pass

        @staticmethod
        def warning(*a, **k):
            pass

        @staticmethod
        def debug(*a, **k):
            pass

    class _StubRA:
        logger = _StubLogger

    monkeypatch.setattr(arh, "_ra", lambda: _StubRA)

    class _FakeAgent:
        provider = "orcarouter"
        base_url = "https://api.orcarouter.ai/v1"
        model = "orcarouter/free"
        _credential_pool = pool

    recovered, has_retried = arh.recover_with_credential_pool(
        _FakeAgent(), status_code=result.status_code, has_retried_429=False,
        classified_reason=result.reason, error_context=result.error_context,
    )
    assert recovered is False  # nothing to rotate to, nor was rotation attempted

    reloaded = load_pool("orcarouter")
    after = reloaded.entries()[0]
    assert after.model_cooldowns is None  # nothing persisted for (key, model)
    assert after.last_status is None  # not marked exhausted
    assert reloaded.has_available() is True
    assert reloaded.select() is not None  # still selectable for the same model
    assert reloaded.select(model="orcarouter/auto") is not None  # and for a different one
