# OrcaRouter plugin for Hermes Agent

> **Status: pre-release.** This plugin has not yet been verified end to end
> inside a running Hermes Agent installation, and it is not yet listed in
> the Hermes plugin catalog. See [What's tested](#whats-tested) below for
> what has and hasn't been exercised so far.

Maintained by Ahmed Bilal (OrcaRouter). This is a community project, not an
OrcaRouter-branded or company-supported product — there is no implied
company support or response-time commitment. Support is via
[GitHub Issues](https://github.com/darth-modder/hermes-orcarouter/issues)
on this repository.

## What it is

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) model-provider
plugin that registers [OrcaRouter](https://orcarouter.ai) as an inference
backend, the same way the in-tree `openrouter` provider works. Dropping this
directory in gives `hermes model` an `orcarouter` entry with no core edits.

Implemented: the static provider profile (endpoints, env vars, auth type,
picker fallback list), live model-catalog fetching with a process-level
cache, a reasoning-effort map onto OrcaRouter's top-level
`reasoning_effort`, per-response cost reporting via `usage.cost_usd`, and one
error-classification override for the free-tier eligibility block. See the
module docstring in `__init__.py` for the full hook-by-hook rationale.

## Install

```bash
hermes plugins install darth-modder/hermes-orcarouter/orcarouter
```

The plugin surface lives in the `orcarouter/` subdirectory of this
repository, not at the repo root — the `owner/repo/subdir` shorthand above
tells `hermes plugins install` to sparse-checkout only that subdirectory
(root-level dev tooling, tests and docs never ship to a user's machine).

Alternatively, copy this directory manually into your Hermes plugins folder:

```bash
cp -r orcarouter "$HERMES_HOME/plugins/orcarouter-provider"
```

## Configure

Set an API key, then pick the provider:

```bash
export ORCAROUTER_API_KEY=...
hermes model  # choose "OrcaRouter", or run: hermes --provider orcarouter -m orcarouter/auto
```

Get a key at <https://docs.orcarouter.ai/getting-started/get-api-key>.

An optional `ORCAROUTER_BASE_URL` env var points the plugin at a different
endpoint (see [OrcaRouter-Lite](#orcarouter-lite) below).

## Choosing a router

Router ids observed on hosted OrcaRouter:

- `orcarouter/auto` — OrcaRouter picks a backend model per request based on
  the routing strategy configured in your OrcaRouter dashboard.
- `orcarouter/free` — the free tier (subject to the eligibility gate
  described under [Troubleshooting](#troubleshooting)).
- `orcarouter/fusion`, `orcarouter/fusion-flash`, `orcarouter/fusion-mini` —
  fusion-family routers.
- `orcarouter/<named>` — pins a specific named router you configured in the
  dashboard. **Untested:** whether a user-named router appears in
  `/v1/models` for its owner's key has not been confirmed.
- Routing strategies (cost-optimized, quality-optimized, custom rules, etc.)
  are configured server-side in the OrcaRouter dashboard, not in this plugin.

**Note:** `orcarouter/auto` (and other auto-selecting routers) may route
different turns of the same conversation to different underlying models.
Because prompt caching is keyed per-backend-model, switching models between
turns can reduce or eliminate the cache hit rate you'd get by pinning a
single model or named router.

## Cost

Per-request cost is read from `usage.cost_usd` on the response and shown as
an **estimate**, provider-reported rather than derived from a static price
table. This plugin sends the `X-OrcaRouter-Include-Cost: true` header
automatically on every request (a static, non-identifying response-shaping
flag — it carries no client identity, version or install id) so `cost_usd`
is present without any configuration.

## Reasoning

A `reasoning_effort` request maps onto OrcaRouter's top-level
`reasoning_effort` field through a small, provider-wide table — no per-model
branching:

| You set      | OrcaRouter receives |
|--------------|----------------------|
| `low`        | `low`                |
| `medium`     | `medium`             |
| `high`       | `high`               |
| `xhigh`      | `xhigh`              |
| `minimal`    | `low`                |
| `max`        | `xhigh`              |
| unset        | *(field omitted)*    |

Not every upstream model accepts every level in the wire vocabulary, so
`minimal` and `max` are mapped to the nearest level that is broadly
accepted, rather than sent unmapped and left to fail.

**Disabling reasoning is not supported through OrcaRouter in v1.** Explicitly
turning reasoning off, or sending `effort: "none"`, is omitted rather than
sent as `"none"` — the field is omitted, so each model uses its own default
(on for models that reason by default, off for those that don't).

Setting a reasoning level on a model that doesn't support reasoning can
cause a failed request, after which Hermes may drop the setting for the
session and retry, or switch to your fallback chain, depending on whether
the rejection is recognizable as a reasoning-specific error. See the mapping
above for how this plugin reduces how often that happens.

## Automatic fallback

Installing this plugin does **not** by itself give you cross-provider
failover. Hermes only fails over to another provider when you configure a
fallback route yourself, via the `fallback_providers` config key (each
entry needs `provider` and `model`).

```yaml
# ~/.hermes/config.yaml (example — provider and model names below are
# illustrative, not a specific recommendation)
model:
  provider: orcarouter
  default: orcarouter/auto
fallback_providers:
  - provider: openai       # example: any other provider you have configured
    model: gpt-5.4-nano    # example: a model on that provider
```

A fallback entry on OrcaRouter cannot help when the whole OrcaRouter key is
out of credit or otherwise unusable — every entry that shares that key
fails the same way. That's why the fallback above is a **different
provider**, so there's still somewhere to go if OrcaRouter itself is down
or exhausted.

Without a `fallback_providers` entry, a `should_fallback` verdict from this
plugin's `classify_api_error` override has nothing to fall back to on its
own. That said, once a fallback chain is configured, Hermes' resilience
layer also falls back on its own in some cases independent of this
plugin's verdicts (for example, eagerly for the rate-limit family, and for
any reason once retries are exhausted). This plugin's own
`fallback_models=("orcarouter/auto",)` field is unrelated to any of the
above: it is picker content only, shown by `/model` and `hermes model` when
the live model-catalog fetch fails, never used for runtime failover.

## Works today without the plugin

OrcaRouter is an OpenAI-compatible endpoint, so you can already reach it
through Hermes' built-in `custom` provider without installing anything:

```yaml
# ~/.hermes/config.yaml
model:
  provider: custom
  base_url: https://api.orcarouter.ai/v1
  api_mode: chat_completions
  default: orcarouter/auto
```

This plugin exists to add first-class discovery (`hermes model` listing,
picker, doctor health check, setup wizard) on top of that same wire path.

## OrcaRouter-Lite

Set `ORCAROUTER_BASE_URL` to use an OrcaRouter-Lite endpoint. On Lite, the
auto router id is `auto` and model ids have no vendor prefix; cost
reporting, account billing and reasoning settings are not available there.
This plugin is tested against hosted OrcaRouter.

## Troubleshooting

- **Free models need account eligibility.** The workspace owner must link
  an established GitHub account in OrcaRouter, or add credits. When a
  request is refused for this reason, the plugin reports it so Hermes can
  switch to your fallback chain (if configured) without penalising your
  key.
- **Authentication errors.** Check that the key is correct and active in
  the OrcaRouter dashboard, and that the account has credit and the key's
  spend limit allows the request.
- **A reasoning level a specific model doesn't support may be rejected.**
  Hermes then drops the setting for the session or switches to your
  fallback chain. The mapping in [Reasoning](#reasoning) above reduces how
  often this happens.

## What's tested

Verified by this repository's automated test suite:

- Provider discovery and registration (canonical name and alias, base URL,
  env vars, auth type) through Hermes' public discovery entry points.
- The reasoning-effort mapping table above, at the request-construction
  level.
- Cost parsing from `usage.cost_usd` into a cost result, including
  fail-open behavior for missing or invalid values.
- The free-access-eligibility classification verdict, including that it
  does not rotate credentials or write a model/credential cooldown.
- The process-level model-catalog cache (dedupe, router ids first,
  per-(base URL, key) isolation, fail-open behavior).

Not yet verified by this suite: the OrcaRouter-Lite notes above (based on
OrcaRouter-Lite's source; not tested by this suite), the "Works today
without the plugin" custom-provider snippet (per OrcaRouter's integration
docs), and `ORCAROUTER_BASE_URL` end-to-end against a real endpoint (only
its declaration and its effect on cache keys are tested).

Not yet observed in a running Hermes installation (designed behavior,
pending end-to-end verification): the live install flow, chat/tool-call/
streaming against a real Hermes session, the fallback behavior described
above actually triggering in practice, and `hermes doctor`/`hermes plugins
validate` against an installed copy of this plugin.

## Credits

Builds on the approach in
[NousResearch/hermes-agent#30191](https://github.com/NousResearch/hermes-agent/pull/30191)
by [@zhenjunchen-png](https://github.com/zhenjunchen-png).

## License

Not yet specified. All rights reserved until a license is added.
