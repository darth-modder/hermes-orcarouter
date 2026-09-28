"""CI guard: fail loudly when Hermes' own version resolution cannot identify
the checked-out release, instead of silently degrading to "unknown".

This matters because `hermes_cli.plugins_manifest.version_satisfies` treats
an unparseable *current* version as universally permissive: every
`requires_hermes` spec is reported satisfied. A CI job that checks out
Hermes without its tag history, then runs the requires_hermes gate check and
sees it pass, would be a false pass -- the gate never actually evaluated
anything. This step catches that case before the rest of the job runs.

Uses `hermes_cli.plugins_manifest.running_hermes_version()` -- the same
function the real `requires_hermes` gate calls -- rather than
`hermes_cli.version_info.get_version_info()` directly: that module does not
exist at every Hermes release this repo might be pointed at (it postdates
the v0.21.5 / v2026.9.24 tag ci.yml pins; at that release `__version__` is a
plain static string in hermes_cli/__init__.py with no version_info module at
all), so importing it directly breaks ci.yml despite working fine against
`main` in nightly.yml. `running_hermes_version()` is present at both.

Usage: python verify_hermes_version.py [expected_version]
  - With an argument, the resolved version must equal it exactly (used on a
    pinned release tag).
  - Without one, only "not unknown / not empty" is required (used on a
    moving target such as the `main` branch in the nightly workflow).
"""

from __future__ import annotations

import sys

from hermes_cli.plugins_manifest import running_hermes_version

expected = sys.argv[1].strip() if len(sys.argv) > 1 else ""

current = running_hermes_version()
print(f"Hermes version resolution: running_hermes_version()={current!r}")

if not current or current == "unknown":
    print(
        f"::error::Hermes version resolution reported {current!r}. This is "
        "the permissive-version-gate trap: an unparseable running version "
        "makes version_satisfies() return True for every requires_hermes "
        "spec, so the requires_hermes gate check later in this job would "
        "pass without ever evaluating anything. Check that hermes-agent was "
        "checked out with full tag history (fetch-depth: 0 / fetch-tags: "
        "true).",
        file=sys.stderr,
    )
    sys.exit(1)

if expected and current != expected:
    print(
        f"::error::Hermes version resolved to {current!r}, expected {expected!r}.",
        file=sys.stderr,
    )
    sys.exit(1)
