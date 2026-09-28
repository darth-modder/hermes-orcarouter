"""CI guard: this plugin's own `requires_hermes` bound must actually be
satisfied by the Hermes version this job just resolved. Exercised through
Hermes' own `running_hermes_version()` / `version_satisfies()` (the same
functions core's plugin loader uses), never reimplemented here.

Usage: python check_requires_hermes.py <plugin_yaml_path>
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml
from hermes_cli.plugins_manifest import running_hermes_version, version_satisfies

manifest_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("orcarouter/plugin.yaml")
manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
spec = manifest["requires_hermes"]
current = running_hermes_version()
satisfied = version_satisfies(spec, current)
print(f"requires_hermes={spec!r} running_hermes_version={current!r} -> version_satisfies={satisfied}")

if not satisfied:
    print(
        f"::error::version_satisfies({spec!r}, {current!r}) is False -- this "
        f"plugin's requires_hermes bound in {manifest_path} rejects the "
        "Hermes version this job resolved.",
        file=sys.stderr,
    )
    sys.exit(1)
