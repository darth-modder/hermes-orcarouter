"""CI guard: `hermes plugins validate` must be clean -- ok=True and an empty
warnings list -- before this plugin can ship. Runs the same
`validate_plugin_dir` function the real `hermes plugins validate` CLI command
calls, so a warning here is exactly what a user installing this plugin would
see.

Usage: python validate_plugin.py <plugin_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

from hermes_cli.plugin_validate import validate_plugin_dir

plugin_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("orcarouter")
report = validate_plugin_dir(plugin_dir)
print(f"validate_plugin_dir({plugin_dir}): ok={report.ok} warnings={report.warnings}")

if not report.ok or report.warnings:
    print(
        f"::error::validate_plugin_dir({plugin_dir}) did not return a clean "
        f"result (ok={report.ok}, warnings={report.warnings}).",
        file=sys.stderr,
    )
    sys.exit(1)
