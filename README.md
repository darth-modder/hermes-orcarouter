# hermes-orcarouter

> **Pre-release.** Not yet verified end to end inside a running Hermes Agent
> installation, and not yet listed in the Hermes plugin catalog.

This repository is a [Hermes Agent](https://github.com/NousResearch/hermes-agent)
model-provider plugin for [OrcaRouter](https://orcarouter.ai). The plugin
itself lives in the [`orcarouter/`](orcarouter/) subdirectory — see
**[orcarouter/README.md](orcarouter/README.md)** for what it does, how to
install it, and how to configure it.

Everything else at the repo root (this file, dev tooling, `tests/`) is
contributor/reviewer material, not part of what ships to a user's
machine — `hermes plugins install` only pulls the `orcarouter/`
subdirectory.

## Contributing

Dev tooling lives at the repo root so it's shared across the plugin surface
and its tests.

1. Set up a Hermes Agent checkout and point tests at it:

   ```bash
   export HERMES_AGENT_DIR=/path/to/hermes-agent
   ```

2. Install dev dependencies into a venv:

   ```bash
   pip install -r requirements-dev.txt
   ```

3. Run checks:

   ```bash
   # Hermetic discovery tests (REQUIRE_HERMES=1 turns a missing hermes-agent
   # checkout into a hard failure instead of a silent skip — use this in CI):
   REQUIRE_HERMES=1 pytest

   ruff check . --config ruff.toml
   ty check --python <path-to-venv> --extra-search-path "$HERMES_AGENT_DIR" .

   # Validate the plugin surface itself (not the whole repo):
   PYTHONPATH="$HERMES_AGENT_DIR" python -c \
     "from pathlib import Path; from hermes_cli.plugin_validate import validate_plugin_dir; \
      print(validate_plugin_dir(Path('orcarouter')).to_dict())"
   ```

## License

Not yet specified. All rights reserved until a license is added.
