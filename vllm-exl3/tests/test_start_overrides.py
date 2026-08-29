#!/usr/bin/env python3
"""Regression test for caller overrides that must win over ``.env``."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# Caller exports must survive the `set -a; source .env` block. Every variable
# listed here needs a matching _cli_* save/restore pair in the start.sh preamble.
OVERRIDES = {
    "MAX_NUM_SEQS": ("2", "4"),
    "GLM53_MIXED_PREFILL_CHUNK": ("skip", "256"),
}


def test_override_wins(variable: str, env_value: str, dotenv_value: str) -> None:
    source = (ROOT / "start.sh").read_text()
    marker = "# ----------------------------- configuration -------------------------------"
    preamble, separator, _rest = source.partition(marker)
    assert separator, "start.sh configuration marker is missing"

    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        script = tmp / "start.sh"
        script.write_text(
            preamble
            + f'\nprintf "{variable}=%s\\n" "${{{variable}:-unset}}"\n'
        )
        script.chmod(0o755)
        (tmp / ".env").write_text(f"{variable}={dotenv_value}\n")

        env = os.environ.copy()
        env[variable] = env_value
        result = subprocess.run(
            ["bash", str(script)],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )

    assert result.stdout.strip() == f"{variable}={env_value}", (
        f"caller export {variable}={env_value} was clobbered by .env "
        f"({dotenv_value}) — add it to the _cli_* save/restore block"
    )


if __name__ == "__main__":
    for variable, (dotenv_value, env_value) in OVERRIDES.items():
        test_override_wins(variable, env_value, dotenv_value)
    print("start.sh caller override regression OK")
