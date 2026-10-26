#!/usr/bin/env python3
"""Static checks for the optional prefill warmup."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "boot-shape-warmup.sh"
START = ROOT / "start.sh"


def test_bash_syntax() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    subprocess.run(["bash", "-n", str(START)], check=True)


def test_prefill_function_and_env() -> None:
    text = SCRIPT.read_text()
    assert "prefill_warmup()" in text
    assert "GLM53_WARMUP_PREFILL_TOKENS" in text
    assert "reset_prefix_cache" in text
    assert "prefill_warmup \"$PREFILL_TOKENS\"" in text
    assert "EXPECTED_CHAT_REQUESTS=$((EXPECTED_CHAT_REQUESTS + 1))" in text


def test_launcher_forwards_it() -> None:
    text = START.read_text()
    assert 'GLM53_WARMUP_PREFILL_TOKENS="${GLM53_WARMUP_PREFILL_TOKENS:-0}"' in text
    assert 'GLM53_WARMUP_PREFILL_TOKENS="$GLM53_WARMUP_PREFILL_TOKENS"' in text
    env = (ROOT / ".env.example").read_text()
    assert "GLM53_WARMUP_PREFILL_TOKENS=0" in env


def test_tally_has_one_slot_for_the_prefill() -> None:
    text = SCRIPT.read_text()
    # The prefill result file must match the *-* tally glob.
    assert 'echo ok > "$tmpdir/prefill-1"' in text
    assert 'echo fail > "$tmpdir/prefill-1"' in text


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("prefill warmup static checks OK")
