#!/usr/bin/env python3
"""Launcher wiring for the agentic prefix-cache v2 bundle.

The four patch scripts are ported from the Apache-2.0 Reederey87 kit. This
test pins the local `start.sh` contract: host paths, preflight, worker copy,
both mounts, both rank scripts, env forwarding, default values and patch order.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "start.sh"
ENV = ROOT / ".env.example"

HOSTS = {
    "FGAPC_PATCH_HOST": "patch_fine_grained_apc.py",
    "APC_PER_GROUP_PATCH_HOST": "patch_apc_per_group_retention.py",
    "APC_NO_STORE_PATCH_HOST": "patch_apc_no_store.py",
    "PREFIX_CACHE_SPARSE_MISS_PATCH_HOST": "patch_prefix_cache_sparse_miss_metric.py",
}


def test_host_paths_and_files_exist() -> None:
    text = START.read_text()
    for var, name in HOSTS.items():
        assert f'{var}="${{{var}:-$SCRIPT_DIR/overlay/{name}}}"' in text
        assert f'[ -f "${var}" ] || die "${var} missing"' in text
        assert (ROOT / "overlay" / name).is_file()


def test_worker_copy_and_both_mounts() -> None:
    text = START.read_text()
    for _var, name in HOSTS.items():
        assert f"${{WORKER_SSH}}:/tmp/{name}" in text
        assert f"-v '/tmp/{name}:/opt/glm53/{name}:ro'" in text
        assert f':/opt/glm53/{name}:ro"' in text


def test_both_rank_scripts_run_each_patch_once() -> None:
    text = START.read_text()
    for _var, name in HOSTS.items():
        assert text.count(f"python3 /opt/glm53/{name}") == 2, name


def test_patch_order_is_pinned() -> None:
    text = START.read_text()
    order = [
        "patch_hybrid_prefix_hit.py",
        "patch_apc_per_group_retention.py",
        "patch_fine_grained_apc.py",
        "patch_prefix_cache_sparse_miss_metric.py",
        "patch_apc_no_store.py",
        "patch_xgrammar_termination.py",
    ]
    positions = [text.index(f"python3 /opt/glm53/{name}") for name in order]
    assert positions == sorted(positions)
    # Both rank scripts must use the same order.
    first = text.index("python3 /opt/glm53/patch_hybrid_prefix_hit.py")
    second = text.index(
        "python3 /opt/glm53/patch_hybrid_prefix_hit.py", first + 1
    )
    for name in order[1:]:
        assert text.index(f"python3 /opt/glm53/{name}", first) < second


def test_env_defaults_and_forwarding() -> None:
    text = START.read_text()
    assert 'GLM53_FINE_GRAINED_APC="${GLM53_FINE_GRAINED_APC:-0}"' in text
    assert (
        'GLM53_APC_PER_GROUP_RETENTION="${GLM53_APC_PER_GROUP_RETENTION:-1}"'
        in text
    )
    assert (
        'VLLM_PREFIX_CACHE_RETENTION_INTERVAL_SWA="${VLLM_PREFIX_CACHE_RETENTION_INTERVAL_SWA-}"'
        in text
    )
    assert 'GLM53_APC_NO_STORE="${GLM53_APC_NO_STORE:-1}"' in text
    assert (
        'GLM53_PREFIX_CACHE_SPARSE_MISS_METRIC="${GLM53_PREFIX_CACHE_SPARSE_MISS_METRIC:-1}"'
        in text
    )
    for knob in (
        "GLM53_FINE_GRAINED_APC",
        "GLM53_APC_PER_GROUP_RETENTION",
        "VLLM_PREFIX_CACHE_RETENTION_INTERVAL_SWA",
        "GLM53_APC_NO_STORE",
        "GLM53_PREFIX_CACHE_SPARSE_MISS_METRIC",
    ):
        assert f'-e "{knob}=' in text or f"-e {knob}=" in text


def test_dotenv_documents_the_bundle() -> None:
    text = ENV.read_text()
    assert "GLM53_FINE_GRAINED_APC=0" in text
    assert "VLLM_PREFIX_CACHE_RETENTION_INTERVAL_SWA=" in text
    assert "GLM53_APC_NO_STORE=1" in text
    assert "GLM53_PREFIX_CACHE_SPARSE_MISS_METRIC=1" in text


def test_kill_switches() -> None:
    per_group = (ROOT / "overlay" / "patch_apc_per_group_retention.py").read_text()
    assert 'GLM53_APC_PER_GROUP_RETENTION", "1"' in per_group
    fine = (ROOT / "overlay" / "patch_fine_grained_apc.py").read_text()
    assert 'GLM53_FINE_GRAINED_APC", "0"' in fine
    no_store = (ROOT / "overlay" / "patch_apc_no_store.py").read_text()
    assert "GLM53_APC_NO_STORE" in no_store
    sparse = (
        ROOT / "overlay" / "patch_prefix_cache_sparse_miss_metric.py"
    ).read_text()
    assert "GLM53_PREFIX_CACHE_SPARSE_MISS_METRIC" in sparse


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("prefix-cache v2 launcher wiring OK")


def test_per_group_kill_switch_is_a_noop() -> None:
    """GLM53_APC_PER_GROUP_RETENTION=0 must not touch the coordinator."""
    import os
    import subprocess
    import sys

    script = ROOT / "overlay" / "patch_apc_per_group_retention.py"
    env = os.environ.copy()
    env["GLM53_APC_PER_GROUP_RETENTION"] = "0"
    env["GLM53_KV_COORDINATOR_PY"] = "/nonexistent/kv_cache_coordinator.py"
    r = subprocess.run(
        [sys.executable, str(script)], env=env, capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr
    assert "skipping" in r.stdout
