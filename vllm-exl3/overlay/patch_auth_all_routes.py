#!/usr/bin/env python3
"""Make ``VLLM_API_KEY`` guard every API route, not only ``/v1``.

This vLLM build (0.1.dev20051+g487ecf187) installs
``AuthenticationMiddleware`` when ``VLLM_API_KEY`` / ``--api-key`` is set,
but the middleware only checks paths under ``GUARDED_PREFIX = ("/v1",
"/v2", "/inference", "/cohere")``. Root-mounted routes answer without a
bearer, among them:

* ``POST /invocations`` (SageMaker router, always mounted) -- dispatches to
  chat / completions, so the model generates for anyone on the LAN;
* ``/reset_prefix_cache`` et al. (``GLM53_EXPOSE_CACHE_RESET=1``, the lane
  default) -- flushes serving state;
* ``/tokenize``, ``/detokenize``, ``/docs``, ``/openapi.json``.

This patch flips the check to default-deny: with a key set, every path
except the keyless probes in ``GLM53_AUTH_OPEN_PATHS`` (``/health``,
``/metrics``) needs ``Authorization: Bearer <key>``. Without a key the
middleware is never installed, so the open API is byte-for-byte stock.
OPTIONS (CORS preflight) and non-HTTP scopes keep their upstream bypass.

Fail closed if the vLLM anchors drift.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

P = Path(
    os.environ.get(
        "GLM53_SERVER_UTILS_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/serve/utils/"
        "server_utils.py",
    )
)
MARK = "# [glm53-auth-all-routes]"

PREFIX_OLD = """GUARDED_PREFIX = ("/v1", "/v2", "/inference", "/cohere")
"""

PREFIX_NEW = """GUARDED_PREFIX = ("/v1", "/v2", "/inference", "/cohere")
# [glm53-auth-all-routes] Keyless probes; every other path needs the bearer
# once an API key is configured (root-mounted /invocations generates too).
GLM53_AUTH_OPEN_PATHS = frozenset({"/health", "/metrics", "/metrics/"})
"""

CHECK_OLD = """        if url_path.startswith(GUARDED_PREFIX) and not self.verify_token(headers):
"""

CHECK_NEW = """        if url_path not in GLM53_AUTH_OPEN_PATHS and not self.verify_token(headers):
"""


def main() -> int:
    if not P.is_file():
        raise SystemExit(f"missing {P}")
    text = P.read_text()
    counts = {
        "prefix_old": text.count(PREFIX_OLD),
        "prefix_new": text.count(PREFIX_NEW),
        "check_old": text.count(CHECK_OLD),
        "check_new": text.count(CHECK_NEW),
        "marker": text.count(MARK),
    }
    if (counts["prefix_new"], counts["check_new"], counts["check_old"],
            counts["marker"]) == (1, 1, 0, 1):
        print(f"{P.name}: auth-all-routes already present — skipping")
        return 0
    if counts["marker"] or counts["prefix_new"] or counts["check_new"]:
        raise SystemExit(
            f"{P}: partial/inconsistent auth-all-routes patch ({counts})"
        )
    if counts["prefix_old"] != 1 or counts["check_old"] != 1:
        raise SystemExit(
            f"{P}: expected exactly one pristine GUARDED_PREFIX and auth "
            f"check, found {counts}"
        )
    text = text.replace(PREFIX_OLD, PREFIX_NEW, 1).replace(CHECK_OLD, CHECK_NEW, 1)
    if (text.count(PREFIX_NEW) != 1 or text.count(CHECK_NEW) != 1
            or text.count(CHECK_OLD) != 0 or text.count(MARK) != 1):
        raise SystemExit(f"{P}: auth-all-routes post-patch verification failed")
    compile(text, str(P), "exec")
    P.write_text(text)
    print(f"patched {P.name} (API key guards every route except "
          "/health and /metrics)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
