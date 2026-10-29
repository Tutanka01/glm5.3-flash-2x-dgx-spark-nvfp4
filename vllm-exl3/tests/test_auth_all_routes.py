#!/usr/bin/env python3
"""Regression tests for the VLLM_API_KEY every-route auth patch."""
from __future__ import annotations

import asyncio
import hashlib
import os
import secrets
import subprocess
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PATCH = next(
    p
    for p in (
        HERE / "patch_auth_all_routes.py",
        ROOT / "overlay" / "patch_auth_all_routes.py",
    )
    if p.is_file()
)
INSTALLED_SERVER_UTILS = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/serve/utils/"
    "server_utils.py"
)
MARK = "# [glm53-auth-all-routes]"
KEY = "s3cret-key"

# Exact vLLM 487ecf187 GUARDED_PREFIX + AuthenticationMiddleware, embedded in
# a dependency-free harness (fake Headers / JSONResponse) so the patched
# source can be exec'd and its routing decisions asserted.
PINNED_SERVER_UTILS_FIXTURE = '''

GUARDED_PREFIX = ("/v1", "/v2", "/inference", "/cohere")


class AuthenticationMiddleware:
    """
    Pure ASGI middleware that authenticates each request by checking
    if the Authorization Bearer token exists and equals anyof "{api_key}".

    Notes
    -----
    There are two cases in which authentication is skipped:
        1. The HTTP method is OPTIONS.
        2. The request path doesn't start with GUARDED_PREFIX (e.g. /health).
    """

    def __init__(self, app: ASGIApp, tokens: list[str]) -> None:
        self.app = app
        self.api_tokens = [hashlib.sha256(t.encode("utf-8")).digest() for t in tokens]

    def verify_token(self, headers: Headers) -> bool:
        authorization_header_value = headers.get("Authorization")
        if not authorization_header_value:
            return False

        scheme, _, param = authorization_header_value.partition(" ")
        if scheme.lower() != "bearer":
            return False

        param_hash = hashlib.sha256(param.encode("utf-8")).digest()

        token_match = False
        for token_hash in self.api_tokens:
            token_match |= secrets.compare_digest(param_hash, token_hash)

        return token_match

    def __call__(self, scope: Scope, receive: Receive, send: Send) -> Awaitable[None]:
        if (
            scope["type"] not in ("http", "websocket")
            or scope.get("method") == "OPTIONS"
        ):
            # scope["type"] can be "lifespan" or "startup" for example,
            # in which case we don't need to do anything
            return self.app(scope, receive, send)
        root_path = scope.get("root_path", "")
        url_path = scope["path"].removeprefix(root_path)
        headers = Headers(scope=scope)
        # Type narrow to satisfy mypy.
        if url_path.startswith(GUARDED_PREFIX) and not self.verify_token(headers):
            response = JSONResponse(content={"error": "Unauthorized"}, status_code=401)
            return response(scope, receive, send)
        return self.app(scope, receive, send)
'''


class FakeHeaders:
    def __init__(self, scope: dict) -> None:
        self._h = {k.lower(): v for k, v in scope.get("headers", {}).items()}

    def get(self, key: str):
        return self._h.get(key.lower())


class FakeJSONResponse:
    def __init__(self, content: dict, status_code: int) -> None:
        self.status_code = status_code

    async def __call__(self, scope, receive, send) -> None:
        await send(self.status_code)


def run_patch(target: Path, *, ok: bool = True) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["GLM53_SERVER_UTILS_PY"] = str(target)
    proc = subprocess.run(
        [sys.executable, str(PATCH)],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if ok and proc.returncode != 0:
        raise AssertionError(proc.stdout + proc.stderr)
    if not ok and proc.returncode == 0:
        raise AssertionError("patch unexpectedly accepted a drifted/partial target")
    return proc


def middleware_from(source: str):
    namespace: dict[str, object] = {
        "hashlib": hashlib,
        "secrets": secrets,
        "Headers": FakeHeaders,
        "JSONResponse": FakeJSONResponse,
        "ASGIApp": object,
        "Scope": dict,
        "Receive": object,
        "Send": object,
        "Awaitable": object,
    }
    exec(compile(source, "patched_server_utils_fixture.py", "exec"), namespace)

    async def app(scope, receive, send) -> None:
        await send(200)

    return namespace["AuthenticationMiddleware"](app, tokens=[KEY])


def status(mw, path: str, *, bearer: str | None = None, method: str = "POST",
           kind: str = "http", root_path: str = "") -> int:
    headers = {"Authorization": f"Bearer {bearer}"} if bearer is not None else {}
    scope = {"type": kind, "method": method, "path": path,
             "root_path": root_path, "headers": headers}
    seen: list[int] = []

    async def send(code: int) -> None:
        seen.append(code)

    asyncio.run(mw(scope, None, send))
    assert len(seen) == 1, seen
    return seen[0]


def test_fixture() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "server_utils.py"
        target.write_text(PINNED_SERVER_UTILS_FIXTURE)

        # Stock behavior, pinned: root-mounted generation bypasses the key.
        stock = middleware_from(PINNED_SERVER_UTILS_FIXTURE)
        assert status(stock, "/invocations") == 200
        assert status(stock, "/v1/chat/completions") == 401

        run_patch(target)
        patched = target.read_text()
        assert patched.count(MARK) == 1
        assert "url_path.startswith(GUARDED_PREFIX)" not in patched
        compile(patched, "patched_fixture.py", "exec")

        mw = middleware_from(patched)
        for path in ("/v1/chat/completions", "/v1/models", "/v1/completions",
                     "/invocations", "/tokenize", "/detokenize",
                     "/reset_prefix_cache", "/reset_mm_cache", "/docs",
                     "/openapi.json", "/ping", "/version", "/", "/healthz",
                     "/health/../invocations"):
            assert status(mw, path) == 401, path
            assert status(mw, path, bearer="wrong") == 401, path
            assert status(mw, path, bearer=KEY) == 200, path
        # Keyless probes stay open for the launcher's health wait and scrapers.
        for path in ("/health", "/metrics", "/metrics/"):
            assert status(mw, path, method="GET") == 200, path
        # Upstream bypasses kept: CORS preflight and non-HTTP scopes.
        assert status(mw, "/v1/chat/completions", method="OPTIONS") == 200
        assert status(mw, "/", kind="lifespan") == 200
        # Websocket (realtime) still needs the bearer.
        assert status(mw, "/v1/realtime", kind="websocket", method=None) == 401
        # root_path is stripped before matching, as upstream.
        assert status(mw, "/api/health", root_path="/api") == 200
        assert status(mw, "/api/invocations", root_path="/api") == 401

        run_patch(target)
        assert target.read_text() == patched


def test_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "server_utils.py"

        drifted = PINNED_SERVER_UTILS_FIXTURE.replace(
            'GUARDED_PREFIX = ("/v1", "/v2", "/inference", "/cohere")',
            'GUARDED_PREFIX = ("/v1", "/v2")',
            1,
        )
        target.write_text(drifted)
        run_patch(target, ok=False)
        assert target.read_text() == drifted

        partial = PINNED_SERVER_UTILS_FIXTURE.replace(
            "\nGUARDED_PREFIX", f"\n{MARK}\nGUARDED_PREFIX", 1
        )
        target.write_text(partial)
        run_patch(target, ok=False)
        assert target.read_text() == partial

        duplicated = PINNED_SERVER_UTILS_FIXTURE + PINNED_SERVER_UTILS_FIXTURE
        target.write_text(duplicated)
        run_patch(target, ok=False)
        assert target.read_text() == duplicated


def test_installed_copy_if_present() -> None:
    source = Path(os.environ.get("GLM53_SERVER_UTILS_PY_SRC", INSTALLED_SERVER_UTILS))
    if not source.is_file():
        return
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "server_utils.py"
        target.write_bytes(source.read_bytes())
        run_patch(target)
        patched = target.read_text()
        compile(patched, str(target), "exec")
        assert patched.count(MARK) == 1
        assert "url_path not in GLM53_AUTH_OPEN_PATHS" in patched
        run_patch(target)


def test_recipe_wiring_if_present() -> None:
    dockerfile = ROOT / "Dockerfile"
    launchers = [ROOT / "start.sh", ROOT / "start-tp4.sh"]
    if not dockerfile.is_file() or not all(p.is_file() for p in launchers):
        return
    for launcher_path in launchers:
        launcher = launcher_path.read_text()
        assert 'AUTH_PATCH_HOST="${AUTH_PATCH_HOST:-' in launcher, launcher_path
        # head only: workers run --headless and serve no HTTP API
        assert launcher.count("python3 /opt/glm53/patch_auth_all_routes.py") == 1
        assert (
            '-v "$AUTH_PATCH_HOST:'
            '/opt/glm53/patch_auth_all_routes.py:ro"' in launcher
        ), launcher_path
    image = dockerfile.read_text()
    assert "COPY overlay/patch_auth_all_routes.py" in image
    assert "RUN python3 /opt/glm53/patch_auth_all_routes.py" in image
    assert "python3 /opt/glm53/test_auth_all_routes.py" in image


def main() -> int:
    test_fixture()
    test_fail_closed()
    test_installed_copy_if_present()
    test_recipe_wiring_if_present()
    print("auth-all-routes patch OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
