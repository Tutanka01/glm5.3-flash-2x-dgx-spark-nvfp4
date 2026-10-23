#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class Handler(BaseHTTPRequestHandler):
    # Every Nth tools request returns blank arguments (issue #10 rehearsal).
    blank_tool_args_every = 0
    _tool_call_count = 0

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def send_json(self, payload: dict) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self.send_json({"status": "ok"})
            return
        if self.path == "/v1/models":
            self.send_json(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "glm-5.3-flash-nvfp4",
                            "object": "model",
                            "max_model_len": 32768,
                        }
                    ],
                }
            )
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length)) if length else {}
        if self.path in ("/flush_cache", "/v1/flush_cache"):
            self.send_json({"success": True})
            return
        if self.path in ("/v1/tokenize", "/tokenize"):
            text = payload.get("prompt") or payload.get("input") or payload.get("text") or ""
            self.send_json({"count": len(str(text).split())})
            return
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        if payload.get("stream"):
            messages = payload.get("messages") or []
            long_context = any(
                "GLM53_LONG_CONTEXT_CAPABILITY_TEST" in str(item.get("content", ""))
                for item in messages
                if isinstance(item, dict)
            )
            pieces = (
                ["ALPHA_7319|", "MIDDLE_2846|", "OMEGA_9052"]
                if long_context
                else ["BENCH", "_OK"]
            )
            events = [
                *({"choices": [{"delta": {"content": piece}}]} for piece in pieces),
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": sum(
                            len(str(item.get("content", "")).split())
                            for item in messages
                            if isinstance(item, dict)
                        )
                        or 12,
                        "completion_tokens": len(pieces),
                    },
                },
            ]
            body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
            body += "data: [DONE]\n\n"
            encoded = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
            return
        if payload.get("tools"):
            message = self.tool_call_message(payload)
        else:
            message = {"role": "assistant", "content": "GLM53_OK"}
        self.send_json(
            {
                "choices": [{"message": message, "finish_reason": "stop", "index": 0}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            }
        )


    @classmethod
    def tool_call_message(cls, payload: dict) -> dict:
        """First offered tool, required args filled from its own schema.

        Legacy compatibility: smoke-glm53.sh asserts the get_temperature call
        echoes the city it asked for, so that tool keeps `{"city": "Paris"}`.
        Other tools get `mock_<key>` placeholders. With blank_tool_args_every
        > 0, every Nth tools request comes back with `arguments: ""` so
        clients can rehearse the issue-#10 mitigation (validation + retry).
        """
        tools = payload.get("tools") or []
        name, required = "get_temperature", ["city"]
        if tools:
            fn = tools[0].get("function") or {}
            name = fn.get("name") or name
            params = fn.get("parameters") or {}
            required = params.get("required") or list(params.get("properties") or {})[:1]
        cls._tool_call_count += 1
        if cls.blank_tool_args_every and cls._tool_call_count % cls.blank_tool_args_every == 0:
            arguments = ""
        elif name == "get_temperature" and required == ["city"]:
            arguments = json.dumps({"city": "Paris"})
        else:
            arguments = json.dumps({key: f"mock_{key}" for key in required})
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call_{cls._tool_call_count}",
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            ],
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--port-file", type=Path, required=True)
    parser.add_argument(
        "--blank-tool-args-every",
        type=int,
        default=0,
        help="every Nth tools request returns blank arguments (0 = off)",
    )
    args = parser.parse_args()
    Handler.blank_tool_args_every = args.blank_tool_args_every
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    args.port_file.write_text(str(server.server_port), encoding="utf-8")
    server.serve_forever()


if __name__ == "__main__":
    main()
