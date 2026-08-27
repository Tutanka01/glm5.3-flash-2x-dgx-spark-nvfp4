#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class Handler(BaseHTTPRequestHandler):
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
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_temperature",
                            "arguments": json.dumps({"city": "Paris"}),
                        },
                    }
                ],
            }
        else:
            message = {"role": "assistant", "content": "GLM53_OK"}
        self.send_json(
            {
                "choices": [{"message": message, "finish_reason": "stop", "index": 0}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--port-file", type=Path, required=True)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    args.port_file.write_text(str(server.server_port), encoding="utf-8")
    server.serve_forever()


if __name__ == "__main__":
    main()
