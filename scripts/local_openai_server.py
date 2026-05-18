"""Tiny local OpenAI-compatible chat-completions server.

This provides a localhost endpoint for scripts/run_baselines.py and
scripts/run_explanation_study.py without requiring vLLM.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def _build_prompt(messages: list[dict]) -> str:
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"[{role}]\n{content}")
    return "\n\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sshleifer/tiny-gpt2")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    class Handler(BaseHTTPRequestHandler):
        def _write_json(self, payload: dict, status: int = 200) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            if self.path in {"/", "/health"}:
                self._write_json({"status": "ok", "model": args.model})
                return
            self._write_json({"error": "not found"}, status=404)

        def do_POST(self):  # noqa: N802
            if self.path != "/v1/chat/completions":
                self._write_json({"error": "not found"}, status=404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            prompt = _build_prompt(body.get("messages", []))
            temperature = float(body.get("temperature", 0.0))
            max_new_tokens = int(body.get("max_tokens", args.max_new_tokens))
            inputs = tokenizer(prompt, return_tensors="pt")
            output = model.generate(
                **inputs,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
            text = tokenizer.decode(output[0], skip_special_tokens=True)
            if text.startswith(prompt):
                text = text[len(prompt):]
            text = text.strip() or "{}"
            self._write_json(
                {
                    "id": "chatcmpl-local",
                    "object": "chat.completion",
                    "model": body.get("model", args.model),
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": "stop",
                        }
                    ],
                }
            )

        def log_message(self, format, *args):  # noqa: A003
            return

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Serving local OpenAI-compatible endpoint on http://127.0.0.1:{args.port}/v1/chat/completions")
    print(f"Model: {args.model}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
