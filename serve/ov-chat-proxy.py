#!/usr/bin/env python
"""Minimal OpenAI-compatible /chat/completions proxy in front of OpenVINO Model Server.

WHY: OVMS (GenAI 2026.2) cannot load a chat template for some tokenizers (e.g.
qwen3_5 tri-modal/omni tokenizers) -- /v3/chat/completions fails with
"Chat template not loaded correctly". But /v3/completions (raw prompt) works fine.

This proxy accepts /v3/chat/completions, applies the chat template ITSELF, forwards
to OVMS /v3/completions, and reshapes the reply into a chat-completions response.
Normal models that template fine in OVMS don't need this -- it's only for the
templating-blocked ones.

  python ov-chat-proxy.py --upstream http://127.0.0.1:8000/v3 --port 8100

Then point any OpenAI client at http://127.0.0.1:8100/v3.
"""
from __future__ import annotations
import argparse, json, time, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ARGS = None


def build_chatml(messages, add_generation_prompt=True):
    """Standard Qwen ChatML. Adds a default system turn if none is present."""
    parts = []
    if not messages or messages[0].get("role") != "system":
        parts.append("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n")
    for m in messages:
        parts.append(f"<|im_start|>{m.get('role','user')}\n{m.get('content','')}<|im_end|>\n")
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
    return "".join(parts)


_TOK = None  # loaded HF tokenizer for faithful template rendering (or None)


def load_tokenizer(template_dir):
    """Load the HF tokenizer so we apply the model's REAL chat template (with
    tools / chat_template_kwargs) — exactly what OVMS's MINJA path would do. This
    is the only piece OVMS's serving path can't do for omni tokenizers; transformers
    handles it fine. Falls back to generic ChatML if unavailable."""
    global _TOK
    if not template_dir:
        return
    try:
        from transformers import AutoTokenizer
        t = AutoTokenizer.from_pretrained(template_dir, trust_remote_code=False)
        if getattr(t, "chat_template", None):
            _TOK = t
            print(f"[proxy] faithful templating ON (real chat_template from {template_dir})", flush=True)
        else:
            print(f"[proxy] tokenizer at {template_dir} has no chat_template; using generic ChatML", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[proxy] tokenizer load failed ({e}); using generic ChatML", flush=True)


def render_prompt(req):
    """Render the chat prompt. Prefer the model's real template (tools-aware);
    fall back to generic ChatML."""
    messages = req.get("messages", [])
    if _TOK is not None:
        try:
            kwargs = req.get("chat_template_kwargs") or {}
            tools = req.get("tools")
            return _TOK.apply_chat_template(
                messages, tools=tools, add_generation_prompt=True, tokenize=False, **kwargs
            )
        except Exception as e:  # noqa: BLE001
            print(f"[proxy] real-template render failed ({e}); falling back to ChatML", flush=True)
    return build_chatml(messages)


def upstream_completions(payload):
    req = urllib.request.Request(
        ARGS.upstream.rstrip("/") + "/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=ARGS.timeout) as r:
        return json.load(r)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        # passthrough /v3/models so clients can discover models
        if self.path.rstrip("/").endswith("/models"):
            try:
                with urllib.request.urlopen(ARGS.upstream.rstrip("/") + "/models", timeout=10) as r:
                    return self._send(200, json.load(r))
            except Exception as e:  # noqa: BLE001
                return self._send(502, {"error": str(e)})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        p = self.path.rstrip("/")
        # passthrough /completions verbatim (readiness probes + raw-prompt clients)
        if p.endswith("/completions") and not p.endswith("/chat/completions"):
            try:
                n = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(n)
                req = urllib.request.Request(ARGS.upstream.rstrip("/") + "/completions",
                                             data=body, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=ARGS.timeout) as r:
                    return self._send(200, json.load(r))
            except Exception as e:  # noqa: BLE001
                return self._send(502, {"error": str(e)})
        if not p.endswith("/chat/completions"):
            return self._send(404, {"error": "only /chat/completions and /completions are proxied"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:  # noqa: BLE001
            return self._send(400, {"error": f"bad json: {e}"})

        prompt = render_prompt(req)
        # forward generation params verbatim; swap messages->prompt
        fwd = {k: v for k, v in req.items() if k not in ("messages", "stream")}
        fwd["prompt"] = prompt
        try:
            up = upstream_completions(fwd)
        except Exception as e:  # noqa: BLE001
            return self._send(502, {"error": f"upstream /completions failed: {e}"})

        ch = (up.get("choices") or [{}])[0]
        text = ch.get("text", "")
        resp = {
            "id": up.get("id", "chatcmpl-proxy"),
            "object": "chat.completion",
            "created": up.get("created", int(time.time())),
            "model": req.get("model", up.get("model", "")),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": ch.get("finish_reason", "stop"),
            }],
            "usage": up.get("usage", {}),
        }
        self._send(200, resp)


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default="http://127.0.0.1:8000/v3", help="OVMS base (…/v3)")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--template-dir", default=None,
                    help="HF tokenizer dir for faithful chat templating (real template + tools). "
                         "Omit for generic ChatML.")
    ARGS = ap.parse_args()
    load_tokenizer(ARGS.template_dir)
    srv = ThreadingHTTPServer((ARGS.host, ARGS.port), Handler)
    print(f"chat-proxy on http://{ARGS.host}:{ARGS.port}/v3  ->  upstream {ARGS.upstream}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
