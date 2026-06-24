# ov-chat-proxy — OVMS `/chat/completions` bridge

A tiny OpenAI-compatible `/chat/completions` proxy that sits in front of OpenVINO
Model Server (OVMS) and applies the chat template itself, then forwards to OVMS
`/v3/completions`.

## Why this exists

OVMS's `_python_on` builds apply chat templates in the serving layer via an
embedded **Python Jinja2** sandbox. For some tokenizers — notably text decoders
extracted from **tri-modal / omni** models (e.g. `qwen3_5`) — that path fails to
load the template and `/v3/chat/completions` returns:

```
Error: Chat template not loaded correctly, so it cannot be applied
```

`/v3/completions` (raw prompt) on the **same** served model works fine — only the
chat-template step is broken. OVMS has a `chat_template_mode: MINJA` option (in
`model_server` `main`) that routes through GenAI's working template engine, but
it isn't in released binaries yet. Until it ships, this proxy does the same job:
apply the template, hit `/completions`.

Upstream tracking: openvinotoolkit/model_server#4322.

## What it does

- Accepts `POST /v3/chat/completions`, renders the prompt, forwards to OVMS
  `/v3/completions`, reshapes the reply into a `chat.completion` response.
- **Faithful templating** (`--template-dir`): applies the model's *real* chat
  template via `transformers.apply_chat_template` (tools- and
  `chat_template_kwargs`-aware) — exactly what fixed-OVMS (`MINJA`) would do.
  Without it, falls back to generic Qwen ChatML.
- Passes through `POST /v3/completions` and `GET /v3/models` unchanged.

## Usage

```bash
# generic ChatML (no extra deps beyond Python stdlib):
python serve/ov-chat-proxy.py --upstream http://127.0.0.1:8000/v3 --port 8100

# faithful templating (needs `transformers`; point at the model's tokenizer dir):
python serve/ov-chat-proxy.py --port 8100 \
    --template-dir /path/to/source-checkpoint-with-tokenizer
```

Then point any OpenAI client at `http://127.0.0.1:8100/v3`. The proxy forwards to
whatever model is loaded on the upstream OVMS port, so load your model on OVMS
first (e.g. `:8000`).

## Notes

- This is a bridge, not a replacement — OVMS still does all inference. Only models
  whose chat template OVMS can't load need to go through it; everything else can
  talk to OVMS directly.
- It does not stream (non-`stream` responses only). Fine for evals/most clients.
- Tool-calling fidelity depends on the model's own template — if the template
  doesn't inject tools, the model won't emit `tool_call`s (same as fixed-OVMS).
