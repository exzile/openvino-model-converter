# transformers version matching (and other conversion traps)

The non-obvious knowledge behind why this tool exists. If you've hit a broken
OpenVINO export, the answer is probably here.

## 1. optimum's exporter must match the installed transformers

optimum-intel exports a model by registering a per-`model_type` **OpenVINO config**
and a **model patcher**. The patcher imports modeling internals directly from
`transformers.models.<arch>.modeling_<arch>` to rewrite the graph for export. Those
internals change between transformers releases, so a patcher written against
transformers X breaks on transformers Y.

optimum encodes the supported window as `MIN_TRANSFORMERS_VERSION` /
`MAX_TRANSFORMERS_VERSION` on each config class and **hard-raises** outside it:

```
ValueError: The current version of Transformers does not allow for the export
of the model. Maximum required is 5.2.*, got: 5.12.1
```

### Worked example: qwen3_5 (the case this tool was built for)

`empero-ai/Qwythos-9B-Claude-Mythos-5-1M` is a `qwen3_5` vision-language model.

- It was **saved with transformers 5.12.1** and only *loads* on transformers that
  know the `qwen3_5` architecture.
- optimum-intel 2.0.0's `Qwen3_5TextOpenVINOConfig` / `Qwen3_5ModelPatcher` target
  **transformers 5.2.x** (`MAX_TRANSFORMERS_VERSION = "5.2.99"`).
- On transformers 5.12.1 the patcher does
  `from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache`
  → **ImportError**. That custom hybrid linear-attention cache existed in 5.2.x but
  was removed by 5.12 (which uses the generic `DynamicCache`). The model genuinely
  has linear-attention layers (`linear_attn.A_log` / `conv1d` / `dt_bias`), so a
  name-only shim would pass the import and then produce a **broken graph**.

**The fix is a matched venv: transformers 5.2.0.** It both loads the 5.12-saved
checkpoint (the newer rope/YaRN config keys just warn) *and* satisfies the patcher
(`Qwen3_5DynamicCache` imports fine). That's why `Setup-Venvs.ps1` builds
`venv-qwen35` on transformers 5.2.0.

### The general rule

> When an export fails with an `ImportError` from a `transformers.models.*` module,
> or a `MIN/MAX_TRANSFORMERS_VERSION` `ValueError`, you have a version mismatch.
> Find the transformers version optimum's patcher targets and pin a venv to it.

`-BypassVersionCeiling` relaxes the MIN/MAX guard at runtime. Use it only to *try*
a near-miss, and **always generation-probe the result** — bypassing the guard turns
a clean error into a silently wrong graph.

### A note on optimum-intel 2.0.0's metadata

PyPI `optimum-intel==2.0.0` declares `transformers<5.1` in its metadata, yet its
code ships `qwen3_5` / `gemma4` exporter configs that require transformers ≥5.2.
The metadata pin is stale. That's why the venv setup installs the stack first
(pulling whatever transformers the metadata allows) and then force-installs the
real target transformers with `--no-deps`.

## 2. `main_export()` silently produces a full-precision IR

Calling optimum's `main_export(..., ov_config=OVConfig(quantization_config=...))`
**does not run weight compression**. The `optimum-cli` command runs a *separate*
`_main_quantize` step that the programmatic entrypoint skips, so a direct
`main_export()` call gives you an FP16/FP32 IR (e.g. 21 GB instead of ~5.6 GB for a
9B model) with no error.

**This tool shells out to `optimum-cli export openvino --weight-format int4 …`**
to get real INT4. To detect the trap yourself:

```bash
grep -c 'element_type="i4"' openvino_model.xml   # >0 means real INT4
```

A proper INT4 IR also shows this in the nncf log:
`int4_sym, group size 128 | 100% ratio-defining params`.

## 3. Weightless IRs

Some community OpenVINO uploads ship `openvino_model.xml` (the graph) **without**
`openvino_model.bin` (the weights) — not loadable. This tool validates that every
`.xml` has a non-trivial `.bin` beside it, recursing into multimodal sub-IRs.

## 4. Vision-language → text-only requires decoder extraction

optimum registers a VL `model_type` (e.g. `qwen3_5`) **only** for the
`image-text-to-text` task — there is no one-command text-only export. The `decoder`
shape:

1. loads the VL model,
2. locates the `language_model` submodule + `lm_head`,
3. grafts them into a standalone CausalLM built from the model's `text_config`,
4. saves that as a text checkpoint,
5. runs the normal `text-generation-with-past` export on it.

For Qwythos this grafted 427/427 tensors cleanly. If your VL model's text decoder
uses an unregistered `model_type` (e.g. `gemma4_unified_text`), the text export
will still need an optimum config for that type — extraction alone isn't enough.
