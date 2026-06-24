# 🔁 openvino-model-converter

> Convert Hugging Face models to **INT4 OpenVINO IRs** that load and serve on Intel
> hardware (CPU / Arc GPU via OVMS, OpenVINO GenAI, or `optimum-intel`).

![License](https://img.shields.io/badge/license-Apache--2.0-blue)
![OpenVINO](https://img.shields.io/badge/OpenVINO-2026.2-purple)
![Tested on](https://img.shields.io/badge/tested-Intel%20Arc%20B70-0071C5)
![Python](https://img.shields.io/badge/python-3.12-green)

A small, reproducible toolkit around `optimum-intel` that handles the parts the
one-line `optimum-cli` examples don't:

- 🖼️ **Vision-language / omni models** — extracts the text decoder into a standalone
  checkpoint so you get a text-only IR your `/chat/completions` stack can serve
  (the full multimodal export is supported too).
- 🧩 **transformers version matching** — different model families need optimum's
  exporter to match a specific transformers version, or the export fails or
  silently produces a broken graph. The bootstrap script pins the right versions.
- 🪶 **Real INT4 quantization** — a programmatic `main_export()` call *silently skips*
  weight compression and yields a full-precision IR; this shells out to the CLI so
  you actually get INT4.
- ✅ **Validation built in** — every IR is checked for present, non-trivial weights
  (the "weightless IR" trap), recursing into multimodal sub-IRs.
- 🚀 **OVMS-ready output** — emits the `graph.pbtxt` + chat-template `rt_info` so the
  IR serves directly, no hand-wiring.

> Verified end-to-end converting `empero-ai/Qwythos-9B-Claude-Mythos-5-1M` (a
> `qwen3_5` vision-language model) to a 5.6 GB INT4 text-decoder IR that loads and
> generates under OpenVINO.

---

## 🤔 Should I convert, or just serve the GGUF?

OVMS 2026.2.1 can serve some GGUFs **directly** (no conversion). So the honest
question is: *when is converting worth it?* We benchmarked both paths head-to-head
on the same model and same Arc GPU — here's the decision in one table.

| Convert to OpenVINO IR when… | Serve the GGUF directly when… |
|---|---|
| 🧠 Your arch/quant **isn't** in OVMS's GGUF allow-list (gemma, mistral, phi, q8, …) | 🏃 It's a mainstream arch + common quant OVMS can already pull |
| 🖼️ You need a **vision/omni model as a text decoder** | 🧪 You just want to **try** a model in minutes |
| 🏭 It's a **production / always-on** model on Intel hardware | 📦 You value portability (GGUF also runs anywhere via llama.cpp) |
| ⚡ You want max efficiency + OVMS-ready output with no footguns | |

**One-liner:** convert when the GGUF-direct path *can't* serve your model — or when
you want maximum efficiency and reliability on the OpenVINO/Arc stack. For a
mainstream model at a common quant, GGUF-direct is simpler and nearly as fast.

### 📊 Benchmark: INT4 IR vs q4_0 GGUF

> Qwen2.5-7B-Instruct · Intel Arc B70 · OVMS 2026.2.1 · `/v3/completions` · temp 0 · 300 tokens · best of 3

| Config | Device | Throughput | VRAM | Disk |
|---|---|---|---|---|
| 🥇 **INT4 IR** (this tool) | GPU | **109.6 tok/s** | **7,869 MB** | 4,250 MB |
| 🥈 **GGUF q4_0** | GPU | 94.1 tok/s | 8,098 MB | 4,226 MB |
| ⚠️ GGUF q4_0 *(as pulled)* | **CPU** | 10.2 tok/s | — | — |

**Takeaways:**

- ⚡ **+16% throughput** and **~230 MB less VRAM** for the converted IR at 7B. Disk is a wash.
- 📉 The gap is **scale-dependent** — at 0.5B the two are a tie (decode isn't yet
  memory-bandwidth-bound). The IR's edge shows at 7B+ where weight bandwidth dominates.
- 🧠 **Efficiency alone is not the headline.** The bigger wins are *coverage* (serving
  models GGUF-direct can't load) and *reliability* on Intel hardware — see the table above.

> ⚠️ **GGUF footgun:** OVMS's GGUF pull writes `graph.pbtxt` with `device: "CPU"` by
> default. Served as-pulled, the 7B silently ran at **10 tok/s on CPU** (~10× slower),
> no error. You must edit `device: "GPU"` yourself. A converted IR served the normal
> way avoids this (and the single-file `models_path: "./"` GGUF load failure).

---

## 📦 Install

Requires Python 3.12 and a recent pip. One-time venv setup (creates `./venvs/`):

```powershell
.\scripts\Setup-Venvs.ps1
```

This builds two venvs with an identical OpenVINO/optimum/nncf/torch stack but
different pinned transformers — see [Why two venvs?](#-why-two-venvs) below.

## 🔧 Convert

```powershell
# Standard text LLM -> INT4 IR (uses venv-standard)
.\Convert-Model.ps1 -Model Qwen/Qwen2.5-Coder-14B-Instruct -Shape text

# A qwen3_5 vision-language model -> BOTH a full multimodal IR and an
# extracted text-only IR (uses venv-qwen35, transformers 5.2.0)
.\Convert-Model.ps1 -Model empero-ai/Qwythos-9B-Claude-Mythos-5-1M -Shape both -Venv qwen35
```

IRs land in `./model-repo/Converted/`. Override with `-OutRoot` or `$env:OVCONV_OUT`.

You can also call the engine directly (any Python env with the deps):

```bash
python convert_model.py --model <hf-repo> --shape text --weight-format int4 --sym --ratio 1.0 --group-size 128
```

## 🧬 Shapes

| `-Shape`     | optimum task                  | output dir          | text-serveable? |
|--------------|-------------------------------|---------------------|-----------------|
| `text`       | `text-generation-with-past`   | `<name>`            | yes             |
| `multimodal` | `image-text-to-text`          | `<name>-multimodal` | no (VL pipeline)|
| `decoder`    | extract LM submodule → text   | `<name>-text`       | yes             |
| `both`       | multimodal + decoder          | both                | the `-text` one |
| `auto`       | text for LLMs, both for VL    | —                   | —               |

A vision-language checkpoint can't be exported text-only in one command — optimum
registers the VL `model_type` only for `image-text-to-text`. The `decoder` shape
loads the VL model, grafts its `language_model` + `lm_head` into a standalone
CausalLM built from `text_config`, saves it, then runs the text export.

## 🗂️ Output layout

```
model-repo/Converted/
  <name>             text IR              (--shape text)
  <name>-multimodal  full VL IR           (--shape multimodal)
  <name>-text        extracted-decoder IR (--shape decoder)
  _src_<name>        downloaded source    (reuse with --src-dir)
  _work_<name>       scratch
```

A valid IR has **both** `openvino_model.xml` and a non-trivial `openvino_model.bin`.
The engine checks this automatically. To verify INT4 by hand, grep the `.xml` for
`element_type="i4"` (a full-precision IR has only `f16`/`f32`).

## 🖥️ Serving on OpenVINO Model Server

optimum produces a loadable IR but not an OVMS-servable one. For the text-serveable
shapes (`text`, `decoder`) the tool runs a `finalize_for_ovms` step that adds the
two missing pieces — a model-agnostic `graph.pbtxt` and a `simplified_chat_template`
in the tokenizer's `rt_info` (which OVMS reads for `/chat/completions`). Set the
device with `--ovms-device GPU|CPU|NPU`, or skip the step with `--no-ovms-finalize`.

> ⚠️ Caveat: OVMS's `_python_on` builds can't load chat templates for some tri-modal/omni
> tokenizers (e.g. `qwen3_5` omni). Those IRs still serve via `/v3/completions`;
> only `/chat/completions` is affected (upstream: openvinotoolkit/model_server#4322).
> See [docs/version-matching.md](docs/version-matching.md#5-serving-on-openvino-model-server-ovms).
> Until the upstream fix ships, **[serve/ov-chat-proxy.py](serve/README.md)** bridges
> `/chat/completions` by applying the template itself and forwarding to `/completions`.

### Serving a GGUF directly (for comparison)

OVMS 2026.2.1 can also serve a GGUF without conversion — useful for quick trials.
Supported (read from the GenAI runtime):

- **Architectures:** `gpt2`, `llama`, `qwen2`, `qwen3`, `qwen3_5`, `qwen3_5_moe`, `qwen3_vl`, `smollm`
- **Quant types:** `q4_0`, `q4_k`, `q5_k`, `q6_k` (no `q8_0`; avoid mixed `*_K_M` — they can fail the tensor loader)

```bash
ovms --pull --source_model Qwen/Qwen2.5-7B-Instruct-GGUF \
     --gguf_filename qwen2.5-7b-instruct-q4_0-00001-of-00002.gguf \
     --model_repository_path ./repo --model_name qwen7b --task text_generation
```

> ⚠️ After pulling, **check `repo/.../graph.pbtxt`** — set `device: "GPU"` (the pull
> defaults to `CPU`), and for a single-file GGUF set `models_path` to the `.gguf`
> filename (the auto-generated `"./"` fails to load). Anything outside the allow-lists
> above → convert with this tool instead.

## 🧪 Why two venvs?

optimum-intel's OpenVINO exporter has a per-model "model patcher" that imports
modeling internals from `transformers`. Those internals **drift across transformers
releases**, so the exporter config must match the installed transformers version:

- **`venv-standard`** (transformers 5.12.1) — most text LLMs.
- **`venv-qwen35`** (transformers 5.2.0) — `qwen3_5` / vision-language. optimum's
  `Qwen3_5ModelPatcher` imports `Qwen3_5DynamicCache`, which existed in transformers
  5.2.x but was **removed in 5.12** (replaced by the generic `DynamicCache`). On
  5.12 the export dies with `ImportError: cannot import name 'Qwen3_5DynamicCache'`.
  transformers 5.2.0 both loads the (5.12-saved) checkpoint *and* satisfies the
  patcher.

If a model hits a version ceiling you can't satisfy, `-BypassVersionCeiling` relaxes
optimum's guard — **but then you must generation-probe the IR**, because a real
internals mismatch produces a silently broken graph instead of a clean error.

See [docs/version-matching.md](docs/version-matching.md) for the full story.

## 🪤 Known traps (baked into the tooling)

1. **`main_export()` silently skips INT4.** Calling optimum's `main_export()`
   programmatically does *not* run the separate weight-compression step the CLI
   does — you get a full-precision IR. This tool shells out to
   `optimum-cli export openvino --weight-format int4 …` instead.
2. **Weightless community IRs.** Some uploaded OpenVINO IRs ship the `.xml` graph
   without the `.bin` weights — not loadable. This tool validates weights exist.
3. **Version matching** — see above.
4. **GGUF on OVMS** — `device:"CPU"` default and single-file `models_path:"./"` both
   fail silently; see the [serving section](#serving-a-gguf-directly-for-comparison).

## 📋 Requirements

Python 3.12 · the pinned stack in [requirements.txt](requirements.txt) ·
~25–40 GB free disk per large model (source + IR + scratch). Tested on Windows with
OpenVINO 2026.2 on an Intel Arc B70. CPU export works anywhere; quantization is
CPU-side regardless of serving device.

## 📄 License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Converted model weights
remain under their original licenses.
