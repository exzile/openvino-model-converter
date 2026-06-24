#!/usr/bin/env python
"""Reusable Hugging Face -> OpenVINO INT4 IR converter.

Handles three export shapes from one tool:
  * text       - standard decoder-only LLM  -> text-generation-with-past IR
  * multimodal - vision-language model       -> image-text-to-text IR (full pipeline)
  * decoder    - VL model, extract the language_model submodule -> standalone
                 text-only IR a /chat/completions stack can actually serve

Detection is automatic from the source config.json; `--shape auto` (default) picks
text for plain LLMs and BOTH multimodal+decoder for VL models. Output IRs are staged
under <repo-root>/Converted/<out-name>[-<shape>].

Drive this through Convert-Model.ps1, which selects the right venv. It can also be
run directly with a venv python that has optimum-intel + openvino + transformers.

This is conversion tooling only - it downloads, converts, and validates. It does not
serve models or run benchmarks; those are separate steps.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[convert] {msg}", flush=True)


def err(msg: str) -> None:
    print(f"[convert][ERROR] {msg}", file=sys.stderr, flush=True)


def default_repo_root() -> Path:
    # IRs are staged under <root>/Converted/. Override with --repo-root or $OVCONV_OUT.
    env = os.environ.get("OVCONV_OUT")
    if env:
        return Path(env)
    return Path.cwd() / "model-repo"


def read_source_config(src: Path) -> dict:
    cfg = src / "config.json"
    if not cfg.is_file():
        return {}
    try:
        return json.loads(cfg.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        err(f"could not parse {cfg}: {e}")
        return {}


def is_vision_language(cfg: dict) -> bool:
    if not cfg:
        return False
    if "vision_config" in cfg:
        return True
    archs = cfg.get("architectures") or []
    for a in archs:
        al = str(a).lower()
        if "conditionalgeneration" in al or "imagetext" in al or "vision" in al:
            return True
    return False


# ----------------------------------------------------------------------------
# download
# ----------------------------------------------------------------------------

def download_model(repo_id: str, dest: Path, revision: str | None, token: str | None) -> Path:
    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    log(f"downloading {repo_id} -> {dest}")
    # Skip formats we never convert from (gguf/awq/onnx/pickle/original sharded mirrors).
    ignore = [
        "*.gguf", "*.onnx", "*.pth", "*.pt", "*.msgpack", "*.h5",
        "*.tflite", "original/*", "*.bin.index.json.bak",
    ]
    path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(dest),
        revision=revision,
        token=token,
        ignore_patterns=ignore,
    )
    return Path(path)


# ----------------------------------------------------------------------------
# export via optimum main_export, signature-filtered
# ----------------------------------------------------------------------------

def bypass_version_ceiling(model_type_hint: str | None = None) -> None:
    """optimum pins MAX_TRANSFORMERS_VERSION on some OV exporter configs (e.g. qwen3_5
    at 5.2.99) and HARD-RAISES if the installed transformers is newer. When the
    installed transformers still loads the model AND the exporter config exists, the
    ceiling is just a conservative guard. This bumps the MAX (and floors the MIN) on
    every OV config class so export can proceed. Opt-in only: the resulting IR MUST
    be generation-probed before trust, since a real patcher/internals mismatch would
    otherwise surface here as a broken graph instead of a clean error."""
    import optimum.exporters.openvino.model_configs as mc

    bumped = []
    for name in dir(mc):
        cls = getattr(mc, name)
        if isinstance(cls, type) and (
            getattr(cls, "MAX_TRANSFORMERS_VERSION", None) or getattr(cls, "MIN_TRANSFORMERS_VERSION", None)
        ):
            if getattr(cls, "MAX_TRANSFORMERS_VERSION", None):
                cls.MAX_TRANSFORMERS_VERSION = "99.99.99"
            if getattr(cls, "MIN_TRANSFORMERS_VERSION", None):
                cls.MIN_TRANSFORMERS_VERSION = "0.0.0"
            bumped.append(name)
    log(f"version-ceiling bypass: relaxed MIN/MAX on {len(bumped)} OV config classes "
        f"(includes {[b for b in bumped if 'Qwen3_5' in b] or '...'})")
    log("  WARNING: bypass is unvalidated by optimum; the IR must pass a real generation probe before use.")


def run_main_export(model_path: Path, output: Path, task: str, q: dict,
                    trust_remote_code: bool, bypass_ceiling: bool = False) -> None:
    """Export to an INT4 OV IR by shelling out to `optimum-cli export openvino`.

    Calling main_export() directly does NOT run the separate weight-compression
    step (_main_quantize) that the CLI performs, so a programmatic call silently
    yields a full-precision IR. The CLI is the proven path (it produced the 32B
    int4 IR), so we invoke it as a subprocess for correct quantization.
    """
    output.mkdir(parents=True, exist_ok=True)
    cli = ["-m", "optimum.commands.optimum_cli", "export", "openvino",
           "--model", str(model_path),
           "--task", task,
           "--weight-format", str(q.get("weight_format", "int4")),
           "--ratio", str(q.get("ratio", 1.0)),
           "--group-size", str(q.get("group_size", 128))]
    if q.get("sym", True):
        cli.append("--sym")
    if trust_remote_code:
        cli.append("--trust-remote-code")
    cli.append(str(output))  # output dir is positional, last

    if bypass_ceiling:
        # The monkeypatch must run inside the export subprocess. Wrap the CLI in a
        # tiny preamble that relaxes optimum's version guards before dispatching.
        preamble = (
            "import sys, optimum.exporters.openvino.model_configs as mc\n"
            "for n in dir(mc):\n"
            " c=getattr(mc,n)\n"
            " if isinstance(c,type):\n"
            "  if getattr(c,'MAX_TRANSFORMERS_VERSION',None): c.MAX_TRANSFORMERS_VERSION='99.99.99'\n"
            "  if getattr(c,'MIN_TRANSFORMERS_VERSION',None): c.MIN_TRANSFORMERS_VERSION='0.0.0'\n"
            "from optimum.commands.optimum_cli import main\n"
            "sys.argv=['optimum-cli']+sys.argv[1:]\n"
            "main()\n"
        )
        cmd = [sys.executable, "-c", preamble] + cli[1:]  # drop the leading -m/module
        log("  (running with version-ceiling bypass inside the export subprocess)")
    else:
        cmd = [sys.executable] + cli

    log(f"export task={task} -> {output}")
    log("  optimum-cli " + " ".join(cli[3:]))  # human-readable: everything after 'export openvino'
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"optimum-cli export failed (exit {res.returncode}) for task={task}")


# ----------------------------------------------------------------------------
# decoder extraction (VL -> standalone text checkpoint)
# ----------------------------------------------------------------------------

def extract_text_decoder(src: Path, work: Path, trust_remote_code: bool) -> Path:
    """Load a VL model, pull out the language model + lm_head, save a standalone
    text CausalLM checkpoint at `work`. Returns the checkpoint path.

    VL internals vary, so this introspects and tries strategies in order, printing
    the module tree so a first run informs any needed follow-up.
    """
    import torch  # noqa: F401
    from transformers import AutoConfig, AutoModelForCausalLM
    try:
        from transformers import AutoModelForImageTextToText as VLLoader
    except Exception:  # noqa: BLE001
        from transformers import AutoModel as VLLoader  # type: ignore

    log("loading VL model on CPU for decoder extraction (this needs RAM ~= model size)")
    vl = VLLoader.from_pretrained(
        str(src), torch_dtype="auto", trust_remote_code=trust_remote_code, low_cpu_mem_usage=True
    )
    vl.eval()

    top = [n for n, _ in vl.named_children()]
    log(f"VL top-level modules: {top}")

    full_cfg = AutoConfig.from_pretrained(str(src), trust_remote_code=trust_remote_code)
    text_cfg = getattr(full_cfg, "text_config", None)
    if text_cfg is None:
        raise RuntimeError("source config has no text_config; cannot derive a text checkpoint")
    log(f"text_config model_type={getattr(text_cfg, 'model_type', '?')} "
        f"hidden={getattr(text_cfg, 'hidden_size', '?')} layers={getattr(text_cfg, 'num_hidden_layers', '?')}")

    # locate the language model submodule
    lm = _find_language_model(vl)
    if lm is None:
        raise RuntimeError(
            "could not locate a language_model submodule on the VL model. "
            f"Inspect the printed module tree (top={top}) and extend _find_language_model()."
        )
    log(f"located language model: {type(lm).__name__}")

    # locate the lm_head (often top-level on the VL wrapper, sometimes inside lm)
    lm_head = _find_lm_head(vl, lm)
    log(f"lm_head: {'found ' + type(lm_head).__name__ if lm_head is not None else 'NOT found (tied embeddings?)'}")

    # Build a standalone CausalLM from text_config and graft weights in.
    try:
        text_cfg.architectures = None  # let AutoModel pick the CausalLM head for this model_type
        standalone = AutoModelForCausalLM.from_config(text_cfg, trust_remote_code=trust_remote_code)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"AutoModelForCausalLM has no CausalLM head registered for text model_type "
            f"'{getattr(text_cfg, 'model_type', '?')}': {e}. "
            "If transformers lacks a *TextForCausalLM, the decoder cannot be exported text-only."
        )

    _graft_weights(standalone, lm, lm_head)

    ckpt = work / "text-decoder-ckpt"
    if ckpt.exists():
        shutil.rmtree(ckpt)
    ckpt.mkdir(parents=True, exist_ok=True)
    log(f"saving standalone text checkpoint -> {ckpt}")
    standalone.save_pretrained(str(ckpt), safe_serialization=True)

    _copy_tokenizer_and_aux(src, ckpt)
    return ckpt


def _find_language_model(vl):
    for path in ("language_model", "model.language_model", "model"):
        obj = vl
        ok = True
        for part in path.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                ok = False
                break
        if ok and obj is not None and any(c in type(obj).__name__.lower() for c in ("model", "decoder", "transformer")):
            # avoid returning the vision tower
            if "vision" not in type(obj).__name__.lower():
                return obj
    return None


def _find_lm_head(vl, lm):
    for owner in (vl, getattr(vl, "model", None), lm):
        if owner is not None and hasattr(owner, "lm_head"):
            head = getattr(owner, "lm_head")
            if head is not None:
                return head
    return None


def _graft_weights(standalone, lm, lm_head) -> None:
    import torch

    sd = standalone.state_dict()
    lm_sd = lm.state_dict()
    matched, missing = 0, []
    new_sd = {}
    for k in sd:
        # standalone keys look like "model.<...>" or "<...>"; the VL language_model
        # keys are relative. Try direct, then strip a leading "model." prefix.
        cand = None
        if k in lm_sd:
            cand = lm_sd[k]
        elif k.startswith("model.") and k[len("model."):] in lm_sd:
            cand = lm_sd[k[len("model."):]]
        elif ("model." + k) in lm_sd:
            cand = lm_sd["model." + k]
        if cand is not None and cand.shape == sd[k].shape:
            new_sd[k] = cand
            matched += 1
        else:
            missing.append(k)
    if lm_head is not None and hasattr(lm_head, "weight"):
        for hk in ("lm_head.weight", "lm_head.bias"):
            attr = hk.split(".")[1]
            if hasattr(lm_head, attr) and getattr(lm_head, attr) is not None and hk in sd:
                if getattr(lm_head, attr).shape == sd[hk].shape:
                    new_sd[hk] = getattr(lm_head, attr)
                    matched += 1
                    if hk in missing:
                        missing.remove(hk)
    log(f"weight graft: matched {matched}/{len(sd)} tensors; {len(missing)} unmatched")
    if missing:
        log(f"  first unmatched keys: {missing[:8]}")
    res = standalone.load_state_dict({**sd, **new_sd}, strict=False)
    if getattr(res, "missing_keys", None):
        log(f"  load_state_dict missing_keys: {list(res.missing_keys)[:8]} (total {len(res.missing_keys)})")


def _copy_tokenizer_and_aux(src: Path, ckpt: Path) -> None:
    keep = [
        "tokenizer.json", "tokenizer_config.json", "tokenizer.model",
        "vocab.json", "merges.txt", "special_tokens_map.json",
        "generation_config.json", "added_tokens.json", "chat_template.jinja",
        "preprocessor_config.json",
    ]
    for name in keep:
        s = src / name
        if s.is_file():
            shutil.copy2(s, ckpt / name)
            log(f"  copied {name}")


# ----------------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------------

def validate_ir(out: Path) -> bool:
    """An IR is usable only if it has BOTH the graph .xml and non-trivial .bin
    weights. Recurse so multimodal sub-IRs (vision/text) are each checked."""
    xmls = list(out.rglob("*.xml"))
    if not xmls:
        err(f"no .xml graph found under {out}")
        return False
    ok = True
    for xml in xmls:
        binf = xml.with_suffix(".bin")
        size = binf.stat().st_size if binf.is_file() else 0
        status = "OK" if size > 1_000_000 else "MISSING/EMPTY WEIGHTS"
        if size <= 1_000_000:
            ok = False
        log(f"  IR {xml.relative_to(out)}: weights {size/1e9:.2f} GB -> {status}")
    return ok


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="HF -> OpenVINO IR converter")
    ap.add_argument("--model", required=True, help="HF repo id OR local checkpoint path")
    ap.add_argument("--out-name", help="output IR folder name (default derived from --model)")
    ap.add_argument("--shape", choices=["auto", "text", "multimodal", "decoder", "both"],
                    default="auto", help="auto: text for LLMs, both(multimodal+decoder) for VL")
    ap.add_argument("--repo-root", default=str(default_repo_root()),
                    help="model-repo root; IRs go to <root>/Converted/")
    ap.add_argument("--src-dir", help="reuse an already-downloaded source dir instead of downloading")
    ap.add_argument("--revision", help="HF revision/branch")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF token (or $HF_TOKEN); public models need none")
    ap.add_argument("--weight-format", default="int4")
    ap.add_argument("--ratio", type=float, default=1.0)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--no-sym", action="store_true", help="disable symmetric quantization")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--bypass-version-ceiling", action="store_true",
                    help="relax optimum's MIN/MAX_TRANSFORMERS_VERSION guards (needed for qwen3_5 "
                         "on transformers>5.2.99); IR MUST then be generation-probed before trust")
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args()

    repo_root = Path(args.repo_root)
    converted = repo_root / "Converted"
    converted.mkdir(parents=True, exist_ok=True)

    base_name = args.out_name or Path(args.model).name
    q = {
        "weight_format": args.weight_format,
        "ratio": args.ratio,
        "sym": not args.no_sym,
        "group_size": args.group_size,
    }

    # 1. acquire source
    if args.src_dir:
        src = Path(args.src_dir)
        if not src.is_dir():
            err(f"--src-dir {src} not found")
            return 2
        log(f"using existing source dir {src}")
    else:
        src = converted / f"_src_{base_name}"
        src = download_model(args.model, src, args.revision, args.token)

    if args.download_only:
        log("download-only: done")
        return 0

    cfg = read_source_config(src)
    vl = is_vision_language(cfg)
    log(f"source detected as {'VISION-LANGUAGE' if vl else 'TEXT'} "
        f"(model_type={cfg.get('model_type','?')}, archs={cfg.get('architectures')})")

    # 2. resolve shapes to produce
    shape = args.shape
    if shape == "auto":
        shape = "both" if vl else "text"
    shapes: list[str] = []
    if shape == "text":
        shapes = ["text"]
    elif shape == "multimodal":
        shapes = ["multimodal"]
    elif shape == "decoder":
        shapes = ["decoder"]
    elif shape == "both":
        shapes = ["multimodal", "decoder"]
    log(f"will produce: {shapes}")

    bp = args.bypass_version_ceiling

    results = {}
    work = converted / f"_work_{base_name}"
    work.mkdir(parents=True, exist_ok=True)

    for sh in shapes:
        try:
            if sh == "text":
                out = converted / base_name
                run_main_export(src, out, "text-generation-with-past", q, args.trust_remote_code, bp)
            elif sh == "multimodal":
                out = converted / f"{base_name}-multimodal"
                run_main_export(src, out, "image-text-to-text", q, args.trust_remote_code, bp)
            elif sh == "decoder":
                out = converted / f"{base_name}-text"
                ckpt = extract_text_decoder(src, work, args.trust_remote_code)
                run_main_export(ckpt, out, "text-generation-with-past", q, args.trust_remote_code, bp)
            ok = True if args.no_validate else validate_ir(out)
            results[sh] = {"output": str(out), "valid": ok}
            log(f"shape '{sh}' -> {out} (valid={ok})")
        except Exception as e:  # noqa: BLE001
            err(f"shape '{sh}' FAILED: {e}")
            traceback.print_exc()
            results[sh] = {"output": None, "valid": False, "error": str(e)}

    print("\n=== CONVERSION SUMMARY ===")
    print(json.dumps({"model": args.model, "source": str(src), "results": results}, indent=2))
    any_ok = any(r.get("valid") for r in results.values())
    return 0 if any_ok else 1


if __name__ == "__main__":
    sys.exit(main())
