<#
.SYNOPSIS
  Convert a Hugging Face model to an INT4 OpenVINO IR.

.DESCRIPTION
  Thin driver around convert_model.py. It:
    1. Picks the right conversion venv (see Setup-Venvs.ps1). Different model
       families need optimum's exporter config to MATCH the installed transformers
       version (the patcher imports model internals that drift across versions) --
       see docs/version-matching.md.
    2. Invokes the Python engine with the requested shape + quantization knobs.

  Run scripts/Setup-Venvs.ps1 once first to create the venvs under ./venvs/.

.EXAMPLE
  # Standard text LLM -> INT4 IR:
  .\Convert-Model.ps1 -Model Qwen/Qwen2.5-Coder-14B-Instruct -Shape text

.EXAMPLE
  # A qwen3_5 vision-language model -> BOTH a full multimodal IR and an
  # extracted text-only IR (uses the transformers-5.2.0 matched venv):
  .\Convert-Model.ps1 -Model empero-ai/Qwythos-9B-Claude-Mythos-5-1M -Shape both -Venv qwen35
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string] $Model,
    [string] $OutName,
    [ValidateSet('auto', 'text', 'multimodal', 'decoder', 'both')] [string] $Shape = 'auto',
    # standard = latest-transformers stack; qwen35 = transformers 5.2.0 (qwen3_5/VL export)
    [ValidateSet('standard', 'qwen35', 'auto')] [string] $Venv = 'auto',
    [string] $VenvRoot,
    [string] $OutRoot,
    [string] $SrcDir,
    [string] $Revision,
    [string] $WeightFormat = 'int4',
    [double] $Ratio = 1.0,
    [int] $GroupSize = 128,
    [switch] $NoSym,
    [switch] $TrustRemoteCode,
    [switch] $BypassVersionCeiling,
    [switch] $DownloadOnly,
    [switch] $NoValidate
)

$ErrorActionPreference = 'Stop'
$engine = Join-Path $PSScriptRoot 'convert_model.py'
if (-not $VenvRoot) { $VenvRoot = Join-Path $PSScriptRoot 'venvs' }
if (-not $OutRoot) { $OutRoot = Join-Path $PSScriptRoot 'model-repo' }

# --- venv resolution ---------------------------------------------------------
# Each venv pins a transformers line so optimum's exporter patcher matches the
# model's modeling code. qwen3_5 / VL export REQUIRES transformers 5.2.0 (see
# docs/version-matching.md); most other text models work on the standard venv.
$venvs = @{
    standard = Join-Path $VenvRoot 'venv-standard\Scripts\python.exe'
    qwen35   = Join-Path $VenvRoot 'venv-qwen35\Scripts\python.exe'
}

$chosen = $Venv
if ($chosen -eq 'auto') {
    if ($Shape -in @('multimodal', 'decoder', 'both')) { $chosen = 'qwen35' } else { $chosen = 'standard' }
}
$python = $venvs[$chosen]
if (-not (Test-Path $python)) {
    throw "Conversion venv '$chosen' not found at: $python`nRun scripts/Setup-Venvs.ps1 first (or pass -VenvRoot)."
}
Write-Host "[Convert-Model] venv=$chosen  out=$OutRoot" -ForegroundColor Cyan

# --- build engine args -------------------------------------------------------
$argList = @($engine, '--model', $Model, '--shape', $Shape, '--repo-root', $OutRoot,
    '--weight-format', $WeightFormat, '--ratio', "$Ratio", '--group-size', "$GroupSize")
if ($OutName) { $argList += @('--out-name', $OutName) }
if ($SrcDir) { $argList += @('--src-dir', $SrcDir) }
if ($Revision) { $argList += @('--revision', $Revision) }
if ($NoSym) { $argList += '--no-sym' }
if ($TrustRemoteCode) { $argList += '--trust-remote-code' }
if ($BypassVersionCeiling) { $argList += '--bypass-version-ceiling' }
if ($DownloadOnly) { $argList += '--download-only' }
if ($NoValidate) { $argList += '--no-validate' }

Write-Host "[Convert-Model] $python $($argList -join ' ')" -ForegroundColor DarkGray
& $python @argList
exit $LASTEXITCODE
