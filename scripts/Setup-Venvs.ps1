<#
.SYNOPSIS
  Create the matched conversion venvs under ./venvs/.

.DESCRIPTION
  Conversion needs optimum-intel's OpenVINO exporter config to MATCH the installed
  transformers version, because the per-model "model patcher" imports modeling
  internals that drift across transformers releases. A mismatch fails the export
  (ImportError or a broken graph) -- see docs/version-matching.md.

  This creates two venvs with an IDENTICAL OpenVINO/optimum/nncf/torch stack but
  DIFFERENT pinned transformers:

    venv-standard  transformers 5.12.1  -- most text LLMs (llama, qwen2/3 dense, ...)
    venv-qwen35    transformers 5.2.0   -- qwen3_5 / vision-language export
                                           (matches optimum's Qwen3_5ModelPatcher,
                                           which imports Qwen3_5DynamicCache --
                                           present in 5.2.x, removed by 5.12)

  The install trick: optimum-intel 2.0.0 declares transformers<5.1, so we install
  the stack first (it pulls an old transformers) and then force the target
  transformers with --no-deps. The qwen3_5/gemma4 exporter support lives in
  optimum-intel 2.0.0's code regardless of that stale metadata pin.

.PARAMETER BasePython
  Python 3.12 interpreter to seed the venvs from. Defaults to "python".

.EXAMPLE
  .\scripts\Setup-Venvs.ps1
.EXAMPLE
  .\scripts\Setup-Venvs.ps1 -BasePython "C:\Python312\python.exe" -Only qwen35
#>
[CmdletBinding()]
param(
    [string] $BasePython = "python",
    [ValidateSet('all', 'standard', 'qwen35')] [string] $Only = 'all'
)
$ErrorActionPreference = 'Stop'
$venvRoot = Join-Path (Split-Path $PSScriptRoot -Parent) 'venvs'
New-Item -ItemType Directory -Force -Path $venvRoot | Out-Null

# Identical across both venvs; only the transformers pin differs.
$stack = @(
    'optimum==2.2.0', 'optimum-intel==2.0.0',
    'openvino==2026.2.1', 'openvino-tokenizers==2026.2.1.0',
    'nncf==3.2.0', 'torch==2.12.1', 'accelerate==1.14.0',
    'safetensors==0.7.0', 'huggingface-hub'
)

function New-ConvVenv {
    param([string] $Name, [string] $TransformersPin)
    $dir = Join-Path $venvRoot $Name
    $py = Join-Path $dir 'Scripts\python.exe'
    Write-Host "`n=== $Name (transformers $TransformersPin) ===" -ForegroundColor Cyan
    if (Test-Path $py) { Write-Warning "$Name already exists; skipping. Delete $dir to rebuild."; return }
    & $BasePython -m venv $dir
    & $py -m pip install --quiet --upgrade pip
    Write-Host "installing OpenVINO/optimum stack..."
    & $py -m pip install --quiet @stack
    Write-Host "forcing transformers==$TransformersPin (--no-deps)..."
    & $py -m pip install --quiet --no-deps --force-reinstall "transformers==$TransformersPin"
    # smoke: confirm the matched pair imports
    & $py -c "import transformers,optimum,openvino,nncf; print('  ok', transformers.__version__)"
}

if ($Only -in @('all', 'standard')) { New-ConvVenv -Name 'venv-standard' -TransformersPin '5.12.1' }
if ($Only -in @('all', 'qwen35'))   { New-ConvVenv -Name 'venv-qwen35'   -TransformersPin '5.2.0' }

Write-Host "`nDone. venvs under $venvRoot" -ForegroundColor Green
Write-Host "Next: .\Convert-Model.ps1 -Model <hf-repo> -Shape text"
