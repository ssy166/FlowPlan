param(
  [string]$Python = "python",
  [string]$ModelPath = "",
  [string]$AdapterDir = "",
  [string]$ModelName = "test800_model",
  [int]$BatchSize = 1,
  [int]$MaxNewTokens = 512,
  [ValidateSet("auto", "bf16", "fp16", "fp32")]
  [string]$Dtype = "bf16",
  [int]$Limit = 0
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

function Invoke-Step {
  param(
    [string]$Label,
    [scriptblock]$Command
  )
  Write-Host $Label
  & $Command
  if ($LASTEXITCODE -ne 0) {
    throw "Step failed with exit code $LASTEXITCODE`: $Label"
  }
}

$Gold = "data\replan_sft\test800\test.jsonl"
$Pack = "data\replan_sft\test800\model_eval_pack.jsonl"
$OutDir = "outputs\test800"
New-Item -ItemType Directory -Force $OutDir | Out-Null

Invoke-Step "[1/3] Audit test800" {
  & $Python scripts\audit_test500.py --input $Gold --expected-rows 800 --expected-format test800_chat_sft_v1
}

Invoke-Step "[2/3] Build model eval pack" {
  & $Python scripts\build_test500_eval_pack.py --input $Gold --out $Pack --format-name test800
}

if ([string]::IsNullOrWhiteSpace($ModelPath)) {
  Write-Host "[3/3] Skip model inference because -ModelPath was not provided."
  Write-Host "Generic HF inference:"
  Write-Host "powershell -ExecutionPolicy Bypass -File scripts\run_test800_experiment.ps1 -ModelPath <hf_model_or_checkpoint> -ModelName <name>"
  Write-Host "LoRA adapter inference:"
  Write-Host "powershell -ExecutionPolicy Bypass -File scripts\run_test800_experiment.ps1 -ModelPath <base_model> -AdapterDir <lora_adapter_dir> -ModelName <name>"
  exit 0
}

$Pred = "$OutDir\predictions.$ModelName.jsonl"
$Raw = "$OutDir\predictions.$ModelName.raw.jsonl"
$Eval = "$OutDir\eval.$ModelName.json"
$Details = "$OutDir\eval_details.$ModelName.jsonl"

if ([string]::IsNullOrWhiteSpace($AdapterDir)) {
  Invoke-Step "[3a/3] Run generic HF inference" {
    & $Python scripts\run_toolrl_inference.py `
      --model-path $ModelPath `
      --input $Pack `
      --tools data\benchmark\tools.jsonl `
      --out $Pred `
      --raw-out $Raw `
      --batch-size $BatchSize `
      --max-new-tokens $MaxNewTokens `
      --dtype $Dtype `
      --model-name $ModelName `
      --limit $Limit
  }
} else {
  Invoke-Step "[3a/3] Run LoRA adapter inference" {
    & $Python scripts\run_toolrl_lora_generation.py `
      --model-path $ModelPath `
      --adapter-dir $AdapterDir `
      --data $Gold `
      --tools data\benchmark\tools.jsonl `
      --out-pred $Pred `
      --out-raw $Raw `
      --out-details $Details `
      --summary-out "$OutDir\generation_summary.$ModelName.json" `
      --batch-size $BatchSize `
      --max-new-tokens $MaxNewTokens `
      --dtype $Dtype `
      --model-name $ModelName `
      --limit $Limit
  }
}

Invoke-Step "[3b/3] Evaluate predictions" {
  & $Python scripts\evaluate_test500_predictions.py --gold $Gold --pred $Pred --out $Eval --details $Details --format-name test800
}
