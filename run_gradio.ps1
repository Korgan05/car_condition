param(
  [string]$Ckpt = "checkpoints/resnet18_multitask.pt",
  [int]$Port = 7860,
  [string]$BindHost = "127.0.0.1"
)

$ErrorActionPreference = 'Stop'

# Ensure we run from the script's directory so relative paths (src/, checkpoints/) resolve correctly
Set-Location -Path $PSScriptRoot

Write-Host ("[Gradio] Starting on http://{0}:{1} with {2}" -f $BindHost, $Port, $Ckpt)
& .\.venv\Scripts\python.exe -m src.app --ckpt $Ckpt --server-name $BindHost --server-port $Port