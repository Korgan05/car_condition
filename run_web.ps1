param(
  [string]$Ckpt = "checkpoints/resnet18_multitask.pt",
  [int]$Port = 8000,
  [string]$BindHost = "127.0.0.1",
  [string]$Backbone = "auto",
  [string]$YoloWeights = ""
)

$ErrorActionPreference = 'Stop'

# Ensure we run from the script's directory so relative paths resolve correctly
Set-Location -Path $PSScriptRoot

# Select Python from GPU venv if available, otherwise default venv
$pythonExe = ".\.venv\Scripts\python.exe"
if (Test-Path ".\.venv-gpu\Scripts\python.exe") {
  $pythonExe = ".\.venv-gpu\Scripts\python.exe"
  Write-Host ("[Web] Using GPU venv: {0}" -f $pythonExe)
} else {
  Write-Host ("[Web] Using venv: {0}" -f $pythonExe)
}

Write-Host ("[Web] Starting on http://{0}:{1} with {2} (backbone={3})" -f $BindHost, $Port, $Ckpt, $Backbone)
if ($YoloWeights -ne "") {
  Write-Host ("[Web] YOLO weights: {0}" -f $YoloWeights)
  & $pythonExe -m src.web.main --ckpt $Ckpt --host $BindHost --port $Port --backbone $Backbone --yolo-weights $YoloWeights
} else {
  & $pythonExe -m src.web.main --ckpt $Ckpt --host $BindHost --port $Port --backbone $Backbone
}