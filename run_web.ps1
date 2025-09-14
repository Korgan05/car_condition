param(
  [string]$Ckpt = "checkpoints/resnet18_multitask.pt",
  [int]$Port = 8000,
  [string]$BindHost = "127.0.0.1",
  [string]$Backbone = "resnet18",
  [string]$YoloWeights = ""
)

$ErrorActionPreference = 'Stop'

# Ensure we run from the script's directory so relative paths resolve correctly
Set-Location -Path $PSScriptRoot

Write-Host ("[Web] Starting on http://{0}:{1} with {2} (backbone={3})" -f $BindHost, $Port, $Ckpt, $Backbone)
if ($YoloWeights -ne "") {
  Write-Host ("[Web] YOLO weights: {0}" -f $YoloWeights)
  & .\.venv\Scripts\python.exe -m src.web.main --ckpt $Ckpt --host $BindHost --port $Port --backbone $Backbone --yolo-weights $YoloWeights
} else {
  & .\.venv\Scripts\python.exe -m src.web.main --ckpt $Ckpt --host $BindHost --port $Port --backbone $Backbone
}