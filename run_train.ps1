param(
  [string]$TrainCsv = "data/splits/train.csv",
  [string]$ValCsv = "data/splits/val.csv",
  [string]$Out = "checkpoints/resnet18_multitask.pt",
  [string]$Backbone = "resnet18",
  [int]$Epochs = 10,
  [int]$Batch = 16,
  [int]$FreezeEpochs = 0,
  [int]$Workers = 2
)

$ErrorActionPreference = 'Stop'

# Ensure we run from the script's directory so relative paths resolve correctly
Set-Location -Path $PSScriptRoot

Write-Host ("[Train] {0} -> {1} (bb={2}, epochs={3})" -f $TrainCsv, $Out, $Backbone, $Epochs)
& .\.venv\Scripts\python.exe -m src.train --train-csv $TrainCsv --val-csv $ValCsv --epochs $Epochs --batch-size $Batch --out $Out --backbone $Backbone --freeze-epochs $FreezeEpochs --workers $Workers