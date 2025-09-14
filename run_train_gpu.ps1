$ErrorActionPreference = 'Stop'

Set-Location $PSScriptRoot

# Defaults (edit if needed)
$EnvPath = ".\.venv-gpu\Scripts\python.exe"
$Data = "Car Scratch and Dent.v1i.yolov8/data.yaml"
$Weights = "yolov8s.pt"
$Project = "yolov8_runs"
$Name = "car_scratch_dent_s960_e30_gpu"
$Img = 960
$Epochs = 30
$Batch = 4
$Device = "0"
$Workers = 2
$Resume = $true

$logDir = Join-Path $Project $Name
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$logFile = Join-Path $logDir "train_${ts}.out.log"
$errFile = Join-Path $logDir "train_${ts}.err.log"

$argsList = @(
    'scripts/train_gpu.py',
    '--weights', $Weights,
    '--data', $Data,
    '--project', $Project,
    '--name', $Name,
    '--imgsz', $Img,
    '--epochs', $Epochs,
    '--batch', $Batch,
    '--device', $Device,
    '--workers', $Workers
)
if ($Resume) { $argsList += '--resume' }

Write-Host "Starting GPU training to $logFile"
Start-Process -FilePath $EnvPath -ArgumentList $argsList -RedirectStandardOutput $logFile -RedirectStandardError $errFile -WindowStyle Minimized
Start-Sleep -Seconds 2
Write-Host "Follow logs with: Get-Content -Path `"$logFile`" -Wait"
