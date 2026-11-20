param(
    [ValidateSet('baseline','h','f','hf','c','hc','fc','hfc')][string]$Configuration = 'hfc',
    [string]$Data = '',
    [string]$Manifest = 'D:\Users\86186\ResearchData\SVRDD-weather-v1\manifest.jsonl',
    [string]$Weights = 'D:\Users\86186\ResearchData\SVRDD-local-runs\svrdd-r18-baseline-officialsplit-local-amp-b8-p20-s0\weights\best.pt',
    [string]$Teacher = '',
    [string]$Project = '',
    [string]$Name = '',
    [string]$Device = '0',
    [int]$Batch = 2,
    [int]$Workers = 0,
    [int]$Epochs = 300,
    [int]$Seed = 0,
    [int]$ImageSize = 640,
    [double]$ConsistencyWeight = 0.1,
    [string]$PythonExe = 'D:\Users\86186\Miniconda3\envs\rtdetr\python.exe',
    [string]$Resume = '',
    [switch]$Amp,
    [switch]$Train
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) { throw "Missing Python: $PythonExe" }
$runner = Join-Path $PSScriptRoot 'train_weather_hfc.py'
if ($Resume) {
    $otherSettings = @($PSBoundParameters.Keys | Where-Object { $_ -notin @('Resume','Train','PythonExe') })
    if ($otherSettings.Count) { throw "Resume restores saved settings; remove: $($otherSettings -join ', ')" }
    $resumeArgs = @($runner, '--resume', $Resume)
    if ($Train) { $resumeArgs += '--train' }
    & $PythonExe @resumeArgs
    exit $LASTEXITCODE
}
if (-not $Data) { $Data = Join-Path $PSScriptRoot 'configs\chapter4\dataset.yaml' }
$runnerArgs = @($runner, '--configuration', $Configuration, '--data', $Data, '--manifest', $Manifest,
    '--weights', $Weights, '--batch', "$Batch", '--workers', "$Workers", '--epochs', "$Epochs",
    '--seed', "$Seed", '--device', $Device, '--imgsz', "$ImageSize", '--consistency-weight', "$ConsistencyWeight")
if ($Teacher) { $runnerArgs += @('--teacher', $Teacher) }
if ($Project) { $runnerArgs += @('--project', $Project) }
if ($Name) { $runnerArgs += @('--name', $Name) }
if ($Amp) { $runnerArgs += '--amp' }
if ($Train) { $runnerArgs += '--train' }
& $PythonExe @runnerArgs
exit $LASTEXITCODE

