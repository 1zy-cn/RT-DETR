param(
    [Parameter(Mandatory=$true)][string]$Weights,
    [ValidateSet('primary','clear','severe','fog','rain','lowlight')][string]$Group='primary',
    [string]$Device='0', [int]$Batch=2, [int]$Workers=0,
    [string]$Project='', [string]$Name='',
    [string]$PythonExe='D:\Users\86186\Miniconda3\envs\rtdetr\python.exe',
    [switch]$Evaluate
)
$ErrorActionPreference='Stop'
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) { throw "Missing Python: $PythonExe" }
$runnerArgs=@((Join-Path $PSScriptRoot 'evaluate_weather_hfc.py'), '--weights', $Weights,
              '--group', $Group, '--device', $Device, '--batch', "$Batch", '--workers', "$Workers")
if ($Project) { $runnerArgs += @('--project', $Project) }
if ($Name) { $runnerArgs += @('--name', $Name) }
if ($Evaluate) { $runnerArgs += '--evaluate' }
& $PythonExe @runnerArgs
exit $LASTEXITCODE
