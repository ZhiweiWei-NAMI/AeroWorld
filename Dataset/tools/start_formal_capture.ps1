param(
  [string]$Repo = "E:\DynamicCityCreatorSamples",
  [string]$Python = "E:\conda\envs\aeroagentsim\python.exe",
  [string]$UnrealEditor = "E:\UE_5.2\Engine\Binaries\Win64\UnrealEditor.exe",
  [string]$UProject = "E:\DynamicCityCreatorSamples\DynamicCityCreatorEx.uproject",
  [string]$AirSimSettings = "E:\HuaweiMoveData\Users\weizhiwei\Documents\AirSim\settings.json",
  [int]$CaptureTicksPerHostRun = 8,
  [int]$HighCaptureTicksPerHostRun = 8,
  [string]$FallbackCaptureTicksPerHostRun = "4,2"
)

$ErrorActionPreference = "Stop"

function Get-FormalCaptureProcess {
  Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python' -and
    ($_.CommandLine -match 'supervise_event_chain_capture|run_semantic_event_chain_every10|episode_render_host')
  }
}

function Invoke-CapturePython {
  param([string]$Code)
  Push-Location $Repo
  try {
    $name = "aw_capture_probe_{0}_{1}.py" -f $PID, ([System.Guid]::NewGuid().ToString("N"))
    $tmp = [System.IO.Path]::Combine([System.IO.Path]::GetTempPath(), $name)
    Set-Content -LiteralPath $tmp -Value $Code -Encoding UTF8
    $proc = Start-Process -FilePath $Python -ArgumentList @($tmp) -WorkingDirectory $Repo -PassThru -Wait -NoNewWindow
    $exitCode = $proc.ExitCode
    Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
    return $exitCode
  }
  finally {
    Pop-Location
  }
}

function Test-PieReady {
  $code = @"
import sys
from pathlib import Path
sys.path.insert(0, str(Path('E:/DynamicCityCreatorSamples/Dataset/tools')))
import supervise_event_chain_capture as supervisor
try:
    supervisor.wait_for_remote(supervisor.ROOT, 20.0)
    supervisor.wait_for_rpc('127.0.0.1', 41451, 8.0)
    supervisor.verify_pie_world(supervisor.ROOT, 20.0)
except Exception as exc:
    print(f'PIE_NOT_READY: {exc}')
    raise SystemExit(1)
print('PIE_READY')
"@
  return (Invoke-CapturePython -Code $code) -eq 0
}

function Test-PieWorld {
  $code = @"
import sys
from pathlib import Path
sys.path.insert(0, str(Path('E:/DynamicCityCreatorSamples/Dataset/tools')))
import supervise_event_chain_capture as supervisor
try:
    supervisor.wait_for_remote(supervisor.ROOT, 20.0)
    supervisor.verify_pie_world(supervisor.ROOT, 20.0)
except Exception as exc:
    print(f'PIE_WORLD_NOT_READY: {exc}')
    raise SystemExit(1)
print('PIE_WORLD_READY')
"@
  return (Invoke-CapturePython -Code $code) -eq 0
}

function Enter-PieOnce {
  $code = @"
import sys
from pathlib import Path
sys.path.insert(0, str(Path('E:/DynamicCityCreatorSamples/Dataset/tools')))
import supervise_event_chain_capture as supervisor
supervisor.wait_for_remote(supervisor.ROOT, 900.0)
supervisor.send_play_hotkey()
supervisor.wait_for_rpc('127.0.0.1', 41451, 600.0)
supervisor.verify_pie_world(supervisor.ROOT, 300.0)
print('PIE_READY')
"@
  $exitCode = Invoke-CapturePython -Code $code
  if ($exitCode -ne 0) {
    throw "Unable to enter or verify PIE. UE is left open for inspection."
  }
}

function Ensure-Pie {
  if (Test-PieReady) {
    Write-Host "[formal-capture] Reusing existing UE/PIE."
    return
  }

  $ue = Get-Process -Name UnrealEditor -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($null -eq $ue) {
    Write-Host "[formal-capture] UnrealEditor is not running; starting it once."
    $argList = @('"' + $UProject + '"', '-settings="' + $AirSimSettings + '"')
    Start-Process -FilePath $UnrealEditor -ArgumentList $argList | Out-Null
    Enter-PieOnce
    return
  }

  if (Test-PieWorld) {
    throw "UE is already in PIE but AirSim RPC is not ready. Leaving UE/PIE untouched for inspection."
  }

  Write-Host "[formal-capture] UnrealEditor is open but not in PIE; entering PIE once without closing UE."
  Enter-PieOnce
}

$existing = @(Get-FormalCaptureProcess)
if ($existing.Count -gt 0) {
  Write-Host "[formal-capture] Formal capture is already running; refusing to start a duplicate."
  $existing | Select-Object ProcessId,CommandLine | Format-Table -AutoSize
  exit 0
}

Ensure-Pie

$argsList = @(
  ".\Dataset\tools\supervise_event_chain_capture.py",
  "--capture-ticks-per-host-run", "$CaptureTicksPerHostRun",
  "--high-capture-ticks-per-host-run", "$HighCaptureTicksPerHostRun",
  "--fallback-capture-ticks-per-host-run", "$FallbackCaptureTicksPerHostRun"
)

Push-Location $Repo
try {
  & $Python @argsList
  $exitCode = $LASTEXITCODE
}
finally {
  Pop-Location
}

if ($exitCode -ne 0) {
  Write-Host "[formal-capture] Supervisor stopped with exit code $exitCode. UE/PIE is intentionally left open."
}
exit $exitCode
