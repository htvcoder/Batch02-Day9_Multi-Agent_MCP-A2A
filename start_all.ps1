$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$logsDir = Join-Path $root "logs"
if (-not (Test-Path $logsDir)) {
    New-Item -ItemType Directory -Path $logsDir | Out-Null
}

$services = @(
    @{ Name = "registry"; Port = 10000; Module = "registry"; Delay = 2 },
    @{ Name = "tax_agent"; Port = 10102; Module = "tax_agent"; Delay = 2 },
    @{ Name = "compliance_agent"; Port = 10103; Module = "compliance_agent"; Delay = 2 },
    @{ Name = "law_agent"; Port = 10101; Module = "law_agent"; Delay = 2 },
    @{ Name = "customer_agent"; Port = 10100; Module = "customer_agent"; Delay = 0 }
)

Write-Host "Checking ports 10000-10103..."
foreach ($port in 10000, 10100, 10101, 10102, 10103) {
    $existing = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if ($existing) {
        $pids = ($existing | Select-Object -ExpandProperty OwningProcess | Sort-Object -Unique) -join ", "
        Write-Warning "Port $port is already in use by PID(s): $pids"
    }
}

foreach ($service in $services) {
    $logPath = Join-Path $logsDir "$($service.Name).log"
    Write-Host "Starting $($service.Name) on port $($service.Port)..."
    $command = "Set-Location '$root'; uv run python -m $($service.Module) *>> '$logPath'"
    $process = Start-Process powershell.exe `
        -ArgumentList @("-NoProfile", "-WindowStyle", "Hidden", "-Command", $command) `
        -PassThru
    Write-Host "  PID: $($process.Id)  Log: $logPath"
    if ($service.Delay -gt 0) {
        Start-Sleep -Seconds $service.Delay
    }
}

Write-Host ""
Write-Host "Services starting:"
Write-Host "  Registry:         http://localhost:10000"
Write-Host "  Customer Agent:   http://localhost:10100"
Write-Host "  Law Agent:        http://localhost:10101"
Write-Host "  Tax Agent:        http://localhost:10102"
Write-Host "  Compliance Agent: http://localhost:10103"
Write-Host ""
Write-Host "Logs are in .\logs\"
Write-Host "To stop the services, run: .\stop_all.ps1"
