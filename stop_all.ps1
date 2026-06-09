$ports = 10000, 10100, 10101, 10102, 10103

foreach ($port in $ports) {
    $connections = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if (-not $connections) {
        Write-Host "No listener found on port $port"
        continue
    }

    $pids = $connections | Select-Object -ExpandProperty OwningProcess | Sort-Object -Unique
    foreach ($procId in $pids) {
        try {
            Stop-Process -Id $procId -Force
            Write-Host "Stopped PID $procId on port $port"
        } catch {
            Write-Warning ("Could not stop PID {0} on port {1}: {2}" -f $procId, $port, $_)
        }
    }
}
