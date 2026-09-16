# Stop services for Marine Debris Sonar AI (backend :8000 and frontend :5173)
$ports = @(8000, 5173)

foreach ($port in $ports) {
    try {
        $connections = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
        if ($connections) {
            $pids = $connections | Select-Object -ExpandProperty OwningProcess -Unique
            foreach ($p in $pids) {
                if ($p -gt 0) {
                    Write-Host "Stopping process PID $p on port $port..."
                    Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
                }
            }
        } else {
            Write-Host "No process listening on port $port."
        }
    } catch {
        Write-Warning "Could not inspect port $port: $_"
    }
}

Write-Host "All services stopped."
