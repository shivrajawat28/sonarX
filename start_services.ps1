# Start services for Marine Debris Sonar AI
$RepoRoot = "C:\Users\kr034\OneDrive\Desktop\SONAR\SONAR"
$PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$FrontendDir = Join-Path $RepoRoot "frontend"

Write-Host "Starting backend on http://127.0.0.1:8000 ..."
Start-Process -FilePath $PythonExe `
    -ArgumentList "-m", "uvicorn", "backend.app.main:app", "--port", "8000", "--host", "127.0.0.1" `
    -WorkingDirectory $RepoRoot `
    -RedirectStandardOutput (Join-Path $RepoRoot "backend_server.log") `
    -RedirectStandardError (Join-Path $RepoRoot "backend_server.err.log")

# Wait for backend health check
Write-Host "Waiting for backend to become ready..."
$ready = $false
for ($i = 0; $i -lt 30; $i++) {
    try {
        $response = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/health" -TimeoutSec 2 -ErrorAction Stop
        if ($response.status -eq "ok" -or $response.model) {
            Write-Host "Backend ready! Model loaded: $($response.model.version) (loaded: $($response.model.loaded))"
            $ready = $true
            break
        }
    } catch {
        # Waiting for server
    }
    Start-Sleep -Seconds 1
}

if (-not $ready) {
    Write-Warning "Backend health check timed out. Inspect backend_server.err.log"
}

# Start frontend
Write-Host "Starting frontend on http://localhost:5173 ..."
Start-Process -FilePath "npm.cmd" `
    -ArgumentList "run", "dev" `
    -WorkingDirectory $FrontendDir `
    -RedirectStandardOutput (Join-Path $RepoRoot "frontend_dev.log") `
    -RedirectStandardError (Join-Path $RepoRoot "frontend_dev.err.log")

Start-Sleep -Seconds 3
Write-Host "Services started! Frontend: http://localhost:5173 | Backend: http://127.0.0.1:8000"