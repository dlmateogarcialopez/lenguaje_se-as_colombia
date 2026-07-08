# Traductor LSC - Startup
# Uso: .\start.ps1

param(
    [int]$Port = 8765,
    [switch]$NoBrowser = $false,
    [switch]$NoWatcher = $false
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Traductor LSC - Prototipo" -ForegroundColor Cyan
Write-Host "  Puerto: $Port" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# --- Verificar Python ---
try {
    $pyv = python --version 2>&1
    Write-Host "[OK] $pyv" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python no encontrado" -ForegroundColor Red
    exit 1
}

# --- Watcher en proceso separado (background) ---
$watcherJob = $null
if (-not $NoWatcher) {
    if (Test-Path "$root\watcher.py") {
        Write-Host "[*] Iniciando watcher (LSCPROPIO -> landmarks)..." -ForegroundColor Green
        $watcherJob = Start-Process python -ArgumentList "watcher.py" `
            -WorkingDirectory $root -PassThru -WindowStyle Hidden `
            -RedirectStandardOutput "$root\watcher.log" `
            -RedirectStandardError "$root\watcher.err.log"
        Write-Host "    PID: $($watcherJob.Id), log: $root\watcher.log" -ForegroundColor Gray
    } else {
        Write-Host "[WARN] watcher.py no encontrado, saltado" -ForegroundColor Yellow
    }
}

# --- Abrir navegador (delay para que servidor este listo) ---
if (-not $NoBrowser) {
    Start-Job -ScriptBlock {
        param($url) Start-Sleep -Seconds 2; Start-Process $url
    } -ArgumentList "http://localhost:$Port/demo.html" | Out-Null
}

# --- Servidor estatico (proceso principal, Ctrl+C para parar) ---
Write-Host ""
Write-Host "[*] Servidor:  http://localhost:$Port/demo.html" -ForegroundColor White
Write-Host "[*] Ctrl+C para detener" -ForegroundColor Gray
Write-Host ""

try {
    Set-Location "$root\static"
    python -m http.server $Port
} finally {
    # Al salir, matar tambien el watcher
    if ($watcherJob -and -not $watcherJob.HasExited) {
        Write-Host "`n[*] Deteniendo watcher (PID $($watcherJob.Id))..." -ForegroundColor Yellow
        Stop-Process -Id $watcherJob.Id -Force -ErrorAction SilentlyContinue
    }
}
