# =============================================================================
# Phase API Production Başlatma (Windows PowerShell)
# =============================================================================

Write-Host "🏭 AFDGCN Phase API — PRODUCTION MODE" -ForegroundColor Magenta
Write-Host ""

# Virtual environment
if (-not $env:VIRTUAL_ENV) {
    Write-Host "⚠️  Virtual environment aktive ediliyor..." -ForegroundColor Yellow
    & ".venv\Scripts\Activate.ps1"
}

# .env production kontrolü
if (-not (Test-Path ".env")) {
    Write-Host "❌ .env dosyası bulunamadı!" -ForegroundColor Red
    exit 1
}

# Network IP
$networkIP = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object {$_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.*"} | Select-Object -First 1).IPAddress

Write-Host ""
Write-Host ("=" * 65) -ForegroundColor Green
Write-Host "🌐 PRODUCTION Phase API Erişim:" -ForegroundColor Green
Write-Host ("=" * 65) -ForegroundColor Green
Write-Host ""
Write-Host "  📍 Local:          http://localhost:9001" -ForegroundColor White
Write-Host "  📍 Network:        http://${networkIP}:9001" -ForegroundColor White
Write-Host "  📍 Swagger UI:     http://localhost:9001/docs" -ForegroundColor White
Write-Host "  📍 Health:         http://localhost:9001/health" -ForegroundColor White
Write-Host "  📍 WebSocket:      ws://localhost:9001/ws/live?token=..." -ForegroundColor White
Write-Host ""
Write-Host ("=" * 65) -ForegroundColor Green
Write-Host ""

# Production modunda başlat (tek worker — AFDGCN thread-safe değil)
Write-Host "🚀 Production server başlatılıyor (1 worker)..." -ForegroundColor Green
Write-Host ""

uvicorn backend.app.main:app `
    --host 0.0.0.0 `
    --port 9001 `
    --workers 1 `
    --log-level warning `
    --no-access-log `
    --timeout-keep-alive 65

