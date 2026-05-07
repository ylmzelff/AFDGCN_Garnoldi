# =============================================================================
# Phase API Başlatma Scripti (Windows PowerShell — Geliştirme)
# =============================================================================

Write-Host "🚀 AFDGCN Phase API Başlatılıyor..." -ForegroundColor Green
Write-Host ""

# Virtual environment aktif mi kontrol et
if (-not $env:VIRTUAL_ENV) {
    Write-Host "⚠️  Virtual environment aktif değil. Aktivating..." -ForegroundColor Yellow
    & ".venv\Scripts\Activate.ps1"
}

# Port kontrolü (9001)
Write-Host "📡 Port 9001 kontrol ediliyor..." -ForegroundColor Cyan
$portCheck = netstat -ano | Select-String ":9001"
if ($portCheck) {
    Write-Host "⚠️  Port 9001 zaten kullanımda!" -ForegroundColor Red
    Write-Host "   Devam etmek için ENTER'a basın" -ForegroundColor Yellow
    Read-Host
    $processId = ($portCheck -split "\s+")[-1]
    Stop-Process -Id $processId -Force
    Write-Host "✅ Process sonlandırıldı" -ForegroundColor Green
}

# .env dosyası var mı?
if (-not (Test-Path ".env")) {
    Write-Host "⚠️  .env dosyası bulunamadı. .env.example kopyalanıyor..." -ForegroundColor Yellow
    Copy-Item ".env.example" ".env"
    Write-Host "✅ .env dosyası oluşturuldu. Lütfen ayarları kontrol edin!" -ForegroundColor Green
    Write-Host ""
}

# Network IP
$networkIP = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object {$_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.*"} | Select-Object -First 1).IPAddress

Write-Host ""
Write-Host ("=" * 65) -ForegroundColor Cyan
Write-Host "🌐 Phase API Erişim Bilgileri:" -ForegroundColor Green
Write-Host ("=" * 65) -ForegroundColor Cyan
Write-Host ""
Write-Host "  📍 Local:          http://localhost:9001" -ForegroundColor White
Write-Host "  📍 Network:        http://${networkIP}:9001" -ForegroundColor White
Write-Host "  📍 Swagger UI:     http://localhost:9001/docs" -ForegroundColor White
Write-Host "  📍 Health Check:   http://localhost:9001/health" -ForegroundColor White
Write-Host "  📍 WebSocket:      ws://localhost:9001/ws/live?token=..." -ForegroundColor White
Write-Host ""
Write-Host ("=" * 65) -ForegroundColor Cyan
Write-Host ""
Write-Host "⏹️  Durdurmak için CTRL+C" -ForegroundColor Red
Write-Host ""

# Phase API başlat (--reload geliştirme modu)
uvicorn backend.app.main:app --host 0.0.0.0 --port 9001 --reload --log-level info
