Write-Host "🚀 Play Buni Platform - Quick Validation" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Yellow

$files = @(
    "app/main.py",
    "requirements.txt", 
    "railway.json",
    "startup.py",
    "app/services/jupiter_service.py",
    "app/services/signal_service.py",
    "DEPLOYMENT_CHECKLIST.md"
)

$missing = 0

foreach ($file in $files) {
    if (Test-Path $file) {
        Write-Host "✅ $file" -ForegroundColor Green
    } else {
        Write-Host "❌ $file" -ForegroundColor Red
        $missing++
    }
}

Write-Host ""
if ($missing -eq 0) {
    Write-Host "🎉 ALL KEY FILES PRESENT!" -ForegroundColor Green
    Write-Host "✅ Ready for Railway deployment!" -ForegroundColor Green
} else {
    Write-Host "❌ Missing $missing files" -ForegroundColor Red
}

Write-Host ""
Write-Host "🚀 DEPLOY TO RAILWAY:" -ForegroundColor Cyan
Write-Host "1. Go to railway.app" -ForegroundColor White
Write-Host "2. Connect GitHub repo" -ForegroundColor White
Write-Host "3. Set root to /backend" -ForegroundColor White
Write-Host "4. Add environment variables" -ForegroundColor White
Write-Host "5. Deploy!" -ForegroundColor White 