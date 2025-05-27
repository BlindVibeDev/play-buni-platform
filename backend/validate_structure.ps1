# Play Buni Platform - Structure Validation Script
# Validates that all necessary files are present for Railway deployment

Write-Host "🚀 Play Buni Platform - Structure Validation" -ForegroundColor Green
Write-Host "===============================================" -ForegroundColor Yellow

$errors = 0
$warnings = 0

# Function to check if file exists
function Test-File {
    param($path, $description)
    if (Test-Path $path) {
        Write-Host "✅ $description" -ForegroundColor Green
        return $true
    } else {
        Write-Host "❌ Missing: $description ($path)" -ForegroundColor Red
        $global:errors++
        return $false
    }
}

# Function to check if directory exists
function Test-Directory {
    param($path, $description)
    if (Test-Path $path -PathType Container) {
        Write-Host "✅ $description" -ForegroundColor Green
        return $true
    } else {
        Write-Host "⚠️ Missing: $description ($path)" -ForegroundColor Yellow
        $global:warnings++
        return $false
    }
}

Write-Host ""
Write-Host "📋 Checking Core Files..." -ForegroundColor Cyan

# Core application files
Test-File "app/main.py" "Main FastAPI application"
Test-File "app/__init__.py" "App package init"
Test-File "app/database.py" "Database configuration"
Test-File "requirements.txt" "Python dependencies"
Test-File "env.example" "Environment variables template"

Write-Host ""
Write-Host "📋 Checking Railway Deployment Files..." -ForegroundColor Cyan

# Railway deployment files
Test-File "railway.json" "Railway configuration"
Test-File "startup.py" "Production startup script"
Test-File "Procfile" "Process definitions"
Test-File "DEPLOYMENT_CHECKLIST.md" "Deployment guide"

Write-Host ""
Write-Host "📋 Checking Core Modules..." -ForegroundColor Cyan

# Core modules
Test-Directory "app/core" "Core configuration module"
Test-Directory "app/models" "Database models"
Test-Directory "app/services" "Business services"
Test-Directory "app/routers" "API routers"
Test-Directory "app/workers" "Background workers"

Write-Host ""
Write-Host "📋 Checking Key Services..." -ForegroundColor Cyan

# Key service files
Test-File "app/services/jupiter_service.py" "Jupiter API service"
Test-File "app/services/signal_service.py" "Signal service"
Test-File "app/services/treasury_manager.py" "Treasury manager"
Test-File "app/routers/jupiter_monitoring.py" "Jupiter monitoring router"

Write-Host ""
Write-Host "📋 Checking Worker Files..." -ForegroundColor Cyan

# Worker files
Test-File "app/workers/celery_app.py" "Celery application"
Test-File "app/workers/signal_processor.py" "Signal processor worker"
Test-File "app/workers/market_monitor.py" "Market monitor worker"

Write-Host ""
Write-Host "📋 Checking Configuration Files..." -ForegroundColor Cyan

# Configuration files
Test-File "app/core/config.py" "Application configuration"
Test-File "app/core/security.py" "Security configuration"
Test-File "app/core/logging.py" "Logging configuration"

# Check file sizes (basic validation)
Write-Host ""
Write-Host "📋 Checking File Sizes..." -ForegroundColor Cyan

$mainSize = (Get-Item "app/main.py" -ErrorAction SilentlyContinue).Length
if ($mainSize -gt 15000) {
    $sizeKB = [math]::Round($mainSize/1024, 1)
    Write-Host "✅ Main app file has substantial content ($sizeKB KB)" -ForegroundColor Green
} else {
    Write-Host "⚠️ Main app file seems small" -ForegroundColor Yellow
    $warnings++
}

$reqSize = (Get-Item "requirements.txt" -ErrorAction SilentlyContinue).Length
if ($reqSize -gt 500) {
    $reqKB = [math]::Round($reqSize/1024, 1)
    Write-Host "✅ Requirements file has dependencies ($reqKB KB)" -ForegroundColor Green
} else {
    Write-Host "⚠️ Requirements file seems small" -ForegroundColor Yellow
    $warnings++
}

# Summary
Write-Host ""
Write-Host "==================================================" -ForegroundColor Yellow
Write-Host "📊 VALIDATION SUMMARY" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Yellow

if ($errors -eq 0 -and $warnings -eq 0) {
    Write-Host "🎉 PERFECT! All files present and ready for deployment!" -ForegroundColor Green
    Write-Host "✅ Your Play Buni Platform is ready for Railway!" -ForegroundColor Green
} elseif ($errors -eq 0) {
    Write-Host "✅ GOOD! Core files present with $warnings warnings" -ForegroundColor Yellow
    Write-Host "🚀 Platform should deploy successfully to Railway" -ForegroundColor Green
} else {
    Write-Host "❌ ISSUES FOUND: $errors errors, $warnings warnings" -ForegroundColor Red
    Write-Host "🔧 Please fix errors before deploying" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🚀 NEXT STEPS:" -ForegroundColor Cyan
Write-Host "1. Review DEPLOYMENT_CHECKLIST.md" -ForegroundColor White
Write-Host "2. Set up Supabase database" -ForegroundColor White  
Write-Host "3. Create Railway project" -ForegroundColor White
Write-Host "4. Configure environment variables" -ForegroundColor White
Write-Host "5. Deploy to Railway!" -ForegroundColor White

Write-Host ""
Write-Host "💡 TIP: Even without local Python, you can deploy directly to Railway!" -ForegroundColor Green
Write-Host "Railway will handle Python installation and dependencies automatically." -ForegroundColor Gray

return $errors 