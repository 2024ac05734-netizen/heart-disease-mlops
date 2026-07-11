<#
 .SYNOPSIS
   One-command Docker build, run and smoke-test for the Heart Disease API.
 .DESCRIPTION
   Requires Docker Desktop installed and running. Trains the model if it is
   missing, builds the image, runs the container and tests the /predict endpoint.
 .EXAMPLE
   ./run_docker.ps1
#>
param(
    [string]$ImageName = "heart-disease-api:latest",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error "Docker is not installed or not on PATH. Install Docker Desktop and retry."
    exit 1
}

if (-not (Test-Path "models/model.joblib")) {
    Write-Host "Model not found - training first..." -ForegroundColor Yellow
    python -m heart_mlops.train
}

Write-Host "Building image $ImageName ..." -ForegroundColor Cyan
docker build -t $ImageName .

Write-Host "Running container on port $Port ..." -ForegroundColor Cyan
docker rm -f heart-api 2>$null | Out-Null
docker run -d --name heart-api -p "${Port}:8000" $ImageName | Out-Null

Write-Host "Waiting for the service to become healthy..." -ForegroundColor Cyan
Start-Sleep -Seconds 15

Write-Host "--- /health ---" -ForegroundColor Green
Invoke-RestMethod "http://localhost:$Port/health" | ConvertTo-Json -Compress

Write-Host "--- /predict ---" -ForegroundColor Green
$body = Get-Content "api/sample_input.json" -Raw
Invoke-RestMethod -Method Post -Uri "http://localhost:$Port/predict" -Body $body -ContentType "application/json" | ConvertTo-Json -Compress

Write-Host "`nContainer 'heart-api' is running. Swagger UI: http://localhost:$Port/docs" -ForegroundColor Cyan
Write-Host "Stop it with: docker rm -f heart-api" -ForegroundColor DarkGray
