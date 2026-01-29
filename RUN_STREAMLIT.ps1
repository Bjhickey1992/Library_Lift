# PowerShell script to run Streamlit app
Set-Location -Path $PSScriptRoot
Write-Host "Starting Streamlit app..." -ForegroundColor Green
Write-Host "Current directory: $(Get-Location)" -ForegroundColor Cyan
Write-Host ""
streamlit run streamlit_app.py
