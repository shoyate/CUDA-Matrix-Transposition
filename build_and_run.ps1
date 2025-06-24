# PowerShell script to build and run CUDA matrix transposition
Write-Host "Setting up Visual Studio 2019 Build Tools environment..." -ForegroundColor Green

# Import Visual Studio 2019 Build Tools environment
$vsPath = "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

if (Test-Path $vsPath) {
    Write-Host "Found Visual Studio Build Tools at: $vsPath" -ForegroundColor Yellow
    
    # Create a temporary batch file to capture environment variables
    $tempBat = "$env:TEMP\setup_vs_env.bat"
    $tempOut = "$env:TEMP\vs_env_vars.txt"
    
    @"
@echo off
call "$vsPath" > nul 2>&1
set > "$tempOut"
"@ | Out-File -FilePath $tempBat -Encoding ASCII
    
    # Execute the batch file
    & cmd /c $tempBat
    
    # Parse and set environment variables
    if (Test-Path $tempOut) {
        Get-Content $tempOut | ForEach-Object {
            if ($_ -match '^([^=]+)=(.*)$') {
                $name = $matches[1]
                $value = $matches[2]
                [System.Environment]::SetEnvironmentVariable($name, $value, "Process")
            }
        }
        Remove-Item $tempOut -Force
    }
    Remove-Item $tempBat -Force
    
    Write-Host "Visual Studio environment variables loaded" -ForegroundColor Green
} else {
    Write-Host "Visual Studio Build Tools not found at expected location!" -ForegroundColor Red
    Write-Host "Please ensure Visual Studio 2019 Build Tools are installed" -ForegroundColor Red
    exit 1
}

Write-Host "`nCompiling CUDA matrix transposition program..." -ForegroundColor Green
Write-Host "Command: nvcc -O3 -arch=sm_50 -o matrix_transpose.exe main.cu" -ForegroundColor Yellow

try {
    & nvcc -O3 -arch=sm_50 -o matrix_transpose.exe main.cu
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Compilation successful!" -ForegroundColor Green
        
        if (Test-Path "matrix_transpose.exe") {
            Write-Host "`nRunning the program..." -ForegroundColor Green
            Write-Host "================================" -ForegroundColor Cyan
            & .\matrix_transpose.exe
            Write-Host "`nProgram execution completed." -ForegroundColor Green
        } else {
            Write-Host "Executable not found after compilation!" -ForegroundColor Red
        }
    } else {
        Write-Host "Compilation failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    }
} catch {
    Write-Host "Error during compilation: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`nPress any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
