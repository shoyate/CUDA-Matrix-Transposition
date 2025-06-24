@echo off
echo Setting up Visual Studio 2019 Build Tools environment...

REM Set up Visual Studio 2019 Build Tools environment
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

if %ERRORLEVEL% neq 0 (
    echo Failed to set up Visual Studio environment
    pause
    exit /b 1
)

echo Visual Studio environment set up successfully
echo.

echo Compiling final CUDA matrix transposition program...
nvcc -O3 -arch=sm_50 -std=c++11 -o matrix_transpose_final.exe main_final.cu

if %ERRORLEVEL% neq 0 (
    echo Compilation failed!
    pause
    exit /b 1
)

echo Compilation successful!
echo.

echo Running the final optimized program...
echo ====================================
matrix_transpose_final.exe

echo.
echo Program execution completed.
pause
