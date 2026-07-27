@echo off
title INR Cash Counter - Model 3 (ONNX)
cd /d "%~dp0"

echo ============================================
echo   INR Cash Counter - Model 3 (best3.onnx)
echo   Running on ONNX Runtime (low CPU usage)
echo ============================================
echo.

:: Activate virtual environment
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo [WARNING] Virtual env not found, trying system Python...
)

:: Run with optimized settings for ONNX CPU inference
python live_count.py ^
    --model models/best3.onnx ^
    --conf 0.35 ^
    --detect_every 2 ^
    --threads 2 ^
    --mode auto ^
    --fps 30 ^
    --verify_time 0.5

echo.
echo App closed. Press any key to exit.
pause >nul
