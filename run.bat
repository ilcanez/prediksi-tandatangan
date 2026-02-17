@echo off
REM BSI Signature Detection - Quick Start Script (Windows)
REM This script will setup and run the Streamlit app

echo.
echo 🖊️  BSI Signature Detection System
echo ====================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed!
    echo Please install Python 3.10 or higher from python.org
    pause
    exit /b 1
)

echo ✅ Python found
python --version
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo 📦 Creating virtual environment...
    python -m venv venv
    echo ✅ Virtual environment created
    echo.
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo 📦 Installing dependencies...
python -m pip install -q --upgrade pip
pip install -q -r requirements.txt

echo ✅ Dependencies installed
echo.

REM Check if model file exists
if not exist "signature_model_final.keras" (
    echo ⚠️  WARNING: Model file not found!
    echo.
    echo Please ensure 'signature_model_final.keras' is in the current directory
    echo You can:
    echo   1. Copy it from Google Drive
    echo   2. Download it using the notebook
    echo   3. Use Google Drive download in app (see DEPLOYMENT_GUIDE.md^)
    echo.
    set /p continue="Continue anyway? (y/n): "
    if /i not "%continue%"=="y" exit /b 1
)

REM Check if label_map.json exists
if not exist "label_map.json" (
    echo ⚠️  WARNING: label_map.json not found!
    echo Please copy it from Google Drive
    echo.
    set /p continue="Continue anyway? (y/n): "
    if /i not "%continue%"=="y" exit /b 1
)

REM Run Streamlit app
echo 🚀 Starting Streamlit app...
echo.
echo App will open at: http://localhost:8501
echo Press Ctrl+C to stop
echo.

streamlit run app.py

pause
