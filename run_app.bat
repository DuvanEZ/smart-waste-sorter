@echo off
REM ===================================================================
REM  Smart Waste Sorter - one-click launcher for Windows
REM  1st run: creates a virtual environment and installs the libraries
REM  (needs internet, about 1-3 minutes). Later runs start immediately.
REM ===================================================================
cd /d "%~dp0"
set PY=python
where py >nul 2>nul && set PY=py -3
%PY% --version >nul 2>nul
if errorlevel 1 (
    echo Python 3.10 or newer is required: https://www.python.org/downloads/
    pause
    exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    %PY% -m venv .venv
    if errorlevel 1 ( echo Could not create the virtual environment. & pause & exit /b 1 )
)
if not exist ".venv\installed.ok" (
    echo Installing the required libraries - first run only...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 ( echo Installing the libraries failed - check your internet connection. & pause & exit /b 1 )
    echo ok> ".venv\installed.ok"
)
echo.
echo Starting Smart Waste Sorter... your browser will open at http://localhost:8501
echo (If it does not, copy the "Local URL" shown below into your browser.)
echo Close this window to stop the application.
start "" /b cmd /c "timeout /t 5 >nul & start http://localhost:8501"
".venv\Scripts\python.exe" -m streamlit run app\streamlit_app.py
pause
