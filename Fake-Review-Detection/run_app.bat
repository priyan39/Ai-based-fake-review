@echo off
setlocal
cd /d "%~dp0"
title Fake Review Detection - Web App Server

echo ========================================================
echo   AI-Based Fake Review Detection - Setup and Launch
echo ========================================================
echo.

:: Detect python executable
set "PY_CMD="
where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set "PY_CMD=python"
    goto :found_py
)

where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set "PY_CMD=py"
    goto :found_py
)

:: Search standard Windows install locations
for %%V in (313 312 311 310 39 38) do (
    if exist "%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe" (
        set "PY_CMD=%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe"
        goto :found_py
    )
    if exist "%ProgramFiles%\Python%%V\python.exe" (
        set "PY_CMD=%ProgramFiles%\Python%%V\python.exe"
        goto :found_py
    )
    if exist "C:\Python%%V\python.exe" (
        set "PY_CMD=C:\Python%%V\python.exe"
        goto :found_py
    )
)

if exist "%USERPROFILE%\anaconda3\python.exe" (
    set "PY_CMD=%USERPROFILE%\anaconda3\python.exe"
    goto :found_py
)
if exist "%USERPROFILE%\miniconda3\python.exe" (
    set "PY_CMD=%USERPROFILE%\miniconda3\python.exe"
    goto :found_py
)
if exist "C:\ProgramData\anaconda3\python.exe" (
    set "PY_CMD=C:\ProgramData\anaconda3\python.exe"
    goto :found_py
)

:found_py
if not defined PY_CMD (
    echo [ERROR] Python is not installed or not in your system PATH!
    echo.
    echo To fix this:
    echo 1. Download Python: https://www.python.org/downloads/
    echo 2. Run the installer and CRITICALLY check the box:
    echo    "[X] Add python.exe to PATH"
    echo 3. Click "Install Now", then re-run this file.
    echo.
    pause
    exit /b 1
)

echo [OK] Using Python: %PY_CMD%
echo.

echo [1/3] Checking and installing dependencies...
%PY_CMD% -m pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Pip returned code %ERRORLEVEL%. Continuing if packages already exist...
)

echo.
echo [2/3] Checking for trained model...
if not exist "models\best_model.pkl" (
    echo Model not found. Training model now on data\reviews.csv...
    %PY_CMD% -m src.train
    if %ERRORLEVEL% NEQ 0 (
        echo.
        echo [ERROR] Training failed. See error above.
        pause
        exit /b %ERRORLEVEL%
    )
) else (
    echo Trained model found at models\best_model.pkl.
)

echo.
echo [3/3] Starting Streamlit server...
echo URL: http://localhost:8501
echo.
echo *** KEEP THIS COMMAND PROMPT WINDOW OPEN ***
echo (Closing this window will stop localhost)
echo.

%PY_CMD% -m streamlit run app.py --server.port 8501
pause
