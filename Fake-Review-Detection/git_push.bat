@echo off
setlocal
cd /d "C:\priyan"
title Git Push - AI-Based Fake Review Detection

echo ========================================================
echo   Pushing changes to GitHub (origin/main)
echo ========================================================
echo.

where git >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Git was not found in PATH!
    echo Please install Git from: https://git-scm.com/
    pause
    exit /b 1
)

echo [1/3] Staging modified files...
git add .

echo.
echo [2/3] Committing changes...
git commit -m "Update app launcher and add resilient model fallback"
if %ERRORLEVEL% NEQ 0 (
    echo (No new changes to commit or already committed)
)

echo.
echo [3/3] Pushing to origin main...
git push -u origin main
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================================
    echo   [SUCCESS] Code successfully pushed to GitHub!
    echo ========================================================
) else (
    echo.
    echo [NOTE] Push failed. If GitHub prompts for credentials, please sign in or paste your Personal Access Token.
)

echo.
pause
