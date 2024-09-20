@echo off

set PYTHON=python
set VENV_DIR=%~dp0venv

mkdir tmp 2>NUL

%PYTHON% -m venv "%VENV_DIR%"
if %ERRORLEVEL% neq 0 (
    echo Unable to create venv
    pause
    exit /b
)

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Creating venv in directory %VENV_DIR%
    %PYTHON% -m venv "%VENV_DIR%"
    if %ERRORLEVEL% neq 0 (
        echo Unable to create venv
        pause
        exit /b
    )
)

set PYTHON=%VENV_DIR%\Scripts\python.exe

call "%VENV_DIR%\Scripts\activate"

%PYTHON% app.py %*

pause