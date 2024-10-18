@echo off

git pull

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

echo Installing pip:
%PYTHON% -m pip install pip==24.0
echo Installing pip-tools:
%PYTHON% -m pip install pip-tools

set /p COMPILE_REQ="[REQUIRED FOR THE FIRST LAUNCH OF INSTALL.BAT] Compile requirements? (Y/N): "

if /I "%COMPILE_REQ%"=="Y" (
    echo Compiling requirements:
    %PYTHON% -m piptools compile requirements.in
) else (
    echo Skipping requirements compilation.
)

echo Installing requirements:
%PYTHON% -m pip install -r requirements.txt
echo Installing torch:
%PYTHON% -m pip install torch torchvision==0.16.2 torchaudio --index-url https://download.pytorch.org/whl/cu121

echo Running app.py:
%PYTHON% app.py %*

pause