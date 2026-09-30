@echo off
rem THALAMUS launcher for Windows: double-click to install (first run) and open THALAMUS in your browser.
setlocal
cd /d "%~dp0"
title THALAMUS
echo.
echo   THALAMUS
echo   --------

set "PY="
call :try_python py -3
if not defined PY call :try_python python
if not defined PY call :try_python "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
if not defined PY call :try_python "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if defined PY goto :have_python

echo   THALAMUS needs Python 3.11 or newer, and it isn't installed yet.
where winget >nul 2>nul
if errorlevel 1 goto :manual_python
echo   Installing Python 3.12 with winget (this may take a few minutes)...
winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
call :try_python "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if defined PY goto :have_python
echo.
echo   Python was installed. Please close this window and double-click Start-THALAMUS again.
goto :fail

:manual_python
echo   Opening python.org. Install the latest Python 3 (tick "Add python.exe to PATH"),
echo   then double-click Start-THALAMUS again.
start "" "https://www.python.org/downloads/"
goto :fail

:have_python
if exist ".venv\Scripts\python.exe" goto :run
echo   Creating a private Python environment (.venv)...
%PY% -m venv .venv
if errorlevel 1 (
  echo   Couldn't create the environment.
  goto :fail
)

:run
".venv\Scripts\python.exe" scripts\launch.py
if errorlevel 1 goto :fail
exit /b 0

:try_python
%* -c "import sys; sys.exit(sys.version_info < (3, 11))" >nul 2>nul
if not errorlevel 1 set "PY=%*"
exit /b 0

:fail
echo.
pause
exit /b 1
