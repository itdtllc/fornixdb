@echo off
rem FornixDB configuration wizard - launcher for Windows.
rem
rem   Double-click fornix-config.cmd, or run it from a terminal.
rem   No arguments needed: the wizard finds your store from the machine
rem   registry and asks which to configure if you have more than one.
rem   Advanced: fornix-config.cmd --db C:\path\to\other.db
setlocal

set "HERE=%~dp0"

rem Prefer the repo venv interpreter (has the deps); else a Python on PATH.
set "PY=%HERE%.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=py"

set "PYTHONPATH=%HERE%;%PYTHONPATH%"

rem Any passthrough args (e.g. --db) are GLOBAL options and so come before the
rem configure subcommand. With no args this just runs the configure wizard.
"%PY%" -m fornixdb %* configure
set "RC=%ERRORLEVEL%"

rem Keep the window open when double-clicked so the result stays readable.
if "%~1"=="" pause
endlocal & exit /b %RC%
