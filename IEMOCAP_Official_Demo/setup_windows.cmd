@echo off
setlocal EnableExtensions DisableDelayedExpansion
rem Use only this package's virtual environment. No activation is required.
set "IEMOCAP_EXIT_CODE=0"
set "IEMOCAP_PAUSE=1"
set "IEMOCAP_PUSHED=0"
set "IEMOCAP_OLD_CP="
:parse_args
if "%~1"=="" goto args_ready
if /i "%~1"=="--non-interactive" goto non_interactive
echo ERROR: Unsupported option. Use --non-interactive to skip the final pause.
set "IEMOCAP_EXIT_CODE=2"
goto finish

:non_interactive
set "IEMOCAP_PAUSE=0"
shift /1
goto parse_args

:args_ready
for /f "tokens=2 delims=:" %%C in ('chcp') do set "IEMOCAP_OLD_CP=%%C"
chcp 65001 >nul
if errorlevel 1 goto code_page_failed
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "JUPYTER_PREFER_ENV_PATH=1"
pushd "%~dp0"
if errorlevel 1 goto directory_failed
set "IEMOCAP_PUSHED=1"
if not exist "requirements.txt" goto missing_package
if not exist "run_demo.py" goto missing_package
if exist ".venv\Scripts\python.exe" goto venv_ready
if exist ".venv" goto invalid_venv

rem Prefer the Windows Python launcher; fall back to a verified python command.
py -3.10 -c "import struct, sys; sys.exit(0 if sys.version_info[:2] == (3, 10) and struct.calcsize('P') == 8 else 1)" >nul 2>&1
if not errorlevel 1 goto create_with_py
python -c "import struct, sys; sys.exit(0 if sys.version_info[:2] == (3, 10) and struct.calcsize('P') == 8 else 1)" >nul 2>&1
if not errorlevel 1 goto create_with_python
echo ERROR: Python 3.10 64-bit was not found by py -3.10 or python.
echo Install Python 3.10 64-bit, then run setup_windows.cmd again.
set "IEMOCAP_EXIT_CODE=1"
goto finish

:create_with_py
set "IEMOCAP_STEP=creating the project virtual environment with py -3.10"
py -3.10 -m venv ".venv"
if errorlevel 1 goto command_failed
goto venv_ready

:create_with_python
set "IEMOCAP_STEP=creating the project virtual environment with python"
python -m venv ".venv"
if errorlevel 1 goto command_failed

:venv_ready
set "IEMOCAP_STEP=checking the project Python version and architecture"
".venv\Scripts\python.exe" -c "import struct, sys; from pathlib import Path; ok = sys.version_info[:2] == (3, 10) and struct.calcsize('P') == 8 and Path(sys.prefix).resolve() == Path('.venv').resolve(); print('Python:', sys.executable); print('Version:', sys.version.split()[0], '| Bits:', struct.calcsize('P') * 8); sys.exit(0 if ok else 1)"
if errorlevel 1 goto command_failed

echo Installing the CPU build of PyTorch 2.7.1...
set "IEMOCAP_STEP=installing CPU PyTorch"
".venv\Scripts\python.exe" -m pip install "torch==2.7.1+cpu" --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto command_failed

echo Installing the package requirements...
set "IEMOCAP_STEP=installing requirements.txt"
".venv\Scripts\python.exe" -m pip install -r "requirements.txt"
if errorlevel 1 goto command_failed

set "IEMOCAP_STEP=checking installed package dependencies"
".venv\Scripts\python.exe" -m pip check
if errorlevel 1 goto command_failed

set "IEMOCAP_STEP=checking the installed CPU PyTorch build"
".venv\Scripts\python.exe" -c "import sys, torch; print('PyTorch:', torch.__version__, '| CUDA build:', torch.version.cuda); sys.exit(0 if torch.__version__ == '2.7.1+cpu' and torch.version.cuda is None else 1)"
if errorlevel 1 goto command_failed

rem Register the kernel inside .venv, never in the user's global kernel folder.
set "IEMOCAP_STEP=registering the project Jupyter kernel"
".venv\Scripts\python.exe" -m ipykernel install --sys-prefix --name iemocap-demo-windows --display-name "IEMOCAP Windows (Python 3.10)"
if errorlevel 1 goto command_failed

echo Verifying the prepared package...
set "IEMOCAP_STEP=verifying the prepared package"
".venv\Scripts\python.exe" "run_demo.py" verify
if errorlevel 1 goto command_failed

echo.
echo Setup completed. Run start_windows.cmd to open the notebook.
goto finish

:invalid_venv
echo ERROR: The existing .venv folder has no Windows Python executable.
echo Keep this folder intact and use a fresh package folder for Windows setup.
set "IEMOCAP_EXIT_CODE=1"
goto finish

:missing_package
echo ERROR: requirements.txt or run_demo.py is missing from this package folder.
set "IEMOCAP_EXIT_CODE=1"
goto finish

:directory_failed
echo ERROR: Cannot open the folder containing setup_windows.cmd.
set "IEMOCAP_EXIT_CODE=1"
goto finish

:code_page_failed
echo ERROR: Cannot select the UTF-8 console code page.
set "IEMOCAP_EXIT_CODE=1"
goto finish

:command_failed
set "IEMOCAP_EXIT_CODE=%errorlevel%"
if "%IEMOCAP_EXIT_CODE%"=="0" set "IEMOCAP_EXIT_CODE=1"
echo.
echo ERROR: Failed while %IEMOCAP_STEP%.
echo Read the error above. No later setup steps were executed.

:finish
if "%IEMOCAP_PUSHED%"=="1" popd
if "%IEMOCAP_PAUSE%"=="1" pause
if defined IEMOCAP_OLD_CP chcp %IEMOCAP_OLD_CP% >nul
endlocal & exit /b %IEMOCAP_EXIT_CODE%
