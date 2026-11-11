@echo off
setlocal EnableExtensions DisableDelayedExpansion
rem Launch the project environment and the package-local Jupyter configuration.
set "IEMOCAP_EXIT_CODE=0"
set "IEMOCAP_PAUSE=1"
set "IEMOCAP_CHECK_ONLY=0"
set "IEMOCAP_PUSHED=0"
set "IEMOCAP_OLD_CP="
:parse_args
if "%~1"=="" goto args_ready
if /i "%~1"=="--non-interactive" goto non_interactive
if /i "%~1"=="--check-only" goto check_only
echo ERROR: Unsupported option. Use --non-interactive and/or --check-only.
set "IEMOCAP_EXIT_CODE=2"
goto finish

:non_interactive
set "IEMOCAP_PAUSE=0"
shift /1
goto parse_args

:check_only
set "IEMOCAP_CHECK_ONLY=1"
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
if not exist ".venv\Scripts\python.exe" goto missing_venv
if not exist "run_demo.py" goto missing_package
if not exist "jupyter_lab_config.py" goto missing_package

set "IEMOCAP_STEP=checking the project Python version and architecture"
".venv\Scripts\python.exe" -c "import struct, sys; from pathlib import Path; ok = sys.version_info[:2] == (3, 10) and struct.calcsize('P') == 8 and Path(sys.prefix).resolve() == Path('.venv').resolve(); print('Python:', sys.executable); print('Version:', sys.version.split()[0], '| Bits:', struct.calcsize('P') * 8); sys.exit(0 if ok else 1)"
if errorlevel 1 goto command_failed

rem Check both the installed kernel file and Jupyter's resolved kernel path.
set "IEMOCAP_STEP=checking the project Jupyter kernel and notebook paths"
".venv\Scripts\python.exe" -c "import json, sys; from pathlib import Path; from jupyter_client.kernelspec import KernelSpecManager; name = 'iemocap-demo-windows'; expected = Path(sys.prefix).resolve() / 'share' / 'jupyter' / 'kernels' / name; saved = json.loads((expected / 'kernel.json').read_text(encoding='utf-8')); spec = KernelSpecManager().get_kernel_spec(name); notebook = Path('\ud604\uc7a5\uc2dc\uc5f0.ipynb').resolve(); ok = Path(spec.resource_dir).resolve() == expected and Path(spec.argv[0]).resolve() == Path(sys.executable).resolve() and Path(saved['argv'][0]).resolve() == Path(sys.executable).resolve() and notebook.is_file(); print('Kernel:', name); print('Kernel directory:', spec.resource_dir); print('Kernel Python:', spec.argv[0]); print('Notebook:', notebook); sys.exit(0 if ok else 1)"
if errorlevel 1 goto command_failed

set "IEMOCAP_STEP=checking the installed CPU PyTorch build"
".venv\Scripts\python.exe" -c "import sys, torch; print('PyTorch:', torch.__version__, '| CUDA build:', torch.version.cuda); sys.exit(0 if torch.__version__ == '2.7.1+cpu' and torch.version.cuda is None else 1)"
if errorlevel 1 goto command_failed

echo Verifying the prepared package...
set "IEMOCAP_STEP=verifying the prepared package"
".venv\Scripts\python.exe" "run_demo.py" verify
if errorlevel 1 goto command_failed
if "%IEMOCAP_CHECK_ONLY%"=="1" goto checks_passed

echo Opening JupyterLab. Keep this window open while using the notebook.
set "IEMOCAP_STEP=running JupyterLab"
".venv\Scripts\python.exe" -m jupyterlab --config "%~dp0jupyter_lab_config.py"
if errorlevel 1 goto command_failed
goto finish

:checks_passed
echo.
echo PASS: Project Python, kernel paths, CPU PyTorch, and package verification.
goto finish

:missing_venv
echo ERROR: The project Windows environment is not ready.
echo Run setup_windows.cmd before start_windows.cmd.
set "IEMOCAP_EXIT_CODE=1"
goto finish

:missing_package
echo ERROR: run_demo.py or jupyter_lab_config.py is missing from this folder.
set "IEMOCAP_EXIT_CODE=1"
goto finish

:directory_failed
echo ERROR: Cannot open the folder containing start_windows.cmd.
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
echo Read the error above. JupyterLab was not continued after this failure.

:finish
if "%IEMOCAP_PUSHED%"=="1" popd
if "%IEMOCAP_PAUSE%"=="1" pause
if defined IEMOCAP_OLD_CP chcp %IEMOCAP_OLD_CP% >nul
endlocal & exit /b %IEMOCAP_EXIT_CODE%
