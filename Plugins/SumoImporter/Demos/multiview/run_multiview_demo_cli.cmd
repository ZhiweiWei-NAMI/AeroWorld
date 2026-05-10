@echo off
setlocal
set SCRIPT_DIR=%~dp0
call conda run -n aeroagentsim python "%SCRIPT_DIR%play_multiview_demo_cli.py"
endlocal
