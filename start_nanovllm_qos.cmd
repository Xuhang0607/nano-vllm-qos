@echo off
setlocal

rem Double-click this file to start the nano-vLLM QoS service.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_nanovllm_qos.ps1"

echo.
echo [nano-vLLM QoS] Launcher exited.
pause
