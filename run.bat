@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

echo [FA-ASMR] 正在启动 GUI 界面...
python FA-ASMR_GUI.py
if %errorlevel% neq 0 (
    echo.
    echo [错误] 程序异常退出，错误代码: %errorlevel%
    pause
)
