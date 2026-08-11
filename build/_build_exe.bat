@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================
echo   FA-ASMR PyInstaller 打包
echo ========================================
echo.

REM 确认 pyinstaller 可用
python -m PyInstaller --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] 未找到 PyInstaller，请先安装: pip install pyinstaller
    pause
    exit /b 1
)

REM 确认 tkinterdnd2 已安装
python -c "from tkinterdnd2 import TkinterDnD" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] 未找到 tkinterdnd2，请先安装: pip install tkinterdnd2
    pause
    exit /b 1
)

echo [1/2] 清理旧构建...
if exist "build" rmdir /s /q "build"
if exist "dist\FA-ASMR" rmdir /s /q "dist\FA-ASMR"

echo [2/2] 开始打包（约 3-5 分钟，取决于机器性能）...
echo.

python _build_exe.py

if %ERRORLEVEL% equ 0 (
    echo.
    echo ========================================
    echo   打包成功!
    echo   输出: dist\FA-ASMR\FA-ASMR.exe
    echo ========================================

    REM 复制 tkinterdnd2 DLL（PyInstaller 可能遗漏）
    for /f "tokens=*" %%i in ('python -c "import tkinterdnd2, os; print(os.path.dirname(tkinterdnd2.__file__))"') do set DND_DIR=%%i
    if exist "%DND_DIR%\tkdnd" (
        xcopy /e /y "%DND_DIR%\tkdnd" "dist\FA-ASMR\tkdnd\" >nul 2>&1
        echo 已复制 tkinterdnd2 DLL 到 dist\FA-ASMR\
    )

    REM 复制配置模板到 exe 同级（advanced_config.json / fa_asmr_settings.json）
    if exist "advanced_config.json" (
        if not exist "dist\FA-ASMR\advanced_config.json" (
            copy /y "advanced_config.json" "dist\FA-ASMR\" >nul 2>&1
            echo 已复制 advanced_config.json 模板到 dist\FA-ASMR\
        )
    )
    if exist "fa_asmr_settings.json" (
        if not exist "dist\FA-ASMR\fa_asmr_settings.json" (
            copy /y "fa_asmr_settings.json" "dist\FA-ASMR\" >nul 2>&1
            echo 已复制 fa_asmr_settings.json 模板到 dist\FA-ASMR\
        )
    )
    if exist "使用说明.txt" (
        copy /y "使用说明.txt" "dist\FA-ASMR\" >nul 2>&1
        echo 已复制 使用说明.txt 到 dist\FA-ASMR\
    )

    start explorer "dist\FA-ASMR"
) else (
    echo.
    echo ========================================
    echo   打包失败! 请查看上方错误信息
    echo ========================================
)

pause
