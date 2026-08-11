@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

echo ============================================
echo   FA-ASMR 启动器
echo ============================================

if not exist "FA-ASMR.exe" (
  echo.
  echo [错误] 当前目录找不到 FA-ASMR.exe
  echo 请确认本启动器与 FA-ASMR.exe 放在同一文件夹。
  echo.
  pause
  exit /b 1
)

if not exist "_internal" (
  echo.
  echo [错误] 缺少 _internal 文件夹！
  echo.
  echo 必须把「整个 FA-ASMR 文件夹」一起复制（包含 FA-ASMR.exe 和 _internal 子文件夹），
  echo 只复制 FA-ASMR.exe 单个文件是无法运行的，会一闪而过或毫无反应。
  echo.
  pause
  exit /b 1
)

if not exist "models\hub\checkpoints\model.pt" (
  echo.
  echo [提示] 未检测到 MMS-FA 权重 model.pt（首次打开界面不受影响，对齐时才需要）。
  echo   GPU 版已内置 CUDA torch，有 N 卡开箱即用；CPU 版为纯 CPU 推理。
  echo   如需其它 CUDA 构建，可自行从源码构筑。
  echo.
)

echo 正在启动 FA-ASMR ...
start "" "FA-ASMR.exe"
