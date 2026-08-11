"""FA-ASMR EXE 构建脚本 — 后台运行 PyInstaller

零删除策略: 构建产物输出到全新的临时 workpath/distpath (PyInstaller 只创建不删除,
不会触发批量删除安全拦截)。构建完成后仅把新的 FA-ASMR.exe / _internal / tkdnd
【合并覆盖】进现有 dist/FA-ASMR, 绝不 rmtree 整个 dist (保护 model.pt / 配置 / 脚本)。
"""
import subprocess, sys, os, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
DIST = os.path.join(HERE, "dist", "FA-ASMR")
# 全新临时目录, 确保 PyInstaller 不会去 rmtree 已存在的输出 (避免安全拦截)
_RUN = "build_run"
WORKPATH = os.path.join(HERE, _RUN, "work")
DISTPATH = os.path.join(HERE, _RUN, "out")

os.makedirs(WORKPATH, exist_ok=True)
os.makedirs(DISTPATH, exist_ok=True)

cmd = [
    sys.executable, "-m", "PyInstaller",
    "--onedir", "--windowed",
    "--name", "FA-ASMR",
    "--workpath", WORKPATH,
    "--distpath", DISTPATH,
    # 子目录
    "--add-data", f"tasks{os.pathsep}tasks",
    "--add-data", f"furigana{os.pathsep}furigana",
    "--add-data", f"lora{os.pathsep}lora",
    # MMS_FA 权重(model.pt ~1.2GB)不进包, 构建后手动放到 dist/FA-ASMR/models/ (exe 同级)
    "--add-data", f"embedded_torch{os.pathsep}embedded_torch",
    # 数据文件
    "--add-data", f"ono_table.json{os.pathsep}.",
    "--add-data", f"fa_asmr_settings.json{os.pathsep}.",
    # 第三方库
    "--collect-all", "tkinterdnd2",
    "--collect-all", "pykakasi",
    "--collect-all", "janome",
    "--collect-all", "fugashi",
    "--collect-all", "sudachidict_core",
    "--collect-all", "sudachipy",
    # hidden imports
    "--hidden-import", "soundfile",
    "--hidden-import", "scipy",
    "--hidden-import", "jaconv",
    "--hidden-import", "align_model",
    "--hidden-import", "align_utils",
    "--hidden-import", "align_post",
    "--hidden-import", "advanced_config",
    "--hidden-import", "lrc_end_extend",
    # torch 被 --exclude-module 排除后, 其启动所需的 timeit 等标准库不会被收集,
    # 这里显式收集, 避免冻结环境 "No module named 'timeit'"
    "--hidden-import", "timeit",
    # 排除无用包
    "--exclude-module", "matplotlib",
    "--exclude-module", "pandas",
    "--exclude-module", "PIL",
    "--exclude-module", "cv2",
    "--exclude-module", "nltk",
    "--exclude-module", "pyphen",
    "--exclude-module", "pypinyin",
    # torch/torchaudio 由 torch_bootstrap.py 运行时从系统 Python / embedded_torch 借用
    "--exclude-module", "torch",
    "--exclude-module", "torchaudio",
    # 以下即便被间接引用也绝不打包: 它们是 CUDA torch / VAD 生态的庞然大物
    "--exclude-module", "peft",
    "--exclude-module", "transformers",
    "--exclude-module", "tokenizers",
    "--exclude-module", "huggingface_hub",
    "--exclude-module", "torchvision",
    "--exclude-module", "onnxruntime",
    "--exclude-module", "onnx",
    # scipy/sklearn 的 array_api_compat.torch 子模块会绕过上面的 --exclude-module torch
    # 被贪婪收集进包（2.3GB CUDA torch 泄漏），这里单独排除它们的 torch 兼容层
    "--exclude-module", "scipy._external.array_api_compat.torch",
    "--exclude-module", "sklearn.externals.array_api_compat.torch",
    "--noconfirm",
    "FA-ASMR_GUI.py",
]

# ── 关键修复: 把 embedded torch 启动所需的【全部标准库模块】强制收进包 ──
# --exclude-module torch 让 PyInstaller 跳过 torch 依赖分析, 导致 timeit / pickletools
# 等标准库缺失, 干净机直接 ModuleNotFoundError (已实测连续报 timeit→pickletools)。
# 清单由 _collect_stdlib.py 实际 import embedded torch 扫描得出, 一次修全。
_stdlib_file = os.path.join(HERE, "_stdlib_imports.txt")
if os.path.isfile(_stdlib_file):
    _n = 0
    with open(_stdlib_file, encoding="utf-8") as _f:
        for _m in _f.read().splitlines():
            _m = _m.strip()
            if _m:
                cmd += ["--hidden-import", _m]
                _n += 1
    print(f"[build] 已追加 {_n} 个标准库 hidden-import (torch 依赖兜底)")
else:
    print("[WARN] 未找到 _stdlib_imports.txt，请先运行 _collect_stdlib.py")

print("[build] 开始打包...")
print(f"[build] 命令: {' '.join(cmd)}")
print("[build] 此过程约需 3-5 分钟，请耐心等待...")
sys.stdout.flush()

result = subprocess.run(cmd, cwd=HERE, env=os.environ.copy())

if result.returncode == 0:
    SRC = os.path.join(DISTPATH, "FA-ASMR")
    if not os.path.isdir(SRC):
        print(f"[FAIL] 未找到构建产物: {SRC}")
        sys.exit(1)
    print(f"[OK] 打包成功 → {SRC}\\FA-ASMR.exe")

    # ── 零删除合并: 仅覆盖冻结代码, 保留 model.pt / 配置 / 脚本 ──
    # copytree(dir_exist_ok=True) 只覆盖/新增, 不删除旧文件 → 不触发批量删除拦截
    _exe_src = os.path.join(SRC, "FA-ASMR.exe")
    if os.path.isfile(_exe_src):
        shutil.copy2(_exe_src, os.path.join(DIST, "FA-ASMR.exe"))
        print(f"[OK] 已更新 FA-ASMR.exe → {DIST}")
    for _item in ["_internal", "tkdnd"]:
        _s = os.path.join(SRC, _item)
        _d = os.path.join(DIST, _item)
        if os.path.isdir(_s):
            shutil.copytree(_s, _d, dirs_exist_ok=True)
            print(f"[OK] 已合并 {_item} → {_d}")

    # 复制 tkinterdnd2 DLL（PyInstaller 可能遗漏）
    try:
        import tkinterdnd2
        dnd_dir = os.path.dirname(tkinterdnd2.__file__)
        tkdnd_src = os.path.join(dnd_dir, "tkdnd")
        if os.path.isdir(tkdnd_src):
            tkdnd_dst = os.path.join(DIST, "tkdnd")
            shutil.copytree(tkdnd_src, tkdnd_dst, dirs_exist_ok=True)
            print(f"[OK] 已复制 tkdnd DLL → {tkdnd_dst}")
    except Exception as e:
        print(f"[WARN] 复制 tkdnd DLL 失败: {e}")
    # 复制底层系统配置模板到 exe 同级（advanced_config.json）
    # 仅当目标不存在时复制，避免覆盖用户已手编的配置
    _cfg_src = os.path.join(HERE, "advanced_config.json")
    _cfg_dst = os.path.join(DIST, "advanced_config.json")
    if os.path.isfile(_cfg_src):
        if not os.path.exists(_cfg_dst):
            shutil.copy2(_cfg_src, _cfg_dst)
            print(f"[OK] 已复制 advanced_config.json 模板 → {_cfg_dst}")
        else:
            print(f"[INFO] 已存在 {_cfg_dst}，保留用户配置（如需恢复默认可重新复制源码模板）")

    # 复制 GUI 设置模板到 exe 同级（fa_asmr_settings.json）
    # 冻结模式下设置在 exe 同级读写，故模板必须落在该处
    _set_src = os.path.join(HERE, "fa_asmr_settings.json")
    _set_dst = os.path.join(DIST, "fa_asmr_settings.json")
    if os.path.isfile(_set_src):
        if not os.path.exists(_set_dst):
            shutil.copy2(_set_src, _set_dst)
            print(f"[OK] 已复制 fa_asmr_settings.json 模板 → {_set_dst}")
        else:
            print(f"[INFO] 已存在 {_set_dst}，保留用户配置")

    # 复制中文使用说明到 exe 同级（论坛分发时用户直接可见）
    _readme_src = os.path.join(HERE, "使用说明.txt")
    _readme_dst = os.path.join(DIST, "使用说明.txt")
    if os.path.isfile(_readme_src):
        shutil.copy2(_readme_src, _readme_dst)
        print(f"[OK] 已复制 使用说明.txt → {_readme_dst}")

    # 复制启动器 bat（校验 _internal 是否齐全，防止只复制 exe 单个文件导致静默失败）
    _launcher_src = os.path.join(HERE, "运行FA-ASMR.bat")
    _launcher_dst = os.path.join(DIST, "运行FA-ASMR.bat")
    if os.path.isfile(_launcher_src):
        shutil.copy2(_launcher_src, _launcher_dst)
        print(f"[OK] 已复制 运行FA-ASMR.bat → {_launcher_dst}")

else:
    print(f"[FAIL] 打包失败，返回码: {result.returncode}")
    sys.exit(1)
