# -*- coding: utf-8 -*-
"""PyTorch 自动加载引导。

分发模型 (每个分发自包含, 不需要用户另行安装 PyTorch):
  - CPU 版: 内置 CPU 版 torch (embedded_torch/), 纯 CPU。
  - GPU 版: 内置 CUDA 版 torch (embedded_torch/), 开箱即用 GPU。

需要其他 CUDA 构建的用户, 可自行从源码构筑对应版本。

DLL 路径用全局 cookie 持有，防止 GC 回收导致 .pyd 加载失败。"""

import sys
import os
import ctypes
import subprocess
import tkinter.messagebox as messagebox

TORCH_DONE = False
_DLL_COOKIES = []  # 全局持有 os.add_dll_directory 返回值
_ORIG_PATH = os.environ.get('PATH', '')  # 启动期 PATH 快照, 供回退时还原
_ADDED_SYS_PATH = []  # 我们插入 sys.path 的条目, 供回退时移除
BOOTSTRAP_NOTES = []  # 启动期提示, 供 GUI 在日志区展示


def _read_build_variant():
    """读取打包时写入的变体标记（_internal/variant.txt）。默认 'cpu'。"""
    if getattr(sys, 'frozen', False):
        _base = getattr(sys, '_MEIPASS', '')
        _vp = os.path.join(_base, 'variant.txt')
        if os.path.isfile(_vp):
            try:
                return open(_vp, encoding='utf-8').read().strip().lower()
            except Exception:
                pass
    return 'cpu'


BUILD_VARIANT = _read_build_variant()


def _note(msg):
    """记录启动期提示 (开发模式可见, GUI 可读取 BOOTSTRAP_NOTES 展示)。"""
    BOOTSTRAP_NOTES.append(msg)
    try:
        print("[FA-ASMR] " + msg, file=sys.stderr)
    except Exception:
        pass


def _write_bootstrap_log():
    """把 BOOTSTRAP_NOTES 写到 exe 同级日志文件 (windowed exe 无控制台, 便于排查)。"""
    try:
        if getattr(sys, 'frozen', False):
            _d = os.path.dirname(os.path.abspath(sys.executable))
        else:
            _d = os.getcwd()
        _p = os.path.join(_d, 'FA-ASMR_torch_bootstrap.log')
        with open(_p, 'w', encoding='utf-8') as _f:
            _f.write("FA-ASMR torch bootstrap 诊断日志\n")
            _f.write("exe: {}\n".format(sys.executable))
            _f.write("frozen: {}\n".format(getattr(sys, 'frozen', False)))
            _f.write("MEIPASS: {}\n".format(getattr(sys, '_MEIPASS', 'None')))
            _f.write("BUILD_VARIANT: {}\n".format(BUILD_VARIANT))
            _f.write("PATH: {}\n\n".format(os.environ.get('PATH', '')))
            _f.write("\n".join(BOOTSTRAP_NOTES) if BOOTSTRAP_NOTES else "(无记录)")
    except Exception:
        pass


def _has_nvidia_gpu():
    """检测系统是否有可用的 NVIDIA GPU（仅用于日志提示，不影响加载逻辑）。"""
    try:
        r = subprocess.run(
            ['nvidia-smi'],
            capture_output=True, text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0,
            timeout=10,
        )
        return r.returncode == 0
    except Exception:
        return False


def _clear_torch_env():
    """撤销对环境的所有修改 (sys.path / DLL 目录 / PATH)。"""
    global _DLL_COOKIES, _ADDED_SYS_PATH

    # 撤销已注册的 DLL 搜索目录 (这些目录优先级高于 PATH, 必须显式移除)
    for _c in list(_DLL_COOKIES):
        try:
            os.remove_dll_directory(_c)
        except Exception:
            pass
    _DLL_COOKIES = []

    # 移除我们插入的 sys.path 条目
    for _p in list(_ADDED_SYS_PATH):
        while _p in sys.path:
            sys.path.remove(_p)
    _ADDED_SYS_PATH.clear()

    # 还原 PATH
    os.environ['PATH'] = _ORIG_PATH


def _pop_torch_modules():
    """从 sys.modules 移除整个 torch / torchaudio 命名空间, 确保重新加载时不被旧安装复用。"""
    _gone = [k for k in sys.modules
             if k == 'torch' or k.startswith('torch.') or
                k == 'torchaudio' or k.startswith('torchaudio.')]
    for k in _gone:
        sys.modules.pop(k, None)


def _try_embedded_torch():
    """加载内置 torch（embedded_torch/）。成功返回 True。

    CPU 版此处为 CPU torch，GPU 版此处为 CUDA torch；二者加载方式一致。
    """
    if not getattr(sys, 'frozen', False):
        return False  # 开发模式不走此路径

    _clear_torch_env()

    # _MEIPASS = _internal/ 目录；embedded_torch/ 在其中
    base = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
    emb = os.path.abspath(os.path.join(base, 'embedded_torch'))
    if not os.path.isdir(emb):
        _note(f"内置 torch 目录不存在: {emb}")
        return False

    # 关键: 清除上一路径加载的 torch/torchaudio 整个命名空间, 否则会复用旧扩展模块
    _pop_torch_modules()

    # 在 frozen 环境下, PyInstaller 的 FrozenImporter 排在 sys.meta_path 前面,
    # 可能先处理 import 请求并导致异常。把 PathFinder 提到最前, 让文件系统优先。
    try:
        import importlib.machinery
        path_finder = None
        for finder in sys.meta_path:
            if isinstance(finder, importlib.machinery.PathFinder):
                path_finder = finder
                break
        if path_finder and sys.meta_path[0] is not path_finder:
            sys.meta_path.remove(path_finder)
            sys.meta_path.insert(0, path_finder)
    except Exception:
        pass

    if emb not in sys.path:
        sys.path.insert(0, emb)
        _ADDED_SYS_PATH.append(emb)

    # 注册所有可能的 DLL 搜索目录 (按优先级): torch/lib > embedded_torch/bin > _internal 根目录
    # 干净 Windows 可能未安装 VC++ Redistributable, 必须把运行时所在目录纳入搜索范围,
    # 否则 torch_python.dll 找不到 vcruntime140.dll 会直接导入失败。
    _dll_dirs = []
    torch_lib = os.path.join(emb, 'torch', 'lib')
    if os.path.isdir(torch_lib):
        _dll_dirs.append(torch_lib)
    emb_bin = os.path.join(emb, 'bin')
    if os.path.isdir(emb_bin):
        _dll_dirs.append(emb_bin)
    # _MEIPASS (=_internal) 根目录也含 PyInstaller 打包的 VC++ 运行时
    if base not in _dll_dirs and os.path.isdir(base):
        _dll_dirs.append(base)

    global _DLL_COOKIES
    for _d in _dll_dirs:
        try:
            cookie = os.add_dll_directory(_d)
            _DLL_COOKIES.append(cookie)
        except Exception as _e:
            _note(f"add_dll_directory 失败({_d}): {_e}")
    # 把上述目录前置到 PATH (某些 .pyd 加载时只查 PATH)
    _orig_path = os.environ.get('PATH', '')
    _prepend = os.pathsep.join([d for d in _dll_dirs if d not in _orig_path])
    if _prepend:
        os.environ['PATH'] = _prepend + os.pathsep + _orig_path

    # PyInstaller 冻结 + --exclude-module torch 组合下, torch 启动所需的
    # 部分标准库模块(如 timeit, 被 torch._strobelight 无条件 import) 未被收集,
    # 直接 "No module named 'timeit'" 会把内置 torch 加载整段打挂。
    # 这里兜底注入最小 stub, 确保 import torch 不被卡死。
    try:
        import timeit  # noqa: F401
    except Exception:
        import sys as _sys, time as _time, types as _types, gc as _gc
        _tmod = _types.ModuleType('timeit')
        _tmod.time = _time
        _tmod.sys = _sys
        _tmod.gc = _gc
        _tmod.default_timer = _time.perf_counter
        class _Timer:
            def __init__(self, stmt='pass', setup='pass', timer=_time.perf_counter, globals=None):
                self.timer = timer
            def timeit(self, number=1000000):
                return 0.0
            def repeat(self, repeat=5, number=1000000):
                return [0.0] * repeat
            def autorange(self, callback=None):
                return (1, 0.0)
        _tmod.Timer = _Timer
        _tmod.timeit = lambda stmt='pass', setup='pass', timer=_time.perf_counter, number=1000000: 0.0
        _sys.modules['timeit'] = _tmod
        _note("已注入 timeit 最小 stub (PyInstaller 未收集该标准库)")

    try:
        import torch      # noqa: F401
    except Exception as _e:
        _note(f"内置 torch 导入失败: {_e}")
        import traceback as _tb
        _note(_tb.format_exc())
        return False

    # torchaudio 不是强制对齐所需 (音频走 soundfile), 即使导入失败也允许回退
    try:
        import torchaudio  # noqa: F401
        _note(f"内置 torch 加载成功: {torch.__version__} (含 torchaudio)")
    except Exception as _ta:
        _note(f"内置 torch 加载成功: {torch.__version__} (torchaudio 不可用, 不影响对齐: {_ta})")
    return True


def ensure_torch(root_window=None):
    """确保 torch / torchaudio 可导入。

    分发均为自包含: 直接加载内置 torch (CPU 版=CPU torch / GPU 版=CUDA torch)。
    """
    global TORCH_DONE
    if TORCH_DONE:
        return True

    # ── 1) 开发模式：直接导入 ──
    if not getattr(sys, 'frozen', False):
        try:
            import torch      # noqa: F401
            import torchaudio # noqa: F401
            TORCH_DONE = True
            return True
        except ImportError:
            pass  # 开发模式没装 torch，继续走安装流程

    # ── 2) 冻结模式：加载内置 torch（CPU 版=CPU torch / GPU 版=CUDA torch）──
    if getattr(sys, 'frozen', False):
        if _try_embedded_torch():
            try:
                import torch as _t
                _cuda_ok = (getattr(_t, 'cuda', None) is not None) and _t.cuda.is_available()
            except Exception:
                _cuda_ok = False
            if _cuda_ok:
                _note("内置 CUDA 版 torch 已加载，已自动启用 GPU 加速。")
            elif BUILD_VARIANT == 'gpu':
                if _has_nvidia_gpu():
                    _note("内置 CUDA 版 torch 已加载，但本机未检测到可用 NVIDIA GPU，已回退 CPU 路径（正常现象）。")
                else:
                    _note("内置 CUDA 版 torch 已加载（本机无 N 卡，走 CPU 路径，属正常）。")
            else:  # CPU 版
                if _has_nvidia_gpu():
                    _note("检测到 NVIDIA 显卡，但本版为纯 CPU 版 FA-ASMR，仅使用内置 CPU torch（无 GPU 加速）。如需 GPU 加速，请改用 GPU 版 FA-ASMR。")
                else:
                    _note("使用内置 CPU 版 torch（纯 CPU 版）。")
            TORCH_DONE = True
            return True

    # ── 3) 最后手段：内置 torch 缺失，弹窗引导联网安装（极少触发）──
    _write_bootstrap_log()  # 落盘诊断, 即使走到弹窗也能拿到失败原因
    if root_window:
        ok = messagebox.askyesno(
            "缺少 PyTorch",
            "未找到可用的 PyTorch（内置 torch 缺失）。\n\n"
            "是否联网下载安装？（需要系统 Python 3.10-3.12）"
        )
    else:
        ok = True

    if not ok:
        sys.exit(0)

    pythons = []
    for cmd in ('python', 'python3', 'py'):
        try:
            r = subprocess.run(
                [cmd, '-c', 'import sys; print(sys.executable)'],
                capture_output=True, text=True, timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            )
            if r.returncode == 0 and r.stdout.strip():
                pythons.append(r.stdout.strip())
        except Exception:
            continue

    if not pythons:
        if root_window:
            messagebox.showerror(
                "未找到 Python",
                "未检测到系统 Python，无法自动安装。请使用带内置 torch 的分发版或从源码构筑。"
            )
        sys.exit(1)

    py_exe = pythons[0]
    if root_window:
        messagebox.showinfo("正在安装", f"正在使用 {py_exe} 安装 PyTorch...")

    ret = subprocess.run(
        [py_exe, '-m', 'pip', 'install', '--user',
         'torch', 'torchaudio',
         '--index-url', 'https://download.pytorch.org/whl/cpu'],
        capture_output=True, text=True,
    )
    if ret.returncode != 0:
        if root_window:
            messagebox.showerror("安装失败", f"安装出错：\n{ret.stderr[-600:]}")
        sys.exit(1)

    try:
        import torch
        import torchaudio
        TORCH_DONE = True
        return True
    except ImportError:
        if root_window:
            messagebox.showerror("导入失败", "PyTorch 已安装但仍无法导入，请检查环境。")
        sys.exit(1)
