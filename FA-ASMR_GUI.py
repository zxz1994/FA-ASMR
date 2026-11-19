# -*- coding: utf-8 -*-
"""FA-ASMR V2 批量强制对齐工具 · Obsidian Acoustic Console 桌面启动器
- 拖放/选择多组台本+音频文件，自动按文件名配对（单人 / 多人 / 双视角）
- 自动多编码嗅探与台本注音 → MMS-FA + LoRA 双声道 Better-Ear 全局对齐 → 标准 LRC/SRT 输出
- 核心无界面管线位于 fa_asmr_pipeline.py，支持 --cli 无界面调用与 --self-test 打包自检
"""
from __future__ import annotations
from typing import Optional, Dict, Any

import sys, os, io, re, time, threading, copy, difflib

# ── windowed (console=False) 冻结模式下绑定标准流 ──────────────────────────
# PyInstaller 的 windowed bootloader 会把 sys.stdout / sys.stderr 置为 None。
# 这会在注音阶段引发一个隐蔽崩溃：
#   yomikata 导入时用 rich 的 RichHandler 做日志，而 RichHandler 默认复用
#   rich 的**全局 Console**（rich/logging.py:98 -> get_console()）。
#   Console.file 在取不到 stdout/stderr 时返回 NULL_FILE，
#   该值会被 Console 缓存（rich/console.py:758-761 只解析一次）。
#   之后 torch_bootstrap._note() 会 `print(..., file=sys.stderr)`，
#   使 sys.stderr 从 None 变为有效对象；此时 logging.Handler.handleError 的
#   守卫 `raiseExceptions and sys.stderr` 成立，而 rich 的 NullFile 分支
#   （rich/logging.py:179-183）会**无条件**调用 handleError(record)。
#   但那里并没有活动异常，sys.exc_info() 返回 (None, None, None)，
#   logging/__init__.py:1086 的 `tb.tb_frame` 于是抛
#   AttributeError: 'NoneType' object has no attribute 'tb_frame'。
#
# 修法：在 import rich（间接由 yomikata 触发）**之前**就把两个流绑到可写文件，
# 使全局 Console 构造时能取到真文件而非 NULL_FILE，从根源消除该分支。
# 必须放在所有业务 import 之前。
if getattr(sys, 'frozen', False):
    for _name in ("stdout", "stderr"):
        if getattr(sys, _name, None) is None:
            try:
                setattr(sys, _name, open(os.devnull, "w", encoding="utf-8",
                                         errors="replace"))
            except Exception:
                pass

if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr is not None and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# NumPy 2.x 移除的别名补齐（原为三处重复副本之一，已统一到 np_compat）。
# 必须在导入期执行，且早于任何依赖它的包。
import np_compat
np_compat.apply()

if getattr(sys, 'frozen', False):
    SCRIPT_DIR = sys._MEIPASS
    EXE_DIR = os.path.dirname(sys.executable)
else:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    EXE_DIR = SCRIPT_DIR
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

SETTINGS_FILE = os.path.join(EXE_DIR, "fa_asmr_settings.json")

# --cli 模式在加载 tkinter / torch 之前直接转交无界面命令行入口，
# 保证 --help / --dry-run / --json 秒级响应且不被 GUI 异常弹窗钩子拦截。
if __name__ == "__main__" and "--cli" in sys.argv:
    from fa_asmr_cli import main as _cli_main
    _cli_argv = [a for a in sys.argv[1:] if a != "--cli"]
    sys.exit(_cli_main(_cli_argv))


def _fatal_error(title, detail):
    try:
        if getattr(sys, 'frozen', False):
            _d = os.path.dirname(os.path.abspath(sys.executable))
        else:
            _d = os.getcwd()
    except Exception:
        _d = os.getcwd()
    _log = os.path.join(_d, "FA-ASMR_crash.log")
    try:
        with open(_log, "w", encoding="utf-8") as _f:
            _f.write("FA-ASMR 启动崩溃日志\n")
            _f.write("time:   " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
            _f.write("frozen: " + str(getattr(sys, 'frozen', False)) + "\n")
            _f.write("MEIPASS:" + str(getattr(sys, '_MEIPASS', 'None')) + "\n\n")
            _f.write(detail)
    except Exception:
        pass
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(
            0,
            "FA-ASMR 启动失败\n\n" + detail[:1800] +
            "\n\n详细日志已保存到：\n" + _log + "\n\n请把该文件发给开发者以便排查。",
            "FA-ASMR 错误", 0x10)
    except Exception:
        pass


def _excepthook(exc_type, exc_val, exc_tb):
    import traceback as _tb
    try:
        _detail = "".join(_tb.format_exception(exc_type, exc_val, exc_tb))
    except Exception:
        _detail = "{}".format(exc_val)
    _fatal_error("FA-ASMR 未捕获异常", _detail)


if "--cli" not in sys.argv:
    sys.excepthook = _excepthook

import torch_bootstrap
from torch_bootstrap import ensure_torch

ensure_torch(None)

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import torch

CUDA_OK = torch.cuda.is_available()

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAS_DND = True
except ImportError:
    HAS_DND = False

try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

import align
import align_utils as au
import haruraw2norm as hn
from clause import F, is_text_token as _is_text_token

# ── 从无界面核心管线模块导入并保持完整向后兼容导出 ──────────────────────────
from fa_asmr_pipeline import (
    AUDIO_EXTS,
    _SPEAKER_BRACKET_RE,
    _SPEAKER_SEP,
    _iter_speaker_marks,
    _split_speaker_segments,
    AutoRuby,
    read_text_auto_encoding,
    _fmt_lrc_time,
    _lrc_time_to_cs,
    build_standard_lrc,
    _fmt_line,
    build_standard_srt,
    _PUNCT_CHARS,
    _strip_orphan_punct,
    _INLINE_RUBY_RE,
    _strip_ruby_for_src,
    _attach_source_spans,
    _release_gpu,
    run_pipeline,
    _parse_speaker_suffix,
    _PERSP_PATTERNS,
    _extract_perspective_tag,
    _clean_stem_for_matching,
    auto_pair_files,
)


def build_gui():
    """创建界面并接入现有对齐管线，保持打包与源码启动路径一致。"""
    from types import SimpleNamespace
    from fa_asmr_ui import build_ui, enable_windows_dpi_awareness

    # 必须在创建 Tk 根窗口之前启用高分屏 DPI 感知，否则 4K 屏下会被系统位图拉伸发糊
    enable_windows_dpi_awareness()
    root = TkinterDnD.Tk() if HAS_DND else tk.Tk()
    ctx = SimpleNamespace(
        root=root,
        script_dir=SCRIPT_DIR,
        settings_file=SETTINGS_FILE,
        cuda_ok=CUDA_OK,
        has_dnd=HAS_DND,
        dnd_files=DND_FILES if HAS_DND else None,
        pil=HAS_PIL,
        auto_pair_files=auto_pair_files,
        run_pipeline=run_pipeline,
        lora_path=str(getattr(align, "DEFAULT_LORA_PATH", "")),
        release_gpu=_release_gpu,
        is_running=lambda: bool(getattr(root, "_fa_running", False)),
        start_worker=lambda target: _start_worker(root, target),
    )
    return build_ui(ctx)


def _start_worker(root, target):
    root._fa_set_running(True)
    threading.Thread(target=target, daemon=True).start()


def _run_self_test() -> int:
    """打包产物自检：在真实 frozen 环境里验证核心链路可用。

    存在的意义：exe 能启动 ≠ 功能可用。曾出现过依赖漏打包
    （yomikata 的 speach 缺失）导致启动正常、一进注音就崩的情况。
    该入口让 CI 在打包后直接跑真实功能，而不是只做启动冒烟。

    退出码 0 = 全部通过；非 0 = 失败。
    """
    import traceback

    reports = []

    def step(name, fn):
        try:
            detail = fn()
            reports.append((True, name, detail or ""))
        except BaseException as e:
            reports.append((False, name, f"{type(e).__name__}: {e}"))
            traceback.print_exc()

    def _deps():
        # 只校验模块**可被定位**，不做真实 import。
        # 原因：yomikata.config 在模块级读 UTF-8 JSON 却未声明 encoding，
        # 直接 import 会在非 UTF-8 locale 下抛 UnicodeDecodeError ——
        # 那属于 yomikata 的缺陷，由 fa_asmr_converter 的补丁在生产路径上兜住。
        # 自检这里若直连 import，等于绕开补丁复现上游缺陷，属误报。
        # 「注音链路实跑」一项才是对 yomikata 的真实检验。
        import importlib.util

        broken = []
        for mod in ("numpy", "torch", "torchaudio", "numba",
                    "transformers", "tokenizers", "huggingface_hub",
                    "safetensors", "soundfile", "fugashi", "unidic_lite",
                    "pykakasi", "janome", "jaconv",
                    "speach", "chirptext", "puchikarui", "rich", "yomikata",
                    "sympy", "mpmath"):
            try:
                if importlib.util.find_spec(mod) is None:
                    broken.append(mod)
            except (ImportError, ValueError):
                broken.append(mod)
        if broken:
            raise RuntimeError("以下模块在打包产物中缺失: " + ", ".join(broken))

        from speach.ttlig import RubyFrag, RubyToken  # noqa: F401
        import torch, transformers, numba
        return (f"torch={torch.__version__} "
                f"transformers={transformers.__version__} "
                f"numba={numba.__version__}")

    def _model_file():
        base = os.path.join(EXE_DIR, "models", "hub", "checkpoints", "model.pt")
        if not os.path.isfile(base):
            raise FileNotFoundError(f"缺少 MMS-FA 权重: {base}")
        size = os.path.getsize(base)
        if size < 1_000_000_000:
            raise RuntimeError(f"MMS-FA 权重体积异常: {size} 字节")
        return f"{size / 1e6:.1f} MB"

    def _lora_file():
        p = os.path.join(EXE_DIR, "lora", "fa_asmr_v9e10.pt")
        if not os.path.isfile(p):
            raise FileNotFoundError(f"缺少 LoRA 适配器: {p}")
        return f"{os.path.getsize(p) / 1e6:.1f} MB"

    def _dbert_weights():
        import yomikata
        d = os.path.join(os.path.dirname(yomikata.__file__),
                         "dbert-artifacts", "pytorch_model.bin")
        if not os.path.isfile(d):
            raise FileNotFoundError(f"缺少 dBert 权重: {d}")
        return f"{os.path.getsize(d) / 1e6:.1f} MB"

    def _annotate():
        import sys as _s
        _s.path.insert(0, os.path.join(SCRIPT_DIR, "furigana"))
        from fa_asmr_converter import RubyAnnotator
        ann = RubyAnnotator(verbose=False, strip_inline_readings=True)
        out = ann.annotate("こんにちは、今日はいい天気ですね。")
        if "{" not in out or "|" not in out:
            raise RuntimeError(f"注音输出格式异常: {out!r}")
        ann.release_gpu()
        return out

    def _align_engine():
        import torch
        import align
        import align_multispeaker  # noqa: F401
        import align_multiperspective  # noqa: F401
        import fa_asmr_pipeline  # noqa: F401
        import fa_asmr_cli  # noqa: F401
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        return f"align + multispeaker + multiperspective + pipeline + cli 可导入，设备={dev}"

    step("第三方依赖导入", _deps)
    step("MMS-FA 权重", _model_file)
    step("LoRA 适配器", _lora_file)
    step("dBert 权重", _dbert_weights)
    step("对齐引擎导入", _align_engine)
    step("注音链路实跑", _annotate)

    print("\n" + "=" * 56)
    print("  FA-ASMR 自检结果")
    print("=" * 56)
    ok = True
    lines = []
    for passed, name, detail in reports:
        lines.append(f"  [{'OK ' if passed else '!!!'}] {name}")
        if detail:
            lines.append(f"        {detail}")
        ok = ok and passed
    lines.append("=" * 56)
    lines.append("  结论: " + ("全部通过" if ok else "存在失败项"))
    lines.append("=" * 56)

    for ln in lines:
        print(ln)

    # windowed（console=False）冻结模式下 stdout 被绑到 devnull，
    # CI 读不到上面的输出，因此额外落盘一份供构建脚本检查。
    try:
        log_path = os.path.join(EXE_DIR, "FA-ASMR_selftest.log")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"  (自检日志已写入 {log_path})")
    except Exception:
        pass

    return 0 if ok else 1


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(_run_self_test())
    app = build_gui()
    app.mainloop()


