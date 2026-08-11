# -*- coding: utf-8 -*-
"""
FA-ASMR 批量对齐工具
- 拖放/选择多组台本+音频文件，自动按文件名配对
- 自动台本注音 → LoRA 对齐 → 标准 LRC 一键输出
- 不保留中间产物（_rlf.lrc, _ruby.lrc, .ass）

用法：
    python FA-ASMR_GUI.py          → GUI（拖放模式）
    python FA-ASMR_GUI.py --cli --script <台本.txt> --audio <音频.wav> [--lora <ckpt.pt>]
"""

import sys, os, io, re, time, threading, queue

if getattr(sys, 'frozen', False):
    # PyInstaller onedir: --add-data 的数据文件落在 _internal/ 下 (sys._MEIPASS)
    SCRIPT_DIR = sys._MEIPASS
    # 设置/配置需持久化到 exe 同级目录（_MEIPASS 是只读临时包，写入不会保留）
    EXE_DIR = os.path.dirname(sys.executable)
else:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    EXE_DIR = SCRIPT_DIR
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

SETTINGS_FILE = os.path.join(EXE_DIR, "fa_asmr_settings.json")


# ════════════════════════════════════════════════════════════════
# 启动期崩溃兜底（windowed exe 无控制台，任何未捕获异常必须弹窗+落盘，
# 否则朋友测试时只会看到"什么都没发生"而无法排查）
# ════════════════════════════════════════════════════════════════
def _fatal_error(title, detail):
    """把崩溃信息写到 exe 同级 FA-ASMR_crash.log，并用原生 Windows 弹窗显示
    （不依赖 tkinter，即使 tkinter 自身导入失败也能弹出）。"""
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


sys.excepthook = _excepthook




def check_mmsfa_weights():
    """检查 MMS-FA 权重 model.pt 是否就位。

    权重不随程序打包，需手动放置到：
      - 冻结模式：exe 同级 models/hub/checkpoints/model.pt
      - 开发模式：脚本同级 models/hub/checkpoints/model.pt
    返回缺失时的预期路径(str)，无缺失返回 None。
    """
    import sys as _sys
    if getattr(_sys, 'frozen', False):
        _base_dir = os.path.dirname(_sys.executable)
    else:
        _base_dir = os.path.dirname(os.path.abspath(__file__))
    _p = os.path.join(_base_dir, 'models', 'hub', 'checkpoints', 'model.pt')
    return _p if not os.path.isfile(_p) else None



# ═══════════════════════════════════════════════════
# PyTorch 自动安装引导：缺失 torch 时自动 pip install
# 必须放在所有 torch 导入之前
# ═══════════════════════════════════════════════════
import torch_bootstrap
from torch_bootstrap import ensure_torch

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ── 确保 torch 可用（弹出安装对话框） ──
_root = tk.Tk()
_root.withdraw()               # 隐藏空窗口
ensure_torch(_root)            # 缺失则弹窗引导安装
_root.destroy()

import torch

# 设备能力探测：CUDA 可用才允许 GUI 选择 GPU（否则设备下拉只有 CPU）
CUDA_OK = torch.cuda.is_available()

# ── 可选拖放库 ──
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAS_DND = True
except ImportError:
    HAS_DND = False

import align
import haruraw2norm as hn

AUDIO_EXTS = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac', '.opus', '.wma'}


# ═══════════════════════════════════════════════════════════════════
#  1. 台本自动注音
# ═══════════════════════════════════════════════════════════════════
class AutoRuby:
    """台本自动注音: 委托 fa_asmr_converter.RubyAnnotator。
    使用 UniDic 主引擎 + yomikata 异读消歧 + 极小已验证修正表,
    取代旧版纯 pykakasi 实现 (含 '同时含 { 和 | 的混合行被整行跳过' 的 bug)。"""

    def __init__(self):
        import os as _os
        import sys as _sys
        # 冻结模式下用 SCRIPT_DIR (=sys._MEIPASS)，开发模式用 __file__ 同级
        _td = _os.path.join(SCRIPT_DIR, "furigana")
        if _td not in _sys.path:
            _sys.path.insert(0, _td)
        from fa_asmr_converter import RubyAnnotator
        self._ann = RubyAnnotator(verbose=False)

    def annotate(self, text: str) -> str:
        return self._ann.annotate(text)

    def release_gpu(self):
        try:
            self._ann.release_gpu()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════
#  2. 标准 LRC 构建
# ═══════════════════════════════════════════════════════════════════
def build_standard_lrc(result_list):
    def _fmt_time(t):
        if not t or t == "[error]":
            return None
        m = re.match(r"\[(\d{2}):(\d{2}):(\d{2})\]", t)
        if m:
            return f"[{m.group(1)}:{m.group(2)}.{m.group(3)}]"
        return None

    lines = []
    group_start = None
    group_end = None
    group_text = []

    for item in result_list:
        io = item.get("orig", "")
        if item.get("type") == 0 and io == "\n":
            if group_text and group_start:
                lines.append(_fmt_line(group_start, group_end, "".join(group_text)))
                if group_end and group_end != group_start:
                    t = _fmt_time(group_end)
                    if t:
                        lines.append(t)
            group_start = None
            group_end = None
            group_text = []
            continue
        if "start" in item and group_start is None:
            group_start = item["start"]
        if "end" in item:
            group_end = item["end"]
        if item.get("type") in (1, 2, 3, 4, 5) and io:
            group_text.append(io)
        elif item.get("type") == 0 and io not in ("\n", ""):
            group_text.append(io)

    if group_text and group_start:
        lines.append(_fmt_line(group_start, group_end, "".join(group_text)))
        if group_end and group_end != group_start:
            t = _fmt_time(group_end)
            if t:
                lines.append(t)
    return "\n".join(lines)


def _fmt_line(start_str, end_str, text):
    clean = re.sub(r"\{([^|]+)\|[^}]+\}", r"\1", text)
    def _t(s):
        if not s or s == "[error]":
            return None
        m = re.match(r"\[(\d{2}):(\d{2}):(\d{2})\]", s)
        return f"[{m.group(1)}:{m.group(2)}.{m.group(3)}]" if m else None
    t = _t(start_str)
    return f"{t}{clean}" if t else clean


# ═══════════════════════════════════════════════════════════════════
#  3. 核心管线
# ═══════════════════════════════════════════════════════════════════
def _release_gpu(ruby=None):
    """一条音频对齐完成后释放所有常驻 GPU 资源:
       - align 的 MMS_FA 模型缓存
       - fa_asmr_converter 的 yomikata dBert (常驻 CUDA)
    引用置空后必须调 torch.cuda.empty_cache() 才能真正归还显存,
    否则 nvidia-smi 仍显示占用 ('什么都挤在 GPU 里')。
    连续任务之间不应调用, 由 persist_gpu=True 保留缓存提速。
    """
    try:
        import align
        align.clear_model_cache()
    except Exception:
        pass
    if ruby is not None:
        try:
            ruby.release_gpu()
        except Exception:
            pass


def run_pipeline(script_path, audio_path, lora_path, log_callback, progress_callback,
                  dev="auto",
                  vram_chunk=6000,
                  persist_gpu=False,
                  line_end_extend=0.0,
                  collapse_prune=True):
    """返回 (ok: bool, output_lrc_path: str, error_msg: str)

    persist_gpu=False (默认): 跑完即释放所有 GPU 资源, 显存归零。
    persist_gpu=True : 保留 MMS_FA/VAD/dBert 缓存, 用于连续多文件批处理提速。
    line_end_extend: 每行字幕结束时间延长秒数（≤下一行开始时间）。0=不延长。
    """
    def log(msg):
        print(msg)

    progress_callback and progress_callback(0, "初始化...")
    total_start = time.time()
    _ruby = None

    try:
        log("[1/4] 读取台本...")
        with open(script_path, encoding="utf-8") as f:
            raw_text = f.read()

        ruby = AutoRuby()
        _ruby = ruby
        # 无论是否含汉字, 均逐行经过 annotate:
        # 含汉字行正常注音, 纯假名/符号/英文行也经 normalize 清洗
        # (浊点/〜/全角/超长促音), 避免 vocab 外非法 token 进入对齐器。
        lines = raw_text.splitlines()
        annotated_lines = [ruby.annotate(line) for line in lines]
        annotated_text = "\n".join(annotated_lines)
        log(f"  台本处理完成 ({len(lines)} 行)")

        log("[2/4] 解析文本...")
        result_list = []
        lang = "auto"
        for line in annotated_text.splitlines(keepends=True):
            if line.strip():
                result_list.extend(hn.process_haruhi_line(line, lang, 0, 1))
        if result_list and result_list[-1].get("orig") != "\n":
            result_list.append({"orig": "\n", "type": 0, "pron": ""})

        alignment_tokens = []
        token_to_index_map = {}
        current_line = []
        flat_idx = 0
        for i, item in enumerate(result_list):
            if item.get("orig") == "\n":
                if current_line:
                    alignment_tokens.append(current_line)
                    current_line = []
            elif "pron" in item and item["pron"]:
                current_line.append(item["pron"])
                token_to_index_map[flat_idx] = i
                flat_idx += 1
        if current_line:
            alignment_tokens.append(current_line)

        n_tokens = sum(len(lt) for lt in alignment_tokens)
        log(f"  解析完成: {n_tokens} tokens, {len(alignment_tokens)} 行")

        progress_callback and progress_callback(10, "加载音频...")

        log("[3/4] 加载音频...")
        # 用 soundfile 读原始波形(避开 librosa.load(sr=None,mono=False)
        # 对 >20min/48k 大文件返回空数组的坑)
        import soundfile as sf
        audio_file, sr = sf.read(audio_path, dtype="float32", always_2d=True)
        audio_file = audio_file.mean(axis=1)  # (N,2)->(N,) 单声道
        dur = audio_file.shape[0] / sr
        log(f"  采样率={sr}, 时长={dur:.1f}s")
        ns_ranges = [(0, dur)]

        progress_callback and progress_callback(25, "对齐中...")

        log(f"[4/4] MMS-FA + LoRA 对齐...")

        # 权重就位检查：model.pt 不随程序打包，需手动放置
        _missing_weights = check_mmsfa_weights()
        if _missing_weights:
            log(f"  [警告] 未找到 MMS-FA 权重 model.pt：{_missing_weights}")
            log(f"  [警告] 程序将尝试联网下载；若离线请手动放置（权重不随程序打包）")

        if not (lora_path and os.path.isfile(lora_path)):
            log("  (无 LoRA，使用基础 MMS-FA)")
            lora_path = None
        else:
            log(f"  LoRA: {os.path.basename(lora_path)}")

        alignment_results = align.align_audio_with_text(
            audio_file, alignment_tokens, ns_ranges, sr, 1,
            lora_path=lora_path,
            device=dev, chunk_frames=int(vram_chunk),
            collapse_prune=collapse_prune,
            log_callback=log
        )


        if not alignment_results:
            if not persist_gpu:
                _release_gpu(_ruby)
            return False, None, "对齐返回空结果"

        for i, result in enumerate(alignment_results):
            if i in token_to_index_map:
                original_index = token_to_index_map[i]
                result_list[original_index]["start"] = result["start"]
                result_list[original_index]["end"] = result["end"]

        for i in range(len(result_list)):
            item = result_list[i]
            if "start" not in item and item.get("type", 0) in [1, 2, 3, 4, 5]:
                for j in range(i - 1, -1, -1):
                    if "start" in result_list[j]:
                        item["start"] = result_list[j]["start"]
                        item["end"] = result_list[j]["end"]
                        break

        lrc_content = build_standard_lrc(result_list)

        # 行尾延长(纯文本后处理, 接在最后、写文件之前):
        # 只把每句「空白结束标记」[mm:ss.xx] 往后推 line_end_extend 秒,
        # 封顶到下一句起点; 绝不改动文本行时间戳(避免整体后移/搞乱对齐结果)。
        # 与 lrc_end_extend.py(直接改最终 .lrc 的脚本)同一逻辑, 必然生效。
        if line_end_extend > 0:
            from lrc_end_extend import extend_lrc
            lrc_content, _n_ext = extend_lrc(lrc_content, line_end_extend)
            if _n_ext:
                log(f"  行尾延长 ({line_end_extend:.1f}s): 处理 {_n_ext} 个空白标记")

        base = os.path.splitext(os.path.basename(script_path))[0]
        out_dir = os.path.dirname(os.path.abspath(audio_path))
        lrc_path = os.path.join(out_dir, f"{base}.lrc")
        with open(lrc_path, "w", encoding="utf-8") as f:
            f.write(lrc_content)

        total_elapsed = time.time() - total_start
        n_lines = lrc_content.count("\n") + 1
        log(f"\n  完成! 总耗时 {total_elapsed:.1f}s")
        log(f"  输出: {lrc_path}  ({n_lines} 行)")

        progress_callback and progress_callback(100, "完成!")
        if not persist_gpu:
            _release_gpu(_ruby)
        return True, lrc_path, None

    except Exception as e:
        import traceback
        err = traceback.format_exc()
        log(f"\n  错误: {err}")
        if not persist_gpu:
            _release_gpu(_ruby)
        return False, None, str(e)


# ═══════════════════════════════════════════════════════════════════
#  4. 文件自动配对
# ═══════════════════════════════════════════════════════════════════
def auto_pair_files(file_paths):
    """将文件列表按 base name 自动配对 (txt ↔ 音频)"""
    txt_files = [f for f in file_paths if os.path.splitext(f)[1].lower() == '.txt']
    audio_files = [f for f in file_paths if os.path.splitext(f)[1].lower() in AUDIO_EXTS]

    txt_by_stem = {}
    for f in txt_files:
        stem = os.path.splitext(os.path.basename(f))[0]
        txt_by_stem.setdefault(stem, []).append(f)

    audio_by_stem = {}
    for f in audio_files:
        stem = os.path.splitext(os.path.basename(f))[0]
        audio_by_stem.setdefault(stem, []).append(f)

    pairs = []
    unmatched = []

    common_stems = set(txt_by_stem.keys()) & set(audio_by_stem.keys())
    for stem in sorted(common_stems):
        for txt in txt_by_stem[stem]:
            for aud in audio_by_stem[stem]:
                pairs.append({
                    'script': txt,
                    'audio': aud,
                    'name': stem,
                })

    for stem in sorted(set(txt_by_stem.keys()) - set(audio_by_stem.keys())):
        for txt in txt_by_stem[stem]:
            unmatched.append({'type': '缺音频', 'file': txt, 'name': stem})

    for stem in sorted(set(audio_by_stem.keys()) - set(txt_by_stem.keys())):
        for aud in audio_by_stem[stem]:
            unmatched.append({'type': '缺台本', 'file': aud, 'name': stem})

    return pairs, unmatched


# ═══════════════════════════════════════════════════════════════════
#  5. GUI
# ═══════════════════════════════════════════════════════════════════
STATUS_ICONS = {
    'pending': '  ⏳ 等待',
    'running': '  ▶  运行中',
    'done':    '  ✓  完成',
    'failed':  '  ✗  失败',
}


def build_gui():
    if HAS_DND:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()

    root.title("FA-ASMR 批量对齐工具")
    root.geometry("860x680")
    root.minsize(700, 500)
    root.resizable(True, True)

    style = ttk.Style()
    style.theme_use("clam")

    # ── 颜色主题 ──
    BG = "#f5f6fa"
    CARD_BG = "#ffffff"
    ACCENT = "#4a6cf7"
    DANGER = "#e74c3c"
    SUCCESS = "#27ae60"
    TEXT = "#2c3e50"
    TEXT_MUTED = "#7f8c8d"
    BORDER = "#e0e4ea"

    root.configure(bg=BG)

    main = tk.Frame(root, bg=BG)
    main.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)

    # ── 顶部标题栏 ──
    header = tk.Frame(main, bg=ACCENT)
    header.pack(fill=tk.X, pady=(0, 12))
    tk.Label(header, text="FA-ASMR 批量对齐工具",
             font=("Microsoft YaHei UI", 15, "bold"),
             fg="white", bg=ACCENT).pack(side=tk.LEFT, padx=16, pady=(10, 4))
    tk.Label(header, text="拖放台本 + 音频自动配对 · 一键生成 LRC",
             font=("Microsoft YaHei UI", 9),
             fg="#cdd6ff", bg=ACCENT).pack(side=tk.LEFT, padx=(8, 16), pady=(12, 6))

    # ── 第一行：LoRA + 按钮 ──
    ctrl_frame = tk.Frame(main, bg=BG)
    ctrl_frame.pack(fill=tk.X, pady=(0, 8))

    # LoRA 固定使用打包目录内的 lora/fa_asmr_e30.pt，无需用户选择
    DEFAULT_LORA = os.path.join(SCRIPT_DIR, "lora", "fa_asmr_e30.pt")
    if not os.path.isfile(DEFAULT_LORA):
        DEFAULT_LORA = ""

    lora_var = tk.StringVar(value=DEFAULT_LORA)

    # ── 高级设置（折叠面板）──
    adv_header_frame = tk.Frame(main, bg=BG)
    adv_header_frame.pack(fill=tk.X, pady=(0, 2))

    adv_show = tk.BooleanVar(value=False)
    adv_toggle = tk.Label(adv_header_frame, text="▶ 高级设置",
                          font=("", 8), fg=TEXT_MUTED, bg=BG, cursor="hand2")
    adv_toggle.pack(side=tk.LEFT)

    adv_panel = tk.Frame(main, bg=CARD_BG, highlightbackground=BORDER,
                         highlightthickness=1)
    adv_panel_inner = tk.Frame(adv_panel, bg=CARD_BG)
    adv_panel_inner.pack(fill=tk.X, padx=10, pady=8)

    def _toggle_adv():
        if adv_show.get():
            adv_show.set(False)
            adv_panel.pack_forget()
            adv_toggle.configure(text="▶ 高级设置")
        else:
            adv_show.set(True)
            adv_panel.pack(fill=tk.X, pady=(0, 8), before=drop_frame)
            adv_toggle.configure(text="▼ 高级设置")

    adv_toggle.bind("<Button-1>", lambda e: _toggle_adv())

    # 设置变量（默认值）
    dev_var = tk.StringVar(value="cuda" if CUDA_OK else "cpu")
    vram_chunk_var = tk.StringVar(value="6000")
    end_extend_var = tk.StringVar(value="0")
    prune_var = tk.BooleanVar(value=True)

    # ── 设置持久化 ──
    # lora 路径固定（lora/fa_asmr_e30.pt），不持久化
    _var_meta = [
        (dev_var,         "dev",         str),
        (vram_chunk_var,  "vram_chunk",  str),
        (end_extend_var, "end_extend", str),
        (prune_var, "prune", bool),
    ]

    def _save_settings():
        try:
            data = {}
            for var, key, _ in _var_meta:
                data[key] = var.get()
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                import json
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _load_settings():
        if not os.path.isfile(SETTINGS_FILE):
            return
        try:
            import json
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            for var, key, cvt in _var_meta:
                if key in data:
                    val = cvt(data[key])
                    if key == "dev":
                        if val == "cuda" and not CUDA_OK:
                            val = "cpu"
                        elif val not in ("cuda", "cpu"):
                            val = "cuda" if CUDA_OK else "cpu"
                    var.set(val)
        except Exception:
            pass

    _load_settings()

    # ── 第一行：设备 ──
    row1 = tk.Frame(adv_panel_inner, bg=CARD_BG)
    row1.pack(fill=tk.X, pady=(0, 4))
    _dev_values = ["cuda", "cpu"] if CUDA_OK else ["cpu"]
    for label, var, vlist, w, readonly in [
        ("设备:", dev_var, _dev_values, 6, True),
    ]:
        tk.Label(row1, text=label, font=("", 8), fg=TEXT_MUTED, bg=CARD_BG).pack(side=tk.LEFT, padx=(0, 2))
        if vlist:
            cb = ttk.Combobox(row1, textvariable=var, values=vlist, state="readonly", width=w)
            cb.pack(side=tk.LEFT, padx=(0, 12))
        else:
            e = tk.Entry(row1, textvariable=var, font=("", 8), width=w,
                         relief=tk.FLAT, bg="white", highlightbackground=BORDER, highlightthickness=1)
            e.pack(side=tk.LEFT, padx=(0, 12))

    # ── 第二行：显存参数(仅GPU) + 优化开关 ──
    row2 = tk.Frame(adv_panel_inner, bg=CARD_BG)
    row2.pack(fill=tk.X)
    if CUDA_OK:
        tk.Label(row2, text="显存缓冲(帧):", font=("", 8), fg=TEXT_MUTED, bg=CARD_BG).pack(side=tk.LEFT, padx=(0, 2))
        e_chunk = tk.Entry(row2, textvariable=vram_chunk_var, font=("", 8), width=6,
                           relief=tk.FLAT, bg="white", highlightbackground=BORDER, highlightthickness=1)
        e_chunk.pack(side=tk.LEFT, padx=(0, 20))

    # ── 第二行末尾：实验开关 ──
    tk.Checkbutton(row2, text="循环切除塌缩行重CTC(≤5次)", variable=prune_var,
                   font=("", 8), bg=CARD_BG, fg=TEXT,
                   selectcolor=CARD_BG, activebackground=CARD_BG,
                   anchor="w").pack(side=tk.LEFT, padx=(0, 8))

    # ── 第三行：行结束延长 ──
    row3 = tk.Frame(adv_panel_inner, bg=CARD_BG)
    row3.pack(fill=tk.X, pady=(4, 0))
    tk.Label(row3, text="行结束延长(秒):", font=("", 8), fg=TEXT_MUTED, bg=CARD_BG).pack(side=tk.LEFT, padx=(0, 2))
    e_ext = tk.Entry(row3, textvariable=end_extend_var, font=("", 8), width=5,
                      relief=tk.FLAT, bg="white", highlightbackground=BORDER, highlightthickness=1)
    e_ext.pack(side=tk.LEFT)
    tk.Label(row3, text="0=不延长  推荐 0.5~1.5  (≤下一行开始)", font=("", 7),
             fg=TEXT_MUTED, bg=CARD_BG).pack(side=tk.LEFT, padx=(4, 0))


    # ── 拖放区 ──
    drop_frame = tk.Frame(main, bg=ACCENT, height=86, highlightthickness=0)
    drop_frame.pack(fill=tk.X, pady=(0, 10))
    drop_frame.pack_propagate(False)

    drop_inner = tk.Frame(drop_frame, bg=ACCENT, highlightbackground="#cdd6ff",
                          highlightthickness=1)
    drop_inner.place(relx=0.012, rely=0.12, relwidth=0.976, relheight=0.76)

    drop_text_var = tk.StringVar(value="拖放台本 (.txt) 和音频 (.wav/.mp3/.flac/.ogg) 到此处\n或点击此处选择文件")
    drop_label = tk.Label(drop_inner, textvariable=drop_text_var,
                           font=("Microsoft YaHei UI", 10),
                           fg="white", bg=ACCENT, cursor="hand2")
    drop_label.pack(expand=True)

    def _drop_hover_enter(e):
        drop_frame.config(bg="#3b5de7")
        drop_inner.config(bg="#3b5de7")
        drop_label.config(bg="#3b5de7")
    def _drop_hover_leave(e):
        drop_frame.config(bg=ACCENT)
        drop_inner.config(bg=ACCENT)
        drop_label.config(bg=ACCENT)
    drop_frame.bind("<Enter>", _drop_hover_enter)
    drop_frame.bind("<Leave>", _drop_hover_leave)

    def _on_drop(event):
        raw = event.data
        files = _parse_drop_data(raw)
        _add_files_to_list(files)

    def _add_files_to_list(files):
        if not files:
            return
        pairs, unmatched = auto_pair_files(files)
        added = 0
        for p in pairs:
            if p['name'] not in _existing_names:
                iid = task_tree.insert("", "end", values=(
                    p['name'], os.path.basename(p['script']),
                    os.path.basename(p['audio']),
                    STATUS_ICONS['pending']), tags=('pending',))
                _task_store[iid] = {**p, 'status': 'pending', 'iid': iid}
                _existing_names.add(p['name'])
                added += 1
        for u in unmatched:
            if u['name'] not in _existing_names and u['type'] == '缺音频':
                # 也加入列表但标记为"无音频"
                iid = task_tree.insert("", "end", values=(
                    u['name'], os.path.basename(u['file']),
                    "— (未匹配)", "  ⚠ 缺音频"), tags=('warn',))
                _task_store[iid] = {
                    'script': u['file'], 'audio': None,
                    'name': u['name'], 'status': 'pending', 'iid': iid}
                _existing_names.add(u['name'])
                added += 1

        if added:
            drop_text_var.set(f"已添加 {added} 项任务 — 可继续拖放追加")
        else:
            drop_text_var.set("未发现新的可配对文件")

    def _parse_drop_data(raw):
        """解析拖放数据（含花括号路径处理）"""
        paths = []
        # tkinterdnd2 格式: {path1} {path2} ...
        for m in re.finditer(r'\{([^}]*)\}', raw):
            paths.append(m.group(1))
        # 纯路径（无花括号）
        if not paths:
            for m in re.finditer(r'\S+', raw):
                p = m.group(0)
                if os.path.isfile(p):
                    paths.append(p)
        return [p for p in paths if os.path.isfile(p)]

    if HAS_DND:
        try:
            drop_frame.drop_target_register(DND_FILES)
            drop_frame.dnd_bind('<<Drop>>', _on_drop)
            drop_label.configure(text="拖放台本(.txt)和音频(.wav/.mp3/.flac/.ogg 等)文件到此处\n或点击选择文件")
        except Exception:
            pass

    def _on_drop_click(e):
        files = filedialog.askopenfilenames(
            title="选择台本和音频文件",
            filetypes=[
                ("台本+音频", "*.txt;*.wav;*.mp3;*.flac;*.m4a;*.ogg;*.aac;*.opus;*.wma"),
                ("All", "*.*"),
            ])
        if files:
            _add_files_to_list(list(files))

    drop_label.bind("<Button-1>", _on_drop_click)

    # ── 任务列表 ──
    list_frame = tk.Frame(main, bg=CARD_BG, highlightbackground=BORDER,
                          highlightthickness=1)
    list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

    # Treeview（使用内置表头，设置列宽）
    tree_container = tk.Frame(list_frame, bg=CARD_BG)
    tree_container.pack(fill=tk.BOTH, expand=True)

    task_tree = ttk.Treeview(tree_container,
                             columns=("name", "script", "audio", "status"),
                             show="headings",
                             selectmode="extended")
    # 隐藏默认的 #0 树形列
    task_tree.column("#0", width=0, stretch=False)
    # 数据列：名称 / 台本 / 音频 / 状态
    task_tree.heading("name", text="任务名称")
    task_tree.column("name", width=150, minwidth=80, stretch=True)
    task_tree.heading("script", text="台本文件")
    task_tree.column("script", width=180, minwidth=100, stretch=True)
    task_tree.heading("audio", text="音频文件")
    task_tree.column("audio", width=180, minwidth=100, stretch=True)
    task_tree.heading("status", text="状态")
    task_tree.column("status", width=100, minwidth=60, stretch=False)

    task_tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

    style.configure("Treeview", rowheight=28, font=("", 9), borderwidth=0)
    style.configure("Treeview.Heading", font=("", 9, "bold"), background="#f0f2f5", borderwidth=0)
    task_tree.tag_configure('pending', foreground=TEXT_MUTED)
    task_tree.tag_configure('running', foreground=ACCENT)
    task_tree.tag_configure('done',    foreground=SUCCESS)
    task_tree.tag_configure('failed',  foreground=DANGER)
    task_tree.tag_configure('warn',    foreground="#e67e22")

    tree_scroll = ttk.Scrollbar(tree_container, orient=tk.VERTICAL, command=task_tree.yview)
    task_tree.configure(yscrollcommand=tree_scroll.set)
    tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    _task_store = {}   # iid → task dict
    _existing_names = set()

    def _set_status(iid, key):
        task_tree.set(iid, 'status', STATUS_ICONS[key])
        task_tree.item(iid, tags=(key,))

    # ── 底部按钮栏 ──
    action_frame = tk.Frame(main, bg=BG)
    action_frame.pack(fill=tk.X, pady=(0, 6))

    def _clear_list():
        for iid in list(_task_store.keys()):
            task_tree.delete(iid)
        _task_store.clear()
        _existing_names.clear()
        drop_text_var.set("拖放台本(.txt)和音频(.wav/.mp3/.flac/.ogg 等)文件到此处\n或点击选择文件")
        update_status("就绪")

    def _del_selected():
        for iid in task_tree.selection():
            if iid in _task_store:
                name = _task_store[iid]['name']
                _existing_names.discard(name)
                del _task_store[iid]
                task_tree.delete(iid)

    tk.Button(action_frame, text="清空列表", command=_clear_list,
              font=("", 8), bg=CARD_BG, relief=tk.FLAT,
              highlightbackground=BORDER, highlightthickness=1,
              cursor="hand2").pack(side=tk.LEFT, padx=(0, 6))
    tk.Button(action_frame, text="删除选中", command=_del_selected,
              font=("", 8), bg=CARD_BG, relief=tk.FLAT,
              highlightbackground=BORDER, highlightthickness=1,
              cursor="hand2").pack(side=tk.LEFT)

    # 进度
    progress = ttk.Progressbar(action_frame, mode="determinate", length=200)
    progress.pack(side=tk.RIGHT, padx=(10, 0))

    status_var = tk.StringVar(value="就绪 — 拖放文件到上方区域开始")
    status_label = tk.Label(action_frame, textvariable=status_var,
                             font=("", 8), fg=TEXT_MUTED, bg=BG)
    status_label.pack(side=tk.RIGHT)

    def update_status(msg):
        root.after(0, lambda: status_var.set(msg))

    # ── 批量运行按钮 ──
    run_btn = tk.Button(action_frame, text="▶  开始全部",
                        font=("Microsoft YaHei UI", 10, "bold"),
                        fg="white", bg=ACCENT, relief=tk.FLAT,
                        padx=20, pady=2, cursor="hand2",
                        activebackground="#3b5de7", activeforeground="white")
    run_btn.pack(side=tk.RIGHT, padx=(10, 0))

    # ── 日志区 ──
    tk.Label(main, text="运行日志:", font=("", 9, "bold"),
             bg=BG, fg=TEXT, anchor="w").pack(fill=tk.X, pady=(0, 3))

    log_frame = tk.Frame(main, bg="#1e1e1e", highlightbackground=BORDER,
                         highlightthickness=1)
    log_frame.pack(fill=tk.BOTH, expand=True)

    log_text = tk.Text(log_frame, height=10, wrap=tk.WORD,
                       font=("Consolas", 9),
                       bg="#1e1e1e", fg="#d4d4d4", insertbackground="white",
                       relief=tk.FLAT, padx=10, pady=8,
                       state=tk.DISABLED)
    log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=log_text.yview)
    log_text.configure(yscrollcommand=log_scroll.set)
    log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    log_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _log_line(line):
        def _do():
            log_text.configure(state=tk.NORMAL)
            log_text.insert(tk.END, line + "\n")
            log_text.see(tk.END)
            log_text.configure(state=tk.DISABLED)
        root.after(0, _do)

    # 启动期提示: 把 torch_bootstrap 收集到的诊断 (如 N 卡未启用 GPU 加速) 打到日志区
    try:
        for _n in torch_bootstrap.BOOTSTRAP_NOTES:
            _log_line("[torch] " + _n)
    except Exception:
        pass

    # ── 开始批量任务 ──
    _running = False

    def _start_batch():
        nonlocal _running
        if _running:
            return
        if not _task_store:
            messagebox.showinfo("提示", "请先添加任务（拖放台本+音频文件）")
            return

        lora = lora_var.get().strip()
        if lora and not os.path.isfile(lora):
            messagebox.showwarning("警告", f"LoRA 文件不存在:\n{lora}\n将使用基础 MMS-FA 模型")
            lora = ""

        # 读取高级参数（主线程安全读取 tkinter 变量）
        cfg = {
            'dev': dev_var.get(),
            'vram_chunk': vram_chunk_var.get(),
            'end_extend': end_extend_var.get(),
            'prune': prune_var.get(),
            }

        # 过滤掉没有音频的任务
        valid_tasks = [(iid, t) for iid, t in _task_store.items()
                       if t.get('audio') and os.path.isfile(t['audio'])]
        if not valid_tasks:
            messagebox.showinfo("提示", "没有可执行的任务（所有任务都缺少音频文件）")
            return

        _running = True
        run_btn.config(state=tk.DISABLED, text="  ⏳ 运行中...")
        progress["value"] = 0
        _log_line(f"\n{'='*60}")
        _log_line(f"开始批量处理 {len(valid_tasks)} 个任务")
        _log_line(f"LoRA: {os.path.basename(lora) if lora else '(无)'}")
        _log_line(f"设备: {'CUDA (GPU 加速)' if cfg['dev'] == 'cuda' else 'CPU (内置)'}")
        _log_line(f"{'='*60}")

        total = len(valid_tasks)

        def _process_next(idx):
            if idx >= total:
                nonlocal _running
                _running = False
                _release_gpu()   # 整批跑完: 释放所有 GPU 资源
                progress["value"] = 100
                done = sum(1 for t in _task_store.values() if t['status'] == 'done')
                failed = sum(1 for t in _task_store.values() if t['status'] == 'failed')
                _log_line(f"\n{'='*60}")
                _log_line(f"全部完成!  成功 {done}, 失败 {failed}")
                _log_line(f"{'='*60}\n")
                update_status(f"完成 — 成功 {done}, 失败 {failed}")
                root.after(0, lambda: run_btn.config(
                    state=tk.NORMAL, text="▶  开始全部"))
                return

            iid, task = valid_tasks[idx]
            name = task['name']

            # 更新列表状态为运行中
            root.after(0, lambda t=task, iid=iid: (
                t.update({'status': 'running'}),
                _set_status(iid, 'running'),
                task_tree.see(iid)
            ))

            _log_line(f"\n{'─'*50}")
            _log_line(f"[{idx+1}/{total}] {name}")
            _log_line(f"  台本: {task['script']}")
            _log_line(f"  音频: {task['audio']}")

            def update_pct(pct, msg):
                # 每任务进度映射到全局进度的对应区间
                global_pct = (idx * 100 + pct) / total
                root.after(0, lambda gp=global_pct, m=msg: (
                    progress.configure(value=gp), update_status(m)
                ))

            def worker_done(ok, out, err):
                nonlocal idx
                if ok:
                    root.after(0, lambda t=task, iid=iid: (
                        t.update({'status': 'done', 'output': out}),
                        _set_status(iid, 'done')
                    ))
                else:
                    _log_line(f"  [X] 失败: {err}")
                    root.after(0, lambda t=task, iid=iid: (
                        t.update({'status': 'failed', 'error': err}),
                        _set_status(iid, 'failed')
                    ))
                root.after(0, lambda: _process_next(idx + 1))

            def worker_thread():
                lora_path = lora if lora else None
                ok, out, err = run_pipeline(
                    task['script'], task['audio'],
                    lora_path, None, update_pct,
                    dev=cfg['dev'],
                    vram_chunk=cfg['vram_chunk'],
                    persist_gpu=True,
                    line_end_extend=float(cfg.get('end_extend', '0') or '0'),
                    collapse_prune=cfg.get('prune', False))
                root.after(0, lambda: worker_done(ok, out, err))

            threading.Thread(target=worker_thread, daemon=True).start()

        _process_next(0)

    run_btn.configure(command=_start_batch)

    # ── 居中窗口 ──
    root.update_idletasks()
    w, h = root.winfo_width(), root.winfo_height()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"+{(sw-w)//2}+{(sh-h)//2}")

    # ── 窗口关闭时保存设置 ──
    def _on_close():
        _save_settings()
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", _on_close)

    root.mainloop()


# ═══════════════════════════════════════════════════════════════════
#  6. CLI 模式
# ═══════════════════════════════════════════════════════════════════
def cli_mode():
    import argparse
    ap = argparse.ArgumentParser(description="FA-ASMR 命令行模式")
    ap.add_argument("--script", required=True, help="台本 .txt")
    ap.add_argument("--audio", required=True, help="音频 .wav/.mp3/.flac")
    ap.add_argument("--lora", default=None, help="LoRA checkpoint")
    args = ap.parse_args()

    lora = args.lora or os.path.join(SCRIPT_DIR, "lora", "fa_asmr_e30.pt")
    if not os.path.isfile(lora):
        lora = None
    ok, out, err = run_pipeline(args.script, args.audio, lora, None, None)
    if ok:
        print(f"→ {out}")
    else:
        print(f"失败: {err}")
        sys.exit(1)


if __name__ == "__main__":
    if "--cli" in sys.argv:
        sys.argv.remove("--cli")
        cli_mode()
    else:
        build_gui()
