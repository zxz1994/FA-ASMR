# -*- coding: utf-8 -*-
"""FA-ASMR V2 批量强制对齐工具 (ASMR · 二次元美化版)
- 拖放/选择多组台本+音频文件，自动按文件名配对
- 自动台本注音 → MMS-FA + E30 LoRA 双声道 Better-Ear 全局对齐 → 标准 LRC/SRT 输出
- 可选 Whisper ASR 音素映射与冻结 E30 V4 守门重解码
- 默认常开静音吸附优化
"""
from __future__ import annotations
from typing import Optional, Dict, Any

import sys, os, io, re, time, threading

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

AUDIO_EXTS = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac', '.opus', '.wma'}

# 台本原文还原用的归一化函数引用（惰性解析，避免导入期强依赖 furigana 包）。
def _RA_NORMALIZE(text: str) -> str:
    try:
        from fa_asmr_converter import RubyAnnotator
        return RubyAnnotator.normalize(text)
    except Exception:
        return text


# ── 说话人标记 ──────────────────────────────────────────────────────
# 统一形态：〖标记〗（U+3016 / U+3017 包裹）。标记内容可为圈号（①）或人名
# （フウカ），项目不关心它具体是什么，只当作分组键。
# 选型依据（均实测）：
#   · 注音前整体剥离 —— 标记内容不产任何 token，天然不进对齐
#   · 〖〗 NFKC 稳定、台本零占用、不与 {}（振假名）/ []（romaji）语法冲突
#   · 有明确边界 —— 刻意不认裸圈号：句中的 ①（「それは①番だ」）会被误判成
#     说话人切换，歧义无法消除；加括号后标记与正文再无混淆
# 文件名后缀同用此形态：原名〖①〗〖②〗.txt
_SPEAKER_BRACKET_RE = re.compile("\u3016\\s*([^\u3016\u3017]*?)\\s*\u3017")
# 圈数字各区：①-⑳ / ⑴-⒇ / ㉑-㉟ / ㊱-㊿ / ➀-➓
_SPEAKER_SEP = "\u3001,\uff0c\u30fb&/ \t"


def _iter_speaker_marks(line):
    """产出 (起点, 终点, 说话人名)。只认〖…〗形式。

    刻意**不认**裸圈号 ①：句中的 ①（如「それは①番だ」）会被误切成说话人切换，
    歧义无法消除。用〖〗界定后，标记与正文再无混淆可能。
    """
    spans = []
    for m in _SPEAKER_BRACKET_RE.finditer(line):
        spans.append((m.start(), m.end(), m.group(1).strip()))
    spans.sort(key=lambda x: x[0])
    return spans


def _split_speaker_segments(line, cur):
    """按说话人标记把一行切成 [(说话人元组, 文本), ...]。

    标记作用范围 = 从标记处到「下一个标记或行尾」。标记之间只有分隔符
    （〖A〗〖B〗 / ①、②）时视为多人并列（同时说话），合并为一个元组。
    未出现标记的文本沿用上一行说话人（cur），故「换人才写标记」可行。
    返回 (切分结果, 行末说话人)。
    """
    marks = _iter_speaker_marks(line)
    if not marks:
        return ([(cur, line)] if line else []), cur
    segs = []
    pos = 0
    names = cur
    i = 0
    n = len(marks)
    while i < n:
        a, b, name = marks[i]
        pre = line[pos:a]
        if pre:
            segs.append((names, pre))
        # 收集连续标记：标记之间只隔分隔符时并入同一组（多人并列）
        group = [name]
        last_end = b
        j = i + 1
        while j < n:
            na, nb, nname = marks[j]
            between = line[last_end:na]
            # 标记之间为空串或纯分隔符 -> 视为同一组（多人并列）
            if all(c in _SPEAKER_SEP for c in between):
                group.append(nname)
                last_end = nb
                j += 1
            else:
                break
        names = tuple(group)
        pos = last_end
        i = j
    rest = line[pos:]
    if rest:
        segs.append((names, rest))
    return segs, names

_RUBY_ANNOTATOR = None


class AutoRuby:
    def __init__(self):
        global _RUBY_ANNOTATOR
        if _RUBY_ANNOTATOR is None:
            import os as _os
            import sys as _sys
            _td = _os.path.join(SCRIPT_DIR, "furigana")
            if _td not in _sys.path:
                _sys.path.insert(0, _td)
            from fa_asmr_converter import RubyAnnotator
            _RUBY_ANNOTATOR = RubyAnnotator(verbose=False, strip_inline_readings=True)
        # 批量任务复用同一实例：fugashi(UniDic) / pykakasi / yomikata(dBert)
        # 只初始化一次，避免每条音频都重载 UniDic 词典与异读模型。
        self._ann = _RUBY_ANNOTATOR

    def annotate(self, text: str) -> str:
        return self._ann.annotate(text)

    def release_gpu(self):
        global _RUBY_ANNOTATOR
        try:
            self._ann.release_gpu()
        except Exception:
            pass
        _RUBY_ANNOTATOR = None


def _fmt_lrc_time(t):
    """把 [MM:SS:CC]（align_utils._format_time 产出）转成 [MM:SS.CC]。

    原为 build_standard_lrc._fmt_time 与 _fmt_line._t 两份完全相同的实现，
    现统一为此函数。两者原先都用 re.match（锚定行首），此处保持一致。

    历史教训（两处曾各自失配，均造成严重后果）：
      - 曾误写成两段式 \\[\\d{2}:\\d{2}\\] 却访问 group(3)，恒不匹配 ->
        返回 None -> 结束标记行被静默丢弃，行尾延长一并失效。
      - 分钟位曾写死 \\d{2}，而 _format_time 的 f"{mins:02d}" 不截断，
        mins >= 100（音频 >= 100 分钟）时产出 3 位 -> 整条字幕退化成
        无时间戳的裸文本。
    故正则统一引用 align_utils.LRC_TS_ANCHORED，不再各自维护字面。
    """
    if not t or t == "[error]":
        return None
    m = au.LRC_TS_ANCHORED.match(t)
    return f"[{m.group(1)}:{m.group(2)}.{m.group(3)}]" if m else None


def build_standard_lrc(result_list):
    lines = []
    group_start = None
    group_end = None
    # 字幕行按【说话人段】收集：同一行内说话人变化处补该说话人的「②：」前缀。
    # 结构 = [[说话人元组或None, 文本], ...]。无标记作品恒为单段 None，输出不变。
    group_parts = []

    def _flush():
        if group_parts and any(t for _, t in group_parts) and group_start:
            lines.append(_fmt_line(group_start, group_end, group_parts))
            if group_end and group_end != group_start:
                t = _fmt_lrc_time(group_end)
                if t:
                    lines.append(t)

    for item in result_list:
        io = item.get("orig", "")
        # 输出层还原：优先用台本原文子串（_src）。orig 是归一化后的字符
        # （NFKC 把 ① 拍平成 1 等），_src 保留原字符，仅换显示、不动对齐标签。
        io_out = item.get(F.SRC, None)
        if io_out is None:
            io_out = io
        if item.get("type") == 0 and io == "\n":
            _flush()
            group_start = None
            group_end = None
            group_parts = []
            continue
        # 被拟声守门删除的 token：不贡献时间边界，也不进入字幕文本
        if item.get(F.ONO_DROPPED):
            continue
        if "start" in item and group_start is None:
            group_start = item["start"]
        if "end" in item:
            group_end = item["end"]
        # 说话人切换：本 token 带 _spk 且与当前段不同 -> 开新段（行内补前缀）。
        # 无 _spk 的 token 沿用当前段（无标记作品全程如此）。
        _spk = tuple(item[F.SPK]) if item.get(F.SPK) else None
        if _spk is not None and (not group_parts or group_parts[-1][0] != _spk):
            group_parts.append([_spk, ""])
        if not group_parts:
            group_parts.append([None, ""])
        if _is_text_token(item) and io:
            if not item.get(F.RUBY_PLACEHOLDER, False):
                group_parts[-1][1] += io_out
        elif item.get("type") == 0 and io not in ("\n", ""):
            group_parts[-1][1] += io_out

    _flush()
    return "\n".join(lines)


def _fmt_line(start_str, end_str, text, speakers=None):
    """组装一行字幕。text 可为字符串（旧式）或 [(说话人元组|None, 文本), ...]。

    逐段处理：每段先剥 ruby 与行首孤立标点，再在段前加「说话人：」前缀。
    行内说话人切换因此表现为 B 段前多一个「②：」，而非另起一行。
    多人并列用 、 连接。无标记作品只有一段 None，输出与原来完全一致。
    """
    if isinstance(text, str):
        parts = [(speakers, text)]
    else:
        parts = [(spk, seg) for spk, seg in text]
    buf = []
    first = True
    for spk, seg in parts:
        if not seg:
            continue
        clean = re.sub(r"\{([^|]+)\|[^}]+\}", r"\1", seg).rstrip()
        if first:
            clean = _strip_orphan_punct(clean)
            first = False
        if spk:
            names = "\u3001".join(s for s in spk if s)
            if names:
                clean = f"{names}\uff1a{clean}"
        buf.append(clean)
    clean = "".join(buf)
    t = _fmt_lrc_time(start_str)
    return f"{t}{clean}" if t else clean


# 行首孤立标点剔除：拟声子句被删后，附着其上的后置标点（、。…?！等）
# 可能残留成行首「、んん」「? 何にって」，读起来是空壳。
#
# 只剔除【行首】——行尾标点是日语的句末终结符（。？！…），必须保留，
# 否则 `だけど。` / `でしょ?` 会被削成 `だけど` / `でしょ`（曾误伤过）。
# 也不含表达性符号：ー/~（长音）、♪♡（语气）、「」『』（）（引用）。
_PUNCT_CHARS = "、。！？…‥・,.:;!?"


def _strip_orphan_punct(text: str) -> str:
    """去掉【行首】的孤立标点（行尾标点保留，它们是句末终结符）。"""
    s = text.lstrip()
    i = 0
    while i < len(s) and s[i] in _PUNCT_CHARS:
        i += 1
    return s[i:].lstrip()


def _source_norm_map(src, norm):
    """建立【归一化文本位置】->【原文字符下标】的单调映射。

    RubyAnnotator.normalize 对绝大多数字符逐字符独立（实测 2626/2627 行），
    故可把每个原文下标映射到它在归一化文本里占据的区间。返回与 norm 等长的
    原文下标列表（第 k 项 = norm[k] 对应的 src 下标）。

    逐位对齐 src 与 norm，三种情形：
      · normalize(src[i]) 非空且与 norm[j:] 前缀相符 -> 正常消费双方；
      · normalize(src[i]) 为空但 src[i] 本身 == norm[j]（如空格：整体 normalize
        因 strip 删了它，norm 里却保留）-> 该字符存活，消费双方；
      · 其余 -> 该源字符被清洗掉，只推进 src。
    """
    m = []
    i = 0
    j = 0
    n_src = len(src)
    n_norm = len(norm)
    while i < n_src and j < n_norm:
        try:
            out = _RA_NORMALIZE(src[i])
        except Exception:
            out = src[i]
        if out and norm.startswith(out, j):
            m.extend([i] * len(out))
            j += len(out)
            i += 1
        elif src[i] == norm[j]:
            m.append(i)
            j += 1
            i += 1
        else:
            i += 1
    while j < n_norm:
        m.append(i - 1 if i > 0 else 0)
        j += 1
    return m


def _attach_source_spans(tokens, annotated_line, source_line, normalize_fn):
    """给每个 token 挂 _src —— 该 token 在【原始台本行】里对应的原文子串。

    对齐必须消费归一化后的 token（NFKC 把 ① 拍平成 1 等），但字幕输出要还原
    台本原字符。此处按 token 的 orig 在归一化文本中的位置反查原文区间，组装
    LRC 时优先用 _src，即可显示原文而不影响对齐标签。

    分两遍：
      1. 逐 token 在归一化文本里定位 orig（多数为直接子串）。定位失败（音变如
         う->ー、tokenizer 丢字如中点）时，用「下一个成功定位的 token」回推边界，
         保证区间连续、无空洞。
      2. 把相邻 token 之间的原文缝隙（被 tokenizer 丢弃的字符）并入前一个 token，
         行首未认领部分并入第一个 token，行尾剩余并入最后一个 token。
    """
    if not source_line:
        return
    try:
        norm = normalize_fn(source_line)
    except Exception:
        return
    if not norm:
        return
    m = _source_norm_map(source_line, norm)
    if len(m) != len(norm):
        return
    # ---- 第一遍：顺序定位每个可见 token 在 norm 中的区间（子串匹配） ----
    anchors = []      # (token_idx, norm_start, norm_end) 或 None
    pos = 0
    for k, t in enumerate(tokens):
        o = t.get("orig", "")
        if not o or t.get(F.RUBY_PLACEHOLDER):
            anchors.append(None)
            continue
        a = norm.find(o, pos)
        if a >= 0:
            anchors.append((k, a, a + len(o)))
            pos = a + len(o)
        else:
            anchors.append(None)
    # 若整行一个都没定位到，放弃（保持 orig 回退）
    if not any(a for a in anchors):
        return
    # ---- 第一遍补：失败项用左右锚点回推，让 norm 区间连续 ----
    n_anchors = len(anchors)
    for k in range(n_anchors):
        if anchors[k] is not None:
            continue
        if not tokens[k].get("orig") or tokens[k].get(F.RUBY_PLACEHOLDER):
            continue
        prev_end = 0
        for j in range(k - 1, -1, -1):
            if anchors[j] is not None:
                prev_end = anchors[j][2]
                break
        next_start = len(norm)
        for j in range(k + 1, n_anchors):
            if anchors[j] is not None:
                next_start = anchors[j][1]
                break
        if next_start < prev_end:
            next_start = prev_end
        anchors[k] = (k, prev_end, next_start)
    # ---- 第二遍：norm 区间 -> 原文下标区间，并补齐缝隙 ----
    last_src = 0
    for k in range(n_anchors):
        a = anchors[k]
        if a is None:
            continue
        _, ns, ne = a
        if ne <= ns:
            continue
        lo = m[ns]
        i2 = min(ne - 1, len(m) - 1)
        hi = m[i2] + 1
        # 缝隙补齐：本 token 起点之前的未认领原文并入本 token
        if lo > last_src:
            lo = last_src
        if hi < lo:
            hi = lo
        tokens[k][F.SRC] = source_line[lo:hi]
        last_src = hi
    # ---- 行尾剩余并入最后一个挂到 _src 的 token ----
    if last_src < len(source_line):
        for k in range(n_anchors - 1, -1, -1):
            if anchors[k] is not None:
                tokens[k][F.SRC] = tokens[k].get(F.SRC, "") + source_line[last_src:]
                break


def _release_gpu(ruby=None):
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


def run_pipeline(
    script_path: str,
    audio_path: str,
    lora_path: Optional[str],
    log_callback,
    progress_callback,
    dev="auto",
    vram_chunk=6000,
    persist_gpu=False,
    line_end_extend=0.0,
    enable_silence_snap=True,
    enable_resegment=True,
    enable_ono_gate=True,
    enable_merge_short=True,
    merge_short_chars=20,
    merge_short_gap=1.5,
    merge_ono_chars=40,
    merge_ono_gap=2.5,
):
    """V2 核心对齐管线执行入口 (默认常开静音吸附与拟声守门)"""
    def log(msg):
        if log_callback:
            log_callback(msg)
        try:
            print(msg)
        except Exception:
            pass

    progress_callback and progress_callback(0, "🌸 初始化台本与环境...")
    total_start = time.time()
    _ruby = None

    try:
        log("[1/4] 🌸 读取台本并进行日文注音转换...")
        with open(script_path, encoding="utf-8") as f:
            raw_text = f.read()

        ruby = AutoRuby()
        _ruby = ruby
        lines = raw_text.splitlines()

        # 说话人标记在【注音前】剥离：〖名字〗整段不进入 annotate / tokenize，
        # 因此天然不产生任何发音 token、不进对齐。剥离后记录每个「注音行」的
        # 说话人归属，供后续按说话人分组对齐使用。
        #   _spk_of_line[k] = 注音行 k 的说话人元组（可能多人并列）
        # 原文还原不依赖此处的下标映射：_attach_source_spans 直接按行号取
        # _body_lines（见下方 _src_line = lines[_li]）。
        _spk_of_line = []
        _body_lines = []
        _cur_spk = ()
        for _raw in lines:
            _segs, _cur_spk = _split_speaker_segments(_raw, _cur_spk)
            for _spk, _body in _segs:
                _body_lines.append(_body)
                _spk_of_line.append(_spk)
        # 逐行清洗「内联振假名泄露」后再注音。
        # 部分台本把振假名写成未标记的行内假名（身体んたい / 沖縄きなわ /
        # 気持ちちいい），这些残留假名会被分词器当成实词，污染注音与对齐标签，
        # 并经 _src 回填漏进字幕正文。
        # 清洗必须在 annotate 之前、且必须同时作用于 _body_lines（供 _src 还原）
        # —— 否则 _attach_source_spans 的缝隙补齐会把刚剥掉的假名塞回正文。
        _leak_cleaner = None
        try:
            from inline_leak_clean import InlineLeakCleaner
            _ann = getattr(ruby, "_ann", None)
            if _ann is not None and getattr(_ann, "_tagger", None):
                _leak_cleaner = InlineLeakCleaner(_ann)
        except Exception as _e:
            log(f"  ⚠ 内联假名清洗器初始化跳过: {_e}")

        if _leak_cleaner is not None:
            _n_leak = 0
            for _k, _b in enumerate(_body_lines):
                _c = _leak_cleaner.clean(_b)
                if _c != _b:
                    _body_lines[_k] = _c
                    _n_leak += 1
            if _n_leak:
                log(f"  ✨ 内联振假名清洗: 修正 {_n_leak} 行")
            # 同步净化 _src 还原源（同一份 _body_lines 已被改写，见下）

        # 逐行注音（正文，不含标记）
        annotated_lines = [ruby.annotate(b) for b in _body_lines]
        annotated_text = "\n".join(annotated_lines)
        # 原文文本行（供 _src 还原）：剥离标记 + 内联假名清洗后的正文
        lines = _body_lines
        log(f"  ✨ 台本处理完成 ({len(lines)} 行)")

        log("[2/4] 🎵 解析音素与规范化...")
        result_list = []
        lang = "auto"

        # 子句假名数统计（供 merge_short 判字数上限）：
        # 汉字在注音阶段已转成读音，只有在这一步（annotated_text 还在时）
        # 才拿得到完整读音。token 化之后 orig 只剩原文、pron 是按 token 压缩的
        # 罗马音（如 {貴方|あなた} 拆成一个 token、pron 只有 'a'），都无法还原读音长度。
        #
        # 口径：
        #   假名     -> 每个字符算 1（ひゃ=2、のー=2、んっ=2）
        #   数字字符 -> 每个算 2（"123" = 6）
        #   英文字母 -> 每个算 1（"ABC" = 3）
        # 统计结果写到该子句内每个 token 的 `_kana_n`（子句内所有 token 相同）。
        _KANA_CH = re.compile(r"[\u3041-\u309F\u30A0-\u30FF]")
        _DIGITS = re.compile(r"[0-9]")
        _ALPHA = re.compile(r"[A-Za-z]")

        def _text_weight(txt: str) -> int:
            """一段文字的权重：假名×1 + 数字字符×2 + 字母×1。"""
            if not txt:
                return 0
            return len(_KANA_CH.findall(txt)) + 2 * len(_DIGITS.findall(txt)) \
                + len(_ALPHA.findall(txt))

        def _line_kana_counts(line: str, toks):
            """按子句切分，返回每个 token 所属子句的假名数列表（与 toks 等长）。

            line 是注音串（`{漢字|かんじ}` 格式）；toks 是它 token 化后的列表。
            两边都按同一判据（无读音的符号 = 子句边界）切分，切点数一致。
            """
            # 注音串剥出读音：{漢字|かんじ} -> かんじ
            plain = re.sub(r"\{([^|]+)\|([^}]+)\}", r"\2", line)
            # 注音串按「非假名非英数」的符号切子句。
            # 延音符号 ー/〜/～/~ 必须并入段内：它们属发音内容（token 侧归入带
            # pron 的 token，如 な~~~ pron=naaaaaaaaa），若在此被当分隔符会与
            # token 侧子句数错位，导致字数取到错误段（实测 な～～～んにも…
            # 被切成 ['な','んにも…']，整行 15 字只算成首段 'な' = 1 字）。
            _SEG_KEEP = "\u3041-\u309F\u30A0-\u30FFA-Za-z0-9\u30FC\u301C\uFF5E~\u2015\u2014\u2010\uFF0D-"
            segs = re.split("[^" + _SEG_KEEP + "]+", plain)
            segs = [s for s in segs if s]
            # token 侧按同一判据切子句
            tsegs, cur = [], []
            for t in toks:
                o = t.get("orig", "")
                if o == "\n":
                    continue
                if t.get("pron"):
                    cur.append(t)
                elif o and not t.get(F.RUBY_PLACEHOLDER, False):
                    if cur:
                        tsegs.append(cur)
                        cur = []
            if cur:
                tsegs.append(cur)

            # 子句数对齐后，把每段的字数摊到该段 token 上
            out = [0] * len(toks)
            idx_of = {id(t): k for k, t in enumerate(toks)}
            for si, seg_toks in enumerate(tsegs):
                n = _text_weight(segs[si] if si < len(segs) else "")
                if n == 0:
                    # 极端情况（注音串切分与 token 侧不同步）：回退数 orig
                    n = sum(_text_weight(t.get("orig", "")) for t in seg_toks)
                for t in seg_toks:
                    out[idx_of[id(t)]] = n
            return out

        for _li, line in enumerate(annotated_text.splitlines(keepends=True)):
            if line.strip():
                _toks = hn.process_haruhi_line(line, lang, 0, 1)
                _counts = _line_kana_counts(line, _toks)
                for _t, _n in zip(_toks, _counts):
                    _t[F.KANA_N] = _n
                # 挂 _src：该 token 对应的【原文台本字符】，供 LRC 还原原字符
                # （对齐仍消费归一化后的 token，仅输出层用原文，故标签不受影响）
                _src_line = lines[_li] if _li < len(lines) else ""
                _attach_source_spans(_toks, line, _src_line, _RA_NORMALIZE)
                # 挂 _spk：本行说话人元组（注音前已从标记剥离得到）。
                # 仅用于最终字幕前缀标注，不参与对齐。
                _spk_line = _spk_of_line[_li] if _li < len(_spk_of_line) else ()
                if _spk_line:
                    for _t in _toks:
                        _t[F.SPK] = _spk_line
                result_list.extend(_toks)
                # 每个源行后插入行边界，避免整轨被并成单一对齐块（会导致 LRC 塌成极少行）。
                result_list.append({"orig": "\n", "type": 0, "pron": ""})
        if result_list and result_list[-1].get("orig") != "\n":
            result_list.append({"orig": "\n", "type": 0, "pron": ""})

        alignment_tokens = []
        line_group = []
        # per-token 易错拟声掩码（与 alignment_tokens 同步累积，按子句整体判定）。
        # 供对齐阶段降低这些 token 的发射分权重，削弱不可信的拟声时长对相邻
        # 语义段的挤压。is_easy_ono 需假名原文，故必须在 GUI 侧（有 orig）算。
        ono_clause_flags = []   # 每子句 1 个 flag (1 = 易错拟声子句)
        _cur_ono_surface = []
        token_to_index_map = {}
        current_line = []
        flat_idx = 0
        src_idx = 0
        # 子句边界判据（已统一到 clause.py；ono_filter / merge_short /
        # resegment 全部引用同一份，此处也直接引 clause，少一层间接）
        from clause import is_clause_boundary as _is_clause_boundary
        for i, item in enumerate(result_list):
            orig = item.get("orig", "")
            pron = item.get("pron", "")
            # 子句边界：换行，或「未转化为 token (无 pron) 且非注音占位符」的符号项
            is_boundary = _is_clause_boundary(item)
            if is_boundary:
                if current_line:
                    alignment_tokens.append(current_line)
                    line_group.append(src_idx)
                    try:
                        from ono_filter import is_easy_ono as _ieo
                        _ez = 1 if _ieo("".join(_cur_ono_surface)) else 0
                    except Exception:
                        _ez = 0
                    ono_clause_flags.append(_ez)
                    _cur_ono_surface = []
                    current_line = []
            if orig == "\n":
                src_idx += 1
            if pron:
                current_line.append(pron)
                _cur_ono_surface.append(orig)
                token_to_index_map[flat_idx] = i
                flat_idx += 1
        if current_line:
            alignment_tokens.append(current_line)
            line_group.append(src_idx)
            try:
                from ono_filter import is_easy_ono as _ieo
                _ez = 1 if _ieo("".join(_cur_ono_surface)) else 0
            except Exception:
                _ez = 0
            ono_clause_flags.append(_ez)

        n_tokens = sum(len(lt) for lt in alignment_tokens)
        log(f"  ✨ 解析完成: {n_tokens} tokens, {len(alignment_tokens)} 行")

        progress_callback and progress_callback(25, "⚡ 双声道对齐计算中...")
        log("[3/4] 🎧 MMS-FA + LoRA 双声道 Better-Ear 全局 CTC 对齐 (静音吸附已启用)...")

        # 注意: 必须先判 None/空值。str(None).upper() == "NONE" 会命中显式禁用分支，
        # 导致下面的 default 回退永不执行（历史逻辑缺陷）。
        #
        # _lora_missing 记录「本想用 LoRA、但权重文件不存在」这一情形 —— 它与
        # 「调用方显式要求不用 LoRA」在结果上都是走基础模型，但含义完全不同：
        # 前者是缺件（精度会下降且不易察觉），必须显式告警；后者是用户的意图。
        _lora_missing = None
        if lora_path is None or str(lora_path).strip() == "":
            default_lora = str(getattr(align, "DEFAULT_LORA_PATH", ""))
            if os.path.isfile(default_lora):
                lora_path = default_lora
            else:
                lora_path = None
                _lora_missing = default_lora
        elif str(lora_path).upper() in ("NO_LORA", "NONE", "DISABLE"):
            lora_path = "NO_LORA"
        elif not os.path.isfile(lora_path):
            default_lora = str(getattr(align, "DEFAULT_LORA_PATH", ""))
            if os.path.isfile(default_lora):
                lora_path = default_lora
            else:
                lora_path = None
                _lora_missing = default_lora

        if lora_path and lora_path != "NO_LORA":
            log(f"  ✨ 使用 LoRA: {os.path.basename(lora_path)}")
        elif _lora_missing:
            # 缺件不等于「用户选择不用」：显式提示，避免精度下降被静默接受。
            log("  ⚠ 未找到 LoRA 权重，已回退到基础 MMS-FA 模型 —— 对齐精度会下降。")
            log(f"     期望路径: {_lora_missing}")
            log("     请把 lora/fa_asmr_v9e10.pt 放到该位置；若确要用基础模型，"
                "传 lora_path='NO_LORA' 可关闭本提示。")
        else:
            log("  ✨ 使用基础 MMS-FA 模型")

        # 标点位置 Blank 豁免掩码：对应相邻两个 token (k 与 k+1) 之间夹着的
        # Blank (下标 k+1)。
        #
        # **当前恒为全 False（豁免停用）**，原因是实测结论而非疏漏：
        # 该机制把标点相邻的 Blank 从 intra_blank_penalty 中豁免，但在 v9e10
        # 权重下 intra_blank_penalty 已重标定为 0.10，豁免反而削弱了正确的
        # 边界约束——关闭后 MAE 0.503->0.501（4/1158 行改善）。
        # 若需恢复旧行为，按下标关系重填掩码即可（判据：k 与 k+1 两个发音
        # token 之间在 result_list 中存在未转化为 token 的标点符号，且排除
        # _ruby_placeholder 占位符），并同步复跑 eval/rerun_check.py。
        blank_punct_mask = [False] * (n_tokens + 1)

        _align_diag: Dict[str, Any] = {}
        alignment_results = align.align_audio_with_text(
            audio_file_path=audio_path,
            text_tokens=alignment_tokens,
            lora_path=lora_path,
            device=dev,
            chunk_frames=int(vram_chunk),
            enable_silence_snap=enable_silence_snap,
            blank_punct_mask=blank_punct_mask,
            line_group=line_group,
            ono_clause_flags=ono_clause_flags,
            log_callback=log,
            diagnostics_out=_align_diag,
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

        # ---- 拟声错位处理 · 阶段一【标记】----------------------------
        # 跑结构判据给可疑拟声子句打 `_ono_mark`，**一个 token 都不删**。
        # 数据保持完整，后续 resegment / 短句合并才能切出真实的最终行结构；
        # 决策放到阶段三、在最终行结构上一次性做（见下方 decide_ono_deletions）。
        if enable_ono_gate:
            try:
                from ono_filter import mark_ono_candidates as _ono_mark
                _ono_mark(
                    result_list=result_list,
                    alignment_results=alignment_results,
                    token_to_index_map=token_to_index_map,
                    log_callback=log,
                )
            except Exception as _e:
                log(f"  ⚠ 拟声标记跳过: {_e}")

        # ---- 标点驱动重分段（候选=标点，证据=两侧 CTC blank 间隔；标点保留） ----
        if enable_resegment:
            try:
                from resegment import apply_punctuation_resegment as _reseg
                # 用全量 token 的停顿证据定行边界。此时还没有任何 token 被删
                # （阶段一只打标记），所以这就是真实的朗读结构。
                _rl_time = {}
                for _k, _r in enumerate(alignment_results):
                    _ri = token_to_index_map.get(_k)
                    if _ri is not None:
                        _rl_time[_ri] = (float(_r.get("original_start", 0.0)),
                                         float(_r.get("original_end", 0.0)))
                _n_before = sum(1 for _it in result_list if _it.get("orig") == "\n") + 1
                result_list = _reseg(result_list, _rl_time, frame_sec=0.02, thr_frames=50, max_line_chars=22)
                _n_after = sum(1 for _it in result_list if _it.get("orig") == "\n") + 1
                log(f"  ✨ 标点重分段: {_n_before} → {_n_after} 行（标点保留）")
            except Exception as _e:
                log(f"  ⚠ 标点重分段跳过: {_e}")

        # ---- 短句合并：把 resegment 切碎的相邻短句重新合并 ----
        # 合并条件：前行可见字符数 <= max_chars 且与下一行起点间隔 <= max_gap。
        # 不区分台本 \n 与 resegment \n —— 台本按语义断行、朗读却是连读的，
        # 被行界切开的两半本是同一口气，硬拆会让后半的拟声被判成「行首问题」。
        if enable_merge_short:
            try:
                from merge_short import merge_short_lines as _merge, count_inserted_breaks as _cb
                _nb = _cb(result_list)
                result_list = _merge(result_list, max_chars=merge_short_chars,
                                     max_gap=merge_short_gap,
                                     ono_max_chars=merge_ono_chars,
                                     ono_max_gap=merge_ono_gap)
                _na = _cb(result_list)
                if _nb != _na:
                    log(f"  ✨ 短句合并: 合并 {_nb - _na} 处碎句")
            except Exception as _e:
                log(f"  ⚠ 短句合并跳过: {_e}")

        # ---- 拟声错位处理 · 阶段三【决策】----------------------------
        # 最终字幕行结构已定，此刻才决定删哪些标记过的拟声，并一次性应用。
        # 规则见 ono_filter.decide_ono_deletions 的文档。
        if enable_ono_gate:
            try:
                from ono_filter import decide_ono_deletions as _ono_decide
                _ono_decide(result_list=result_list, log_callback=log)
            except Exception as _e:
                log(f"  ⚠ 拟声决策跳过: {_e}")

        for i in range(len(result_list)):
            item = result_list[i]
            # 被拟声守门删除的 token：不借时间，保持无时间状态以从 LRC 剔除
            if item.get(F.ONO_DROPPED):
                continue
            if "start" not in item and item.get("type", 0) in [1, 2, 3, 4, 5]:
                # 行内有界时间借用：绝不越过换行符 \n 跨行偷时间
                found = False
                # 1. 优先在当前行内向左寻找前一个有时间的 item
                for j in range(i - 1, -1, -1):
                    if result_list[j].get("type") == 0 and result_list[j].get("orig") == "\n":
                        break
                    # 不借被删拟声 token 的时间（本就不可靠）
                    if result_list[j].get(F.ONO_DROPPED):
                        continue
                    if "start" in result_list[j]:
                        item["start"] = result_list[j]["start"]
                        item["end"] = result_list[j]["end"]
                        found = True
                        break
                # 2. 若左侧已至行首，在当前行内向右寻找第一个有时间的 item
                if not found:
                    for j in range(i + 1, len(result_list)):
                        if result_list[j].get("type") == 0 and result_list[j].get("orig") == "\n":
                            break
                        if result_list[j].get(F.ONO_DROPPED):
                            continue
                        if "start" in result_list[j]:
                            item["start"] = result_list[j]["start"]
                            item["end"] = result_list[j]["end"]
                            break

        log("[4/4] 📜 组装标准 LRC 与 SRT 字幕...")
        lrc_content = build_standard_lrc(result_list)

        if line_end_extend > 0:
            from lrc_end_extend import extend_lrc
            lrc_content, _n_ext = extend_lrc(lrc_content, line_end_extend)
            if _n_ext:
                log(f"  ✨ 行尾延长 ({line_end_extend:.1f}s): 处理 {_n_ext} 个空白标记")

        # 输出字幕与【音频】同名，去掉台本的说话人后缀（原名_{①}{②}.txt -> 原名.lrc）。
        # 多说话人台本会给文件名加 _{标记} 后缀，若直接沿用会产出 原名_{①}{②}.lrc，
        # 与音频名不一致，播放器无法自动挂载。
        _base, _ = _parse_speaker_suffix(os.path.splitext(os.path.basename(script_path))[0])
        base = _base or os.path.splitext(os.path.basename(script_path))[0]
        out_dir = os.path.dirname(os.path.abspath(audio_path))
        lrc_path = os.path.join(out_dir, f"{base}.lrc")
        with open(lrc_path, "w", encoding="utf-8") as f:
            f.write(lrc_content)

        total_elapsed = time.time() - total_start
        n_lines = lrc_content.count("\n") + 1
        log(f"\n  💖 对齐完成! 总耗时 {total_elapsed:.1f}s")
        log(f"  🎯 输出字幕: {lrc_path} ({n_lines} 行)")

        progress_callback and progress_callback(100, "💖 全部完成!")
        if not persist_gpu:
            _release_gpu(_ruby)
        return True, lrc_path, None

    except Exception as e:
        import traceback
        err = traceback.format_exc()
        log(f"\n  💔 发生错误: {err}")
        if not persist_gpu:
            _release_gpu(_ruby)
        return False, None, str(e)


def _parse_speaker_suffix(stem):
    """从文件名 stem 尾部解析 〖标记〗〖标记〗... 后缀。

    形态: 音频名〖标记A〗〖标记B〗 或 音频名〖人名1〗〖人名2〗。
    标记一律用 〖〗 包裹，与正文标记同形，项目只看文件名即知本台本
    有哪些说话人及用什么标记书写。返回 (基名, 标记列表)；无后缀时返回 (stem, [])。
    """
    m = re.match("^(.*?)((?:\u3016[^\u3016\u3017]*\u3017)+)$", stem)
    if not m:
        return stem, []
    base = m.group(1)
    marks = [x.strip() for x in re.findall("\u3016([^\u3016\u3017]*)\u3017", m.group(2))]
    return base, [x for x in marks if x]


def auto_pair_files(file_paths):
    clean_paths = [os.path.abspath(f) for f in file_paths if f and str(f).strip()]
    # 排除本工具生成的派生产物（.dropped.txt 删除清单 / .lrc / .srt），
    # 否则它们会被当成台本误配对（.dropped.txt 的 splitext 后缀也是 .txt）。
    txt_files = [f for f in clean_paths
                 if os.path.splitext(f)[1].lower() == '.txt'
                 and not f.lower().endswith('.dropped.txt')]
    audio_files = [f for f in clean_paths if os.path.splitext(f)[1].lower() in AUDIO_EXTS]

    # 台本按「基名」聚合：文件名尾部 _{标记}{标记} 声明说话人，
    # 基名（剥掉后缀）用于与音频配对。无后缀的保持原样（单人作品）。
    txt_by_stem = {}
    for f in txt_files:
        stem = os.path.splitext(os.path.basename(f))[0]
        base, marks = _parse_speaker_suffix(stem)
        if base not in txt_by_stem:
            txt_by_stem[base] = {'files': [], 'marks': marks}
        txt_by_stem[base]['files'].append(f)
        if marks:
            txt_by_stem[base]['marks'] = marks

    audio_by_stem = {}
    for f in audio_files:
        stem = os.path.splitext(os.path.basename(f))[0]
        audio_by_stem.setdefault(stem, []).append(f)

    pairs = []
    unmatched = []

    common_stems = set(txt_by_stem.keys()) & set(audio_by_stem.keys())
    for stem in sorted(common_stems):
        info = txt_by_stem[stem]
        for txt in info['files']:
            for aud in audio_by_stem[stem]:
                pairs.append({'script': txt, 'audio': aud, 'name': stem,
                              'speaker_marks': list(info['marks'])})

    for stem in sorted(set(txt_by_stem.keys()) - set(audio_by_stem.keys())):
        for txt in txt_by_stem[stem]['files']:
            unmatched.append({'type': '缺音频', 'file': txt, 'name': stem})

    for stem in sorted(set(audio_by_stem.keys()) - set(txt_by_stem.keys())):
        for aud in audio_by_stem[stem]:
            unmatched.append({'type': '缺台本', 'file': aud, 'name': stem})

    return pairs, unmatched


# ═══════════════════════════════════════════════════════════════════
#  ASMR · 二次元 (Anime / ACG) 视觉设计系统
# ═══════════════════════════════════════════════════════════════════
BG = "#181926"             # 深邃夜空紫底色 (Catppuccin Macchiato / Tokyo Night)
CARD_BG = "#24273a"        # 柔和丝绒深紫卡片
CARD_BG_ALT = "#1e2030"    # 表头与较深次级卡片
CARD_INNER = "#2d3149"     # 输入与内嵌微亮区域
TEXT = "#cad3f5"           # 梦幻奶白紫文字
TEXT_MUTED = "#939ab7"     # 薰衣草淡灰
TEXT_ACCENT = "#f5c2e7"    # 浅樱粉高亮文字
ACCENT_PINK = "#f7768e"    # 樱花主粉色
ACCENT_PINK_HOVER = "#ff94a9"
ACCENT_PURPLE = "#bb9af7"  # 薰衣草梦幻紫
ACCENT_PURPLE_HOVER = "#cba6f7"
ACCENT_CYAN = "#7dcfff"    # 梦幻天青
BORDER = "#3b3f5c"         # 柔和描边
BORDER_ACTIVE = "#c6a0f6"  # 高亮紫描边
SUCCESS = "#a6e3a1"        # 抹茶嫩绿
WARN = "#f9e2af"           # 奶油暖黄
ERROR = "#f38ba8"          # 樱桃红

STATUS_ICONS = {
    'pending': '  ⏳ 待命中',
    'running': '  ⚡ 对齐中...',
    'done':    '  💖 已完成',
    'failed':  '  💔 失败',
    'warn':    '  ⚠️ 缺音频',
}


def build_gui():
    root = (TkinterDnD.Tk() if HAS_DND else tk.Tk())
    root.title("🌸 FA-ASMR V2 · 二次元音声对齐工具 ✨")
    root.geometry("880x700")
    root.minsize(720, 520)
    root.configure(bg=BG)

    # ── 载入 ASMR 看板娘徽章与图标 ──
    mascot_img = None
    if HAS_PIL:
        try:
            mascot_path = os.path.join(SCRIPT_DIR, "assets", "mascot_icon.png")
            if not os.path.isfile(mascot_path):
                mascot_path = os.path.join(SCRIPT_DIR, "assets", "asmr_mascot.png")
            if os.path.isfile(mascot_path):
                _im = Image.open(mascot_path).convert("RGBA")
                _im = _im.resize((56, 56), Image.Resampling.LANCZOS)
                mascot_img = ImageTk.PhotoImage(_im)
                root.mascot_img = mascot_img  # 保留引用避免 GC
                try:
                    root.wm_iconphoto(True, mascot_img)
                except Exception:
                    pass
        except Exception:
            mascot_img = None

    style = ttk.Style()
    style.theme_use("clam")

    # ── TTK 全局样式微调 ──
    style.configure("Treeview",
                    background=CARD_BG,
                    foreground=TEXT,
                    fieldbackground=CARD_BG,
                    borderwidth=0,
                    rowheight=28,
                    font=("Microsoft YaHei UI", 9))
    style.configure("Treeview.Heading",
                    background=CARD_BG_ALT,
                    foreground=ACCENT_PURPLE,
                    font=("Microsoft YaHei UI", 9, "bold"),
                    relief=tk.FLAT,
                    borderwidth=0)
    style.map("Treeview.Heading",
              background=[("active", CARD_INNER)],
              foreground=[("active", TEXT_ACCENT)])
    style.map("Treeview",
              background=[("selected", ACCENT_PURPLE)],
              foreground=[("selected", "#181926")])

    style.configure("TProgressbar",
                    thickness=14,
                    borderwidth=0,
                    troughcolor=CARD_BG_ALT,
                    background=ACCENT_PINK)
    style.map("TProgressbar",
              background=[("active", ACCENT_PINK_HOVER)])

    style.configure("TScrollbar",
                    background=CARD_BG_ALT,
                    troughcolor=BG,
                    borderwidth=0,
                    arrowsize=11,
                    relief=tk.FLAT)
    style.map("TScrollbar",
              background=[("active", ACCENT_PURPLE), ("pressed", ACCENT_PINK)])

    main = tk.Frame(root, bg=BG)
    main.pack(fill=tk.BOTH, expand=True, padx=16, pady=12)

    # ── 顶栏 Banner ──
    header = tk.Frame(main, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1)
    header.pack(fill=tk.X, pady=(0, 10))

    header_inner = tk.Frame(header, bg=CARD_BG)
    header_inner.pack(fill=tk.X, padx=12, pady=8)

    title_left = tk.Frame(header_inner, bg=CARD_BG)
    title_left.pack(side=tk.LEFT)

    if mascot_img:
        mascot_label = tk.Label(title_left, image=mascot_img, bg=CARD_BG)
        mascot_label.pack(side=tk.LEFT, padx=(0, 10))

    title_text_box = tk.Frame(title_left, bg=CARD_BG)
    title_text_box.pack(side=tk.LEFT)

    tk.Label(title_text_box, text="🌸 FA-ASMR V2 · 二次元音声对齐",
             font=("Microsoft YaHei UI", 15, "bold"),
             fg=ACCENT_PINK, bg=CARD_BG).pack(anchor="w")
    tk.Label(title_text_box, text="✨ 双声道 Better-Ear 全局声学对齐 · 梦幻输出 LRC / SRT 💫",
             font=("Microsoft YaHei UI", 9),
             fg=TEXT_MUTED, bg=CARD_BG).pack(anchor="w", pady=(2, 0))

    # 右侧设备徽标
    dev_badge_text = "✨ CUDA 加速模式" if CUDA_OK else "💤 CPU 运算模式"
    dev_badge_fg = SUCCESS if CUDA_OK else TEXT_MUTED
    tk.Label(header_inner, text=dev_badge_text, font=("Microsoft YaHei UI", 9, "bold"),
             fg=dev_badge_fg, bg=CARD_INNER, padx=12, pady=4, relief=tk.FLAT).pack(side=tk.RIGHT)

    # ── 拖放区 Drop Zone ──
    drop_frame = tk.Frame(main, bg=CARD_INNER, height=88,
                          highlightbackground=ACCENT_PURPLE, highlightthickness=1)
    drop_frame.pack(fill=tk.X, pady=(0, 10))
    drop_frame.pack_propagate(False)

    drop_inner = tk.Frame(drop_frame, bg=CARD_INNER)
    drop_inner.place(relx=0.01, rely=0.08, relwidth=0.98, relheight=0.84)

    drop_text_var = tk.StringVar(
        value="🌸 拖放台本 (.txt) 与音频文件 (.wav / .mp3 / .flac / .m4a / .ogg) 到此处 ~\n✨ 或点击此处挑选文件 · 自动按名称智能配对 🎵"
    )
    drop_label = tk.Label(drop_inner, textvariable=drop_text_var,
                          font=("Microsoft YaHei UI", 10),
                          fg=TEXT_ACCENT, bg=CARD_INNER, cursor="hand2")
    drop_label.pack(expand=True)

    def _drop_hover_enter(e):
        drop_frame.config(highlightbackground=ACCENT_PINK, bg="#362f4b")
        drop_inner.config(bg="#362f4b")
        drop_label.config(bg="#362f4b", fg="#ffffff")

    def _drop_hover_leave(e):
        drop_frame.config(highlightbackground=ACCENT_PURPLE, bg=CARD_INNER)
        drop_inner.config(bg=CARD_INNER)
        drop_label.config(bg=CARD_INNER, fg=TEXT_ACCENT)

    drop_frame.bind("<Enter>", _drop_hover_enter)
    drop_frame.bind("<Leave>", _drop_hover_leave)
    drop_inner.bind("<Enter>", _drop_hover_enter)
    drop_inner.bind("<Leave>", _drop_hover_leave)
    drop_label.bind("<Enter>", _drop_hover_enter)
    drop_label.bind("<Leave>", _drop_hover_leave)

    _existing_names = set()
    _task_store = {}

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
                iid = task_tree.insert("", "end", values=(
                    u['name'], os.path.basename(u['file']),
                    "— (未匹配)", STATUS_ICONS['warn']), tags=('warn',))
                _task_store[iid] = {
                    'script': u['file'], 'audio': None,
                    'name': u['name'], 'status': 'pending', 'iid': iid}
                _existing_names.add(u['name'])
                added += 1
        if added:
            drop_text_var.set(f"💖 已成功添加 {added} 项任务 ~ 可继续拖放追加 ✨")
            update_status(f"已载入 {len(_task_store)} 项任务")
        else:
            drop_text_var.set("🌸 拖放台本 (.txt) 与音频文件到此处 ~\n✨ 点击挑选文件 · 自动按名称智能配对 🎵")

    def _browse_files(e=None):
        files = filedialog.askopenfilenames(
            title="选择台本与音频文件",
            filetypes=[("支持的台本与音频", "*.txt;*.wav;*.mp3;*.flac;*.m4a;*.ogg;*.aac;*.opus;*.wma"), ("所有文件", "*.*")])
        if files:
            _add_files_to_list(list(files))

    drop_label.bind("<Button-1>", _browse_files)
    drop_inner.bind("<Button-1>", _browse_files)
    drop_frame.bind("<Button-1>", _browse_files)

    if HAS_DND:
        def _on_drop(event):
            raw = event.data
            paths = []
            try:
                # 优先使用 Tk 原生 splitlist 解析标准带空格和花括号的路径列表
                items = root.tk.splitlist(raw)
                for item in items:
                    p = str(item).strip().strip('"').strip("'")
                    if p:
                        paths.append(os.path.abspath(p))
            except Exception:
                # 备选正则兜底
                for m in re.finditer(r'\{([^}]*)\}', raw):
                    paths.append(os.path.abspath(m.group(1).strip()))
                cleaned = re.sub(r'\{[^}]*\}', '', raw)
                for p in cleaned.split():
                    if p.strip():
                        paths.append(os.path.abspath(p.strip()))

            _add_files_to_list(paths)

        drop_frame.drop_target_register(DND_FILES)
        drop_frame.dnd_bind('<<Drop>>', _on_drop)
        drop_inner.drop_target_register(DND_FILES)
        drop_inner.dnd_bind('<<Drop>>', _on_drop)
        drop_label.drop_target_register(DND_FILES)
        drop_label.dnd_bind('<<Drop>>', _on_drop)

    # ── 高级设置面板 (折叠面板，静音吸附默认开启无需配置) ──
    adv_header_frame = tk.Frame(main, bg=BG)
    adv_header_frame.pack(fill=tk.X, pady=(0, 4))

    adv_show = tk.BooleanVar(value=False)
    adv_toggle = tk.Label(adv_header_frame, text="🎀 高级设置 (点击展开)",
                          font=("Microsoft YaHei UI", 9), fg=TEXT_MUTED, bg=BG, cursor="hand2")
    adv_toggle.pack(side=tk.LEFT)

    adv_panel = tk.Frame(main, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1)
    adv_panel_inner = tk.Frame(adv_panel, bg=CARD_BG)
    adv_panel_inner.pack(fill=tk.X, padx=12, pady=8)

    def _toggle_adv():
        if adv_show.get():
            adv_show.set(False)
            adv_panel.pack_forget()
            adv_toggle.configure(text="🎀 高级设置 (点击展开)", fg=TEXT_MUTED)
        else:
            adv_show.set(True)
            adv_panel.pack(fill=tk.X, pady=(0, 8), before=list_frame)
            adv_toggle.configure(text="🎀 高级设置 (点击收起)", fg=ACCENT_PURPLE)

    adv_toggle.bind("<Button-1>", lambda e: _toggle_adv())

    dev_var = tk.StringVar(value="cuda" if CUDA_OK else "cpu")
    vram_chunk_var = tk.StringVar(value="6000")
    end_extend_var = tk.StringVar(value="0")
    ono_gate_var = tk.BooleanVar(value=True)

    _var_meta = [
        (dev_var, "dev", str),
        (vram_chunk_var, "vram_chunk", str),
        (end_extend_var, "end_extend", str),
        (ono_gate_var, "ono_gate", lambda v: v in (True, "true", "True", 1, "1")),
    ]

    def _save_settings():
        try:
            data = {key: var.get() for var, key, _ in _var_meta}
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
                    var.set(cvt(data[key]))
        except Exception:
            pass

    _load_settings()

    row1 = tk.Frame(adv_panel_inner, bg=CARD_BG)
    row1.pack(fill=tk.X, pady=(0, 4))
    tk.Label(row1, text="计算设备:", font=("Microsoft YaHei UI", 9), fg=TEXT_MUTED, bg=CARD_BG).pack(side=tk.LEFT, padx=(0, 4))
    cb = ttk.Combobox(row1, textvariable=dev_var, values=["cuda", "cpu"] if CUDA_OK else ["cpu"], state="readonly", width=6)
    cb.pack(side=tk.LEFT, padx=(0, 16))

    if CUDA_OK:
        tk.Label(row1, text="显存分块(帧):", font=("Microsoft YaHei UI", 9), fg=TEXT_MUTED, bg=CARD_BG).pack(side=tk.LEFT, padx=(0, 4))
        tk.Entry(row1, textvariable=vram_chunk_var, font=("Consolas", 9), width=6,
                 relief=tk.FLAT, bg=CARD_BG_ALT, fg=TEXT, insertbackground=ACCENT_PINK,
                 highlightbackground=BORDER, highlightthickness=1).pack(side=tk.LEFT, padx=(0, 16))

    row2 = tk.Frame(adv_panel_inner, bg=CARD_BG)
    row2.pack(fill=tk.X, pady=(4, 0))
    tk.Label(row2, text="行结束延长(秒):", font=("Microsoft YaHei UI", 9), fg=TEXT_MUTED, bg=CARD_BG).pack(side=tk.LEFT, padx=(0, 4))
    tk.Entry(row2, textvariable=end_extend_var, font=("Consolas", 9), width=5,
             relief=tk.FLAT, bg=CARD_BG_ALT, fg=TEXT, insertbackground=ACCENT_PINK,
             highlightbackground=BORDER, highlightthickness=1).pack(side=tk.LEFT)
    tk.Label(row2, text="0=不延长  推荐 0.5~1.5s  (≤下一句起点)", font=("Microsoft YaHei UI", 8),
             fg=TEXT_MUTED, bg=CARD_BG).pack(side=tk.LEFT, padx=(8, 0))

    row3 = tk.Frame(adv_panel_inner, bg=CARD_BG)
    row3.pack(fill=tk.X, pady=(4, 0))
    tk.Checkbutton(row3, text="拟声词删除 (删除错位的喘息/呻吟拟声，保留语义)",
                   variable=ono_gate_var, font=("Microsoft YaHei UI", 9),
                   fg=TEXT, bg=CARD_BG, selectcolor=CARD_BG_ALT,
                   activebackground=CARD_BG, activeforeground=TEXT,
                   highlightthickness=0, bd=0).pack(side=tk.LEFT)

    # ── 任务列表 (Treeview) ──
    list_frame = tk.Frame(main, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1)
    list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

    cols = ("name", "script", "audio", "status")
    task_tree = ttk.Treeview(list_frame, columns=cols, show="headings", height=8, selectmode="extended")
    task_tree.heading("name", text="🌸 任务配对")
    task_tree.heading("script", text="📄 台本文件 (Script)")
    task_tree.heading("audio", text="🎵 音频文件 (Audio)")
    task_tree.heading("status", text="📌 状态 (Status)")

    task_tree.column("name", width=150, minwidth=100, stretch=True)
    task_tree.column("script", width=220, minwidth=140, stretch=True)
    task_tree.column("audio", width=220, minwidth=140, stretch=True)
    task_tree.column("status", width=110, minwidth=80, stretch=False, anchor="center")

    task_tree.tag_configure('pending', foreground=TEXT_MUTED)
    task_tree.tag_configure('running', foreground=ACCENT_CYAN)
    task_tree.tag_configure('done',    foreground=SUCCESS)
    task_tree.tag_configure('failed',  foreground=ERROR)
    task_tree.tag_configure('warn',    foreground=WARN)

    sb = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=task_tree.yview)
    task_tree.configure(yscrollcommand=sb.set)
    task_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    sb.pack(side=tk.RIGHT, fill=tk.Y)

    if HAS_DND:
        task_tree.drop_target_register(DND_FILES)
        task_tree.dnd_bind('<<Drop>>', _on_drop)

    # ── 控制与进度工具栏 ──
    btn_bar = tk.Frame(main, bg=BG)
    btn_bar.pack(fill=tk.X, pady=(0, 8))

    def _clear_tasks():
        task_tree.delete(*task_tree.get_children())
        _task_store.clear()
        _existing_names.clear()
        drop_text_var.set("🌸 拖放台本 (.txt) 与音频文件到此处 ~\n✨ 点击挑选文件 · 自动按名称智能配对 🎵")
        progress.configure(value=0)
        update_status("🌸 就绪 ~ 请添加任务")

    def _delete_selected():
        for iid in task_tree.selection():
            if iid in _task_store:
                name = _task_store[iid]['name']
                _existing_names.discard(name)
                del _task_store[iid]
                task_tree.delete(iid)
        update_status(f"当前剩余 {len(_task_store)} 项任务")

    clear_btn = tk.Button(btn_bar, text="🗑️ 清空列表", font=("Microsoft YaHei UI", 9),
                          fg=TEXT, bg=CARD_BG, activebackground=CARD_INNER, activeforeground=TEXT_ACCENT,
                          relief=tk.FLAT, padx=12, pady=4, cursor="hand2",
                          highlightbackground=BORDER, highlightthickness=1,
                          command=_clear_tasks)
    clear_btn.pack(side=tk.LEFT, padx=(0, 6))

    del_btn = tk.Button(btn_bar, text="❌ 删除选中", font=("Microsoft YaHei UI", 9),
                        fg=TEXT, bg=CARD_BG, activebackground=CARD_INNER, activeforeground=TEXT_ACCENT,
                        relief=tk.FLAT, padx=12, pady=4, cursor="hand2",
                        highlightbackground=BORDER, highlightthickness=1,
                        command=_delete_selected)
    del_btn.pack(side=tk.LEFT)

    # 进度条与状态文案
    start_btn = tk.Button(btn_bar, text="🌸  开始全部对齐  ✨",
                          font=("Microsoft YaHei UI", 10, "bold"),
                          fg="white", bg=ACCENT_PINK, activebackground=ACCENT_PINK_HOVER,
                          activeforeground="white", relief=tk.FLAT,
                          padx=20, pady=4, cursor="hand2")
    start_btn.pack(side=tk.RIGHT, padx=(10, 0))

    progress = ttk.Progressbar(btn_bar, mode="determinate", length=180)
    progress.pack(side=tk.RIGHT, padx=(10, 0))

    status_var = tk.StringVar(value="🌸 准备就绪 ~ 拖放文件开始")
    status_label = tk.Label(btn_bar, textvariable=status_var,
                            font=("Microsoft YaHei UI", 9), fg=TEXT_MUTED, bg=BG)
    status_label.pack(side=tk.RIGHT)

    def update_status(msg):
        root.after(0, lambda: status_var.set(msg))

    # ── 日志控制台 ──
    log_header = tk.Frame(main, bg=BG)
    log_header.pack(fill=tk.X, pady=(0, 2))
    tk.Label(log_header, text="📜 运行日志 (Log Console):",
             font=("Microsoft YaHei UI", 9, "bold"),
             fg=TEXT, bg=BG, anchor="w").pack(side=tk.LEFT)

    log_frame = tk.Frame(main, bg="#141521", highlightbackground=BORDER, highlightthickness=1)
    log_frame.pack(fill=tk.BOTH, expand=True)

    log_text = tk.Text(log_frame, bg="#141521", fg=TEXT,
                       font=("Consolas", 9), relief=tk.FLAT,
                       wrap=tk.WORD, height=6, padx=8, pady=6,
                       insertbackground=ACCENT_PINK)
    log_sb = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=log_text.yview)
    log_text.configure(yscrollcommand=log_sb.set)
    log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    log_sb.pack(side=tk.RIGHT, fill=tk.Y)

    def append_log(msg):
        def _do():
            log_text.insert(tk.END, msg + "\n")
            log_text.see(tk.END)
        root.after(0, _do)

    # ── 线程池执行 ──
    _running = False

    def _run_all():
        nonlocal _running
        if _running:
            return
        _save_settings()
        items = list(_task_store.values())
        valid_items = [it for it in items if it.get("audio") and it.get("script") and it.get("status") != "done"]
        if not valid_items:
            messagebox.showinfo("提示", "🌸 没有待处理的有效配对任务哦！请先拖放台本与音频文件。")
            return

        _running = True
        start_btn.config(state=tk.DISABLED, text="⚡ 正在对齐中...", bg=CARD_INNER)
        clear_btn.config(state=tk.DISABLED)
        del_btn.config(state=tk.DISABLED)
        progress.configure(value=0)

        total_cnt = len(valid_items)
        append_log("\n" + "═" * 58)
        append_log(f"🌸 开始批量处理 {total_cnt} 个任务 (默认开启静音吸附优化)")
        append_log(f"⚡ 计算设备: {'CUDA (GPU 加速)' if dev_var.get() == 'cuda' else 'CPU 模式'}")
        append_log("═" * 58)

        def _worker():
            nonlocal _running
            succ_cnt = 0
            fail_cnt = 0
            for idx, it in enumerate(valid_items):
                iid = it["iid"]
                root.after(0, lambda id=iid: (task_tree.set(id, "status", STATUS_ICONS["running"]),
                                              task_tree.item(id, tags=('running',))))
                it["status"] = "running"
                append_log(f"\n[{idx + 1}/{total_cnt}] 🌸 正在对齐: {it['name']}")
                append_log(f"  📄 台本: {it['script']}")
                append_log(f"  🎵 音频: {it['audio']}")

                def _pct(pct, m):
                    gp = (idx * 100 + pct) / total_cnt
                    root.after(0, lambda v=gp, msg=m: (progress.configure(value=v), update_status(msg)))

                ext_sec = float(end_extend_var.get() or 0.0)
                ok, out_path, err = run_pipeline(
                    script_path=it["script"],
                    audio_path=it["audio"],
                    lora_path=str(getattr(align, "DEFAULT_LORA_PATH", "")),
                    log_callback=append_log,
                    progress_callback=_pct,
                    dev=dev_var.get(),
                    vram_chunk=int(vram_chunk_var.get() or 6000),
                    persist_gpu=True,
                    line_end_extend=ext_sec,
                    enable_silence_snap=True,
                    enable_resegment=True,
                    enable_ono_gate=bool(ono_gate_var.get()),
                )
                if ok:
                    root.after(0, lambda id=iid: (task_tree.set(id, "status", STATUS_ICONS["done"]),
                                                  task_tree.item(id, tags=('done',))))
                    it["status"] = "done"
                    succ_cnt += 1
                else:
                    root.after(0, lambda id=iid: (task_tree.set(id, "status", STATUS_ICONS["failed"]),
                                                  task_tree.item(id, tags=('failed',))))
                    it["status"] = "failed"
                    fail_cnt += 1
                    append_log(f"  💔 失败原因: {err}")

            _release_gpu()
            progress.configure(value=100)
            _running = False
            append_log("\n" + "═" * 58)
            append_log(f"💖 全部任务处理完成！成功 {succ_cnt} 个，失败 {fail_cnt} 个 ✨")
            append_log("═" * 58 + "\n")
            update_status(f"💖 处理完成: 成功 {succ_cnt}, 失败 {fail_cnt}")

            root.after(0, lambda: (
                start_btn.config(state=tk.NORMAL, text="🌸  开始全部对齐  ✨", bg=ACCENT_PINK),
                clear_btn.config(state=tk.NORMAL),
                del_btn.config(state=tk.NORMAL),
            ))

        threading.Thread(target=_worker, daemon=True).start()

    start_btn.config(command=_run_all)

    # ── 居中显示与关闭持久化 ──
    root.update_idletasks()
    w, h = root.winfo_width(), root.winfo_height()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"+{(sw-w)//2}+{(sh-h)//2}")

    def _on_close():
        _save_settings()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)
    return root


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
        import numpy, torch, torchaudio, numba
        import transformers, tokenizers, huggingface_hub, safetensors
        import soundfile, fugashi, unidic_lite, pykakasi, janome, jaconv
        import speach, chirptext, puchikarui, rich
        from speach.ttlig import RubyFrag, RubyToken
        from yomikata.dbert import dBert
        return (f"torch={torch.__version__} transformers={transformers.__version__} "
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
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        return f"align.py 可导入，设备={dev}"

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


