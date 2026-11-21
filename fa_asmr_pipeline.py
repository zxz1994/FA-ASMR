# -*- coding: utf-8 -*-
"""FA-ASMR V2 — 无界面核心对齐流水线与通用文本/字幕工具。

从 FA-ASMR_GUI.py 抽离的非 GUI 业务逻辑：
- 台本多编码嗅探读取、说话人标记拆分、AutoRuby 注音单例管理
- LRC / SRT 标准字幕组装、原文双序列锚点切片还原 (_attach_source_spans)
- 单人 / 多人乱序 / 多视角联合对齐核心管线 (run_pipeline)
- 音频与台本智能配对 (auto_pair_files)

设计原则：
- 不导入 tkinter，不注册 GUI 弹窗异常钩子，可供 GUI、CLI 与单元测试无副作用复用；
- 仅在实际执行 run_pipeline() 时延迟加载 torch / align / haruraw2norm，保持轻量导入。
"""
from __future__ import annotations

import copy
import difflib
import json
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

if getattr(sys, "frozen", False):
    SCRIPT_DIR = sys._MEIPASS
    EXE_DIR = os.path.dirname(sys.executable)
else:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    EXE_DIR = SCRIPT_DIR

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import align_utils as au
from clause import F, is_text_token as _is_text_token

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".opus", ".wma"}


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
            _td = os.path.join(SCRIPT_DIR, "furigana")
            if _td not in sys.path:
                sys.path.insert(0, _td)
            from fa_asmr_converter import RubyAnnotator
            _RUBY_ANNOTATOR = RubyAnnotator(verbose=False, strip_inline_readings=False)
        # 批量任务复用同一实例：fugashi(UniDic) / pykakasi / yomikata(dBert)
        # 只初始化一次，避免每条音频都重载 UniDic 词典与异读模型。
        self._ann = _RUBY_ANNOTATOR

    def annotate(self, text: str) -> str:
        return self._ann.annotate(text)

    def release_gpu(self):
        global _RUBY_ANNOTATOR
        try:
            self._ann.release_gpu()
        except Exception as e:
            logger.warning("释放 AutoRuby GPU 资源失败: %s: %s", type(e).__name__, e)
        _RUBY_ANNOTATOR = None


def read_text_auto_encoding(file_path: str) -> tuple[str, str]:
    """智能多编码降级嗅探读取台本文本。

    支持格式：
      - UTF-8（标准及带 BOM utf-8-sig）
      - Shift-JIS / CP932（DLsite 与日本声优事务所原生台本常用编码）
      - GB18030 / GBK / GB2312（中文 Windows 常见编码）
      - UTF-16 LE / BE
    返回 (解码文本, 识别编码名称)。
    """
    with open(file_path, "rb") as f:
        raw = f.read()

    # 1. 明确的 BOM 探测
    if raw.startswith(b"\xef\xbb\xbf"):
        try:
            text = raw[3:].decode("utf-8")
            return (text[1:] if text.startswith("\ufeff") else text), "utf-8-sig"
        except UnicodeDecodeError:
            text = raw[3:].decode("utf-8", errors="replace")
            return (text[1:] if text.startswith("\ufeff") else text), "utf-8-sig-fallback"
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        try:
            text = raw.decode("utf-16")
            return (text[1:] if text.startswith("\ufeff") else text), "utf-16"
        except UnicodeDecodeError:
            pass

    # 2. 严格 UTF-8 校验（覆盖现代大多数台本）
    try:
        text = raw.decode("utf-8")
        return (text[1:] if text.startswith("\ufeff") else text), "utf-8"
    except UnicodeDecodeError:
        pass

    # 3. 日文本地编码 CP932 / Shift-JIS（覆盖绝大多数 DLsite 原生日文台本）
    try:
        return raw.decode("cp932"), "cp932"
    except UnicodeDecodeError:
        pass

    # 4. 中文编码 GB18030（覆盖 GBK / GB2312）
    try:
        return raw.decode("gb18030"), "gb18030"
    except UnicodeDecodeError:
        pass

    # 5. UTF-16 无 BOM 尝试
    try:
        text = raw.decode("utf-16")
        return (text[1:] if text.startswith("\ufeff") else text), "utf-16"
    except UnicodeDecodeError:
        pass

    # 6. 最终兜底：UTF-8 容错替换（防止抛出未捕获异常导致对齐中断）
    return raw.decode("utf-8", errors="replace"), "utf-8-fallback"


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


def _lrc_time_to_cs(t):
    if not t or t == "[error]":
        return None
    m = au.LRC_TS_ANCHORED.match(t)
    if not m:
        m = re.match(r"\[(\d+):(\d+)[.:](\d+)\]", str(t))
    if not m:
        return None
    return (int(m.group(1)) * 60 + int(m.group(2))) * 100 + int(m.group(3))


def build_standard_lrc(result_list):
    groups = []
    min_start_cs = None
    min_start_str = None
    max_end_cs = None
    max_end_str = None
    # 字幕行按【说话人段】收集：同一行内说话人变化处补该说话人的「②：」前缀。
    # 结构 = [[说话人元组或None, 文本], ...]。无标记作品恒为单段 None，输出不变。
    group_parts = []

    def _flush():
        if group_parts and any(t for _, t in group_parts) and min_start_str:
            best_end = max_end_str if max_end_str else min_start_str
            # 若 max_end 比 min_start 还要早（防御极端声学错位），保证 end >= start
            if max_end_cs is not None and min_start_cs is not None and max_end_cs < min_start_cs:
                best_end = min_start_str
            groups.append((min_start_str, best_end, [list(p) for p in group_parts]))

    for item in result_list:
        io = item.get("orig", "")
        # 输出层还原：优先用台本原文子串（_src）。orig 是归一化后的字符
        # （NFKC 把 ① 拍平成 1 等），_src 保留原字符，仅换显示、不动对齐标签。
        io_out = item.get(F.SRC, None)
        if io_out is None:
            io_out = io
        if item.get("type") == 0 and io == "\n":
            _flush()
            min_start_cs = None
            min_start_str = None
            max_end_cs = None
            max_end_str = None
            group_parts = []
            continue
        # 被拟声守门删除的 token：不贡献时间边界，也不进入字幕文本
        if item.get(F.ONO_DROPPED):
            continue
        st = item.get("start")
        if st:
            cs_s = _lrc_time_to_cs(st)
            if cs_s is not None:
                if min_start_cs is None or cs_s < min_start_cs:
                    min_start_cs = cs_s
                    min_start_str = st
        en = item.get("end")
        if en:
            cs_e = _lrc_time_to_cs(en)
            if cs_e is not None:
                if max_end_cs is None or cs_e > max_end_cs:
                    max_end_cs = cs_e
                    max_end_str = en
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

    lines = []
    active_max_end_cs = None
    for idx, (g_start, g_end, g_parts) in enumerate(groups):
        eff_end = g_end
        cs_start = _lrc_time_to_cs(g_start)
        cs_end = _lrc_time_to_cs(eff_end)
        cs_next = _lrc_time_to_cs(groups[idx + 1][0]) if idx + 1 < len(groups) else None

        # 与 SRT 对齐最低 1.0s 驻留时长（仅当后续存在真实空白时补足，避免 0.12s 短句闪现即消失）
        if cs_start is not None and (cs_end is None or cs_end < cs_start + 100):
            target_cs = cs_start + 100
            if cs_next is not None:
                target_cs = min(target_cs, cs_next)
            if cs_end is None or target_cs > cs_end:
                cs_end = max(cs_start, target_cs)
                eff_end = f"[{cs_end // 6000:02d}:{(cs_end // 100) % 60:02d}:{cs_end % 100:02d}]"

        # 追踪跨行全局活跃最大结束时间，防止前一句仍在持续时因较短的嵌套插话先结束而提前输出空白清屏行
        if cs_end is not None:
            if active_max_end_cs is None or cs_end > active_max_end_cs:
                active_max_end_cs = cs_end
            clear_cs = max(cs_end, active_max_end_cs)
            if cs_next is not None and clear_cs > cs_next:
                clear_cs = cs_next
            cs_end = clear_cs
            eff_end = f"[{cs_end // 6000:02d}:{(cs_end // 100) % 60:02d}:{cs_end % 100:02d}]"

        lines.append(_fmt_line(g_start, eff_end, g_parts))
        if eff_end and eff_end != g_start:
            # 仅当所有已激活语句全部结束且下一行之前存在真实停顿（或为末行）时，输出空白下轴时间戳
            has_gap = cs_next is None or (cs_end is not None and cs_end < cs_next)
            if has_gap:
                t = _fmt_lrc_time(eff_end)
                if t:
                    lines.append(t)
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


def build_standard_srt(result_list, line_end_extend=0.0):
    """组装标准 SRT 字幕文本（单角色与多角色独立时间流通用）。"""
    from align_multispeaker import build_standard_srt as _ams_build_srt
    return _ams_build_srt(result_list, line_end_extend=line_end_extend)


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


_INLINE_RUBY_RE = re.compile(r"\{([^|{}]+)\|([^|{}]*)\}")
_RUBY_READING_VALID_RE = re.compile(r"[ぁ-んァ-ヶー一-龯々A-Za-z0-9]")


def _validate_ruby_pair(surface: Any, reading: Any, where: str) -> tuple[str, str]:
    if not isinstance(surface, str) or not surface.strip():
        raise ValueError(f"ruby-map ({where}) 的词条键必须为非空字符串: {surface!r}")
    if not isinstance(reading, str) or not reading.strip():
        raise ValueError(f"ruby-map ({where}) 词条 {surface!r} 的读音必须为非空字符串: {reading!r}")
    for bad in ("{", "}", "|", "\n", "\r"):
        if bad in surface:
            raise ValueError(f"ruby-map ({where}) 词条键含非法字符 {bad!r}: {surface!r}")
        if bad in reading:
            raise ValueError(f"ruby-map ({where}) 词条 {surface!r} 的读音含非法字符 {bad!r}: {reading!r}")
    rd = reading.strip()
    if not _RUBY_READING_VALID_RE.search(rd):
        raise ValueError(f"ruby-map ({where}) 词条 {surface!r} 的读音缺少有效假名/发音字符: {reading!r}")
    return surface, rd


def _parse_by_line_dict(raw_bl: Any, where: str) -> Dict[int, Dict[str, str]]:
    if not isinstance(raw_bl, dict):
        raise ValueError(f"ruby-map 的 '{where}' 字段必须为字典对象")
    by_line: Dict[int, Dict[str, str]] = {}
    for ln_k, sub in raw_bl.items():
        try:
            ln_no = int(str(ln_k).strip())
        except ValueError as e:
            raise ValueError(f"ruby-map '{where}' 行号必须为正整数: {ln_k!r}") from e
        if ln_no <= 0:
            raise ValueError(f"ruby-map '{where}' 行号必须 >= 1: {ln_k!r}")
        if not isinstance(sub, dict):
            raise ValueError(f"ruby-map '{where}[{ln_no}]' 必须为字典对象")
        ln_dict: Dict[str, str] = {}
        for k, v in sub.items():
            s, r = _validate_ruby_pair(k, v, f"{where}.{ln_no}")
            ln_dict[s] = r
        if ln_dict:
            by_line[ln_no] = ln_dict
    return by_line


def load_ruby_map(source: Any) -> Dict[str, Any]:
    """加载并严格校验注音覆盖表（支持 JSON 文件路径或 dict）。

    支持结构：
      1) 扁平全局表: ``{"陽菜羽": "ひなは", "コソ練": "こそれん"}``
      2) 分层表:     ``{"global": {"私": "あたし"}, "by_line": {"12": {"本気": "まじ"}}, "by_file": {"04": {"by_line": {"197": {"晶": "あきら"}}}}}``
    返回归一化结构 ``{"global": Dict[str, str], "by_line": Dict[int, Dict[str, str]], "by_file": Dict[str, Dict[str, Any]]}``。
    """
    if source is None:
        return {"global": {}, "by_line": {}, "by_file": {}}
    if isinstance(source, (str, os.PathLike)):
        path = os.path.abspath(str(source))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"指定的注音覆盖表不存在: {path}")
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                raw = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"注音覆盖表 JSON 解析失败 ({path}): {e}") from e
    elif isinstance(source, dict):
        raw = source
    else:
        raise ValueError(f"ruby-map 必须为 JSON 文件路径或字典对象，实际为: {type(source).__name__}")

    if not isinstance(raw, dict):
        raise ValueError(f"ruby-map 顶层必须为 JSON 对象 (dict)，实际为: {type(raw).__name__}")

    g_map: Dict[str, str] = {}
    by_line: Dict[int, Dict[str, str]] = {}
    by_file: Dict[str, Dict[str, Any]] = {}

    _STRUCT_KEYS = ("global", "by_line", "by_file", "files", "comment", "_comment")
    is_structured = (
        any(k in raw for k in ("global", "by_line", "by_file", "files"))
        and all(k in _STRUCT_KEYS for k in raw.keys())
        and (
            ("global" in raw and isinstance(raw["global"], dict))
            or ("by_line" in raw and isinstance(raw["by_line"], dict))
            or ("by_file" in raw and isinstance(raw["by_file"], dict))
            or ("files" in raw and isinstance(raw["files"], dict))
        )
    )
    if is_structured:
        raw_g = raw.get("global")
        if raw_g is not None:
            if not isinstance(raw_g, dict):
                raise ValueError("ruby-map 的 'global' 字段必须为字典对象")
            for k, v in raw_g.items():
                s, r = _validate_ruby_pair(k, v, "global")
                g_map[s] = r
        raw_bl = raw.get("by_line")
        if raw_bl is not None:
            by_line = _parse_by_line_dict(raw_bl, "by_line")
        for bf_key in ("by_file", "files"):
            raw_bf = raw.get(bf_key)
            if raw_bf is None:
                continue
            if not isinstance(raw_bf, dict):
                raise ValueError(f"ruby-map 的 '{bf_key}' 字段必须为字典对象")
            for fk, fv in raw_bf.items():
                fk_str = str(fk).strip()
                if not fk_str:
                    raise ValueError(f"ruby-map '{bf_key}' 的文件键不能为空")
                if not isinstance(fv, dict):
                    raise ValueError(f"ruby-map '{bf_key}[{fk_str}]' 必须为字典对象")
                f_g: Dict[str, str] = {}
                f_bl: Dict[int, Dict[str, str]] = {}
                is_sub_struct = (
                    ("global" in fv or "by_line" in fv)
                    and all(k in ("global", "by_line", "comment", "_comment") for k in fv.keys())
                )
                if is_sub_struct:
                    if fv.get("global") is not None:
                        if not isinstance(fv["global"], dict):
                            raise ValueError(f"ruby-map '{bf_key}[{fk_str}].global' 必须为字典对象")
                        for k, v in fv["global"].items():
                            s, r = _validate_ruby_pair(k, v, f"{bf_key}.{fk_str}.global")
                            f_g[s] = r
                    if fv.get("by_line") is not None:
                        f_bl = _parse_by_line_dict(fv["by_line"], f"{bf_key}.{fk_str}.by_line")
                else:
                    # 简写形式：如果所有键都是正整数行号且值为 dict，视为 by_line；否则视为该文件的 global
                    if fv and all(str(k).strip().isdigit() and isinstance(v, dict) for k, v in fv.items()):
                        f_bl = _parse_by_line_dict(fv, f"{bf_key}.{fk_str}")
                    else:
                        for k, v in fv.items():
                            s, r = _validate_ruby_pair(k, v, f"{bf_key}.{fk_str}")
                            f_g[s] = r
                entry = by_file.setdefault(fk_str, {"global": {}, "by_line": {}})
                entry["global"].update(f_g)
                for ln_no, ln_d in f_bl.items():
                    entry["by_line"].setdefault(ln_no, {}).update(ln_d)
    else:
        for k, v in raw.items():
            s, r = _validate_ruby_pair(k, v, "root")
            g_map[s] = r

    return {"global": g_map, "by_line": by_line, "by_file": by_file}


def resolve_ruby_map_for_script(ruby_map: Any, script_path: Optional[str] = None) -> Dict[str, Any]:
    """根据当前台本路径解析合并后的生效 {global, by_line}（支持 by_file 文件名/基名/序号前缀匹配）。"""
    norm = load_ruby_map(ruby_map) if ruby_map is not None else {"global": {}, "by_line": {}, "by_file": {}}
    g_map = dict(norm.get("global", {}))
    bl_map: Dict[int, Dict[str, str]] = {ln: dict(d) for ln, d in norm.get("by_line", {}).items()}
    by_file = norm.get("by_file", {})
    if not by_file or not script_path:
        return {"global": g_map, "by_line": bl_map, "by_file": by_file}

    bname = os.path.basename(script_path)
    stem = os.path.splitext(bname)[0]
    bname_l = bname.lower()
    stem_l = stem.lower()
    m_num = re.match(r"^(\d{1,3})(?=[_\-\s.「【]|$)", stem)
    track_prefix = m_num.group(1) if m_num else None

    for fk, f_entry in by_file.items():
        fk_norm = os.path.basename(fk.replace("\\", "/")).strip()
        fk_l = fk_norm.lower()
        fk_stem_l = os.path.splitext(fk_l)[0]
        matched = (
            fk_l == bname_l
            or fk_l == stem_l
            or fk_stem_l == stem_l
            or (track_prefix is not None and fk_l == track_prefix.lower())
            or (len(fk_l) >= 2 and stem_l.startswith(fk_l) and (len(stem_l) == len(fk_l) or stem_l[len(fk_l)] in "_- 　.「【"))
        )
        if matched:
            g_map.update(f_entry.get("global", {}))
            for ln_no, ln_d in f_entry.get("by_line", {}).items():
                bl_map.setdefault(ln_no, {}).update(ln_d)

    return {"global": g_map, "by_line": bl_map, "by_file": by_file}


def apply_ruby_overrides_to_line(
    line: str,
    overrides: Dict[str, str],
    replace_existing: bool = False,
) -> tuple[str, int]:
    """将 {surface: reading} 精准应用到单行文本，绝不嵌套或破坏已有 {surface|reading} 结构。

    - ``replace_existing=False``（注音前预注入）：对裸文本按最长词优先包为 ``{surface|reading}``，
      已有 ``{s|r}`` 中若 ``s`` 命中 overrides 则直接更新其读音。
    - ``replace_existing=True``（注音后权威覆盖）：对已注音行中所有 ``{s|r}`` 若 ``s`` 命中 overrides
      强制改写为 ``{s|reading}``（防止 yomikata 异读层回写覆盖人工/LLM 修正），并对尚未包裹的纯文本片段补包。
    """
    if not line or not overrides:
        return line, 0

    keys_sorted = sorted(overrides.keys(), key=len, reverse=True)
    plain_pat = re.compile("|".join(re.escape(k) for k in keys_sorted))

    out_parts: List[str] = []
    hits = 0
    pos = 0
    for m in _INLINE_RUBY_RE.finditer(line):
        seg = line[pos:m.start()]
        if seg:
            def _rep_plain(pm):
                nonlocal hits
                surf = pm.group(0)
                hits += 1
                return f"{{{surf}|{overrides[surf]}}}"
            out_parts.append(plain_pat.sub(_rep_plain, seg))
        surf_ex, read_ex = m.group(1), m.group(2)
        if surf_ex in overrides:
            new_rd = overrides[surf_ex]
            if replace_existing or read_ex != new_rd:
                hits += 1
                out_parts.append(f"{{{surf_ex}|{new_rd}}}")
            else:
                out_parts.append(m.group(0))
        else:
            out_parts.append(m.group(0))
        pos = m.end()

    tail = line[pos:]
    if tail:
        def _rep_tail(pm):
            nonlocal hits
            surf = pm.group(0)
            hits += 1
            return f"{{{surf}|{overrides[surf]}}}"
        out_parts.append(plain_pat.sub(_rep_tail, tail))

    return "".join(out_parts), hits


def annotate_lines_with_overrides(
    ruby: Any,
    body_lines: Sequence[str],
    src_line_nos: Optional[Sequence[int]] = None,
    ruby_map: Optional[Dict[str, Any]] = None,
    script_path: Optional[str] = None,
) -> tuple[List[str], int]:
    """统一执行「预注入复合词 → AutoRuby 注音 → 权威后置覆写」，保证 --verify-script 与 run_pipeline 行为一致。"""
    norm_map = resolve_ruby_map_for_script(ruby_map, script_path=script_path) if ruby_map else {"global": {}, "by_line": {}}
    g_map: Dict[str, str] = norm_map["global"]
    bl_map: Dict[int, Dict[str, str]] = norm_map["by_line"]
    has_any = bool(g_map or bl_map)

    annotated: List[str] = []
    applied_total = 0
    for idx, b in enumerate(body_lines):
        ln_no = src_line_nos[idx] if (src_line_nos is not None and idx < len(src_line_nos)) else (idx + 1)
        if not has_any or not b.strip():
            annotated.append(ruby.annotate(b))
            continue
        eff = dict(g_map)
        if ln_no in bl_map:
            eff.update(bl_map[ln_no])
        if not eff:
            annotated.append(ruby.annotate(b))
            continue
        # 预注入阶段仅注入 >=2 字的复合词/短语，单字词条留给 AutoRuby 分词后再由后置阶段对独立 {单字|...} 覆写，
        # 彻底防止单字覆写（如 {"中": "なか"}）在分词前把裸文本中的复合词（如「途中」）从中间劈成「途{中|なか}」。
        pre_eff = {k: v for k, v in eff.items() if len(k) >= 2}
        if pre_eff:
            pre_b, pre_hits = apply_ruby_overrides_to_line(b, pre_eff, replace_existing=False)
        else:
            pre_b, pre_hits = b, 0
        ann_b = ruby.annotate(pre_b)
        post_b, post_hits = apply_ruby_overrides_to_line(ann_b, eff, replace_existing=True)
        annotated.append(post_b)
        applied_total += max(pre_hits, post_hits)
    return annotated, applied_total


_COLLOC_STOP_RE = re.compile(r"[、。！？!?,.\s「」『』（）()【】〖〗♡♥♪・…‥ー～〜~\-—―]")
_AFFIX_PREFIX_CHARS = frozenset("おご御")
_AFFIX_SUFFIX_CHARS = frozenset("様さんちゃんくん達共目中内外的性化者員家屋所場側方用式風系")
_PARTICLE_STOP_CHARS = frozenset("はがをにでとのもへやかねよぞぜわさ")


def _is_kanji_char(ch: str) -> bool:
    return bool(ch) and ("\u4e00" <= ch <= "\u9fff" or ch == "々")


def _is_kana_char(ch: str) -> bool:
    return bool(ch) and ("\u3040" <= ch <= "\u309f" or "\u30a0" <= ch <= "\u30ff")


def _extract_local_collocation(plain_line: str, p_st: int, p_en: int) -> tuple[str, bool]:
    """提取目标词在原句中的紧邻局部搭配（如 お+母+様 -> お母様），并判断是否存在单字切碎/接辞风险。"""
    surf = plain_line[p_st:p_en]
    left_ch = plain_line[p_st - 1] if p_st > 0 else ""
    right_ch = plain_line[p_en] if p_en < len(plain_line) else ""
    left_ok = bool(left_ch) and not _COLLOC_STOP_RE.match(left_ch)
    right_ok = bool(right_ch) and not _COLLOC_STOP_RE.match(right_ch)

    affix_risk = False
    if len(surf) == 1:
        if left_ok and (left_ch in _AFFIX_PREFIX_CHARS or _is_kanji_char(left_ch) or ("\u30a0" <= left_ch <= "\u30ff")):
            affix_risk = True
        if right_ok and (right_ch in _AFFIX_SUFFIX_CHARS or _is_kanji_char(right_ch) or ("\u30a0" <= right_ch <= "\u30ff")):
            affix_risk = True

        # 单汉字：
        #   - 左侧优先保留敬语前缀(お/ご/御)、相邻汉字或片假名；若右侧已构成汉字/接辞搭配（如 母親），
        #     则不把左侧无关助词/形容动词尾（如 な/た）混入搭配键，保证「な母親 / た母親」归并成同一「母親」。
        left_strong = left_ok and (
            left_ch in _AFFIX_PREFIX_CHARS or _is_kanji_char(left_ch) or ("\u30a0" <= left_ch <= "\u30ff")
        )
        right_strong = right_ok and (
            right_ch in _AFFIX_SUFFIX_CHARS or _is_kanji_char(right_ch) or ("\u30a0" <= right_ch <= "\u30ff")
        )
        if right_ok and (right_ch not in _PARTICLE_STOP_CHARS or right_strong):
            r_part = right_ch
        else:
            r_part = ""
        if left_strong:
            l_part = left_ch
        elif left_ok and not right_strong and left_ch not in ("な", "た", "て", "で", "を", "が", "は", "も", "と"):
            l_part = left_ch
        else:
            l_part = ""
        colloc = f"{l_part}{surf}{r_part}"
        return colloc, affix_risk

    # 多字词：若前后紧贴敬语接辞（如 お+兄+様）或汉字则补上 1 字，否则取词形本身
    l_part = left_ch if (left_ok and left_ch in _AFFIX_PREFIX_CHARS) else ""
    r_part = right_ch if (right_ok and right_ch in _AFFIX_SUFFIX_CHARS) else ""
    colloc = f"{l_part}{surf}{r_part}"
    return colloc, bool(l_part or r_part)


def _classify_ruby_priority(
    surf: str,
    is_polyphone: bool,
    has_affix_risk: bool,
    distinct_collocs: int,
    speaker_names: Sequence[str] = (),
) -> tuple[str, List[str]]:
    """按误读风险将注音词条分为 high（高危必查）与 normal（常规单读音词）。"""
    reasons: List[str] = []
    if is_polyphone:
        reasons.append("polyphone")
    if has_affix_risk:
        reasons.append("affix_split")
    if any(ch in surf for ch in "ー～〜~") or (
        any("\u30a0" <= ch <= "\u30ff" for ch in surf) and any(_is_kanji_char(ch) for ch in surf)
    ):
        reasons.append("mixed_script")
    if speaker_names and any(surf in spk or spk in surf for spk in speaker_names if spk):
        reasons.append("speaker_name")
    return ("high" if reasons else "normal"), reasons


def extract_ruby_glossary(
    annotated_lines: Sequence[str],
    src_line_nos: Optional[Sequence[int]] = None,
    max_examples: int = 6,
    speaker_names: Optional[Sequence[str]] = None,
    file_label: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """从注音后的行列表提取去重词表（供 LLM 审校，不输出整篇原文）。

    - 按「(读音, 局部搭配)」去重采样上下文（例如 `母->はは` 同时出现在 `母親` 与 `お母様` 时各保留一条，
      消除此前仅取前 2 条导致后文错词被隐藏的盲区）；
    - 自动分层标记 ``priority``（``high`` / ``normal``）：多音词排最前，其次高危单字/接辞切碎/角色名，
      最后常规多字单读音熟语；
    - ``lines_by_reading`` 完整记录每种读音对应的全部 1-based 原台本行号，供直接构造 ``by_line``。
    """
    spk_list = [str(s).strip() for s in (speaker_names or ()) if str(s).strip()]
    entries: Dict[str, Dict[str, Any]] = {}
    for idx, ann in enumerate(annotated_lines):
        if not ann or "{" not in ann:
            continue
        ln_no = src_line_nos[idx] if (src_line_nos is not None and idx < len(src_line_nos)) else (idx + 1)
        plain_parts: List[str] = []
        match_spans: List[tuple[int, int, str, str]] = []
        pos = 0
        plain_len = 0
        for m in _INLINE_RUBY_RE.finditer(ann):
            seg = ann[pos:m.start()]
            if seg:
                plain_parts.append(seg)
                plain_len += len(seg)
            surf, rd = m.group(1), m.group(2)
            match_spans.append((plain_len, plain_len + len(surf), surf, rd))
            plain_parts.append(surf)
            plain_len += len(surf)
            pos = m.end()
        if pos < len(ann):
            plain_parts.append(ann[pos:])
        plain_line = "".join(plain_parts).strip()

        for p_st, p_en, surf, rd in match_spans:
            colloc, affix_risk = _extract_local_collocation(plain_line, p_st, p_en)
            rec = entries.get(surf)
            if rec is None:
                rec = {
                    "surface": surf,
                    "readings": {},
                    "lines_by_reading": {},
                    "total_count": 0,
                    "examples": [],
                    "collocations": [],
                    "_colloc_set": set(),
                    "_seen_rd_colloc": set(),
                    "_ex_count_by_rd": {},
                    "_affix_risk": False,
                }
                entries[surf] = rec
            rec["readings"][rd] = rec["readings"].get(rd, 0) + 1
            ln_list = rec["lines_by_reading"].setdefault(rd, [])
            if not ln_list or ln_list[-1] != ln_no:
                ln_list.append(ln_no)
            rec["total_count"] += 1
            if affix_risk:
                rec["_affix_risk"] = True
            if colloc not in rec["_colloc_set"]:
                rec["_colloc_set"].add(colloc)
                if len(rec["collocations"]) < 8:
                    rec["collocations"].append(colloc)

            # 按 (reading, collocation) 去重采样上下文，每种读音最多保留 max_examples 条不同搭配
            rd_ex_cnt = rec["_ex_count_by_rd"].get(rd, 0)
            pair_key = (rd, colloc)
            if pair_key not in rec["_seen_rd_colloc"] and rd_ex_cnt < max(1, int(max_examples)):
                ctx_lo = max(0, p_st - 10)
                ctx_hi = min(len(plain_line), p_en + 12)
                snippet = plain_line[ctx_lo:ctx_hi]
                if ctx_lo > 0:
                    snippet = "…" + snippet
                if ctx_hi < len(plain_line):
                    snippet = snippet + "…"
                ex_item: Dict[str, Any] = {
                    "line": ln_no,
                    "reading": rd,
                    "collocation": colloc,
                    "context": snippet,
                }
                if file_label is not None:
                    ex_item["file"] = file_label
                rec["examples"].append(ex_item)
                rec["_seen_rd_colloc"].add(pair_key)
                rec["_ex_count_by_rd"][rd] = rd_ex_cnt + 1

    glossary: List[Dict[str, Any]] = []
    for rec in entries.values():
        rds = dict(sorted(rec["readings"].items(), key=lambda kv: (-kv[1], kv[0])))
        lbr = {r: rec["lines_by_reading"].get(r, []) for r in rds.keys()}
        is_poly = len(rds) > 1
        priority, p_reasons = _classify_ruby_priority(
            rec["surface"],
            is_polyphone=is_poly,
            has_affix_risk=rec["_affix_risk"],
            distinct_collocs=len(rec["_colloc_set"]),
            speaker_names=spk_list,
        )
        glossary.append({
            "surface": rec["surface"],
            "priority": priority,
            "priority_reasons": p_reasons,
            "readings": rds,
            "lines_by_reading": lbr,
            "count": rec["total_count"],
            "is_polyphone": is_poly,
            "collocations": rec["collocations"],
            "examples": rec["examples"],
        })
    glossary.sort(key=lambda x: (not x["is_polyphone"], 0 if x["priority"] == "high" else 1, -x["count"], x["surface"]))
    return glossary


def merge_ruby_glossaries(
    per_file_glossaries: Sequence[tuple[str, Sequence[Dict[str, Any]]]],
    speaker_names: Optional[Sequence[str]] = None,
    max_examples_per_reading: int = 6,
) -> List[Dict[str, Any]]:
    """将批量多轨的注音词表合并去重为单份全局词表，保留按文件分桶的 ``lines_by_file_reading``。"""
    spk_list = [str(s).strip() for s in (speaker_names or ()) if str(s).strip()]
    merged: Dict[str, Dict[str, Any]] = {}
    for file_key, gl in per_file_glossaries:
        for item in gl:
            surf = item["surface"]
            rec = merged.get(surf)
            if rec is None:
                rec = {
                    "surface": surf,
                    "readings": {},
                    "lines_by_file_reading": {},
                    "total_count": 0,
                    "collocations": [],
                    "_colloc_set": set(),
                    "examples": [],
                    "_seen_rd_colloc": set(),
                    "_ex_count_by_rd": {},
                    "_affix_risk": False,
                }
                merged[surf] = rec
            if "affix_split" in item.get("priority_reasons", ()):
                rec["_affix_risk"] = True
            for rd, cnt in item.get("readings", {}).items():
                rec["readings"][rd] = rec["readings"].get(rd, 0) + int(cnt)
            rec["total_count"] += int(item.get("count", 0))
            for rd, lns in item.get("lines_by_reading", {}).items():
                f_bucket = rec["lines_by_file_reading"].setdefault(file_key, {})
                dst = f_bucket.setdefault(rd, [])
                for ln in lns:
                    if ln not in dst:
                        dst.append(ln)
            for c in item.get("collocations", ()):
                if c not in rec["_colloc_set"]:
                    rec["_colloc_set"].add(c)
                    if len(rec["collocations"]) < 8:
                        rec["collocations"].append(c)
            for ex in item.get("examples", ()):
                rd = ex["reading"]
                colloc = ex.get("collocation", surf)
                pair_key = (rd, colloc)
                rd_cnt = rec["_ex_count_by_rd"].get(rd, 0)
                if pair_key not in rec["_seen_rd_colloc"] and rd_cnt < max(1, int(max_examples_per_reading)):
                    ex_copy = dict(ex)
                    ex_copy.setdefault("file", file_key)
                    rec["examples"].append(ex_copy)
                    rec["_seen_rd_colloc"].add(pair_key)
                    rec["_ex_count_by_rd"][rd] = rd_cnt + 1

    out: List[Dict[str, Any]] = []
    for rec in merged.values():
        rds = dict(sorted(rec["readings"].items(), key=lambda kv: (-kv[1], kv[0])))
        is_poly = len(rds) > 1
        priority, p_reasons = _classify_ruby_priority(
            rec["surface"],
            is_polyphone=is_poly,
            has_affix_risk=rec["_affix_risk"],
            distinct_collocs=len(rec["_colloc_set"]),
            speaker_names=spk_list,
        )
        out.append({
            "surface": rec["surface"],
            "priority": priority,
            "priority_reasons": p_reasons,
            "readings": rds,
            "lines_by_file_reading": rec["lines_by_file_reading"],
            "count": rec["total_count"],
            "is_polyphone": is_poly,
            "collocations": rec["collocations"],
            "examples": rec["examples"],
        })
    out.sort(key=lambda x: (not x["is_polyphone"], 0 if x["priority"] == "high" else 1, -x["count"], x["surface"]))
    return out


def _strip_ruby_for_src(text: str) -> str:
    """把原台本里已有的 {漢字|かな} 折叠成表层字，避免 _src 把花括号与读音拆挂到相邻 token。"""
    if "{" not in text:
        return text
    return _INLINE_RUBY_RE.sub(r"\1", text)


def _attach_source_spans(tokens, annotated_line, source_line, normalize_fn=None):
    """给每个 token 挂 _src —— 直接把可见 token 串与原台本行做双序列锚点对齐并切片。

    不再尝试逐字逆向推算 normalize 行为（曾因 ゛/゜ 变空格后 .strip() 失步导致整行雪崩、
    或因 ℃/㍿ 展开后左边界回退导致符号翻倍）：直接通过全局序列比对生成单调递增边界数组
    ``bound[0..len(vis_str)] -> [0..len(plain_src)]``，再按边界严格切分原始台本。
    """
    if not source_line:
        return
    plain_src = _strip_ruby_for_src(source_line)
    if not plain_src:
        return

    vis_indices = []
    token_ranges = []
    vis_parts = []
    v_pos = 0
    for k, t in enumerate(tokens):
        o = t.get("orig", "")
        if not o or o == "\n" or t.get(F.RUBY_PLACEHOLDER):
            continue
        vis_indices.append(k)
        vis_parts.append(o)
        token_ranges.append((v_pos, v_pos + len(o)))
        v_pos += len(o)

    if not vis_indices:
        return
    vis_str = "".join(vis_parts)
    n_v = len(vis_str)

    sm = difflib.SequenceMatcher(None, vis_str, plain_src, autojunk=False)
    bound = [0] * (n_v + 1)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for d in range(1, i2 - i1 + 1):
                bound[i1 + d] = j1 + d
        elif tag == "delete":
            for d in range(1, i2 - i1 + 1):
                bound[i1 + d] = j1
        elif tag == "insert":
            # 行首未匹配字符保留在 bound[0]=0 之前归属首 token；行中/行尾缝隙并入前一个 token
            if i1 > 0:
                bound[i1] = j2
        elif tag == "replace":
            lv = i2 - i1
            lp = j2 - j1
            for d in range(1, lv):
                bound[i1 + d] = j1 + min(lp, (d * lp + lv - 1) // lv)
            bound[i2] = j2

    for k, (ts, te) in zip(vis_indices, token_ranges):
        tokens[k][F.SRC] = plain_src[bound[ts]:bound[te]]


def _release_gpu(ruby=None):
    global _RUBY_ANNOTATOR
    try:
        import align
        align.clear_model_cache()
    except Exception as e:
        logger.warning("释放 MMS-FA 模型缓存失败: %s: %s", type(e).__name__, e)
    if ruby is not None:
        try:
            ruby.release_gpu()
        except Exception as e:
            logger.warning("释放 AutoRuby GPU 资源失败: %s: %s", type(e).__name__, e)
    elif _RUBY_ANNOTATOR is not None:
        try:
            _RUBY_ANNOTATOR.release_gpu()
        except Exception as e:
            logger.warning("释放全局 AutoRuby GPU 资源失败: %s: %s", type(e).__name__, e)
        _RUBY_ANNOTATOR = None


def _ensure_runtime_ready():
    """确保 NumPy 兼容垫片与内置/系统 PyTorch 已就绪（幂等）。"""
    import np_compat
    np_compat.apply()
    from torch_bootstrap import ensure_torch
    ensure_torch(None)


def run_pipeline(
    script_path: str,
    audio_path: str,
    lora_path: Optional[str],
    log_callback,
    progress_callback,
    dev="auto",
    vram_chunk=3000,
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
    enable_multi_order=False,
    audio_path_b: Optional[str] = None,
    output_dir: Optional[str] = None,
    summary_out: Optional[Dict[str, Any]] = None,
    echo_stdout: bool = True,
    ruby_map: Optional[Any] = None,
    enable_audit: bool = False,
    audit_clips_dir: Optional[str] = None,
    audit_separate: bool = False,
    audit_sep_sources: int = 0,
    audit_clips_all: bool = False,
):
    """V2 核心对齐管线执行入口 (默认常开静音吸附与拟声守门；传入 audio_path_b 时启用多视角联合对齐)"""
    _ensure_runtime_ready()
    import align
    import haruraw2norm as hn

    if audio_path_b:
        enable_multi_order = True

    def log(msg):
        if log_callback:
            log_callback(msg)
        if echo_stdout:
            try:
                print(msg)
            except Exception:
                pass

    progress_callback and progress_callback(0, "🌸 初始化台本与环境...")
    total_start = time.time()
    _ruby = None
    warnings_list: list[str] = []

    def warn(msg: str):
        warnings_list.append(msg)
        log(f"  ⚠ {msg}")

    try:
        if not script_path or not os.path.isfile(script_path):
            raise FileNotFoundError(f"台本文件不存在: {script_path}")
        if not audio_path or not os.path.isfile(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        _norm_ruby_map = load_ruby_map(ruby_map) if ruby_map is not None else None
        if audio_path_b:
            if not os.path.isfile(audio_path_b):
                raise FileNotFoundError(f"视角 B 音频文件不存在: {audio_path_b}")
            if os.path.normcase(os.path.abspath(audio_path)) == os.path.normcase(os.path.abspath(audio_path_b)):
                raise ValueError(f"双视角模式的视角 A 与视角 B 不能指向同一个音频文件: {audio_path}")
        if output_dir and os.path.isfile(os.path.abspath(output_dir)):
            raise ValueError(f"输出目录路径已存在且为普通文件，无法写入目录: {output_dir}")
        if float(line_end_extend) < 0:
            raise ValueError(f"行尾延长秒数不能为负数: {line_end_extend}")
        if int(vram_chunk) <= 0:
            raise ValueError(f"显存分块帧数必须 > 0: {vram_chunk}")

        import advanced_config as _adv
        _adv.load()
        for _adv_w in list(getattr(_adv, "LAST_WARNINGS", [])):
            warn(f"[advanced_config] {_adv_w}")

        log("[1/4] 🌸 读取台本并进行日文注音转换...")
        raw_text, detected_enc = read_text_auto_encoding(script_path)
        if detected_enc.endswith("-fallback") or "\ufffd" in raw_text:
            bad_cnt = raw_text.count("\ufffd")
            warn(
                f"台本编码异常（识别为 {detected_enc}，含 {bad_cnt} 处无法解码的替换字符 ），"
                "建议先用 UTF-8 / Shift-JIS 重新保存台本，否则可能导致对齐错位。"
            )
        elif detected_enc not in ("utf-8", "utf-8-sig"):
            log(f"  ✨ 自动识别并兼容台本编码: {detected_enc.upper()} → UTF-8")

        ruby = AutoRuby()
        _ruby = ruby
        _ann_inst = getattr(ruby, "_ann", None)
        if _ann_inst is not None:
            if getattr(_ann_inst, "fugashi_error", None):
                warn(
                    f"fugashi/UniDic 未就绪 ({_ann_inst.fugashi_error})，"
                    "当前已降级为 pykakasi 逐字注音（注音精度会下降）"
                )
            if getattr(_ann_inst, "_OVERRIDES_WARNING", None):
                warn(str(_ann_inst._OVERRIDES_WARNING))

        lines = raw_text.splitlines()

        if enable_multi_order:
            from align_multispeaker import auto_normalize_speaker_headers
            lines, _auto_spks = auto_normalize_speaker_headers(lines)
            if _auto_spks:
                log("  ✨ 自动识别独立角色标题: " + " / ".join(f"〖{s}〗" for s in _auto_spks))

        # 说话人标记在【注音前】剥离：〖名字〗整段不进入 annotate / tokenize，
        # 因此天然不产生任何发音 token、不进对齐。剥离后记录每个「注音行」的
        # 说话人归属，供后续按说话人分组对齐使用。
        #   _spk_of_line[k] = 注音行 k 的说话人元组（可能多人并列）
        # 原文还原不依赖此处的下标映射：_attach_source_spans 直接按行号取
        # _body_lines（见下方 _src_line = lines[_li]）。
        _spk_of_line = []
        _body_lines = []
        _src_line_nos = []
        _cur_spk = ()
        for _raw_ln_no, _raw in enumerate(lines, 1):
            _segs, _cur_spk = _split_speaker_segments(_raw, _cur_spk)
            for _spk, _body in _segs:
                _body_lines.append(_body)
                _spk_of_line.append(_spk)
                _src_line_nos.append(_raw_ln_no)
        if not any(b.strip() for b in _body_lines):
            raise ValueError(f"台本不包含任何有效文本行: {script_path}")

        # 逐行清洗「内联振假名泄露」后再注音。
        # 部分台本把振假名写成未标记的行内假名（身体んたい / 沖縄きなわ /
        # 気持ちちいい），这些残留假名会被分词器当成实词，污染注音与对齐标签，
        # 并经 _src 回填漏进字幕正文。
        # 清洗必须在 annotate 之前、且必须同时作用于 _body_lines（供 _src 还原）
        # —— 否则 _attach_source_spans 的缝隙补齐会把刚剥掉的假名塞回正文。
        _leak_cleaner = None
        if _ann_inst is not None and getattr(_ann_inst, "_tagger", None):
            from inline_leak_clean import InlineLeakCleaner
            _leak_cleaner = InlineLeakCleaner(_ann_inst)

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

        # 逐行注音（正文，不含标记；若提供 --ruby-map 则执行预注入+权威后置覆写）
        annotated_lines, _ruby_override_hits = annotate_lines_with_overrides(
            ruby, _body_lines, src_line_nos=_src_line_nos, ruby_map=_norm_ruby_map, script_path=script_path
        )
        if _norm_ruby_map is not None:
            log(f"  ✨ 注音覆写表已应用: 命中 {_ruby_override_hits} 处")
        annotated_text = "\n".join(annotated_lines)
        # 原文文本行（供 _src 还原）：剥离标记 + 内联假名清洗后的正文
        lines = _body_lines
        log(f"  ✨ 台本处理完成 ({len(lines)} 行)")

        # 及时释放 yomikata dBert 显存（步骤 2~4 的声学对齐不再需要 dBert，同时保留 UniDic 单例供批处理复用）
        if _ann_inst is not None:
            try:
                _ann_inst.release_gpu()
            except Exception as _e:
                warn(f"释放 yomikata dBert 显存失败: {type(_e).__name__}: {_e}")

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
            return (
                len(_KANA_CH.findall(txt))
                + 2 * len(_DIGITS.findall(txt))
                + len(_ALPHA.findall(txt))
            )

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

            # 子句数对齐后，把每段的字数与子句首偏移摊到该段 token 上
            out = [0] * len(toks)
            heads = [None] * len(toks)
            idx_of = {id(t): k for k, t in enumerate(toks)}
            for si, seg_toks in enumerate(tsegs):
                n = _text_weight(segs[si] if si < len(segs) else "")
                if n == 0:
                    # 极端情况（注音串切分与 token 侧不同步）：回退数 orig
                    n = sum(_text_weight(t.get("orig", "")) for t in seg_toks)
                seg_head = idx_of[id(seg_toks[0])]
                for t in seg_toks:
                    ti = idx_of[id(t)]
                    out[ti] = n
                    heads[ti] = seg_head
            return out, heads

        _src_idx_to_lineno = []
        _src_idx_to_ann = []
        for _li, line in enumerate(annotated_text.splitlines(keepends=True)):
            if line.strip():
                _toks = hn.process_haruhi_line(line, lang, 0, 1)
                _counts, _heads = _line_kana_counts(line, _toks)
                _base_idx = len(result_list)
                for _t, _n, _h in zip(_toks, _counts, _heads):
                    _t[F.KANA_N] = _n
                    if _h is not None:
                        _t[F.CLAUSE_HEAD] = _base_idx + _h
                # 挂 _src：该 token 对应的【原文台本字符】，供 LRC 还原原字符
                # （对齐仍消费归一化后的 token，仅输出层用原文，故标签不受影响）
                _src_line = lines[_li] if _li < len(lines) else ""
                _attach_source_spans(_toks, line, _src_line)
                # 挂 _spk：本行说话人元组（注音前已从标记剥离得到）。
                # 仅用于最终字幕前缀标注，不参与对齐。
                _spk_line = _spk_of_line[_li] if _li < len(_spk_of_line) else ()
                if _spk_line:
                    for _t in _toks:
                        _t[F.SPK] = _spk_line
                result_list.extend(_toks)
                # 每个源行后插入行边界，避免整轨被并成单一对齐块（会导致 LRC 塌成极少行）。
                result_list.append({"orig": "\n", "type": 0, "pron": ""})
                _src_idx_to_lineno.append(_src_line_nos[_li] if _li < len(_src_line_nos) else (_li + 1))
                _src_idx_to_ann.append(line.strip())
        if result_list and result_list[-1].get("orig") != "\n":
            result_list.append({"orig": "\n", "type": 0, "pron": ""})

        alignment_tokens = []
        line_group = []
        clause_surfaces = []
        # per-token 易错拟声掩码（与 alignment_tokens 同步累积，按子句整体判定）。
        # 供对齐阶段降低这些 token 的发射分权重，削弱不可信的拟声时长对相邻
        # 语义段的挤压。is_easy_ono 需假名原文，故必须在管线侧（有 orig）算。
        ono_clause_flags = []   # 每子句 1 个 flag (1 = 易错拟声子句)
        clause_speakers = []    # 每子句对应的说话人元组
        _cur_ono_surface = []
        _cur_clause_spk = ()
        token_to_index_map = {}
        current_line = []
        flat_idx = 0
        src_idx = 0
        # 子句边界判据（已统一到 clause.py；ono_filter / merge_short /
        # resegment 全部引用同一份，此处也直接引 clause，少一层间接）
        from clause import is_clause_boundary as _is_clause_boundary, mark_short_comma_bridges as _mark_comma_bridges
        from ono_filter import is_easy_ono as _ieo
        _mark_comma_bridges(result_list)
        for i, item in enumerate(result_list):
            orig = item.get("orig", "")
            pron = item.get("pron", "")
            # 子句边界：换行，或「未转化为 token (无 pron) 且非注音占位符」的符号项
            is_boundary = _is_clause_boundary(item)
            if is_boundary:
                if current_line:
                    alignment_tokens.append(current_line)
                    line_group.append(src_idx)
                    clause_speakers.append(_cur_clause_spk)
                    _surf = "".join(_cur_ono_surface)
                    clause_surfaces.append(_surf)
                    ono_clause_flags.append(1 if _ieo(_surf) else 0)
                    _cur_ono_surface = []
                    _cur_clause_spk = ()
                    current_line = []
            if orig == "\n":
                src_idx += 1
            if pron:
                if not current_line:
                    _cur_clause_spk = tuple(item.get(F.SPK, ())) if item.get(F.SPK) else ()
                current_line.append(pron)
                _cur_ono_surface.append(orig)
                token_to_index_map[flat_idx] = i
                flat_idx += 1
        if current_line:
            alignment_tokens.append(current_line)
            line_group.append(src_idx)
            clause_speakers.append(_cur_clause_spk)
            _surf = "".join(_cur_ono_surface)
            clause_surfaces.append(_surf)
            ono_clause_flags.append(1 if _ieo(_surf) else 0)

        n_tokens = sum(len(lt) for lt in alignment_tokens)
        log(f"  ✨ 解析完成: {n_tokens} tokens, {len(alignment_tokens)} 行")
        if n_tokens == 0 or not alignment_tokens:
            raise ValueError(
                f"台本未解析出任何可对齐的发音音素 (tokens=0)，请检查台本是否为空或仅含标点/注释: {script_path}"
            )

        progress_callback and progress_callback(25, "⚡ 双声道对齐计算中...")
        log("[3/4] 🎧 MMS-FA + LoRA 双声道 Better-Ear 全局 CTC 对齐 (静音吸附已启用)...")

        # 注意: 必须先判 None/空值。str(None).upper() == "NONE" 会命中显式禁用分支，
        # 导致下面的 default 回退永不执行（历史逻辑缺陷）。
        #
        # _lora_missing 记录「本想用默认 LoRA、但权重文件不存在」这一情形；
        # 若用户显式指定了自定义的 lora_path 且文件不存在，则直接报错终止，绝不静默切换其他模型。
        _lora_missing = None
        default_lora = str(getattr(align, "DEFAULT_LORA_PATH", ""))
        _norm_default = os.path.normcase(os.path.abspath(default_lora)) if default_lora else ""
        if lora_path is None or str(lora_path).strip() == "":
            if default_lora and os.path.isfile(default_lora):
                lora_path = default_lora
            else:
                lora_path = None
                _lora_missing = default_lora
        elif str(lora_path).upper() in ("NO_LORA", "NONE", "DISABLE"):
            lora_path = "NO_LORA"
        elif not os.path.isfile(str(lora_path)):
            _norm_req = os.path.normcase(os.path.abspath(str(lora_path)))
            if _norm_default and _norm_req == _norm_default:
                lora_path = None
                _lora_missing = default_lora
            else:
                raise FileNotFoundError(
                    f"指定的 LoRA 权重文件不存在: {lora_path}。"
                    "请检查路径是否正确；若需使用基础模型，请传 NO_LORA。"
                )

        if lora_path and lora_path != "NO_LORA":
            log(f"  ✨ 使用 LoRA: {os.path.basename(lora_path)}")
        elif _lora_missing:
            # 缺件不等于「用户选择不用」：显式提示并记录到 warnings，避免精度下降被静默接受。
            warn(
                f"未找到默认 LoRA 权重 ({_lora_missing})，已回退到基础 MMS-FA 模型 —— 对齐精度会下降。"
                "请将 lora/fa_asmr_v9e10.pt 放到该位置；若确要用基础模型，传 lora_path='NO_LORA' 可关闭本提示。"
            )
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
        _stream_clause_lists = [list(range(len(alignment_tokens)))]
        _mp_meta: Optional[Dict[str, Any]] = None
        if audio_path_b:
            from align_multispeaker import (
                apply_skipped_ono_marks,
                post_process_and_merge_streams,
            )
            from align_multiperspective import (
                align_multiperspective_joint,
                project_result_list_to_perspective,
            )
            alignment_results, _stream_clause_lists, _mp_meta = align_multiperspective_joint(
                audio_path_A=audio_path,
                audio_path_B=audio_path_b,
                alignment_tokens=alignment_tokens,
                line_group=line_group,
                ono_clause_flags=ono_clause_flags,
                clause_speakers=clause_speakers,
                lora_path=lora_path,
                dev=dev,
                vram_chunk=int(vram_chunk),
                enable_silence_snap=enable_silence_snap,
                enable_ono_gate=enable_ono_gate,
                log_callback=log,
                clause_surfaces=clause_surfaces,
            )
        elif enable_multi_order:
            from align_multispeaker import (
                align_multispeaker_or_reordered,
                apply_skipped_ono_marks,
                post_process_and_merge_streams,
            )
            alignment_results, _stream_clause_lists = align_multispeaker_or_reordered(
                audio_path=audio_path,
                alignment_tokens=alignment_tokens,
                line_group=line_group,
                ono_clause_flags=ono_clause_flags,
                clause_speakers=clause_speakers,
                lora_path=lora_path,
                dev=dev,
                vram_chunk=int(vram_chunk),
                enable_silence_snap=enable_silence_snap,
                enable_ono_gate=enable_ono_gate,
                blank_punct_mask=blank_punct_mask,
                log_callback=log,
                diagnostics_out=_align_diag,
                clause_surfaces=clause_surfaces,
            )
        else:
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

        def _finalize_post_rl(base_rl, align_res, side_label: str = ""):
            rl = copy.deepcopy(base_rl)
            for i, result in enumerate(align_res):
                if i in token_to_index_map and not result.get("skipped") and "start" in result and "end" in result:
                    original_index = token_to_index_map[i]
                    rl[original_index]["start"] = result["start"]
                    rl[original_index]["end"] = result["end"]

            ind_rl = None
            if enable_multi_order and len(_stream_clause_lists) > 1:
                rl, ind_rl = post_process_and_merge_streams(
                    result_list=rl,
                    alignment_results=align_res,
                    token_to_index_map=token_to_index_map,
                    stream_clause_lists=_stream_clause_lists,
                    enable_ono_gate=enable_ono_gate,
                    enable_resegment=enable_resegment,
                    enable_merge_short=enable_merge_short,
                    merge_short_chars=merge_short_chars,
                    merge_short_gap=merge_short_gap,
                    merge_ono_chars=merge_ono_chars,
                    merge_ono_gap=merge_ono_gap,
                    log_callback=log,
                    return_independent_rl=True,
                )
            else:
                if enable_multi_order and enable_ono_gate:
                    apply_skipped_ono_marks(rl, align_res, token_to_index_map)

                # ---- 拟声错位处理 · 阶段一【标记】----------------------------
                # 跑结构判据给可疑拟声子句打 `_ono_mark`，**一个 token 都不删**。
                # 数据保持完整，后续 resegment / 短句合并才能切出真实的最终行结构；
                # 决策放到阶段三、在最终行结构上一次性做（见下方 decide_ono_deletions）。
                if enable_ono_gate:
                    from ono_filter import mark_ono_candidates as _ono_mark
                    _ono_mark(
                        result_list=rl,
                        alignment_results=align_res,
                        token_to_index_map=token_to_index_map,
                        log_callback=log,
                    )

                # ---- 标点驱动重分段（候选=标点，证据=两侧 CTC blank 间隔；标点保留） ----
                if enable_resegment:
                    from resegment import apply_punctuation_resegment as _reseg
                    _rl_time = {}
                    for _k, _r in enumerate(align_res):
                        _ri = token_to_index_map.get(_k)
                        if (
                            _ri is not None
                            and not _r.get("skipped")
                            and _r.get("original_start") is not None
                            and _r.get("original_end") is not None
                        ):
                            _rl_time[_ri] = (
                                float(_r["original_start"]),
                                float(_r["original_end"]),
                            )
                    _n_before = sum(1 for _it in rl if _it.get("orig") == "\n") + 1
                    rl = _reseg(rl, _rl_time, frame_sec=0.02, thr_frames=50, max_line_chars=22)
                    _n_after = sum(1 for _it in rl if _it.get("orig") == "\n") + 1
                    _pfx = f"[{side_label}] " if side_label else ""
                    log(f"  ✨ {_pfx}标点重分段: {_n_before} → {_n_after} 行（标点保留）")

                # ---- 短句合并：把 resegment 切碎的相邻短句重新合并 ----
                if enable_merge_short:
                    from merge_short import merge_short_lines as _merge, count_inserted_breaks as _cb
                    _nb = _cb(rl)
                    rl = _merge(
                        rl,
                        max_chars=merge_short_chars,
                        max_gap=merge_short_gap,
                        ono_max_chars=merge_ono_chars,
                        ono_max_gap=merge_ono_gap,
                    )
                    _na = _cb(rl)
                    if _nb != _na:
                        _pfx = f"[{side_label}] " if side_label else ""
                        log(f"  ✨ {_pfx}短句合并: 合并 {_nb - _na} 处碎句")

                # ---- 拟声错位处理 · 阶段三【决策】----------------------------
                if enable_ono_gate:
                    from ono_filter import decide_ono_deletions as _ono_decide
                    _ono_decide(result_list=rl, log_callback=log)

                for i in range(len(rl)):
                    item = rl[i]
                    if item.get(F.ONO_DROPPED):
                        continue
                    if "start" not in item and item.get("type", 0) in [1, 2, 3, 4, 5]:
                        found = False
                        for j in range(i - 1, -1, -1):
                            if rl[j].get("type") == 0 and rl[j].get("orig") == "\n":
                                break
                            if rl[j].get(F.ONO_DROPPED):
                                continue
                            if "start" in rl[j]:
                                item["start"] = rl[j]["start"]
                                item["end"] = rl[j]["end"]
                                found = True
                                break
                        if not found:
                            for j in range(i + 1, len(rl)):
                                if rl[j].get("type") == 0 and rl[j].get("orig") == "\n":
                                    break
                                if rl[j].get(F.ONO_DROPPED):
                                    continue
                                if "start" in rl[j]:
                                    item["start"] = rl[j]["start"]
                                    item["end"] = rl[j]["end"]
                                    break
            return rl, ind_rl

        log("[4/4] 📜 组装标准 LRC 与 SRT 字幕...")
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        if audio_path_b and _mp_meta is not None:
            from align_multiperspective import project_result_list_to_perspective
            _spk_pref = _mp_meta.get("spk_persp_pref")

            if "flat_A" in _mp_meta and "flat_B" in _mp_meta:
                _rl_lrc_A, _rl_ind_A = _finalize_post_rl(result_list, _mp_meta["flat_A"], side_label="视角A")
                _rl_lrc_B, _rl_ind_B = _finalize_post_rl(result_list, _mp_meta["flat_B"], side_label="视角B")
                _target_ind_A = _rl_ind_A if _rl_ind_A is not None else _rl_lrc_A
                _target_ind_B = _rl_ind_B if _rl_ind_B is not None else _rl_lrc_B
            else:
                _rl_lrc_A, _rl_ind_raw = _finalize_post_rl(result_list, alignment_results)
                _rl_lrc_B = _rl_lrc_A
                _target_ind_A = _rl_ind_raw if _rl_ind_raw is not None else _rl_lrc_A
                _target_ind_B = _target_ind_A

            # 1. LRC 视轨生成（水平归并）
            rl_A = project_result_list_to_perspective(
                _rl_lrc_A,
                _mp_meta["u_A_st_sec"],
                _mp_meta["dur_A_sec"],
                perspective_side="A",
                spk_persp_pref=_spk_pref,
            )
            rl_B = project_result_list_to_perspective(
                _rl_lrc_B,
                _mp_meta["u_B_st_sec"],
                _mp_meta["dur_B_sec"],
                perspective_side="B",
                spk_persp_pref=_spk_pref,
            )
            lrc_A = build_standard_lrc(rl_A)
            lrc_B = build_standard_lrc(rl_B)
            if line_end_extend > 0:
                from lrc_end_extend import extend_lrc
                lrc_A, _ = extend_lrc(lrc_A, line_end_extend)
                lrc_B, _ = extend_lrc(lrc_B, line_end_extend)

            # 2. SRT 视轨生成（多角色完全自治独立流）
            rl_indep_A = project_result_list_to_perspective(
                _target_ind_A,
                _mp_meta["u_A_st_sec"],
                _mp_meta["dur_A_sec"],
                perspective_side="A",
                spk_persp_pref=_spk_pref,
            )
            rl_indep_B = project_result_list_to_perspective(
                _target_ind_B,
                _mp_meta["u_B_st_sec"],
                _mp_meta["dur_B_sec"],
                perspective_side="B",
                spk_persp_pref=_spk_pref,
            )
            srt_A = build_standard_srt(rl_indep_A, line_end_extend=line_end_extend)
            srt_B = build_standard_srt(rl_indep_B, line_end_extend=line_end_extend)

            base_A = os.path.splitext(os.path.basename(audio_path))[0]
            base_B = os.path.splitext(os.path.basename(audio_path_b))[0]
            dir_A = os.path.abspath(output_dir) if output_dir else os.path.dirname(os.path.abspath(audio_path))
            dir_B = os.path.abspath(output_dir) if output_dir else os.path.dirname(os.path.abspath(audio_path_b))
            lrc_path = os.path.join(dir_A, f"{base_A}.lrc")
            lrc_path_b = os.path.join(dir_B, f"{base_B}.lrc")
            srt_path = os.path.join(dir_A, f"{base_A}.srt")
            srt_path_b = os.path.join(dir_B, f"{base_B}.srt")

            if os.path.normcase(os.path.abspath(lrc_path)) == os.path.normcase(os.path.abspath(lrc_path_b)):
                raise ValueError(
                    f"双视角输出路径冲突（视角 A 与视角 B 将覆盖写入同一文件: {lrc_path}），"
                    "请区分两轨音频文件名或不要使用同一 --output-dir。"
                )

            if not lrc_A.strip() and not lrc_B.strip() and not srt_A.strip() and not srt_B.strip():
                raise RuntimeError("双视角对齐后未生成任何有效字幕内容（视角 A/B 均为空），请检查音频与台本是否匹配")
            if not lrc_A.strip() and not srt_A.strip():
                warn("视角 A 未生成有效字幕内容，请检查视角 A 音频与台本时间范围")
            if not lrc_B.strip() and not srt_B.strip():
                warn("视角 B 未生成有效字幕内容，请检查视角 B 音频与台本时间范围")

            with open(lrc_path, "w", encoding="utf-8") as f:
                f.write(lrc_A)
            with open(lrc_path_b, "w", encoding="utf-8") as f:
                f.write(lrc_B)
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(srt_A)
            with open(srt_path_b, "w", encoding="utf-8") as f:
                f.write(srt_B)

            total_elapsed = time.time() - total_start
            n_lines_a = len(lrc_A.splitlines())
            n_lines_b = len(lrc_B.splitlines())
            n_cues_a = srt_A.count(" --> ")
            n_cues_b = srt_B.count(" --> ")
            log(f"\n  💖 多视角联合对齐完成! Δt = {_mp_meta['delta_sec']:+.2f}s | 总耗时 {total_elapsed:.1f}s")
            log(f"  🎯 视角 A: {lrc_path} ({n_lines_a} 行) | {srt_path} ({n_cues_a} 条独立字幕)")
            log(f"  🎯 视角 B: {lrc_path_b} ({n_lines_b} 行) | {srt_path_b} ({n_cues_b} 条独立字幕)")
            if summary_out is not None:
                summary_out.update({
                    "mode": "multi_persp",
                    "detected_encoding": detected_enc,
                    "script_lines": len(lines),
                    "tokens": n_tokens,
                    "clauses": len(alignment_tokens),
                    "delta_sec": float(_mp_meta["delta_sec"]),
                    "elapsed_sec": round(total_elapsed, 2),
                    "outputs": {
                        "lrc": lrc_path,
                        "srt": srt_path,
                        "lrc_b": lrc_path_b,
                        "srt_b": srt_path_b,
                    },
                    "counts": {
                        "lrc_lines": n_lines_a,
                        "srt_cues": n_cues_a,
                        "lrc_b_lines": n_lines_b,
                        "srt_b_cues": n_cues_b,
                    },
                    "warnings": list(warnings_list),
                })
                if _norm_ruby_map is not None:
                    summary_out["ruby_overrides_applied"] = _ruby_override_hits
        else:
            result_list, independent_rl = _finalize_post_rl(result_list, alignment_results)
            target_rl_for_srt = independent_rl if independent_rl is not None else result_list
            # 1. 组装 LRC 字幕
            lrc_content = build_standard_lrc(result_list)
            if line_end_extend > 0:
                from lrc_end_extend import extend_lrc
                lrc_content, _n_ext = extend_lrc(lrc_content, line_end_extend)
                if _n_ext:
                    log(f"  ✨ LRC 行尾延长 ({line_end_extend:.1f}s): 处理 {_n_ext} 个空白标记")

            # 2. 组装 SRT 字幕（角色完全独立）
            srt_content = build_standard_srt(target_rl_for_srt, line_end_extend=line_end_extend)

            # 统一以音频基名输出，确保模糊匹配/手动配对时播放器能自动挂载，且与双视角模式及 UI 查找一致
            base = os.path.splitext(os.path.basename(audio_path))[0]
            out_dir = os.path.abspath(output_dir) if output_dir else os.path.dirname(os.path.abspath(audio_path))
            lrc_path = os.path.join(out_dir, f"{base}.lrc")
            srt_path = os.path.join(out_dir, f"{base}.srt")

            if not lrc_content.strip() and not srt_content.strip():
                raise RuntimeError("对齐后未生成任何有效字幕内容（LRC 与 SRT 均为空），请检查音频与台本是否匹配")

            with open(lrc_path, "w", encoding="utf-8") as f:
                f.write(lrc_content)
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(srt_content)

            total_elapsed = time.time() - total_start
            n_lines_lrc = len(lrc_content.splitlines())
            n_cues_srt = srt_content.count(" --> ")
            log(f"\n  💖 对齐完成! 总耗时 {total_elapsed:.1f}s")
            log(f"  🎯 输出 LRC: {lrc_path} ({n_lines_lrc} 行)")
            log(f"  🎯 输出 SRT: {srt_path} ({n_cues_srt} 条独立字幕)")
            if summary_out is not None:
                summary_out.update({
                    "mode": "multi_speaker" if enable_multi_order else "single",
                    "detected_encoding": detected_enc,
                    "script_lines": len(lines),
                    "tokens": n_tokens,
                    "clauses": len(alignment_tokens),
                    "elapsed_sec": round(total_elapsed, 2),
                    "outputs": {
                        "lrc": lrc_path,
                        "srt": srt_path,
                    },
                    "counts": {
                        "lrc_lines": n_lines_lrc,
                        "srt_cues": n_cues_srt,
                    },
                    "warnings": list(warnings_list),
                })
                if _norm_ruby_map is not None:
                    summary_out["ruby_overrides_applied"] = _ruby_override_hits

        if enable_audit or audit_clips_dir or audit_separate:
            from fa_asmr_audit import (
                build_alignment_audit,
                build_multiperspective_audit,
                export_audit_clips,
            )
            _cl_line_nos = [
                _src_idx_to_lineno[lg] if (0 <= lg < len(_src_idx_to_lineno)) else (lg + 1)
                for lg in line_group
            ]
            _cl_ann = [
                _src_idx_to_ann[lg] if (0 <= lg < len(_src_idx_to_ann)) else ""
                for lg in line_group
            ]
            _eff_clips_dir = (
                os.path.abspath(audit_clips_dir)
                if audit_clips_dir
                else (
                    os.path.join(
                        os.path.abspath(output_dir) if output_dir else os.path.dirname(os.path.abspath(audio_path)),
                        "audit_clips",
                    )
                    if audit_separate
                    else None
                )
            )
            _uniq_spks = {
                s
                for sp in clause_speakers
                for s in (sp if isinstance(sp, (tuple, list)) else ((sp,) if sp else ()))
                if s
            }
            if int(audit_sep_sources) > 0:
                _eff_sep_sources = max(2, min(4, int(audit_sep_sources)))
                _eff_enable_separate = bool(audit_separate)
            else:
                _eff_sep_sources = 3 if len(_uniq_spks) >= 3 else 2
                # 单人轨（<=1 说话人且非多人模式）照常完整体检与切原声 WAV，但自动跳过立体声空间拆轨
                _eff_enable_separate = bool(audit_separate) and (len(_uniq_spks) >= 2 or bool(enable_multi_order))

            if audio_path_b and _mp_meta is not None and "flat_B" in _mp_meta:
                _diag_a = {
                    "ctc_nb_active": _mp_meta.get("ctc_nb_active_A"),
                    "duration_sec": float(_mp_meta.get("dur_A_sec", 0.0)),
                    "frame_sec": float(_mp_meta.get("frame_sec", 0.02)),
                }
                _diag_b = {
                    "ctc_nb_active": _mp_meta.get("ctc_nb_active_B"),
                    "duration_sec": float(_mp_meta.get("dur_B_sec", 0.0)),
                    "frame_sec": float(_mp_meta.get("frame_sec", 0.02)),
                }
                _audit_report = build_multiperspective_audit(
                    alignment_tokens=alignment_tokens,
                    alignment_results_a=_mp_meta.get("flat_A", alignment_results),
                    alignment_results_b=_mp_meta["flat_B"],
                    ono_clause_flags=ono_clause_flags,
                    clause_surfaces=clause_surfaces,
                    clause_speakers=clause_speakers,
                    clause_line_nos=_cl_line_nos,
                    clause_annotated=_cl_ann,
                    diagnostics_a=_diag_a,
                    diagnostics_b=_diag_b,
                    delta_sec=float(_mp_meta.get("delta_sec", 0.0)),
                    sim_segs=_mp_meta.get("sim_segs"),
                )
            else:
                _audit_report = build_alignment_audit(
                    alignment_tokens=alignment_tokens,
                    alignment_results=alignment_results,
                    ono_clause_flags=ono_clause_flags,
                    clause_surfaces=clause_surfaces,
                    clause_speakers=clause_speakers,
                    clause_line_nos=_cl_line_nos,
                    clause_annotated=_cl_ann,
                    diagnostics=_align_diag,
                )

            if _eff_clips_dir:
                _n_clips = export_audit_clips(
                    audio_path,
                    _audit_report,
                    _eff_clips_dir,
                    enable_separate=_eff_enable_separate,
                    sep_sources=_eff_sep_sources,
                    audio_path_b=audio_path_b,
                    priority_only=not bool(audit_clips_all),
                )
                _audit_report["exported_clips_count"] = _n_clips
                _audit_report["clips_dir"] = _eff_clips_dir
            if audit_separate and _eff_clips_dir:
                if _eff_enable_separate:
                    _sep_msg = f" | 空间分离触发={_audit_report.get('separated_clips_count', 0)}/{_audit_report.get('exported_clips_count', 0)} (K={_eff_sep_sources})"
                else:
                    _sep_msg = " | 空间分离=单人轨已跳过"
            else:
                _sep_msg = ""
            _mp_tag = "·双视角联合" if _audit_report.get("is_multiperspective") else ""
            log(
                f"  🔎 [声学体检{_mp_tag}] 基线中位分={_audit_report['track_median_score']:.2f} | "
                f"高危必审={_audit_report.get('priority_review_count', 0)} | "
                f"可疑实词句={_audit_report['suspect_clauses_count']} | "
                f"未覆盖活跃语音空档={_audit_report['unaligned_speech_gaps_count']}{_sep_msg}"
            )
            if summary_out is not None:
                summary_out["audit"] = _audit_report

        progress_callback and progress_callback(100, "💖 全部完成!")
        if not persist_gpu:
            _release_gpu(_ruby)
        return True, lrc_path, None

    except Exception as e:
        import traceback
        err = traceback.format_exc()
        err_msg = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
        log(f"\n  💔 发生错误 ({err_msg}):\n{err}")
        if summary_out is not None:
            summary_out.update({
                "error_type": type(e).__name__,
                "warnings": list(warnings_list),
            })
        if not persist_gpu:
            _release_gpu(_ruby)
        return False, None, err_msg


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


_PERSP_PATTERNS = [
    re.compile(r"^(\d+)[_\-\s]*([ABab])(?:[_\s\-\.](.*))?$"),
    re.compile(r"^(.*?)[_\s\-\.]+(?:perspective|cam|视角)?([ABab])(?:[_\s\-\.](.*))?$", re.IGNORECASE),
    re.compile(r"^(.*?)[_\s\-\.]+(?:perspective|cam|视角)?([12])$", re.IGNORECASE),
]


def _extract_perspective_tag(stem):
    """从音频文件名中识别是否带有视角 A / B 或日文「視点」标记。返回 (base_key, side_tag, desc)

    例如: '15A_超绝差距_龟男' -> ('15', 'A', '超绝差距_龟男')
          'Lesson_1_A'        -> ('Lesson_1', 'A', '')
          'トラック7_行ってきますのＮＴＲキス‗主人公視点' -> ('トラック7_行ってきますのNTRキス', '主人公', '')
          'トラック7_行ってきますのＮＴＲキス‗夫視点'   -> ('トラック7_行ってきますのNTRキス', '夫', '')
    """
    import unicodedata
    # 先处理 DLsite 等常用的 Unicode 特殊连接符（双下划线 U+2017、全角下划线 U+FF3F），避免 normalize 丢失连接
    s = stem.replace("\u2017", "_").replace("\uff3f", "_")
    s = unicodedata.normalize("NFKC", s)

    # 1. 匹配常见的日文视点后缀（贪婪匹配 base，提取最后一个连接符后面的视点名）
    m_persp = re.match(r"^(.*)[\-_\s]+(.*?)(?:視点|视点|视角)(?:[_\s]*(?:集中)?Ver)?$", s, re.IGNORECASE)
    if m_persp:
        base = m_persp.group(1).strip(" \t\r\n_")
        tag = m_persp.group(2).strip(" \t\r\n_")
        return base, tag or "persp", "perspective"

    # 2. 匹配英文字母 A/B/1/2
    m2 = re.match(r"^(.*)[\-_\s]+(?:perspective|cam|视角)?([ABab12])(?:[_\-\s](.*))?$", s, re.IGNORECASE)
    if m2:
        return m2.group(1).strip(" \t\r\n_"), m2.group(2).upper(), m2.group(3) or ""

    for pat in _PERSP_PATTERNS:
        m = pat.match(stem)
        if m:
            base = m.group(1).rstrip("_- ")
            raw_side = m.group(2).upper()
            side = "A" if raw_side in ("A", "1") else "B"
            desc = m.group(3) if len(m.groups()) >= 3 and m.group(3) else ""
            return base, side, desc
    m_num = re.match(r"^(\d+)(?:[_\s\-\.](.*))?$", stem)
    if m_num:
        return m_num.group(1), None, m_num.group(2) or ""
    return stem, None, ""


def _clean_stem_for_matching(name: str) -> str:
    """字符与修饰词归一化，用于跨格式模糊比对。"""
    import unicodedata
    s = name.replace("\u2017", "_").replace("\uff3f", "_")
    s = unicodedata.normalize("NFKC", s).strip().lower()
    s = re.sub(r"[\(\[\{【〖（].*?[\)\]\}】〗）]", "", s)
    # 清洗掉视点后缀
    s = re.sub(r"[_\-\s\.]+(?:主人公|夫|彼|彼女|男|女|私|僕|俺)?(?:視点|视点|视角)(?:[_\s]*(?:集中)?ver)?$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[_\-\s\.]+(?:perspective|cam)?[_\-\s]*[ab12]$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[_\-\s\.]+(台本|script|raw|clean|正文|字幕|text)$", "", s)
    s = re.sub(r"[_\-\s\.]+(master|audio|vocal|48k|16bit)$", "", s)
    s = re.sub(r"\b0+(\d+)\b", r"\1", s)
    s = re.sub(r"[_\-\s\.]+", " ", s).strip()
    return s


def auto_pair_files(file_paths, mode="single"):
    """智能配对台本与音频文件，支持核心锚点提取与模糊初筛。
    - mode="single" 或 "multi_speaker": 常规一对一配对
    - mode="multi_persp": 双视角模式，自动合并同一基名下的视角 A 与 视角 B 音频
    """
    clean_paths = [os.path.abspath(f) for f in file_paths if f and str(f).strip()]
    txt_files = [
        f for f in clean_paths
        if os.path.splitext(f)[1].lower() == ".txt"
        and not f.lower().endswith(".dropped.txt")
    ]
    audio_files = [f for f in clean_paths if os.path.splitext(f)[1].lower() in AUDIO_EXTS]

    pairs = []
    unmatched = []

    if mode == "multi_persp":
        # ── 双视角模式配对逻辑 ──
        aud_by_base = {}
        for aud in audio_files:
            stem = os.path.splitext(os.path.basename(aud))[0]
            base, side, desc = _extract_perspective_tag(stem)
            aud_by_base.setdefault(base, []).append((aud, side, stem))

        txt_by_base = {}
        for txt in txt_files:
            stem = os.path.splitext(os.path.basename(txt))[0]
            base, side, desc = _extract_perspective_tag(stem)
            txt_by_base.setdefault(base, []).append(txt)
            if stem != base:
                txt_by_base.setdefault(stem, []).append(txt)

        used_txts = set()
        used_auds = set()
        all_bases = sorted(set(aud_by_base.keys()) | set(txt_by_base.keys()))
        for base in all_bases:
            aud_list = [item for item in aud_by_base.get(base, []) if item[0] not in used_auds]
            txts = [t for t in txt_by_base.get(base, []) if t not in used_txts]
            script_file = txts[0] if txts else None

            if script_file and aud_list:
                aud_a, aud_b = None, None
                sides = {item[1]: item[0] for item in aud_list if item[1]}
                if "A" in sides and "B" in sides:
                    aud_a, aud_b = sides["A"], sides["B"]
                elif "1" in sides and "2" in sides:
                    aud_a, aud_b = sides["1"], sides["2"]
                elif len(aud_list) >= 2:
                    p1_kw = ("主人公", "主角", "男", "彼", "第一", "1", "a")
                    first_idx = 0
                    for idx, (aud_f, side_tag, _) in enumerate(aud_list):
                        if side_tag and any(k in str(side_tag).lower() for k in p1_kw):
                            first_idx = idx
                            break
                    aud_a = aud_list[first_idx][0]
                    other_auds = [aud_list[i][0] for i in range(len(aud_list)) if i != first_idx]
                    aud_b = other_auds[0] if other_auds else None
                elif len(aud_list) == 1:
                    aud_a = aud_list[0][0]

                pairs.append({
                    "name": os.path.splitext(os.path.basename(script_file))[0],
                    "script": script_file,
                    "audio": aud_a,
                    "audio_b": aud_b,
                    "confidence": 100,
                    "speaker_marks": [],
                })
                used_txts.add(script_file)
                if aud_a:
                    used_auds.add(aud_a)
                if aud_b:
                    used_auds.add(aud_b)

        paired_auds = {p["audio"] for p in pairs if p.get("audio")} | {p["audio_b"] for p in pairs if p.get("audio_b")}
        paired_txts = {p["script"] for p in pairs if p.get("script")}
        rem_auds = [a for a in audio_files if a not in paired_auds]
        rem_txts = [t for t in txt_files if t not in paired_txts]

        # 对剩余未配对项尝试双视角模糊匹配
        if rem_txts and rem_auds:
            for txt in list(rem_txts):
                t_stem = os.path.splitext(os.path.basename(txt))[0]
                t_clean = _clean_stem_for_matching(t_stem)
                scores = []
                for aud in rem_auds:
                    a_stem = os.path.splitext(os.path.basename(aud))[0]
                    a_clean = _clean_stem_for_matching(a_stem)
                    score = difflib.SequenceMatcher(None, t_clean, a_clean).ratio()
                    scores.append((score, aud))
                scores.sort(key=lambda x: x[0], reverse=True)
                if scores and scores[0][0] >= 0.40:
                    best_aud = scores[0][1]
                    best_score = scores[0][0]
                    second_aud = None
                    if len(scores) >= 2 and scores[1][0] >= 0.40:
                        second_aud = scores[1][1]
                    pairs.append({
                        "name": t_stem,
                        "script": txt,
                        "audio": best_aud,
                        "audio_b": second_aud,
                        "confidence": int(best_score * 100),
                        "speaker_marks": [],
                    })
                    rem_txts.remove(txt)
                    rem_auds.remove(best_aud)
                    if second_aud:
                        rem_auds.remove(second_aud)

        for aud in rem_auds:
            unmatched.append({"type": "缺台本", "file": aud, "name": os.path.splitext(os.path.basename(aud))[0]})
        for txt in rem_txts:
            unmatched.append({"type": "缺音频", "file": txt, "name": os.path.splitext(os.path.basename(txt))[0]})

    else:
        # ── 单人顺序 / 多人分角色常规一对一配对 ──
        txt_by_stem = {}
        for f in txt_files:
            stem = os.path.splitext(os.path.basename(f))[0]
            base, marks = _parse_speaker_suffix(stem)
            if base not in txt_by_stem:
                txt_by_stem[base] = {"files": [], "marks": marks}
            txt_by_stem[base]["files"].append(f)

        audio_by_stem = {}
        for f in audio_files:
            stem = os.path.splitext(os.path.basename(f))[0]
            audio_by_stem.setdefault(stem, []).append(f)

        # 1. 严格基名精确匹配
        common_stems = set(txt_by_stem.keys()) & set(audio_by_stem.keys())
        paired_txts = set()
        paired_auds = set()
        for stem in sorted(common_stems):
            info = txt_by_stem[stem]
            for txt in info["files"]:
                for aud in audio_by_stem[stem]:
                    if txt not in paired_txts and aud not in paired_auds:
                        pairs.append({
                            "script": txt,
                            "audio": aud,
                            "audio_b": None,
                            "name": stem,
                            "confidence": 100,
                            "speaker_marks": list(info["marks"]),
                        })
                        paired_txts.add(txt)
                        paired_auds.add(aud)

        rem_txts = [f for f in txt_files if f not in paired_txts]
        rem_auds = [f for f in audio_files if f not in paired_auds]

        # 2. 若双方各只剩 1 个，直接保底配对
        if len(rem_txts) == 1 and len(rem_auds) == 1:
            t = rem_txts[0]
            a = rem_auds[0]
            s_t = os.path.splitext(os.path.basename(t))[0]
            pairs.append({
                "script": t,
                "audio": a,
                "audio_b": None,
                "name": s_t,
                "confidence": 90,
                "speaker_marks": [],
            })
            rem_txts.clear()
            rem_auds.clear()

        # 3. 规约模糊相似度配对
        if rem_txts and rem_auds:
            candidates = []
            for t in rem_txts:
                t_stem = os.path.splitext(os.path.basename(t))[0]
                t_clean = _clean_stem_for_matching(t_stem)
                for a in rem_auds:
                    a_stem = os.path.splitext(os.path.basename(a))[0]
                    a_clean = _clean_stem_for_matching(a_stem)
                    score = difflib.SequenceMatcher(None, t_clean, a_clean).ratio()
                    if score >= 0.45:
                        candidates.append((score, t, a, t_stem))

            candidates.sort(key=lambda x: x[0], reverse=True)
            used_t = set()
            used_a = set()
            for score, t, a, t_stem in candidates:
                if t not in used_t and a not in used_a:
                    pairs.append({
                        "script": t,
                        "audio": a,
                        "audio_b": None,
                        "name": t_stem,
                        "confidence": int(score * 100),
                        "speaker_marks": [],
                    })
                    used_t.add(t)
                    used_a.add(a)

            rem_txts = [t for t in rem_txts if t not in used_t]
            rem_auds = [a for a in rem_auds if a not in used_a]

        for txt in rem_txts:
            unmatched.append({"type": "缺音频", "file": txt, "name": os.path.splitext(os.path.basename(txt))[0]})
        for aud in rem_auds:
            unmatched.append({"type": "缺台本", "file": aud, "name": os.path.splitext(os.path.basename(aud))[0]})

    return pairs, unmatched
