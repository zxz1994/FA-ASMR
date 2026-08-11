# -*- coding: utf-8 -*-
"""对齐器组件 — token 文本工具 / span 打分 / OnoWave 波形计数 / 重复型行检测

从 align.py 拆分。纯工具函数，不依赖 torch 模型。
"""

import dataclasses
import logging
from typing import Any, Dict

SYMBOL_CHARS = set('♥♡☆★♪♫♬♩♭♮†‡※§∞≈≡≠≤≥←→↑↓…⋯·•○●◎◇◆□■△▲▽▼＊※*')

# 行分隔符（调用方用这些 token 标记台本换行）
LINE_SEPARATORS = {'\n', '\\n', '<newline>', 'NEWLINE', '↵', '<br>', '<br/>'}


def _sum_score(spans):
    return sum(s.score for s in spans) if spans else -999.0


def _pick_best_spans(spans_L, spans_R):
    """逐 token 跨声道择优（L/R 取总分高者）。返回 (spans, score)。"""
    if not spans_L and not spans_R:
        return None, -999.0
    if not spans_L:
        return spans_R, _sum_score(spans_R)
    if not spans_R:
        return spans_L, _sum_score(spans_L)
    sl, sr = _sum_score(spans_L), _sum_score(spans_R)
    return (spans_L, sl) if sl >= sr else (spans_R, sr)


def _hybrid_spans(base, bonus):
    """方案 B 混合：每词「起点恒取 base（精确），时长取 bonus（拉长后的宽度）」。

    base  = 无奖励的标准 CTC 对齐（起点准，但 token 易塌缩成 1 帧）
    bonus = 注入自环奖励后的对齐（时长被拉到接近真实，但整体边界前移、漂移大）
    关键修正：bonus 的起点不可信（整体前移），因此
      · base 该 token 有效 → 起点 100% 保留 base，仅用 bonus 的「时长」拉伸每个 span 的 end；
      · base 该 token 缺失([error]) → 保持 [error]，绝不借用 bonus 的起点（否则会把
        bonus 的前移整段带进来，导致该行起点灾难性跳变 + 漂移爆炸）。
    这样起点精度 = base（与 global 零漂移），时长比 = bonus，互不牺牲。
    """
    out = []
    n = len(base)
    for i in range(n):
        bs = base[i]
        if not bs:
            out.append(bs)                     # base 缺 → 仍 [error]，不借 bonus
            continue
        bn = bonus[i] if i < len(bonus) else None
        base_start = bs[0].start
        base_end = bs[-1].end
        base_dur = max(base_end - base_start, 1)
        if bn:
            bonus_dur = max(bn[-1].end - bn[0].start, 1)
            target_end = base_start + bonus_dur        # 只用时长，起点恒为 base
        else:
            target_end = base_end
        # 跨 token 防重叠：end 不得超过下一 token 的 base 起点（否则下游 Refine 会把
        # 起点整体挪动，重新引入漂移）。约束下时长回收受相邻 token 紧密打包限制，
        # 但仍 > base（0.68→~0.77），且零漂移。
        if i + 1 < n and base[i + 1]:
            next_start = base[i + 1][0].start
            if target_end >= next_start:
                target_end = next_start - 1
        target_end = max(target_end, base_start + 1)
        ratio = (target_end - base_start) / base_dur
        a = []
        for s in bs:
            rel = s.end - base_start
            new_end = base_start + round(rel * ratio)
            if new_end <= s.start:
                new_end = s.start + 1         # 防重叠/倒退
            a.append(dataclasses.replace(s, end=new_end))
        out.append(a)
    return out


def _is_symbol_token(tok):
    return bool(tok) and all(c in SYMBOL_CHARS for c in tok)


def _split_lines(text_tokens):
    """支持两种输入：
    1) 扁平列表（无换行标记）→ legacy 全局对齐
    2) 嵌套列表（每行一个子列表）→ 行模式（逐行 assemble token 序列）
    """
    # 嵌套列表：调用方已按行切好（main.py 传的就是这种）
    if text_tokens and isinstance(text_tokens[0], (list, tuple)):
        lines = []
        for raw in text_tokens:
            line = [t for t in raw if t != '']
            if line:
                lines.append(line)
        return lines, True
    # 扁平列表 + 换行标记
    lines, cur, has_sep = [], [], False
    for t in text_tokens:
        if t == '':
            continue
        if t in LINE_SEPARATORS:
            has_sep = True
            if cur:
                lines.append(cur)
                cur = []
        else:
            cur.append(t)
    if cur:
        lines.append(cur)
    return lines, has_sep


def _strip_symbols(line_tokens):
    """剔除纯符号 token，返回 (可对齐 token 列表, [(行内位置, 符号), ...])"""
    align, symbols = [], []
    for i, t in enumerate(line_tokens):
        if _is_symbol_token(t):
            symbols.append((i, t))
        else:
            align.append(t)
    return align, symbols


import re as _re
import numpy as _np
import sys as _sys, json as _json, os as _os

_ONO_TABLE = {}
import sys as _sys, json as _json, os as _os
try:
    if getattr(_sys, 'frozen', False):
        # PyInstaller onedir: --add-data 的文件落在 _internal/ (sys._MEIPASS)
        _base = _sys._MEIPASS
    else:
        _base = _os.path.dirname(_os.path.abspath(__file__))
    _tp = _os.path.join(_base, "ono_table.json")
    with open(_tp, "r", encoding="utf-8") as _tf:
        _ONO_TABLE = _json.load(_tf)
    assert isinstance(_ONO_TABLE, dict) and len(_ONO_TABLE) > 0, f"table empty from {_tp}"
except Exception as _e:
    _ONO_TABLE = {}
    _sys.stderr.write(f"[align] ono_table: {_e}\n")


def _ono_count_energy_peak(waveform, sr, threshold_factor, min_dist_ms):
    """数能量峰：RMS > median × threshold_factor，峰距 ≥ min_dist_ms"""
    win = int(0.02 * sr); hop = win // 2
    nf = (len(waveform) - win) // hop + 1
    if nf < 10: return 0
    energy = _np.zeros(nf)
    for i in range(nf):
        seg = waveform[i * hop : i * hop + win]
        energy[i] = _np.sqrt(_np.mean(seg ** 2))
    thresh = _np.median(energy) * threshold_factor
    min_dist = int(min_dist_ms / 1000 / (hop / sr))
    peaks = []
    for i in range(1, nf - 1):
        if energy[i] > thresh and energy[i] > energy[i - 1] and energy[i] > energy[i + 1]:
            if not peaks or i - peaks[-1] > min_dist:
                peaks.append(i)
    return len(peaks)


def _ono_count_envelope_autocorr(waveform, sr, autocorr_min_ms, autocorr_max_ms):
    """能量包络自相关：时长 ÷ 主周期 = 组数"""
    win = int(0.02 * sr); hop = win // 2
    nf = (len(waveform) - win) // hop + 1
    if nf < 20: return 0
    energy = _np.zeros(nf)
    for i in range(nf):
        seg = waveform[i * hop : i * hop + win]
        energy[i] = _np.sqrt(_np.mean(seg ** 2))
    e_c = energy - _np.mean(energy)
    ac = _np.correlate(e_c, e_c, mode="full")
    ac = ac[len(ac) // 2:]
    mn = max(1, int(autocorr_min_ms / 1000 / (hop / sr)))
    mx = min(len(ac), int(autocorr_max_ms / 1000 / (hop / sr)))
    if mn >= mx: return 0
    peak = int(_np.argmax(ac[mn:mx])) + mn
    period_sec = peak * hop / sr
    if period_sec < 0.05: return 0
    dur = len(waveform) / sr
    return max(1, int(dur / period_sec + 0.5))


def _ono_count_peak_pairs(waveform, sr, rule):
    """数成对峰+静音分组：2尖峰<100ms→1组，峰间>静音阈值→新组"""
    win = int(0.02 * sr); hop = win // 2
    nf = (len(waveform) - win) // hop + 1
    if nf < 10: return 0
    energy = _np.zeros(nf)
    for i in range(nf):
        seg = waveform[i * hop : i * hop + win]
        energy[i] = _np.sqrt(_np.mean(seg ** 2))
    thresh = _np.median(energy) * 2.0
    min_dist = int(rule.get('peak_min_dist_ms', 80) / 1000 / (hop / sr))
    silence = int(rule.get('silence_min_ms', 300) / 1000 / (hop / sr))
    # 找所有峰
    peaks = []
    for i in range(1, nf - 1):
        if energy[i] > thresh and energy[i] > energy[i-1] and energy[i] > energy[i+1]:
            peaks.append(i)
    if len(peaks) < 2: return len(peaks)
    # 合并成对峰（< min_dist）
    groups = []
    i = 0
    while i < len(peaks):
        j = i + 1
        while j < len(peaks) and peaks[j] - peaks[j-1] < min_dist:
            j += 1
        groups.append(peaks[i:j])
        i = j
    # 数有效组（组间距 > silence）
    count = 1
    for k in range(1, len(groups)):
        if groups[k][0] - groups[k-1][-1] > silence:
            count += 1
    return count


def _ono_count_rms_valley(waveform, sr, rule):
    """RMS谷切分：RMS下降到中位×阈值以下=边界"""
    win = int(0.02 * sr); hop = win // 2
    nf = (len(waveform) - win) // hop + 1
    if nf < 10: return 0
    energy = _np.zeros(nf)
    for i in range(nf):
        seg = waveform[i * hop : i * hop + win]
        energy[i] = _np.sqrt(_np.mean(seg ** 2))
    med = _np.median(energy)
    valley_th = med * rule.get('valley_threshold', 0.4)
    min_group = int(rule.get('min_group_ms', 300) / 1000 / (hop / sr))
    # 找谷（RMS低于阈值）
    valleys = [i for i in range(1, nf-1) if energy[i] < valley_th
               and energy[i-1] >= valley_th]
    # 去重：合并<min_group的谷
    groups = 1
    for i in range(1, len(valleys)):
        if valleys[i] - valleys[i-1] > min_group:
            groups += 1
    return max(1, groups + 1)


def _ono_count_rms_segment(waveform, sr, rule):
    """RMS段切分：高段(>中位×high)>min_ms → 低段(<中位×low) → 1组"""
    win = int(0.02 * sr); hop = win // 2
    nf = (len(waveform) - win) // hop + 1
    if nf < 10: return 0
    energy = _np.zeros(nf)
    for i in range(nf):
        seg = waveform[i * hop : i * hop + win]
        energy[i] = _np.sqrt(_np.mean(seg ** 2))
    med = _np.median(energy)
    high_th = med * rule.get('rms_high_factor', 1.5)
    min_high = int(rule.get('min_high_ms', 300) / 1000 / (hop / sr))
    # 找连续高段
    groups = 0
    i = 0
    while i < nf:
        if energy[i] > high_th:
            j = i
            while j < nf and energy[j] > high_th:
                j += 1
            if j - i >= min_high:
                groups += 1
            i = j
        else:
            i += 1
    return max(1, groups)


_SEP = _re.compile(r"[…・、。,.！!？?♡♥\s〜~ー－\-]+")
# 拟声行允许出现的字符：假名 + 长音 + 促音 + 浊点/半浊点
_KANA_OK = _re.compile(
    r"^[\u3040-\u309F\u30A0-\u30FF\uFF66-\uFF9F\u3099\u309A\u309B\u309C]+$")
_KANJI = _re.compile(r"[\u4E00-\u9FFF\u3400-\u4DBF]")


def _is_repeat_line(raw_tokens):
    """判断是否为「重复型拟声行」（基于罗马音 token，不依赖假名文本）。

    返回 (is_repeat, n_tokens, unit_len)。
    判据：
    1) 过滤标点后 token 种类 ≤ 4；
    2) 严格模式：token 序列被长度 1-3 的单元整除（如 kurikuri→ku,ri×2）；
    3) 宽松模式：去掉尾部 1-2 个非重复 token 后判整除
       （如 ha,ha,ha,heart → 去掉 heart → ha×3 整除）。
    """
    punct = {'.', '…', '！', '!', '？', '?', '、', '。', '♡', '♥', '-', 'ー', '~', '〜'}
    tokens = [t for t in raw_tokens if t and t != '_' and t not in punct]
    n = len(tokens)
    if n < 4:
        return False, 0, 0
    from collections import Counter
    cnt = Counter(tokens)
    if len(cnt) > 4:
        return False, 0, 0
    # 严格模式：完全整除
    for unit_len in range(1, min(5, n)):
        if n % unit_len != 0:
            continue
        unit = tokens[:unit_len]
        if all(tokens[j] == unit[j % unit_len] for j in range(unit_len, n)):
            return True, n, unit_len
    # 宽松模式：去掉末尾 1-2 个 token 后判整除
    # 例如 ha,ha,ha,heart → 去 heart 后 ha,ha,ha → unit_len=1 整除
    for trim in (1, 2):
        if n - trim < 4:
            continue
        core = tokens[:n - trim]
        cn = len(core)
        for unit_len in range(1, min(5, cn)):
            if cn % unit_len != 0:
                continue
            unit = core[:unit_len]
            if all(core[j] == unit[j % unit_len] for j in range(unit_len, cn)):
                return True, n, unit_len
    return False, 0, 0
