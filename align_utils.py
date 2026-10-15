# -*- coding: utf-8 -*-
"""FA-ASMR V2 — 基础文本符号清洗与 Token 处理工具"""
from __future__ import annotations

import re
from typing import List, Tuple, Any

_PUNCT_RE = re.compile(r"[\s\.,?!~…\-—、。！？～・♪♡♥☆★\[\]()（）「」『』]")

# ── LRC 时间戳正则（全项目唯一定义）────────────────────────────────────
# 下游有两处解析方：FA-ASMR_GUI._fmt_lrc_time、lrc_end_extend._TS（后者带
# ImportError 兜底，因该文件支持独立 CLI 运行）。两者必须与本定义语义一致，
# 但分散维护时极易漂移——曾出现过一处写成两段式 \[(\d{2}):(\d{2})\]
# 却访问 group(3)，导致结束标记行被静默丢弃（见 README 第十节第 2 条）；
# 也出现过分钟位写死 \d{2}，让 100 分钟以上的音频时间戳全部失配。
#
# 故在此统一定义，两处改为引用。分钟位允许 1~3 位：_format_time 的
# f"{mins:02d}" 不截断，2 小时音频会产出 [120:00:00]。
LRC_TS_ANYWHERE = re.compile(r"\[(\d{1,3}):(\d{2})(?:[.:](\d{1,2}))?\]")
LRC_TS_ANCHORED = re.compile(r"\[(\d{2,3}):(\d{2}):(\d{2})\]")


def _is_repeat_line(tokens: List[str]) -> Tuple[bool, int, int]:
    """检测 token 序列是否为重复拟声结构（如 A-A, A-B-A-B 等）。"""
    cleaned = [_PUNCT_RE.sub("", t) for t in tokens if _PUNCT_RE.sub("", t)]
    n = len(cleaned)
    if n < 2:
        return False, 1, n
    for unit_len in range(1, n // 2 + 1):
        if n % unit_len == 0:
            count = n // unit_len
            unit = cleaned[:unit_len]
            if unit * count == cleaned:
                return True, count, unit_len
    return False, 1, n


def find_subclause_ii_folds(tokens: List[str]) -> List[Tuple[int, int]]:
    """扫描识别该行中每一个由标点分割的短句句首（包括整行开头，以及所有标点后的短句首）连续出现的 ii 连读。

    返回被折叠的配对列表: [(first_idx, second_idx), ...]
    其中 first_idx 为长音首拍，second_idx 为被折叠的长音第二拍。
    """
    n = len(tokens)
    if n < 2:
        return []

    # 1. 判定哪些位置是非标点 token，哪些位置是标点
    # is_subclause_start 记录是否为某个短句的首个非标点音素
    is_subclause_start = [False] * n
    after_punct = True
    for j in range(n):
        cleaned = _PUNCT_RE.sub("", tokens[j])
        if not cleaned:
            after_punct = True
        else:
            if after_punct:
                is_subclause_start[j] = True
                after_punct = False

    folds = []
    # 2. 检查每个短句句首的连续两个非标点 token 是否均为 'i'
    for j in range(n - 1):
        if is_subclause_start[j]:
            tok1 = _PUNCT_RE.sub("", tokens[j]).lower()
            tok2 = _PUNCT_RE.sub("", tokens[j + 1]).lower()
            if tok1 == "i" and tok2 == "i":
                folds.append((j, j + 1))
    return folds


def _canonicalize_line_tokens(
    tokens: List[str], tokenizer: Any, blank: int = 0, fold_subclause_ii: bool = True
) -> Tuple[List[str], List[int], List[int]]:
    """将台本 token 过滤并映射为 MMS-FA 词表 token id 列表。

    当 fold_subclause_ii=True 时，自动对每个标点分割的短句句首出现的连续 ii 连读进行长音折叠，
    仅向 CTC targets 送入单长音 i，从源头杜绝同音素被硬塞 Blank 并向前抛射吸附。

    返回 (canonical_tokens, flat_ids, kept_indices, token_id_spans):
        kept_indices[k] = canonical[k] 在原始 tokens 列表中的下标。

    kept_indices 是「输出条数与输入 token 数保持一致」的关键：被过滤掉的
    token（纯标点/词表外字符）必须靠它回映射到相邻的 canonical token，
    否则调用方按下标对齐时会整体错位（历史上丢过一次）。
    """
    vocab = getattr(tokenizer, "dictionary", None)
    folded_second_indices = set()
    if fold_subclause_ii:
        folds = find_subclause_ii_folds(tokens)
        for _, second_idx in folds:
            folded_second_indices.add(second_idx)

    canonical = []
    flat_ids = []
    kept_indices = []
    token_id_spans = []
    for idx, tok in enumerate(tokens):
        if idx in folded_second_indices:
            # 短句句首 ii 的第二拍被折叠为单长音，不进入 CTC targets
            continue
        cleaned = _PUNCT_RE.sub("", tok)
        if not cleaned:
            continue
        chars = [c for c in cleaned if vocab is None or c in vocab]
        if not chars:
            continue
        ids = [i for i in tokenizer([chars])[0] if i != blank]
        if ids:
            st_id = len(flat_ids)
            canonical.append(tok)
            flat_ids.extend(ids)
            en_id = len(flat_ids)
            kept_indices.append(idx)
            token_id_spans.append((st_id, en_id))
    return canonical, flat_ids, kept_indices, token_id_spans


def _build_head_standalone_n_flag(line_ids, token_id_ranges, vowel_ids,
                                  nasal_token_id):
    """判定单行台词是否为「行首独立鼻音 ん」。

    语义与 C->V 跨度相互独立，因此单独成函数、单独成数组：

    - ``len(line_ids) == 1``：单音素行，天然独立（如独立成句的「ん」）。
    - 否则看该行首个 token 的 label 跨度里有没有元音：``ん`` 只有辅音，
      跨度内无元音 → 独立（如「んれるっ」「んっ、んん」）；``na``/``ne``
      跨度内含元音 → 属正常音节开头，**不算**独立。

    ``line_ids`` 为该行展平后的 label 序列，``token_id_ranges`` 为
    每个原始 token 在其中的 (start, end) 跨度。返回 0/1。
    """
    if not line_ids or int(line_ids[0]) != nasal_token_id:
        return 0
    if len(line_ids) == 1:
        return 1
    if token_id_ranges:
        start, end = token_id_ranges[0]
        vowels = set(vowel_ids)
        if not any(int(line_ids[pos]) in vowels for pos in range(start, end)):
            return 1
    return 0


def _format_time(seconds: float) -> str:
    """格式化秒数为 LRC 时间戳 [MM:SS:CC]（方括号 + 两位百分秒）。

    契约：下游两处解析 —— FA-ASMR_GUI._fmt_lrc_time 用
    r"\\[(\\d{2,3}):(\\d{2}):(\\d{2})\\]"（模块级 LRC_TS_ANCHORED），
    lrc_end_extend 用 LRC_TS_ANYWHERE 的等价字面。
    切勿改回 HH:MM:SS,mmm（SRT 风格）——下游正则会全部失配，LRC 静默丢掉
    所有时间戳（踩过一次）。

    分钟位允许 3 位：``f"{mins:02d}"`` 不截断，mins >= 100 时自然产出 3 位
    （如 2 小时的 ``[120:00:00]``）。旧的下游正则把分钟写死成 ``\\d{2}``，
    对 3 位分钟恒不匹配 -> ``_flush()`` 的 ``if t:`` 判定为假 -> **整条字幕
    退化成无时间戳的裸文本**（100 分钟以上的作品全部受影响）。两处正则已统一
    放开为 ``\\d{2,3}`` / ``\\d{1,3}``。
    """
    total_cs = max(0, int(round(seconds * 100.0)))
    mins = total_cs // 6000
    total_cs %= 6000
    secs = total_cs // 100
    cs = total_cs % 100
    return f"[{mins:02d}:{secs:02d}:{cs:02d}]"
