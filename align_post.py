# -*- coding: utf-8 -*-
"""FA-ASMR V2 — 纯净后处理与时间戳聚合模块

负责：
1. Pre-expansion 跨度回映射与单行起止聚合
2. 选择性长静音吸附（Selective Silence Snapping）
3. 严格单调无重叠时间轴保障
"""
from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict, Any, Optional

FRAME_SEC: float = 0.02


def _get_max_consonant_lead_sec(consonant_str: str) -> float:
    """计算假名音节中辅音声母相对韵母元音的最大生理提前量 (秒)。

    参考语音学文献与生理发音统计：
    - 清擦音族 (/s/, /sh/, /f/, /h/): 80~180ms，上限 220ms
    - 破擦音族 (/ts/, /ch/, /j/, /z/): 80~150ms，上限 180ms
    - 塞音族 (/t/, /k/, /p/, /b/, /d/, /g/): 40~100ms，上限 150ms
    - 鼻音/流音/半元音 (/m/, /n/, /r/, /w/, /y/): 上限 150ms
    """
    if not consonant_str:
        return 0.20
    c_low = consonant_str.lower()
    if c_low.startswith(("sh", "s", "f", "h")):
        return 0.22
    if c_low.startswith(("ts", "ch", "j", "z")):
        return 0.18
    if c_low.startswith(("t", "k", "p", "b", "d", "g")):
        return 0.15
    if c_low.startswith(("m", "n", "r", "w", "y")):
        return 0.15
    # 非典型或多字符辅音组合仍需返回有限的保守上限，避免调用方
    # 在元音起振点回溯时收到 None 并触发 TypeError。
    return 0.15


def _is_isolated_gasp_line(raw_tokens: List[str]) -> bool:
    """判定当前行是否为独立促音惊叹句（如 ......っ!）。

    特征：包含促音 ('* 或 *)，且没有任何元音 (a, i, u, e, o)。
    """
    if not raw_tokens:
        return False
    joined = "".join(raw_tokens).lower()
    if any(v in joined for v in "aiueo"):
        return False
    return all(t in {"'*", "*", "'"} for t in raw_tokens)


def _snap_isolated_gasp_db(
    tok_s: float,
    tok_e: float,
    prev_speech_end: float,
    next_speech_start: float,
    frame_rms_db: Any,
    frame_duration: float = FRAME_SEC,
    min_gasp_sec: float = 0.06,  # 60ms
    max_gasp_sec: float = 0.45,  # 450ms 严格时间上限，防止误判为长杂音/长水声
    lead_margin_sec: float = 0.25,
    trail_margin_sec: float = 0.15,
) -> Tuple[float, float]:
    """对独立促音惊叹句（如 ......っ!）进行短脉冲声学吸附。"""
    rms = np.asarray(frame_rms_db)
    total_frames = len(rms)

    search_st = prev_speech_end + lead_margin_sec
    search_en = next_speech_start - trail_margin_sec
    if search_en <= search_st + min_gasp_sec:
        return tok_s, tok_e

    st_f = max(0, int(round(search_st / frame_duration)))
    en_f = min(total_frames, int(round(search_en / frame_duration)))
    window_rms = rms[st_f:en_f]
    if len(window_rms) < 3:
        return tok_s, tok_e

    local_floor_db = float(np.percentile(window_rms, 25))
    burst_thr_db = max(local_floor_db + 8.0, -60.0)

    min_f = int(round(min_gasp_sec / frame_duration))
    max_f = int(round(max_gasp_sec / frame_duration))

    bursts = []
    in_b = False
    b_st = 0
    for idx, val in enumerate(window_rms):
        if val >= burst_thr_db and not in_b:
            in_b = True
            b_st = idx
        elif val < burst_thr_db and in_b:
            in_b = False
            dur_f = idx - b_st
            if min_f <= dur_f <= max_f:
                p_val = np.max(window_rms[b_st:idx])
                bursts.append((st_f + b_st, st_f + idx, p_val))

    if in_b:
        dur_f = len(window_rms) - b_st
        if min_f <= dur_f <= max_f:
            p_val = np.max(window_rms[b_st:])
            bursts.append((st_f + b_st, st_f + len(window_rms), p_val))

    if bursts:
        bursts.sort(key=lambda x: x[2], reverse=True)
        best_st_f, best_en_f, _ = bursts[0]
        new_s = best_st_f * frame_duration
        new_e = best_en_f * frame_duration
        return new_s, max(new_s + 0.15, new_e)

    # 未检测到符合时长的短爆发，退化为紧贴下一句前 0.6s
    fallback_s = max(search_st, next_speech_start - 0.60)
    fallback_e = fallback_s + 0.20
    return fallback_s, fallback_e


def _build_raw_index_map(kept_indices, n_raw):
    """原始 token 下标 → canonical token 下标。

    被 _canonicalize_line_tokens 过滤掉的 token 继承「前一个」canonical token 的时间；
    位于首个 canonical 之前的 token 继承首个 canonical。保证输出条数 == 输入 token 数，
    调用方按下标对齐时不会整体错位。
    """
    if n_raw <= 0:
        return []
    mapping = [None] * n_raw
    ptr = 0
    last = None
    for j in range(n_raw):
        if ptr < len(kept_indices) and kept_indices[ptr] == j:
            last = ptr
            ptr += 1
        mapping[j] = last
    first = next((m for m in mapping if m is not None), None)
    if first is not None:
        for j in range(n_raw):
            if mapping[j] is not None:
                break
            mapping[j] = first
    return mapping


def _post_process_tokens(
    grouped_spans: List[List[Any]],
    lines: List[Dict[str, Any]],
    frame_duration: float = FRAME_SEC,
    frame_rms_db: Optional[Any] = None,
    silence_threshold_db: Optional[float] = None,
    enable_silence_snap: bool = True,
) -> List[Dict[str, Any]]:
    """将全局 CTC 的 token spans 映射并聚合成逐行结果。

    输出条数严格等于各行原始 token 数之和（lines[i]["n_raw"]），
    调用方按下标与输入 text_tokens 一一对应。
    """
    raw_results = []

    for li, (spans, pl) in enumerate(zip(grouped_spans, lines)):
        nt_orig = len(pl.get("tokens", []))
        if nt_orig == 0:
            continue

        st = float(min(s.start for s in spans)) * frame_duration
        en = float(max(s.end for s in spans)) * frame_duration

        # 行级选择性长静音吸附：若整句台词前导在深静音中且超过 0.5s，吸附至起振点前 120ms
        if enable_silence_snap and frame_rms_db is not None and silence_threshold_db is not None:
            rms = np.asarray(frame_rms_db)
            total_f = len(rms)
            st_f = max(0, int(round(st / frame_duration)))
            en_f = min(total_f, int(round(en / frame_duration)))
            if st_f < total_f and rms[st_f] <= silence_threshold_db:
                active_f = st_f
                while active_f < en_f and rms[active_f] <= silence_threshold_db:
                    active_f += 1
                silence_dur = (active_f - st_f) * frame_duration
                if silence_dur >= 0.5 and active_f < en_f:
                    speech_sec = active_f * frame_duration
                    st = max(st, speech_sec - 0.12)

        n_raw = int(pl.get("n_raw", nt_orig))
        raw_tokens = pl.get("raw_tokens") or pl["tokens"]

        # 独立促音惊叹句（如 ......っ!）短脉冲物理声学吸附：
        # 若整行仅包含促音且无任何元音，CTC 会在静音中向左塌缩至上一句尾。
        # 此时以前后句发音边界为屏障，在静音带中严格搜索 60ms~450ms 的信噪比突跃短脉冲。
        is_gasp = _is_isolated_gasp_line(raw_tokens)

        if is_gasp and frame_rms_db is not None:
            prev_en = float(max(s.end for s in grouped_spans[li - 1])) * frame_duration if li > 0 else 0.0
            next_st = float(min(s.start for s in grouped_spans[li + 1])) * frame_duration if (li + 1 < len(grouped_spans) and len(grouped_spans[li + 1]) > 0) else en
            st, en = _snap_isolated_gasp_db(st, en, prev_en, next_st, frame_rms_db, frame_duration)
        idx_map = _build_raw_index_map(pl.get("orig_idx"), n_raw)
        ii_folds = pl.get("ii_folds") or []
        first_to_second = {f: s for f, s in ii_folds}
        second_to_first = {s: f for f, s in ii_folds}

        ranges = pl.get("token_id_ranges")
        # 逐 token 真实物理时间映射（短句句首 ii 紧耦合平分，多音素 token 跨度聚合，其余保留 CTC 真实发音起止与声学时长）
        for j in range(n_raw):
            tok_score = None
            if is_gasp:
                tok_s = st
                tok_e = en
            else:
                ci = idx_map[j] if j < len(idx_map) else None
                if ci is None:
                    ci = 0
                ci = min(ci, len(spans) - 1)
                if ranges and ci < len(ranges):
                    st_id, en_id = ranges[ci]
                    st_id = min(st_id, len(spans) - 1)
                    en_id = min(max(st_id + 1, en_id), len(spans))
                    tok_s = float(spans[st_id].start) * frame_duration
                    tok_e = float(spans[en_id - 1].end) * frame_duration
                    # 多音素 token 的置信度取跨度内最弱一帧（最不利原则）：
                    # 只要有一个音素塌陷，整 token 不可信。
                    tok_score = float(min(s.score for s in spans[st_id:en_id]))

                    # 方案 1：元音起振点反向锚定法 (Vowel Onset Time Back-Tracking)
                    # 若假名音节包含辅音声母 + 韵母元音，辅音起点严格受限于元音起振点前生理上限 max_lead 秒，
                    # 杜绝清擦音/破擦音/塞音在静音区向左漫延与提前抢跑。
                    c_tok = pl["tokens"][ci] if ci < len(pl.get("tokens", [])) else ""
                    chars = list(c_tok)
                    first_vowel_offset = -1
                    for k, ch in enumerate(chars):
                        if ch in {"a", "i", "u", "e", "o"}:
                            first_vowel_offset = k
                            break
                    if first_vowel_offset > 0 and (st_id + first_vowel_offset) < len(spans):
                        v_span = spans[st_id + first_vowel_offset]
                        vowel_onset = float(v_span.start) * frame_duration
                        c_str = "".join(chars[:first_vowel_offset])
                        max_lead = _get_max_consonant_lead_sec(c_str)
                        earliest_allowed = vowel_onset - max_lead
                        if tok_s < earliest_allowed:
                            tok_s = earliest_allowed
                else:
                    span = spans[ci]
                    tok_s = float(span.start) * frame_duration
                    tok_e = float(span.end) * frame_duration
                    tok_score = float(span.score)
            if tok_s < st:
                tok_s = st

            # 处理每个由标点分割的短句句首 ii 折叠还原：平分物理长音区间
            if j in first_to_second:
                tot_dur = max(tok_e - tok_s, 2 * frame_duration)
                mid = tok_s + tot_dur * 0.5
                tok_e = mid
            elif j in second_to_first:
                tot_dur = max(tok_e - tok_s, 2 * frame_duration)
                mid = tok_s + tot_dur * 0.5
                tok_s = mid
                tok_e = tok_s + tot_dur * 0.5
            else:
                if tok_e < tok_s + frame_duration:
                    tok_e = tok_s + frame_duration

            raw_results.append({
                "token": raw_tokens[j] if j < len(raw_tokens) else "",
                "original_start": tok_s,
                "original_end": tok_e,
                "line_index": li,
                "stage": "global_ctc",
                # CTC 对齐置信度（log-softmax 尺度，越接近 0 越可信）。
                # 供下游「拟声词守门器」做声学证伪；is_gasp 行无独立 span 分数，
                # 用 0.0 占位（gasps 走独立的物理吸附判定，不经守门器）。
                "score": (tok_score if tok_score is not None else 0.0),
            })

    return raw_results
