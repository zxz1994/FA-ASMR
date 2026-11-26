# -*- coding: utf-8 -*-
"""FA-ASMR V2 — 对齐后声学雷达体检模块（纯只读分析，不修改对齐结果）。

核心职责：
  1. 检出低分/被跳过/被异常压扁或拉伸的非拟声实词子句（suspect_clauses）：
     对应「注音读错」「台本多写 / 声优实际未录」或局部错位；
  2. 检出字幕间隙内 CTC 非 blank 音素密集活跃的未覆盖语音岛（unaligned_speech_gaps）：
     对应「台本缺句 / 漏句 / 声优即兴加戏」；
  3. 可选切片导出（export_audit_clips）：仅按需切出异常点前后短 WAV，供多模态 Agent 局部听证。
"""
from __future__ import annotations

import math
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from fa_asmr.text.clause import F

_INLINE_RUBY_RE = re.compile(r"\{([^|{}]+)\|([^|{}]*)\}")
_REPEAT_VOWEL_RE = re.compile(r"([aiueo])\1{2,}")


def _effective_phone_len(toks: Sequence[str]) -> int:
    """统计折叠长音重复元音（如 tooooo -> to）后的有效音素数，避免单字拖音被误判为长实词。"""
    return sum(len(_REPEAT_VOWEL_RE.sub(r"\1", t)) for t in toks)


def _fmt_mmss(sec: float) -> str:
    sec = max(0.0, float(sec))
    m = int(sec // 60)
    s = sec - m * 60
    return f"{m:02d}:{s:05.2f}"


def _merge_intervals(spans: Sequence[Tuple[float, float]], pad: float = 0.0) -> List[Tuple[float, float]]:
    if not spans:
        return []
    pts = sorted((max(0.0, float(a) - pad), float(b) + pad) for a, b in spans if b >= a)
    if not pts:
        return []
    merged: List[Tuple[float, float]] = [pts[0]]
    for st, en in pts[1:]:
        if st <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], en))
        else:
            merged.append((st, en))
    return merged


def _extract_clause_meta_from_alignment(
    alignment_tokens: Sequence[Sequence[str]],
    alignment_results: Sequence[Dict[str, Any]],
    ono_clause_flags: Sequence[int],
    clause_surfaces: Sequence[str],
    clause_speakers: Sequence[Any],
    clause_line_nos: Sequence[int],
    clause_annotated: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """按子句聚合对齐结果（起止时间、CTC 均值分、最差词、字形/注音）。"""
    out: List[Dict[str, Any]] = []
    flat_idx = 0
    n_res = len(alignment_results)
    for ci, toks in enumerate(alignment_tokens):
        n_tok = len(toks)
        cl_res = list(alignment_results[flat_idx: min(n_res, flat_idx + n_tok)])
        flat_idx += n_tok
        is_ono = bool(ci < len(ono_clause_flags) and ono_clause_flags[ci])
        surf = str(clause_surfaces[ci]) if ci < len(clause_surfaces) else ""
        spk_raw = clause_speakers[ci] if ci < len(clause_speakers) else ()
        spk = "、".join(s for s in spk_raw if s) if isinstance(spk_raw, (list, tuple)) else str(spk_raw or "")
        ln_no = int(clause_line_nos[ci]) if ci < len(clause_line_nos) else (ci + 1)
        ann = str(clause_annotated[ci]) if (clause_annotated is not None and ci < len(clause_annotated)) else ""
        n_phones = _effective_phone_len(toks)

        valid_r = [
            (ti, r) for ti, r in enumerate(cl_res)
            if not r.get("skipped")
            and r.get("original_start") is not None
            and r.get("original_end") is not None
        ]
        all_skipped = bool(cl_res) and not valid_r
        if valid_r:
            st = min(float(r["original_start"]) for _, r in valid_r)
            en = max(float(r["original_end"]) for _, r in valid_r)
            dur = max(0.0, en - st)
            scores = [float(r.get("score", -10.0)) for _, r in valid_r]
            mean_sc = float(np.mean(scores))
            worst_ti, worst_r = min(valid_r, key=lambda x: float(x[1].get("score", -10.0)))
            worst_tok = {
                "pron": toks[worst_ti] if worst_ti < len(toks) else "",
                "score": round(float(worst_r.get("score", -10.0)), 2),
                "start": round(float(worst_r["original_start"]), 2),
                "end": round(float(worst_r["original_end"]), 2),
            }
            max_internal_gap = 0.0
            internal_gap_info: Optional[Dict[str, Any]] = None
            if len(valid_r) >= 2:
                ordered_r = sorted(valid_r, key=lambda x: (float(x[1]["original_start"]), float(x[1]["original_end"]), x[0]))
                for (ti0, r0), (ti1, r1) in zip(ordered_r, ordered_r[1:]):
                    g = float(r1["original_start"]) - float(r0["original_end"])
                    if g > max_internal_gap:
                        max_internal_gap = g
                        internal_gap_info = {
                            "gap_sec": round(g, 2),
                            "before_end": round(float(r0["original_end"]), 2),
                            "after_start": round(float(r1["original_start"]), 2),
                            "before_pron": toks[ti0] if ti0 < len(toks) else "",
                            "after_pron": toks[ti1] if ti1 < len(toks) else "",
                        }
            if max_internal_gap < 1.0:
                internal_gap_info = None
        else:
            st, en, dur, mean_sc = None, None, 0.0, -10.0
            worst_tok = None
            max_internal_gap = 0.0
            internal_gap_info = None

        ruby_terms = [
            {"surface": m.group(1), "reading": m.group(2)}
            for m in _INLINE_RUBY_RE.finditer(ann)
        ] if ann and "{" in ann else []

        out.append({
            "clause_index": ci,
            "line": ln_no,
            "speaker": spk,
            "text": surf,
            "is_ono": is_ono,
            "skipped": all_skipped,
            "start": st,
            "end": en,
            "duration": dur,
            "n_tokens": n_tok,
            "n_phones": n_phones,
            "mean_score": mean_sc,
            "worst_token": worst_tok,
            "max_internal_gap": max_internal_gap,
            "internal_gap_info": internal_gap_info,
            "ruby_terms": ruby_terms,
        })
    return out


def _detect_unaligned_speech_gaps(
    clause_metas: Sequence[Dict[str, Any]],
    ctc_nb_active: Optional[np.ndarray],
    frame_sec: float,
    duration_sec: float,
    min_gap_sec: float = 1.20,
    min_active_frames: int = 10,
    min_island_sec: float = 0.80,
) -> List[Dict[str, Any]]:
    """在已对齐字幕的空白间隙中，查找 CTC 非 blank 活跃音素帧密集出现的未覆盖语音岛。"""
    if ctc_nb_active is None or len(ctc_nb_active) == 0 or frame_sec <= 0:
        return []
    nb = np.asarray(ctc_nb_active, dtype=bool)
    n_frames = len(nb)
    tot_dur = max(float(duration_sec), n_frames * frame_sec)

    covered_raw = [
        (float(m["start"]), float(m["end"]))
        for m in clause_metas
        if not m["skipped"] and m["start"] is not None and m["end"] is not None
    ]
    # 对已有字幕两端加 0.25s 缓冲，避免把正常尾音/起音换气误报成缺句
    covered = _merge_intervals(covered_raw, pad=0.25)

    # 构造未覆盖间隙区间
    gaps: List[Tuple[float, float]] = []
    cur = 0.0
    for c_st, c_en in covered:
        if c_st > cur:
            gaps.append((cur, min(c_st, tot_dur)))
        cur = max(cur, c_en)
    if cur < tot_dur:
        gaps.append((cur, tot_dur))

    # 预按时间排序有效子句，便于为每个缺句空档挂出前后文锚点
    dated = sorted(
        (m for m in clause_metas if not m["skipped"] and m["start"] is not None and m["end"] is not None),
        key=lambda x: (float(x["start"]), float(x["end"])),
    )

    bridge_f = max(1, int(round(0.45 / frame_sec)))
    findings: List[Dict[str, Any]] = []

    for g_st, g_en in gaps:
        if (g_en - g_st) < min_gap_sec:
            continue
        f0 = max(0, int(round(g_st / frame_sec)))
        f1 = min(n_frames, int(round(g_en / frame_sec)))
        if f1 <= f0:
            continue
        active_idxs = np.flatnonzero(nb[f0:f1]) + f0
        if len(active_idxs) < min_active_frames:
            continue

        # 把空档内的活跃音素尖峰按 <= 0.45s 聚类成连续语音岛（去掉零星单帧噪声）
        islands: List[Tuple[int, int, int]] = []
        isl_s = int(active_idxs[0])
        isl_e = int(active_idxs[0])
        isl_cnt = 1
        for k_f in active_idxs[1:]:
            kf = int(k_f)
            if kf - isl_e <= bridge_f:
                isl_e = kf
                isl_cnt += 1
            else:
                islands.append((isl_s, isl_e + 1, isl_cnt))
                isl_s = kf
                isl_e = kf
                isl_cnt = 1
        islands.append((isl_s, isl_e + 1, isl_cnt))

        for i_f0, i_f1, i_cnt in islands:
            i_st = i_f0 * frame_sec
            i_en = i_f1 * frame_sec
            i_dur = i_en - i_st
            if i_cnt < min_active_frames or i_dur < min_island_sec:
                continue
            # 找前后紧邻台本行作为定位锚点
            prev_m = None
            next_m = None
            for dm in dated:
                if float(dm["end"]) <= i_st + 0.30:
                    prev_m = dm
                elif next_m is None and float(dm["start"]) >= i_en - 0.30:
                    next_m = dm
                    break
            findings.append({
                "start": round(i_st, 2),
                "end": round(i_en, 2),
                "time_range": f"{_fmt_mmss(i_st)}-{_fmt_mmss(i_en)}",
                "duration": round(i_dur, 2),
                "active_frames": int(i_cnt),
                "active_ratio": round(i_cnt / float(max(1, i_f1 - i_f0)), 3),
                "prev_line": (
                    {"line": prev_m["line"], "speaker": prev_m["speaker"], "text": prev_m["text"], "end": round(float(prev_m["end"]), 2)}
                    if prev_m is not None else None
                ),
                "next_line": (
                    {"line": next_m["line"], "speaker": next_m["speaker"], "text": next_m["text"], "start": round(float(next_m["start"]), 2)}
                    if next_m is not None else None
                ),
            })

    findings.sort(key=lambda x: x["start"])
    return findings


def _evaluate_clause_reasons(
    m: Dict[str, Any],
    eff_low_thr: float,
    short_low_thr: float,
) -> List[str]:
    """判定单句是否存在声学异常（对 4~5 音素极短语气词启用更严格阈值以压低误报）。"""
    reasons: List[str] = []
    if m["skipped"]:
        reasons.append("skipped_semantic")
        return reasons
    n_ph = int(m["n_phones"])
    dur = float(m["duration"])
    sc = float(m["mean_score"])
    if n_ph >= 6 and sc < eff_low_thr:
        reasons.append("low_score")
    elif 4 <= n_ph <= 5 and sc < short_low_thr:
        reasons.append("low_score")
    if n_ph >= 5 and dur < max(0.18, 0.032 * n_ph) and sc < -2.40:
        reasons.append("compressed")
    if n_ph >= 4 and dur > max(4.50, 0.55 * n_ph) and sc < -2.50:
        reasons.append("stretched")
    if n_ph >= 4 and float(m.get("max_internal_gap", 0.0)) >= 2.50:
        reasons.append("internal_gap_stretch")
    return reasons


def _detect_cross_speaker_engulfs(
    clause_metas: Sequence[Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    """多人轨专用：检测因首/尾未录词前吸或后吸导致单句畸长并完整包夹他方台词的异常。

    仅在满足「完整包夹另一角色 >=1 句非拟声台词」且「句内存在 >=1.2s 断层或单音素均时长异常拉长」时触发，
    避免误报正常双人同步叠声。
    """
    valid = [
        m for m in clause_metas
        if not m["is_ono"]
        and not m["skipped"]
        and m["start"] is not None
        and m["end"] is not None
        and m.get("speaker")
        and "、" not in str(m["speaker"])
    ]
    speakers = {str(m["speaker"]) for m in valid}
    if len(speakers) < 2:
        return {}

    engulfs: Dict[int, Dict[str, Any]] = {}
    for m in valid:
        st = float(m["start"])
        en = float(m["end"])
        dur = float(m["duration"])
        n_ph = max(1, int(m["n_phones"]))
        max_gap = float(m.get("max_internal_gap", 0.0))
        if max_gap < 1.20 and not (dur >= 5.0 and (dur / float(n_ph)) >= 0.28):
            continue
        spk = str(m["speaker"])
        hit_lines: List[int] = []
        hit_spks: List[str] = []
        for om in valid:
            if str(om["speaker"]) == spk:
                continue
            ost = float(om["start"])
            oen = float(om["end"])
            if float(om["duration"]) < 0.25:
                continue
            if ost >= st - 0.05 and oen <= en + 0.05:
                hit_lines.append(int(om["line"]))
                if str(om["speaker"]) not in hit_spks:
                    hit_spks.append(str(om["speaker"]))
        if hit_lines:
            engulfs[int(m["clause_index"])] = {
                "engulfed_count": len(hit_lines),
                "engulfed_lines": sorted(set(hit_lines)),
                "engulfed_speakers": hit_spks,
                "max_internal_gap": round(max_gap, 2),
                "sec_per_phone": round(dur / float(n_ph), 3),
            }
    return engulfs


def _stereo_coherence_stats(
    wav_stereo: np.ndarray,
    st_s: float,
    en_s: float,
    sr: int = 16000,
) -> Tuple[float, float, Optional[float], float]:
    """在 180..1800Hz 频带计算区间局部 STFT 双耳相干度 (MSC)、线性消去残差 (cres) 与 ILD。"""
    from scipy import signal as _sig

    n_fft = 1024
    hop = 256
    hop_sec = float(hop) / float(sr)
    n_samples = int(wav_stereo.shape[1])
    dur_tot = float(n_samples) / float(sr)
    t_frames = (n_samples // hop) + 1

    dur = max(0.0, float(en_s) - float(st_s))
    if dur < 0.24:
        mid = 0.5 * (float(st_s) + float(en_s))
        w_st = max(0.0, mid - 0.12)
        w_en = min(dur_tot, mid + 0.12)
    else:
        w_st = max(0.0, float(st_s) - 0.04)
        w_en = min(dur_tot, float(en_s) + 0.04)
    t0 = max(0, int(round(w_st / hop_sec)))
    t1 = min(t_frames, int(round(w_en / hop_sec)))
    if t1 - t0 < 8:
        return 1.0, 1.0, None, 0.0

    half_fft = n_fft // 2
    s_lo = t0 * hop - half_fft
    s_hi = (t1 - 1) * hop + half_fft
    pad_l = max(0, -s_lo)
    pad_r = max(0, s_hi - n_samples)
    seg = wav_stereo[:, max(0, s_lo): min(n_samples, s_hi)]
    if pad_l > 0 or pad_r > 0:
        seg = np.pad(seg, ((0, 0), (pad_l, pad_r)), mode="constant")
    if seg.shape[1] < n_fft:
        return 1.0, 1.0, None, 0.0

    freqs, _, zl = _sig.stft(
        seg[0], fs=sr, nperseg=n_fft, noverlap=n_fft - hop,
        window="hann", boundary=None, padded=False,
    )
    _, _, zr = _sig.stft(
        seg[1], fs=sr, nperseg=n_fft, noverlap=n_fft - hop,
        window="hann", boundary=None, padded=False,
    )
    sb_mask = (freqs >= 180.0) & (freqs <= 1800.0)
    fl_w = zl[sb_mask]
    fr_w = zr[sb_mask]
    fe_w = np.sum((np.abs(fl_w) ** 2) + (np.abs(fr_w) ** 2) + 1e-24, axis=0)
    if fe_w.size < 8:
        return 1.0, 1.0, None, 0.0

    e_floor = max(1e-24, 0.08 * float(np.percentile(fe_w, 90)))
    T_w = fl_w.shape[1]
    win_f = 16
    step_f = 4
    cres_list: List[float] = []
    msc_list: List[float] = []
    ild_list: List[float] = []
    for k0 in range(0, max(1, T_w - win_f + 1), step_f):
        k1 = min(T_w, k0 + win_f)
        if k1 - k0 < 8 or float(np.mean(fe_w[k0:k1])) < e_floor:
            continue
        xl, xr = fl_w[:, k0:k1], fr_w[:, k0:k1]
        rll = np.mean(np.abs(xl) ** 2, axis=1)
        rrr = np.mean(np.abs(xr) ** 2, axis=1)
        rlr = np.mean(xl * np.conj(xr), axis=1)
        sum_l = float(np.sum(rll))
        sum_r = float(np.sum(rrr))
        abs_ild_db = abs(10.0 * float(np.log10(max(1e-24, sum_l) / max(1e-24, sum_r))))
        tr = rll + rrr + 1e-24
        top_f = tr >= np.percentile(tr, 40)
        msc = float(
            np.sum((np.abs(rlr[top_f]) ** 2) / np.maximum(1e-24, rll[top_f] * rrr[top_f]) * tr[top_f])
            / float(np.sum(tr[top_f]))
        )
        res_l = np.mean(np.abs(xl - (rlr / np.maximum(1e-24, rrr))[:, None] * xr) ** 2, axis=1)
        res_r = np.mean(np.abs(xr - (np.conj(rlr) / np.maximum(1e-24, rll))[:, None] * xl) ** 2, axis=1)
        cres = float(min(np.sum(res_l) / max(1e-24, sum_l), np.sum(res_r) / max(1e-24, sum_r)))
        cres_list.append(cres)
        msc_list.append(msc)
        ild_list.append(abs_ild_db)
    if not msc_list:
        return 1.0, 1.0, None, 0.0
    return (
        float(np.max(cres_list)),
        float(np.percentile(cres_list, 85)),
        float(np.percentile(msc_list, 15)),
        float(np.percentile(ild_list, 25)),
    )


def _detect_solo_speaker_overlaps(
    clause_metas: Sequence[Dict[str, Any]],
    stereo_wav_16k: Optional[np.ndarray],
    sr: int = 16000,
) -> Dict[int, Dict[str, Any]]:
    """立体声多人轨专用：检出「声学上为单人独白（高 MSC + 低 cres），却出现跨角色非拟声实词重叠」的异常。

    设计原则（经实测标定）：
      - 严格依赖立体声双耳相干度 (MSC) 与线性消去残差 (cres)，**不走 ILD>=12dB 快速通道**，避免高潮单耳贴麦真叠声误报；
      - 单声道音频自动跳过；仅检查双方均非拟声、重叠 >=0.50s（或占较短句 >=50% 且 >=0.25s）；
      - 分两档：
        * strict_solo=True（极纯净单声源，如 msc15>=0.88 且 cres85<=0.08）-> priority='high' 进 priority_review；
        * strict_solo=False（中等边界区，如轻声居中叠声）-> 记入 suspect_clauses (priority='normal') 留档。
    """
    if stereo_wav_16k is None:
        return {}
    wav = np.asarray(stereo_wav_16k, dtype=np.float32)
    if wav.ndim != 2 or wav.shape[0] < 2 or wav.shape[1] < sr:
        return {}
    wav = wav[:2]

    valid = [
        m for m in clause_metas
        if not m["is_ono"]
        and not m["skipped"]
        and m["start"] is not None
        and m["end"] is not None
        and m.get("speaker")
        and "、" not in str(m["speaker"])
    ]
    if len({str(m["speaker"]) for m in valid}) < 2:
        return {}
    valid.sort(key=lambda x: (float(x["start"]), float(x["end"])))

    cand_pairs: List[Tuple[float, float, float, Dict[str, Any], Dict[str, Any]]] = []
    for i, m1 in enumerate(valid):
        s1, e1 = float(m1["start"]), float(m1["end"])
        spk1 = str(m1["speaker"])
        for j in range(i + 1, len(valid)):
            m2 = valid[j]
            s2, e2 = float(m2["start"]), float(m2["end"])
            if s2 >= e1:
                break
            if str(m2["speaker"]) == spk1:
                continue
            ov_st = max(s1, s2)
            ov_en = min(e1, e2)
            ov = ov_en - ov_st
            short_dur = max(0.05, min(e1 - s1, e2 - s2))
            if ov >= 0.50 or (ov >= 0.25 and ov >= 0.50 * short_dur):
                cand_pairs.append((ov_st, ov_en, ov, m1, m2))
    if not cand_pairs:
        return {}

    # 自适应标定本轨双耳相干基线（排除单侧极强贴麦段）
    ref_spans = [(float(m["start"]), float(m["end"])) for m in valid if float(m["duration"]) >= 0.40]
    step_ref = max(1, len(ref_spans) // 40 or 1)
    cres_vals: List[float] = []
    msc_vals: List[float] = []
    for s0, s1 in ref_spans[::step_ref]:
        _, c_p85, m_p15, ild_p25 = _stereo_coherence_stats(wav, s0, s1, sr=sr)
        if m_p15 is not None and ild_p25 < 12.0:
            cres_vals.append(c_p85)
            msc_vals.append(m_p15)
    base_cres = min(0.16, float(np.percentile(cres_vals, 30))) if cres_vals else 0.10
    base_msc = max(0.70, float(np.percentile(msc_vals, 70))) if msc_vals else 0.75

    overlaps: Dict[int, Dict[str, Any]] = {}
    for ov_st, ov_en, ov, m1, m2 in cand_pairs:
        c_max, c_p85, m_p15, ild_p25 = _stereo_coherence_stats(wav, ov_st, ov_en, sr=sr)
        if m_p15 is None:
            continue
        is_mild_solo = (
            m_p15 >= max(0.66, 0.85 * base_msc)
            and c_p85 <= max(0.18, 1.35 * base_cres)
            and c_max <= max(0.28, 1.75 * base_cres)
        )
        if not is_mild_solo:
            continue
        is_strict_solo = (
            m_p15 >= max(0.88, 0.94 * base_msc)
            and c_p85 <= min(0.08, max(0.05, 1.10 * base_cres))
            and c_max <= max(0.14, 1.50 * base_cres)
        )
        # 将嫌疑归属到更可疑的一方（优先存在句内大断层者，否则归给均分更低者）
        g1 = float(m1.get("max_internal_gap", 0.0))
        g2 = float(m2.get("max_internal_gap", 0.0))
        if (g1 >= 1.20) != (g2 >= 1.20):
            suspect_m, peer_m = (m1, m2) if g1 >= 1.20 else (m2, m1)
        elif float(m1["mean_score"]) <= float(m2["mean_score"]):
            suspect_m, peer_m = m1, m2
        else:
            suspect_m, peer_m = m2, m1

        ci = int(suspect_m["clause_index"])
        info = {
            "strict_solo": bool(is_strict_solo),
            "overlap_start": round(ov_st, 2),
            "overlap_end": round(ov_en, 2),
            "overlap_sec": round(ov, 2),
            "msc15": round(m_p15, 3),
            "cres85": round(c_p85, 3),
            "ild25_db": round(ild_p25, 1),
            "peer_clause_index": int(peer_m["clause_index"]),
            "peer_line": int(peer_m["line"]),
            "peer_speaker": str(peer_m["speaker"]),
            "peer_text": str(peer_m["text"]),
        }
        prev = overlaps.get(ci)
        if prev is None or (info["strict_solo"] and not prev["strict_solo"]) or (
            info["strict_solo"] == prev["strict_solo"] and info["overlap_sec"] > prev["overlap_sec"]
        ):
            overlaps[ci] = info
    return overlaps


def _detect_isolated_speaker_jumps(
    clause_metas: Sequence[Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    """多人轨专用：检测孤立插队到另一角色连续独白区间里的极短句（常见于短称呼词前吸错位）。

    两类高置信度形态（均跳过正常持续双耳叠声与正常短交替）：
      - 形态 A（跨角色重叠孤立跳）：短句与另一角色对齐句重叠 >0.05s，且距己方前后句一侧 >=20s、另一侧 >=4s；
      - 形态 B（孤立楔入对侧独白）：短句紧贴另一角色对齐句（<=4s），且距己方前后句一侧 >=25s、另一侧 >=8s。
    """
    valid = [
        m for m in clause_metas
        if not m["is_ono"]
        and not m["skipped"]
        and m["start"] is not None
        and m["end"] is not None
        and m.get("speaker")
        and "、" not in str(m["speaker"])
    ]
    by_spk: Dict[str, List[Dict[str, Any]]] = {}
    for m in valid:
        by_spk.setdefault(str(m["speaker"]), []).append(m)
    if len(by_spk) < 2:
        return {}
    for arr in by_spk.values():
        arr.sort(key=lambda x: (float(x["start"]), float(x["end"])))

    jumps: Dict[int, Dict[str, Any]] = {}
    for spk, arr in by_spk.items():
        if len(arr) < 2:
            continue
        others = [m for k, v in by_spk.items() if k != spk for m in v]
        for idx, m in enumerate(arr):
            dur = float(m["duration"])
            n_ph = int(m["n_phones"])
            if dur > 1.60 or n_ph > 14:
                continue
            st = float(m["start"])
            en = float(m["end"])
            has_prev = idx > 0
            has_next = (idx + 1) < len(arr)
            gap_prev = (st - float(arr[idx - 1]["end"])) if has_prev else None
            gap_next = (float(arr[idx + 1]["start"]) - en) if has_next else None

            overlaps_other = False
            near_other = False
            for om in others:
                ost = float(om["start"])
                oen = float(om["end"])
                if min(en, oen) - max(st, ost) > 0.05:
                    overlaps_other = True
                    near_other = True
                    break
                if min(abs(st - oen), abs(ost - en)) <= 4.0:
                    near_other = True

            if has_prev and has_next:
                max_gap = max(gap_prev, gap_next)
                min_gap = min(gap_prev, gap_next)
                is_case_a = overlaps_other and max_gap >= 20.0 and min_gap >= 4.0
                is_case_b = near_other and max_gap >= 25.0 and min_gap >= 8.0
            else:
                # 轨首/轨尾仅单侧有己方邻句，绝不把缺失侧当 inf；仅在跨角色重叠且单侧远距 >=25s 时判定前吸/后吸
                single_gap = gap_next if gap_next is not None else gap_prev
                is_case_a = bool(overlaps_other and single_gap is not None and single_gap >= 25.0)
                is_case_b = False

            if is_case_a or is_case_b:
                jumps[int(m["clause_index"])] = {
                    "gap_prev_sec": (round(gap_prev, 2) if gap_prev is not None else None),
                    "gap_next_sec": (round(gap_next, 2) if gap_next is not None else None),
                    "overlaps_other_speaker": overlaps_other,
                }
    return jumps


def _is_priority_suspect(entry: Dict[str, Any]) -> bool:
    """判定该可疑实词句是否属于高危必审项（供多模态精准听审，过滤轻度气音/促音噪音）。"""
    reasons = entry.get("reasons", ())
    sc = float(entry.get("mean_score", 0.0))
    if "skipped_semantic" in reasons:
        return True
    if "isolated_speaker_jump" in reasons:
        return True
    if "internal_gap_stretch" in reasons or "cross_speaker_engulf" in reasons:
        return True
    if "solo_speaker_overlap" in reasons and bool((entry.get("overlap_info") or {}).get("strict_solo")):
        return True
    if "compressed" in reasons and sc < -5.5:
        return True
    if "low_score" in reasons and sc < -5.8:
        return True
    return False


def _fmt_context_clause(m: Dict[str, Any]) -> str:
    ln = m.get("line")
    spk = str(m.get("speaker") or "")
    txt = str(m.get("text") or "").strip()
    st, en = m.get("start"), m.get("end")
    t_part = f"({float(st):.1f}-{float(en):.1f}s)" if (st is not None and en is not None) else ""
    s_part = f"[{spk}]" if spk else ""
    return f"L{ln}{t_part}{s_part}: {txt}"


def _neighbor_context_for_suspect(
    clause_index: Any,
    metas: Sequence[Dict[str, Any]],
    ci_to_pos: Dict[int, int],
    window: int = 2,
) -> Tuple[List[str], List[str]]:
    if clause_index is None or int(clause_index) not in ci_to_pos:
        return [], []
    pos = ci_to_pos[int(clause_index)]
    before = [_fmt_context_clause(metas[k]) for k in range(max(0, pos - window), pos)]
    after = [_fmt_context_clause(metas[k]) for k in range(pos + 1, min(len(metas), pos + 1 + window))]
    return before, after


def _neighbor_context_for_gap(
    g_st: Any,
    g_en: Any,
    aligned_metas: Sequence[Dict[str, Any]],
    window: int = 2,
) -> Tuple[List[str], List[str]]:
    if g_st is None or g_en is None or not aligned_metas:
        return [], []
    st_f, en_f = float(g_st), float(g_en)
    before_m = [m for m in aligned_metas if float(m["end"]) <= st_f + 0.5][-window:]
    after_m = [m for m in aligned_metas if float(m["start"]) >= en_f - 0.5][:window]
    return [_fmt_context_clause(m) for m in before_m], [_fmt_context_clause(m) for m in after_m]


def _build_priority_review(
    suspects: Sequence[Dict[str, Any]],
    gaps: Sequence[Dict[str, Any]],
    clause_metas: Optional[Sequence[Dict[str, Any]]] = None,
    metas_by_side: Optional[Dict[str, Sequence[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    """汇总全部缺句空档与高危可疑句（附前后各 2 句邻句上下文），供多模态 Agent 直接定向听审。"""
    def _prep(seq: Optional[Sequence[Dict[str, Any]]]) -> Tuple[List[Dict[str, Any]], Dict[int, int], List[Dict[str, Any]]]:
        non_empty = [m for m in (seq or ()) if str(m.get("text", "")).strip()]
        ci_map = {int(m["clause_index"]): i for i, m in enumerate(non_empty) if m.get("clause_index") is not None}
        aligned = sorted(
            [m for m in non_empty if not m.get("skipped") and m.get("start") is not None and m.get("end") is not None],
            key=lambda x: (float(x["start"]), float(x["end"])),
        )
        return non_empty, ci_map, aligned

    side_cache: Dict[str, Tuple[List[Dict[str, Any]], Dict[int, int], List[Dict[str, Any]]]] = {}
    if metas_by_side:
        for k, v in metas_by_side.items():
            side_cache[k] = _prep(v)
    default_pack = _prep(clause_metas)

    def _pick_pack(persp: Optional[str]) -> Tuple[List[Dict[str, Any]], Dict[int, int], List[Dict[str, Any]]]:
        if persp and persp in side_cache:
            return side_cache[persp]
        if "A" in side_cache:
            return side_cache["A"]
        return default_pack

    review: List[Dict[str, Any]] = []
    for g in gaps:
        item = {
            "kind": "gap",
            "start": g.get("start"),
            "end": g.get("end"),
            "time_range": g.get("time_range"),
            "duration": g.get("duration"),
            "reasons": ["unaligned_speech_gap"],
            "prev_line": g.get("prev_line"),
            "next_line": g.get("next_line"),
        }
        if g.get("perspective") is not None:
            item["perspective"] = g["perspective"]
        _, _, aligned = _pick_pack(g.get("perspective"))
        cb, ca = _neighbor_context_for_gap(g.get("start"), g.get("end"), aligned)
        if cb:
            item["context_before"] = cb
        if ca:
            item["context_after"] = ca
        review.append(item)

    for s in suspects:
        if s.get("priority") != "high":
            continue
        item = {
            "kind": "suspect",
            "clause_index": s.get("clause_index"),
            "line": s.get("line"),
            "speaker": s.get("speaker"),
            "text": s.get("text"),
            "start": s.get("start"),
            "end": s.get("end"),
            "time_range": s.get("time_range"),
            "mean_score": s.get("mean_score"),
            "reasons": list(s.get("reasons", ())),
        }
        if s.get("perspective") is not None:
            item["perspective"] = s["perspective"]
        if s.get("jump_info") is not None:
            item["jump_info"] = s["jump_info"]
        if s.get("internal_gap_info") is not None:
            item["internal_gap_info"] = s["internal_gap_info"]
        if s.get("engulf_info") is not None:
            item["engulf_info"] = s["engulf_info"]
        if s.get("overlap_info") is not None:
            item["overlap_info"] = s["overlap_info"]
        non_empty, ci_map, _ = _pick_pack(s.get("perspective"))
        cb, ca = _neighbor_context_for_suspect(s.get("clause_index"), non_empty, ci_map)
        if cb:
            item["context_before"] = cb
        if ca:
            item["context_after"] = ca
        review.append(item)
    return review


def _make_suspect_entry(
    m: Dict[str, Any],
    reasons: List[str],
    perspective: Optional[str] = None,
    jump_info: Optional[Dict[str, Any]] = None,
    engulf_info: Optional[Dict[str, Any]] = None,
    overlap_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    st = round(float(m["start"]), 2) if m["start"] is not None else None
    en = round(float(m["end"]), 2) if m["end"] is not None else None
    entry: Dict[str, Any] = {
        "clause_index": m["clause_index"],
        "line": m["line"],
        "speaker": m["speaker"],
        "text": m["text"],
        "start": st,
        "end": en,
        "time_range": (f"{_fmt_mmss(st)}-{_fmt_mmss(en)}" if st is not None and en is not None else None),
        "duration": round(float(m["duration"]), 2),
        "n_phones": m["n_phones"],
        "mean_score": round(float(m["mean_score"]), 2),
        "reasons": reasons,
        "worst_token": m["worst_token"],
        "ruby_terms": m["ruby_terms"],
    }
    if perspective is not None:
        entry["perspective"] = perspective
    if jump_info is not None:
        entry["jump_info"] = jump_info
    if m.get("internal_gap_info") is not None:
        entry["internal_gap_info"] = m["internal_gap_info"]
    if engulf_info is not None:
        entry["engulf_info"] = engulf_info
    if overlap_info is not None:
        entry["overlap_info"] = overlap_info
    entry["priority"] = "high" if _is_priority_suspect(entry) else "normal"
    return entry


def build_alignment_audit(
    alignment_tokens: Sequence[Sequence[str]],
    alignment_results: Sequence[Dict[str, Any]],
    ono_clause_flags: Sequence[int],
    clause_surfaces: Sequence[str],
    clause_speakers: Sequence[Any],
    clause_line_nos: Sequence[int],
    clause_annotated: Optional[Sequence[str]] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
    low_score_threshold: float = -3.20,
) -> Dict[str, Any]:
    """生成对齐后声学体检报告（低分可疑句/注音错/未录句 + 未覆盖活跃语音空档）。"""
    metas = _extract_clause_meta_from_alignment(
        alignment_tokens=alignment_tokens,
        alignment_results=alignment_results,
        ono_clause_flags=ono_clause_flags,
        clause_surfaces=clause_surfaces,
        clause_speakers=clause_speakers,
        clause_line_nos=clause_line_nos,
        clause_annotated=clause_annotated,
    )

    sem_scores = [
        m["mean_score"] for m in metas
        if not m["is_ono"] and not m["skipped"] and m["n_phones"] >= 4 and m["start"] is not None
    ]
    med_score = float(np.median(sem_scores)) if sem_scores else -1.50
    eff_low_thr = min(float(low_score_threshold), med_score - 1.80)
    short_low_thr = min(-4.80, eff_low_thr - 1.40)

    diag = diagnostics or {}
    jumps = _detect_isolated_speaker_jumps(metas)
    engulfs = _detect_cross_speaker_engulfs(metas)
    solo_overlaps = _detect_solo_speaker_overlaps(metas, diag.get("stereo_wav_16k"))

    suspects: List[Dict[str, Any]] = []
    for m in metas:
        if m["is_ono"]:
            continue
        reasons = _evaluate_clause_reasons(m, eff_low_thr, short_low_thr)
        jinfo = jumps.get(int(m["clause_index"]))
        if jinfo is not None:
            reasons.append("isolated_speaker_jump")
        einfo = engulfs.get(int(m["clause_index"]))
        if einfo is not None:
            reasons.append("cross_speaker_engulf")
        oinfo = solo_overlaps.get(int(m["clause_index"]))
        if oinfo is not None:
            reasons.append("solo_speaker_overlap")
        if reasons:
            suspects.append(_make_suspect_entry(m, reasons, jump_info=jinfo, engulf_info=einfo, overlap_info=oinfo))

    suspects.sort(
        key=lambda x: (
            0 if "skipped_semantic" in x["reasons"] else (
                1 if any(r in x["reasons"] for r in ("isolated_speaker_jump", "internal_gap_stretch", "cross_speaker_engulf", "solo_speaker_overlap")) else 2
            ),
            0 if x.get("priority") == "high" else 1,
            x["mean_score"],
            x["clause_index"],
        )
    )

    frame_sec = float(diag.get("frame_sec", 0.02))
    dur_sec = float(diag.get("duration_sec", 0.0))
    ctc_nb = diag.get("ctc_nb_active")
    gaps = _detect_unaligned_speech_gaps(
        clause_metas=metas,
        ctc_nb_active=ctc_nb,
        frame_sec=frame_sec,
        duration_sec=dur_sec,
    )
    priority_review = _build_priority_review(suspects, gaps, clause_metas=metas)

    return {
        "track_median_score": round(med_score, 2),
        "effective_low_score_threshold": round(eff_low_thr, 2),
        "semantic_clauses_total": sum(1 for m in metas if not m["is_ono"]),
        "priority_review_count": len(priority_review),
        "suspect_clauses_count": len(suspects),
        "unaligned_speech_gaps_count": len(gaps),
        "priority_review": priority_review,
        "suspect_clauses": suspects,
        "unaligned_speech_gaps": gaps,
    }


def _safe_finite_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        num = float(val)
    except (TypeError, ValueError, OverflowError):
        return None
    return num if math.isfinite(num) else None


def _peer_shift(start: float, end: float, side: str,
                sim_segs: Optional[Sequence[Dict[str, Any]]],
                delta_sec: Optional[float]) -> Optional[float]:
    """Map a whole interval within one synchronized scene (bijective on both sides). dt_AB is A minus B."""
    sign = -1.0 if side == "A" else 1.0
    if sim_segs is None:
        # Older callers may supply an explicit offset for a known fixed-clock pair.
        d = _safe_finite_float(delta_sec)
        return None if d is None else sign * d
    peer = "B" if side == "A" else "A"
    shifts = set()
    for seg in sim_segs:
        left = _safe_finite_float(seg.get(f"{side}_st"))
        right = _safe_finite_float(seg.get(f"{side}_en"))
        if left is None or right is None:
            # 源侧边界本身缺失时无法判断该段是否与 [start, end] 相交，跳过继续检查其余段
            continue
        if end <= left or start >= right:
            continue
        # 仅对与源区间相交的同步段严格校验完整元数据
        dt_v = _safe_finite_float(seg.get("dt_AB"))
        p_left = _safe_finite_float(seg.get(f"{peer}_st"))
        p_right = _safe_finite_float(seg.get(f"{peer}_en"))
        if dt_v is None or p_left is None or p_right is None:
            return None
        shift = sign * dt_v
        if not (left <= start and end <= right and p_left <= start + shift and end + shift <= p_right):
            return None
        shifts.add(shift)
    if len(shifts) != 1:
        return None
    shift = shifts.pop()
    # 目标侧双向唯一性校验：防止多个不同偏移的源段（含缺单侧边界的残缺竞争段）映射到同一对侧窗口
    p_start, p_end = start + shift, end + shift
    for seg in sim_segs:
        pl = _safe_finite_float(seg.get(f"{peer}_st"))
        pr = _safe_finite_float(seg.get(f"{peer}_en"))
        sl = _safe_finite_float(seg.get(f"{side}_st"))
        sr = _safe_finite_float(seg.get(f"{side}_en"))
        dt_v = _safe_finite_float(seg.get("dt_AB"))
        # 当目标侧仅给出一侧边界时，结合源侧区间+dt_AB推算缺失边界，防止残缺竞争段绕过多对一检查
        eff_pl = pl if pl is not None else (sl + sign * dt_v if (sl is not None and dt_v is not None) else None)
        eff_pr = pr if pr is not None else (sr + sign * dt_v if (sr is not None and dt_v is not None) else None)
        if eff_pl is not None and eff_pr is not None:
            if p_end <= eff_pl or p_start >= eff_pr:
                continue
        elif eff_pl is not None:
            if p_end <= eff_pl:
                continue
        elif eff_pr is not None:
            if p_start >= eff_pr:
                continue
        else:
            # 既无目标侧边界也无法推算，无法确认与目标区间相交，跳过无关段
            continue
        if pl is None or pr is None or sl is None or sr is None or dt_v is None:
            return None
        other_shift = sign * dt_v
        if other_shift != shift:
            return None
    return shift


def _is_covered_on_peer(
    g_st: float,
    g_en: float,
    peer_metas: Sequence[Dict[str, Any]],
    shift: float,
    margin: float = 0.45,
) -> bool:
    """判断本视角未覆盖的活跃语音区段，映射至对侧视角后是否已被台词对齐覆盖（隔墙闷音串扰）。"""
    p_st = g_st + shift
    p_en = g_en + shift
    dur = max(1e-6, p_en - p_st)
    overlaps = []
    for m in peer_metas:
        if m["skipped"] or m["start"] is None or m["end"] is None:
            continue
        start = max(p_st, float(m["start"]) - margin)
        end = min(p_en, float(m["end"]) + margin)
        if start < end:
            overlaps.append((start, end))
    return sum(end - start for start, end in _merge_intervals(overlaps)) >= 0.35 * dur


def build_multiperspective_audit(
    alignment_tokens: Sequence[Sequence[str]],
    alignment_results_a: Sequence[Dict[str, Any]],
    alignment_results_b: Sequence[Dict[str, Any]],
    ono_clause_flags: Sequence[int],
    clause_surfaces: Sequence[str],
    clause_speakers: Sequence[Any],
    clause_line_nos: Sequence[int],
    clause_annotated: Optional[Sequence[str]] = None,
    diagnostics_a: Optional[Dict[str, Any]] = None,
    diagnostics_b: Optional[Dict[str, Any]] = None,
    delta_sec: Optional[float] = None,
    sim_segs: Optional[Sequence[Dict[str, Any]]] = None,
    low_score_threshold: float = -3.20,
) -> Dict[str, Any]:
    """双视角联合声学体检：
    - 仅当某实词句在视角 A 与 B **两侧均被跳过** 时才判 ``skipped_semantic``（消除客视角正常跳过误报）；
    - 按每句更清晰的主视角（A/B 中得分更高的一侧）评估 ``low_score / compressed / stretched``；
    - 分别扫描 A/B 两侧未覆盖活跃语音空档，并剔除“仅是对侧视角已对齐台词的隔墙闷音串扰”及双侧重复空档。
    """
    metas_a = _extract_clause_meta_from_alignment(
        alignment_tokens=alignment_tokens,
        alignment_results=alignment_results_a,
        ono_clause_flags=ono_clause_flags,
        clause_surfaces=clause_surfaces,
        clause_speakers=clause_speakers,
        clause_line_nos=clause_line_nos,
        clause_annotated=clause_annotated,
    )
    metas_b = _extract_clause_meta_from_alignment(
        alignment_tokens=alignment_tokens,
        alignment_results=alignment_results_b,
        ono_clause_flags=ono_clause_flags,
        clause_surfaces=clause_surfaces,
        clause_speakers=clause_speakers,
        clause_line_nos=clause_line_nos,
        clause_annotated=clause_annotated,
    )

    chosen: List[Tuple[str, Dict[str, Any]]] = []
    for ma, mb in zip(metas_a, metas_b):
        if ma["is_ono"]:
            continue
        if ma["skipped"] and mb["skipped"]:
            chosen.append(("BOTH", ma))
        elif not ma["skipped"] and mb["skipped"]:
            chosen.append(("A", ma))
        elif ma["skipped"] and not mb["skipped"]:
            chosen.append(("B", mb))
        else:
            if float(ma["mean_score"]) >= float(mb["mean_score"]):
                chosen.append(("A", ma))
            else:
                chosen.append(("B", mb))

    sem_scores = [
        m["mean_score"] for _, m in chosen
        if not m["skipped"] and m["n_phones"] >= 4 and m["start"] is not None
    ]
    med_score = float(np.median(sem_scores)) if sem_scores else -1.50
    eff_low_thr = min(float(low_score_threshold), med_score - 1.80)
    short_low_thr = min(-4.80, eff_low_thr - 1.40)

    da = diagnostics_a or {}
    db = diagnostics_b or {}
    jumps_a = _detect_isolated_speaker_jumps([m for side, m in chosen if side == "A"])
    jumps_b = _detect_isolated_speaker_jumps([m for side, m in chosen if side == "B"])
    engulfs_a = _detect_cross_speaker_engulfs([m for side, m in chosen if side == "A"])
    engulfs_b = _detect_cross_speaker_engulfs([m for side, m in chosen if side == "B"])
    ov_a = _detect_solo_speaker_overlaps([m for side, m in chosen if side == "A"], da.get("stereo_wav_16k"))
    ov_b = _detect_solo_speaker_overlaps([m for side, m in chosen if side == "B"], db.get("stereo_wav_16k"))

    suspects: List[Dict[str, Any]] = []
    for side, m in chosen:
        reasons = _evaluate_clause_reasons(m, eff_low_thr, short_low_thr)
        jinfo = (jumps_a if side == "A" else (jumps_b if side == "B" else {})).get(int(m["clause_index"]))
        if jinfo is not None:
            reasons.append("isolated_speaker_jump")
        einfo = (engulfs_a if side == "A" else (engulfs_b if side == "B" else {})).get(int(m["clause_index"]))
        if einfo is not None:
            reasons.append("cross_speaker_engulf")
        oinfo = (ov_a if side == "A" else (ov_b if side == "B" else {})).get(int(m["clause_index"]))
        if oinfo is not None:
            reasons.append("solo_speaker_overlap")
        if reasons:
            suspects.append(_make_suspect_entry(m, reasons, perspective=side, jump_info=jinfo, engulf_info=einfo, overlap_info=oinfo))

    suspects.sort(
        key=lambda x: (
            0 if "skipped_semantic" in x["reasons"] else (
                1 if any(r in x["reasons"] for r in ("isolated_speaker_jump", "internal_gap_stretch", "cross_speaker_engulf", "solo_speaker_overlap")) else 2
            ),
            0 if x.get("priority") == "high" else 1,
            x["mean_score"],
            x["clause_index"],
        )
    )

    raw_gaps_a = _detect_unaligned_speech_gaps(
        clause_metas=metas_a,
        ctc_nb_active=da.get("ctc_nb_active"),
        frame_sec=float(da.get("frame_sec", 0.02)),
        duration_sec=float(da.get("duration_sec", 0.0)),
    )
    raw_gaps_b = _detect_unaligned_speech_gaps(
        clause_metas=metas_b,
        ctc_nb_active=db.get("ctc_nb_active"),
        frame_sec=float(db.get("frame_sec", 0.02)),
        duration_sec=float(db.get("duration_sec", 0.0)),
    )

    gaps: List[Dict[str, Any]] = []
    kept_a_bcoords: List[Tuple[float, float, float]] = []
    for g in raw_gaps_a:
        shift = _peer_shift(g["start"], g["end"], "A", sim_segs, delta_sec)
        if shift is not None and _is_covered_on_peer(g["start"], g["end"], metas_b, shift):
            continue
        ga = dict(g)
        ga["perspective"] = "A"
        gaps.append(ga)
        if shift is not None:
            kept_a_bcoords.append((g["start"] + shift, g["end"] + shift, shift))

    for g in raw_gaps_b:
        shift = _peer_shift(g["start"], g["end"], "B", sim_segs, delta_sec)
        if shift is not None and _is_covered_on_peer(g["start"], g["end"], metas_a, shift):
            continue
        # 避免与视角 A 已报的同一处缺失语音重复报警
        if shift is not None and any(
            shift == -a_shift and min(g["end"], be) - max(g["start"], bs) > 0.35 * max(0.1, g["duration"])
            for bs, be, a_shift in kept_a_bcoords
        ):
            continue
        gb = dict(g)
        gb["perspective"] = "B"
        gaps.append(gb)

    gaps.sort(key=lambda x: (x["start"], x.get("perspective", "A")))
    priority_review = _build_priority_review(
        suspects,
        gaps,
        metas_by_side={"A": metas_a, "B": metas_b, "BOTH": metas_a},
    )

    return {
        "is_multiperspective": True,
        "track_median_score": round(med_score, 2),
        "effective_low_score_threshold": round(eff_low_thr, 2),
        "semantic_clauses_total": len(chosen),
        "priority_review_count": len(priority_review),
        "suspect_clauses_count": len(suspects),
        "unaligned_speech_gaps_count": len(gaps),
        "priority_review": priority_review,
        "suspect_clauses": suspects,
        "unaligned_speech_gaps": gaps,
    }


def _downsample_to_mono_16k(wav_arr: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
    """将审听切片转为 16kHz 单声道 float32，显著降低体积与多模态听审开销。"""
    arr = np.asarray(wav_arr, dtype=np.float32)
    mono = np.mean(arr, axis=0) if arr.ndim == 2 else arr.reshape(-1)
    target_sr = 16000
    if sr <= target_sr or mono.size < 2:
        return mono.astype(np.float32), int(sr)
    try:
        from math import gcd
        from scipy.signal import resample_poly

        g = gcd(int(sr), target_sr)
        out = resample_poly(mono, target_sr // g, int(sr) // g).astype(np.float32)
        return out, target_sr
    except Exception:
        n_out = max(1, int(round(mono.size * (float(target_sr) / float(sr)))))
        x_old = np.linspace(0.0, 1.0, num=mono.size, dtype=np.float64)
        x_new = np.linspace(0.0, 1.0, num=n_out, dtype=np.float64)
        out = np.interp(x_new, x_old, mono).astype(np.float32)
        return out, target_sr


def export_audit_clips(
    audio_path: str,
    audit_report: Dict[str, Any],
    clips_dir: str,
    pad_sec: float = 1.0,
    max_clips_per_kind: int = 30,
    merge_gap_sec: float = 1.5,
    max_window_sec: float = 12.0,
    enable_separate: bool = False,
    sep_sources: int = 2,
    force_separate: bool = False,
    sigma2: float = 0.45,
    audio_path_b: Optional[str] = None,
    priority_only: bool = False,
    compact_clips: bool = False,
) -> int:
    """按体检报告合并相邻异常窗、切出原生采样率审听 WAV，并可选联动立体声空间分离。

    当 ``enable_separate=True`` 时：
      - 若切片为双声道且检测到显著立体声多方位分布（或 ``force_separate=True``），
        自动调用 ``fa_asmr_spatial_sep.separate_stereo_spatial`` 导出各声像干声 WAV，
        并将文件路径列表回填至每个异常条目的 ``separated_wavs`` 字段以及
        ``audit_report["review_clips"]`` 中；
      - 若为单声道或单声源居中独白，则跳过分离（避免无意义噪声与多余文件）。
    """
    import soundfile as sf
    from fa_asmr_spatial_sep import (
        _merge_interval_items,
        _probe_audio_duration,
        _read_stereo_audio,
        has_stereo_spatial_spread,
        make_neutral_clip_prefix,
        separate_stereo_spatial,
    )

    os.makedirs(clips_dir, exist_ok=True)
    dur_a = _probe_audio_duration(audio_path)
    if dur_a <= 0.0:
        return 0
    dur_b = _probe_audio_duration(audio_path_b) if audio_path_b else 0.0
    stem_a = os.path.splitext(os.path.basename(audio_path))[0]
    stem_b = os.path.splitext(os.path.basename(audio_path_b))[0] if audio_path_b else stem_a
    exported = 0
    separated_count = 0
    review_clips: List[Dict[str, Any]] = []

    def _export_window(kind: str, idx: int, win: Dict[str, Any], side: str) -> None:
        nonlocal exported, separated_count
        use_b = (side == "B" and bool(audio_path_b) and dur_b > 0.0)
        aud_p = str(audio_path_b) if use_b else audio_path
        dur_tot = dur_b if use_b else dur_a
        stem = stem_b if use_b else stem_a

        a_st = round(max(0.0, float(win["start"]) - float(pad_sec)), 2)
        a_en = round(min(dur_tot, float(win["end"]) + float(pad_sec)), 2)
        if a_en <= a_st + 0.05:
            return
        wav_data, sr = _read_stereo_audio(aud_p, start_sec=a_st, end_sec=a_en)
        if wav_data.size == 0:
            return

        members = win["items"]
        lines_hit = sorted({int(it["line"]) for it in members if it.get("line") is not None})
        spks_hit = sorted({str(it["speaker"]) for it in members if it.get("speaker")})
        line_tag = f"_L{lines_hit[0]}" if (kind == "suspect" and lines_hit) else ""
        n_pfx = make_neutral_clip_prefix(stem, side=("B" if use_b else "A"), dual_pov=bool(audio_path_b))
        base_tag = f"{n_pfx}_{kind}_{idx:02d}{line_tag}_{a_st:.1f}s_{a_en:.1f}s"
        out_p = os.path.join(clips_dir, f"{base_tag}.wav")
        sep_files: List[Dict[str, Any]] = []
        spread_meta: Optional[Dict[str, Any]] = None
        did_sep = False
        if enable_separate and wav_data.ndim == 2 and wav_data.shape[0] >= 2 and wav_data.shape[1] >= 1024:
            is_spread, spread_info = has_stereo_spatial_spread(wav_data[:2], sr=sr)
            spread_meta = {**spread_info, "is_spread": is_spread}
            if force_separate or is_spread:
                sep_wavs, ch_meta = separate_stereo_spatial(
                    wav_data[:2], sr=sr, n_sources=int(sep_sources), sigma2=float(sigma2)
                )
                for k, cm in enumerate(ch_meta):
                    ch_p = os.path.join(clips_dir, f"{base_tag}_K{k+1}_{cm['position_tag']}.wav")
                    if compact_clips:
                        c_sig, c_sr = _downsample_to_mono_16k(sep_wavs[k], sr)
                        sf.write(ch_p, c_sig, c_sr, subtype="PCM_16")
                    else:
                        sf.write(ch_p, sep_wavs[k], sr)
                    sep_files.append({**cm, "wav": ch_p})
                did_sep = True
                separated_count += 1
            spread_meta["separated"] = did_sep
        elif enable_separate:
            spread_meta = {"ild_span_db": 0.0, "ild_iqr_db": 0.0, "ild_median_db": 0.0, "msc": 1.0, "cres": 0.0, "is_spread": False, "separated": False}

        out_sr = int(sr)
        if compact_clips:
            c_mono, out_sr = _downsample_to_mono_16k(wav_data, sr)
            sf.write(out_p, c_mono, out_sr, subtype="PCM_16")
        else:
            sf.write(out_p, wav_data.T.astype(np.float32), sr)
        exported += 1

        for it in members:
            it["clip_wav"] = out_p
            if spread_meta is not None:
                it["stereo_spread"] = dict(spread_meta)
            if sep_files:
                it["separated_wavs"] = list(sep_files)

        clip_entry: Dict[str, Any] = {
            "kind": kind,
            "index": idx,
            "start": a_st,
            "end": a_en,
            "duration": round(a_en - a_st, 2),
            "sample_rate": int(out_sr),
            "lines": lines_hit,
            "speakers": spks_hit,
            "member_count": len(members),
            "source_stem": stem,
            "source_audio": os.path.abspath(aud_p),
            "clip_wav": out_p,
        }
        if compact_clips:
            clip_entry["compact"] = True
        if audio_path_b:
            clip_entry["perspective"] = "B" if use_b else "A"
        if spread_meta is not None:
            clip_entry["stereo_spread"] = spread_meta
        if sep_files:
            clip_entry["separated_wavs"] = sep_files
        review_clips.append(clip_entry)

    def _export_kind(kind: str, items: Sequence[Dict[str, Any]]) -> None:
        if audio_path_b:
            by_side: Dict[str, List[Dict[str, Any]]] = {"A": [], "B": []}
            for it in items:
                s = "B" if str(it.get("perspective", "A")).upper() == "B" else "A"
                by_side[s].append(it)
            idx_counter = 0
            for side in ("A", "B"):
                for win in _merge_interval_items(
                    by_side[side], merge_gap_sec=merge_gap_sec, max_window_sec=max_window_sec
                ):
                    if idx_counter >= max_clips_per_kind:
                        break
                    idx_counter += 1
                    _export_window(kind, idx_counter, win, side)
        else:
            wins = _merge_interval_items(items, merge_gap_sec=merge_gap_sec, max_window_sec=max_window_sec)
            for idx, win in enumerate(wins[:max_clips_per_kind], 1):
                _export_window(kind, idx, win, "A")

    all_suspects = audit_report.get("suspect_clauses", [])
    if priority_only:
        suspects_to_export = [s for s in all_suspects if s.get("priority", "high") == "high"]
    else:
        suspects_to_export = all_suspects
    all_gaps = audit_report.get("unaligned_speech_gaps", [])

    _export_kind("suspect", suspects_to_export)
    _export_kind("gap", all_gaps)

    # 将已切出的 clip_wav / stereo_spread / separated_wavs 同步回填到 priority_review 摘要条目中
    pr_list = audit_report.get("priority_review")
    if isinstance(pr_list, list) and pr_list:
        suspect_by_ci = {
            (s.get("clause_index"), s.get("perspective")): s
            for s in all_suspects
            if s.get("clause_index") is not None
        }
        gap_by_span = {
            (g.get("start"), g.get("end"), g.get("perspective")): g
            for g in all_gaps
        }
        for pr in pr_list:
            src_item = (
                suspect_by_ci.get((pr.get("clause_index"), pr.get("perspective")))
                if pr.get("kind") == "suspect"
                else gap_by_span.get((pr.get("start"), pr.get("end"), pr.get("perspective")))
            )
            if not src_item:
                continue
            for k in ("clip_wav", "stereo_spread", "separated_wavs"):
                if k in src_item:
                    pr[k] = src_item[k]

    audit_report["review_clips"] = review_clips
    audit_report["separated_clips_count"] = separated_count
    audit_report["clips_priority_only"] = bool(priority_only)
    return exported
