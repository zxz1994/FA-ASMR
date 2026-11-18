# -*- coding: utf-8 -*-
"""FA-ASMR V2 — 多视角联合对齐与动态融合引擎 (Multi-Perspective Joint Alignment)

四步干净架构（完全独立，不修改 align.py 与 align_trellis.py）：
1. 纯声学分段相似度校准 (detect_clean_audio_segments):
   使用 PyTorch GPU/CPU 批处理 FFT 互相关 + 有效音素余弦相似度 (>= 0.22) 过滤静音/喘息假峰，
   按时间序聚类生成紧密的纯声学同步段 SimSeg（天然支持中途切换房间的跳跃 Δt）。
2. 按说话人与场景切块及绝对清晰度判主视角 (segment_and_assign_home):
   结合双视角清晰锚点 (SHARED 平台)、持续主场翻转切分 (flip_cuts) 与 SimSeg 场景边界切分时间段，
   通过绝对 CTC 清晰度阈值独立判定每块归属 (`A` / `B` / `SHARED` / `None`)，支持重音效同步双麦救援。
3. 主视角块全量采纳与立体声单声源并发修复 (refine_home_segments):
   100% 保留主视角对齐质量；仅针对被另一方抢词撞坏的弱语义行，使用纯 STFT 双声道相干性检测器
   (_AudioSoloChecker) 精确识别单声源独占区间，在局部邻域窗口内执行有界重对齐。
4. 客视角局部 Δt 小窗口对齐与独立导出 (align_away_speaker_by_dt):
   仅沿 SimSeg 给出的真实 Δt 候选及局部邻域漂移窗口执行 Numba 加速局部切片 CTC 检索，
   以背景对照增益筛选高置信度击中并严格按时序与主视角交错过滤，输出 A/B 原生双视角时间轴。
"""
from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchaudio

try:
    import numba as _nb
    _HAS_NUMBA = True
except Exception:
    _nb = None
    _HAS_NUMBA = False

from align_acoustic import FRAME_SEC
from align_utils import _format_time
import align_multispeaker as ams
from clause import F


def _njit_cache(fn: Callable) -> Callable:
    """Use numba.njit(cache=True, nogil=True) when available; otherwise fall back cleanly to Python/NumPy."""
    if _HAS_NUMBA:
        return _nb.njit(cache=True, nogil=True)(fn)
    return fn


_NUMBA_WARMED_UP = False


_CACHED_TOKENIZER_DICT: Optional[Dict[str, int]] = None


def _get_tokenizer_dict(ctx: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
    """Retrieve MMS-FA tokenizer dictionary cleanly with module-level caching."""
    global _CACHED_TOKENIZER_DICT
    if ctx is not None and isinstance(ctx.get("tokenizer"), object):
        tok = ctx.get("tokenizer")
        if hasattr(tok, "dictionary"):
            return tok.dictionary
    if _CACHED_TOKENIZER_DICT is None:
        _CACHED_TOKENIZER_DICT = torchaudio.pipelines.MMS_FA.get_tokenizer().dictionary
    return _CACHED_TOKENIZER_DICT


def _norm_spk(raw_sp: Any) -> str:
    """Normalize speaker metadata (str, tuple, list, or empty) into a stable string key."""
    if isinstance(raw_sp, (tuple, list)):
        parts = [str(x).strip().lstrip("　") for x in raw_sp if str(x).strip()]
        return "/".join(parts) if parts else "UNK"
    s = str(raw_sp or "").strip().lstrip("　")
    return s if s else "UNK"


def _load_audio_16k_once(audio_file_path: str | Path) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray]]:
    """Load and resample audio once into 16kHz tensor (for MMS-FA) and (2, N) float32 numpy (for STFT)."""
    try:
        wav, sr = torchaudio.load(str(audio_file_path))
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        elif wav.shape[0] > 2:
            wav = wav[:2]
        if int(sr) != 16000:
            wav = torchaudio.functional.resample(wav, int(sr), 16000)
        wav = wav.float()
        stereo_wav = wav.repeat(2, 1) if wav.shape[0] == 1 else wav
        return wav, stereo_wav.detach().cpu().numpy().astype(np.float32)
    except Exception:
        return None, None


def _load_stereo_16k(audio_file_path: str | Path) -> Optional[np.ndarray]:
    """Load audio as 16kHz float32 (2, N) numpy array for pure-audio STFT solo detection."""
    _, stereo_np = _load_audio_16k_once(audio_file_path)
    return stereo_np


# ══════════════════════════════════════════════════════════════════════════
#  0. 轻量声学特征工具与 STFT 单声源并发检测器
# ══════════════════════════════════════════════════════════════════════════

def _is_pure_sfx_clause(
    surface: str,
    tokens: Optional[Sequence[str]] = None,
    tok_dict: Optional[Dict[str, int]] = None,
) -> bool:
    """识别纯拟声/吻声/喘息短句，仅用于 Step 4 锚点隔离（绝不从主视角剔除）。"""
    s = re.sub(r"[\s…〜～ー・、。！？!?\.,♡♥♪()（）]+", "", str(surface or ""))
    if s and re.fullmatch(
        r"(?:ちゅ|ちゅっ|ちゅぱ|ちゅる|ちゅぅ|じゅ|じゅる|じゅぼ|ずぼ|れろ|れりゅ|ねりゅ|ねぢゅ|れぢゅ|ぺろ|ぴちゃ|くちゅ|ぐちゅ|ごっくん|んちゅ|んっ|んぁ|んふ|んは|はぁ|はぅ|あぁ|ぁっ|ぁぁ|っ|あっ|ふぅ|ぅっ|くぅ|きゅ|びゅ|ぴゅ|ぷは|ぼっ|ずぶ)+[っぅぁぃぇぉー〜]*",
        s,
    ):
        return True
    if tokens:
        t_dict = tok_dict if tok_dict is not None else _get_tokenizer_dict()
        t_clean = [t for t in tokens if any(c in t_dict for c in t)]
        if len(t_clean) <= 1:
            return True
    return False


# 与 align_multispeaker 共用 STFT 单声源检测器及未遮罩原始分检验，避免逻辑重复
_unmasked_clause_mean_score = ams._unmasked_clause_mean_score
_AudioSoloChecker = ams._AudioSoloChecker


def compute_phoneme_advantage_matrix(coop_np: np.ndarray, acoustic_ctx: Dict[str, Any]) -> np.ndarray:
    """计算相对于背景有效音素中位数的逐帧音素净优势矩阵。"""
    blank = int(acoustic_ctx.get("blank", 0))
    ignored = set(acoustic_ctx.get("ignored_classes", ()))
    valid_cls = [c for c in range(coop_np.shape[1]) if c != blank and c not in ignored]
    adv = coop_np - np.median(coop_np[:, valid_cls], axis=1, keepdims=True)
    if "emission_L" in acoustic_ctx and "emission_R" in acoustic_ctx:
        for ch_k in ("emission_L", "emission_R"):
            em_ch = acoustic_ctx[ch_k][0].detach().cpu().numpy()
            adv = np.maximum(adv, em_ch - np.median(em_ch[:, valid_cls], axis=1, keepdims=True))
    return adv.astype(np.float32)


def _prep_phoneme_probs(coop_np: np.ndarray, ctx: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """预处理L2归一化非Blank音素概率矩阵与激活帧掩码。"""
    blank = int(ctx.get("blank", 0))
    ignored = set(ctx.get("ignored_classes", ()))
    valid_cls = [c for c in range(coop_np.shape[1]) if c != blank and c not in ignored]
    logp = np.clip(coop_np[:, valid_cls], -7.5, 0.0)
    act = np.max(logp, axis=1) > -3.8
    exp_p = np.exp(logp - np.max(logp, axis=1, keepdims=True))
    norm_p = exp_p / (np.linalg.norm(exp_p, axis=1, keepdims=True) + 1e-8)
    return norm_p.astype(np.float32), act


def _active_phoneme_sim(
    norm_src: np.ndarray,
    act_src: np.ndarray,
    tgt_channels: Sequence[Tuple[np.ndarray, np.ndarray]],
    st_src: float,
    en_src: float,
    dt_src_to_tgt: float,
    pad_steps: int = 15,
) -> float:
    """在 ±pad_steps 帧微抖动范围内测算局部发声窗跨视角活跃音素余弦相似度。"""
    f_st = int(round(st_src / FRAME_SEC))
    f_en = int(round(en_src / FRAME_SEC))
    wf = max(1, f_en - f_st)
    pad_f = 5 if wf < 20 else 0
    f0 = max(0, f_st - pad_f)
    wf_p = wf + 2 * pad_f
    if f0 + wf_p > norm_src.shape[0]:
        return 0.0
    act_s = act_src[f0 : f0 + wf_p]
    n_s = int(np.sum(act_s))
    if n_s == 0:
        return 0.0
    p_s = norm_src[f0 : f0 + wf_p]
    shift_f = int(round(dt_src_to_tgt / FRAME_SEC))
    best = 0.0
    for norm_t, act_t_all in tgt_channels:
        t_len = norm_t.shape[0]
        for df in range(-pad_steps, pad_steps + 1):
            ft = f0 + shift_f + df
            if ft < 0 or ft + wf_p > t_len:
                continue
            act_t = act_t_all[ft : ft + wf_p]
            n_t = int(np.sum(act_t))
            both = act_s & act_t
            n_both = int(np.sum(both))
            if n_both < 1:
                continue
            cos = np.sum(p_s[both] * norm_t[ft : ft + wf_p][both], axis=1)
            ov = min(1.0, float(n_both) / max(1.0, 0.5 * (n_s + n_t)))
            best = max(best, float(np.mean(cos)) * ov)
    return best


_EMPTY_F2D = np.empty((0, 0), dtype=np.float32)


@_njit_cache
def _run_ctc_dp_nb(
    sl_em: np.ndarray,
    sl_adv: np.ndarray,
    has_fg: bool,
    sl_fg: np.ndarray,
    tid_arr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute local CTC step/stay score matrices for one channel slice."""
    W = sl_em.shape[0]
    M = tid_arr.shape[0]
    char_em = np.empty((W, M), dtype=np.float32)
    step_score = np.empty((W, M), dtype=np.float32)
    stay_score = np.empty((W, M), dtype=np.float32)
    f060 = np.float32(0.60)
    f040 = np.float32(0.40)
    f200 = np.float32(2.0)
    f180 = np.float32(1.80)
    f012 = np.float32(0.12)
    f005 = np.float32(0.05)

    for t in range(W):
        b_stay = sl_em[t, 0] - f012
        for m in range(M):
            tid = tid_arr[m]
            ce = sl_em[t, tid]
            char_em[t, m] = ce
            st_sc = f060 * ce + f040 * (sl_adv[t, tid] - f200)
            if has_fg:
                st_sc -= f180 * sl_fg[t, tid]
            step_score[t, m] = st_sc
            stay_score[t, m] = (st_sc if st_sc >= b_stay else b_stay) - f005

    return char_em, step_score, stay_score, sl_em


@_njit_cache
def _eval_ctc_channel_nb(
    sl_em: np.ndarray,
    sl_adv: np.ndarray,
    has_fg: bool,
    sl_fg: np.ndarray,
    tid_arr: np.ndarray,
    expected_mid_f: float,
    min_dur_f: int,
    max_dur_f: int,
    max_m0_span: int,
    dur_home: float,
    frame_sec: float,
) -> Tuple[bool, float, int, int, float, float, float, float]:
    """Evaluate local CTC Viterbi alignment for one channel and return summary metrics."""
    W = sl_em.shape[0]
    M = tid_arr.shape[0]
    char_em, step_score, stay_score, _ = _run_ctc_dp_nb(sl_em, sl_adv, has_fg, sl_fg, tid_arr)

    dp = np.full((W, M), np.float32(-1e9), dtype=np.float32)
    bp = np.zeros((W, M), dtype=np.int32)
    s0 = np.zeros((W, M), dtype=np.int32)
    for t in range(W):
        dp[t, 0] = step_score[t, 0]
        s0[t, 0] = t

    for t in range(1, W):
        ext0 = dp[t - 1, 0] + stay_score[t, 0]
        fresh0 = step_score[t, 0]
        if ext0 >= fresh0 and (t - s0[t - 1, 0]) <= max_m0_span:
            dp[t, 0] = ext0
            bp[t, 0] = 0
            s0[t, 0] = s0[t - 1, 0]
        else:
            dp[t, 0] = fresh0
            bp[t, 0] = -1
            s0[t, 0] = t
        for m in range(1, M):
            c_stay = dp[t - 1, m] + stay_score[t, m]
            c_step = dp[t - 1, m - 1] + step_score[t, m]
            if c_step >= c_stay:
                dp[t, m] = c_step
                bp[t, m] = m - 1
                s0[t, m] = s0[t - 1, m - 1]
            else:
                dp[t, m] = c_stay
                bp[t, m] = m
                s0[t, m] = s0[t - 1, m]

    best_t1 = -1
    best_obj = -1e18
    for t1 in range(M - 1, W):
        t0_c = s0[t1, M - 1]
        span = t1 - t0_c + 1
        if span < min_dur_f or span > max_dur_f:
            continue
        dur_cand = span * frame_sec
        if dur_cand < 0.15:
            dur_cand = 0.15
        mid_f = 0.5 * (t0_c + t1)
        obj = (
            float(dp[t1, M - 1]) / float(span)
            - 0.08 * abs(mid_f - expected_mid_f) * frame_sec
            - 0.25 * abs(np.log(dur_cand / dur_home))
        )
        if obj > best_obj:
            best_obj = obj
            best_t1 = t1

    if best_t1 < 0:
        return False, -99.0, 0, 0, -10.0, -10.0, -10.0, 0.0

    has_f = np.zeros(M, dtype=np.uint8)
    max_em = np.full(M, np.float32(-1e9), dtype=np.float32)
    max_av = np.full(M, np.float32(-1e9), dtype=np.float32)
    m0_min_t = W
    t_cur = best_t1
    m_cur = M - 1
    while t_cur >= 0 and m_cur >= 0:
        has_f[m_cur] = 1
        if m_cur == 0 and t_cur < m0_min_t:
            m0_min_t = t_cur
        tid = tid_arr[m_cur]
        em_v = char_em[t_cur, m_cur]
        av_v = sl_adv[t_cur, tid]
        if em_v > max_em[m_cur]:
            max_em[m_cur] = em_v
        if av_v > max_av[m_cur]:
            max_av[m_cur] = av_v
        pm = bp[t_cur, m_cur]
        if pm == -1:
            break
        m_cur = pm
        t_cur -= 1

    t0_rel = m0_min_t if has_f[0] != 0 else int(s0[best_t1, M - 1])
    cnt = 0
    sum_em = 0.0
    sum_av = 0.0
    for m in range(M):
        if has_f[m] != 0:
            sum_em += float(max_em[m])
            sum_av += float(max_av[m])
            cnt += 1
    if cnt == 0:
        sc_m = -10.0
        p25_s = -10.0
        av_m = 0.0
    else:
        sc_m = sum_em / cnt
        av_m = sum_av / cnt
        vals = np.empty(cnt, dtype=np.float64)
        vi = 0
        for m in range(M):
            if has_f[m] != 0:
                vals[vi] = float(max_em[m])
                vi += 1
        p25_s = float(np.percentile(vals, 25.0))

    sc_bal = 0.5 * sc_m + 0.5 * p25_s
    dur_t = float((best_t1 + 1 - t0_rel) * frame_sec)
    dur_c = dur_t if dur_t >= 0.15 else 0.15
    q_m = sc_bal + 0.45 * av_m - 0.25 * abs(np.log(dur_c / dur_home))
    return True, q_m, t0_rel, best_t1, sc_m, p25_s, sc_bal, av_m


@_njit_cache
def _backtrack_ctc_bounds_nb(
    sl_em: np.ndarray,
    sl_adv: np.ndarray,
    has_fg: bool,
    sl_fg: np.ndarray,
    tid_arr: np.ndarray,
    max_m0_span: int,
    best_t1: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct per-character min/max relative frames for the winning channel."""
    W = sl_em.shape[0]
    M = tid_arr.shape[0]
    _, step_score, stay_score, _ = _run_ctc_dp_nb(sl_em, sl_adv, has_fg, sl_fg, tid_arr)

    dp = np.full((W, M), np.float32(-1e9), dtype=np.float32)
    bp = np.zeros((W, M), dtype=np.int32)
    s0 = np.zeros((W, M), dtype=np.int32)
    for t in range(W):
        dp[t, 0] = step_score[t, 0]
        s0[t, 0] = t

    for t in range(1, W):
        ext0 = dp[t - 1, 0] + stay_score[t, 0]
        fresh0 = step_score[t, 0]
        if ext0 >= fresh0 and (t - s0[t - 1, 0]) <= max_m0_span:
            dp[t, 0] = ext0
            bp[t, 0] = 0
            s0[t, 0] = s0[t - 1, 0]
        else:
            dp[t, 0] = fresh0
            bp[t, 0] = -1
            s0[t, 0] = t
        for m in range(1, M):
            c_stay = dp[t - 1, m] + stay_score[t, m]
            c_step = dp[t - 1, m - 1] + step_score[t, m]
            if c_step >= c_stay:
                dp[t, m] = c_step
                bp[t, m] = m - 1
            else:
                dp[t, m] = c_stay
                bp[t, m] = m

    ch_min = np.full(M, W, dtype=np.int32)
    ch_max = np.full(M, -1, dtype=np.int32)
    t_cur = best_t1
    m_cur = M - 1
    while t_cur >= 0 and m_cur >= 0:
        if t_cur < ch_min[m_cur]:
            ch_min[m_cur] = t_cur
        if t_cur > ch_max[m_cur]:
            ch_max[m_cur] = t_cur
        pm = bp[t_cur, m_cur]
        if pm == -1:
            break
        m_cur = pm
        t_cur -= 1
    return ch_min, ch_max


def slice_ctc_verify_clause(
    tokens: Sequence[str],
    ctx: Dict[str, Any],
    channels: Sequence[np.ndarray],
    phoneme_adv_np: np.ndarray,
    center_st_sec: float,
    center_en_sec: float,
    search_pad_sec: float,
    fg_occupied_mask: Optional[np.ndarray] = None,
    fg_char_mask: Optional[np.ndarray] = None,
    tid_seq: Optional[Sequence[int]] = None,
    build_tokens: bool = False,
) -> Dict[str, Any]:
    """在局部 Δt 搜索窗内跨 (coop, L, R) 三通道执行切片 CTC 对齐验证。"""
    if not tokens:
        return {"matched": False, "score": -10.0, "adv": 0.0, "fg_overlap": 1.0, "tokens": []}
    tok_dict = _get_tokenizer_dict(ctx)
    if tid_seq is None:
        tid_seq = [int(tok_dict[ch]) for t in tokens for ch in t if ch in tok_dict]
    if not tid_seq:
        return {"matched": False, "score": -10.0, "adv": 0.0, "fg_overlap": 1.0, "tokens": []}

    T_total = int(channels[0].shape[0])
    dur_home = max(0.25, center_en_sec - center_st_sec)
    win_st_f = max(0, int(round((center_st_sec - search_pad_sec) / FRAME_SEC)))
    win_en_f = min(T_total, int(round((center_en_sec + search_pad_sec) / FRAME_SEC)))
    W, M = win_en_f - win_st_f, len(tid_seq)
    if W < max(8, len(tokens) * 2, M + 2):
        return {"matched": False, "score": -10.0, "adv": 0.0, "fg_overlap": 1.0, "tokens": []}

    min_dur_f = max(4, int(round(min(0.35, dur_home * 0.45) / FRAME_SEC)))
    max_dur_f = max(min_dur_f + 6, int(round(max(dur_home * 1.85, dur_home + 1.2) / FRAME_SEC)))
    expected_mid_f = 0.5 * ((center_st_sec + center_en_sec) / FRAME_SEC) - win_st_f
    max_m0_span = int(round(dur_home * 1.65 / FRAME_SEC)) + 15
    tid_arr = np.asarray(tid_seq, dtype=np.int32)
    sl_adv = phoneme_adv_np[win_st_f:win_en_f]
    has_fg = fg_char_mask is not None
    sl_fg = fg_char_mask[win_st_f:win_en_f] if has_fg else _EMPTY_F2D

    best_out: Dict[str, Any] = {
        "matched": False, "score": -10.0, "p25_score": -10.0, "sc_bal": -10.0,
        "adv": 0.0, "dur_ratio": 1.0, "fg_overlap": 1.0, "tokens": [],
    }
    best_q = -99.0
    best_ch_sl = None
    best_t1_rel = -1

    for ch_np in channels:
        sl_em = ch_np[win_st_f:win_en_f]
        matched, q_m, t0_rel, best_t1, sc_m, p25_s, sc_bal, av_m = _eval_ctc_channel_nb(
            sl_em, sl_adv, has_fg, sl_fg, tid_arr,
            float(expected_mid_f), int(min_dur_f), int(max_dur_f), int(max_m0_span),
            float(dur_home), float(FRAME_SEC),
        )
        if not matched or q_m <= best_q:
            continue
        best_q = q_m
        best_ch_sl = sl_em
        best_t1_rel = int(best_t1)
        t0_abs = win_st_f + int(t0_rel)
        t1_abs = win_st_f + int(best_t1)
        dur_t = float((t1_abs + 1 - t0_abs) * FRAME_SEC)
        fg_ov = float(np.mean(fg_occupied_mask[t0_abs : t1_abs + 1])) if fg_occupied_mask is not None else 0.0
        best_out = {
            "matched": True, "score": float(sc_m), "p25_score": float(p25_s), "sc_bal": float(sc_bal),
            "adv": float(av_m), "dur_ratio": dur_t / dur_home, "fg_overlap": fg_ov,
            "start_sec": t0_abs * FRAME_SEC, "end_sec": (t1_abs + 1) * FRAME_SEC, "tokens": [],
        }

    if build_tokens and best_out["matched"] and best_ch_sl is not None:
        ch_min, ch_max = _backtrack_ctc_bounds_nb(best_ch_sl, sl_adv, has_fg, sl_fg, tid_arr, int(max_m0_span), best_t1_rel)
        t0_abs = int(round(best_out["start_sec"] / FRAME_SEC))
        t1_abs = int(round(best_out["end_sec"] / FRAME_SEC)) - 1
        sc_m = float(best_out["score"])
        tok_res: List[Dict[str, Any]] = []
        cur = 0
        for tok in tokens:
            k_len = sum(1 for c in tok if c in tok_dict)
            mn_f = W
            mx_f = -1
            for mj in range(cur, min(M, cur + k_len)):
                if ch_max[mj] >= ch_min[mj]:
                    if ch_min[mj] < mn_f:
                        mn_f = int(ch_min[mj])
                    if ch_max[mj] > mx_f:
                        mx_f = int(ch_max[mj])
            cur += max(0, k_len)
            st_s = (win_st_f + mn_f if mx_f >= mn_f else t0_abs) * FRAME_SEC
            en_s = ((win_st_f + mx_f if mx_f >= mn_f else t1_abs) + 1) * FRAME_SEC
            tok_res.append({
                "token": tok, "start": _format_time(st_s), "end": _format_time(en_s),
                "original_start": st_s, "original_end": en_s, "score": sc_m,
            })
        best_out["tokens"] = tok_res
    return best_out


# ══════════════════════════════════════════════════════════════════════════
#  1. 纯声学跨视角分段相似度对齐 (Clean Pure-Audio SimSeg)
# ══════════════════════════════════════════════════════════════════════════

def _extract_audio_feat(ctx: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    coop_em = ams.build_cooperative_stereo_emission(ctx, ear_bias=0.0).detach().cpu().numpy()
    blank = int(ctx.get("blank", 0))
    ignored = set(ctx.get("ignored_classes", ()))
    valid_cls = [c for c in range(coop_em.shape[1]) if c != blank and c not in ignored]
    ph_logp = np.clip(coop_em[:, valid_cls], -7.5, 0.0).astype(np.float32)
    rms_max = np.clip(
        np.maximum(np.asarray(ctx["rms_l"], dtype=np.float32), np.asarray(ctx["rms_r"], dtype=np.float32)),
        -65.0,
        0.0,
    ).reshape(-1, 1)
    feat = np.concatenate([ph_logp, np.max(ph_logp, axis=1, keepdims=True), rms_max], axis=1)
    feat = (feat - np.mean(feat, axis=0, keepdims=True)) / (np.std(feat, axis=0, keepdims=True) + 1e-6)
    return feat.astype(np.float32), ph_logp


def _phoneme_cos_sim(ph_A: np.ndarray, ph_B: np.ndarray, st_A: int, st_B: int, win_f: int) -> float:
    sl_A, sl_B = ph_A[st_A : st_A + win_f], ph_B[st_B : st_B + win_f]
    if len(sl_A) == 0 or len(sl_A) != len(sl_B):
        return 0.0
    both = (np.max(sl_A, axis=1) > -3.2) & (np.max(sl_B, axis=1) > -3.2)
    if int(np.sum(both)) < 15:
        return 0.0
    pA = np.exp(sl_A[both] - np.max(sl_A[both], axis=1, keepdims=True))
    pB = np.exp(sl_B[both] - np.max(sl_B[both], axis=1, keepdims=True))
    pA /= np.sum(pA, axis=1, keepdims=True) + 1e-8
    pB /= np.sum(pB, axis=1, keepdims=True) + 1e-8
    return float(np.mean(np.sum(pA * pB, axis=1) / (np.linalg.norm(pA, axis=1) * np.linalg.norm(pB, axis=1) + 1e-8)))


def detect_clean_audio_segments(
    ctx_A: Dict[str, Any],
    ctx_B: Dict[str, Any],
    win_sec: float = 20.0,
    step_sec: float = 5.0,
) -> List[Dict[str, Any]]:
    """纯声学 A<->B 相似度分段：批处理 FFT 扫描与活跃音素余弦门槛 (ctc_sim >= 0.22)。"""
    feat_A, ph_A = _extract_audio_feat(ctx_A)
    feat_B, ph_B = _extract_audio_feat(ctx_B)
    T_A, D = feat_A.shape
    T_B = feat_B.shape[0]
    win_f = int(round(win_sec / FRAME_SEC))
    step_f = max(50, int(round(step_sec / FRAME_SEC)))
    if T_A < win_f or T_B < win_f:
        return []

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cs_A = np.concatenate([[0.0], np.cumsum(np.sum(feat_A.astype(np.float64) ** 2, axis=1))])
    n_valid_A = T_A - win_f + 1
    denom = np.sqrt(np.maximum(1e-6, cs_A[win_f:] - cs_A[:-win_f])[:n_valid_A] * float(D * win_f)).astype(np.float32)
    n_fft = 1 << int(np.ceil(np.log2(T_A + win_f + 1)))
    sep_f = int(round(1.5 / FRAME_SEC))

    feat_A_t = torch.from_numpy(feat_A).to(dev, dtype=torch.float32)
    F_A_t = torch.fft.rfft(feat_A_t, n=n_fft, dim=0)
    denom_t = torch.from_numpy(denom).to(dev, dtype=torch.float32)
    del feat_A_t

    win_starts_B = [
        st_B for st_B in range(0, T_B - win_f + 1, step_f)
        if float(np.std(feat_B[st_B : st_B + win_f])) >= 0.12
    ]

    hits: List[Dict[str, Any]] = []
    batch_sz = 16 if dev.type == "cuda" else 8
    with torch.no_grad():
        for b0 in range(0, len(win_starts_B), batch_sz):
            batch_sts = win_starts_B[b0 : b0 + batch_sz]
            segs = np.stack([feat_B[st_B : st_B + win_f] for st_B in batch_sts], axis=0)
            segs_n = (segs - np.mean(segs, axis=1, keepdims=True)) / (np.std(segs, axis=1, keepdims=True) + 1e-6)
            segs_t = torch.from_numpy(segs_n).to(dev, dtype=torch.float32)
            F_B_batch = torch.fft.rfft(segs_t, n=n_fft, dim=1)
            cross = torch.sum(F_A_t.unsqueeze(0) * torch.conj(F_B_batch), dim=2)
            ncc_batch = (torch.fft.irfft(cross, n=n_fft, dim=1)[:, :n_valid_A] / denom_t.unsqueeze(0)).detach().cpu().numpy()
            del segs_t, F_B_batch, cross

            for bi, st_B in enumerate(batch_sts):
                ncc = ncc_batch[bi]
                med, std = float(np.median(ncc)), float(np.std(ncc) + 1e-6)
                nw = ncc.copy()
                kept_w = 0
                for _ in range(8):
                    pk = int(np.argmax(nw))
                    pv = float(nw[pk])
                    z = (pv - med) / std
                    if z < 3.0 or pv < 0.045:
                        break
                    sim = _phoneme_cos_sim(ph_A, ph_B, pk, st_B, win_f)
                    if sim >= 0.22:
                        hits.append({
                            "st_B": st_B * FRAME_SEC, "en_B": (st_B + win_f) * FRAME_SEC,
                            "st_A": pk * FRAME_SEC, "en_A": (pk + win_f) * FRAME_SEC,
                            "dt_AB": float((pk - st_B) * FRAME_SEC), "w": float(z * sim),
                        })
                        kept_w += 1
                        if kept_w >= 3:
                            break
                    nw[max(0, pk - sep_f) : min(n_valid_A, pk + sep_f + 1)] = -1e9
    del F_A_t, denom_t

    hits.sort(key=lambda h: (h["st_B"], -h["w"]))
    clusters: List[Dict[str, Any]] = []
    for h in hits:
        best_cl, best_d = None, 1e9
        for cl in clusters:
            if abs(h["dt_AB"] - cl["dt_cur"]) <= 2.0 and (cl["last_st_B"] - 0.5) <= h["st_B"] <= (cl["last_en_B"] + 45.0):
                d = abs(h["dt_AB"] - cl["dt_cur"])
                if d < best_d:
                    best_d, best_cl = d, cl
        if best_cl is not None:
            best_cl["items"].append(h)
            best_cl["last_st_B"] = max(best_cl["last_st_B"], h["st_B"])
            best_cl["last_en_B"] = max(best_cl["last_en_B"], h["en_B"])
            best_cl["dt_cur"] = float(np.median([x["dt_AB"] for x in best_cl["items"][-4:]]))
        else:
            clusters.append({"items": [h], "last_st_B": h["st_B"], "last_en_B": h["en_B"], "dt_cur": h["dt_AB"]})

    sim_segs: List[Dict[str, Any]] = []
    for cl in clusters:
        its = cl["items"]
        tw = float(sum(x["w"] for x in its))
        if (len(its) >= 3 and tw >= 18.0) or tw >= 25.0:
            sim_segs.append({
                "A_st": min(x["st_A"] for x in its), "A_en": max(x["en_A"] for x in its),
                "B_st": min(x["st_B"] for x in its), "B_en": max(x["en_B"] for x in its),
                "dt_AB": float(np.median([x["dt_AB"] for x in its])),
                "weight": tw, "count": len(its),
            })
    sim_segs.sort(key=lambda s: (s["A_st"], s["B_st"]))
    return sim_segs


# ══════════════════════════════════════════════════════════════════════════
#  2. 按角色场景分块与绝对清晰度主视角判定 (Segment & Assign Home)
# ══════════════════════════════════════════════════════════════════════════

def segment_and_assign_home(
    spk: str,
    c_list: List[int],
    probe_A_spk: Dict[int, List[Dict[str, Any]]],
    probe_B_spk: Dict[int, List[Dict[str, Any]]],
    tokens_all: Sequence[Sequence[str]],
    onos_all: Sequence[int],
    clause_tid_seqs: Dict[int, List[int]],
    norm_p_A: np.ndarray,
    act_A: np.ndarray,
    norm_p_B: np.ndarray,
    act_B: np.ndarray,
    coop_A_np: np.ndarray,
    coop_B_np: np.ndarray,
    sim_segs: Sequence[Dict[str, Any]],
) -> Tuple[List[Optional[str]], List[Dict[str, Any]]]:
    """切分 SHARED 平台、持续主导权翻转与 SimSeg 场景转折点，并通过绝对阈值判定主视角。"""
    n_c = len(c_list)
    if n_c == 0:
        return [], []
    dur_A, dur_B = coop_A_np.shape[0] * FRAME_SEC, coop_B_np.shape[0] * FRAME_SEC

    sc_a = np.full(n_c, np.nan, dtype=np.float32)
    sc_b = np.full(n_c, np.nan, dtype=np.float32)
    ta = np.full(n_c, np.nan, dtype=np.float32)
    tb = np.full(n_c, np.nan, dtype=np.float32)
    sta = np.full(n_c, np.nan, dtype=np.float32)
    ena = np.full(n_c, np.nan, dtype=np.float32)
    stb = np.full(n_c, np.nan, dtype=np.float32)
    enb = np.full(n_c, np.nan, dtype=np.float32)
    bc_flag = np.zeros(n_c, dtype=bool)
    matched_dt = np.full(n_c, np.nan, dtype=np.float32)

    for k, ci in enumerate(c_list):
        toks, tids = tokens_all[ci], clause_tid_seqs[ci]
        if onos_all[ci] != 0 or len(toks) < 2 or len(tids) < 2:
            continue
        ia, ib = ams._clause_summary(probe_A_spk[ci]), ams._clause_summary(probe_B_spk[ci])
        max_d = max(4.5, 0.65 * len(tids))
        if not ia["skipped"] and 0.12 <= ia["dur"] <= max_d:
            sc_a[k], sta[k], ena[k], ta[k] = ia["score"], ia["st"], ia["en"], 0.5 * (ia["st"] + ia["en"])
        if not ib["skipped"] and 0.12 <= ib["dur"] <= max_d:
            sc_b[k], stb[k], enb[k], tb[k] = ib["score"], ib["st"], ib["en"], 0.5 * (ib["st"] + ib["en"])
        if not (np.isnan(sc_a[k]) or np.isnan(sc_b[k])) and sc_a[k] >= -2.55 and sc_b[k] >= -2.55:
            dt_v = float(ta[k] - tb[k])
            hits = [
                s for s in sim_segs
                if abs(dt_v - s["dt_AB"]) <= 3.5 and (s["A_st"] - 20.0) <= ta[k] <= (s["A_en"] + 20.0)
            ]
            if hits:
                dt_c = float(hits[0]["dt_AB"])
                dom_B = bool(sc_b[k] >= sc_a[k])
                s0_d, s1_d = (float(stb[k]), float(enb[k])) if dom_B else (float(sta[k]), float(ena[k]))
                ps = _active_phoneme_sim(
                    norm_p_B if dom_B else norm_p_A, act_B if dom_B else act_A,
                    [(norm_p_A, act_A)] if dom_B else [(norm_p_B, act_B)],
                    s0_d, s1_d, dt_c if dom_B else -dt_c, pad_steps=12,
                )
                if ps >= 0.25:
                    bc_flag[k], matched_dt[k] = True, dt_c

    raw_cl: List[List[int]] = []
    for k in range(n_c):
        if not bc_flag[k]:
            continue
        if raw_cl:
            pk = raw_cl[-1][-1]
            if abs(matched_dt[k] - matched_dt[pk]) <= 2.5 and abs((ta[k] - ta[pk]) - (tb[k] - tb[pk])) <= 3.0:
                raw_cl[-1].append(k)
                continue
        raw_cl.append([k])
    sh_cl = [cl for cl in raw_cl if len(cl) >= 3]

    base_ints: List[Tuple[int, int, bool, Optional[List[int]]]] = []
    cur = 0
    for cl in sh_cl:
        if cl[0] > cur:
            base_ints.append((cur, cl[0], False, None))
        base_ints.append((cl[0], cl[-1] + 1, True, cl))
        cur = cl[-1] + 1
    if cur < n_c:
        base_ints.append((cur, n_c, False, None))

    # 切分持续主场切换边界（例如中途换房或双角色镜头切换）
    dom_ints: List[Tuple[int, int, bool, Optional[List[int]]]] = []
    for ks, ke, is_sh, cl in base_ints:
        if is_sh or (ke - ks) < 16:
            dom_ints.append((ks, ke, is_sh, cl))
            continue
        sem_votes: List[Tuple[int, str]] = []
        for k in range(ks, ke):
            va = float(sc_a[k]) if not np.isnan(sc_a[k]) else -10.0
            vb = float(sc_b[k]) if not np.isnan(sc_b[k]) else -10.0
            if va >= -2.55 and va - vb >= 1.5:
                sem_votes.append((k, "A"))
            elif vb >= -2.55 and vb - va >= 1.5:
                sem_votes.append((k, "B"))
        flip_cuts = [ks]
        if len(sem_votes) >= 8:
            side_runs: List[Tuple[str, int, int, int]] = []
            for k, sd in sem_votes:
                if side_runs and side_runs[-1][0] == sd:
                    ps, fk, _, cnt = side_runs[-1]
                    side_runs[-1] = (ps, fk, k, cnt + 1)
                else:
                    side_runs.append((sd, k, k, 1))
            strong_runs = [r for r in side_runs if r[3] >= 4]
            for r_i in range(len(strong_runs) - 1):
                if strong_runs[r_i][0] != strong_runs[r_i + 1][0]:
                    cut_k = strong_runs[r_i + 1][1]
                    if cut_k - flip_cuts[-1] >= 4 and ke - cut_k >= 4:
                        flip_cuts.append(cut_k)
        flip_cuts.append(ke)
        for u in range(len(flip_cuts) - 1):
            dom_ints.append((flip_cuts[u], flip_cuts[u + 1], False, None))

    final_ints: List[Tuple[int, int, bool, Optional[List[int]]]] = []
    sim_bounds_A = sorted({float(s["A_st"]) for s in sim_segs})
    sim_bounds_B = sorted({float(s["B_st"]) for s in sim_segs})
    for ks, ke, is_sh, cl in dom_ints:
        if is_sh or (ke - ks) < 30:
            final_ints.append((ks, ke, is_sh, cl))
            continue
        sa_k = [float(sc_a[k]) for k in range(ks, ke) if not np.isnan(sc_a[k])]
        sb_k = [float(sc_b[k]) for k in range(ks, ke) if not np.isnan(sc_b[k])]
        use_B = (float(np.median(sb_k)) if sb_k else -10.0) > (float(np.median(sa_k)) if sa_k else -10.0)
        t_arr = tb if use_B else ta
        b_list = sim_bounds_B if use_B else sim_bounds_A
        t_cuts: List[int] = [ks]
        for b_t in b_list:
            last_k = t_cuts[-1]
            if ke - last_k < 24:
                break
            hit_k = next(
                (j for j in range(last_k + 12, ke - 12) if not np.isnan(t_arr[j]) and t_arr[j] >= b_t),
                None,
            )
            if hit_k is not None:
                t_cuts.append(hit_k)
        t_cuts.append(ke)
        for u in range(len(t_cuts) - 1):
            final_ints.append((t_cuts[u], t_cuts[u + 1], False, None))

    def _is_clear_home(vals: List[float]) -> bool:
        if not vals:
            return False
        if len(vals) < 4:
            return max(vals) >= -2.00
        return float(np.percentile(vals, 75)) >= -2.55 and float(np.median(vals)) >= -3.40

    seg_records: List[Dict[str, Any]] = []
    for ks, ke, is_sh, cl in final_ints:
        sa = [float(sc_a[k]) for k in range(ks, ke) if not np.isnan(sc_a[k])]
        sb = [float(sc_b[k]) for k in range(ks, ke) if not np.isnan(sc_b[k])]
        med_a = float(np.median(sa)) if sa else -10.0
        med_b = float(np.median(sb)) if sb else -10.0
        p75_a = float(np.percentile(sa, 75)) if sa else -10.0
        p75_b = float(np.percentile(sb, 75)) if sb else -10.0
        clr_a, clr_b = _is_clear_home(sa), _is_clear_home(sb)

        if is_sh and clr_a and clr_b:
            home = "SHARED"
            seg_type = f"SHARED(dt={float(np.median(matched_dt[cl])):+.2f}s)"
            ta_span = (float(np.min(ta[cl])), float(np.max(ta[cl])))
            tb_span = (float(np.min(tb[cl])), float(np.max(tb[cl])))
        elif clr_a and clr_b:
            home = "SHARED" if abs(med_a - med_b) <= 0.85 else ("A" if med_a >= med_b else "B")
            seg_type = "DUAL_CLEAR"
            ta_span = (float(np.nanmin(ta[ks:ke])) if sa else 0.0, float(np.nanmax(ta[ks:ke])) if sa else dur_A)
            tb_span = (float(np.nanmin(tb[ks:ke])) if sb else 0.0, float(np.nanmax(tb[ks:ke])) if sb else dur_B)
        elif clr_a:
            home, seg_type = "A", "SPLIT"
            ta_span, tb_span = None, None
        elif clr_b:
            home, seg_type = "B", "SPLIT"
            ta_span, tb_span = None, None
        else:
            home, seg_type = None, "SUBTHRESH"
            ta_span, tb_span = None, None

        seg_records.append({
            "k_start": ks, "k_end": ke - 1, "home": home, "raw_clear": (clr_a or clr_b),
            "seg_type": seg_type, "q_A": med_a, "q_B": med_b, "p75_A": p75_a, "p75_B": p75_b,
            "ta_span": ta_span, "tb_span": tb_span,
            "dt_med": float(np.median(matched_dt[cl])) if (is_sh and cl) else None,
        })

    cnt_a = sum(s["k_end"] - s["k_start"] + 1 for s in seg_records if s["raw_clear"] and s["home"] == "A")
    cnt_b = sum(s["k_end"] - s["k_start"] + 1 for s in seg_records if s["raw_clear"] and s["home"] == "B")
    pref_nonshared = "B" if cnt_b >= cnt_a else "A"
    for i, sg in enumerate(seg_records):
        if sg["home"] is not None:
            continue
        left_nh, right_nh = None, None
        for ni in range(i - 1, -1, -1):
            if seg_records[ni]["raw_clear"]:
                left_nh = seg_records[ni]["home"]
                break
        for ni in range(i + 1, len(seg_records)):
            if seg_records[ni]["raw_clear"]:
                right_nh = seg_records[ni]["home"]
                break
        if left_nh in ("A", "B") and right_nh in ("A", "B") and left_nh != right_nh:
            sg["home"] = "B" if sg["p75_B"] >= sg["p75_A"] else "A"
        elif left_nh in ("A", "B"):
            sg["home"] = left_nh
        elif right_nh in ("A", "B"):
            sg["home"] = right_nh
        elif left_nh == "SHARED" or right_nh == "SHARED":
            sg["home"] = ("A" if sg["p75_A"] > sg["p75_B"] else "B") if abs(sg["p75_A"] - sg["p75_B"]) >= 0.80 else pref_nonshared

    # 同步双麦高频拟声/音效救援（当全轨拟声偏多拉低 p75，但同步覆盖率或 p85 明确清晰时）
    if not any(sg["raw_clear"] for sg in seg_records):
        sa_all = [float(sc_a[k]) for k in range(n_c) if not np.isnan(sc_a[k])]
        sb_all = [float(sc_b[k]) for k in range(n_c) if not np.isnan(sc_b[k])]
        p85_a = float(np.percentile(sa_all, 85)) if sa_all else -10.0
        p85_b = float(np.percentile(sb_all, 85)) if sb_all else -10.0
        p75_a_all = float(np.percentile(sa_all, 75)) if sa_all else -10.0
        p75_b_all = float(np.percentile(sb_all, 75)) if sb_all else -10.0
        n_sync = int(np.sum(bc_flag))
        sync_cov = sum(
            max(0.0, min(float(s["A_en"]) - float(s["A_st"]), float(s["B_en"]) - float(s["B_st"])))
            for s in sim_segs
        ) / max(dur_A, dur_B, 1.0)
        if max(p85_a, p85_b) >= -2.20 or (sync_cov >= 0.70 and max(p85_a, p85_b) >= -2.60):
            home_fb = (
                "SHARED"
                if (n_sync >= 4 or sync_cov >= 0.70) and abs(p85_a - p85_b) <= 0.85
                else ("B" if (p85_b + p75_b_all) >= (p85_a + p75_a_all) else "A")
            )
            for sg in seg_records:
                sg["home"] = home_fb

    clause_sides: List[Optional[str]] = [None] * n_c
    for sg in seg_records:
        for k in range(sg["k_start"], sg["k_end"] + 1):
            clause_sides[k] = sg["home"]
    return clause_sides, seg_records


# ══════════════════════════════════════════════════════════════════════════
#  3. 主视角全量采纳与立体声单声源碰撞修复 (Refine Home Segments)
# ══════════════════════════════════════════════════════════════════════════

def _build_fg_masks(
    clauses_by_spk: Dict[str, List[int]],
    probe_by_spk: Dict[str, Dict[int, List[Dict[str, Any]]]],
    T_target: int,
    C_target: int,
    min_score: Optional[float],
    tok_dict: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[float, float]]]:
    fg_mask = np.zeros(T_target, dtype=np.float32)
    fg_char = np.zeros((T_target, C_target), dtype=np.float32)
    spans: List[Tuple[float, float]] = []
    for spk, cl in clauses_by_spk.items():
        for ci in cl:
            inf = ams._clause_summary(probe_by_spk[spk][ci])
            if inf["skipped"] or inf["dur"] <= 0.05:
                continue
            if min_score is not None and inf["score"] < min_score:
                continue
            spans.append((float(inf["st"]), float(inf["en"])))
            for r in probe_by_spk[spk][ci]:
                if r.get("skipped") or r.get("original_start") is None or r.get("original_end") is None:
                    continue
                f0 = max(0, min(T_target - 1, int(round(float(r["original_start"]) / FRAME_SEC))))
                f1 = max(f0 + 1, min(T_target, int(round(float(r["original_end"]) / FRAME_SEC))))
                fg_mask[f0:f1] = 1.0
                for ch in str(r.get("token", "")):
                    if ch in tok_dict:
                        fg_char[f0:f1, int(tok_dict[ch])] = 1.0
    return fg_mask, fg_char, spans


def refine_home_segments(
    active_home_spks: List[str],
    spk_home_clauses: Dict[str, List[int]],
    spk_all_clauses: Dict[str, List[int]],
    probe_target: Dict[str, Dict[int, List[Dict[str, Any]]]],
    away_raw_hits: Sequence[Tuple[float, float, float]],
    alignment_tokens: Sequence[Sequence[str]],
    line_group: Sequence[int],
    ono_clause_flags: Sequence[int],
    clause_surfaces: Optional[Sequence[str]],
    clause_speakers: Sequence[Any],
    ctx_target: Dict[str, Any],
    coop_target: torch.Tensor,
    solo_checker: _AudioSoloChecker,
) -> Dict[str, Dict[int, List[Dict[str, Any]]]]:
    """全量接受 Home/SHARED 块对齐结果，仅对落入单声源独占碰撞区的弱句在 ±15s 局部窗内修复。"""
    out = {s: copy.deepcopy(probe_target[s]) for s in active_home_spks}
    if not active_home_spks or (not away_raw_hits and len(active_home_spks) < 2):
        return out

    tok_dict = _get_tokenizer_dict(ctx_target)
    coop_np = coop_target.detach().cpu().numpy()
    em_L_np = ctx_target["emission_L"][0].detach().cpu().numpy() if "emission_L" in ctx_target else coop_np
    em_R_np = ctx_target["emission_R"][0].detach().cpu().numpy() if "emission_R" in ctx_target else coop_np
    blank_id = int(ctx_target.get("blank", 0))
    ignored = set(ctx_target.get("ignored_classes", ()))
    valid_cls = [c for c in range(coop_np.shape[1]) if c != blank_id and c not in ignored]
    T_all = int(coop_np.shape[0])
    max_nb = np.max(coop_np[:, valid_cls], axis=1).astype(np.float32)

    def _merge_spans(sps: Sequence[Tuple[float, float]], gap: float = 1.5) -> List[Tuple[float, float]]:
        res: List[Tuple[float, float]] = []
        for a, b in sorted(sps):
            if res and a <= res[-1][1] + gap:
                res[-1] = (res[-1][0], max(res[-1][1], b))
            else:
                res.append((a, b))
        return res

    solo_cache: Dict[Tuple[float, float], bool] = {}

    def _is_solo_cached(st_s: float, en_s: float) -> bool:
        key = (round(st_s, 2), round(en_s, 2))
        if key not in solo_cache:
            solo_cache[key] = solo_checker.is_clearly_solo(st_s, en_s)
        return solo_cache[key]

    def _skip_clause(ci: int) -> List[Dict[str, Any]]:
        return [{"token": t, "score": 0.0, "skipped": True, "ono_skipped": True, "line_index": ci} for t in alignment_tokens[ci]]

    for spk in active_home_spks:
        spk_hits: List[Tuple[float, float, float]] = list(away_raw_hits)
        for other_spk in active_home_spks:
            if other_spk == spk:
                continue
            for oci in spk_home_clauses[other_spk]:
                if ono_clause_flags[oci] != 0:
                    continue
                oinf = ams._clause_summary(probe_target[other_spk][oci])
                if not oinf["skipped"] and oinf["dur"] > 0.05 and oinf["score"] >= -2.30:
                    spk_hits.append((float(oinf["st"]), float(oinf["en"]), float(oinf["score"])))
        if not spk_hits:
            continue

        solo_spans = [(st_a, en_a) for st_a, en_a, _ in spk_hits if _is_solo_cached(st_a, en_a)]
        simul_spans = [(st_a, en_a) for st_a, en_a, _ in spk_hits if not _is_solo_cached(st_a, en_a)]
        m_solo = _merge_spans(solo_spans, 2.0)
        m_simul = _merge_spans(simul_spans, 1.2)
        all_claims = m_solo + m_simul
        raw_solo_spans = solo_spans

        l_mod, r_mod, st_mod = em_L_np.copy(), em_R_np.copy(), coop_np.copy()
        filler_logp = (max_nb - np.float32(1.35)).astype(np.float32)

        for st_s, en_s in m_solo:
            f0 = max(0, int(round((st_s - 0.05) / FRAME_SEC)))
            f1 = min(T_all, int(round((en_s + 0.05) / FRAME_SEC)))
            if f1 <= f0:
                continue
            fl = np.float32(-5.8)
            for arr in (l_mod, r_mod, st_mod):
                cur = arr[f0:f1, valid_cls]
                arr[f0:f1, valid_cls] = np.maximum(cur - np.float32(2.6), np.minimum(cur, fl))
            filler_logp[f0:f1] = np.maximum(
                filler_logp[f0:f1],
                np.maximum(coop_np[f0:f1, blank_id], max_nb[f0:f1] - np.float32(0.10)),
            )

        for st_s, en_s in m_simul:
            f0 = max(0, int(round((st_s - 0.05) / FRAME_SEC)))
            f1 = min(T_all, int(round((en_s + 0.05) / FRAME_SEC)))
            if f1 > f0:
                filler_logp[f0:f1] = np.maximum(
                    filler_logp[f0:f1],
                    np.maximum(coop_np[f0:f1, blank_id], max_nb[f0:f1] - np.float32(0.10)),
                )

        em_mod = torch.from_numpy(np.maximum(st_mod, np.maximum(l_mod, r_mod) - np.float32(0.18)).astype(np.float32))
        em_mod[:, blank_id] = torch.from_numpy(st_mod[:, blank_id])

        c_list = spk_home_clauses[spk]
        rank_map = {ci: rk for rk, ci in enumerate(spk_all_clauses[spk])}
        runs: List[List[int]] = []
        for ci in c_list:
            if runs and rank_map.get(ci, -9) == rank_map.get(runs[-1][-1], -99) + 1:
                runs[-1].append(ci)
            else:
                runs.append([ci])

        for run_cls in runs:
            sem_pos = [
                idx_c for idx_c, ci in enumerate(run_cls)
                if ono_clause_flags[ci] == 0 and not ams._clause_summary(out[spk][ci])["skipped"]
            ]
            if not sem_pos:
                continue
            valid_sts = [ams._clause_summary(out[spk][ci]) for ci in run_cls]
            valid_sts = [x for x in valid_sts if not x["skipped"] and x["dur"] > 0.05]
            if not valid_sts:
                continue
            run_lo = max(0.0, min(float(x["st"]) for x in valid_sts) - 15.0)
            run_hi = min(T_all * FRAME_SEC, max(float(x["en"]) for x in valid_sts) + 15.0)

            is_col: List[bool] = []
            is_short_col: List[bool] = []
            for p_idx in sem_pos:
                ci_p = run_cls[p_idx]
                inf = ams._clause_summary(out[spk][ci_p])
                n_phones = sum(len(t) for t in alignment_tokens[ci_p])
                ov_solo = any(min(ce, inf["en"]) - max(cs, inf["st"]) >= 0.25 for cs, ce in m_solo)
                short_wrapped_solo = (
                    n_phones <= 6 and float(inf["dur"]) <= 0.65
                    and any(
                        (min(ce, inf["en"]) - max(cs, inf["st"])) >= 0.60 * max(0.05, float(inf["dur"]))
                        and (ce - cs) >= 1.5 * max(0.05, float(inf["dur"]))
                        for cs, ce in raw_solo_spans
                    )
                )
                col_any = (inf["score"] < -2.30 and any(ce > inf["st"] - 1.0 and cs < inf["en"] + 1.0 for cs, ce in all_claims))
                col_solo = (inf["score"] < -1.80 and ov_solo) or short_wrapped_solo
                is_col.append(col_any or col_solo)
                is_short_col.append(short_wrapped_solo)

            k = 0
            while k < len(sem_pos):
                if not is_col[k]:
                    k += 1
                    continue
                k0 = k
                while (
                    k + 1 < len(sem_pos)
                    and is_col[k + 1]
                    and all(_norm_spk(clause_speakers[idx]) == spk for idx in range(run_cls[sem_pos[k]] + 1, run_cls[sem_pos[k + 1]]))
                ):
                    k += 1
                k1 = k
                k += 1

                left_idx = sem_pos[k0 - 1] if k0 > 0 else -1
                right_idx = sem_pos[k1 + 1] if k1 + 1 < len(sem_pos) else len(run_cls)
                w_lo = float(ams._clause_summary(out[spk][run_cls[left_idx]])["en"]) if left_idx >= 0 else run_lo
                w_hi = float(ams._clause_summary(out[spk][run_cls[right_idx]])["st"]) if right_idx < len(run_cls) else run_hi
                f_lo = max(0, int(np.ceil(w_lo / FRAME_SEC)))
                f_hi = min(T_all, int(np.floor(w_hi / FRAME_SEC)))
                if f_hi <= f_lo + 2:
                    continue

                sub_sem = [run_cls[p_idx] for p_idx in sem_pos[k0 : k1 + 1]]
                em_win = em_mod.clone()
                if f_lo > 0:
                    em_win[:f_lo, valid_cls] = -15.0
                    em_win[:f_lo, blank_id] = 0.0
                if f_hi < T_all:
                    em_win[f_hi:, valid_cls] = -15.0
                    em_win[f_hi:, blank_id] = 0.0

                realigned_blk = ams._align_subset_clauses(
                    sub_indices=sub_sem,
                    alignment_tokens=alignment_tokens,
                    line_group=line_group,
                    ono_clause_flags=ono_clause_flags,
                    enable_silence_snap=True,
                    acoustic_ctx=ctx_target,
                    custom_emission_2d=em_win,
                    enable_inter_line_filler=True,
                    filler_frame_logp=filler_logp,
                    enable_ono_skip=False,
                    ono_skip_token_penalty=0.5,
                    enable_word_skip=False,
                    clause_surfaces=clause_surfaces,
                    clause_speakers=clause_speakers,
                )
                old_scs = [_unmasked_clause_mean_score(out[spk][ci], coop_np, tok_dict) for ci in sub_sem]
                old_solo_ov = sum(
                    max(0.0, min(ce, float(ams._clause_summary(out[spk][ci])["en"])) - max(cs, float(ams._clause_summary(out[spk][ci])["st"])))
                    for ci in sub_sem for cs, ce in m_solo
                )
                new_scs: List[float] = []
                new_solo_ov = 0.0
                valid_blk = True
                for ci in sub_sem:
                    if ci not in realigned_blk or ams._clause_summary(realigned_blk[ci])["skipped"]:
                        valid_blk = False
                        break
                    new_scs.append(_unmasked_clause_mean_score(realigned_blk[ci], coop_np, tok_dict))
                    n_inf = ams._clause_summary(realigned_blk[ci])
                    new_solo_ov += sum(max(0.0, min(ce, float(n_inf["en"])) - max(cs, float(n_inf["st"]))) for cs, ce in m_solo)
                solo_margin = 1.80 if any(is_short_col[pi] for pi in range(k0, k1 + 1)) else 0.50
                cleared_solo = (
                    old_solo_ov >= 0.25
                    and new_solo_ov <= 0.10
                    and float(np.mean(new_scs)) >= max(-3.50, float(np.mean(old_scs)) - solo_margin)
                )
                if not valid_blk or (not cleared_solo and float(np.mean(new_scs)) <= float(np.mean(old_scs)) + 0.15):
                    continue

                for ci in sub_sem:
                    out[spk][ci] = realigned_blk[ci]

                sem_in_span = list(sem_pos[k0 : k1 + 1])
                anchors_idx = ([left_idx] if left_idx >= 0 else []) + sem_in_span + ([right_idx] if right_idx < len(run_cls) else [])
                for pos_i in range(max(0, left_idx + 1), min(len(run_cls), right_idx)):
                    ci_o = run_cls[pos_i]
                    if ono_clause_flags[ci_o] == 0:
                        continue
                    inf_o = ams._clause_summary(out[spk][ci_o])
                    if inf_o["skipped"]:
                        continue
                    prev_a = max((ai for ai in anchors_idx if ai < pos_i), default=None)
                    next_a = min((ai for ai in anchors_idx if ai > pos_i), default=None)
                    lo_t = float(ams._clause_summary(out[spk][run_cls[prev_a]])["st"]) if prev_a is not None else -1e9
                    hi_t = float(ams._clause_summary(out[spk][run_cls[next_a]])["st"]) if next_a is not None else 1e9
                    if inf_o["st"] < lo_t or inf_o["st"] > hi_t:
                        out[spk][ci_o] = _skip_clause(ci_o)
    return out


# ══════════════════════════════════════════════════════════════════════════
#  4. 客视角局部 Δt 小窗检索与过滤 (Align Away Speaker by Δt)
# ══════════════════════════════════════════════════════════════════════════

def _candidate_dts_at_time(t_h: float, sim_segs: Sequence[Dict[str, Any]], sign: float) -> List[float]:
    """从纯声学 SimSeg 提取主视角时刻 t_h 处的候选 Δt 列表。"""
    if not sim_segs:
        return []
    min_h = min(float(s["B_st"] if sign > 0 else s["A_st"]) for s in sim_segs)
    max_h = max(float(s["B_en"] if sign > 0 else s["A_en"]) for s in sim_segs)
    if t_h < min_h - 45.0 or t_h > max_h + 45.0:
        return []

    cands: List[float] = [
        float(sign * s["dt_AB"])
        for s in sim_segs
        if ((s["B_st"] if sign > 0 else s["A_st"]) - 15.0) <= t_h <= ((s["B_en"] if sign > 0 else s["A_en"]) + 15.0)
    ]
    has_active_cov = len(cands) > 0
    modes: List[float] = []
    for s in sorted(
        sim_segs,
        key=lambda x: min(
            abs(t_h - (x["B_st"] if sign > 0 else x["A_st"])),
            abs(t_h - (x["B_en"] if sign > 0 else x["A_en"])),
        ),
    ):
        st_s, en_s = (s["B_st"], s["B_en"]) if sign > 0 else (s["A_st"], s["A_en"])
        dist = 0.0 if st_s - 15.0 <= t_h <= en_s + 15.0 else min(abs(t_h - st_s), abs(t_h - en_s))
        is_exp = t_h > en_s + 15.0
        if dist > 650.0 or (has_active_cov and is_exp):
            continue
        if not any(abs(s["dt_AB"] - m_dt) <= 4.0 for m_dt in modes):
            modes.append(float(s["dt_AB"]))
            cands.append(float(sign * s["dt_AB"]))

    unique: List[float] = []
    for dt in cands:
        if not any(abs(dt - u) <= 0.8 for u in unique):
            unique.append(dt)
    return unique


def align_away_speaker_by_dt(
    c_away: Sequence[int],
    target_side: str,
    probe_home_spk: Dict[int, List[Dict[str, Any]]],
    sim_segs: Sequence[Dict[str, Any]],
    ctx_target: Dict[str, Any],
    channels_target: Sequence[np.ndarray],
    adv_target: np.ndarray,
    norm_p_home: np.ndarray,
    act_home: np.ndarray,
    tgt_norm_acts: Sequence[Tuple[np.ndarray, np.ndarray]],
    fg_mask: np.ndarray,
    fg_char_mask: np.ndarray,
    toks_all: Sequence[Sequence[str]],
    onos_all: Sequence[int],
    surfaces_all: Sequence[str],
    clause_tid_seqs: Dict[int, List[int]],
    is_final_pass: bool = True,
    eval_cache: Optional[Dict[Tuple[Any, ...], Tuple[Optional[Dict[str, Any]], List[Tuple[int, int]]]]] = None,
    fg_changed_cs: Optional[np.ndarray] = None,
) -> Tuple[Dict[int, List[Dict[str, Any]]], List[Tuple[float, float, float]]]:
    """在纯声学 SimSeg 提供的 Δt 局部小窗口内对齐隔壁视角子句并保留高置信度命中。"""
    sign = 1.0 if target_side == "A" else -1.0
    tok_dict = _get_tokenizer_dict(ctx_target)
    dur_target = float(channels_target[0].shape[0]) * FRAME_SEC
    out_map: Dict[int, List[Dict[str, Any]]] = {
        ci: [{"token": t, "score": 0.0, "skipped": True, "ono_skipped": True, "line_index": ci} for t in toks_all[ci]]
        for ci in c_away
    }

    if not sim_segs or not c_away:
        return out_map, []

    n_c = len(c_away)
    cand_dt_pool = sorted({round(float(sign * s["dt_AB"]), 2) for s in sim_segs})

    home_times = np.zeros((n_c, 2), dtype=np.float32)
    is_sem = np.zeros(n_c, dtype=bool)
    local_r_lists: List[List[Tuple[int, bool]]] = [[] for _ in range(n_c)]
    for k, ci in enumerate(c_away):
        tids = clause_tid_seqs[ci]
        toks = toks_all[ci]
        surf_str = surfaces_all[ci] if ci < len(surfaces_all) else "".join(toks)
        is_sfx = _is_pure_sfx_clause(surf_str, toks, tok_dict=tok_dict)
        is_sem[k] = (onos_all[ci] == 0 and not is_sfx and len(tids) >= 3)
        inf_h = ams._clause_summary(probe_home_spk[ci])
        if inf_h["skipped"] or not tids:
            continue
        hst, hen = float(inf_h["st"]), float(inf_h["en"])
        hmid = 0.5 * (hst + hen)
        home_times[k, 0], home_times[k, 1] = hst, hen
        local_dts = _candidate_dts_at_time(hmid, sim_segs, sign)
        cov_dts = [
            float(sign * s["dt_AB"])
            for s in sim_segs
            if ((s["B_st"] if sign > 0 else s["A_st"]) - 15.0) <= hmid <= ((s["B_en"] if sign > 0 else s["A_en"]) + 15.0)
        ]
        for r_idx, dt_cand in enumerate(cand_dt_pool):
            if any(abs(dt_cand - ldt) <= 1.2 for ldt in local_dts):
                is_cov = any(abs(dt_cand - cdt) <= 1.2 for cdt in cov_dts)
                local_r_lists[k].append((r_idx, is_cov))

    def _eval_at_dt(
        k: int, ci: int, hst: float, hen: float, dt_use: float, r_idx: int, is_cov: bool
    ) -> Optional[Dict[str, Any]]:
        if hst + dt_use < -1.0 or hen + dt_use > dur_target + 1.0:
            return None
        ckey = (ci, round(hst, 4), round(hen, 4), round(dt_use, 4), bool(is_sem[k]))
        if eval_cache is not None and fg_changed_cs is not None and ckey in eval_cache:
            cached_res, touched_wins = eval_cache[ckey]
            if all(int(fg_changed_cs[w1]) == int(fg_changed_cs[w0]) for (w0, w1) in touched_wins):
                if cached_res is None:
                    return None
                return {**cached_res, "k": k, "r_idx": r_idx, "is_cov": is_cov}

        toks, tids = toks_all[ci], clause_tid_seqs[ci]
        hmid = 0.5 * (hst + hen)
        pad = (3.5 if len(tids) > 4 else 2.2) if is_sem[k] else 1.8
        T_t_len = int(channels_target[0].shape[0])
        touched_wins: List[Tuple[int, int]] = [(
            max(0, int(round((hst + dt_use - pad) / FRAME_SEC))),
            min(T_t_len, int(round((hen + dt_use + pad) / FRAME_SEC))),
        )]

        def _store_cache(val: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            if eval_cache is not None and fg_changed_cs is None:
                eval_cache[ckey] = (val, touched_wins)
            return val

        vr = slice_ctc_verify_clause(
            toks, ctx_target, channels_target, adv_target, hst + dt_use, hen + dt_use, pad,
            fg_mask, fg_char_mask, tids, build_tokens=False,
        )
        if not vr["matched"]:
            return _store_cache(None)
        st_v, en_v = float(vr["start_sec"]), float(vr["end_sec"])
        dto = 0.5 * (st_v + en_v) - hmid
        sc = float(vr["score"])
        sc_bal = float(vr["sc_bal"])
        adv = float(vr["adv"])
        fov = float(vr["fg_overlap"])
        dur_ratio = float(vr["dur_ratio"])
        q_bal = sc_bal + 0.45 * adv
        dur_pen = 0.25 * abs(float(np.log(max(0.15, dur_ratio))))
        ph_s = _active_phoneme_sim(norm_p_home, act_home, tgt_norm_acts, hst, hen, dto) if is_sem[k] else 0.0
        q_eff = q_bal - 2.0 * fov - dur_pen + 1.5 * max(0.0, ph_s - 0.30)

        if is_sem[k]:
            if fov >= 0.85 or adv < 3.20 or q_bal < -3.95 or sc_bal < -6.00:
                return _store_cache(None)
        else:
            if fov > 0.45 or q_bal < -1.20 or adv < 2.30:
                return _store_cache(None)

        ctrl_qs: List[float] = []
        for sh in (-37.0, -21.0, 21.0, 37.0):
            if 2.0 <= hst + dto + sh and hen + dto + sh <= dur_target - 2.0:
                touched_wins.append((
                    max(0, int(round((hst + dto + sh - 2.0) / FRAME_SEC))),
                    min(T_t_len, int(round((hen + dto + sh + 2.0) / FRAME_SEC))),
                ))
                vc = slice_ctc_verify_clause(
                    toks, ctx_target, channels_target, adv_target, hst + dto + sh, hen + dto + sh, 2.0,
                    fg_mask, fg_char_mask, tids, build_tokens=False,
                )
                if vc["matched"]:
                    ctrl_qs.append(float(vc["sc_bal"]) + 0.45 * float(vc["adv"]))
        q_null = float(np.median(ctrl_qs)) if ctrl_qs else -3.5
        gain = q_bal - q_null

        if is_sem[k]:
            solo_clear = (sc_bal >= -4.55 and fov < 0.70 and (gain >= -0.20 or sc_bal >= -3.80))
            qual_ok = (q_eff >= -3.60 and (gain >= 0.20 or adv >= 4.70 or (ph_s >= 0.38 and gain >= -0.25)))
            if not (solo_clear or qual_ok):
                return _store_cache(None)
        else:
            if gain < 0.30:
                return _store_cache(None)

        res = {
            "k": k, "ci": ci, "r_idx": r_idx, "dt_used": float(dt_use), "is_cov": is_cov,
            "st": st_v, "en": en_v, "hmid": hmid, "dt": dto, "sc": sc, "sc_bal": sc_bal,
            "adv": adv, "dur_ratio": dur_ratio, "q": q_eff, "gain": gain, "fg_ov": fov,
            "ph_sim": ph_s, "is_sem": bool(is_sem[k]),
        }
        return _store_cache(res)

    cands_by_k_r: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for k, ci in enumerate(c_away):
        hst, hen = float(home_times[k, 0]), float(home_times[k, 1])
        if hen <= hst:
            continue
        for r_idx, is_cov in local_r_lists[k]:
            it = _eval_at_dt(k, ci, hst, hen, cand_dt_pool[r_idx], r_idx, is_cov)
            if it is not None:
                cands_by_k_r[(k, r_idx)] = it

    strong_hits = {(k, r): it for (k, r), it in cands_by_k_r.items() if it["is_sem"] and it["q"] >= -2.50}
    for k, ci in enumerate(c_away):
        hst, hen = float(home_times[k, 0]), float(home_times[k, 1])
        if hen <= hst:
            continue
        hmid = 0.5 * (hst + hen)
        for r_idx, is_cov in local_r_lists[k]:
            dt_cand = cand_dt_pool[r_idx]
            neigh_dts = [
                strong_hits[(j, r_idx)]["dt"]
                for j in range(max(0, k - 10), min(n_c, k + 11))
                if (j, r_idx) in strong_hits
                and abs(strong_hits[(j, r_idx)]["hmid"] - hmid) <= 45.0
                and abs(strong_hits[(j, r_idx)]["dt"] - dt_cand) <= 2.0
            ]
            if len(neigh_dts) < 2:
                continue
            dt_loc = float(np.median(neigh_dts))
            if abs(dt_loc - dt_cand) <= 0.25 or abs(dt_loc - dt_cand) > 2.0:
                continue
            it_loc = _eval_at_dt(k, ci, hst, hen, dt_loc, r_idx, is_cov)
            if it_loc is not None:
                it_prev = cands_by_k_r.get((k, r_idx))
                if it_prev is None or it_loc["q"] >= (it_prev["q"] - 0.1):
                    cands_by_k_r[(k, r_idx)] = it_loc

    def _chrono_ok(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        if not (a["st"] + 0.06 <= b["st"] and a["en"] <= b["en"] + 0.15):
            return False
        ov = min(a["en"], b["en"]) - max(a["st"], b["st"])
        if ov <= 0.0:
            return True
        min_dur = max(0.10, min(a["en"] - a["st"], b["en"] - b["st"]))
        return ov <= 0.50 * min_dur

    def _dedup_and_lis(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not items:
            return []
        best_unique: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
        for it in items:
            key = (it["k"], round(it["st"] * 20.0), round(it["en"] * 20.0))
            prev = best_unique.get(key)
            if prev is None or it["q"] > prev["q"]:
                best_unique[key] = it
        cand_list = sorted(best_unique.values(), key=lambda x: (x["k"], x["st"], x["r_idx"]))
        m_p = len(cand_list)
        w_base = np.zeros(m_p, dtype=np.float64)
        for i, it in enumerate(cand_list):
            w_base[i] = max(0.5, float(len(clause_tid_seqs[it["ci"]]))) + min(0.9, 0.10 * max(0.0, 5.0 + it["q"]))
        dp_w = w_base.copy()
        dp_bp = np.full(m_p, -1, dtype=np.int32)
        for i in range(m_p):
            for j in range(i):
                if (
                    cand_list[j]["k"] < cand_list[i]["k"]
                    and _chrono_ok(cand_list[j], cand_list[i])
                    and dp_w[j] + w_base[i] > dp_w[i]
                ):
                    dp_w[i], dp_bp[i] = dp_w[j] + w_base[i], j
        chosen_idx: List[int] = []
        cm = int(np.argmax(dp_w))
        while cm >= 0:
            chosen_idx.append(cm)
            cm = int(dp_bp[cm])
        chosen_idx.reverse()
        return [cand_list[i] for i in chosen_idx]

    kept_sem = _dedup_and_lis([it for it in cands_by_k_r.values() if it["is_sem"]])
    sfx_valid = [
        it for it in cands_by_k_r.values()
        if not it["is_sem"]
        and any(abs(it["st"] - a["st"]) <= 8.0 for a in kept_sem)
        and all(_chrono_ok(a, it) if a["k"] < it["k"] else _chrono_ok(it, a) for a in kept_sem)
    ]
    kept_all = sorted(kept_sem + _dedup_and_lis(sfx_valid), key=lambda x: x["k"])
    raw_hits: List[Tuple[float, float, float]] = []
    for it in kept_all:
        ci = it["ci"]
        if is_final_pass:
            hst, hen = float(home_times[it["k"], 0]), float(home_times[it["k"], 1])
            pad = (3.5 if len(clause_tid_seqs[ci]) > 4 else 2.2) if it["is_sem"] else 1.8
            vr_tok = slice_ctc_verify_clause(
                toks_all[ci], ctx_target, channels_target, adv_target,
                hst + it["dt_used"], hen + it["dt_used"], pad, fg_mask, fg_char_mask,
                clause_tid_seqs[ci], build_tokens=True,
            )
            if vr_tok["matched"]:
                tok_out = []
                for r in vr_tok["tokens"]:
                    r2 = dict(r)
                    r2["line_index"] = ci
                    tok_out.append(r2)
                out_map[ci] = tok_out
        if it["is_sem"]:
            raw_hits.append((it["st"], it["en"], it["sc"]))
    return out_map, raw_hits


# ══════════════════════════════════════════════════════════════════════════
#  5. 主编排管线与视角投影导出 (Pipeline Orchestration & Projection)
# ══════════════════════════════════════════════════════════════════════════

def run_multiperspective_pipeline(
    alignment_tokens: List[List[str]],
    line_group: Sequence[int],
    ono_clause_flags: Sequence[int],
    clause_speakers: Sequence[Any],
    ctx_A: Dict[str, Any],
    ctx_B: Dict[str, Any],
    enable_silence_snap: bool = True,
    enable_ono_gate: bool = True,
    log_callback: Optional[Callable[[str], None]] = None,
    enable_word_skip: bool = False,
    word_skip_token_penalty: float = 4.8,
    clause_surfaces: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[List[int]], Dict[str, Any]]:
    """执行四步多视角联合对齐，返回原生双视角 flat_A / flat_B 及其元信息。"""
    def log(msg: str) -> None:
        if log_callback:
            log_callback(msg)

    t_pipe0 = time.time()
    n_c = len(alignment_tokens)
    if n_c == 0:
        return [], [], {
            "flat_A": [], "flat_B": [], "dur_A_sec": 0.0, "dur_B_sec": 0.0,
            "u_A_st_sec": 0.0, "u_B_st_sec": 0.0, "delta_sec": 0.0,
            "sim_segs": [], "spk_persp_pref": {}, "spk_stats": {},
        }

    if clause_surfaces is None:
        clause_surfaces = ["".join(toks) for toks in alignment_tokens]

    tok_dict = _get_tokenizer_dict(ctx_A)
    clause_tid_seqs: Dict[int, List[int]] = {
        ci: [int(tok_dict[ch]) for t in alignment_tokens[ci] for ch in t if ch in tok_dict]
        for ci in range(n_c)
    }

    spk_to_clauses: Dict[str, List[int]] = {}
    norm_speakers: List[str] = []
    for ci in range(n_c):
        sp = _norm_spk(clause_speakers[ci] if ci < len(clause_speakers) else None)
        norm_speakers.append(sp)
        spk_to_clauses.setdefault(sp, []).append(ci)

    coop_A_t = ams.build_cooperative_stereo_emission(ctx_A, ear_bias=0.0)
    coop_B_t = ams.build_cooperative_stereo_emission(ctx_B, ear_bias=0.0)
    coop_A_np = coop_A_t.detach().cpu().numpy()
    coop_B_np = coop_B_t.detach().cpu().numpy()

    # ── 预热本地 Numba 核（确保多线程并发前已完成首次 JIT 加载，避免编译锁竞争）──
    global _NUMBA_WARMED_UP
    if _HAS_NUMBA and not _NUMBA_WARMED_UP:
        _dummy_em = np.zeros((10, max(4, coop_A_np.shape[1])), dtype=np.float32)
        _dummy_tid = np.array([1, 2], dtype=np.int32)
        _eval_ctc_channel_nb(_dummy_em, _dummy_em, False, _EMPTY_F2D, _dummy_tid, 5.0, 2, 8, 8, 0.25, float(FRAME_SEC))
        _backtrack_ctc_bounds_nb(_dummy_em, _dummy_em, False, _EMPTY_F2D, _dummy_tid, 8, 5)
        _NUMBA_WARMED_UP = True

    # ── 初始全角色探针（A/B 双视角 nogil 双线程并行解码，100% 逐比特无损）──
    probe_A: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}
    probe_B: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}
    if len(spk_to_clauses) >= 2:
        n_tok = sum(len(t) for t in alignment_tokens)

        def _probe_multi_side(ctx_side: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
            flat_init, _ = ams.align_multispeaker_or_reordered(
                audio_path="",
                alignment_tokens=alignment_tokens,
                line_group=list(line_group),
                ono_clause_flags=list(ono_clause_flags),
                clause_speakers=list(clause_speakers),
                lora_path=None,
                dev="auto",
                vram_chunk=6000,
                enable_silence_snap=enable_silence_snap,
                enable_ono_gate=enable_ono_gate,
                blank_punct_mask=[False] * (n_tok + 1),
                log_callback=None,
                enable_word_skip=enable_word_skip,
                word_skip_token_penalty=word_skip_token_penalty,
                acoustic_ctx=ctx_side,
                clause_surfaces=clause_surfaces,
            )
            return ams._group_results_by_clause(alignment_tokens, flat_init)

        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_A = ex.submit(_probe_multi_side, ctx_A)
            fut_B = ex.submit(_probe_multi_side, ctx_B)
            grp_A = fut_A.result()
            grp_B = fut_B.result()
        for sp, cl in spk_to_clauses.items():
            probe_A[sp] = {ci: grp_A[ci] for ci in cl}
            probe_B[sp] = {ci: grp_B[ci] for ci in cl}
    else:
        for sp, cl in spk_to_clauses.items():
            def _probe_single_side(ctx_side: Dict[str, Any], coop_t: torch.Tensor) -> Dict[int, List[Dict[str, Any]]]:
                return ams._align_subset_clauses(
                    sub_indices=cl,
                    alignment_tokens=alignment_tokens,
                    line_group=list(line_group),
                    ono_clause_flags=list(ono_clause_flags),
                    enable_silence_snap=enable_silence_snap,
                    acoustic_ctx=ctx_side,
                    custom_emission_2d=coop_t,
                    enable_ono_skip=enable_ono_gate,
                    enable_word_skip=enable_word_skip,
                    word_skip_token_penalty=word_skip_token_penalty,
                    clause_surfaces=clause_surfaces,
                    clause_speakers=list(clause_speakers),
                )

            with ThreadPoolExecutor(max_workers=2) as ex:
                fut_A = ex.submit(_probe_single_side, ctx_A, coop_A_t)
                fut_B = ex.submit(_probe_single_side, ctx_B, coop_B_t)
                probe_A[sp] = fut_A.result()
                probe_B[sp] = fut_B.result()

    spk_stats: Dict[str, Dict[str, Any]] = {}
    for sp, cl in spk_to_clauses.items():
        scA = [ams._clause_summary(probe_A[sp][ci])["score"] for ci in cl if ono_clause_flags[ci] == 0 and not ams._clause_summary(probe_A[sp][ci])["skipped"]]
        scB = [ams._clause_summary(probe_B[sp][ci])["score"] for ci in cl if ono_clause_flags[ci] == 0 and not ams._clause_summary(probe_B[sp][ci])["skipped"]]
        mA = float(np.median(scA)) if scA else -10.0
        mB = float(np.median(scB)) if scB else -10.0
        spk_stats[sp] = {"sc_A": mA, "sc_B": mB, "home": "A" if mA >= mB else "B"}

    adv_A_np = compute_phoneme_advantage_matrix(coop_A_np, ctx_A)
    adv_B_np = compute_phoneme_advantage_matrix(coop_B_np, ctx_B)
    norm_p_A, act_A = _prep_phoneme_probs(coop_A_np, ctx_A)
    norm_p_B, act_B = _prep_phoneme_probs(coop_B_np, ctx_B)
    em_AL = ctx_A["emission_L"][0].detach().cpu().numpy() if "emission_L" in ctx_A else None
    em_AR = ctx_A["emission_R"][0].detach().cpu().numpy() if "emission_R" in ctx_A else None
    em_BL = ctx_B["emission_L"][0].detach().cpu().numpy() if "emission_L" in ctx_B else None
    em_BR = ctx_B["emission_R"][0].detach().cpu().numpy() if "emission_R" in ctx_B else None
    ch_A = [coop_A_np] + [x for x in (em_AL, em_AR) if x is not None]
    ch_B = [coop_B_np] + [x for x in (em_BL, em_BR) if x is not None]
    acts_A = [_prep_phoneme_probs(x, ctx_A) for x in ch_A]
    acts_B = [_prep_phoneme_probs(x, ctx_B) for x in ch_B]

    # ── Step 1: 纯声学相似段 SimSeg 检测 ───────────────────────────────────
    sim_segs = detect_clean_audio_segments(ctx_A, ctx_B)
    if sim_segs:
        seg_desc = ", ".join(
            f"{s['dt_AB']:+.2f}s(A={s['A_st']:.0f}..{s['A_en']:.0f},B={s['B_st']:.0f}..{s['B_en']:.0f})"
            for s in sim_segs[:8]
        )
        log(f"  🔗 [Step 1 声学同步段] {len(sim_segs)} 段: {seg_desc}{' ...' if len(sim_segs) > 8 else ''}")
    else:
        log("  🔗 [Step 1 声学同步段] 未检测到高相似同步段 (各自独立视角处理)")

    # ── Step 2: 角色分块与主视角判定 ───────────────────────────────────────
    spk_sides: Dict[str, List[Optional[str]]] = {}
    for spk, cl in spk_to_clauses.items():
        sides, segs = segment_and_assign_home(
            spk, cl, probe_A[spk], probe_B[spk],
            alignment_tokens, ono_clause_flags, clause_tid_seqs,
            norm_p_A, act_A, norm_p_B, act_B, coop_A_np, coop_B_np, sim_segs,
        )
        spk_sides[spk] = sides
        chunk_desc = ", ".join(
            f"[{s['k_start']}..{s['k_end']}]({s['seg_type']})->{s['home']}"
            for s in segs
        )
        log(f"  🎭 [Step 2 主视角切块] [{spk}]: {chunk_desc}")

    # ── Step 3: Home 全量采纳与纯 STFT 单声源碰撞局部修复 (A/B 双线程并行) ───
    refined_probes: Dict[str, Dict[str, Dict[int, List[Dict[str, Any]]]]] = {}
    side_meta: Dict[str, Dict[str, Any]] = {}

    def _step3_one_side(target_side: str) -> Tuple[str, Dict[str, Dict[int, List[Dict[str, Any]]]], Dict[str, Any]]:
        ctx_t = ctx_A if target_side == "A" else ctx_B
        coop_t_t = coop_A_t if target_side == "A" else coop_B_t
        coop_t_np = coop_A_np if target_side == "A" else coop_B_np
        ch_t = ch_A if target_side == "A" else ch_B
        adv_t = adv_A_np if target_side == "A" else adv_B_np
        acts_t = acts_A if target_side == "A" else acts_B
        probe_t = probe_A if target_side == "A" else probe_B
        probe_h_other = probe_B if target_side == "A" else probe_A
        np_h_oth, act_h_oth = (norm_p_B, act_B) if target_side == "A" else (norm_p_A, act_A)
        T_t, C_t = coop_t_np.shape

        home_cls = {s: [cl[k] for k, sd in enumerate(spk_sides[s]) if sd in (target_side, "SHARED")] for s, cl in spk_to_clauses.items()}
        away_cls = {s: [cl[k] for k, sd in enumerate(spk_sides[s]) if sd not in (target_side, "SHARED")] for s, cl in spk_to_clauses.items()}
        active_home = [s for s, cl in home_cls.items() if cl]
        away_spks = [s for s, cl in away_cls.items() if cl]

        fg0, fgc0, _ = _build_fg_masks(home_cls, probe_t, T_t, C_t, min_score=-2.05, tok_dict=tok_dict)
        ref_spans = [
            (ams._clause_summary(probe_t[s][ci])["st"], ams._clause_summary(probe_t[s][ci])["en"])
            for s in active_home for ci in home_cls[s]
            if ono_clause_flags[ci] == 0 and not ams._clause_summary(probe_t[s][ci])["skipped"]
        ]
        solo_chk = _AudioSoloChecker(ctx_t.get("_stereo_wav"), ref_spans)

        away_hits_p1: List[Tuple[float, float, float]] = []
        away_eval_cache: Dict[Tuple[Any, ...], Tuple[Optional[Dict[str, Any]], List[Tuple[int, int]]]] = {}
        for s in away_spks:
            _, hits_s = align_away_speaker_by_dt(
                away_cls[s], target_side, probe_h_other[s], sim_segs,
                ctx_t, ch_t, adv_t, np_h_oth, act_h_oth, acts_t, fg0, fgc0,
                alignment_tokens, ono_clause_flags, clause_surfaces, clause_tid_seqs,
                is_final_pass=False, eval_cache=away_eval_cache,
            )
            away_hits_p1.extend(hits_s)

        ref_probe = refine_home_segments(
            active_home, home_cls, spk_to_clauses, probe_t, away_hits_p1,
            alignment_tokens, line_group, ono_clause_flags, clause_surfaces, clause_speakers,
            ctx_t, coop_t_t, solo_chk,
        )
        fg_full, fgc_full, _ = _build_fg_masks(home_cls, ref_probe, T_t, C_t, min_score=-2.35, tok_dict=tok_dict)
        fg_diff_cs = np.concatenate([[0], np.cumsum(((fg0 != fg_full) | np.any(fgc0 != fgc_full, axis=1)).astype(np.int32))])
        meta_s = {
            "home_cls": home_cls, "away_cls": away_cls, "active_home": active_home,
            "away_spks": away_spks, "fg": fg_full, "fgc": fgc_full,
            "eval_cache": away_eval_cache, "fg_diff_cs": fg_diff_cs,
        }
        return target_side, ref_probe, meta_s

    with ThreadPoolExecutor(max_workers=2) as ex:
        for t_side, ref_p, m_s in ex.map(_step3_one_side, ("A", "B")):
            refined_probes[t_side] = ref_p
            side_meta[t_side] = m_s

    # ── Step 4: 客视角局部 Δt 对齐与原生双视角输出组装 (A/B 双线程并行) ───────
    streams = list(spk_to_clauses.values())
    spk_persp_pref = {s: float(spk_stats[s]["sc_B"] - spk_stats[s]["sc_A"]) for s in spk_to_clauses}
    flat_by_side: Dict[str, List[Dict[str, Any]]] = {}

    def _step4_one_side(target_side: str) -> Tuple[str, List[Dict[str, Any]], List[str]]:
        opp_side = "B" if target_side == "A" else "A"
        meta = side_meta[target_side]
        ctx_t = ctx_A if target_side == "A" else ctx_B
        ch_t = ch_A if target_side == "A" else ch_B
        adv_t = adv_A_np if target_side == "A" else adv_B_np
        acts_t = acts_A if target_side == "A" else acts_B
        np_h_oth, act_h_oth = (norm_p_B, act_B) if target_side == "A" else (norm_p_A, act_A)
        side_logs: List[str] = []

        final_map: Dict[int, List[Dict[str, Any]]] = {}
        for s in meta["active_home"]:
            for ci in meta["home_cls"][s]:
                final_map[ci] = copy.deepcopy(refined_probes[target_side][s][ci])

        for s in meta["away_spks"]:
            probe_h = refined_probes[opp_side].get(s, (probe_B if opp_side == "B" else probe_A)[s])
            amap, _ = align_away_speaker_by_dt(
                meta["away_cls"][s], target_side, probe_h, sim_segs,
                ctx_t, ch_t, adv_t, np_h_oth, act_h_oth, acts_t,
                meta["fg"], meta["fgc"],
                alignment_tokens, ono_clause_flags, clause_surfaces, clause_tid_seqs,
                is_final_pass=True, eval_cache=meta["eval_cache"], fg_changed_cs=meta["fg_diff_cs"],
            )
            full_s = spk_to_clauses[s]
            home_set_s = set(meta["home_cls"].get(s, ()))
            home_st = {}
            for ci_s in full_s:
                if ci_s in home_set_s:
                    ih = ams._clause_summary(final_map[ci_s])
                    if not ih["skipped"]:
                        home_st[ci_s] = float(ih["st"])
            prev_h = -1e9
            next_h_map: Dict[int, float] = {}
            nh = 1e9
            for ci_s in reversed(full_s):
                if ci_s in home_st:
                    nh = home_st[ci_s]
                next_h_map[ci_s] = nh
            last_st = -1e9
            for ci_s in full_s:
                if ci_s in home_st:
                    prev_h = home_st[ci_s]
                    last_st = max(last_st, prev_h)
                elif ci_s in amap:
                    iaw = ams._clause_summary(amap[ci_s])
                    if not iaw["skipped"]:
                        st_aw = float(iaw["st"])
                        if st_aw <= max(prev_h, last_st) + 0.05 or st_aw >= next_h_map[ci_s] - 0.05:
                            amap[ci_s] = [
                                {"token": t, "score": 0.0, "skipped": True, "ono_skipped": True, "line_index": ci_s}
                                for t in alignment_tokens[ci_s]
                            ]
                        else:
                            last_st = st_aw
            kept_n = sum(1 for ci in amap if not ams._clause_summary(amap[ci])["skipped"])
            side_logs.append(f"  🎯 [Step 4 -> 视角 {target_side}] 客视角 [{s}] 保留 {kept_n}/{len(meta['away_cls'][s])} 句")
            final_map.update(amap)

        flat_side: List[Dict[str, Any]] = []
        for ci in range(n_c):
            if ci in final_map:
                flat_side.extend(final_map[ci])
            else:
                flat_side.extend(
                    {"token": t, "score": 0.0, "skipped": True, "ono_skipped": True, "line_index": ci}
                    for t in alignment_tokens[ci]
                )
        return target_side, flat_side, side_logs

    with ThreadPoolExecutor(max_workers=2) as ex:
        for t_side, fl_s, s_logs in ex.map(_step4_one_side, ("A", "B")):
            flat_by_side[t_side] = fl_s
            for msg in s_logs:
                log(msg)

    dur_A_sec = float(coop_A_np.shape[0] * FRAME_SEC)
    dur_B_sec = float(coop_B_np.shape[0] * FRAME_SEC)
    if sim_segs:
        # 按权重取主要同步段的典型 Δt 仅作日志展示参考
        best_seg = max(sim_segs, key=lambda s: float(s.get("weight", 0.0)))
        delta_sec = float(best_seg["dt_AB"])
    else:
        delta_sec = 0.0

    log(f"  ✅ 多视角核心计算完成 (耗时 {time.time() - t_pipe0:.1f}s)")
    mp_meta: Dict[str, Any] = {
        "flat_A": flat_by_side["A"],
        "flat_B": flat_by_side["B"],
        "dur_A_sec": dur_A_sec,
        "dur_B_sec": dur_B_sec,
        "u_A_st_sec": 0.0,
        "u_B_st_sec": 0.0,
        "delta_sec": delta_sec,
        "sim_segs": sim_segs,
        "spk_persp_pref": spk_persp_pref,
        "spk_stats": spk_stats,
        "probe_streams": streams,
    }
    return flat_by_side["A"], streams, mp_meta


# ══════════════════════════════════════════════════════════════════════════
#  6. 按视角时间轴投影与越界裁剪导出 (Perspective Projection & Clipping)
# ══════════════════════════════════════════════════════════════════════════

def project_result_list_to_perspective(
    master_result_list: List[Dict[str, Any]],
    perspective_origin_sec: float,
    perspective_duration_sec: float,
    boundary_tolerance_sec: float = 0.35,
    perspective_side: str = "A",
    spk_persp_pref: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """将 result_list 裁剪到指定视角时间坐标系 [0, perspective_duration_sec] 并规范同屏左右顺序。

    - 若整行台词全部落在 [-boundary_tolerance_sec, perspective_duration_sec + boundary_tolerance_sec] 之外，
      自动标记为遗弃以防越界词条堆在开头或结尾。
    - 对于同屏多角色并列行（以全角空格连接），按起音顺序及该视角前景主次自动排列左右位置。
    """
    cloned = copy.deepcopy(master_result_list)
    n = len(cloned)
    line_start = 0
    side_is_B = (perspective_side.upper() == "B")

    for i in range(n + 1):
        is_nl = (i == n) or (cloned[i].get("orig") == "\n")
        if not is_nl:
            continue
        line_end = i
        if line_end > line_start:
            sts: List[float] = []
            ens: List[float] = []
            for k in range(line_start, line_end):
                it = cloned[k]
                if it.get(F.ONO_DROPPED):
                    continue
                st_s = ams._parse_lrc_ts_sec(it.get("start"))
                en_s = ams._parse_lrc_ts_sec(it.get("end"))
                if st_s is not None:
                    sts.append(st_s - perspective_origin_sec)
                if en_s is not None:
                    ens.append(en_s - perspective_origin_sec)

            if sts:
                min_st = min(sts)
                max_en = max(ens) if ens else min_st
                out_of_bounds = (
                    min_st < -0.15
                    or max_en < 0.05
                    or min_st > perspective_duration_sec - 0.05
                    or max_en > perspective_duration_sec + boundary_tolerance_sec
                )
                if out_of_bounds:
                    for k in range(line_start, line_end):
                        cloned[k].pop("start", None)
                        cloned[k].pop("end", None)
                        cloned[k][F.ONO_DROPPED] = True
                else:
                    for k in range(line_start, line_end):
                        it = cloned[k]
                        st_s = ams._parse_lrc_ts_sec(it.get("start"))
                        en_s = ams._parse_lrc_ts_sec(it.get("end"))
                        if st_s is not None:
                            new_st = max(0.0, min(perspective_duration_sec, st_s - perspective_origin_sec))
                            it["start"] = _format_time(new_st)
                        if en_s is not None:
                            new_en = max(0.0, min(perspective_duration_sec, en_s - perspective_origin_sec))
                            it["end"] = _format_time(new_en)

                    if spk_persp_pref:
                        segments: List[Tuple[str, List[Dict[str, Any]]]] = []
                        cur_spk_clean: Optional[str] = None
                        cur_seg: List[Dict[str, Any]] = []
                        for k in range(line_start, line_end):
                            it = cloned[k]
                            sp_tup = it.get(F.SPK)
                            sp_clean = str(sp_tup[0]).strip().lstrip("　") if sp_tup else (cur_spk_clean or "")
                            if cur_spk_clean is not None and sp_clean != cur_spk_clean and cur_seg:
                                segments.append((cur_spk_clean, cur_seg))
                                cur_seg = [it]
                                cur_spk_clean = sp_clean
                            else:
                                cur_spk_clean = sp_clean
                                cur_seg.append(it)
                        if cur_seg:
                            segments.append((cur_spk_clean or "", cur_seg))

                        if len(segments) >= 2:
                            seg_spans: List[Tuple[float, float]] = []
                            for _, seg_items in segments:
                                s_vals = [
                                    ams._parse_lrc_ts_sec(x.get("start"))
                                    for x in seg_items
                                    if not x.get(F.ONO_DROPPED) and x.get("start")
                                ]
                                e_vals = [
                                    ams._parse_lrc_ts_sec(x.get("end"))
                                    for x in seg_items
                                    if not x.get(F.ONO_DROPPED) and x.get("end")
                                ]
                                s_vals_clean = [v for v in s_vals if v is not None]
                                e_vals_clean = [v for v in e_vals if v is not None]
                                s_min = min(s_vals_clean) if s_vals_clean else min_st
                                e_max = max(e_vals_clean) if e_vals_clean else (s_min + 0.1)
                                seg_spans.append((s_min, e_max))

                            import functools

                            def _cmp_segs(a_info: Tuple[int, str], b_info: Tuple[int, str]) -> int:
                                idx_a, sp_a = a_info
                                idx_b, sp_b = b_info
                                st_a, en_a = seg_spans[idx_a]
                                st_b, en_b = seg_spans[idx_b]

                                if en_a <= st_b + 0.15 and st_a < st_b:
                                    return -1
                                elif en_b <= st_a + 0.15 and st_b < st_a:
                                    return 1

                                if abs(st_a - st_b) >= 0.40:
                                    return -1 if st_a < st_b else 1

                                pref_a = float(spk_persp_pref.get(sp_a, 0.0))
                                pref_b = float(spk_persp_pref.get(sp_b, 0.0))
                                score_a = pref_a if side_is_B else -pref_a
                                score_b = pref_b if side_is_B else -pref_b
                                if abs(score_a - score_b) > 1e-4:
                                    return -1 if score_a > score_b else 1

                                return -1 if st_a < st_b else (1 if st_a > st_b else (idx_a - idx_b))

                            sorted_indices = sorted(
                                [(idx_s, segments[idx_s][0]) for idx_s in range(len(segments))],
                                key=functools.cmp_to_key(_cmp_segs),
                            )
                            sorted_segs = [segments[idx_s] for idx_s, _ in sorted_indices]
                            rebuilt_line: List[Dict[str, Any]] = []
                            for seg_idx, (sp_c, seg_items) in enumerate(sorted_segs):
                                for it in seg_items:
                                    sp_tup = it.get(F.SPK)
                                    if sp_tup:
                                        base_sp0 = str(sp_tup[0]).strip().lstrip("　")
                                        new_sp0 = base_sp0 if seg_idx == 0 else ("　" + base_sp0)
                                        it[F.SPK] = (new_sp0,) + tuple(sp_tup[1:])
                                    rebuilt_line.append(it)
                            first_st_str = _format_time(max(0.0, min(perspective_duration_sec, min_st)))
                            last_en_str = _format_time(max(0.0, min(perspective_duration_sec, max_en)))
                            surviving_toks = [
                                it for it in rebuilt_line
                                if not it.get(F.ONO_DROPPED) and ("start" in it or "end" in it)
                            ]
                            if surviving_toks:
                                surviving_toks[0]["start"] = first_st_str
                                surviving_toks[-1]["end"] = last_en_str
                                for _, seg_items in sorted_segs:
                                    s_running = max(0.0, min(perspective_duration_sec, min_st))
                                    for it in seg_items:
                                        if it.get(F.ONO_DROPPED):
                                            continue
                                        st_v = ams._parse_lrc_ts_sec(it.get("start"))
                                        en_v = ams._parse_lrc_ts_sec(it.get("end"))
                                        if st_v is not None and st_v < s_running:
                                            it["start"] = _format_time(s_running)
                                        elif st_v is not None:
                                            s_running = st_v
                                        if en_v is not None and en_v < s_running:
                                            it["end"] = _format_time(s_running)
                                        elif en_v is not None:
                                            s_running = min(max_en, en_v)
                                surviving_toks[0]["start"] = first_st_str
                                surviving_toks[-1]["end"] = last_en_str
                            cloned[line_start:line_end] = rebuilt_line
        line_start = i + 1

    return cloned


def align_multiperspective_joint(
    audio_path_A: str | Path,
    audio_path_B: str | Path,
    alignment_tokens: List[List[str]],
    line_group: Sequence[int],
    ono_clause_flags: Sequence[int],
    clause_speakers: Sequence[Any],
    lora_path: Optional[str] = None,
    dev: str = "auto",
    vram_chunk: int = 6000,
    enable_silence_snap: bool = True,
    enable_ono_gate: bool = True,
    log_callback: Optional[Callable[[str], None]] = None,
    enable_word_skip: bool = False,
    word_skip_token_penalty: float = 4.8,
    clause_surfaces: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[List[int]], Dict[str, Any]]:
    """从两个视角的音频文件提取声学上下文（含 STFT 立体声波），并执行四步多视角动态联合对齐。"""
    def log(msg: str) -> None:
        if log_callback:
            log_callback(msg)

    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_wA = ex.submit(_load_audio_16k_once, audio_path_A)
        fut_wB = ex.submit(_load_audio_16k_once, audio_path_B)
        wav_A_t, stereo_A_np = fut_wA.result()
        wav_B_t, stereo_B_np = fut_wB.result()

    log(f"  🎧 [视角 A 声学提取] {Path(audio_path_A).name}")
    ctx_A = ams.extract_acoustic_context(
        audio_file_path=wav_A_t if wav_A_t is not None else audio_path_A,
        lora_path=lora_path,
        device=dev,
        chunk_frames=int(vram_chunk),
    )
    ctx_A["_stereo_wav"] = stereo_A_np

    log(f"  🎧 [视角 B 声学提取] {Path(audio_path_B).name}")
    ctx_B = ams.extract_acoustic_context(
        audio_file_path=wav_B_t if wav_B_t is not None else audio_path_B,
        lora_path=lora_path,
        device=dev,
        chunk_frames=int(vram_chunk),
    )
    ctx_B["_stereo_wav"] = stereo_B_np

    return run_multiperspective_pipeline(
        alignment_tokens=alignment_tokens,
        line_group=line_group,
        ono_clause_flags=ono_clause_flags,
        clause_speakers=clause_speakers,
        ctx_A=ctx_A,
        ctx_B=ctx_B,
        enable_silence_snap=enable_silence_snap,
        enable_ono_gate=enable_ono_gate,
        log_callback=log_callback,
        enable_word_skip=enable_word_skip,
        word_skip_token_penalty=word_skip_token_penalty,
        clause_surfaces=clause_surfaces,
    )
