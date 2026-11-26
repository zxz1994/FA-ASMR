# -*- coding: utf-8 -*-
"""FA-ASMR V2 — 多视角声学特征、Numba CTC 切片验证与纯声学同步段对齐。"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)

try:
    import numba as _nb
    _HAS_NUMBA = True
except Exception as _nb_err:
    _nb = None
    _HAS_NUMBA = False
    _nb_msg = f"  ⚠ [align_multiperspective] Numba 不可用 ({type(_nb_err).__name__}: {_nb_err})，已降级为纯 Python 执行（速度将明显下降）"
    logger.warning(_nb_msg)
    try:
        print(_nb_msg, file=sys.stderr)
    except Exception:
        pass

from fa_asmr.aligners.align_acoustic import FRAME_SEC
from fa_asmr.aligners.align_utils import _format_time
from fa_asmr.text.ono_block_corpus import is_spoken_interjection

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


def _load_audio_16k_once(audio_file_path: str | Path) -> Tuple[torch.Tensor, np.ndarray]:
    """Load and resample audio once into 16kHz tensor (for MMS-FA) and (2, N) float32 numpy (for STFT)."""
    try:
        wav, sr = torchaudio.load(str(audio_file_path))
        if wav.numel() == 0 or wav.shape[-1] == 0:
            raise ValueError("音频波形为空（0 采样）")
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        elif wav.shape[0] > 2:
            wav = wav[:2]
        if int(sr) != 16000:
            wav = torchaudio.functional.resample(wav, int(sr), 16000)
        wav = wav.float()
        stereo_wav = wav.repeat(2, 1) if wav.shape[0] == 1 else wav
        return wav, stereo_wav.detach().cpu().numpy().astype(np.float32)
    except Exception as e:
        raise RuntimeError(
            f"加载双视角音频失败 ({audio_file_path}): {type(e).__name__}: {e}"
        ) from e


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
    if surface and is_spoken_interjection(surface):
        return False
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



from fa_asmr.aligners.multispeaker_acoustic import (  # noqa: F401
    _AudioSoloChecker,
    _unmasked_clause_mean_score,
    build_cooperative_stereo_emission,
)
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
    coop_em = build_cooperative_stereo_emission(ctx, ear_bias=0.0).detach().cpu().numpy()
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




def ensure_numba_warmed_up(n_classes: int) -> None:
    """预热本地 Numba 核（确保多线程并发前已完成首次 JIT 加载，避免编译锁竞争）。"""
    global _NUMBA_WARMED_UP
    if _HAS_NUMBA and not _NUMBA_WARMED_UP:
        _dummy_em = np.zeros((10, max(4, int(n_classes))), dtype=np.float32)
        _dummy_tid = np.array([1, 2], dtype=np.int32)
        _eval_ctc_channel_nb(_dummy_em, _dummy_em, False, _EMPTY_F2D, _dummy_tid, 5.0, 2, 8, 8, 0.25, float(FRAME_SEC))
        _backtrack_ctc_bounds_nb(_dummy_em, _dummy_em, False, _EMPTY_F2D, _dummy_tid, 8, 5)
        _NUMBA_WARMED_UP = True
