# -*- coding: utf-8 -*-
"""FA-ASMR V2 — 多视角联合对齐与动态融合引擎 (Multi-Perspective Joint Alignment)

设计原则（完全隔离，不修改 align.py 与 align_trellis.py）：
1. 默认开启并复用 align_multispeaker.py 的多人分流与乱序重排能力。
2. 自动测算两视角音频之间的时间轴错位 Δt = t_A - t_B（结合 CTC 音素发射矩阵 FFT 互相关、
   滑窗能量/声学包络互相关与高置信度公共台词锚点）。
3. 构建覆盖两视角并集的联合主时间轴 (Union Master Timeline)，绝不丢失任一视角的独有首尾片段。
4. 无角色-音轨硬绑定：通过子句级动态视角质量评估 + 跨视角互斥占用 (Perspective-Aware Claims)，
   在每一帧/每一子句自动向更清晰的视角动态倾斜，完美支持音频中途换人、串房或同场交换。
5. 一次联合对齐后，自动按各自视角的时间坐标系平移并裁剪越界台词，同时生成两视角的高精度字幕。
"""
from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from align_acoustic import (
    FRAME_SEC,
    build_next_stable_silence_end,
    compute_track_noise_floor,
)
from align_post import _post_process_tokens
from align_utils import _format_time
import align_multispeaker as ams
from clause import F


# ══════════════════════════════════════════════════════════════════════════
#  1. 跨视角时间偏移 Δt 自动校准 (Offset Calibration)
# ══════════════════════════════════════════════════════════════════════════

def _standardize_2d(mat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """按列（特征维）标准化矩阵 (T, D) 为零均值单位方差。"""
    mean = np.mean(mat, axis=0, keepdims=True)
    std = np.std(mat, axis=0, keepdims=True)
    return ((mat - mean) / (std + eps)).astype(np.float32)


def _extract_perspective_feature_matrix(acoustic_ctx: Dict[str, Any]) -> np.ndarray:
    """从 acoustic_ctx 提取用于跨视角时间对齐的多通道帧级特征矩阵 (T, D)。

    融合：
    - 双耳协同 CTC 非 Blank 音素 log-prob 分布（截断底噪后标准化）
    - 左右耳最大 RMS 能量包络与双耳差包络
    """
    coop_em = ams.build_cooperative_stereo_emission(acoustic_ctx, ear_bias=0.0).detach().cpu().numpy()
    total_frames, n_classes = coop_em.shape
    blank = int(acoustic_ctx.get("blank", 0))
    ignored = set(acoustic_ctx.get("ignored_classes", ()))
    valid_cls = [c for c in range(n_classes) if c != blank and c not in ignored]

    # 截断极低概率底噪，保留真实发音激活形态
    phoneme_mat = np.clip(coop_em[:, valid_cls], -7.5, 0.0).astype(np.float32)
    max_phoneme = np.max(phoneme_mat, axis=1, keepdims=True)

    rms_l = np.asarray(acoustic_ctx["rms_l"], dtype=np.float32)
    rms_r = np.asarray(acoustic_ctx["rms_r"], dtype=np.float32)
    rms_max = np.clip(np.maximum(rms_l, rms_r), -65.0, 0.0).reshape(-1, 1)

    feat = np.concatenate([phoneme_mat, max_phoneme, rms_max], axis=1)
    return _standardize_2d(feat)


def estimate_perspective_offset_frames(
    ctx_A: Dict[str, Any],
    ctx_B: Dict[str, Any],
    max_offset_sec: float = 120.0,
    win_sec: float = 30.0,
    step_sec: float = 10.0,
) -> Tuple[int, float, Dict[str, Any]]:
    """自动估计视角 A 与视角 B 的帧级时间偏移 Δf = f_A - f_B (即 t_A = t_B + Δf * FRAME_SEC)。

    结合：
    1) 全轨 FFT 多通道互相关曲线 R_global(Δf)
    2) 滑动窗口互相关峰值投票共识 (Windowed Peak Consensus)，抗局部隔墙静音干扰
    返回: (delta_frames, delta_sec, debug_info)
    """
    feat_A = _extract_perspective_feature_matrix(ctx_A)
    feat_B = _extract_perspective_feature_matrix(ctx_B)
    T_A, D = feat_A.shape
    T_B, _ = feat_B.shape

    max_lag = min(int(round(max_offset_sec / FRAME_SEC)), max(T_A, T_B) - 50)
    if max_lag <= 0:
        return 0, 0.0, {"method": "zero_fallback", "confidence": 0.0}

    # ── 1) 全轨 FFT 归一化互相关 (NCC) ──
    n_fft = 1 << int(np.ceil(np.log2(T_A + T_B + 1)))
    F_A = np.fft.rfft(feat_A, n=n_fft, axis=0)
    F_B = np.fft.rfft(feat_B, n=n_fft, axis=0)
    cc_full = np.fft.irfft(np.sum(F_A * np.conj(F_B), axis=1), n=n_fft)

    sq_A = np.sum(feat_A.astype(np.float64) ** 2, axis=1)
    sq_B = np.sum(feat_B.astype(np.float64) ** 2, axis=1)
    cs_sq_A = np.concatenate([[0.0], np.cumsum(sq_A)])
    cs_sq_B = np.concatenate([[0.0], np.cumsum(sq_B)])

    lags = np.arange(-max_lag, max_lag + 1, dtype=np.int32)
    global_scores = np.full(len(lags), -1e9, dtype=np.float32)
    min_overlap = min(600, min(T_A, T_B) // 3)
    ref_len = float(min(T_A, T_B))

    for idx, lag in enumerate(lags):
        s_A = max(0, int(lag))
        s_B = max(0, -int(lag))
        ov = min(T_A - s_A, T_B - s_B)
        if ov < min_overlap:
            continue
        raw_c = float(cc_full[lag % n_fft])
        e_A = float(cs_sq_A[s_A + ov] - cs_sq_A[s_A])
        e_B = float(cs_sq_B[s_B + ov] - cs_sq_B[s_B])
        denom = np.sqrt(max(1e-6, e_A * e_B))
        # 余弦相似度 × 温和重叠率覆盖因子，杜绝高能量区段拉偏与边缘短窗伪峰
        cov_weight = (float(ov) / ref_len) ** 0.20
        global_scores[idx] = float((raw_c / denom) * cov_weight)

    best_global_idx = int(np.argmax(global_scores))
    best_global_lag = int(lags[best_global_idx])
    valid_g = global_scores[global_scores > -1e8]
    g_zscore = (
        float((global_scores[best_global_idx] - np.median(valid_g)) / (np.std(valid_g) + 1e-6))
        if len(valid_g) > 10
        else 0.0
    )

    # ── 2) 滑动窗口局部归一化互相关 (Windowed NCC) 峰值投票 ──
    win_f = int(round(win_sec / FRAME_SEC))
    step_f = max(50, int(round(step_sec / FRAME_SEC)))
    win_votes: List[Tuple[int, float]] = []

    if T_B >= win_f and T_A >= win_f:
        n_fft_w = 1 << int(np.ceil(np.log2(T_A + win_f + 1)))
        F_A_w = np.fft.rfft(feat_A, n=n_fft_w, axis=0)
        win_e_A = np.maximum(1e-6, cs_sq_A[win_f:] - cs_sq_A[:-win_f])
        for st_B in range(0, T_B - win_f + 1, step_f):
            seg_B = feat_B[st_B : st_B + win_f]
            if float(np.std(seg_B)) < 0.15:
                continue
            seg_B_norm = _standardize_2d(seg_B)
            e_seg_B = float(D * win_f)
            F_seg = np.fft.rfft(seg_B_norm, n=n_fft_w, axis=0)
            cc_w = np.fft.irfft(np.sum(F_A_w * np.conj(F_seg), axis=1), n=n_fft_w)

            # st_A = st_B + lag => 局部余弦归一化，防止被 A 轨高能量区间吸引
            w_lags = []
            w_vals = []
            for lag in range(-max_lag, max_lag + 1):
                st_A = st_B + lag
                if 0 <= st_A <= T_A - win_f:
                    denom_w = float(np.sqrt(win_e_A[st_A] * e_seg_B))
                    w_lags.append(lag)
                    w_vals.append(float(cc_w[st_A]) / denom_w)
            if len(w_vals) < 20:
                continue
            w_vals_np = np.asarray(w_vals, dtype=np.float32)
            pk_i = int(np.argmax(w_vals_np))
            pk_val = float(w_vals_np[pk_i])
            z_w = float((pk_val - np.median(w_vals_np)) / (np.std(w_vals_np) + 1e-6))
            if z_w >= 3.2 and pk_val >= 0.06:
                win_votes.append((int(w_lags[pk_i]), z_w * pk_val))

    # 聚类滑窗投票（容差 ±4 帧 = ±0.08s）
    best_cluster_lag = best_global_lag
    best_cluster_weight = 0.0
    if win_votes:
        for cand_lag, _ in win_votes:
            inliers = [(l, w) for l, w in win_votes if abs(l - cand_lag) <= 4]
            tot_w = sum(w for _, w in inliers)
            if tot_w > best_cluster_weight:
                best_cluster_weight = tot_w
                best_cluster_lag = int(round(sum(l * w for l, w in inliers) / tot_w))

    # 若滑窗共识与全局峰值一致或滑窗共识极强，采用滑窗精修值
    if win_votes and (abs(best_cluster_lag - best_global_lag) <= 6 or best_cluster_weight >= 1.5):
        final_lag = best_cluster_lag
        method = "window_consensus"
    else:
        final_lag = best_global_lag
        method = "global_fft"

    # 若偏移极小 (<= 2 帧 = 0.04s) 且 0 帧本身得分极高，归零消除抖动
    zero_idx = int(np.where(lags == 0)[0][0]) if 0 in lags else -1
    if abs(final_lag) <= 2 and zero_idx >= 0 and global_scores[zero_idx] >= global_scores[best_global_idx] * 0.97:
        final_lag = 0

    return int(final_lag), float(final_lag * FRAME_SEC), {
        "method": method,
        "global_lag_sec": float(best_global_lag * FRAME_SEC),
        "cluster_lag_sec": float(best_cluster_lag * FRAME_SEC),
        "global_zscore": g_zscore,
        "win_votes_count": len(win_votes),
        "cluster_weight": best_cluster_weight,
    }


def _refine_offset_with_clause_anchors(
    initial_lag_frames: int,
    map_A: Dict[int, List[Dict[str, Any]]],
    map_B: Dict[int, List[Dict[str, Any]]],
    ono_clause_flags: Sequence[int],
    max_deviation_sec: float = 1.5,
) -> int:
    """利用两视角下均取得高声学置信度的实词子句对齐时间差，对帧偏移做二次精修。"""
    anchor_diffs: List[float] = []
    anchor_weights: List[float] = []
    init_sec = initial_lag_frames * FRAME_SEC

    for ci, res_a in map_A.items():
        if ci not in map_B:
            continue
        if ci < len(ono_clause_flags) and ono_clause_flags[ci]:
            continue
        inf_a = ams._clause_summary(res_a)
        inf_b = ams._clause_summary(map_B[ci])
        if inf_a["skipped"] or inf_b["skipped"]:
            continue
        if inf_a["n"] < 4 or inf_a["dur"] < 0.25 or inf_b["dur"] < 0.25:
            continue
        if inf_a["score"] > -1.25 and inf_b["score"] > -1.25:
            diff_st = inf_a["st"] - inf_b["st"]
            if abs(diff_st - init_sec) <= max_deviation_sec:
                anchor_diffs.append(diff_st)
                anchor_weights.append(float(inf_a["n"]) * (2.0 + min(inf_a["score"], inf_b["score"])))

    if len(anchor_diffs) >= 3 and sum(anchor_weights) >= 15.0:
        med_sec = float(np.median(anchor_diffs))
        return int(round(med_sec / FRAME_SEC))
    return initial_lag_frames


# ══════════════════════════════════════════════════════════════════════════
#  2. 联合主时间轴构建与动态无绑定特征融合 (Dynamic Perspective Fusion)
# ══════════════════════════════════════════════════════════════════════════

def build_union_master_context(
    ctx_A: Dict[str, Any],
    ctx_B: Dict[str, Any],
    delta_frames: int,
) -> Dict[str, Any]:
    """构建覆盖视角 A 与视角 B 并集的联合主时间轴上下文 (Union Master Timeline)。

    关系定义：
    - f_A = f_B + delta_frames
    - Master 帧 u 对应的视角 A 帧: f_A(u) = u + min(0, delta_frames)
    - Master 帧 u 对应的视角 B 帧: f_B(u) = f_A(u) - delta_frames
    """
    em_LA = ctx_A["emission_L"][0].cpu()
    em_RA = ctx_A["emission_R"][0].cpu()
    st_A = ctx_A["stereo_emission"][0].cpu()
    coop_A = ams.build_cooperative_stereo_emission(ctx_A, ear_bias=0.0).cpu()

    em_LB = ctx_B["emission_L"][0].cpu()
    em_RB = ctx_B["emission_R"][0].cpu()
    st_B = ctx_B["stereo_emission"][0].cpu()
    coop_B = ams.build_cooperative_stereo_emission(ctx_B, ear_bias=0.0).cpu()

    T_A, C = coop_A.shape
    T_B, _ = coop_B.shape
    blank = int(ctx_A.get("blank", 0))

    f_A_min = min(0, int(delta_frames))
    f_A_max = max(T_A, T_B + int(delta_frames))
    T_union = int(f_A_max - f_A_min)

    # 在主时间轴上视角 A 与视角 B 各自的有效起止帧区间 [u_st, u_en)
    u_A_st = -f_A_min
    u_A_en = u_A_st + T_A
    u_B_st = int(delta_frames) - f_A_min
    u_B_en = u_B_st + T_B

    # 初始化静音底值发射矩阵
    default_col = torch.full((T_union, C), -12.0, dtype=torch.float32)
    default_col[:, blank] = -0.01

    aligned_coop_A = default_col.clone()
    aligned_coop_B = default_col.clone()
    aligned_LA = default_col.clone()
    aligned_RA = default_col.clone()
    aligned_LB = default_col.clone()
    aligned_RB = default_col.clone()

    active_A = np.zeros(T_union, dtype=bool)
    active_B = np.zeros(T_union, dtype=bool)

    aligned_coop_A[u_A_st:u_A_en] = coop_A
    aligned_LA[u_A_st:u_A_en] = em_LA
    aligned_RA[u_A_st:u_A_en] = em_RA
    active_A[u_A_st:u_A_en] = True

    aligned_coop_B[u_B_st:u_B_en] = coop_B
    aligned_LB[u_B_st:u_B_en] = em_LB
    aligned_RB[u_B_st:u_B_en] = em_RB
    active_B[u_B_st:u_B_en] = True

    # 对齐 RMS 与静音掩码
    floor_A = float(ctx_A["mono_floor"])
    floor_B = float(ctx_B["mono_floor"])
    rms_A_mono = np.full(T_union, floor_A, dtype=np.float32)
    rms_B_mono = np.full(T_union, floor_B, dtype=np.float32)
    rms_A_max = np.full(T_union, floor_A, dtype=np.float32)
    rms_B_max = np.full(T_union, floor_B, dtype=np.float32)

    rms_A_mono[u_A_st:u_A_en] = ctx_A["mono_rms"]
    rms_B_mono[u_B_st:u_B_en] = ctx_B["mono_rms"]
    rms_A_max[u_A_st:u_A_en] = np.maximum(ctx_A["rms_l"], ctx_A["rms_r"])
    rms_B_max[u_B_st:u_B_en] = np.maximum(ctx_B["rms_l"], ctx_B["rms_r"])

    # 构建基础跨视角协同发射矩阵 joint_coop:
    # 两个视角都有效时，音素通道取 max，Blank 通道取双视角联合静音（二者均静音才为 Blank）
    joint_coop = default_col.clone()
    only_A = active_A & (~active_B)
    only_B = active_B & (~active_A)
    both_AB = active_A & active_B

    if np.any(only_A):
        joint_coop[only_A] = aligned_coop_A[only_A]
    if np.any(only_B):
        joint_coop[only_B] = aligned_coop_B[only_B]
    if np.any(both_AB):
        joint_coop[both_AB] = torch.maximum(aligned_coop_A[both_AB], aligned_coop_B[both_AB])
        # Blank 通道遵循 AND 静音逻辑：取较小（更负）的 blank log-prob，防止静音视角稀释发声视角
        joint_coop[both_AB, blank] = torch.minimum(
            st_A[ np.where(both_AB)[0] - u_A_st, blank ],
            st_B[ np.where(both_AB)[0] - u_B_st, blank ],
        )

    mask_A = np.zeros(T_union, dtype=np.uint8)
    mask_B = np.zeros(T_union, dtype=np.uint8)
    mask_A[u_A_st:u_A_en] = ctx_A["speech_frame_mask"]
    mask_B[u_B_st:u_B_en] = ctx_B["speech_frame_mask"]
    joint_speech_mask = np.maximum(mask_A, mask_B).astype(np.uint8)

    joint_rms = np.maximum(rms_A_max, rms_B_max)
    joint_floor = compute_track_noise_floor(joint_rms, percentile=10.0)
    joint_head_silence_end = build_next_stable_silence_end(
        joint_rms,
        joint_floor,
        max_spike_db=None,
        require_pre_silence_island=True,
        island_active_margin_db=4.0,
        island_smooth_frames=5,
        island_bridge_frames=5,
        island_lookback_frames=50,
        island_prepad_frames=4,
    )
    joint_mono_rms = np.maximum(rms_A_mono, rms_B_mono)
    joint_mono_floor = float(compute_track_noise_floor(joint_mono_rms))

    return {
        # 标准 acoustic_ctx 接口字段（供 align_multispeaker 直接调用）
        "emission_L": aligned_coop_A.unsqueeze(0),
        "emission_R": aligned_coop_B.unsqueeze(0),
        "rms_l": rms_A_max,
        "rms_r": rms_B_max,
        "mono_rms": joint_mono_rms,
        "mono_floor": joint_mono_floor,
        "speech_frame_mask": joint_speech_mask,
        "stereo_emission": joint_coop.unsqueeze(0),
        "head_silence_end": joint_head_silence_end,
        "ignored_classes": ctx_A["ignored_classes"],
        "blank": blank,
        "tokenizer": ctx_A["tokenizer"],
        # 多视角扩展字段
        "T_union": T_union,
        "T_A": T_A,
        "T_B": T_B,
        "delta_frames": int(delta_frames),
        "f_A_min": int(f_A_min),
        "u_A_range": (int(u_A_st), int(u_A_en)),
        "u_B_range": (int(u_B_st), int(u_B_en)),
        "active_A": active_A,
        "active_B": active_B,
        "aligned_coop_A": aligned_coop_A,
        "aligned_coop_B": aligned_coop_B,
        "aligned_LA": aligned_LA,
        "aligned_RA": aligned_RA,
        "aligned_LB": aligned_LB,
        "aligned_RB": aligned_RB,
        "joint_coop": joint_coop,
        "rms_A_mono": rms_A_mono,
        "rms_B_mono": rms_B_mono,
        "floor_A": floor_A,
        "floor_B": floor_B,
        "mask_A": mask_A,
        "mask_B": mask_B,
    }


def _shift_clause_map_time(
    clause_map: Dict[int, List[Dict[str, Any]]],
    shift_sec: float,
) -> Dict[int, List[Dict[str, Any]]]:
    """将子句结果映射中的所有时间戳平移 shift_sec 秒。"""
    if abs(shift_sec) < 1e-6:
        return copy.deepcopy(clause_map)
    out: Dict[int, List[Dict[str, Any]]] = {}
    for ci, cl_res in clause_map.items():
        new_cl = []
        for r in cl_res:
            nr = dict(r)
            if nr.get("original_start") is not None:
                nr["original_start"] = float(nr["original_start"]) + shift_sec
                nr["original_end"] = float(nr["original_end"]) + shift_sec
                nr["start"] = _format_time(nr["original_start"])
                nr["end"] = _format_time(nr["original_end"])
            new_cl.append(nr)
        out[ci] = new_cl
    return out


# ══════════════════════════════════════════════════════════════════════════
#  3. 动态视角感知与跨视角互斥对齐管线 (Dynamic Multi-Perspective Pipeline)
# ══════════════════════════════════════════════════════════════════════════

def _compute_clause_perspective_affinity(
    ci: int,
    cl_res: List[Dict[str, Any]],
    alignment_tokens: List[List[str]],
    union_ctx: Dict[str, Any],
) -> float:
    """计算某个已对齐子句在主时间轴对应帧区间内对视角 A (+1) 与视角 B (-1) 的声学亲和度差值。

    不仅看能量，而是直接比较该子句真实目标音素在视角 A 与视角 B 发射矩阵上的 log-prob 得分差。
    正值表示视角 A 更清晰，负值表示视角 B 更清晰。
    """
    inf = ams._clause_summary(cl_res)
    if inf["skipped"] or inf["dur"] <= 0.02:
        return 0.0

    T_union = union_ctx["T_union"]
    coop_A = union_ctx["aligned_coop_A"]
    coop_B = union_ctx["aligned_coop_B"]
    active_A = union_ctx["active_A"]
    active_B = union_ctx["active_B"]
    tokenizer = union_ctx["tokenizer"]
    blank = int(union_ctx["blank"])

    c_toks, single_ids, _, _ = ams._canonicalize_line_tokens(
        alignment_tokens[ci], tokenizer, blank, fold_subclause_ii=True
    )
    valid_Mask = [r for r in cl_res if not r.get("skipped") and r.get("original_start") is not None]
    if not valid_Mask or not single_ids:
        return 0.0

    diffs = []
    for k, r in enumerate(cl_res):
        if r.get("skipped") or r.get("original_start") is None:
            continue
        st_f = max(0, min(T_union - 1, int(round(float(r["original_start"]) / FRAME_SEC))))
        en_f = max(st_f + 1, min(T_union, int(round(float(r["original_end"]) / FRAME_SEC))))
        # 若该区间仅在单一视角有效，直接返回强亲和度
        if not np.any(active_B[st_f:en_f]) and np.any(active_A[st_f:en_f]):
            diffs.append(4.0)
            continue
        if not np.any(active_A[st_f:en_f]) and np.any(active_B[st_f:en_f]):
            diffs.append(-4.0)
            continue
        tid = single_ids[min(k, len(single_ids) - 1)]
        sc_a = float(torch.max(coop_A[st_f:en_f, tid]).item())
        sc_b = float(torch.max(coop_B[st_f:en_f, tid]).item())
        diffs.append(sc_a - sc_b)

    return float(np.mean(diffs)) if diffs else 0.0


def _build_dynamic_perspective_emission(
    chain_indices: List[int],
    map_A_master: Dict[int, List[Dict[str, Any]]],
    map_B_master: Dict[int, List[Dict[str, Any]]],
    map_curr_master: Dict[int, List[Dict[str, Any]]],
    ono_clause_flags: Sequence[int],
    union_ctx: Dict[str, Any],
    other_perspective_claims: List[Tuple[float, float, float, float]],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], float]:
    """为一条说话人子句链构建时间动态变化的发射矩阵 (T_union, C)、行间 Filler 概率与动态声学上下文。

    核心特性（无角色与音轨硬绑定）：
    1) 对链上每一个实词子句 ci，动态比较其在视角 A (map_A_master) 与视角 B (map_B_master) 的声学质量；
       即便同一角色中途换房间/换搭档（前半段在 A 清晰、后半段在 B 清晰），其锚点偏好也会随时间平滑切换。
    2) 结合其他说话人在视角 A / 视角 B 上已宣告占用的时间区间 (other_perspective_claims: st, en, pen_A, pen_B)，
       在每一帧动态压低被他人占用的视角，同时抬升已被他人占用区间的行间 Filler 通过概率。
    3) 动态解耦 Blank 通道、静音掩码 (speech_frame_mask) 与能量包络 (mono_rms)：
       当说话人在时刻 u 偏向某视角时，Blank 概率与静音检测自动跟随该视角，彻底杜绝另一房间说话声阻断当前角色静音收尾。
    """
    T_union = union_ctx["T_union"]
    coop_A = union_ctx["aligned_coop_A"].numpy()
    coop_B = union_ctx["aligned_coop_B"].numpy()
    joint_coop = union_ctx["joint_coop"].numpy()
    active_A = union_ctx["active_A"]
    active_B = union_ctx["active_B"]
    blank = int(union_ctx["blank"])
    ignored = set(union_ctx.get("ignored_classes", ()))
    C = joint_coop.shape[1]
    valid_cls = [c for c in range(C) if c != blank and c not in ignored]

    # ── 1) 从链上高置信度实词锚点提取随时间变化的动态视角偏好轨迹 pref_curve(u) ∈ [-1, +1] ──
    # +1 代表完全偏向视角 A，-1 代表完全偏向视角 B，0 代表双视角均衡
    anchor_times: List[float] = []
    anchor_prefs: List[float] = []

    for ci in chain_indices:
        if ci < len(ono_clause_flags) and ono_clause_flags[ci]:
            continue
        inf_a = ams._clause_summary(map_A_master.get(ci, []))
        inf_b = ams._clause_summary(map_B_master.get(ci, []))
        if inf_a["n"] < 2:
            continue

        # 评估该子句在视角 A 与视角 B 的有效声学质量
        q_a = inf_a["score"] if (not inf_a["skipped"] and inf_a["dur"] >= 0.12) else -6.0
        q_b = inf_b["score"] if (not inf_b["skipped"] and inf_b["dur"] >= 0.12) else -6.0
        best_q = max(q_a, q_b)
        if best_q < -1.85:
            continue

        diff_q = q_a - q_b
        if abs(diff_q) >= 0.25:
            # 选择得分更高一侧的时间点作为该子句在主时间轴上的锚点时刻
            t_mid = 0.5 * (inf_a["st"] + inf_a["en"]) if q_a >= q_b else 0.5 * (inf_b["st"] + inf_b["en"])
            pref_val = float(np.clip(diff_q / 1.5, -1.0, 1.0))
            anchor_times.append(t_mid)
            anchor_prefs.append(pref_val)

    pref_curve = np.zeros(T_union, dtype=np.float32)
    if anchor_times:
        # 按时间排序并对锚点做中值/移动平均滤波，既保留中途换人的动态转折，又滤除单句噪声
        order = np.argsort(anchor_times)
        t_arr = np.asarray([anchor_times[i] for i in order], dtype=np.float32)
        p_arr = np.asarray([anchor_prefs[i] for i in order], dtype=np.float32)
        if len(p_arr) >= 3:
            k_win = min(5, len(p_arr))
            pad = k_win // 2
            p_pad = np.pad(p_arr, (pad, pad), mode="edge")
            p_smooth = np.convolve(p_pad, np.ones(k_win, dtype=np.float32) / k_win, mode="valid")[: len(p_arr)]
        else:
            p_smooth = p_arr
        u_sec = np.arange(T_union, dtype=np.float32) * FRAME_SEC
        pref_curve = np.interp(u_sec, t_arr, p_smooth).astype(np.float32)

    # ── 2) 按帧合成动态视角发射矩阵与视角解耦 Blank ──
    # 在双视角共存区间，根据 pref_curve(u) 动态调整视角 A 与视角 B 的惩罚权重
    atten_A = np.maximum(0.0, -pref_curve * 2.40).astype(np.float32)  # 当偏向 B (<0) 时压低 A 的隔壁串音
    atten_B = np.maximum(0.0, pref_curve * 2.40).astype(np.float32)   # 当偏向 A (>0) 时压低 B 的隔壁串音

    # 叠加来自其他说话人的跨视角占用惩罚 (st, en, pen_A, pen_B)
    claimed_boost = np.zeros(T_union, dtype=np.float32)
    for st_c, en_c, pen_a, pen_b in other_perspective_claims:
        st_f = max(0, min(T_union, int(round((st_c - 0.04) / FRAME_SEC))))
        en_f = max(st_f, min(T_union, int(round((en_c + 0.04) / FRAME_SEC))))
        if en_f > st_f:
            if pen_a > 0:
                atten_A[st_f:en_f] = np.maximum(atten_A[st_f:en_f], np.float32(pen_a))
            if pen_b > 0:
                atten_B[st_f:en_f] = np.maximum(atten_B[st_f:en_f], np.float32(pen_b))
            claimed_boost[st_f:en_f] = np.maximum(
                claimed_boost[st_f:en_f], np.float32(0.55 * max(pen_a, pen_b))
            )

    em_out = joint_coop.copy()
    dyn_rms = union_ctx["mono_rms"].copy()
    dyn_mask = union_ctx["speech_frame_mask"].copy()
    rms_A_mono = union_ctx["rms_A_mono"]
    rms_B_mono = union_ctx["rms_B_mono"]
    mask_A = union_ctx["mask_A"]
    mask_B = union_ctx["mask_B"]

    both_idx = np.where(active_A & active_B)[0]
    if len(both_idx) > 0:
        # 计算消除整体混音电平差后的跨视角声压级差 ild_AB(u) = (rms_A - med_A) - (rms_B - med_B)
        # 因为 Wav2Vec2 内部带 LayerNorm，会将 -50dB 的微弱串音放大成高置信度音素；
        # 利用物理声压差 ild_AB 可直接识别出“在麦克风 A 中虽能听见、但实际上离麦克风 B 近得多”的串音帧！
        sp_A_vals = rms_A_mono[both_idx][mask_A[both_idx] > 0]
        sp_B_vals = rms_B_mono[both_idx][mask_B[both_idx] > 0]
        med_A = float(np.median(sp_A_vals)) if len(sp_A_vals) >= 10 else -30.0
        med_B = float(np.median(sp_B_vals)) if len(sp_B_vals) >= 10 else -30.0
        ild_offset = np.clip(med_A - med_B, -8.0, 8.0)
        ild_AB = (rms_A_mono[both_idx] - rms_B_mono[both_idx]) - np.float32(ild_offset)

        # 当物理声压级明显偏向 B (ild_AB < -4.5dB) 时，说明该帧声源在视角 B，视角 A 上的能量属于串音泄露
        bleed_on_A = np.clip((-ild_AB - 4.5) * 0.35, 0.0, 3.2).astype(np.float32)
        # 当物理声压级明显偏向 A (ild_AB > +4.5dB) 时，说明该帧声源在视角 A，视角 B 上的能量属于串音泄露
        bleed_on_B = np.clip((ild_AB - 4.5) * 0.35, 0.0, 3.2).astype(np.float32)

        net_pref = np.clip(
            pref_curve[both_idx] + 0.35 * (atten_B[both_idx] - atten_A[both_idx]),
            -1.0,
            1.0,
        )
        # 若当前说话人在时刻 u 偏向 A (net_pref > 0.15)，则对 A 上的跨房串音帧施加 bleed_on_A 压制；反之亦然
        eff_atten_A = atten_A[both_idx].copy()
        eff_atten_B = atten_B[both_idx].copy()
        favor_A = net_pref > 0.15
        favor_B = net_pref < -0.15
        if np.any(favor_A):
            eff_atten_A[favor_A] = np.maximum(eff_atten_A[favor_A], bleed_on_A[favor_A])
        if np.any(favor_B):
            eff_atten_B[favor_B] = np.maximum(eff_atten_B[favor_B], bleed_on_B[favor_B])

        adj_A = coop_A[both_idx] - eff_atten_A[:, None]
        adj_B = coop_B[both_idx] - eff_atten_B[:, None]
        em_out[both_idx] = np.maximum(adj_A, adj_B)

        # 动态解耦 Blank 通道与静音掩码：当说话人在帧 u 明确归属某视角（或另一视角被他人占用）时，
        # Blank 概率与 RMS/静音掩码跟随本视角，避免隔壁房间的持续说话声阻断本房间的静音收尾
        blank_both = joint_coop[both_idx, blank].copy()
        if np.any(favor_A):
            alpha_A = np.clip((net_pref[favor_A] - 0.15) / 0.45, 0.0, 1.0)
            idx_fA = both_idx[favor_A]
            blank_both[favor_A] = (
                alpha_A * coop_A[idx_fA, blank] + (1.0 - alpha_A) * joint_coop[idx_fA, blank]
            )
            dyn_rms[idx_fA] = rms_A_mono[idx_fA]
            dyn_mask[idx_fA] = mask_A[idx_fA]
            # 若该帧本身是 B 视角强发声泄露到 A 的串音帧 (bleed_on_A > 1.2)，将其对 A 视角的静音掩码置 0 并抬升 Filler
            is_bleed_A = bleed_on_A[favor_A] > 1.2
            if np.any(is_bleed_A):
                dyn_mask[idx_fA[is_bleed_A]] = 0
                claimed_boost[idx_fA[is_bleed_A]] = np.maximum(
                    claimed_boost[idx_fA[is_bleed_A]], bleed_on_A[favor_A][is_bleed_A] * 0.55
                )

        if np.any(favor_B):
            alpha_B = np.clip((-net_pref[favor_B] - 0.15) / 0.45, 0.0, 1.0)
            idx_fB = both_idx[favor_B]
            blank_both[favor_B] = (
                alpha_B * coop_B[idx_fB, blank] + (1.0 - alpha_B) * joint_coop[idx_fB, blank]
            )
            dyn_rms[idx_fB] = rms_B_mono[idx_fB]
            dyn_mask[idx_fB] = mask_B[idx_fB]
            is_bleed_B = bleed_on_B[favor_B] > 1.2
            if np.any(is_bleed_B):
                dyn_mask[idx_fB[is_bleed_B]] = 0
                claimed_boost[idx_fB[is_bleed_B]] = np.maximum(
                    claimed_boost[idx_fB[is_bleed_B]], bleed_on_B[favor_B][is_bleed_B] * 0.55
                )

        em_out[both_idx, blank] = blank_both

    # 单一视角有效区间若被其他同视角角色占用，同样施加对应惩罚
    only_A_idx = np.where(active_A & (~active_B) & (atten_A > 0))[0]
    if len(only_A_idx) > 0:
        em_out[only_A_idx] = coop_A[only_A_idx] - atten_A[only_A_idx, None]
        em_out[only_A_idx, blank] = coop_A[only_A_idx, blank]

    only_B_idx = np.where(active_B & (~active_A) & (atten_B > 0))[0]
    if len(only_B_idx) > 0:
        em_out[only_B_idx] = coop_B[only_B_idx] - atten_B[only_B_idx, None]
        em_out[only_B_idx, blank] = coop_B[only_B_idx, blank]

    max_nb = np.max(em_out[:, valid_cls], axis=1).astype(np.float32)
    filler_logp = (max_nb - np.float32(1.35) + claimed_boost).astype(np.float32)

    dyn_ctx = copy.copy(union_ctx)
    dyn_ctx["mono_rms"] = dyn_rms
    dyn_ctx["mono_floor"] = float(compute_track_noise_floor(dyn_rms))
    dyn_ctx["speech_frame_mask"] = dyn_mask.astype(np.uint8)
    mean_pref = float(np.mean(pref_curve[both_idx])) if len(both_idx) > 0 else float(np.mean(pref_curve))
    return em_out, filler_logp, dyn_ctx, mean_pref


def _build_single_perspective_union_ctx(
    union_ctx: Dict[str, Any],
    which: str,
) -> Dict[str, Any]:
    """在联合主时间轴 [0, T_union) 上构建单视角优先探针上下文 (which='A' 或 'B')。

    关键设计：
    - 在双视角重叠区间 (active_A & active_B)，严格使用该视角自身的声学特征；
    - 在另一视角独有的首尾区间（例如视角 B 尚未开机录制的前 34 秒），自动借用另一视角的声学特征兜底，
      从而让台本中属于独有区间的台词自然锚定在独有区间内，彻底杜绝未录制台词挤压侵入重叠区间！
    """
    ctx = copy.copy(union_ctx)
    T_union = union_ctx["T_union"]
    blank = int(union_ctx["blank"])
    active_A = union_ctx["active_A"]
    active_B = union_ctx["active_B"]
    both_AB = active_A & active_B

    if which.upper() == "A":
        em_L = union_ctx["aligned_LA"].clone()
        em_R = union_ctx["aligned_RA"].clone()
        coop = union_ctx["aligned_coop_A"].clone()
        rms_mono = union_ctx["rms_A_mono"].copy()
        sp_mask = union_ctx["mask_A"].copy()
        only_other = active_B & (~active_A)
        if np.any(only_other):
            em_L[only_other] = union_ctx["aligned_LB"][only_other]
            em_R[only_other] = union_ctx["aligned_RB"][only_other]
            coop[only_other] = union_ctx["aligned_coop_B"][only_other]
            rms_mono[only_other] = union_ctx["rms_B_mono"][only_other]
            sp_mask[only_other] = union_ctx["mask_B"][only_other]
    else:
        em_L = union_ctx["aligned_LB"].clone()
        em_R = union_ctx["aligned_RB"].clone()
        coop = union_ctx["aligned_coop_B"].clone()
        rms_mono = union_ctx["rms_B_mono"].copy()
        sp_mask = union_ctx["mask_B"].copy()
        only_other = active_A & (~active_B)
        if np.any(only_other):
            em_L[only_other] = union_ctx["aligned_LA"][only_other]
            em_R[only_other] = union_ctx["aligned_RA"][only_other]
            coop[only_other] = union_ctx["aligned_coop_A"][only_other]
            rms_mono[only_other] = union_ctx["rms_A_mono"][only_other]
            sp_mask[only_other] = union_ctx["mask_A"][only_other]

    ctx["emission_L"] = em_L.unsqueeze(0)
    ctx["emission_R"] = em_R.unsqueeze(0)
    ctx["stereo_emission"] = coop.unsqueeze(0)
    ctx["mono_rms"] = rms_mono
    ctx["mono_floor"] = float(compute_track_noise_floor(rms_mono))
    ctx["speech_frame_mask"] = sp_mask.astype(np.uint8)
    return ctx


def run_multiperspective_pipeline(
    alignment_tokens: List[List[str]],
    line_group: Sequence[int],
    ono_clause_flags: Sequence[int],
    clause_speakers: Sequence[Optional[str]],
    ctx_A: Dict[str, Any],
    ctx_B: Dict[str, Any],
    enable_silence_snap: bool = True,
    enable_ono_gate: bool = True,
    log_callback: Optional[Callable[[str], None]] = None,
    enable_word_skip: bool = False,
    word_skip_token_penalty: float = 4.8,
    clause_surfaces: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[List[int]], Dict[str, Any]]:
    """执行双视角动态联合对齐（默认开启多人分流与乱序重排）。

    返回:
    - flat_out: 在联合主时间轴 (Union Master Timeline) 上的逐 token 对齐结果列表
    - final_streams: 分流子句索引列表
    - mp_meta: 包含 delta_frames, delta_sec, union_ctx 等供双视角导出使用的元数据
    """
    def log(msg: str) -> None:
        if log_callback:
            log_callback(msg)

    n_clauses = len(alignment_tokens)
    if n_clauses == 0:
        union_ctx = build_union_master_context(ctx_A, ctx_B, 0)
        return [], [], {"delta_frames": 0, "delta_sec": 0.0, "union_ctx": union_ctx}

    # ── 步骤 1：声学互相关初测跨视角时间差 Δt ──
    delta_f, delta_sec, calib_info = estimate_perspective_offset_frames(ctx_A, ctx_B)
    log(
        f"  🧭 [多视角时间校准] 声学偏移 Δt = {delta_sec:+.2f}s ({delta_f:+d} 帧, "
        f"方式={calib_info['method']}, 投票窗={calib_info['win_votes_count']})"
    )

    # ── 步骤 2：构建联合主时间轴 (Union Master Timeline) 及单视角探针上下文 ──
    union_ctx = build_union_master_context(ctx_A, ctx_B, delta_f)
    u_A_st_sec = union_ctx["u_A_range"][0] * FRAME_SEC
    u_B_st_sec = union_ctx["u_B_range"][0] * FRAME_SEC

    ctx_A_union = _build_single_perspective_union_ctx(union_ctx, "A")
    ctx_B_union = _build_single_perspective_union_ctx(union_ctx, "B")
    n_tokens_flat = sum(len(lt) for lt in alignment_tokens)
    blank_punct_mask = [False] * (n_tokens_flat + 1)

    log("  🎧 [多视角独立探针] 正在评估各子句在视角 A 与视角 B 下的动态清晰度分布...")
    flat_A, streams_A = ams.align_multispeaker_or_reordered(
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
        blank_punct_mask=blank_punct_mask,
        log_callback=None,
        acoustic_ctx=ctx_A_union,
        enable_word_skip=enable_word_skip,
        word_skip_token_penalty=word_skip_token_penalty,
        clause_surfaces=clause_surfaces,
    )
    flat_B, _ = ams.align_multispeaker_or_reordered(
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
        blank_punct_mask=blank_punct_mask,
        log_callback=None,
        acoustic_ctx=ctx_B_union,
        enable_word_skip=enable_word_skip,
        word_skip_token_penalty=word_skip_token_penalty,
        clause_surfaces=clause_surfaces,
    )

    grouped_A = ams._group_results_by_clause(alignment_tokens, flat_A)
    grouped_B = ams._group_results_by_clause(alignment_tokens, flat_B)
    map_A_master = {ci: grouped_A[ci] for ci in range(n_clauses)}
    map_B_master = {ci: grouped_B[ci] for ci in range(n_clauses)}

    # 用双视角共同高分台词检验/微调 Δt（此时 map_A_master 与 map_B_master 已同处主时间轴，理想差值为 0）
    residual_f = _refine_offset_with_clause_anchors(0, map_A_master, map_B_master, ono_clause_flags)
    if abs(residual_f) >= 2 and abs(residual_f * FRAME_SEC) <= 1.2:
        new_delta_f = delta_f + residual_f
        log(
            f"  🎯 [多视角锚点精修] Δt 由 {delta_f * FRAME_SEC:+.2f}s 微调至 {new_delta_f * FRAME_SEC:+.2f}s "
            f"(残差 {residual_f * FRAME_SEC:+.2f}s)"
        )
        delta_f = new_delta_f
        delta_sec = float(delta_f * FRAME_SEC)
        union_ctx = build_union_master_context(ctx_A, ctx_B, delta_f)
        u_A_st_sec = union_ctx["u_A_range"][0] * FRAME_SEC
        u_B_st_sec = union_ctx["u_B_range"][0] * FRAME_SEC
        map_B_master = _shift_clause_map_time(map_B_master, residual_f * FRAME_SEC)

    # 确定分流链（若台本含角色标签则使用角色链，否则使用乱序拆流链）
    streams = streams_A if streams_A else [list(range(n_clauses))]

    # ── 步骤 4：动态视角融合 + 跨视角互斥迭代对齐 ──
    per_stream_results: Dict[int, Dict[int, List[Dict[str, Any]]]] = {}
    stream_persp_prefs: Dict[int, float] = {}
    for s_idx, chain in enumerate(streams):
        em_dyn, filler_dyn, dyn_ctx, mean_pref = _build_dynamic_perspective_emission(
            chain_indices=chain,
            map_A_master=map_A_master,
            map_B_master=map_B_master,
            map_curr_master={},
            ono_clause_flags=ono_clause_flags,
            union_ctx=union_ctx,
            other_perspective_claims=[],
        )
        stream_persp_prefs[s_idx] = mean_pref
        per_stream_results[s_idx] = ams._align_subset_clauses(
            sub_indices=chain,
            alignment_tokens=alignment_tokens,
            line_group=line_group,
            ono_clause_flags=ono_clause_flags,
            enable_silence_snap=enable_silence_snap,
            acoustic_ctx=dyn_ctx,
            custom_emission_2d=em_dyn,
            enable_inter_line_filler=True,
            inter_line_filler_margin=1.35,
            enable_ono_skip=enable_ono_gate,
            ono_skip_token_penalty=2.6,
            filler_frame_logp=filler_dyn,
            enable_word_skip=enable_word_skip,
            word_skip_token_penalty=word_skip_token_penalty,
            clause_surfaces=clause_surfaces,
            clause_speakers=clause_speakers,
            log_callback=log_callback,
        )

    # 若存在多条说话人流，执行 2 轮跨视角感知的互斥迭代
    if len(streams) >= 2:
        def _stream_mean_score(s_i: int) -> float:
            vals = [
                ams._clause_summary(per_stream_results[s_i][ci])["score"]
                for ci in streams[s_i]
                if not (ci < len(ono_clause_flags) and ono_clause_flags[ci])
                and not ams._clause_summary(per_stream_results[s_i][ci])["skipped"]
            ]
            return float(np.mean(vals)) if vals else -10.0

        def _collect_other_perspective_claims(target_s: int) -> List[Tuple[float, float, float, float]]:
            own_set = set(streams[target_s])
            claims: List[Tuple[float, float, float, float]] = []
            for oth_s, oth_map in per_stream_results.items():
                if oth_s == target_s:
                    continue
                for ci, cl_res in oth_map.items():
                    if ci in own_set:
                        continue
                    is_ono = (ci < len(ono_clause_flags) and ono_clause_flags[ci])
                    inf = ams._clause_summary(cl_res)
                    if inf["skipped"] or inf["dur"] < 0.10 or inf["dur"] > 12.0:
                        continue
                    if is_ono and inf["score"] <= -1.80:
                        continue
                    if inf["score"] <= -2.30:
                        continue
                    aff = _compute_clause_perspective_affinity(ci, cl_res, alignment_tokens, union_ctx)
                    if aff > 0.45:
                        # 声源偏向 A，但若在走廊/开门场景下 B 视角也有明显串音，同样对 B 施加适度互斥保护
                        pen_a, pen_b = 2.20, 1.15
                    elif aff < -0.45:
                        # 声源偏向 B，但对 A 视角上的串音泄露同样施加 1.15 互斥保护，防止 A 视角短元音词抢拍吸附
                        pen_a, pen_b = 1.15, 2.20
                    else:
                        pen_a, pen_b = 1.75, 1.75
                    claims.append((inf["st"], inf["en"], pen_a, pen_b))
            return claims

        for _rnd in range(2):
            for s_idx in sorted(range(len(streams)), key=_stream_mean_score):
                claims = _collect_other_perspective_claims(s_idx)
                em_dyn, filler_dyn, dyn_ctx, mean_pref = _build_dynamic_perspective_emission(
                    chain_indices=streams[s_idx],
                    map_A_master=map_A_master,
                    map_B_master=map_B_master,
                    map_curr_master=per_stream_results[s_idx],
                    ono_clause_flags=ono_clause_flags,
                    union_ctx=union_ctx,
                    other_perspective_claims=claims,
                )
                stream_persp_prefs[s_idx] = mean_pref
                per_stream_results[s_idx] = ams._align_subset_clauses(
                    sub_indices=streams[s_idx],
                    alignment_tokens=alignment_tokens,
                    line_group=line_group,
                    ono_clause_flags=ono_clause_flags,
                    enable_silence_snap=enable_silence_snap,
                    acoustic_ctx=dyn_ctx,
                    custom_emission_2d=em_dyn,
                    enable_inter_line_filler=True,
                    inter_line_filler_margin=1.35,
                    enable_ono_skip=enable_ono_gate,
                    ono_skip_token_penalty=2.6,
                    filler_frame_logp=filler_dyn,
                    enable_word_skip=enable_word_skip,
                    word_skip_token_penalty=word_skip_token_penalty,
                    clause_surfaces=clause_surfaces,
                    clause_speakers=clause_speakers,
                    log_callback=log_callback,
                )

    # ── 步骤 5：行级单视角保真择优回填（若某一行在单一视角独立对齐得分显著更高且不破坏前后单调性，直接采用该清晰视角的时间戳） ──
    merged_clause_map: Dict[int, List[Dict[str, Any]]] = {}
    rms_A_all = union_ctx["rms_A_mono"]
    rms_B_all = union_ctx["rms_B_mono"]
    T_union_all = union_ctx["T_union"]
    for s_idx, chain in enumerate(streams):
        s_map = per_stream_results[s_idx]
        s_pref = stream_persp_prefs.get(s_idx, 0.0)
        m_len = len(chain)
        for pos, ci in enumerate(chain):
            cur_res = s_map[ci]
            cur_inf = ams._clause_summary(cur_res)
            cand_a = map_A_master.get(ci, [])
            cand_b = map_B_master.get(ci, [])
            inf_a = ams._clause_summary(cand_a)
            inf_b = ams._clause_summary(cand_b)

            # 前后单调边界
            prev_en = 0.0
            for p_back in range(pos - 1, -1, -1):
                p_inf = ams._clause_summary(merged_clause_map.get(chain[p_back], s_map[chain[p_back]]))
                if not p_inf["skipped"] and p_inf["en"] > 0:
                    prev_en = p_inf["en"]
                    break
            next_st = float(T_union_all * FRAME_SEC)
            for p_fwd in range(pos + 1, m_len):
                n_inf = ams._clause_summary(s_map[chain[p_fwd]])
                if not n_inf["skipped"] and n_inf["st"] > 0:
                    next_st = n_inf["st"]
                    break

            best_res = cur_res
            best_sc = cur_inf["score"] if not cur_inf["skipped"] else -10.0
            max_reasonable_dur = max(2.6, 0.32 * max(1, cur_inf["n"]))

            for cand_side, cand_res, cand_inf in (("A", cand_a, inf_a), ("B", cand_b, inf_b)):
                if cand_inf["skipped"] or cand_inf["dur"] <= 0.02 or cand_inf["dur"] > max_reasonable_dur:
                    continue
                # 检查候选区间是否落在另一视角强发声的串音泄露区（防止单视角探针因 LayerNorm 误吸对门声音）
                st_f = max(0, min(T_union_all - 1, int(round(cand_inf["st"] / FRAME_SEC))))
                en_f = max(st_f + 1, min(T_union_all, int(round(cand_inf["en"] / FRAME_SEC))))
                mean_ild = float(np.mean(rms_A_all[st_f:en_f] - rms_B_all[st_f:en_f]))
                if s_pref > 0.15 and mean_ild < -4.5:
                    continue
                if s_pref < -0.15 and mean_ild > 4.5:
                    continue
                if cand_inf["st"] >= prev_en - 0.04 and cand_inf["en"] <= next_st + 0.04:
                    if cand_inf["score"] > best_sc + 0.35 or (
                        cur_inf["dur"] > max_reasonable_dur and cand_inf["score"] >= best_sc - 0.25
                    ):
                        best_res = cand_res
                        best_sc = cand_inf["score"]

            merged_clause_map[ci] = best_res

    flat_out: List[Dict[str, Any]] = []
    for ci in range(n_clauses):
        flat_out.extend(merged_clause_map[ci])

    spk_persp_pref: Dict[str, float] = {}
    for s_idx, chain in enumerate(streams):
        pref_s = stream_persp_prefs.get(s_idx, 0.0)
        for ci in chain:
            if ci < len(clause_speakers) and clause_speakers[ci]:
                raw_sp = clause_speakers[ci]
                sp0 = raw_sp[0] if isinstance(raw_sp, (tuple, list)) else raw_sp
                spk_name = str(sp0).strip().lstrip("　")
                if spk_name:
                    spk_persp_pref[spk_name] = pref_s

    mp_meta = {
        "delta_frames": delta_f,
        "delta_sec": delta_sec,
        "u_A_st_sec": u_A_st_sec,
        "u_B_st_sec": u_B_st_sec,
        "dur_A_sec": float(union_ctx["T_A"] * FRAME_SEC),
        "dur_B_sec": float(union_ctx["T_B"] * FRAME_SEC),
        "union_ctx": union_ctx,
        "calib_info": calib_info,
        "spk_persp_pref": spk_persp_pref,
    }
    return flat_out, streams, mp_meta


# ══════════════════════════════════════════════════════════════════════════
#  4. 按视角时间轴投影与越界裁剪导出 (Perspective Projection & Clipping)
# ══════════════════════════════════════════════════════════════════════════

def project_result_list_to_perspective(
    master_result_list: List[Dict[str, Any]],
    perspective_origin_sec: float,
    perspective_duration_sec: float,
    boundary_tolerance_sec: float = 0.35,
    perspective_side: str = "A",
    spk_persp_pref: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """将联合主时间轴上的 result_list 投影到指定视角的时间坐标系 [0, perspective_duration_sec]。

    - 每个 token 的时间戳平移: t_persp = t_master - perspective_origin_sec
    - 若整行台词全部落在 [-boundary_tolerance_sec, perspective_duration_sec + boundary_tolerance_sec] 之外
      （例如另一视角开机前或关机后的独有台词），自动剔除该行，防止未录制台词堆积在 00:00.00 或末尾。
    - 对于同屏多角色并列行（以全角空格连接），自动将该视角的主视角角色固定排在左侧、隔壁视角角色排在右侧。
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
            sts = []
            ens = []
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
                # 判断该行是否处于当前视角的录音时间范围之外（含开机前已发声或关机后才结束的越界台词）
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

                    # 若该行包含多个说话人并列，按当前视角的主次角色固定左右顺序（本视角前景角色居左）
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
                            # 提取各说话人段的首个有效起音时间与最晚有效尾音时间
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
                                s_vals = [v for v in s_vals if v is not None]
                                e_vals = [v for v in e_vals if v is not None]
                                s_min = min(s_vals) if s_vals else min_st
                                e_max = max(e_vals) if e_vals else (s_min + 0.1)
                                seg_spans.append((s_min, e_max))

                            import functools

                            def _cmp_segs(a_info: Tuple[int, str], b_info: Tuple[int, str]) -> int:
                                idx_a, sp_a = a_info
                                idx_b, sp_b = b_info
                                st_a, en_a = seg_spans[idx_a]
                                st_b, en_b = seg_spans[idx_b]

                                # 规则 1：尾音检测。若一方的尾音在另一方起音之前（先说完后开口），严格按对话先后顺序排列
                                if en_a <= st_b + 0.15 and st_a < st_b:
                                    return -1
                                elif en_b <= st_a + 0.15 and st_b < st_a:
                                    return 1

                                # 规则 2：起音时间差 >= 1.0s，非声学微小抖动，先开口者必须居左
                                if abs(st_a - st_b) >= 1.00:
                                    return -1 if st_a < st_b else 1

                                # 规则 3：起音差 < 1.0s 且时间高度重叠的并发抢话，遵循主视角角色居左
                                pref_a = float(spk_persp_pref.get(sp_a, 0.0))
                                pref_b = float(spk_persp_pref.get(sp_b, 0.0))
                                score_a = pref_a if side_is_B else -pref_a
                                score_b = pref_b if side_is_B else -pref_b
                                if abs(score_a - score_b) > 1e-4:
                                    return -1 if score_a < score_b else 1

                                return -1 if st_a < st_b else (1 if st_a > st_b else (idx_a - idx_b))

                            sorted_indices = sorted(
                                [(i, segments[i][0]) for i in range(len(segments))],
                                key=functools.cmp_to_key(_cmp_segs)
                            )
                            sorted_segs = [segments[idx] for idx, _ in sorted_indices]
                            rebuilt_line: List[Dict[str, Any]] = []
                            for seg_idx, (sp_c, seg_items) in enumerate(sorted_segs):
                                for it in seg_items:
                                    sp_tup = it.get(F.SPK)
                                    if sp_tup:
                                        base_sp0 = str(sp_tup[0]).strip().lstrip("　")
                                        new_sp0 = base_sp0 if seg_idx == 0 else ("　" + base_sp0)
                                        it[F.SPK] = (new_sp0,) + tuple(sp_tup[1:])
                                    rebuilt_line.append(it)
                            # 统一规范化行内各有效 token 的时间戳，确保首尾严格为 [min_st, max_en]
                            first_st_str = _format_time(max(0.0, min(perspective_duration_sec, min_st)))
                            last_en_str = _format_time(max(0.0, min(perspective_duration_sec, max_en)))
                            surviving_toks = [
                                it for it in rebuilt_line if not it.get(F.ONO_DROPPED) and ("start" in it or "end" in it)
                            ]
                            if surviving_toks:
                                surviving_toks[0]["start"] = first_st_str
                                surviving_toks[-1]["end"] = last_en_str
                                # 各说话人段内部保持各自时间单调
                                for seg_idx, (sp_c, seg_items) in enumerate(sorted_segs):
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
    clause_speakers: Sequence[Optional[str]],
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
    """从两个视角的音频文件提取声学上下文，并执行多视角动态联合对齐。"""
    def log(msg: str) -> None:
        if log_callback:
            log_callback(msg)

    log(f"  🎧 [视角 A 声学提取] {Path(audio_path_A).name}")
    ctx_A = ams.extract_acoustic_context(
        audio_file_path=audio_path_A,
        lora_path=lora_path,
        device=dev,
        chunk_frames=int(vram_chunk),
    )
    log(f"  🎧 [视角 B 声学提取] {Path(audio_path_B).name}")
    ctx_B = ams.extract_acoustic_context(
        audio_file_path=audio_path_B,
        lora_path=lora_path,
        device=dev,
        chunk_frames=int(vram_chunk),
    )
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

