# -*- coding: utf-8 -*-
"""FA-ASMR V2 — 零依赖立体声时频空间分离工具 (ILD + IPD 软掩码, 支持 K=2 / K=3 路)。

面向无台本多人人头麦 ASMR 听写与重叠段辅助分析：
  - 纯 numpy / scipy / soundfile 实现，零大模型权重、零 GPU 依赖、原生采样率无损保高频气音；
  - 从双声道人头麦立体声按 ILD（双耳声级差）+ 低频无模糊 ITD（双耳时间差）聚类，
    支持将双人（左/右）或三人（左/中/右）叠声拆为 2~3 条独立干声 WAV；
  - 支持单段切分（--start / --end）或全轨自动定位真叠声区间（--auto-overlaps）。

用法示例：
  # 1. 分离指定重叠片段（3 人左/中/右）
  python fa_asmr_spatial_sep.py --audio track.flac --start 1267.0 --end 1275.0 --sources 3 --out-dir ./sep_out --json

  # 2. 自动扫描整轨真叠声区间，并仅对叠声段做 3 路空间分离
  python fa_asmr_spatial_sep.py --audio track.flac --auto-overlaps --sources 3 --out-dir ./sep_overlaps --json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.signal as sig
from scipy.ndimage import uniform_filter


def _choose_stft_params(sr: int) -> Tuple[int, int]:
    """按采样率自适应 ~64ms 窗长与 ~16ms 步进，兼顾 16kHz / 44.1kHz / 48kHz 原生保真。"""
    if sr >= 40000:
        return 2048, 512
    if sr >= 22000:
        return 1024, 256
    return 1024, 256


def separate_stereo_spatial(
    wav_lr: np.ndarray,
    sr: int = 16000,
    n_sources: int = 3,
    sigma2: float = 0.45,
    smooth_size: int = 3,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """零依赖双耳立体声 ILD+IPD 时频空间软掩码分离（从 2 声道拆 K=2 或 K=3 路）。

    参数：
      wav_lr: shape ``(2, N)`` float32 立体声波形；
      sr:     采样率（原生支持 16kHz / 44.1kHz / 48kHz）；
      n_sources: 目标分离路数（>= 2，常用 2=左/右，3=左/中/右）；
      sigma2: 高斯软掩码温度（越大越柔和、越少音乐噪声，越小分离越锐利）；
      smooth_size: 时频掩码平滑核尺寸（抑制耳语清辅音处的音乐噪声）。

    返回：
      ``(sep_wavs, channel_meta)``，其中 ``sep_wavs`` 形状 ``(K, N)``，按从左到右（ILD 降序）排列。
    """
    if wav_lr.ndim != 2 or wav_lr.shape[0] < 2:
        raise ValueError(f"输入必须为双声道立体声 (2, N)，实际形状: {wav_lr.shape}")
    if n_sources < 2 or n_sources > 4:
        raise ValueError(f"n_sources 必须在 2..4 之间，实际为: {n_sources}")

    _, N = wav_lr.shape
    if N < 1024:
        raise ValueError(f"音频片段过短 ({N} 采样点)，无法进行 STFT 空间分离")

    n_fft, hop = _choose_stft_params(sr)
    freqs, _, Xl = sig.stft(wav_lr[0].astype(np.float64), fs=sr, nperseg=n_fft, noverlap=n_fft - hop, window="hann")
    _, _, Xr = sig.stft(wav_lr[1].astype(np.float64), fs=sr, nperseg=n_fft, noverlap=n_fft - hop, window="hann")
    F_bins, T_frames = Xl.shape
    eps = 1e-12

    Pl = np.abs(Xl) ** 2
    Pr = np.abs(Xr) ** 2
    Ptot = Pl + Pr + eps

    # 1. 双耳声级差 ILD (dB): 正值=偏左，负值=偏右
    ild_db = np.clip(10.0 * np.log10((Pl + eps) / (Pr + eps)), -18.0, 18.0)

    # 2. 低频无相位模糊双耳时间差 ITD (ms): 仅在 200..1600 Hz 启用
    ipd = np.angle(Xl * np.conj(Xr))
    f_safe = np.maximum(freqs[:, None], 50.0)
    itd_ms = np.clip(-ipd / (2.0 * np.pi * f_safe) * 1000.0, -0.8, 0.8)
    itd_w = np.where((freqs >= 200.0) & (freqs <= 1600.0), 1.0, 0.0)[:, None]

    # 3. 抽取语音频带高能量时频点做幅度加权 K-means 聚类
    sb = (freqs >= 200.0) & (freqs <= 4000.0)
    P_sb = Ptot[sb]
    thr = float(np.percentile(P_sb, 70))
    sel = P_sb >= thr
    if not np.any(sel):
        sel = np.ones_like(P_sb, dtype=bool)
    ild_pts = ild_db[sb][sel]
    itd_pts = (itd_ms[sb] * itd_w[sb])[sel]
    w_pts = np.sqrt(P_sb[sel])

    f1 = ild_pts / 6.0
    f2 = itd_pts / 0.35
    feats = np.stack([f1, f2], axis=1)

    qs = np.linspace(15.0, 85.0, n_sources)
    c_ild = np.percentile(f1, qs)
    centers = np.stack([c_ild, np.zeros(n_sources, dtype=np.float64)], axis=1)

    for _ in range(20):
        d2 = np.sum((feats[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
        lbl = np.argmin(d2, axis=1)
        for k in range(n_sources):
            mk = lbl == k
            if np.any(mk):
                wk = w_pts[mk]
                centers[k] = np.sum(feats[mk] * wk[:, None], axis=0) / (float(np.sum(wk)) + eps)

    # 按从左到右（ILD 降序）固定通道顺序
    order = np.argsort(-centers[:, 0])
    centers = centers[order]

    # 4. 全频带高斯软掩码（>1600Hz 仅用 ILD，<=1600Hz 融合 ILD+ITD）
    f1_all = ild_db / 6.0
    f2_all = itd_ms / 0.35
    log_resp = np.zeros((n_sources, F_bins, T_frames), dtype=np.float64)
    for k in range(n_sources):
        d_ild = (f1_all - centers[k, 0]) ** 2
        d_itd = ((f2_all - centers[k, 1]) ** 2) * itd_w
        log_resp[k] = -(d_ild + 0.6 * d_itd) / (2.0 * float(sigma2))

    log_resp -= np.max(log_resp, axis=0, keepdims=True)
    masks = np.exp(log_resp)
    masks /= np.sum(masks, axis=0, keepdims=True) + eps

    if smooth_size >= 2:
        for k in range(n_sources):
            masks[k] = uniform_filter(masks[k], size=(smooth_size, smooth_size))
        masks /= np.sum(masks, axis=0, keepdims=True) + eps

    # 5. 按各路空间方位择优参考声道重构时域波形
    out_wavs: List[np.ndarray] = []
    meta: List[Dict[str, Any]] = []
    for k in range(n_sources):
        c_db = float(centers[k, 0] * 6.0)
        if c_db > 1.5:
            X_ref = Xl
            pos_tag = "L"
        elif c_db < -1.5:
            X_ref = Xr
            pos_tag = "R"
        else:
            X_ref = 0.5 * (Xl + Xr)
            pos_tag = "C"
        Z_k = X_ref * masks[k]
        _, y_t = sig.istft(Z_k, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, window="hann")
        y_t = y_t[:N].astype(np.float32)
        if len(y_t) < N:
            y_t = np.pad(y_t, (0, N - len(y_t)))
        rms_db = float(10.0 * np.log10(float(np.mean(y_t.astype(np.float64) ** 2)) + 1e-12))
        out_wavs.append(y_t)
        meta.append({
            "channel_index": k + 1,
            "position_tag": pos_tag,
            "centroid_ild_db": round(c_db, 1),
            "centroid_itd_ms": round(float(centers[k, 1] * 0.35), 3),
            "rms_db": round(rms_db, 1),
        })

    return np.stack(out_wavs, axis=0), meta


def detect_overlap_windows(
    wav_lr: np.ndarray,
    sr: int,
    min_dur_sec: float = 1.5,
    merge_gap_sec: float = 1.2,
    pad_sec: float = 0.4,
) -> List[Tuple[float, float]]:
    """纯音频扫描整轨中的多声源重叠区间（基于 180..1800Hz STFT 相干性 MSC、消去残差与 ILD）。"""
    if wav_lr.ndim != 2 or wav_lr.shape[0] < 2:
        return []
    n_samples = wav_lr.shape[1]
    dur_tot = n_samples / float(sr)
    if dur_tot < min_dur_sec:
        return []

    n_fft, hop = _choose_stft_params(sr)
    hop_sec = hop / float(sr)
    freqs, _, zl = sig.stft(wav_lr[0].astype(np.float32), fs=sr, nperseg=n_fft, noverlap=n_fft - hop, window="hann")
    _, _, zr = sig.stft(wav_lr[1].astype(np.float32), fs=sr, nperseg=n_fft, noverlap=n_fft - hop, window="hann")
    sb = (freqs >= 180.0) & (freqs <= 1800.0)
    fl = zl[sb]
    fr = zr[sb]
    fe = np.sum((np.abs(fl) ** 2) + (np.abs(fr) ** 2) + 1e-24, axis=0)
    T = fl.shape[1]
    e_floor = max(1e-24, 0.05 * float(np.percentile(fe, 85)))

    win_f = max(12, int(round(0.32 / hop_sec)))
    step_f = max(3, int(round(0.10 / hop_sec)))

    rows: List[Tuple[float, float, float, float, float]] = []
    for k0 in range(0, max(1, T - win_f + 1), step_f):
        k1 = min(T, k0 + win_f)
        if k1 - k0 < 8 or float(np.mean(fe[k0:k1])) < e_floor:
            continue
        xl, xr = fl[:, k0:k1], fr[:, k0:k1]
        rll = np.mean(np.abs(xl) ** 2, axis=1)
        rrr = np.mean(np.abs(xr) ** 2, axis=1)
        rlr = np.mean(xl * np.conj(xr), axis=1)
        sum_l = float(np.sum(rll))
        sum_r = float(np.sum(rrr))
        abs_ild = abs(10.0 * float(np.log10(max(1e-24, sum_l) / max(1e-24, sum_r))))
        tr = rll + rrr + 1e-24
        top_f = tr >= np.percentile(tr, 40)
        msc = float(
            np.sum((np.abs(rlr[top_f]) ** 2) / np.maximum(1e-24, rll[top_f] * rrr[top_f]) * tr[top_f])
            / float(np.sum(tr[top_f]))
        )
        res_l = np.mean(np.abs(xl - (rlr / np.maximum(1e-24, rrr))[:, None] * xr) ** 2, axis=1)
        res_r = np.mean(np.abs(xr - (np.conj(rlr) / np.maximum(1e-24, rll))[:, None] * xl) ** 2, axis=1)
        cres = float(min(np.sum(res_l) / max(1e-24, sum_l), np.sum(res_r) / max(1e-24, sum_r)))
        rows.append((k0 * hop_sec, k1 * hop_sec, cres, msc, abs_ild))

    if not rows:
        return []

    # 仅用非极强单耳贴麦窗标定本轨双耳基线
    non_extreme = [(c, m) for _, _, c, m, ild in rows if ild < 12.0]
    if not non_extreme:
        return []
    base_cres = min(0.16, float(np.percentile([c for c, _ in non_extreme], 30)))
    base_msc = max(0.70, float(np.percentile([m for _, m in non_extreme], 70)))
    msc_thr = max(0.66, 0.85 * base_msc)
    cres_thr = max(0.22, 1.50 * base_cres)

    ov_spans: List[Tuple[float, float]] = []
    for st, en, cres, msc, abs_ild in rows:
        is_solo = (abs_ild >= 12.0) or (msc >= msc_thr and cres <= cres_thr)
        if not is_solo:
            if ov_spans and st <= ov_spans[-1][1] + merge_gap_sec:
                ov_spans[-1] = (ov_spans[-1][0], max(ov_spans[-1][1], en))
            else:
                ov_spans.append((st, en))

    out: List[Tuple[float, float]] = []
    for st, en in ov_spans:
        if (en - st) >= min_dur_sec:
            out.append((round(max(0.0, st - pad_sec), 2), round(min(dur_tot, en + pad_sec), 2)))
    return out


def _probe_audio_duration(audio_path: str) -> float:
    """获取音频总时长（秒），优先 soundfile，mp3/m4a 自动回退 torchaudio。"""
    import soundfile as sf

    try:
        info = sf.info(audio_path)
        return (info.frames / float(info.samplerate)) if info.samplerate > 0 else 0.0
    except Exception:
        import torchaudio
        info = torchaudio.info(audio_path)
        return (info.num_frames / float(info.sample_rate)) if info.sample_rate > 0 else 0.0


def _read_stereo_audio(audio_path: str, start_sec: Optional[float] = None, end_sec: Optional[float] = None) -> Tuple[np.ndarray, int]:
    """读取完整或局部立体声音频（优先 soundfile，自动兼容 flac/wav/ogg；mp3/m4a 回退 torchaudio）。"""
    import soundfile as sf

    try:
        info = sf.info(audio_path)
        sr = int(info.samplerate)
        total_f = int(info.frames)
        f_start = max(0, int(round((start_sec or 0.0) * sr)))
        f_stop = total_f if end_sec is None else min(total_f, int(round(end_sec * sr)))
        if f_stop <= f_start:
            raise ValueError(f"无效时间区间: start={start_sec}, end={end_sec} (总时长 {total_f / sr:.2f}s)")
        data, sr = sf.read(audio_path, start=f_start, stop=f_stop, dtype="float32", always_2d=True)
        return data.T, sr
    except Exception:
        import torchaudio
        info = torchaudio.info(audio_path)
        sr = int(info.sample_rate)
        total_f = int(info.num_frames)
        f_start = max(0, int(round((start_sec or 0.0) * sr)))
        f_stop = total_f if end_sec is None else min(total_f, int(round(end_sec * sr)))
        if f_stop <= f_start:
            raise ValueError(f"无效时间区间: start={start_sec}, end={end_sec}")
        wav, sr = torchaudio.load(audio_path, frame_offset=f_start, num_frames=f_stop - f_start)
        return wav.detach().cpu().numpy().astype(np.float32), int(sr)


def has_stereo_spatial_spread(
    wav_lr: np.ndarray,
    sr: int,
    min_ild_span_db: float = 6.0,
    min_cres: float = 0.20,
    max_msc: float = 0.72,
) -> Tuple[bool, Dict[str, float]]:
    """快速判定一段双声道音频是否存在多声源立体声重叠/双耳分立（适合做空间分离）。

    原理（针对真实人头麦 KU100 HRTF 校准）：
      单一声源即使偏耳（不同频段因头影效应产生 ILD 倾斜），在单频点跨时间上仍满足
      秩-1 线性传递函数 X_L(f,t) ≈ H(f)·X_R(f,t)，因此相干度 MSC 极高 (>0.80)、
      跨声道消去残差 cres 极低 (<0.15)；
      而多人叠声或不同方位交替发声会破坏秩-1 关系（cres 升至 >=0.20 / MSC 降至 <=0.72），
      或在左右双耳同时出现强能量分立（p90 >= +6dB 且 p10 <= -6dB）。
    """
    zero_info = {
        "ild_span_db": 0.0,
        "ild_iqr_db": 0.0,
        "ild_median_db": 0.0,
        "msc": 1.0,
        "cres": 0.0,
    }
    if wav_lr.ndim != 2 or wav_lr.shape[0] < 2 or wav_lr.shape[1] < 1024:
        return False, zero_info

    n_fft, hop = _choose_stft_params(sr)
    freqs, _, Xl = sig.stft(wav_lr[0].astype(np.float64), fs=sr, nperseg=n_fft, noverlap=n_fft - hop, window="hann")
    _, _, Xr = sig.stft(wav_lr[1].astype(np.float64), fs=sr, nperseg=n_fft, noverlap=n_fft - hop, window="hann")
    eps = 1e-12
    Pl = np.abs(Xl) ** 2
    Pr = np.abs(Xr) ** 2
    Ptot = Pl + Pr + eps

    # 1. 语音频带高能量时频点 ILD 分布
    sb = (freqs >= 200.0) & (freqs <= 4000.0)
    P_sb = Ptot[sb]
    if P_sb.size < 16 or float(np.max(P_sb)) < 1e-10:
        return False, zero_info

    thr = max(1e-10, float(np.percentile(P_sb, 75)))
    sel = P_sb >= thr
    if int(np.sum(sel)) < 8:
        return False, zero_info

    ild_db = np.clip(10.0 * np.log10((Pl[sb][sel] + eps) / (Pr[sb][sel] + eps)), -18.0, 18.0)
    p10, p25, p50, p75, p90 = [float(x) for x in np.percentile(ild_db, [10, 25, 50, 75, 90])]
    span = p90 - p10
    iqr = p75 - p25

    # 2. 低中频 (180..1800Hz) 双耳相干度 MSC 与跨声道线性消去残差 cres
    sb_coh = (freqs >= 180.0) & (freqs <= 1800.0)
    xl, xr = Xl[sb_coh], Xr[sb_coh]
    if xl.size > 0 and xl.shape[1] >= 4:
        rll = np.mean(np.abs(xl) ** 2, axis=1)
        rrr = np.mean(np.abs(xr) ** 2, axis=1)
        rlr = np.mean(xl * np.conj(xr), axis=1)
        tr = rll + rrr + 1e-24
        top_f = tr >= np.percentile(tr, 40)
        msc = float(
            np.sum((np.abs(rlr[top_f]) ** 2) / np.maximum(1e-24, rll[top_f] * rrr[top_f]) * tr[top_f])
            / max(1e-24, float(np.sum(tr[top_f])))
        )
        res_l = np.mean(np.abs(xl - (rlr / np.maximum(1e-24, rrr))[:, None] * xr) ** 2, axis=1)
        res_r = np.mean(np.abs(xr - (np.conj(rlr) / np.maximum(1e-24, rll))[:, None] * xl) ** 2, axis=1)
        cres = float(min(
            np.sum(res_l) / max(1e-24, float(np.sum(rll))),
            np.sum(res_r) / max(1e-24, float(np.sum(rrr))),
        ))
    else:
        msc, cres = 1.0, 0.0

    # 3. 综合判据：
    #    (a) 真实人头麦多源叠声：破秩-1 (cres>=0.20 或 msc<=0.72) 且有显著 ILD 跨度；
    #    (b) 强左右分立双声源：同时存在显著偏左 (p90>=+6dB) 与显著偏右 (p10<=-6dB) 且整体能量非底噪
    rms_all = float(np.sqrt(np.mean(wav_lr[:2].astype(np.float64) ** 2)))
    rank2_overlap = (cres >= float(min_cres) or msc <= float(max_msc)) and (span >= float(min_ild_span_db))
    bilateral_split = (p90 >= 6.0) and (p10 <= -6.0) and (iqr >= 6.0) and (rms_all >= 3e-3)
    is_spread = bool(rank2_overlap or bilateral_split)
    return is_spread, {
        "ild_span_db": round(span, 2),
        "ild_iqr_db": round(iqr, 2),
        "ild_median_db": round(p50, 2),
        "msc": round(msc, 3),
        "cres": round(cres, 3),
    }


def _merge_interval_items(
    items: Sequence[Dict[str, Any]],
    merge_gap_sec: float = 1.5,
) -> List[Dict[str, Any]]:
    """把时间相邻或重叠的体检条目合并为连续审听窗口，避免重复切出高度重叠的短片段。"""
    valid = [
        it for it in items
        if it.get("start") is not None and it.get("end") is not None and float(it["end"]) > float(it["start"])
    ]
    valid.sort(key=lambda x: (float(x["start"]), float(x["end"])))
    merged: List[Dict[str, Any]] = []
    for it in valid:
        st, en = float(it["start"]), float(it["end"])
        if merged and st <= merged[-1]["end"] + float(merge_gap_sec):
            merged[-1]["end"] = max(merged[-1]["end"], en)
            merged[-1]["items"].append(it)
        else:
            merged.append({"start": st, "end": en, "items": [it]})
    return merged


def _load_audit_tasks(audit_json_path: str, audio_override: Optional[str] = None) -> List[Dict[str, Any]]:
    """兼容读取 fa_asmr_cli --audit-out / --json-out 报告或单轨 audit dict。"""
    with open(audit_json_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"体检报告顶层必须为 JSON 对象: {audit_json_path}")

    tasks_out: List[Dict[str, Any]] = []
    if isinstance(data.get("tasks"), list):
        for t in data["tasks"]:
            if not isinstance(t, dict):
                continue
            aud = audio_override or t.get("audio")
            aud_b = None if audio_override else t.get("audio_b")
            audit_obj = t.get("audit")
            if isinstance(audit_obj, dict) and aud:
                if audit_obj.get("is_multiperspective") and aud_b:
                    # 双视角联合体检：按条目 perspective 字段分别映射到视角 A / 视角 B 音频
                    for side_tag, side_aud in (("A", aud), ("B", aud_b)):
                        sub_audit = {
                            "suspect_clauses": [
                                it for it in audit_obj.get("suspect_clauses", [])
                                if (str(it.get("perspective", "A")).upper() == side_tag)
                            ],
                            "unaligned_speech_gaps": [
                                it for it in audit_obj.get("unaligned_speech_gaps", [])
                                if (str(it.get("perspective", "A")).upper() == side_tag)
                            ],
                        }
                        if sub_audit["suspect_clauses"] or sub_audit["unaligned_speech_gaps"]:
                            tasks_out.append({
                                "name": os.path.splitext(os.path.basename(str(side_aud)))[0],
                                "audio": os.path.abspath(str(side_aud)),
                                "audit": sub_audit,
                            })
                else:
                    tasks_out.append({
                        "name": t.get("name") or os.path.splitext(os.path.basename(str(aud)))[0],
                        "audio": os.path.abspath(str(aud)),
                        "audit": audit_obj,
                    })
            # 兼容旧版双视角独立 audit_b 结构
            audit_b = t.get("audit_b")
            if isinstance(audit_b, dict) and aud_b:
                tasks_out.append({
                    "name": os.path.splitext(os.path.basename(str(aud_b)))[0],
                    "audio": os.path.abspath(str(aud_b)),
                    "audit": audit_b,
                })
    elif "suspect_clauses" in data or "unaligned_speech_gaps" in data:
        if not audio_override:
            raise ValueError("单轨 audit JSON 未包含音频路径，请同时提供 --audio <音频文件>")
        tasks_out.append({
            "name": os.path.splitext(os.path.basename(audio_override))[0],
            "audio": os.path.abspath(audio_override),
            "audit": data,
        })
    else:
        raise ValueError(f"无法识别的体检报告格式（缺少 tasks 或 suspect_clauses）: {audit_json_path}")
    return tasks_out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fa_asmr_spatial_sep.py",
        description="FA-ASMR 零依赖立体声 ILD+IPD 时频空间分离工具（支持 2/3 路分离、自动叠声定位与 --from-audit 联动）",
    )
    p.add_argument("--audio", default=None, help="输入双声道立体声音频路径（使用 --from-audit 且报告已含 audio 时可省略）")
    p.add_argument("--out-dir", required=True, help="分离后 WAV 输出目录")
    p.add_argument("--sources", type=int, default=3, choices=(2, 3, 4), help="分离路数（默认 3=左/中/右，2=左/右）")
    p.add_argument("--start", type=float, default=None, help="单段模式：起始秒数（默认 0）")
    p.add_argument("--end", type=float, default=None, help="单段模式：结束秒数（默认到末尾）")
    p.add_argument("--auto-overlaps", action="store_true", help="自动扫描整轨真叠声区间，仅对叠声段切片并分离")
    p.add_argument("--from-audit", metavar="JSON", default=None, help="直接读取第二步 --audit-out / --json-out 体检报告，按异常句与漏句空档定向切片并分离")
    p.add_argument("--only-stereo-spread", action="store_true", help="配合 --from-audit 使用：仅对检测到立体声多方位分布的重叠片段做分离，跳过居中单声源")
    p.add_argument("--pad-sec", type=float, default=1.0, help="--from-audit 切片前后缓冲秒数（默认 1.0）")
    p.add_argument("--merge-gap-sec", type=float, default=1.5, help="--from-audit 相邻异常窗口合并间隔秒数（默认 1.5）")
    p.add_argument("--min-overlap-sec", type=float, default=1.5, help="--auto-overlaps 最小叠声持续秒数（默认 1.5）")
    p.add_argument("--sigma2", type=float, default=0.45, help="软掩码温度参数（默认 0.45）")
    p.add_argument("--json", action="store_true", help="以结构化 JSON 输出到 stdout")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    import soundfile as sf

    parser = build_parser()
    args = parser.parse_args(argv)
    out_dir = os.path.abspath(args.out_dir)

    if args.from_audit and args.auto_overlaps:
        payload = {"ok": False, "error": "不能同时指定 --from-audit 与 --auto-overlaps"}
        print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else f"[ERROR] {payload['error']}", file=(sys.stdout if args.json else sys.stderr))
        return 2

    if not args.from_audit and not args.audio:
        payload = {"ok": False, "error": "请提供 --audio <音频路径>，或通过 --from-audit <audit.json> 传入体检报告"}
        print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else f"[ERROR] {payload['error']}", file=(sys.stdout if args.json else sys.stderr))
        return 2

    if args.from_audit:
        audit_abs = os.path.abspath(args.from_audit)
        if not os.path.isfile(audit_abs):
            payload = {"ok": False, "error": f"体检报告不存在: {audit_abs}"}
            print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else f"[ERROR] {payload['error']}", file=(sys.stdout if args.json else sys.stderr))
            return 2
        os.makedirs(out_dir, exist_ok=True)
        try:
            audit_tasks = _load_audit_tasks(audit_abs, audio_override=os.path.abspath(args.audio) if args.audio else None)
            all_segments: List[Dict[str, Any]] = []
            sr_last = 0
            for t in audit_tasks:
                aud_p = t["audio"]
                if not os.path.isfile(aud_p):
                    raise FileNotFoundError(f"体检报告指向的音频文件不存在: {aud_p}")
                stem = t["name"]
                audit_obj = t["audit"]
                dur_tot = _probe_audio_duration(aud_p)
                for kind, key in (("suspect", "suspect_clauses"), ("gap", "unaligned_speech_gaps")):
                    windows = _merge_interval_items(audit_obj.get(key, []), merge_gap_sec=args.merge_gap_sec)
                    for idx, win in enumerate(windows, 1):
                        a_st = round(max(0.0, win["start"] - float(args.pad_sec)), 2)
                        a_en = round(min(dur_tot, win["end"] + float(args.pad_sec)), 2) if dur_tot > 0 else round(win["end"] + float(args.pad_sec), 2)
                        if a_en <= a_st + 0.1:
                            continue
                        wav_lr, sr = _read_stereo_audio(aud_p, start_sec=a_st, end_sec=a_en)
                        sr_last = sr
                        if wav_lr.shape[0] < 2:
                            raise ValueError(f"输入音频为单声道，无法进行立体声空间分离: {aud_p}")
                        is_spread, spread_info = has_stereo_spatial_spread(wav_lr[:2], sr=sr)
                        tag = f"{stem}_{kind}{idx:02d}_{a_st:.1f}s_{a_en:.1f}s"
                        orig_p = os.path.join(out_dir, f"{tag}_orig.wav")
                        sf.write(orig_p, wav_lr[:2].T, sr)
                        ch_files: List[Dict[str, Any]] = []
                        separated = False
                        if (not args.only_stereo_spread) or is_spread:
                            sep_wavs, meta = separate_stereo_spatial(
                                wav_lr[:2], sr=sr, n_sources=args.sources, sigma2=args.sigma2
                            )
                            for k, cm in enumerate(meta):
                                ch_p = os.path.join(out_dir, f"{tag}_K{k+1}_{cm['position_tag']}.wav")
                                sf.write(ch_p, sep_wavs[k], sr)
                                ch_files.append({**cm, "wav": ch_p})
                            separated = True
                        lines_hit = sorted({int(it["line"]) for it in win["items"] if it.get("line") is not None})
                        spks_hit = sorted({str(it["speaker"]) for it in win["items"] if it.get("speaker")})
                        all_segments.append({
                            "index": len(all_segments) + 1,
                            "task": stem,
                            "audio": aud_p,
                            "kind": kind,
                            "start": a_st,
                            "end": a_en,
                            "duration": round(a_en - a_st, 2),
                            "lines": lines_hit,
                            "speakers": spks_hit,
                            "orig_wav": orig_p,
                            "stereo_spread": {**spread_info, "is_spread": is_spread},
                            "separated": separated,
                            "channels": ch_files,
                        })

            payload = {
                "ok": True,
                "from_audit": audit_abs,
                "sample_rate": sr_last,
                "sources": args.sources,
                "only_stereo_spread": bool(args.only_stereo_spread),
                "segments_count": len(all_segments),
                "separated_count": sum(1 for s in all_segments if s["separated"]),
                "segments": all_segments,
            }
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            else:
                print(
                    f"=== FA-ASMR 体检联动空间分离完成 "
                    f"({payload['segments_count']} 段, 实际分离 {payload['separated_count']} 段, {args.sources} 路) ==="
                )
                for seg in all_segments:
                    if seg["separated"]:
                        chs = ", ".join(f"K{c['channel_index']}({c['position_tag']} {c['centroid_ild_db']:+.1f}dB)" for c in seg["channels"])
                    else:
                        sp_i = seg["stereo_spread"]
                        chs = f"跳过分离(单声源 msc={sp_i['msc']:.2f} cres={sp_i['cres']:.2f})"
                    print(f"  #{seg['index']:02d} [{seg['kind']}] [{seg['start']:.2f}s - {seg['end']:.2f}s] -> {chs}")
            return 0
        except Exception as e:
            payload = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else f"[ERROR] {payload['error']}", file=(sys.stdout if args.json else sys.stderr))
            return 1

    aud_abs = os.path.abspath(args.audio)
    if not os.path.isfile(aud_abs):
        payload = {"ok": False, "error": f"音频文件不存在: {aud_abs}"}
        print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else f"[ERROR] {payload['error']}", file=(sys.stdout if args.json else sys.stderr))
        return 2

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(aud_abs))[0]

    try:
        segments_out: List[Dict[str, Any]] = []
        if args.auto_overlaps:
            wav_full, sr = _read_stereo_audio(aud_abs)
            if wav_full.shape[0] < 2:
                raise ValueError("输入音频为单声道，无法进行立体声空间分离")
            windows = detect_overlap_windows(wav_full[:2], sr=sr, min_dur_sec=args.min_overlap_sec)
            for idx, (st, en) in enumerate(windows, 1):
                s0 = max(0, int(round(st * sr)))
                s1 = min(wav_full.shape[1], int(round(en * sr)))
                seg_lr = wav_full[:2, s0:s1]
                sep_wavs, meta = separate_stereo_spatial(seg_lr, sr=sr, n_sources=args.sources, sigma2=args.sigma2)
                orig_p = os.path.join(out_dir, f"{stem}_ov{idx:02d}_{st:.1f}s_{en:.1f}s_orig.wav")
                sf.write(orig_p, seg_lr.T, sr)
                ch_files = []
                for k, cm in enumerate(meta):
                    ch_p = os.path.join(out_dir, f"{stem}_ov{idx:02d}_{st:.1f}s_{en:.1f}s_K{k+1}_{cm['position_tag']}.wav")
                    sf.write(ch_p, sep_wavs[k], sr)
                    ch_files.append({**cm, "wav": ch_p})
                segments_out.append({"index": idx, "start": st, "end": en, "duration": round(en - st, 2), "orig_wav": orig_p, "channels": ch_files})
        else:
            wav_lr, sr = _read_stereo_audio(aud_abs, start_sec=args.start, end_sec=args.end)
            if wav_lr.shape[0] < 2:
                raise ValueError("输入音频为单声道，无法进行立体声空间分离")
            sep_wavs, meta = separate_stereo_spatial(wav_lr[:2], sr=sr, n_sources=args.sources, sigma2=args.sigma2)
            st = float(args.start or 0.0)
            en = round(st + wav_lr.shape[1] / float(sr), 2)
            tag = f"{st:.1f}s_{en:.1f}s" if (args.start is not None or args.end is not None) else "full"
            ch_files = []
            for k, cm in enumerate(meta):
                ch_p = os.path.join(out_dir, f"{stem}_{tag}_K{k+1}_{cm['position_tag']}.wav")
                sf.write(ch_p, sep_wavs[k], sr)
                ch_files.append({**cm, "wav": ch_p})
            segments_out.append({"index": 1, "start": st, "end": en, "duration": round(en - st, 2), "channels": ch_files})

        payload = {
            "ok": True,
            "audio": aud_abs,
            "sample_rate": sr,
            "sources": args.sources,
            "auto_overlaps": bool(args.auto_overlaps),
            "segments_count": len(segments_out),
            "segments": segments_out,
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(f"=== FA-ASMR 立体声空间分离完成 ({len(segments_out)} 段, {args.sources} 路, {sr}Hz) ===")
            for seg in segments_out:
                chs = ", ".join(f"K{c['channel_index']}({c['position_tag']} {c['centroid_ild_db']:+.1f}dB)" for c in seg["channels"])
                print(f"  #{seg['index']:02d} [{seg['start']:.2f}s - {seg['end']:.2f}s] -> {chs}")
        return 0
    except Exception as e:
        payload = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else f"[ERROR] {payload['error']}", file=(sys.stdout if args.json else sys.stderr))
        return 1


if __name__ == "__main__":
    sys.exit(main())
