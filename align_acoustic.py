# -*- coding: utf-8 -*-
"""FA-ASMR V2 — 纯声学特征提取与双声道 Better-Ear 融合模块

包含：
1. 左右声道独立 RMS 能量计算与自适应底噪门限
2. 线性软渐变边缘保护先验（Soft-Ramped Silence Prior）
3. 基于 MMS-FA CTC 活跃度优势比的双耳融合（Better-Ear Log Mixture, 500ms / T=1.5）
"""
from __future__ import annotations

import math
from typing import Tuple, List, Any, Optional

import numpy as np
import torch

try:
    from numba import njit as _njit
    _HAS_NUMBA = True
except ImportError:  # 无 numba 时回退到纯 numpy 实现 (功能一致, 稍慢)
    _HAS_NUMBA = False

FRAME_SEC: float = 0.02  # 20ms frame step for MMS-FA (hop length = 320 at 16kHz)
DEFAULT_SILENCE_MARGIN_DB: float = 4.0
DEFAULT_ABS_SILENCE_CEILING_DB: float = -45.0


def _frame_rms_grid(audio: np.ndarray, sample_rate: int, target_frames: int,
                    win_sec: float, hop_sec: float):
    """构造与 CTC 帧严格对齐的窗口网格 (starts / ends / valid)."""
    win_len = int(round(win_sec * sample_rate))
    hop_len = int(round(hop_sec * sample_rate))
    total = len(audio)
    starts = np.arange(target_frames, dtype=np.int64) * hop_len
    ends = np.minimum(starts + win_len, total)
    valid = starts < total
    return win_len, hop_len, starts, ends, valid


def _frame_rms_reduce(sums: np.ndarray, starts: np.ndarray, ends: np.ndarray,
                      valid: np.ndarray) -> np.ndarray:
    """把窗口平方和转成与旧实现逐比特一致的 dB 数组。"""
    counts = np.where(valid, ends - starts, 1)
    rms = np.sqrt(sums / counts)
    db = np.where(valid, 20.0 * np.log10(np.maximum(rms, 1e-5)), -100.0)
    return db.astype(np.float32)


if _HAS_NUMBA:

    @_njit(cache=True, fastmath=False)
    def _scan_window_sums(audio, starts, ends, n):
        """单遍扫描取窗口平方和, 省掉 cumsum 中间数组。

        原实现要先构造 float64 cumsum(与音频同长)。对 96kHz 长轨, 那是一个
        与音频等大的临时数组(轨 04 单声道 = 946MB), 写一遍再读两遍。
        这里只需要窗口两端的累积值, 于是在一遍扫描里维护 running sum,
        命中窗口端点时取值 —— float64 加法顺序与 np.cumsum 完全相同,
        因此结果与旧实现逐比特一致, 但省掉整个中间数组的读写。

        游标用 ``<=`` 而非 ``==`` 推进: starts[0] - 1 == -1 永不等于任何
        i >= 0, 用 ``==`` 会让游标永久卡死(实测误差可达 115 dB)。
        """
        total = audio.shape[0]
        s_prev = np.zeros(n, dtype=np.float64)
        s_end = np.zeros(n, dtype=np.float64)
        acc = 0.0
        p = 0
        q = 0
        for i in range(total):
            v = np.float64(audio[i])
            acc += v * v
            while p < n and starts[p] - 1 <= i:
                if starts[p] > 0:
                    s_prev[p] = acc
                p += 1
            while q < n and ends[q] - 1 <= i:
                s_end[q] = acc
                q += 1
            if q >= n and p >= n:
                break
        out = np.empty(n, dtype=np.float64)
        for f in range(n):
            if starts[f] > 0:
                out[f] = s_end[f] - s_prev[f]
            else:
                out[f] = s_end[f]
        return out

    @_njit(cache=True, fastmath=False)
    def _scan_window_sums_stereo(lr, starts, ends, n):
        """立体声交织缓冲 (零拷贝, 单遍): 同时得到 L / R / mono 三路窗口平方和。

        ``torchaudio.load`` 返回的 96kHz 立体声在内存里是**采样交织**的
        (底层缓冲 L0 R0 L1 R1 ...), ``waveform.numpy().T`` 恰是 (N, 2) 的
        C 连续视图, 但 ``[:, 0]`` 取出来是 stride=8 字节的跨步视图 —— numba
        对跨步切片无法向量化, 实测比连续内存慢 3.8x (轨 04: 0.268s -> 1.020s);
        若先 ``ascontiguousarray`` 拷贝, 又要为每个声道多读写一份与音频等大的
        内存 (轨 04 = 946MB/声道)。

        这里直接遍历交织缓冲, L / R 两个 running sum 连同 mono 一起推进:
        零拷贝、单遍, 且 float64 加法顺序与各自独立的 cumsum 版本一致。

        mono 语义: 与旧实现 ``audio_np.mean(axis=1)`` 完全相同 —— numpy 对
        2 个 float32 求均值即 (L+R)/2, 而 /2 在二进制下精确, 故在 float32
        内先算 (L+R)/2 再转 float64 平方, 结果逐比特一致(已实测验证)。
        """
        total = lr.shape[0]
        pa = np.zeros(n, dtype=np.float64)
        pb = np.zeros(n, dtype=np.float64)
        pc = np.zeros(n, dtype=np.float64)
        ea = np.zeros(n, dtype=np.float64)
        eb = np.zeros(n, dtype=np.float64)
        ec = np.zeros(n, dtype=np.float64)
        aa = 0.0
        ab = 0.0
        ac = 0.0
        p = 0
        q = 0
        for i in range(total):
            va = np.float64(lr[i, 0])
            vb = np.float64(lr[i, 1])
            vm = np.float64(np.float32(lr[i, 0] + lr[i, 1]) / np.float32(2.0))
            aa += va * va
            ab += vb * vb
            ac += vm * vm
            while p < n and starts[p] - 1 <= i:
                if starts[p] > 0:
                    pa[p] = aa
                    pb[p] = ab
                    pc[p] = ac
                p += 1
            while q < n and ends[q] - 1 <= i:
                ea[q] = aa
                eb[q] = ab
                ec[q] = ac
                q += 1
            if q >= n and p >= n:
                break
        oa = np.empty(n, dtype=np.float64)
        ob = np.empty(n, dtype=np.float64)
        oc = np.empty(n, dtype=np.float64)
        for f in range(n):
            if starts[f] > 0:
                oa[f] = ea[f] - pa[f]
                ob[f] = eb[f] - pb[f]
                oc[f] = ec[f] - pc[f]
            else:
                oa[f] = ea[f]
                ob[f] = eb[f]
                oc[f] = ec[f]
        return oa, ob, oc


def compute_frame_rms(
    audio_1d: np.ndarray,
    sample_rate: int,
    target_frames: int,
    win_sec: float = 0.04,
    hop_sec: float = 0.02,
) -> np.ndarray:
    """计算与 CTC 帧严格对齐的 1D 音频逐帧 RMS（以 dB 为单位）。

    用平方累积和整段计算，替代逐帧 Python 循环：长音频（数万帧以上）时
    开销从 O(target_frames) 次 Python 循环 + 每帧 numpy 调用降为一次线性扫描。
    最后一帧窗口越过音频末尾时仍按真实（截断）长度求均值，与旧逐帧逻辑一致。

    numba 可用时走单遍扫描内核（省掉 float64 cumsum 中间数组，实测快 2.6x）；
    不可用时回退到 cumsum 实现，两者输出逐比特一致。
    """
    audio = np.asarray(audio_1d, dtype=np.float32)
    if target_frames <= 0:
        return np.zeros(0, dtype=np.float32)

    win_len, hop_len, starts, ends, valid = _frame_rms_grid(
        audio, sample_rate, target_frames, win_sec, hop_sec)

    if _HAS_NUMBA:
        contig = np.ascontiguousarray(audio)
        sums = _scan_window_sums(contig, starts, ends, target_frames)
        return _frame_rms_reduce(sums, starts, ends, valid)

    # --- 纯 numpy 回退: float64 累积平方和 ---
    # float32 对 1 小时级音频 cumsum 会累积到不可接受的误差。
    # csum[k] = sum(sq[0..k-1])，帧 [st, en) 的平方和 = csum[en-1] - csum[st-1]。
    sq = np.square(audio, dtype=np.float64)
    csum = np.cumsum(sq)
    del sq

    sums = np.zeros(target_frames, dtype=np.float64)
    if valid.any():
        sums[valid] = csum[ends[valid] - 1]
        prev = starts > 0
        if prev.any():
            idx = np.flatnonzero(valid & prev)
            sums[idx] -= csum[starts[idx] - 1]
    del csum
    return _frame_rms_reduce(sums, starts, ends, valid)


def compute_frame_rms_triple(
    stereo: np.ndarray,
    sample_rate: int,
    target_frames: int,
    win_sec: float = 0.04,
    hop_sec: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """一次扫描同时得到 (rms_l, rms_r, rms_mono)，与三次独立调用逐比特一致。

    ``stereo`` 必须是 ``(samples, 2)`` 的 C 连续立体声数组 (即
    ``waveform.numpy().T``)。传入交织布局而非两个声道切片，是为了让内核
    顺序访问内存: 逐声道切片是 stride=8 的跨步视图, 实测慢 3.8x。

    mono 与旧实现 ``audio_np.mean(axis=1)`` 语义完全相同 (2 个 float32 求均值)。
    """
    if target_frames <= 0:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty.copy(), empty.copy()

    lr = np.ascontiguousarray(stereo, dtype=np.float32)
    if lr.ndim != 2 or lr.shape[1] < 2:
        raise ValueError("stereo array must be (samples, 2)")
    if lr.shape[1] > 2:
        lr = lr[:, :2]

    win_len, hop_len, starts, ends, valid = _frame_rms_grid(
        lr[:, 0], sample_rate, target_frames, win_sec, hop_sec)

    if not _HAS_NUMBA:
        mono = lr.mean(axis=1)
        return (compute_frame_rms(lr[:, 0], sample_rate, target_frames, win_sec, hop_sec),
                compute_frame_rms(lr[:, 1], sample_rate, target_frames, win_sec, hop_sec),
                compute_frame_rms(mono, sample_rate, target_frames, win_sec, hop_sec))

    sa, sb, sc = _scan_window_sums_stereo(lr, starts, ends, target_frames)
    return (_frame_rms_reduce(sa, starts, ends, valid),
            _frame_rms_reduce(sb, starts, ends, valid),
            _frame_rms_reduce(sc, starts, ends, valid))


def compute_track_noise_floor(frame_rms_db: np.ndarray, percentile: float = 10.0) -> float:
    """以整轨音频 RMS 的第 10 百分位数作为稳态底噪基准。"""
    valid = np.asarray(frame_rms_db, dtype=np.float64)
    if valid.size == 0:
        return -80.0
    return float(np.percentile(valid, percentile))


def make_silence_mask(
    frame_rms_db: np.ndarray,
    noise_floor_db: float,
    silence_margin_db: float = DEFAULT_SILENCE_MARGIN_DB,
    absolute_ceiling_db: float = DEFAULT_ABS_SILENCE_CEILING_DB,
) -> np.ndarray:
    """生成物理深静音掩码。低于 threshold 即判为物理静音。"""
    threshold = min(float(noise_floor_db) + float(silence_margin_db), float(absolute_ceiling_db))
    return np.asarray(frame_rms_db, dtype=np.float64) <= threshold


def build_head_token_silence_penalties(
    targets: List[List[int]],
    line_group: Optional[Any] = None,
    discount: float = 1.0,
) -> Tuple[np.ndarray, int]:
    """Mark only heads of source lines for the in-trellis silence guard."""
    flat_len = sum(len(line) for line in targets)
    penalties = np.zeros(flat_len, dtype=np.float32)
    offset = len(targets[0]) if targets else 0
    count = 0
    for i in range(len(targets) - 1):
        next_line = targets[i + 1]
        if line_group is not None and line_group[i] == line_group[i + 1]:
            offset += len(next_line)
            continue
        if next_line:
            penalties[offset] = np.float32(discount)
            count += 1
        offset += len(next_line)
    return penalties, count


def build_next_stable_silence_end(
    frame_rms_db: np.ndarray,
    noise_floor_db: float,
    min_run_frames: int = 10,
    max_search_frames: int = 150,
    floor_margin_db: float = 2.5,
    max_spike_db: Optional[float] = None,
    require_pre_silence_island: bool = True,
    island_active_margin_db: float = 4.0,
    island_smooth_frames: int = 5,
    island_bridge_frames: int = 5,
    island_lookback_frames: int = 50,
    island_prepad_frames: int = 4,
) -> np.ndarray:
    """For every frame, return the end of its next qualifying silence run.

    A zero means no stable silence begins within the original three-second
    search horizon.  This is audio-only and can therefore be consumed by the
    trellis while it evaluates each candidate path.

    For a pre-silence candidate, the normal gate requires it to belong to the
    final contiguous acoustic island before that silence. This preserves the
    useful "do not cross an intervening sound event" behavior without treating
    the CTC token boundary as an acoustic peak. ``max_spike_db`` remains only
    as an optional legacy A/B veto; it is not used by the production profile.
    """
    rms = np.asarray(frame_rms_db, dtype=np.float64)
    result = np.zeros(len(rms), dtype=np.int32)
    threshold = float(noise_floor_db) + float(floor_margin_db)
    smooth_n = max(1, int(island_smooth_frames))
    if smooth_n > 1:
        pad_left = smooth_n // 2
        pad_right = smooth_n - 1 - pad_left
        padded = np.pad(rms, (pad_left, pad_right), mode="edge")
        island_rms = np.convolve(
            padded, np.full(smooth_n, 1.0 / smooth_n, dtype=np.float64),
            mode="valid")
    else:
        island_rms = rms
    island_active = island_rms > (float(noise_floor_db) + float(island_active_margin_db))
    runs: List[Tuple[int, int]] = []
    t = 0
    while t + min_run_frames <= len(rms):
        window = rms[t:t + min_run_frames]
        if np.all(window <= threshold) and float(np.max(window) - np.min(window)) <= 2.5:
            end = t + min_run_frames
            while end < len(rms) and rms[end] <= threshold:
                end += 1
            runs.append((t, end))
            t = end
        else:
            t += 1

    for start, end in runs:
        lo = max(0, start - max_search_frames + min_run_frames + 1)
        anchor = lo
        if require_pre_silence_island:
            island_lo = max(lo, start - max(1, int(island_lookback_frames)))
            last_active = -1
            for frame in range(start - 1, island_lo - 1, -1):
                if island_active[frame]:
                    last_active = frame
                    break
            if last_active >= 0:
                island_start = last_active
                inactive = 0
                for frame in range(last_active - 1, island_lo - 1, -1):
                    if island_active[frame]:
                        island_start = frame
                        inactive = 0
                    else:
                        inactive += 1
                        if inactive > max(0, int(island_bridge_frames)):
                            break
                        island_start = frame
                anchor = max(island_lo, island_start - max(0, int(island_prepad_frames)))
            else:
                # A stable silence without a nearby preceding sound island is
                # not evidence that a much earlier token owns that silence.
                anchor = start

        largest_rise = -np.inf
        for frame in range(start - 1, anchor - 1, -1):
            largest_rise = max(largest_rise, float(rms[frame + 1] - rms[frame]))
            if (result[frame] == 0 and
                    (max_spike_db is None or largest_rise <= max_spike_db)):
                result[frame] = min(end, frame + max_search_frames)
        valid_end = end - min_run_frames + 1
        for frame in range(start, valid_end):
            result[frame] = min(end, frame + max_search_frames)
    return result


def ctc_nonblank_log_odds(
    log_emission: torch.Tensor,
    blank: int = 0,
    ignored_classes: Tuple[int, ...] = (),
) -> np.ndarray:
    """计算 MMS-FA 的非 blank 与 blank 对数优势比（语音活跃度特征）。"""
    ignored = {int(v) for v in ignored_classes}
    classes = [
        idx for idx in range(int(log_emission.shape[-1]))
        if idx != int(blank) and idx not in ignored
    ]
    if not classes:
        raise ValueError("CTC emission has no usable non-blank classes")
    scores = log_emission[0]
    nonblank = torch.logsumexp(scores[:, classes], dim=-1)
    odds = nonblank - scores[:, int(blank)]
    return odds.detach().cpu().numpy().astype(np.float32)


def smooth_frame_values(values: np.ndarray, window_frames: int) -> np.ndarray:
    """居中边缘填充移动平均平滑。"""
    data = np.asarray(values, dtype=np.float64)
    width = max(1, int(window_frames))
    if width == 1 or len(data) == 0:
        return data.astype(np.float32)
    if width % 2 == 0:
        width += 1
    radius = width // 2
    padded = np.pad(data, (radius, radius), mode="edge")
    kernel = np.full(width, 1.0 / width, dtype=np.float64)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def stereo_ctc_better_ear_log_mixture(
    left_log_emission: torch.Tensor,
    right_log_emission: torch.Tensor,
    *,
    blank: int = 0,
    ignored_classes: Tuple[int, ...] = (),
    smoothing_frames: int = 25,
    temperature: float = 1.5,
) -> torch.Tensor:
    """双耳 Better-Ear 动态对数加权融合。

    利用平滑后的左右声道 CTC 活跃度优势比差值，通过 Sigmoid 计算每帧声道权重，
    并在发射层执行对数混合，完美支持 ASMR 声优左右耳穿梭与耳语交替。
    """
    if left_log_emission.shape != right_log_emission.shape:
        raise ValueError("stereo emission shape mismatch")
    left_score = smooth_frame_values(
        ctc_nonblank_log_odds(left_log_emission, blank, ignored_classes),
        smoothing_frames,
    )
    right_score = smooth_frame_values(
        ctc_nonblank_log_odds(right_log_emission, blank, ignored_classes),
        smoothing_frames,
    )
    scale = max(float(temperature), 1e-6)
    delta = np.clip((left_score - right_score) / scale, -8.0, 8.0)
    left_weight = 1.0 / (1.0 + np.exp(-delta))
    right_weight = 1.0 - left_weight
    eps = np.finfo(np.float32).tiny
    left_log_weight = torch.as_tensor(
        np.log(np.maximum(left_weight, eps)),
        dtype=left_log_emission.dtype,
        device=left_log_emission.device,
    )
    right_log_weight = torch.as_tensor(
        np.log(np.maximum(right_weight, eps)),
        dtype=right_log_emission.dtype,
        device=right_log_emission.device,
    )
    return torch.logaddexp(
        left_log_emission + left_log_weight[None, :, None],
        right_log_emission + right_log_weight[None, :, None],
    )


