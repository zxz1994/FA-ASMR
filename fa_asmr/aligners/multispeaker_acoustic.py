# -*- coding: utf-8 -*-
"""FA-ASMR V2 — 多人声学上下文提取、STFT 单声源并发检测与浅层声纹判别。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchaudio

import fa_asmr.aligners.align as align
from fa_asmr.aligners.align_acoustic import (
    FRAME_SEC,
    build_head_token_silence_penalties,
    build_next_stable_silence_end,
    compute_frame_rms_triple,
    compute_track_noise_floor,
    make_silence_mask,
    stereo_ctc_better_ear_log_mixture,
)
from fa_asmr.aligners.align_model import compute_emission
from fa_asmr.aligners.align_post import _post_process_tokens
from fa_asmr.aligners.align_utils import (
    _build_head_standalone_n_flag,
    _canonicalize_line_tokens,
    _format_time,
    find_subclause_ii_folds,
)
from fa_asmr.aligners.multispeaker_decoder import MultiSpeakerCTCAligner

# ══════════════════════════════════════════════════════════════════════════
#  2. 独立声学提取与子句链对齐封装（全轨只跑一次模型前向）
# ══════════════════════════════════════════════════════════════════════════

def _unmasked_clause_mean_score(
    cl_res: Sequence[Dict[str, Any]],
    orig_coop_np: np.ndarray,
    tok_dict: Dict[str, int],
) -> float:
    """在未遮罩原始 CTC 概率图上计算子句真实平均得分，多字符音节按帧比例切分。"""
    vals: List[float] = []
    T_max = int(orig_coop_np.shape[0])
    for r in cl_res:
        if r.get("skipped") or r.get("original_start") is None or r.get("original_end") is None:
            continue
        ch_ids = [tok_dict[ch] for ch in str(r.get("token", "")) if ch in tok_dict]
        if not ch_ids:
            continue
        f_s = max(0, min(T_max - 1, int(round(float(r["original_start"]) / FRAME_SEC))))
        f_e = max(f_s + 1, min(T_max, int(round(float(r["original_end"]) / FRAME_SEC))))
        if len(ch_ids) == 1:
            vals.append(float(np.max(orig_coop_np[f_s:f_e, ch_ids[0]])))
        else:
            bounds = np.linspace(f_s, f_e, len(ch_ids) + 1)
            for idx_ch, cid in enumerate(ch_ids):
                sub_s = max(f_s, min(f_e - 1, int(np.floor(bounds[idx_ch]))))
                sub_e = max(sub_s + 1, min(f_e, int(np.ceil(bounds[idx_ch + 1]))))
                vals.append(float(np.max(orig_coop_np[sub_s:sub_e, cid])))
    return float(np.mean(vals)) if vals else -10.0


class _AudioSoloChecker:
    """基于 16kHz 立体声按需局部 STFT (180..1800Hz) 相干性、消去残差与单耳声级差(ILD)的纯音频单声源判定器。"""

    def __init__(
        self,
        wav_stereo: Optional[np.ndarray],
        ref_spans: Sequence[Tuple[float, float]],
        sr: int = 16000,
    ) -> None:
        self.enabled = False
        if wav_stereo is None or wav_stereo.ndim != 2 or wav_stereo.shape[1] < sr:
            return
        from scipy import signal as _sig

        self._sig = _sig
        self.wav = wav_stereo
        self.sr = int(sr)
        self.n_samples = int(wav_stereo.shape[1])
        self.dur_sec = float(self.n_samples) / float(self.sr)
        self.n_fft = 1024
        self.hop = 256
        self.hop_sec = float(self.hop) / float(self.sr)
        self.t_frames = (self.n_samples // self.hop) + 1
        self._ref_spans = list(ref_spans)
        self._base_ready = False
        self.base_cres = 0.10
        self.base_msc = 0.75
        self._sb_mask: Optional[np.ndarray] = None
        self.enabled = True

    def _ensure_baseline(self) -> None:
        if self._base_ready or not self.enabled:
            return
        self._base_ready = True
        cres_vals: List[float] = []
        msc_vals: List[float] = []
        step_ref = max(1, len(self._ref_spans) // 40 or 1)
        for s0, s1 in self._ref_spans[::step_ref]:
            if s1 - s0 >= 0.40:
                _, c_p85, m_p15, ild_p25 = self._window_stats(s0, s1)
                # 排除单侧极强贴麦段（头影效应下 MSC 天然偏低）后再标定双耳相干基线
                if m_p15 is not None and ild_p25 < 12.0:
                    cres_vals.append(c_p85)
                    msc_vals.append(m_p15)
        # 取高纯净端分位（MSC P70 / cres P30），避免全程高并发轨污染基线
        self.base_cres = min(0.16, float(np.percentile(cres_vals, 30))) if cres_vals else 0.10
        self.base_msc = max(0.70, float(np.percentile(msc_vals, 70))) if msc_vals else 0.75

    def _slice_stft(self, t0: int, t1: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        half_fft = self.n_fft // 2
        s_lo = t0 * self.hop - half_fft
        s_hi = (t1 - 1) * self.hop + half_fft
        pad_l = max(0, -s_lo)
        pad_r = max(0, s_hi - self.n_samples)
        c0 = max(0, s_lo)
        c1 = min(self.n_samples, s_hi)
        seg = self.wav[:, c0:c1]
        if pad_l > 0 or pad_r > 0:
            seg = np.pad(seg, ((0, 0), (pad_l, pad_r)), mode="constant")
        freqs, _, zl = self._sig.stft(
            seg[0], fs=self.sr, nperseg=self.n_fft, noverlap=self.n_fft - self.hop,
            window="hann", boundary=None, padded=False,
        )
        _, _, zr = self._sig.stft(
            seg[1], fs=self.sr, nperseg=self.n_fft, noverlap=self.n_fft - self.hop,
            window="hann", boundary=None, padded=False,
        )
        if self._sb_mask is None:
            # 聚焦语音基频与第一/二共振峰带 (180..1800Hz)，避开高频水声/气音与头影衰减
            self._sb_mask = (freqs >= 180.0) & (freqs <= 1800.0)
        fl_w = zl[self._sb_mask]
        fr_w = zr[self._sb_mask]
        fe_w = np.sum((np.abs(fl_w) ** 2) + (np.abs(fr_w) ** 2) + 1e-24, axis=0)
        return fl_w, fr_w, fe_w

    def _subwindow_metrics(
        self, st_s: float, en_s: float
    ) -> List[Tuple[float, float, float, float, float]]:
        """返回句内活跃短子窗（~256ms，步进 64ms）列表: [(sub_st, sub_en, cres, msc, abs_ild_db), ...]"""
        # 紧贴句子真实边界（最多外扩 ±0.04s，极短句仅补足到 0.24s），避免蹭进相邻声优台词
        dur = max(0.0, en_s - st_s)
        if dur < 0.24:
            mid = 0.5 * (st_s + en_s)
            w_st = max(0.0, mid - 0.12)
            w_en = min(self.dur_sec, mid + 0.12)
        else:
            w_st = max(0.0, st_s - 0.04)
            w_en = min(self.dur_sec, en_s + 0.04)
        t0 = max(0, int(round(w_st / self.hop_sec)))
        t1 = min(self.t_frames, int(round(w_en / self.hop_sec)))
        if t1 - t0 < 8:
            return []
        fl_w, fr_w, fe_w = self._slice_stft(t0, t1)
        e_floor = max(1e-24, 0.08 * float(np.percentile(fe_w, 90)))
        T_w = fl_w.shape[1]
        win_f = 16   # ~256ms，抗转头/快节奏换人
        step_f = 4   # ~64ms 细粒度步进
        metrics: List[Tuple[float, float, float, float, float]] = []
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
            sub_st = max(st_s, (t0 + k0) * self.hop_sec)
            sub_en = min(en_s, (t0 + k1) * self.hop_sec)
            metrics.append((sub_st, sub_en, cres, msc, abs_ild_db))
        return metrics

    def _window_stats(self, st_s: float, en_s: float) -> Tuple[float, float, Optional[float], float]:
        metrics = self._subwindow_metrics(st_s, en_s)
        if not metrics:
            return 1.0, 1.0, None, 0.0
        cres_list = [m[2] for m in metrics]
        msc_list = [m[3] for m in metrics]
        ild_list = [m[4] for m in metrics]
        return (
            float(np.max(cres_list)),
            float(np.percentile(cres_list, 85)),
            float(np.percentile(msc_list, 15)),
            float(np.percentile(ild_list, 25)),
        )

    def _is_subwindow_solo(self, cres: float, msc: float, abs_ild_db: float) -> bool:
        # 单耳极强贴麦（头影效应：安静耳低 >= 12dB），直接判为单侧主导单声源
        if abs_ild_db >= 12.0:
            return True
        return (
            msc >= max(0.66, 0.85 * self.base_msc)
            and cres <= max(0.22, 1.50 * self.base_cres)
        )

    def is_clearly_solo(self, st_s: float, en_s: float) -> bool:
        if not self.enabled:
            return True
        self._ensure_baseline()
        c_max, c_p85, m_p15, ild_p25 = self._window_stats(st_s, en_s)
        if m_p15 is None:
            return False
        # 单耳强偏置快速通道（避免贴耳低语因头影耳底噪拉低 MSC 而误判为并发）
        if ild_p25 >= 12.0:
            return True
        return (
            m_p15 >= max(0.66, 0.85 * self.base_msc)
            and c_p85 <= max(0.18, 1.35 * self.base_cres)
            and c_max <= max(0.28, 1.75 * self.base_cres)
        )

    def split_solo_subspans(self, st_s: float, en_s: float) -> List[Tuple[float, float, bool]]:
        """对长句做帧级单声源/并发分段，避免句尾短暂插话让整句独白失去保护。"""
        if not self.enabled or (en_s - st_s) < 1.50:
            return [(st_s, en_s, self.is_clearly_solo(st_s, en_s))]
        self._ensure_baseline()
        metrics = self._subwindow_metrics(st_s, en_s)
        if len(metrics) < 3:
            return [(st_s, en_s, self.is_clearly_solo(st_s, en_s))]
        flags = [self._is_subwindow_solo(m[2], m[3], m[4]) for m in metrics]
        if all(flags):
            return [(st_s, en_s, True)]
        if not any(flags):
            return [(st_s, en_s, False)]
        # 3 点中值平滑消除单帧毛刺
        smooth = list(flags)
        for i in range(1, len(flags) - 1):
            smooth[i] = (int(flags[i - 1]) + int(flags[i]) + int(flags[i + 1])) >= 2
        segs: List[Tuple[float, float, bool]] = []
        cur_st, cur_en, cur_flag = metrics[0][0], metrics[0][1], smooth[0]
        for idx in range(1, len(metrics)):
            s0, s1 = metrics[idx][0], metrics[idx][1]
            fl = smooth[idx]
            if fl == cur_flag:
                cur_en = max(cur_en, s1)
            else:
                if cur_en - cur_st >= 0.12:
                    segs.append((cur_st, cur_en, cur_flag))
                cur_st, cur_en, cur_flag = s0, s1, fl
        if cur_en - cur_st >= 0.12:
            segs.append((cur_st, cur_en, cur_flag))
        return segs if segs else [(st_s, en_s, False)]


def extract_acoustic_context(
    audio_file_path: str | Path,
    lora_path: Optional[str | Path] = None,
    device: str = "auto",
    chunk_frames: int = 3000,
) -> Dict[str, Any]:
    """提取一次双声道发射矩阵、逐帧 RMS、静音掩码与句首静音保护数组，供多流复用。"""
    bundle = torchaudio.pipelines.MMS_FA
    tokenizer = bundle.get_tokenizer()
    blank = 0
    wildcard = tokenizer.dictionary.get("*")
    ignored_classes = (() if wildcard is None else (int(wildcard),))

    if isinstance(audio_file_path, (str, Path)):
        waveform, sample_rate = torchaudio.load(str(audio_file_path))
    else:
        waveform = torch.as_tensor(audio_file_path).float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        sample_rate = bundle.sample_rate

    if lora_path is None or str(lora_path).strip() == "":
        resolved_lora = str(align.DEFAULT_LORA_PATH) if align.DEFAULT_LORA_PATH.is_file() else None
    elif str(lora_path).upper() in ("NO_LORA", "NONE", "DISABLE"):
        resolved_lora = None
    else:
        resolved_lora = lora_path

    raw_L, raw_R = compute_emission(
        waveform, sample_rate, device=device, lora_path=resolved_lora, chunk_frames=chunk_frames
    )
    emission_L = torch.log_softmax(raw_L.float(), dim=-1)
    emission_R = torch.log_softmax(raw_R.float(), dim=-1)

    audio_np = (
        waveform.numpy().T
        if waveform.ndim == 2
        else np.column_stack([waveform.numpy(), waveform.numpy()])
    )
    if audio_np.shape[1] < 2:
        audio_np = np.column_stack([audio_np[:, 0], audio_np[:, 0]])

    total_frames = int(emission_L.shape[1])
    rms_l, rms_r, mono_rms = compute_frame_rms_triple(audio_np, sample_rate, total_frames)
    floor_l = compute_track_noise_floor(rms_l)
    floor_r = compute_track_noise_floor(rms_r)
    mask_l = make_silence_mask(rms_l, floor_l)
    mask_r = make_silence_mask(rms_r, floor_r)
    silence_active = mask_l & mask_r
    speech_frame_mask = (~silence_active).astype(np.uint8)

    stereo_emission = stereo_ctc_better_ear_log_mixture(
        emission_L,
        emission_R,
        blank=blank,
        ignored_classes=ignored_classes,
        smoothing_frames=75,
        temperature=1.5,
    )

    stereo_rms = np.maximum(rms_l, rms_r)
    stereo_floor = compute_track_noise_floor(stereo_rms, percentile=10.0)
    head_silence_end = build_next_stable_silence_end(
        stereo_rms,
        stereo_floor,
        max_spike_db=None,
        require_pre_silence_island=True,
        island_active_margin_db=4.0,
        island_smooth_frames=5,
        island_bridge_frames=5,
        island_lookback_frames=50,
        island_prepad_frames=4,
    )
    mono_floor = compute_track_noise_floor(mono_rms)

    wav_16k = waveform[:2] if waveform.shape[0] >= 2 else waveform.repeat(2, 1)
    if int(sample_rate) != int(bundle.sample_rate):
        wav_16k = torchaudio.functional.resample(wav_16k, int(sample_rate), int(bundle.sample_rate))
    stereo_wav_16k = wav_16k.detach().cpu().numpy().astype(np.float32)

    return {
        "emission_L": emission_L,
        "emission_R": emission_R,
        "rms_l": rms_l,
        "rms_r": rms_r,
        "mono_rms": mono_rms,
        "mono_floor": float(mono_floor),
        "speech_frame_mask": speech_frame_mask,
        "stereo_emission": stereo_emission,
        "head_silence_end": head_silence_end,
        "ignored_classes": ignored_classes,
        "blank": blank,
        "tokenizer": tokenizer,
        "_stereo_wav": stereo_wav_16k,
        "_device": str(device),
        "_lora_path": resolved_lora,
    }


class _SpeakerVoiceprintJudge:
    """零新依赖说话人声纹原型判别器：复用已缓存的 MMS-FA (wav2vec2) 第 2~4 浅层隐状态。

    设计原则：
      1. 严格按需提取（只跑各角色至多 20 条纯净独白句 + 少数争议候选句），不常驻整轨隐状态，显存/内存开销极低；
      2. 轨内去均值中心化（消除 Transformer 共享偏置）+ 留一法自检门控（LOO < 0.72 自动关闭），
         确保声线极近或全程强畸变轨零负优化，自动退回纯声学基线。
    """

    def __init__(
        self,
        acoustic_ctx: Dict[str, Any],
        spk_chains: Dict[str, List[int]],
        per_spk_results: Dict[str, Dict[int, List[Dict[str, Any]]]],
        ono_clause_flags: Sequence[int],
        is_solo_cached: Callable[[float, float], bool],
        log: Callable[[str], None],
        clause_speakers: Optional[Sequence[Any]] = None,
    ) -> None:
        self.ready = False
        self.loo_acc = 0.0
        self.protos: Dict[str, np.ndarray] = {}
        self._mean_vec: Optional[np.ndarray] = None
        self._span_cache: Dict[Tuple[float, float], Optional[Dict[str, float]]] = {}
        self._wav = acoustic_ctx.get("_stereo_wav")
        self._sr = 16000
        self._w2v2 = None
        self._device = None

        if self._wav is None or self._wav.ndim != 2 or self._wav.shape[1] < self._sr:
            return
        active_spks = [s for s, ch in spk_chains.items() if ch]
        if len(active_spks) < 2:
            return

        # 复用已缓存在 align_model 中的 MMS_FA 模型（0ms 重载）
        from fa_asmr.aligners.align_model import get_mms_model, _resolve_device
        dev_str = acoustic_ctx.get("_device", "auto")
        lora_p = acoustic_ctx.get("_lora_path", None)
        self._device = _resolve_device(dev_str)
        bundle = torchaudio.pipelines.MMS_FA
        mms_model = get_mms_model(bundle, self._device, lora_path=lora_p)
        self._w2v2 = mms_model.model

        # 收集每个角色干净、无他方重叠的单声源独白句用于在线标定原型
        all_spans_by_spk: Dict[str, List[Tuple[float, float, float, int]]] = {}
        for sp in active_spks:
            lst: List[Tuple[float, float, float, int]] = []
            for ci in spk_chains[sp]:
                if ci < len(ono_clause_flags) and ono_clause_flags[ci]:
                    continue
                if clause_speakers and ci < len(clause_speakers) and len(clause_speakers[ci]) >= 2:
                    continue  # 排除齐声句
                inf = _clause_summary(per_spk_results[sp].get(ci, []))
                if inf["skipped"] or inf["dur"] < 0.70 or inf["score"] < -2.35:
                    continue
                lst.append((float(inf["st"]), float(inf["en"]), float(inf["score"]), ci))
            all_spans_by_spk[sp] = lst

        calib_items: List[Tuple[str, float, float]] = []
        for sp in active_spks:
            cands = []
            for st_s, en_s, sc_s, _ in all_spans_by_spk[sp]:
                # 排除与他方重叠的句子
                ov_other = any(
                    min(en_s, e2) - max(st_s, s2) > 0.15
                    for osp, olst in all_spans_by_spk.items() if osp != sp
                    for s2, e2, _, _ in olst
                )
                if ov_other:
                    continue
                solo_bonus = 1.0 if is_solo_cached(st_s, en_s) else 0.0
                cands.append((solo_bonus, sc_s, min(4.0, en_s - st_s), st_s, en_s))
            cands.sort(reverse=True)
            for _, _, _, st_s, en_s in cands[:20]:
                calib_items.append((sp, st_s, en_s))

        per_spk_cnt = {sp: sum(1 for s, _, _ in calib_items if s == sp) for sp in active_spks}
        if any(v < 2 for v in per_spk_cnt.values()):
            return

        raw_embs: List[np.ndarray] = []
        raw_labels: List[str] = []
        for sp, st_s, en_s in calib_items:
            v = self._extract_raw_embedding(st_s, en_s)
            if v is not None:
                raw_embs.append(v)
                raw_labels.append(sp)

        final_cnt = {sp: sum(1 for s in raw_labels if s == sp) for sp in active_spks}
        if any(v < 2 for v in final_cnt.values()):
            return

        mat = np.stack(raw_embs, axis=0)
        mean_v = mat.mean(axis=0, keepdims=True)
        mat_c = mat - mean_v
        mat_c = mat_c / (np.linalg.norm(mat_c, axis=1, keepdims=True) + 1e-8)

        # 留一法自检门控：仅当本轨浅层声纹真实可分（LOO >= 72%）时启用
        n_cal = len(raw_labels)
        ok_cnt = 0
        for i in range(n_cal):
            p_loo: Dict[str, np.ndarray] = {}
            for sp in active_spks:
                idxs = [k for k in range(n_cal) if k != i and raw_labels[k] == sp]
                if not idxs:
                    continue
                pv = mat_c[idxs].mean(axis=0)
                p_loo[sp] = pv / (np.linalg.norm(pv) + 1e-8)
            if len(p_loo) < 2:
                continue
            pred = max(p_loo.keys(), key=lambda s: float(np.dot(mat_c[i], p_loo[s])))
            if pred == raw_labels[i]:
                ok_cnt += 1
        self.loo_acc = ok_cnt / float(max(1, n_cal))
        if self.loo_acc < 0.72:
            log(f"  ℹ [浅层声纹门控] L2-L4 留一法区分率 {self.loo_acc*100:.1f}% < 72%，本轨自动关闭声纹裁决")
            return

        self._mean_vec = mean_v[0]
        for sp in active_spks:
            idxs = [k for k in range(n_cal) if raw_labels[k] == sp]
            pv = mat_c[idxs].mean(axis=0)
            self.protos[sp] = pv / (np.linalg.norm(pv) + 1e-8)
        self.ready = True
        log(f"  🧬 [浅层声纹原型] 已复用 MMS-FA L2-L4 特征标定 {len(active_spks)} 位说话人 (LOO={self.loo_acc*100:.1f}%, 样本={final_cnt})")

    def _extract_raw_embedding(self, st_s: float, en_s: float) -> Optional[np.ndarray]:
        if self._w2v2 is None or self._wav is None:
            return None
        sr = self._sr
        n_tot = self._wav.shape[1]
        # 限制单句最长截取 8.0s（取中心窗），防止超长独白句在小显存卡上引发自注意力显存尖峰
        if en_s - st_s > 8.0:
            mid_cap = 0.5 * (st_s + en_s)
            st_s, en_s = mid_cap - 4.0, mid_cap + 4.0
        pad = 0.12
        a0 = max(0, int((st_s - pad) * sr))
        a1 = min(n_tot, int((en_s + pad) * sr))
        if a1 - a0 < int(0.30 * sr):
            # 极短句对称扩展至 0.32s 确保 CNN 感受野完整
            mid = 0.5 * (st_s + en_s)
            a0 = max(0, int((mid - 0.16) * sr))
            a1 = min(n_tot, int((mid + 0.16) * sr))
            if a1 - a0 < int(0.24 * sr):
                return None
        seg = self._wav[:, a0:a1]
        el = float(np.mean(seg[0] ** 2))
        er = float(np.mean(seg[1] ** 2))
        mono_np = seg[0] if el >= er else seg[1]
        mx = float(np.max(np.abs(mono_np)))
        if mx < 1e-5:
            return None
        mono_t = torch.from_numpy((mono_np / mx).astype(np.float32)).unsqueeze(0).to(self._device)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=(self._device.type == "cuda")
        ):
            feats, _ = self._w2v2.extract_features(mono_t, num_layers=4)
        left_f = int(round((st_s - a0 / float(sr)) / 0.02))
        right_f = int(round((en_s - a0 / float(sr)) / 0.02))
        parts: List[np.ndarray] = []
        for ly in (2, 3, 4):
            h = feats[ly - 1][0].float()
            T = h.shape[0]
            f0 = max(0, min(T - 1, left_f))
            f1 = max(f0 + 1, min(T, right_f))
            sub = h[f0:f1]
            mu = sub.mean(dim=0)
            sd = sub.std(dim=0) if sub.shape[0] > 1 else torch.zeros_like(mu)
            v = torch.cat([mu, sd], dim=0)
            v = v / (v.norm() + 1e-8)
            parts.append(v.cpu().numpy())
        return np.concatenate(parts, axis=0).astype(np.float32)

    def score_span(self, st_s: float, en_s: float) -> Optional[Dict[str, float]]:
        if not self.ready or self._mean_vec is None:
            return None
        key = (round(st_s, 2), round(en_s, 2))
        if key in self._span_cache:
            return self._span_cache[key]
        raw = self._extract_raw_embedding(st_s, en_s)
        if raw is None:
            self._span_cache[key] = None
            return None
        vc = raw - self._mean_vec
        vc = vc / (np.linalg.norm(vc) + 1e-8)
        scores = {sp: float(np.dot(vc, pv)) for sp, pv in self.protos.items()}
        self._span_cache[key] = scores
        return scores

    def margin_for(self, st_s: float, en_s: float, spk: str) -> Optional[float]:
        scs = self.score_span(st_s, en_s)
        if not scs or spk not in scs or len(scs) < 2:
            return None
        own = scs[spk]
        best_oth = max(v for k, v in scs.items() if k != spk)
        return own - best_oth


def _resolve_same_text_swaps(
    spk_chains: Dict[str, List[int]],
    per_spk_results: Dict[str, Dict[int, List[Dict[str, Any]]]],
    alignment_tokens: List[List[str]],
    ono_clause_flags: Sequence[int],
    vp_judge: _SpeakerVoiceprintJudge,
    log: Callable[[str], None],
    clause_speakers: Optional[Sequence[Any]] = None,
) -> int:
    """修正台本紧邻、发音完全相同、时间紧挨却发生角色对调的回声/跟读句。"""
    active_spks = [s for s, ch in spk_chains.items() if ch]
    if len(active_spks) < 2:
        return 0
    # 构造 ci -> spk 映射（仅单人非拟声非跳过句）
    ci_owner: Dict[int, str] = {}
    for sp in active_spks:
        for ci in spk_chains[sp]:
            if ci < len(ono_clause_flags) and ono_clause_flags[ci]:
                continue
            if clause_speakers and ci < len(clause_speakers) and len(clause_speakers[ci]) >= 2:
                continue
            inf = _clause_summary(per_spk_results[sp].get(ci, []))
            if not inf["skipped"] and inf["n"] > 0:
                ci_owner[ci] = sp

    swapped = 0
    cis_sorted = sorted(ci_owner.keys())
    used = set()
    for idx_i, ci_a in enumerate(cis_sorted):
        if ci_a in used:
            continue
        sp_a = ci_owner[ci_a]
        ph_a = "".join(alignment_tokens[ci_a])
        if len(ph_a) < 4:
            continue
        for ci_b in cis_sorted[idx_i + 1:]:
            if ci_b - ci_a > 3:
                break
            if ci_b in used:
                continue
            sp_b = ci_owner[ci_b]
            if sp_b == sp_a:
                continue
            if "".join(alignment_tokens[ci_b]) != ph_a:
                continue
            if len(alignment_tokens[ci_a]) != len(alignment_tokens[ci_b]):
                continue
            inf_a = _clause_summary(per_spk_results[sp_a][ci_a])
            inf_b = _clause_summary(per_spk_results[sp_b][ci_b])
            # 必须时间邻近 (<= 4.0s) 且与台本顺序恰好相反 (ci_a < ci_b 但 st_a > st_b + 0.15)
            gap_ab = max(0.0, max(inf_a["st"], inf_b["st"]) - min(inf_a["en"], inf_b["en"]))
            if gap_ab > 4.0 or inf_a["st"] <= inf_b["st"] + 0.15:
                continue
            # 检查声纹证据（若声纹可用，要求交换后总声纹匹配度不降反升或至少不强烈反驳）
            if vp_judge.ready:
                sc_a_pos = vp_judge.score_span(inf_a["st"], inf_a["en"])
                sc_b_pos = vp_judge.score_span(inf_b["st"], inf_b["en"])
                if sc_a_pos and sc_b_pos:
                    vp_cur = sc_a_pos.get(sp_a, 0.0) + sc_b_pos.get(sp_b, 0.0)
                    vp_swp = sc_b_pos.get(sp_a, 0.0) + sc_a_pos.get(sp_b, 0.0)
                    if vp_swp < vp_cur - 0.05:
                        continue
            # 检查交换后不破坏各自单链单调顺序
            def _fits_chain(sp_t: str, ci_t: int, new_st: float, new_en: float) -> bool:
                ch = spk_chains[sp_t]
                pos = ch.index(ci_t)
                for p in range(pos - 1, -1, -1):
                    pi = _clause_summary(per_spk_results[sp_t].get(ch[p], []))
                    if not pi["skipped"] and pi["n"] > 0:
                        if new_st < pi["en"] - 0.15:
                            return False
                        break
                for p in range(pos + 1, len(ch)):
                    ni = _clause_summary(per_spk_results[sp_t].get(ch[p], []))
                    if not ni["skipped"] and ni["n"] > 0:
                        if new_en > ni["st"] + 0.15:
                            return False
                        break
                return True

            if not (_fits_chain(sp_a, ci_a, inf_b["st"], inf_b["en"]) and _fits_chain(sp_b, ci_b, inf_a["st"], inf_a["en"])):
                continue

            res_a = per_spk_results[sp_a][ci_a]
            res_b = per_spk_results[sp_b][ci_b]
            if len(res_a) != len(res_b):
                continue
            for ra, rb in zip(res_a, res_b):
                for key in ("original_start", "original_end", "start", "end", "score"):
                    has_a, has_b = key in ra, key in rb
                    if has_a and has_b:
                        ra[key], rb[key] = rb[key], ra[key]
                    elif has_a:
                        rb[key] = ra.pop(key)
                    elif has_b:
                        ra[key] = rb.pop(key)
            used.add(ci_a)
            used.add(ci_b)
            swapped += 1
            break
    if swapped:
        log(f"  🔀 [同词角色裁决] 修正 {swapped} 对紧邻同词反序对调")
    return swapped


def _rescue_out_of_order_short_clauses(
    spk_chains: Dict[str, List[int]],
    per_spk_results: Dict[str, Dict[int, List[Dict[str, Any]]]],
    alignment_tokens: List[List[str]],
    line_group: Sequence[int],
    ono_clause_flags: Sequence[int],
    acoustic_ctx: Dict[str, Any],
    base_emission_2d: torch.Tensor,
    vp_judge: _SpeakerVoiceprintJudge,
    log: Callable[[str], None],
    clause_surfaces: Optional[Sequence[str]] = None,
    clause_speakers: Optional[Sequence[Any]] = None,
) -> int:
    """对违背台本前后高置信他方锚点顺序的低分短句（如短回声/短接话），在未遮罩原始图上做有界窗口重对齐与严格验收。"""
    active_spks = [s for s, ch in spk_chains.items() if ch]
    if len(active_spks) < 2:
        return 0
    tokenizer = acoustic_ctx["tokenizer"]
    tok_dict = tokenizer.dictionary
    blank_id = int(acoustic_ctx["blank"])
    raw_np = base_emission_2d.detach().cpu().numpy()
    T_all = int(raw_np.shape[0])
    n_cls = int(raw_np.shape[1])
    valid_cls = [c for c in range(n_cls) if c != blank_id]

    # 全局各 ci 的归属与摘要
    ci_info: Dict[int, Tuple[str, Dict[str, Any]]] = {}
    for sp in active_spks:
        for ci in spk_chains[sp]:
            inf = _clause_summary(per_spk_results[sp].get(ci, []))
            if not inf["skipped"] and inf["n"] > 0:
                ci_info[ci] = (sp, inf)

    repaired = 0
    for sp in active_spks:
        ch = spk_chains[sp]
        for pos, ci in enumerate(ch):
            if ci < len(ono_clause_flags) and ono_clause_flags[ci]:
                continue
            if clause_speakers and ci < len(clause_speakers) and len(clause_speakers[ci]) >= 2:
                continue
            inf = _clause_summary(per_spk_results[sp].get(ci, []))
            if inf["skipped"] or inf["n"] <= 0:
                continue
            n_phones = sum(len(t) for t in alignment_tokens[ci])
            if n_phones > 14:
                continue

            # 找台本中紧邻前一句与后一句（属于他方角色、高置信、顺序正常且间隔 <= 12s）
            prev_anc = ci_info.get(ci - 1)
            next_anc = ci_info.get(ci + 1)
            if prev_anc is None or next_anc is None:
                continue
            sp_prev, inf_prev = prev_anc
            sp_next, inf_next = next_anc
            if sp_prev == sp or sp_next == sp:
                continue
            if inf_prev["score"] < -2.10 or inf_next["score"] < -2.10:
                continue
            if inf_prev["st"] >= inf_next["en"] or (inf_next["en"] - inf_prev["st"]) > 12.0:
                continue

            # 仅当本句明显越出了台本紧邻前后锚点区间才视为候选
            out_left = inf["en"] < inf_prev["st"] + 0.15
            out_right = inf["st"] > inf_next["en"] - 0.15
            if not (out_left or out_right):
                continue

            old_raw = _unmasked_clause_mean_score(per_spk_results[sp][ci], raw_np, tok_dict)
            old_vp = vp_judge.margin_for(inf["st"], inf["en"], sp) if vp_judge.ready else None
            # 门禁：若原落点原始 CTC 极高且声纹不反对（可能是真乱序），绝不妄动
            if old_raw >= -1.25 and (old_vp is None or old_vp >= -0.05):
                continue

            # 结合本角色单链前后邻句，构造安全搜索窗
            own_lo = 0.0
            for p in range(pos - 1, -1, -1):
                pi = _clause_summary(per_spk_results[sp].get(ch[p], []))
                if not pi["skipped"] and pi["n"] > 0:
                    own_lo = float(pi["en"])
                    break
            own_hi = float(T_all * FRAME_SEC)
            for p in range(pos + 1, len(ch)):
                ni = _clause_summary(per_spk_results[sp].get(ch[p], []))
                if not ni["skipped"] and ni["n"] > 0:
                    own_hi = float(ni["st"])
                    break

            w_lo = max(own_lo, float(inf_prev["st"]) - 0.20)
            w_hi = min(own_hi, float(inf_next["en"]) + 0.20)
            if w_hi - w_lo < 0.35:
                continue

            f_lo = max(0, int(np.floor(w_lo / FRAME_SEC)))
            f_hi = min(T_all, int(np.ceil(w_hi / FRAME_SEC)))
            if f_hi <= f_lo + 4:
                continue

            em_win = base_emission_2d.clone()
            if f_lo > 0:
                em_win[:f_lo, valid_cls] = -15.0
                em_win[:f_lo, blank_id] = 0.0
            if f_hi < T_all:
                em_win[f_hi:, valid_cls] = -15.0
                em_win[f_hi:, blank_id] = 0.0

            realigned = _align_subset_clauses(
                sub_indices=[ci],
                alignment_tokens=alignment_tokens,
                line_group=line_group,
                ono_clause_flags=ono_clause_flags,
                enable_silence_snap=True,
                acoustic_ctx=acoustic_ctx,
                custom_emission_2d=em_win,
                enable_inter_line_filler=True,
                enable_ono_skip=False,
                enable_word_skip=False,
                clause_surfaces=clause_surfaces,
                clause_speakers=clause_speakers,
                log_callback=None,
            )
            new_cl = realigned.get(ci)
            if not new_cl:
                continue
            new_inf = _clause_summary(new_cl)
            if new_inf["skipped"] or new_inf["n"] <= 0 or new_inf["dur"] > max(3.0, 0.45 * n_phones):
                continue

            new_raw = _unmasked_clause_mean_score(new_cl, raw_np, tok_dict)
            new_vp = vp_judge.margin_for(new_inf["st"], new_inf["en"], sp) if vp_judge.ready else None

            ctc_win = (new_raw >= -2.35 and new_raw >= old_raw + 0.30)
            vp_win = (
                vp_judge.ready
                and new_vp is not None
                and old_vp is not None
                and new_raw >= -2.40
                and new_raw >= old_raw - 0.25
                and new_vp >= 0.08
                and (new_vp - old_vp) >= 0.15
            )
            if ctc_win or vp_win:
                per_spk_results[sp][ci] = new_cl
                ci_info[ci] = (sp, new_inf)
                repaired += 1
    if repaired:
        log(f"  🎯 [越序短句局部修复] 共校正 {repaired} 句错位短接话")
    return repaired


def _align_clauses_with_context(
    text_tokens: List[List[str]],
    line_group: Sequence[int],
    ono_clause_flags: Sequence[int],
    acoustic_ctx: Dict[str, Any],
    enable_silence_snap: bool = True,
    custom_emission_2d: Optional[Any] = None,
    enable_inter_line_filler: bool = True,
    inter_line_filler_margin: float = 1.35,
    enable_ono_skip: bool = True,
    ono_skip_token_penalty: float = 2.6,
    filler_frame_logp: Optional[np.ndarray] = None,
    enable_word_skip: bool = False,
    word_skip_token_penalty: float = 4.8,
    clause_surfaces: Optional[Sequence[str]] = None,
    clause_speakers: Optional[Sequence[Any]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
) -> List[Dict[str, Any]]:
    """使用预提取的 acoustic_ctx 对给定的 text_tokens 执行全局 CTC 对齐。"""
    tokenizer = acoustic_ctx["tokenizer"]
    blank = int(acoustic_ctx["blank"])
    ignored_classes = acoustic_ctx["ignored_classes"]

    aligner = MultiSpeakerCTCAligner(
        blank=blank,
        intra_blank_penalty=0.10,
        n_pure_continuation_penalty=0.20,
        n_pure_continuation_lookahead_frames=4,
        n_pure_continuation_drop_threshold=1.0,
        n_pure_continuation_drop_ramp=2.0,
        frame_sec=FRAME_SEC,
    )

    lines = []
    targets = []
    head_standalone_n_flag = []
    ono_flags = []
    _nasal_token_id = 4
    _vowel_ids = {tokenizer.dictionary[v] for v in "aiueo"}

    for li, line_tokens in enumerate(text_tokens):
        if isinstance(line_tokens, str):
            line_tokens = line_tokens.split()
        ii_folds = find_subclause_ii_folds(line_tokens)
        c_toks, single_line_ids, kept_idx, token_id_ranges = _canonicalize_line_tokens(
            line_tokens, tokenizer, blank, fold_subclause_ii=True
        )
        if not c_toks:
            c_toks = ["a"]
            single_line_ids = [1]
            kept_idx = []
            token_id_ranges = [(0, 1)]

        targets.append(single_line_ids)
        _f = 1 if (ono_clause_flags is not None and li < len(ono_clause_flags) and ono_clause_flags[li]) else 0
        ono_flags.extend([_f] * len(single_line_ids))
        head_standalone_n_flag.append(
            _build_head_standalone_n_flag(single_line_ids, token_id_ranges, _vowel_ids, _nasal_token_id)
        )
        lines.append({
            "line_index": li,
            "tokens": c_toks,
            "n_raw": len(line_tokens),
            "raw_tokens": list(line_tokens),
            "orig_idx": list(kept_idx),
            "token_id_ranges": list(token_id_ranges),
            "ii_folds": ii_folds,
        })

    if custom_emission_2d is not None:
        emission_for_align = (
            custom_emission_2d.cpu()
            if hasattr(custom_emission_2d, "cpu")
            else torch.as_tensor(custom_emission_2d)
        )
    else:
        emission_for_align = acoustic_ctx["stereo_emission"][0].cpu()

    _flat_n = sum(len(line) for line in targets)
    head_silence_end = acoustic_ctx["head_silence_end"]
    head_silence_penalty, _ = build_head_token_silence_penalties(
        targets=targets,
        line_group=line_group,
        discount=4.0,
    )

    filler_blank_mask = None
    if enable_inter_line_filler and _flat_n > 0:
        filler_blank_mask = np.zeros(_flat_n + 1, dtype=np.uint8)
        filler_blank_mask[0] = 1
        filler_blank_mask[_flat_n] = 1
        _cur_b = 0
        for _li, _line in enumerate(targets):
            _cur_b += len(_line)
            filler_blank_mask[_cur_b] = 1
        if filler_frame_logp is None:
            _em_np_tmp = (
                emission_for_align.detach().cpu().numpy()
                if hasattr(emission_for_align, "detach")
                else np.asarray(emission_for_align, dtype=np.float32)
            )
            _valid_cls = [
                c for c in range(_em_np_tmp.shape[1]) if c != blank and c not in ignored_classes
            ]
            _max_nb = np.max(_em_np_tmp[:, _valid_cls], axis=1).astype(np.float32)
            filler_frame_logp = (_max_nb - np.float32(inter_line_filler_margin)).astype(np.float32)

    skip_prev_state = None
    skip_penalty = None
    _clause_tok_spans: List[Tuple[int, int]] = []
    _cur_t = 0
    for _line in targets:
        _clause_tok_spans.append((_cur_t, _cur_t + len(_line)))
        _cur_t += len(_line)

    if (enable_ono_skip or enable_word_skip) and _flat_n > 0:
        _states_n = _flat_n * 2 + 1
        skip_prev_state = np.full(_states_n, -1, dtype=np.int32)
        skip_penalty = np.zeros(_states_n, dtype=np.float32)
        for _li, (_st_t, _en_t) in enumerate(_clause_tok_spans):
            if _en_t <= _st_t:
                continue
            is_ono = bool(
                _li < len(ono_clause_flags) and ono_clause_flags[_li]
            ) if ono_clause_flags is not None else False
            if is_ono and enable_ono_skip:
                skip_prev_state[2 * _en_t] = np.int32(2 * _st_t)
                skip_penalty[2 * _en_t] = np.float32((_en_t - _st_t) * ono_skip_token_penalty)
            elif not is_ono and enable_word_skip:
                skip_prev_state[2 * _en_t] = np.int32(2 * _st_t)
                skip_penalty[2 * _en_t] = np.float32((_en_t - _st_t) * word_skip_token_penalty)

    grouped_spans = aligner(
        emission_for_align,
        targets,
        blank_punct_mask=[False] * (_flat_n + 1),
        speech_frame_mask=acoustic_ctx["speech_frame_mask"],
        line_group=line_group,
        head_token_silence_end=head_silence_end,
        head_token_silence_penalty=head_silence_penalty,
        head_standalone_n_flag=head_standalone_n_flag,
        ono_token_mask=ono_flags,
        ono_score_scale=1.0,
        filler_blank_mask=filler_blank_mask,
        filler_frame_logp=filler_frame_logp,
        skip_prev_state=skip_prev_state,
        skip_penalty=skip_penalty,
    )

    mono_rms = acoustic_ctx["mono_rms"]
    mono_floor = acoustic_ctx["mono_floor"]
    silence_thresh = min(mono_floor + 4.0, -45.0)

    results = _post_process_tokens(
        grouped_spans=grouped_spans,
        lines=lines,
        frame_duration=FRAME_SEC,
        frame_rms_db=mono_rms,
        silence_threshold_db=silence_thresh,
        enable_silence_snap=enable_silence_snap,
    )

    _hit_mask = aligner._last_hit_mask
    if _hit_mask is not None:
        _skipped_lines = set()
        _skipped_word_lines = []
        for _li, (_st_t, _en_t) in enumerate(_clause_tok_spans):
            if _en_t > _st_t and not bool(np.any(_hit_mask[_st_t:_en_t])):
                is_ono = bool(
                    _li < len(ono_clause_flags) and ono_clause_flags[_li]
                ) if ono_clause_flags is not None else False
                if (is_ono and enable_ono_skip) or (not is_ono and enable_word_skip):
                    _skipped_lines.add(_li)
                    if not is_ono:
                        _skipped_word_lines.append(_li)
        if _skipped_lines:
            for r in results:
                if r.get("line_index") in _skipped_lines:
                    r["skipped"] = True
        if _skipped_word_lines and log_callback:
            for _wli in _skipped_word_lines:
                txt = clause_surfaces[_wli] if clause_surfaces and _wli < len(clause_surfaces) else (
                    "".join(text_tokens[_wli]) if _wli < len(text_tokens) else f"clause_{_wli}"
                )
                spk = clause_speakers[_wli] if clause_speakers and _wli < len(clause_speakers) else ""
                spk_str = f"【{spk[0] if isinstance(spk, (tuple, list)) else spk}】" if spk else ""
                log_callback(
                    f"  ⏭ [未发音台词跳过] 检测到音频中未念该台词，已自动跳过避免挤压正常字幕: {spk_str}「{txt}」"
                )

    for r in results:
        if r.get("original_start") is not None:
            r["start"] = _format_time(r["original_start"])
            r["end"] = _format_time(r["original_end"])

    return results




def build_cooperative_stereo_emission(
    acoustic_ctx: Dict[str, Any],
    ear_bias: float = 0.0,
) -> torch.Tensor:
    """构建双耳协同发射矩阵 (T, C)，防止一侧高能量拟声/舔耳通过 Sigmoid 把另一侧轻声台词压低。"""
    em_L = acoustic_ctx["emission_L"][0].cpu()
    em_R = acoustic_ctx["emission_R"][0].cpu()
    stereo = acoustic_ctx["stereo_emission"][0].cpu()
    blank = int(acoustic_ctx.get("blank", 0))

    both_max = torch.maximum(em_L, em_R) - 0.18
    coop = torch.maximum(stereo, both_max)

    if abs(ear_bias) > 1e-3:
        if ear_bias > 0:
            w_pref = min(0.85, 0.50 + 0.25 * float(ear_bias))
            pref, other = em_L, em_R
        else:
            w_pref = min(0.85, 0.50 + 0.25 * float(-ear_bias))
            pref, other = em_R, em_L
        biased = torch.logaddexp(
            pref + float(np.log(w_pref)),
            other + float(np.log(max(1e-4, 1.0 - w_pref))),
        )
        coop = torch.maximum(biased, coop - 0.35)

    # Blank 通道必须保持双耳联合静音定义（两耳均静音才为 Blank），绝不可取 max(em_L[0], em_R[0])，
    # 否则单侧舔耳/单耳说话时另一侧静音耳的 Blank≈0 会让词内/句内 Blank 无代价横跨整段语音
    coop[:, blank] = stereo[:, blank]
    return coop


def _group_results_by_clause(
    alignment_tokens: List[List[str]],
    flat_results: List[Dict[str, Any]],
) -> List[List[Dict[str, Any]]]:
    by_clause: List[List[Dict[str, Any]]] = []
    cursor = 0
    for cl_toks in alignment_tokens:
        k = len(cl_toks)
        by_clause.append(flat_results[cursor : cursor + k])
        cursor += k
    return by_clause


def _clause_summary(clause_res: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not clause_res:
        return {"st": 0.0, "en": 0.0, "dur": 0.0, "score": 0.0, "skipped": True, "n": 0}
    skipped = any(bool(r.get("skipped", False)) for r in clause_res)
    sts = [float(r["original_start"]) for r in clause_res if r.get("original_start") is not None]
    ens = [float(r["original_end"]) for r in clause_res if r.get("original_end") is not None]
    scores = [float(r.get("score", 0.0)) for r in clause_res]
    st = min(sts) if sts else 0.0
    en = max(ens) if ens else st
    return {
        "st": st,
        "en": en,
        "dur": max(0.0, en - st),
        "score": float(np.mean(scores)) if scores else -10.0,
        "min_score": float(np.min(scores)) if scores else -10.0,
        "skipped": skipped,
        "n": len(clause_res),
    }


def _estimate_speaker_ear_bias(
    clause_indices: List[int],
    by_clause: List[List[Dict[str, Any]]],
    ono_clause_flags: Sequence[int],
    acoustic_ctx: Dict[str, Any],
) -> float:
    rms_l = acoustic_ctx["rms_l"]
    rms_r = acoustic_ctx["rms_r"]
    total_frames = len(rms_l)

    diffs = []
    weights = []
    for ci in clause_indices:
        if ci < len(ono_clause_flags) and ono_clause_flags[ci]:
            continue
        info = _clause_summary(by_clause[ci])
        if info["skipped"] or info["score"] < -1.6 or info["dur"] < 0.15:
            continue
        st_f = max(0, min(total_frames - 1, int(round(info["st"] / FRAME_SEC))))
        en_f = max(st_f + 1, min(total_frames, int(round(info["en"] / FRAME_SEC))))
        db_diff = float(np.mean(rms_l[st_f:en_f] - rms_r[st_f:en_f]))
        diffs.append(db_diff)
        weights.append(float(info["n"]))

    if not diffs or sum(weights) < 6:
        return 0.0
    avg_db_diff = float(np.average(diffs, weights=weights))
    if abs(avg_db_diff) < 3.5:
        return 0.0
    return float(np.clip(avg_db_diff / 8.0, -1.2, 1.2))


def _align_subset_clauses(
    sub_indices: List[int],
    alignment_tokens: List[List[str]],
    line_group: Sequence[int],
    ono_clause_flags: Sequence[int],
    enable_silence_snap: bool,
    acoustic_ctx: Dict[str, Any],
    custom_emission_2d: Optional[Any] = None,
    enable_inter_line_filler: bool = True,
    inter_line_filler_margin: float = 1.35,
    enable_ono_skip: bool = True,
    ono_skip_token_penalty: float = 2.6,
    filler_frame_logp: Optional[np.ndarray] = None,
    enable_word_skip: bool = False,
    word_skip_token_penalty: float = 4.8,
    clause_surfaces: Optional[Sequence[str]] = None,
    clause_speakers: Optional[Sequence[Any]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
) -> Dict[int, List[Dict[str, Any]]]:
    if not sub_indices:
        return {}
    sub_tokens = [alignment_tokens[i] for i in sub_indices]
    sub_groups = [line_group[i] for i in sub_indices]
    sub_ono = [ono_clause_flags[i] if i < len(ono_clause_flags) else 0 for i in sub_indices]
    sub_surfaces = [clause_surfaces[i] if i < len(clause_surfaces) else "" for i in sub_indices] if clause_surfaces else None
    sub_spk = [clause_speakers[i] if i < len(clause_speakers) else () for i in sub_indices] if clause_speakers else None

    sub_res = _align_clauses_with_context(
        text_tokens=sub_tokens,
        line_group=sub_groups,
        ono_clause_flags=sub_ono,
        acoustic_ctx=acoustic_ctx,
        enable_silence_snap=enable_silence_snap,
        custom_emission_2d=custom_emission_2d,
        enable_inter_line_filler=enable_inter_line_filler,
        inter_line_filler_margin=inter_line_filler_margin,
        enable_ono_skip=enable_ono_skip,
        ono_skip_token_penalty=ono_skip_token_penalty,
        filler_frame_logp=filler_frame_logp,
        enable_word_skip=enable_word_skip,
        word_skip_token_penalty=word_skip_token_penalty,
        clause_surfaces=sub_surfaces,
        clause_speakers=sub_spk,
        log_callback=log_callback,
    )
    grouped = _group_results_by_clause(sub_tokens, sub_res)
    out: Dict[int, List[Dict[str, Any]]] = {}
    for sub_i, glob_ci in enumerate(sub_indices):
        cl_list = grouped[sub_i]
        for r in cl_list:
            r["line_index"] = glob_ci
        out[glob_ci] = cl_list
    return out


