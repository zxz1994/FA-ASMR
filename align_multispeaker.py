# -*- coding: utf-8 -*-
"""FA-ASMR V2 — 独立的多人分轨混音 / 台本乱序 / 拟声与台词重叠对齐引擎

设计原则（方案 B 完全隔离）：
- 不修改原版 align.py 与 align_trellis.py 的任何一行代码，确保常规单向对齐 100% 零影响。
- 本模块自带独立的 Viterbi 解码核 (_viterbi_multispeaker_duration) 与声学上下文缓存，
  仅在用户在 GUI 高级设置中显式勾选「多人/乱序重排对齐」时启用。

解决痛点：
1. 多人音声分轨录制、后期混音导致的「台本跨角色顺序不可靠」（单人内部顺序可靠，但不同声优交错/颠倒）。
2. 一侧声优持续发出呻吟/舔耳等易错拟声词，另一侧声优同时说语义台词，且两边内容都写在台本里。
3. 即使台本未标注 〖角色〗（或已在清洗阶段剥去角色前缀），遇到局部台本块与音频顺序颠倒时仍可自动拆流重排。
"""
from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch
import torchaudio

try:
    from numba import njit, prange
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

import align
from align_acoustic import (
    FRAME_SEC,
    build_head_token_silence_penalties,
    build_next_stable_silence_end,
    compute_frame_rms_triple,
    compute_track_noise_floor,
    make_silence_mask,
    stereo_ctc_better_ear_log_mixture,
)
from align_model import compute_emission
from align_post import _post_process_tokens
from align_trellis import (
    TokenSpan,
    _n_pure_continuation_penalty,
    _run_decoder,
)
from align_utils import (
    _build_head_standalone_n_flag,
    _canonicalize_line_tokens,
    _format_time,
    find_subclause_ii_folds,
)
import clause as _clause_mod
from clause import F


# ══════════════════════════════════════════════════════════════════════════
#  1. 独立的多人/乱序专用 Viterbi 解码器（支持行间 Filler Blank 与易错拟声跳过弧）
# ══════════════════════════════════════════════════════════════════════════

if _HAS_NUMBA:
    @njit(cache=True, fastmath=True, nogil=True)
    def _viterbi_multispeaker_duration(
        emission: np.ndarray,
        tokens: np.ndarray,
        n_pure_token_mask: np.ndarray = np.empty(0, dtype=np.uint8),
        n_pure_entry_penalty: np.ndarray = np.empty(0, dtype=np.float32),
        blank_cost: np.ndarray = np.empty(0, dtype=np.float32),
        speech_frame_mask: np.ndarray = np.empty(0, dtype=np.uint8),
        subclause_blank_cost: np.ndarray = np.empty(0, dtype=np.float32),
        subclause_blank_free_frames: int = 50,
        head_token_silence_end: np.ndarray = np.empty(0, dtype=np.int32),
        head_token_silence_penalty: np.ndarray = np.empty(0, dtype=np.float32),
        blank: int = 0,
        ono_token_mask: np.ndarray = np.empty(0, dtype=np.uint8),
        ono_score_scale: float = 1.0,
        filler_blank_mask: np.ndarray = np.empty(0, dtype=np.uint8),
        filler_frame_logp: np.ndarray = np.empty(0, dtype=np.float32),
        skip_prev_state: np.ndarray = np.empty(0, dtype=np.int32),
        skip_penalty: np.ndarray = np.empty(0, dtype=np.float32),
    ):
        frames = emission.shape[0]
        token_count = tokens.shape[0]
        states = token_count * 2 + 1

        labels = np.empty(states, dtype=np.int32)
        state_cost = np.empty(states, dtype=np.float32)
        state_sub_cost = np.zeros(states, dtype=np.float32)
        state_is_filler = np.zeros(states, dtype=np.uint8)

        has_speech_mask = speech_frame_mask.size > 0
        has_n_pure_penalty = bool(n_pure_token_mask.any()) and n_pure_entry_penalty.size > 0
        has_ono_scale = bool(ono_token_mask.any()) and ono_score_scale != 1.0
        has_sub_cost = subclause_blank_cost.size > 0
        has_head_token_silence_penalty = (
            head_token_silence_end.size > 0 and head_token_silence_penalty.size > 0
        )
        has_filler_blank = (
            filler_blank_mask.size == (token_count + 1)
            and filler_frame_logp.size == frames
            and bool(filler_blank_mask.any())
        )
        has_skip = (
            skip_prev_state.size == states
            and skip_penalty.size == states
            and bool((skip_prev_state >= 0).any())
        )

        for state in range(states):
            if state % 2 == 0:
                blank_idx = state // 2
                labels[state] = blank
                state_cost[state] = blank_cost[blank_idx]
                state_sub_cost[state] = (
                    subclause_blank_cost[blank_idx] if has_sub_cost else np.float32(0.0)
                )
                if has_filler_blank:
                    state_is_filler[state] = filler_blank_mask[blank_idx]
            else:
                tok_idx = state // 2
                labels[state] = tokens[tok_idx]
                state_cost[state] = np.float32(0.0)
                state_sub_cost[state] = np.float32(0.0)

        neg_inf = np.float32(-1.0e30)
        prev_dp = np.full(states, neg_inf, dtype=np.float32)
        curr_dp = np.full(states, neg_inf, dtype=np.float32)
        prev_dur = np.zeros(states, dtype=np.int32)
        curr_dur = np.zeros(states, dtype=np.int32)
        prev_clause_gap = np.zeros(states, dtype=np.int32)
        curr_clause_gap = np.zeros(states, dtype=np.int32)

        # 2-bit packed moves: 0 (stay), 1 (s-1 -> s), 2 (s-2 -> s), 3 (skip_prev_state[s] -> s)
        back = np.zeros(((frames + 3) // 4, states), dtype=np.uint8)

        is_speech_0 = (speech_frame_mask[0] != 0) if has_speech_mask else True
        c0 = state_cost[0] if is_speech_0 else np.float32(0.0)
        b0_emit = emission[0, blank]
        if has_filler_blank and state_is_filler[0] != 0:
            if filler_frame_logp[0] > b0_emit:
                b0_emit = filler_frame_logp[0]
        prev_dp[0] = b0_emit - c0
        prev_dur[0] = 1

        if states > 1:
            prev_dp[1] = emission[0, labels[1]]
            if has_ono_scale and ono_token_mask[0]:
                prev_dp[1] = prev_dp[1] * np.float32(ono_score_scale)
            if has_n_pure_penalty and n_pure_token_mask[0]:
                prev_dp[1] -= n_pure_entry_penalty[0]
            prev_dur[1] = 1

        for t in range(1, frames):
            max_state = states if has_skip else min(states, 2 * t + 2)
            is_speech = (speech_frame_mask[t] != 0) if has_speech_mask else True
            em_row = emission[t]

            for s in prange(max_state):
                is_nonblank = (s % 2) == 1
                is_blank = not is_nonblank

                # Move 0: s -> s
                best_val = prev_dp[s]
                best_move = 0
                best_clause_gap = prev_clause_gap[s]
                best_head_guard_end = prev_clause_gap[s]
                best_d = prev_dur[s] + 1

                # Move 1: s - 1 -> s
                if s > 0:
                    ps = s - 1
                    val1 = prev_dp[ps]
                    if has_n_pure_penalty and is_nonblank and n_pure_token_mask[s // 2]:
                        val1 -= n_pure_entry_penalty[t]
                    m1_clause_gap = prev_clause_gap[ps]
                    m1_head_guard_end = prev_clause_gap[ps]
                    if has_head_token_silence_penalty:
                        if is_blank:
                            next_token = s // 2
                            if (next_token < token_count
                                    and head_token_silence_penalty[next_token] > np.float32(0.0)):
                                m1_head_guard_end = head_token_silence_end[t]
                            else:
                                m1_head_guard_end = 0
                        elif (head_token_silence_penalty[s // 2] <= np.float32(0.0)
                              or m1_head_guard_end <= t):
                            m1_head_guard_end = 0

                    if val1 > best_val:
                        best_val = val1
                        best_move = 1
                        best_d = 1
                        best_clause_gap = m1_clause_gap
                        best_head_guard_end = m1_head_guard_end

                # Move 2: s - 2 -> s
                if s > 1 and is_nonblank and labels[s] != labels[s - 2]:
                    ps = s - 2
                    val2 = prev_dp[ps]
                    if has_n_pure_penalty and n_pure_token_mask[s // 2]:
                        val2 -= n_pure_entry_penalty[t]
                    m2_clause_gap = prev_clause_gap[ps]
                    m2_head_guard_end = 0
                    if (has_head_token_silence_penalty
                            and head_token_silence_penalty[s // 2] > np.float32(0.0)):
                        m2_head_guard_end = head_token_silence_end[t]

                    if val2 > best_val:
                        best_val = val2
                        best_move = 2
                        best_d = 1
                        best_clause_gap = m2_clause_gap
                        best_head_guard_end = m2_head_guard_end

                # Move 3: 易错拟声子句整句跳过弧 (skip_prev_state[s] -> s)
                if has_skip and is_blank:
                    ps3 = skip_prev_state[s]
                    if ps3 >= 0:
                        val3 = prev_dp[ps3] - skip_penalty[s]
                        if val3 > best_val:
                            best_val = val3
                            best_move = 3
                            best_d = prev_dur[ps3] + 1
                            best_clause_gap = prev_clause_gap[ps3]
                            best_head_guard_end = prev_clause_gap[ps3]

                sub_c = np.float32(0.0)
                if is_blank and state_sub_cost[s] > 0.0:
                    if best_d > subclause_blank_free_frames:
                        sub_c = state_sub_cost[s]

                c = (state_cost[s] + sub_c) if is_speech else np.float32(0.0)
                position_c = np.float32(0.0)
                if (has_head_token_silence_penalty and is_nonblank
                        and best_head_guard_end > t):
                    position_c += head_token_silence_penalty[s // 2]

                frame_value = em_row[labels[s]]
                if has_filler_blank and is_blank and state_is_filler[s] != 0:
                    f_val = filler_frame_logp[t]
                    if f_val > frame_value:
                        frame_value = f_val
                if has_ono_scale and is_nonblank and ono_token_mask[s // 2]:
                    frame_value = frame_value * np.float32(ono_score_scale)

                curr_dp[s] = best_val + frame_value - c - position_c
                curr_dur[s] = best_d
                if has_head_token_silence_penalty:
                    curr_clause_gap[s] = best_head_guard_end
                else:
                    curr_clause_gap[s] = best_clause_gap

                if best_move > 0:
                    back[t >> 2, s] |= np.uint8(best_move << ((t & 3) << 1))

            if max_state < states:
                curr_dp[max_state:states] = neg_inf
                curr_dur[max_state:states] = 0
                curr_clause_gap[max_state:states] = 0

            prev_dp, curr_dp = curr_dp, prev_dp
            prev_dur, curr_dur = curr_dur, prev_dur
            prev_clause_gap, curr_clause_gap = curr_clause_gap, prev_clause_gap

        final_state = states - 1 if prev_dp[states - 1] >= prev_dp[states - 2] else states - 2
        path = np.empty(frames, dtype=np.int32)
        path[frames - 1] = final_state
        curr_s = final_state

        for t in range(frames - 1, 0, -1):
            move = (int(back[t >> 2, curr_s]) >> ((t & 3) << 1)) & 3
            if has_skip and move == 3:
                curr_s = int(skip_prev_state[curr_s])
                if curr_s < 0:
                    curr_s = 0
            else:
                if curr_s < move:
                    curr_s = 0
                else:
                    curr_s -= move
            path[t - 1] = curr_s

        return path, prev_dp[final_state]
else:
    def _viterbi_multispeaker_duration(*args, **kwargs):
        raise RuntimeError("Numba is required for MultiSpeakerCTCAligner")


class MultiSpeakerCTCAligner:
    """独立于 ConstrainedCTCAligner 的多人/乱序专用对齐器。"""

    def __init__(
        self,
        blank: int = 0,
        nasal_token_id: int = 4,
        subclause_blank_penalty: float = 0.0,
        subclause_blank_free_sec: float = 1.0,
        subclause_blank_max_tokens: int = 3,
        intra_blank_penalty: float = 0.10,
        min_phonemes_for_penalty: int = 2,
        frame_sec: float = FRAME_SEC,
        n_pure_continuation_penalty: float = 0.20,
        n_pure_continuation_lookahead_frames: int = 4,
        n_pure_continuation_drop_threshold: float = 1.0,
        n_pure_continuation_drop_ramp: float = 2.0,
    ):
        self.blank = int(blank)
        self.frame_sec = float(frame_sec)
        self.nasal_token_id = int(nasal_token_id)
        self.intra_blank_penalty = float(intra_blank_penalty)
        self.min_phonemes_for_penalty = int(min_phonemes_for_penalty)
        self.subclause_blank_penalty = float(subclause_blank_penalty)
        self.subclause_blank_free_sec = float(subclause_blank_free_sec)
        self.subclause_blank_max_tokens = int(subclause_blank_max_tokens)
        self.subclause_blank_free_frames = (
            int(round(self.subclause_blank_free_sec / self.frame_sec))
            if self.subclause_blank_free_sec > 0
            else 0
        )
        self.n_pure_continuation_penalty = float(n_pure_continuation_penalty)
        self.n_pure_continuation_lookahead_frames = int(n_pure_continuation_lookahead_frames)
        self.n_pure_continuation_drop_threshold = float(n_pure_continuation_drop_threshold)
        self.n_pure_continuation_drop_ramp = float(n_pure_continuation_drop_ramp)
        self._last_path: Optional[np.ndarray] = None
        self._last_viterbi_score: float = 0.0
        self._last_hit_mask: Optional[np.ndarray] = None

    def __call__(
        self,
        emission: Any,
        targets: List[List[int]],
        blank_punct_mask: Optional[Any] = None,
        speech_frame_mask: Optional[Any] = None,
        line_group: Optional[Any] = None,
        head_token_silence_end: Optional[Any] = None,
        head_token_silence_penalty: Optional[Any] = None,
        head_standalone_n_flag: Optional[Any] = None,
        ono_token_mask: Optional[Any] = None,
        ono_score_scale: float = 1.0,
        filler_blank_mask: Optional[Any] = None,
        filler_frame_logp: Optional[Any] = None,
        skip_prev_state: Optional[Any] = None,
        skip_penalty: Optional[Any] = None,
    ) -> List[List[TokenSpan]]:
        if hasattr(emission, "detach"):
            em_np = emission.detach().cpu().numpy().astype(np.float32)
        else:
            em_np = np.asarray(emission, dtype=np.float32)
        if em_np.ndim == 3:
            em_np = em_np[0]

        flat_tokens = np.asarray([tok for line in targets for tok in line], dtype=np.int32)
        if flat_tokens.size == 0:
            self._last_hit_mask = np.zeros(0, dtype=bool)
            return [[] for _ in targets]

        blank_cost = np.zeros(len(flat_tokens) + 1, dtype=np.float32)
        if self.intra_blank_penalty > 0:
            cursor = 0
            for line in targets:
                line_len = len(line)
                if line_len >= self.min_phonemes_for_penalty:
                    for b_idx in range(cursor + 1, cursor + line_len):
                        if blank_punct_mask is not None and b_idx < len(blank_punct_mask) and blank_punct_mask[b_idx]:
                            blank_cost[b_idx] = 0.0
                        else:
                            blank_cost[b_idx] = np.float32(self.intra_blank_penalty)
                cursor += line_len

        sp_mask = (
            np.asarray(speech_frame_mask, dtype=np.uint8)
            if speech_frame_mask is not None
            else np.empty(0, dtype=np.uint8)
        )
        head_silence_end = (
            np.asarray(head_token_silence_end, dtype=np.int32)
            if head_token_silence_end is not None
            else np.empty(0, dtype=np.int32)
        )
        head_silence_penalty = (
            np.asarray(head_token_silence_penalty, dtype=np.float32)
            if head_token_silence_penalty is not None
            else np.empty(0, dtype=np.float32)
        )

        n_pure_token_mask = np.zeros(len(flat_tokens), dtype=np.uint8)
        cursor = 0
        for line_index, line in enumerate(targets):
            n_line = len(line)
            is_head_standalone_n = False
            if n_line > 0 and int(line[0]) == self.nasal_token_id:
                if head_standalone_n_flag is not None and len(head_standalone_n_flag) > line_index:
                    is_head_standalone_n = bool(int(head_standalone_n_flag[line_index]))
                elif n_line == 1:
                    is_head_standalone_n = True
            if is_head_standalone_n and self.n_pure_continuation_penalty > 0.0:
                n_pure_token_mask[cursor] = 1
            cursor += n_line

        if n_pure_token_mask.any():
            n_pure_entry_penalty = _n_pure_continuation_penalty(
                em_np[:, self.nasal_token_id],
                self.n_pure_continuation_lookahead_frames,
                self.n_pure_continuation_penalty,
                self.n_pure_continuation_drop_threshold,
                self.n_pure_continuation_drop_ramp,
            )
        else:
            n_pure_entry_penalty = np.zeros(em_np.shape[0], dtype=np.float32)

        subclause_blank_cost = np.zeros(len(flat_tokens) + 1, dtype=np.float32)
        if self.subclause_blank_penalty > 0.0 and line_group is not None:
            cursor = 0
            for i, line in enumerate(targets):
                n_line = len(line)
                if i + 1 < len(targets) and line_group[i] == line_group[i + 1]:
                    b_idx = cursor + n_line
                    subclause_blank_cost[b_idx] = np.float32(self.subclause_blank_penalty)
                cursor += n_line
        if ono_token_mask is not None and len(ono_token_mask) == flat_tokens.shape[0]:
            ono_mask_arr = np.asarray(ono_token_mask, dtype=np.uint8)
        else:
            ono_mask_arr = np.zeros(flat_tokens.shape[0], dtype=np.uint8)

        if filler_blank_mask is not None and len(filler_blank_mask) == (flat_tokens.shape[0] + 1):
            filler_mask_arr = np.asarray(filler_blank_mask, dtype=np.uint8)
        else:
            filler_mask_arr = np.empty(0, dtype=np.uint8)

        if filler_frame_logp is not None and len(filler_frame_logp) == em_np.shape[0]:
            filler_logp_arr = np.asarray(filler_frame_logp, dtype=np.float32)
        else:
            filler_logp_arr = np.empty(0, dtype=np.float32)

        states_n = flat_tokens.shape[0] * 2 + 1
        if skip_prev_state is not None and len(skip_prev_state) == states_n:
            skip_prev_arr = np.asarray(skip_prev_state, dtype=np.int32)
        else:
            skip_prev_arr = np.empty(0, dtype=np.int32)

        if skip_penalty is not None and len(skip_penalty) == states_n:
            skip_pen_arr = np.asarray(skip_penalty, dtype=np.float32)
        else:
            skip_pen_arr = np.empty(0, dtype=np.float32)

        decoder_kwargs = dict(
            n_pure_token_mask=n_pure_token_mask,
            n_pure_entry_penalty=n_pure_entry_penalty,
            blank_cost=blank_cost,
            speech_frame_mask=sp_mask,
            subclause_blank_cost=subclause_blank_cost,
            subclause_blank_free_frames=self.subclause_blank_free_frames,
            head_token_silence_end=head_silence_end,
            head_token_silence_penalty=head_silence_penalty,
            blank=self.blank,
            ono_token_mask=ono_mask_arr,
            ono_score_scale=np.float32(ono_score_scale),
            filler_blank_mask=filler_mask_arr,
            filler_frame_logp=filler_logp_arr,
            skip_prev_state=skip_prev_arr,
            skip_penalty=skip_pen_arr,
        )

        path, final_score = _run_decoder(
            _viterbi_multispeaker_duration, em_np, flat_tokens, decoder_kwargs
        )
        self._last_path = np.array(path, copy=True)
        self._last_viterbi_score = float(final_score)

        grouped_spans: List[List[TokenSpan]] = []
        token_cursor = 0
        total_tokens = flat_tokens.shape[0]
        wanted = 2 * np.arange(total_tokens, dtype=np.int64) + 1
        sorted_path = np.sort(path)
        lo = np.searchsorted(sorted_path, wanted, side="left")
        hi = np.searchsorted(sorted_path, wanted, side="right")
        hit_mask = hi > lo
        self._last_hit_mask = np.array(hit_mask, copy=True)

        for line in targets:
            line_spans = []
            for tok in line:
                state_idx = 2 * token_cursor + 1
                if hit_mask[token_cursor]:
                    st = int(lo[token_cursor])
                    en = int(hi[token_cursor])
                    avg_score = float(np.mean(em_np[st:en, tok]))
                else:
                    pos = int(np.searchsorted(sorted_path, state_idx, side="left"))
                    st = pos if pos < len(sorted_path) else 0
                    en = st + 1
                    avg_score = float(em_np[st, tok])
                line_spans.append(TokenSpan(token=int(tok), start=st, end=en, score=avg_score))
                token_cursor += 1
            grouped_spans.append(line_spans)

        return grouped_spans


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
        from align_model import get_mms_model, _resolve_device
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


# ══════════════════════════════════════════════════════════════════════════
#  3. 角色标题自动识别、双耳协同发射与乱序重排核心管线
# ══════════════════════════════════════════════════════════════════════════

_NON_SPEAKER_HEADER_RE = re.compile(
    r"(?i)^(?:トラック|track|tr\.?\s*\d|scene|シーン|第?\s*[0-9０-９一二三四五六七八九十]+\s*[章話節回部編]|"
    r"プロローグ|エピローグ|タイトル|オープニング|エンディング|おまけ|特典|フリートーク|free\s*talk|"
    r"se|bgm|効果音|環境音|音声|台本|本編|目次|登場人物|キャスト|cast|キャラクター|設定|注意|紹介)"
)

_ACTION_SUFFIX_RE = re.compile(
    r"[・/／：:\s]+(?:吐息|耳舐め|耳かき|耳キス|キス|フェラ|フェラチオ|パイズリ|手コキ|足コキ|"
    r"オナニー|喘ぎ|囁き|舐め|しゃぶり|射精|絶頂|イキ|クンニ|アナル|挿入|本番|素股|乳首|乳首責め).*$"
)

_STANDALONE_BRACKET_HEADER_RE = re.compile(r"^\s*(?:【([^】]{1,20})】|《([^》]{1,20})》)\s*$")


def _clean_speaker_header_name(raw_inside: str) -> Tuple[List[str], bool]:
    """从【...】内部文本提取角色名列表，并返回是否带有动作后缀。"""
    s = (raw_inside or "").strip()
    if not s or _NON_SPEAKER_HEADER_RE.search(s):
        return [], False
    if any(ch in s for ch in "。！？!?「」『』（）()"):
        return [], False
    cleaned = _ACTION_SUFFIX_RE.sub("", s).strip()
    has_action_suffix = (cleaned != s)
    if not cleaned or _NON_SPEAKER_HEADER_RE.search(cleaned):
        return [], False
    parts = [p.strip() for p in re.split(r"[＆&、，,＋+×/／]", cleaned) if p.strip()]
    valid_parts = [p for p in parts if 1 <= len(p) <= 12 and not _NON_SPEAKER_HEADER_RE.search(p)]
    return valid_parts, has_action_suffix


def auto_normalize_speaker_headers(raw_lines: List[str]) -> Tuple[List[str], List[str]]:
    """若台本尚未使用 〖角色〗，自动检测高频复现的独立【角色名】标题行并转为 〖角色名〗。"""
    if any("\u3016" in ln and "\u3017" in ln for ln in raw_lines):
        return raw_lines, []

    header_hits: List[Tuple[int, List[str], bool]] = []
    name_counts: Dict[str, int] = {}

    for idx, ln in enumerate(raw_lines):
        m = _STANDALONE_BRACKET_HEADER_RE.match(ln)
        if not m:
            continue
        inside = m.group(1) if m.group(1) is not None else m.group(2)
        names, has_act = _clean_speaker_header_name(inside)
        if not names:
            continue
        header_hits.append((idx, names, has_act))
        for nm in names:
            name_counts[nm] = name_counts.get(nm, 0) + 1

    recurring = {k for k, cnt in name_counts.items() if cnt >= 2}
    if len(recurring) < 2 or len(header_hits) < 4:
        return raw_lines, []

    ordered_speakers: List[str] = []
    for _, names, _ in header_hits:
        for nm in names:
            if nm in recurring and nm not in ordered_speakers:
                ordered_speakers.append(nm)

    if len(ordered_speakers) < 2:
        return raw_lines, []

    hit_map = {idx: (names, has_act) for idx, names, has_act in header_hits}
    out_lines: List[str] = []
    for idx, ln in enumerate(raw_lines):
        if idx in hit_map:
            names, _ = hit_map[idx]
            kept = [nm for nm in names if nm in recurring]
            if kept:
                tag_str = "".join(f"\u3016{nm}\u3017" for nm in kept)
                out_lines.append(tag_str)
                continue
        out_lines.append(ln)

    return out_lines, ordered_speakers


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


def _build_claimed_acoustic_ctx(
    acoustic_ctx: Dict[str, Any],
    claimed_spans: List[Tuple[float, float, float]],
    pad_sec: float = 0.08,
) -> Dict[str, Any]:
    if not claimed_spans:
        return acoustic_ctx
    rms_l = acoustic_ctx["rms_l"]
    rms_r = acoustic_ctx["rms_r"]
    stereo_rms = np.maximum(rms_l, rms_r).copy()
    total_frames = len(stereo_rms)
    stereo_floor = compute_track_noise_floor(stereo_rms, percentile=10.0)
    speech_mask = acoustic_ctx["speech_frame_mask"].copy()

    for st_sec, en_sec, penalty in claimed_spans:
        if penalty <= 0.0:
            continue
        st_f = max(0, min(total_frames, int(round((st_sec - pad_sec) / FRAME_SEC))))
        en_f = max(st_f, min(total_frames, int(round((en_sec + pad_sec) / FRAME_SEC))))
        if en_f > st_f:
            stereo_rms[st_f:en_f] = stereo_floor
            speech_mask[st_f:en_f] = 0

    new_head_silence_end = build_next_stable_silence_end(
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
    new_ctx = dict(acoustic_ctx)
    new_ctx["speech_frame_mask"] = speech_mask
    new_ctx["head_silence_end"] = new_head_silence_end
    return new_ctx


def _build_claimed_emission(
    base_emission_2d: torch.Tensor,
    acoustic_ctx: Dict[str, Any],
    claimed_spans: List[Tuple[float, float, float]],
    window_bounds: Optional[Tuple[float, float]] = None,
    base_filler_margin: float = 1.35,
    pad_sec: float = 0.06,
) -> Tuple[np.ndarray, np.ndarray]:
    em_np = (
        base_emission_2d.detach().cpu().numpy().copy()
        if hasattr(base_emission_2d, "detach")
        else np.array(base_emission_2d, dtype=np.float32, copy=True)
    )
    total_frames, n_classes = em_np.shape
    blank = int(acoustic_ctx.get("blank", 0))
    ignored = set(acoustic_ctx.get("ignored_classes", ()))
    valid_cls = [c for c in range(n_classes) if c != blank and c not in ignored]

    max_nb = np.max(em_np[:, valid_cls], axis=1).astype(np.float32)
    if window_bounds is not None:
        filler_logp = (max_nb - np.float32(base_filler_margin)).astype(np.float32)
    else:
        filler_logp = np.full(total_frames, -1.0e30, dtype=np.float32)

    for st_sec, en_sec, penalty in claimed_spans:
        st_f = max(0, min(total_frames, int(round((st_sec - pad_sec) / FRAME_SEC))))
        en_f = max(st_f, min(total_frames, int(round((en_sec + pad_sec) / FRAME_SEC))))
        if en_f > st_f and penalty > 0.0:
            orig_blank = em_np[st_f:en_f, blank].copy()
            em_np[st_f:en_f, valid_cls] -= np.float32(penalty)
            em_np[st_f:en_f, blank] -= np.float32(penalty)
            filler_logp[st_f:en_f] = np.maximum(
                filler_logp[st_f:en_f],
                np.maximum(orig_blank, max_nb[st_f:en_f] - np.float32(0.10)),
            )

    if window_bounds is not None:
        win_lo, win_hi = window_bounds
        lo_f = max(0, min(total_frames, int(round(win_lo / FRAME_SEC))))
        hi_f = max(lo_f, min(total_frames, int(round(win_hi / FRAME_SEC))))
        if lo_f > 0:
            em_np[:lo_f, valid_cls] -= np.float32(3.0)
            filler_logp[:lo_f] = np.maximum(filler_logp[:lo_f], max_nb[:lo_f] - np.float32(0.10))
        if hi_f < total_frames:
            em_np[hi_f:, valid_cls] -= np.float32(3.0)
            filler_logp[hi_f:] = np.maximum(filler_logp[hi_f:], max_nb[hi_f:] - np.float32(0.10))

    return em_np, filler_logp


def _slide_window_curve(
    ids: List[int],
    em_np: np.ndarray,
    lo_sec: float,
    hi_sec: float,
    step_sec: float = 0.2,
    sec_per_tok: float = 0.09,
) -> List[Tuple[float, float]]:
    """在 [lo,hi] 内逐位置滑窗打分, 返回 [(起始秒, 平均分)] 曲线。"""
    n = len(ids)
    if n < 2:
        return []
    need_f = max(1, int(round(sec_per_tok * n / FRAME_SEC)))
    id_arr = np.asarray(ids, dtype=np.int64)
    total_frames = em_np.shape[0]
    lo_f = max(0, int(lo_sec / FRAME_SEC))
    hi_f = min(total_frames - need_f - 1, int(hi_sec / FRAME_SEC))
    if hi_f <= lo_f:
        return []
    cuts = np.linspace(0, need_f, n + 1).astype(int)
    step_f = max(1, int(step_sec / FRAME_SEC))
    out: List[Tuple[float, float]] = []
    t = lo_f
    while t <= hi_f:
        seg = em_np[t : t + need_f]
        vals = [
            float(np.max(seg[cuts[i] : cuts[i + 1], id_arr[i]]))
            for i in range(n)
            if cuts[i + 1] > cuts[i]
        ]
        out.append((t * FRAME_SEC, float(np.mean(vals)) if vals else -30.0))
        t += step_f
    return out


def _slide_window_peak(
    ids: List[int],
    em_np: np.ndarray,
    lo_sec: float,
    hi_sec: float,
    step_sec: float = 0.2,
    sec_per_tok: float = 0.09,
) -> Optional[Tuple[float, float]]:
    """滑窗峰值扫描: 假设子句连续发音, 在 [lo,hi] 内找音素平均分最高的位置。

    返回 (best_score, best_start_sec); 窗口无效时返回 None。
    这是块级重定位的独立声学评估, 完全脱离 Viterbi 链的单调约束,
    用于回答"这段音素序列在整个窗口内最像在哪里被说出来"。
    """
    n = len(ids)
    if n < 2:
        return None
    need_f = max(1, int(round(sec_per_tok * n / FRAME_SEC)))
    id_arr = np.asarray(ids, dtype=np.int64)
    total_frames = em_np.shape[0]
    lo_f = max(0, int(lo_sec / FRAME_SEC))
    hi_f = min(total_frames - need_f - 1, int(hi_sec / FRAME_SEC))
    if hi_f <= lo_f:
        return None
    cuts = np.linspace(0, need_f, n + 1).astype(int)
    step_f = max(1, int(step_sec / FRAME_SEC))
    best_score = -1.0e30
    best_t = lo_sec
    t = lo_f
    while t <= hi_f:
        seg = em_np[t : t + need_f]
        vals = []
        for i in range(n):
            if cuts[i + 1] > cuts[i]:
                vals.append(float(np.max(seg[cuts[i] : cuts[i + 1], id_arr[i]])))
        v = float(np.mean(vals)) if vals else -30.0
        if v > best_score:
            best_score = v
            best_t = t * FRAME_SEC
        t += step_f
    return best_score, best_t


def _clause_token_ids(
    ci: int,
    alignment_tokens: List[List[str]],
    acoustic_ctx: Dict[str, Any],
) -> List[int]:
    """把子句的发音 token 展开为 CTC 音素 id 序列 (与解码器同一词典口径)。"""
    tokenizer = acoustic_ctx["tokenizer"]
    blank = int(acoustic_ctx.get("blank", 0))
    ignored = set(acoustic_ctx.get("ignored_classes", ()))
    ids: List[int] = []
    for p in alignment_tokens[ci]:
        for ch in str(p):
            tid = tokenizer.dictionary.get(ch)
            if tid is not None and tid != blank and tid not in ignored:
                ids.append(int(tid))
    return ids


def _build_torn_line_blocks(
    semantic: List[int],
    res_map: Dict[int, List[Dict[str, Any]]],
    line_group: Sequence[int],
    alignment_tokens: List[List[str]],
    clause_surfaces: Optional[Sequence[str]],
    max_gap_sec: float,
) -> List[List[int]]:
    """把同一台本行拆成相隔过远的两个时间簇。

    顿号、逗号、省略号附近出现超长间隔时，前一簇通常被提前吸走。
    无法取得 surface 时只依赖行号和真实时间间隔，不凭文本猜断点。
    """
    groups: Dict[int, List[int]] = {}
    for ci in semantic:
        inf = _clause_summary(res_map.get(ci, []))
        if not inf["skipped"]:
            groups.setdefault(int(line_group[ci]), []).append(ci)

    blocks: List[List[int]] = []
    for group in groups.values():
        if len(group) < 2:
            continue
        for left, right in zip(group, group[1:]):
            left_inf = _clause_summary(res_map.get(left, []))
            right_inf = _clause_summary(res_map.get(right, []))
            gap = right_inf["st"] - left_inf["en"]
            if gap <= max_gap_sec or len(alignment_tokens[right]) < 3:
                continue
            # 只收集当前断裂的较早时间簇。靠后的完整行尾作为锚，
            # 防止把两个原本都正确的子句簇互换或整体乱搬。
            moved = [left]
            for prev in reversed(group[: group.index(left)]):
                prev_inf = _clause_summary(res_map.get(prev, []))
                next_inf = _clause_summary(res_map.get(moved[0], []))
                if next_inf["st"] - prev_inf["en"] > max_gap_sec:
                    break
                moved.insert(0, prev)
            anchor = [
                ci for ci in group[group.index(right):]
                if not _clause_summary(res_map.get(ci, []))["skipped"]
            ]
            if moved and anchor:
                blocks.append(moved + anchor)
            break
    return blocks


def _repair_tail_gap_clauses(
    spk_chains: Dict[str, List[int]],
    per_spk_results: Dict[str, Dict[int, List[Dict[str, Any]]]],
    alignment_tokens: List[List[str]],
    ono_clause_flags: Sequence[int],
    acoustic_ctx: Dict[str, Any],
    base_emission_2d: torch.Tensor,
    log: Callable[[str], None],
    enable_silence_snap: bool = True,
    enable_ono_gate: bool = True,
    enable_word_skip: bool = False,
    word_skip_token_penalty: float = 4.8,
    clause_surfaces: Optional[Sequence[str]] = None,
    clause_speakers: Optional[Sequence[Any]] = None,
    line_group: Optional[Sequence[int]] = None,
    gap_thr_sec: float = 30.0,
    low_score_margin: float = 1.0,
    min_accept_frac: float = 0.55,
) -> None:
    """修复角色链内部的"提前吸附": 按同角色空档 + 原句低分双判据重定位。

    判据 (只用该角色自身链, 不掺跨角色信息):
      a) 该句结束到同角色下一句开始 (无下一句则到音频结尾) 超过 gap_thr_sec;
      b) 该句原对齐得分明显低于本轨实词正常水平 (低于中位数 low_score_margin)。

    处理: 对空档区间做滑窗扫描, 取"得分接近实词正常水平"的位置,
    而不是取最高分 —— 最高分往往是别的角色的清晰语音。只有候选位置
    得分达到正常水平的 min_accept_frac 以上才接受, 否则保持原位不动。
    """
    em_np = (
        base_emission_2d.detach().cpu().numpy()
        if hasattr(base_emission_2d, "detach")
        else np.asarray(base_emission_2d, dtype=np.float32)
    )
    total_dur = em_np.shape[0] * FRAME_SEC

    # 实词正常得分基线按"句长档位"取: 短句(音素少)的平均分天然低于长句,
    # 用全轨中位数会把短句全部挡在门外 (01 轨 ci248 n=11 原分 -5.30,
    # 全轨中位 -1.57, 而它在真实位置也只有 -5.39 —— 是句长效应而非对齐错误)。
    baseline_by_n: Dict[int, List[float]] = {}
    for result_map in per_spk_results.values():
        for ci, cl_res in result_map.items():
            if ci < len(ono_clause_flags) and ono_clause_flags[ci]:
                continue
            inf = _clause_summary(cl_res)
            if not inf["skipped"] and inf["n"] >= 4:
                baseline_by_n.setdefault(int(inf["n"]), []).append(inf["score"])
    if sum(len(v) for v in baseline_by_n.values()) < 10:
        return

    def _baseline_for(n_tok: int) -> float:
        """取最接近的同长档位基线; 无同档时回退到全量中位数。"""
        best_key = min(baseline_by_n, key=lambda k: abs(k - n_tok))
        if abs(best_key - n_tok) <= max(4, n_tok // 3):
            return float(np.median(baseline_by_n[best_key]))
        all_vals = [v for vals in baseline_by_n.values() for v in vals]
        return float(np.median(all_vals))

    moved_total = 0
    for spk, chain in spk_chains.items():
        res_map = per_spk_results.get(spk, {})
        placed = [
            ci for ci in chain
            if not (ci < len(ono_clause_flags) and ono_clause_flags[ci])
            and not _clause_summary(res_map.get(ci, []))["skipped"]
        ]
        for pos, ci in enumerate(placed):
            inf = _clause_summary(res_map.get(ci, []))
            if inf["n"] < 4:
                continue
            # (a) 同角色空档: 本句结束 -> 该角色下一句开始; 无下一句则到音频结尾
            if pos + 1 < len(placed):
                nxt = _clause_summary(res_map.get(placed[pos + 1], []))
                gap_lo, gap_hi = inf["en"], nxt["st"]
            else:
                gap_lo, gap_hi = inf["en"], total_dur
            if gap_hi - gap_lo <= gap_thr_sec:
                continue
            # (b) 原句得分低于同长档位的实词正常水平
            baseline = _baseline_for(inf["n"])
            if inf["score"] >= baseline - low_score_margin:
                continue

            # 候选位置用真正的 Viterbi 单句重对齐评估: 固定时长切片滑窗
            # 对语速不均匀的句子会严重错位 (01 轨 ci248 真位置 Viterbi
            # 得分 -1.74, 固定切片只有 -5.39), 因此以对齐器为准。
            em_cand, filler_cand = _build_claimed_emission(
                base_emission_2d=base_emission_2d,
                acoustic_ctx=acoustic_ctx,
                claimed_spans=[],
                window_bounds=(gap_lo + 0.2, max(gap_lo + 0.3, gap_hi - 0.1)),
                base_filler_margin=1.35,
            )
            cand_res = _align_subset_clauses(
                sub_indices=[ci],
                alignment_tokens=alignment_tokens,
                line_group=line_group,
                ono_clause_flags=ono_clause_flags,
                enable_silence_snap=enable_silence_snap,
                acoustic_ctx=acoustic_ctx,
                custom_emission_2d=em_cand,
                enable_inter_line_filler=True,
                enable_ono_skip=enable_ono_gate,
                filler_frame_logp=filler_cand,
                enable_word_skip=enable_word_skip,
                word_skip_token_penalty=word_skip_token_penalty,
                clause_surfaces=clause_surfaces,
                clause_speakers=clause_speakers,
                log_callback=None,
            )
            cand_inf = _clause_summary(cand_res.get(ci, []))
            if cand_inf["skipped"] or cand_inf["dur"] < 0.10:
                continue
            best_t, best_s = cand_inf["st"], cand_inf["score"]
            # 只接受得分达到同长档位实词正常水平的位置; 找不到就不动。
            accept_bar = baseline - low_score_margin
            if best_s < accept_bar:
                log(
                    f"  🧭 [尾部空档] 〖{spk}〗ci{ci} 空档 {gap_hi - gap_lo:.1f}s 内"
                    f"无达标位置(重对齐最高 {best_s:.2f} < {accept_bar:.2f}), 保持不动"
                )
                continue
            shift = best_t - inf["st"]
            if shift < 1.0:
                continue
            for r in res_map.get(ci, []):
                if r.get("original_start") is not None:
                    r["original_start"] = float(r["original_start"]) + shift
                if r.get("original_end") is not None:
                    r["original_end"] = float(r["original_end"]) + shift
                if r.get("start") is not None:
                    s = _parse_lrc_ts_sec(r.get("start"))
                    if s is not None:
                        r["start"] = _format_time(s + shift)
                if r.get("end") is not None:
                    e = _parse_lrc_ts_sec(r.get("end"))
                    if e is not None:
                        r["end"] = _format_time(e + shift)
            moved_total += 1
            log(
                f"  🧭 [尾部空档] 〖{spk}〗ci{ci} {inf['st']:.2f}s → {best_t:.2f}s "
                f"位移 {shift:+.2f}s (空档 {gap_hi - gap_lo:.1f}s, 原分 {inf['score']:.2f}, "
                f"基线 {baseline:.2f}, 新分 {best_s:.2f})"
            )

    if moved_total:
        log(f"  🧭 [尾部空档] 共重定位 {moved_total} 句")


def _repair_overlapping_clause_blocks(
    spk_chains: Dict[str, List[int]],
    per_spk_results: Dict[str, Dict[int, List[Dict[str, Any]]]],
    alignment_tokens: List[List[str]],
    line_group: Sequence[int],
    ono_clause_flags: Sequence[int],
    acoustic_ctx: Dict[str, Any],
    base_emission_2d: torch.Tensor,
    log: Callable[[str], None],
    clause_surfaces: Optional[Sequence[str]] = None,
    min_move_sec: float = 5.0,
    min_gain: float = 0.30,
    overlap_thr_sec: float = 1.0,
    scan_pad_sec: float = 30.0,
    line_tear_thr_sec: float = 8.0,
    min_distinct_gain: float = 0.50,
) -> None:
    """修复"提前吸附": 链内时间倒挂 / 同台本行子句撕裂时, 用滑窗峰值独立重定位。

    成因: 角色链内单调放置 + 行间 filler 零代价, 弱/短子句会被更清晰的
    异角色语音吸走。两种表现:
    a) 倒挂: cur_st < prev_en - overlap_thr (被吸进前一句的时间段),
       锚是前一句, 被搬的是 cur (往后移回真实位置);
    b) 撕裂: 同一台本行(line_group 相同)的相邻子句间隔 >> 人声一口气
       的合理上限, 说明行中途被拽走。被吸走的是时间靠前的簇 (链内
       单调使它只能更早), 锚是时间靠后的簇, 修复方向同样是往后移。
       (02 轨 ci532 1141s / ci533 1197s 同属一行, 真实位置 1194s。)
    修复: 对问题子句在窗口内做滑窗峰值扫描; 峰值位置比当前位置后移
    >= min_move_sec (只接受正向位移 —— 吸附方向永远是提前) 且峰值得分
    比"当前位置窗口峰"高 >= min_gain 时整体搬迁。
    只接受声学证据; 无显著峰一律保持原位, 不做文本结构猜测。
    """
    em_np = (
        base_emission_2d.detach().cpu().numpy()
        if hasattr(base_emission_2d, "detach")
        else np.asarray(base_emission_2d, dtype=np.float32)
    )
    total_dur = em_np.shape[0] * FRAME_SEC

    by_speaker: Dict[str, Dict[int, List[Dict[str, Any]]]] = per_spk_results
    moved_total = 0
    for spk, chain in spk_chains.items():
        semantic = [ci for ci in chain if not (ci < len(ono_clause_flags) and ono_clause_flags[ci])]
        if len(semantic) < 3:
            continue
        res_map = by_speaker.get(spk, {})
        # 连续倒挂的子句聚成块, 整块一起搬, 避免短句被单独拉走
        blocks: List[List[int]] = []
        cur_block: List[int] = []
        for pos, ci in enumerate(semantic):
            inf = _clause_summary(res_map.get(ci, []))
            if inf["skipped"]:
                if cur_block:
                    blocks.append(cur_block)
                    cur_block = []
                continue
            prev_ci = None
            for back in range(pos - 1, -1, -1):
                pinf = _clause_summary(res_map.get(semantic[back], []))
                if not pinf["skipped"]:
                    prev_ci = semantic[back]
                    break
            prev_inf = _clause_summary(res_map.get(prev_ci, [])) if prev_ci is not None else None
            overlapped = (
                prev_inf is not None and not prev_inf["skipped"]
                and inf["st"] < prev_inf["en"] - overlap_thr_sec
            )
            # 同台本行相邻子句撕裂: line_group 相同但间隔超一口气上限
            if (
                not overlapped
                and prev_ci is not None
                and line_group[ci] == line_group[prev_ci]
                and prev_inf is not None and not prev_inf["skipped"]
                and inf["st"] - prev_inf["en"] > line_tear_thr_sec
                and len(alignment_tokens[ci]) >= 3
            ):
                overlapped = True  # 行中途被拽走, 同样按块处理
            if overlapped:
                if not cur_block:
                    # 块的起点必须包含"被倒挂的锚"(前一句), 否则无法定义倒挂
                    cur_block.append(prev_ci) if prev_ci is not None else None
                cur_block.append(ci)
            else:
                if cur_block:
                    blocks.append(cur_block)
                    cur_block = []
        if cur_block:
            blocks.append(cur_block)

        # 行级补充: 台本同一行被拆成两个远隔时间簇时，即使链内
        # 中间夹了其他台词，也仍然属于同一口气的断裂，不能被链上
        # 邻接关系漏掉。与倒挂块按 ci 去重，避免二次搬迁。
        covered = {ci for block in blocks for ci in block}
        for torn_block in _build_torn_line_blocks(
            semantic=semantic,
            res_map=res_map,
            line_group=line_group,
            alignment_tokens=alignment_tokens,
            clause_surfaces=clause_surfaces,
            max_gap_sec=line_tear_thr_sec,
        ):
            if not any(ci in covered for ci in torn_block):
                blocks.append(torn_block)
                covered.update(torn_block)

        for block in blocks:
            if len(block) < 2:
                continue
            # 倒挂块: block[0]=锚(前一句, 位置正确), 搬 block[1:] 往后移。
            # 撕裂块: 同一 line_group 的子句被聚进来时, 时间靠后的簇是锚,
            # 时间靠前的簇是被吸走的 (链内单调使吸附只能往提前方向发生),
            # 搬时间靠前的簇, 使其向靠后簇的真实位置靠拢。
            anchor_ci = block[0]
            anchor_inf = _clause_summary(res_map.get(anchor_ci, []))
            if anchor_inf["skipped"]:
                continue
            moved_block = list(block[1:])
            anchor_lg = line_group[anchor_ci]
            tear_line = [ci for ci in moved_block if line_group[ci] == anchor_lg]
            if tear_line:
                # 撕裂: 以同行中时间最靠后的子句为锚
                torn_sorted = sorted(tear_line, key=lambda c: _clause_summary(res_map.get(c, []))["st"])
                late_ci = torn_sorted[-1]
                late_inf = _clause_summary(res_map.get(late_ci, []))
                if late_inf["st"] > anchor_inf["st"]:
                    # 交换: late 成为锚, 其余(含原锚)都是被搬对象
                    anchor_ci, anchor_inf = late_ci, late_inf
                    moved_block = [c for c in block if c != late_ci]
                    # 只搬比锚早的子句; 比锚晚的(位置可能正确)保持
                    moved_block = [
                        c for c in moved_block
                        if _clause_summary(res_map.get(c, []))["st"] < anchor_inf["st"]
                    ]
            if not moved_block:
                continue
            # 扫描代表句选择: 撕裂块用"时间距锚最远的子句"(它是被吸走的),
            # 不能用音素最多的 —— 音素最多者可能恰是位置已正确的后半句
            # (05 轨 ci58 n=20 位置已对, 真正被吸走的是音素少的 ci56)。
            is_tear_block = bool(tear_line)
            head_ci = None
            head_inf = None
            if is_tear_block:
                for ci in moved_block:
                    inf = _clause_summary(res_map.get(ci, []))
                    if inf["skipped"] or inf["n"] < 2 or len(alignment_tokens[ci]) < 3:
                        continue
                    if head_inf is None or inf["st"] < head_inf["st"]:
                        head_ci = ci
                        head_inf = inf
            else:
                for ci in moved_block:
                    inf = _clause_summary(res_map.get(ci, []))
                    if inf["skipped"] or inf["n"] < 2:
                        continue
                    if head_inf is None or inf["n"] > head_inf["n"]:
                        head_ci = ci
                        head_inf = inf
            if head_ci is None:
                continue
            ids = _clause_token_ids(head_ci, alignment_tokens, acoustic_ctx)
            if len(ids) < 4:
                # 短句音素太少, 峰值扫描不可靠
                continue

            # 窗口: 锚句结束 ~ 块后第一个未跳过句开始(各留 pad)
            anchor_pos = semantic.index(anchor_ci)
            next_anchor_st = total_dur
            tail_ci = block[-1]
            tail_pos = semantic.index(tail_ci)
            for fwd in range(tail_pos + 1, len(semantic)):
                ninf = _clause_summary(res_map.get(semantic[fwd], []))
                if not ninf["skipped"]:
                    next_anchor_st = ninf["st"]
                    break
            lo = max(0.0, anchor_inf["st"] - 2.0)
            hi = min(total_dur, next_anchor_st + scan_pad_sec)
            if hi - lo < min_move_sec * 2:
                continue

            cur_start = head_inf["st"]
            cand = _slide_window_peak(ids, em_np, lo, hi)
            if cand is None:
                continue
            best_score, best_t = cand
            # 当前位置附近窗口峰(±5s)作对照基准, 而非直接用链上得分
            at_cur = _slide_window_peak(ids, em_np, max(0.0, cur_start - 5.0), cur_start + 5.0)
            cur_score = at_cur[0] if at_cur else -1.0e30

            move = best_t - cur_start
            gain = best_score - cur_score
            if move < min_move_sec or gain < min_gain:
                # 只接受显著的正向声学峰: 反向位移说明当前位置已对或峰值
                # 落在另一段相似语音 (05 轨描ける人即此类); 增益不足则
                # 说明声学证据弱。两种情况都保持原位, 不做结构猜测。
                continue
            if gain < min_distinct_gain and move < 25.0:
                # 小位移要求更显著峰值, 防止高频元音句在喘息密集区
                # 形成弱伪峰。
                continue
            log_reason = f"峰值增益 {gain:+.2f}"

            shift = best_t - head_inf["st"]
            for ci in moved_block:
                cl_res = res_map.get(ci)
                if not cl_res:
                    continue
                for r in cl_res:
                    if r.get("original_start") is not None:
                        r["original_start"] = float(r["original_start"]) + shift
                    if r.get("original_end") is not None:
                        r["original_end"] = float(r["original_end"]) + shift
                    if r.get("start") is not None:
                        s = _parse_lrc_ts_sec(r.get("start"))
                        if s is not None:
                            r["start"] = _format_time(s + shift)
                    if r.get("end") is not None:
                        e = _parse_lrc_ts_sec(r.get("end"))
                        if e is not None:
                            r["end"] = _format_time(e + shift)
            moved_total += len(moved_block)
            log(
                f"  🧭 [重叠吸附修复] 〖{spk}〗块 {len(moved_block)} 句 "
                f"「{head_inf['st']:.2f}s → {best_t:.2f}s」位移 {shift:+.2f}s "
                f"({log_reason})"
            )

    if moved_total:
        log(f"  🧭 [重叠吸附修复] 共搬迁 {moved_total} 句")


def _is_anomalous_semantic_clause(
    ci: int,
    by_clause: List[List[Dict[str, Any]]],
    ono_clause_flags: Sequence[int],
) -> bool:
    if ci < len(ono_clause_flags) and ono_clause_flags[ci]:
        return False
    info = _clause_summary(by_clause[ci])
    if info["n"] <= 0:
        return False
    if info["score"] < -1.85:
        return True
    if info["n"] >= 3 and info["min_score"] < -3.2 and info["score"] < -1.45:
        return True
    if info["n"] >= 2 and info["dur"] < max(0.09, 0.048 * info["n"]):
        return True
    return False


def _semantic_quality_objective(
    clause_indices: Iterable[int],
    by_clause: Dict[int, List[Dict[str, Any]]] | List[List[Dict[str, Any]]],
    ono_clause_flags: Sequence[int],
) -> float:
    total = 0.0
    for ci in clause_indices:
        if ci < len(ono_clause_flags) and ono_clause_flags[ci]:
            continue
        cl_res = by_clause[ci]
        info = _clause_summary(cl_res)
        if info["n"] <= 0:
            continue
        tok_sum = sum(float(r.get("score", -5.0)) for r in cl_res)
        min_need = 0.055 * info["n"]
        squeeze_pen = 0.0
        if info["dur"] < min_need:
            squeeze_pen = (min_need - info["dur"]) * 18.0 * info["n"]
        total += (tok_sum - squeeze_pen)
    return total


def _rescue_out_of_order_clauses(
    chain_indices: List[int],
    current_map: Dict[int, List[Dict[str, Any]]],
    alignment_tokens: List[List[str]],
    line_group: Sequence[int],
    ono_clause_flags: Sequence[int],
    enable_silence_snap: bool,
    acoustic_ctx: Dict[str, Any],
    base_emission_2d: torch.Tensor,
    enable_ono_gate: bool,
    log: Callable[[str], None],
    extra_claims: Optional[List[Tuple[float, float, float]]] = None,
    enable_word_skip: bool = False,
    word_skip_token_penalty: float = 4.8,
    clause_surfaces: Optional[Sequence[str]] = None,
    clause_speakers: Optional[Sequence[Any]] = None,
) -> Tuple[Dict[int, List[Dict[str, Any]]], List[List[int]]]:
    if len(chain_indices) < 2:
        return current_map, [chain_indices]

    by_clause_proxy = [current_map.get(i, []) for i in range(len(alignment_tokens))]
    anomalous_set: Set[int] = set()
    for ci in chain_indices:
        if _is_anomalous_semantic_clause(ci, by_clause_proxy, ono_clause_flags):
            anomalous_set.add(ci)

    if not anomalous_set:
        return current_map, [chain_indices]

    expanded_anom: Set[int] = set(anomalous_set)
    anom_groups = {line_group[ci] for ci in anomalous_set}
    for ci in chain_indices:
        if line_group[ci] in anom_groups:
            expanded_anom.add(ci)

    main_chain = [ci for ci in chain_indices if ci not in expanded_anom]
    detached_chain = [ci for ci in chain_indices if ci in expanded_anom]

    if not main_chain or not detached_chain:
        return current_map, [chain_indices]

    base_obj = _semantic_quality_objective(chain_indices, current_map, ono_clause_flags)

    # 多人分链时：其他说话人已占用的时间段在本链重解时同样被压低，
    # 防止重排步骤把子句放回别人的声音上（互斥约束不能被重排覆盖）。
    main_em: Any = base_emission_2d
    main_filler: Optional[np.ndarray] = None
    if extra_claims:
        main_em, main_filler = _build_claimed_emission(
            base_emission_2d=base_emission_2d,
            acoustic_ctx=acoustic_ctx,
            claimed_spans=list(extra_claims),
            base_filler_margin=1.35,
        )

    main_res_map = _align_subset_clauses(
        sub_indices=main_chain,
        alignment_tokens=alignment_tokens,
        line_group=line_group,
        ono_clause_flags=ono_clause_flags,
        enable_silence_snap=enable_silence_snap,
        acoustic_ctx=acoustic_ctx,
        custom_emission_2d=main_em,
        enable_inter_line_filler=True,
        enable_ono_skip=enable_ono_gate,
        filler_frame_logp=main_filler,
        enable_word_skip=enable_word_skip,
        word_skip_token_penalty=word_skip_token_penalty,
        clause_surfaces=clause_surfaces,
        clause_speakers=clause_speakers,
        log_callback=log,
    )

    claimed_spans: List[Tuple[float, float, float]] = list(extra_claims) if extra_claims else []
    valid_times: List[Tuple[float, float]] = []
    for ci in main_chain:
        info = _clause_summary(main_res_map[ci])
        if info["skipped"]:
            continue
        valid_times.append((info["st"], info["en"]))
        is_ono = (ci < len(ono_clause_flags) and ono_clause_flags[ci])
        if not is_ono and info["score"] > -1.35 and info["dur"] >= 0.12:
            claimed_spans.append((info["st"], info["en"], 1.65))

    total_dur = float(acoustic_ctx["emission_L"].shape[1]) * FRAME_SEC
    if valid_times:
        pos_map = {ci: k for k, ci in enumerate(chain_indices)}
        min_pos = min(pos_map[ci] for ci in detached_chain)
        max_pos = max(pos_map[ci] for ci in detached_chain)
        prev_main = [chain_indices[k] for k in range(min_pos - 1, -1, -1) if chain_indices[k] in main_res_map]
        next_main = [chain_indices[k] for k in range(max_pos + 1, len(chain_indices)) if chain_indices[k] in main_res_map]
        anchor_lo = _clause_summary(main_res_map[prev_main[0]])["st"] if prev_main else 0.0
        anchor_hi = _clause_summary(main_res_map[next_main[0]])["en"] if next_main else total_dur
        win_lo = max(0.0, min(anchor_lo, anchor_hi) - 75.0)
        win_hi = min(total_dur, max(anchor_lo, anchor_hi) + 75.0)
        win_bounds = (win_lo, win_hi)
    else:
        win_bounds = None

    det_em_np, det_filler_logp = _build_claimed_emission(
        base_emission_2d=base_emission_2d,
        acoustic_ctx=acoustic_ctx,
        claimed_spans=claimed_spans,
        window_bounds=win_bounds,
        base_filler_margin=1.25,
    )

    det_res_map = _align_subset_clauses(
        sub_indices=detached_chain,
        alignment_tokens=alignment_tokens,
        line_group=line_group,
        ono_clause_flags=ono_clause_flags,
        enable_silence_snap=enable_silence_snap,
        acoustic_ctx=acoustic_ctx,
        custom_emission_2d=det_em_np,
        enable_inter_line_filler=True,
        enable_ono_skip=enable_ono_gate,
        filler_frame_logp=det_filler_logp,
        enable_word_skip=enable_word_skip,
        word_skip_token_penalty=word_skip_token_penalty,
        clause_surfaces=clause_surfaces,
        clause_speakers=clause_speakers,
        log_callback=log,
    )

    cand_map = dict(current_map)
    cand_map.update(main_res_map)
    cand_map.update(det_res_map)
    cand_obj = _semantic_quality_objective(chain_indices, cand_map, ono_clause_flags)

    n_anom_toks = max(1, sum(len(alignment_tokens[ci]) for ci in anomalous_set))
    gain_per_tok = (cand_obj - base_obj) / float(n_anom_toks)

    if gain_per_tok >= 0.30:
        log(
            f"  ✨ [乱序重排] 拆出 {len(detached_chain)} 个受挤压子句独立重对齐 "
            f"(声学增益 +{gain_per_tok:.2f}/tok)"
        )
        return cand_map, [main_chain, detached_chain]

    return current_map, [chain_indices]


def _refine_solo_collisions(
    spk_chains: Dict[str, List[int]],
    per_spk_results: Dict[str, Dict[int, List[Dict[str, Any]]]],
    alignment_tokens: List[List[str]],
    line_group: Sequence[int],
    ono_clause_flags: Sequence[int],
    acoustic_ctx: Dict[str, Any],
    base_emission_2d: torch.Tensor,
    solo_checker: _AudioSoloChecker,
    is_solo_cached: Callable[[float, float], bool],
    log: Callable[[str], None],
    clause_surfaces: Optional[Sequence[str]] = None,
    clause_speakers: Optional[Sequence[Any]] = None,
) -> None:
    """在互斥迭代后，对落入他方单声源独占区间的弱语义/短被吞句做有界局部重对齐与原始概率验收。"""
    active_spks = [s for s, cl in spk_chains.items() if cl]
    if len(active_spks) < 2:
        return

    coop_np = base_emission_2d.detach().cpu().numpy()
    em_L_np = acoustic_ctx["emission_L"][0].detach().cpu().numpy()
    em_R_np = acoustic_ctx["emission_R"][0].detach().cpu().numpy()
    tok_dict = acoustic_ctx["tokenizer"].dictionary
    blank_id = int(acoustic_ctx.get("blank", 0))
    ignored = set(acoustic_ctx.get("ignored_classes", ()))
    valid_cls = [c for c in range(coop_np.shape[1]) if c != blank_id and c not in ignored]
    T_all = int(coop_np.shape[0])
    max_nb = np.max(coop_np[:, valid_cls], axis=1).astype(np.float32)

    def _merge_spans(sps: Sequence[Tuple[float, float]], gap: float) -> List[Tuple[float, float]]:
        res: List[Tuple[float, float]] = []
        for a, b in sorted(sps):
            if res and a <= res[-1][1] + gap:
                res[-1] = (res[-1][0], max(res[-1][1], b))
            else:
                res.append((a, b))
        return res

    repaired_total = 0
    for spk in active_spks:
        own_set = set(spk_chains[spk])
        raw_hits: List[Tuple[float, float]] = []
        for other_spk in active_spks:
            if other_spk == spk:
                continue
            for oci, cl_res in per_spk_results[other_spk].items():
                if oci in own_set or (oci < len(ono_clause_flags) and ono_clause_flags[oci]):
                    continue
                oinf = _clause_summary(cl_res)
                if not oinf["skipped"] and oinf["dur"] > 0.05 and oinf["score"] >= -2.30:
                    raw_hits.append((float(oinf["st"]), float(oinf["en"])))
        if not raw_hits:
            continue

        solo_spans = [(st_a, en_a) for st_a, en_a in raw_hits if is_solo_cached(st_a, en_a)]
        simul_spans = [(st_a, en_a) for st_a, en_a in raw_hits if not is_solo_cached(st_a, en_a)]
        m_solo = _merge_spans(solo_spans, 2.0)
        m_simul = _merge_spans(simul_spans, 1.2)
        if not m_solo:
            continue
        all_claims = m_solo + m_simul

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

        c_list = spk_chains[spk]
        sem_pos = [
            idx_c for idx_c, ci in enumerate(c_list)
            if not (ci < len(ono_clause_flags) and ono_clause_flags[ci])
            and not _clause_summary(per_spk_results[spk][ci])["skipped"]
        ]
        if not sem_pos:
            continue

        is_col: List[bool] = []
        is_short_col: List[bool] = []
        for p_idx in sem_pos:
            ci_p = c_list[p_idx]
            inf = _clause_summary(per_spk_results[spk][ci_p])
            n_phones = sum(len(t) for t in alignment_tokens[ci_p])
            ov_solo = any(min(ce, inf["en"]) - max(cs, inf["st"]) >= 0.25 for cs, ce in m_solo)
            short_wrapped_solo = (
                n_phones <= 6 and float(inf["dur"]) <= 0.65
                and any(
                    (min(ce, inf["en"]) - max(cs, inf["st"])) >= 0.60 * max(0.05, float(inf["dur"]))
                    and (ce - cs) >= 1.5 * max(0.05, float(inf["dur"]))
                    for cs, ce in solo_spans
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
                and all(
                    clause_speakers and idx < len(clause_speakers) and spk in clause_speakers[idx]
                    for idx in range(c_list[sem_pos[k]] + 1, c_list[sem_pos[k + 1]])
                )
            ):
                k += 1
            k1 = k
            k += 1

            left_idx = sem_pos[k0 - 1] if k0 > 0 else -1
            right_idx = sem_pos[k1 + 1] if k1 + 1 < len(sem_pos) else len(c_list)
            w_lo = float(_clause_summary(per_spk_results[spk][c_list[left_idx]])["en"]) if left_idx >= 0 else 0.0
            w_hi = float(_clause_summary(per_spk_results[spk][c_list[right_idx]])["st"]) if right_idx < len(c_list) else float(T_all * FRAME_SEC)
            f_lo = max(0, int(np.ceil(w_lo / FRAME_SEC)))
            f_hi = min(T_all, int(np.floor(w_hi / FRAME_SEC)))
            if f_hi <= f_lo + 2:
                continue

            sub_sem = [c_list[p_idx] for p_idx in sem_pos[k0 : k1 + 1]]
            em_win = em_mod.clone()
            if f_lo > 0:
                em_win[:f_lo, valid_cls] = -15.0
                em_win[:f_lo, blank_id] = 0.0
            if f_hi < T_all:
                em_win[f_hi:, valid_cls] = -15.0
                em_win[f_hi:, blank_id] = 0.0

            realigned_blk = _align_subset_clauses(
                sub_indices=sub_sem,
                alignment_tokens=alignment_tokens,
                line_group=line_group,
                ono_clause_flags=ono_clause_flags,
                enable_silence_snap=True,
                acoustic_ctx=acoustic_ctx,
                custom_emission_2d=em_win,
                enable_inter_line_filler=True,
                filler_frame_logp=filler_logp,
                enable_ono_skip=False,
                ono_skip_token_penalty=0.5,
                enable_word_skip=False,
                clause_surfaces=clause_surfaces,
                clause_speakers=clause_speakers,
            )
            old_scs = [_unmasked_clause_mean_score(per_spk_results[spk][ci], coop_np, tok_dict) for ci in sub_sem]
            old_solo_ov = sum(
                max(0.0, min(ce, float(_clause_summary(per_spk_results[spk][ci])["en"])) - max(cs, float(_clause_summary(per_spk_results[spk][ci])["st"])))
                for ci in sub_sem for cs, ce in m_solo
            )
            new_scs: List[float] = []
            new_solo_ov = 0.0
            valid_blk = True
            for ci in sub_sem:
                if ci not in realigned_blk or _clause_summary(realigned_blk[ci])["skipped"]:
                    valid_blk = False
                    break
                new_scs.append(_unmasked_clause_mean_score(realigned_blk[ci], coop_np, tok_dict))
                n_inf = _clause_summary(realigned_blk[ci])
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
                per_spk_results[spk][ci] = realigned_blk[ci]
            repaired_total += len(sub_sem)

    if repaired_total:
        log(f"  🛡 [单声源局部碰撞修复] 共修正 {repaired_total} 句误撞弱句")


    if repaired_total:
        log(f"  🛡 [单声源局部碰撞修复] 共修正 {repaired_total} 句误撞弱句")


def align_multispeaker_or_reordered(
    audio_path: Any,
    alignment_tokens: List[List[str]],
    line_group: List[int],
    ono_clause_flags: List[int],
    clause_speakers: List[Tuple[str, ...]],
    lora_path: Optional[str],
    dev: str,
    vram_chunk: int,
    enable_silence_snap: bool,
    enable_ono_gate: bool,
    blank_punct_mask: List[bool],
    log_callback: Optional[Callable[[str], None]] = None,
    diagnostics_out: Optional[Dict[str, Any]] = None,
    acoustic_ctx: Optional[Dict[str, Any]] = None,
    enable_word_skip: bool = False,
    word_skip_token_penalty: float = 4.8,
    clause_surfaces: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[List[int]]]:
    """多人分链与乱序重排主入口（仅在用户开启多人/乱序模式时调用）。"""
    def log(msg: str):
        if log_callback:
            log_callback(msg)

    n_clauses = len(alignment_tokens)
    all_clause_indices = list(range(n_clauses))
    if n_clauses == 0:
        return [], []

    # 保护单音素/单假名口吃子句（如「い、いや」「い、挿れて」「う、嘘」）：
    # 若某短子句与下一非拟声子句同属一行，且其音素是下一子句音素的前缀，则视为实词口吃而非易错拟声
    ono_clause_flags = list(ono_clause_flags)
    for ci_k in range(n_clauses - 1):
        if ci_k < len(ono_clause_flags) and ono_clause_flags[ci_k]:
            toks_cur = alignment_tokens[ci_k]
            toks_nxt = alignment_tokens[ci_k + 1]
            if (1 <= len(toks_cur) <= 2
                    and ci_k + 1 < len(line_group)
                    and line_group[ci_k] == line_group[ci_k + 1]
                    and (ci_k + 1 >= len(ono_clause_flags) or not ono_clause_flags[ci_k + 1])
                    and toks_nxt
                    and "".join(toks_nxt).startswith("".join(toks_cur))):
                ono_clause_flags[ci_k] = 0

    # 保护被顿号/喘息劈成两半的短实词（如「うる、さいっ」「やめ、てっ」）：
    # 前半段是非拟声极短实词(<=2字)、紧邻后缀虽<=3字却被误标 ono=1 时，合并语义判断若为非易错实词则撤销后缀的 ono 标记
    if clause_surfaces:
        from ono_filter import is_easy_ono as _is_easy_ono_fn
        for ci_k in range(n_clauses - 1):
            if (
                ono_clause_flags[ci_k] == 0
                and ci_k + 1 < len(ono_clause_flags)
                and ono_clause_flags[ci_k + 1] == 1
                and ci_k + 1 < len(line_group)
                and line_group[ci_k] == line_group[ci_k + 1]
                and ci_k + 1 < len(clause_surfaces)
                and len(str(clause_surfaces[ci_k]).strip()) <= 2
                and len(str(clause_surfaces[ci_k + 1]).strip()) <= 3
                and not _is_easy_ono_fn(str(clause_surfaces[ci_k]) + str(clause_surfaces[ci_k + 1]))
            ):
                ono_clause_flags[ci_k + 1] = 0

    distinct_speakers: List[str] = []
    for spk_tup in clause_speakers:
        for s in spk_tup:
            if s and s not in distinct_speakers:
                distinct_speakers.append(s)

    # ── 情况 0：仅 1 位说话人（或无角色标记）且未传入预计算联合声学上下文，自动回归标准单向顺序对齐 ──
    if len(distinct_speakers) <= 1 and acoustic_ctx is None and audio_path:
        spk_desc = f"〖{distinct_speakers[0]}〗" if distinct_speakers else "未标注角色"
        log(f"  ℹ [多人模式] 检测到独立说话人 <= 1 位 ({spk_desc})，自动回归标准单向顺序对齐")
        seq_res = align.align_audio_with_text(
            audio_file_path=audio_path,
            text_tokens=alignment_tokens,
            lora_path=lora_path,
            device=dev,
            chunk_frames=int(vram_chunk),
            enable_silence_snap=enable_silence_snap,
            blank_punct_mask=blank_punct_mask,
            line_group=line_group,
            ono_clause_flags=ono_clause_flags,
            log_callback=log_callback,
            diagnostics_out=diagnostics_out,
        )
        return seq_res, [all_clause_indices]

    if acoustic_ctx is None:
        acoustic_ctx = extract_acoustic_context(
            audio_file_path=audio_path,
            lora_path=lora_path,
            device=dev,
            chunk_frames=int(vram_chunk),
        )
    if isinstance(diagnostics_out, dict):
        diagnostics_out["mono_rms_db"] = acoustic_ctx["mono_rms"]
        diagnostics_out["mono_noise_floor_db"] = acoustic_ctx["mono_floor"]
        diagnostics_out["frame_sec"] = float(FRAME_SEC)

    coop_emission = build_cooperative_stereo_emission(acoustic_ctx, ear_bias=0.0)
    if isinstance(diagnostics_out, dict):
        _c_np = coop_emission.detach().cpu().numpy()
        _b_id = int(acoustic_ctx.get("blank", 0))
        _ign = set(acoustic_ctx.get("ignored_classes", ()))
        _v_cls = [c for c in range(_c_np.shape[1]) if c != _b_id and c not in _ign]
        diagnostics_out["ctc_nb_active"] = (
            (np.max(_c_np[:, _v_cls], axis=1) > (_c_np[:, _b_id] - np.float32(0.5)))
            if _v_cls else np.zeros(_c_np.shape[0], dtype=bool)
        )
        diagnostics_out["duration_sec"] = float(_c_np.shape[0] * FRAME_SEC)

    # ── 情况 A：显式多说话人台本 (>= 2 个角色) ──────────────────────────
    if len(distinct_speakers) >= 2:
        log(
            f"  🎭 [多人分链模式] 检测到 {len(distinct_speakers)} 位说话人: "
            + " / ".join(f"〖{s}〗" for s in distinct_speakers)
        )

        spk_chains: Dict[str, List[int]] = {s: [] for s in distinct_speakers}
        first_valid_spk = distinct_speakers[0]
        for ci, spk_tup in enumerate(clause_speakers):
            valid_spks = [s for s in spk_tup if s in spk_chains]
            if not valid_spks:
                spk_chains[first_valid_spk].append(ci)
            else:
                for s in valid_spks:
                    spk_chains[s].append(ci)

        active_speakers = [s for s in distinct_speakers if spk_chains[s]]

        per_spk_results: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}
        for spk in active_speakers:
            c_idx = spk_chains[spk]
            res_map = _align_subset_clauses(
                sub_indices=c_idx,
                alignment_tokens=alignment_tokens,
                line_group=line_group,
                ono_clause_flags=ono_clause_flags,
                enable_silence_snap=enable_silence_snap,
                acoustic_ctx=acoustic_ctx,
                custom_emission_2d=coop_emission,
                enable_inter_line_filler=False,
                inter_line_filler_margin=1.35,
                enable_ono_skip=enable_ono_gate,
                ono_skip_token_penalty=2.6,
                enable_word_skip=enable_word_skip,
                word_skip_token_penalty=word_skip_token_penalty,
                clause_surfaces=clause_surfaces,
                clause_speakers=clause_speakers,
                log_callback=log_callback,
            )
            per_spk_results[spk] = res_map

        ear_biases: Dict[str, float] = {}
        temp_by_clause: List[List[Dict[str, Any]]] = [[] for _ in range(n_clauses)]
        for spk in active_speakers:
            for ci, cl_res in per_spk_results[spk].items():
                temp_by_clause[ci] = cl_res
            ear_biases[spk] = _estimate_speaker_ear_bias(
                spk_chains[spk], temp_by_clause, ono_clause_flags, acoustic_ctx
            )

        em_L_np = acoustic_ctx["emission_L"][0].detach().cpu().numpy()
        em_R_np = acoustic_ctx["emission_R"][0].detach().cpu().numpy()
        stereo_base_np = acoustic_ctx["stereo_emission"][0].detach().cpu().numpy()
        tok_dict = acoustic_ctx["tokenizer"].dictionary
        total_frames_all, n_classes_all = em_L_np.shape
        blank_id = int(acoustic_ctx.get("blank", 0))
        ignored_set = set(acoustic_ctx.get("ignored_classes", ()))
        valid_cls_all = [c for c in range(n_classes_all) if c != blank_id and c not in ignored_set]
        _coop_ref_np = coop_emission.detach().cpu().numpy()
        _ctc_nb_active = (
            np.max(_coop_ref_np[:, valid_cls_all], axis=1) > (_coop_ref_np[:, blank_id] - np.float32(0.5))
            if valid_cls_all else np.zeros(total_frames_all, dtype=bool)
        )

        def _clause_phoneme_lr_diff(ci: int, cl_res: List[Dict[str, Any]]) -> float:
            """用目标音素在左右声道的神经网络发射得分差 (em_L - em_R) 判定声道，完全不受舔耳物理分贝干扰。"""
            if not cl_res or ci >= len(alignment_tokens):
                return 0.0
            raw_toks = alignment_tokens[ci]
            diffs_tok: List[float] = []
            for r_tok, t_str in zip(cl_res, raw_toks):
                if r_tok.get("skipped") or r_tok.get("original_start") is None or r_tok.get("original_end") is None:
                    continue
                t_id = tok_dict.get(t_str)
                if t_id is None:
                    continue
                st_f = max(0, min(total_frames_all - 1, int(round(float(r_tok["original_start"]) / FRAME_SEC))))
                en_f = max(st_f + 1, min(total_frames_all, int(round(float(r_tok["original_end"]) / FRAME_SEC))))
                sc_l = float(np.mean(em_L_np[st_f:en_f, t_id]))
                sc_r = float(np.mean(em_R_np[st_f:en_f, t_id]))
                diffs_tok.append(sc_l - sc_r)
            return float(np.mean(diffs_tok)) if diffs_tok else 0.0

        def _build_channel_claimed_emission(
            ear_bias: float,
            channel_claims: List[Tuple[float, float, float, float]],
            pad_sec: float = 0.05,
        ) -> Tuple[np.ndarray, np.ndarray]:
            """在左右声道上分别施加音素互斥惩罚后再合成协同发射，且保证扣分不低于静音底噪 (-3.5)。"""
            l_mod = em_L_np.copy()
            r_mod = em_R_np.copy()
            st_mod = stereo_base_np.copy()

            base_coop = build_cooperative_stereo_emission(acoustic_ctx, ear_bias=ear_bias).detach().cpu().numpy()
            max_nb = np.max(base_coop[:, valid_cls_all], axis=1).astype(np.float32)
            filler_logp = np.full(total_frames_all, -1.0e30, dtype=np.float32)
            silence_floor = np.float32(-3.50)

            for st_sec, en_sec, pen_l, pen_r in channel_claims:
                st_f = max(0, min(total_frames_all, int(round((st_sec - pad_sec) / FRAME_SEC))))
                en_f = max(st_f, min(total_frames_all, int(round((en_sec + pad_sec) / FRAME_SEC))))
                if en_f <= st_f:
                    continue
                if pen_l > 0.0:
                    cur_l = l_mod[st_f:en_f, valid_cls_all]
                    l_mod[st_f:en_f, valid_cls_all] = np.maximum(
                        cur_l - np.float32(pen_l),
                        np.minimum(cur_l, silence_floor),
                    )
                if pen_r > 0.0:
                    cur_r = r_mod[st_f:en_f, valid_cls_all]
                    r_mod[st_f:en_f, valid_cls_all] = np.maximum(
                        cur_r - np.float32(pen_r),
                        np.minimum(cur_r, silence_floor),
                    )
                pen_st = 0.5 * (pen_l + pen_r)
                if pen_st > 0.0:
                    cur_s = st_mod[st_f:en_f, valid_cls_all]
                    st_mod[st_f:en_f, valid_cls_all] = np.maximum(
                        cur_s - np.float32(pen_st),
                        np.minimum(cur_s, silence_floor),
                    )
                pen_both = min(pen_l, pen_r)
                if pen_both > 0.0:
                    l_mod[st_f:en_f, blank_id] -= np.float32(pen_both)
                    r_mod[st_f:en_f, blank_id] -= np.float32(pen_both)
                    st_mod[st_f:en_f, blank_id] -= np.float32(pen_both)
                orig_b = base_coop[st_f:en_f, blank_id]
                filler_logp[st_f:en_f] = np.maximum(
                    filler_logp[st_f:en_f],
                    np.maximum(orig_b, max_nb[st_f:en_f] - np.float32(0.10)),
                )

            both_max = np.maximum(l_mod, r_mod) - np.float32(0.18)
            coop_np = np.maximum(st_mod, both_max)
            if abs(ear_bias) > 1e-3:
                if ear_bias > 0:
                    w_pref = min(0.85, 0.50 + 0.25 * float(ear_bias))
                    pref, oth = l_mod, r_mod
                else:
                    w_pref = min(0.85, 0.50 + 0.25 * float(-ear_bias))
                    pref, oth = r_mod, l_mod
                biased = np.logaddexp(
                    pref + np.float32(np.log(w_pref)),
                    oth + np.float32(np.log(max(1e-4, 1.0 - w_pref))),
                )
                coop_np = np.maximum(biased, coop_np - np.float32(0.35))
            coop_np[:, blank_id] = st_mod[:, blank_id]
            return coop_np.astype(np.float32), filler_logp

        # ── 互斥迭代解码（支持乱序 + 音素级声道归属 + STFT 单声源并发保护） ───────
        _ref_spans = [
            (float(_inf["st"]), float(_inf["en"]))
            for _spk in active_speakers
            for _ci in spk_chains[_spk]
            if not (_ci < len(ono_clause_flags) and ono_clause_flags[_ci])
            for _inf in (_clause_summary(per_spk_results[_spk].get(_ci, [])),)
            if not _inf["skipped"] and _inf["score"] >= -2.05
        ]
        solo_checker = _AudioSoloChecker(acoustic_ctx.get("_stereo_wav"), _ref_spans)
        _solo_cache: Dict[Tuple[float, float], bool] = {}

        def _is_solo_cached(st_s: float, en_s: float) -> bool:
            key = (round(st_s, 2), round(en_s, 2))
            if key not in _solo_cache:
                _solo_cache[key] = solo_checker.is_clearly_solo(st_s, en_s)
            return _solo_cache[key]
        def _chain_mean_score(spk_name: str) -> float:
            tot, wt = 0.0, 0.0
            for ci, cl_res in per_spk_results[spk_name].items():
                if ci < len(ono_clause_flags) and ono_clause_flags[ci]:
                    continue
                inf = _clause_summary(cl_res)
                if inf["skipped"] or inf["n"] <= 0:
                    continue
                tot += inf["score"] * inf["n"]
                wt += inf["n"]
            return tot / wt if wt > 0 else -10.0

        def _clause_mass(inf: Dict[str, Any]) -> float:
            if inf["skipped"] or inf["n"] <= 0:
                return 0.0
            return float(inf["n"]) * max(0.15, float(inf["score"]) + 2.5)

        def _is_valid_span(inf: Dict[str, Any]) -> bool:
            if inf["skipped"] or inf["n"] <= 0:
                return False
            return 0.08 <= inf["dur"] <= max(4.5, 0.75 * float(inf["n"]))

        def _claims_from_others(spk_name: str) -> List[Tuple[float, float, float, float]]:
            own = set(spk_chains[spk_name])
            own_spans: List[Tuple[float, float, float, float]] = []
            for ci_own, cl_own in per_spk_results[spk_name].items():
                if ci_own < len(ono_clause_flags) and ono_clause_flags[ci_own]:
                    continue
                inf_o = _clause_summary(cl_own)
                if not _is_valid_span(inf_o):
                    continue
                lr_o = _clause_phoneme_lr_diff(ci_own, cl_own)
                own_spans.append((inf_o["st"], inf_o["en"], _clause_mass(inf_o), lr_o))

            out_claims: List[Tuple[float, float, float, float]] = []
            for other in active_speakers:
                if other == spk_name:
                    continue
                other_raw: List[Tuple[float, float, float, float]] = []
                for ci, cl_res in per_spk_results[other].items():
                    if ci in own:
                        continue
                    is_ono = (ci < len(ono_clause_flags) and ono_clause_flags[ci])
                    inf = _clause_summary(cl_res)
                    if not _is_valid_span(inf) or inf["dur"] < 0.10:
                        continue
                    if is_ono and inf["score"] <= -1.80:
                        continue
                    m_other = _clause_mass(inf)
                    lr_diff = _clause_phoneme_lr_diff(ci, cl_res)
                    beaten_by_own = False
                    for ost, oen, m_own, lr_own in own_spans:
                        ov = min(inf["en"], oen) - max(inf["st"], ost)
                        if ov > 0.04 and m_own > m_other * 1.15:
                            # 仅当本方与对方在同一声道竞争且本方质量更高时豁免
                            if lr_diff * lr_own >= -0.25:
                                beaten_by_own = True
                                break
                    if beaten_by_own:
                        continue
                    def _pen_for_solo(solo_flag: bool) -> Tuple[float, float]:
                        if solo_flag:
                            if lr_diff > 0.55:
                                return 1.85, 0.0
                            if lr_diff < -0.55:
                                return 0.0, 1.85
                            return 1.65, 1.65
                        # 真·多人同时叠声区间：仅轻度抑制单耳偏向，居中叠声完全不抹除音素发射（只抬高 filler 底线）
                        if lr_diff > 0.55:
                            return 1.10, 0.0
                        if lr_diff < -0.55:
                            return 0.0, 1.10
                        return 0.0, 0.0

                    if inf["score"] > -2.20 or (not is_ono and inf["n"] >= 3 and inf["score"] > -2.60):
                        # 长句按子窗拆分单声源段与并发段，避免局部抢话导致整句独白失去保护
                        if inf["dur"] >= 1.50 and not _is_solo_cached(inf["st"], inf["en"]):
                            for sub_st, sub_en, sub_solo in solo_checker.split_solo_subspans(inf["st"], inf["en"]):
                                pl_s, pr_s = _pen_for_solo(sub_solo)
                                other_raw.append((sub_st, sub_en, pl_s, pr_s))
                        else:
                            pl_s, pr_s = _pen_for_solo(_is_solo_cached(inf["st"], inf["en"]))
                            other_raw.append((inf["st"], inf["en"], pl_s, pr_s))
                    elif not is_ono:
                        pl_s, pr_s = _pen_for_solo(_is_solo_cached(inf["st"], inf["en"]))
                        for r in cl_res:
                            if (not r.get("skipped")
                                    and float(r.get("score", -10.0)) > -2.0
                                    and r.get("original_start") is not None
                                    and r.get("original_end") is not None):
                                other_raw.append((float(r["original_start"]), float(r["original_end"]), pl_s, pr_s))
                other_raw.sort(key=lambda x: (x[0], x[1]))
                bridged: List[Tuple[float, float, float, float]] = []
                for st_c, en_c, pl_c, pr_c in other_raw:
                    if bridged and (st_c - bridged[-1][1]) <= 2.20:
                        gap_lo, gap_hi = bridged[-1][1], st_c
                        same_ch = (bridged[-1][2] > 0 and pl_c > 0) or (bridged[-1][3] > 0 and pr_c > 0)
                        has_own_in_gap = any(
                            min(gap_hi, oen) - max(gap_lo, ost) > 0.04
                            for ost, oen, _, _ in own_spans
                        )
                        # CTC 音素活跃守卫：仅当缝隙内确实检出非 blank 音素尖峰（极可能是对方短插话/接龙）时才阻止跨缝桥接，
                        # 避免在含混响/底噪的普通换气停顿处被低阈值 speech_frame_mask 误断
                        has_active_speech_in_gap = False
                        if (gap_hi - gap_lo) >= 0.35:
                            gf0 = max(0, int(round((gap_lo + 0.06) / FRAME_SEC)))
                            gf1 = min(len(_ctc_nb_active), int(round((gap_hi - 0.06) / FRAME_SEC)))
                            if gf1 > gf0 and int(np.sum(_ctc_nb_active[gf0:gf1])) >= 2:
                                has_active_speech_in_gap = True
                        if same_ch and not has_own_in_gap and not has_active_speech_in_gap:
                            bridged[-1] = (
                                bridged[-1][0],
                                max(bridged[-1][1], en_c),
                                max(bridged[-1][2], pl_c),
                                max(bridged[-1][3], pr_c),
                            )
                            continue
                    bridged.append((st_c, en_c, pl_c, pr_c))
                out_claims.extend(bridged)
            return out_claims

        MAX_EXCLUSION_ROUNDS = 3
        prev_sig = None
        for rnd in range(MAX_EXCLUSION_ROUNDS):
            for spk in sorted(active_speakers, key=_chain_mean_score):
                bias = ear_biases[spk]
                claims = _claims_from_others(spk)
                if abs(bias) <= 0.25 and not claims:
                    continue
                if claims:
                    em_np, filler_logp = _build_channel_claimed_emission(
                        ear_bias=bias,
                        channel_claims=claims,
                    )
                else:
                    em_np = build_cooperative_stereo_emission(acoustic_ctx, ear_bias=bias)
                    filler_logp = None
                per_spk_results[spk] = _align_subset_clauses(
                    sub_indices=spk_chains[spk],
                    alignment_tokens=alignment_tokens,
                    line_group=line_group,
                    ono_clause_flags=ono_clause_flags,
                    enable_silence_snap=enable_silence_snap,
                    acoustic_ctx=acoustic_ctx,
                    custom_emission_2d=em_np,
                    enable_inter_line_filler=bool(claims),
                    enable_ono_skip=enable_ono_gate,
                    filler_frame_logp=filler_logp,
                    enable_word_skip=enable_word_skip,
                    word_skip_token_penalty=word_skip_token_penalty,
                    clause_surfaces=clause_surfaces,
                    clause_speakers=clause_speakers,
                    log_callback=log_callback,
                )
            sig = tuple(
                (s, tuple(sorted((round(a, 1), round(b, 1)) for a, b, _, _ in _claims_from_others(s))))
                for s in active_speakers
            )
            if sig == prev_sig:
                log(f"  🔁 [互斥迭代] 第 {rnd + 1} 轮占用结果不再变化，收敛")
                break
            prev_sig = sig
        else:
            log(f"  🔁 [互斥迭代] 已执行 {MAX_EXCLUSION_ROUNDS} 轮")

        _refine_solo_collisions(
            spk_chains=spk_chains,
            per_spk_results=per_spk_results,
            alignment_tokens=alignment_tokens,
            line_group=line_group,
            ono_clause_flags=ono_clause_flags,
            acoustic_ctx=acoustic_ctx,
            base_emission_2d=coop_emission,
            solo_checker=solo_checker,
            is_solo_cached=_is_solo_cached,
            log=log,
            clause_surfaces=clause_surfaces,
            clause_speakers=clause_speakers,
        )

        _repair_tail_gap_clauses(
            spk_chains=spk_chains,
            per_spk_results=per_spk_results,
            alignment_tokens=alignment_tokens,
            ono_clause_flags=ono_clause_flags,
            acoustic_ctx=acoustic_ctx,
            base_emission_2d=coop_emission,
            log=log,
            enable_silence_snap=enable_silence_snap,
            enable_ono_gate=enable_ono_gate,
            enable_word_skip=enable_word_skip,
            word_skip_token_penalty=word_skip_token_penalty,
            clause_surfaces=clause_surfaces,
            clause_speakers=clause_speakers,
            line_group=line_group,
        )

        _repair_overlapping_clause_blocks(
            spk_chains=spk_chains,
            per_spk_results=per_spk_results,
            alignment_tokens=alignment_tokens,
            line_group=line_group,
            ono_clause_flags=ono_clause_flags,
            acoustic_ctx=acoustic_ctx,
            base_emission_2d=coop_emission,
            log=log,
            clause_surfaces=clause_surfaces,
        )

        # ── 零依赖 MMS-FA 浅层声纹原型裁决 + 局部越序短句校正 ──
        vp_judge = _SpeakerVoiceprintJudge(
            acoustic_ctx=acoustic_ctx,
            spk_chains=spk_chains,
            per_spk_results=per_spk_results,
            ono_clause_flags=ono_clause_flags,
            is_solo_cached=_is_solo_cached,
            log=log,
            clause_speakers=clause_speakers,
        )
        _resolve_same_text_swaps(
            spk_chains=spk_chains,
            per_spk_results=per_spk_results,
            alignment_tokens=alignment_tokens,
            ono_clause_flags=ono_clause_flags,
            vp_judge=vp_judge,
            log=log,
            clause_speakers=clause_speakers,
        )
        _rescue_out_of_order_short_clauses(
            spk_chains=spk_chains,
            per_spk_results=per_spk_results,
            alignment_tokens=alignment_tokens,
            line_group=line_group,
            ono_clause_flags=ono_clause_flags,
            acoustic_ctx=acoustic_ctx,
            base_emission_2d=coop_emission,
            vp_judge=vp_judge,
            log=log,
            clause_surfaces=clause_surfaces,
            clause_speakers=clause_speakers,
        )

        merged_clause_map: Dict[int, List[Dict[str, Any]]] = {}
        final_streams: List[List[int]] = []
        assigned_clauses: Set[int] = set()

        for spk in active_speakers:
            spk_map = per_spk_results[spk]
            for ci, cl_res in spk_map.items():
                if ci not in merged_clause_map:
                    merged_clause_map[ci] = cl_res
                else:
                    old_inf = _clause_summary(merged_clause_map[ci])
                    new_inf = _clause_summary(cl_res)
                    if new_inf["score"] > old_inf["score"]:
                        merged_clause_map[ci] = cl_res

            dedup_stream = [ci for ci in spk_chains[spk] if ci not in assigned_clauses]
            if dedup_stream:
                final_streams.append(dedup_stream)
                assigned_clauses.update(dedup_stream)

        n_skipped = sum(
            1
            for ci in range(n_clauses)
            if ci in merged_clause_map and _clause_summary(merged_clause_map[ci])["skipped"]
        )
        if n_skipped:
            log(f"  ✨ [跳过弧] 解码阶段直接跳过 {n_skipped} 句未发音实词/失配拟声")

        flat_out: List[Dict[str, Any]] = []
        for ci in range(n_clauses):
            flat_out.extend(merged_clause_map[ci])
        return flat_out, final_streams

    # ── 情况 B：无显式角色标记的乱序台本（如已剥除角色前缀的多人混音台本） ──
    log("  🔀 [乱序重排模式] 正在执行双耳协同 + 拟声跳过弧 + 乱序拆流检测...")
    pass1_map = _align_subset_clauses(
        sub_indices=all_clause_indices,
        alignment_tokens=alignment_tokens,
        line_group=line_group,
        ono_clause_flags=ono_clause_flags,
        enable_silence_snap=enable_silence_snap,
        acoustic_ctx=acoustic_ctx,
        custom_emission_2d=coop_emission,
        enable_inter_line_filler=True,
        inter_line_filler_margin=1.35,
        enable_ono_skip=enable_ono_gate,
        ono_skip_token_penalty=2.6,
        enable_word_skip=enable_word_skip,
        word_skip_token_penalty=word_skip_token_penalty,
        clause_surfaces=clause_surfaces,
        clause_speakers=clause_speakers,
        log_callback=log,
    )

    rescued_map, streams = _rescue_out_of_order_clauses(
        chain_indices=all_clause_indices,
        current_map=pass1_map,
        alignment_tokens=alignment_tokens,
        line_group=line_group,
        ono_clause_flags=ono_clause_flags,
        enable_silence_snap=enable_silence_snap,
        acoustic_ctx=acoustic_ctx,
        base_emission_2d=coop_emission,
        enable_ono_gate=enable_ono_gate,
        log=log,
        enable_word_skip=enable_word_skip,
        word_skip_token_penalty=word_skip_token_penalty,
        clause_surfaces=clause_surfaces,
        clause_speakers=clause_speakers,
    )

    n_skipped = sum(
        1 for ci in all_clause_indices if _clause_summary(rescued_map[ci])["skipped"]
    )
    if n_skipped:
        log(f"  ✨ [跳过弧] 解码阶段直接跳过 {n_skipped} 句未发音实词/失配拟声")

    flat_out = []
    for ci in all_clause_indices:
        flat_out.extend(rescued_map[ci])
    return flat_out, streams


# ══════════════════════════════════════════════════════════════════════════
#  4. 分流独立后处理与时间轴归并
# ══════════════════════════════════════════════════════════════════════════

def _parse_lrc_ts_sec(ts_str: Optional[str]) -> Optional[float]:
    if not ts_str or ts_str == "[error]":
        return None
    m = re.match(r"\[(\d+):(\d+)[.:](\d+)\]", str(ts_str))
    if m:
        return int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 100.0
    try:
        return float(ts_str)
    except (TypeError, ValueError):
        return None


def apply_skipped_ono_marks(
    result_list: List[Dict[str, Any]],
    alignment_results: Sequence[Dict[str, Any]],
    token_to_index_map: Dict[int, int],
) -> int:
    """将被 Viterbi 跳过弧 (skipped=True) 跳过的易错拟声子句在 result_list 中标为 ONO_DROPPED；
    但对于同台本行内严格夹在两段已对齐存活内容之间的“句中夹心 ono”（且间距 <= 4.5s），
    保留其显示并以内插边界时间护送，防止整句话被挖空。
    """
    skipped_ri_set: Set[int] = set()
    for fi, r in enumerate(alignment_results):
        if r.get("skipped"):
            ri = token_to_index_map.get(fi)
            if ri is not None and 0 <= ri < len(result_list):
                skipped_ri_set.add(ri)

    if not skipped_ri_set:
        return 0

    preserved_ri_set: Set[int] = set()
    n = len(result_list)
    ln_start = 0
    for idx_i in range(n + 1):
        is_nl = (idx_i == n) or (result_list[idx_i].get("orig") == "\n")
        if not is_nl:
            continue
        ln_end = idx_i
        if ln_end > ln_start:
            timed_alive = [
                k for k in range(ln_start, ln_end)
                if k not in skipped_ri_set
                and result_list[k].get("pron")
                and not result_list[k].get(F.ONO_DROPPED)
                and _parse_lrc_ts_sec(result_list[k].get("start")) is not None
                and _parse_lrc_ts_sec(result_list[k].get("end")) is not None
            ]
            if len(timed_alive) >= 2:
                first_alive, last_alive = timed_alive[0], timed_alive[-1]
                k = first_alive + 1
                while k < last_alive:
                    if k not in skipped_ri_set:
                        k += 1
                        continue
                    # 收集同处一片连续夹心区域内的所有 skipped token（允许中间夹杂标点/注音占位符）
                    run_skips: List[int] = [k]
                    j = k + 1
                    while j < last_alive:
                        if j in skipped_ri_set:
                            run_skips.append(j)
                            j += 1
                        elif not result_list[j].get("pron"):
                            j += 1
                        else:
                            break
                    l_anchor = max((a for a in timed_alive if a < run_skips[0]), default=None)
                    r_anchor = min((a for a in timed_alive if a > run_skips[-1]), default=None)
                    if l_anchor is not None and r_anchor is not None:
                        spk_l = result_list[l_anchor].get(F.SPK)
                        spk_r = result_list[r_anchor].get(F.SPK)
                        same_spk = (spk_l == spk_r) or (not spk_l) or (not spk_r)
                        en_l = _parse_lrc_ts_sec(result_list[l_anchor].get("end"))
                        st_r = _parse_lrc_ts_sec(result_list[r_anchor].get("start"))
                        if same_spk and en_l is not None and st_r is not None and -0.25 <= (st_r - en_l) <= 4.50:
                            span_lo = min(en_l, st_r)
                            span_hi = max(span_lo + 0.04, st_r)
                            m_cnt = len(run_skips)
                            step_t = (span_hi - span_lo) / float(m_cnt)
                            for pos_s, s_ri in enumerate(run_skips):
                                t_s = span_lo + pos_s * step_t
                                t_e = span_lo + (pos_s + 1) * step_t
                                result_list[s_ri]["start"] = _format_time(t_s)
                                result_list[s_ri]["end"] = _format_time(t_e)
                                preserved_ri_set.add(s_ri)
                    k = j
        ln_start = idx_i + 1

    dropped = 0
    dropped_indices: List[int] = []
    for ri in sorted(skipped_ri_set):
        if ri in preserved_ri_set:
            continue
        it = result_list[ri]
        if "start" in it and F.ONO_SAVED not in it:
            it[F.ONO_SAVED] = (it.get("start"), it.get("end"))
        it.pop("start", None)
        it.pop("end", None)
        if not it.get(F.ONO_DROPPED):
            it[F.ONO_DROPPED] = True
            dropped += 1
            dropped_indices.append(ri)

    if not dropped_indices:
        return 0

    def _touches_only_dropped_or_boundary(idx: int) -> bool:
        for d in (-1, +1):
            j = idx + d
            while 0 <= j < len(result_list):
                if result_list[j].get("orig") == "\n":
                    break
                if result_list[j].get("pron"):
                    if not result_list[j].get(F.ONO_DROPPED):
                        return False
                    break
                j += d
        return True

    for ri in dropped_indices:
        for d in (-1, +1):
            j = ri + d
            while 0 <= j < len(result_list):
                if result_list[j].get("orig") == "\n":
                    break
                if not _clause_mod.is_punct_symbol(result_list[j]):
                    break
                if _touches_only_dropped_or_boundary(j):
                    result_list[j][F.ONO_DROPPED] = True
                j += d

    return dropped


def _borrow_intra_line_timestamps(result_list: List[Dict[str, Any]]) -> None:
    n = len(result_list)
    for i in range(n):
        item = result_list[i]
        if item.get(F.ONO_DROPPED):
            continue
        if "start" not in item and item.get("type", 0) in (1, 2, 3, 4, 5):
            found = False
            for j in range(i - 1, -1, -1):
                if result_list[j].get("type") == 0 and result_list[j].get("orig") == "\n":
                    break
                if result_list[j].get(F.ONO_DROPPED):
                    continue
                if "start" in result_list[j]:
                    item["start"] = result_list[j]["start"]
                    item["end"] = result_list[j]["end"]
                    found = True
                    break
            if not found:
                for j in range(i + 1, n):
                    if result_list[j].get("type") == 0 and result_list[j].get("orig") == "\n":
                        break
                    if result_list[j].get(F.ONO_DROPPED):
                        continue
                    if "start" in result_list[j]:
                        item["start"] = result_list[j]["start"]
                        item["end"] = result_list[j]["end"]
                        break


def _extract_clause_item_ranges(result_list: List[Dict[str, Any]]) -> List[Tuple[int, int, bool]]:
    ranges: List[Tuple[int, int, bool]] = []
    n = len(result_list)
    seg_start = 0
    has_pron_in_seg = False

    i = 0
    while i < n:
        it = result_list[i]
        orig = it.get("orig", "")
        is_bound = _clause_mod.is_clause_boundary(it)
        if is_bound:
            if has_pron_in_seg:
                j = i
                saw_nl = (orig == "\n")
                while j < n and _clause_mod.is_clause_boundary(result_list[j]):
                    if result_list[j].get("orig") == "\n":
                        saw_nl = True
                        j += 1
                        break
                    j += 1
                ranges.append((seg_start, j, saw_nl))
                seg_start = j
                has_pron_in_seg = False
                i = j
                continue
            else:
                if orig == "\n":
                    seg_start = i + 1
                i += 1
                continue
        if it.get("pron"):
            has_pron_in_seg = True
        i += 1

    if has_pron_in_seg and seg_start < n:
        ranges.append((seg_start, n, False))
    return ranges


def post_process_and_merge_streams(
    result_list: List[Dict[str, Any]],
    alignment_results: List[Dict[str, Any]],
    token_to_index_map: Dict[int, int],
    stream_clause_lists: List[List[int]],
    enable_ono_gate: bool,
    enable_resegment: bool,
    enable_merge_short: bool,
    merge_short_chars: int,
    merge_short_gap: float,
    merge_ono_chars: int,
    merge_ono_gap: float,
    log_callback: Optional[Callable[[str], None]] = None,
    return_independent_rl: bool = False,
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
    def log(msg: str):
        if log_callback:
            log_callback(msg)

    apply_skipped_ono_marks(result_list, alignment_results, token_to_index_map)

    clause_ranges = _extract_clause_item_ranges(result_list)
    if len(stream_clause_lists) <= 1 or len(clause_ranges) == 0:
        return (result_list, result_list) if return_independent_rl else result_list

    orig_rl_time: Dict[int, Tuple[float, float]] = {}
    orig_rl_ar: Dict[int, Dict[str, Any]] = {}
    for fi, r in enumerate(alignment_results):
        ri = token_to_index_map.get(fi)
        if ri is not None:
            orig_rl_ar[ri] = r
            if r.get("original_start") is not None and r.get("original_end") is not None:
                orig_rl_time[ri] = (float(r["original_start"]), float(r["original_end"]))

    # 识别每个 clause 所属的原始台本行号 (line_group)
    clause_to_lg: List[int] = []
    cur_lg = 0
    for c_idx, (s_i, e_i, had_nl) in enumerate(clause_ranges):
        clause_to_lg.append(cur_lg)
        if had_nl:
            cur_lg += 1

    all_final_lines: List[Tuple[float, float, int, int, int, List[Dict[str, Any]]]] = []
    stream_sub_rls: Dict[int, List[Dict[str, Any]]] = {}
    stream_line_spans: Dict[int, List[Tuple[float, float]]] = {}

    # 预计算各角色子流的原始有效子句时间区间，供单流 merge_short 避让跨角色插入台词
    stream_raw_spans: Dict[int, List[Tuple[float, float]]] = {}
    for sid_pre, cl_idxs_pre in enumerate(stream_clause_lists):
        sp_list: List[Tuple[float, float]] = []
        for ci_pre in cl_idxs_pre:
            if ci_pre >= len(clause_ranges):
                continue
            s_p, e_p, _ = clause_ranges[ci_pre]
            sts_p: List[float] = []
            ens_p: List[float] = []
            for oi_p in range(s_p, e_p):
                if result_list[oi_p].get(F.ONO_DROPPED):
                    continue
                if oi_p in orig_rl_time:
                    t_s, t_e = orig_rl_time[oi_p]
                    sts_p.append(t_s)
                    ens_p.append(t_e)
            if sts_p:
                sp_list.append((min(sts_p), max(ens_p)))
        stream_raw_spans[sid_pre] = sp_list

    for stream_id, cl_indices in enumerate(stream_clause_lists):
        sub_rl: List[Dict[str, Any]] = []
        sub_ar: List[Dict[str, Any]] = []
        sub_tok_map: Dict[int, int] = {}
        sub_rl_time: Dict[int, Tuple[float, float]] = {}

        for pos, ci in enumerate(cl_indices):
            if ci >= len(clause_ranges):
                continue
            s_idx, e_idx, had_nl = clause_ranges[ci]
            lg_val = clause_to_lg[ci] if ci < len(clause_to_lg) else ci
            for orig_i in range(s_idx, e_idx):
                it_orig = result_list[orig_i]
                if it_orig.get("orig") == "\n":
                    continue
                it_copy = copy.copy(it_orig)
                it_copy["_ms_lg"] = lg_val
                new_i = len(sub_rl)
                sub_rl.append(it_copy)
                if orig_i in orig_rl_ar:
                    sub_fi = len(sub_ar)
                    sub_ar.append(orig_rl_ar[orig_i])
                    sub_tok_map[sub_fi] = new_i
                    it_copy["_ms_score"] = float(orig_rl_ar[orig_i].get("score", -10.0))
                if orig_i in orig_rl_time:
                    sub_rl_time[new_i] = orig_rl_time[orig_i]

            next_ci = cl_indices[pos + 1] if (pos + 1 < len(cl_indices)) else -1
            if had_nl or (next_ci != ci + 1):
                sub_rl.append({"orig": "\n", "type": 0, "pron": ""})

        if not sub_rl:
            continue
        if sub_rl[-1].get("orig") != "\n":
            sub_rl.append({"orig": "\n", "type": 0, "pron": ""})

        if enable_ono_gate:
            from ono_filter import mark_ono_candidates as _ono_mark
            _ono_mark(
                result_list=sub_rl,
                alignment_results=sub_ar,
                token_to_index_map=sub_tok_map,
                log_callback=None,
            )
            # 保护含明确谓语/句尾助词的短句（如「ビクってしたぁ」），避免因跨角色轮次间隔被误标为飞走拟声
            _cl_buf: List[int] = []
            for _idx_s, _it_s in enumerate(sub_rl):
                if _clause_mod.is_clause_boundary(_it_s):
                    if _cl_buf:
                        _surf = "".join(sub_rl[k].get("orig", "") for k in _cl_buf)
                        if any(kw in _surf for kw in ("した", "して", "てる", "ちゃ", "だよ", "だね", "かな", "もん")):
                            for k in _cl_buf:
                                sub_rl[k].pop(F.ONO_MARK, None)
                                sub_rl[k][F.CLAUSE_ONO] = False
                        _cl_buf = []
                elif _it_s.get("pron"):
                    _cl_buf.append(_idx_s)

        if enable_resegment:
            from resegment import apply_punctuation_resegment as _reseg
            sub_rl = _reseg(sub_rl, sub_rl_time, frame_sec=0.02, thr_frames=50, max_line_chars=22)
            # 若重分段把行中单假名口吃（如「…、き、\n綺麗…」）切在了上一行末尾，将其并回下一行行首
            _k_nl = 0
            while _k_nl < len(sub_rl):
                if sub_rl[_k_nl].get("orig") == "\n" and sub_rl[_k_nl].get(F.INSERTED):
                    # 找 \n 左侧最后一个子句的发音 token
                    _p_end = _k_nl - 1
                    while _p_end >= 0 and sub_rl[_p_end].get("orig") != "\n" and _clause_mod.is_punct_symbol(sub_rl[_p_end]):
                        _p_end -= 1
                    _p_st = _p_end
                    while _p_st >= 0 and sub_rl[_p_st].get("orig") != "\n" and not _clause_mod.is_clause_boundary(sub_rl[_p_st]):
                        _p_st -= 1
                    _cl_left = [m for m in range(_p_st + 1, _p_end + 1) if 0 <= m < len(sub_rl) and sub_rl[m].get("pron")]
                    # 找 \n 右侧第一个发音 token
                    _r_first = _k_nl + 1
                    while _r_first < len(sub_rl) and sub_rl[_r_first].get("orig") != "\n" and not sub_rl[_r_first].get("pron"):
                        _r_first += 1
                    if (_cl_left
                            and _p_st >= 0
                            and sub_rl[_p_st].get("orig") != "\n"
                            and _r_first < len(sub_rl)
                            and sub_rl[_r_first].get("pron")):
                        _s_left = "".join(sub_rl[m].get("orig", "") for m in _cl_left)
                        _p_left = "".join(sub_rl[m].get("pron", "") for m in _cl_left)
                        _p_right = str(sub_rl[_r_first].get("pron", ""))
                        if len(_s_left) == 1 and _p_left and _p_right.startswith(_p_left):
                            _nl_item = sub_rl.pop(_k_nl)
                            sub_rl.insert(_p_st + 1, _nl_item)
                _k_nl += 1

        if enable_merge_short:
            from merge_short import merge_short_lines as _merge
            _other_spans = [
                sp
                for other_sid, sps in stream_raw_spans.items()
                if other_sid != stream_id
                for sp in sps
            ]
            sub_rl = _merge(
                sub_rl,
                max_chars=merge_short_chars,
                max_gap=merge_short_gap,
                ono_max_chars=merge_ono_chars,
                ono_max_gap=merge_ono_gap,
                forbidden_spans=_other_spans if _other_spans else None,
            )

        if enable_ono_gate:
            from ono_filter import decide_ono_deletions as _ono_decide
            _ono_decide(result_list=sub_rl, log_callback=None)

        stream_sub_rls[stream_id] = sub_rl
        spans_this: List[Tuple[float, float]] = []
        tmp_ln: List[Dict[str, Any]] = []
        for it in sub_rl:
            if it.get("orig") == "\n":
                if tmp_ln:
                    sts_t = [_parse_lrc_ts_sec(x.get("start")) for x in tmp_ln if not x.get(F.ONO_DROPPED) and x.get("start")]
                    ens_t = [_parse_lrc_ts_sec(x.get("end")) for x in tmp_ln if not x.get(F.ONO_DROPPED) and x.get("end")]
                    sts_t = [t for t in sts_t if t is not None]
                    ens_t = [t for t in ens_t if t is not None]
                    if sts_t:
                        spans_this.append((min(sts_t), max(ens_t) if ens_t else min(sts_t)))
                    tmp_ln = []
            else:
                tmp_ln.append(it)
        stream_line_spans[stream_id] = spans_this

    # ── 重叠部分的易错拟声词剔除 + 脱离正文的远距首尾喘息与孤立短杂音清理 ──
    if enable_ono_gate:
        from ono_filter import is_easy_ono as _is_easy_ono_fn
        from ono_block_corpus import is_blocked as _is_blocked_fn
        _KANJI_ALNUM_RE = re.compile(r"[一-鿿㐀-䶿々A-Za-zＡ-Ｚａ-ｚ0-9０-９]")
        _WET_SUBSTRS = ("くぷ", "くぽ", "ぐぷ", "ちゅぷ", "じゅぷ", "れちゅ", "りゅ", "ぴゅる")
        _CONV_KEEP_SET = frozenset(("ねー", "ねえ", "はーい", "はい", "うん", "ええ", "あはっ", "えへっ"))

        def _is_wet_or_ono_clause(cl_idxs: List[int], rl_arr: List[Dict[str, Any]], in_overlap: bool) -> bool:
            surf = "".join(rl_arr[p].get("orig", "") for p in cl_idxs)
            if not surf or _KANJI_ALNUM_RE.search(surf):
                return False
            if all(bool(rl_arr[p].get(F.CLAUSE_ONO)) for p in cl_idxs) or _is_easy_ono_fn(surf):
                return True
            if _is_blocked_fn(surf):
                return False
            if any(w in surf for w in _WET_SUBSTRS) and len(surf) <= 6:
                return True
            if surf[0] in "ぁぃぅぇぉ" and len(surf) <= 4:
                return True
            return False

        def _drop_clause_and_punct(cl_idxs: List[int], rl_arr: List[Dict[str, Any]], l_bound: int, r_bound: int) -> None:
            for p in cl_idxs:
                if "start" in rl_arr[p] and F.ONO_SAVED not in rl_arr[p]:
                    rl_arr[p][F.ONO_SAVED] = (rl_arr[p].get("start"), rl_arr[p].get("end"))
                rl_arr[p].pop("start", None)
                rl_arr[p].pop("end", None)
                rl_arr[p][F.ONO_DROPPED] = True
            r_p = cl_idxs[-1] + 1
            while r_p < r_bound and _clause_mod.is_punct_symbol(rl_arr[r_p]):
                rl_arr[r_p].pop("start", None)
                rl_arr[r_p].pop("end", None)
                rl_arr[r_p][F.ONO_DROPPED] = True
                r_p += 1
            l_p = cl_idxs[0] - 1
            has_left_alive = any(
                rl_arr[m].get("pron") and not rl_arr[m].get(F.ONO_DROPPED)
                for m in range(l_bound, cl_idxs[0])
            )
            if not has_left_alive:
                while l_p >= l_bound and _clause_mod.is_punct_symbol(rl_arr[l_p]):
                    rl_arr[l_p].pop("start", None)
                    rl_arr[l_p].pop("end", None)
                    rl_arr[l_p][F.ONO_DROPPED] = True
                    l_p -= 1

        for stream_id, sub_rl in stream_sub_rls.items():
            other_spans = [
                sp for oth_id, sps in stream_line_spans.items() if oth_id != stream_id for sp in sps
            ]
            n_sub = len(sub_rl)
            ln_start = 0
            for idx_i in range(n_sub):
                if sub_rl[idx_i].get("orig") != "\n":
                    continue
                ln_end = idx_i
                if ln_end > ln_start:
                    sts_l = [
                        _parse_lrc_ts_sec(sub_rl[k].get("start"))
                        for k in range(ln_start, ln_end)
                        if not sub_rl[k].get(F.ONO_DROPPED) and sub_rl[k].get("start")
                    ]
                    ens_l = [
                        _parse_lrc_ts_sec(sub_rl[k].get("end"))
                        for k in range(ln_start, ln_end)
                        if not sub_rl[k].get(F.ONO_DROPPED) and sub_rl[k].get("end")
                    ]
                    sts_l = [t for t in sts_l if t is not None]
                    ens_l = [t for t in ens_l if t is not None]
                    if sts_l:
                        l_st, l_en = min(sts_l), (max(ens_l) if ens_l else min(sts_l))
                        line_overlaps = any(
                            min(l_en, o_en) - max(l_st, o_st) > -1e-4
                            for o_st, o_en in other_spans
                        )
                        # 提取该行内所有存活子句
                        line_clauses: List[Tuple[List[int], bool, float, float]] = []
                        cl_pron_indices: List[int] = []
                        for k in range(ln_start, ln_end + 1):
                            it_k = sub_rl[k] if k < ln_end else {"orig": "\n", "type": 0, "pron": ""}
                            if k == ln_end or _clause_mod.is_clause_boundary(it_k):
                                if cl_pron_indices:
                                    surf_c = "".join(sub_rl[p].get("orig", "") for p in cl_pron_indices)
                                    is_ono_c = _is_wet_or_ono_clause(cl_pron_indices, sub_rl, line_overlaps)
                                    # 保护单假名口吃前缀（如「い、挿れて」「う、嘘」）
                                    if is_ono_c and len(surf_c) == 1:
                                        c_pron = "".join(sub_rl[p].get("pron", "") for p in cl_pron_indices)
                                        nxt_p = k + 1
                                        while nxt_p < ln_end and not sub_rl[nxt_p].get("pron"):
                                            nxt_p += 1
                                        if (nxt_p < ln_end
                                                and not sub_rl[nxt_p].get(F.CLAUSE_ONO)
                                                and c_pron
                                                and str(sub_rl[nxt_p].get("pron", "")).startswith(c_pron)):
                                            is_ono_c = False
                                    c_sts = [_parse_lrc_ts_sec(sub_rl[p].get("start")) for p in cl_pron_indices if sub_rl[p].get("start")]
                                    c_ens = [_parse_lrc_ts_sec(sub_rl[p].get("end")) for p in cl_pron_indices if sub_rl[p].get("end")]
                                    c_sts = [t for t in c_sts if t is not None]
                                    c_ens = [t for t in c_ens if t is not None]
                                    c_st = min(c_sts) if c_sts else l_st
                                    c_en = max(c_ens) if c_ens else c_st
                                    line_clauses.append((list(cl_pron_indices), is_ono_c, c_st, c_en))
                                    cl_pron_indices = []
                            elif it_k.get("pron") and not it_k.get(F.ONO_DROPPED):
                                cl_pron_indices.append(k)

                        has_sem = any(not is_o for _, is_o, _, _ in line_clauses)
                        if line_overlaps:
                            if has_sem:
                                # 含实词的重叠混合行：仅清理行首/行尾边缘（不在实词内部区间）的拟声，严禁挖空句中字词
                                first_sem_idx = next(i_c for i_c, (_, is_o, _, _) in enumerate(line_clauses) if not is_o)
                                last_sem_idx = max(i_c for i_c, (_, is_o, _, _) in enumerate(line_clauses) if not is_o)
                                for i_c, (c_idxs, is_ono_c, _, _) in enumerate(line_clauses):
                                    if is_ono_c and (i_c < first_sem_idx or i_c > last_sem_idx):
                                        surf_edge = "".join(sub_rl[p].get("orig", "") for p in c_idxs)
                                        if surf_edge not in _CONV_KEEP_SET:
                                            _drop_clause_and_punct(c_idxs, sub_rl, ln_start, ln_end)
                            else:
                                for c_idxs, is_ono_c, _, _ in line_clauses:
                                    if is_ono_c:
                                        _drop_clause_and_punct(c_idxs, sub_rl, ln_start, ln_end)
                        elif line_clauses:
                            if has_sem:
                                # 混合行：1) 若行首喘息子句距首个实词子句过远(>=1.2s 或跨度>=2.0s)，剔除行首喘息防止提前出字
                                first_sem_idx = next(i_c for i_c, (_, is_o, _, _) in enumerate(line_clauses) if not is_o)
                                sem_st = line_clauses[first_sem_idx][2]
                                if first_sem_idx > 0:
                                    lead_st = line_clauses[0][2]
                                    lead_en = line_clauses[first_sem_idx - 1][3]
                                    if (sem_st - lead_en) >= 1.20 or (sem_st - lead_st) >= 2.00:
                                        for i_c in range(first_sem_idx):
                                            c_i_toks = line_clauses[i_c][0]
                                            surf_lead = "".join(sub_rl[p].get("orig", "") for p in c_i_toks)
                                            if surf_lead not in _CONV_KEEP_SET:
                                                _drop_clause_and_punct(c_i_toks, sub_rl, ln_start, ln_end)
                                # 2) 若行尾被 merge_short 拼入了下一台本行的拟声子句（或间隔>=0.85s），剔除行尾拟声
                                last_sem_idx = max(i_c for i_c, (_, is_o, _, _) in enumerate(line_clauses) if not is_o)
                                sem_en = line_clauses[last_sem_idx][3]
                                sem_lg = sub_rl[line_clauses[last_sem_idx][0][0]].get("_ms_lg")
                                for i_c in range(last_sem_idx + 1, len(line_clauses)):
                                    tail_idxs, _, tail_st, _ = line_clauses[i_c]
                                    tail_lg = sub_rl[tail_idxs[0]].get("_ms_lg")
                                    if tail_lg != sem_lg or (tail_st - sem_en) >= 0.85:
                                        _drop_clause_and_punct(tail_idxs, sub_rl, ln_start, ln_end)
                            else:
                                # 纯拟声/湿音行：若持续时间极短 (<0.55s) 或音素得分低下 (<-2.20)，整行剔除
                                all_p = [p for c_idxs, _, _, _ in line_clauses for p in c_idxs]
                                scs_all = [float(sub_rl[p].get("_ms_score", -10.0)) for p in all_p]
                                mean_sc = (sum(scs_all) / len(scs_all)) if scs_all else -10.0
                                if (l_en - l_st) < 0.55 or mean_sc < -2.20:
                                    for c_idxs, _, _, _ in line_clauses:
                                        _drop_clause_and_punct(c_idxs, sub_rl, ln_start, ln_end)
                ln_start = idx_i + 1

    for stream_id, sub_rl in stream_sub_rls.items():
        # 若中间拟声被剔除后导致单行内部跨过了其他角色的台本轮次 (_ms_lg 跳跃 > 2)，在子句边界处自动拆回独立行
        re_split_rl: List[Dict[str, Any]] = []
        tmp_seg: List[Dict[str, Any]] = []
        prev_lg_val: Optional[int] = None
        prev_spk_val: Optional[Tuple[str, ...]] = None
        at_clause_bound = True
        for it_x in sub_rl:
            if it_x.get("orig") == "\n":
                if tmp_seg:
                    re_split_rl.extend(tmp_seg)
                    tmp_seg = []
                re_split_rl.append(it_x)
                prev_lg_val = None
                prev_spk_val = None
                at_clause_bound = True
                continue
            if _clause_mod.is_clause_boundary(it_x):
                at_clause_bound = True
                tmp_seg.append(it_x)
                continue
            if it_x.get("pron") and not it_x.get(F.ONO_DROPPED):
                lg_x = it_x.get("_ms_lg")
                spk_x = tuple(it_x[F.SPK]) if it_x.get(F.SPK) else None
                spk_diff = (
                    spk_x is not None
                    and prev_spk_val is not None
                    and spk_x != prev_spk_val
                )
                lg_jump = (
                    lg_x is not None
                    and prev_lg_val is not None
                    and abs(int(lg_x) - int(prev_lg_val)) > 2
                )
                if at_clause_bound and (lg_jump or spk_diff):
                    re_split_rl.extend(tmp_seg)
                    re_split_rl.append({"orig": "\n", "type": 0, "pron": "", F.INSERTED: True})
                    tmp_seg = []
                if lg_x is not None:
                    prev_lg_val = int(lg_x)
                if spk_x is not None:
                    prev_spk_val = spk_x
                at_clause_bound = False
            tmp_seg.append(it_x)
        if tmp_seg:
            re_split_rl.extend(tmp_seg)
        sub_rl = re_split_rl
        stream_sub_rls[stream_id] = sub_rl

        _borrow_intra_line_timestamps(sub_rl)

        cur_line: List[Dict[str, Any]] = []
        line_seq = 0
        for it in sub_rl:
            if it.get("orig") == "\n":
                if cur_line:
                    has_word_char = any(
                        not x.get(F.ONO_DROPPED)
                        and not x.get(F.RUBY_PLACEHOLDER)
                        and bool(re.search(r"[ぁ-んァ-ヶー一-龯々A-Za-z0-9０-９]", str(_clause_mod.src_text(x))))
                        for x in cur_line
                    )
                    sts = [
                        _parse_lrc_ts_sec(x.get("start"))
                        for x in cur_line
                        if not x.get(F.ONO_DROPPED) and x.get("start")
                    ]
                    ens = [
                        _parse_lrc_ts_sec(x.get("end"))
                        for x in cur_line
                        if not x.get(F.ONO_DROPPED) and x.get("end")
                    ]
                    sts = [t for t in sts if t is not None]
                    ens = [t for t in ens if t is not None]
                    if has_word_char and sts:
                        st_sec = min(sts)
                        en_sec = max(ens) if ens else st_sec
                        lgs_in_line = [
                            int(x["_ms_lg"]) for x in cur_line
                            if "_ms_lg" in x and not x.get(F.ONO_DROPPED)
                        ]
                        lg_line = lgs_in_line[0] if lgs_in_line else 0
                        all_final_lines.append((st_sec, en_sec, stream_id, line_seq, lg_line, cur_line))
                        line_seq += 1
                    cur_line = []
            else:
                cur_line.append(it)

    def _visible_len(blk: List[Dict[str, Any]]) -> int:
        return sum(
            len(_clause_mod.src_text(x))
            for x in blk
            if not x.get(F.ONO_DROPPED) and not x.get(F.RUBY_PLACEHOLDER) and x.get("orig") != "\n"
        )

    # ── 同屏重叠区内同角色碎句预聚合 ─────────────────────────────────────────
    # resegment 或台本分行会把同角色的一整句话切成相邻小句；当处于跨角色重叠区且间隔很短时先行还原合并。
    def _overlaps_other_stream(idx_a: int, lines_pool: Sequence[Tuple[float, float, int, int, int, List[Dict[str, Any]]]]) -> bool:
        st_a, en_a, sid_a, _, _, _ = lines_pool[idx_a]
        dur_a = max(0.10, en_a - st_a)
        for idx_b, (st_b, en_b, sid_b, _, _, _) in enumerate(lines_pool):
            if sid_b == sid_a:
                continue
            ov = min(en_a, en_b) - max(st_a, st_b)
            min_d = min(dur_a, max(0.10, en_b - st_b))
            if (ov >= 0.20 and (ov / min_d >= 0.35 or abs(st_a - st_b) <= 1.40)) or (ov >= -0.15 and abs(st_a - st_b) <= 0.80):
                return True
        return False

    ov_flags = [_overlaps_other_stream(k, all_final_lines) for k in range(len(all_final_lines))]
    by_stream: Dict[int, List[Tuple[float, float, int, int, int, List[Dict[str, Any]], bool]]] = {}
    for k, item_tup in enumerate(all_final_lines):
        by_stream.setdefault(item_tup[2], []).append((*item_tup, ov_flags[k]))

    from merge_short import append_tokens_with_space as _append_with_space

    def _blk_spk(blk: List[Dict[str, Any]]) -> Optional[Tuple[str, ...]]:
        for x in blk:
            if not x.get(F.ONO_DROPPED) and x.get(F.SPK):
                return tuple(x[F.SPK])
        return None

    consolidated_lines: List[Tuple[float, float, int, int, int, List[Dict[str, Any]]]] = []
    for sid, s_items in by_stream.items():
        idx_s = 0
        while idx_s < len(s_items):
            st_c, en_c, _, seq_c, lg_c, blk_c, is_ov_c = s_items[idx_s]
            blk_merged = [copy.copy(x) for x in blk_c]
            spk_c = _blk_spk(blk_merged)
            last_lg = lg_c
            idx_next = idx_s + 1
            while idx_next < len(s_items):
                st_n, en_n, _, _, lg_n, blk_n, is_ov_n = s_items[idx_next]
                spk_n = _blk_spk(blk_n)
                # 严禁将不同说话人组合（例如多人齐声【A、B】与单人【A】）聚合进同一行，避免前缀误覆盖
                if spk_c is not None and spk_n is not None and spk_c != spk_n:
                    break
                gap_cn = st_n - en_c
                dur_n = max(0.10, en_n - st_n)
                vis_n = _visible_len(blk_n)
                vis_merged = _visible_len(blk_merged)

                # 判断同角色相邻句是否可以聚合：
                # 1) 跨台本行（即使行号因舞台词、环境音或轮次跳跃），只要声学停顿短 (gap <= 0.65s，或重叠区 gap <= 0.85s)
                # 2) 同一台本行且停顿较短 (gap <= 0.85s)
                # 3) 极短的碎屑词（如短于 1.0s 或 <= 4 字的喘息/感叹/接话），只要 gap <= 1.10s 必须吸纳，避免单独成行闪退
                can_merge = (
                    (gap_cn <= 0.65)
                    or (lg_n == last_lg and gap_cn <= 0.85)
                    or ((is_ov_c or is_ov_n) and gap_cn <= 0.85)
                    or ((vis_n <= 4 or dur_n <= 1.0) and gap_cn <= 1.10)
                )

                if not can_merge:
                    break
                # 控制可见字数不超过 38 字（由字数与停顿自然约束边界，不设人为硬性秒数截断）
                if vis_merged + vis_n > 38:
                    break

                _append_with_space(blk_merged, [copy.copy(x) for x in blk_n])
                st_c = min(st_c, st_n)
                en_c = max(en_c, en_n)
                last_lg = lg_n
                is_ov_c = is_ov_c or is_ov_n
                idx_next += 1

            if en_c > s_items[idx_s][1] or st_c < s_items[idx_s][0]:
                st_str = _format_time(st_c)
                en_str = _format_time(en_c)
                for x in blk_merged:
                    if not x.get(F.ONO_DROPPED) and "start" in x:
                        cur_s = _parse_lrc_ts_sec(x.get("start"))
                        if cur_s is not None and cur_s > st_c:
                            x["start"] = st_str
                        break
                for idx_rev in range(len(blk_merged) - 1, -1, -1):
                    if not blk_merged[idx_rev].get(F.ONO_DROPPED) and "end" in blk_merged[idx_rev]:
                        blk_merged[idx_rev]["end"] = en_str
                        break
            consolidated_lines.append((st_c, en_c, sid, seq_c, lg_c, blk_merged))
            idx_s = idx_next

    all_final_lines = consolidated_lines
    all_final_lines.sort(key=lambda x: (x[0], x[1], x[2], x[3]))

    # 构建未被水平拼接、各角色完全独立的字幕结构流 (用于生成 SRT 字幕)
    independent_rl: List[Dict[str, Any]] = []
    for st_c, en_c, sid, seq_c, lg_c, blk in all_final_lines:
        independent_rl.extend([copy.copy(x) for x in blk])
        independent_rl.append({"orig": "\n", "type": 0, "pron": "", F.INSERTED: True})

    def _append_spaced_block(
        base_blk: List[Dict[str, Any]],
        next_blk: List[Dict[str, Any]],
        spk_next: Tuple[str, ...],
        merged_en_sec: float,
        merged_st_sec: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        spaced_spk = (
            (("　" + spk_next[0] if not spk_next[0].startswith("　") else spk_next[0]),)
            + spk_next[1:]
        )
        blk_b_spaced: List[Dict[str, Any]] = []
        seen_first_surviving = False
        for x in next_blk:
            x_cp = copy.copy(x)
            if not x_cp.get(F.ONO_DROPPED):
                if not seen_first_surviving and _clause_mod.is_punct_symbol(x_cp):
                    x_cp[F.ONO_DROPPED] = True
                elif x_cp.get("pron") or (x_cp.get("orig") and not x_cp.get(F.RUBY_PLACEHOLDER)):
                    seen_first_surviving = True
            if x_cp.get(F.SPK) and tuple(x_cp[F.SPK]) == spk_next:
                x_cp[F.SPK] = spaced_spk
            blk_b_spaced.append(x_cp)
        combined = [copy.copy(x) for x in base_blk] + blk_b_spaced
        if merged_st_sec is not None:
            st_str = _format_time(merged_st_sec)
            for x in combined:
                if not x.get(F.ONO_DROPPED) and "start" in x:
                    cur_s = _parse_lrc_ts_sec(x.get("start"))
                    if cur_s is not None and cur_s > merged_st_sec:
                        x["start"] = st_str
                    break
        en_str = _format_time(merged_en_sec)
        for idx_rev in range(len(combined) - 1, -1, -1):
            if not combined[idx_rev].get(F.ONO_DROPPED) and "end" in combined[idx_rev]:
                combined[idx_rev]["end"] = en_str
                break
        return combined

    def _cuts_internal_subclause(blk: List[Dict[str, Any]], cut_t: float) -> bool:
        """检查在 cut_t 处切换字幕是否会腰斩 blk 内部尚未说完（或尚未开口）的后续子句。"""
        sub_clauses: List[Tuple[float, float]] = []
        c_sts: List[float] = []
        c_ens: List[float] = []
        for x in blk:
            if _clause_mod.is_clause_boundary(x):
                if c_sts:
                    sub_clauses.append((min(c_sts), max(c_ens) if c_ens else min(c_sts)))
                    c_sts, c_ens = [], []
            elif x.get("pron") and not x.get(F.ONO_DROPPED):
                ts_s = _parse_lrc_ts_sec(x.get("start"))
                ts_e = _parse_lrc_ts_sec(x.get("end"))
                if ts_s is not None:
                    c_sts.append(ts_s)
                if ts_e is not None:
                    c_ens.append(ts_e)
        if c_sts:
            sub_clauses.append((min(c_sts), max(c_ens) if c_ens else min(c_sts)))
        if len(sub_clauses) < 2:
            return False
        for c_st, c_en in sub_clauses[1:]:
            if (c_en - cut_t) >= 0.25 and (cut_t - c_st) <= 0.65:
                return True
        return False

    merged_blocks: List[List[Dict[str, Any]]] = []
    consumed = set()
    for i in range(len(all_final_lines)):
        if i in consumed:
            continue
        cur_st, cur_en, sid_a, seq_a, lg_a, cur_blk = all_final_lines[i]
        consumed.add(i)
        spk_a = next((tuple(x[F.SPK]) for x in cur_blk if x.get(F.SPK)), None)
        used_sids = {sid_a}
        used_spks = {spk_a} if spk_a is not None else set()

        j = i + 1
        while j < len(all_final_lines):
            if j in consumed:
                j += 1
                continue
            st_b, en_b, sid_b, seq_b, lg_b, blk_b = all_final_lines[j]
            # 允许在 <= 2.20s 内接纳被当前长句完整嵌套包住的短插话，避免长句播到 1.7s 就被腰斩頂掉
            if st_b > cur_en + 0.35 or (st_b - cur_st) > 2.20:
                break

            spk_b = next((tuple(x[F.SPK]) for x in blk_b if x.get(F.SPK)), None)
            if spk_b is None or spk_b in used_spks:
                break

            vis_cur = _visible_len(cur_blk)
            vis_b = _visible_len(blk_b)
            # 放宽多说话人同屏最大字数至 68 字符（同步起音 <= 0.80s 时放宽至 75 字符）；超限时直接停止，严禁跳过 j 去抢合并 j+1 导致倒序
            max_vis = 75 if (st_b - cur_st) <= 0.80 else 68
            if vis_cur + vis_b > max_vis:
                break

            # ── 不同角色的并发重叠句（同一角色已在 consolidated_lines 完成顺接，此处不再跨句回拉）──
            if len(used_spks) < 3:
                overlap = cur_en - st_b
                dur_b = max(0.20, en_b - st_b)
                min_dur = max(0.20, min(cur_en - cur_st, dur_b))
                dt_st = st_b - cur_st
                is_simul_near = (
                    dt_st <= 1.65
                    and (
                        (dt_st <= 0.45 and overlap >= 0.0)
                        or (overlap >= 0.35 and (overlap / min_dur) >= 0.35)
                        or (overlap >= 1.20 and (overlap / min_dur) >= 0.25)
                    )
                )
                # 嵌套包容重叠：后句被前句（剩余 >= 1.5s 且几乎完全覆盖）包在内部，合并同屏防止长句瞬间消失
                is_nested_inside = (
                    (en_b <= cur_en + 0.25)
                    and (cur_en - st_b) >= 1.50
                    and (overlap / dur_b) >= 0.85
                )
                is_overlap_simul = is_simul_near or is_nested_inside
                if is_overlap_simul:
                    cur_st = min(cur_st, st_b)
                    cur_en = max(cur_en, en_b)
                    cur_blk = _append_spaced_block(cur_blk, blk_b, spk_b, cur_en, cur_st)
                    used_sids.add(sid_b)
                    used_spks.add(spk_b)
                    consumed.add(j)
                    j += 1
                    continue

            break

        merged_blocks.append(cur_blk)

    out_rl: List[Dict[str, Any]] = []
    for blk in merged_blocks:
        out_rl.extend(blk)
        out_rl.append({"orig": "\n", "type": 0, "pron": "", F.INSERTED: True})

    log(f"  ✨ [多流归并] 已将 {len(stream_clause_lists)} 条时间流按时间轴归并为 {len(merged_blocks)} 行字幕")
    if return_independent_rl:
        return out_rl, independent_rl
    return out_rl


# ══════════════════════════════════════════════════════════════════════════
#  5. 多角色独立 SRT 字幕生成引擎
# ══════════════════════════════════════════════════════════════════════════

_PUNCT_CHARS = "、。！？…‥・,.:;!?"


def _strip_orphan_punct(text: str) -> str:
    """去掉【行首】的孤立标点（行尾标点保留，它们是句末终结符）。"""
    s = text.lstrip()
    i = 0
    while i < len(s) and s[i] in _PUNCT_CHARS:
        i += 1
    return s[i:].lstrip()


def _format_srt_time(sec: float) -> str:
    """把秒数格式化为 SRT 标准时间戳：HH:MM:SS,mmm"""
    if sec is None or sec < 0:
        sec = 0.0
    ms_total = int(round(sec * 1000.0))
    s_total, ms = divmod(ms_total, 1000)
    m_total, s = divmod(s_total, 60)
    h, m = divmod(m_total, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_srt_cues(cues: Sequence[Tuple[float, float, str]]) -> str:
    """将 (st_sec, en_sec, text) 列表组装为标准 SRT 文件文本。"""
    blocks = []
    for idx, (st_sec, en_sec, text) in enumerate(cues, start=1):
        if not text.strip():
            continue
        st_str = _format_srt_time(st_sec)
        en_str = _format_srt_time(en_sec)
        blocks.append(f"{idx}\n{st_str} --> {en_str}\n{text}\n")
    return "\n".join(blocks)


def build_independent_multispeaker_srt(
    result_list: List[Dict[str, Any]],
    line_end_extend: float = 0.0,
) -> str:
    """从独立角色流 result_list 组装多角色完全自治的 SRT 字幕。

    核心准则：
    1. 提取所有有效字幕行，按角色（_spk）分组。
    2. 对每个角色内部的台词流独立进行行尾自然延伸（line_end_extend）与最短驻留保护；
       【核心独立准则】：角色的结束时间仅受该角色自身下一句开嗓限制，绝对不被其他并发角色截断。
    3. 将所有角色的 SRT Cue 块汇入全局池，按起音时间严格升序排序，赋予自增序号。
    4. 视频播放器加载后将原生支持多角色垂直重叠分层渲染。
    """
    _KANJI_ALNUM = re.compile(r"[ぁ-んァ-ヶー一-龯々A-Za-z0-9０-９]")

    lines: List[Dict[str, Any]] = []
    cur_tokens: List[Dict[str, Any]] = []

    def _flush_line(tokens: List[Dict[str, Any]]):
        if not tokens:
            return
        active = [
            x for x in tokens
            if not x.get(F.ONO_DROPPED) and not x.get(F.RUBY_PLACEHOLDER)
        ]
        if not active:
            return

        has_word = any(
            _KANJI_ALNUM.search(str(_clause_mod.src_text(x)))
            for x in active if x.get("orig") != "\n"
        )
        if not has_word:
            return

        sts = [_parse_lrc_ts_sec(x.get("start")) for x in active if x.get("start")]
        ens = [_parse_lrc_ts_sec(x.get("end")) for x in active if x.get("end")]
        sts = [t for t in sts if t is not None]
        ens = [t for t in ens if t is not None]
        if not sts:
            return

        st_sec = min(sts)
        en_sec = max(ens) if ens else st_sec

        spk = next((tuple(x[F.SPK]) for x in active if x.get(F.SPK)), None)
        spk_key = spk if spk else ()

        raw_chars = []
        for x in tokens:
            if x.get(F.ONO_DROPPED) or x.get(F.RUBY_PLACEHOLDER) or x.get("orig") == "\n":
                continue
            raw_chars.append(str(_clause_mod.src_text(x)))
        raw_text = "".join(raw_chars)
        clean_text = re.sub(r"\{([^|]+)\|[^}]+\}", r"\1", raw_text).rstrip()
        clean_text = _strip_orphan_punct(clean_text)

        if not clean_text:
            return

        if spk_key:
            names = "、".join(s for s in spk_key if s).strip()
            clean_name = names.lstrip("　 〖【").rstrip("　 〗】")
            if clean_name:
                pfx = f"【{clean_name}】"
                if not (clean_text.startswith(pfx)
                        or clean_text.startswith(f"〖{clean_name}〗")
                        or clean_text.startswith(f"{clean_name}：")
                        or clean_text.startswith(f"{clean_name}:")):
                    clean_text = f"{pfx}{clean_text}"

        lines.append({
            "st_sec": st_sec,
            "en_sec": en_sec,
            "spk_key": spk_key,
            "text": clean_text,
        })

    for item in result_list:
        if item.get("type") == 0 and item.get("orig") == "\n":
            _flush_line(cur_tokens)
            cur_tokens = []
        else:
            cur_tokens.append(item)
    _flush_line(cur_tokens)

    if not lines:
        return ""

    # 按角色分组，各自进行完全独立的时间延伸与防闪退保护
    by_spk: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}
    for ln in lines:
        by_spk.setdefault(ln["spk_key"], []).append(ln)

    all_cues: List[Tuple[float, float, str]] = []

    for spk_key, spk_lines in by_spk.items():
        spk_lines.sort(key=lambda x: (x["st_sec"], x["en_sec"]))
        for idx, ln in enumerate(spk_lines):
            st = ln["st_sec"]
            en = ln["en_sec"]
            eff_en = en

            # 1. 行尾自然延伸
            if line_end_extend > 0:
                eff_en += line_end_extend

            # 2. 角色内部下一句截断（绝对不被其他并发角色截断；间隔 <= 50ms 时避免产生 eff_en < st 倒挂时间轴）
            if idx + 1 < len(spk_lines):
                next_st_same_spk = spk_lines[idx + 1]["st_sec"]
                if next_st_same_spk > st:
                    gap_cap = next_st_same_spk - 0.05 if (next_st_same_spk - st) > 0.05 else max(next_st_same_spk, st + 0.02)
                    eff_en = min(eff_en, gap_cap)
                else:
                    eff_en = max(eff_en, st + 0.5)

            # 3. 最短驻留时长保护（1.0s），防止极短词或尾音一闪而过
            if eff_en < st + 1.0:
                if idx + 1 < len(spk_lines) and spk_lines[idx + 1]["st_sec"] > st:
                    next_st_same_spk = spk_lines[idx + 1]["st_sec"]
                    gap_cap = next_st_same_spk - 0.05 if (next_st_same_spk - st) > 0.05 else max(next_st_same_spk, st + 0.02)
                    eff_en = min(st + 1.0, gap_cap)
                else:
                    eff_en = st + 1.0

            eff_en = max(eff_en, st + 0.02)
            all_cues.append((st, eff_en, ln["text"]))

    # 全局汇流：按起音时间严格升序排序
    all_cues.sort(key=lambda x: (x[0], x[1]))
    return _format_srt_cues(all_cues)


# 单人模式与多视角模式通用别名
build_standard_srt = build_independent_multispeaker_srt
