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

def extract_acoustic_context(
    audio_file_path: str | Path,
    lora_path: Optional[str | Path] = None,
    device: str = "auto",
    chunk_frames: int = 6000,
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
    }


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

    distinct_speakers: List[str] = []
    for spk_tup in clause_speakers:
        for s in spk_tup:
            if s and s not in distinct_speakers:
                distinct_speakers.append(s)

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
                if max(pen_l, pen_r) > 0.0:
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

        # ── 互斥迭代解码（支持乱序 + 音素级声道归属 + 静音底噪保护） ─────────────
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
                    if lr_diff > 0.55:
                        pen_l, pen_r = 1.85, 0.0
                    elif lr_diff < -0.55:
                        pen_l, pen_r = 0.0, 1.85
                    else:
                        pen_l, pen_r = 1.65, 1.65

                    if inf["score"] > -2.20 or (not is_ono and inf["n"] >= 3 and inf["score"] > -2.60):
                        other_raw.append((inf["st"], inf["en"], pen_l, pen_r))
                    elif not is_ono:
                        for r in cl_res:
                            if (not r.get("skipped")
                                    and float(r.get("score", -10.0)) > -2.0
                                    and r.get("original_start") is not None
                                    and r.get("original_end") is not None):
                                other_raw.append((float(r["original_start"]), float(r["original_end"]), pen_l, pen_r))
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
                        if same_ch and not has_own_in_gap:
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
    """将被 Viterbi 跳过弧 (skipped=True) 跳过的易错拟声子句在 result_list 中直接标为 ONO_DROPPED。"""
    dropped = 0
    dropped_indices: List[int] = []
    for fi, r in enumerate(alignment_results):
        if r.get("skipped"):
            ri = token_to_index_map.get(fi)
            if ri is not None and 0 <= ri < len(result_list):
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
            try:
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
            except Exception:
                pass

        if enable_resegment:
            try:
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
            except Exception:
                pass

        if enable_merge_short:
            try:
                from merge_short import merge_short_lines as _merge
                sub_rl = _merge(
                    sub_rl,
                    max_chars=merge_short_chars,
                    max_gap=merge_short_gap,
                    ono_max_chars=merge_ono_chars,
                    ono_max_gap=merge_ono_gap,
                )
            except Exception:
                pass

        if enable_ono_gate:
            try:
                from ono_filter import decide_ono_deletions as _ono_decide
                _ono_decide(result_list=sub_rl, log_callback=None)
            except Exception:
                pass

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

                        if line_overlaps:
                            for c_idxs, is_ono_c, _, _ in line_clauses:
                                if is_ono_c:
                                    _drop_clause_and_punct(c_idxs, sub_rl, ln_start, ln_end)
                        elif line_clauses:
                            has_sem = any(not is_o for _, is_o, _, _ in line_clauses)
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
        at_clause_bound = True
        for it_x in sub_rl:
            if it_x.get("orig") == "\n":
                if tmp_seg:
                    re_split_rl.extend(tmp_seg)
                    tmp_seg = []
                re_split_rl.append(it_x)
                prev_lg_val = None
                at_clause_bound = True
                continue
            if _clause_mod.is_clause_boundary(it_x):
                at_clause_bound = True
                tmp_seg.append(it_x)
                continue
            if it_x.get("pron") and not it_x.get(F.ONO_DROPPED):
                lg_x = it_x.get("_ms_lg")
                if (at_clause_bound
                        and lg_x is not None
                        and prev_lg_val is not None
                        and abs(int(lg_x) - int(prev_lg_val)) > 2):
                    re_split_rl.extend(tmp_seg)
                    re_split_rl.append({"orig": "\n", "type": 0, "pron": "", F.INSERTED: True})
                    tmp_seg = []
                if lg_x is not None:
                    prev_lg_val = int(lg_x)
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
                        and bool(re.search(r"[ぁ-んァ-ヶー一-龯々A-Za-z0-9０-９]", str(x.get(F.SRC) or x.get("orig", ""))))
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
            len(x.get(F.SRC) or x.get("orig", ""))
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

    consolidated_lines: List[Tuple[float, float, int, int, int, List[Dict[str, Any]]]] = []
    for sid, s_items in by_stream.items():
        idx_s = 0
        while idx_s < len(s_items):
            st_c, en_c, _, seq_c, lg_c, blk_c, is_ov_c = s_items[idx_s]
            blk_merged = [copy.copy(x) for x in blk_c]
            last_lg = lg_c
            idx_next = idx_s + 1
            while idx_next < len(s_items):
                st_n, en_n, _, _, lg_n, blk_n, is_ov_n = s_items[idx_next]
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

                blk_merged.extend(copy.copy(x) for x in blk_n)
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
            # 若候选行的起始时间明显晚于当前合并块的结束时间，后续行更晚，直接终止向后探测
            if st_b > cur_en + 0.35:
                break

            spk_b = next((tuple(x[F.SPK]) for x in blk_b if x.get(F.SPK)), None)
            if spk_b is None:
                j += 1
                continue

            vis_cur = _visible_len(cur_blk)
            vis_b = _visible_len(blk_b)
            # 放宽多说话人同屏最大字数至 65 字符（当换行会导致严重闪退截断时放宽至 75 字符）
            max_vis = 65
            if (st_b - cur_st) < 1.80:
                max_vis = 75
            if vis_cur + vis_b > max_vis:
                j += 1
                continue

            # ── 情况 1：同一角色的紧邻/重叠后续台词 ──
            if spk_b in used_spks:
                # 无论当前块是单角色还是已包含多角色，只要当前说话人紧接着继续开口 (st_b <= cur_en + 0.50s)，将其并入对应角色的语段中，防止被后续行切碎截断
                if st_b <= cur_en + 0.50:
                    last_spk_idx = -1
                    for idx_k, x_k in enumerate(cur_blk):
                        if x_k.get(F.SPK) and tuple(x_k[F.SPK]) == spk_b:
                            last_spk_idx = idx_k
                    if last_spk_idx >= 0:
                        cur_blk = cur_blk[:last_spk_idx + 1] + [copy.copy(x) for x in blk_b] + cur_blk[last_spk_idx + 1:]
                    else:
                        cur_blk = cur_blk + [copy.copy(x) for x in blk_b]
                    cur_st = min(cur_st, st_b)
                    cur_en = max(cur_en, en_b)
                    # 重新刷新当前块的 end 标记
                    en_str = _format_time(cur_en)
                    for idx_rev in range(len(cur_blk) - 1, -1, -1):
                        if not cur_blk[idx_rev].get(F.ONO_DROPPED) and "end" in cur_blk[idx_rev]:
                            cur_blk[idx_rev]["end"] = en_str
                            break
                    consumed.add(j)
                    j += 1
                    continue

            # ── 情况 2：不同角色的并发重叠句 ──
            if spk_b not in used_spks and len(used_spks) < 3:
                overlap = cur_en - st_b
                # 判定重叠并发：
                # (1) 实质性时间重叠（后句在前句尚未说完前开口 >= 0.15s）
                # (2) 起音高度接近的抢话（起音差 <= 0.65s 且重叠 >= 0.10s）
                # (3) 驻留保护：后句在前句显示不足 1.8s 时开口（若不合并前句就会被瞬间腰斩闪退）
                is_overlap_simul = (
                    (overlap >= 0.15)
                    or (abs(st_b - cur_st) <= 0.65 and overlap >= 0.10)
                    or ((st_b - cur_st) < 1.80 and overlap >= -0.05)
                )
                if is_overlap_simul:
                    cur_st = min(cur_st, st_b)
                    cur_en = max(cur_en, en_b)
                    cur_blk = _append_spaced_block(cur_blk, blk_b, spk_b, cur_en, cur_st)
                    used_sids.add(sid_b)
                    used_spks.add(spk_b)
                    consumed.add(j)
                    j += 1
                    continue

            j += 1

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
            _KANJI_ALNUM.search(str(x.get(F.SRC) or x.get("orig", "")))
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
            raw_chars.append(str(x.get(F.SRC) or x.get("orig", "")))
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

            # 2. 角色内部下一句截断（绝对不被其他并发角色截断）
            if idx + 1 < len(spk_lines):
                next_st_same_spk = spk_lines[idx + 1]["st_sec"]
                if next_st_same_spk > st:
                    eff_en = min(eff_en, next_st_same_spk - 0.05)
                else:
                    eff_en = max(eff_en, st + 0.5)

            # 3. 最短驻留时长保护（1.0s），防止极短词或尾音一闪而过
            if eff_en < st + 1.0:
                if idx + 1 < len(spk_lines) and spk_lines[idx + 1]["st_sec"] > st:
                    eff_en = min(st + 1.0, spk_lines[idx + 1]["st_sec"] - 0.05)
                else:
                    eff_en = st + 1.0

            all_cues.append((st, eff_en, ln["text"]))

    # 全局汇流：按起音时间严格升序排序
    all_cues.sort(key=lambda x: (x[0], x[1]))
    return _format_srt_cues(all_cues)


# 单人模式与多视角模式通用别名
build_standard_srt = build_independent_multispeaker_srt
