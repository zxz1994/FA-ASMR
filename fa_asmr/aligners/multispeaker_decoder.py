# -*- coding: utf-8 -*-
"""FA-ASMR V2 — 多人/乱序专用 Numba Viterbi 解码器。"""
from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

try:
    from numba import njit, prange
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

from fa_asmr.aligners.align_acoustic import FRAME_SEC
from fa_asmr.aligners.align_trellis import (
    TokenSpan,
    _n_pure_continuation_penalty,
    _run_decoder,
)

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


