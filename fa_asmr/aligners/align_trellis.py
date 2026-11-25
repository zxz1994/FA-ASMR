# -*- coding: utf-8 -*-

"""FA-ASMR V2 — 全局 CTC 对齐器 (Global CTC Aligner)

核心机制

1. 保持全局 CTC 端到端全时域连续拓扑, 不进行任何硬窗口切块。

2. 软性声学证据惩罚 (非硬性时长上限):
   本模块不施加任何 per-token 的帧数上限。所有约束都是加在路径分数上的
   软惩罚, 由 Viterbi 自行权衡:

   - 词内 Blank 软降权 (intra_blank_penalty): 抑制假名过渡处的 Blank 抢跑
   - 短句向心 Blank 成本 (subclause_blank_*): 同属一大行时抑制子句间停顿
   - 独立鼻音 ん 入口惩罚 (n_pure_continuation_penalty): 抑制 n 在长静音区打滑
   - 句首末端声学岛保护 (head_token_silence_*): 由 align.py 提供掩码

   注意: 早期版本存在的 consonant_max_sec / nasal_max_sec / vowel_max_sec
   主动时长上限机制在当前实现中并不存在, 相关参数已移除。

3. 接口与 torchaudio.pipelines.MMS_FA.get_aligner() 完全兼容。

"""

from __future__ import annotations


import logging

from dataclasses import dataclass

from typing import Any, List, Optional


import numpy as np


try:

    from numba import njit, prange

    _HAS_NUMBA = True

except ImportError:

    _HAS_NUMBA = False


logger = logging.getLogger(__name__)

FRAME_SEC: float = 0.02



@dataclass
class TokenSpan:
    """与 torchaudio MMS_FA TokenSpan 保持完全相同的字段契约."""
    token: int
    start: int
    end: int
    score: float


def _n_pure_continuation_penalty(
    n_logp,
    lookahead_frames: int = 5,
    penalty_scale: float = 0.15,
    drop_threshold: float = 1.0,
    drop_ramp: float = 2.0,
):
    """Entry-only bounded additive penalty for singleton [n] targets.

    Compare each frame with the median of the following frames, excluding
    itself. If the future support collapses by more than drop_threshold
    log units, apply an additive bounded penalty up to penalty_scale.
    No blank comparison is used, preserving soft nasals in ASMR.
    """
    if lookahead_frames < 1:
        raise ValueError("n continuation lookahead must be at least one frame")
    scores = np.asarray(n_logp, dtype=np.float32)
    penalties = np.zeros_like(scores)
    if len(scores) > lookahead_frames and penalty_scale > 0.0:
        future = np.lib.stride_tricks.sliding_window_view(scores[1:], lookahead_frames)
        support = np.median(future, axis=1)
        current = scores[:-lookahead_frames]
        drop = np.zeros_like(current)
        finite_mask = np.isfinite(current) & np.isfinite(support)
        drop[finite_mask] = np.maximum(0.0, current[finite_mask] - support[finite_mask])
        collapsed_mask = np.isfinite(current) & np.isneginf(support)
        drop[collapsed_mask] = np.float32(10.0)
        excess = np.maximum(0.0, drop - np.float32(drop_threshold))
        if drop_ramp > 0.0:
            ramp_factor = np.minimum(1.0, excess / np.float32(drop_ramp))
        else:
            ramp_factor = (excess > 0.0).astype(np.float32)
        penalties[:-lookahead_frames] = np.float32(penalty_scale) * ramp_factor
    return penalties


if _HAS_NUMBA:

    # Viterbi 的并行线程数上限。`parallel=True` 默认吃满 numba 的全部线程
    # （本机 12）。实测 1h 音频的 DP：1 线程 28.1s / 2 线程 17.5s / 4 线程
    # 11.5s / **8 线程 9.4s（最优）** / 12 线程 11.2s（劣化 18%）。
    # 超过 8 线程后，每帧的 prange 区域启动/同步开销开始抵消并行收益
    # （states 达 2.4 万，每帧并行区极短）。
    # 结果逐位一致（score 恒为 -607477.8750），属零行为变更的纯性能改动。
    _VITERBI_MAX_THREADS = 8

    class _viterbi_threads:
        """把 numba 线程数限制在最优值，作用域内生效、退出后还原。

        只在需要收紧时才 set_num_threads：若当前线程数已 <= 上限则完全不动，
        避免对全局 numba 状态做任何无谓的写入。
        """

        def __init__(self, n):
            self.n = int(n)
            self.prev = None

        def __enter__(self):
            try:
                import numba
                cur = numba.get_num_threads()
                if cur > self.n:
                    self.prev = cur
                    numba.set_num_threads(self.n)
            except Exception:
                self.prev = None
            return self

        def __exit__(self, *exc):
            if self.prev is not None:
                try:
                    import numba
                    numba.set_num_threads(self.prev)
                except Exception:
                    pass
            return False

    @njit(cache=True, fastmath=True, parallel=True)
    def _viterbi_phoneme_category_duration(
        emission: np.ndarray,          # (T, num_classes) float32 log-probabilities
        tokens: np.ndarray,            # (L,) int32 target token sequence
        blank_cost: np.ndarray,        # (L+1,) float32 每个 blank 状态每帧的成本 (词内 blank 软降权)
        speech_frame_mask: np.ndarray, # (T,) uint8 清晰语音帧掩码 (方案 A 能量解耦)
        subclause_blank_cost: np.ndarray = np.empty(0, dtype=np.float32), # (L+1,) float32 短句与下一句之间 blank 的向心成本惩罚
        subclause_blank_free_frames: int = 50, # (帧) 死区免罚自由额度 (默认 1.0s = 50 帧 在此帧数内停留 0 成本 保留自然换气高精)

        head_token_silence_end: np.ndarray = np.empty(0, dtype=np.int32),

        head_token_silence_penalty: np.ndarray = np.empty(0, dtype=np.float32),

        blank: int = 0,

        n_pure_token_mask: np.ndarray = np.empty(0, dtype=np.uint8),
        n_pure_entry_penalty: np.ndarray = np.empty(0, dtype=np.float32),

        # 易错拟声声学降权: 命中 mask 的非 blank 状态, 其每帧发射分乘 ono_score_scale。
        # 对齐 Kaldi DecodableAmDiagGmmScaled 的 scale_ * loglikelihood 乘性范式:
        # emission 为 log 概率(<=0), 乘 >1 的系数即等比例压低该 token 的声学权重,
        # 分数越负压得越多(按比例), 而非等量平移。拟声时长与边界本不可信,
        # 降低其支配力可减少对相邻语义段起止位置的挤压。
        ono_token_mask: np.ndarray = np.empty(0, dtype=np.uint8),
        ono_score_scale: float = 1.0,
    ) -> np.ndarray:

        frames = emission.shape[0]

        token_count = tokens.shape[0]

        states = token_count * 2 + 1

        labels = np.empty(states, dtype=np.int32)

        state_cost = np.empty(states, dtype=np.float32)

        state_sub_cost = np.zeros(states, dtype=np.float32)

        has_speech_mask = speech_frame_mask.size > 0
        # 注意: 这里必须是「真有位生效」而不是「数组非空」。
        # n_pure_entry_penalty 恒为 (T,) 长度、n_pure_token_mask 恒为 (L,) 长度,
        # 二者长度永远 > 0, 所以写成 size > 0 会恒为 True —— 那样掩码部分失效时
        # 不会有任何降级信号 (踩过一次: 曾误判为「机制在跑」而实际已静默失效)。
        has_n_pure_penalty = bool(n_pure_token_mask.any()) and n_pure_entry_penalty.size > 0
        # 同样要求「真有位生效」: mask 恒为 (L,) 长度, 只看 size 会恒 True。
        has_ono_scale = bool(ono_token_mask.any()) and ono_score_scale != 1.0

        has_sub_cost = subclause_blank_cost.size > 0

        has_head_token_silence_penalty = (
            head_token_silence_end.size > 0 and head_token_silence_penalty.size > 0)

        for state in range(states):

            if state % 2 == 0:

                blank_idx = state // 2

                labels[state] = blank

                state_cost[state] = blank_cost[blank_idx]

                state_sub_cost[state] = subclause_blank_cost[blank_idx] if has_sub_cost else np.float32(0.0)

            else:

                tok_idx = state // 2

                tok = tokens[tok_idx]

                labels[state] = tok

                state_cost[state] = np.float32(0.0)

                state_sub_cost[state] = np.float32(0.0)

        neg_inf = np.float32(-1.0e30)

        
        prev_dp = np.full(states, neg_inf, dtype=np.float32)

        curr_dp = np.full(states, neg_inf, dtype=np.float32)

        
        prev_dur = np.zeros(states, dtype=np.int32)

        curr_dur = np.zeros(states, dtype=np.int32)

        # 该缓冲不参与声学评分, 仅用于承载句首保护窗口的结束帧 (head_guard_end)。
        prev_clause_gap = np.zeros(states, dtype=np.int32)
        curr_clause_gap = np.zeros(states, dtype=np.int32)


        # Every transition move is 0, 1, or 2.  Pack four 2-bit moves per
        # byte along each state's time axis.
        #
        # 布局为 (frames/4, states) 而非 (states, frames/4): 每一帧只写
        # back[t>>2, s] 这一行, 元素在 s 方向连续。反过来写会让同一帧的
        # 24071 次写入跨 24071 条缓存行 (轨 04 实测: 转置后解码 5.34s -> 3.91s,
        # 且 Viterbi 路径逐帧完全一致)。
        back = np.zeros(((frames + 3) // 4, states), dtype=np.uint8)


        # Frame 0 初始化

        is_speech_0 = (speech_frame_mask[0] != 0) if has_speech_mask else True

        c0 = state_cost[0] if is_speech_0 else np.float32(0.0)

        prev_dp[0] = emission[0, blank] - c0

        prev_dur[0] = 1

        
        if states > 1:

            prev_dp[1] = emission[0, labels[1]]
            if has_ono_scale and ono_token_mask[0]:
                prev_dp[1] = prev_dp[1] * np.float32(ono_score_scale)
            if has_n_pure_penalty and n_pure_token_mask[0]:
                prev_dp[1] -= n_pure_entry_penalty[0]

            prev_dur[1] = 1
        for t in range(1, frames):

            max_state = min(states, 2 * t + 2)

            is_speech = (speech_frame_mask[t] != 0) if has_speech_mask else True

            # 取当前帧 emission 行引用: 逐状态用 emission[t, labels[s]] 会让每次
            # 访问都重算 2D 偏移 (states 达 2.4 万时是热点)。
            em_row = emission[t]

            for s in prange(max_state):

                is_nonblank = (s % 2) == 1

                is_blank = not is_nonblank


                # --- Move 0: 自循环 (s -> s) ---

                best_val = prev_dp[s]
                best_prev_state = np.int32(s)
                best_move = 0
                best_clause_gap = prev_clause_gap[s]
                # 句首保护窗口的结束帧复用同一缓冲, 省去另一对全状态数组。
                best_head_guard_end = prev_clause_gap[s]

                best_d = prev_dur[s] + 1

                
                # --- Move 1: 从上一状态进入 (s-1 -> s) ---

                # 完全自由转移 恢复毫秒级边界精度

                if s > 0:

                    ps = s - 1

                    val1 = prev_dp[ps]
                    # Pay once on entry, before comparing with the self-loop.
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
                        best_prev_state = np.int32(ps)

                        best_d = 1  # 重置为新状态第 1 帧
                        best_clause_gap = m1_clause_gap
                        best_head_guard_end = m1_head_guard_end


                # --- Move 2: 跳过中间 Blank 进入 (s-2 -> s) ---

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
                        best_prev_state = np.int32(ps)

                        best_d = 1  # 重置为新状态第 1 帧
                        best_clause_gap = m2_clause_gap
                        best_head_guard_end = m2_head_guard_end


                # 累加当前帧 emission 并扣除词内 Blank 成本与子句间向心 Blank 惩罚 (微调 blank 数值, 1.0s死区门限)
                sub_c = np.float32(0.0)
                if is_blank and state_sub_cost[s] > 0.0:
                    if best_d > subclause_blank_free_frames:
                        sub_c = state_sub_cost[s]

                c = (state_cost[s] if is_speech else np.float32(0.0)) + sub_c

                # 句首保护只作用于 transcript 中的目标位置 不改同一音素类别的

                # 整列 emission 因此不会误伤上一句尾音或其他同音素位置.

                position_c = np.float32(0.0)

                # Sentence-head protection is evaluated in the same global
                # trellis.  The target-position marker guarantees that the
                # preceding transcript token has already been consumed; the
                # candidate carries its own following-silence window.
                if (has_head_token_silence_penalty and is_nonblank
                        and best_head_guard_end > t):
                    position_c += head_token_silence_penalty[s // 2]

                frame_value = em_row[labels[s]]
                # 易错拟声降权: 命中则每帧发射分乘 ono_score_scale (Kaldi 乘性范式)。
                # 只作用于非 blank 状态(拟声 token 自身的停留), blank 不受影响。
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


            # 尾部状态清零用切片赋值, 不再开第二个 prange 区域。
            # 每帧启动一次并行区域的开销在长轨上很可观 (轨 01 实测 1.58s -> 1.10s)。
            if max_state < states:
                curr_dp[max_state:states] = neg_inf
                curr_dur[max_state:states] = 0
                curr_clause_gap[max_state:states] = 0


            # Both state loops overwrite every current slot before it is read.
            # Swap references instead of copying and clearing all state arrays.
            prev_dp, curr_dp = curr_dp, prev_dp
            prev_dur, curr_dur = curr_dur, prev_dur
            prev_clause_gap, curr_clause_gap = curr_clause_gap, prev_clause_gap


        # 回溯最优路径

        final_state = states - 1 if prev_dp[states - 1] >= prev_dp[states - 2] else states - 2

        path = np.empty(frames, dtype=np.int32)

        path[frames - 1] = final_state

        curr_s = final_state

        for t in range(frames - 1, 0, -1):

            move = (int(back[t >> 2, curr_s]) >> ((t & 3) << 1)) & 3

            curr_s -= move
            path[t - 1] = curr_s

        return path, prev_dp[final_state]

else:

    def _viterbi_phoneme_category_duration(*args, **kwargs):
        raise RuntimeError("Numba is required for ConstrainedCTCAligner")

    class _viterbi_threads:
        """无 numba 时的空实现（此时解码器必然抛错，无需管理线程）。"""

        def __init__(self, n):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    _VITERBI_MAX_THREADS = 8


def _run_decoder(decoder, em_np, flat_tokens, decoder_kwargs):
    """在受限线程数下运行 Viterbi 解码器。

    把 numba 线程数上限单独提出来（而非在 njit 上写死），是为了让线程策略
    可被测试与调优，同时保证作用域外完全还原。
    """
    with _viterbi_threads(_VITERBI_MAX_THREADS):
        return decoder(em_np, flat_tokens, **decoder_kwargs)


class ConstrainedCTCAligner:

    def __init__(
        self,
        blank: int = 0,

        nasal_token_id: int = 4,

        subclause_blank_penalty: float = 0.080,

        subclause_blank_free_sec: float = 1.0,

        subclause_blank_max_tokens: int = 3,
        disable_subclause_lt3: bool = False,

        intra_blank_penalty: float = 0.015,

        min_phonemes_for_penalty: int = 10,

        frame_sec: float = FRAME_SEC,
        n_pure_continuation_penalty: float = 0.20,
        n_pure_continuation_lookahead_frames: int = 4,
        n_pure_continuation_drop_threshold: float = 1.0,
        n_pure_continuation_drop_ramp: float = 2.0,

        **kwargs,

    ):

        self.blank = int(blank)

        self.frame_sec = float(frame_sec)


        self.nasal_token_id = int(kwargs.get("nasal_token_id", nasal_token_id))
        self.n_pure_continuation_penalty = float(kwargs.get("n_pure_continuation_penalty", n_pure_continuation_penalty))
        self.n_pure_continuation_lookahead_frames = int(kwargs.get("n_pure_continuation_lookahead_frames", n_pure_continuation_lookahead_frames))
        self.n_pure_continuation_drop_threshold = float(kwargs.get("n_pure_continuation_drop_threshold", n_pure_continuation_drop_threshold))
        self.n_pure_continuation_drop_ramp = float(kwargs.get("n_pure_continuation_drop_ramp", n_pure_continuation_drop_ramp))
        if self.n_pure_continuation_lookahead_frames < 1:
            raise ValueError("n continuation lookahead must be at least one frame")

        self.intra_blank_penalty = float(intra_blank_penalty)
        self.disable_subclause_lt3 = bool(disable_subclause_lt3)

        self.min_phonemes_for_penalty = int(min_phonemes_for_penalty)

        self.subclause_blank_penalty = float(kwargs.get("subclause_blank_penalty", subclause_blank_penalty))

        self.subclause_blank_free_sec = float(kwargs.get("subclause_blank_free_sec", subclause_blank_free_sec))
        self.subclause_blank_max_tokens = int(kwargs.get("subclause_blank_max_tokens", subclause_blank_max_tokens))
        self.subclause_blank_free_frames = (
            int(round(self.subclause_blank_free_sec / self.frame_sec))
            if self.subclause_blank_free_sec > 0
            else 0
        )

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

    ) -> List[List[TokenSpan]]:

        """执行分类约束全局对齐 返回与 torchaudio aligner 一致的 List[List[TokenSpan]]."""

        if hasattr(emission, "detach"):

            em_np = emission.detach().cpu().numpy().astype(np.float32)

        else:

            em_np = np.asarray(emission, dtype=np.float32)


        if em_np.ndim == 3:

            em_np = em_np[0]


        flat_tokens = np.asarray([tok for line in targets for tok in line], dtype=np.int32)

        if flat_tokens.size == 0:

            return [[] for _ in targets]

        # 构建词内 Blank 软降权成本 

        # 句首/句末 Blank 保持 0.0 

        # 短片段豁免 若该断句片段音素数 < min_phonemes_for_penalty 如 <= 4 个音素的弱叹词/短拟声  

        #   内部 Blank 成本完全设为 0.0 防止其在长距离水声/空白区因惩罚而提前跳变抢跑 

        # 若相邻 token 之间存在标点 (blank_punct_mask 为 True) 完全豁免 (0.0) 保留声优自然的呼吸停顿与气口 

        # 仅对实质长对话内部的假名过渡施加微弱向心力.

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


        # 处理能量解耦语音帧掩码 (方案 A)

        if speech_frame_mask is not None:

            sp_mask = np.asarray(speech_frame_mask, dtype=np.uint8)

        else:

            sp_mask = np.empty(0, dtype=np.uint8)


        head_silence_end = (np.asarray(head_token_silence_end, dtype=np.int32)
                            if head_token_silence_end is not None else np.empty(0, dtype=np.int32))
        head_silence_penalty = (np.asarray(head_token_silence_penalty, dtype=np.float32)
                                if head_token_silence_penalty is not None else np.empty(0, dtype=np.float32))
        if head_silence_penalty.size:
            if len(head_silence_penalty) != len(flat_tokens):
                raise ValueError("head token silence penalties must match flattened target length")
            if len(head_silence_end) != len(em_np):
                raise ValueError("head token silence ends must match emission frame count")


        # 短句内独立鼻音 ん 判定: 仅 n_pure_token_mask 有实际消费方
        # (njit 内的 _n_pure_continuation_penalty 入口惩罚)。其余语义已无读取点。

        n_pure_token_mask = np.zeros(len(flat_tokens), dtype=np.uint8)

        cursor = 0

        for line_index, line in enumerate(targets):

            n_line = len(line)

            # 行首独立 ん 判定, 输入源是 head_standalone_n_flag (每行一个 0/1)。
            # 未传该数组时回退为「单 token 行」判定, 保持向后兼容。
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

        # 构建短句 len <= subclause_blank_max_tokens 默认 3 与下一句同属一大行台词时的向心 Blank 成本惩罚 (微调 blank 数值) 

        subclause_blank_cost = np.zeros(len(flat_tokens) + 1, dtype=np.float32)

        if self.subclause_blank_penalty > 0.0 and line_group is not None:

            cursor = 0

            for i, line in enumerate(targets):

                n_line = len(line)

                if 1 <= n_line <= self.subclause_blank_max_tokens:

                    if i + 1 < len(targets) and line_group[i] == line_group[i + 1]:

                        b_idx = cursor + n_line

                        subclause_blank_cost[b_idx] = np.float32(self.subclause_blank_penalty)

                cursor += n_line

            if self.disable_subclause_lt3:
                group_totals = {}
                for i, line in enumerate(targets):
                    g = int(line_group[i])
                    group_totals[g] = group_totals.get(g, 0) + len(line)
                cursor = 0
                for i, line in enumerate(targets):
                    n_line = len(line)
                    g = int(line_group[i])
                    # 阈值原为 short_clause_acoustic_max_tokens 的默认值 6。
                    # 该参数随短句声学加权机制一并移除, 此处取其原值以保持行为。
                    if (group_totals[g] > 6
                            and 0 < n_line < 3 and i + 1 < len(targets)
                            and line_group[i] == line_group[i + 1]):
                        subclause_blank_cost[cursor + n_line] = np.float32(0.0)
                    cursor += n_line


        # 运行分类约束 Viterbi DP（短行元音簇末端着陆势阱）

        # The whole-clause experiment uses the fast trellis with a small
        # amount of per-state bookkeeping inside the global Viterbi search.
        # 易错拟声掩码: 长度必须等于 flat_tokens, 越界会让 numba 段错误。
        if ono_token_mask is not None and len(ono_token_mask) == flat_tokens.shape[0]:
            ono_mask_arr = np.asarray(ono_token_mask, dtype=np.uint8)
        else:
            ono_mask_arr = np.zeros(flat_tokens.shape[0], dtype=np.uint8)

        decoder = _viterbi_phoneme_category_duration
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
        )

        path, final_score = _run_decoder(
            decoder, em_np, flat_tokens, decoder_kwargs)
        # Keep the final state path available for offline score diagnostics.
        # This is a copy so callers cannot mutate the Viterbi result.
        self._last_path = np.array(path, copy=True)
        self._last_viterbi_score = float(final_score)


        # 转换为按行分组的 TokenSpan 列表
        #
        # 原实现对每个 token 做一次 `np.flatnonzero(path == state_idx)`，即对每个
        # token 全扫一遍 frames -> O(L x T)。1h 音频（T=180000, L=12000）实测
        # 0.608s。改为一次 sort 后用 searchsorted 求每个 state 的首/末出现位置，
        # 复杂度降到 O(T log T)，实测 0.016s（37x），12000 个 span 逐一对等。
        #
        # 关键点：path 是 Viterbi 回溯出的状态序列，**单调不减**（每帧最多前进
        # 2），因此 sort 后同一 state 的帧必然聚成一段连续区间，searchsorted
        # 的左右边界即该 state 的首帧与末帧+1，与原实现语义完全一致。

        grouped_spans: List[List[TokenSpan]] = []

        token_cursor = 0

        total_tokens = flat_tokens.shape[0]
        # 所有非 blank 状态（奇数 state_idx = 2*token+1），一次性批量求边界
        wanted = 2 * np.arange(total_tokens, dtype=np.int64) + 1
        sorted_path = np.sort(path)
        lo = np.searchsorted(sorted_path, wanted, side="left")
        hi = np.searchsorted(sorted_path, wanted, side="right")
        # state 在 path 中出现过 <=> 右边界严格大于左边界
        hit_mask = hi > lo

        for line in targets:

            line_spans = []

            for tok in line:

                state_idx = 2 * token_cursor + 1

                if hit_mask[token_cursor]:

                    st = int(lo[token_cursor])

                    en = int(hi[token_cursor])

                    avg_score = float(np.mean(em_np[st:en, tok]))

                else:

                    # 回退分支：该 state 从未被进入。原实现用
                    # `np.flatnonzero(path >= state_idx)` 取首个 >= 该 state 的帧，
                    # 在排序数组上等价于 searchsorted(..., side="left") 的位置。
                    pos = int(np.searchsorted(sorted_path, state_idx, side="left"))

                    st = pos if pos < len(sorted_path) else 0

                    en = st + 1

                    avg_score = float(em_np[st, tok])


                line_spans.append(TokenSpan(token=int(tok), start=st, end=en, score=avg_score))

                token_cursor += 1

            grouped_spans.append(line_spans)


        return grouped_spans

