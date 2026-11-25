# -*- coding: utf-8 -*-
"""FA-ASMR V2 — 多人双耳协同调度、局部碰撞修复与乱序重排核心管线。"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

import fa_asmr.aligners.align as align
from fa_asmr.aligners.align_acoustic import (
    FRAME_SEC,
    build_next_stable_silence_end,
    compute_track_noise_floor,
)
from fa_asmr.aligners.align_utils import _format_time
from fa_asmr.aligners.multispeaker_acoustic import (
    _AudioSoloChecker,
    _SpeakerVoiceprintJudge,
    _align_subset_clauses,
    _clause_summary,
    _estimate_speaker_ear_bias,
    _rescue_out_of_order_short_clauses,
    _resolve_same_text_swaps,
    _unmasked_clause_mean_score,
    build_cooperative_stereo_emission,
    extract_acoustic_context,
)
from fa_asmr.pipeline.subtitles import _parse_lrc_ts_sec

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
        from fa_asmr.text.ono_filter import is_easy_ono as _is_easy_ono_fn
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
        diagnostics_out["stereo_wav_16k"] = acoustic_ctx.get("_stereo_wav")

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


