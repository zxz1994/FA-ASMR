# -*- coding: utf-8 -*-
"""对齐器组件 — 后处理: segment_align / anchor_realign / auto_repeat

从 align.py 拆分。这些函数接收全局 CTC 结果并做修正/重对齐。
"""

import dataclasses
import logging

logger = logging.getLogger(__name__)

# 从 align_utils 导入共享工具（同一目录，非 package）
from align_utils import (
    _sum_score, _pick_best_spans, _hybrid_spans,
    _is_repeat_line, _is_symbol_token,
)

def _segment_align(emission_L, emission_R, all_token_ids, lines,
                   aligner, frame_duration, map_to_original_time,
                   token_spans_L, token_spans_R, n_tokens,
                   all_align_tokens=None,
                   blank_th=0.4, min_sil_frames=30, min_seg_frames=40,
                   anchor_min_frames=4, min_unit_ratio=0.5, log_callback=None):
    """基于单位时长的 duration-constrained realignment（治段内塌缩）。

    背景：MMS_FA 发射饱和（top1≈1.0），全局单次 CTC 在重复拟声词/匀质段会塌缩成
    1 帧（label 后紧跟 blank，label 只占 1 帧）。切段独立 CTC 在数学上等价于全局，
    无效。真正缺的是**时长先验**：CTC 知道顺序和相对位置，但不知道每个词该多长。

    本函数注入时长先验：
    1) 收集全局 CTC 每词的 (start,end) 帧 + 字符数。
    2) 低频独特词（整轨出现≤2 次）自身 span 不塌缩，其全局时间可信，作锚点；
       由锚点估计单位字符时长 u = median(锚点词时长 / 锚点词字符数)。
    3) 每个词目标时长 = u × 字符数（绝对时长先验）。
    4) 从首词起点起，按目标时长依次单调排布，强制不重叠；
       锚点词起点被钉在 CTC 真实起点（顺序/边界信息保留），其余词相对锚点插值。
    结果：保留 CTC 的相对顺序，恢复绝对时长比例（时长比 0.67→~1.0）。
    完全基于信号自身（不依赖 GT 时间轴）。

    返回 override: list[(adj_start, adj_end)]（调整后帧），未覆盖为 None。
    """
    from collections import Counter
    # 1) 收集每词全局 (start,end) 帧 + 字符数
    g_start, g_end, g_chars = [None] * n_tokens, [None] * n_tokens, [0] * n_tokens
    for i in range(n_tokens):
        if all_align_tokens and i < len(all_align_tokens):
            g_chars[i] = max(1, len(str(all_align_tokens[i])))
        sL = token_spans_L[i] if i < len(token_spans_L) else None
        sR = token_spans_R[i] if i < len(token_spans_R) else None
        best, _ = _pick_best_spans(sL, sR)
        if best:
            g_start[i] = best[0].start
            g_end[i] = best[-1].end
    valid_idx = [i for i in range(n_tokens) if g_start[i] is not None]
    if len(valid_idx) < 4:
        return None

    word_texts = [str(all_align_tokens[i]) if i < len(all_align_tokens) else '' for i in range(n_tokens)]
    cnt = Counter(w for w in word_texts if w)

    # 2) 锚点：低频独特词 + 自身 span 非塌缩（span 帧 / 字符数 ≥ 阈值，说明没被压成 1 帧）
    anchors = []  # (idx, center_frame, duration_frames)
    for i in valid_idx:
        w = word_texts[i]
        if not w or cnt[w] > 2:
            continue
        span = g_end[i] - g_start[i]
        # 锚点只需：低频独特 + 自身 span 达到绝对最小宽（≥anchor_min_frames 帧），
        # 即该词在 CTC 里没有被压成 1 帧——其全局时间可信。
        if span >= anchor_min_frames:
            anchors.append((i, (g_start[i] + g_end[i]) / 2.0, span))

    if len(anchors) < 4:
        msg = (f"[AnchorStretch] 锚点不足({len(anchors)})，跳过长宽恢复"
               f"（独特非塌缩词太少，退化为全局）")
        logger.info(msg)
        if log_callback:
            log_callback(msg)
        return None

    # 3) 估计单位字符时长 u（帧/字符），用锚点词估算
    #    用较高分位（75%）而非 median，避免被轻微塌缩锚点拉低；
    #    只有 ratio 本身较高的锚点才参与，排除仍塌缩的。
    ratios = sorted(a[2] / g_chars[a[0]] for a in anchors)
    if not ratios:
        return None
    u = float(ratios[min(len(ratios) - 1, int(len(ratios) * 0.75))])  # 75 分位
    if u <= 0:
        return None

    # 4) 锚点固定 + 区间内按比例拉伸（治塌缩，不重排锚点）
    #    锚点词保持 CTC 真实 (start,end) 不动；非锚点词位于相邻锚点之间，
    #    按其在 CTC 里的相对顺序（中心帧在 [lo_c, hi_c] 的比例）映射到
    # 4) 最稳健的后处理：中心固定 + 时长先验（治塌缩，绝不移动/爆炸）
    #    每个词的中心 = CTC 中心（起点可信，保留顺序与相对位置）。
    #    词的时长直接用先验 target=u×字符数（CTC 的塌缩时长不可信，弃用），
    #    锚点词也用先验（其 CTC 时长本就接近先验，几乎不变）。
    #    不重排、不移动中心，杜绝越界/累积漂移。
    u = max(u, 4.0)  # 兜底：日语 ASMR ≈ 0.16~0.20s/字 ≈ 4~5 帧@0.04s
    target = [u * g_chars[i] for i in range(n_tokens)]
    override = [None] * n_tokens
    # 只对明显塌缩的词（orig 显著小于 target）拉长，非塌缩/锚点词保持 CTC 原值，
    # 避免一刀切把原本起点准的词也移动、引入新误差。
    raw = {}
    for i in valid_idx:
        orig_dur = g_end[i] - g_start[i]
        c = (g_start[i] + g_end[i]) / 2.0
        # 塌缩判定：原时长不足 target 的 60% 才拉伸
        if orig_dur > 0 and orig_dur < 0.6 * target[i]:
            new_dur = target[i]
        else:
            new_dur = orig_dur
        raw[i] = (c - new_dur / 2.0, c + new_dur / 2.0)
    # 单调收口：从左到右，避免重叠（上一词终点推后本词起点）
    last_end = 0.0
    for i in valid_idx:
        s, e = raw[i]
        if s < last_end:
            shift = last_end - s
            s += shift
            e += shift
        override[i] = (max(0, s) * frame_duration, e * frame_duration)
        last_end = e

    n_ov = sum(1 for x in override if x is not None)
    msg = (f"[AnchorStretch] 锚点{len(anchors)}个，单位时长{u:.1f}f/字，"
           f"覆盖 {n_ov}/{n_tokens} token")
    logger.info(msg)
    if log_callback:
        log_callback(msg)
    return override


def _anchor_realign(emission_L, emission_R, all_token_ids, token_spans_L,
                    token_spans_R, n_tokens, aligner, frame_duration,
                    map_to_original_time, total_frames,
                    max_tokens=40, overlap_tokens=2, pad_frames=12,
                    gap_min_frames=6, log_callback=None):
    g_start, g_end, g_score = [None] * n_tokens, [None] * n_tokens, [0.0] * n_tokens
    valid = [False] * n_tokens
    for i in range(n_tokens):
        sL = token_spans_L[i] if i < len(token_spans_L) else None
        sR = token_spans_R[i] if i < len(token_spans_R) else None
        best, sc = _pick_best_spans(sL, sR)
        if best:
            g_start[i] = best[0].start
            g_end[i] = best[-1].end
            g_score[i] = sc
            valid[i] = True
    valid_idx = [i for i in range(n_tokens) if valid[i]]
    if len(valid_idx) < 4:
        return token_spans_L, token_spans_R

    # 1) 词间空白间隙 → 候选切割点（word i 与 i+1 之间）
    cuts = set()
    for a, b in zip(valid_idx[:-1], valid_idx[1:]):
        if b != a + 1:
            continue
        if g_start[b] - g_end[a] >= gap_min_frames:
            cuts.add(b)
    boundaries = set(cuts)

    # 2) 按自然停顿切出初窗，再对超长窗按内部最大间隙细分（每段 ≤ max_tokens）
    runs = []
    cur = [valid_idx[0]]
    for i in valid_idx[1:]:
        if i in boundaries:
            runs.append(cur)
            cur = [i]
        else:
            cur.append(i)
    runs.append(cur)

    final_runs = []
    for run in runs:
        if len(run) <= max_tokens:
            final_runs.append(run)
            continue
        segs = [run]
        changed = True
        while changed:
            changed = False
            nxt = []
            for seg in segs:
                if len(seg) <= max_tokens:
                    nxt.append(seg)
                    continue
                best_g, best_k = -1, 1
                for k in range(1, len(seg)):
                    ga, gb = seg[k - 1], seg[k]
                    gg = g_start[gb] - g_end[ga]
                    if gg > best_g:
                        best_g, best_k = gg, k
                nxt.append(seg[:best_k])
                nxt.append(seg[best_k:])
                changed = True
            segs = nxt
        final_runs.extend(segs)

    # 3) 逐窗重 CTC（重叠扩展）+ 合并
    #    非重叠词必采窗口结果；重叠词取两窗中置信更高者；未覆盖词回退全局。
    best_L = [None] * n_tokens
    best_R = [None] * n_tokens
    win_sc = [-1e9] * n_tokens
    covered = [False] * n_tokens

    def _shift(spans, f0):
        return [dataclasses.replace(s, start=s.start + f0, end=s.end + f0)
                for s in spans]

    for run in final_runs:
        a0, b0 = run[0], run[-1]
        a = max(0, a0 - overlap_tokens)
        b = min(n_tokens - 1, b0 + overlap_tokens)
        ws = [i for i in range(a, b + 1) if valid[i]]
        if not ws:
            continue
        f0 = max(0, g_start[ws[0]] - pad_frames)
        f1 = min(total_frames, g_end[ws[-1]] + pad_frames)
        if f1 <= f0 + 1:
            continue
        win_ids = all_token_ids[a:b + 1]
        if not any(win_ids):
            continue
        try:
            spL = aligner(emission_L[0, f0:f1], win_ids)
            spR = aligner(emission_R[0, f0:f1], win_ids)
        except Exception as e:
            logger.warning(f"[AnchorRealign] 窗 {a}-{b} 对齐失败跳过: {e}")
            continue
        for off, wi in enumerate(range(a, b + 1)):
            sL = spL[off] if off < len(spL) else None
            sR = spR[off] if off < len(spR) else None
            if not sL and not sR:
                continue
            best_sp, sc = _pick_best_spans(sL, sR)
            if not best_sp:
                continue
            if not covered[wi]:
                best_L[wi] = _shift(sL or [], f0)
                best_R[wi] = _shift(sR or [], f0)
                win_sc[wi] = sc
                covered[wi] = True
            elif sc > win_sc[wi]:
                best_L[wi] = _shift(sL or [], f0)
                best_R[wi] = _shift(sR or [], f0)
                win_sc[wi] = sc

    # 4) 未覆盖词（极端边界情况）回退全局
    for i in range(n_tokens):
        if not covered[i]:
            best_L[i] = token_spans_L[i] if i < len(token_spans_L) else None
            best_R[i] = token_spans_R[i] if i < len(token_spans_R) else None

    msg = (f"[AnchorRealign] 窗 {len(final_runs)} 个 (max_tokens={max_tokens}, "
           f"overlap={overlap_tokens})，覆盖 {sum(covered)}/{n_tokens} token")
    logger.info(msg)
    if log_callback:
        log_callback(msg)
    return best_L, best_R


def _auto_repeat(emission_L, emission_R, lines, aligner, tokenizer,
                 frame_duration, map_to_original_time, format_time,
                 all_token_ids, n_tokens, results,
                 token_spans_L, token_spans_R,
                 original_to_adjusted=None, waveform=None, sample_rate=16000,
                 log_callback=None, auto_repeat_burst=True):
    """检测塌缩的重复型拟声行与短 token 喘息行，自动展开 token 重跑 CTC。

    返回 (n_repeat_lines, n_fixed_total)。
    """
    # 1) 从 results 收集每行的 pred 时长 (秒)
    line_starts = [l["g_start"] for l in lines]
    line_ends = [l["g_end"] for l in lines]
    n_lines = len(lines)
    line_sec = [0.0] * n_lines   # 每行 pred 秒
    for li in range(n_lines):
        g0, g1 = line_starts[li], line_ends[li]
        if g1 <= g0:
            continue
        t0 = t1 = None
        for gi in range(g0, min(g1, n_tokens)):
            sL = token_spans_L[gi] if gi < len(token_spans_L) else None
            sR = token_spans_R[gi] if gi < len(token_spans_R) else None
            best, _ = _pick_best_spans(sL, sR)
            if best:
                fs = best[0].start * frame_duration
                fe = best[-1].end * frame_duration
                if t0 is None or fs < t0:
                    t0 = fs
                if t1 is None or fe > t1:
                    t1 = fe
        if t0 is not None and t1 is not None:
            line_sec[li] = max(0.0, map_to_original_time(t1)
                               - map_to_original_time(t0))

    # 2) 检测重复型行 + 塌缩判据
    repeat_lines = []
    for li in range(n_lines):
        raw = lines[li]["raw"]
        is_rep, n_groups, unit = _is_repeat_line(raw)
        if not is_rep:
            continue
        repeat_lines.append(li)

    if not repeat_lines:
        return 0, 0

    n_fixed = 0
    for li in repeat_lines:
        g0, g1 = line_starts[li], line_ends[li]
        nt = g1 - g0
        if nt <= 0:
            continue
        # 塌缩判据：重复型行的每 token 时长 vs 下一非重复行的每 token 时长比。
        # 若比值 < 0.3 说明该行帧被下游行大量吸走（典型くりくり 422ms/tok
        # 而下游行可能是 2000+ms/tok，比值 0.2）。
        next_nt = next_sec = 0
        for nj in range(li + 1, n_lines):
            if nj in repeat_lines:
                continue
            nxt_gs, nxt_ge = line_starts[nj], line_ends[nj]
            next_nt = nxt_ge - nxt_gs
            next_sec = line_sec[nj]
            break
        dur_per_tok = line_sec[li] / nt * 1000 if nt > 0 else 0
        next_dur_per_tok = next_sec / next_nt * 1000 if next_nt > 0 else 0
        ratio = dur_per_tok / next_dur_per_tok if next_dur_per_tok > 0 else 1.0
        logger.info(f" [AutoRepeat] 行{li}: nt={nt} pred={line_sec[li]:.2f}s "
                    f"({dur_per_tok:.0f}ms/tok) next={next_nt}t/{next_sec:.2f}s "
                    f"({next_dur_per_tok:.0f}ms/tok) ratio={ratio:.2f} "
                    f"tokens={lines[li]['tokens'][:8]}...")
        if ratio >= 0.6:
            continue

        # 3) 下游 "联合重对齐"：把塌缩行 (K-exp) + 下行 拼在一起跑 CTC
        line_ids = all_token_ids[g0:g1]
        if not line_ids or not any(line_ids):
            continue
        # 下一非重复行
        _nxt = li + 1
        while _nxt < n_lines and _nxt in repeat_lines:
            _nxt += 1
        nxt_ids = all_token_ids[line_starts[_nxt]:line_ends[_nxt]] if _nxt < n_lines and line_starts[_nxt] < n_tokens else None
        if nxt_ids is None or not nxt_ids or not any(nxt_ids):
            continue
        rs = results[g0].get("original_start")
        nxt_old_end = results[min(line_ends[_nxt]-1, n_tokens-1)].get("original_end", rs+5)
        if rs is None or rs == "[error]" or nxt_old_end == "[error]":
            continue

        # 窗口: 塌缩行 start → 下行 end+1s
        pad_f = int(1.0 / frame_duration)
        max_frames = emission_L.shape[1]
        if original_to_adjusted is None:
            f0 = max(0, int(rs / frame_duration) - pad_f // 2)
            f1 = min(max_frames, int(nxt_old_end / frame_duration) + pad_f)
        else:
            f0 = max(0, int(original_to_adjusted(rs) / frame_duration) - pad_f // 2)
            f1 = min(max_frames, int(original_to_adjusted(nxt_old_end) / frame_duration) + pad_f)
        if f1 <= f0 + 4:
            continue



        # 试几个 K，塌缩行 K-exp token + 下行原始 token，CTC 看总 span
        best_K, best_score = 1, 0.0
        for K in range(2, 11):
            joint_ids = line_ids * K + nxt_ids
            try:
                sp = aligner(emission_L[0, f0:f1, :], joint_ids)
            except Exception:
                continue
            if not sp:
                continue
            # 得分：塌缩行最后一个 token 的 end vs 下行第一个 token 的 start 间隙
            coll_end = sp[K * nt - 1][-1].end if K * nt <= len(sp) and sp[K*nt-1] else None
            nxt_s = sp[K * nt][0].start if K * nt < len(sp) and sp[K*nt] else None
            if coll_end is None or nxt_s is None:
                continue
            if not sp[-1] or not sp[0]:
                continue
            gap = nxt_s - coll_end
            total_width = (sp[-1][-1].end - sp[0][0].start) * frame_duration
            # 最优策略：间隙尽量小（CTC 自然边界紧）且总跨度不过大
            score = total_width / (1 + gap)
            if score > best_score:
                best_score, best_K = score, K

        if best_K <= 1:
            continue

        joint_ids = line_ids * best_K + nxt_ids
        try:
            sp = aligner(emission_L[0, f0:f1, :], joint_ids)
        except Exception:
            continue
        if not sp:
            continue

        total = len(joint_ids)
        first_s_adj = sp[0][0].start + f0
        last_e_adj = sp[-1][-1].end + f0

        # 塌缩行：取复制 token 的首次出现 start / 末次出现 end
        rc_start = first_s_adj * frame_duration
        rc_end = sp[best_K * nt - 1][-1].end + f0
        rc_end = rc_end * frame_duration

        # 下行：取从第 (best_K*nt) 个 token 起的 start / end
        nx_start = sp[best_K * nt][0].start + f0 if best_K * nt < len(sp) and sp[best_K*nt] else rc_end + f0
        nx_start *= frame_duration
        nx_end = last_e_adj * frame_duration

        os_prev = map_to_original_time(rc_start)
        oe_prev = map_to_original_time(rc_end)
        os_nxt  = map_to_original_time(nx_start)
        oe_nxt  = map_to_original_time(nx_end)

        logger.info(f" [Joint] 行{li}+{_nxt}: K={best_K} "
                    f"new={os_prev:.1f}-{oe_prev:.1f}s / "
                    f"{os_nxt:.1f}-{oe_nxt:.1f}s {lines[li]['tokens'][:6]}")

        # 回填塌缩行
        for gi in range(g0, min(g1, n_tokens)):
            r = results[gi]
            frac_start = (gi - g0) / nt; frac_end = (gi - g0 + 1) / nt
            r["original_start"] = os_prev + frac_start * (oe_prev - os_prev)
            r["original_end"]   = os_prev + frac_end   * (oe_prev - os_prev)
            r["start"] = format_time(r["original_start"])
            r["end"]   = format_time(r["original_end"])
        # 回填下行
        _n0, _n1 = line_starts[_nxt], line_ends[_nxt]
        _n_nt = _n1 - _n0
        for gi in range(_n0, min(_n1, n_tokens)):
            r = results[gi]
            frac_start = (gi - _n0) / _n_nt; frac_end = (gi - _n0 + 1) / _n_nt
            r["original_start"] = os_nxt + frac_start * (oe_nxt - os_nxt)
            r["original_end"]   = os_nxt + frac_end   * (oe_nxt - os_nxt)
            r["start"] = format_time(r["original_start"])
            r["end"]   = format_time(r["original_end"])
        n_fixed += 1

    # 3) Burst-line 扩展：非重复型的短 token 行（喘息/轻哼）在 CTC 下极易塌缩，
    #    导致下游行吸入空洞 → 大幅偏移。对这些行做 token 重复扩展（类 auto_repeat
    #    但触发条件改为 per-token duration < 阈值，不要求 repeat 模式）。
    #    实验结论（2026-08-09）：全指标退步(±200ms 71.3→67.8%), 因非重复行的
    #    盲扩 token 序列与实际持続音不匹配。保留代码但默认不开启。
    burst_fixed = 0
    if auto_repeat_burst:
        for li in range(n_lines):
            if li in repeat_lines:
                continue
            g0, g1 = line_starts[li], line_ends[li]
            nt = g1 - g0
            if nt <= 0 or nt > 6:
                continue
            dur = line_sec[li]
            if dur <= 0:
                continue
            dur_per_tok = dur / nt * 1000
            if dur_per_tok >= 80:
                continue

            line_ids = all_token_ids[g0:g1]
            if not line_ids or not any(line_ids):
                continue

            rs = results[g0].get("original_start")
            re_val = results[min(g1 - 1, n_tokens - 1)].get("original_end", rs + 2) if g1 > 0 else rs + 2
            if rs is None or rs == "[error]" or re_val == "[error]" or re_val is None:
                continue

            pad_f = int(1.0 / frame_duration)
            max_frames = emission_L.shape[1]
            if original_to_adjusted is None:
                f0 = max(0, int(rs / frame_duration) - pad_f // 2)
                f1 = min(max_frames, int(re_val / frame_duration) + pad_f + int(5.0 / frame_duration))
            else:
                f0 = max(0, int(original_to_adjusted(rs) / frame_duration) - pad_f // 2)
                f1 = min(max_frames, int(original_to_adjusted(re_val) / frame_duration) + pad_f + int(5.0 / frame_duration))
            if f1 <= f0 + 4:
                continue

            best_K, best_width = 1, 0.0
            for K in range(2, 8):
                expanded_ids = line_ids * K
                try:
                    sp = aligner(emission_L[0, f0:f1, :], expanded_ids)
                except Exception:
                    continue
                if not sp or not sp[0] or not sp[-1]:
                    continue
                w = (sp[-1][-1].end - sp[0][0].start) * frame_duration
                if K > 2 and best_width > 0 and w < best_width * 1.15:
                    break
                if w > best_width:
                    best_width, best_K = w, K

            if best_K <= 1:
                continue

            expanded_ids = line_ids * best_K
            try:
                sp = aligner(emission_L[0, f0:f1, :], expanded_ids)
            except Exception:
                continue
            if not sp or not sp[0] or not sp[-1]:
                continue

            total_exp = len(expanded_ids)
            first_s_adj = sp[0][0].start + f0
            first_e_adj = sp[total_exp - 1][-1].end + f0
            os_burst = map_to_original_time(first_s_adj * frame_duration)
            oe_burst = map_to_original_time(first_e_adj * frame_duration)

            for gi in range(g0, min(g1, n_tokens)):
                r = results[gi]
                frac_start = (gi - g0) / nt
                frac_end = (gi - g0 + 1) / nt
                r["original_start"] = os_burst + frac_start * (oe_burst - os_burst)
                r["original_end"]   = os_burst + frac_end   * (oe_burst - os_burst)
                r["start"] = format_time(r["original_start"])
                r["end"]   = format_time(r["original_end"])

            logger.info(f" [Burst] 行{li}: K={best_K} nt={nt} {dur:.2f}s→{(oe_burst - os_burst):.2f}s "
                        f"({dur_per_tok:.0f}→{(oe_burst - os_burst)/nt*1000:.0f}ms/tok) "
                        f"tokens={lines[li]['tokens'][:8]}...")
            burst_fixed += 1

    return len(repeat_lines), n_fixed + burst_fixed
