# -*- coding: utf-8 -*-
"""FA-ASMR V2 — 多流分流独立后处理与时间轴归并。"""
from __future__ import annotations

import copy
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

from fa_asmr.aligners.align_utils import _format_time
import fa_asmr.text.clause as _clause_mod
from fa_asmr.text.clause import F
from fa_asmr.pipeline.subtitles import _parse_lrc_ts_sec

# ══════════════════════════════════════════════════════════════════════════
#  4. 分流独立后处理与时间轴归并
# ══════════════════════════════════════════════════════════════════════════


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
            if "_mp_home" in r and 0 <= ri < len(result_list):
                result_list[ri]["_mp_home"] = bool(r["_mp_home"])
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
            from fa_asmr.text.ono_filter import mark_ono_candidates as _ono_mark
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
            from fa_asmr.text.resegment import apply_punctuation_resegment as _reseg
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
            from fa_asmr.text.merge_short import merge_short_lines as _merge
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
            from fa_asmr.text.ono_filter import decide_ono_deletions as _ono_decide
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
        from fa_asmr.text.ono_filter import is_easy_ono as _is_easy_ono_fn
        from fa_asmr.text.ono_block_corpus import is_blocked as _is_blocked_fn
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

    from fa_asmr.text.merge_short import append_tokens_with_space as _append_with_space

    def _blk_spk(blk: List[Dict[str, Any]]) -> Optional[Tuple[str, ...]]:
        for x in blk:
            if not x.get(F.ONO_DROPPED) and x.get(F.SPK):
                return tuple(x[F.SPK])
        return None

    _TERMINAL_ENDS = frozenset("。！？!?♪♡」』）)")

    def _ends_unfinished(blk: List[Dict[str, Any]]) -> bool:
        """判断当前片段是否以非句末终结符收尾（如省略号/顿号/无标点，适合慢语速主谓顺接）。"""
        tail = ""
        for x in reversed(blk):
            if x.get(F.ONO_DROPPED) or x.get(F.RUBY_PLACEHOLDER) or x.get("orig") == "\n":
                continue
            s = str(_clause_mod.src_text(x)).strip()
            if s:
                tail = s[-1]
                break
        return bool(tail) and (tail not in _TERMINAL_ENDS)

    def _other_stream_intervenes(sid_self: int, t0: float, t1: float) -> bool:
        """检查 (t0, t1) 空隙期间是否有其他角色独立开口插话（避免跨别人台词硬合并）。"""
        if t1 <= t0 + 0.10:
            return False
        for o_sid, o_list in by_stream.items():
            if o_sid == sid_self:
                continue
            for o_st, o_en, _, _, _, _, _ in o_list:
                if o_st >= t0 - 0.05 and o_st <= t1 - 0.10:
                    return True
                if o_st > t1:
                    break
        return False

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
                # 4) ASMR 慢语速语义未完结短前缀（<= 8 字且非 。！？ 结尾，或同台本行未完结），在无其他角色插话时放宽至 gap <= 1.65s；
                #    若为极短悬念短语（<= 6 字且合后 <= 18 字），容忍至 gap <= 2.25s（如「それまで……」「ガマン……」「私も……」「ガマン、するから……」）
                is_unfin = _ends_unfinished(blk_merged)
                no_other_int = (gap_cn <= 2.25 and is_unfin and not _other_stream_intervenes(sid, en_c, st_n))
                slow_unfinished_merge = no_other_int and (
                    (gap_cn <= 1.65 and (vis_merged + vis_n) <= 32 and (vis_merged <= 8 or lg_n == last_lg))
                    or (gap_cn <= 2.25 and vis_merged <= 6 and (vis_merged + vis_n) <= 18)
                )
                can_merge = (
                    (gap_cn <= 0.65)
                    or (lg_n == last_lg and gap_cn <= 0.85)
                    or ((is_ov_c or is_ov_n) and gap_cn <= 0.85)
                    or ((vis_n <= 4 or dur_n <= 1.0) and gap_cn <= 1.10)
                    or slow_unfinished_merge
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
                    dt_st <= 1.25
                    and (
                        (dt_st <= 0.45 and overlap >= 0.0)
                        or (overlap >= 0.40 and (overlap / min_dur) >= 0.40)
                        or (overlap >= 1.20 and (overlap / min_dur) >= 0.30)
                    )
                )
                # 嵌套包容重叠：后句被前句包在内部，但排除较晚冒出(>0.80s)的极短碎插话强行揉进长句
                is_nested_inside = (
                    (en_b <= cur_en + 0.25)
                    and (cur_en - st_b) >= 1.50
                    and (overlap / dur_b) >= 0.85
                    and not (vis_b <= 4 and dt_st > 0.80)
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


