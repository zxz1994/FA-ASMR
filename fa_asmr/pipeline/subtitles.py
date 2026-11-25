# -*- coding: utf-8 -*-
"""FA-ASMR V2 — LRC / SRT 字幕格式化、齐声合并与行首孤立标点清理。

从 fa_asmr_pipeline.py 与 align_multispeaker.py 抽离的无界面字幕导出层：
- LRC 时间戳转换与标准 LRC 组装 (build_standard_lrc / _merge_unison_lrc_groups)
- SRT 时间戳与 Cue 组装、多角色独立时间流 SRT 生成 (build_independent_multispeaker_srt / build_standard_srt)
- 同步齐声前后缀只读拆分 (split_unison_prefix_suffix_rl)
- 统一的行首孤立标点清理 (_strip_orphan_punct)
"""
from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import fa_asmr.aligners.align_utils as au
import fa_asmr.text.clause as _clause_mod
from fa_asmr.text.clause import F, is_text_token as _is_text_token


# 行首孤立标点剔除：拟声子句被删后，附着其上的后置标点（、。…?！等）
# 可能残留成行首「、んん」「? 何にって」，读起来是空壳。
#
# 只剔除【行首】——行尾标点是日语的句末终结符（。？！…），必须保留，
# 否则 `だけど。` / `でしょ?` 会被削成 `だけど` / `でしょ`（曾误伤过）。
# 也不含表达性符号：ー/~（长音）、♪♡（语气）、「」『』（）（引用）。
_PUNCT_CHARS = "、。！？…‥・,.:;!?"


def _strip_orphan_punct(text: str) -> str:
    """去掉【行首】的孤立标点（行尾标点保留，它们是句末终结符）。"""
    s = text.lstrip()
    i = 0
    while i < len(s) and s[i] in _PUNCT_CHARS:
        i += 1
    return s[i:].lstrip()


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


def _fmt_lrc_time(t):
    """把 [MM:SS:CC]（align_utils._format_time 产出）转成 [MM:SS.CC]。

    原为 build_standard_lrc._fmt_time 与 _fmt_line._t 两份完全相同的实现，
    现统一为此函数。两者原先都用 re.match（锚定行首），此处保持一致。

    历史教训（两处曾各自失配，均造成严重后果）：
      - 曾误写成两段式 \\[\\d{2}:\\d{2}\\] 却访问 group(3)，恒不匹配 ->
        返回 None -> 结束标记行被静默丢弃，行尾延长一并失效。
      - 分钟位曾写死 \\d{2}，而 _format_time 的 f"{mins:02d}" 不截断，
        mins >= 100（音频 >= 100 分钟）时产出 3 位 -> 整条字幕退化成
        无时间戳的裸文本。
    故正则统一引用 align_utils.LRC_TS_ANCHORED，不再各自维护字面。
    """
    if not t or t == "[error]":
        return None
    m = au.LRC_TS_ANCHORED.match(t)
    return f"[{m.group(1)}:{m.group(2)}.{m.group(3)}]" if m else None


def _lrc_time_to_cs(t):
    if not t or t == "[error]":
        return None
    m = au.LRC_TS_ANCHORED.match(t)
    if not m:
        m = re.match(r"\[(\d+):(\d+)[.:](\d+)\]", str(t))
    if not m:
        return None
    return (int(m.group(1)) * 60 + int(m.group(2))) * 100 + int(m.group(3))


def _cs_to_raw_lrc_ts(cs: int) -> str:
    cs_i = max(0, int(cs))
    return f"[{cs_i // 6000:02d}:{(cs_i // 100) % 60:02d}:{cs_i % 100:02d}]"


def _merge_unison_lrc_groups(groups: list) -> list:
    """LRC 组装出口统一执行「1 对多（保留较细分段）+ 1 对 1」同步齐声说话人合并。

    入参 groups 每项为 [start_cs, end_cs, parts]，
    其中 parts 每项为 [spk_tuple_or_None, seg_text, seg_st_cs, seg_en_cs]。
    """
    if not groups:
        return groups

    def _clean_spks(spk):
        if not spk:
            return []
        return [s.lstrip("　 ") for s in spk if s and s.lstrip("　 ")]

    def _strip_outer_q(seg: str) -> str:
        t = seg.strip()
        for oq, cq in (("「", "」"), ("『", "』")):
            if len(t) >= 2 and t.startswith(oq) and t.endswith(cq):
                inner = t[1:-1].strip()
                if inner and oq not in inner and cq not in inner:
                    return inner
        return t

    def _norm_seg(seg: str) -> str:
        clean = re.sub(r"\{([^|]+)\|[^}]+\}", r"\1", seg).rstrip()
        clean = _strip_orphan_punct(clean)
        clean = _strip_outer_q(clean)
        return re.sub(r"[\s\u3000]+", "", clean)

    def _prefer_unquoted_seg(s1: str, s2: str) -> str:
        c1 = re.sub(r"\{([^|]+)\|[^}]+\}", r"\1", s1).strip()
        c2 = re.sub(r"\{([^|]+)\|[^}]+\}", r"\1", s2).strip()
        if (_strip_outer_q(c1) != c1) and (_strip_outer_q(c2) == c2):
            return s2
        return s1

    def _make_spk_tuple(names: list, lead_space: bool):
        if not names:
            return None
        if lead_space:
            return ("　" + names[0],) + tuple(names[1:])
        return tuple(names)

    def _merge_names(a: list, b: list, b_first: bool = False) -> list:
        res = list(b) if b_first else list(a)
        for s in (a if b_first else b):
            if s not in res:
                res.append(s)
        return res

    def _recalc_group_bounds(g) -> None:
        sts = [p[2] for p in g[2] if p[2] is not None]
        ens = [p[3] for p in g[2] if p[3] is not None]
        if sts:
            g[0] = min(sts)
        if ens:
            g[1] = max(ens)
        if g[1] < g[0]:
            g[1] = g[0]

    def _normalize_first_part_space(g) -> None:
        if g[2] and g[2][0][0]:
            names = _clean_spks(g[2][0][0])
            g[2][0][0] = _make_spk_tuple(names, lead_space=False)

    # 快速路径：整轨无说话人标记时直接跳过
    if not any(p[0] for g in groups for p in g[2]):
        return groups

    work = [[g[0], g[1], [list(p) for p in g[2]]] for g in groups]

    # Step 1A: 行内并发拼行中「1 条长整句 vs 本行首段 + 后续 1..2 行续段」（方案 A：保留较细分段）
    for i, g in enumerate(work):
        parts = g[2]
        if len(parts) < 2:
            continue
        removed_p = set()
        for pi_long, p_long in enumerate(parts):
            if pi_long in removed_p:
                continue
            spks_l = _clean_spks(p_long[0])
            norm_l = _norm_seg(p_long[1])
            if not spks_l or not norm_l:
                continue
            long_st = p_long[2] if p_long[2] is not None else g[0]
            long_en = p_long[3] if p_long[3] is not None else g[1]
            for pi_sh, p_sh in enumerate(parts):
                if pi_sh == pi_long or pi_sh in removed_p:
                    continue
                spks_s = _clean_spks(p_sh[0])
                norm_s = _norm_seg(p_sh[1])
                if not spks_s or not norm_s or any(s in spks_l for s in spks_s):
                    continue
                if not norm_l.startswith(norm_s) or norm_s == norm_l:
                    continue
                # 沿后续行查找同一说话人的 1..2 个单段续行
                acc = norm_s
                cont_rows = []
                for k in range(i + 1, min(len(work), i + 8)):
                    gk = work[k]
                    if gk[0] > long_en + 45:
                        break
                    if len(gk[2]) != 1:
                        continue
                    spks_k = _clean_spks(gk[2][0][0])
                    if spks_k != spks_s:
                        continue
                    norm_k = _norm_seg(gk[2][0][1])
                    if not norm_k:
                        continue
                    acc += norm_k
                    cont_rows.append(k)
                    if not norm_l.startswith(acc) or len(cont_rows) >= 3:
                        break
                    if acc == norm_l:
                        last_en = gk[2][0][3] if gk[2][0][3] is not None else gk[1]
                        if abs(last_en - long_en) <= 85:
                            sh_st = p_sh[2] if p_sh[2] is not None else g[0]
                            l_first = long_st < sh_st
                            merged_names = _merge_names(spks_s, spks_l, b_first=l_first)
                            p_sh[0] = _make_spk_tuple(merged_names, lead_space=(pi_sh > 0))
                            if p_sh[2] is not None and long_st is not None:
                                p_sh[2] = min(p_sh[2], long_st)
                            for rk in cont_rows:
                                work[rk][2][0][0] = _make_spk_tuple(merged_names, lead_space=False)
                            last_g = work[cont_rows[-1]]
                            last_g[1] = max(last_g[1], long_en)
                            if last_g[2][0][3] is not None:
                                last_g[2][0][3] = max(last_g[2][0][3], long_en)
                            removed_p.add(pi_long)
                        break
                if pi_long in removed_p:
                    break
        if removed_p:
            g[2] = [p for idx_p, p in enumerate(parts) if idx_p not in removed_p]
            _normalize_first_part_space(g)
            _recalc_group_bounds(g)

    # Step 1B: 独立成行的「1 条长整句 vs 另一角色连续 2..3 行拆句」（方案 A：保留较细分段）
    dropped = [False] * len(work)
    single_by_spk = {}
    for idx_g, g in enumerate(work):
        if len(g[2]) == 1:
            spks = tuple(_clean_spks(g[2][0][0]))
            if spks:
                single_by_spk.setdefault(spks, []).append(idx_g)

    for i, g in enumerate(work):
        if dropped[i] or len(g[2]) != 1:
            continue
        spks_l = _clean_spks(g[2][0][0])
        norm_l = _norm_seg(g[2][0][1])
        if not spks_l or not norm_l:
            continue
        st_l, en_l = g[0], g[1]
        matched_chain = None
        for other_key, o_idxs in single_by_spk.items():
            if any(s in spks_l for s in other_key):
                continue
            for pos, first_idx in enumerate(o_idxs):
                if dropped[first_idx]:
                    continue
                st_first = work[first_idx][0]
                if st_first < st_l - 45:
                    continue
                if st_first > st_l + 45:
                    break
                acc = _norm_seg(work[first_idx][2][0][1])
                if not acc or not norm_l.startswith(acc) or acc == norm_l:
                    continue
                chain = [first_idx]
                for step in range(1, 3):
                    if pos + step >= len(o_idxs):
                        break
                    nxt_idx = o_idxs[pos + step]
                    if dropped[nxt_idx]:
                        break
                    acc += _norm_seg(work[nxt_idx][2][0][1])
                    chain.append(nxt_idx)
                    if not norm_l.startswith(acc):
                        break
                    if acc == norm_l:
                        if abs(work[nxt_idx][1] - en_l) <= 85:
                            matched_chain = chain
                        break
                if matched_chain is not None:
                    break
            if matched_chain is not None:
                break
        if matched_chain is not None:
            dropped[i] = True
            first_i, last_i = matched_chain[0], matched_chain[-1]
            l_first = st_l < work[first_i][0]
            for ci_m in matched_chain:
                cur_s = _clean_spks(work[ci_m][2][0][0])
                merged_names = _merge_names(cur_s, spks_l, b_first=l_first)
                work[ci_m][2][0][0] = _make_spk_tuple(merged_names, lead_space=False)
            work[first_i][0] = min(work[first_i][0], st_l)
            work[last_i][1] = max(work[last_i][1], en_l)

    stage2 = [g for idx_g, g in enumerate(work) if not dropped[idx_g] and g[2]]

    # Step 2A: 行内「1 对 1」同文段合并
    for g in stage2:
        parts = g[2]
        if len(parts) < 2:
            continue
        out_parts = []
        for p in parts:
            spks_p = _clean_spks(p[0])
            norm_p = _norm_seg(p[1])
            merged = False
            if spks_p and norm_p:
                for prev in out_parts:
                    spks_prev = _clean_spks(prev[0])
                    if spks_prev and not any(s in spks_prev for s in spks_p) and _norm_seg(prev[1]) == norm_p:
                        merged_names = _merge_names(spks_prev, spks_p, b_first=False)
                        prev[0] = _make_spk_tuple(merged_names, lead_space=(prev is not out_parts[0]))
                        prev[1] = _prefer_unquoted_seg(prev[1], p[1])
                        if p[2] is not None:
                            prev[2] = p[2] if prev[2] is None else min(prev[2], p[2])
                        if p[3] is not None:
                            prev[3] = p[3] if prev[3] is None else max(prev[3], p[3])
                        merged = True
                        break
            if not merged:
                out_parts.append(p)
        g[2] = out_parts
        _normalize_first_part_space(g)
        _recalc_group_bounds(g)

    # Step 2B: 跨行「1 对 1」同步齐声合并
    stage2.sort(key=lambda x: (x[0], x[1]))
    final_groups = []
    used = [False] * len(stage2)
    for i, g in enumerate(stage2):
        if used[i]:
            continue
        used[i] = True
        if len(g[2]) != 1:
            final_groups.append(g)
            continue
        spks_i = _clean_spks(g[2][0][0])
        norm_i = _norm_seg(g[2][0][1])
        if not spks_i or not norm_i:
            final_groups.append(g)
            continue
        g_st, g_en = g[0], g[1]
        g_spks = list(spks_i)
        for j in range(i + 1, len(stage2)):
            if used[j]:
                continue
            gj = stage2[j]
            if gj[0] - g_st > 45:
                break
            if len(gj[2]) != 1:
                continue
            spks_j = _clean_spks(gj[2][0][0])
            if not spks_j or any(s in g_spks for s in spks_j):
                continue
            if _norm_seg(gj[2][0][1]) != norm_i:
                continue
            ov = min(g_en, gj[1]) - max(g_st, gj[0])
            min_dur = max(5, min(g_en - g_st, gj[1] - gj[0]))
            if ov >= 0.60 * min_dur:
                used[j] = True
                g_st = min(g_st, gj[0])
                g_en = max(g_en, gj[1])
                g_spks = _merge_names(g_spks, spks_j, b_first=False)
                g[2][0][1] = _prefer_unquoted_seg(g[2][0][1], gj[2][0][1])
        g[0], g[1] = g_st, g_en
        g[2][0][0] = _make_spk_tuple(g_spks, lead_space=False)
        final_groups.append(g)

    final_groups.sort(key=lambda x: (x[0], x[1]))
    return final_groups


def build_standard_lrc(result_list):
    if result_list and any(it.get(F.SPK) for it in result_list):
        result_list = split_unison_prefix_suffix_rl(result_list)

    raw_groups = []
    min_start_cs = None
    max_end_cs = None
    # 字幕行按【说话人段】收集：同一行内说话人变化处补该说话人的「②：」前缀。
    # 结构 = [[说话人元组或None, 文本, 段起cs, 段止cs], ...]。无标记作品恒为单段 None，输出不变。
    group_parts = []

    def _flush():
        if group_parts and any(p[1] for p in group_parts) and min_start_cs is not None:
            best_end_cs = max_end_cs if max_end_cs is not None else min_start_cs
            # 若 max_end 比 min_start 还要早（防御极端声学错位），保证 end >= start
            if best_end_cs < min_start_cs:
                best_end_cs = min_start_cs
            raw_groups.append([min_start_cs, best_end_cs, [list(p) for p in group_parts]])

    for item in result_list:
        io = item.get("orig", "")
        # 输出层还原：优先用台本原文子串（_src）。orig 是归一化后的字符
        # （NFKC 把 ① 拍平成 1 等），_src 保留原字符，仅换显示、不动对齐标签。
        io_out = item.get(F.SRC, None)
        if io_out is None:
            io_out = io
        if item.get("type") == 0 and io == "\n":
            _flush()
            min_start_cs = None
            max_end_cs = None
            group_parts = []
            continue
        # 被拟声守门删除的 token：不贡献时间边界，也不进入字幕文本
        if item.get(F.ONO_DROPPED):
            continue
        # 说话人切换：本 token 带 _spk 且与当前段不同 -> 开新段（行内补前缀）。
        # 无 _spk 的 token 沿用当前段（无标记作品全程如此）。
        _spk = tuple(item[F.SPK]) if item.get(F.SPK) else None
        if _spk is not None and (not group_parts or group_parts[-1][0] != _spk):
            group_parts.append([_spk, "", None, None])
        if not group_parts:
            group_parts.append([None, "", None, None])
        st = item.get("start")
        if st:
            cs_s = _lrc_time_to_cs(st)
            if cs_s is not None:
                if min_start_cs is None or cs_s < min_start_cs:
                    min_start_cs = cs_s
                if group_parts[-1][2] is None or cs_s < group_parts[-1][2]:
                    group_parts[-1][2] = cs_s
        en = item.get("end")
        if en:
            cs_e = _lrc_time_to_cs(en)
            if cs_e is not None:
                if max_end_cs is None or cs_e > max_end_cs:
                    max_end_cs = cs_e
                if group_parts[-1][3] is None or cs_e > group_parts[-1][3]:
                    group_parts[-1][3] = cs_e
        if _is_text_token(item) and io:
            if not item.get(F.RUBY_PLACEHOLDER, False):
                group_parts[-1][1] += io_out
        elif item.get("type") == 0 and io not in ("\n", ""):
            group_parts[-1][1] += io_out

    _flush()

    merged_groups = _merge_unison_lrc_groups(raw_groups)
    groups = [
        (_cs_to_raw_lrc_ts(g[0]), _cs_to_raw_lrc_ts(g[1]), [(p[0], p[1]) for p in g[2]])
        for g in merged_groups
    ]

    lines = []
    active_max_end_cs = None
    for idx, (g_start, g_end, g_parts) in enumerate(groups):
        eff_end = g_end
        cs_start = _lrc_time_to_cs(g_start)
        cs_end = _lrc_time_to_cs(eff_end)
        cs_next = _lrc_time_to_cs(groups[idx + 1][0]) if idx + 1 < len(groups) else None

        # 与 SRT 对齐最低 1.0s 驻留时长（仅当后续存在真实空白时补足，避免 0.12s 短句闪现即消失）
        if cs_start is not None and (cs_end is None or cs_end < cs_start + 100):
            target_cs = cs_start + 100
            if cs_next is not None:
                target_cs = min(target_cs, cs_next)
            if cs_end is None or target_cs > cs_end:
                cs_end = max(cs_start, target_cs)
                eff_end = f"[{cs_end // 6000:02d}:{(cs_end // 100) % 60:02d}:{cs_end % 100:02d}]"

        # 追踪跨行全局活跃最大结束时间，防止前一句仍在持续时因较短的嵌套插话先结束而提前输出空白清屏行
        if cs_end is not None:
            if active_max_end_cs is None or cs_end > active_max_end_cs:
                active_max_end_cs = cs_end
            clear_cs = max(cs_end, active_max_end_cs)
            if cs_next is not None and clear_cs > cs_next:
                clear_cs = cs_next
            cs_end = clear_cs
            eff_end = f"[{cs_end // 6000:02d}:{(cs_end // 100) % 60:02d}:{cs_end % 100:02d}]"

        lines.append(_fmt_line(g_start, eff_end, g_parts))
        if eff_end and eff_end != g_start:
            # 仅当所有已激活语句全部结束且下一行之前存在真实停顿（或为末行）时，输出空白下轴时间戳
            has_gap = cs_next is None or (cs_end is not None and cs_end < cs_next)
            if has_gap:
                t = _fmt_lrc_time(eff_end)
                if t:
                    lines.append(t)
    return "\n".join(lines)


def _fmt_line(start_str, end_str, text, speakers=None):
    """组装一行字幕。text 可为字符串（旧式）或 [(说话人元组|None, 文本), ...]。

    逐段处理：每段先剥 ruby 与行首孤立标点，再在段前加「说话人：」前缀。
    行内说话人切换因此表现为 B 段前多一个「②：」，而非另起一行。
    多人并列用 、 连接。无标记作品只有一段 None，输出与原来完全一致。
    """
    if isinstance(text, str):
        parts = [(speakers, text)]
    else:
        parts = [(spk, seg) for spk, seg in text]
    buf = []
    first = True
    for spk, seg in parts:
        if not seg:
            continue
        clean = re.sub(r"\{([^|]+)\|[^}]+\}", r"\1", seg).rstrip()
        if first:
            clean = _strip_orphan_punct(clean)
            first = False
        if spk:
            names = "\u3001".join(s for s in spk if s)
            if names:
                clean = f"{names}\uff1a{clean}"
        buf.append(clean)
    clean = "".join(buf)
    t = _fmt_lrc_time(start_str)
    return f"{t}{clean}" if t else clean


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


_OUTER_QUOTE_PAIRS = (("「", "」"), ("『", "』"))
_SPLIT_BOUNDARY_PUNCT = set("。！？!?、,…♡♪」』）)")
_OPEN_QUOTE_CHARS = set("「『（(")
_UNISON_WORD_RE = re.compile(r"[ぁ-んァ-ヶー一-龯々A-Za-z0-9０-９]")


def _strip_outer_quotes_text(text: str) -> str:
    t = text.strip()
    while len(t) >= 2:
        changed = False
        for oq, cq in _OUTER_QUOTE_PAIRS:
            if t.startswith(oq) and t.endswith(cq):
                inner = t[1:-1].strip()
                if inner and oq not in inner and cq not in inner:
                    t = inner
                    changed = True
                    break
        if not changed:
            break
    return t


def _norm_unison_text(text: str) -> str:
    """齐声正文归一化：去注音壳、去行首孤立标点、去外层成对引号与全/半角空白。"""
    t = re.sub(r"\{([^|]+)\|[^}]+\}", r"\1", text).strip()
    t = _strip_orphan_punct(t)
    t = _strip_outer_quotes_text(t)
    return re.sub(r"[\s\u3000]+", "", t)


def _tokens_time_bounds(tokens: Sequence[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
    sts: List[float] = []
    ens: List[float] = []
    for x in tokens:
        if x.get(F.ONO_DROPPED) or x.get(F.RUBY_PLACEHOLDER):
            continue
        ts = _parse_lrc_ts_sec(x.get("start"))
        te = _parse_lrc_ts_sec(x.get("end"))
        if ts is not None:
            sts.append(ts)
        if te is not None:
            ens.append(te)
    if not sts:
        return None, None
    st = min(sts)
    en = max(ens) if ens else st
    return st, max(st, en)


def _tokens_display_text(tokens: Sequence[Dict[str, Any]]) -> str:
    out: List[str] = []
    for x in tokens:
        if x.get(F.ONO_DROPPED) or x.get(F.RUBY_PLACEHOLDER) or x.get("orig") == "\n":
            continue
        out.append(str(_clause_mod.src_text(x)))
    return "".join(out)


def split_unison_prefix_suffix_rl(result_list: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """组装层纯只读预处理：利用逐 token 真实时间戳，在子句/引号边界拆分「单人前/后缀 + 同步齐声段」。

    不触碰对齐与 merge_short 核心；切分后的两段均保留各自原始逐 token 时间戳，
    随后交给 SRT/LRC 组装层的 1 对 1 齐声合并自动合出双人前缀。
    """
    if not result_list or not any(it.get(F.SPK) for it in result_list):
        return list(result_list)

    # 1. 先按 \n 拆行，再按行内 F.SPK 切换拆成独立说话人段（兼容 LRC 行内并发段与 SRT 单段行）
    rows: List[List[Dict[str, Any]]] = []
    cur_row: List[Dict[str, Any]] = []
    for it in result_list:
        if it.get("type") == 0 and it.get("orig") == "\n":
            if cur_row:
                rows.append(cur_row)
                cur_row = []
        else:
            cur_row.append(it)
    if cur_row:
        rows.append(cur_row)

    def _clean_spks(spk_t) -> Tuple[str, ...]:
        if not spk_t:
            return ()
        return tuple(s.lstrip("　 〖【").rstrip("　 〗】") for s in spk_t if s and s.lstrip("　 〖【").rstrip("　 〗】"))

    segs: List[Dict[str, Any]] = []
    for r_idx, row in enumerate(rows):
        row_segs: List[List[Dict[str, Any]]] = []
        cur_seg: List[Dict[str, Any]] = []
        cur_raw_spk = None
        for it in row:
            raw_spk = tuple(it[F.SPK]) if it.get(F.SPK) else None
            if raw_spk is not None and cur_seg and cur_raw_spk is not None and raw_spk != cur_raw_spk:
                row_segs.append(cur_seg)
                cur_seg = []
            if raw_spk is not None:
                cur_raw_spk = raw_spk
            cur_seg.append(it)
        if cur_seg:
            row_segs.append(cur_seg)
        for s_idx, toks in enumerate(row_segs):
            raw_spk = next((tuple(x[F.SPK]) for x in toks if x.get(F.SPK)), None)
            spks = _clean_spks(raw_spk)
            st, en = _tokens_time_bounds(toks)
            disp = _tokens_display_text(toks)
            segs.append({
                "row_idx": r_idx,
                "seg_idx": s_idx,
                "row_len": len(row_segs),
                "raw_spk": raw_spk,
                "spks": spks,
                "tokens": toks,
                "st": st,
                "en": en,
                "norm": _norm_unison_text(disp),
            })

    if len({s["spks"] for s in segs if s["spks"]}) < 2:
        return list(result_list)

    def _is_valid_cut(toks: Sequence[Dict[str, Any]], cut_idx: int) -> bool:
        if cut_idx <= 0 or cut_idx >= len(toks):
            return False
        left_txt = _tokens_display_text(toks[:cut_idx]).strip()
        right_txt = _tokens_display_text(toks[cut_idx:]).strip()
        if not left_txt or not right_txt:
            return False
        if not _UNISON_WORD_RE.search(left_txt) or not _UNISON_WORD_RE.search(right_txt):
            return False
        return (left_txt[-1] in _SPLIT_BOUNDARY_PUNCT) or (right_txt[0] in _OPEN_QUOTE_CHARS)

    def _strip_leading_space_spk(toks: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for x in toks:
            xc = copy.copy(x)
            if xc.get(F.SPK):
                cleaned = _clean_spks(xc[F.SPK])
                if cleaned:
                    xc[F.SPK] = cleaned
            out.append(xc)
        return out

    # 2. 对每个说话人段检测「单人前缀 + 同步齐声后缀」或「同步齐声前缀 + 单人后缀」
    split_plans: Dict[int, Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]] = {}
    affected_rows: Set[int] = set()

    for i, seg_l in enumerate(segs):
        spks_l = seg_l["spks"]
        norm_l = seg_l["norm"]
        st_l, en_l = seg_l["st"], seg_l["en"]
        if not spks_l or not norm_l or st_l is None or en_l is None:
            continue
        toks_l = seg_l["tokens"]
        if len(toks_l) < 2:
            continue

        # 找时间范围有交集、说话人不重合、且正文严格短于 norm_l 的候选齐声短段
        best_cut: Optional[int] = None
        for j, seg_s in enumerate(segs):
            if j == i:
                continue
            spks_s = seg_s["spks"]
            norm_s = seg_s["norm"]
            st_s, en_s = seg_s["st"], seg_s["en"]
            if not spks_s or not norm_s or st_s is None or en_s is None:
                continue
            if any(s in spks_l for s in spks_s):
                continue
            if len(norm_s) >= len(norm_l):
                continue
            if en_s < st_l - 0.50 or st_s > en_l + 0.50:
                continue
            norm_l_noq = re.sub(r"[「」『』]", "", norm_l)
            if not (norm_l_noq.endswith(norm_s) or norm_l_noq.startswith(norm_s)):
                continue

            # 扫描合法 token 边界，核对逐 token 真实时间是否与 seg_s 同步起音且高度重叠
            for cut_idx in range(1, len(toks_l)):
                if not _is_valid_cut(toks_l, cut_idx):
                    continue
                left_toks = toks_l[:cut_idx]
                right_toks = toks_l[cut_idx:]
                l_norm = _norm_unison_text(_tokens_display_text(left_toks))
                r_norm = _norm_unison_text(_tokens_display_text(right_toks))
                cand_toks = None
                other_toks = None
                if r_norm == norm_s and l_norm:
                    cand_toks, other_toks = right_toks, left_toks
                elif l_norm == norm_s and r_norm:
                    cand_toks, other_toks = left_toks, right_toks
                else:
                    continue

                c_st, c_en = _tokens_time_bounds(cand_toks)
                o_st, o_en = _tokens_time_bounds(other_toks)
                if c_st is None or c_en is None or o_st is None or o_en is None:
                    continue
                if abs(c_st - st_s) > 0.45:
                    continue
                ov = min(c_en, en_s) - max(c_st, st_s)
                min_dur = max(0.05, min(c_en - c_st, en_s - st_s))
                if ov >= 0.60 * min_dur:
                    best_cut = cut_idx
                    affected_rows.add(seg_l["row_idx"])
                    affected_rows.add(seg_s["row_idx"])
                    break
            if best_cut is not None:
                break

        if best_cut is not None:
            split_plans[i] = (
                _strip_leading_space_spk(toks_l[:best_cut]),
                _strip_leading_space_spk(toks_l[best_cut:]),
            )

    if not split_plans:
        return list(result_list)

    # 3. 重建 result_list：未受影响的行 100% 原样保留；受影响的行按切分后的独立行顺序输出
    segs_by_row: Dict[int, List[Tuple[int, Dict[str, Any]]]] = {}
    for idx_s, s in enumerate(segs):
        segs_by_row.setdefault(s["row_idx"], []).append((idx_s, s))

    out_rl: List[Dict[str, Any]] = []
    for r_idx, row in enumerate(rows):
        if r_idx not in affected_rows:
            out_rl.extend(row)
            out_rl.append({"orig": "\n", "type": 0, "pron": "", F.INSERTED: True})
            continue
        emitted_blocks: List[Tuple[float, List[Dict[str, Any]]]] = []
        for idx_s, s in segs_by_row.get(r_idx, []):
            if idx_s in split_plans:
                left_t, right_t = split_plans[idx_s]
                ls, _ = _tokens_time_bounds(left_t)
                rs, _ = _tokens_time_bounds(right_t)
                emitted_blocks.append((ls if ls is not None else 0.0, left_t))
                emitted_blocks.append((rs if rs is not None else 0.0, right_t))
            else:
                clean_t = _strip_leading_space_spk(s["tokens"])
                cs, _ = _tokens_time_bounds(clean_t)
                emitted_blocks.append((cs if cs is not None else 0.0, clean_t))
        emitted_blocks.sort(key=lambda x: x[0])
        for _, blk in emitted_blocks:
            out_rl.extend(blk)
            out_rl.append({"orig": "\n", "type": 0, "pron": "", F.INSERTED: True})
    return out_rl


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
    result_list = split_unison_prefix_suffix_rl(result_list)
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

        body_text = clean_text
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
            "body": body_text,
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

    raw_cues: List[Tuple[float, float, Tuple[str, ...], str, str]] = []

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
            raw_cues.append((st, eff_en, spk_key, ln["text"], ln["body"]))

    # 全局汇流：按起音时间严格升序排序，并在 SRT 组装出口统一执行「1 对多（保留较细分段）+ 1 对 1」同步齐声合并
    raw_cues.sort(key=lambda x: (x[0], x[1]))

    def _clean_spks(spk_t: Tuple[str, ...]) -> List[str]:
        return [s.lstrip("　 〖【").rstrip("　 〗】") for s in spk_t if s.strip()]

    def _norm_body(b: str) -> str:
        return _norm_unison_text(b)

    def _prefer_unquoted_body(b1: str, b2: str) -> str:
        """齐声合并时若一方带外层成对引号、另一方不带，优先展示不带外层引号的文本。"""
        q1 = _strip_outer_quotes_text(b1) != b1.strip()
        q2 = _strip_outer_quotes_text(b2) != b2.strip()
        if q1 and not q2:
            return b2
        return b1

    # 转为可变列表：[st, en, spks_list, orig_txt, body, norm]
    working: List[List[Any]] = [
        [st, en, _clean_spks(spks), txt, body, _norm_body(body)]
        for st, en, spks, txt, body in raw_cues
    ]

    # Step 1: 1 对多齐声归并（方案 A：保留被切开一方的较细分段与精确时间，将长整句说话人并入每段）
    dropped = [False] * len(working)
    # 预先按角色建立索引顺序，确保只匹配同一说话人的连续相邻句
    by_spk_indices: Dict[Tuple[str, ...], List[int]] = {}
    for idx_w, item in enumerate(working):
        if item[2]:
            by_spk_indices.setdefault(tuple(item[2]), []).append(idx_w)

    for i, item_l in enumerate(working):
        if dropped[i]:
            continue
        st_l, en_l, spks_l, _, _, norm_l = item_l
        if not spks_l or not norm_l:
            continue
        matched_chain: Optional[List[int]] = None
        for other_key, o_idxs in by_spk_indices.items():
            if any(s in spks_l for s in other_key):
                continue
            for pos, first_idx in enumerate(o_idxs):
                if dropped[first_idx]:
                    continue
                st_first = working[first_idx][0]
                if st_first < st_l - 0.45:
                    continue
                if st_first > st_l + 0.45:
                    break
                # 尝试拼合连续 2..3 句
                acc_norm = working[first_idx][5]
                if not acc_norm or not norm_l.startswith(acc_norm) or acc_norm == norm_l:
                    continue
                chain = [first_idx]
                for step in range(1, 3):
                    if pos + step >= len(o_idxs):
                        break
                    nxt_idx = o_idxs[pos + step]
                    if dropped[nxt_idx]:
                        break
                    acc_norm += working[nxt_idx][5]
                    chain.append(nxt_idx)
                    if not norm_l.startswith(acc_norm):
                        break
                    if acc_norm == norm_l:
                        en_last = working[nxt_idx][1]
                        if abs(en_last - en_l) <= 0.85:
                            matched_chain = chain
                        break
                if matched_chain is not None:
                    break
            if matched_chain is not None:
                break

        if matched_chain is not None:
            dropped[i] = True
            first_i = matched_chain[0]
            last_i = matched_chain[-1]
            # 决定合并后说话人顺序：先起音者在前
            l_first = st_l < working[first_i][0]
            for ci_m in matched_chain:
                cur_spks = working[ci_m][2]
                merged_spks = list(spks_l) if l_first else list(cur_spks)
                for s in (cur_spks if l_first else spks_l):
                    if s not in merged_spks:
                        merged_spks.append(s)
                working[ci_m][2] = merged_spks
            working[first_i][0] = min(working[first_i][0], st_l)
            working[last_i][1] = max(working[last_i][1], en_l)

    stage2 = [w for idx_w, w in enumerate(working) if not dropped[idx_w]]
    stage2.sort(key=lambda x: (x[0], x[1]))

    # Step 2: 1 对 1 同步齐声归并
    all_cues: List[Tuple[float, float, str]] = []
    used = [False] * len(stage2)
    for i, (st_i, en_i, spks_i, txt_i, body_i, norm_i) in enumerate(stage2):
        if used[i]:
            continue
        used[i] = True
        if not spks_i or not norm_i:
            all_cues.append((st_i, en_i, txt_i))
            continue

        g_st, g_en = st_i, en_i
        g_spks = list(spks_i)
        g_body = body_i

        for j in range(i + 1, len(stage2)):
            if used[j]:
                continue
            st_j, en_j, spks_j, _, body_j, norm_j = stage2[j]
            if st_j - g_st > 0.45:
                break
            if not spks_j or any(s in g_spks for s in spks_j):
                continue
            if norm_j != norm_i:
                continue
            ov = min(g_en, en_j) - max(g_st, st_j)
            min_dur = max(0.05, min(g_en - g_st, en_j - st_j))
            if ov >= 0.60 * min_dur:
                used[j] = True
                g_st = min(g_st, st_j)
                g_en = max(g_en, en_j)
                g_body = _prefer_unquoted_body(g_body, body_j)
                for s in spks_j:
                    if s not in g_spks:
                        g_spks.append(s)

        clean_name = "、".join(g_spks)
        if clean_name:
            all_cues.append((g_st, g_en, f"【{clean_name}】{g_body}"))
        else:
            all_cues.append((g_st, g_en, txt_i))

    all_cues.sort(key=lambda x: (x[0], x[1]))
    return _format_srt_cues(all_cues)


# 单人模式与多视角模式通用别名
build_standard_srt = build_independent_multispeaker_srt
