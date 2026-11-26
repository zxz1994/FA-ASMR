# -*- coding: utf-8 -*-
"""短句合并（按子句重排字幕行）。

问题：台本一行里的子句可能被 CV 一口气念完（resegment 按停顿判据不切），
或反过来被 resegment 切得过碎。两种都要按【子句】重排。

**子句** = 换行或标点切开的片段（判据统一在 ono_filter.is_clause_boundary，
与 GUI 解析阶段构建 alignment_tokens / line_group 时同一口径）。

流程：
  ① 把 result_list 切成子句（标点归属前一个子句）
  ② 从左到右累积相邻子句成一行，满足以下条件才继续累积：
       ① 累积后总字数 <= 上限
       ② 本子句末字 end → 下一子句首字 start 的真实停顿 <= 间隔上限（且未跨说话人）
       ③ 前瞻防「句尾吸句首」：若并入下一子句 C_{i+1} 后无法再容纳其后续子句
          C_{i+2}（即必须切开 [C_{i+1}, C_{i+2}]），且 [C_{i+1}, C_{i+2}] 在下一行
          可合并、且前侧边界阻力 D_prev(i, i+1) > 后侧边界阻力 D_next(i+1, i+2)，
          则在此断行，把句首 C_{i+1} 留给下一行。
  ③ 输出：每个 LRC 行 = 若干连续子句，行间插 \\n

**两个标准**（按两侧子句类型取上限）：
  1. 两侧都是【易错拟声行】 -> ono_max_chars=40 / ono_max_gap=2.5（宽松）。
     易错拟声无语义，合成更长的连续呻吟更连贯，也不切碎任何语义单位。
  2. 其余（任一侧**非**易错拟声行 —— 含台词行、含汉字行、含英文/数字行、
     命中黑名单的实词行、任何非拟声行）-> max_chars=20 / max_gap=1.5（严格）。
     含三种组合：非易错+非易错、非易错+易错、易错+非易错。

纯函数，返回重排后的 result_list。
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import re
import sys

# 子句概念统一到 clause.py（全项目唯一判据）
import fa_asmr.text.clause as clause
from fa_asmr.text.clause import F


_FALLBACK_RE = re.compile(r"[\u3041-\u309F\u30A0-\u30FF]")

_STRONG_END_CHARS = set("。！？!?")
_EXPRESSIVE_END_CHARS = set("♪♡♥☆★")
_ELLIPSIS_CHARS = set("…‥.―—")
_WEAK_COMMA_CHARS = set("、，,・")

# 跨行合并替换内部的 \n 时，仅当左右均为实词/句子且不紧邻这些符号才插半角空格，
# 既修复无标点短句直接粘连（如「お疲れ様布団」→「お疲れ様 布団」），又保留「、」「……」「（）」紧排。
_NO_SPACE_AFTER = set("、，,・…‥.―—「『（([【《〈“‘」』）)]】》〉”’~～〜ーっッ゛゜")
_NO_SPACE_BEFORE = set("、。！？!?,，.・…‥」』）)]】》〉”’ーっッゃゅょゎぁぃぅぇぉャュョヮァィゥェォ~～〜゛゜")
_RUBY_SURFACE_RE = re.compile(r"\{([^|]+)\|[^}]+\}")


def _visible_chars(result_list: List[Dict[str, Any]], start: int, end: int) -> int:
    """[start,end) 区间的【字数】—— 每个子句取其 `_kana_n`，只算一次。"""
    seen = set()
    n = 0
    fallback_head = None
    for i in range(start, end):
        it = result_list[i]
        if clause.is_clause_boundary(it):
            fallback_head = None
            continue
        if not it.get("pron"):
            continue
        if fallback_head is None:
            fallback_head = i
        head = it.get(F.CLAUSE_HEAD)
        if head is None:
            head = fallback_head
        if head in seen:
            continue
        seen.add(head)
        k = it.get(F.KANA_N)
        if k is None:
            k = len(_FALLBACK_RE.findall(it.get("orig", "")))
        n += k
    raw_k = sum(
        len(_FALLBACK_RE.findall(result_list[i].get("orig", "")))
        for i in range(start, end)
        if result_list[i].get("pron")
    )
    return max(n, raw_k)


_ONO_IMPORT_WARNED = False


def _is_easy_ono(result_list: List[Dict[str, Any]], start: int, end: int) -> bool:
    """该子句是否为【易错拟声行】—— True / False，只有两种。

    取不到 ono_filter.is_easy_ono 时降级为 False（视为非拟声）。
    该降级会让拟声合并的宽松上限（ono_max_chars/ono_max_gap）失效、
    一律走严格上限，且**结果不会有任何异常**——故必须打印一次告警，
    否则机制静默失效、极难察觉。
    """
    global _ONO_IMPORT_WARNED
    from fa_asmr.text.clause import is_voicing_mark as _is_voicing_mark, ono_token_surface as _ono_token_surface
    toks = []
    for i in range(start, end):
        it = result_list[i]
        if it.get("pron") and it.get("orig"):
            toks.append(_ono_token_surface(it))
        elif toks and _is_voicing_mark(it):
            toks.append("゛")
    if not toks:
        return False
    try:
        from fa_asmr.text.ono_filter import is_easy_ono
    except ImportError as e:
        if not _ONO_IMPORT_WARNED:
            _ONO_IMPORT_WARNED = True
            print(
                f"  ⚠ [merge_short] 易错拟声判定不可用（{type(e).__name__}: {e}），"
                "已降级为「一律非拟声」——拟声合并的宽松上限失效。",
                file=sys.stderr,
            )
        return False
    return is_easy_ono("".join(toks))


def _to_seconds(v) -> Optional[float]:
    """把时间值统一成秒。兼容浮点秒与 LRC 字符串 [MM:SS:CC] / [MM:SS.CC]。"""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    m = re.match(r"\[(\d+):(\d+)[.:](\d+)\]", str(v))
    if m:
        return int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 100.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _start_time(result_list: List[Dict[str, Any]], start: int, end: int) -> Optional[float]:
    """该子句第一个带时间的 token 的 start（秒）。"""
    for i in range(start, end):
        t = _to_seconds(result_list[i].get("start"))
        if t is not None:
            return t
        sv = result_list[i].get(F.ONO_SAVED)
        if sv:
            t = _to_seconds(sv[0])
            if t is not None:
                return t
    return None


def _end_time(result_list: List[Dict[str, Any]], start: int, end: int) -> Optional[float]:
    """该子句最后一个带时间的 token 的 end（秒）。与 `_start_time` 对称。"""
    for i in range(end - 1, start - 1, -1):
        t = _to_seconds(result_list[i].get("end"))
        if t is not None:
            return t
        sv = result_list[i].get(F.ONO_SAVED)
        if sv:
            t = _to_seconds(sv[1])
            if t is not None:
                return t
    return None


def _is_punct_only_clause(result_list: List[Dict[str, Any]],
                          start: int, end: int) -> bool:
    """该子句是否为【纯标点的台本行】—— 整行没有发音 token、只有标点。"""
    if _visible_chars(result_list, start, end) != 0:
        return False
    has_punct = False
    for i in range(start, end):
        it = result_list[i]
        o = it.get("orig", "")
        if not o or o == "\n" or it.get(F.RUBY_PLACEHOLDER, False):
            continue
        if it.get("pron") or clause.is_text_token(it) or clause.is_open_bracket(it) or clause.is_prolong_mark(it) or clause.is_voicing_mark(it):
            return False
        has_punct = True
    return has_punct


def _split_clauses(result_list: List[Dict[str, Any]]) -> List[Any]:
    """按【子句】切分：换行或标点为边界，**分隔符归入前一个子句尾部**。

    边界判据统一引用 clause.is_clause_boundary（全项目统一子句概念）。
    原先此处有内联副本，且 import 失败时会静默退化为「只看换行」——
    那会让标点不再切分子句，与 ono_filter 口径不一致。统一后无退化分支。

    （本函数无退化分支；但同模块的 _is_easy_ono 仍保留一层降级——
      取不到 ono_filter 时判「非拟声」，现已改为打印告警，见该函数注释。）
    """
    is_clause_boundary = clause.is_clause_boundary

    clauses = []
    seg_start = 0
    i = 0
    n = len(result_list)
    while i < n:
        it = result_list[i]
        o = it.get("orig", "")
        if o == "\n":
            if i > seg_start:
                clauses.append((seg_start, i + 1))
            elif clauses and clauses[-1][1] == i:
                clauses[-1] = (clauses[-1][0], i + 1)
            seg_start = i + 1
            i += 1
            continue
        if not is_clause_boundary(it):
            i += 1
            continue
        # 标点：吞掉紧随其后的连续标点，整段归入本子句
        j = i
        while (j + 1 < n and result_list[j + 1].get("orig", "") != "\n"
               and is_clause_boundary(result_list[j + 1])):
            j += 1
        clauses.append((seg_start, j + 1))
        seg_start = j + 1
        i = j + 1
    if seg_start < n:
        clauses.append((seg_start, n))
    return clauses


def _clause_meta(result_list: List[Dict[str, Any]], s: int, e: int) -> Dict[str, Any]:
    """提取子句的时间、字数、标点层级、原台本换行、说话人信息。"""
    chars = _visible_chars(result_list, s, e)
    easy = _is_easy_ono(result_list, s, e)
    st = _start_time(result_list, s, e)
    en = _end_time(result_list, s, e)
    has_script_nl = any(
        result_list[k].get("orig") == "\n" and not result_list[k].get(F.INSERTED)
        for k in range(s, e)
    )
    spks = [tuple(result_list[k][F.SPK]) for k in range(s, e) if result_list[k].get(F.SPK)]
    spk = spks[-1] if spks else None
    first_spk = spks[0] if spks else None

    last_pron_idx = -1
    for k in range(e - 1, s - 1, -1):
        if result_list[k].get("pron"):
            last_pron_idx = k
            break

    tail_p = []
    if last_pron_idx >= 0:
        for k in range(last_pron_idx + 1, e):
            it = result_list[k]
            o = it.get(F.SRC, it.get("orig", ""))
            if o and o != "\n" and not it.get(F.RUBY_PLACEHOLDER):
                tail_p.append(o)
    tail_punct = "".join(tail_p)
    has_terminal = any(c in _STRONG_END_CHARS for c in tail_punct)
    has_expressive = any(c in _EXPRESSIVE_END_CHARS for c in tail_punct)
    has_ellipsis = any(c in _ELLIPSIS_CHARS for c in tail_punct)
    is_weak_comma = (
        not has_terminal
        and not has_expressive
        and not has_ellipsis
        and any(c in _WEAK_COMMA_CHARS for c in tail_punct)
    )
    return {
        "s": s,
        "e": e,
        "chars": chars,
        "easy": easy,
        "st": st,
        "en": en,
        "has_script_nl": has_script_nl,
        "spk": spk,
        "first_spk": first_spk,
        "has_terminal": has_terminal,
        "has_expressive": has_expressive,
        "has_ellipsis": has_ellipsis,
        "is_weak_comma": is_weak_comma,
        "is_sent_end": has_terminal or has_expressive or (has_ellipsis and has_script_nl),
    }


def _boundary_distance(m_prev: Dict[str, Any], m_next: Dict[str, Any]) -> float:
    """在必须断行二选一时，用于比较两个候选切点(i, i+1)与(i+1, i+2)的边界疏远度。

    仅在「字数溢出必须切一刀」的向心力对比中使用，不作为硬门限阻断正常合并。
    """
    if m_prev["en"] is None or m_next["st"] is None:
        return 999.0
    g_raw = max(0.0, m_next["st"] - m_prev["en"])
    w = 0.0
    if m_prev["has_terminal"] and m_prev["has_expressive"]:
        w += 0.50
    elif m_prev["has_terminal"]:
        w += 0.45
    elif m_prev["has_expressive"]:
        w += 0.35
    elif m_prev["has_ellipsis"]:
        w += 0.30
    if m_prev["has_script_nl"]:
        w += 0.30
    if (m_prev["spk"] is not None and m_next["first_spk"] is not None
            and m_prev["spk"] != m_next["first_spk"]):
        w += 1.50
    return g_raw + w


def merge_short_lines(
    result_list: List[Dict[str, Any]],
    max_chars: int = 20,
    max_gap: float = 1.5,
    ono_max_chars: int = 40,
    ono_max_gap: float = 2.5,
    forbidden_spans: Optional[List[tuple]] = None,
) -> List[Dict[str, Any]]:
    """按【子句】重排字幕行（见模块文档）。

    可选 ``forbidden_spans``：多人模式传入他方角色的真实发声区间 ``[(st, en), ...]``。
    当本角色相邻两子句之间存在明显停顿（>= 0.35s）且停顿内被他方角色占据 >= 0.30s 时，
    禁止跨着对方台词把前后两句合并成一条长字幕。
    """
    clauses = _split_clauses(result_list)
    if not clauses:
        return list(result_list)

    # 预先把空子句（行内只有标点，且非独立纯标点行）并入相邻子句，便于后续前瞻
    merged_clauses = []
    idx = 0
    while idx < len(clauses):
        s, e = clauses[idx]
        if _is_punct_only_clause(result_list, s, e):
            merged_clauses.append((s, e, True))
            idx += 1
            continue
        while idx + 1 < len(clauses):
            ns, ne = clauses[idx + 1]
            if _is_punct_only_clause(result_list, ns, ne):
                break
            if _visible_chars(result_list, ns, ne) == 0:
                e = ne
                idx += 1
                continue
            if _visible_chars(result_list, s, e) == 0:
                s, e = ns, ne
                idx += 1
                continue
            break
        merged_clauses.append((s, e, False))
        idx += 1

    metas = [_clause_meta(result_list, s, e) for (s, e, _) in merged_clauses]
    n = len(metas)

    out: List[Dict[str, Any]] = []
    i = 0
    while i < n:
        cur_start, cur_end, is_punct_only = merged_clauses[i]
        if is_punct_only or metas[i]["chars"] == 0:
            out.extend(it for it in result_list[cur_start:cur_end]
                       if it.get("orig") != "\n")
            out.append({"orig": "\n", "type": 0, "pron": "", F.INSERTED: True})
            i += 1
            continue

        cur_easy = metas[i]["easy"]
        cur_chars = metas[i]["chars"]

        # 累积后续子句，直到超限
        while i + 1 < n:
            ns, ne, nxt_punct_only = merged_clauses[i + 1]
            if nxt_punct_only or metas[i + 1]["chars"] == 0:
                break

            m_prev = metas[i]
            m_next = metas[i + 1]

            # 说话人切换：不同说话人加 1.0s 软阻力；若任一侧为多人齐声(len>=2)则硬断行，避免齐声前缀误盖单人句
            spk_changed = (
                m_prev["spk"] is not None
                and m_next["first_spk"] is not None
                and m_prev["spk"] != m_next["first_spk"]
            )
            if spk_changed and (len(m_prev["spk"]) >= 2 or len(m_next["first_spk"]) >= 2):
                break

            nxt_easy = m_next["easy"]
            both_easy = cur_easy and nxt_easy
            lim_c, lim_g = ((ono_max_chars, ono_max_gap) if both_easy
                            else (max_chars, max_gap))

            # 判据 ①：累积后总字数
            if cur_chars + m_next["chars"] > lim_c:
                break

            # 判据 ②：本子句末字 end -> 下一子句首字 start 的真实停顿
            if m_prev["en"] is None or m_next["st"] is None:
                break
            raw_gap = m_next["st"] - m_prev["en"]
            gap = raw_gap + (1.0 if spk_changed else 0.0)
            if gap > lim_g:
                break

            # 判据 ②b（多人模式跨角色避让）：两子句间存在停顿且缝隙内夹着他方角色发言时禁止合并
            if forbidden_spans and raw_gap >= 0.35:
                t_a, t_b = m_prev["en"], m_next["st"]
                if any(min(t_b, o_en) - max(t_a, o_st) >= 0.30 for o_st, o_en in forbidden_spans):
                    break

            # 判据 ③：防「句尾吸句首」前瞻竞争（仅在 cur + C_{i+1} + C_{i+2} 无法同时塞进一行时生效）
            # 如果 cur + C_{i+1} + C_{i+2} 本身就能全部装进当前行（如短拟声串），绝不拆开！
            if not both_easy and (i + 2 < n) and not merged_clauses[i + 2][2] and metas[i + 2]["chars"] > 0:
                m_after = metas[i + 2]
                spk_same_prev_next = not spk_changed
                # 若 m_prev 是句内未完逗号子句，且 m_next 是完结该句的句尾，则 m_next 归属 m_prev
                m_next_completes_m_prev = (
                    m_prev["is_weak_comma"]
                    and spk_same_prev_next
                    and (m_next["is_sent_end"] or (not m_prev["has_script_nl"] and m_next["has_script_nl"]))
                )
                if not m_next_completes_m_prev:
                    # 检查 [i+1, i+2] 在下一行是否能合并
                    next_both_easy = m_next["easy"] and m_after["easy"]
                    next_lim_c, next_lim_g = (
                        (ono_max_chars, ono_max_gap) if next_both_easy else (max_chars, max_gap)
                    )
                    spk_changed_next = (
                        m_next["spk"] is not None
                        and m_after["first_spk"] is not None
                        and m_next["spk"] != m_after["first_spk"]
                    )
                    gap_next = (
                        (m_after["st"] - m_next["en"]) + (1.0 if spk_changed_next else 0.0)
                        if (m_after["st"] is not None and m_next["en"] is not None)
                        else 999.0
                    )
                    can_next_pair_merge = (
                        (m_next["chars"] + m_after["chars"] <= next_lim_c)
                        and (gap_next <= next_lim_g)
                    )

                    # 检查当前行是否能把 i+1 和 i+2 一起吞下
                    after_both_easy = both_easy and m_after["easy"]
                    after_lim_c, after_lim_g = (
                        (ono_max_chars, ono_max_gap) if after_both_easy else (max_chars, max_gap)
                    )
                    can_cur_fit_both = (
                        (cur_chars + m_next["chars"] + m_after["chars"] <= after_lim_c)
                        and (gap_next <= after_lim_g)
                    )

                    if can_next_pair_merge and not can_cur_fit_both:
                        # 必须在 (i | i+1) 与 (i+1 | i+2) 之间二选一切一刀：比较两侧边界距离
                        d_prev = _boundary_distance(m_prev, m_next)
                        d_next = _boundary_distance(m_next, m_after)
                        if (m_prev["is_sent_end"] and m_next["is_weak_comma"]) or (d_prev > d_next + 0.10):
                            break

            # 合并：累积行的类型 = 所有子句都易错才算易错
            cur_end = ne
            cur_chars += m_next["chars"]
            cur_easy = both_easy
            i += 1

        # 输出本行：将合并区间内部原有的 \n 去掉，必要时补半角空格避免无标点短句粘连
        _append_merged_line_tokens(out, result_list[cur_start:cur_end])
        out.append({"orig": "\n", "type": 0, "pron": "", F.INSERTED: True})
        i += 1

    return out


def _visible_surface(item: Dict[str, Any]) -> str:
    """提取 token 在最终字幕里的可见文本（展开 ruby 注音、忽略占位与已删项）。"""
    if item.get(F.ONO_DROPPED) or item.get(F.RUBY_PLACEHOLDER):
        return ""
    raw = str(item.get(F.SRC, item.get("orig", "")) or "")
    if not raw or raw == "\n":
        return ""
    return _RUBY_SURFACE_RE.sub(r"\1", raw)


def _make_join_space_token(
    left_tokens: List[Dict[str, Any]],
    right_tokens: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """判断两个相邻句段合并成同一字幕行时是否需要插入半角空格分隔符。"""
    last_vis_item: Optional[Dict[str, Any]] = None
    last_vis_ch = ""
    for it in reversed(left_tokens):
        if it.get("orig") == "\n":
            continue
        surf = _visible_surface(it)
        if surf:
            last_vis_item = it
            last_vis_ch = surf[-1]
            break

    if not last_vis_ch or last_vis_ch.isspace() or last_vis_ch in _NO_SPACE_AFTER:
        return None

    next_ch = ""
    for it in right_tokens:
        if it.get("orig") == "\n":
            continue
        surf = _visible_surface(it)
        if surf:
            next_ch = surf[0]
            break

    if not next_ch or next_ch.isspace() or next_ch in _NO_SPACE_BEFORE:
        return None

    space_tok: Dict[str, Any] = {
        "orig": " ",
        "type": 0,
        "pron": "",
        F.INSERTED: True,
    }
    if last_vis_item is not None and last_vis_item.get(F.SPK):
        space_tok[F.SPK] = last_vis_item[F.SPK]
    return space_tok


def append_tokens_with_space(
    dst: List[Dict[str, Any]],
    next_tokens: List[Dict[str, Any]],
) -> None:
    """将下一句段的 token 列表并入 dst；若衔接处无分隔标点则自动补一个半角空格。"""
    sp = _make_join_space_token(dst, next_tokens)
    if sp is not None:
        dst.append(sp)
    dst.extend(next_tokens)


def _append_merged_line_tokens(
    out: List[Dict[str, Any]],
    line_tokens: List[Dict[str, Any]],
) -> None:
    """把已合并为单行的 token 序列写入 out；原内部换行在无逗号/省略号分隔时补半角空格。"""
    n_tok = len(line_tokens)
    for idx, it in enumerate(line_tokens):
        if it.get("orig") != "\n":
            out.append(it)
            continue
        sp = _make_join_space_token(out, line_tokens[idx + 1:n_tok])
        if sp is not None:
            out.append(sp)


def count_inserted_breaks(result_list: List[Dict[str, Any]]) -> int:
    """结果列表里的行界总数（\\n 项数）—— 用于比较合并前后的行数变化。

    注意：统计的是**全部** \\n（含台本原有的与算法插入的），不只是 F.INSERTED。
    调用方（FA-ASMR_GUI）比较的是调用前后的差值，两次口径一致故差值正确；
    函数名里的「inserted」指的是「合并后重建的行界」，不是 F.INSERTED 标记。
    """
    return sum(1 for it in result_list if it.get("orig") == "\n")
