# -*- coding: utf-8 -*-
"""FA-ASMR V2 — 拟声词错位处理（标记/决策分离版）

架构（两阶段）：
  ① mark_ono_candidates —— resegment 前打标记，不改数据
  ② decide_ono_deletions —— 最终行结构上决策删除

【唯一判据】is_easy_ono(surface) -> True / False，只有两种：
    易错拟声行 / 非易错拟声行
  判定顺序：
    1. 含汉字/英文/数字      -> False（不参与拟声判断）
    2. 命中黑名单词根        -> False（实词，是台词）
    3. 状声词（有可翻语义）   -> False（保留）
    4. ono_base.is_ono       -> True/False（单音素 + 词干 + 变换 + 黑名单）
  resegment / merge_short / ono_filter 全部调它，不各自写一套。

【结构判据】断裂 / 拉伸 / 内部撕裂 / 夹塞 / 坍缩 / 压扁
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

FRAME_SEC: float = 0.02

# ── 结构判据阈值（沿用旧版 5 轨实测精调）──
_GAP_THR: float = 2.5        # 断裂：与相邻发音 token 间隙（秒）
_DUR_THR: float = 3.0        # 拉伸：子句时长（秒）
_TIGHT_GAP: float = 0.2      # 紧贴三明治：与前后拟声的紧贴上限（秒）
_INNER_GAP_THR: float = 1.0  # 内部撕裂：子句内部相邻 token 的最大间隙（秒）
_COLLAPSE_SEC: float = 2.0   # 坍缩：单 token 独占时长（秒）
# 压扁（物理容量判据）：该行的显示窗口 < 物理上塞下这些内容所需的时长。
#   所需时长 = 假名数 × _MIN_KANA_SEC + 停顿点数 × _MIN_PAUSE_SEC
#   「停顿点」= 标点把假名串【分割出的段数 - 1】
#     （ん、チュッ、 切成 2 段 -> 1 个停顿点；
#       あ、い、う、 切成 3 段 -> 2 个停顿点）
# 用户案例：ん、チュッ、 显示窗口 0.16s，
#   需 4 字×0.10 + 1 停顿×0.30 = 0.7s —— 物理上塞不下，判定被压扁。
_MIN_KANA_SEC: float = 0.10    # 每个假名最少占用的时长（秒）
_MIN_PAUSE_SEC: float = 0.30   # 每个停顿点（标点分段处）最少占用的时长（秒）
_MAX_CLAUSE_TOKENS: int = 4      # 子句 token 数上限
_MAX_ONO_LINE_CHARS: int = 24    # 整行长度上限（字符）
# 易错拟声段「连接点」的时间邻近上限（秒）。
#   保留条件 = 类型相邻(N) 且 【上一句句尾 → 下一句句首】间隔 <= 此值。
#   超过说明该拟声其实悬空离台词很远（如 E 距上一句 N 达 45s），不算连接，删。
_LINK_GAP_SEC: float = 10.0

# ── 拟声判定（用新架构）──
from ono_base import is_ono as _is_ono_token
from ono_block_corpus import is_blocked as _is_blocked
# 子句概念已统一到 clause.py；本模块保留 is_clause_boundary 兼容入口。
import clause as _clause
from clause import F

# 公开导出（merge_short 等外部模块用）
is_ono_token = _is_ono_token

# 汉字 / 英文 / 数字（含全角）
_KANJI_RE = re.compile(r"[一-鿿㐀-䶿豈-﫿々]")
_LATIN_RE = re.compile(r"[A-Za-zＡ-Ｚａ-ｚ]")
_DIGIT_RE = re.compile(r"[0-9０-９]")


def _has_kanji(text: str) -> bool:
    return bool(_KANJI_RE.search(text or ""))


def is_clause_boundary(item: dict) -> bool:
    """该项是否构成【子句边界】—— 含换行的完整子句视角。

    实现已迁移到 clause.py（全项目统一的子句概念模块），此处保留同名函数
    作为兼容入口：本模块及 merge_short 的多处调用都走这个名字。

    语义：子句 = 换行，或「未转化为 token (无 pron) 且非注音占位符」的符号项。
    与 FA-ASMR_GUI.py 解析阶段（line_group / alignment_tokens 的构建）同一口径。

    若需要「不含换行、排除文本字」的标点切点视角，请用
    clause.is_punct_cut_point —— 两者是同一概念的两种视角，见 clause.py 模块文档。
    """
    return _clause.is_clause_boundary(item)


def is_easy_ono(surface: str) -> bool:
    """该子句是否为【易错拟声行】—— 全项目唯一的二分判据。

    只有两种：易错拟声行 / 非易错拟声行。
    **没有命中易错拟声判定的，一律是非易错拟声行** —— 不再细分原因。

    判定顺序：
      1. 含汉字/英文/数字  -> False
      2. 不是拟声         -> False（ono_base 内部已含黑名单：实词/动作音/喷射音…）
      其余 -> True（易错拟声行）
    """
    if not surface:
        return False
    # 1. 汉字/英文/数字 —— 直接排除
    if _has_kanji(surface) or _LATIN_RE.search(surface) or _DIGIT_RE.search(surface):
        return False
    # 2. 不是拟声（ono_base.is_ono 内部已查黑名单，此处不再重复查）
    return _is_ono_token(surface)





def _is_vague_pron(pron: str) -> bool:
    """该 token 的发音是否为「无明确辅音的模糊呻吟」（纯元音）。"""
    p = (pron or "").lower()
    if not p:
        return False
    return all(ch in "aiueo" for ch in p)


# ══════════════════════════════════════════════════════════════
# 阶段一【标记】resegment 前打标记，不改数据
# ══════════════════════════════════════════════════════════════

def mark_ono_candidates(
    result_list: List[Dict[str, Any]],
    alignment_results: Sequence[Dict[str, Any]],
    token_to_index_map: Dict[int, int],
    gap_threshold: float = _GAP_THR,
    dur_threshold: float = _DUR_THR,
    log_callback=None,
) -> Dict[str, Any]:
    """给可疑拟声子句打 `_ono_mark` 标记，**一个 token 都不删**。

    数据保持完整，后续 resegment / 短句合并才能切出真实的最终行结构；
    决策放到阶段三、在最终行结构上一次性做（见 decide_ono_deletions）。

    返回 {"records": [...], "scanned": N, "marked": M}。
    """
    def log(msg):
        if log_callback:
            log_callback(msg)

    if not result_list or not alignment_results:
        return {"scanned": 0, "marked": 0, "records": []}

    # flat alignment 下标 → 时间
    flat_time: Dict[int, Tuple[float, float]] = {}
    for fi, r in enumerate(alignment_results):
        ri = token_to_index_map.get(fi)
        if ri is not None and r.get("original_start") is not None:
            flat_time[ri] = (float(r["original_start"]), float(r["original_end"]))

    # 重建对齐子句（换行/标点为界）—— 用全项目统一判据 is_clause_boundary
    clauses: List[List[int]] = []
    cur: List[int] = []
    for i, item in enumerate(result_list):
        if is_clause_boundary(item):
            if cur:
                clauses.append(cur)
                cur = []
            continue
        if item.get("pron"):
            cur.append(i)
    if cur:
        clauses.append(cur)

    # 把【子句结构 + 类型】存到 token 上，供 decide 阶段直接用，
    # 避免决策时再按标点重新切分重判（两处口径易打架）。
    #   _clause_head = 子句首个发音 token 的下标（同一子句内所有 token 相同）
    #   _clause_ono  = 该子句是否为易错拟声子句
    for _cl in clauses:
        _surface = "".join(result_list[x].get("orig", "") for x in _cl)
        _is_ono_cl = is_easy_ono(_surface)
        for _ri in _cl:
            result_list[_ri][F.CLAUSE_HEAD] = _cl[0]
            result_list[_ri][F.CLAUSE_ONO] = _is_ono_cl

    all_pron_ri: List[int] = [ri for clause in clauses for ri in clause]
    pos_of = {ri: k for k, ri in enumerate(all_pron_ri)}

    def _crosses_newline(idx_a, idx_b):
        lo, hi = min(idx_a, idx_b), max(idx_a, idx_b)
        return any(result_list[k].get("orig") == "\n" for k in range(lo, hi))

    marked = 0
    scanned = 0
    records: List[Dict[str, Any]] = []

    for clause in clauses:
        head = clause[0]
        if result_list[head].get(F.ONO_MARK):
            continue

        # 子句级判定：走全项目统一判据 is_easy_ono
        # （含汉字/英文/数字排除 -> 黑名单拦实词 -> 状声词保留 -> ono_base 判定）
        surface = "".join(result_list[x].get("orig", "") for x in clause)
        if not is_easy_ono(surface):
            continue
        # 错位拟声串都很短，长子句不判
        if len(clause) > _MAX_CLAUSE_TOKENS:
            continue

        timed = [ri for ri in clause if ri in flat_time]
        if not timed:
            continue
        scanned += 1
        st = flat_time[timed[0]][0]
        en = flat_time[timed[-1]][1]

        # 结构判据（与旧版相同）
        pk = pos_of.get(timed[0], 0)
        has_prev = (pk > 0 and all_pron_ri[pk - 1] in flat_time
                    and not _crosses_newline(all_pron_ri[pk - 1], timed[0]))
        gap_before = (st - flat_time[all_pron_ri[pk - 1]][1]) if has_prev else None
        nk = pos_of.get(timed[-1], len(all_pron_ri) - 1)
        has_next = (nk + 1 < len(all_pron_ri) and all_pron_ri[nk + 1] in flat_time)
        gap_after = (flat_time[all_pron_ri[nk + 1]][0] - en) if has_next else None

        fly_before = (gap_before is not None and gap_before > gap_threshold)
        fly_after = (gap_after is not None and gap_after > gap_threshold)
        flyaway = fly_before or fly_after
        stretched = (en - st) > dur_threshold

        inner_gap = 0.0
        if len(timed) >= 2:
            times = [flat_time[ri] for ri in timed]
            for k in range(len(times) - 1):
                g = times[k + 1][0] - times[k][1]
                if g > inner_gap:
                    inner_gap = g
        torn = inner_gap > _INNER_GAP_THR

        prev_ono = False
        if has_prev:
            prev_ono = _is_ono_token(result_list[all_pron_ri[pk - 1]].get("orig", ""))
        next_ono = False
        if has_next:
            next_ono = _is_ono_token(result_list[all_pron_ri[nk + 1]].get("orig", ""))
        tight_sandwich = (prev_ono and next_ono
                          and gap_before is not None and gap_before < _TIGHT_GAP
                          and gap_after is not None and gap_after < _TIGHT_GAP)
        all_vague = bool(timed) and all(
            _is_vague_pron(result_list[ri].get("pron", "")) for ri in clause)

        if flyaway or stretched or tight_sandwich or torn or all_vague:
            reason = []
            if flyaway:
                gb_s = f"{gap_before:.1f}" if gap_before is not None else "-"
                ga_s = f"{gap_after:.1f}" if gap_after is not None else "-"
                reason.append(f"断裂(前{gb_s}s/后{ga_s}s)")
            if stretched:
                reason.append(f"拉伸({en-st:.1f}s)")
            if tight_sandwich:
                reason.append(f"夹塞(前后拟声紧贴)")
            if torn:
                reason.append(f"内部撕裂({inner_gap:.1f}s)")
            if all_vague:
                reason.append("无明确发音(纯元音)")
            for ri in clause:
                result_list[ri][F.ONO_MARK] = True
                result_list[ri][F.ONO_REASON] = "；".join(reason)
            marked += 1
            records.append({
                "kind": "标记",
                "start": st,
                "text": surface,
                "reason": "；".join(reason),
            })
            log(f"  [拟声标记] 「{surface[:20]}」{'；'.join(reason)}")

    log(f"  ✨ 拟声标记(子句级): 扫描 {scanned} 句, 标记 {marked} 句")
    return {"scanned": scanned, "marked": marked, "records": records}


# ══════════════════════════════════════════════════════════════
# 阶段三【决策】最终行结构上决策删除
# ══════════════════════════════════════════════════════════════

def decide_ono_deletions(
    result_list: List[Dict[str, Any]],
    line_collapse_sec: float = _COLLAPSE_SEC,
    log_callback=None,
) -> Dict[str, Any]:
    """在【最终字幕结构】（resegment + merge_short 之后）上决策删除。

    规则：
      ① 行级结构异常：坍缩 / 抢时（沿用旧版）
      ② 行首易错拟声：无汉字 + 全是拟声 + 行首高危 + 非状声词 + 非语义 + 具语形
      ③ 已标记的子句（阶段一打的 `_ono_mark`）优先删除

    返回 {"dropped": N, "records": [...]}。
    """
    def log(msg):
        if log_callback:
            log_callback(msg)

    if not result_list:
        return {"dropped": 0, "records": []}

    # ── 按【最终字幕行】分组 ──
    # groups[li]      = 该行发音 token 的下标列表（用于时间/删除）
    # line_spans[li]  = 该行的完整区间 [start, end)（用于按 is_clause_boundary 切子句）
    groups: List[List[int]] = []
    line_spans: List[Tuple[int, int]] = []
    cur: List[int] = []
    seg_start = 0
    for i, it in enumerate(result_list):
        if it.get("orig") == "\n":
            groups.append(cur)
            line_spans.append((seg_start, i))
            cur = []
            seg_start = i + 1
        else:
            if it.get("pron"):
                cur.append(i)
    groups.append(cur)
    line_spans.append((seg_start, len(result_list)))

    def _to_sec(v):
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

    def _tok_time(i: int):
        s = _to_sec(result_list[i].get("start"))
        e = _to_sec(result_list[i].get("end"))
        if s is not None and e is not None:
            return (s, e)
        sv = result_list[i].get(F.ONO_SAVED)
        if sv:
            s2 = _to_sec(sv[0])
            e2 = _to_sec(sv[1])
            if s2 is not None and e2 is not None:
                return (s2, e2)
        return None

    def _times(g: List[int]):
        return [t for t in (_tok_time(i) for i in g) if t]

    def _is_easy_ono(li: int) -> bool:
        """第 li 行是否为【易错拟声行】—— 行内每个子句都是易错拟声子句才算。

        字幕行只有两种：① 易错拟声行（本函数 True）② 剩余的行（False）。
        含台词、含黑名单实词、无发音内容的行，都归②。

        **直接用阶段一存下的子句类型**（`_clause_ono`），不重新切分重判 ——
        子句划分与类型判定在 mark_ono_candidates 已按 is_clause_boundary 做过一次，
        这里再切一遍只会让两处口径打架。
        """
        if li < 0 or li >= len(groups):
            return False
        s, e = line_spans[li]
        seen = False
        for i in range(s, e):
            it = result_list[i]
            if it.get(F.ONO_DROPPED) or not it.get("pron"):
                continue
            if not it.get(F.CLAUSE_ONO, False):
                return False
            seen = True
        return seen

    def _line_need_sec(li: int) -> Optional[float]:
        """该行【物理上塞下这些内容】所需的最短时长（秒）。

        物理容量判据：
          所需时长 = 假名数 × _MIN_KANA_SEC + 停顿点数 × _MIN_PAUSE_SEC

        「停顿点」= 该行被标点【分割出的段数 - 1】。
        即：标点把假名串切成几段，段与段之间就有一个停顿（CTC blank）。

          ん、チュッ、      -> 切成 2 段（ん | チュッ）-> 1 个停顿点
          あ、い、う、      -> 切成 3 段（あ | い | う）-> 2 个停顿点
          はぁ、           -> 切成 1 段（标点在行尾，后面没有内容）-> 0 个停顿点

        每个停顿点至少要占 _MIN_PAUSE_SEC 的时长，否则两侧发音会被挤没。
        """
        g = groups[li]
        if not g:
            return None
        lo, hi = min(g), max(g)

        # 假名/汉字等可见字符（发音内容）
        kana = 0
        for i in g:
            o = result_list[i].get("orig", "")
            kana += len(o) if o else 0
        need = kana * _MIN_KANA_SEC

        # 停顿点 = 标点把内容切成的【段数 - 1】
        # 扫描整行（含非发音项），数「发音 token 段」的个数
        segs = 0
        in_seg = False
        for j in range(lo, hi + 1):
            it = result_list[j]
            if it.get("pron"):
                if not in_seg:
                    segs += 1
                    in_seg = True
            elif _clause.is_punct_symbol(it):
                # 标点 -> 断开当前段（换行由 is_punct_symbol 内部排除，
                # 它在此属行界而非停顿点）
                in_seg = False
        pauses = max(0, segs - 1)
        need += pauses * _MIN_PAUSE_SEC
        return need

    def _is_squeezed_ono(li: int) -> bool:
        """本行是否【被压扁】—— 窗口塞不下它的内容。

        判据（两条同时满足）：
          1. 本行是易错拟声行；
          2. 本行的显示窗口 < 物理所需时长（假名 × _MIN_KANA_SEC
             + 停顿点 × _MIN_PAUSE_SEC）—— 内容塞不下，必然错位。

        用户案例：ん、チュッ、 窗口 0.16s，
          需 4 字×0.10 + 1 停顿×0.30 = 0.70s —— 塞不下。
        """
        g = groups[li]
        if not g or not _is_easy_ono(li):
            return False
        ts = _times(g)
        if not ts:
            return False
        window = max(e for _, e in ts) - min(s for s, _ in ts)
        need = _line_need_sec(li)
        if need is None or need <= 0:
            return False
        return window < need

    def _ono_segments():
        """切出【夹在两段「剩余的行」之间的连续易错拟声行段】。

        字幕行只有两种：
          ① 易错拟声行 —— 由易错拟声子句组成（_is_easy_ono 为 True）
          ② 剩余的行   —— 其余全部（含台词行、非拟声行、无发音内容的行）

        返回 [(lo, hi), ...]：段内每行都是①，段的紧邻前行、后行都是②
        （行首/行尾也算②的边界）。

          ② 剩余行 | [① ① ①] | ② 剩余行
        """
        segs = []
        li = 0
        while li < len(groups):
            if not _is_easy_ono(li):
                li += 1
                continue
            lo = li
            hi = li
            while hi + 1 < len(groups) and _is_easy_ono(hi + 1):
                hi += 1
            segs.append((lo, hi))
            li = hi + 1
        return segs

    dropped = 0
    records: List[Dict[str, Any]] = []

    def _drop(ri):
        # 删除原因由返回值的 records[].reason 记录；不在 item 上挂冗余字段。
        nonlocal dropped
        it = result_list[ri]
        if "start" in it and F.ONO_SAVED not in it:
            it[F.ONO_SAVED] = (it.get("start"), it.get("end"))
        it.pop("start", None)
        it.pop("end", None)
        if not it.get(F.ONO_DROPPED):
            it[F.ONO_DROPPED] = True
            dropped += 1

    def _is_punct(idx):
        return _clause.is_punct_symbol(result_list[idx])

    def _touches_dropped(idx):
        for d in (-1, +1):
            j = idx + d
            while 0 <= j < len(result_list):
                o = result_list[j].get("orig", "")
                if o == "\n":
                    return True
                if result_list[j].get("pron"):
                    if result_list[j].get(F.ONO_DROPPED):
                        return True
                    break
                j += d
            else:
                return True
        return False

    def _cascade_punct(clause_ris):
        lo, hi = min(clause_ris), max(clause_ris)
        for j in range(lo - 1, max(-1, lo - 8), -1):
            if j < 0 or not _is_punct(j):
                break
            if _touches_dropped(j):
                _drop(j)
            else:
                break
        for j in range(hi + 1, min(len(result_list), hi + 8)):
            if j >= len(result_list) or not _is_punct(j):
                break
            if _touches_dropped(j):
                _drop(j)
            else:
                break

    def _drop_group(g: List[int], reason: str):
        nonlocal dropped
        alive = [i for i in g if not result_list[i].get(F.ONO_DROPPED)
                 and (result_list[i].get(F.CLAUSE_ONO, False)
                      or _is_ono_token(result_list[i].get("orig", "")))]
        if not alive:
            return
        surface = "".join(result_list[i].get("orig", "") for i in alive)
        # 整块黑名单兜底：token 拆成单字后逐个看都像拟声，整块才是实词
        if _is_blocked(surface):
            return
        ts = _times(g)
        for i in alive:
            _drop(i)
        _cascade_punct(alive)
        records.append({
            "kind": "决策",
            "start": ts[0][0] if ts else None,
            "text": surface,
            "reason": reason,
        })
        log(f"  [拟声删除·决策] 「{surface[:26]}」{reason}")

    # 第一优先：阶段一标记过的子句 —— **仅限最终字幕行首**
    #
    # 阶段一是【子句级、不限位置】判定，行中间的子句（んんちゅん、ちゅるんん）
    # 也会被标记。但在最终字幕结构里，只有【行首】的拟声会因错位而不可信；
    # 行中内容跟着行首走，删了就是误删语义。故此处必须过滤「是否行首」。
    #
    # 删除范围按行类型区分：
    #   · 纯易错拟声行 -> **整行删除**（行内全是呻吟，行首错位说明整行时间轴
    #     都是错的，只删行首会留下对不上时间的残串）
    #   · 其余行       -> 只删被标记的【行首子句】，后面跟着的台词不动
    #
    # 关键：行首必须用【删除前的原始行结构】一次性算出，
    # 不能边删边判 —— 否则删掉行首后，原本的行中会变成新行首而被级联误删
    # （实测：ふぅ 行首被删 → んんちゅん 变成新行首 → 被误删）。
    #
    # 行号直接用 groups 的现成结构反查（token 下标 -> groups 下标），
    # 不要另建一套编号 —— 两套编号极易错位，且 groups 末尾会多 append 一行，
    # 行数定义与「遇 \n 才推进」的计数不一致。
    _line_heads = {}                 # 行首 token 下标 -> groups 下标
    _line_of_idx = {}                # token 下标      -> groups 下标
    for _li, _g in enumerate(groups):
        for _ri in _g:
            _line_of_idx[_ri] = _li
        # 行首 = 该行第一个非空、非标点的项（标点也算在行的区间内，故用 line_spans）
        _s, _e = line_spans[_li]
        for _ri in range(_s, _e):
            _o = result_list[_ri].get("orig", "")
            if _o and _o != "\n":
                _line_heads[_ri] = _li
                break

    def _apply_marked_clause(marked_clause: List[int]):
        """对被标记的行首子句执行删除。

        纯易错拟声行 -> 整行删除；混合行 -> 只删该子句。
        """
        if not marked_clause:
            return
        _li = _line_of_idx.get(marked_clause[0])
        if _li is not None and _is_easy_ono(_li):
            # 纯易错拟声行：整行删除
            _drop_group(groups[_li], "纯拟声行首异常(整行删)")
        else:
            # 混合行：只删被标记的行首子句
            for ri in marked_clause:
                _drop(ri)
        _cascade_punct(marked_clause)

    marked_clause: List[int] = []
    for i, it in enumerate(result_list):
        if it.get(F.ONO_MARK) and not it.get(F.ONO_DROPPED):
            if not marked_clause:
                # 子句起点：非行首则整段跳过
                if i not in _line_heads:
                    marked_clause.append(-1)      # 占位标记「跳过中」
                    continue
            if marked_clause and marked_clause[-1] == -1:
                continue                          # 仍在跳过的子句内
            marked_clause.append(i)
        else:
            if marked_clause:
                if marked_clause[-1] != -1:
                    _apply_marked_clause(marked_clause)
                marked_clause = []
    if marked_clause and marked_clause[-1] != -1:
        _apply_marked_clause(marked_clause)

    # 第二优先：行级结构异常
    #
    # ① 压扁段：夹在两个语义行之间的一整段连续拟声行，段内只要有【任一行】
    #    被压扁（窗口塞不下它的内容），说明这段时间轴整体错乱 —— 整段拟声行
    #    都不可信，全部删除。
    #      语义行 | [拟声A 拟声B 拟声C] | 语义行
    #                        ^被压扁          -> A B C 全删
    # ② 坍缩：单行内某 token 独占超长 -> 删该行及其后连续拟声行
    _seg_handled = set()
    for (lo, hi) in _ono_segments():
        squeezed_at = [j for j in range(lo, hi + 1) if _is_squeezed_ono(j)]
        if not squeezed_at:
            continue
        for j in range(lo, hi + 1):
            _drop_group(groups[j],
                        "压扁段(段内被压扁)" if j in squeezed_at else "压扁段连带")
            _seg_handled.add(j)

    for li, g in enumerate(groups):
        if li in _seg_handled:
            continue
        if not g or not _is_easy_ono(li):
            continue
        ts = _times(g)
        if not any((e - s) > line_collapse_sec for s, e in ts):
            continue
        _drop_group(g, "含坍缩token")
        for j in range(li + 1, len(groups)):
            ng = groups[j]
            if not ng:
                continue
            if not _is_easy_ono(j):
                break
            _drop_group(ng, "坍缩段连带")

    # 第三优先：连续易错拟声行段 [头段 / 尾段 / 中段] 连接点保留
    #
    # 只在【预生产的字幕行】上判定，不回到台本行 —— 行类型由该行原始发音构成决定：
    #   E 行 = 该行所有发音子句都是易错拟声子句（`_clause_ono` 全 True）
    #   N 行 = 其余全部（含台词、含实词、含清晰发音拟声、混排行）
    #
    # 规则：一段连续 E 行，只保留「与非易错行直接相邻」的行 ——
    #
    #   ① 中段（E 左右都有 N）：N | E E E E E | N  ->  N | E       E | N
    #      保留首尾各 1 行，中间全删。
    #   ② 头段（轨道开始处就是 E，右边接 N）：只有尾端 1 行能连上 N，前面全删。
    #   ③ 尾段（E 一直延伸到轨道结束，左边接 N）：只有首端 1 行能连上 N，后面全删。
    #   ④ 全弧（整轨都是 E、两侧都没有 N）：没有任何连接点，整弧删除。
    #
    # 这样既保住「台词行旁边那一声呻吟」的衔接感，又不留下悬在半空、
    # 离前后台词十几秒远的孤立拟声。
    def _survives(li: int) -> bool:
        g = groups[li]
        if not g:
            return False
        return any(not result_list[i].get(F.ONO_DROPPED) for i in g)

    def _line_span_time(li: int) -> Optional[Tuple[float, float]]:
        """该行存活发音内容的 [最早 start, 最晚 end]。无内容则 None。"""
        alive = [i for i in groups[li] if not result_list[i].get(F.ONO_DROPPED)]
        ts = _times(alive)
        if not ts:
            return None
        return (min(s for s, _ in ts), max(e for _, e in ts))

    def _neighbor_content(li: int, step: int) -> Optional[int]:
        """沿 step(-1 左 / +1 右) 找第一个【有存活发音内容】的行。

        跳过空行（无发音内容的行）—— 空行不能充当连接点，
        否则「E 空 E 空 E」里每个 E 都会被误判为「紧邻 N」而全部保留。
        """
        p = li + step
        while 0 <= p < len(groups):
            if _survives(p):
                return p
            p += step
        return None

    _li = 0
    while _li < len(groups):
        if not (_is_easy_ono(_li) and _survives(_li)):
            _li += 1
            continue
        lo = _li
        hi = _li
        while hi + 1 < len(groups) and _is_easy_ono(hi + 1) and _survives(hi + 1):
            hi += 1
        p = _neighbor_content(lo, -1)
        q = _neighbor_content(hi, 1)
        # 连接点必须【类型相邻】且【时间邻近】：
        #   按「上一句句尾 → 下一句句首」算真实间隔，> _LINK_GAP_SEC 视为悬空，不算连接。
        #   N 在前 E 在后：gap = E.start - N.end
        #   E 在前 N 在后：gap = N.start - E.end
        def _linked(n_li: int, e_li: int) -> bool:
            if n_li is None or _is_easy_ono(n_li):
                return False
            tn = _line_span_time(n_li)
            te = _line_span_time(e_li)
            if tn is None or te is None:
                return False
            if n_li < e_li:                     # N 在前
                gap = te[0] - tn[1]
            else:                               # N 在后
                gap = tn[0] - te[1]
            return gap <= _LINK_GAP_SEC

        has_prev_n = _linked(p, lo) if p is not None else False
        has_next_n = _linked(q, hi) if q is not None else False
        if not has_prev_n and not has_next_n:
            for j in range(lo, hi + 1):
                _drop_group(groups[j], "易错拟声段无连接点(整段删)")
        else:
            for j in range(lo, hi + 1):
                if (j == hi and has_next_n) or (j == lo and has_prev_n):
                    continue                      # 与 N 类型相邻且时间邻近，保留
                _drop_group(groups[j], "易错拟声段中间行(无连接)")
        _li = hi + 1

    if dropped:
        log(f"  ✨ 拟声决策删除: 共删 {dropped} token")
    return {"dropped": dropped, "records": records}
