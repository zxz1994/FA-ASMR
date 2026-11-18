# -*- coding: utf-8 -*-
"""子句（clause）—— 全项目统一的切分判据。

为什么独立成模块
----------------
子句是本项目最核心的中间概念，注音、对齐、拟声守门、重分段、短句合并
全都建立在它之上。但判据原先散落三处：

    ono_filter.is_clause_boundary   「这是子句边界吗」—— 含换行
    merge_short 内联副本             导入成功时同上，失败时退化为只看换行
    resegment._is_cut_candidate      「这是标点切点吗」—— 不含换行、排除 type 1-5

三份实现让「统一子句概念」无从谈起：改一处要同步三处，且无法判断改的是
否同一个概念。本模块把它们收敛为同一套底层判定上的两个视角。

两个视角（不是两份重复实现）
----------------------------
调研数据（3 条轨、10096 项）：三者差异 98.3% 来自换行符，而那**不是分歧**，
是分工——

- ``is_clause_boundary``   回答「这个位置是不是子句边界」。换行当然是边界。
  供 ono_filter、merge_short、GUI 解析阶段使用。
- ``is_punct_cut_point``   回答「这个位置是不是标点切点」。resegment 在调用
  前已于别处处理换行（重置行字数、推进子句起点），故此处必须排除换行；
  且要排除 type 1-5 的文本字（即使是促音 っ 这类无读音的）。
  供 resegment 使用。

两者共用底层判定（is_symbol / is_newline / is_text_token），从结构上
保证不会各自漂移。

真实分歧只剩 9 个促音
--------------------
type=3 且 pron 为空的 ``っ``（形如 ``...って、答は``）是唯一的实质分歧点。
实测：三轨 1377 个子句，两种口径差异 4 个（0.29%），且这 4 个的拟声判定
全部相同（ono 值不变）。

默认行为保持现状：``is_punct_cut_point`` 不把促音当切点，``って`` 保持完整，
不被切成 ``っ`` + ``て``。

如需切换口径，改 ``PUNCT_CUT_INCLUDES_EMPTY_TEXT = True``。
**改后必须跑 `python eval\\rerun_check.py --track 01` 验证。**

对外接口
--------
    is_clause_boundary(item)        子句边界（含换行）
    is_punct_cut_point(item)        标点切点（不含换行、排除文本字）
    split_clauses(result_list)      切出子句，返回发音 token 下标列表
    clause_surface(result_list, c)  取子句文本
    clauses_with_surface(...)       切分并附带文本

底层判定（供上层组合，也可单独使用）：
    is_newline / is_symbol / is_punct_symbol / is_text_token / has_pron /
    is_ruby_placeholder
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

# ── result_list 扩展字段名（全项目单一事实源）─────────────────────────────
#
# 这些 key 以字符串形式承载【阶段间状态】，跨模块读写。历史上它们以裸字面量
# 散落在 6 个文件里（_ruby_placeholder 共 14 处；一处拼错即静默失效、极难
# 排查），故在此统一定义。各模块请从此处导入，不要再写字面量。
#
# 契约与语义详见 README「五、核心数据契约」。
class F:
    """result_list 元素的扩展字段名（命名空间，避免污染模块级名字）。"""

    RUBY_PLACEHOLDER = "_ruby_placeholder"   # 注音块的非首音节占位（无可见文本）
    SRC = "_src"                             # 输出层还原用的台本原文子串
    SPK = "_spk"                             # 说话人标记元组
    KANA_N = "_kana_n"                       # 所属子句的假名数（字数判据）
    ONO_MARK = "_ono_mark"                   # 拟声守门阶段一：标记
    ONO_DROPPED = "_ono_dropped"             # 拟声守门阶段三：决定删除
    ONO_SAVED = "_ono_saved"                 # 被删 token 的原始 (start, end)
    ONO_REASON = "_ono_reason"               # 拟声决策原因（诊断）
    CLAUSE_HEAD = "_clause_head"             # 该 token 是否为子句首
    CLAUSE_ONO = "_clause_ono"               # 所在子句是否为拟声
    INSERTED = "_inserted"                   # 该 "\n" 由算法插入（非台本原有）
    NO_CLAUSE_CUT = "_no_clause_cut"         # 行内短实词顿号桥接：不在此标点处切分子句


# ── 可调开关 ────────────────────────────────────────────────────────────
# 标点切点是否把「无读音的文本字」(type 1-5 且 pron 为空，典型是促音 っ)
# 也算作切点。
#   False（默认，现状）：って 保持完整
#   True              ：って 被切成 っ / て
#
# 调研数据：开启后 1377 个子句中有 4 个会被多切一刀（0.29%），
# 且这 4 个的拟声判定不变。属产品口味选择，非技术对错。
#
# 改动后必须跑：python eval\rerun_check.py --track 01
PUNCT_CUT_INCLUDES_EMPTY_TEXT = False

# 「文本字」的 type 取值：1..5 为实体文本，0 为符号/控制字符。
# 与 README 数据契约一致。
_TEXT_TYPES = (1, 2, 3, 4, 5)


# ── 底层判定 ────────────────────────────────────────────────────────────

def is_text_token(item: Dict[str, Any]) -> bool:
    """该项是否为实体文本（type 1..5），而非符号/控制字符。"""
    return item.get("type") in _TEXT_TYPES


def has_pron(item: Dict[str, Any]) -> bool:
    """该项是否带发音（会被对齐消费）。"""
    return bool(item.get("pron"))


def is_ruby_placeholder(item: Dict[str, Any]) -> bool:
    """该项是否为注音占位符（{漢字|かな} 的汉字侧，不参与发音）。"""
    return bool(item.get("_ruby_placeholder", False))


def is_newline(item: Dict[str, Any]) -> bool:
    """该项是否为台本行分隔。"""
    return item.get("orig", "") == "\n"


def is_symbol(item: Dict[str, Any]) -> bool:
    """该项是否为「被 tokenizer 丢弃的符号」。

    判据刻意**不依赖任何字符表**：凡无发音、非空、非注音占位符的项都算。
    这样不同台本用任何符号都能识别，不会「碰到特殊台本就不认」。

    注意：换行也满足本判据（非空、无发音）。需要「符号但非换行」时用
    ``is_punct_symbol``。
    """
    orig = item.get("orig", "")
    return bool(orig) and not has_pron(item) and not is_ruby_placeholder(item)


def is_punct_symbol(item: Dict[str, Any]) -> bool:
    """该项是否为【标点符号】—— 是符号且不是换行。

    与 ``is_symbol`` 的唯一区别是排除换行。调用方（ono_filter 的停顿点统计、
    merge_short 的可见字符统计）都需要「标点断开内容」这一语义，而换行是
    行界、另有专门处理，混在一起会多算一个停顿点。
    """
    if is_newline(item):
        return False
    return is_symbol(item)


# ── 两个视角 ────────────────────────────────────────────────────────────

def is_clause_boundary(item: Dict[str, Any]) -> bool:
    """该项是否构成【子句边界】（含换行）。

    子句 = 换行，或「无发音且非空且非注音占位符」的符号项。
    供 ono_filter.mark_ono_candidates / merge_short._split_clauses 使用
    —— 它们需要把换行也当作子句边界。
    """
    if is_newline(item):
        return True
    if item.get(F.NO_CLAUSE_CUT):
        return False
    return is_symbol(item)


def is_punct_cut_point(item: Dict[str, Any]) -> bool:
    """该项是否构成【标点切点】（不含换行，排除文本字）。

    供 resegment 使用：它已在调用前单独处理换行（重置行字数、推进子句
    起点），故此处必须排除换行，否则会重复处理。

    排除 type 1-5 的原因：假名/汉字/文本字绝不是标点切点，即使它因为
    促音收缩等原因没有读音（如 ``...って`` 的 ``っ``）。
    """
    if is_newline(item):
        return False
    if item.get(F.NO_CLAUSE_CUT):
        return False
    if is_text_token(item) and not PUNCT_CUT_INCLUDES_EMPTY_TEXT:
        return False
    return is_symbol(item)


def mark_short_comma_bridges(
    result_list: Sequence[Dict[str, Any]],
    max_merged_chars: int = 10,
) -> int:
    """标记行内被单个顿号（、/，）切开的短实词片段，使其不作为子句切点。

    规则：
    - 仅在同一行内、相邻两段仅由单个顿号（``、`` 或 ``，``）分隔、且说话人相同时触发；
    - 合并后总字数 ``<= max_merged_chars``（默认 10 字）且整体非拟声词（``not is_easy_ono``）；
    - 满足以下任一条件即桥接：
      1) 左右两侧均为非拟声实词（如 ``ダメ、です``、``口の、中``、``早い、ですね``）；
      2) 左侧为 1~2 字口吃前缀且右侧实词读音以其开头（如 ``ぃ、行ってらっしゃいませ``、``ち、ちが``）；
      3) 左侧为 <=2 字实词且右侧为 <=3 字词尾（如 ``うる、さいっ``、``やめ、てっ``）。
    """
    try:
        from ono_filter import is_easy_ono as _is_easy_ono
    except Exception:
        return 0

    for it in result_list:
        it.pop(F.NO_CLAUSE_CUT, None)

    n = len(result_list)
    raw_segs: List[Tuple[List[int], List[int], bool]] = []
    cur_pron: List[int] = []
    i = 0
    while i < n:
        it = result_list[i]
        if is_newline(it) or is_symbol(it):
            if cur_pron:
                j = i
                p_idxs: List[int] = []
                saw_nl = False
                while j < n and (is_newline(result_list[j]) or is_symbol(result_list[j])):
                    if is_newline(result_list[j]):
                        saw_nl = True
                        j += 1
                        break
                    p_idxs.append(j)
                    j += 1
                raw_segs.append((cur_pron, p_idxs, saw_nl))
                cur_pron = []
                i = j
                continue
        elif has_pron(it):
            cur_pron.append(i)
        i += 1
    if cur_pron:
        raw_segs.append((cur_pron, [], False))

    bridged = 0
    acc_surf = ""
    acc_pron = ""
    acc_ono = True
    for k in range(len(raw_segs) - 1):
        c1, p_idxs, saw_nl = raw_segs[k]
        c2, _, _ = raw_segs[k + 1]
        s1_raw = "".join(result_list[x].get("orig", "") for x in c1)
        p1_raw = "".join(str(result_list[x].get("pron", "")) for x in c1)
        if not acc_surf:
            acc_surf = s1_raw
            acc_pron = p1_raw
            acc_ono = bool(_is_easy_ono(s1_raw))

        if saw_nl or not p_idxs:
            acc_surf = ""
            continue

        punct_str = "".join(result_list[p].get("orig", "") for p in p_idxs)
        if punct_str not in ("、", "，"):
            acc_surf = ""
            continue

        spk1 = tuple(result_list[c1[0]].get(F.SPK, ())) if result_list[c1[0]].get(F.SPK) else ()
        spk2 = tuple(result_list[c2[0]].get(F.SPK, ())) if result_list[c2[0]].get(F.SPK) else ()
        if spk1 != spk2:
            acc_surf = ""
            continue

        s2 = "".join(result_list[x].get("orig", "") for x in c2)
        p2 = "".join(str(result_list[x].get("pron", "")) for x in c2)
        ono2 = bool(_is_easy_ono(s2))
        comb_surf = acc_surf + s2
        if len(comb_surf) > max_merged_chars or _is_easy_ono(comb_surf):
            acc_surf = ""
            continue

        both_sem = (not acc_ono) and (not ono2)
        is_stutter = (1 <= len(acc_surf) <= 2 and not ono2 and bool(acc_pron) and p2.startswith(acc_pron))
        is_split_word = (not acc_ono and len(acc_surf) <= 2 and len(s2) <= 3)

        if both_sem or is_stutter or is_split_word:
            for p in p_idxs:
                result_list[p][F.NO_CLAUSE_CUT] = True
            bridged += 1
            acc_surf = comb_surf
            acc_pron = acc_pron + p2
            acc_ono = False
        else:
            acc_surf = ""

    return bridged


# ── 基于统一判据的子句切分 ──────────────────────────────────────────────

def split_clauses(
    result_list: Sequence[Dict[str, Any]],
    boundary=None,
) -> List[List[int]]:
    """按给定边界判据切出子句，返回每个子句的【发音 token 下标列表】。

    ``boundary`` 缺省用 ``is_clause_boundary``（含换行的完整子句视角）。
    分隔符本身不进入子句（只保留带 pron 的项），与原有各处行为一致。

    返回的下标可直接索引回 result_list，便于取 orig / 时间。
    """
    judge = boundary or is_clause_boundary
    clauses: List[List[int]] = []
    cur: List[int] = []
    for i, item in enumerate(result_list):
        if judge(item):
            if cur:
                clauses.append(cur)
                cur = []
            continue
        if has_pron(item):
            cur.append(i)
    if cur:
        clauses.append(cur)
    return clauses


def clause_surface(result_list: Sequence[Dict[str, Any]],
                   clause: Sequence[int]) -> str:
    """取出一个子句的可见文本（拼 orig）。"""
    return "".join(result_list[i].get("orig", "") for i in clause)


def clauses_with_surface(
    result_list: Sequence[Dict[str, Any]],
    boundary=None,
) -> List[Tuple[List[int], str]]:
    """切子句并附带文本，返回 [(下标列表, surface), ...]。"""
    return [(c, clause_surface(result_list, c))
            for c in split_clauses(result_list, boundary)]
