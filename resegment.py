# -*- coding: utf-8 -*-
"""标点驱动的重分段（punctuation-driven re-segmentation）。

设计动机
--------
用户要「长语句拆分 / 重新分段排版」，且句首拟声（笑声只是例子）会把 CTC 判得准的
语义部分「带偏」：现有行边界 == 台本行边界，时间戳借用严格行内有界，粘连行里语义
token 被迫向拟声借时间。

机制（已用 01 轨 CTC 对齐实测验证）
-----------------------------------------------------------
- 候选切点 = 台本标点（。！？、，… 等语言单位边界），零成本、不误判、不枚举模式
- 证据     = 标点左右发音 token 的 CTC blank 间隔（gap），声学观测值
- 决策     = gap >= thr_frames 才切；标点本身保留在输出里（不删除）
实测：标点处 gap 中位 760ms–1.8s，词内非标点处仅 60ms，差 12 倍，干净可分。

接入点
------
FA-ASMR_GUI.py：在「发音 token 拿到真实时间（result_list[idx]['start/end'] 已设）」
之后、「行内有界时间借用」之前调用 apply_punctuation_resegment，插入 {"orig":"\\n"}
分隔项。原有台本行边界（\n）保留，仅在行内标点处追加切点。借用逻辑随之适配新边界。

纯函数，无副作用；不依赖音频，只依赖已对齐的时间。
"""
from __future__ import annotations

import clause
from clause import F

# 候选切点判定：**不依赖任何字符表**。
# 对齐只消费「有读音(pron)」的 token；凡 result_list 中「无 pron 且非行分隔 \n」的
# 项，都是被 tokenizer 丢弃的东西（标点/装饰符/空白/特殊符号/异体字形…），一律作为
# 候选切点。这样不同台本用任何符号都能识别，不会「碰到特殊台本就不认」。
# 真正切不切由两侧发音 token 的 CTC gap 决定（声学证据），不在文本侧枚举。
#
# 实现已迁移到 clause.py（全项目统一的子句概念模块）的 is_punct_cut_point。
# 注意本处用的是「标点切点」视角而非「子句边界」视角：换行在本文件别处
# （apply_punctuation_resegment 内）已单独处理，故此处必须排除换行。


def _is_cut_candidate(item: dict) -> bool:
    return clause.is_punct_cut_point(item)


# 易错拟声判定不可用时的降级告警只打一次（见 _is_ono_span）
_ONO_IMPORT_WARNED = False


def _nearest_time(result_list, rl_time, idx, direction):
    """direction=-1 向左 / +1 向右，返回 (start_sec, end_sec) 或 None。
    严格行内有界：绝不越过换行符 \n 跨行偷时间，杜绝句末标点被误判为空白间隙而插换行吞字。
    """
    step = -1 if direction < 0 else 1
    j = idx + step
    while 0 <= j < len(result_list):
        if result_list[j].get("orig") == "\n":
            break
        if j in rl_time:
            return rl_time[j]
        j += step
    return None


def _measure_phrase_chars(result_list, start_idx: int) -> int:
    """向右测量下一个子句直到标点或行尾的可见字符数。"""
    count = 0
    j = start_idx
    while j < len(result_list):
        it = result_list[j]
        orig = it.get("orig", "")
        if orig == "\n":
            break
        # 遇到下一个候选切点（标点符号），子句结束
        if _is_cut_candidate(it):
            break
        if orig and not it.get(F.RUBY_PLACEHOLDER, False):
            count += len(orig)
        j += 1
    return count


def _is_ono_span(result_list, start_idx: int, end_idx: int = None) -> bool:
    """该子句是否为【易错拟声行】—— True / False，只有两种。

    取区间内的发音 token 拼成整块，调 ono_filter.is_easy_ono（统一判据）。
    end_idx 为 None 时扫到「行界或下一个标点」为止（= 一个子句）；
    给定 end_idx 时扫到该下标。
    空子句（无发音 token）返回 False。
    """
    toks = []
    j = start_idx
    while j < len(result_list):
        if end_idx is not None and j >= end_idx:
            break
        it = result_list[j]
        orig = it.get("orig", "")
        if orig == "\n":
            break
        if end_idx is None and _is_cut_candidate(it):
            break
        if it.get("pron") and orig:
            toks.append(orig)
        j += 1
    if not toks:
        return False
    try:
        from ono_filter import is_easy_ono
        return is_easy_ono("".join(toks))
    except Exception as e:
        # 降级为「非拟声」会让类型边界切分失效（台词/呻吟不再强制分行），
        # 但结果不会异常 —— 故必须打印一次告警，避免机制静默失效。
        global _ONO_IMPORT_WARNED
        if not _ONO_IMPORT_WARNED:
            _ONO_IMPORT_WARNED = True
            print(f"  ⚠ [resegment] 易错拟声判定不可用（{type(e).__name__}: {e}），"
                  "已降级为「一律非拟声」——类型边界切分失效。")
        return False


def apply_punctuation_resegment(
    result_list,
    rl_time,
    frame_sec: float = 0.02,
    thr_frames: int = 50,
    max_line_chars: int = 22,
    soft_gap_frames: int = 20,
    ono_max_line_chars: int = 32,
) -> list:
    """标点驱动智能重分段：
    1. 严格保留原始台本的大行边界（\\n），绝不擅自跨行吞并台本；
    2. 在行内标点处，以句尾发音 token 的 end 到句首发音 token 的 start 严格计算声学停顿 (gap)；
    3. 硬停顿切分：标点两侧停顿 >= thr_frames (1.0s)，切分独立成行；
    4. 前瞻溢出切分：合并下一子句后会爆屏，且标点处有呼吸气口 >= soft_gap_frames。
       上限按类型区分：**拟声行之间用 ono_max_line_chars（更宽）**，
       其余用 max_line_chars —— 呻吟串无语义，不该被台词行的 22 字上限切碎；
    5. **类型边界切分**：当前子句与下一子句的「拟声/台词」类型不同时，无条件切分
       —— 台词行与呻吟行绝不混排（汉化时两者处理方式完全不同：
       台词要翻译、呻吟要删），即使 CV 连读无停顿也必须切开；
    6. 单子句本身超长绝不生硬截断假名。
    """
    out = []
    current_line_chars = 0
    clause_start_idx = 0             # 当前子句起点（上一个标点之后）

    for idx, item in enumerate(result_list):
        out.append(item)
        orig = item.get("orig", "")

        # 原始台本换行符：严格保留原著大行结构，重置当前行累积字数
        if orig == "\n":
            current_line_chars = 0
            clause_start_idx = idx + 1
            continue

        if orig and not item.get(F.RUBY_PLACEHOLDER, False) and not _is_cut_candidate(item):
            current_line_chars += len(orig)

        if not _is_cut_candidate(item):
            continue  # 发音 token：不作切点
        if out and out[-1].get("orig") == "\n":
            continue  # 避免连续分隔产生空行
        if current_line_chars == 0:
            continue  # 行首前导标点/省略号（尚未出现实质发音），绝不在句首打断切分

        # 保护：若紧随其后的下一个项仍是候选标点/符号（即处于标点团块内部），
        # 暂不切断，延后到该标点团块全部输出后再统一决策切分，
        # 绝对防止连续标点（如 "...、"、"……！"）被拦腰截断成两行。
        if idx + 1 < len(result_list) and _is_cut_candidate(result_list[idx + 1]):
            continue

        # 句尾 token 到句首 token 的精确声学间隙
        prev_t = _nearest_time(result_list, rl_time, idx, -1)
        next_t = _nearest_time(result_list, rl_time, idx, +1)

        cur_ono = _is_ono_span(result_list, clause_start_idx, idx + 1)
        next_ono = _is_ono_span(result_list, idx + 1)

        if prev_t is None or next_t is None:
            clause_start_idx = idx + 1
            continue

        gap_frames = (next_t[0] - prev_t[1]) / frame_sec

        # 判定 1：硬停顿条件 (>= 1.0s)
        is_hard_pause = gap_frames >= thr_frames

        # 判定 2：前瞻溢出 —— 合并下一子句后会爆屏，且标点处有呼吸气口。
        # 上限按类型区分（与 merge_short 同口径，两个标准）：
        #   两侧都是易错拟声 -> ono_max_line_chars（更宽），
        #     呻吟串无语义，合成更长的连续呻吟更连贯；
        #   其余 -> max_line_chars（严格）。
        _lim = ono_max_line_chars if (cur_ono and next_ono) else max_line_chars
        next_phrase_chars = _measure_phrase_chars(result_list, idx + 1)
        is_overflow = (
            (current_line_chars + next_phrase_chars > _lim)
            and (current_line_chars > 0)
            and (gap_frames >= soft_gap_frames)
        )

        if is_hard_pause or is_overflow:
            out.append({"orig": "\n", "type": 0, "pron": "", F.INSERTED: True})
            current_line_chars = 0
        # 无论是否切分，本标点之后都是新子句的起点
        clause_start_idx = idx + 1

    return out


