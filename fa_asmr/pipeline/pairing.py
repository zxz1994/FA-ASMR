# -*- coding: utf-8 -*-
"""FA-ASMR V2 — 台本编码识别、说话人解析、原文锚点映射与批量配对。

从 fa_asmr_pipeline.py 与 align_multispeaker.py 抽离的无界面前置输入域逻辑：
- 多编码自动降级嗅探读取 (read_text_auto_encoding)
- 说话人标记拆分 (_iter_speaker_marks / _split_speaker_segments) 与独立标题行归一化 (auto_normalize_speaker_headers)
- 原文双序列锚点切片还原 (_attach_source_spans)
- 单人 / 多人 / 双视角批量文件智能配对 (auto_pair_files)
"""
from __future__ import annotations

import difflib
import os
import re
import unicodedata
from typing import Dict, List, Tuple

from fa_asmr.text.clause import F
from fa_asmr.pipeline.ruby import _INLINE_RUBY_RE

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".opus", ".wma"}


# ── 说话人标记 ──────────────────────────────────────────────────────
# 统一形态：〖标记〗（U+3016 / U+3017 包裹）。标记内容可为圈号（①）或人名
# （フウカ），项目不关心它具体是什么，只当作分组键。
# 选型依据（均实测）：
#   · 注音前整体剥离 —— 标记内容不产任何 token，天然不进对齐
#   · 〖〗 NFKC 稳定、台本零占用、不与 {}（振假名）/ []（romaji）语法冲突
#   · 有明确边界 —— 刻意不认裸圈号：句中的 ①（「それは①番だ」）会被误判成
#     说话人切换，歧义无法消除；加括号后标记与正文再无混淆
# 文件名后缀同用此形态：原名〖①〗〖②〗.txt
_SPEAKER_BRACKET_RE = re.compile("\u3016\\s*([^\u3016\u3017]*?)\\s*\u3017")
# 圈数字各区：①-⑳ / ⑴-⒇ / ㉑-㉟ / ㊱-㊿ / ➀-➓
_SPEAKER_SEP = "\u3001,\uff0c\u30fb&/ \t"


def _iter_speaker_marks(line):
    """产出 (起点, 终点, 说话人名)。只认〖…〗形式。

    刻意**不认**裸圈号 ①：句中的 ①（如「それは①番だ」）会被误切成说话人切换，
    歧义无法消除。用〖〗界定后，标记与正文再无混淆可能。
    """
    spans = []
    for m in _SPEAKER_BRACKET_RE.finditer(line):
        spans.append((m.start(), m.end(), m.group(1).strip()))
    spans.sort(key=lambda x: x[0])
    return spans


def _split_speaker_segments(line, cur):
    """按说话人标记把一行切成 [(说话人元组, 文本), ...]。

    标记作用范围 = 从标记处到「下一个标记或行尾」。标记之间只有分隔符
    （〖A〗〖B〗 / ①、②）时视为多人并列（同时说话），合并为一个元组。
    未出现标记的文本沿用上一行说话人（cur），故「换人才写标记」可行。
    返回 (切分结果, 行末说话人)。
    """
    marks = _iter_speaker_marks(line)
    if not marks:
        return ([(cur, line)] if line else []), cur
    segs = []
    pos = 0
    names = cur
    i = 0
    n = len(marks)
    while i < n:
        a, b, name = marks[i]
        pre = line[pos:a]
        if pre:
            segs.append((names, pre))
        # 收集连续标记：标记之间只隔分隔符时并入同一组（多人并列）
        group = [name]
        last_end = b
        j = i + 1
        while j < n:
            na, nb, nname = marks[j]
            between = line[last_end:na]
            # 标记之间为空串或纯分隔符 -> 视为同一组（多人并列）
            if all(c in _SPEAKER_SEP for c in between):
                group.append(nname)
                last_end = nb
                j += 1
            else:
                break
        names = tuple(group)
        pos = last_end
        i = j
    rest = line[pos:]
    if rest:
        segs.append((names, rest))
    return segs, names


# ── 角色标题自动识别与归一化 ──────────────────────────────────────────

_NON_SPEAKER_HEADER_RE = re.compile(
    r"(?i)^(?:トラック|track|tr\.?\s*\d|scene|シーン|第?\s*[0-9０-９一二三四五六七八九十]+\s*[章話節回部編]|"
    r"プロローグ|エピローグ|タイトル|オープニング|エンディング|おまけ|特典|フリートーク|free\s*talk|"
    r"se|bgm|効果音|環境音|音声|台本|本編|目次|登場人物|キャスト|cast|キャラクター|設定|注意|紹介)"
)

_ACTION_SUFFIX_RE = re.compile(
    r"[・/／：:\s]+(?:吐息|耳舐め|耳かき|耳キス|キス|フェラ|フェラチオ|パイズリ|手コキ|足コキ|"
    r"オナニー|喘ぎ|囁き|舐め|しゃぶり|射精|絶頂|イキ|クンニ|アナル|挿入|本番|素股|乳首|乳首責め).*$"
)

_STANDALONE_BRACKET_HEADER_RE = re.compile(r"^\s*(?:【([^】]{1,20})】|《([^》]{1,20})》)\s*$")


def _clean_speaker_header_name(raw_inside: str) -> Tuple[List[str], bool]:
    """从【...】内部文本提取角色名列表，并返回是否带有动作后缀。"""
    s = (raw_inside or "").strip()
    if not s or _NON_SPEAKER_HEADER_RE.search(s):
        return [], False
    if any(ch in s for ch in "。！？!?「」『』（）()"):
        return [], False
    cleaned = _ACTION_SUFFIX_RE.sub("", s).strip()
    has_action_suffix = (cleaned != s)
    if not cleaned or _NON_SPEAKER_HEADER_RE.search(cleaned):
        return [], False
    parts = [p.strip() for p in re.split(r"[＆&、，,＋+×/／]", cleaned) if p.strip()]
    valid_parts = [p for p in parts if 1 <= len(p) <= 12 and not _NON_SPEAKER_HEADER_RE.search(p)]
    return valid_parts, has_action_suffix


def auto_normalize_speaker_headers(raw_lines: List[str]) -> Tuple[List[str], List[str]]:
    """若台本尚未使用 〖角色〗，自动检测高频复现的独立【角色名】标题行并转为 〖角色名〗。"""
    if any("\u3016" in ln and "\u3017" in ln for ln in raw_lines):
        return raw_lines, []

    header_hits: List[Tuple[int, List[str], bool]] = []
    name_counts: Dict[str, int] = {}

    for idx, ln in enumerate(raw_lines):
        m = _STANDALONE_BRACKET_HEADER_RE.match(ln)
        if not m:
            continue
        inside = m.group(1) if m.group(1) is not None else m.group(2)
        names, has_act = _clean_speaker_header_name(inside)
        if not names:
            continue
        header_hits.append((idx, names, has_act))
        for nm in names:
            name_counts[nm] = name_counts.get(nm, 0) + 1

    recurring = {k for k, cnt in name_counts.items() if cnt >= 2}
    if len(recurring) < 2 or len(header_hits) < 4:
        return raw_lines, []

    ordered_speakers: List[str] = []
    for _, names, _ in header_hits:
        for nm in names:
            if nm in recurring and nm not in ordered_speakers:
                ordered_speakers.append(nm)

    if len(ordered_speakers) < 2:
        return raw_lines, []

    hit_map = {idx: (names, has_act) for idx, names, has_act in header_hits}
    out_lines: List[str] = []
    for idx, ln in enumerate(raw_lines):
        if idx in hit_map:
            names, _ = hit_map[idx]
            kept = [nm for nm in names if nm in recurring]
            if kept:
                tag_str = "".join(f"\u3016{nm}\u3017" for nm in kept)
                out_lines.append(tag_str)
                continue
        out_lines.append(ln)

    return out_lines, ordered_speakers


def read_text_auto_encoding(file_path: str) -> tuple[str, str]:
    """智能多编码降级嗅探读取台本文本。

    支持格式：
      - UTF-8（标准及带 BOM utf-8-sig）
      - Shift-JIS / CP932（DLsite 与日本声优事务所原生台本常用编码）
      - GB18030 / GBK / GB2312（中文 Windows 常见编码）
      - UTF-16 LE / BE
    返回 (解码文本, 识别编码名称)。
    """
    with open(file_path, "rb") as f:
        raw = f.read()

    # 1. 明确的 BOM 探测
    if raw.startswith(b"\xef\xbb\xbf"):
        try:
            text = raw[3:].decode("utf-8")
            return (text[1:] if text.startswith("\ufeff") else text), "utf-8-sig"
        except UnicodeDecodeError:
            text = raw[3:].decode("utf-8", errors="replace")
            return (text[1:] if text.startswith("\ufeff") else text), "utf-8-sig-fallback"
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        try:
            text = raw.decode("utf-16")
            return (text[1:] if text.startswith("\ufeff") else text), "utf-16"
        except UnicodeDecodeError:
            pass

    # 2. 严格 UTF-8 校验（覆盖现代大多数台本）
    try:
        text = raw.decode("utf-8")
        return (text[1:] if text.startswith("\ufeff") else text), "utf-8"
    except UnicodeDecodeError:
        pass

    # 3. 日文本地编码 CP932 / Shift-JIS（覆盖绝大多数 DLsite 原生日文台本）
    try:
        return raw.decode("cp932"), "cp932"
    except UnicodeDecodeError:
        pass

    # 4. 中文编码 GB18030（覆盖 GBK / GB2312）
    try:
        return raw.decode("gb18030"), "gb18030"
    except UnicodeDecodeError:
        pass

    # 5. UTF-16 无 BOM 尝试
    try:
        text = raw.decode("utf-16")
        return (text[1:] if text.startswith("\ufeff") else text), "utf-16"
    except UnicodeDecodeError:
        pass

    # 6. 最终兜底：UTF-8 容错替换（防止抛出未捕获异常导致对齐中断）
    return raw.decode("utf-8", errors="replace"), "utf-8-fallback"


def _strip_ruby_for_src(text: str) -> str:
    """把原台本里已有的 {漢字|かな} 折叠成表层字，避免 _src 把花括号与读音拆挂到相邻 token。"""
    if "{" not in text:
        return text
    return _INLINE_RUBY_RE.sub(r"\1", text)


def _attach_source_spans(tokens, annotated_line, source_line, normalize_fn=None):
    """给每个 token 挂 _src —— 直接把可见 token 串与原台本行做双序列锚点对齐并切片。

    不再尝试逐字逆向推算 normalize 行为（曾因 ゛/゜ 变空格后 .strip() 失步导致整行雪崩、
    或因 ℃/㍿ 展开后左边界回退导致符号翻倍）：直接通过全局序列比对生成单调递增边界数组
    ``bound[0..len(vis_str)] -> [0..len(plain_src)]``，再按边界严格切分原始台本。
    """
    if not source_line:
        return
    plain_src = _strip_ruby_for_src(source_line)
    if not plain_src:
        return

    vis_indices = []
    token_ranges = []
    vis_parts = []
    v_pos = 0
    for k, t in enumerate(tokens):
        o = t.get("orig", "")
        if not o or o == "\n" or t.get(F.RUBY_PLACEHOLDER):
            continue
        vis_indices.append(k)
        vis_parts.append(o)
        token_ranges.append((v_pos, v_pos + len(o)))
        v_pos += len(o)

    if not vis_indices:
        return
    vis_str = "".join(vis_parts)
    n_v = len(vis_str)

    sm = difflib.SequenceMatcher(None, vis_str, plain_src, autojunk=False)
    bound = [0] * (n_v + 1)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for d in range(1, i2 - i1 + 1):
                bound[i1 + d] = j1 + d
        elif tag == "delete":
            for d in range(1, i2 - i1 + 1):
                bound[i1 + d] = j1
        elif tag == "insert":
            # 行首未匹配字符保留在 bound[0]=0 之前归属首 token；行中/行尾缝隙并入前一个 token
            if i1 > 0:
                bound[i1] = j2
        elif tag == "replace":
            lv = i2 - i1
            lp = j2 - j1
            for d in range(1, lv):
                bound[i1 + d] = j1 + min(lp, (d * lp + lv - 1) // lv)
            bound[i2] = j2

    for k, (ts, te) in zip(vis_indices, token_ranges):
        tokens[k][F.SRC] = plain_src[bound[ts]:bound[te]]


def _parse_speaker_suffix(stem):
    """从文件名 stem 尾部解析 〖标记〗〖标记〗... 后缀。

    形态: 音频名〖标记A〗〖标记B〗 或 音频名〖人名1〗〖人名2〗。
    标记一律用 〖〗 包裹，与正文标记同形，项目只看文件名即知本台本
    有哪些说话人及用什么标记书写。返回 (基名, 标记列表)；无后缀时返回 (stem, [])。
    """
    m = re.match("^(.*?)((?:\u3016[^\u3016\u3017]*\u3017)+)$", stem)
    if not m:
        return stem, []
    base = m.group(1)
    marks = [x.strip() for x in re.findall("\u3016([^\u3016\u3017]*)\u3017", m.group(2))]
    return base, [x for x in marks if x]


_PERSP_PATTERNS = [
    re.compile(r"^(\d+)[_\-\s]*([ABab])(?:[_\s\-\.](.*))?$"),
    re.compile(r"^(.*?)[_\s\-\.]+(?:perspective|cam|视角)?([ABab])(?:[_\s\-\.](.*))?$", re.IGNORECASE),
    re.compile(r"^(.*?)[_\s\-\.]+(?:perspective|cam|视角)?([12])$", re.IGNORECASE),
]


def _extract_perspective_tag(stem):
    """从音频文件名中识别是否带有视角 A / B 或日文「視点」标记。返回 (base_key, side_tag, desc)

    例如: '15A_超绝差距_龟男' -> ('15', 'A', '超绝差距_龟男')
          'Lesson_1_A'        -> ('Lesson_1', 'A', '')
          'トラック7_行ってきますのＮＴＲキス‗主人公視点' -> ('トラック7_行ってきますのNTRキス', '主人公', '')
          'トラック7_行ってきますのＮＴＲキス‗夫視点'   -> ('トラック7_行ってきますのNTRキス', '夫', '')
    """
    # 先处理 DLsite 等常用的 Unicode 特殊连接符（双下划线 U+2017、全角下划线 U+FF3F），避免 normalize 丢失连接
    s = stem.replace("\u2017", "_").replace("\uff3f", "_")
    s = unicodedata.normalize("NFKC", s)

    # 1. 匹配常见的日文视点后缀（贪婪匹配 base，提取最后一个连接符后面的视点名）
    m_persp = re.match(r"^(.*)[\-_\s]+(.*?)(?:視点|视点|视角)(?:[_\s]*(?:集中)?Ver)?$", s, re.IGNORECASE)
    if m_persp:
        base = m_persp.group(1).strip(" \t\r\n_")
        tag = m_persp.group(2).strip(" \t\r\n_")
        return base, tag or "persp", "perspective"

    # 2. 匹配英文字母 A/B/1/2
    m2 = re.match(r"^(.*)[\-_\s]+(?:perspective|cam|视角)?([ABab12])(?:[_\-\s](.*))?$", s, re.IGNORECASE)
    if m2:
        return m2.group(1).strip(" \t\r\n_"), m2.group(2).upper(), m2.group(3) or ""

    for pat in _PERSP_PATTERNS:
        m = pat.match(stem)
        if m:
            base = m.group(1).rstrip("_- ")
            raw_side = m.group(2).upper()
            side = "A" if raw_side in ("A", "1") else "B"
            desc = m.group(3) if len(m.groups()) >= 3 and m.group(3) else ""
            return base, side, desc
    m_num = re.match(r"^(\d+)(?:[_\s\-\.](.*))?$", stem)
    if m_num:
        return m_num.group(1), None, m_num.group(2) or ""
    return stem, None, ""


def _clean_stem_for_matching(name: str) -> str:
    """字符与修饰词归一化，用于跨格式模糊比对。"""
    s = name.replace("\u2017", "_").replace("\uff3f", "_")
    s = unicodedata.normalize("NFKC", s).strip().lower()
    s = re.sub(r"[\(\[\{【〖（].*?[\)\]\}】〗）]", "", s)
    # 清洗掉视点后缀
    s = re.sub(r"[_\-\s\.]+(?:主人公|夫|彼|彼女|男|女|私|僕|俺)?(?:視点|视点|视角)(?:[_\s]*(?:集中)?ver)?$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[_\-\s\.]+(?:perspective|cam)?[_\-\s]*[ab12]$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[_\-\s\.]+(台本|script|raw|clean|正文|字幕|text)$", "", s)
    s = re.sub(r"[_\-\s\.]+(master|audio|vocal|48k|16bit)$", "", s)
    s = re.sub(r"\b0+(\d+)\b", r"\1", s)
    s = re.sub(r"[_\-\s\.]+", " ", s).strip()
    return s


def auto_pair_files(file_paths, mode="single"):
    """智能配对台本与音频文件，支持核心锚点提取与模糊初筛。
    - mode="single" 或 "multi_speaker": 常规一对一配对
    - mode="multi_persp": 双视角模式，自动合并同一基名下的视角 A 与 视角 B 音频
    """
    clean_paths = [os.path.abspath(f) for f in file_paths if f and str(f).strip()]
    txt_files = [
        f for f in clean_paths
        if os.path.splitext(f)[1].lower() == ".txt"
        and not f.lower().endswith(".dropped.txt")
    ]
    audio_files = [f for f in clean_paths if os.path.splitext(f)[1].lower() in AUDIO_EXTS]

    pairs = []
    unmatched = []

    if mode == "multi_persp":
        # ── 双视角模式配对逻辑 ──
        aud_by_base = {}
        for aud in audio_files:
            stem = os.path.splitext(os.path.basename(aud))[0]
            base, side, desc = _extract_perspective_tag(stem)
            aud_by_base.setdefault(base, []).append((aud, side, stem))

        txt_by_base = {}
        for txt in txt_files:
            stem = os.path.splitext(os.path.basename(txt))[0]
            base, side, desc = _extract_perspective_tag(stem)
            txt_by_base.setdefault(base, []).append(txt)
            if stem != base:
                txt_by_base.setdefault(stem, []).append(txt)

        used_txts = set()
        used_auds = set()
        all_bases = sorted(set(aud_by_base.keys()) | set(txt_by_base.keys()))
        for base in all_bases:
            aud_list = [item for item in aud_by_base.get(base, []) if item[0] not in used_auds]
            txts = [t for t in txt_by_base.get(base, []) if t not in used_txts]
            script_file = txts[0] if txts else None

            if script_file and aud_list:
                aud_a, aud_b = None, None
                sides = {item[1]: item[0] for item in aud_list if item[1]}
                if "A" in sides and "B" in sides:
                    aud_a, aud_b = sides["A"], sides["B"]
                elif "1" in sides and "2" in sides:
                    aud_a, aud_b = sides["1"], sides["2"]
                elif len(aud_list) >= 2:
                    p1_kw = ("主人公", "主角", "男", "彼", "第一", "1", "a")
                    first_idx = 0
                    for idx, (aud_f, side_tag, _) in enumerate(aud_list):
                        if side_tag and any(k in str(side_tag).lower() for k in p1_kw):
                            first_idx = idx
                            break
                    aud_a = aud_list[first_idx][0]
                    other_auds = [aud_list[i][0] for i in range(len(aud_list)) if i != first_idx]
                    aud_b = other_auds[0] if other_auds else None
                elif len(aud_list) == 1:
                    aud_a = aud_list[0][0]

                pairs.append({
                    "name": os.path.splitext(os.path.basename(script_file))[0],
                    "script": script_file,
                    "audio": aud_a,
                    "audio_b": aud_b,
                    "confidence": 100,
                    "speaker_marks": [],
                })
                used_txts.add(script_file)
                if aud_a:
                    used_auds.add(aud_a)
                if aud_b:
                    used_auds.add(aud_b)

        paired_auds = {p["audio"] for p in pairs if p.get("audio")} | {p["audio_b"] for p in pairs if p.get("audio_b")}
        paired_txts = {p["script"] for p in pairs if p.get("script")}
        rem_auds = [a for a in audio_files if a not in paired_auds]
        rem_txts = [t for t in txt_files if t not in paired_txts]

        # 对剩余未配对项尝试双视角模糊匹配
        if rem_txts and rem_auds:
            for txt in list(rem_txts):
                t_stem = os.path.splitext(os.path.basename(txt))[0]
                t_clean = _clean_stem_for_matching(t_stem)
                scores = []
                for aud in rem_auds:
                    a_stem = os.path.splitext(os.path.basename(aud))[0]
                    a_clean = _clean_stem_for_matching(a_stem)
                    score = difflib.SequenceMatcher(None, t_clean, a_clean).ratio()
                    scores.append((score, aud))
                scores.sort(key=lambda x: x[0], reverse=True)
                if scores and scores[0][0] >= 0.40:
                    best_aud = scores[0][1]
                    best_score = scores[0][0]
                    second_aud = None
                    if len(scores) >= 2 and scores[1][0] >= 0.40:
                        second_aud = scores[1][1]
                    pairs.append({
                        "name": t_stem,
                        "script": txt,
                        "audio": best_aud,
                        "audio_b": second_aud,
                        "confidence": int(best_score * 100),
                        "speaker_marks": [],
                    })
                    rem_txts.remove(txt)
                    rem_auds.remove(best_aud)
                    if second_aud:
                        rem_auds.remove(second_aud)

        for aud in rem_auds:
            unmatched.append({"type": "缺台本", "file": aud, "name": os.path.splitext(os.path.basename(aud))[0]})
        for txt in rem_txts:
            unmatched.append({"type": "缺音频", "file": txt, "name": os.path.splitext(os.path.basename(txt))[0]})

    else:
        # ── 单人顺序 / 多人分角色常规一对一配对 ──
        txt_by_stem = {}
        for f in txt_files:
            stem = os.path.splitext(os.path.basename(f))[0]
            base, marks = _parse_speaker_suffix(stem)
            if base not in txt_by_stem:
                txt_by_stem[base] = {"files": [], "marks": marks}
            txt_by_stem[base]["files"].append(f)

        audio_by_stem = {}
        for f in audio_files:
            stem = os.path.splitext(os.path.basename(f))[0]
            audio_by_stem.setdefault(stem, []).append(f)

        # 1. 严格基名精确匹配
        common_stems = set(txt_by_stem.keys()) & set(audio_by_stem.keys())
        paired_txts = set()
        paired_auds = set()
        for stem in sorted(common_stems):
            info = txt_by_stem[stem]
            for txt in info["files"]:
                for aud in audio_by_stem[stem]:
                    if txt not in paired_txts and aud not in paired_auds:
                        pairs.append({
                            "script": txt,
                            "audio": aud,
                            "audio_b": None,
                            "name": stem,
                            "confidence": 100,
                            "speaker_marks": list(info["marks"]),
                        })
                        paired_txts.add(txt)
                        paired_auds.add(aud)

        rem_txts = [f for f in txt_files if f not in paired_txts]
        rem_auds = [f for f in audio_files if f not in paired_auds]

        # 2. 若双方各只剩 1 个，直接保底配对
        if len(rem_txts) == 1 and len(rem_auds) == 1:
            t = rem_txts[0]
            a = rem_auds[0]
            s_t = os.path.splitext(os.path.basename(t))[0]
            pairs.append({
                "script": t,
                "audio": a,
                "audio_b": None,
                "name": s_t,
                "confidence": 90,
                "speaker_marks": [],
            })
            rem_txts.clear()
            rem_auds.clear()

        # 3. 规约模糊相似度配对
        if rem_txts and rem_auds:
            candidates = []
            for t in rem_txts:
                t_stem = os.path.splitext(os.path.basename(t))[0]
                t_clean = _clean_stem_for_matching(t_stem)
                for a in rem_auds:
                    a_stem = os.path.splitext(os.path.basename(a))[0]
                    a_clean = _clean_stem_for_matching(a_stem)
                    score = difflib.SequenceMatcher(None, t_clean, a_clean).ratio()
                    if score >= 0.45:
                        candidates.append((score, t, a, t_stem))

            candidates.sort(key=lambda x: x[0], reverse=True)
            used_t = set()
            used_a = set()
            for score, t, a, t_stem in candidates:
                if t not in used_t and a not in used_a:
                    pairs.append({
                        "script": t,
                        "audio": a,
                        "audio_b": None,
                        "name": t_stem,
                        "confidence": int(score * 100),
                        "speaker_marks": [],
                    })
                    used_t.add(t)
                    used_a.add(a)

            rem_txts = [t for t in rem_txts if t not in used_t]
            rem_auds = [a for a in rem_auds if a not in used_a]

        for txt in rem_txts:
            unmatched.append({"type": "缺音频", "file": txt, "name": os.path.splitext(os.path.basename(txt))[0]})
        for aud in rem_auds:
            unmatched.append({"type": "缺台本", "file": aud, "name": os.path.splitext(os.path.basename(aud))[0]})

    return pairs, unmatched
