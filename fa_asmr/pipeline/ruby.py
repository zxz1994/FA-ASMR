# -*- coding: utf-8 -*-
"""FA-ASMR V2 — 注音单实例管理、覆盖表 (ruby-map) 与去重词表提取。

从 fa_asmr_pipeline.py 抽离的无界面注音域逻辑：
- AutoRuby 单例（复用 fugashi / pykakasi / yomikata 模型与 GPU 资源释放）
- ruby-map JSON 校验、按文件/行号解析及预注入 + 后置权威覆写
- 分级去重注音词表提取与跨轨合并 (extract_ruby_glossary / merge_ruby_glossaries)
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

if getattr(sys, "frozen", False):
    SCRIPT_DIR = sys._MEIPASS
else:
    SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_RUBY_ANNOTATOR = None


class AutoRuby:
    def __init__(self):
        global _RUBY_ANNOTATOR
        if _RUBY_ANNOTATOR is None:
            _td = os.path.join(SCRIPT_DIR, "furigana")
            if _td not in sys.path:
                sys.path.insert(0, _td)
            from fa_asmr_converter import RubyAnnotator
            _RUBY_ANNOTATOR = RubyAnnotator(verbose=False, strip_inline_readings=False)
        # 批量任务复用同一实例：fugashi(UniDic) / pykakasi / yomikata(dBert)
        # 只初始化一次，避免每条音频都重载 UniDic 词典与异读模型。
        self._ann = _RUBY_ANNOTATOR

    def annotate(self, text: str) -> str:
        return self._ann.annotate(text)

    def release_gpu(self):
        global _RUBY_ANNOTATOR
        try:
            self._ann.release_gpu()
        except Exception as e:
            logger.warning("释放 AutoRuby GPU 资源失败: %s: %s", type(e).__name__, e)
        _RUBY_ANNOTATOR = None


def release_global_ruby(ruby: Any = None) -> None:
    """释放传入实例或全局缓存的 AutoRuby GPU 资源。"""
    global _RUBY_ANNOTATOR
    if ruby is not None:
        try:
            ruby.release_gpu()
        except Exception as e:
            logger.warning("释放 AutoRuby GPU 资源失败: %s: %s", type(e).__name__, e)
    elif _RUBY_ANNOTATOR is not None:
        try:
            _RUBY_ANNOTATOR.release_gpu()
        except Exception as e:
            logger.warning("释放全局 AutoRuby GPU 资源失败: %s: %s", type(e).__name__, e)
        _RUBY_ANNOTATOR = None


_INLINE_RUBY_RE = re.compile(r"\{([^|{}]+)\|([^|{}]*)\}")
_RUBY_READING_VALID_RE = re.compile(r"[ぁ-んァ-ヶー一-龯々A-Za-z0-9]")


def _validate_ruby_pair(surface: Any, reading: Any, where: str) -> tuple[str, str]:
    if not isinstance(surface, str) or not surface.strip():
        raise ValueError(f"ruby-map ({where}) 的词条键必须为非空字符串: {surface!r}")
    if not isinstance(reading, str) or not reading.strip():
        raise ValueError(f"ruby-map ({where}) 词条 {surface!r} 的读音必须为非空字符串: {reading!r}")
    for bad in ("{", "}", "|", "\n", "\r"):
        if bad in surface:
            raise ValueError(f"ruby-map ({where}) 词条键含非法字符 {bad!r}: {surface!r}")
        if bad in reading:
            raise ValueError(f"ruby-map ({where}) 词条 {surface!r} 的读音含非法字符 {bad!r}: {reading!r}")
    rd = reading.strip()
    if not _RUBY_READING_VALID_RE.search(rd):
        raise ValueError(f"ruby-map ({where}) 词条 {surface!r} 的读音缺少有效假名/发音字符: {reading!r}")
    return surface, rd


def _parse_by_line_dict(raw_bl: Any, where: str) -> Dict[int, Dict[str, str]]:
    if not isinstance(raw_bl, dict):
        raise ValueError(f"ruby-map 的 '{where}' 字段必须为字典对象")
    by_line: Dict[int, Dict[str, str]] = {}
    for ln_k, sub in raw_bl.items():
        try:
            ln_no = int(str(ln_k).strip())
        except ValueError as e:
            raise ValueError(f"ruby-map '{where}' 行号必须为正整数: {ln_k!r}") from e
        if ln_no <= 0:
            raise ValueError(f"ruby-map '{where}' 行号必须 >= 1: {ln_k!r}")
        if not isinstance(sub, dict):
            raise ValueError(f"ruby-map '{where}[{ln_no}]' 必须为字典对象")
        ln_dict: Dict[str, str] = {}
        for k, v in sub.items():
            s, r = _validate_ruby_pair(k, v, f"{where}.{ln_no}")
            ln_dict[s] = r
        if ln_dict:
            by_line[ln_no] = ln_dict
    return by_line


def load_ruby_map(source: Any) -> Dict[str, Any]:
    """加载并严格校验注音覆盖表（支持 JSON 文件路径或 dict）。

    支持结构：
      1) 扁平全局表: ``{"陽菜羽": "ひなは", "コソ練": "こそれん"}``
      2) 分层表:     ``{"global": {"私": "あたし"}, "by_line": {"12": {"本気": "まじ"}}, "by_file": {"04": {"by_line": {"197": {"晶": "あきら"}}}}}``
    返回归一化结构 ``{"global": Dict[str, str], "by_line": Dict[int, Dict[str, str]], "by_file": Dict[str, Dict[str, Any]]}``。
    """
    if source is None:
        return {"global": {}, "by_line": {}, "by_file": {}}
    if isinstance(source, (str, os.PathLike)):
        path = os.path.abspath(str(source))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"指定的注音覆盖表不存在: {path}")
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                raw = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"注音覆盖表 JSON 解析失败 ({path}): {e}") from e
    elif isinstance(source, dict):
        raw = source
    else:
        raise ValueError(f"ruby-map 必须为 JSON 文件路径或字典对象，实际为: {type(source).__name__}")

    if not isinstance(raw, dict):
        raise ValueError(f"ruby-map 顶层必须为 JSON 对象 (dict)，实际为: {type(raw).__name__}")

    g_map: Dict[str, str] = {}
    by_line: Dict[int, Dict[str, str]] = {}
    by_file: Dict[str, Dict[str, Any]] = {}

    _STRUCT_KEYS = ("global", "by_line", "by_file", "files", "comment", "_comment")
    is_structured = (
        any(k in raw for k in ("global", "by_line", "by_file", "files"))
        and all(k in _STRUCT_KEYS for k in raw.keys())
        and (
            ("global" in raw and isinstance(raw["global"], dict))
            or ("by_line" in raw and isinstance(raw["by_line"], dict))
            or ("by_file" in raw and isinstance(raw["by_file"], dict))
            or ("files" in raw and isinstance(raw["files"], dict))
        )
    )
    if is_structured:
        raw_g = raw.get("global")
        if raw_g is not None:
            if not isinstance(raw_g, dict):
                raise ValueError("ruby-map 的 'global' 字段必须为字典对象")
            for k, v in raw_g.items():
                s, r = _validate_ruby_pair(k, v, "global")
                g_map[s] = r
        raw_bl = raw.get("by_line")
        if raw_bl is not None:
            by_line = _parse_by_line_dict(raw_bl, "by_line")
        for bf_key in ("by_file", "files"):
            raw_bf = raw.get(bf_key)
            if raw_bf is None:
                continue
            if not isinstance(raw_bf, dict):
                raise ValueError(f"ruby-map 的 '{bf_key}' 字段必须为字典对象")
            for fk, fv in raw_bf.items():
                fk_str = str(fk).strip()
                if not fk_str:
                    raise ValueError(f"ruby-map '{bf_key}' 的文件键不能为空")
                if not isinstance(fv, dict):
                    raise ValueError(f"ruby-map '{bf_key}[{fk_str}]' 必须为字典对象")
                f_g: Dict[str, str] = {}
                f_bl: Dict[int, Dict[str, str]] = {}
                is_sub_struct = (
                    ("global" in fv or "by_line" in fv)
                    and all(k in ("global", "by_line", "comment", "_comment") for k in fv.keys())
                )
                if is_sub_struct:
                    if fv.get("global") is not None:
                        if not isinstance(fv["global"], dict):
                            raise ValueError(f"ruby-map '{bf_key}[{fk_str}].global' 必须为字典对象")
                        for k, v in fv["global"].items():
                            s, r = _validate_ruby_pair(k, v, f"{bf_key}.{fk_str}.global")
                            f_g[s] = r
                    if fv.get("by_line") is not None:
                        f_bl = _parse_by_line_dict(fv["by_line"], f"{bf_key}.{fk_str}.by_line")
                else:
                    # 简写形式：如果所有键都是正整数行号且值为 dict，视为 by_line；否则视为该文件的 global
                    if fv and all(str(k).strip().isdigit() and isinstance(v, dict) for k, v in fv.items()):
                        f_bl = _parse_by_line_dict(fv, f"{bf_key}.{fk_str}")
                    else:
                        for k, v in fv.items():
                            s, r = _validate_ruby_pair(k, v, f"{bf_key}.{fk_str}")
                            f_g[s] = r
                entry = by_file.setdefault(fk_str, {"global": {}, "by_line": {}})
                entry["global"].update(f_g)
                for ln_no, ln_d in f_bl.items():
                    entry["by_line"].setdefault(ln_no, {}).update(ln_d)
    else:
        for k, v in raw.items():
            s, r = _validate_ruby_pair(k, v, "root")
            g_map[s] = r

    return {"global": g_map, "by_line": by_line, "by_file": by_file}


def resolve_ruby_map_for_script(ruby_map: Any, script_path: Optional[str] = None) -> Dict[str, Any]:
    """根据当前台本路径解析合并后的生效 {global, by_line}（支持 by_file 文件名/基名/序号前缀匹配）。"""
    norm = load_ruby_map(ruby_map) if ruby_map is not None else {"global": {}, "by_line": {}, "by_file": {}}
    g_map = dict(norm.get("global", {}))
    bl_map: Dict[int, Dict[str, str]] = {ln: dict(d) for ln, d in norm.get("by_line", {}).items()}
    by_file = norm.get("by_file", {})
    if not by_file or not script_path:
        return {"global": g_map, "by_line": bl_map, "by_file": by_file}

    bname = os.path.basename(script_path)
    stem = os.path.splitext(bname)[0]
    bname_l = bname.lower()
    stem_l = stem.lower()
    m_num = re.match(r"^(\d{1,3})(?=[_\-\s.「【]|$)", stem)
    track_prefix = m_num.group(1) if m_num else None

    for fk, f_entry in by_file.items():
        fk_norm = os.path.basename(fk.replace("\\", "/")).strip()
        fk_l = fk_norm.lower()
        fk_stem_l = os.path.splitext(fk_l)[0]
        matched = (
            fk_l == bname_l
            or fk_l == stem_l
            or fk_stem_l == stem_l
            or (track_prefix is not None and fk_l == track_prefix.lower())
            or (len(fk_l) >= 2 and stem_l.startswith(fk_l) and (len(stem_l) == len(fk_l) or stem_l[len(fk_l)] in "_- 　.「【"))
        )
        if matched:
            g_map.update(f_entry.get("global", {}))
            for ln_no, ln_d in f_entry.get("by_line", {}).items():
                bl_map.setdefault(ln_no, {}).update(ln_d)

    return {"global": g_map, "by_line": bl_map, "by_file": by_file}


def apply_ruby_overrides_to_line(
    line: str,
    overrides: Dict[str, str],
    replace_existing: bool = False,
) -> tuple[str, int]:
    """将 {surface: reading} 精准应用到单行文本，绝不嵌套或破坏已有 {surface|reading} 结构。

    - ``replace_existing=False``（注音前预注入）：对裸文本按最长词优先包为 ``{surface|reading}``，
      已有 ``{s|r}`` 中若 ``s`` 命中 overrides 则直接更新其读音。
    - ``replace_existing=True``（注音后权威覆盖）：对已注音行中所有 ``{s|r}`` 若 ``s`` 命中 overrides
      强制改写为 ``{s|reading}``（防止 yomikata 异读层回写覆盖人工/LLM 修正），并对尚未包裹的纯文本片段补包。
    """
    if not line or not overrides:
        return line, 0

    keys_sorted = sorted(overrides.keys(), key=len, reverse=True)
    plain_pat = re.compile("|".join(re.escape(k) for k in keys_sorted))

    out_parts: List[str] = []
    hits = 0
    pos = 0
    for m in _INLINE_RUBY_RE.finditer(line):
        seg = line[pos:m.start()]
        if seg:
            def _rep_plain(pm):
                nonlocal hits
                surf = pm.group(0)
                hits += 1
                return f"{{{surf}|{overrides[surf]}}}"
            out_parts.append(plain_pat.sub(_rep_plain, seg))
        surf_ex, read_ex = m.group(1), m.group(2)
        if surf_ex in overrides:
            new_rd = overrides[surf_ex]
            if replace_existing or read_ex != new_rd:
                hits += 1
                out_parts.append(f"{{{surf_ex}|{new_rd}}}")
            else:
                out_parts.append(m.group(0))
        else:
            out_parts.append(m.group(0))
        pos = m.end()

    tail = line[pos:]
    if tail:
        def _rep_tail(pm):
            nonlocal hits
            surf = pm.group(0)
            hits += 1
            return f"{{{surf}|{overrides[surf]}}}"
        out_parts.append(plain_pat.sub(_rep_tail, tail))

    return "".join(out_parts), hits


def annotate_lines_with_overrides(
    ruby: Any,
    body_lines: Sequence[str],
    src_line_nos: Optional[Sequence[int]] = None,
    ruby_map: Optional[Dict[str, Any]] = None,
    script_path: Optional[str] = None,
) -> tuple[List[str], int]:
    """统一执行「预注入复合词 → AutoRuby 注音 → 权威后置覆写」，保证 --verify-script 与 run_pipeline 行为一致。"""
    norm_map = resolve_ruby_map_for_script(ruby_map, script_path=script_path) if ruby_map else {"global": {}, "by_line": {}}
    g_map: Dict[str, str] = norm_map["global"]
    bl_map: Dict[int, Dict[str, str]] = norm_map["by_line"]
    has_any = bool(g_map or bl_map)

    annotated: List[str] = []
    applied_total = 0
    for idx, b in enumerate(body_lines):
        ln_no = src_line_nos[idx] if (src_line_nos is not None and idx < len(src_line_nos)) else (idx + 1)
        if not has_any or not b.strip():
            annotated.append(ruby.annotate(b))
            continue
        eff = dict(g_map)
        if ln_no in bl_map:
            eff.update(bl_map[ln_no])
        if not eff:
            annotated.append(ruby.annotate(b))
            continue
        # 预注入阶段仅注入 >=2 字的复合词/短语，单字词条留给 AutoRuby 分词后再由后置阶段对独立 {单字|...} 覆写，
        # 彻底防止单字覆写（如 {"中": "なか"}）在分词前把裸文本中的复合词（如「途中」）从中间劈成「途{中|なか}」。
        pre_eff = {k: v for k, v in eff.items() if len(k) >= 2}
        if pre_eff:
            pre_b, pre_hits = apply_ruby_overrides_to_line(b, pre_eff, replace_existing=False)
        else:
            pre_b, pre_hits = b, 0
        ann_b = ruby.annotate(pre_b)
        post_b, post_hits = apply_ruby_overrides_to_line(ann_b, eff, replace_existing=True)
        annotated.append(post_b)
        applied_total += max(pre_hits, post_hits)
    return annotated, applied_total


_COLLOC_STOP_RE = re.compile(r"[、。！？!?,.\s「」『』（）()【】〖〗♡♥♪・…‥ー～〜~\-—―]")
_AFFIX_PREFIX_CHARS = frozenset("おご御")
_AFFIX_SUFFIX_CHARS = frozenset("様さんちゃんくん達共目中内外的性化者員家屋所場側方用式風系")
_PARTICLE_STOP_CHARS = frozenset("はがをにでとのもへやかねよぞぜわさ")


def _is_kanji_char(ch: str) -> bool:
    return bool(ch) and ("\u4e00" <= ch <= "\u9fff" or ch == "々")


def _is_kana_char(ch: str) -> bool:
    return bool(ch) and ("\u3040" <= ch <= "\u309f" or "\u30a0" <= ch <= "\u30ff")


def _extract_local_collocation(plain_line: str, p_st: int, p_en: int) -> tuple[str, bool]:
    """提取目标词在原句中的紧邻局部搭配（如 お+母+様 -> お母様），并判断是否存在单字切碎/接辞风险。"""
    surf = plain_line[p_st:p_en]
    left_ch = plain_line[p_st - 1] if p_st > 0 else ""
    right_ch = plain_line[p_en] if p_en < len(plain_line) else ""
    left_ok = bool(left_ch) and not _COLLOC_STOP_RE.match(left_ch)
    right_ok = bool(right_ch) and not _COLLOC_STOP_RE.match(right_ch)

    affix_risk = False
    if len(surf) == 1:
        if left_ok and (left_ch in _AFFIX_PREFIX_CHARS or _is_kanji_char(left_ch) or ("\u30a0" <= left_ch <= "\u30ff")):
            affix_risk = True
        if right_ok and (right_ch in _AFFIX_SUFFIX_CHARS or _is_kanji_char(right_ch) or ("\u30a0" <= right_ch <= "\u30ff")):
            affix_risk = True

        # 单汉字：
        #   - 左侧优先保留敬语前缀(お/ご/御)、相邻汉字或片假名；若右侧已构成汉字/接辞搭配（如 母親），
        #     则不把左侧无关助词/形容动词尾（如 な/た）混入搭配键，保证「な母親 / た母親」归并成同一「母親」。
        left_strong = left_ok and (
            left_ch in _AFFIX_PREFIX_CHARS or _is_kanji_char(left_ch) or ("\u30a0" <= left_ch <= "\u30ff")
        )
        right_strong = right_ok and (
            right_ch in _AFFIX_SUFFIX_CHARS or _is_kanji_char(right_ch) or ("\u30a0" <= right_ch <= "\u30ff")
        )
        if right_ok and (right_ch not in _PARTICLE_STOP_CHARS or right_strong):
            r_part = right_ch
        else:
            r_part = ""
        if left_strong:
            l_part = left_ch
        elif left_ok and not right_strong and left_ch not in ("な", "た", "て", "で", "を", "が", "は", "も", "と"):
            l_part = left_ch
        else:
            l_part = ""
        colloc = f"{l_part}{surf}{r_part}"
        return colloc, affix_risk

    # 多字词：若前后紧贴敬语接辞（如 お+兄+様）或汉字则补上 1 字，否则取词形本身
    l_part = left_ch if (left_ok and left_ch in _AFFIX_PREFIX_CHARS) else ""
    r_part = right_ch if (right_ok and right_ch in _AFFIX_SUFFIX_CHARS) else ""
    colloc = f"{l_part}{surf}{r_part}"
    return colloc, bool(l_part or r_part)


def _classify_ruby_priority(
    surf: str,
    is_polyphone: bool,
    has_affix_risk: bool,
    distinct_collocs: int,
    speaker_names: Sequence[str] = (),
) -> tuple[str, List[str]]:
    """按误读风险将注音词条分为 high（高危必查）与 normal（常规单读音词）。"""
    reasons: List[str] = []
    if is_polyphone:
        reasons.append("polyphone")
    if has_affix_risk:
        reasons.append("affix_split")
    if any(ch in surf for ch in "ー～〜~") or (
        any("\u30a0" <= ch <= "\u30ff" for ch in surf) and any(_is_kanji_char(ch) for ch in surf)
    ):
        reasons.append("mixed_script")
    if speaker_names and any(surf in spk or spk in surf for spk in speaker_names if spk):
        reasons.append("speaker_name")
    return ("high" if reasons else "normal"), reasons


def extract_ruby_glossary(
    annotated_lines: Sequence[str],
    src_line_nos: Optional[Sequence[int]] = None,
    max_examples: int = 6,
    speaker_names: Optional[Sequence[str]] = None,
    file_label: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """从注音后的行列表提取去重词表（供 LLM 审校，不输出整篇原文）。

    - 按「(读音, 局部搭配)」去重采样上下文（例如 `母->はは` 同时出现在 `母親` 与 `お母様` 时各保留一条，
      消除此前仅取前 2 条导致后文错词被隐藏的盲区）；
    - 自动分层标记 ``priority``（``high`` / ``normal``）：多音词排最前，其次高危单字/接辞切碎/角色名，
      最后常规多字单读音熟语；
    - ``lines_by_reading`` 完整记录每种读音对应的全部 1-based 原台本行号，供直接构造 ``by_line``。
    """
    spk_list = [str(s).strip() for s in (speaker_names or ()) if str(s).strip()]
    entries: Dict[str, Dict[str, Any]] = {}
    for idx, ann in enumerate(annotated_lines):
        if not ann or "{" not in ann:
            continue
        ln_no = src_line_nos[idx] if (src_line_nos is not None and idx < len(src_line_nos)) else (idx + 1)
        plain_parts: List[str] = []
        match_spans: List[tuple[int, int, str, str]] = []
        pos = 0
        plain_len = 0
        for m in _INLINE_RUBY_RE.finditer(ann):
            seg = ann[pos:m.start()]
            if seg:
                plain_parts.append(seg)
                plain_len += len(seg)
            surf, rd = m.group(1), m.group(2)
            match_spans.append((plain_len, plain_len + len(surf), surf, rd))
            plain_parts.append(surf)
            plain_len += len(surf)
            pos = m.end()
        if pos < len(ann):
            plain_parts.append(ann[pos:])
        plain_line = "".join(plain_parts).strip()

        for p_st, p_en, surf, rd in match_spans:
            colloc, affix_risk = _extract_local_collocation(plain_line, p_st, p_en)
            rec = entries.get(surf)
            if rec is None:
                rec = {
                    "surface": surf,
                    "readings": {},
                    "lines_by_reading": {},
                    "total_count": 0,
                    "examples": [],
                    "collocations": [],
                    "_colloc_set": set(),
                    "_seen_rd_colloc": set(),
                    "_ex_count_by_rd": {},
                    "_affix_risk": False,
                }
                entries[surf] = rec
            rec["readings"][rd] = rec["readings"].get(rd, 0) + 1
            ln_list = rec["lines_by_reading"].setdefault(rd, [])
            if not ln_list or ln_list[-1] != ln_no:
                ln_list.append(ln_no)
            rec["total_count"] += 1
            if affix_risk:
                rec["_affix_risk"] = True
            if colloc not in rec["_colloc_set"]:
                rec["_colloc_set"].add(colloc)
                if len(rec["collocations"]) < 8:
                    rec["collocations"].append(colloc)

            # 按 (reading, collocation) 去重采样上下文，每种读音最多保留 max_examples 条不同搭配
            rd_ex_cnt = rec["_ex_count_by_rd"].get(rd, 0)
            pair_key = (rd, colloc)
            if pair_key not in rec["_seen_rd_colloc"] and rd_ex_cnt < max(1, int(max_examples)):
                ctx_lo = max(0, p_st - 10)
                ctx_hi = min(len(plain_line), p_en + 12)
                snippet = plain_line[ctx_lo:ctx_hi]
                if ctx_lo > 0:
                    snippet = "…" + snippet
                if ctx_hi < len(plain_line):
                    snippet = snippet + "…"
                ex_item: Dict[str, Any] = {
                    "line": ln_no,
                    "reading": rd,
                    "collocation": colloc,
                    "context": snippet,
                }
                if file_label is not None:
                    ex_item["file"] = file_label
                rec["examples"].append(ex_item)
                rec["_seen_rd_colloc"].add(pair_key)
                rec["_ex_count_by_rd"][rd] = rd_ex_cnt + 1

    glossary: List[Dict[str, Any]] = []
    for rec in entries.values():
        rds = dict(sorted(rec["readings"].items(), key=lambda kv: (-kv[1], kv[0])))
        lbr = {r: rec["lines_by_reading"].get(r, []) for r in rds.keys()}
        is_poly = len(rds) > 1
        priority, p_reasons = _classify_ruby_priority(
            rec["surface"],
            is_polyphone=is_poly,
            has_affix_risk=rec["_affix_risk"],
            distinct_collocs=len(rec["_colloc_set"]),
            speaker_names=spk_list,
        )
        glossary.append({
            "surface": rec["surface"],
            "priority": priority,
            "priority_reasons": p_reasons,
            "readings": rds,
            "lines_by_reading": lbr,
            "count": rec["total_count"],
            "is_polyphone": is_poly,
            "collocations": rec["collocations"],
            "examples": rec["examples"],
        })
    glossary.sort(key=lambda x: (not x["is_polyphone"], 0 if x["priority"] == "high" else 1, -x["count"], x["surface"]))
    return glossary


def merge_ruby_glossaries(
    per_file_glossaries: Sequence[tuple[str, Sequence[Dict[str, Any]]]],
    speaker_names: Optional[Sequence[str]] = None,
    max_examples_per_reading: int = 6,
) -> List[Dict[str, Any]]:
    """将批量多轨的注音词表合并去重为单份全局词表，保留按文件分桶的 ``lines_by_file_reading``。"""
    spk_list = [str(s).strip() for s in (speaker_names or ()) if str(s).strip()]
    merged: Dict[str, Dict[str, Any]] = {}
    for file_key, gl in per_file_glossaries:
        for item in gl:
            surf = item["surface"]
            rec = merged.get(surf)
            if rec is None:
                rec = {
                    "surface": surf,
                    "readings": {},
                    "lines_by_file_reading": {},
                    "total_count": 0,
                    "collocations": [],
                    "_colloc_set": set(),
                    "examples": [],
                    "_seen_rd_colloc": set(),
                    "_ex_count_by_rd": {},
                    "_affix_risk": False,
                }
                merged[surf] = rec
            if "affix_split" in item.get("priority_reasons", ()):
                rec["_affix_risk"] = True
            for rd, cnt in item.get("readings", {}).items():
                rec["readings"][rd] = rec["readings"].get(rd, 0) + int(cnt)
            rec["total_count"] += int(item.get("count", 0))
            for rd, lns in item.get("lines_by_reading", {}).items():
                f_bucket = rec["lines_by_file_reading"].setdefault(file_key, {})
                dst = f_bucket.setdefault(rd, [])
                for ln in lns:
                    if ln not in dst:
                        dst.append(ln)
            for c in item.get("collocations", ()):
                if c not in rec["_colloc_set"]:
                    rec["_colloc_set"].add(c)
                    if len(rec["collocations"]) < 8:
                        rec["collocations"].append(c)
            for ex in item.get("examples", ()):
                rd = ex["reading"]
                colloc = ex.get("collocation", surf)
                pair_key = (rd, colloc)
                rd_cnt = rec["_ex_count_by_rd"].get(rd, 0)
                if pair_key not in rec["_seen_rd_colloc"] and rd_cnt < max(1, int(max_examples_per_reading)):
                    ex_copy = dict(ex)
                    ex_copy.setdefault("file", file_key)
                    rec["examples"].append(ex_copy)
                    rec["_seen_rd_colloc"].add(pair_key)
                    rec["_ex_count_by_rd"][rd] = rd_cnt + 1

    out: List[Dict[str, Any]] = []
    for rec in merged.values():
        rds = dict(sorted(rec["readings"].items(), key=lambda kv: (-kv[1], kv[0])))
        is_poly = len(rds) > 1
        priority, p_reasons = _classify_ruby_priority(
            rec["surface"],
            is_polyphone=is_poly,
            has_affix_risk=rec["_affix_risk"],
            distinct_collocs=len(rec["_colloc_set"]),
            speaker_names=spk_list,
        )
        out.append({
            "surface": rec["surface"],
            "priority": priority,
            "priority_reasons": p_reasons,
            "readings": rds,
            "lines_by_file_reading": rec["lines_by_file_reading"],
            "count": rec["total_count"],
            "is_polyphone": is_poly,
            "collocations": rec["collocations"],
            "examples": rec["examples"],
        })
    out.sort(key=lambda x: (not x["is_polyphone"], 0 if x["priority"] == "high" else 1, -x["count"], x["surface"]))
    return out
