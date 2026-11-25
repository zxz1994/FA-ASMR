# -*- coding: utf-8 -*-
"""FA-ASMR 字幕原地微调、紧凑翻译导出、回填与深度校验工具。

核心子命令：
  1. patch  : 在定稿 .srt + .lrc 上原地同步执行删句/插句/改时/改字并重排序号（无重大结构错位无需全局重跑）
  2. export : 将日文定稿目录导出为紧凑待翻文本（支持 --dedup-pov 自动提取跨视角共句去重）
  3. build  : 将翻译 JSON 回填生成中文 .srt + .lrc（支持 idx| 防串行锚点校验、BOM 清洗与 @Sxxxx 共句展开）
  4. verify : 批量终检 1:1 时间轴一致性 + 纯汉字日文残留/禁用拟声词/跨视角同句异译体检 + 双语抽检
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

KANA_RE = re.compile(r"[\u3040-\u309F\u30A0-\u30FF]")
SRT_SPK_RE = re.compile(r"^(【[^】]+】)(.*)$")
LRC_TS_RE = re.compile(r"^(\[\d+:\d+[.:]\d+\])(.*)$")
LRC_META_RE = re.compile(r"^\[[a-zA-Z]{2,8}:.*\]$")
SRT_TS_LINE_RE = re.compile(
    r"^\s*(\d{2,}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2,}:\d{2}:\d{2}[,.]\d{3})\s*$"
)
INDEXED_LINE_RE = re.compile(r"^(\d+)\|(.*)$")
SHARED_REF_RE = re.compile(r"^@(S\d{4,})$")
SHARED_DEF_RE = re.compile(r"^(S\d{4,})\|(.*)$")
TRACK_GROUP_RE = re.compile(r"(?:^|(?:track|trk|トラック|第)\s*0*)(\d{1,3})(?=\D|$)", re.IGNORECASE)

# 简体中文字幕高置信度日文残留检测（仅作用于剥离说话人前缀后的译文正文）
JP_SHINJITAI_RE = re.compile(r"[気変悪発対駅嬢]")
JP_UNTRANSLATED_WORD_RE = re.compile(r"(?:大丈夫|我慢|邪魔|先輩|後輩|彼氏|旦那|無理矢理|一生懸命)")
BANNED_ONO_RE = re.compile(r"(?:滋溜|嘞咯|舔舐)")
FULLWIDTH_LATIN_TYPO_RE = re.compile(r"[ｘＸ]")


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _clean_bom(s: str) -> str:
    return str(s).replace("\ufeff", "")


def parse_srt_time_sec(val: Any) -> float:
    """支持 'HH:MM:SS,mmm' / 'MM:SS.xx' 或秒数 float/str 转为浮点秒。"""
    if isinstance(val, (int, float)):
        return max(0.0, float(val))
    s = _clean_bom(str(val)).strip()
    m = re.match(r"^(\d+):(\d{2}):(\d{2})[,.](\d{1,3})$", s)
    if m:
        ms_s = m.group(4).ljust(3, "0")[:3]
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)) + int(ms_s) / 1000.0
    m2 = re.match(r"^\[?(\d+):(\d{2})[.:](\d{1,3})\]?$", s)
    if m2:
        frac = m2.group(3)
        scale = 100.0 if len(frac) <= 2 else 1000.0
        return int(m2.group(1)) * 60 + int(m2.group(2)) + int(frac.ljust(2, "0")) / scale
    return max(0.0, float(s))


def format_srt_time(sec: float) -> str:
    ms_total = max(0, int(round(float(sec) * 1000.0)))
    hh = ms_total // 3600000
    rem = ms_total % 3600000
    mm = rem // 60000
    rem %= 60000
    ss = rem // 1000
    ms = rem % 1000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def format_lrc_time_from_cs(cs: int) -> str:
    cs_i = max(0, int(cs))
    return f"[{cs_i // 6000:02d}:{(cs_i // 100) % 60:02d}.{cs_i % 100:02d}]"


def parse_ts_range_sec(ts_line: str) -> Tuple[float, float]:
    m = SRT_TS_LINE_RE.match(ts_line.strip())
    if not m:
        raise ValueError(f"无效 SRT 时间轴行: {ts_line!r}")
    return parse_srt_time_sec(m.group(1)), parse_srt_time_sec(m.group(2))


def parse_srt(path: pathlib.Path) -> List[Tuple[str, str, str]]:
    text = _clean_bom(path.read_text(encoding="utf-8-sig")).strip()
    if not text:
        return []
    blocks = [b.splitlines() for b in re.split(r"\r?\n\r?\n", text) if b.strip()]
    out: List[Tuple[str, str, str]] = []
    for i, blk in enumerate(blocks, 1):
        if len(blk) < 3:
            raise ValueError(f"[{path.name}] 第 {i} 块 SRT 格式异常: {blk}")
        out.append((blk[0].strip(), blk[1].strip(), " ".join(x.strip() for x in blk[2:]).strip()))
    return out


def strip_srt_spk(text: str) -> Tuple[str, str]:
    s = _clean_bom(text).strip()
    m = SRT_SPK_RE.match(s)
    if m:
        return m.group(1), m.group(2).strip()
    return "", s


def _track_group_key(stem: str) -> Optional[str]:
    m = re.match(r"^0*(\d{1,3})(?=[_\-\u2017\s.「【]|$)", stem) or TRACK_GROUP_RE.search(stem)
    return f"T{int(m.group(1)):02d}" if m else None


def _filter_srts_by_only(srts: Sequence[pathlib.Path], only_raw: str) -> List[pathlib.Path]:
    only_tokens = [x.strip().lower() for x in re.split(r"[,，;；]+", only_raw or "") if x.strip()]
    if not only_tokens:
        return list(srts)
    filtered: List[pathlib.Path] = []
    for p in srts:
        stem = p.stem
        stem_l = stem.lower()
        m_num = re.match(r"^0*(\d+)(?=[_\-\u2017\s.「【]|$)", stem) or TRACK_GROUP_RE.search(stem)
        s_num = int(m_num.group(1)) if m_num else None
        if any(
            (tok.isdigit() and s_num is not None and s_num == int(tok)) or (not tok.isdigit() and tok in stem_l)
            for tok in only_tokens
        ):
            filtered.append(p)
    return filtered


# ---------------------------------------------------------------------------
# 1. patch: 原地同步微调 .srt + .lrc（避免无重大错位时重跑整轨全局对齐）
# ---------------------------------------------------------------------------

def render_lrc_from_cues(
    cues_sec: Sequence[Tuple[float, float, str, str]],
    header_lines: Optional[Sequence[str]] = None,
) -> str:
    """由 (start_sec, end_sec, speaker, body) 列表确定性重建标准 LRC（含驻留保底与空行清屏）。"""
    out: List[str] = list(header_lines or [])
    active_max_end_cs: Optional[int] = None
    n = len(cues_sec)
    for idx, (st_s, en_s, spk, body) in enumerate(cues_sec):
        cs_start = max(0, int(round(st_s * 100.0)))
        cs_end = max(cs_start, int(round(en_s * 100.0)))
        cs_next = max(0, int(round(cues_sec[idx + 1][0] * 100.0))) if (idx + 1) < n else None

        if cs_end < cs_start + 100:
            target_cs = cs_start + 100
            if cs_next is not None:
                target_cs = min(target_cs, cs_next)
            if target_cs > cs_end:
                cs_end = max(cs_start, target_cs)

        if active_max_end_cs is None or cs_end > active_max_end_cs:
            active_max_end_cs = cs_end
        clear_cs = max(cs_end, active_max_end_cs)
        if cs_next is not None and clear_cs > cs_next:
            clear_cs = cs_next
        cs_end = clear_cs

        content = f"{spk}：{body}" if spk else body
        out.append(f"{format_lrc_time_from_cs(cs_start)}{content}")
        if cs_end > cs_start and (cs_next is None or cs_end < cs_next):
            out.append(format_lrc_time_from_cs(cs_end))
    return "\n".join(out) + ("\n" if out else "")


def apply_patch_ops_to_files(
    srt_path: pathlib.Path,
    ops: Sequence[Dict[str, Any]],
    lrc_path: Optional[pathlib.Path] = None,
) -> Tuple[int, int]:
    """对单条轨道的 .srt（及同基名 .lrc）执行原子原地修轴操作。

    所有 cue 索引均基于修改前的原始 1-based SRT 序号定位，同一批次内删改互不干扰偏移。
    """
    if not srt_path.is_file():
        raise FileNotFoundError(f"找不到待修补的 SRT 文件: {srt_path}")
    if lrc_path is None:
        cand_lrc = srt_path.with_suffix(".lrc")
        lrc_path = cand_lrc if cand_lrc.is_file() else None

    raw_cues = parse_srt(srt_path)
    items: List[Dict[str, Any]] = []
    for idx, (_num, ts, full_txt) in enumerate(raw_cues, 1):
        st_s, en_s = parse_ts_range_sec(ts)
        pfx, body = strip_srt_spk(full_txt)
        items.append({
            "orig_idx": idx,
            "start": st_s,
            "end": en_s,
            "speaker": pfx[1:-1] if pfx else "",
            "body": body,
            "deleted": False,
        })

    n_orig = len(items)

    def _get_item(c_idx: Any, op_name: str, expect: Optional[str] = None) -> Dict[str, Any]:
        ci = int(c_idx)
        if ci < 1 or ci > n_orig:
            raise ValueError(f"[{srt_path.stem}] {op_name} 目标序号越界: cue={ci} (总条数={n_orig})")
        target = items[ci - 1]
        if expect and str(expect) not in target["body"] and str(expect) not in target["speaker"]:
            raise ValueError(
                f"[{srt_path.stem}] {op_name} cue={ci} 内容不匹配预期 {expect!r}: 实际为 {target['body']!r}"
            )
        return target

    inserts: List[Dict[str, Any]] = []
    for step_no, op_obj in enumerate(ops, 1):
        if not isinstance(op_obj, dict) or not op_obj.get("op"):
            raise ValueError(f"[{srt_path.stem}] 第 {step_no} 条 patch 指令缺少 'op' 字段: {op_obj}")
        kind = str(op_obj["op"]).strip().lower()
        expect = op_obj.get("expect")

        if kind == "delete":
            raw_ids = op_obj.get("cues") if "cues" in op_obj else [op_obj.get("cue")]
            if not raw_ids or raw_ids == [None]:
                raise ValueError(f"[{srt_path.stem}] delete 指令必须提供 cue 或 cues")
            for cid in raw_ids:
                it = _get_item(cid, "delete", expect=expect if len(raw_ids) == 1 else None)
                it["deleted"] = True

        elif kind == "update":
            if "cue" not in op_obj:
                raise ValueError(f"[{srt_path.stem}] update 指令必须提供 cue")
            it = _get_item(op_obj["cue"], "update", expect=expect)
            if "start" in op_obj and op_obj["start"] is not None:
                it["start"] = parse_srt_time_sec(op_obj["start"])
            if "end" in op_obj and op_obj["end"] is not None:
                it["end"] = parse_srt_time_sec(op_obj["end"])
            if "speaker" in op_obj and op_obj["speaker"] is not None:
                it["speaker"] = _clean_bom(str(op_obj["speaker"])).strip().strip("【】")
            if "text" in op_obj and op_obj["text"] is not None:
                pfx_n, body_n = strip_srt_spk(str(op_obj["text"]))
                if pfx_n and "speaker" not in op_obj:
                    it["speaker"] = pfx_n[1:-1]
                it["body"] = body_n

        elif kind == "replace":
            old_s = _clean_bom(str(op_obj.get("old", "")))
            new_s = _clean_bom(str(op_obj.get("new", "")))
            if not old_s:
                raise ValueError(f"[{srt_path.stem}] replace 指令的 'old' 不能为空")
            targets = [_get_item(op_obj["cue"], "replace", expect=expect)] if "cue" in op_obj else items
            replaced_any = False
            for it in targets:
                if not it["deleted"] and old_s in it["body"]:
                    it["body"] = it["body"].replace(old_s, new_s).strip()
                    replaced_any = True
            if not replaced_any:
                raise ValueError(f"[{srt_path.stem}] replace 未命中目标子串 {old_s!r}")

        elif kind == "insert":
            if op_obj.get("start") is None or op_obj.get("end") is None or op_obj.get("text") is None:
                raise ValueError(f"[{srt_path.stem}] insert 指令必须提供 start, end, text")
            st_i = parse_srt_time_sec(op_obj["start"])
            en_i = parse_srt_time_sec(op_obj["end"])
            pfx_i, body_i = strip_srt_spk(str(op_obj["text"]))
            spk_i = _clean_bom(str(op_obj.get("speaker") or (pfx_i[1:-1] if pfx_i else ""))).strip().strip("【】")
            inserts.append({
                "orig_idx": n_orig + len(inserts) + 1,
                "start": st_i,
                "end": en_i,
                "speaker": spk_i,
                "body": body_i,
                "deleted": False,
            })
        else:
            raise ValueError(f"[{srt_path.stem}] 不支持的 patch op={kind!r}（支持: delete/update/replace/insert）")

    kept = [it for it in (items + inserts) if not it["deleted"]]
    for it in kept:
        if not it["body"]:
            raise ValueError(f"[{srt_path.stem}] 修补后存在空字幕正文 (orig_idx={it['orig_idx']})")
        if it["end"] < it["start"]:
            raise ValueError(
                f"[{srt_path.stem}] 修补后结束时间早于开始时间 (orig_idx={it['orig_idx']}: {it['start']} > {it['end']})"
            )

    kept.sort(key=lambda x: (round(x["start"], 3), round(x["end"], 3), x["orig_idx"]))
    srt_blocks: List[str] = []
    cues_for_lrc: List[Tuple[float, float, str, str]] = []
    for new_idx, it in enumerate(kept, 1):
        ts_str = f"{format_srt_time(it['start'])} --> {format_srt_time(it['end'])}"
        full_line = f"【{it['speaker']}】{it['body']}" if it["speaker"] else it["body"]
        srt_blocks.append(f"{new_idx}\n{ts_str}\n{full_line}\n")
        cues_for_lrc.append((it["start"], it["end"], it["speaker"], it["body"]))

    srt_path.write_text("\n".join(srt_blocks), encoding="utf-8")

    lrc_lines_written = 0
    if lrc_path is not None:
        headers: List[str] = []
        if lrc_path.is_file():
            for ln in _clean_bom(lrc_path.read_text(encoding="utf-8-sig")).splitlines():
                if LRC_META_RE.match(ln.strip()) and not LRC_TS_RE.match(ln.strip()):
                    headers.append(ln.strip())
                else:
                    break
        lrc_text = render_lrc_from_cues(cues_for_lrc, header_lines=headers)
        lrc_path.write_text(lrc_text, encoding="utf-8")
        lrc_lines_written = len(lrc_text.splitlines())

    return len(srt_blocks), lrc_lines_written


def cmd_patch(args: argparse.Namespace) -> int:
    if args.ops_json:
        if not args.sub_dir:
            print("[ERROR] 使用 --ops-json 批量修轴时必须指定 --sub-dir", file=sys.stderr)
            return 2
        sub_dir = pathlib.Path(args.sub_dir)
        payload = json.loads(_clean_bom(pathlib.Path(args.ops_json).read_text(encoding="utf-8-sig")))
        tracks_map = payload.get("tracks") if isinstance(payload, dict) and "tracks" in payload else payload
        if not isinstance(tracks_map, dict):
            print("[ERROR] --ops-json 顶层必须为 {stem: [ops...]} 对象", file=sys.stderr)
            return 2
        for stem, ops in tracks_map.items():
            if not isinstance(ops, list):
                raise ValueError(f"[{stem}] ops 必须为列表")
            n_srt, n_lrc = apply_patch_ops_to_files(sub_dir / f"{stem}.srt", ops)
            print(f"[PATCH OK] {stem} -> SRT({n_srt} cues), LRC({n_lrc} lines)")
        return 0

    if not args.srt:
        print("[ERROR] 请提供 --ops-json + --sub-dir，或提供单文件 --srt", file=sys.stderr)
        return 2

    srt_p = pathlib.Path(args.srt)
    ops_single: List[Dict[str, Any]] = []
    if args.delete:
        c_ids = [int(x.strip()) for x in re.split(r"[,，\s]+", args.delete) if x.strip()]
        ops_single.append({"op": "delete", "cues": c_ids})
    if args.replace:
        if "=" not in args.replace:
            print("[ERROR] --replace 格式须为 '旧文本=新文本'（如 'ｘ='）", file=sys.stderr)
            return 2
        old_s, new_s = args.replace.split("=", 1)
        rep_op: Dict[str, Any] = {"op": "replace", "old": old_s, "new": new_s}
        if args.cue is not None:
            rep_op["cue"] = args.cue
        ops_single.append(rep_op)
    elif args.cue is not None and any(v is not None for v in (args.start, args.end, args.text, args.speaker)):
        upd: Dict[str, Any] = {"op": "update", "cue": args.cue}
        if args.start is not None:
            upd["start"] = args.start
        if args.end is not None:
            upd["end"] = args.end
        if args.text is not None:
            upd["text"] = args.text
        if args.speaker is not None:
            upd["speaker"] = args.speaker
        ops_single.append(upd)
    if args.insert_text is not None:
        if args.start is None or args.end is None:
            print("[ERROR] --insert-text 必须同时配合 --start 与 --end", file=sys.stderr)
            return 2
        ops_single.append({
            "op": "insert",
            "start": args.start,
            "end": args.end,
            "speaker": args.speaker or "",
            "text": args.insert_text,
        })

    if not ops_single:
        print("[ERROR] 未指定任何修补动作（--delete / --replace / --cue + --start|--end|--text / --insert-text）", file=sys.stderr)
        return 2

    n_srt, n_lrc = apply_patch_ops_to_files(srt_p, ops_single)
    print(f"[PATCH OK] {srt_p.stem} -> SRT({n_srt} cues), LRC({n_lrc} lines)")
    return 0


# ---------------------------------------------------------------------------
# 2. export: 紧凑导出（支持 --dedup-pov 双视角/跨轨共句去重）
# ---------------------------------------------------------------------------

def cmd_export(args: argparse.Namespace) -> int:
    jp_dir = pathlib.Path(args.jp_dir)
    if not jp_dir.is_dir():
        print(f"[ERROR] 目录不存在: {jp_dir}", file=sys.stderr)
        return 2

    srts = _filter_srts_by_only(sorted(jp_dir.glob("*.srt")), args.only or "")
    parsed_tracks: List[Tuple[str, List[Tuple[int, str, str]]]] = []
    total_cues = 0
    for p in srts:
        cues = parse_srt(p)
        rows: List[Tuple[int, str, str]] = []
        for idx, (_num, _ts, body) in enumerate(cues, 1):
            pfx, txt = strip_srt_spk(body)
            spk = pfx[1:-1] if pfx else ""
            rows.append((idx, spk, txt))
        parsed_tracks.append((p.stem, rows))
        total_cues += len(rows)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if getattr(args, "dedup_pov", False):
        # 统计出现在 >=2 条不同音轨中的完全相同 (speaker, text) 台词
        track_presence: Dict[Tuple[str, str], set] = {}
        ordered_keys: List[Tuple[str, str]] = []
        for stem, rows in parsed_tracks:
            for _idx, spk, txt in rows:
                key = (spk, txt)
                if key not in track_presence:
                    track_presence[key] = set()
                    ordered_keys.append(key)
                track_presence[key].add(stem)

        shared_id_map: Dict[Tuple[str, str], str] = {}
        shared_list: List[str] = []
        for key in ordered_keys:
            if len(track_presence[key]) >= 2:
                sid = f"S{len(shared_list) + 1:04d}"
                shared_id_map[key] = sid
                spk, txt = key
                shared_list.append(f"{sid}|{spk}|{txt}" if spk else f"{sid}|{txt}")

        tracks_out: Dict[str, List[str]] = {}
        ref_count = 0
        for stem, rows in parsed_tracks:
            t_lines: List[str] = []
            for idx, spk, txt in rows:
                sid = shared_id_map.get((spk, txt))
                if sid is not None:
                    t_lines.append(f"{idx}|@{sid}")
                    ref_count += 1
                else:
                    t_lines.append(f"{idx}|{spk}|{txt}" if spk else f"{idx}|{txt}")
            tracks_out[stem] = t_lines

        saved = max(0, ref_count - len(shared_list))
        dedup_payload = {
            "format": "dedup_v1",
            "stats": {
                "tracks": len(parsed_tracks),
                "total_cues": total_cues,
                "shared_unique": len(shared_list),
                "shared_refs": ref_count,
                "saved_duplicate_cues": saved,
            },
            "shared": shared_list,
            "tracks": tracks_out,
        }
        out_path.write_text(json.dumps(dedup_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"[OK] 已导出 {len(parsed_tracks)} 轨去重紧凑包（共 {total_cues} 条，提取跨轨共句 {len(shared_list)} 条，省去重复翻译 {saved} 条） -> {out_path}"
        )
        return 0

    bundle: Dict[str, List[str]] = {}
    for stem, rows in parsed_tracks:
        bundle[stem] = [f"{idx}|{spk}|{txt}" if spk else f"{idx}|{txt}" for idx, spk, txt in rows]

    if out_path.suffix.lower() == ".txt":
        chunks: List[str] = []
        for stem, lines in bundle.items():
            chunks.append(f"=== {stem} ({len(lines)}) ===")
            chunks.extend(lines)
            chunks.append("")
        out_path.write_text("\n".join(chunks), encoding="utf-8")
    else:
        out_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] 已导出 {len(bundle)} 轨紧凑文本（共 {total_cues} 条） -> {out_path}")
    return 0


# ---------------------------------------------------------------------------
# 3. build & 4. verify: 锚点防错回填 + 质量体检 + 双语抽检
# ---------------------------------------------------------------------------

def _parse_shared_translations(raw_shared: Any) -> Dict[str, str]:
    if not raw_shared:
        return {}
    out: Dict[str, str] = {}
    if isinstance(raw_shared, dict):
        for k, v in raw_shared.items():
            sid = _clean_bom(str(k)).strip().lstrip("@")
            val = _clean_bom(str(v)).strip()
            # 兼容 "S0001|凛咲|译文" 或纯译文
            m_def = SHARED_DEF_RE.match(val)
            if m_def and m_def.group(1) == sid:
                val = m_def.group(2).strip()
            out[sid] = val
        return out
    if isinstance(raw_shared, list):
        for item in raw_shared:
            s = _clean_bom(str(item)).strip()
            m = SHARED_DEF_RE.match(s)
            if not m:
                raise ValueError(f"shared 列表项必须以 'S0001|译文' 开头，实际为: {s!r}")
            out[m.group(1)] = m.group(2).strip()
        return out
    raise ValueError(f"shared 字段必须为 dict 或 list，实际为 {type(raw_shared).__name__}")


def _strip_indexed_entry(
    stem: str,
    expected_idx: int,
    raw_entry: str,
    expected_spk: str,
    shared_map: Dict[str, str],
    require_index: bool = False,
) -> Tuple[str, bool]:
    """解析单条译文：剥除 BOM、校验可选的 'idx|' 行号锚点与说话人列、展开 '@S0001' 共句引用。"""
    s = _clean_bom(str(raw_entry)).strip()
    has_idx = False
    m_idx = INDEXED_LINE_RE.match(s)
    if m_idx:
        has_idx = True
        got_idx = int(m_idx.group(1))
        if got_idx != expected_idx:
            raise ValueError(
                f"[{stem}] 译文行号锚点错位: 期望第 {expected_idx} 条，实际读到 '{got_idx}|...'（疑似前文漏行或合行）"
            )
        s = m_idx.group(2).strip()
    elif require_index:
        raise ValueError(f"[{stem}] 第 {expected_idx} 条缺少 '{expected_idx}|' 行号锚点前缀: {s!r}")

    m_ref = SHARED_REF_RE.match(s)
    if m_ref:
        sid = m_ref.group(1)
        if sid not in shared_map:
            raise ValueError(f"[{stem}] 第 {expected_idx} 条引用了未定义的共句 ID '@{sid}'")
        s = shared_map[sid]

    # 兼容 "说话人|译文"（如从 export 模板原样保留了第二列说话人）
    if expected_spk and s.startswith(f"{expected_spk}|"):
        s = s[len(expected_spk) + 1 :].strip()
    return s, has_idx


def _normalize_zh_lines(
    stem: str,
    raw_val: Any,
    cues: Sequence[Tuple[str, str, str]],
    shared_map: Dict[str, str],
    require_index: bool = False,
) -> Tuple[List[str], bool]:
    expected_len = len(cues)
    exp_spks = [(strip_srt_spk(c[2])[0][1:-1] if strip_srt_spk(c[2])[0] else "") for c in cues]

    if isinstance(raw_val, list):
        if len(raw_val) != expected_len:
            raise ValueError(f"[{stem}] 译文行数 ({len(raw_val)}) != 日文 SRT 条数 ({expected_len})")
        out: List[str] = []
        any_idx = False
        all_idx = True
        for i, (item, espk) in enumerate(zip(raw_val, exp_spks), 1):
            # 若整轨启用了部分 idx| 锚点，则要求全轨统一带锚点，避免半途漏标掩盖错位
            val, has_i = _strip_indexed_entry(
                stem, i, item, espk, shared_map, require_index=(require_index or (any_idx and all_idx))
            )
            any_idx = any_idx or has_i
            all_idx = all_idx and has_i
            out.append(val)
        return out, all_idx

    if isinstance(raw_val, dict):
        if len(raw_val) != expected_len:
            raise ValueError(f"[{stem}] 译文字典条数 ({len(raw_val)}) != 日文 SRT 条数 ({expected_len})")
        out = []
        for i, espk in enumerate(exp_spks, 1):
            key = str(i)
            if key not in raw_val:
                raise ValueError(f"[{stem}] 缺少第 {i} 条译文键")
            val, _ = _strip_indexed_entry(stem, i, raw_val[key], espk, shared_map, require_index=False)
            out.append(val)
        return out, True

    raise ValueError(f"[{stem}] 译文值必须为 list 或 dict，实际为 {type(raw_val).__name__}")


def lint_cue_pair(stem: str, idx: int, jp_body: str, zh_body: str) -> List[str]:
    """对单条日->中译文正文做非假名语义/规范质量巡检（返回 warning 列表）。"""
    warns: List[str] = []
    m_shin = JP_SHINJITAI_RE.search(zh_body)
    if m_shin:
        warns.append(f"[{stem}] #{idx} 疑似残留日文新字体汉字 '{m_shin.group(0)}': {zh_body}")
    m_jpw = JP_UNTRANSLATED_WORD_RE.search(zh_body)
    if m_jpw:
        warns.append(f"[{stem}] #{idx} 疑似未翻译日文汉字词 '{m_jpw.group(0)}': {zh_body}")
    m_ono = BANNED_ONO_RE.search(zh_body)
    if m_ono:
        warns.append(f"[{stem}] #{idx} 含规范禁用拟声词 '{m_ono.group(0)}': {zh_body}")
    if FULLWIDTH_LATIN_TYPO_RE.search(zh_body):
        warns.append(f"[{stem}] #{idx} 含疑似全角拉丁错字: {zh_body}")

    jp_clean = re.sub(r"[\s…‥ー～〜♡♪！!？?、。，,「」『』（）()──—\-]+", "", jp_body)
    zh_clean = re.sub(r"[\s…‥—～〜♡♪！!？?、。，,「」『』（）()\-]+", "", zh_body)
    # 排除纯拟声长拖音（如 んんんぅぅぅぅ）被误报为长句漏翻
    if len(jp_clean) >= 18 and len(set(jp_clean)) >= 7 and len(zh_clean) <= 2:
        warns.append(f"[{stem}] #{idx} 译文疑似过短漏翻 (日={len(jp_clean)}字 vs 中={len(zh_clean)}字): {jp_body} -> {zh_body}")
    elif 1 <= len(jp_clean) <= 3 and len(zh_clean) >= 18:
        warns.append(f"[{stem}] #{idx} 译文疑似过度扩写或串行 (日={len(jp_clean)}字 vs 中={len(zh_clean)}字): {jp_body} -> {zh_body}")
    return warns


def build_track(
    jp_dir: pathlib.Path,
    cn_dir: pathlib.Path,
    stem: str,
    raw_zh: Any,
    allow_kana: bool = False,
    shared_map: Optional[Dict[str, str]] = None,
    require_index: bool = False,
) -> Tuple[int, int, List[str]]:
    jp_srt = jp_dir / f"{stem}.srt"
    jp_lrc = jp_dir / f"{stem}.lrc"
    if not jp_srt.is_file():
        raise FileNotFoundError(f"找不到对应日文 SRT: {jp_srt}")

    cues = parse_srt(jp_srt)
    zh_lines, had_index = _normalize_zh_lines(
        stem, raw_zh, cues, shared_map=shared_map or {}, require_index=require_index
    )
    warns: List[str] = []
    if not had_index and len(cues) >= 30:
        warns.append(f"[{stem}] 传入了无行号前缀的纯数组（{len(cues)} 条）；推荐使用 '1|译文' 锚点格式防错位")

    out_srt_blocks: List[str] = []
    jp_to_zh_queue: Dict[Tuple[str, str], List[str]] = {}
    speakers_seen: List[str] = []

    for idx, ((num, ts, jp_full), zh_raw) in enumerate(zip(cues, zh_lines), 1):
        jp_pfx, jp_body = strip_srt_spk(jp_full)
        zh_pfx, zh_body = strip_srt_spk(zh_raw)
        if jp_pfx:
            spk_name = jp_pfx[1:-1]
            if spk_name not in speakers_seen:
                speakers_seen.append(spk_name)
            if zh_pfx and zh_pfx != jp_pfx:
                raise ValueError(f"[{stem}] 第 {idx} 条说话人前缀不匹配: 日={jp_pfx} vs 中={zh_pfx}")
            zh_full = jp_pfx + zh_body
        else:
            spk_name = ""
            zh_full = zh_body

        if not zh_body:
            raise ValueError(f"[{stem}] 第 {idx} 条译文为空")
        if "\ufffd" in zh_full:
            raise ValueError(f"[{stem}] 第 {idx} 条包含乱码替换符 U+FFFD: {zh_full}")
        if not allow_kana and KANA_RE.search(zh_body):
            raise ValueError(f"[{stem}] 第 {idx} 条残留日文假名/长音/间隔号: {zh_full}")

        warns.extend(lint_cue_pair(stem, idx, jp_body, zh_body))
        out_srt_blocks.append(f"{num}\n{ts}\n{zh_full}\n")
        jp_to_zh_queue.setdefault((spk_name, jp_body), []).append(zh_body)

    cn_dir.mkdir(parents=True, exist_ok=True)
    (cn_dir / f"{stem}.srt").write_text("\n".join(out_srt_blocks), encoding="utf-8")

    lrc_count = 0
    if jp_lrc.is_file():
        lrc_lines = _clean_bom(jp_lrc.read_text(encoding="utf-8-sig")).splitlines()
        spk_pat = (
            re.compile("(" + "|".join(re.escape(s) for s in sorted(speakers_seen, key=len, reverse=True)) + ")：")
            if speakers_seen
            else None
        )
        out_lrc: List[str] = []
        unmatched: List[Tuple[int, str]] = []
        for li, line in enumerate(lrc_lines, 1):
            m_ts = LRC_TS_RE.match(line)
            if not m_ts:
                out_lrc.append(line)
                continue
            ts, content = m_ts.group(1), m_ts.group(2)
            if not content.strip():
                out_lrc.append(ts)
                continue
            matches = list(spk_pat.finditer(content)) if spk_pat is not None else []
            if not matches:
                q = jp_to_zh_queue.get(("", content.strip()))
                if q:
                    out_lrc.append(f"{ts}{q.pop(0)}")
                else:
                    norm_t = re.sub(r"\s+", "", content.strip())
                    found: Optional[str] = None
                    for (ks, kb), qv in jp_to_zh_queue.items():
                        if ks == "" and qv and re.sub(r"\s+", "", kb) == norm_t:
                            found = qv.pop(0)
                            break
                    if found is not None:
                        out_lrc.append(f"{ts}{found}")
                    else:
                        unmatched.append((li, content))
                        out_lrc.append(line)
                continue

            parts_out: List[str] = []
            for mi, mm in enumerate(matches):
                spk = mm.group(1)
                seg_st = mm.end()
                seg_en = matches[mi + 1].start() if mi + 1 < len(matches) else len(content)
                jp_seg = content[seg_st:seg_en].strip()
                q = jp_to_zh_queue.get((spk, jp_seg))
                if q:
                    zh_seg = q.pop(0)
                else:
                    norm_t = re.sub(r"\s+", "", jp_seg)
                    found = None
                    for (ks, kb), qv in jp_to_zh_queue.items():
                        if ks == spk and qv and re.sub(r"\s+", "", kb) == norm_t:
                            found = qv.pop(0)
                            break
                    if found is not None:
                        zh_seg = found
                    else:
                        unmatched.append((li, f"{spk}：{jp_seg}"))
                        zh_seg = jp_seg
                parts_out.append(f"{spk}：{zh_seg}")
            out_lrc.append(ts + ("　".join(parts_out) if len(parts_out) > 1 else parts_out[0]))

        if unmatched:
            raise ValueError(f"[{stem}] LRC 有 {len(unmatched)} 段未匹配: {unmatched[:3]}")
        (cn_dir / f"{stem}.lrc").write_text("\n".join(out_lrc) + "\n", encoding="utf-8")
        lrc_count = len(out_lrc)

    return len(out_srt_blocks), lrc_count, warns


def cmd_build(args: argparse.Namespace) -> int:
    jp_dir = pathlib.Path(args.jp_dir)
    cn_dir = pathlib.Path(args.cn_dir)
    payload = json.loads(_clean_bom(pathlib.Path(args.in_json).read_text(encoding="utf-8-sig")))
    if not isinstance(payload, dict):
        print("[ERROR] 输入 JSON 顶层必须为 {stem: [译文...]} 或 {shared: ..., tracks: {...}} 对象", file=sys.stderr)
        return 2

    shared_map: Dict[str, str] = {}
    if getattr(args, "shared_json", None):
        ext_sh = json.loads(_clean_bom(pathlib.Path(args.shared_json).read_text(encoding="utf-8-sig")))
        if isinstance(ext_sh, dict) and "shared" in ext_sh:
            ext_sh = ext_sh["shared"]
        shared_map.update(_parse_shared_translations(ext_sh))

    if "tracks" in payload and isinstance(payload["tracks"], dict):
        if "shared" in payload:
            shared_map.update(_parse_shared_translations(payload["shared"]))
        tracks_dict = payload["tracks"]
    else:
        tracks_dict = {k: v for k, v in payload.items() if k not in ("format", "stats", "shared")}

    all_warns: List[str] = []
    for stem, lines in tracks_dict.items():
        n_srt, n_lrc, warns = build_track(
            jp_dir,
            cn_dir,
            stem,
            lines,
            allow_kana=bool(args.allow_kana),
            shared_map=shared_map,
            require_index=bool(args.require_index),
        )
        all_warns.extend(warns)
        w_sfx = f" [警告={len(warns)}]" if warns else ""
        print(f"[OK] {stem} -> SRT({n_srt} cues), LRC({n_lrc} lines){w_sfx}")

    for w in all_warns[:20]:
        print(f"  [WARN] {w}", file=sys.stderr)
    if len(all_warns) > 20:
        print(f"  [WARN] ... 其余 {len(all_warns) - 20} 条警告已省略", file=sys.stderr)
    if args.strict and all_warns:
        return 1
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    jp_dir = pathlib.Path(args.jp_dir)
    cn_dir = pathlib.Path(args.cn_dir)
    jp_srts = _filter_srts_by_only(sorted(jp_dir.glob("*.srt")), getattr(args, "only", "") or "")
    if not jp_srts:
        print(f"[ERROR] 未找到日文 SRT: {jp_dir}", file=sys.stderr)
        return 2

    errors: List[str] = []
    warnings: List[str] = []
    total_cues = 0
    # 跨视角同句一致性追踪: {(track_group, spk, jp_body): {zh_body: stem}}
    pov_memory: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    spot_samples: List[Tuple[str, List[Tuple[int, str, str]]]] = []

    for jp_s in jp_srts:
        stem = jp_s.stem
        cn_s = cn_dir / f"{stem}.srt"
        if not cn_s.is_file():
            errors.append(f"缺少中文 SRT: {cn_s.name}")
            continue
        raw_cn_bytes = cn_s.read_bytes()
        # 检查非文件头的内部残留 U+FEFF BOM（常见于多段文件直接拼接）
        cn_text_no_head_bom = raw_cn_bytes.decode("utf-8-sig", errors="replace")
        if "\ufeff" in cn_text_no_head_bom:
            errors.append(f"[{stem}] SRT 正文内部残留 BOM (U+FEFF)")

        js = parse_srt(jp_s)
        cs = parse_srt(cn_s)
        total_cues += len(js)
        grp = _track_group_key(stem)

        if len(js) != len(cs):
            errors.append(f"[{stem}] SRT 条数不一致: 日={len(js)} 中={len(cs)}")
        else:
            track_pairs: List[Tuple[int, str, str]] = []
            for i, (a, b) in enumerate(zip(js, cs), 1):
                if a[0] != str(i) or b[0] != str(i) or a[1] != b[1]:
                    errors.append(f"[{stem}] 第 {i} 条序号或时间戳不一致: 日=({a[0]},{a[1]}) 中=({b[0]},{b[1]})")
                p1, jb = strip_srt_spk(a[2])
                p2, zb = strip_srt_spk(b[2])
                if p1 != p2:
                    errors.append(f"[{stem}] 第 {i} 条说话人前缀不一致: 日={p1} 中={p2}")
                if not zb:
                    errors.append(f"[{stem}] 第 {i} 条译文正文为空")
                if "\ufffd" in b[2]:
                    errors.append(f"[{stem}] 第 {i} 条含乱码字符 U+FFFD: {b[2]}")
                if not args.allow_kana and KANA_RE.search(zb):
                    errors.append(f"[{stem}] 第 {i} 条残留假名: {b[2]}")

                warnings.extend(lint_cue_pair(stem, i, jb, zb))
                track_pairs.append((i, a[2], b[2]))
                if grp and len(jb) >= 4:
                    spk_k = p1[1:-1] if p1 else ""
                    seen_z = pov_memory.setdefault((grp, spk_k, jb), {})
                    seen_z.setdefault(zb, stem)

            if args.spot_check > 0 and track_pairs:
                k = min(int(args.spot_check), len(track_pairs))
                idxs = sorted({round(j * (len(track_pairs) - 1) / max(1, k - 1)) for j in range(k)})
                spot_samples.append((stem, [track_pairs[ix] for ix in idxs]))

        jp_l = jp_dir / f"{stem}.lrc"
        cn_l = cn_dir / f"{stem}.lrc"
        if jp_l.is_file():
            if not cn_l.is_file():
                errors.append(f"缺少中文 LRC: {cn_l.name}")
            else:
                cn_l_raw = cn_l.read_text(encoding="utf-8-sig")
                if "\ufeff" in cn_l_raw:
                    errors.append(f"[{stem}] LRC 正文内部残留 BOM (U+FEFF)")
                jl = _clean_bom(jp_l.read_text(encoding="utf-8-sig")).splitlines()
                cl = _clean_bom(cn_l_raw).splitlines()
                if len(jl) != len(cl):
                    errors.append(f"[{stem}] LRC 行数不一致: 日={len(jl)} 中={len(cl)}")
                else:
                    # 剥离 LRC 行内说话人名字后再查假名，避免角色名自带假名误报
                    spk_names = sorted(
                        {strip_srt_spk(c[2])[0][1:-1] for c in js if strip_srt_spk(c[2])[0]},
                        key=len,
                        reverse=True,
                    )
                    lrc_spk_re = (
                        re.compile("(" + "|".join(re.escape(s) for s in spk_names) + ")：")
                        if spk_names
                        else None
                    )
                    for li, (la, lb) in enumerate(zip(jl, cl), 1):
                        ma, mb = LRC_TS_RE.match(la), LRC_TS_RE.match(lb)
                        ta, ca = (ma.group(1), ma.group(2)) if ma else ("", la)
                        tb, cb = (mb.group(1), mb.group(2)) if mb else ("", lb)
                        if ta != tb or bool(ca.strip()) != bool(cb.strip()):
                            errors.append(f"[{stem}] LRC 第 {li} 行时间戳或空行不一致")
                        cb_body = lrc_spk_re.sub("", cb) if lrc_spk_re else cb
                        if not args.allow_kana and KANA_RE.search(cb_body):
                            errors.append(f"[{stem}] LRC 第 {li} 行残留假名: {cb}")

    for (grp, spk_k, jb), zh_variants in pov_memory.items():
        if len(zh_variants) > 1:
            desc = " vs ".join(f"{z!r}({st})" for z, st in list(zh_variants.items())[:3])
            warnings.append(f"[{grp} 跨视角同句异译] {spk_k + '：' if spk_k else ''}{jb!r} -> {desc}")

    print(
        f"校验完成: {len(jp_srts)} 轨 / 共 {total_cues} 条字幕 | 异常={len(errors)} | 质量警告={len(warnings)}"
    )
    for e in errors[:20]:
        print(f"  [ERR] {e}", file=sys.stderr)
    for w in warnings[:25]:
        print(f"  [WARN] {w}", file=sys.stderr)
    if len(warnings) > 25:
        print(f"  [WARN] ... 其余 {len(warnings) - 25} 条质量警告已省略", file=sys.stderr)

    if spot_samples:
        print("\n=== 双语对照抽样巡检 ===")
        for stem, rows in spot_samples:
            print(f"[{stem}]")
            for idx, j_full, z_full in rows:
                print(f"  #{idx:03d} JP: {j_full}")
                print(f"       CN: {z_full}")

    if errors or (args.strict and warnings):
        return 1
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    _configure_stdio()
    p = argparse.ArgumentParser(description="FA-ASMR 原地修轴 / 紧凑翻译导出 / 防错回填 / 深度校验工具")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("patch", help="在定稿 .srt + .lrc 上原地同步执行删句/插句/改时/改文（免全局重跑）")
    pp.add_argument("--sub-dir", help="批量修轴：字幕所在目录（配合 --ops-json）")
    pp.add_argument("--ops-json", help="批量修轴操作 JSON（格式: {stem: [{op: delete|update|replace|insert, ...}]}）")
    pp.add_argument("--srt", help="单轨快速修轴：目标 .srt 文件路径（自动同步同目录同名 .lrc）")
    pp.add_argument("--cue", type=int, help="单轨快速修轴：1-based SRT 条目序号")
    pp.add_argument("--delete", help="单轨快速删除：逗号分隔的 1-based 条目序号（如 114,115）")
    pp.add_argument("--replace", help="单轨文本替换：格式 '旧子串=新子串'（如 'ｘ='）")
    pp.add_argument("--start", help="修改或插入条目的起始时间（支持 HH:MM:SS,mmm 或浮点秒数）")
    pp.add_argument("--end", help="修改或插入条目的结束时间（支持 HH:MM:SS,mmm 或浮点秒数）")
    pp.add_argument("--speaker", help="修改或插入条目的说话人名称")
    pp.add_argument("--text", help="修改目标条目的正文")
    pp.add_argument("--insert-text", help="插入新条目的正文（需配合 --start 与 --end）")
    pp.set_defaults(func=cmd_patch)

    pe = sub.add_parser("export", help="将日文 .srt 目录导出为紧凑待翻文本")
    pe.add_argument("--jp-dir", required=True)
    pe.add_argument("--out", required=True)
    pe.add_argument("--only", default="", help="可选：逗号分隔轨号/关键词筛选")
    pe.add_argument(
        "--dedup-pov",
        action="store_true",
        help="自动提取跨视角/跨轨重复台词为 shared 共句表（S0001...），消除双视角重复翻译并保证同句同译",
    )
    pe.set_defaults(func=cmd_export)

    pb = sub.add_parser("build", help="将中文译文 JSON 回填生成严格对齐的 .srt 和 .lrc")
    pb.add_argument("--jp-dir", required=True)
    pb.add_argument("--cn-dir", required=True)
    pb.add_argument("--in-json", required=True)
    pb.add_argument("--shared-json", help="可选：外部已翻好的跨轨共句 JSON（配合 --dedup-pov 分批回填）")
    pb.add_argument("--require-index", action="store_true", help="强制要求列表每行携带 'idx|' 序号锚点，彻底杜绝抵消型串行")
    pb.add_argument("--allow-kana", action="store_true")
    pb.add_argument("--strict", action="store_true", help="将疑似日文汉字残留/禁用词等质量警告也视为失败返回码")
    pb.set_defaults(func=cmd_build)

    pv = sub.add_parser("verify", help="校验日文与中文字幕目录的 1:1 一致性与译文质量")
    pv.add_argument("--jp-dir", required=True)
    pv.add_argument("--cn-dir", required=True)
    pv.add_argument("--only", default="", help="可选：逗号分隔轨号/关键词筛选")
    pv.add_argument("--allow-kana", action="store_true")
    pv.add_argument("--strict", action="store_true", help="存在质量警告时返回非零退出码")
    pv.add_argument("--spot-check", type=int, default=0, metavar="N", help="每轨均匀抽样打印 N 条中日双语对照供人工/Agent 终审")
    pv.set_defaults(func=cmd_verify)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
