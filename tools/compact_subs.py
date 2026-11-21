# -*- coding: utf-8 -*-
"""FA-ASMR 紧凑翻译辅助脚本（导出精简待翻文本 + 一键回填双格式 .srt/.lrc + 严格一致性校验）。

用法示例：
  # 1. 将日文定稿目录导出为紧凑待翻文本包（剥离时间戳与空行，省 ~35%-40% 翻译输入 Token）
  python compact_subs.py export --jp-dir "E:\\work\\字幕（日文）" --out "compact_jp.json"

  # 2. 将翻译好的 JSON 回填生成中文 .srt + .lrc（自动校验行数、时间戳、说话人前缀与 0 假名残留）
  python compact_subs.py build --jp-dir "E:\\work\\字幕（日文）" --cn-dir "E:\\work\\字幕（中文）" --in-json "zh_batch.json"

  # 3. 批量终检日文目录与中文目录的 1:1 时间轴一致性
  python compact_subs.py verify --jp-dir "E:\\work\\字幕（日文）" --cn-dir "E:\\work\\字幕（中文）"
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
from typing import Dict, List, Optional, Tuple

KANA_RE = re.compile(r"[\u3040-\u309F\u30A0-\u30FF]")
SRT_SPK_RE = re.compile(r"^(【[^】]+】)(.*)$")
LRC_TS_RE = re.compile(r"^(\[[^\]]+\])(.*)$")


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def parse_srt(path: pathlib.Path) -> List[Tuple[str, str, str]]:
    text = path.read_text(encoding="utf-8-sig").strip()
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
    m = SRT_SPK_RE.match(text.strip())
    if m:
        return m.group(1), m.group(2).strip()
    return "", text.strip()


def cmd_export(args: argparse.Namespace) -> int:
    jp_dir = pathlib.Path(args.jp_dir)
    if not jp_dir.is_dir():
        print(f"[ERROR] 目录不存在: {jp_dir}", file=sys.stderr)
        return 2

    only_tokens = [x.strip().lower() for x in re.split(r"[,，;；]+", args.only or "") if x.strip()]
    srts = sorted(jp_dir.glob("*.srt"))
    if only_tokens:
        filtered = []
        for p in srts:
            stem_l = p.stem.lower()
            m_num = re.match(r"^0*(\d+)(?=[_\-\s.「【]|$)", stem_l)
            s_num = int(m_num.group(1)) if m_num else None
            if any(
                (tok.isdigit() and s_num is not None and s_num == int(tok)) or (not tok.isdigit() and tok in stem_l)
                for tok in only_tokens
            ):
                filtered.append(p)
        srts = filtered

    bundle: Dict[str, List[str]] = {}
    total_cues = 0
    for p in srts:
        cues = parse_srt(p)
        lines: List[str] = []
        for idx, (_num, _ts, body) in enumerate(cues, 1):
            pfx, txt = strip_srt_spk(body)
            spk = pfx[1:-1] if pfx else ""
            lines.append(f"{idx}|{spk}|{txt}" if spk else f"{idx}|{txt}")
        bundle[p.stem] = lines
        total_cues += len(lines)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".txt":
        chunks = []
        for stem, lines in bundle.items():
            chunks.append(f"=== {stem} ({len(lines)}) ===")
            chunks.extend(lines)
            chunks.append("")
        out_path.write_text("\n".join(chunks), encoding="utf-8")
    else:
        out_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] 已导出 {len(bundle)} 轨紧凑文本（共 {total_cues} 条） -> {out_path}")
    return 0


def _normalize_zh_lines(stem: str, raw_val, expected_len: int) -> List[str]:
    if isinstance(raw_val, list):
        if len(raw_val) != expected_len:
            raise ValueError(f"[{stem}] 译文行数 ({len(raw_val)}) != 日文 SRT 条数 ({expected_len})")
        return [str(x).strip() for x in raw_val]
    if isinstance(raw_val, dict):
        if len(raw_val) != expected_len:
            raise ValueError(f"[{stem}] 译文字典条数 ({len(raw_val)}) != 日文 SRT 条数 ({expected_len})")
        out: List[str] = []
        for i in range(1, expected_len + 1):
            key = str(i)
            if key not in raw_val:
                raise ValueError(f"[{stem}] 缺少第 {i} 条译文键")
            out.append(str(raw_val[key]).strip())
        return out
    raise ValueError(f"[{stem}] 译文值必须为 list 或 dict，实际为 {type(raw_val).__name__}")


def build_track(
    jp_dir: pathlib.Path,
    cn_dir: pathlib.Path,
    stem: str,
    raw_zh,
    allow_kana: bool = False,
) -> Tuple[int, int]:
    jp_srt = jp_dir / f"{stem}.srt"
    jp_lrc = jp_dir / f"{stem}.lrc"
    if not jp_srt.is_file():
        raise FileNotFoundError(f"找不到对应日文 SRT: {jp_srt}")

    cues = parse_srt(jp_srt)
    zh_lines = _normalize_zh_lines(stem, raw_zh, len(cues))

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
            zh_full = zh_raw.strip()
            zh_body = zh_full

        if not zh_body:
            raise ValueError(f"[{stem}] 第 {idx} 条译文为空")
        if not allow_kana and KANA_RE.search(zh_full):
            raise ValueError(f"[{stem}] 第 {idx} 条残留日文假名: {zh_full}")

        out_srt_blocks.append(f"{num}\n{ts}\n{zh_full}\n")
        jp_to_zh_queue.setdefault((spk_name, jp_body), []).append(zh_body)

    cn_dir.mkdir(parents=True, exist_ok=True)
    (cn_dir / f"{stem}.srt").write_text("\n".join(out_srt_blocks), encoding="utf-8")

    lrc_count = 0
    if jp_lrc.is_file():
        lrc_lines = jp_lrc.read_text(encoding="utf-8-sig").splitlines()
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

    return len(out_srt_blocks), lrc_count


def cmd_build(args: argparse.Namespace) -> int:
    jp_dir = pathlib.Path(args.jp_dir)
    cn_dir = pathlib.Path(args.cn_dir)
    payload = json.loads(pathlib.Path(args.in_json).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        print("[ERROR] 输入 JSON 顶层必须为 {stem: [译文...]} 对象", file=sys.stderr)
        return 2
    for stem, lines in payload.items():
        n_srt, n_lrc = build_track(jp_dir, cn_dir, stem, lines, allow_kana=bool(args.allow_kana))
        print(f"[OK] {stem} -> SRT({n_srt} cues), LRC({n_lrc} lines)")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    jp_dir = pathlib.Path(args.jp_dir)
    cn_dir = pathlib.Path(args.cn_dir)
    jp_srts = sorted(jp_dir.glob("*.srt"))
    if not jp_srts:
        print(f"[ERROR] 未找到日文 SRT: {jp_dir}", file=sys.stderr)
        return 2

    errors: List[str] = []
    total_cues = 0
    for jp_s in jp_srts:
        stem = jp_s.stem
        cn_s = cn_dir / f"{stem}.srt"
        if not cn_s.is_file():
            errors.append(f"缺少中文 SRT: {cn_s.name}")
            continue
        js = parse_srt(jp_s)
        cs = parse_srt(cn_s)
        total_cues += len(js)
        if len(js) != len(cs):
            errors.append(f"[{stem}] SRT 条数不一致: 日={len(js)} 中={len(cs)}")
        else:
            for i, (a, b) in enumerate(zip(js, cs), 1):
                if a[0] != str(i) or b[0] != str(i) or a[1] != b[1]:
                    errors.append(f"[{stem}] 第 {i} 条序号或时间戳不一致: 日=({a[0]},{a[1]}) 中=({b[0]},{b[1]})")
                p1, _ = strip_srt_spk(a[2])
                p2, _ = strip_srt_spk(b[2])
                if p1 != p2:
                    errors.append(f"[{stem}] 第 {i} 条说话人前缀不一致: 日={p1} 中={p2}")
                if not args.allow_kana and KANA_RE.search(b[2]):
                    errors.append(f"[{stem}] 第 {i} 条残留假名: {b[2]}")

        jp_l = jp_dir / f"{stem}.lrc"
        cn_l = cn_dir / f"{stem}.lrc"
        if jp_l.is_file():
            if not cn_l.is_file():
                errors.append(f"缺少中文 LRC: {cn_l.name}")
            else:
                jl = jp_l.read_text(encoding="utf-8-sig").splitlines()
                cl = cn_l.read_text(encoding="utf-8-sig").splitlines()
                if len(jl) != len(cl):
                    errors.append(f"[{stem}] LRC 行数不一致: 日={len(jl)} 中={len(cl)}")
                else:
                    for li, (la, lb) in enumerate(zip(jl, cl), 1):
                        ma, mb = LRC_TS_RE.match(la), LRC_TS_RE.match(lb)
                        ta, ca = (ma.group(1), ma.group(2)) if ma else ("", la)
                        tb, cb = (mb.group(1), mb.group(2)) if mb else ("", lb)
                        if ta != tb or bool(ca.strip()) != bool(cb.strip()):
                            errors.append(f"[{stem}] LRC 第 {li} 行时间戳或空行不一致")
                        if not args.allow_kana and KANA_RE.search(cb):
                            errors.append(f"[{stem}] LRC 第 {li} 行残留假名: {cb}")

    print(f"校验完成: {len(jp_srts)} 轨 / 共 {total_cues} 条字幕 | 异常={len(errors)}")
    for e in errors[:20]:
        print(f"  - {e}", file=sys.stderr)
    return 0 if not errors else 1


def main(argv: Optional[List[str]] = None) -> int:
    _configure_stdio()
    p = argparse.ArgumentParser(description="FA-ASMR 紧凑翻译导出 / 回填 / 校验工具")
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("export", help="将日文 .srt 目录导出为紧凑待翻文本")
    pe.add_argument("--jp-dir", required=True)
    pe.add_argument("--out", required=True)
    pe.add_argument("--only", default="", help="可选：逗号分隔轨号/关键词筛选")
    pe.set_defaults(func=cmd_export)

    pb = sub.add_parser("build", help="将中文译文 JSON 回填生成严格对齐的 .srt 和 .lrc")
    pb.add_argument("--jp-dir", required=True)
    pb.add_argument("--cn-dir", required=True)
    pb.add_argument("--in-json", required=True)
    pb.add_argument("--allow-kana", action="store_true")
    pb.set_defaults(func=cmd_build)

    pv = sub.add_parser("verify", help="校验日文与中文字幕目录的 1:1 一致性")
    pv.add_argument("--jp-dir", required=True)
    pv.add_argument("--cn-dir", required=True)
    pv.add_argument("--allow-kana", action="store_true")
    pv.set_defaults(func=cmd_verify)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
