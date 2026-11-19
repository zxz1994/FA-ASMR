# -*- coding: utf-8 -*-
"""FA-ASMR V2 — 无界面命令行入口（面向批量脚本与外部 AI Agent 调用）。

用法示例：
  # 1. 深度校验单个清洗后的台本（注音 / 说话人 / 顿号桥接 / 子句数 / 推荐模式）
  python fa_asmr_cli.py --verify-script 01.txt --json

  # 2. 单条自动判模式对齐（有 --audio-b 走 multi_persp，>=2 角色走 multi_speaker，否则 single）
  python fa_asmr_cli.py --audio 01.wav --script 01.txt --json

  # 3. 双视角联合对齐
  python fa_asmr_cli.py --audio 01_A.wav --audio-b 01_B.wav --script 01.txt --json

  # 4. 整目录自动配对预检（不加载模型，秒级检查配对、编码、角色与推荐模式）
  python fa_asmr_cli.py --batch ./作品目录 --dry-run --json

  # 5. 整目录批量执行并输出机器可读 JSON 摘要
  python fa_asmr_cli.py --batch ./作品目录 --json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence

if getattr(sys, "frozen", False):
    _SCRIPT_DIR = sys._MEIPASS
else:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from fa_asmr_pipeline import (
    AUDIO_EXTS,
    _extract_perspective_tag,
    _iter_speaker_marks,
    _release_gpu,
    _split_speaker_segments,
    auto_pair_files,
    read_text_auto_encoding,
    run_pipeline,
)

_TITLE_SPEAKER_RE = re.compile(r"^\s*【([^【】\r\n]{1,20})】\s*$")
_DEFAULT_VRAM_CHUNK = 3000


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _collect_batch_files(inputs: Sequence[str], recursive: bool = False) -> List[str]:
    files: List[str] = []
    seen = set()
    for raw in inputs:
        path = os.path.abspath(raw)
        if os.path.isdir(path):
            if recursive:
                for root, _dirs, names in os.walk(path):
                    for name in sorted(names):
                        full = os.path.join(root, name)
                        ext = os.path.splitext(name)[1].lower()
                        if (ext == ".txt" or ext in AUDIO_EXTS) and full not in seen:
                            seen.add(full)
                            files.append(full)
            else:
                for name in sorted(os.listdir(path)):
                    full = os.path.join(path, name)
                    if not os.path.isfile(full):
                        continue
                    ext = os.path.splitext(name)[1].lower()
                    if (ext == ".txt" or ext in AUDIO_EXTS) and full not in seen:
                        seen.add(full)
                        files.append(full)
        elif os.path.isfile(path):
            if path not in seen:
                seen.add(path)
                files.append(path)
    return files


def _inspect_script_light(script_path: str) -> Dict[str, Any]:
    """轻量读取台本元信息（编码、行数、说话人列表与推荐模式），不加载 PyTorch/注音模型。"""
    text, enc = read_text_auto_encoding(script_path)
    raw_lines = text.splitlines()
    non_empty = [ln for ln in raw_lines if ln.strip()]

    speakers: List[str] = []
    seen_spk = set()
    cur_spk = ()
    for ln in non_empty:
        segs, cur_spk = _split_speaker_segments(ln, cur_spk)
        for spk, _ in segs:
            for name in spk:
                if name and name not in seen_spk:
                    seen_spk.add(name)
                    speakers.append(name)

    if not speakers:
        # 兼容独立 【角色名】 标题行（按与 align_multispeaker 相同的常见分隔拆角色）
        header_counts: Dict[str, int] = {}
        header_order: List[str] = []
        for ln in non_empty:
            m = _TITLE_SPEAKER_RE.match(ln)
            if not m:
                continue
            inside = m.group(1).strip()
            if any(ch in inside for ch in "。！？!?「」『』（）()"):
                continue
            for part in re.split(r"[＆&、，,＋+×/／]", inside):
                nm = part.strip()
                if 1 <= len(nm) <= 12:
                    header_counts[nm] = header_counts.get(nm, 0) + 1
                    if nm not in header_order:
                        header_order.append(nm)
        recurring = [nm for nm in header_order if header_counts.get(nm, 0) >= 2]
        if len(recurring) >= 2:
            speakers = recurring
        else:
            for ln in non_empty:
                for _a, _b, name in _iter_speaker_marks(ln):
                    if name and name not in seen_spk:
                        seen_spk.add(name)
                        speakers.append(name)

    recommended = "multi_speaker" if len(speakers) >= 2 else "single"
    return {
        "encoding": enc,
        "total_lines": len(raw_lines),
        "non_empty_lines": len(non_empty),
        "detected_speakers": speakers,
        "recommended_mode": recommended,
    }


def _detect_task_mode(requested_mode: str, script_path: str, audio_b: Optional[str]) -> str:
    """根据用户指定模式或台本/音频特征确定实际执行模式。"""
    if audio_b:
        return "multi_persp"
    if requested_mode != "auto":
        return requested_mode
    info = _inspect_script_light(script_path)
    return info["recommended_mode"]


def _looks_like_multi_persp_batch(file_paths: Sequence[str]) -> bool:
    """判断批量目录中是否存在共享同一基名的双视角音频对（如 _A/_B 或 主人公視点/夫視点）。"""
    sides_by_base: Dict[str, set] = {}
    for p in file_paths:
        if os.path.splitext(p)[1].lower() not in AUDIO_EXTS:
            continue
        stem = os.path.splitext(os.path.basename(p))[0]
        base, side, _ = _extract_perspective_tag(stem)
        if side:
            sides_by_base.setdefault(base, set()).add(side)
    return any(len(sides) >= 2 for sides in sides_by_base.values())


def _run_verify_script(script_path: str, as_json: bool, json_out: Optional[str]) -> int:
    """深度校验单个台本：完整运行自动编码嗅探、角色标题归一化、AutoRuby 注音、顿号桥接与子句切分。"""
    abs_path = os.path.abspath(script_path)
    if not os.path.isfile(abs_path):
        err = f"台本文件不存在: {abs_path}"
        payload = {"ok": False, "error": err}
        if as_json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(f"[ERROR] {err}", file=sys.stderr)
        if json_out:
            with open(json_out, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        return 2

    from fa_asmr_pipeline import AutoRuby, _ensure_runtime_ready
    _ensure_runtime_ready()
    import clause
    import haruraw2norm as hn
    from align_multispeaker import auto_normalize_speaker_headers

    raw_text, detected_enc = read_text_auto_encoding(abs_path)
    lines = raw_text.splitlines()
    lines, _auto_spks = auto_normalize_speaker_headers(lines)

    body_lines: List[str] = []
    speakers: List[str] = []
    cur_spk = ()
    for raw in lines:
        segs, cur_spk = _split_speaker_segments(raw, cur_spk)
        for spk, body in segs:
            for s in spk:
                if s and s not in speakers:
                    speakers.append(s)
            if body.strip():
                body_lines.append(body)

    ruby = AutoRuby()
    try:
        annotated_lines = [ruby.annotate(b) for b in body_lines]
    finally:
        ruby.release_gpu()

    result_list: List[Dict[str, Any]] = []
    for line in annotated_lines:
        if line.strip():
            toks = hn.process_haruhi_line(line, "auto", 0, 1)
            result_list.extend(toks)
            result_list.append({"orig": "\n", "type": 0, "pron": ""})

    bridged_count = clause.mark_short_comma_bridges(result_list)
    clauses = clause.split_clauses(result_list)
    total_tokens = sum(len(c) for c in clauses)
    recommended_mode = "multi_speaker" if len(speakers) >= 2 else "single"

    payload = {
        "ok": True,
        "verify_script": True,
        "script": abs_path,
        "encoding": detected_enc,
        "dialogue_lines": len(body_lines),
        "detected_speakers": speakers,
        "speaker_count": len(speakers),
        "short_comma_bridges": bridged_count,
        "clauses": len(clauses),
        "align_tokens": total_tokens,
        "recommended_mode": recommended_mode,
    }

    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        spks_disp = ", ".join(speakers) if speakers else "(单人 / 无标记)"
        print("=== FA-ASMR 台本深度校验 ===")
        print(f"文件        : {abs_path}")
        print(f"识别编码    : {detected_enc}")
        print(f"有效台词行  : {len(body_lines)}")
        print(f"识别角色    : {spks_disp} (共 {len(speakers)} 位)")
        print(f"短句顿号桥接: {bridged_count}")
        print(f"切分子句数  : {len(clauses)}")
        print(f"对齐音素项  : {total_tokens}")
        print(f"推荐模式    : {recommended_mode}（若有双视角 A/B 音频则自动走 multi_persp）")

    if json_out:
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fa_asmr_cli.py",
        description="FA-ASMR V2 无界面声学强制对齐命令行工具（内置自动编码识别与模式推断）",
    )
    p.add_argument(
        "--verify-script",
        metavar="TXT",
        help="深度校验单个清洗后台本（实际运行注音、角色识别、顿号桥接与子句统计，不执行音频对齐）",
    )
    p.add_argument("--audio", help="单条模式：主音频路径（多视角模式下为视角 A 音频）")
    p.add_argument("--audio-b", dest="audio_b", help="多视角模式：视角 B 音频路径")
    p.add_argument("--script", help="单条模式：台本文本路径（自动识别 UTF-8 / Shift-JIS / GB18030 / UTF-16）")
    p.add_argument(
        "--batch",
        nargs="+",
        help="批量模式：传入一个或多个目录/文件路径，自动按文件名配对台本与音频",
    )
    p.add_argument("--recursive", action="store_true", help="批量模式下递归扫描子目录")
    p.add_argument(
        "--mode",
        choices=("auto", "single", "multi_speaker", "multi_persp"),
        default="auto",
        help="对齐模式：auto（默认，自动判定）、single、multi_speaker、multi_persp",
    )
    p.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="推理设备（默认 auto）",
    )
    p.add_argument(
        "--lora",
        default=None,
        help="LoRA 权重路径；留空自动使用 lora/fa_asmr_v9e10.pt；传 NO_LORA 禁用 LoRA",
    )
    # 底层固定最优 3000 帧分块（v2.2.1 起从常规界面移除），此处仅保留隐藏兼容参数
    p.add_argument("--vram-chunk", type=int, default=_DEFAULT_VRAM_CHUNK, help=argparse.SUPPRESS)
    p.add_argument(
        "--line-end-extend",
        "--extend",
        dest="line_end_extend",
        type=float,
        default=0.0,
        help="字幕行尾自然延长秒数（默认 0.0，常用 0.5~1.5）",
    )
    p.add_argument("--no-silence-snap", action="store_true", help="关闭静音吸附")
    p.add_argument("--no-resegment", action="store_true", help="关闭标点驱动重分段")
    p.add_argument("--no-ono-gate", action="store_true", help="关闭拟声词守门")
    p.add_argument("--no-merge-short", action="store_true", help="关闭短句智能合并")
    p.add_argument("--output-dir", help="自定义字幕输出目录（默认输出到各音频同级目录）")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="仅做文件配对、编码嗅探与角色/模式预检，不加载模型、不执行对齐",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="将结构化 JSON 摘要输出到 stdout（运行日志自动转至 stderr）",
    )
    p.add_argument(
        "--json-out",
        help="将结构化 JSON 摘要写入指定文件（适合冻结版 exe 无控制台场景）",
    )
    p.add_argument("--quiet", action="store_true", help="静默模式，减少过程日志输出")
    return p


def _resolve_tasks(args: argparse.Namespace) -> tuple[str, List[Dict[str, Any]], List[Dict[str, Any]], Optional[str]]:
    requested_mode = args.mode
    if args.audio_b and requested_mode != "multi_persp":
        requested_mode = "multi_persp"

    has_single = bool(args.audio or args.script or args.audio_b)
    has_batch = bool(args.batch)
    if has_single and has_batch:
        return requested_mode, [], [], "不能同时指定 --batch 与 --audio/--script"
    if not has_single and not has_batch:
        return requested_mode, [], [], "请指定 --verify-script <TXT>、--audio/--script 或 --batch <目录/文件>"

    if has_single:
        if not args.audio or not args.script:
            return requested_mode, [], [], "单条模式必须同时提供 --audio 和 --script"
        audio = os.path.abspath(args.audio)
        script = os.path.abspath(args.script)
        audio_b = os.path.abspath(args.audio_b) if args.audio_b else None
        for label, p in (("audio", audio), ("script", script)):
            if not os.path.isfile(p):
                return requested_mode, [], [], f"文件不存在 ({label}): {p}"
        if audio_b and not os.path.isfile(audio_b):
            return requested_mode, [], [], f"文件不存在 (audio_b): {audio_b}"
        if requested_mode == "multi_persp" and not audio_b:
            return requested_mode, [], [], "multi_persp 模式需要同时提供 --audio 与 --audio-b"
        resolved_mode = _detect_task_mode(requested_mode, script, audio_b)
        task = {
            "name": os.path.splitext(os.path.basename(audio))[0],
            "mode": resolved_mode,
            "script": script,
            "audio": audio,
            "audio_b": audio_b,
            "confidence": 100,
            "speaker_marks": [],
        }
        return resolved_mode, [task], [], None

    # batch 模式
    missing = [p for p in args.batch if not os.path.exists(p)]
    if missing:
        return requested_mode, [], [], f"批量路径不存在: {', '.join(missing)}"
    collected = _collect_batch_files(args.batch, recursive=args.recursive)
    pair_mode = requested_mode
    if requested_mode == "auto":
        pair_mode = "multi_persp" if _looks_like_multi_persp_batch(collected) else "single"
    pairs, unmatched = auto_pair_files(collected, mode=pair_mode)
    for t in pairs:
        t["mode"] = _detect_task_mode(requested_mode, t["script"], t.get("audio_b"))
    effective_mode = pair_mode if requested_mode == "auto" and pair_mode == "multi_persp" else requested_mode
    return effective_mode, pairs, unmatched, None


def main(argv: Optional[Sequence[str]] = None) -> int:
    _configure_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.verify_script:
        return _run_verify_script(args.verify_script, as_json=args.json, json_out=args.json_out)

    mode, tasks, unmatched, err = _resolve_tasks(args)
    if err:
        if args.json:
            payload = {"ok": False, "error": err, "mode": mode, "tasks": [], "unmatched": []}
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            if args.json_out:
                with open(args.json_out, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
        else:
            print(f"[ERROR] {err}", file=sys.stderr)
        return 2

    log_stream = sys.stderr if args.json else sys.stdout

    def _emit_log(msg: str) -> None:
        if args.quiet:
            return
        try:
            print(msg, file=log_stream, flush=True)
        except Exception:
            pass

    if args.dry_run:
        preview_tasks = []
        for t in tasks:
            info = _inspect_script_light(t["script"])
            preview_tasks.append({
                "name": t["name"],
                "mode": t["mode"],
                "audio": t["audio"],
                "audio_b": t.get("audio_b"),
                "script": t["script"],
                "confidence": t.get("confidence", 100),
                "script_info": info,
            })
        payload = {
            "ok": len(preview_tasks) > 0,
            "dry_run": True,
            "mode": mode,
            "matched_count": len(preview_tasks),
            "unmatched_count": len(unmatched),
            "tasks": preview_tasks,
            "unmatched": unmatched,
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(f"[DRY-RUN] mode={mode} | 已配对={len(preview_tasks)} | 未配对={len(unmatched)}")
            for i, pt in enumerate(preview_tasks, 1):
                si = pt["script_info"]
                spks = ", ".join(si["detected_speakers"]) if si["detected_speakers"] else "无"
                aud_desc = pt["audio"] + (f" + {pt['audio_b']}" if pt.get("audio_b") else "")
                print(
                    f"  {i}. {pt['name']} -> {pt['mode']} (匹配度 {pt['confidence']}%) | "
                    f"编码={si['encoding']} 行数={si['non_empty_lines']} 角色=[{spks}]\n"
                    f"     台本: {pt['script']}\n"
                    f"     音频: {aud_desc}"
                )
            for u in unmatched:
                print(f"  [未配对] {u['type']}: {u['file']}")
        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        return 0 if preview_tasks else 1

    if not tasks:
        payload = {
            "ok": False,
            "error": "未找到可配对的音频与台本文件",
            "mode": mode,
            "tasks": [],
            "unmatched": unmatched,
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print("[ERROR] 未找到可配对的音频与台本文件", file=sys.stderr)
        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        return 1

    results = []
    all_ok = True
    is_batch = len(tasks) > 1
    try:
        for idx, t in enumerate(tasks, 1):
            t_mode = t["mode"]
            _emit_log(f"\n=== [{idx}/{len(tasks)}] {t['name']} (mode={t_mode}) ===")
            summary: Dict[str, Any] = {}
            orig_stdout = sys.stdout
            if args.json:
                sys.stdout = sys.stderr
            try:
                ok, lrc_path, error = run_pipeline(
                    script_path=t["script"],
                    audio_path=t["audio"],
                    lora_path=args.lora,
                    log_callback=_emit_log if not args.quiet else (lambda _m: None),
                    progress_callback=None,
                    dev=args.device,
                    vram_chunk=args.vram_chunk,
                    persist_gpu=is_batch,
                    line_end_extend=args.line_end_extend,
                    enable_silence_snap=not args.no_silence_snap,
                    enable_resegment=not args.no_resegment,
                    enable_ono_gate=not args.no_ono_gate,
                    enable_merge_short=not args.no_merge_short,
                    enable_multi_order=(t_mode in ("multi_speaker", "multi_persp")),
                    audio_path_b=t.get("audio_b"),
                    output_dir=args.output_dir,
                    summary_out=summary,
                    echo_stdout=False,
                )
            finally:
                sys.stdout = orig_stdout

            if not ok:
                all_ok = False
            results.append({
                "name": t["name"],
                "ok": ok,
                "mode": t_mode,
                "audio": t["audio"],
                "audio_b": t.get("audio_b"),
                "script": t["script"],
                "primary_lrc": lrc_path,
                "error": error,
                **summary,
            })
    finally:
        if is_batch:
            _release_gpu()

    payload = {
        "ok": all_ok,
        "dry_run": False,
        "mode": mode,
        "succeeded": sum(1 for r in results if r["ok"]),
        "failed": sum(1 for r in results if not r["ok"]),
        "unmatched_count": len(unmatched),
        "tasks": results,
        "unmatched": unmatched,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
