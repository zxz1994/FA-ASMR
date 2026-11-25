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

from fa_asmr.pipeline.runner import (
    AUDIO_EXTS,
    _extract_perspective_tag,
    _iter_speaker_marks,
    _release_gpu,
    _split_speaker_segments,
    annotate_lines_with_overrides,
    auto_pair_files,
    extract_ruby_glossary,
    load_ruby_map,
    merge_ruby_glossaries,
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


def _collect_batch_files(inputs: Sequence[str], recursive: bool = False) -> tuple[List[str], List[Dict[str, Any]]]:
    files: List[str] = []
    unsupported: List[Dict[str, Any]] = []
    seen = set()
    for raw in inputs:
        path = os.path.abspath(raw)
        if os.path.isdir(path):
            if recursive:
                for root, _dirs, names in os.walk(path):
                    for name in sorted(names):
                        full = os.path.join(root, name)
                        ext = os.path.splitext(name)[1].lower()
                        if (ext == ".txt" or ext in AUDIO_EXTS) and not name.lower().endswith(".dropped.txt") and full not in seen:
                            seen.add(full)
                            files.append(full)
            else:
                for name in sorted(os.listdir(path)):
                    full = os.path.join(path, name)
                    if not os.path.isfile(full):
                        continue
                    ext = os.path.splitext(name)[1].lower()
                    if (ext == ".txt" or ext in AUDIO_EXTS) and not name.lower().endswith(".dropped.txt") and full not in seen:
                        seen.add(full)
                        files.append(full)
        elif os.path.isfile(path):
            ext = os.path.splitext(path)[1].lower()
            if (ext == ".txt" or ext in AUDIO_EXTS) and not path.lower().endswith(".dropped.txt"):
                if path not in seen:
                    seen.add(path)
                    files.append(path)
            else:
                unsupported.append({
                    "type": "不支持的格式",
                    "file": path,
                    "name": os.path.splitext(os.path.basename(path))[0],
                })
    return files, unsupported


def _write_json_out(json_out: Optional[str], payload: Dict[str, Any]) -> None:
    if not json_out:
        return
    out_dir = os.path.dirname(os.path.abspath(json_out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _inspect_script_light(script_path: str) -> Dict[str, Any]:
    """轻量读取台本元信息（编码、行数、说话人列表与推荐模式），不加载 PyTorch/注音模型。"""
    text, enc = read_text_auto_encoding(script_path)
    raw_lines = text.splitlines()
    non_empty = [ln for ln in raw_lines if ln.strip()]
    warnings: List[str] = []
    if enc.endswith("-fallback") or "\ufffd" in text:
        warnings.append(f"台本编码异常 ({enc})，包含 {text.count(chr(0xFFFD))} 处无法解码字符")
    if not non_empty:
        warnings.append("台本不包含任何非空文本行")

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
        "warnings": warnings,
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


def _parse_only_filters(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [tok.strip() for tok in re.split(r"[,，;；]+", str(raw)) if tok.strip()]


_EMBEDDED_TRACK_NUM_RE = re.compile(r"(?:track|trk|トラック|第)\s*0*(\d{1,3})(?=\D|$)", re.IGNORECASE)


def _matches_only_filter(name_or_path: str, filters: Sequence[str]) -> bool:
    if not filters:
        return True
    bname = os.path.basename(str(name_or_path))
    stem = os.path.splitext(bname)[0]
    bname_l = bname.lower()
    stem_l = stem.lower()
    m_num = re.match(r"^0*(\d+)(?=[_\-\u2017\s.「【]|$)", stem) or _EMBEDDED_TRACK_NUM_RE.search(stem)
    stem_num = int(m_num.group(1)) if m_num else None
    for tok in filters:
        tok_l = tok.lower()
        if tok_l.isdigit():
            if stem_num is not None and stem_num == int(tok_l):
                return True
        elif tok_l in stem_l or tok_l in bname_l:
            return True
    return False


def _load_manifest_tasks(
    manifest_path: str,
    default_mode: str = "auto",
    output_dir: Optional[str] = None,
) -> tuple[List[Dict[str, Any]], Optional[str]]:
    """读取显式配对清单 JSON 并校验每个任务的路径与模式。"""
    abs_m = os.path.abspath(manifest_path)
    if not os.path.isfile(abs_m):
        return [], f"配对清单文件不存在: {abs_m}"
    try:
        with open(abs_m, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except Exception as e:
        return [], f"解析配对清单 JSON 失败 ({abs_m}): {type(e).__name__}: {e}"

    manifest_dir = os.path.dirname(abs_m)
    base_dir = manifest_dir
    if isinstance(data, list):
        raw_items = data
    elif isinstance(data, dict):
        if data.get("base_dir"):
            bd = str(data["base_dir"])
            base_dir = os.path.abspath(bd) if os.path.isabs(bd) else os.path.abspath(os.path.join(manifest_dir, bd))
        raw_items = data.get("tasks") if "tasks" in data else data.get("pairs")
        if not isinstance(raw_items, list):
            return [], "配对清单对象必须包含 'tasks'（或 'pairs'）数组"
    else:
        return [], "配对清单顶层必须为任务数组 [...] 或对象 {base_dir?, tasks: [...]}"

    if not raw_items:
        return [], f"配对清单不包含任何任务: {abs_m}"

    def _res_path(p_val: Any) -> str:
        s = str(p_val).strip()
        if os.path.isabs(s):
            return os.path.abspath(s)
        cand_base = os.path.abspath(os.path.join(base_dir, s))
        if os.path.exists(cand_base):
            return cand_base
        cand_cwd = os.path.abspath(s)
        if os.path.exists(cand_cwd):
            return cand_cwd
        return cand_base

    valid_modes = {"auto", "single", "multi_speaker", "multi_persp"}
    tasks: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_items, 1):
        if not isinstance(item, dict):
            return [], f"清单第 {idx} 项必须为 JSON 对象"
        if not item.get("script") or not item.get("audio"):
            return [], f"清单第 {idx} 项必须同时提供 'script' 与 'audio'"
        script = _res_path(item["script"])
        audio = _res_path(item["audio"])
        audio_b = _res_path(item["audio_b"]) if item.get("audio_b") else None
        for label, p in (("script", script), ("audio", audio)):
            if not os.path.isfile(p):
                return [], f"清单第 {idx} 项文件不存在 ({label}): {p}"
        if audio_b:
            if not os.path.isfile(audio_b):
                return [], f"清单第 {idx} 项文件不存在 (audio_b): {audio_b}"
            if os.path.normcase(audio) == os.path.normcase(audio_b):
                return [], f"清单第 {idx} 项 audio 与 audio_b 不能指向同一文件: {audio}"
            if output_dir:
                ba = os.path.splitext(os.path.basename(audio))[0]
                bb = os.path.splitext(os.path.basename(audio_b))[0]
                if os.path.normcase(ba) == os.path.normcase(bb):
                    return [], f"清单第 {idx} 项双视角音频基名相同 ({ba}) 且指定了 --output-dir，输出将互相覆盖"

        req_m = str(item.get("mode") or default_mode).strip()
        if req_m not in valid_modes:
            return [], f"清单第 {idx} 项包含无效 mode='{req_m}'（可选: auto/single/multi_speaker/multi_persp）"
        if audio_b and req_m != "multi_persp":
            req_m = "multi_persp"
        if req_m == "multi_persp" and not audio_b:
            return [], f"清单第 {idx} 项指定了 multi_persp，但未提供 audio_b"

        t_name = str(item.get("name") or os.path.splitext(os.path.basename(audio))[0]).strip()
        tasks.append({
            "id": str(item["id"]).strip() if item.get("id") is not None else "",
            "name": t_name,
            "requested_mode": req_m,
            "script": script,
            "audio": audio,
            "audio_b": audio_b,
            "confidence": 100,
            "speaker_marks": [],
        })
    return tasks, None


def _run_batch_export_ruby(
    batch_inputs: Sequence[str],
    recursive: bool,
    only_filters: Sequence[str],
    ruby_map: Optional[str],
    as_json: bool,
    json_out: Optional[str],
    explicit_txt_files: Optional[Sequence[str]] = None,
) -> int:
    """整目录或清单一次加载 AutoRuby 合并导出去重注音词表（避免逐轨重复冷启动与重复审校）。"""
    if explicit_txt_files is None:
        missing = [p for p in batch_inputs if not os.path.exists(p)]
        if missing:
            err = f"批量路径不存在: {', '.join(missing)}"
            payload = {"ok": False, "export_ruby": True, "batch": True, "error": err}
            if as_json:
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            else:
                print(f"[ERROR] {err}", file=sys.stderr)
            _write_json_out(json_out, payload)
            return 2

    norm_ruby_map = None
    if ruby_map:
        try:
            norm_ruby_map = load_ruby_map(ruby_map)
        except Exception as e:
            payload = {"ok": False, "export_ruby": True, "batch": True, "error": str(e), "error_type": type(e).__name__}
            if as_json:
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            else:
                print(f"[ERROR] {e}", file=sys.stderr)
            _write_json_out(json_out, payload)
            return 2

    if explicit_txt_files is not None:
        txt_files = list(explicit_txt_files)
    else:
        collected, _ = _collect_batch_files(batch_inputs, recursive=recursive)
        txt_files = [f for f in collected if os.path.splitext(f)[1].lower() == ".txt"]
        if only_filters:
            txt_files = [f for f in txt_files if _matches_only_filter(f, only_filters)]
    if not txt_files:
        err = "批量目录中未找到任何匹配的 .txt 台本文件"
        payload = {"ok": False, "export_ruby": True, "batch": True, "error": err, "scripts": []}
        if as_json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(f"[ERROR] {err}", file=sys.stderr)
        _write_json_out(json_out, payload)
        return 1

    # 生成紧凑且无冲突的 file_key（优先使用唯一数字前缀如 "05"，否则用文件基名）
    raw_prefixes: List[Optional[str]] = []
    for p in txt_files:
        stem = os.path.splitext(os.path.basename(p))[0]
        m = re.match(r"^(\d{1,3})(?=[_\-\s.「【]|$)", stem)
        raw_prefixes.append(m.group(1) if m else None)
    pref_counts: Dict[str, int] = {}
    for pr in raw_prefixes:
        if pr is not None:
            pref_counts[pr] = pref_counts.get(pr, 0) + 1
    file_keys = [
        (pr if (pr is not None and pref_counts.get(pr, 0) == 1) else os.path.splitext(os.path.basename(p))[0])
        for p, pr in zip(txt_files, raw_prefixes)
    ]

    orig_stdout = sys.stdout
    if as_json:
        sys.stdout = sys.stderr
    try:
        from fa_asmr.pipeline.runner import AutoRuby, _ensure_runtime_ready
        _ensure_runtime_ready()
        from fa_asmr.pipeline.pairing import auto_normalize_speaker_headers

        ruby = AutoRuby()
        ann_inst = getattr(ruby, "_ann", None)
        cleaner = None
        if ann_inst is not None and getattr(ann_inst, "_tagger", None):
            from fa_asmr.text.inline_leak_clean import InlineLeakCleaner
            cleaner = InlineLeakCleaner(ann_inst)

        scripts_meta: List[Dict[str, Any]] = []
        per_file_gl: List[tuple[str, List[Dict[str, Any]]]] = []
        all_speakers: List[str] = []
        total_override_hits = 0
        warnings: List[str] = []
        if ann_inst is not None and getattr(ann_inst, "fugashi_error", None):
            warnings.append(f"fugashi/UniDic 未就绪 ({ann_inst.fugashi_error})，已降级为 pykakasi 逐字注音")

        try:
            for fk, txt_p in zip(file_keys, txt_files):
                raw_text, detected_enc = read_text_auto_encoding(txt_p)
                f_warns: List[str] = []
                if detected_enc.endswith("-fallback") or "\ufffd" in raw_text:
                    f_warns.append(f"台本编码异常 ({detected_enc})")
                lines = raw_text.splitlines()
                lines, _ = auto_normalize_speaker_headers(lines)

                body_lines: List[str] = []
                src_line_nos: List[int] = []
                spks: List[str] = []
                cur_spk = ()
                for raw_ln_no, raw in enumerate(lines, 1):
                    segs, cur_spk = _split_speaker_segments(raw, cur_spk)
                    for spk, body in segs:
                        for s in spk:
                            if s and s not in spks:
                                spks.append(s)
                            if s and s not in all_speakers:
                                all_speakers.append(s)
                        if body.strip():
                            body_lines.append(body)
                            src_line_nos.append(raw_ln_no)

                if cleaner is not None:
                    body_lines = [cleaner.clean(b) for b in body_lines]

                ann_lines, hits = annotate_lines_with_overrides(
                    ruby, body_lines, src_line_nos=src_line_nos, ruby_map=norm_ruby_map, script_path=txt_p
                )
                total_override_hits += hits
                gl = extract_ruby_glossary(
                    ann_lines, src_line_nos=src_line_nos, speaker_names=spks, file_label=fk
                )
                per_file_gl.append((fk, gl))
                scripts_meta.append({
                    "file_key": fk,
                    "script": txt_p,
                    "encoding": detected_enc,
                    "dialogue_lines": len(body_lines),
                    "detected_speakers": spks,
                    "warnings": f_warns,
                })
        finally:
            ruby.release_gpu()

        merged_gl = merge_ruby_glossaries(per_file_gl, speaker_names=all_speakers)
        payload: Dict[str, Any] = {
            "ok": True,
            "export_ruby": True,
            "batch": True,
            "scripts_count": len(scripts_meta),
            "detected_speakers": all_speakers,
            "warnings": warnings,
            "ruby_stats": {
                "unique_terms": len(merged_gl),
                "high_priority_terms": sum(1 for g in merged_gl if g.get("priority") == "high"),
                "polyphone_terms": sum(1 for g in merged_gl if g["is_polyphone"]),
                "total_occurrences": sum(g["count"] for g in merged_gl),
            },
            "scripts": scripts_meta,
            "ruby_glossary": merged_gl,
        }
        if norm_ruby_map is not None:
            payload["ruby_overrides_applied"] = total_override_hits

        sys.stdout = orig_stdout
        if as_json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            st = payload["ruby_stats"]
            print(
                f"=== FA-ASMR 批量合并注音词表 ({len(scripts_meta)} 份台本) ===\n"
                f"去重词条: {st['unique_terms']} (高危必查 {st['high_priority_terms']} / 多音词 {st['polyphone_terms']} / 总计 {st['total_occurrences']} 处)"
            )
            for g in merged_gl[:40]:
                tag = "[多音] " if g["is_polyphone"] else ("[高危] " if g.get("priority") == "high" else "")
                rds = ", ".join(f"{r}×{c}" for r, c in g["readings"].items())
                cols = "/".join(g.get("collocations", [])[:3])
                ex_txt = g["examples"][0]["context"] if g["examples"] else ""
                print(f"  {tag}{g['surface']} ({cols}) -> {rds} | {ex_txt}")
            if len(merged_gl) > 40:
                print(f"  ...（其余 {len(merged_gl) - 40} 词请加 --json 或 --json-out 查看完整词表）")
        _write_json_out(json_out, payload)
        return 0
    except Exception as e:
        sys.stdout = orig_stdout
        err_msg = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
        payload = {"ok": False, "export_ruby": True, "batch": True, "error": err_msg, "error_type": type(e).__name__}
        if as_json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(f"[ERROR] 批量注音导出失败: {err_msg}", file=sys.stderr)
        _write_json_out(json_out, payload)
        return 1
    finally:
        sys.stdout = orig_stdout


def _run_verify_script(
    script_path: str,
    as_json: bool,
    json_out: Optional[str],
    export_ruby: bool = False,
    ruby_map: Optional[str] = None,
) -> int:
    """深度校验单个台本：完整运行自动编码嗅探、角色标题归一化、AutoRuby 注音、顿号桥接与子句切分。"""
    abs_path = os.path.abspath(script_path)
    if not os.path.isfile(abs_path):
        err = f"台本文件不存在: {abs_path}"
        payload = {"ok": False, "verify_script": True, "script": abs_path, "error": err, "error_type": "FileNotFoundError"}
        if as_json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(f"[ERROR] {err}", file=sys.stderr)
        _write_json_out(json_out, payload)
        return 2

    norm_ruby_map = None
    if ruby_map:
        try:
            norm_ruby_map = load_ruby_map(ruby_map)
        except Exception as e:
            err = str(e)
            payload = {
                "ok": False,
                "verify_script": True,
                "script": abs_path,
                "error": err,
                "error_type": type(e).__name__,
            }
            if as_json:
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            else:
                print(f"[ERROR] {err}", file=sys.stderr)
            _write_json_out(json_out, payload)
            return 2

    orig_stdout = sys.stdout
    if as_json:
        sys.stdout = sys.stderr
    try:
        from fa_asmr.pipeline.runner import AutoRuby, _ensure_runtime_ready
        _ensure_runtime_ready()
        import fa_asmr.text.clause as clause
        import fa_asmr.text.haruraw2norm as hn
        from fa_asmr.pipeline.pairing import auto_normalize_speaker_headers

        warnings: List[str] = []
        raw_text, detected_enc = read_text_auto_encoding(abs_path)
        if detected_enc.endswith("-fallback") or "\ufffd" in raw_text:
            warnings.append(f"台本编码异常 ({detected_enc})，包含 {raw_text.count(chr(0xFFFD))} 处无法解码字符")

        lines = raw_text.splitlines()
        lines, _auto_spks = auto_normalize_speaker_headers(lines)

        body_lines: List[str] = []
        src_line_nos: List[int] = []
        speakers: List[str] = []
        cur_spk = ()
        for raw_ln_no, raw in enumerate(lines, 1):
            segs, cur_spk = _split_speaker_segments(raw, cur_spk)
            for spk, body in segs:
                for s in spk:
                    if s and s not in speakers:
                        speakers.append(s)
                if body.strip():
                    body_lines.append(body)
                    src_line_nos.append(raw_ln_no)

        ruby = AutoRuby()
        ann_inst = getattr(ruby, "_ann", None)
        if ann_inst is not None and getattr(ann_inst, "fugashi_error", None):
            warnings.append(f"fugashi/UniDic 未就绪 ({ann_inst.fugashi_error})，已降级为 pykakasi 逐字注音")

        if ann_inst is not None and getattr(ann_inst, "_tagger", None):
            from fa_asmr.text.inline_leak_clean import InlineLeakCleaner
            _cleaner = InlineLeakCleaner(ann_inst)
            body_lines = [_cleaner.clean(b) for b in body_lines]

        try:
            annotated_lines, ruby_override_hits = annotate_lines_with_overrides(
                ruby, body_lines, src_line_nos=src_line_nos, ruby_map=norm_ruby_map, script_path=abs_path
            )
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

        ok = len(body_lines) > 0 and total_tokens > 0
        err_msg = None if ok else "台本未解析出任何可对齐的发音音素（有效台词行或音素数为 0）"

        payload: Dict[str, Any] = {
            "ok": ok,
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
            "warnings": warnings,
            "error": err_msg,
        }
        if norm_ruby_map is not None:
            payload["ruby_overrides_applied"] = ruby_override_hits
        if export_ruby:
            glossary = extract_ruby_glossary(annotated_lines, src_line_nos=src_line_nos, speaker_names=speakers)
            payload["ruby_stats"] = {
                "unique_terms": len(glossary),
                "high_priority_terms": sum(1 for g in glossary if g.get("priority") == "high"),
                "polyphone_terms": sum(1 for g in glossary if g["is_polyphone"]),
                "total_occurrences": sum(g["count"] for g in glossary),
            }
            payload["ruby_glossary"] = glossary

        sys.stdout = orig_stdout
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
            if norm_ruby_map is not None:
                print(f"注音覆写命中: {ruby_override_hits} 处")
            if export_ruby:
                st = payload["ruby_stats"]
                print(
                    f"注音去重词表: {st['unique_terms']} 词 "
                    f"(多音词 {st['polyphone_terms']} / 总计 {st['total_occurrences']} 处)"
                )
                for g in payload["ruby_glossary"][:30]:
                    flag = "[多音] " if g["is_polyphone"] else ""
                    rds = ", ".join(f"{r}×{c}" for r, c in g["readings"].items())
                    ex_txt = g["examples"][0]["context"] if g["examples"] else ""
                    print(f"  {flag}{g['surface']} -> {rds} | {ex_txt}")
                if len(payload["ruby_glossary"]) > 30:
                    print(f"  ...（其余 {len(payload['ruby_glossary']) - 30} 词请加 --json 或 --json-out 查看完整词表）")
            for w in warnings:
                print(f"[WARN] {w}", file=sys.stderr)
            if err_msg:
                print(f"[ERROR] {err_msg}", file=sys.stderr)

        _write_json_out(json_out, payload)
        return 0 if ok else 1
    except Exception as e:
        sys.stdout = orig_stdout
        err_msg = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
        payload = {
            "ok": False,
            "verify_script": True,
            "script": abs_path,
            "error": err_msg,
            "error_type": type(e).__name__,
        }
        if as_json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(f"[ERROR] 台本校验失败: {err_msg}", file=sys.stderr)
        _write_json_out(json_out, payload)
        return 1
    finally:
        sys.stdout = orig_stdout


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
    p.add_argument(
        "--export-ruby",
        action="store_true",
        help="配合 --verify-script 或 --batch 使用：导出分级去重注音词表与局部搭配上下文（ruby_glossary），供外部 LLM 审校读音",
    )
    p.add_argument(
        "--ruby-map",
        metavar="JSON",
        help="外部注音覆写字典 JSON（支持扁平 {词: 读音} 或 {global: {...}, by_line: {\"12\": {...}}}），可用于 --verify-script / 单条 / 批量对齐",
    )
    p.add_argument("--audio", help="单条模式：主音频路径（多视角模式下为视角 A 音频）")
    p.add_argument("--audio-b", dest="audio_b", help="多视角模式：视角 B 音频路径")
    p.add_argument("--script", help="单条模式：台本文本路径（自动识别 UTF-8 / Shift-JIS / GB18030 / UTF-16）")
    p.add_argument(
        "--batch",
        nargs="+",
        help="批量模式：传入一个或多个目录/文件路径，自动按文件名配对台本与音频",
    )
    p.add_argument(
        "--manifest",
        metavar="JSON",
        help="显式配对清单模式：传入配对 JSON（支持 [{script, audio, audio_b?, mode?, name?}] 或 {base_dir?, tasks: [...]}），适用于混合单轨/双视角/差分轨目录",
    )
    p.add_argument("--recursive", action="store_true", help="批量模式下递归扫描子目录")
    p.add_argument(
        "--only",
        metavar="FILTERS",
        help="批量模式下按逗号分隔的轨道序号或文件名关键词筛选局部重跑（如 --only 05,07,13）",
    )
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
        default=1.2,
        help="字幕行尾自然延长秒数（默认 1.2，传 0 关闭）",
    )
    p.add_argument("--no-silence-snap", action="store_true", help="关闭静音吸附")
    p.add_argument("--no-resegment", action="store_true", help="关闭标点驱动重分段")
    p.add_argument("--no-ono-gate", action="store_true", help="关闭拟声词守门")
    p.add_argument("--no-merge-short", action="store_true", help="关闭短句智能合并")
    p.add_argument("--output-dir", help="自定义字幕输出目录（默认输出到各音频同级目录）")
    p.add_argument(
        "--audit",
        action="store_true",
        help="对齐结束后输出声学体检报告（低分可疑词/疑似未录句 + 字幕间隙未覆盖活跃语音空档）",
    )
    p.add_argument(
        "--audit-out",
        metavar="JSON",
        help="将声学体检报告单独写入指定 JSON 文件（自动开启 --audit）",
    )
    p.add_argument(
        "--audit-clips-dir",
        metavar="DIR",
        help="按体检报告自动切出异常点前后短音频 WAV 到指定目录供人工/多模态听证（自动开启 --audit）",
    )
    p.add_argument(
        "--audit-separate",
        action="store_true",
        help="联动第三步空间分离：体检切片若检测到立体声多方位重叠，自动附带输出左/中/右干声 WAV 并回填 audit.json（自动开启 --audit）",
    )
    p.add_argument(
        "--audit-sep-sources",
        type=int,
        default=0,
        choices=(0, 2, 3, 4),
        help="联动空间分离路数：0=按台本说话人数自动判定（单人轨跳过分离，多人轨选 2 或 3，默认），或手动指定 2/3/4",
    )
    p.add_argument(
        "--audit-clips-all",
        action="store_true",
        help="体检切片时导出全部可疑句 WAV（默认仅切出 priority_review 高危必审项与缺句空档，减少多模态听审冗余）",
    )
    p.add_argument(
        "--audit-clips-compact",
        action="store_true",
        help="将体检审听切片转为 16kHz 单声道 PCM16（先在原生立体声上完成空间分离再瘦身，减少多模态听审体积与上下文开销）",
    )
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

    if args.line_end_extend < 0:
        return requested_mode, [], [], f"--line-end-extend 不能为负数: {args.line_end_extend}"
    if args.vram_chunk <= 0:
        return requested_mode, [], [], f"--vram-chunk 必须为正整数: {args.vram_chunk}"
    if args.output_dir and os.path.isfile(os.path.abspath(args.output_dir)):
        return requested_mode, [], [], f"--output-dir 指向已存在的普通文件，无法作为输出目录: {os.path.abspath(args.output_dir)}"
    if args.lora is not None and str(args.lora).strip() != "":
        if str(args.lora).upper() not in ("NO_LORA", "NONE", "DISABLE"):
            lora_abs = os.path.abspath(args.lora)
            if not os.path.isfile(lora_abs):
                return requested_mode, [], [], f"指定的 LoRA 权重文件不存在: {lora_abs}"

    has_single = bool(args.audio or args.script or args.audio_b)
    has_batch = bool(args.batch)
    has_manifest = bool(getattr(args, "manifest", None))
    if sum(1 for flag in (has_single, has_batch, has_manifest) if flag) > 1:
        return requested_mode, [], [], "不能同时指定 --manifest、--batch 与 --audio/--script"
    if not (has_single or has_batch or has_manifest):
        return requested_mode, [], [], "请指定 --verify-script <TXT>、--audio/--script、--batch <目录/文件> 或 --manifest <JSON>"

    if has_manifest:
        m_tasks, m_err = _load_manifest_tasks(args.manifest, default_mode=requested_mode, output_dir=args.output_dir)
        if m_err:
            return requested_mode, [], [], m_err
        only_filters = _parse_only_filters(getattr(args, "only", None))
        if only_filters:
            m_tasks = [
                t for t in m_tasks
                if (t.get("id") and _matches_only_filter(t["id"], only_filters))
                or _matches_only_filter(t["name"], only_filters)
                or _matches_only_filter(t["script"], only_filters)
                or _matches_only_filter(t["audio"], only_filters)
                or (t.get("audio_b") and _matches_only_filter(t["audio_b"], only_filters))
            ]
            if not m_tasks:
                return requested_mode, [], [], f"--only 过滤条件 ({args.only}) 未匹配到任何清单任务"
        for t in m_tasks:
            try:
                t["mode"] = _detect_task_mode(t["requested_mode"], t["script"], t.get("audio_b"))
            except Exception as e:
                return requested_mode, [], [], f"读取台本失败 ({t['script']}): {type(e).__name__}: {e}"
        seen_outputs: Dict[str, str] = {}
        for t in m_tasks:
            for aud_key in ("audio", "audio_b"):
                aud_p = t.get(aud_key)
                if not aud_p:
                    continue
                out_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.dirname(os.path.abspath(aud_p))
                out_lrc = os.path.normcase(os.path.join(out_dir, os.path.splitext(os.path.basename(aud_p))[0] + ".lrc"))
                if out_lrc in seen_outputs:
                    return requested_mode, [], [], (
                        f"输出字幕路径冲突: {aud_p} 与 {seen_outputs[out_lrc]} 将覆盖同一目标文件 ({out_lrc})"
                    )
                seen_outputs[out_lrc] = aud_p
        uniq_modes = {t["mode"] for t in m_tasks}
        eff_mode = next(iter(uniq_modes)) if len(uniq_modes) == 1 else "mixed"
        return eff_mode, m_tasks, [], None

    if has_single:
        if not args.audio or not args.script:
            return requested_mode, [], [], "单条模式必须同时提供 --audio 和 --script"
        audio = os.path.abspath(args.audio)
        script = os.path.abspath(args.script)
        audio_b = os.path.abspath(args.audio_b) if args.audio_b else None
        for label, p in (("audio", audio), ("script", script)):
            if not os.path.isfile(p):
                return requested_mode, [], [], f"文件不存在 ({label}): {p}"
        if audio_b:
            if not os.path.isfile(audio_b):
                return requested_mode, [], [], f"文件不存在 (audio_b): {audio_b}"
            if os.path.normcase(audio) == os.path.normcase(audio_b):
                return requested_mode, [], [], f"--audio 与 --audio-b 不能指向同一个文件: {audio}"
            if args.output_dir:
                base_a = os.path.splitext(os.path.basename(audio))[0]
                base_b = os.path.splitext(os.path.basename(audio_b))[0]
                if os.path.normcase(base_a) == os.path.normcase(base_b):
                    return requested_mode, [], [], (
                        f"双视角音频基名相同 ({base_a}) 且指定了 --output-dir，输出字幕将互相覆盖"
                    )
        if requested_mode == "multi_persp" and not audio_b:
            return requested_mode, [], [], "multi_persp 模式需要同时提供 --audio 与 --audio-b"
        try:
            resolved_mode = _detect_task_mode(requested_mode, script, audio_b)
        except Exception as e:
            return requested_mode, [], [], f"读取台本失败 ({script}): {type(e).__name__}: {e}"
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
    collected, unsupported_files = _collect_batch_files(args.batch, recursive=args.recursive)
    pair_mode = requested_mode
    if requested_mode == "auto":
        pair_mode = "multi_persp" if _looks_like_multi_persp_batch(collected) else "single"
    pairs, unmatched = auto_pair_files(collected, mode=pair_mode)
    if unsupported_files:
        unmatched = list(unmatched) + unsupported_files
    only_filters = _parse_only_filters(getattr(args, "only", None))
    if only_filters:
        pairs = [
            t for t in pairs
            if _matches_only_filter(t["name"], only_filters)
            or _matches_only_filter(t["script"], only_filters)
            or _matches_only_filter(t["audio"], only_filters)
        ]
        if not pairs:
            return requested_mode, [], unmatched, f"--only 过滤条件 ({args.only}) 未匹配到任何批处理任务"
    for t in pairs:
        try:
            t["mode"] = _detect_task_mode(requested_mode, t["script"], t.get("audio_b"))
        except Exception as e:
            return requested_mode, [], unmatched, f"读取台本失败 ({t['script']}): {type(e).__name__}: {e}"
    # 校验批量输出路径是否冲突（常见于 --recursive + --output-dir 时不同子目录含同名音频）
    seen_outputs: Dict[str, str] = {}
    for t in pairs:
        for aud_key in ("audio", "audio_b"):
            aud_p = t.get(aud_key)
            if not aud_p:
                continue
            out_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.dirname(os.path.abspath(aud_p))
            out_lrc = os.path.normcase(os.path.join(out_dir, os.path.splitext(os.path.basename(aud_p))[0] + ".lrc"))
            if out_lrc in seen_outputs:
                return requested_mode, [], unmatched, (
                    f"输出字幕路径冲突: {aud_p} 与 {seen_outputs[out_lrc]} 将覆盖同一目标文件 ({out_lrc})"
                )
            seen_outputs[out_lrc] = aud_p
    effective_mode = pair_mode if requested_mode == "auto" and pair_mode == "multi_persp" else requested_mode
    return effective_mode, pairs, unmatched, None


def main(argv: Optional[Sequence[str]] = None) -> int:
    _configure_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.verify_script:
        return _run_verify_script(
            args.verify_script,
            as_json=args.json,
            json_out=args.json_out,
            export_ruby=bool(args.export_ruby),
            ruby_map=args.ruby_map,
        )

    if args.export_ruby:
        if args.manifest and args.batch:
            err = "不能同时指定 --manifest 与 --batch"
            payload = {"ok": False, "export_ruby": True, "error": err}
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            else:
                print(f"[ERROR] {err}", file=sys.stderr)
            _write_json_out(args.json_out, payload)
            return 2
        if args.manifest:
            m_tasks, m_err = _load_manifest_tasks(args.manifest, default_mode=args.mode, output_dir=args.output_dir)
            if m_err:
                payload = {"ok": False, "export_ruby": True, "batch": True, "error": m_err}
                if args.json:
                    print(json.dumps(payload, ensure_ascii=False, indent=2))
                else:
                    print(f"[ERROR] {m_err}", file=sys.stderr)
                _write_json_out(args.json_out, payload)
                return 2
            only_f = _parse_only_filters(args.only)
            if only_f:
                m_tasks = [
                    t for t in m_tasks
                    if (t.get("id") and _matches_only_filter(t["id"], only_f))
                    or _matches_only_filter(t["name"], only_f)
                    or _matches_only_filter(t["script"], only_f)
                    or _matches_only_filter(t["audio"], only_f)
                ]
            seen_sc = set()
            uniq_scripts: List[str] = []
            for t in m_tasks:
                nc = os.path.normcase(t["script"])
                if nc not in seen_sc:
                    seen_sc.add(nc)
                    uniq_scripts.append(t["script"])
            return _run_batch_export_ruby(
                batch_inputs=[],
                recursive=False,
                only_filters=[],
                ruby_map=args.ruby_map,
                as_json=args.json,
                json_out=args.json_out,
                explicit_txt_files=uniq_scripts,
            )
        if args.batch:
            return _run_batch_export_ruby(
                batch_inputs=args.batch,
                recursive=bool(args.recursive),
                only_filters=_parse_only_filters(args.only),
                ruby_map=args.ruby_map,
                as_json=args.json,
                json_out=args.json_out,
            )
        err = "--export-ruby 仅支持配合 --verify-script <TXT>、--batch <目录> 或 --manifest <JSON> 使用"
        payload = {"ok": False, "error": err, "mode": args.mode, "tasks": [], "unmatched": []}
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(f"[ERROR] {err}", file=sys.stderr)
        _write_json_out(args.json_out, payload)
        return 2

    parsed_ruby_map = None
    if args.ruby_map:
        try:
            parsed_ruby_map = load_ruby_map(args.ruby_map)
        except Exception as e:
            err = str(e)
            payload = {"ok": False, "error": err, "error_type": type(e).__name__, "mode": args.mode, "tasks": [], "unmatched": []}
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            else:
                print(f"[ERROR] {err}", file=sys.stderr)
            _write_json_out(args.json_out, payload)
            return 2

    mode, tasks, unmatched, err = _resolve_tasks(args)
    if err:
        payload = {"ok": False, "error": err, "mode": mode, "tasks": [], "unmatched": []}
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(f"[ERROR] {err}", file=sys.stderr)
        _write_json_out(args.json_out, payload)
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
        inspect_errors = []
        for t in tasks:
            try:
                info = _inspect_script_light(t["script"])
            except Exception as e:
                err_msg = f"读取台本失败 ({t['script']}): {type(e).__name__}: {e}"
                inspect_errors.append(err_msg)
                info = {"error": err_msg, "warnings": []}
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
            "ok": len(preview_tasks) > 0 and not inspect_errors,
            "dry_run": True,
            "mode": mode,
            "matched_count": len(preview_tasks),
            "unmatched_count": len(unmatched),
            "tasks": preview_tasks,
            "unmatched": unmatched,
        }
        if inspect_errors:
            payload["errors"] = inspect_errors
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(f"[DRY-RUN] mode={mode} | 已配对={len(preview_tasks)} | 未配对={len(unmatched)}")
            for i, pt in enumerate(preview_tasks, 1):
                si = pt["script_info"]
                aud_desc = pt["audio"] + (f" + {pt['audio_b']}" if pt.get("audio_b") else "")
                if "error" in si:
                    print(f"  {i}. {pt['name']} -> [ERROR] {si['error']}", file=sys.stderr)
                    continue
                spks = ", ".join(si["detected_speakers"]) if si["detected_speakers"] else "无"
                print(
                    f"  {i}. {pt['name']} -> {pt['mode']} (匹配度 {pt['confidence']}%) | "
                    f"编码={si['encoding']} 行数={si['non_empty_lines']} 角色=[{spks}]\n"
                    f"     台本: {pt['script']}\n"
                    f"     音频: {aud_desc}"
                )
                for w in si.get("warnings", []):
                    print(f"     [WARN] {w}", file=sys.stderr)
            for u in unmatched:
                print(f"  [未配对] {u['type']}: {u['file']}")
        _write_json_out(args.json_out, payload)
        return 0 if (preview_tasks and not inspect_errors) else 1

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
        _write_json_out(args.json_out, payload)
        return 1

    results = []
    all_ok = True
    is_batch = len(tasks) > 1
    enable_audit = bool(args.audit or args.audit_out or args.audit_clips_dir or args.audit_separate)
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
                    ruby_map=parsed_ruby_map,
                    enable_audit=enable_audit,
                    audit_clips_dir=args.audit_clips_dir,
                    audit_separate=bool(args.audit_separate),
                    audit_sep_sources=int(args.audit_sep_sources),
                    audit_clips_all=bool(args.audit_clips_all),
                    audit_clips_compact=bool(args.audit_clips_compact),
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
        "warnings_count": sum(len(r.get("warnings", [])) for r in results),
        "unmatched_count": len(unmatched),
        "tasks": results,
        "unmatched": unmatched,
    }
    if args.audit_out:
        audit_payload = {
            "ok": all_ok,
            "mode": mode,
            "tasks": [
                {
                    "name": r["name"],
                    "script": r["script"],
                    "audio": r["audio"],
                    **({"audio_b": r["audio_b"]} if r.get("audio_b") else {}),
                    "audit": r.get("audit"),
                    **({"audit_b": r["audit_b"]} if r.get("audit_b") is not None else {}),
                }
                for r in results
            ],
        }
        _write_json_out(args.audit_out, audit_payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    _write_json_out(args.json_out, payload)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
