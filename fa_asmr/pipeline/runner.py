# -*- coding: utf-8 -*-
"""FA-ASMR V2 — 无界面核心对齐流水线与门面导出。

本模块作为核心流水线编排入口，与拆分出的子域模块协同工作：
- `fa_asmr/pipeline/pairing.py`：台本多编码嗅探、说话人标记拆分、原文锚点映射与批量配对
- `fa_asmr/pipeline/ruby.py`：AutoRuby 单例管理、ruby-map 校验与注音词表提取/合并
- `fa_asmr/pipeline/subtitles.py`：LRC / SRT 标准字幕组装与同步齐声合并

设计原则：
- 不导入 tkinter，不注册 GUI 弹窗异常钩子，可供 GUI、CLI 与单元测试无副作用复用；
- 仅在实际执行 `run_pipeline()` 时延迟加载 torch / align / haruraw2norm，保持轻量导入；
- 保持原有公开及内部导出符号 100% 向后兼容。
"""
from __future__ import annotations

import copy
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

if getattr(sys, "frozen", False):
    SCRIPT_DIR = sys._MEIPASS
    EXE_DIR = os.path.dirname(sys.executable)
else:
    SCRIPT_DIR = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    EXE_DIR = SCRIPT_DIR

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from fa_asmr.text.clause import F

# ── 向后兼容重导出：拆出子域的符号统一在门面模块可见 ─────────────────────
from fa_asmr.pipeline.pairing import (  # noqa: F401
    AUDIO_EXTS,
    _PERSP_PATTERNS,
    _SPEAKER_BRACKET_RE,
    _SPEAKER_SEP,
    _attach_source_spans,
    _clean_stem_for_matching,
    _extract_perspective_tag,
    _iter_speaker_marks,
    _parse_speaker_suffix,
    _split_speaker_segments,
    _strip_ruby_for_src,
    auto_normalize_speaker_headers,
    auto_pair_files,
    read_text_auto_encoding,
)
from fa_asmr.pipeline.ruby import (  # noqa: F401
    _INLINE_RUBY_RE,
    AutoRuby,
    annotate_lines_with_overrides,
    apply_ruby_overrides_to_line,
    extract_ruby_glossary,
    load_ruby_map,
    merge_ruby_glossaries,
    release_global_ruby,
    resolve_ruby_map_for_script,
)
from fa_asmr.pipeline.subtitles import (  # noqa: F401
    _PUNCT_CHARS,
    _cs_to_raw_lrc_ts,
    _fmt_line,
    _fmt_lrc_time,
    _lrc_time_to_cs,
    _merge_unison_lrc_groups,
    _strip_orphan_punct,
    build_independent_multispeaker_srt,
    build_standard_lrc,
    build_standard_srt,
    split_unison_prefix_suffix_rl,
)

_KANA_CH_RE = re.compile(r"[\u3041-\u309F\u30A0-\u30FF]")
_DIGITS_RE = re.compile(r"[0-9]")
_ALPHA_RE = re.compile(r"[A-Za-z]")


def _release_gpu(ruby=None):
    try:
        import fa_asmr.aligners.align as align
        align.clear_model_cache()
    except Exception as e:
        logger.warning("释放 MMS-FA 模型缓存失败: %s: %s", type(e).__name__, e)
    release_global_ruby(ruby)


def _ensure_runtime_ready():
    """确保 NumPy 兼容垫片与内置/系统 PyTorch 已就绪（幂等）。"""
    import fa_asmr.core.np_compat as np_compat
    np_compat.apply()
    from fa_asmr.core.torch_bootstrap import ensure_torch
    ensure_torch(None)


def _text_weight(txt: str) -> int:
    """一段文字的权重：假名×1 + 数字字符×2 + 字母×1。"""
    if not txt:
        return 0
    return (
        len(_KANA_CH_RE.findall(txt))
        + 2 * len(_DIGITS_RE.findall(txt))
        + len(_ALPHA_RE.findall(txt))
    )


def _line_kana_counts(line: str, toks):
    """按子句切分，返回每个 token 所属子句的假名数列表（与 toks 等长）。

    line 是注音串（`{漢字|かんじ}` 格式）；toks 是它 token 化后的列表。
    两边都按同一判据（无读音的符号 = 子句边界）切分，切点数一致。
    """
    plain = re.sub(r"\{([^|]+)\|([^}]+)\}", r"\2", line)
    # 延音符号 ー/〜/～/~ 必须并入段内：它们属发音内容（token 侧归入带
    # pron 的 token，如 な~~~ pron=naaaaaaaaa），若在此被当分隔符会与
    # token 侧子句数错位，导致字数取到错误段（实测 な～～～んにも…
    # 被切成 ['な','んにも…']，整行 15 字只算成首段 'な' = 1 字）。
    _seg_keep = "\u3041-\u309F\u30A0-\u30FFA-Za-z0-9\u30FC\u301C\uFF5E~\u2015\u2014\u2010\uFF0D-"
    segs = [s for s in re.split("[^" + _seg_keep + "]+", plain) if s]

    tsegs, cur = [], []
    for t in toks:
        o = t.get("orig", "")
        if o == "\n":
            continue
        if t.get("pron"):
            cur.append(t)
        elif o and not t.get(F.RUBY_PLACEHOLDER, False):
            if cur:
                tsegs.append(cur)
                cur = []
    if cur:
        tsegs.append(cur)

    out = [0] * len(toks)
    heads = [None] * len(toks)
    idx_of = {id(t): k for k, t in enumerate(toks)}
    for si, seg_toks in enumerate(tsegs):
        n = _text_weight(segs[si] if si < len(segs) else "")
        if n == 0:
            n = sum(_text_weight(t.get("orig", "")) for t in seg_toks)
        seg_head = idx_of[id(seg_toks[0])]
        for t in seg_toks:
            ti = idx_of[id(t)]
            out[ti] = n
            heads[ti] = seg_head
    return out, heads


def _validate_pipeline_inputs(
    script_path: str,
    audio_path: str,
    audio_path_b: Optional[str],
    output_dir: Optional[str],
    ruby_map: Optional[Any],
    line_end_extend: float,
    vram_chunk: int,
) -> Optional[Dict[str, Any]]:
    """前置校验所有管线入参，并返回规范化后的 ruby_map（如有）。"""
    if not script_path or not os.path.isfile(script_path):
        raise FileNotFoundError(f"台本文件不存在: {script_path}")
    if not audio_path or not os.path.isfile(audio_path):
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")
    norm_ruby_map = load_ruby_map(ruby_map) if ruby_map is not None else None
    if audio_path_b:
        if not os.path.isfile(audio_path_b):
            raise FileNotFoundError(f"视角 B 音频文件不存在: {audio_path_b}")
        if os.path.normcase(os.path.abspath(audio_path)) == os.path.normcase(os.path.abspath(audio_path_b)):
            raise ValueError(f"双视角模式的视角 A 与视角 B 不能指向同一个音频文件: {audio_path}")
    if output_dir and os.path.isfile(os.path.abspath(output_dir)):
        raise ValueError(f"输出目录路径已存在且为普通文件，无法写入目录: {output_dir}")
    if float(line_end_extend) < 0:
        raise ValueError(f"行尾延长秒数不能为负数: {line_end_extend}")
    if int(vram_chunk) <= 0:
        raise ValueError(f"显存分块帧数必须 > 0: {vram_chunk}")
    return norm_ruby_map


def _resolve_lora_path(align_mod: Any, lora_path: Optional[str], log, warn) -> Optional[str]:
    """解析默认 / 显式指定的 LoRA 权重路径，缺件时显式告警或报错。"""
    lora_missing = None
    default_lora = str(getattr(align_mod, "DEFAULT_LORA_PATH", ""))
    norm_default = os.path.normcase(os.path.abspath(default_lora)) if default_lora else ""
    if lora_path is None or str(lora_path).strip() == "":
        if default_lora and os.path.isfile(default_lora):
            lora_path = default_lora
        else:
            lora_path = None
            lora_missing = default_lora
    elif str(lora_path).upper() in ("NO_LORA", "NONE", "DISABLE"):
        lora_path = "NO_LORA"
    elif not os.path.isfile(str(lora_path)):
        norm_req = os.path.normcase(os.path.abspath(str(lora_path)))
        if norm_default and norm_req == norm_default:
            lora_path = None
            lora_missing = default_lora
        else:
            raise FileNotFoundError(
                f"指定的 LoRA 权重文件不存在: {lora_path}。"
                "请检查路径是否正确；若需使用基础模型，请传 NO_LORA。"
            )

    if lora_path and lora_path != "NO_LORA":
        log(f"  ✨ 使用 LoRA: {os.path.basename(lora_path)}")
    elif lora_missing:
        warn(
            f"未找到默认 LoRA 权重 ({lora_missing})，已回退到基础 MMS-FA 模型 —— 对齐精度会下降。"
            "请将 lora/fa_asmr_v9e10.pt 放到该位置；若确要用基础模型，传 lora_path='NO_LORA' 可关闭本提示。"
        )
    else:
        log("  ✨ 使用基础 MMS-FA 模型")
    return lora_path


def _tokenize_for_alignment(
    hn_mod: Any,
    annotated_text: str,
    lines: Sequence[str],
    spk_of_line: Sequence[tuple],
    src_line_nos: Sequence[int],
    script_path: str,
) -> Dict[str, Any]:
    """将注音后的行转为音素 token，切分子句边界并标记易错拟声。"""
    result_list: List[Dict[str, Any]] = []
    src_idx_to_lineno: List[int] = []
    src_idx_to_ann: List[str] = []

    for li, line in enumerate(annotated_text.splitlines(keepends=True)):
        if line.strip():
            toks = hn_mod.process_haruhi_line(line, "auto", 0, 1)
            counts, heads = _line_kana_counts(line, toks)
            base_idx = len(result_list)
            for t, n, h in zip(toks, counts, heads):
                t[F.KANA_N] = n
                if h is not None:
                    t[F.CLAUSE_HEAD] = base_idx + h
            src_line = lines[li] if li < len(lines) else ""
            _attach_source_spans(toks, line, src_line)
            spk_line = spk_of_line[li] if li < len(spk_of_line) else ()
            if spk_line:
                for t in toks:
                    t[F.SPK] = spk_line
            result_list.extend(toks)
            result_list.append({"orig": "\n", "type": 0, "pron": ""})
            src_idx_to_lineno.append(src_line_nos[li] if li < len(src_line_nos) else (li + 1))
            src_idx_to_ann.append(line.strip())
    if result_list and result_list[-1].get("orig") != "\n":
        result_list.append({"orig": "\n", "type": 0, "pron": ""})

    alignment_tokens: List[List[str]] = []
    line_group: List[int] = []
    clause_surfaces: List[str] = []
    ono_clause_flags: List[int] = []
    clause_speakers: List[tuple] = []
    cur_ono_surface: List[str] = []
    cur_clause_spk: tuple = ()
    token_to_index_map: Dict[int, int] = {}
    current_line: List[str] = []
    flat_idx = 0
    src_idx = 0

    from fa_asmr.text.clause import (
        is_clause_boundary as _is_clause_boundary,
        is_voicing_mark as _is_voicing_mark,
        mark_short_comma_bridges as _mark_comma_bridges,
        ono_token_surface as _ono_token_surface,
    )
    from fa_asmr.text.ono_base import is_easy_ono as _ieo

    _mark_comma_bridges(result_list)
    for i, item in enumerate(result_list):
        orig = item.get("orig", "")
        pron = item.get("pron", "")
        if _is_clause_boundary(item):
            if current_line:
                alignment_tokens.append(current_line)
                line_group.append(src_idx)
                clause_speakers.append(cur_clause_spk)
                surf = "".join(cur_ono_surface)
                clause_surfaces.append(surf)
                ono_clause_flags.append(1 if _ieo(surf) else 0)
                cur_ono_surface = []
                cur_clause_spk = ()
                current_line = []
        if orig == "\n":
            src_idx += 1
        if pron:
            if not current_line:
                cur_clause_spk = tuple(item.get(F.SPK, ())) if item.get(F.SPK) else ()
            current_line.append(pron)
            cur_ono_surface.append(_ono_token_surface(item))
            token_to_index_map[flat_idx] = i
            flat_idx += 1
        elif current_line and _is_voicing_mark(item):
            cur_ono_surface.append("゛")
    if current_line:
        alignment_tokens.append(current_line)
        line_group.append(src_idx)
        clause_speakers.append(cur_clause_spk)
        surf = "".join(cur_ono_surface)
        clause_surfaces.append(surf)
        ono_clause_flags.append(1 if _ieo(surf) else 0)

    n_tokens = sum(len(lt) for lt in alignment_tokens)
    if n_tokens == 0 or not alignment_tokens:
        raise ValueError(
            f"台本未解析出任何可对齐的发音音素 (tokens=0)，请检查台本是否为空或仅含标点/注释: {script_path}"
        )
    return {
        "result_list": result_list,
        "alignment_tokens": alignment_tokens,
        "line_group": line_group,
        "clause_surfaces": clause_surfaces,
        "ono_clause_flags": ono_clause_flags,
        "clause_speakers": clause_speakers,
        "token_to_index_map": token_to_index_map,
        "src_idx_to_lineno": src_idx_to_lineno,
        "src_idx_to_ann": src_idx_to_ann,
        "n_tokens": n_tokens,
    }


def _finalize_post_rl(
    base_rl: List[Dict[str, Any]],
    align_res: Sequence[Dict[str, Any]],
    token_to_index_map: Dict[int, int],
    stream_clause_lists: Sequence[Sequence[int]],
    enable_multi_order: bool,
    enable_ono_gate: bool,
    enable_resegment: bool,
    enable_merge_short: bool,
    merge_short_chars: int,
    merge_short_gap: float,
    merge_ono_chars: int,
    merge_ono_gap: float,
    log,
    side_label: str = "",
):
    """回填对齐时间戳并执行多流归并 / 标点重分段 / 短句合并 / 拟声守门。"""
    rl = copy.deepcopy(base_rl)
    for i, result in enumerate(align_res):
        if i in token_to_index_map and not result.get("skipped") and "start" in result and "end" in result:
            original_index = token_to_index_map[i]
            rl[original_index]["start"] = result["start"]
            rl[original_index]["end"] = result["end"]

    ind_rl = None
    if enable_multi_order and len(stream_clause_lists) > 1:
        from fa_asmr.aligners.multispeaker import post_process_and_merge_streams
        rl, ind_rl = post_process_and_merge_streams(
            result_list=rl,
            alignment_results=align_res,
            token_to_index_map=token_to_index_map,
            stream_clause_lists=stream_clause_lists,
            enable_ono_gate=enable_ono_gate,
            enable_resegment=enable_resegment,
            enable_merge_short=enable_merge_short,
            merge_short_chars=merge_short_chars,
            merge_short_gap=merge_short_gap,
            merge_ono_chars=merge_ono_chars,
            merge_ono_gap=merge_ono_gap,
            log_callback=log,
            return_independent_rl=True,
        )
    else:
        if enable_multi_order and enable_ono_gate:
            from fa_asmr.aligners.multispeaker import apply_skipped_ono_marks
            apply_skipped_ono_marks(rl, align_res, token_to_index_map)

        # ---- 拟声错位处理 · 阶段一【标记】----------------------------
        # 跑结构判据给可疑拟声子句打 `_ono_mark`，**一个 token 都不删**。
        if enable_ono_gate:
            from fa_asmr.text.ono_filter import mark_ono_candidates as _ono_mark
            _ono_mark(
                result_list=rl,
                alignment_results=align_res,
                token_to_index_map=token_to_index_map,
                log_callback=log,
            )

        # ---- 标点驱动重分段（候选=标点，证据=两侧 CTC blank 间隔；标点保留） ----
        if enable_resegment:
            from fa_asmr.text.resegment import apply_punctuation_resegment as _reseg
            rl_time = {}
            for k, r in enumerate(align_res):
                ri = token_to_index_map.get(k)
                if (
                    ri is not None
                    and not r.get("skipped")
                    and r.get("original_start") is not None
                    and r.get("original_end") is not None
                ):
                    rl_time[ri] = (float(r["original_start"]), float(r["original_end"]))
            n_before = sum(1 for it in rl if it.get("orig") == "\n") + 1
            rl = _reseg(rl, rl_time, frame_sec=0.02, thr_frames=50, max_line_chars=22)
            n_after = sum(1 for it in rl if it.get("orig") == "\n") + 1
            pfx = f"[{side_label}] " if side_label else ""
            log(f"  ✨ {pfx}标点重分段: {n_before} → {n_after} 行（标点保留）")

        # ---- 短句合并：把 resegment 切碎的相邻短句重新合并 ----
        if enable_merge_short:
            from fa_asmr.text.merge_short import merge_short_lines as _merge, count_inserted_breaks as _cb
            nb = _cb(rl)
            rl = _merge(
                rl,
                max_chars=merge_short_chars,
                max_gap=merge_short_gap,
                ono_max_chars=merge_ono_chars,
                ono_max_gap=merge_ono_gap,
            )
            na = _cb(rl)
            if nb != na:
                pfx = f"[{side_label}] " if side_label else ""
                log(f"  ✨ {pfx}短句合并: 合并 {nb - na} 处碎句")

        # ---- 拟声错位处理 · 阶段三【决策】----------------------------
        if enable_ono_gate:
            from fa_asmr.text.ono_filter import decide_ono_deletions as _ono_decide
            _ono_decide(result_list=rl, log_callback=log)

        for i in range(len(rl)):
            item = rl[i]
            if item.get(F.ONO_DROPPED):
                continue
            if "start" not in item and item.get("type", 0) in [1, 2, 3, 4, 5]:
                found = False
                for j in range(i - 1, -1, -1):
                    if rl[j].get("type") == 0 and rl[j].get("orig") == "\n":
                        break
                    if rl[j].get(F.ONO_DROPPED):
                        continue
                    if "start" in rl[j]:
                        item["start"] = rl[j]["start"]
                        item["end"] = rl[j]["end"]
                        found = True
                        break
                if not found:
                    for j in range(i + 1, len(rl)):
                        if rl[j].get("type") == 0 and rl[j].get("orig") == "\n":
                            break
                        if rl[j].get(F.ONO_DROPPED):
                            continue
                        if "start" in rl[j]:
                            item["start"] = rl[j]["start"]
                            item["end"] = rl[j]["end"]
                            break
    return rl, ind_rl


def _run_audit_stage(
    audio_path: str,
    audio_path_b: Optional[str],
    output_dir: Optional[str],
    enable_multi_order: bool,
    tok_data: Dict[str, Any],
    alignment_results: Any,
    align_diag: Dict[str, Any],
    mp_meta: Optional[Dict[str, Any]],
    audit_clips_dir: Optional[str],
    audit_separate: bool,
    audit_sep_sources: int,
    audit_clips_all: bool,
    audit_clips_compact: bool,
    summary_out: Optional[Dict[str, Any]],
    log,
) -> None:
    """生成对齐后声学体检报告并按需导出审听 / 空间分离 WAV 切片。"""
    from fa_asmr.pipeline.audit import (
        build_alignment_audit,
        build_multiperspective_audit,
        export_audit_clips,
    )

    line_group = tok_data["line_group"]
    src_idx_to_lineno = tok_data["src_idx_to_lineno"]
    src_idx_to_ann = tok_data["src_idx_to_ann"]
    clause_speakers = tok_data["clause_speakers"]

    cl_line_nos = [
        src_idx_to_lineno[lg] if (0 <= lg < len(src_idx_to_lineno)) else (lg + 1)
        for lg in line_group
    ]
    cl_ann = [
        src_idx_to_ann[lg] if (0 <= lg < len(src_idx_to_ann)) else ""
        for lg in line_group
    ]
    eff_clips_dir = (
        os.path.abspath(audit_clips_dir)
        if audit_clips_dir
        else (
            os.path.join(
                os.path.abspath(output_dir) if output_dir else os.path.dirname(os.path.abspath(audio_path)),
                "audit_clips",
            )
            if audit_separate
            else None
        )
    )
    uniq_spks = {
        s
        for sp in clause_speakers
        for s in (sp if isinstance(sp, (tuple, list)) else ((sp,) if sp else ()))
        if s
    }
    if int(audit_sep_sources) > 0:
        eff_sep_sources = max(2, min(4, int(audit_sep_sources)))
        eff_enable_separate = bool(audit_separate)
    else:
        eff_sep_sources = 3 if len(uniq_spks) >= 3 else 2
        # 单人轨（<=1 说话人且非多人模式）照常完整体检与切原声 WAV，但自动跳过立体声空间拆轨
        eff_enable_separate = bool(audit_separate) and (len(uniq_spks) >= 2 or bool(enable_multi_order))

    if audio_path_b and mp_meta is not None and "flat_B" in mp_meta:
        diag_a = {
            "ctc_nb_active": mp_meta.get("ctc_nb_active_A"),
            "stereo_wav_16k": mp_meta.get("stereo_wav_16k_A"),
            "duration_sec": float(mp_meta.get("dur_A_sec", 0.0)),
            "frame_sec": float(mp_meta.get("frame_sec", 0.02)),
        }
        diag_b = {
            "ctc_nb_active": mp_meta.get("ctc_nb_active_B"),
            "stereo_wav_16k": mp_meta.get("stereo_wav_16k_B"),
            "duration_sec": float(mp_meta.get("dur_B_sec", 0.0)),
            "frame_sec": float(mp_meta.get("frame_sec", 0.02)),
        }
        audit_report = build_multiperspective_audit(
            alignment_tokens=tok_data["alignment_tokens"],
            alignment_results_a=mp_meta.get("flat_A", alignment_results),
            alignment_results_b=mp_meta["flat_B"],
            ono_clause_flags=tok_data["ono_clause_flags"],
            clause_surfaces=tok_data["clause_surfaces"],
            clause_speakers=clause_speakers,
            clause_line_nos=cl_line_nos,
            clause_annotated=cl_ann,
            diagnostics_a=diag_a,
            diagnostics_b=diag_b,
            delta_sec=float(mp_meta.get("delta_sec", 0.0)),
            sim_segs=mp_meta.get("sim_segs"),
        )
    else:
        audit_report = build_alignment_audit(
            alignment_tokens=tok_data["alignment_tokens"],
            alignment_results=alignment_results,
            ono_clause_flags=tok_data["ono_clause_flags"],
            clause_surfaces=tok_data["clause_surfaces"],
            clause_speakers=clause_speakers,
            clause_line_nos=cl_line_nos,
            clause_annotated=cl_ann,
            diagnostics=align_diag,
        )

    if eff_clips_dir:
        n_clips = export_audit_clips(
            audio_path,
            audit_report,
            eff_clips_dir,
            enable_separate=eff_enable_separate,
            sep_sources=eff_sep_sources,
            audio_path_b=audio_path_b,
            priority_only=not bool(audit_clips_all),
            compact_clips=bool(audit_clips_compact),
        )
        audit_report["exported_clips_count"] = n_clips
        audit_report["clips_dir"] = eff_clips_dir
    if audit_separate and eff_clips_dir:
        if eff_enable_separate:
            sep_msg = f" | 空间分离触发={audit_report.get('separated_clips_count', 0)}/{audit_report.get('exported_clips_count', 0)} (K={eff_sep_sources})"
        else:
            sep_msg = " | 空间分离=单人轨已跳过"
    else:
        sep_msg = ""
    mp_tag = "·双视角联合" if audit_report.get("is_multiperspective") else ""
    log(
        f"  🔎 [声学体检{mp_tag}] 基线中位分={audit_report['track_median_score']:.2f} | "
        f"高危必审={audit_report.get('priority_review_count', 0)} | "
        f"可疑实词句={audit_report['suspect_clauses_count']} | "
        f"未覆盖活跃语音空档={audit_report['unaligned_speech_gaps_count']}{sep_msg}"
    )
    if summary_out is not None:
        summary_out["audit"] = audit_report


def run_pipeline(
    script_path: str,
    audio_path: str,
    lora_path: Optional[str],
    log_callback,
    progress_callback,
    dev="auto",
    vram_chunk=3000,
    persist_gpu=False,
    line_end_extend=0.0,
    enable_silence_snap=True,
    enable_resegment=True,
    enable_ono_gate=True,
    enable_merge_short=True,
    merge_short_chars=20,
    merge_short_gap=1.5,
    merge_ono_chars=40,
    merge_ono_gap=2.5,
    enable_multi_order=False,
    audio_path_b: Optional[str] = None,
    output_dir: Optional[str] = None,
    summary_out: Optional[Dict[str, Any]] = None,
    echo_stdout: bool = True,
    ruby_map: Optional[Any] = None,
    enable_audit: bool = False,
    audit_clips_dir: Optional[str] = None,
    audit_separate: bool = False,
    audit_sep_sources: int = 0,
    audit_clips_all: bool = False,
    audit_clips_compact: bool = False,
):
    """V2 核心对齐管线执行入口 (默认常开静音吸附与拟声守门；传入 audio_path_b 时启用多视角联合对齐)"""
    _ensure_runtime_ready()
    import fa_asmr.aligners.align as align
    import fa_asmr.text.haruraw2norm as hn

    if audio_path_b:
        enable_multi_order = True

    def log(msg):
        if log_callback:
            log_callback(msg)
        if echo_stdout:
            try:
                print(msg)
            except Exception:
                pass

    progress_callback and progress_callback(0, "🌸 初始化台本与环境...")
    total_start = time.time()
    _ruby = None
    warnings_list: list[str] = []

    def warn(msg: str):
        warnings_list.append(msg)
        log(f"  ⚠ {msg}")

    try:
        _norm_ruby_map = _validate_pipeline_inputs(
            script_path=script_path,
            audio_path=audio_path,
            audio_path_b=audio_path_b,
            output_dir=output_dir,
            ruby_map=ruby_map,
            line_end_extend=line_end_extend,
            vram_chunk=vram_chunk,
        )

        import fa_asmr.core.advanced_config as _adv
        _adv.load()
        for _adv_w in list(getattr(_adv, "LAST_WARNINGS", [])):
            warn(f"[advanced_config] {_adv_w}")

        progress_callback and progress_callback(5, "🌸 读取台本并注音...")
        log("[1/4] 🌸 读取台本并进行日文注音转换...")
        raw_text, detected_enc = read_text_auto_encoding(script_path)
        if detected_enc.endswith("-fallback") or "\ufffd" in raw_text:
            bad_cnt = raw_text.count("\ufffd")
            warn(
                f"台本编码异常（识别为 {detected_enc}，含 {bad_cnt} 处无法解码的替换字符 ），"
                "建议先用 UTF-8 / Shift-JIS 重新保存台本，否则可能导致对齐错位。"
            )
        elif detected_enc not in ("utf-8", "utf-8-sig"):
            log(f"  ✨ 自动识别并兼容台本编码: {detected_enc.upper()} → UTF-8")

        ruby = AutoRuby()
        _ruby = ruby
        _ann_inst = getattr(ruby, "_ann", None)
        if _ann_inst is not None:
            if getattr(_ann_inst, "fugashi_error", None):
                warn(
                    f"fugashi/UniDic 未就绪 ({_ann_inst.fugashi_error})，"
                    "当前已降级为 pykakasi 逐字注音（注音精度会下降）"
                )
            if getattr(_ann_inst, "_OVERRIDES_WARNING", None):
                warn(str(_ann_inst._OVERRIDES_WARNING))

        lines = raw_text.splitlines()
        if enable_multi_order:
            lines, _auto_spks = auto_normalize_speaker_headers(lines)
            if _auto_spks:
                log("  ✨ 自动识别独立角色标题: " + " / ".join(f"〖{s}〗" for s in _auto_spks))

        # 说话人标记在【注音前】剥离：〖名字〗整段不进入 annotate / tokenize
        _spk_of_line: List[tuple] = []
        _body_lines: List[str] = []
        _src_line_nos: List[int] = []
        _cur_spk: tuple = ()
        for _raw_ln_no, _raw in enumerate(lines, 1):
            _segs, _cur_spk = _split_speaker_segments(_raw, _cur_spk)
            for _spk, _body in _segs:
                _body_lines.append(_body)
                _spk_of_line.append(_spk)
                _src_line_nos.append(_raw_ln_no)
        if not any(b.strip() for b in _body_lines):
            raise ValueError(f"台本不包含任何有效文本行: {script_path}")

        # 逐行清洗「内联振假名泄露」后再注音（同时作用于 _body_lines 供 _src 还原）
        if _ann_inst is not None and getattr(_ann_inst, "_tagger", None):
            from fa_asmr.text.inline_leak_clean import InlineLeakCleaner
            _leak_cleaner = InlineLeakCleaner(_ann_inst)
            _n_leak = 0
            for _k, _b in enumerate(_body_lines):
                _c = _leak_cleaner.clean(_b)
                if _c != _b:
                    _body_lines[_k] = _c
                    _n_leak += 1
            if _n_leak:
                log(f"  ✨ 内联振假名清洗: 修正 {_n_leak} 行")

        annotated_lines, _ruby_override_hits = annotate_lines_with_overrides(
            ruby, _body_lines, src_line_nos=_src_line_nos, ruby_map=_norm_ruby_map, script_path=script_path
        )
        if _norm_ruby_map is not None:
            log(f"  ✨ 注音覆写表已应用: 命中 {_ruby_override_hits} 处")
        annotated_text = "\n".join(annotated_lines)
        lines = _body_lines
        log(f"  ✨ 台本处理完成 ({len(lines)} 行)")

        # 及时释放 yomikata dBert 显存（步骤 2~4 的声学对齐不再需要 dBert，同时保留 UniDic 单例供批处理复用）
        if _ann_inst is not None:
            try:
                _ann_inst.release_gpu()
            except Exception as _e:
                warn(f"释放 yomikata dBert 显存失败: {type(_e).__name__}: {_e}")

        progress_callback and progress_callback(15, "🎵 解析音素与规范化...")
        log("[2/4] 🎵 解析音素与规范化...")
        tok_data = _tokenize_for_alignment(
            hn, annotated_text, lines, _spk_of_line, _src_line_nos, script_path
        )
        result_list = tok_data["result_list"]
        alignment_tokens = tok_data["alignment_tokens"]
        line_group = tok_data["line_group"]
        clause_surfaces = tok_data["clause_surfaces"]
        ono_clause_flags = tok_data["ono_clause_flags"]
        clause_speakers = tok_data["clause_speakers"]
        token_to_index_map = tok_data["token_to_index_map"]
        n_tokens = tok_data["n_tokens"]
        log(f"  ✨ 解析完成: {n_tokens} tokens, {len(alignment_tokens)} 行")

        progress_callback and progress_callback(25, "⚡ 双声道对齐计算中...")
        log("[3/4] 🎧 MMS-FA + LoRA 双声道 Better-Ear 全局 CTC 对齐 (静音吸附已启用)...")
        lora_path = _resolve_lora_path(align, lora_path, log, warn)

        # 标点位置 Blank 豁免掩码：当前恒为全 False（在 v9e10 下关闭后 MAE 更优）
        blank_punct_mask = [False] * (n_tokens + 1)

        _align_diag: Dict[str, Any] = {}
        _stream_clause_lists = [list(range(len(alignment_tokens)))]
        _mp_meta: Optional[Dict[str, Any]] = None
        if audio_path_b:
            from fa_asmr.aligners.multiperspective import (
                align_multiperspective_joint,
                project_result_list_to_perspective,
            )
            alignment_results, _stream_clause_lists, _mp_meta = align_multiperspective_joint(
                audio_path_A=audio_path,
                audio_path_B=audio_path_b,
                alignment_tokens=alignment_tokens,
                line_group=line_group,
                ono_clause_flags=ono_clause_flags,
                clause_speakers=clause_speakers,
                lora_path=lora_path,
                dev=dev,
                vram_chunk=int(vram_chunk),
                enable_silence_snap=enable_silence_snap,
                enable_ono_gate=enable_ono_gate,
                log_callback=log,
                clause_surfaces=clause_surfaces,
            )
        elif enable_multi_order:
            from fa_asmr.aligners.multispeaker import align_multispeaker_or_reordered
            alignment_results, _stream_clause_lists = align_multispeaker_or_reordered(
                audio_path=audio_path,
                alignment_tokens=alignment_tokens,
                line_group=line_group,
                ono_clause_flags=ono_clause_flags,
                clause_speakers=clause_speakers,
                lora_path=lora_path,
                dev=dev,
                vram_chunk=int(vram_chunk),
                enable_silence_snap=enable_silence_snap,
                enable_ono_gate=enable_ono_gate,
                blank_punct_mask=blank_punct_mask,
                log_callback=log,
                diagnostics_out=_align_diag,
                clause_surfaces=clause_surfaces,
            )
        else:
            alignment_results = align.align_audio_with_text(
                audio_file_path=audio_path,
                text_tokens=alignment_tokens,
                lora_path=lora_path,
                device=dev,
                chunk_frames=int(vram_chunk),
                enable_silence_snap=enable_silence_snap,
                blank_punct_mask=blank_punct_mask,
                line_group=line_group,
                ono_clause_flags=ono_clause_flags,
                log_callback=log,
                diagnostics_out=_align_diag,
            )

        if not alignment_results:
            if not persist_gpu:
                _release_gpu(_ruby)
            return False, None, "对齐返回空结果"

        progress_callback and progress_callback(80, "✂ 断句、短句合并与拟声守门...")
        log("[4/4] 📜 组装标准 LRC 与 SRT 字幕...")
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        def _post(base_rl, align_res, side_label: str = ""):
            return _finalize_post_rl(
                base_rl=base_rl,
                align_res=align_res,
                token_to_index_map=token_to_index_map,
                stream_clause_lists=_stream_clause_lists,
                enable_multi_order=enable_multi_order,
                enable_ono_gate=enable_ono_gate,
                enable_resegment=enable_resegment,
                enable_merge_short=enable_merge_short,
                merge_short_chars=merge_short_chars,
                merge_short_gap=merge_short_gap,
                merge_ono_chars=merge_ono_chars,
                merge_ono_gap=merge_ono_gap,
                log=log,
                side_label=side_label,
            )

        if audio_path_b and _mp_meta is not None:
            from fa_asmr.aligners.multiperspective import project_result_list_to_perspective
            _spk_pref = _mp_meta.get("spk_persp_pref")

            if "flat_A" in _mp_meta and "flat_B" in _mp_meta:
                _rl_lrc_A, _rl_ind_A = _post(result_list, _mp_meta["flat_A"], side_label="视角A")
                _rl_lrc_B, _rl_ind_B = _post(result_list, _mp_meta["flat_B"], side_label="视角B")
                _target_ind_A = _rl_ind_A if _rl_ind_A is not None else _rl_lrc_A
                _target_ind_B = _rl_ind_B if _rl_ind_B is not None else _rl_lrc_B
            else:
                _rl_lrc_A, _rl_ind_raw = _post(result_list, alignment_results)
                _rl_lrc_B = _rl_lrc_A
                _target_ind_A = _rl_ind_raw if _rl_ind_raw is not None else _rl_lrc_A
                _target_ind_B = _target_ind_A

            rl_A = project_result_list_to_perspective(
                _rl_lrc_A, _mp_meta["u_A_st_sec"], _mp_meta["dur_A_sec"],
                perspective_side="A", spk_persp_pref=_spk_pref,
            )
            rl_B = project_result_list_to_perspective(
                _rl_lrc_B, _mp_meta["u_B_st_sec"], _mp_meta["dur_B_sec"],
                perspective_side="B", spk_persp_pref=_spk_pref,
            )
            lrc_A = build_standard_lrc(rl_A)
            lrc_B = build_standard_lrc(rl_B)
            if line_end_extend > 0:
                from lrc_end_extend import extend_lrc
                lrc_A, _ = extend_lrc(lrc_A, line_end_extend)
                lrc_B, _ = extend_lrc(lrc_B, line_end_extend)

            rl_indep_A = project_result_list_to_perspective(
                _target_ind_A, _mp_meta["u_A_st_sec"], _mp_meta["dur_A_sec"],
                perspective_side="A", spk_persp_pref=_spk_pref,
            )
            rl_indep_B = project_result_list_to_perspective(
                _target_ind_B, _mp_meta["u_B_st_sec"], _mp_meta["dur_B_sec"],
                perspective_side="B", spk_persp_pref=_spk_pref,
            )
            srt_A = build_standard_srt(rl_indep_A, line_end_extend=line_end_extend)
            srt_B = build_standard_srt(rl_indep_B, line_end_extend=line_end_extend)

            base_A = os.path.splitext(os.path.basename(audio_path))[0]
            base_B = os.path.splitext(os.path.basename(audio_path_b))[0]
            dir_A = os.path.abspath(output_dir) if output_dir else os.path.dirname(os.path.abspath(audio_path))
            dir_B = os.path.abspath(output_dir) if output_dir else os.path.dirname(os.path.abspath(audio_path_b))
            lrc_path = os.path.join(dir_A, f"{base_A}.lrc")
            lrc_path_b = os.path.join(dir_B, f"{base_B}.lrc")
            srt_path = os.path.join(dir_A, f"{base_A}.srt")
            srt_path_b = os.path.join(dir_B, f"{base_B}.srt")

            if os.path.normcase(os.path.abspath(lrc_path)) == os.path.normcase(os.path.abspath(lrc_path_b)):
                raise ValueError(
                    f"双视角输出路径冲突（视角 A 与视角 B 将覆盖写入同一文件: {lrc_path}），"
                    "请区分两轨音频文件名或不要使用同一 --output-dir。"
                )

            if not lrc_A.strip() and not lrc_B.strip() and not srt_A.strip() and not srt_B.strip():
                raise RuntimeError("双视角对齐后未生成任何有效字幕内容（视角 A/B 均为空），请检查音频与台本是否匹配")
            if not lrc_A.strip() and not srt_A.strip():
                warn("视角 A 未生成有效字幕内容，请检查视角 A 音频与台本时间范围")
            if not lrc_B.strip() and not srt_B.strip():
                warn("视角 B 未生成有效字幕内容，请检查视角 B 音频与台本时间范围")

            with open(lrc_path, "w", encoding="utf-8") as f:
                f.write(lrc_A)
            with open(lrc_path_b, "w", encoding="utf-8") as f:
                f.write(lrc_B)
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(srt_A)
            with open(srt_path_b, "w", encoding="utf-8") as f:
                f.write(srt_B)

            total_elapsed = time.time() - total_start
            n_lines_a = len(lrc_A.splitlines())
            n_lines_b = len(lrc_B.splitlines())
            n_cues_a = srt_A.count(" --> ")
            n_cues_b = srt_B.count(" --> ")
            log(f"\n  💖 多视角联合对齐完成! Δt = {_mp_meta['delta_sec']:+.2f}s | 总耗时 {total_elapsed:.1f}s")
            log(f"  🎯 视角 A: {lrc_path} ({n_lines_a} 行) | {srt_path} ({n_cues_a} 条独立字幕)")
            log(f"  🎯 视角 B: {lrc_path_b} ({n_lines_b} 行) | {srt_path_b} ({n_cues_b} 条独立字幕)")
            if summary_out is not None:
                summary_out.update({
                    "mode": "multi_persp",
                    "detected_encoding": detected_enc,
                    "script_lines": len(lines),
                    "tokens": n_tokens,
                    "clauses": len(alignment_tokens),
                    "delta_sec": float(_mp_meta["delta_sec"]),
                    "elapsed_sec": round(total_elapsed, 2),
                    "outputs": {
                        "lrc": lrc_path,
                        "srt": srt_path,
                        "lrc_b": lrc_path_b,
                        "srt_b": srt_path_b,
                    },
                    "counts": {
                        "lrc_lines": n_lines_a,
                        "srt_cues": n_cues_a,
                        "lrc_b_lines": n_lines_b,
                        "srt_b_cues": n_cues_b,
                    },
                    "warnings": list(warnings_list),
                })
                if _norm_ruby_map is not None:
                    summary_out["ruby_overrides_applied"] = _ruby_override_hits
        else:
            result_list, independent_rl = _post(result_list, alignment_results)
            target_rl_for_srt = independent_rl if independent_rl is not None else result_list
            lrc_content = build_standard_lrc(result_list)
            if line_end_extend > 0:
                from lrc_end_extend import extend_lrc
                lrc_content, _n_ext = extend_lrc(lrc_content, line_end_extend)
                if _n_ext:
                    log(f"  ✨ LRC 行尾延长 ({line_end_extend:.1f}s): 处理 {_n_ext} 个空白标记")

            srt_content = build_standard_srt(target_rl_for_srt, line_end_extend=line_end_extend)

            base = os.path.splitext(os.path.basename(audio_path))[0]
            out_dir = os.path.abspath(output_dir) if output_dir else os.path.dirname(os.path.abspath(audio_path))
            lrc_path = os.path.join(out_dir, f"{base}.lrc")
            srt_path = os.path.join(out_dir, f"{base}.srt")

            if not lrc_content.strip() and not srt_content.strip():
                raise RuntimeError("对齐后未生成任何有效字幕内容（LRC 与 SRT 均为空），请检查音频与台本是否匹配")

            with open(lrc_path, "w", encoding="utf-8") as f:
                f.write(lrc_content)
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(srt_content)

            total_elapsed = time.time() - total_start
            n_lines_lrc = len(lrc_content.splitlines())
            n_cues_srt = srt_content.count(" --> ")
            log(f"\n  💖 对齐完成! 总耗时 {total_elapsed:.1f}s")
            log(f"  🎯 输出 LRC: {lrc_path} ({n_lines_lrc} 行)")
            log(f"  🎯 输出 SRT: {srt_path} ({n_cues_srt} 条独立字幕)")
            if summary_out is not None:
                summary_out.update({
                    "mode": "multi_speaker" if enable_multi_order else "single",
                    "detected_encoding": detected_enc,
                    "script_lines": len(lines),
                    "tokens": n_tokens,
                    "clauses": len(alignment_tokens),
                    "elapsed_sec": round(total_elapsed, 2),
                    "outputs": {
                        "lrc": lrc_path,
                        "srt": srt_path,
                    },
                    "counts": {
                        "lrc_lines": n_lines_lrc,
                        "srt_cues": n_cues_srt,
                    },
                    "warnings": list(warnings_list),
                })
                if _norm_ruby_map is not None:
                    summary_out["ruby_overrides_applied"] = _ruby_override_hits

        if enable_audit or audit_clips_dir or audit_separate:
            progress_callback and progress_callback(92, "🔎 声学体检与审听切片...")
            _run_audit_stage(
                audio_path=audio_path,
                audio_path_b=audio_path_b,
                output_dir=output_dir,
                enable_multi_order=enable_multi_order,
                tok_data=tok_data,
                alignment_results=alignment_results,
                align_diag=_align_diag,
                mp_meta=_mp_meta,
                audit_clips_dir=audit_clips_dir,
                audit_separate=audit_separate,
                audit_sep_sources=audit_sep_sources,
                audit_clips_all=audit_clips_all,
                audit_clips_compact=audit_clips_compact,
                summary_out=summary_out,
                log=log,
            )

        progress_callback and progress_callback(100, "💖 全部完成!")
        if not persist_gpu:
            _release_gpu(_ruby)
        return True, lrc_path, None

    except Exception as e:
        import traceback
        err = traceback.format_exc()
        err_msg = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
        log(f"\n  💔 发生错误 ({err_msg}):\n{err}")
        if summary_out is not None:
            summary_out.update({
                "error_type": type(e).__name__,
                "warnings": list(warnings_list),
            })
        if not persist_gpu:
            _release_gpu(_ruby)
        return False, None, err_msg
