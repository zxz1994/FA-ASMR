# -*- coding: utf-8 -*-
"""FA-ASMR V2 — 多视角联合对齐主编排管线与视角时间轴投影导出。"""
from __future__ import annotations

import copy
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from fa_asmr.aligners.align_acoustic import FRAME_SEC
from fa_asmr.aligners.align_utils import _format_time
from fa_asmr.text.clause import F
from fa_asmr.aligners.multipersp_acoustic import (
    _AudioSoloChecker,
    _get_tokenizer_dict,
    _load_audio_16k_once,
    _norm_spk,
    _prep_phoneme_probs,
    compute_phoneme_advantage_matrix,
    detect_clean_audio_segments,
    ensure_numba_warmed_up,
)
from fa_asmr.aligners.multipersp_segments import (
    _build_fg_masks,
    align_away_speaker_by_dt,
    refine_home_segments,
    segment_and_assign_home,
)
from fa_asmr.aligners.multispeaker_acoustic import (
    _align_subset_clauses,
    _clause_summary,
    _group_results_by_clause,
    build_cooperative_stereo_emission,
    extract_acoustic_context,
)
from fa_asmr.aligners.multispeaker_schedule import align_multispeaker_or_reordered
from fa_asmr.pipeline.subtitles import _parse_lrc_ts_sec

# ══════════════════════════════════════════════════════════════════════════
#  5. 主编排管线与视角投影导出 (Pipeline Orchestration & Projection)
# ══════════════════════════════════════════════════════════════════════════

def run_multiperspective_pipeline(
    alignment_tokens: List[List[str]],
    line_group: Sequence[int],
    ono_clause_flags: Sequence[int],
    clause_speakers: Sequence[Any],
    ctx_A: Dict[str, Any],
    ctx_B: Dict[str, Any],
    enable_silence_snap: bool = True,
    enable_ono_gate: bool = True,
    log_callback: Optional[Callable[[str], None]] = None,
    enable_word_skip: bool = False,
    word_skip_token_penalty: float = 4.8,
    clause_surfaces: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[List[int]], Dict[str, Any]]:
    """执行四步多视角联合对齐，返回原生双视角 flat_A / flat_B 及其元信息。"""
    def log(msg: str) -> None:
        if log_callback:
            log_callback(msg)

    t_pipe0 = time.time()
    n_c = len(alignment_tokens)
    if n_c == 0:
        return [], [], {
            "flat_A": [], "flat_B": [], "dur_A_sec": 0.0, "dur_B_sec": 0.0,
            "u_A_st_sec": 0.0, "u_B_st_sec": 0.0, "delta_sec": 0.0,
            "sim_segs": [], "spk_persp_pref": {}, "spk_stats": {},
        }

    if clause_surfaces is None:
        clause_surfaces = ["".join(toks) for toks in alignment_tokens]

    tok_dict = _get_tokenizer_dict(ctx_A)
    clause_tid_seqs: Dict[int, List[int]] = {
        ci: [int(tok_dict[ch]) for t in alignment_tokens[ci] for ch in t if ch in tok_dict]
        for ci in range(n_c)
    }

    spk_to_clauses: Dict[str, List[int]] = {}
    norm_speakers: List[str] = []
    for ci in range(n_c):
        sp = _norm_spk(clause_speakers[ci] if ci < len(clause_speakers) else None)
        norm_speakers.append(sp)
        spk_to_clauses.setdefault(sp, []).append(ci)

    coop_A_t = build_cooperative_stereo_emission(ctx_A, ear_bias=0.0)
    coop_B_t = build_cooperative_stereo_emission(ctx_B, ear_bias=0.0)
    coop_A_np = coop_A_t.detach().cpu().numpy()
    coop_B_np = coop_B_t.detach().cpu().numpy()

    # ── 预热本地 Numba 核（确保多线程并发前已完成首次 JIT 加载，避免编译锁竞争）──
    ensure_numba_warmed_up(coop_A_np.shape[1])

    # ── 初始全角色探针（A/B 双视角 nogil 双线程并行解码，100% 逐比特无损）──
    probe_A: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}
    probe_B: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}
    if len(spk_to_clauses) >= 2:
        n_tok = sum(len(t) for t in alignment_tokens)

        def _probe_multi_side(ctx_side: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
            flat_init, _ = align_multispeaker_or_reordered(
                audio_path="",
                alignment_tokens=alignment_tokens,
                line_group=list(line_group),
                ono_clause_flags=list(ono_clause_flags),
                clause_speakers=list(clause_speakers),
                lora_path=None,
                dev="auto",
                vram_chunk=3000,
                enable_silence_snap=enable_silence_snap,
                enable_ono_gate=enable_ono_gate,
                blank_punct_mask=[False] * (n_tok + 1),
                log_callback=None,
                enable_word_skip=enable_word_skip,
                word_skip_token_penalty=word_skip_token_penalty,
                acoustic_ctx=ctx_side,
                clause_surfaces=clause_surfaces,
            )
            return _group_results_by_clause(alignment_tokens, flat_init)

        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_A = ex.submit(_probe_multi_side, ctx_A)
            fut_B = ex.submit(_probe_multi_side, ctx_B)
            grp_A = fut_A.result()
            grp_B = fut_B.result()
        for sp, cl in spk_to_clauses.items():
            probe_A[sp] = {ci: grp_A[ci] for ci in cl}
            probe_B[sp] = {ci: grp_B[ci] for ci in cl}
    else:
        for sp, cl in spk_to_clauses.items():
            def _probe_single_side(ctx_side: Dict[str, Any], coop_t: torch.Tensor) -> Dict[int, List[Dict[str, Any]]]:
                return _align_subset_clauses(
                    sub_indices=cl,
                    alignment_tokens=alignment_tokens,
                    line_group=list(line_group),
                    ono_clause_flags=list(ono_clause_flags),
                    enable_silence_snap=enable_silence_snap,
                    acoustic_ctx=ctx_side,
                    custom_emission_2d=coop_t,
                    enable_ono_skip=enable_ono_gate,
                    enable_word_skip=enable_word_skip,
                    word_skip_token_penalty=word_skip_token_penalty,
                    clause_surfaces=clause_surfaces,
                    clause_speakers=list(clause_speakers),
                )

            with ThreadPoolExecutor(max_workers=2) as ex:
                fut_A = ex.submit(_probe_single_side, ctx_A, coop_A_t)
                fut_B = ex.submit(_probe_single_side, ctx_B, coop_B_t)
                probe_A[sp] = fut_A.result()
                probe_B[sp] = fut_B.result()

    spk_stats: Dict[str, Dict[str, Any]] = {}
    for sp, cl in spk_to_clauses.items():
        scA = [_clause_summary(probe_A[sp][ci])["score"] for ci in cl if ono_clause_flags[ci] == 0 and not _clause_summary(probe_A[sp][ci])["skipped"]]
        scB = [_clause_summary(probe_B[sp][ci])["score"] for ci in cl if ono_clause_flags[ci] == 0 and not _clause_summary(probe_B[sp][ci])["skipped"]]
        mA = float(np.median(scA)) if scA else -10.0
        mB = float(np.median(scB)) if scB else -10.0
        spk_stats[sp] = {"sc_A": mA, "sc_B": mB, "home": "A" if mA >= mB else "B"}

    adv_A_np = compute_phoneme_advantage_matrix(coop_A_np, ctx_A)
    adv_B_np = compute_phoneme_advantage_matrix(coop_B_np, ctx_B)
    norm_p_A, act_A = _prep_phoneme_probs(coop_A_np, ctx_A)
    norm_p_B, act_B = _prep_phoneme_probs(coop_B_np, ctx_B)
    em_AL = ctx_A["emission_L"][0].detach().cpu().numpy() if "emission_L" in ctx_A else None
    em_AR = ctx_A["emission_R"][0].detach().cpu().numpy() if "emission_R" in ctx_A else None
    em_BL = ctx_B["emission_L"][0].detach().cpu().numpy() if "emission_L" in ctx_B else None
    em_BR = ctx_B["emission_R"][0].detach().cpu().numpy() if "emission_R" in ctx_B else None
    ch_A = [coop_A_np] + [x for x in (em_AL, em_AR) if x is not None]
    ch_B = [coop_B_np] + [x for x in (em_BL, em_BR) if x is not None]
    acts_A = [_prep_phoneme_probs(x, ctx_A) for x in ch_A]
    acts_B = [_prep_phoneme_probs(x, ctx_B) for x in ch_B]

    # ── Step 1: 纯声学相似段 SimSeg 检测 ───────────────────────────────────
    sim_segs = detect_clean_audio_segments(ctx_A, ctx_B)
    if sim_segs:
        seg_desc = ", ".join(
            f"{s['dt_AB']:+.2f}s(A={s['A_st']:.0f}..{s['A_en']:.0f},B={s['B_st']:.0f}..{s['B_en']:.0f})"
            for s in sim_segs[:8]
        )
        log(f"  🔗 [Step 1 声学同步段] {len(sim_segs)} 段: {seg_desc}{' ...' if len(sim_segs) > 8 else ''}")
    else:
        log("  🔗 [Step 1 声学同步段] 未检测到高相似同步段 (各自独立视角处理)")

    # ── Step 2: 角色分块与主视角判定 ───────────────────────────────────────
    spk_sides: Dict[str, List[Optional[str]]] = {}
    for spk, cl in spk_to_clauses.items():
        sides, segs = segment_and_assign_home(
            spk, cl, probe_A[spk], probe_B[spk],
            alignment_tokens, ono_clause_flags, clause_tid_seqs,
            norm_p_A, act_A, norm_p_B, act_B, coop_A_np, coop_B_np, sim_segs,
        )
        spk_sides[spk] = sides
        chunk_desc = ", ".join(
            f"[{s['k_start']}..{s['k_end']}]({s['seg_type']})->{s['home']}"
            for s in segs
        )
        log(f"  🎭 [Step 2 主视角切块] [{spk}]: {chunk_desc}")

    # ── Step 3: Home 全量采纳与纯 STFT 单声源碰撞局部修复 (A/B 双线程并行) ───
    refined_probes: Dict[str, Dict[str, Dict[int, List[Dict[str, Any]]]]] = {}
    side_meta: Dict[str, Dict[str, Any]] = {}

    def _step3_one_side(target_side: str) -> Tuple[str, Dict[str, Dict[int, List[Dict[str, Any]]]], Dict[str, Any]]:
        ctx_t = ctx_A if target_side == "A" else ctx_B
        coop_t_t = coop_A_t if target_side == "A" else coop_B_t
        coop_t_np = coop_A_np if target_side == "A" else coop_B_np
        ch_t = ch_A if target_side == "A" else ch_B
        adv_t = adv_A_np if target_side == "A" else adv_B_np
        acts_t = acts_A if target_side == "A" else acts_B
        probe_t = probe_A if target_side == "A" else probe_B
        probe_h_other = probe_B if target_side == "A" else probe_A
        np_h_oth, act_h_oth = (norm_p_B, act_B) if target_side == "A" else (norm_p_A, act_A)
        T_t, C_t = coop_t_np.shape

        home_cls = {s: [cl[k] for k, sd in enumerate(spk_sides[s]) if sd in (target_side, "SHARED")] for s, cl in spk_to_clauses.items()}
        away_cls = {s: [cl[k] for k, sd in enumerate(spk_sides[s]) if sd not in (target_side, "SHARED")] for s, cl in spk_to_clauses.items()}
        active_home = [s for s, cl in home_cls.items() if cl]
        away_spks = [s for s, cl in away_cls.items() if cl]

        fg0, fgc0, _ = _build_fg_masks(home_cls, probe_t, T_t, C_t, min_score=-2.05, tok_dict=tok_dict)
        ref_spans = [
            (_clause_summary(probe_t[s][ci])["st"], _clause_summary(probe_t[s][ci])["en"])
            for s in active_home for ci in home_cls[s]
            if ono_clause_flags[ci] == 0 and not _clause_summary(probe_t[s][ci])["skipped"]
        ]
        solo_chk = _AudioSoloChecker(ctx_t.get("_stereo_wav"), ref_spans)

        away_hits_p1: List[Tuple[float, float, float]] = []
        away_eval_cache: Dict[Tuple[Any, ...], Tuple[Optional[Dict[str, Any]], List[Tuple[int, int]]]] = {}
        for s in away_spks:
            _, hits_s = align_away_speaker_by_dt(
                away_cls[s], target_side, probe_h_other[s], sim_segs,
                ctx_t, ch_t, adv_t, np_h_oth, act_h_oth, acts_t, fg0, fgc0,
                alignment_tokens, ono_clause_flags, clause_surfaces, clause_tid_seqs,
                is_final_pass=False, eval_cache=away_eval_cache,
            )
            away_hits_p1.extend(hits_s)

        ref_probe = refine_home_segments(
            active_home, home_cls, spk_to_clauses, probe_t, away_hits_p1,
            alignment_tokens, line_group, ono_clause_flags, clause_surfaces, clause_speakers,
            ctx_t, coop_t_t, solo_chk,
        )
        fg_full, fgc_full, _ = _build_fg_masks(home_cls, ref_probe, T_t, C_t, min_score=-2.35, tok_dict=tok_dict)
        fg_diff_cs = np.concatenate([[0], np.cumsum(((fg0 != fg_full) | np.any(fgc0 != fgc_full, axis=1)).astype(np.int32))])
        meta_s = {
            "home_cls": home_cls, "away_cls": away_cls, "active_home": active_home,
            "away_spks": away_spks, "fg": fg_full, "fgc": fgc_full,
            "eval_cache": away_eval_cache, "fg_diff_cs": fg_diff_cs,
        }
        return target_side, ref_probe, meta_s

    with ThreadPoolExecutor(max_workers=2) as ex:
        for t_side, ref_p, m_s in ex.map(_step3_one_side, ("A", "B")):
            refined_probes[t_side] = ref_p
            side_meta[t_side] = m_s

    # ── Step 4: 客视角局部 Δt 对齐与原生双视角输出组装 (A/B 双线程并行) ───────
    streams = list(spk_to_clauses.values())
    spk_persp_pref = {s: float(spk_stats[s]["sc_B"] - spk_stats[s]["sc_A"]) for s in spk_to_clauses}
    flat_by_side: Dict[str, List[Dict[str, Any]]] = {}

    def _step4_one_side(target_side: str) -> Tuple[str, List[Dict[str, Any]], List[str]]:
        opp_side = "B" if target_side == "A" else "A"
        meta = side_meta[target_side]
        ctx_t = ctx_A if target_side == "A" else ctx_B
        ch_t = ch_A if target_side == "A" else ch_B
        adv_t = adv_A_np if target_side == "A" else adv_B_np
        acts_t = acts_A if target_side == "A" else acts_B
        np_h_oth, act_h_oth = (norm_p_B, act_B) if target_side == "A" else (norm_p_A, act_A)
        side_logs: List[str] = []

        final_map: Dict[int, List[Dict[str, Any]]] = {}
        for s in meta["active_home"]:
            for ci in meta["home_cls"][s]:
                toks_h = copy.deepcopy(refined_probes[target_side][s][ci])
                for _tk in toks_h:
                    _tk["_mp_home"] = True
                final_map[ci] = toks_h

        for s in meta["away_spks"]:
            probe_h = refined_probes[opp_side].get(s, (probe_B if opp_side == "B" else probe_A)[s])
            amap, _ = align_away_speaker_by_dt(
                meta["away_cls"][s], target_side, probe_h, sim_segs,
                ctx_t, ch_t, adv_t, np_h_oth, act_h_oth, acts_t,
                meta["fg"], meta["fgc"],
                alignment_tokens, ono_clause_flags, clause_surfaces, clause_tid_seqs,
                is_final_pass=True, eval_cache=meta["eval_cache"], fg_changed_cs=meta["fg_diff_cs"],
                line_group=line_group,
            )
            full_s = spk_to_clauses[s]
            home_set_s = set(meta["home_cls"].get(s, ()))
            home_st = {}
            for ci_s in full_s:
                if ci_s in home_set_s:
                    ih = _clause_summary(final_map[ci_s])
                    if not ih["skipped"]:
                        home_st[ci_s] = float(ih["st"])
            prev_h = -1e9
            next_h_map: Dict[int, float] = {}
            nh = 1e9
            for ci_s in reversed(full_s):
                if ci_s in home_st:
                    nh = home_st[ci_s]
                next_h_map[ci_s] = nh
            last_st = -1e9
            for ci_s in full_s:
                if ci_s in home_st:
                    prev_h = home_st[ci_s]
                    last_st = max(last_st, prev_h)
                elif ci_s in amap:
                    iaw = _clause_summary(amap[ci_s])
                    if not iaw["skipped"]:
                        st_aw = float(iaw["st"])
                        if st_aw <= max(prev_h, last_st) + 0.05 or st_aw >= next_h_map[ci_s] - 0.05:
                            amap[ci_s] = [
                                {"token": t, "score": 0.0, "skipped": True, "ono_skipped": True, "line_index": ci_s}
                                for t in alignment_tokens[ci_s]
                            ]
                        else:
                            last_st = st_aw
            kept_n = sum(1 for ci in amap if not _clause_summary(amap[ci])["skipped"])
            side_logs.append(f"  🎯 [Step 4 -> 视角 {target_side}] 客视角 [{s}] 保留 {kept_n}/{len(meta['away_cls'][s])} 句")
            for _ci_a, _toks_a in amap.items():
                for _tk_a in _toks_a:
                    _tk_a["_mp_home"] = False
            final_map.update(amap)

        flat_side: List[Dict[str, Any]] = []
        for ci in range(n_c):
            if ci in final_map:
                flat_side.extend(final_map[ci])
            else:
                flat_side.extend(
                    {"token": t, "score": 0.0, "skipped": True, "ono_skipped": True, "line_index": ci}
                    for t in alignment_tokens[ci]
                )
        return target_side, flat_side, side_logs

    with ThreadPoolExecutor(max_workers=2) as ex:
        for t_side, fl_s, s_logs in ex.map(_step4_one_side, ("A", "B")):
            flat_by_side[t_side] = fl_s
            for msg in s_logs:
                log(msg)

    dur_A_sec = float(coop_A_np.shape[0] * FRAME_SEC)
    dur_B_sec = float(coop_B_np.shape[0] * FRAME_SEC)
    if sim_segs:
        # 按权重取主要同步段的典型 Δt 仅作日志展示参考
        best_seg = max(sim_segs, key=lambda s: float(s.get("weight", 0.0)))
        delta_sec = float(best_seg["dt_AB"])
    else:
        delta_sec = 0.0

    def _calc_nb_active(c_np: np.ndarray, ctx: Dict[str, Any]) -> Optional[np.ndarray]:
        if c_np.ndim != 2 or c_np.shape[1] <= 1:
            return None
        b_id = int(ctx.get("blank", 0))
        ign = set(ctx.get("ignored_classes", ()))
        v_cls = [c for c in range(c_np.shape[1]) if c != b_id and c not in ign]
        if not v_cls:
            return np.zeros(c_np.shape[0], dtype=bool)
        return (np.max(c_np[:, v_cls], axis=1) > (c_np[:, b_id] - np.float32(0.5)))

    log(f"  ✅ 多视角核心计算完成 (耗时 {time.time() - t_pipe0:.1f}s)")
    mp_meta: Dict[str, Any] = {
        "flat_A": flat_by_side["A"],
        "flat_B": flat_by_side["B"],
        "dur_A_sec": dur_A_sec,
        "dur_B_sec": dur_B_sec,
        "u_A_st_sec": 0.0,
        "u_B_st_sec": 0.0,
        "delta_sec": delta_sec,
        "sim_segs": sim_segs,
        "spk_persp_pref": spk_persp_pref,
        "spk_stats": spk_stats,
        "probe_streams": streams,
        "frame_sec": float(FRAME_SEC),
        "ctc_nb_active_A": _calc_nb_active(coop_A_np, ctx_A),
        "ctc_nb_active_B": _calc_nb_active(coop_B_np, ctx_B),
        "stereo_wav_16k_A": ctx_A.get("_stereo_wav"),
        "stereo_wav_16k_B": ctx_B.get("_stereo_wav"),
    }
    return flat_by_side["A"], streams, mp_meta


# ══════════════════════════════════════════════════════════════════════════
#  6. 按视角时间轴投影与越界裁剪导出 (Perspective Projection & Clipping)
# ══════════════════════════════════════════════════════════════════════════

def project_result_list_to_perspective(
    master_result_list: List[Dict[str, Any]],
    perspective_origin_sec: float,
    perspective_duration_sec: float,
    boundary_tolerance_sec: float = 0.35,
    perspective_side: str = "A",
    spk_persp_pref: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """将 result_list 裁剪到指定视角时间坐标系 [0, perspective_duration_sec] 并规范同屏左右顺序。

    - 若整行台词全部落在 [-boundary_tolerance_sec, perspective_duration_sec + boundary_tolerance_sec] 之外，
      自动标记为遗弃以防越界词条堆在开头或结尾。
    - 对于同屏多角色并列行（以全角空格连接），按起音顺序及该视角前景主次自动排列左右位置。
    """
    cloned = copy.deepcopy(master_result_list)
    n = len(cloned)
    line_start = 0
    side_is_B = (perspective_side.upper() == "B")

    for i in range(n + 1):
        is_nl = (i == n) or (cloned[i].get("orig") == "\n")
        if not is_nl:
            continue
        line_end = i
        if line_end > line_start:
            sts: List[float] = []
            ens: List[float] = []
            for k in range(line_start, line_end):
                it = cloned[k]
                if it.get(F.ONO_DROPPED):
                    continue
                st_s = _parse_lrc_ts_sec(it.get("start"))
                en_s = _parse_lrc_ts_sec(it.get("end"))
                if st_s is not None:
                    sts.append(st_s - perspective_origin_sec)
                if en_s is not None:
                    ens.append(en_s - perspective_origin_sec)

            if sts:
                min_st = min(sts)
                max_en = max(ens) if ens else min_st
                out_of_bounds = (
                    min_st < -0.15
                    or max_en < 0.05
                    or min_st > perspective_duration_sec - 0.05
                    or max_en > perspective_duration_sec + boundary_tolerance_sec
                )
                if out_of_bounds:
                    for k in range(line_start, line_end):
                        cloned[k].pop("start", None)
                        cloned[k].pop("end", None)
                        cloned[k][F.ONO_DROPPED] = True
                else:
                    for k in range(line_start, line_end):
                        it = cloned[k]
                        st_s = _parse_lrc_ts_sec(it.get("start"))
                        en_s = _parse_lrc_ts_sec(it.get("end"))
                        if st_s is not None:
                            new_st = max(0.0, min(perspective_duration_sec, st_s - perspective_origin_sec))
                            it["start"] = _format_time(new_st)
                        if en_s is not None:
                            new_en = max(0.0, min(perspective_duration_sec, en_s - perspective_origin_sec))
                            it["end"] = _format_time(new_en)

                    if spk_persp_pref:
                        segments: List[Tuple[str, List[Dict[str, Any]]]] = []
                        cur_spk_clean: Optional[str] = None
                        cur_seg: List[Dict[str, Any]] = []
                        for k in range(line_start, line_end):
                            it = cloned[k]
                            sp_tup = it.get(F.SPK)
                            sp_clean = str(sp_tup[0]).strip().lstrip("　") if sp_tup else (cur_spk_clean or "")
                            if cur_spk_clean is not None and sp_clean != cur_spk_clean and cur_seg:
                                segments.append((cur_spk_clean, cur_seg))
                                cur_seg = [it]
                                cur_spk_clean = sp_clean
                            else:
                                cur_spk_clean = sp_clean
                                cur_seg.append(it)
                        if cur_seg:
                            segments.append((cur_spk_clean or "", cur_seg))

                        if len(segments) >= 2:
                            seg_spans: List[Tuple[float, float]] = []
                            for _, seg_items in segments:
                                s_vals = [
                                    _parse_lrc_ts_sec(x.get("start"))
                                    for x in seg_items
                                    if not x.get(F.ONO_DROPPED) and x.get("start")
                                ]
                                e_vals = [
                                    _parse_lrc_ts_sec(x.get("end"))
                                    for x in seg_items
                                    if not x.get(F.ONO_DROPPED) and x.get("end")
                                ]
                                s_vals_clean = [v for v in s_vals if v is not None]
                                e_vals_clean = [v for v in e_vals if v is not None]
                                s_min = min(s_vals_clean) if s_vals_clean else min_st
                                e_max = max(e_vals_clean) if e_vals_clean else (s_min + 0.1)
                                seg_spans.append((s_min, e_max))

                            seg_homes: List[Optional[bool]] = []
                            for _, seg_items in segments:
                                h_vals = [x.get("_mp_home") for x in seg_items if not x.get(F.ONO_DROPPED) and "_mp_home" in x]
                                if h_vals:
                                    seg_homes.append(sum(1 for v in h_vals if v) >= (len(h_vals) / 2.0))
                                else:
                                    seg_homes.append(None)

                            import functools

                            def _cmp_segs(a_info: Tuple[int, str], b_info: Tuple[int, str]) -> int:
                                idx_a, sp_a = a_info
                                idx_b, sp_b = b_info
                                st_a, en_a = seg_spans[idx_a]
                                st_b, en_b = seg_spans[idx_b]

                                if en_a <= st_b + 0.15 and st_a < st_b:
                                    return -1
                                elif en_b <= st_a + 0.15 and st_b < st_a:
                                    return 1

                                # 当前段实时主客归属优先（主场永远居左，防止换偶前后全局均值颠倒）
                                ha, hb = seg_homes[idx_a], seg_homes[idx_b]
                                if ha is not None and hb is not None and ha != hb:
                                    return -1 if ha else 1

                                if abs(st_a - st_b) >= 0.40:
                                    return -1 if st_a < st_b else 1

                                pref_a = float(spk_persp_pref.get(sp_a, 0.0))
                                pref_b = float(spk_persp_pref.get(sp_b, 0.0))
                                score_a = pref_a if side_is_B else -pref_a
                                score_b = pref_b if side_is_B else -pref_b
                                if abs(score_a - score_b) > 1e-4:
                                    return -1 if score_a > score_b else 1

                                return -1 if st_a < st_b else (1 if st_a > st_b else (idx_a - idx_b))

                            sorted_indices = sorted(
                                [(idx_s, segments[idx_s][0]) for idx_s in range(len(segments))],
                                key=functools.cmp_to_key(_cmp_segs),
                            )
                            sorted_segs = [segments[idx_s] for idx_s, _ in sorted_indices]
                            rebuilt_line: List[Dict[str, Any]] = []
                            for seg_idx, (sp_c, seg_items) in enumerate(sorted_segs):
                                for it in seg_items:
                                    sp_tup = it.get(F.SPK)
                                    if sp_tup:
                                        base_sp0 = str(sp_tup[0]).strip().lstrip("　")
                                        new_sp0 = base_sp0 if seg_idx == 0 else ("　" + base_sp0)
                                        it[F.SPK] = (new_sp0,) + tuple(sp_tup[1:])
                                    rebuilt_line.append(it)
                            first_st_str = _format_time(max(0.0, min(perspective_duration_sec, min_st)))
                            last_en_str = _format_time(max(0.0, min(perspective_duration_sec, max_en)))
                            surviving_toks = [
                                it for it in rebuilt_line
                                if not it.get(F.ONO_DROPPED) and ("start" in it or "end" in it)
                            ]
                            if surviving_toks:
                                surviving_toks[0]["start"] = first_st_str
                                surviving_toks[-1]["end"] = last_en_str
                                for _, seg_items in sorted_segs:
                                    s_running = max(0.0, min(perspective_duration_sec, min_st))
                                    for it in seg_items:
                                        if it.get(F.ONO_DROPPED):
                                            continue
                                        st_v = _parse_lrc_ts_sec(it.get("start"))
                                        en_v = _parse_lrc_ts_sec(it.get("end"))
                                        if st_v is not None and st_v < s_running:
                                            it["start"] = _format_time(s_running)
                                        elif st_v is not None:
                                            s_running = st_v
                                        if en_v is not None and en_v < s_running:
                                            it["end"] = _format_time(s_running)
                                        elif en_v is not None:
                                            s_running = min(max_en, en_v)
                                surviving_toks[0]["start"] = first_st_str
                                surviving_toks[-1]["end"] = last_en_str
                            cloned[line_start:line_end] = rebuilt_line
        line_start = i + 1

    return cloned


def align_multiperspective_joint(
    audio_path_A: str | Path,
    audio_path_B: str | Path,
    alignment_tokens: List[List[str]],
    line_group: Sequence[int],
    ono_clause_flags: Sequence[int],
    clause_speakers: Sequence[Any],
    lora_path: Optional[str] = None,
    dev: str = "auto",
    vram_chunk: int = 3000,
    enable_silence_snap: bool = True,
    enable_ono_gate: bool = True,
    log_callback: Optional[Callable[[str], None]] = None,
    enable_word_skip: bool = False,
    word_skip_token_penalty: float = 4.8,
    clause_surfaces: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[List[int]], Dict[str, Any]]:
    """从两个视角的音频文件提取声学上下文（含 STFT 立体声波），并执行四步多视角动态联合对齐。"""
    def log(msg: str) -> None:
        if log_callback:
            log_callback(msg)

    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_wA = ex.submit(_load_audio_16k_once, audio_path_A)
        fut_wB = ex.submit(_load_audio_16k_once, audio_path_B)
        wav_A_t, stereo_A_np = fut_wA.result()
        wav_B_t, stereo_B_np = fut_wB.result()

    log(f"  🎧 [视角 A 声学提取] {Path(audio_path_A).name}")
    ctx_A = extract_acoustic_context(
        audio_file_path=wav_A_t if wav_A_t is not None else audio_path_A,
        lora_path=lora_path,
        device=dev,
        chunk_frames=int(vram_chunk),
    )
    ctx_A["_stereo_wav"] = stereo_A_np

    log(f"  🎧 [视角 B 声学提取] {Path(audio_path_B).name}")
    ctx_B = extract_acoustic_context(
        audio_file_path=wav_B_t if wav_B_t is not None else audio_path_B,
        lora_path=lora_path,
        device=dev,
        chunk_frames=int(vram_chunk),
    )
    ctx_B["_stereo_wav"] = stereo_B_np

    return run_multiperspective_pipeline(
        alignment_tokens=alignment_tokens,
        line_group=line_group,
        ono_clause_flags=ono_clause_flags,
        clause_speakers=clause_speakers,
        ctx_A=ctx_A,
        ctx_B=ctx_B,
        enable_silence_snap=enable_silence_snap,
        enable_ono_gate=enable_ono_gate,
        log_callback=log_callback,
        enable_word_skip=enable_word_skip,
        word_skip_token_penalty=word_skip_token_penalty,
        clause_surfaces=clause_surfaces,
    )
