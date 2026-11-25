# -*- coding: utf-8 -*-
"""FA-ASMR V2 — 多视角主/客视角分块归属、单声源并发修复与 Δt 局部检索。"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from fa_asmr.aligners.align_acoustic import FRAME_SEC
from fa_asmr.aligners.align_utils import _format_time
from fa_asmr.aligners.multipersp_acoustic import (
    _AudioSoloChecker,
    _active_phoneme_sim,
    _get_tokenizer_dict,
    _is_pure_sfx_clause,
    _norm_spk,
    _unmasked_clause_mean_score,
    slice_ctc_verify_clause,
)
from fa_asmr.aligners.multispeaker_acoustic import (
    _align_subset_clauses,
    _clause_summary,
)

# ══════════════════════════════════════════════════════════════════════════
#  2. 按角色场景分块与绝对清晰度主视角判定 (Segment & Assign Home)
# ══════════════════════════════════════════════════════════════════════════

def segment_and_assign_home(
    spk: str,
    c_list: List[int],
    probe_A_spk: Dict[int, List[Dict[str, Any]]],
    probe_B_spk: Dict[int, List[Dict[str, Any]]],
    tokens_all: Sequence[Sequence[str]],
    onos_all: Sequence[int],
    clause_tid_seqs: Dict[int, List[int]],
    norm_p_A: np.ndarray,
    act_A: np.ndarray,
    norm_p_B: np.ndarray,
    act_B: np.ndarray,
    coop_A_np: np.ndarray,
    coop_B_np: np.ndarray,
    sim_segs: Sequence[Dict[str, Any]],
) -> Tuple[List[Optional[str]], List[Dict[str, Any]]]:
    """切分 SHARED 平台、持续主导权翻转与 SimSeg 场景转折点，并通过绝对阈值判定主视角。"""
    n_c = len(c_list)
    if n_c == 0:
        return [], []
    dur_A, dur_B = coop_A_np.shape[0] * FRAME_SEC, coop_B_np.shape[0] * FRAME_SEC

    sc_a = np.full(n_c, np.nan, dtype=np.float32)
    sc_b = np.full(n_c, np.nan, dtype=np.float32)
    ta = np.full(n_c, np.nan, dtype=np.float32)
    tb = np.full(n_c, np.nan, dtype=np.float32)
    sta = np.full(n_c, np.nan, dtype=np.float32)
    ena = np.full(n_c, np.nan, dtype=np.float32)
    stb = np.full(n_c, np.nan, dtype=np.float32)
    enb = np.full(n_c, np.nan, dtype=np.float32)
    bc_flag = np.zeros(n_c, dtype=bool)
    matched_dt = np.full(n_c, np.nan, dtype=np.float32)

    for k, ci in enumerate(c_list):
        toks, tids = tokens_all[ci], clause_tid_seqs[ci]
        if onos_all[ci] != 0 or len(toks) < 2 or len(tids) < 2:
            continue
        ia, ib = _clause_summary(probe_A_spk[ci]), _clause_summary(probe_B_spk[ci])
        max_d = max(4.5, 0.65 * len(tids))
        if not ia["skipped"] and 0.12 <= ia["dur"] <= max_d:
            sc_a[k], sta[k], ena[k], ta[k] = ia["score"], ia["st"], ia["en"], 0.5 * (ia["st"] + ia["en"])
        if not ib["skipped"] and 0.12 <= ib["dur"] <= max_d:
            sc_b[k], stb[k], enb[k], tb[k] = ib["score"], ib["st"], ib["en"], 0.5 * (ib["st"] + ib["en"])
        if not (np.isnan(sc_a[k]) or np.isnan(sc_b[k])) and sc_a[k] >= -2.55 and sc_b[k] >= -2.55:
            dt_v = float(ta[k] - tb[k])
            hits = [
                s for s in sim_segs
                if abs(dt_v - s["dt_AB"]) <= 3.5 and (s["A_st"] - 20.0) <= ta[k] <= (s["A_en"] + 20.0)
            ]
            if hits:
                dt_c = float(hits[0]["dt_AB"])
                dom_B = bool(sc_b[k] >= sc_a[k])
                s0_d, s1_d = (float(stb[k]), float(enb[k])) if dom_B else (float(sta[k]), float(ena[k]))
                ps = _active_phoneme_sim(
                    norm_p_B if dom_B else norm_p_A, act_B if dom_B else act_A,
                    [(norm_p_A, act_A)] if dom_B else [(norm_p_B, act_B)],
                    s0_d, s1_d, dt_c if dom_B else -dt_c, pad_steps=12,
                )
                if ps >= 0.25:
                    bc_flag[k], matched_dt[k] = True, dt_c

    raw_cl: List[List[int]] = []
    for k in range(n_c):
        if not bc_flag[k]:
            continue
        if raw_cl:
            pk = raw_cl[-1][-1]
            # 严禁把相隔大量非共享句（中间可能是单侧主导段）的偶发同步点跨空连成一个巨无霸 SHARED 块
            non_sh_between = sum(
                1 for j in range(pk + 1, k)
                if onos_all[c_list[j]] == 0 and len(tokens_all[c_list[j]]) >= 2 and not bc_flag[j]
            )
            dt_span = abs(float(ta[k] - ta[pk]))
            if (
                non_sh_between <= 6
                and (k - pk) <= 14
                and dt_span <= 35.0
                and abs(matched_dt[k] - matched_dt[pk]) <= 2.5
                and abs((ta[k] - ta[pk]) - (tb[k] - tb[pk])) <= 3.0
            ):
                raw_cl[-1].append(k)
                continue
        raw_cl.append([k])
    sh_cl = [cl for cl in raw_cl if len(cl) >= 3]

    base_ints: List[Tuple[int, int, bool, Optional[List[int]]]] = []
    cur = 0
    for cl in sh_cl:
        if cl[0] > cur:
            base_ints.append((cur, cl[0], False, None))
        base_ints.append((cl[0], cl[-1] + 1, True, cl))
        cur = cl[-1] + 1
    if cur < n_c:
        base_ints.append((cur, n_c, False, None))

    # 切分持续主场切换边界（例如中途换房或双角色镜头切换）
    dom_ints: List[Tuple[int, int, bool, Optional[List[int]]]] = []
    for ks, ke, is_sh, cl in base_ints:
        if is_sh or (ke - ks) < 16:
            dom_ints.append((ks, ke, is_sh, cl))
            continue
        sem_votes: List[Tuple[int, str]] = []
        for k in range(ks, ke):
            va = float(sc_a[k]) if not np.isnan(sc_a[k]) else -10.0
            vb = float(sc_b[k]) if not np.isnan(sc_b[k]) else -10.0
            if va >= -2.75 and va - vb >= 0.85:
                sem_votes.append((k, "A"))
            elif vb >= -2.75 and vb - va >= 0.85:
                sem_votes.append((k, "B"))
        flip_cuts = [ks]
        if len(sem_votes) >= 7:
            side_runs: List[Tuple[str, int, int, int]] = []
            for k, sd in sem_votes:
                if side_runs and side_runs[-1][0] == sd:
                    ps, fk, _, cnt = side_runs[-1]
                    side_runs[-1] = (ps, fk, k, cnt + 1)
                else:
                    side_runs.append((sd, k, k, 1))
            strong_runs = [r for r in side_runs if r[3] >= 4]
            for r_i in range(len(strong_runs) - 1):
                if strong_runs[r_i][0] != strong_runs[r_i + 1][0]:
                    cut_k = strong_runs[r_i + 1][1]
                    if cut_k - flip_cuts[-1] >= 4 and ke - cut_k >= 4:
                        flip_cuts.append(cut_k)
        flip_cuts.append(ke)
        for u in range(len(flip_cuts) - 1):
            dom_ints.append((flip_cuts[u], flip_cuts[u + 1], False, None))

    final_ints: List[Tuple[int, int, bool, Optional[List[int]]]] = []
    sim_bounds_A = sorted({float(s["A_st"]) for s in sim_segs})
    sim_bounds_B = sorted({float(s["B_st"]) for s in sim_segs})
    for ks, ke, is_sh, cl in dom_ints:
        if is_sh or (ke - ks) < 30:
            final_ints.append((ks, ke, is_sh, cl))
            continue
        sa_k = [float(sc_a[k]) for k in range(ks, ke) if not np.isnan(sc_a[k])]
        sb_k = [float(sc_b[k]) for k in range(ks, ke) if not np.isnan(sc_b[k])]
        use_B = (float(np.median(sb_k)) if sb_k else -10.0) > (float(np.median(sa_k)) if sa_k else -10.0)
        t_arr = tb if use_B else ta
        b_list = sim_bounds_B if use_B else sim_bounds_A
        t_cuts: List[int] = [ks]
        for b_t in b_list:
            last_k = t_cuts[-1]
            if ke - last_k < 24:
                break
            hit_k = next(
                (j for j in range(last_k + 12, ke - 12) if not np.isnan(t_arr[j]) and t_arr[j] >= b_t),
                None,
            )
            if hit_k is not None:
                t_cuts.append(hit_k)
        t_cuts.append(ke)
        for u in range(len(t_cuts) - 1):
            final_ints.append((t_cuts[u], t_cuts[u + 1], False, None))

    def _is_clear_home(vals: List[float]) -> bool:
        if not vals:
            return False
        if len(vals) < 4:
            return max(vals) >= -2.00
        return float(np.percentile(vals, 75)) >= -2.55 and float(np.median(vals)) >= -3.40

    seg_records: List[Dict[str, Any]] = []
    for ks, ke, is_sh, cl in final_ints:
        sa = [float(sc_a[k]) for k in range(ks, ke) if not np.isnan(sc_a[k])]
        sb = [float(sc_b[k]) for k in range(ks, ke) if not np.isnan(sc_b[k])]
        med_a = float(np.median(sa)) if sa else -10.0
        med_b = float(np.median(sb)) if sb else -10.0
        p75_a = float(np.percentile(sa, 75)) if sa else -10.0
        p75_b = float(np.percentile(sb, 75)) if sb else -10.0
        clr_a, clr_b = _is_clear_home(sa), _is_clear_home(sb)

        if is_sh and clr_a and clr_b:
            dt_m = float(np.median(matched_dt[cl]))
            if abs(med_a - med_b) <= 1.15:
                home = "SHARED"
                seg_type = f"SHARED(dt={dt_m:+.2f}s)"
            else:
                home = "A" if med_a >= med_b else "B"
                seg_type = f"SPLIT_SYNC(dt={dt_m:+.2f}s)"
            ta_span = (float(np.min(ta[cl])), float(np.max(ta[cl])))
            tb_span = (float(np.min(tb[cl])), float(np.max(tb[cl])))
        elif clr_a and clr_b:
            home = "SHARED" if abs(med_a - med_b) <= 0.85 else ("A" if med_a >= med_b else "B")
            seg_type = "DUAL_CLEAR"
            ta_span = (float(np.nanmin(ta[ks:ke])) if sa else 0.0, float(np.nanmax(ta[ks:ke])) if sa else dur_A)
            tb_span = (float(np.nanmin(tb[ks:ke])) if sb else 0.0, float(np.nanmax(tb[ks:ke])) if sb else dur_B)
        elif clr_a:
            home, seg_type = "A", "SPLIT"
            ta_span, tb_span = None, None
        elif clr_b:
            home, seg_type = "B", "SPLIT"
            ta_span, tb_span = None, None
        else:
            home, seg_type = None, "SUBTHRESH"
            ta_span, tb_span = None, None

        seg_records.append({
            "k_start": ks, "k_end": ke - 1, "home": home, "raw_clear": (clr_a or clr_b),
            "seg_type": seg_type, "q_A": med_a, "q_B": med_b, "p75_A": p75_a, "p75_B": p75_b,
            "ta_span": ta_span, "tb_span": tb_span,
            "dt_med": float(np.median(matched_dt[cl])) if (is_sh and cl) else None,
        })

    # 孤立噪声小块反抖平滑：防止 1 个极小 SPLIT 块（<=16 句且无压倒性清晰度优势）劫持大片 SUBTHRESH
    cnt_a0 = sum(s["k_end"] - s["k_start"] + 1 for s in seg_records if s["raw_clear"] and s["home"] == "A")
    cnt_b0 = sum(s["k_end"] - s["k_start"] + 1 for s in seg_records if s["raw_clear"] and s["home"] == "B")
    dom_tot0 = cnt_a0 + cnt_b0
    strong_prior = None
    if dom_tot0 >= 30:
        if cnt_a0 >= 0.72 * dom_tot0:
            strong_prior = "A"
        elif cnt_b0 >= 0.72 * dom_tot0:
            strong_prior = "B"
    for i, sg in enumerate(seg_records):
        seg_len = sg["k_end"] - sg["k_start"] + 1
        if not sg["raw_clear"] or sg["home"] not in ("A", "B") or seg_len > 18:
            continue
        opp = "B" if sg["home"] == "A" else "A"
        margin = (sg["q_A"] - sg["q_B"]) if sg["home"] == "A" else (sg["q_B"] - sg["q_A"])
        # 左右最近的大体量清晰块 (>= 20 句)
        left_big = next((seg_records[j]["home"] for j in range(i - 1, -1, -1) if seg_records[j]["raw_clear"] and (seg_records[j]["k_end"] - seg_records[j]["k_start"] + 1) >= 20), None)
        right_big = next((seg_records[j]["home"] for j in range(i + 1, len(seg_records)) if seg_records[j]["raw_clear"] and (seg_records[j]["k_end"] - seg_records[j]["k_start"] + 1) >= 20), None)
        flanked_by_opp = (left_big == opp and right_big in (opp, "SHARED", None)) or (right_big == opp and left_big in (opp, "SHARED", None))
        if strong_prior == opp and flanked_by_opp and margin < 1.35:
            sg["home"] = None
            sg["raw_clear"] = False
            sg["seg_type"] = "SUBTHRESH"

    cnt_a = sum(s["k_end"] - s["k_start"] + 1 for s in seg_records if s["raw_clear"] and s["home"] == "A")
    cnt_b = sum(s["k_end"] - s["k_start"] + 1 for s in seg_records if s["raw_clear"] and s["home"] == "B")
    pref_nonshared = "B" if cnt_b >= cnt_a else "A"
    for i, sg in enumerate(seg_records):
        if sg["home"] is not None:
            continue
        left_nh, left_w = None, 0
        for ni in range(i - 1, -1, -1):
            if seg_records[ni]["raw_clear"]:
                left_nh = seg_records[ni]["home"]
                left_w = seg_records[ni]["k_end"] - seg_records[ni]["k_start"] + 1
                break
        right_nh, right_w = None, 0
        for ni in range(i + 1, len(seg_records)):
            if seg_records[ni]["raw_clear"]:
                right_nh = seg_records[ni]["home"]
                right_w = seg_records[ni]["k_end"] - seg_records[ni]["k_start"] + 1
                break
        if left_nh in ("A", "B") and right_nh in ("A", "B") and left_nh != right_nh:
            if abs(sg["p75_A"] - sg["p75_B"]) >= 0.45:
                sg["home"] = "B" if sg["p75_B"] >= sg["p75_A"] else "A"
            else:
                sg["home"] = left_nh if left_w >= right_w else right_nh
        elif left_nh in ("A", "B"):
            sg["home"] = left_nh
        elif right_nh in ("A", "B"):
            sg["home"] = right_nh
        elif left_nh == "SHARED" or right_nh == "SHARED":
            sg["home"] = ("A" if sg["p75_A"] > sg["p75_B"] else "B") if abs(sg["p75_A"] - sg["p75_B"]) >= 0.80 else pref_nonshared

    # 同步双麦高频拟声/音效救援（当全轨拟声偏多拉低 p75，但同步覆盖率或 p85 明确清晰时）
    if not any(sg["raw_clear"] for sg in seg_records):
        sa_all = [float(sc_a[k]) for k in range(n_c) if not np.isnan(sc_a[k])]
        sb_all = [float(sc_b[k]) for k in range(n_c) if not np.isnan(sc_b[k])]
        p85_a = float(np.percentile(sa_all, 85)) if sa_all else -10.0
        p85_b = float(np.percentile(sb_all, 85)) if sb_all else -10.0
        p75_a_all = float(np.percentile(sa_all, 75)) if sa_all else -10.0
        p75_b_all = float(np.percentile(sb_all, 75)) if sb_all else -10.0
        n_sync = int(np.sum(bc_flag))
        sync_cov = sum(
            max(0.0, min(float(s["A_en"]) - float(s["A_st"]), float(s["B_en"]) - float(s["B_st"])))
            for s in sim_segs
        ) / max(dur_A, dur_B, 1.0)
        if max(p85_a, p85_b) >= -2.20 or (sync_cov >= 0.70 and max(p85_a, p85_b) >= -2.60):
            home_fb = (
                "SHARED"
                if (n_sync >= 4 or sync_cov >= 0.70) and abs(p85_a - p85_b) <= 0.85
                else ("B" if (p85_b + p75_b_all) >= (p85_a + p75_a_all) else "A")
            )
            for sg in seg_records:
                sg["home"] = home_fb

    clause_sides: List[Optional[str]] = [None] * n_c
    for sg in seg_records:
        for k in range(sg["k_start"], sg["k_end"] + 1):
            clause_sides[k] = sg["home"]
    return clause_sides, seg_records


# ══════════════════════════════════════════════════════════════════════════
#  3. 主视角全量采纳与立体声单声源碰撞修复 (Refine Home Segments)
# ══════════════════════════════════════════════════════════════════════════

def _build_fg_masks(
    clauses_by_spk: Dict[str, List[int]],
    probe_by_spk: Dict[str, Dict[int, List[Dict[str, Any]]]],
    T_target: int,
    C_target: int,
    min_score: Optional[float],
    tok_dict: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[float, float]]]:
    fg_mask = np.zeros(T_target, dtype=np.float32)
    fg_char = np.zeros((T_target, C_target), dtype=np.float32)
    spans: List[Tuple[float, float]] = []
    for spk, cl in clauses_by_spk.items():
        for ci in cl:
            inf = _clause_summary(probe_by_spk[spk][ci])
            if inf["skipped"] or inf["dur"] <= 0.05:
                continue
            if min_score is not None and inf["score"] < min_score:
                continue
            spans.append((float(inf["st"]), float(inf["en"])))
            for r in probe_by_spk[spk][ci]:
                if r.get("skipped") or r.get("original_start") is None or r.get("original_end") is None:
                    continue
                f0 = max(0, min(T_target - 1, int(round(float(r["original_start"]) / FRAME_SEC))))
                f1 = max(f0 + 1, min(T_target, int(round(float(r["original_end"]) / FRAME_SEC))))
                fg_mask[f0:f1] = 1.0
                for ch in str(r.get("token", "")):
                    if ch in tok_dict:
                        fg_char[f0:f1, int(tok_dict[ch])] = 1.0
    return fg_mask, fg_char, spans


def refine_home_segments(
    active_home_spks: List[str],
    spk_home_clauses: Dict[str, List[int]],
    spk_all_clauses: Dict[str, List[int]],
    probe_target: Dict[str, Dict[int, List[Dict[str, Any]]]],
    away_raw_hits: Sequence[Tuple[float, float, float]],
    alignment_tokens: Sequence[Sequence[str]],
    line_group: Sequence[int],
    ono_clause_flags: Sequence[int],
    clause_surfaces: Optional[Sequence[str]],
    clause_speakers: Sequence[Any],
    ctx_target: Dict[str, Any],
    coop_target: torch.Tensor,
    solo_checker: _AudioSoloChecker,
) -> Dict[str, Dict[int, List[Dict[str, Any]]]]:
    """全量接受 Home/SHARED 块对齐结果，仅对落入单声源独占碰撞区的弱句在 ±15s 局部窗内修复。"""
    out = {s: copy.deepcopy(probe_target[s]) for s in active_home_spks}
    if not active_home_spks or (not away_raw_hits and len(active_home_spks) < 2):
        return out

    tok_dict = _get_tokenizer_dict(ctx_target)
    coop_np = coop_target.detach().cpu().numpy()
    em_L_np = ctx_target["emission_L"][0].detach().cpu().numpy() if "emission_L" in ctx_target else coop_np
    em_R_np = ctx_target["emission_R"][0].detach().cpu().numpy() if "emission_R" in ctx_target else coop_np
    blank_id = int(ctx_target.get("blank", 0))
    ignored = set(ctx_target.get("ignored_classes", ()))
    valid_cls = [c for c in range(coop_np.shape[1]) if c != blank_id and c not in ignored]
    T_all = int(coop_np.shape[0])
    max_nb = np.max(coop_np[:, valid_cls], axis=1).astype(np.float32)

    def _merge_spans(sps: Sequence[Tuple[float, float]], gap: float = 1.5) -> List[Tuple[float, float]]:
        res: List[Tuple[float, float]] = []
        for a, b in sorted(sps):
            if res and a <= res[-1][1] + gap:
                res[-1] = (res[-1][0], max(res[-1][1], b))
            else:
                res.append((a, b))
        return res

    solo_cache: Dict[Tuple[float, float], bool] = {}

    def _is_solo_cached(st_s: float, en_s: float) -> bool:
        key = (round(st_s, 2), round(en_s, 2))
        if key not in solo_cache:
            solo_cache[key] = solo_checker.is_clearly_solo(st_s, en_s)
        return solo_cache[key]

    def _skip_clause(ci: int) -> List[Dict[str, Any]]:
        return [{"token": t, "score": 0.0, "skipped": True, "ono_skipped": True, "line_index": ci} for t in alignment_tokens[ci]]

    for spk in active_home_spks:
        spk_hits: List[Tuple[float, float, float]] = list(away_raw_hits)
        for other_spk in active_home_spks:
            if other_spk == spk:
                continue
            for oci in spk_home_clauses[other_spk]:
                if ono_clause_flags[oci] != 0:
                    continue
                oinf = _clause_summary(probe_target[other_spk][oci])
                if not oinf["skipped"] and oinf["dur"] > 0.05 and oinf["score"] >= -2.30:
                    spk_hits.append((float(oinf["st"]), float(oinf["en"]), float(oinf["score"])))
        if not spk_hits:
            continue

        solo_spans = [(st_a, en_a) for st_a, en_a, _ in spk_hits if _is_solo_cached(st_a, en_a)]
        simul_spans = [(st_a, en_a) for st_a, en_a, _ in spk_hits if not _is_solo_cached(st_a, en_a)]
        m_solo = _merge_spans(solo_spans, 2.0)
        m_simul = _merge_spans(simul_spans, 1.2)
        all_claims = m_solo + m_simul
        raw_solo_spans = solo_spans

        l_mod, r_mod, st_mod = em_L_np.copy(), em_R_np.copy(), coop_np.copy()
        filler_logp = (max_nb - np.float32(1.35)).astype(np.float32)

        for st_s, en_s in m_solo:
            f0 = max(0, int(round((st_s - 0.05) / FRAME_SEC)))
            f1 = min(T_all, int(round((en_s + 0.05) / FRAME_SEC)))
            if f1 <= f0:
                continue
            fl = np.float32(-5.8)
            for arr in (l_mod, r_mod, st_mod):
                cur = arr[f0:f1, valid_cls]
                arr[f0:f1, valid_cls] = np.maximum(cur - np.float32(2.6), np.minimum(cur, fl))
            filler_logp[f0:f1] = np.maximum(
                filler_logp[f0:f1],
                np.maximum(coop_np[f0:f1, blank_id], max_nb[f0:f1] - np.float32(0.10)),
            )

        for st_s, en_s in m_simul:
            f0 = max(0, int(round((st_s - 0.05) / FRAME_SEC)))
            f1 = min(T_all, int(round((en_s + 0.05) / FRAME_SEC)))
            if f1 > f0:
                filler_logp[f0:f1] = np.maximum(
                    filler_logp[f0:f1],
                    np.maximum(coop_np[f0:f1, blank_id], max_nb[f0:f1] - np.float32(0.10)),
                )

        em_mod = torch.from_numpy(np.maximum(st_mod, np.maximum(l_mod, r_mod) - np.float32(0.18)).astype(np.float32))
        em_mod[:, blank_id] = torch.from_numpy(st_mod[:, blank_id])

        c_list = spk_home_clauses[spk]
        rank_map = {ci: rk for rk, ci in enumerate(spk_all_clauses[spk])}
        runs: List[List[int]] = []
        for ci in c_list:
            if runs and rank_map.get(ci, -9) == rank_map.get(runs[-1][-1], -99) + 1:
                runs[-1].append(ci)
            else:
                runs.append([ci])

        for run_cls in runs:
            sem_pos = [
                idx_c for idx_c, ci in enumerate(run_cls)
                if ono_clause_flags[ci] == 0 and not _clause_summary(out[spk][ci])["skipped"]
            ]
            if not sem_pos:
                continue
            valid_sts = [_clause_summary(out[spk][ci]) for ci in run_cls]
            valid_sts = [x for x in valid_sts if not x["skipped"] and x["dur"] > 0.05]
            if not valid_sts:
                continue
            run_lo = max(0.0, min(float(x["st"]) for x in valid_sts) - 15.0)
            run_hi = min(T_all * FRAME_SEC, max(float(x["en"]) for x in valid_sts) + 15.0)

            is_col: List[bool] = []
            is_short_col: List[bool] = []
            for p_idx in sem_pos:
                ci_p = run_cls[p_idx]
                inf = _clause_summary(out[spk][ci_p])
                n_phones = sum(len(t) for t in alignment_tokens[ci_p])
                ov_solo = any(min(ce, inf["en"]) - max(cs, inf["st"]) >= 0.25 for cs, ce in m_solo)
                short_wrapped_solo = (
                    n_phones <= 6 and float(inf["dur"]) <= 0.65
                    and any(
                        (min(ce, inf["en"]) - max(cs, inf["st"])) >= 0.60 * max(0.05, float(inf["dur"]))
                        and (ce - cs) >= 1.5 * max(0.05, float(inf["dur"]))
                        for cs, ce in raw_solo_spans
                    )
                )
                col_any = (inf["score"] < -2.30 and any(ce > inf["st"] - 1.0 and cs < inf["en"] + 1.0 for cs, ce in all_claims))
                col_solo = (inf["score"] < -1.80 and ov_solo) or short_wrapped_solo
                is_col.append(col_any or col_solo)
                is_short_col.append(short_wrapped_solo)

            k = 0
            while k < len(sem_pos):
                if not is_col[k]:
                    k += 1
                    continue
                k0 = k
                while (
                    k + 1 < len(sem_pos)
                    and is_col[k + 1]
                    and all(_norm_spk(clause_speakers[idx]) == spk for idx in range(run_cls[sem_pos[k]] + 1, run_cls[sem_pos[k + 1]]))
                ):
                    k += 1
                k1 = k
                k += 1

                left_idx = sem_pos[k0 - 1] if k0 > 0 else -1
                right_idx = sem_pos[k1 + 1] if k1 + 1 < len(sem_pos) else len(run_cls)
                w_lo = float(_clause_summary(out[spk][run_cls[left_idx]])["en"]) if left_idx >= 0 else run_lo
                w_hi = float(_clause_summary(out[spk][run_cls[right_idx]])["st"]) if right_idx < len(run_cls) else run_hi
                f_lo = max(0, int(np.ceil(w_lo / FRAME_SEC)))
                f_hi = min(T_all, int(np.floor(w_hi / FRAME_SEC)))
                if f_hi <= f_lo + 2:
                    continue

                sub_sem = [run_cls[p_idx] for p_idx in sem_pos[k0 : k1 + 1]]
                em_win = em_mod.clone()
                if f_lo > 0:
                    em_win[:f_lo, valid_cls] = -15.0
                    em_win[:f_lo, blank_id] = 0.0
                if f_hi < T_all:
                    em_win[f_hi:, valid_cls] = -15.0
                    em_win[f_hi:, blank_id] = 0.0

                realigned_blk = _align_subset_clauses(
                    sub_indices=sub_sem,
                    alignment_tokens=alignment_tokens,
                    line_group=line_group,
                    ono_clause_flags=ono_clause_flags,
                    enable_silence_snap=True,
                    acoustic_ctx=ctx_target,
                    custom_emission_2d=em_win,
                    enable_inter_line_filler=True,
                    filler_frame_logp=filler_logp,
                    enable_ono_skip=False,
                    ono_skip_token_penalty=0.5,
                    enable_word_skip=False,
                    clause_surfaces=clause_surfaces,
                    clause_speakers=clause_speakers,
                )
                old_scs = [_unmasked_clause_mean_score(out[spk][ci], coop_np, tok_dict) for ci in sub_sem]
                old_solo_ov = sum(
                    max(0.0, min(ce, float(_clause_summary(out[spk][ci])["en"])) - max(cs, float(_clause_summary(out[spk][ci])["st"])))
                    for ci in sub_sem for cs, ce in m_solo
                )
                new_scs: List[float] = []
                new_solo_ov = 0.0
                valid_blk = True
                for ci in sub_sem:
                    if ci not in realigned_blk or _clause_summary(realigned_blk[ci])["skipped"]:
                        valid_blk = False
                        break
                    new_scs.append(_unmasked_clause_mean_score(realigned_blk[ci], coop_np, tok_dict))
                    n_inf = _clause_summary(realigned_blk[ci])
                    new_solo_ov += sum(max(0.0, min(ce, float(n_inf["en"])) - max(cs, float(n_inf["st"]))) for cs, ce in m_solo)
                solo_margin = 1.80 if any(is_short_col[pi] for pi in range(k0, k1 + 1)) else 0.50
                cleared_solo = (
                    old_solo_ov >= 0.25
                    and new_solo_ov <= 0.10
                    and float(np.mean(new_scs)) >= max(-3.50, float(np.mean(old_scs)) - solo_margin)
                )
                if not valid_blk or (not cleared_solo and float(np.mean(new_scs)) <= float(np.mean(old_scs)) + 0.15):
                    continue

                for ci in sub_sem:
                    out[spk][ci] = realigned_blk[ci]

                sem_in_span = list(sem_pos[k0 : k1 + 1])
                anchors_idx = ([left_idx] if left_idx >= 0 else []) + sem_in_span + ([right_idx] if right_idx < len(run_cls) else [])
                for pos_i in range(max(0, left_idx + 1), min(len(run_cls), right_idx)):
                    ci_o = run_cls[pos_i]
                    if ono_clause_flags[ci_o] == 0:
                        continue
                    inf_o = _clause_summary(out[spk][ci_o])
                    if inf_o["skipped"]:
                        continue
                    prev_a = max((ai for ai in anchors_idx if ai < pos_i), default=None)
                    next_a = min((ai for ai in anchors_idx if ai > pos_i), default=None)
                    lo_t = float(_clause_summary(out[spk][run_cls[prev_a]])["st"]) if prev_a is not None else -1e9
                    hi_t = float(_clause_summary(out[spk][run_cls[next_a]])["st"]) if next_a is not None else 1e9
                    if inf_o["st"] < lo_t or inf_o["st"] > hi_t:
                        out[spk][ci_o] = _skip_clause(ci_o)
    return out


# ══════════════════════════════════════════════════════════════════════════
#  4. 客视角局部 Δt 小窗检索与过滤 (Align Away Speaker by Δt)
# ══════════════════════════════════════════════════════════════════════════

def _candidate_dts_at_time(t_h: float, sim_segs: Sequence[Dict[str, Any]], sign: float) -> List[float]:
    """从纯声学 SimSeg 提取主视角时刻 t_h 处的候选 Δt 列表。"""
    if not sim_segs:
        return []
    min_h = min(float(s["B_st"] if sign > 0 else s["A_st"]) for s in sim_segs)
    max_h = max(float(s["B_en"] if sign > 0 else s["A_en"]) for s in sim_segs)
    if t_h < min_h - 45.0 or t_h > max_h + 45.0:
        return []

    cands: List[float] = [
        float(sign * s["dt_AB"])
        for s in sim_segs
        if ((s["B_st"] if sign > 0 else s["A_st"]) - 15.0) <= t_h <= ((s["B_en"] if sign > 0 else s["A_en"]) + 15.0)
    ]
    has_active_cov = len(cands) > 0
    modes: List[float] = []
    for s in sorted(
        sim_segs,
        key=lambda x: min(
            abs(t_h - (x["B_st"] if sign > 0 else x["A_st"])),
            abs(t_h - (x["B_en"] if sign > 0 else x["A_en"])),
        ),
    ):
        st_s, en_s = (s["B_st"], s["B_en"]) if sign > 0 else (s["A_st"], s["A_en"])
        dist = 0.0 if st_s - 15.0 <= t_h <= en_s + 15.0 else min(abs(t_h - st_s), abs(t_h - en_s))
        is_exp = t_h > en_s + 15.0
        if dist > 650.0 or (has_active_cov and is_exp):
            continue
        if not any(abs(s["dt_AB"] - m_dt) <= 4.0 for m_dt in modes):
            modes.append(float(s["dt_AB"]))
            cands.append(float(sign * s["dt_AB"]))

    unique: List[float] = []
    for dt in cands:
        if not any(abs(dt - u) <= 0.8 for u in unique):
            unique.append(dt)
    return unique


def align_away_speaker_by_dt(
    c_away: Sequence[int],
    target_side: str,
    probe_home_spk: Dict[int, List[Dict[str, Any]]],
    sim_segs: Sequence[Dict[str, Any]],
    ctx_target: Dict[str, Any],
    channels_target: Sequence[np.ndarray],
    adv_target: np.ndarray,
    norm_p_home: np.ndarray,
    act_home: np.ndarray,
    tgt_norm_acts: Sequence[Tuple[np.ndarray, np.ndarray]],
    fg_mask: np.ndarray,
    fg_char_mask: np.ndarray,
    toks_all: Sequence[Sequence[str]],
    onos_all: Sequence[int],
    surfaces_all: Sequence[str],
    clause_tid_seqs: Dict[int, List[int]],
    is_final_pass: bool = True,
    eval_cache: Optional[Dict[Tuple[Any, ...], Tuple[Optional[Dict[str, Any]], List[Tuple[int, int]]]]] = None,
    fg_changed_cs: Optional[np.ndarray] = None,
    line_group: Optional[Sequence[int]] = None,
) -> Tuple[Dict[int, List[Dict[str, Any]]], List[Tuple[float, float, float]]]:
    """在纯声学 SimSeg 提供的 Δt 局部小窗口内对齐隔壁视角子句并保留高置信度命中。"""
    sign = 1.0 if target_side == "A" else -1.0
    tok_dict = _get_tokenizer_dict(ctx_target)
    dur_target = float(channels_target[0].shape[0]) * FRAME_SEC
    out_map: Dict[int, List[Dict[str, Any]]] = {
        ci: [{"token": t, "score": 0.0, "skipped": True, "ono_skipped": True, "line_index": ci} for t in toks_all[ci]]
        for ci in c_away
    }

    if not sim_segs or not c_away:
        return out_map, []

    n_c = len(c_away)
    cand_dt_pool = sorted({round(float(sign * s["dt_AB"]), 2) for s in sim_segs})

    home_times = np.zeros((n_c, 2), dtype=np.float32)
    is_sem = np.zeros(n_c, dtype=bool)
    local_r_lists: List[List[Tuple[int, bool]]] = [[] for _ in range(n_c)]
    for k, ci in enumerate(c_away):
        tids = clause_tid_seqs[ci]
        toks = toks_all[ci]
        surf_str = surfaces_all[ci] if ci < len(surfaces_all) else "".join(toks)
        is_sfx = _is_pure_sfx_clause(surf_str, toks, tok_dict=tok_dict)
        is_sem[k] = (onos_all[ci] == 0 and not is_sfx and len(tids) >= 3)
        inf_h = _clause_summary(probe_home_spk[ci])
        if inf_h["skipped"] or not tids:
            continue
        hst, hen = float(inf_h["st"]), float(inf_h["en"])
        hmid = 0.5 * (hst + hen)
        home_times[k, 0], home_times[k, 1] = hst, hen
        local_dts = _candidate_dts_at_time(hmid, sim_segs, sign)
        cov_dts = [
            float(sign * s["dt_AB"])
            for s in sim_segs
            if ((s["B_st"] if sign > 0 else s["A_st"]) - 15.0) <= hmid <= ((s["B_en"] if sign > 0 else s["A_en"]) + 15.0)
        ]
        for r_idx, dt_cand in enumerate(cand_dt_pool):
            if any(abs(dt_cand - ldt) <= 1.2 for ldt in local_dts):
                is_cov = any(abs(dt_cand - cdt) <= 1.2 for cdt in cov_dts)
                local_r_lists[k].append((r_idx, is_cov))

    def _eval_at_dt(
        k: int, ci: int, hst: float, hen: float, dt_use: float, r_idx: int, is_cov: bool
    ) -> Optional[Dict[str, Any]]:
        if hst + dt_use < -1.0 or hen + dt_use > dur_target + 1.0:
            return None
        ckey = (ci, round(hst, 4), round(hen, 4), round(dt_use, 4), bool(is_sem[k]), bool(is_cov))
        if eval_cache is not None and fg_changed_cs is not None and ckey in eval_cache:
            cached_res, touched_wins = eval_cache[ckey]
            if all(int(fg_changed_cs[w1]) == int(fg_changed_cs[w0]) for (w0, w1) in touched_wins):
                if cached_res is None:
                    return None
                return {**cached_res, "k": k, "r_idx": r_idx, "is_cov": is_cov}

        toks, tids = toks_all[ci], clause_tid_seqs[ci]
        hmid = 0.5 * (hst + hen)
        pad = (3.5 if len(tids) > 4 else 2.2) if is_sem[k] else 1.8
        T_t_len = int(channels_target[0].shape[0])
        touched_wins: List[Tuple[int, int]] = [(
            max(0, int(round((hst + dt_use - pad) / FRAME_SEC))),
            min(T_t_len, int(round((hen + dt_use + pad) / FRAME_SEC))),
        )]

        def _store_cache(val: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            if eval_cache is not None and fg_changed_cs is None:
                eval_cache[ckey] = (val, touched_wins)
            return val

        vr = slice_ctc_verify_clause(
            toks, ctx_target, channels_target, adv_target, hst + dt_use, hen + dt_use, pad,
            fg_mask, fg_char_mask, tids, build_tokens=False,
        )
        if not vr["matched"]:
            return _store_cache(None)
        st_v, en_v = float(vr["start_sec"]), float(vr["end_sec"])
        dto = 0.5 * (st_v + en_v) - hmid
        drift = abs(dto - dt_use)
        sc = float(vr["score"])
        sc_bal = float(vr["sc_bal"])
        adv = float(vr["adv"])
        fov = float(vr["fg_overlap"])
        dur_ratio = float(vr["dur_ratio"])
        q_bal = sc_bal + 0.45 * adv
        dur_pen = 0.25 * abs(float(np.log(max(0.15, dur_ratio))))
        drift_pen = 0.65 * max(0.0, drift - 0.65)
        ph_s = _active_phoneme_sim(norm_p_home, act_home, tgt_norm_acts, hst, hen, dto) if is_sem[k] else 0.0
        q_eff = q_bal - 2.0 * fov - dur_pen - drift_pen + 1.5 * max(0.0, ph_s - 0.30)

        if is_sem[k]:
            if fov >= 0.85 or adv < 3.20 or q_bal < -3.95 or sc_bal < -6.00:
                return _store_cache(None)
            # 标准 1：SimSeg 同步漂移硬截断（禁止在 ±3.5s 宽窗内大漂移碰瓷前景角色）
            max_drift = 1.35 if is_cov else 2.20
            if drift > max_drift and not (sc_bal >= -2.35 and ph_s >= 0.42):
                return _store_cache(None)
            # 标准 2：跨视角音素同源度兜底（低同源 + 低声学分 + 有前景重叠或偏离同步点 -> 碰瓷假阳性）
            if ph_s < 0.22 and sc_bal < -3.60 and (fov >= 0.30 or drift >= 0.75):
                return _store_cache(None)
        else:
            if fov > 0.45 or q_bal < -1.20 or adv < 2.30 or drift > 1.35:
                return _store_cache(None)

        ctrl_qs: List[float] = []
        for sh in (-37.0, -21.0, 21.0, 37.0):
            if 2.0 <= hst + dto + sh and hen + dto + sh <= dur_target - 2.0:
                touched_wins.append((
                    max(0, int(round((hst + dto + sh - 2.0) / FRAME_SEC))),
                    min(T_t_len, int(round((hen + dto + sh + 2.0) / FRAME_SEC))),
                ))
                vc = slice_ctc_verify_clause(
                    toks, ctx_target, channels_target, adv_target, hst + dto + sh, hen + dto + sh, 2.0,
                    fg_mask, fg_char_mask, tids, build_tokens=False,
                )
                if vc["matched"]:
                    ctrl_qs.append(float(vc["sc_bal"]) + 0.45 * float(vc["adv"]))
        q_null = float(np.median(ctrl_qs)) if ctrl_qs else -3.5
        gain = q_bal - q_null

        if is_sem[k]:
            solo_clear = (sc_bal >= -4.55 and fov < 0.70 and (gain >= -0.20 or sc_bal >= -3.80))
            qual_ok = (q_eff >= -3.60 and (gain >= 0.20 or adv >= 4.70 or (ph_s >= 0.38 and gain >= -0.25)))
            if not (solo_clear or qual_ok):
                return _store_cache(None)
        else:
            if gain < 0.30:
                return _store_cache(None)

        res = {
            "k": k, "ci": ci, "r_idx": r_idx, "dt_used": float(dt_use), "is_cov": is_cov,
            "st": st_v, "en": en_v, "hmid": hmid, "dt": dto, "sc": sc, "sc_bal": sc_bal,
            "adv": adv, "dur_ratio": dur_ratio, "q": q_eff, "gain": gain, "fg_ov": fov,
            "ph_sim": ph_s, "is_sem": bool(is_sem[k]),
        }
        return _store_cache(res)

    cands_by_k_r: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for k, ci in enumerate(c_away):
        hst, hen = float(home_times[k, 0]), float(home_times[k, 1])
        if hen <= hst:
            continue
        for r_idx, is_cov in local_r_lists[k]:
            it = _eval_at_dt(k, ci, hst, hen, cand_dt_pool[r_idx], r_idx, is_cov)
            if it is not None:
                cands_by_k_r[(k, r_idx)] = it

    strong_hits = {(k, r): it for (k, r), it in cands_by_k_r.items() if it["is_sem"] and it["q"] >= -2.50}
    for k, ci in enumerate(c_away):
        hst, hen = float(home_times[k, 0]), float(home_times[k, 1])
        if hen <= hst:
            continue
        hmid = 0.5 * (hst + hen)
        for r_idx, is_cov in local_r_lists[k]:
            dt_cand = cand_dt_pool[r_idx]
            neigh_dts = [
                strong_hits[(j, r_idx)]["dt"]
                for j in range(max(0, k - 10), min(n_c, k + 11))
                if (j, r_idx) in strong_hits
                and abs(strong_hits[(j, r_idx)]["hmid"] - hmid) <= 45.0
                and abs(strong_hits[(j, r_idx)]["dt"] - dt_cand) <= 2.0
            ]
            if len(neigh_dts) < 2:
                continue
            dt_loc = float(np.median(neigh_dts))
            if abs(dt_loc - dt_cand) <= 0.25 or abs(dt_loc - dt_cand) > 2.0:
                continue
            it_loc = _eval_at_dt(k, ci, hst, hen, dt_loc, r_idx, is_cov)
            if it_loc is not None:
                it_prev = cands_by_k_r.get((k, r_idx))
                if it_prev is None or it_loc["q"] >= (it_prev["q"] - 0.1):
                    cands_by_k_r[(k, r_idx)] = it_loc

    def _chrono_ok(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        if not (a["st"] + 0.06 <= b["st"] and a["en"] <= b["en"] + 0.15):
            return False
        ov = min(a["en"], b["en"]) - max(a["st"], b["st"])
        if ov <= 0.0:
            return True
        min_dur = max(0.10, min(a["en"] - a["st"], b["en"] - b["st"]))
        return ov <= 0.50 * min_dur

    def _dedup_and_lis(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not items:
            return []
        best_unique: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
        for it in items:
            key = (it["k"], round(it["st"] * 20.0), round(it["en"] * 20.0))
            prev = best_unique.get(key)
            if prev is None or it["q"] > prev["q"]:
                best_unique[key] = it
        cand_list = sorted(best_unique.values(), key=lambda x: (x["k"], x["st"], x["r_idx"]))
        m_p = len(cand_list)
        w_base = np.zeros(m_p, dtype=np.float64)
        for i, it in enumerate(cand_list):
            w_base[i] = max(0.5, float(len(clause_tid_seqs[it["ci"]]))) + min(0.9, 0.10 * max(0.0, 5.0 + it["q"]))
        dp_w = w_base.copy()
        dp_bp = np.full(m_p, -1, dtype=np.int32)
        for i in range(m_p):
            for j in range(i):
                if (
                    cand_list[j]["k"] < cand_list[i]["k"]
                    and _chrono_ok(cand_list[j], cand_list[i])
                    and dp_w[j] + w_base[i] > dp_w[i]
                ):
                    dp_w[i], dp_bp[i] = dp_w[j] + w_base[i], j
        chosen_idx: List[int] = []
        cm = int(np.argmax(dp_w))
        while cm >= 0:
            chosen_idx.append(cm)
            cm = int(dp_bp[cm])
        chosen_idx.reverse()
        return [cand_list[i] for i in chosen_idx]

    kept_sem_raw = _dedup_and_lis([it for it in cands_by_k_r.values() if it["is_sem"]])
    # ── P0-2: 客视角段级/滑窗存在密度门控 + 孤立碎词防幽灵 ───────────────────
    # 当某客视角大段命中密度极低（如不在场/完全隔音段里零星碰瓷 3~5 句）且无局部集群支撑时，整段抑制。
    sem_ks_all = [k for k in range(n_c) if is_sem[k] and home_times[k, 1] > home_times[k, 0]]
    sem_k_to_pos = {k: idx for idx, k in enumerate(sem_ks_all)}
    hit_k_map = {it["k"]: it for it in kept_sem_raw}

    def _is_strong_sync_anchor(it_x: Dict[str, Any]) -> bool:
        dr = abs(float(it_x["dt"]) - float(it_x["dt_used"]))
        return dr <= 0.90 and (float(it_x["ph_sim"]) >= 0.30 or float(it_x["sc_bal"]) >= -2.60)

    kept_sem: List[Dict[str, Any]] = []
    for idx_k, it in enumerate(kept_sem_raw):
        n_tids = len(clause_tid_seqs[it["ci"]])
        prev_gap = (it["st"] - kept_sem_raw[idx_k - 1]["en"]) if idx_k > 0 else 999.0
        next_gap = (kept_sem_raw[idx_k + 1]["st"] - it["en"]) if idx_k + 1 < len(kept_sem_raw) else 999.0
        isolated_6s = (prev_gap > 6.0 and next_gap > 6.0)
        has_cluster_12s = (prev_gap <= 12.0 or next_gap <= 12.0)

        # 标准 4：定宽对称滑窗“强同源锚点”密度门控（25 实词句窗内无 >=2 个强同步锚点且密度 < 16% -> 整段抑制）
        pos_s = sem_k_to_pos.get(it["k"], -1)
        if pos_s >= 0:
            n_sem_all = len(sem_ks_all)
            lo_p = max(0, min(pos_s - 12, n_sem_all - 25))
            hi_p = min(n_sem_all, lo_p + 25)
            hmid_cur = it["hmid"]
            win_ks = [
                sem_ks_all[p] for p in range(lo_p, hi_p)
                if abs(0.5 * (float(home_times[sem_ks_all[p], 0]) + float(home_times[sem_ks_all[p], 1])) - hmid_cur) <= 120.0
            ]
            win_tot = len(win_ks)
            win_items = [hit_k_map[wk] for wk in win_ks if wk in hit_k_map]
            win_hits = len(win_items)
            win_strong_sync = sum(1 for wi in win_items if _is_strong_sync_anchor(wi))
            density = (win_hits / float(win_tot)) if win_tot > 0 else 1.0
            is_ultra_clear_solo = (it["sc_bal"] >= -1.85 and it["fg_ov"] < 0.15 and it["gain"] >= 1.40)
            if win_tot >= 12 and density < 0.16 and (win_strong_sync < 2 or not has_cluster_12s):
                if not is_ultra_clear_solo:
                    continue

        if isolated_6s:
            if n_tids <= 4 and not (it["sc_bal"] >= -2.40 and it["fg_ov"] < 0.20):
                continue
            if n_tids <= 6 and it["fg_ov"] >= 0.45 and it["sc_bal"] < -2.60:
                continue
        kept_sem.append(it)

    # ── P0-1: 客视角行级连坐 / 整句补全（消除断肢残句，如「他の人の……」「ただの……」「目で……」）──
    rescued_map: Dict[int, Tuple[float, float, float]] = {}  # ci -> (st_proj, en_proj, dt_line)
    if is_final_pass and line_group is not None and kept_sem:
        lg_to_ks: Dict[int, List[int]] = {}
        for k, ci in enumerate(c_away):
            if is_sem[k] and home_times[k, 1] > home_times[k, 0] and ci < len(line_group):
                lg_to_ks.setdefault(int(line_group[ci]), []).append(k)
        kept_by_lg: Dict[int, List[Dict[str, Any]]] = {}
        for it in kept_sem:
            ci = it["ci"]
            if ci < len(line_group):
                kept_by_lg.setdefault(int(line_group[ci]), []).append(it)

        drop_cis: set = set()
        # 已有保留句的时间顺序索引（用于投影单调性检查）
        kept_by_k = {it["k"]: it for it in kept_sem}
        kept_ks_sorted = sorted(kept_by_k.keys())

        for lg_id, lg_ks in lg_to_ks.items():
            hits_lg = kept_by_lg.get(lg_id)
            if not hits_lg or len(lg_ks) < 2:
                continue
            tot_tids_lg = sum(len(clause_tid_seqs[c_away[k]]) for k in lg_ks)
            hit_tids_lg = sum(len(clause_tid_seqs[it["ci"]]) for it in hits_lg)
            hit_ratio = hit_tids_lg / max(1.0, float(tot_tids_lg))
            best_q = max(float(it["q"]) for it in hits_lg)
            best_sc = max(float(it["sc_bal"]) for it in hits_lg)
            min_drift_lg = min(abs(float(it["dt"]) - float(it["dt_used"])) for it in hits_lg)
            dt_line = float(np.median([float(it["dt"]) for it in hits_lg]))

            # 标准 3：严控整句补全锚点（必须至少含 1 个低漂移 <= 0.95s 的可靠锚点，杜绝烂分大漂移幽灵“买一送二”）
            can_rescue_line = (min_drift_lg <= 0.95) and (
                len(hits_lg) >= 2
                or (hit_ratio >= 0.35 and best_q >= -3.10 and best_sc >= -3.80)
                or (best_q >= -2.35 and best_sc >= -3.50)
            )
            if can_rescue_line:
                hit_ks_lg = {it["k"] for it in hits_lg}
                for km in lg_ks:
                    if km in hit_ks_lg:
                        continue
                    ci_m = c_away[km]
                    hst_m, hen_m = float(home_times[km, 0]), float(home_times[km, 1])
                    # 必须与同行已命中子句在主视角紧邻（间隙 <= 3.5s），防跨剪辑跳切点
                    min_hgap = min(
                        max(0.0, max(hst_m - float(home_times[kh, 1]), float(home_times[kh, 0]) - hen_m))
                        for kh in hit_ks_lg
                    )
                    if min_hgap > 3.5:
                        continue
                    st_p, en_p = hst_m + dt_line, hen_m + dt_line
                    if st_p < -0.10 or en_p > dur_target + 0.15 or en_p <= st_p + 0.05:
                        continue
                    st_p = max(0.0, st_p)
                    en_p = min(dur_target, en_p)
                    # 检查与前后保留句不倒序
                    prev_ks = [pk for pk in kept_ks_sorted if pk < km]
                    next_ks = [nk for nk in kept_ks_sorted if nk > km]
                    if prev_ks and st_p < kept_by_k[prev_ks[-1]]["st"] + 0.03:
                        continue
                    if next_ks and st_p > kept_by_k[next_ks[0]]["st"] - 0.03:
                        continue
                    rescued_map[ci_m] = (st_p, en_p, dt_line)
            else:
                # 整句连坐放弃：一行含多子句却仅剩 1 个极短弱残词（占整行音素 < 30%），避免吐出断肢废词
                if len(hits_lg) == 1 and len(clause_tid_seqs[hits_lg[0]["ci"]]) <= 5 and hit_ratio < 0.30:
                    drop_cis.add(hits_lg[0]["ci"])

        if drop_cis:
            kept_sem = [it for it in kept_sem if it["ci"] not in drop_cis]

    sfx_valid = [
        it for it in cands_by_k_r.values()
        if not it["is_sem"]
        and any(abs(it["st"] - a["st"]) <= 8.0 for a in kept_sem)
        and all(_chrono_ok(a, it) if a["k"] < it["k"] else _chrono_ok(it, a) for a in kept_sem)
    ]
    kept_all = sorted(kept_sem + _dedup_and_lis(sfx_valid), key=lambda x: x["k"])
    raw_hits: List[Tuple[float, float, float]] = []
    for it in kept_all:
        ci = it["ci"]
        if is_final_pass:
            hst, hen = float(home_times[it["k"], 0]), float(home_times[it["k"], 1])
            pad = (3.5 if len(clause_tid_seqs[ci]) > 4 else 2.2) if it["is_sem"] else 1.8
            vr_tok = slice_ctc_verify_clause(
                toks_all[ci], ctx_target, channels_target, adv_target,
                hst + it["dt_used"], hen + it["dt_used"], pad, fg_mask, fg_char_mask,
                clause_tid_seqs[ci], build_tokens=True,
            )
            if vr_tok["matched"]:
                tok_out = []
                for r in vr_tok["tokens"]:
                    r2 = dict(r)
                    r2["line_index"] = ci
                    tok_out.append(r2)
                out_map[ci] = tok_out
        if it["is_sem"]:
            raw_hits.append((it["st"], it["en"], it["sc"]))

    # 将整句救回的子句按主视角精准时间戳 + dt_line 直接投影写入 out_map
    if is_final_pass and rescued_map:
        for ci_r, (st_p, en_p, dt_line) in rescued_map.items():
            src_toks = probe_home_spk.get(ci_r)
            if not src_toks:
                continue
            proj_toks: List[Dict[str, Any]] = []
            for r in src_toks:
                r2 = dict(r)
                r2["line_index"] = ci_r
                if not r2.get("skipped") and r2.get("original_start") is not None and r2.get("original_end") is not None:
                    os_t = max(0.0, min(dur_target, float(r2["original_start"]) + dt_line))
                    oe_t = max(os_t + 0.02, min(dur_target, float(r2["original_end"]) + dt_line))
                    r2["original_start"] = os_t
                    r2["original_end"] = oe_t
                    r2["start"] = _format_time(os_t)
                    r2["end"] = _format_time(oe_t)
                proj_toks.append(r2)
            out_map[ci_r] = proj_toks
    return out_map, raw_hits


