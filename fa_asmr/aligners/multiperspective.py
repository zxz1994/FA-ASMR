# -*- coding: utf-8 -*-
"""FA-ASMR V2 — 多视角联合对齐与动态融合引擎（门面重导出）。

核心实现已按职责无环拆分至 `fa_asmr/aligners/`：
- `fa_asmr.aligners.multipersp_acoustic`：声学特征工具、Numba CTC 验证核与纯声学同步段对齐
- `fa_asmr.aligners.multipersp_segments`：主/客视角归属切块、并发碰撞修复与 Δt 局部检索
- `fa_asmr.aligners.multipersp_pipeline`：主编排管线、视角时间轴投影与联合对齐入口
"""
from __future__ import annotations

from fa_asmr.aligners.align_acoustic import FRAME_SEC  # noqa: F401
from fa_asmr.aligners.align_utils import _format_time  # noqa: F401
from fa_asmr.text.clause import F  # noqa: F401
from fa_asmr.aligners.multipersp_acoustic import (  # noqa: F401
    _CACHED_TOKENIZER_DICT,
    _EMPTY_F2D,
    _HAS_NUMBA,
    _NUMBA_WARMED_UP,
    _AudioSoloChecker,
    _active_phoneme_sim,
    _backtrack_ctc_bounds_nb,
    _eval_ctc_channel_nb,
    _extract_audio_feat,
    _get_tokenizer_dict,
    _is_pure_sfx_clause,
    _load_audio_16k_once,
    _load_stereo_16k,
    _njit_cache,
    _norm_spk,
    _phoneme_cos_sim,
    _prep_phoneme_probs,
    _run_ctc_dp_nb,
    _unmasked_clause_mean_score,
    compute_phoneme_advantage_matrix,
    detect_clean_audio_segments,
    ensure_numba_warmed_up,
    slice_ctc_verify_clause,
)
from fa_asmr.aligners.multipersp_segments import (  # noqa: F401
    _build_fg_masks,
    _candidate_dts_at_time,
    align_away_speaker_by_dt,
    refine_home_segments,
    segment_and_assign_home,
)
from fa_asmr.aligners.multipersp_pipeline import (  # noqa: F401
    align_multiperspective_joint,
    project_result_list_to_perspective,
    run_multiperspective_pipeline,
)
