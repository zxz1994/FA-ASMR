# -*- coding: utf-8 -*-
"""FA-ASMR V2 — 独立的多人分轨混音 / 台本乱序 / 拟声与台词重叠对齐引擎（门面重导出）。

核心实现已按职责无环拆分至 `fa_asmr/aligners/` 与 `fa_asmr/pipeline/`：
- `fa_asmr.aligners.multispeaker_decoder`：Viterbi 跳过弧解码器与 `MultiSpeakerCTCAligner`
- `fa_asmr.aligners.multispeaker_acoustic`：全轨声学特征提取、STFT 单声源检测与浅层声纹判别
- `fa_asmr.aligners.multispeaker_schedule`：双耳协同、碰撞修复、乱序抢救与 `align_multispeaker_or_reordered`
- `fa_asmr.aligners.multispeaker_post`：多流分流独立后处理与时间轴归并 (`post_process_and_merge_streams`)
- `fa_asmr.pipeline.pairing` / `fa_asmr.pipeline.subtitles`：角色标题归一化与多角色独立 SRT 导出
"""
from __future__ import annotations

from fa_asmr.aligners.align_acoustic import FRAME_SEC  # noqa: F401
from fa_asmr.aligners.align_utils import _format_time  # noqa: F401
from fa_asmr.text.clause import F  # noqa: F401
from fa_asmr.aligners.multispeaker_decoder import (  # noqa: F401
    _HAS_NUMBA,
    MultiSpeakerCTCAligner,
    _viterbi_multispeaker_duration,
)
from fa_asmr.aligners.multispeaker_acoustic import (  # noqa: F401
    _AudioSoloChecker,
    _SpeakerVoiceprintJudge,
    _align_clauses_with_context,
    _align_subset_clauses,
    _clause_summary,
    _estimate_speaker_ear_bias,
    _group_results_by_clause,
    _rescue_out_of_order_short_clauses,
    _resolve_same_text_swaps,
    _unmasked_clause_mean_score,
    build_cooperative_stereo_emission,
    extract_acoustic_context,
)
from fa_asmr.aligners.multispeaker_schedule import (  # noqa: F401
    _build_claimed_acoustic_ctx,
    _build_claimed_emission,
    _build_torn_line_blocks,
    _clause_token_ids,
    _is_anomalous_semantic_clause,
    _refine_solo_collisions,
    _repair_overlapping_clause_blocks,
    _repair_tail_gap_clauses,
    _rescue_out_of_order_clauses,
    _semantic_quality_objective,
    _slide_window_curve,
    _slide_window_peak,
    align_multispeaker_or_reordered,
)
from fa_asmr.aligners.multispeaker_post import (  # noqa: F401
    _borrow_intra_line_timestamps,
    _extract_clause_item_ranges,
    apply_skipped_ono_marks,
    post_process_and_merge_streams,
)
from fa_asmr.pipeline.pairing import (  # noqa: F401
    _ACTION_SUFFIX_RE,
    _NON_SPEAKER_HEADER_RE,
    _STANDALONE_BRACKET_HEADER_RE,
    _clean_speaker_header_name,
    auto_normalize_speaker_headers,
)
from fa_asmr.pipeline.subtitles import (  # noqa: F401
    _OPEN_QUOTE_CHARS,
    _OUTER_QUOTE_PAIRS,
    _PUNCT_CHARS,
    _SPLIT_BOUNDARY_PUNCT,
    _UNISON_WORD_RE,
    _format_srt_cues,
    _format_srt_time,
    _norm_unison_text,
    _parse_lrc_ts_sec,
    _strip_orphan_punct,
    _strip_outer_quotes_text,
    _tokens_display_text,
    _tokens_time_bounds,
    build_independent_multispeaker_srt,
    build_standard_srt,
    split_unison_prefix_suffix_rl,
)
