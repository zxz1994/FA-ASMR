# -*- coding: utf-8 -*-
"""FA-ASMR V2 — 强制对齐主引擎（可选接入冻结 E30 V4 ASR 守门链路）"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
import torchaudio

from align_acoustic import (
    compute_frame_rms_triple,
    compute_track_noise_floor,
    make_silence_mask,
    stereo_ctc_better_ear_log_mixture,
    build_head_token_silence_penalties,
    build_next_stable_silence_end,
    FRAME_SEC,
)
from align_model import (
    compute_emission,
    clear_model_cache,
)
from align_utils import (
    _canonicalize_line_tokens,
    _build_head_standalone_n_flag,
    find_subclause_ii_folds,
    _format_time,
)
from align_post import _post_process_tokens
from align_trellis import ConstrainedCTCAligner
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent
# 生产使用的 LoRA 权重。v9/E10 为 18 个候选模型中下游实测最优
# （消融条件下 <=500ms 79.2% / MAE 0.543，四项指标综合最好）。
# 旧基线 e30 保留为 lora/fa_asmr_e30.pt（及 .bak），可随时切回。
DEFAULT_LORA_NAME = "fa_asmr_v9e10.pt"
DEFAULT_LORA_PATH = ROOT_DIR / "lora" / DEFAULT_LORA_NAME


def align_audio_with_text(
    audio_file_path: str | Path,
    text_tokens: List[List[str]] | List[str],
    sr: Optional[int] = None,
    lora_path: Optional[str | Path] = None,
    device: str = "auto",
    chunk_frames: int = 6000,
    enable_silence_snap: bool = True,
    log_callback: Optional[Any] = None,
    precomputed_emission: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    blank_punct_mask: Optional[Any] = None,
    line_group: Optional[Any] = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """FA-ASMR V2 核心入口：双声道 Better-Ear 纯声学强制对齐。

    参数:
        audio_file_path: 音频路径（支持双声道 WAV/FLAC/MP3 等）
        text_tokens: 台本注音 token 列表（每行一个 token 列表）
        lora_path: LoRA 权重路径（默认加载 align.DEFAULT_LORA_PATH）
        device: "auto" / "cuda" / "cpu"
        line_group: 可选各对齐子行对应的原始台本行序号 (用于短句向心 Blank 判定与归组)
    """
    t0 = time.time()
    bundle = torchaudio.pipelines.MMS_FA
    tokenizer = bundle.get_tokenizer()
    blank = 0
    # 同一原始行内的短子句边界 Blank 固定惩罚 (subclause_blank_*)。
    # 默认已关闭: 该机制在 v9e10 权重下对整体 MAE 无正贡献 (关掉后 MAE 反而略降)。
    # 传 enable_subclause_blank=True 可恢复原行为。
    _enable_subclause_blank = bool(kwargs.get("enable_subclause_blank", False))
    aligner = ConstrainedCTCAligner(
        blank=blank,
        subclause_blank_penalty=(
            float(kwargs.get("subclause_blank_penalty", 0.080))
            if _enable_subclause_blank else 0.0),
        subclause_blank_free_sec=float(kwargs.get("subclause_blank_free_sec", 1.0)),
        subclause_blank_max_tokens=int(kwargs.get("subclause_blank_max_tokens", 3)),
        # 词内 Blank 软降权。原值 0.015 是按旧权重 e30 的 log 尺度标定的；
        # v9e10 的 emission 尺度约为 e30 的 6.5 倍，同值惩罚的相对影响力被稀释
        # 到 0.22%。按尺度比例补偿到 0.10（实测最优点，MAE 0.518->0.510）。
        intra_blank_penalty=float(kwargs.get("intra_blank_penalty", 0.10)),
        n_pure_continuation_penalty=float(kwargs.get("n_pure_continuation_penalty", 0.20)),
        n_pure_continuation_lookahead_frames=int(kwargs.get("n_pure_continuation_lookahead_frames", 4)),
        n_pure_continuation_drop_threshold=float(kwargs.get("n_pure_continuation_drop_threshold", 1.0)),
        n_pure_continuation_drop_ramp=float(kwargs.get("n_pure_continuation_drop_ramp", 2.0)),
        frame_sec=FRAME_SEC,
    )
    wildcard = tokenizer.dictionary.get("*")
    ignored_classes = (() if wildcard is None else (int(wildcard),))

    # 1. 加载音频
    if isinstance(audio_file_path, (str, Path)):
        waveform, sample_rate = torchaudio.load(str(audio_file_path))
    else:
        waveform = torch.as_tensor(audio_file_path).float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        sample_rate = sr or bundle.sample_rate

    # 2. 发射提取
    #   注意: 必须先判 None/空值 —— str(None).upper() == "NONE" 会误命中禁用分支。
    if lora_path is None or str(lora_path).strip() == "":
        resolved_lora = str(DEFAULT_LORA_PATH) if DEFAULT_LORA_PATH.is_file() else None
    elif str(lora_path).upper() in ("NO_LORA", "NONE", "DISABLE"):
        resolved_lora = None
    else:
        resolved_lora = lora_path
    if precomputed_emission is not None:
        raw_L, raw_R = precomputed_emission
    else:
        raw_L, raw_R = compute_emission(
            waveform, sample_rate, device=device, lora_path=resolved_lora, chunk_frames=chunk_frames
        )

    emission_L = torch.log_softmax(raw_L.float(), dim=-1)
    emission_R = torch.log_softmax(raw_R.float(), dim=-1)

    # 3. 左右声道声学先验与 Better-Ear 融合
    audio_np = waveform.numpy().T if waveform.ndim == 2 else np.column_stack([waveform.numpy(), waveform.numpy()])
    if audio_np.shape[1] < 2:
        audio_np = np.column_stack([audio_np[:, 0], audio_np[:, 0]])

    total_frames = int(emission_L.shape[1])
    # 注意: 这里必须用原始 sample_rate（通常是 96kHz），而不是 bundle.sample_rate。
    # emission 帧数与「原始采样率 + 20ms hop」自洽（total_frames * hop == 音频长度），
    # 换成重采样后的 16kHz 会让窗口网格与帧数错位。
    #
    # 传入 (samples, 2) 交织缓冲而非两个声道切片: 切片是 stride=8 的跨步视图,
    # numba 无法向量化(实测慢 3.8x), 而拷贝又要多读写一份与音频等大的内存。
    # 本次调用同时算出左 / 右 / 单声道三路逐帧 RMS, 省掉两遍全轨扫描。
    rms_l, rms_r, mono_rms = compute_frame_rms_triple(
        audio_np, sample_rate, total_frames)
    floor_l = compute_track_noise_floor(rms_l)
    floor_r = compute_track_noise_floor(rms_r)

    mask_l = make_silence_mask(rms_l, floor_l)
    mask_r = make_silence_mask(rms_r, floor_r)

    # 能量解耦掩码：判定清晰语音帧。在低能量双耳底噪区，词内 Blank 成本完全归零，消除反向撕扯
    silence_active = mask_l & mask_r
    speech_frame_mask = (~silence_active).astype(np.uint8)

    # smoothing_frames 由 25 提到 75：v9e10 的两声道 log-odds 差异幅度约为 e30 的
    # 3 倍，25 帧平滑不足以致稳融合权重。实测 75 帧为局部最优
    # （MAE 0.518->0.511、P90 1.65->1.58，两次独立复现一致）。
    stereo_emission = stereo_ctc_better_ear_log_mixture(
        emission_L, emission_R, blank=blank, ignored_classes=ignored_classes,
        smoothing_frames=75, temperature=1.5
    )

    # 4. 构建台本 CTC targets（纯净 1:1 映射，无重复词展开）
    lines = []
    targets = []
    head_standalone_n_flag = []
    # 易错拟声掩码: 输入是【子句级】flags (来自 GUI), 此处按每个子句展开后的
    # 音素 id 数扩展到音素级, 长度与 flat_tokens 严格一致。
    _ono_clause_flags = kwargs.get("ono_clause_flags")
    ono_flags = []   # 音素级 0/1 掩码
    # 与 ConstrainedCTCAligner 的 nasal_token_id 默认值必须一致（MMS-FA 中 n == 4）。
    _nasal_token_id = 4
    _vowel_ids = {tokenizer.dictionary[v] for v in "aiueo"}
    for li, line_tokens in enumerate(text_tokens):
        if isinstance(line_tokens, str):
            line_tokens = line_tokens.split()
        ii_folds = find_subclause_ii_folds(line_tokens)
        c_toks, single_line_ids, kept_idx, token_id_ranges = _canonicalize_line_tokens(
            line_tokens, tokenizer, blank, fold_subclause_ii=True
        )
        if not c_toks:
            c_toks = ["a"]
            single_line_ids = [1]
            kept_idx = []
            token_id_ranges = [(0, 1)]

        targets.append(single_line_ids)
        # 易错拟声掩码: 本子句 flag 展开到其音素 id 序列长度。
        if _ono_clause_flags is not None and li < len(_ono_clause_flags):
            _f = 1 if _ono_clause_flags[li] else 0
        else:
            _f = 0
        ono_flags.extend([_f] * len(single_line_ids))
        # 行首独立 ん 判定。语义与 C->V 跨度无关，故直接用本行的 label 跨度判定，
        # 不与任何 C->V 约束数组混用（混用会导致关闭该约束时连带误伤本判定）。
        head_standalone_n_flag.append(_build_head_standalone_n_flag(
            single_line_ids, token_id_ranges, _vowel_ids, _nasal_token_id))
        lines.append({
            "line_index": li,
            "tokens": c_toks,
            # 原始 token 数与回映射下标：保证 _post_process_tokens 输出条数与
            # 输入 text_tokens 严格一致（被过滤的 token 也要占一条，否则错位）
            "n_raw": len(line_tokens),
            "raw_tokens": list(line_tokens),
            "orig_idx": list(kept_idx),
            "token_id_ranges": list(token_id_ranges),
            "ii_folds": ii_folds,
        })

    # 5. Global CTC 解码。句首保护由路径中的前句末 token、静音前最后声学岛
    # 与稳定静音帧共同判定，直接参与首轮全局 Viterbi 解码。
    emission_for_align = stereo_emission[0].cpu()
    # 关闭时必须传"长度匹配的全 0 数组"而非 None：align_trellis 把 None 转成
    # np.empty(0) 后交给 numba，会越界段错误(0xC0000005，已实测复现)。全 0 数组
    # 走 has_=True 分支但不产生任何扣分，等价于关闭。
    _flat_n = sum(len(line) for line in targets)
    _n_frames = int(emission_for_align.shape[0])
    head_silence_end = np.zeros(_n_frames, dtype=np.int32)
    head_silence_penalty = np.zeros(_flat_n, dtype=np.float32)
    n_adj = 0
    if kwargs.get("enable_boundary_discount", True):
        stereo_rms = np.maximum(rms_l, rms_r)
        stereo_floor = compute_track_noise_floor(stereo_rms, percentile=10.0)
        # 毛刺 veto 默认关闭：CTC 所选的前句末 token 边界不是能量峰值，
        # 不应用它到后方静音间的上升来否定一个真实稳定静音。
        max_spike_db = kwargs.get("boundary_max_spike_db", None)
        head_silence_end = build_next_stable_silence_end(
            stereo_rms,
            stereo_floor,
            max_spike_db=(None if max_spike_db is None else float(max_spike_db)),
            require_pre_silence_island=bool(
                kwargs.get("boundary_require_pre_silence_island", True)),
            island_active_margin_db=float(
                kwargs.get("boundary_island_active_margin_db", 4.0)),
            island_smooth_frames=int(kwargs.get("boundary_island_smooth_frames", 5)),
            island_bridge_frames=int(kwargs.get("boundary_island_bridge_frames", 5)),
            island_lookback_frames=int(kwargs.get("boundary_island_lookback_frames", 50)),
            island_prepad_frames=int(kwargs.get("boundary_island_prepad_frames", 4)),
        )
        head_silence_penalty, n_adj = build_head_token_silence_penalties(
            targets=targets,
            line_group=line_group,
            # 句首末端保护强度。原值 1.0 在旧权重 e30 上标定；v9e10 的 emission
            # 尺度约为其 6.5 倍，同值折扣的相对力度被稀释。实测 4.0 为拐点
            # （MAE 0.504->0.503、P90 1.55->1.54），8.0 无额外增益。
            discount=float(kwargs.get("boundary_discount", 4.0)),
        )

    grouped_spans = aligner(
        emission_for_align,
        targets,
        blank_punct_mask=blank_punct_mask,
        speech_frame_mask=speech_frame_mask,
        line_group=line_group,
        head_token_silence_end=head_silence_end,
        head_token_silence_penalty=head_silence_penalty,
        head_standalone_n_flag=head_standalone_n_flag,
        ono_token_mask=ono_flags,
        ono_score_scale=float(kwargs.get("ono_score_scale", 1.0)),
    )
    if n_adj > 0 and log_callback:
        log_callback(
            f"  [声学保护] 首轮全局 CTC 已启用 {n_adj} 处"
            "句首末端声学岛-稳定静音保护")
    # 6. 纯净后处理与格式化
    # mono_rms 已在第 3 步与左右声道融合计算（同一次遍历），此处不再重复扫描全轨。
    mono_floor = compute_track_noise_floor(mono_rms)
    silence_thresh = min(mono_floor + 4.0, -45.0)

    results = _post_process_tokens(
        grouped_spans=grouped_spans,
        lines=lines,
        frame_duration=FRAME_SEC,
        frame_rms_db=mono_rms,
        silence_threshold_db=silence_thresh,
        enable_silence_snap=enable_silence_snap,
    )

    for r in results:
        if r.get("original_start") is not None:
            r["start"] = _format_time(r["original_start"])
            r["end"] = _format_time(r["original_end"])

    elapsed = time.time() - t0
    logger.info(f"FA-ASMR V2 对齐完成，耗时 {elapsed:.2f}s，共处理 {len(lines)} 行。")
    if log_callback:
        log_callback(f"FA-ASMR V2 对齐完成 ({elapsed:.2f}s)")

    # 可选声学诊断包：供下游「拟声词守门器」做声学证伪。
    # 仅在调用方显式传入可变 dict（如 {}）时填充，默认零开销、不破坏契约。
    diag = kwargs.get("diagnostics_out")
    if isinstance(diag, dict):
        diag["mono_rms_db"] = mono_rms
        diag["mono_noise_floor_db"] = float(mono_floor)
        diag["frame_sec"] = float(FRAME_SEC)

    return results


align_stereo_audio = align_audio_with_text

