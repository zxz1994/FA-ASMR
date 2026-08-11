# -*- coding: utf-8 -*-
"""FA-ASMR 强制对齐主模块

本文件只含主入口 align_audio_with_text；其余组件已拆分至:
  - align_model.py : LoRA 适配器 / MMS_FA 模型缓存 / emission 计算
  - align_utils.py : token 文本工具 / span 打分 / OnoWave 波形计数 / 重复型行检测
  - align_post.py  : 后处理 (segment_align / anchor_realign / auto_repeat)
"""

import torch
import torchaudio
import math
import time
import dataclasses
import torch.nn as nn
import torch.nn.functional as F
import numpy as _np
import logging
from typing import Optional, List, Tuple, Dict, Any

logger = logging.getLogger(__name__)

import advanced_config as adv

# ── 从拆分模块导入 ──────────────────────────────────────────────
from align_model import (
    _get_model, clear_model_cache, compute_emission,
    _resolve_device, RebuiltSpan, _CPU_CHUNK_CAP,
)
from align_utils import (
    _sum_score, _pick_best_spans, _hybrid_spans,
    _split_lines, _strip_symbols, _is_symbol_token,
    _is_repeat_line, _ONO_TABLE,
    _ono_count_energy_peak, _ono_count_envelope_autocorr,
    _ono_count_peak_pairs, _ono_count_rms_valley, _ono_count_rms_segment,
)
from align_post import (
    _segment_align, _anchor_realign, _auto_repeat,
)


def align_audio_with_text(audio_file_path, text_tokens, non_silent_ranges=[], sr=None, speed=1,
                         lora_path=None, device="auto", chunk_frames=6000,
                         use_stereo_pick=True,
                         min_token_dur_ms=40, log_callback=None,
                         rate_quantile=0.95, left_pad=0.25, window_min=0.3,
                         segment_align=False, blank_th=0.6, min_sil_frames=30, min_seg_frames=40,
                         emission_align=False, emission_self_loop=2.0,
                         anchor_realign=False, anchor_max_tokens=40,
                         anchor_overlap=2, anchor_pad=12, anchor_gap=6,
                         auto_repeat=True,
                         auto_repeat_burst=False,
                         prep_sandwich=False,
                         line_floor=0,
                         prep_f1=5, prep_f2=3,
                         ono_wave=True,
                         collapse_prune=True,
                         cpu_threads=None, quantize_int8=None,
                         precomputed_emission=None):
    start_time = time.time()

    # 底层系统配置兜底：调用方未显式传入的旋钮，从 exe 同级 advanced_config.json 读取。
    # GUI 已显式传入的值优先；此处仅作为「未指定」时的二级默认值。
    # cpu_threads / quantize_int8 在 _get_model 内统一查 adv，这里只准备分块上限供闭包使用。
    _cpu_chunk_cap = int(adv.get("cpu_chunk_cap", _CPU_CHUNK_CAP))

    if device == "cuda":
        device = torch.device("cuda")
    elif device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        bundle = torchaudio.pipelines.MMS_FA
        if isinstance(audio_file_path, str):
            waveform, sample_rate = torchaudio.load(audio_file_path)
        else:
            waveform = torch.tensor(audio_file_path).float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            sample_rate = sr

        # 立体声全局全动态等比归一化
        max_val = torch.max(torch.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val

        is_stereo = waveform.shape[0] > 1

        # 🔧 模型缓存：批量处理时复用同一份权重，避免每个文件都重载 ~1GB 模型
        # 评测复用模式（precomputed_emission 已给）跳过模型加载与前向，直接喂发射，
        # 从而 global 与 emB 共用同一份发射 → 干净消除 GPU 非确定性造成的跨 config 漂移。
        if precomputed_emission is not None:
            emission_L, emission_R = precomputed_emission
        else:
            global_model = _get_model(bundle, device, lora_path, log_callback,
                                      cpu_threads=cpu_threads, quantize_int8=quantize_int8)

        # =======================================================================
        # 单声道特征提取（显存随产随销防爆版）
        # =======================================================================
        # VAD 段拼接信息（供时间映射使用）：实际保留的段，按真实边界还原时间。
        # 直接首尾相接真实语音段，不插入合成静音——合成零静音会误导 MMS_FA 的
        # CTC 发射（不保证是 blank 峰），反而劣化边界对齐；时间映射仍按真实段边界。
        _seg_ranges = []   # [(start_sec, end_sec), ...] 实际保留段（与拼接顺序一致）
        _seg_gaps = []     # 段间插入静音秒数（当前为 0，直接相接）

        def get_emission_single_channel(single_channel_wf, current_sr):
            nonlocal _seg_ranges, _seg_gaps
            if non_silent_ranges:
                total_samples = single_channel_wf.shape[1]
                dur_orig = total_samples / current_sr * speed
                sample_ranges = []  # (start_sec, start_sample, end_sample, end_sec)
                min_chunk = 320 * 8  # 至少 8 帧(~80ms)才认为区间有效
                for start_sec, end_sec in non_silent_ranges:
                    start_sample = int(start_sec * current_sr / speed)
                    end_sample = min(int(end_sec * current_sr / speed), total_samples)
                    end_sec_eff = min(end_sec, dur_orig)  # 末端可能超出，截断到音频长度
                    if end_sample - start_sample >= min_chunk:
                        sample_ranges.append((start_sec, start_sample, end_sample, end_sec_eff))

                if sample_ranges:
                    # 暴露实际保留段(原始秒)与段间插入静音，供 map_to_original_time 使用
                    _seg_ranges = [(s, e_sec) for (s, _, _, e_sec) in sample_ranges]
                    _seg_gaps = [0.0] * len(sample_ranges)
                    # 拼接：直接首尾相接真实语音段（不插入合成静音）
                    pieces = [single_channel_wf[:, ss:ee] for (_s, ss, ee, _e_sec) in sample_ranges]
                    single_channel_wf = torch.cat(pieces, dim=1)
                else:
                    # VAD 失效(空/亚阈值区间) → 退回整段对齐, 不依赖 VAD
                    logger.warning(" [VAD] 有效语音段为空, 退回整段对齐")
            else:
                pass  # 无 VAD 区间 → 整段对齐

            if current_sr != bundle.sample_rate:
                single_channel_wf = torchaudio.functional.resample(single_channel_wf, current_sr, bundle.sample_rate)

            chunk_frames_eff = chunk_frames
            if str(device) == "cpu" and chunk_frames_eff > _cpu_chunk_cap:
                chunk_frames_eff = _cpu_chunk_cap
                logger.info(f" [CPU] 分块窗口自动收紧为 {chunk_frames_eff} 帧以降低注意力开销")
            chunk_size = 320 * chunk_frames_eff
            context_size = max(320 * 100, chunk_size // 24)  # 自适应上下文
            total_samples = single_channel_wf.shape[1]

            local_emissions = []

            with torch.inference_mode():
                for start in range(0, total_samples, chunk_size):
                    end = min(start + chunk_size, total_samples)

                    keep_frames = math.ceil((end - start) / 320)
                    if keep_frames == 0:
                        continue

                    pad_start = max(0, start - context_size)
                    pad_end = min(total_samples, end + context_size)

                    chunk_audio = single_channel_wf[:, pad_start:pad_end].to(device)
                    emission_chunk, _ = global_model(chunk_audio)

                    left_frames = (start - pad_start) // 320
                    valid_emission = emission_chunk[:, left_frames: left_frames + keep_frames, :]

                    local_emissions.append(valid_emission.cpu())

                    del chunk_audio, emission_chunk, valid_emission

            return torch.cat(local_emissions, dim=1)
        # =======================================================================

        # 特征提取（复用模式已设置 emission，跳过前向）
        if precomputed_emission is None:
            if is_stereo and use_stereo_pick:
                logger.info(" [Smart Stereo] 分时串行特征提取...")
                logger.info(" -> 正在处理 [左声道 L]...")
                emission_L = get_emission_single_channel(waveform[0:1, :], sample_rate)

                logger.info(" -> 正在处理 [右声道 R]...")
                emission_R = get_emission_single_channel(waveform[1:2, :], sample_rate)
            else:
                if is_stereo:
                    # 立体声但不择优 → 合并为单声道
                    waveform = waveform.mean(dim=0, keepdim=True)
                logger.info(" [Mono Mode] 单声道特征提取...")
                emission_L = get_emission_single_channel(waveform, sample_rate)
                emission_R = emission_L

        # 模型保留在缓存中供后续文件复用（见 _get_model），此处不再销毁

        # =======================================================================
        # 避风港：在 CPU 上安全跑 CTC 对齐与级联二次打准星
        # =======================================================================
        logger.info(" [CPU Alignment] 全局对齐与级联锁定...")
        tokenizer = bundle.get_tokenizer()
        aligner = bundle.get_aligner()
        blank = getattr(aligner, "blank", 0)  # MMS_FA 的 blank 索引(=0)

        # =======================================================================
        # 📝 文本预处理：按换行切行 + 剔除纯符号 token（对齐后插回）
        # =======================================================================
        raw_lines, has_sep = _split_lines(text_tokens)
        if not has_sep:
            # 兼容旧调用：无换行标记 → legacy 全局对齐 + post_process
            lines = [{'tokens': list(raw_lines[0]) if raw_lines else [],
                      'symbols': [], 'raw': list(raw_lines[0]) if raw_lines else []}]
            all_align_tokens = list(lines[0]['tokens'])
            line_mode = False
        else:
            lines = []
            all_align_tokens = []
            for raw in raw_lines:
                align, symbols = _strip_symbols(raw)
                lines.append({'tokens': align, 'symbols': symbols, 'raw': raw})
                all_align_tokens.extend(align)
            line_mode = True

        if not all_align_tokens:
            msg = " [Warn] 没有可对齐的 token，返回空结果"
            logger.warning(msg)
            if log_callback:
                log_callback(msg)
            return []

        # tokenize（每行独立 token_ids，区域 CTC 直接复用）
        all_token_ids = []
        gidx = 0
        for li, line in enumerate(lines):
            line['g_start'] = gidx
            line['g_end'] = gidx + len(line['tokens'])
            if line['tokens']:
                vocab = getattr(tokenizer, 'dictionary', None)
                if vocab is not None:
                    _aligned = [list(w) for w in line['tokens']]
                    _filtered = [[c for c in w if c in vocab] for w in _aligned]
                else:
                    _filtered = line['tokens']
                _filtered = [w for w in _filtered if w]
                line['token_ids'] = tokenizer(_filtered) if _filtered else []
            else:
                line['token_ids'] = []
            line['token_ids'] = [[t for t in w if t != blank] for w in line['token_ids'] if w]
            all_token_ids.extend(line['token_ids'])
            gidx = line['g_end']

        # ═══════════════════════════════════════════════════════════════════
        # 📝 pre-expand：第一轮 CTC 前展开重复型行 token
        # 因子来源优先级: 波形检测(ono_wave) > 自适应(prep_f1/f2)
        # ═══════════════════════════════════════════════════════════════════
        _prep_map = {}  # line_idx → (expand_factor, (unit_tokens_tuple, unit_len))
        if line_mode:
            _punct_set = {'.', '…', '！', '!', '？', '?', '、', '。', '♡', '♥', '-', 'ー', '~', '〜'}
            for _pli, _pline in enumerate(lines):
                _is_rep, _ntok, _ulen = _is_repeat_line(_pline['raw'])
                if _is_rep:
                    _fac = prep_f1 if _ulen == 1 else (prep_f2 if _ulen == 2 else 2)
                    # 记录 unit tokens 供 OnoWave 查表
                    _tok_strip = tuple(t for t in _pline['raw'] if t and t != '_' and t not in _punct_set)
                    _unit = _tok_strip[:_ulen] if len(_tok_strip) >= _ulen else ()
                    _prep_map[_pli] = (_fac, _unit, _ulen)
            # 夹心爆发行扩展（prep_sandwich，2026-08-09 A/B 结论：净持平，默认关）
            # Repeat行之间的短 token 喘息行（くりくり[OK]→んふ[塌缩]→くりくり[溢出]）
            # 给中间喘息行 2x token 展开，融入全局 CTC 路径。
            if prep_sandwich:
                _nlines = len(lines)
                for _pli in range(_nlines):
                    if _pli in _prep_map:
                        continue
                    _nt = len(lines[_pli].get("tokens", []))
                    if _nt <= 0 or _nt > 4:
                        continue
                    _near_rep = False
                    for _d in range(max(0, _pli - 2), min(_nlines, _pli + 3)):
                        if _d in _prep_map:
                            _near_rep = True
                            break
                    if _near_rep:
                        _prep_map[_pli] = (2, (), 0)
            if _prep_map:
                _rep_count = {f: sum(1 for v in _prep_map.values() if v[0] == f)
                              for f in sorted(set(v[0] for v in _prep_map.values()))}
                _desc = ", ".join(f"×{f}:{c}" for f, c in sorted(_rep_count.items()))
                logger.info(f" [PreExpand] 自适应展开: {_desc}")
                _orig_all_ids = all_token_ids  # 保存原始版本供回映射后恢复
                _exp_ids = []; _exp_at = []
                for _pli, _pline in enumerate(lines):
                    _fac = _prep_map.get(_pli, (1, (), 0))[0]
                    for _c in range(_fac):
                        _exp_ids.extend(_pline['token_ids'])
                        _exp_at.extend(_pline['tokens'])
                all_token_ids = _exp_ids
                all_align_tokens = _exp_at
                use_prep = True
            else:
                use_prep = False
        else:
            use_prep = False

        frame_duration = 1.0 / bundle.sample_rate * 320 * speed
        fps = 1.0 / frame_duration if frame_duration > 0 else 50.0
        total_frames = emission_L.shape[1]

        def format_time(time_sec):
            minutes, remainder = divmod(time_sec, 60)
            seconds, centiseconds = divmod(remainder, 1)
            return f"[{int(minutes):02d}:{int(seconds):02d}:{math.floor(centiseconds * 100):02d}]"

        def map_to_original_time(adjusted_time):
            if not _seg_ranges:
                return adjusted_time
            cumulative = 0.0
            for i, (s, e) in enumerate(_seg_ranges):
                dur = e - s
                if adjusted_time < cumulative + dur:
                    return s + (adjusted_time - cumulative)
                cumulative += dur
                # 段后插入的静音区：把压缩轴时间比例映射回原始间隙
                if i < len(_seg_ranges) - 1:
                    g = _seg_gaps[i + 1]
                    if g > 0 and adjusted_time < cumulative + g:
                        frac = (adjusted_time - cumulative) / g
                        nxt_s = _seg_ranges[i + 1][0]
                        return e + frac * (nxt_s - e)
                    cumulative += g
            excess_time = adjusted_time - cumulative
            return _seg_ranges[-1][1] + excess_time

        def original_to_adjusted(orig):
            if not _seg_ranges:
                return orig
            cumulative = 0.0
            for i, (s, e) in enumerate(_seg_ranges):
                if s <= orig <= e:
                    return cumulative + (orig - s)
                cumulative += (e - s)
                # 段间原始间隙：比例映射到插入的静音区
                if i < len(_seg_ranges) - 1:
                    g = _seg_gaps[i + 1]
                    gap_start, gap_end = e, _seg_ranges[i + 1][0]
                    if gap_start <= orig <= gap_end:
                        frac = (orig - gap_start) / (gap_end - gap_start) if gap_end > gap_start else 0.0
                        return cumulative + frac * g
                    cumulative += g
            return cumulative + (orig - _seg_ranges[-1][1])

        # =======================================================================
        # 原 region_align(逐行独立窗口 CTC) 已于 2026-08-09 删除：
        # 其"锚定全局 CTC + 单调"版本与 two_pass 行为重合、无增量收益；
        # 其"独立放窗"版本会把重复短语锁到错误出现位置。
        # （two_pass 递推窗口逻辑亦已于后续删除，仅保留整段全局 CTC 主路径。）
        # =======================================================================

        # =======================================================================
        # 方案 B：发射自环时长先验（emission_align=True）
        # 思路：跑两次强制对齐——
        #   (1) 基础对齐（无奖励）：起点精确，但 token 易塌缩成 1 帧；
        #   (2) 奖励对齐：给发射的非 blank 列加 self_loop 自环奖励（等价于
        #       ctc_crf 的 crf_self_loop 在推理端应用），把 token 时长拉到接近
        #       真实，但整体边界前移、累积漂移大。
        # 两者都在 aligner 内部完成（只改喂给 aligner 的发射）。混合：
        #   每词「起点取基础对齐，时长取奖励对齐」→ 起点准 + 不塌缩。
        # 实测：基础 global 中位|e|115ms/时长比0.69；纯奖励中位|e|745ms/时长比0.98；
        #       混合应同时拿到 115ms 与 ~0.98。
        # emission_align 与 segment_align 互斥：segment_align 优先时跳过。
        # =======================================================================
        # 一阶段全局基础路径铺设
        token_spans_L = aligner(emission_L[0], all_token_ids)
        token_spans_R = aligner(emission_R[0], all_token_ids)

        if emission_align:
            _sl = float(emission_self_loop)
            if _sl:
                _emL = emission_L[0].clone()
                _emL[:, 1:] = _emL[:, 1:] + _sl
                _emR = emission_R[0].clone()
                _emR[:, 1:] = _emR[:, 1:] + _sl
                bonus_spans_L = aligner(_emL, all_token_ids)
                bonus_spans_R = aligner(_emR, all_token_ids)
                token_spans_L = _hybrid_spans(token_spans_L, bonus_spans_L)
                token_spans_R = _hybrid_spans(token_spans_R, bonus_spans_R)
                logger.info(f" [EmissionAlign] 自环奖励 self_loop={_sl} 已混合(起点取基础对齐)")

        # =======================================================================
        # 逐 token 结果组装（L/R 择优 → 时间映射 → 格式化）
        # =======================================================================
        n_tokens = len(all_token_ids)

        # =======================================================================
        # 方案 C：窗口化重 CTC（锚点切窗 + 重叠合并，治累积漂移 / 大幅错误行）
        # 在全局 CTC（含 emission_align 修正起点）结果基础上，按高置信空白间隔
        # 切窗重对齐，原地改写 token_spans_L/R 的全局帧坐标。下游组装/后处理不变。
        # =======================================================================
        if anchor_realign:
            token_spans_L, token_spans_R = _anchor_realign(
                emission_L, emission_R, all_token_ids, token_spans_L,
                token_spans_R, n_tokens, aligner, frame_duration,
                map_to_original_time, total_frames,
                max_tokens=anchor_max_tokens, overlap_tokens=anchor_overlap,
                pad_frames=anchor_pad, gap_min_frames=anchor_gap,
                log_callback=log_callback)

        results = []

        for i in range(n_tokens):
            spans_L = token_spans_L[i] if i < len(token_spans_L) else None
            spans_R = token_spans_R[i] if i < len(token_spans_R) else None
            best_spans, best_score = _pick_best_spans(spans_L, spans_R)

            if best_spans is None:
                results.append({
                    'token': all_align_tokens[i] if i < len(all_align_tokens) else '',
                    'start': '[error]',
                    'end': '[error]'
                })
                continue

            adjusted_start = best_spans[0].start * frame_duration
            adjusted_end = best_spans[-1].end * frame_duration

            original_start = map_to_original_time(adjusted_start)
            original_end = map_to_original_time(adjusted_end)

            results.append({
                'token': all_align_tokens[i] if i < len(all_align_tokens) else '',
                'start': format_time(original_start),
                'end': format_time(original_end),
                    'original_start': original_start,
                    'original_end': original_end
                })

        # ── pre-expand 回映射 ──────────────────────────────────────
        if line_mode and _prep_map:
            _orig_results = results
            # 构建原始 all_align_tokens（未展开版本），供回填 token 字段
            _orig_aat = []
            for _pli, _pline in enumerate(lines):
                _orig_aat.extend(_pline['tokens'])

            _orig_line_counts = [0] * len(lines)
            for _pli, _pline in enumerate(lines):
                _nt_orig = len(_pline.get('token_ids', []))
                _orig_line_counts[_pli] = _nt_orig

            _new_results = []
            _ri = 0; _oi = 0   # expanded idx, original aat idx
            for _pli, _pline in enumerate(lines):
                _fac = _prep_map.get(_pli, (1, (), 0))[0]
                _nt_orig = _orig_line_counts[_pli]
                _nt_exp = _nt_orig * _fac
                if _nt_orig == 0 or _fac == 1:
                    for _j in range(_nt_orig):
                        _new_results.append(_orig_results[_ri + _j])
                        _oi += 1
                    _ri += _nt_orig
                    continue
                # 展开行：取所有副本跨段 min-start / max-end
                _ls = _le = None
                for _j in range(_nt_exp):
                    r = _orig_results[_ri + _j]
                    rs = r.get("original_start"); re_ = r.get("original_end")
                    if rs is not None and rs != "[error]" and re_ is not None and re_ != "[error]":
                        if _ls is None or rs < _ls: _ls = rs
                        if _le is None or re_ > _le: _le = re_
                _ri += _nt_exp
                if _ls is None: _ls = _le = 0
                for _j in range(_nt_orig):
                    fs = _j / _nt_orig; fe = (_j + 1) / _nt_orig
                    s = _ls + fs * (_le - _ls); e = _ls + fe * (_le - _ls)
                    _tk = _orig_aat[_oi] if _oi < len(_orig_aat) else ''
                    _new_results.append({'token': _tk, 'start': format_time(s),
                                          'end': format_time(e),
                                          'original_start': s, 'original_end': e})
                    _oi += 1
            results = _new_results
            n_tokens = len(results)
            all_token_ids = _orig_all_ids
            # 恢复原始行级索引（OnoWave 两步数峰需要）
            _orig_gt0 = 0
            for _j, _pl in enumerate(lines):
                _cnt = _orig_line_counts[_j] if _j < len(_orig_line_counts) else (lines[_j].get('g_end', 0) - lines[_j].get('g_start', 0))
                _pl['g_start'] = _orig_gt0
                _pl['g_end'] = _orig_gt0 + _cnt
                _orig_gt0 += _cnt

        # (OnoWave moved after auto_repeat — see below)

        # =======================================================================
        # 结构性修复：按静音切段独立 CTC（segment_align=True）
        # 调用方：GUI 高级面板「静音切段对齐(实验)」复选框；eval 的 seg 配置。
        # 仅支持全局模式（non_silent_ranges 为空）。在全局结果基础上，用
        # blank 概率定位静音边界切段，逐段重对齐并覆盖（未覆盖处保留全局）。
        # segment_align 仅支持全局模式（non_silent_ranges 为空）。
        # =======================================================================
        if segment_align:
            if len(non_silent_ranges) > 1:
                logger.warning(" [SegAlign] 检测到多段 VAD 分段，segment_align 仅支持整段全局模式，已跳过")
            else:
                try:
                    _ov = _segment_align(
                        emission_L, emission_R, all_token_ids, lines,
                        aligner, frame_duration, map_to_original_time,
                        token_spans_L, token_spans_R, n_tokens,
                        all_align_tokens=all_align_tokens,
                        blank_th=blank_th, min_sil_frames=min_sil_frames,
                        min_seg_frames=min_seg_frames, log_callback=log_callback)
                except Exception as e:
                    logger.warning(f" [SegAlign] 段对齐异常，回退全局: {e}")
                    _ov = None
                if _ov:
                    for i in range(n_tokens):
                        ov = _ov[i]
                        if ov is None:
                            continue
                        a_s, a_e = ov
                        o_s = map_to_original_time(a_s)
                        o_e = map_to_original_time(a_e)
                        r = results[i]
                        r['original_start'] = o_s
                        r['original_end'] = o_e
                        r['start'] = format_time(o_s)
                        r['end'] = format_time(o_e)

        # =======================================================================
        # LineFloor（已弃用，2026-08-09 结论）：
        # 尝试在 auto_repeat 前/后做溢出回流→塌缩行推后，降低下游连锁偏移。
        # 实测：LineFloor 与 auto_repeat 结构性冲突（auto_repeat joint CTC 覆写
        # LineFloor 推后的 results），无论前后顺序均无法协同。保留参数接口，
        # 实现留作实验参考，默认不执行。
        # =======================================================================
            _line_starts = [l["g_start"] for l in lines]
            _line_ends = [l["g_end"] for l in lines]
            _nline = len(lines)
            _line_dur = [0.0] * _nline
            for _li in range(_nline):
                _g0, _g1 = _line_starts[_li], _line_ends[_li]
                if _g1 <= _g0:
                    continue
                _t0 = results[_g0].get("original_start")
                _t1 = results[_g1 - 1].get("original_end")
                if (isinstance(_t0, (int, float)) and isinstance(_t1, (int, float)) and _t1 > _t0):
                    _line_dur[_li] = _t1 - _t0
            _nt_per_line = [_line_ends[_li] - _line_starts[_li] for _li in range(_nline)]
            _flag_rep = set(_prep_map.keys()) if _prep_map else set()
            _ntok_vals = [_line_dur[i] / max(1, _nt_per_line[i])
                          for i in range(_nline) if _line_dur[i] > 0 and i not in _flag_rep]
            _med = sorted(_ntok_vals)[len(_ntok_vals) // 2] if _ntok_vals else 0.1
            if _med > 0:
                _redist_total = 0
                _li = 0
                _li_checked = 0
                while _li < _nline:
                    _g0, _g1 = _line_starts[_li], _line_ends[_li]
                    _nt = _g1 - _g0
                    _expected = _nt * _med
                    if _li in _flag_rep and _nt > 0:
                        _li_checked += 1
                        _rpt_dur_x = _line_dur[_li] / _expected if _expected > 0 else 999
                        logger.info(f" [LineFloor] 行{_li} nt={_nt} pred={_line_dur[_li]:.1f}s "
                                    f"expected={_expected:.1f}s ratio={_rpt_dur_x:.1f}x")
                    if _li in _flag_rep and _nt > 0 and _line_dur[_li] > _expected * 3.0:
                        _excess = _line_dur[_li] - _expected * 3.0
                        _donors = []
                        _di = _li - 1
                        while _di >= 0 and _di not in _flag_rep:
                            _dg0, _dg1 = _line_starts[_di], _line_ends[_di]
                            _dnt = _dg1 - _dg0
                            if _dnt <= 0 or _dnt > 6:
                                break
                            _dntok = _line_dur[_di] / _dnt if _dnt > 0 else 999
                            if _dntok >= _med * 0.3:
                                break
                            _donors.insert(0, _di)
                            _di -= 1
                        if _donors:
                            _donor_tokens = sum(_nt_per_line[_di] for _di in _donors)
                            if _donor_tokens > 0 and _excess > 0.1:
                                _push_per_tok = _excess * line_floor / _donor_tokens
                                for _di in _donors:
                                    _dg0, _dg1 = _line_starts[_di], _line_ends[_di]
                                    _push = _push_per_tok * (_dg1 - _dg0)
                                    _last = _dg1 - 1
                                    _r = results[_last]
                                    _r["original_end"] = _r.get("original_end", 0) + _push
                                    _r["end"] = format_time(_r["original_end"])
                                    _redist_total += 1
                                    logger.info(f" [LineFloor] 行{_di}(nt={_dg1-_dg0})"
                                                f" ← 行{_li} 回流+{_push:.2f}s")
                                _li += 1
                                continue
                    _li += 1
                if _redist_total:
                    logger.info(f" [LineFloor] 回流了 {_redist_total} 个 burst 行 "
                                f"(回收溢出×{line_floor}，中位 {_med*1000:.0f}ms/tok)")


        # (collapse_prune moved after OnoWave)


        # =======================================================================
        # 自动重复型拟声行修复（auto_repeat）：形态 A 的 CTC 塌缩
        # =======================================================================
        # —— 快照原始位置（OnoWave 两步数峰用）——
        _snap_pos = {}  # line_idx → (original_start, original_end)
        if ono_wave and _ONO_TABLE and waveform is not None and line_mode:
            for _pli, _val in _prep_map.items():
                if not _val or not _val[1]: continue
                _pl = lines[_pli]
                _g0, _g1 = _pl['g_start'], _pl['g_end']
                if _g1 <= _g0 or _g0 >= n_tokens: continue
                _rs = results[_g0].get('original_start')
                _re = results[_g1 - 1].get('original_end')
                if isinstance(_rs, (int, float)) and isinstance(_re, (int, float)):
                    _snap_pos[_pli] = (_rs, _re)

        if line_mode and auto_repeat:
            _ar_repeat, _ar_fixed = _auto_repeat(
                emission_L, emission_R, lines, aligner, tokenizer,
                frame_duration, map_to_original_time, format_time,
                all_token_ids, n_tokens, results,
                token_spans_L, token_spans_R,
                original_to_adjusted=original_to_adjusted,
                waveform=waveform, sample_rate=sample_rate,
                log_callback=log_callback,
                auto_repeat_burst=auto_repeat_burst)
            if _ar_fixed:
                logger.info(f" [AutoRepeat] 检测到 {_ar_repeat} 个重复型行, "
                            f"修复了 {_ar_fixed} 个塌缩行 (含 burst 修复)")
            elif _ar_repeat:
                logger.info(f" [AutoRepeat] 检测到 {_ar_repeat} 个重复型行, "
                            f"全部无需修复（时长正常）")

        # =======================================================================
        # 两步波形数峰（ono_wave）：auto_repeat 修完后，对仍塌缩/溢出的行数峰修正。
        # =======================================================================
        if ono_wave and _ONO_TABLE and waveform is not None and line_mode:
            _wf_np = waveform[0].cpu().numpy() if hasattr(waveform, 'cpu') else _np.asarray(waveform).ravel()
            _nredo = 0
            _not_found = []
            for _pli in range(len(lines)):
                _val = _prep_map.get(_pli)
                if not _val: continue
                _pline = lines[_pli]
                _g0, _g1 = _pline['g_start'], _pline['g_end']
                if _g1 <= _g0 or _g0 >= n_tokens: continue
                # 用 auto_repeat 前的快照位置（auto_repeat 已改写 results）
                _snap = _snap_pos.get(_pli)
                if _snap:
                    _rs, _re = _snap
                else:
                    _rs = results[_g0].get('original_start')
                    _re = results[_g1 - 1].get('original_end')
                if _rs is None or _re is None or _rs == '[error]' or _re == '[error]':
                    continue
                _unit = _val[1]; _ulen = _val[2]; _key = ''.join(_unit) if _unit else ''
                _rule = _ONO_TABLE.get(_key)
                if not _rule:
                    if _key: _not_found.append(_key)
                    continue
                _punct = {'.','…','！','!','？','?','、','。','♡','♥','-','ー','~','〜'}
                _raw = [t for t in _pline['raw'] if t and t != '_' and t not in _punct]
                _nw = max(1, len(_raw) // max(1, _ulen))
                _pad = 2.0
                _f0 = max(0, int((_rs - _pad) * sample_rate))
                _f1 = min(_wf_np.shape[0], int((_re + _pad) * sample_rate))
                _chunk = _wf_np[_f0:_f1]
                if len(_chunk) < sample_rate * 0.5: continue
                _det = _rule.get('detect', '')
                if _det == 'peak_pairs_with_silence':
                    _act = _ono_count_peak_pairs(_chunk, sample_rate, _rule)
                elif _det == 'rms_autocorr':
                    _act = _ono_count_envelope_autocorr(_chunk, sample_rate,
                        _rule.get('min_period_ms', 400), _rule.get('max_period_ms', 1800))
                elif _det == 'rms_valley_cut':
                    _act = _ono_count_rms_valley(_chunk, sample_rate, _rule)
                elif _det == 'peak_count':
                    _act = _ono_count_energy_peak(_chunk, sample_rate,
                        _rule.get('peak_threshold', 2.0), _rule.get('min_peak_dist_ms', 400))
                elif _det == 'rms_segment':
                    _act = _ono_count_rms_segment(_chunk, sample_rate, _rule)
                else: continue
                if _act <= _nw or _act < 2: continue
                _new_f = max(2, min(int(_act / _nw + 0.5), 8))
                if _new_f <= _val[0]: continue
                # 用 frame_duration 转帧坐标（emission 是帧维度，不是样本维度）
                _f0_f = max(0, int(_rs / frame_duration) - int(1.0 / frame_duration))
                _f1_f = min(emission_L.shape[1], int(_re / frame_duration) + int(1.0 / frame_duration))
                if _f1_f - _f0_f < int(0.5 / frame_duration): continue
                try:
                    _exp_tokens = _pline['token_ids'] * _new_f
                    if len(_exp_tokens) > 50:
                        logger.info(f" [OnoWave] L{_pli} tokens too long: {len(_exp_tokens)}")
                        continue
                    _sp = aligner(emission_L[0, _f0_f:_f1_f, :], _exp_tokens)
                    _ok = bool(_sp and _sp[0] and _sp[-1])
                    if _ok:
                        _os_w = map_to_original_time((_sp[0][0].start + _f0_f) * frame_duration)
                        _oe_w = map_to_original_time((_sp[-1][-1].end + _f0_f) * frame_duration)
                        _nt_o = _g1 - _g0
                        for _gi in range(_g0, min(_g1, n_tokens)):
                            _r = results[_gi]; _frac = (_gi - _g0) / max(1, _nt_o)
                            _r['original_start'] = _os_w + _frac * (_oe_w - _os_w)
                            _r['original_end'] = _os_w + (_frac + 1/max(1, _nt_o)) * (_oe_w - _os_w)
                            _r['start'] = format_time(_r['original_start'])
                            _r['end'] = format_time(_r['original_end'])
                        _nredo += 1
                        logger.info(f" [OnoWave] L{_pli} [{_key}] 数{_act}组/台{_nw}→×{_new_f}")
                except Exception:
                    pass
            if _nredo:
                logger.info(f" [OnoWave] 修正了 {_nredo} 行")
            if _not_found:
                logger.info(f" [OnoWave] 表外key: {_not_found[:5]}")

        # =======================================================================
        # 塌缩行切除重跑全局 CTC（collapse_prune）：
        # 检测仍塌缩的喘息行→去掉其token→全局 CTC →仅重写下游。
        # 循环直到无新塌缩或5次上限。
        # =======================================================================
        if line_mode and collapse_prune:
            _done = set()
            for _cp_iter in range(5):
                _found = set()
                for _pli in range(len(lines)):
                    if _pli in _done or _pli in _prep_map: continue
                    _pl = lines[_pli]; _g0, _g1 = _pl['g_start'], _pl['g_end']
                    _nt = _g1 - _g0
                    if _nt <= 0 or _nt > 4: continue
                    _rs = results[_g0].get('original_start')
                    _re = results[_g1 - 1].get('original_end')
                    if not (isinstance(_rs, (int,float)) and isinstance(_re, (int,float))): continue
                    if _re - _rs > 0.25: continue
                    _found.add(_pli)
                if not _found:
                    break
                # 一次全局 CTC：切除所有塌缩行 token
                _cp_ids = []; _cp_map = []
                for _gi in range(n_tokens):
                    _li = None
                    for _j in range(len(lines)):
                        if lines[_j]['g_start'] <= _gi < lines[_j]['g_end']:
                            _li = _j; break
                    if _li is not None and _li in _found:
                        continue
                    _cp_ids.append(all_token_ids[_gi])
                    _cp_map.append(_gi)
                if len(_cp_ids) < 10: break
                _first_rm = min(_g for _pli in _found
                                for _g in range(lines[_pli]['g_start'], lines[_pli]['g_end']))
                logger.info(f" [Prune] 回合{_cp_iter+1}: 切除{len(_found)}行 "
                            f"(token {n_tokens}→{len(_cp_ids)}), 重跑全局 CTC")
                try:
                    _cp_spans = aligner(emission_L[0], _cp_ids)
                    if _cp_spans and len(_cp_spans) == len(_cp_ids):
                        _cp_rewritten = 0
                        for _new_idx, _old_gi in enumerate(_cp_map):
                            if _old_gi < _first_rm: continue
                            _sp = _cp_spans[_new_idx]
                            if not _sp: continue
                            _r = results[_old_gi]
                            _os = map_to_original_time(_sp[0].start * frame_duration)
                            _oe = map_to_original_time(_sp[-1].end * frame_duration)
                            _r['original_start'] = _os
                            _r['original_end'] = _oe
                            _r['start'] = format_time(_os)
                            _r['end'] = format_time(_oe)
                            _cp_rewritten += 1
                        _done |= _found
                        logger.info(f" [Prune] 重写了 {_cp_rewritten} 个下游 token")
                except Exception as _ex:
                    logger.info(f" [Prune] 失败: {_ex}")
                    break

        # =======================================================================
        # 对齐后处理（零重训、纯推理侧）：单调性 + 不重叠 + 最小持续宽度
        # CTC 峰值塌缩会让短音节（ん/あ/吐息）塌成 0~1 帧宽；相邻 token 重叠
        # 几乎必为对齐错误。此处仅对"过短/重叠"的 span 做约束，长句不受影响。
        # 模型与 LoRA 完全不动，只改 emission → 时间轴的映射。
        # =======================================================================
        _min_dur = max(frame_duration, min_token_dur_ms / 1000.0)
        _prev_end = None
        _refined = 0
        for r in results:
            if r.get('original_start') is None or r['original_start'] == '[error]':
                _prev_end = None
                continue
            s, e = r['original_start'], r['original_end']
            changed = False
            if _prev_end is not None and s < _prev_end:
                s = _prev_end
                changed = True
            if e < s + _min_dur:
                e = s + _min_dur
                changed = True
            if changed:
                _refined += 1
                r['original_start'], r['original_end'] = s, e
                r['start'], r['end'] = format_time(s), format_time(e)
            _prev_end = e
        if _refined:
            logger.info(f" [Refine] 约束了 {_refined} 个过短/重叠 token（最小宽度 {_min_dur*1000:.0f}ms）")

        # 清理 emission
        del emission_L, emission_R

        end_time = time.time()
        logger.info("Alignment inference executed in %s seconds", round(end_time - start_time, 3))
        return results
    except Exception:
        import traceback
        tb = traceback.format_exc()
        logger.exception("强制对齐失败")
        if log_callback:
            log_callback("[对齐异常] 强制对齐失败，详细 traceback:")
            for line in tb.splitlines():
                log_callback(line)
        # 同时把 traceback 写到 exe/工作目录，方便离线排查
        try:
            import sys
            if getattr(sys, 'frozen', False):
                _d = os.path.dirname(os.path.abspath(sys.executable))
            else:
                _d = os.getcwd()
            _p = os.path.join(_d, 'FA-ASMR_align_error.log')
            with open(_p, 'a', encoding='utf-8') as _f:
                _f.write(f"\n{'='*40}\n")
                _f.write(time.strftime("%Y-%m-%d %H:%M:%S\n"))
                _f.write(tb)
                _f.write("\n")
        except Exception:
            pass
        return []
