# -*- coding: utf-8 -*-
"""对齐器组件 — LoRA 适配器 / MMS_FA 模型缓存 / emission 计算

从 align.py 拆分。本模块只含模型加载、LoRA 注入、emission 前向。
主对齐管线见 align.py；后处理见 align_post.py；工具函数见 align_utils.py。
"""

import torch
import torch.nn as nn
import torchaudio
import math
import os
import sys
import logging
from typing import Optional, List, Dict, Any

import fa_asmr.core.advanced_config as adv

logger = logging.getLogger(__name__)


class SimpleLoraLinear(nn.Module):
    """
    最小化 LoRA Linear 层，与 peft 训练的 checkpoint 参数命名兼容。
    使用 ModuleDict 匹配 peft 的 lora_A.default.weight / lora_B.default.weight 命名，
    确保 load_state_dict 无缝工作，且不依赖 peft 库。
    """
    def __init__(self, base_layer: nn.Linear, r: int = 8, lora_alpha: int = 16,
                 lora_dropout: float = 0.1):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # ModuleDict 使 state_dict key 变成 lora_A.default.weight (与 peft 一致)
        self.lora_A = nn.ModuleDict()
        self.lora_B = nn.ModuleDict()
        self.lora_A["default"] = nn.Linear(in_features, r, bias=False)
        self.lora_B["default"] = nn.Linear(r, out_features, bias=False)

        if lora_dropout > 0:
            self.lora_dropout = nn.Dropout(lora_dropout)
        else:
            self.lora_dropout = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        lora = self.lora_B["default"](self.lora_A["default"](self.lora_dropout(x)))
        return base + lora * self.scaling


class LoraAdapter:
    """
    LoRA 适配器: 向 MMS-FA wav2vec2 注入 LoRA 权重并替换 aux 头。
    用法:
        lora = LoraAdapter(model, lora_path="best_model.pt")
        model = lora.apply()   # 原地注入 LoRA，返回模型
    注入目标由 checkpoint 的 target_modules 决定，支持 Attention 与 FFN。
    """
    def __init__(self, model: nn.Module, lora_path: Optional[str] = None):
        self.model = model          # 外层 bundle model (Wav2Vec2FASTBundle.Model)
        self.w2v2 = model.model     # 内层 wav2vec2
        self.aux = self.w2v2.aux    # 线性分类头
        self.lora_path = lora_path
        self._loaded = False
        self._injected = False
        self._merged = False
        self.last_error = None      # 公开字段：最近一次加载错误信息

    def apply(self) -> nn.Module:
        if self.lora_path is not None:
            self._try_load()
        return self.model

    def reload(self, lora_path: str) -> nn.Module:
        self.lora_path = lora_path
        return self.apply()

    def _try_load(self) -> bool:
        try:
            if not os.path.isfile(str(self.lora_path)):
                raise FileNotFoundError(f"LoRA 权重文件不存在: {self.lora_path}")
            ckpt = torch.load(self.lora_path, map_location="cpu", weights_only=False)
            if not isinstance(ckpt, dict):
                raise ValueError(f"LoRA 权重格式非法：期望 dict，实际为 {type(ckpt).__name__}")
            missing_sections = [k for k in ("w2v2_lora", "aux") if k not in ckpt]
            if missing_sections:
                raise KeyError(f"LoRA 权重缺少必要字段: {', '.join(missing_sections)}")

            config = ckpt.get("config", {})
            if not isinstance(config, dict):
                raise ValueError(f"LoRA config 字段非法：期望 dict，实际为 {type(config).__name__}")
            target_modules = config.get("target_modules", ["k_proj", "q_proj", "v_proj", "out_proj"])
            r = config.get("r", 8)
            alpha = config.get("alpha", 16)
            device = next(self.model.parameters()).device

            logger.info(f"[LoRA] 加载 {self.lora_path}")
            logger.info(f"[LoRA] 配置: r={r}, alpha={alpha}, targets={target_modules}")

            for p in self.w2v2.parameters():
                p.requires_grad = False

            replaced = self._inject_lora(target_modules, r, alpha, device)
            if replaced == 0:
                raise RuntimeError(
                    f"未匹配到任何可注入的 Linear 目标层 (targets={target_modules})"
                )

            missing, unexpected = self.w2v2.load_state_dict(ckpt["w2v2_lora"], strict=False)
            if missing:
                logger.info(f"[LoRA] w2v2 缺失 {len(missing)} keys (正常，只加载 lora)")
            if unexpected:
                logger.warning(f"[LoRA] w2v2 多余 {len(unexpected)} keys")

            self.aux.load_state_dict(ckpt["aux"])

            # In inference mode dropout is disabled, so each LoRA adapter can
            # be folded into its base Linear once.  This removes two extra
            # Linear calls (A and B) from every transformer forward.
            self.w2v2.eval()
            self.aux.eval()
            self._merge_for_inference()

            trainable = sum(p.numel() for p in self.w2v2.parameters() if p.requires_grad)
            logger.info(f"[LoRA] 可训参数: {trainable:,}")

            self._loaded = True
            self.last_error = None
            loss_val = ckpt.get("loss", "?")
            loss_str = f"{loss_val:.4f}" if isinstance(loss_val, (int, float)) else str(loss_val)
            logger.info(f"[LoRA] 加载成功 (epoch={ckpt.get('epoch','?')}, loss={loss_str})")
            return True

        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            logger.error(f"[LoRA] 加载失败 ({self.lora_path}): {self.last_error}")
            raise RuntimeError(
                f"[LoRA] 权重加载失败 ({self.lora_path}): {self.last_error}。"
                "为避免模型状态损坏或静默降级，已终止加载。"
            ) from e

    def _merge_for_inference(self) -> int:
        """Fold eval-mode LoRA weights into base Linear modules."""
        if self._merged:
            return 0
        merged = 0
        with torch.no_grad():
            for name, module in list(self.w2v2.named_modules()):
                if not isinstance(module, SimpleLoraLinear):
                    continue
                a = module.lora_A["default"].weight
                b = module.lora_B["default"].weight
                module.base_layer.weight.add_(torch.matmul(b, a) * module.scaling)
                parent_name, attr_name = name.rsplit(".", 1)
                parent = self.w2v2.get_submodule(parent_name)
                setattr(parent, attr_name, module.base_layer)
                merged += 1
        self._merged = True
        logger.info(f"[LoRA] 推理合并 {merged} 个 Linear 层")
        return merged

    def _inject_lora(
        self, target_modules: List[str], r: int, alpha: int, device: torch.device
    ) -> int:
        if self._injected:
            return 1

        replaced = 0

        for name, module in self.w2v2.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            basename = name.split(".")[-1]
            if basename not in target_modules:
                continue
            in_f, out_f = module.in_features, module.out_features
            has_bias = module.bias is not None
            new_base = nn.Linear(in_f, out_f, bias=has_bias)
            new_base.weight.data.copy_(module.weight.data)
            if has_bias:
                new_base.bias.data.copy_(module.bias.data)
            new_base.to(device, dtype=module.weight.dtype)

            lora_layer = SimpleLoraLinear(new_base, r=r, lora_alpha=alpha)
            lora_layer.base_layer.weight.requires_grad = False
            if has_bias:
                lora_layer.base_layer.bias.requires_grad = False

            parent_name = ".".join(name.split(".")[:-1])
            attr_name = name.split(".")[-1]
            parent = self.w2v2.get_submodule(parent_name)
            setattr(parent, attr_name, lora_layer)
            replaced += 1

        self._injected = True
        logger.info(f"[LoRA] 注入 {replaced} 个线性层 (r={r}, alpha={alpha})")
        return replaced


_MODEL_CACHE: Dict[Any, nn.Module] = {}
# CPU 推理时单块帧数上限：wav2vec2 自注意力为 O(n²)，超过 ~2*d(=2048) 帧后
# 注意力开销开始主导前向；收紧到 2000 帧可在质量无损(上下文 padding 保证 CNN
# 边界连续)的前提下显著降低 CPU 算力。GPU 不受影响(用户 vram_chunk 仍然生效)。
_CPU_CHUNK_CAP = 2000

# MMS-FA 的固有帧移：16kHz × 20ms = 320 采样/帧。
# CNN 下采样后每帧恰好对应 320 个输入采样，故「采样数 ↔ 帧数」的换算与
# 分块边界都必须用它。与 align_acoustic.FRAME_SEC（0.02s）是同一事实的两种表述。
_HOP_SAMPLES = 320


def _get_model(bundle, device, lora_path=None, log_callback=None,
               cpu_threads=None, quantize_int8=None) -> nn.Module:
    """按 (lora_path, device, quant, threads) 缓存 MMS_FA 模型，批量处理时复用同一份权重。

    CPU 加速：当 device 为 cpu 时自动对 wav2vec2 主干的 Linear 层(int8 动态量化)
    并按 cpu_threads(默认物理核数)设置 torch 线程数，显著降低前向耗时。
    """
    is_cpu = (str(device) == "cpu")
    # quantize_int8: 调用方未指定(None)时查 advanced_config，仍无则默认 CPU 自动开启
    if quantize_int8 is None:
        _q = adv.get("quantize_int8", None)
        quantize_int8 = (is_cpu if _q is None else bool(_q))
    key = (lora_path, str(device), bool(quantize_int8), cpu_threads)
    if key in _MODEL_CACHE:
        logger.info(" [VRAM] 复用已缓存的 MMS_FA 模型")
        return _MODEL_CACHE[key]

    logger.info(" [VRAM] 加载 MMS_FA 模型 ...")
    # 优先使用分发目录内的 MMS_FA 模型权重 (离线模式, 免下载)
    # frozen(onedir) 模式: models/ 手动放在 exe 同级目录; 开发模式: 在脚本同级目录
    import sys as _sys
    if getattr(_sys, 'frozen', False):
        # 权重由人工放到 dist/FA-ASMR/models/ (exe 同级), 不走 _internal
        _base_dir = os.path.dirname(_sys.executable)
    else:
        _base_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    _local_torch_home = os.path.join(_base_dir, 'models')
    if os.path.isfile(os.path.join(_local_torch_home, 'hub', 'checkpoints', 'model.pt')):
        os.environ['TORCH_HOME'] = _local_torch_home
        logger.info(f" [VRAM] 使用本地 MMS_FA 权重: {_local_torch_home}/hub/checkpoints/model.pt")
    model = bundle.get_model().to(device)

    if lora_path:
        original_device = next(model.parameters()).device
        adapter = LoraAdapter(model.cpu(), lora_path)
        model = adapter.apply()
        msg = f"[LoRA] 已注入，模型位于 {original_device}"
        logger.info(msg)
        if log_callback:
            log_callback(msg)
        model = model.to(original_device)

    # ---- CPU 推理加速 ----
    if is_cpu:
        if cpu_threads is None:
            cpu_threads = adv.get("cpu_threads", None)
        if cpu_threads is None:
            try:
                cpu_threads = max(1, (os.cpu_count() or 4))
            except Exception as e:
                logger.warning(f" [CPU] 获取 CPU 核数失败，回退默认 4 线程: {e}")
                cpu_threads = 4
        try:
            torch.set_num_threads(int(cpu_threads))
        except Exception as e:
            warn_msg = f"  ⚠ [CPU] 设置推理线程数 ({cpu_threads}) 失败: {type(e).__name__}: {e}"
            logger.warning(warn_msg)
            if log_callback:
                log_callback(warn_msg)
        try:
            torch.set_num_interop_threads(max(1, int(cpu_threads) // 2))
        except RuntimeError:
            # PyTorch 在并行任务启动后禁止重复修改 interop 线程数，属预期行为
            pass
        except Exception as e:
            warn_msg = f"  ⚠ [CPU] 设置 interop 线程数失败: {type(e).__name__}: {e}"
            logger.warning(warn_msg)
            if log_callback:
                log_callback(warn_msg)
        if quantize_int8:
            try:
                import torch.ao.quantization as aq
                # 仅量化 wav2vec2 主干 Linear(自注意力/FFN)；CNN 特征提取器保持 float32
                model = aq.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
                logger.info(f" [CPU] 已启用 int8 动态量化 + 线程数={cpu_threads}")
            except Exception as e:
                warn_msg = f"  ⚠ [CPU] int8 量化失败，已回退 float32: {type(e).__name__}: {e}"
                logger.warning(warn_msg)
                if log_callback:
                    log_callback(warn_msg)
                else:
                    print(warn_msg, file=sys.stderr)

    _MODEL_CACHE[key] = model
    return model


get_mms_model = _get_model


def clear_model_cache():
    """释放缓存的 MMS_FA 模型（释放显存或切换 LoRA 前调用）。"""
    _MODEL_CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _resolve_device(device):
    if isinstance(device, torch.device):
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "指定了 CUDA 设备，但当前环境未检测到可用的 NVIDIA GPU / CUDA 支持。请改用 cpu 或 auto。"
            )
        return device
    dev_str = str(device).strip().lower()
    if dev_str == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "指定了 CUDA 设备，但当前环境未检测到可用的 NVIDIA GPU / CUDA 支持。请改用 cpu 或 auto。"
            )
        return torch.device("cuda")
    if dev_str == "cpu":
        return torch.device("cpu")
    if dev_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"不支持的推理设备: {device!r}（仅支持 'auto' / 'cuda' / 'cpu'）")


def compute_emission(waveform, sample_rate, device="auto", lora_path=None,
                     chunk_frames=3000, use_stereo_pick=True,
                     cpu_threads=None, quantize_int8=None, cpu_chunk_cap=None,
                     log_callback=None):
    """与 align_audio_with_text 内部的发射计算完全一致（vad=None 路径），
    但可被外部复用：同一份发射喂给多个 config，消除 GPU 卷积非确定性导致的
    跨 config 起点漂移。返回 (emission_L, emission_R)，均为 CPU 张量。

    注意：本函数不处理 VAD 分段（评测对比的 global/emB/seg 均为整段 CTC）。
    """
    device = _resolve_device(device)
    bundle = getattr(torchaudio.pipelines, "MMS_FA", None)
    if bundle is None:
        raise RuntimeError(
            "当前 torchaudio 不含 MMS_FA pipeline。FA-ASMR 强制对齐依赖它，"
            "需要 torchaudio==2.6.0（与打包 exe 内置版本一致）。\n"
            "请执行: pip install torch==2.6.0 torchaudio==2.6.0")
    if waveform is None or waveform.numel() == 0 or waveform.shape[-1] == 0:
        raise ValueError("音频波形为空（0 采样），无法执行声学对齐")
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    max_val = torch.max(torch.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val
    # 整轨重采样一次（在拆声道之前）。旧实现让左右声道各自重采样，而
    # torchaudio 对 2-D 波形本就是逐声道独立操作，等价但多跑一遍全轨。
    # 重采样后帧数换算只需 _HOP_SAMPLES（帧移与采样率无关），故不必回写
    # sample_rate —— 这也是原先那个「二次重采样」死分支被删掉的原因。
    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(
            waveform, sample_rate, bundle.sample_rate)
    is_stereo = waveform.shape[0] > 1
    model = _get_model(bundle, device, lora_path, log_callback,
                       cpu_threads=cpu_threads, quantize_int8=quantize_int8)

    def _chunk_params():
        chunk_frames_eff = chunk_frames
        _cap = cpu_chunk_cap if cpu_chunk_cap is not None else adv.get("cpu_chunk_cap", _CPU_CHUNK_CAP)
        _cap = int(_cap)
        if str(device) == "cpu" and chunk_frames_eff > _cap:
            chunk_frames_eff = _cap
        chunk_size = _HOP_SAMPLES * chunk_frames_eff
        # 上下文的帧数下限 100 帧：短块也要给 CNN 边界留足够感受野，
        # 否则块首尾的卷积上下文不足会让该块的发射质量下降。
        context_size = max(_HOP_SAMPLES * 100, chunk_size // 24)
        return chunk_size, context_size

    def get_emission_single_channel(single_channel_wf):
        chunk_size, context_size = _chunk_params()
        total_samples = single_channel_wf.shape[1]
        local = []
        # On CUDA, keep the small per-chunk valid slices on device and copy
        # once after the channel is complete.  Copying every chunk forces a
        # synchronization point between the GPU forward and Python, while
        # the accumulated valid emissions are only tens of MB for a full
        # track.  The CPU path keeps its original bounded-memory behavior.
        accumulate_on_device = (device.type == "cuda")
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")
        ):
            for start in range(0, total_samples, chunk_size):
                end = min(start + chunk_size, total_samples)
                keep_frames = math.ceil((end - start) / _HOP_SAMPLES)
                if keep_frames == 0:
                    continue
                pad_start = max(0, start - context_size)
                pad_end = min(total_samples, end + context_size)
                chunk_audio = single_channel_wf[:, pad_start:pad_end].to(device)
                emission_chunk, _ = model(chunk_audio)
                left_frames = (start - pad_start) // _HOP_SAMPLES
                valid = emission_chunk[:, left_frames:left_frames + keep_frames, :]
                local.append(valid if accumulate_on_device else valid.cpu())
                del chunk_audio, emission_chunk, valid
        if not local:
            raise ValueError("音频过短，未产生任何有效声学帧，无法执行对齐")
        merged = torch.cat(local, dim=1)
        del local
        res = merged.cpu() if accumulate_on_device else merged
        del merged
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return res

    def get_emission_stereo_pair(wf_L, wf_R):
        """左右声道各跑一遍前向，不打包成 batch=2。

        不要试图把两声道 stack 成 (2, N) 一次前向来"省一遍"——两个原因，都实测踩过：

        1. torchaudio 2.6 的 MMS_FA wrapper（pipelines/_wav2vec2/utils.py
           _Wav2Vec2Model.forward）在 append_star=True 时把 star_dim 写死成
           ``torch.zeros((1, T, 1))``，batch=2 时 ``torch.cat`` 直接抛
           "Expected size 2 but got size 1"。该版本根本不支持 batch>1。
           （在那之前还有一关：feature_extractor 只接受 2D (batch, time)。）

        2. 就算绕过上面那关也不等价：wrapper 前向第一步是
           ``layer_norm(waveforms, waveforms.shape)``，对 (2, N) 整块归一化
           意味着左右声道共用一套统计量；分两次前向则是各声道独立归一化。
           打包会偏离 E30/V4 验证时的行为，不是逐位一致的优化。
        """
        return (get_emission_single_channel(wf_L),
                get_emission_single_channel(wf_R))

    if is_stereo and use_stereo_pick:
        emission_L, emission_R = get_emission_stereo_pair(
            waveform[0:1, :], waveform[1:2, :])
    else:
        if is_stereo:
            waveform = waveform.mean(dim=0, keepdim=True)
        emission_L = get_emission_single_channel(waveform)
        emission_R = emission_L
    return emission_L, emission_R
