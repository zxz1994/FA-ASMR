# -*- coding: utf-8 -*-
"""FA-ASMR 底层系统配置（进阶用户手编，非 GUI）

读取 exe 同级的 advanced_config.json。该文件只被读取、从不被 GUI 写回，
因此用户手编的内容不会被 fa_asmr_settings.json 的保存覆盖（两者刻意分离）。

JSON 标准不支持注释，这里做宽松解析：以 // 或 # 开头的整行视为注释并忽略。

仅暴露「系统/性能层面」、GUI 高级面板没有的旋钮，全部可选，缺省回退代码默认值：

  cpu_threads    : 整数 / null
                  CPU 推理前向线程数。null = 自动取物理核数。
                  机器核心很多、但想给其它程序（或并发任务）留资源时可设较小值（如 4）。

  quantize_int8  : true / false / null
                  CPU 推理 INT8 动态量化。null = 自动（CPU 开启 / GPU 关闭）。
                  个别老旧 CPU 上 INT8 反而更慢或异常时可设 false 关闭。

  cpu_chunk_cap  : 整数
                  CPU 模式单块帧数上限（默认 2000）。
                  wav2vec2 自注意力为 O(n²)，数值越大越占内存、越小越慢；
                  仅 CPU 模式生效，GPU 模式由 GUI 的「显存分块(vram_chunk)」控制。
"""

import os
import sys
import json
import logging

logger = logging.getLogger(__name__)

# 与 align_model._CPU_CHUNK_CAP 原值一致，作为未配置时的回退
DEFAULTS = {
    "cpu_threads": None,
    "quantize_int8": None,
    "cpu_chunk_cap": 2000,
}

_CACHE = None
LAST_WARNINGS = []


def _warn(msg):
    LAST_WARNINGS.append(msg)
    logger.warning(msg)
    try:
        print(f"  ⚠ [advanced_config] {msg}", file=sys.stderr)
    except Exception:
        pass


def reset_cache():
    """清空配置缓存与历史告警（供测试或热重载使用）。"""
    global _CACHE
    _CACHE = None
    LAST_WARNINGS.clear()


def config_path():
    """定位 advanced_config.json：冻结后取 exe 同级，开发时取本模块同级。"""
    if getattr(sys, "frozen", False):
        # onedir: sys.executable = dist/FA-ASMR/FA-ASMR.exe，
        # 权重 model.pt / 本配置均放在该 exe 同级目录（非 _internal）。
        return os.path.join(os.path.dirname(sys.executable), "advanced_config.json")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "advanced_config.json")


def _strip_comments(text):
    """剔除 // 与 # 开头的整行注释，保留 JSON 主体。"""
    out = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("//") or s.startswith("#"):
            continue
        out.append(line)
    return "\n".join(out)


def _validate_value(key, val):
    """校验单字段类型与合法范围，返回 (是否合法, 规范化值, 错误说明)。"""
    if key == "cpu_threads":
        if isinstance(val, bool) or not isinstance(val, int) or val < 1:
            return False, None, f"cpu_threads 必须为 >= 1 的整数或 null，收到 {val!r}"
        return True, val, None
    if key == "quantize_int8":
        if not isinstance(val, bool):
            return False, None, f"quantize_int8 必须为布尔值 (true/false) 或 null，收到 {val!r}"
        return True, val, None
    if key == "cpu_chunk_cap":
        if isinstance(val, bool) or not isinstance(val, int) or val < 100:
            return False, None, f"cpu_chunk_cap 必须为 >= 100 的整数，收到 {val!r}"
        return True, val, None
    return True, val, None


def load():
    """加载配置（带缓存），缺字段回退 DEFAULTS；格式或字段错误时显式告警。"""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    LAST_WARNINGS.clear()
    data = dict(DEFAULTS)
    path = config_path()
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                user = json.loads(_strip_comments(f.read()))
        except Exception as e:
            _warn(f"读取或解析 {path} 失败 ({type(e).__name__}: {e})，已回退默认配置")
        else:
            if not isinstance(user, dict):
                _warn(f"{path} 根节点必须为 JSON 对象，实际为 {type(user).__name__}，已回退默认配置")
            else:
                unknown = [k for k in user if k not in DEFAULTS]
                if unknown:
                    _warn(f"{path} 包含未识别的配置键 {unknown}，已忽略（有效键: {list(DEFAULTS.keys())}）")
                for k in DEFAULTS:
                    if k in user and user[k] is not None:
                        ok, norm_val, err_msg = _validate_value(k, user[k])
                        if ok:
                            data[k] = norm_val
                        else:
                            _warn(f"{path} 配置项 {err_msg}，已回退默认值 {DEFAULTS[k]!r}")
    _CACHE = data
    return data


def get(key, default=None):
    """取某项配置；未配置或不存在时返回 default。"""
    return load().get(key, default)


