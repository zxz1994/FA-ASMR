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

# 与 align_model._CPU_CHUNK_CAP 原值一致，作为未配置时的回退
DEFAULTS = {
    "cpu_threads": None,
    "quantize_int8": None,
    "cpu_chunk_cap": 2000,
}

_CACHE = None


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


def load():
    """加载配置（带缓存），缺字段回退 DEFAULTS。"""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    data = dict(DEFAULTS)
    path = config_path()
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                user = json.loads(_strip_comments(f.read()))
            if isinstance(user, dict):
                for k in DEFAULTS:
                    if k in user and user[k] is not None:
                        data[k] = user[k]
        except Exception:
            # 解析失败静默回退默认值，不影响主流程
            pass
    _CACHE = data
    return data


def get(key, default=None):
    """取某项配置；未配置或不存在时返回 default。"""
    return load().get(key, default)


def reload():
    """清空缓存，下次 load 重新读取磁盘（运行时改配置后调用生效）。"""
    global _CACHE
    _CACHE = None
