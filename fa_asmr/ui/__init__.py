# -*- coding: utf-8 -*-
"""FA-ASMR V2 桌面控制台 UI 子包（主题控件、多轨编排矩阵、主控制台窗口）。"""


def __getattr__(name: str):
    if name in ("build_ui", "enable_windows_dpi_awareness"):
        from .console import build_ui, enable_windows_dpi_awareness
        return build_ui if name == "build_ui" else enable_windows_dpi_awareness
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
