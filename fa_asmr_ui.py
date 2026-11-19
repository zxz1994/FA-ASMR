# -*- coding: utf-8 -*-
"""FA-ASMR Acoustic Mastering Console — 全新桌面前端界面。

与强制对齐管线完全解耦，仅通过 build_ui(ctx) 接入业务回调。
采用暗色专业音频控制台架构：左侧控制导轨 + 中央多轨编排矩阵 + 底部走带与实时终端。
支持三模式即时切换、界面内外拖拽配对/互换、多语言热切换与配置自动持久化。
"""
from __future__ import annotations

import json
import math
import os
import queue
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from fa_asmr_i18n import (
    I18N,
    LANG_MAP,
    LANG_NAMES,
    detect_system_lang,
    tr,
)

# ══════════════════════════════════════════════════════════════════════════
#  色板与排版系统 — Obsidian Acoustic Console
# ══════════════════════════════════════════════════════════════════════════

BG_APP = "#0B0D12"
BG_RAIL = "#11151D"
BG_STAGE = "#0D1017"
BG_PANEL = "#151A24"
BG_PANEL_ALT = "#12161F"
BG_SLOT = "#1A2130"
BG_SLOT_HOVER = "#232D40"
BG_SLOT_EMPTY = "#121722"
BG_CONSOLE = "#080A0E"

BORDER = "#222B3B"
BORDER_SUBTLE = "#1B2230"
BORDER_STRONG = "#334155"

ACCENT = "#0EA5E9"
ACCENT_HOVER = "#38BDF8"
ACCENT_DIM = "#0F293B"
ACCENT_GLOW = "#7DD3FC"

AMBER = "#F59E0B"
AMBER_HOVER = "#FBBF24"
AMBER_DIM = "#292011"

EMERALD = "#10B981"
EMERALD_DIM = "#0E2920"

CRIMSON = "#F43F5E"
CRIMSON_DIM = "#2E131B"

TEXT_PRI = "#F1F5F9"
TEXT_SEC = "#94A3B8"
TEXT_MUTED = "#64748B"
INK_DARK = "#04111A"

FONT = "Microsoft YaHei UI"
MONO = "Consolas"

DEFAULT_VRAM_CHUNK = 3000

# 本地补充文案（覆盖原 UI 中未入字典的细粒度操作文案，保证中/日/英 100% 同步）
_EXTRA_I18N = {
    "import_files": {"zh": "＋ 导入素材", "ja": "＋ ファイル追加", "en": "+ Import"},
    "auto_pair": {"zh": "自动配对", "ja": "自動ペアリング", "en": "Auto-Pair"},
    "new_track_drop": {
        "zh": "＋ 拖拽素材至此创建新轨道，或点击导入文件",
        "ja": "＋ ここにドロップして新規トラック作成、またはクリックして追加",
        "en": "+ Drop files here to create a new track, or click to import",
    },
    "empty_sub": {
        "zh": "支持 .txt 台本与 .wav / .flac / .mp3 / .m4a / .ogg 音频 · 按文件名自动匹配",
        "ja": ".txt 台本と .wav / .flac / .mp3 / .m4a / .ogg 音声に対応 · ファイル名で自動照合",
        "en": "Supports .txt scripts & .wav / .flac / .mp3 / .m4a / .ogg · Auto-matched by filename",
    },
    "rail_section_engine": {"zh": "对齐引擎", "ja": "エンジンモード", "en": "ENGINE MODE"},
    "rail_section_params": {"zh": "运行参数", "ja": "パラメータ", "en": "PARAMETERS"},
    "rail_section_sys": {"zh": "系统与规范", "ja": "システム", "en": "SYSTEM"},
    "kpi_total": {"zh": "总计", "ja": "合計", "en": "TOTAL"},
    "kpi_ready": {"zh": "就绪", "ja": "準備完了", "en": "READY"},
    "kpi_done": {"zh": "完成", "ja": "完了", "en": "DONE"},
    "col_act": {"zh": "操作", "ja": "操作", "en": "ACT"},
    "console_btn": {"zh": "终端日志", "ja": "ログ", "en": "CONSOLE"},
    "console_header": {"zh": "实时管线输出", "ja": "パイプライン出力", "en": "PIPELINE CONSOLE"},
    "pick_script_title": {"zh": "选择台本文件", "ja": "台本ファイルを選択", "en": "Select Script File"},
    "pick_audio_title": {"zh": "选择音频文件", "ja": "音声ファイルを選択", "en": "Select Audio File"},
    "pick_audio_b_title": {"zh": "选择副视角音频文件", "ja": "視点 B 音声を選択", "en": "Select Perspective B Audio"},
    "tip_open": {"zh": "打开输出字幕或所在目录", "ja": "字幕またはフォルダを開く", "en": "Open subtitle or folder"},
    "tip_retry": {"zh": "重新加入待对齐队列", "ja": "再実行キューへ戻す", "en": "Re-queue for alignment"},
    "tip_remove": {"zh": "移除此轨道", "ja": "トラックを削除", "en": "Remove track"},
    "untitled_track": {"zh": "未命名轨道", "ja": "名称未設定", "en": "Untitled Track"},
    "cuda_badge": {"zh": "CUDA FP16", "ja": "CUDA FP16", "en": "CUDA FP16"},
    "cpu_badge": {"zh": "CPU MODE", "ja": "CPU MODE", "en": "CPU MODE"},
    "dnd_hint": {"zh": "支持拖拽导入与槽位互换", "ja": "D&D・スロット入替対応", "en": "Drag & Drop · Slot Swap"},
}


def _etr(key: str) -> str:
    entry = _EXTRA_I18N.get(key)
    if not entry:
        return key
    return entry.get(I18N.current_lang, entry.get("zh", key))


# ══════════════════════════════════════════════════════════════════════════
#  高分屏 (4K / High-DPI) 原生感知与等比缩放系统
# ══════════════════════════════════════════════════════════════════════════

_UI_SCALE: float = 1.0
_DPI_ENABLED: bool = False


def enable_windows_dpi_awareness() -> None:
    """在创建 Tk() 之前启用 Windows Per-Monitor V2 DPI 感知，消除 4K 拉伸发糊。"""
    global _DPI_ENABLED
    if _DPI_ENABLED or sys.platform != "win32":
        return
    _DPI_ENABLED = True
    try:
        import ctypes
        # Win10 1703+ Per-Monitor V2 (-4)
        res = ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        if res:
            return
    except Exception:
        pass
    try:
        import ctypes
        # Win8.1+ Per-Monitor (2) -> System (1)
        for mode in (2, 1):
            if ctypes.windll.shcore.SetProcessDpiAwareness(mode) == 0:
                return
    except Exception:
        pass
    try:
        import ctypes
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def _detect_window_dpi(root: tk.Misc) -> float:
    """读取当前窗口所在显示器的真实 DPI（默认 96.0；4K 150%=144 / 200%=192）。"""
    env_dpi = os.environ.get("FA_ASMR_TEST_DPI", "").strip()
    if env_dpi:
        try:
            v = float(env_dpi)
            if 72.0 <= v <= 384.0:
                return v
        except ValueError:
            pass
    if sys.platform == "win32":
        try:
            import ctypes
            root.update_idletasks()
            hwnd = ctypes.windll.user32.GetParent(root.winfo_id()) or root.winfo_id()
            if hasattr(ctypes.windll.user32, "GetDpiForWindow"):
                dpi = ctypes.windll.user32.GetDpiForWindow(hwnd)
                if dpi and dpi >= 72:
                    return float(dpi)
        except Exception:
            pass
        try:
            import ctypes
            hdc = ctypes.windll.user32.GetDC(0)
            if hdc:
                dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)  # LOGPIXELSX
                ctypes.windll.user32.ReleaseDC(0, hdc)
                if dpi and dpi >= 72:
                    return float(dpi)
        except Exception:
            pass
    try:
        return max(72.0, float(root.winfo_fpixels("1i")))
    except Exception:
        return 96.0


def _init_dpi_scaling(root: tk.Misc) -> float:
    """同步 Tk 内部字体点距缩放与全局像素缩放倍率 _UI_SCALE。"""
    global _UI_SCALE
    dpi = _detect_window_dpi(root)
    _UI_SCALE = max(1.0, round(dpi / 96.0, 3))
    try:
        # 让所有点值字号 (pt) 与物理像素缩放倍率严格对齐
        root.tk.call("tk", "scaling", dpi / 72.0)
    except Exception:
        pass
    return _UI_SCALE


def _s(px: float) -> int:
    """按当前屏幕 DPI 倍率缩放像素整数。"""
    if px == 0:
        return 0
    val = int(round(px * _UI_SCALE))
    return max(1, val) if px > 0 else min(-1, val)


def _sf(px: float) -> float:
    """按当前屏幕 DPI 倍率缩放像素浮点。"""
    return float(px) * _UI_SCALE


def _apply_dark_titlebar(win: tk.Misc) -> None:
    """Windows 10/11 原生深色标题栏适配（失败静默降级）。"""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        win.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(win.winfo_id())
        if not hwnd:
            hwnd = win.winfo_id()
        val = ctypes.c_int(1)
        for attr in (20, 19):
            res = ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, attr, ctypes.byref(val), ctypes.sizeof(val)
            )
            if res == 0:
                break
    except Exception:
        pass


def _build_studio_icon(root: tk.Misc, pil_ok: bool) -> None:
    """用 Pillow 4× 超采样绘制高清示波器窗口图标，适配 4K 标题栏与任务栏。"""
    if not pil_ok:
        return
    try:
        from PIL import Image, ImageDraw, ImageTk

        hi = 256
        im = Image.new("RGBA", (hi, hi), (0, 0, 0, 0))
        draw = ImageDraw.Draw(im)
        draw.rounded_rectangle((8, 8, hi - 9, hi - 9), radius=56,
                               fill=(14, 18, 26, 255), outline=(56, 189, 248, 235), width=8)
        bars = [48, 96, 152, 80, 176, 120, 64]
        bx = 48
        for idx, bh in enumerate(bars):
            y0 = (hi - bh) // 2
            y1 = y0 + bh
            col = (56, 189, 248, 255) if idx % 2 == 0 else (245, 158, 11, 245)
            draw.rounded_rectangle((bx, y0, bx + 16, y1), radius=8, fill=col)
            bx += 24
        im64 = im.resize((64, 64), Image.LANCZOS)
        im32 = im.resize((32, 32), Image.LANCZOS)
        p64 = ImageTk.PhotoImage(im64)
        p32 = ImageTk.PhotoImage(im32)
        root._icon_photos = (p64, p32)
        root.wm_iconphoto(True, p64, p32)
    except Exception:
        pass


def _truncate_filename(name: str, max_len: int = 24) -> str:
    if len(name) <= max_len:
        return name
    head = max(6, max_len - 8)
    return name[:head] + "…" + name[-7:]


# ══════════════════════════════════════════════════════════════════════════
#  自定义暗色控件：ToolTip / ToggleSwitch / SegmentedControl / Oscilloscope
# ══════════════════════════════════════════════════════════════════════════

class ToolTip:
    """暗色控制台风格轻量悬停提示。"""

    def __init__(self, widget: tk.Widget, text: str):
        self.widget = widget
        self.text = text
        self.window = None
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def set_text(self, text: str) -> None:
        self.text = text

    def _show(self, _event=None):
        if self.window or not self.text:
            return
        try:
            self.window = tk.Toplevel(self.widget)
            self.window.wm_overrideredirect(True)
            self.window.attributes("-topmost", True)
            x = self.widget.winfo_rootx() + _s(6)
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + _s(6)
            self.window.wm_geometry(f"+{x}+{y}")
            frame = tk.Frame(self.window, bg="#1E293B", highlightbackground=BORDER_STRONG,
                             highlightthickness=1)
            frame.pack()
            tk.Label(frame, text=self.text, bg="#1E293B", fg=TEXT_PRI,
                     font=(MONO if "\\" in self.text or "/" in self.text else FONT, 8),
                     padx=_s(10), pady=_s(5), justify=tk.LEFT).pack()
        except tk.TclError:
            self._hide()

    def _hide(self, _event=None):
        if self.window:
            try:
                self.window.destroy()
            except tk.TclError:
                pass
            self.window = None


class ToggleSwitch(tk.Canvas):
    """自定义滑动胶囊开关（高分屏优先 4× PIL 超采样抗锯齿，无 PIL 时平滑降级）。"""

    def __init__(self, parent, variable: tk.BooleanVar, command=None, bg=BG_RAIL, pil_ok: bool = True):
        self._sw = _s(42)
        self._sh = _s(22)
        super().__init__(parent, width=self._sw, height=self._sh, bg=bg, bd=0,
                         highlightthickness=0, cursor="hand2")
        self._var = variable
        self._cmd = command
        self._bg = bg
        self._pil_ok = pil_ok
        self._disabled = False
        self._hover = False
        self._photo_cache: dict[tuple, object] = {}
        self.bind("<Button-1>", self._toggle)
        self.bind("<Enter>", lambda _e: self._set_hover(True))
        self.bind("<Leave>", lambda _e: self._set_hover(False))
        self.redraw()

    def set_disabled(self, disabled: bool) -> None:
        self._disabled = disabled
        self.configure(cursor="" if disabled else "hand2")
        self.redraw()

    def _set_hover(self, h: bool) -> None:
        self._hover = h
        self.redraw()

    def _toggle(self, _event=None):
        if self._disabled:
            return
        self._var.set(not self._var.get())
        self.redraw()
        if self._cmd:
            self._cmd()

    @staticmethod
    def _hex_rgba(hx: str) -> tuple[int, int, int, int]:
        hx = hx.lstrip("#")
        return (int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16), 255)

    def redraw(self):
        self.delete("all")
        on = bool(self._var.get())
        if self._disabled:
            track = "#1E2533"
            knob = "#475569"
        else:
            track = (ACCENT_HOVER if self._hover else ACCENT) if on else ("#2A3448" if self._hover else "#21293A")
            knob = "#F8FAFC"

        if self._pil_ok:
            key = (on, self._disabled, self._hover, self._sw, self._sh)
            photo = self._photo_cache.get(key)
            if photo is None:
                try:
                    from PIL import Image, ImageDraw, ImageTk
                    ss = 4
                    W, H = self._sw * ss, self._sh * ss
                    pad = max(2, int(round(ss * _UI_SCALE)))
                    im = Image.new("RGBA", (W, H), self._hex_rgba(self._bg))
                    d = ImageDraw.Draw(im)
                    rad = (H - 2 * pad) // 2
                    d.rounded_rectangle((pad, pad, W - pad - 1, H - pad - 1),
                                        radius=rad, fill=self._hex_rgba(track))
                    kr = int(round((H - 2 * pad) * 0.36))
                    cy = H // 2
                    cx = (W - pad - rad) if on else (pad + rad)
                    d.ellipse((cx - kr, cy - kr, cx + kr, cy + kr), fill=self._hex_rgba(knob))
                    im_out = im.resize((self._sw, self._sh), Image.LANCZOS)
                    photo = ImageTk.PhotoImage(im_out)
                    self._photo_cache[key] = photo
                except Exception:
                    self._pil_ok = False
                    photo = None
            if photo is not None:
                self.create_image(0, 0, anchor="nw", image=photo)
                return

        # 无 PIL 时的等比 Canvas 回退绘制
        pad = _s(1)
        r = (self._sh - 2 * pad) // 2
        self.create_oval(pad, pad, pad + 2 * r, pad + 2 * r, fill=track, outline="")
        self.create_oval(self._sw - pad - 2 * r, pad, self._sw - pad, pad + 2 * r, fill=track, outline="")
        self.create_rectangle(pad + r, pad, self._sw - pad - r, pad + 2 * r, fill=track, outline="")
        kr = _s(7)
        cx = (self._sw - pad - r) if on else (pad + r)
        cy = self._sh // 2
        self.create_oval(cx - kr, cy - kr, cx + kr, cy + kr, fill=knob, outline="")


class SegmentedControl(tk.Frame):
    """横向分段胶囊选择器，用于设备切换（CUDA/CPU）与语言切换（中/日/EN）。"""

    def __init__(self, parent, options, value: str, on_change, bg=BG_APP):
        super().__init__(parent, bg=bg, highlightbackground=BORDER, highlightthickness=1,
                         padx=_s(2), pady=_s(2))
        self._seg_items = list(options)  # [(key, label), ...]
        self._value = value
        self._on_change = on_change
        self._disabled_keys: set[str] = set()
        self._disabled_all: bool = False
        self._buttons: dict[str, tk.Label] = {}
        self._build()

    def _build(self):
        for w in self.winfo_children():
            w.destroy()
        self._buttons.clear()
        for idx, (key, label) in enumerate(self._seg_items):
            lbl = tk.Label(self, text=label, font=(FONT, 8, "bold"),
                           pady=_s(5), padx=_s(6), cursor="hand2")
            lbl.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0 if idx == 0 else _s(2), 0))
            lbl.bind("<Button-1>", lambda _e, k=key: self._select(k))
            lbl.bind("<Enter>", lambda _e, k=key: self._paint(hover_key=k))
            lbl.bind("<Leave>", lambda _e: self._paint())
            self._buttons[key] = lbl
        self._paint()

    def set_disabled_keys(self, keys) -> None:
        self._disabled_keys = set(keys)
        self._paint()

    def set_disabled(self, disabled: bool) -> None:
        self._disabled_all = bool(disabled)
        self._paint()

    def set_value(self, value: str) -> None:
        self._value = value
        self._paint()

    def _select(self, key: str):
        if self._disabled_all or key in self._disabled_keys or key == self._value:
            return
        self._value = key
        self._paint()
        if self._on_change:
            self._on_change(key)

    def _paint(self, hover_key: str | None = None):
        for key, lbl in self._buttons.items():
            if self._disabled_all or key in self._disabled_keys:
                if key == self._value:
                    lbl.configure(bg="#132230", fg="#4F708A", cursor="")
                else:
                    lbl.configure(bg=BG_APP, fg="#3B4559", cursor="")
            elif key == self._value:
                lbl.configure(bg=ACCENT_DIM, fg=ACCENT_HOVER, cursor="hand2")
            elif key == hover_key:
                lbl.configure(bg=BG_SLOT, fg=TEXT_PRI, cursor="hand2")
            else:
                lbl.configure(bg=BG_APP, fg=TEXT_MUTED, cursor="hand2")


class OscilloscopeHeader(tk.Canvas):
    """左侧导轨顶部：双声道示波器美学抬头 + 版本与硬件遥测（全 DPI 自适应测量，零重叠）。"""

    def __init__(self, parent, cuda_ok: bool):
        super().__init__(parent, height=_s(84), bg="#0D1118", bd=0, highlightthickness=0)
        self._cuda_ok = cuda_ok
        self._phase = 0.0
        self._running = False
        self._after_id = None
        self.bind("<Configure>", lambda _e: self.redraw())

    def set_running(self, running: bool) -> None:
        self._running = running
        if running and self._after_id is None:
            self._tick()
        elif not running and self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None
            self.redraw()

    def stop(self) -> None:
        self._running = False
        if self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None

    def _tick(self):
        if not self._running:
            self._after_id = None
            return
        self._phase += 0.24
        self.redraw()
        try:
            self._after_id = self.after(65, self._tick)
        except tk.TclError:
            self._after_id = None

    def redraw(self):
        self.delete("all")
        w = max(self.winfo_width(), _s(240))
        h = max(self.winfo_height(), _s(84))

        # 微弱示波器网格线（置于底部波段，与抬头文案零重叠）
        mid_y = int(h * 0.78)
        self.create_line(0, mid_y, w, mid_y, fill="#182030", width=1)
        grid_step = _s(36)
        tick_h = _s(10)
        for gx in range(_s(28), w, grid_step):
            self.create_line(gx, mid_y - tick_h, gx, mid_y + tick_h, fill="#141B27", width=1)

        # 双声道立体声包络曲线（L = Cyan, R = Amber）
        amp = _sf(10.5 if self._running else 6.5)
        pts_l = []
        pts_r = []
        step = max(3, _s(5))
        for x in range(0, w + step, step):
            u = min(1.0, x / max(w, 1))
            env = max(0.0, math.sin(u * math.pi)) ** 1.3
            yl = mid_y + math.sin(u * 15.0 + self._phase) * math.cos(u * 5.0 - self._phase * 0.4) * amp * env
            yr = mid_y + math.sin(u * 12.0 - self._phase * 0.8 + 1.2) * (amp * 0.72) * env
            pts_l.extend((x, yl))
            pts_r.extend((x, yr))
        if len(pts_r) >= 4:
            self.create_line(*pts_r, fill="#63481A" if not self._running else "#D97706",
                             width=max(1.0, _sf(1.2)), smooth=True)
        if len(pts_l) >= 4:
            self.create_line(*pts_l, fill="#1E5878" if not self._running else ACCENT_HOVER,
                             width=max(1.2, _sf(1.6)), smooth=True)

        # 品牌字标（实时测量 bbox 放置版本号，避免任何 DPI 下文字重叠）
        pad_x = _s(14)
        top_y = _s(13)
        brand_id = self.create_text(pad_x, top_y, anchor="nw", text="FA-ASMR", fill=TEXT_PRI,
                                    font=(MONO, 13, "bold"))
        bb = self.bbox(brand_id) or (pad_x, top_y, pad_x + _s(84), top_y + _s(20))
        mid_row = (bb[1] + bb[3]) // 2

        ver = tr("banner_badge")
        vx0 = bb[2] + _s(8)
        ver_txt_id = self.create_text(vx0 + _s(6), mid_row, anchor="w", text=ver,
                                      fill=ACCENT_HOVER, font=(MONO, 8, "bold"))
        vbb = self.bbox(ver_txt_id) or (vx0 + _s(6), mid_row - _s(7), vx0 + _s(44), mid_row + _s(7))
        v_rect = self.create_rectangle(vbb[0] - _s(5), vbb[1] - _s(2), vbb[2] + _s(5), vbb[3] + _s(2),
                                       fill=ACCENT_DIM, outline="#1C4F6E", width=1)
        self.tag_lower(v_rect, ver_txt_id)

        # 硬件状态 LED 胶囊（右对齐动态测量宽度）
        hw_color = EMERALD if self._cuda_ok else AMBER
        hw_bg = EMERALD_DIM if self._cuda_ok else AMBER_DIM
        hw_txt = _etr("cuda_badge") if self._cuda_ok else _etr("cpu_badge")
        rx = w - pad_x
        hw_txt_id = self.create_text(rx - _s(6), mid_row, anchor="e", text=hw_txt,
                                     fill=hw_color, font=(MONO, 7, "bold"))
        hbb = self.bbox(hw_txt_id) or (rx - _s(68), mid_row - _s(6), rx - _s(6), mid_row + _s(6))
        dot_r = _s(3)
        pill_x0 = hbb[0] - _s(14)
        pill_y0 = hbb[1] - _s(3)
        pill_y1 = hbb[3] + _s(3)
        hw_rect = self.create_rectangle(pill_x0, pill_y0, rx, pill_y1,
                                        fill=hw_bg, outline=hw_color, width=1)
        self.tag_lower(hw_rect, hw_txt_id)
        dot_cx = pill_x0 + _s(7)
        self.create_oval(dot_cx - dot_r, mid_row - dot_r, dot_cx + dot_r, mid_row + dot_r,
                         fill=hw_color, outline="")

        # 副标题
        self.create_text(pad_x, bb[3] + _s(4), anchor="nw", text=tr("banner_subtitle"),
                         fill=TEXT_SEC, font=(FONT, 8))
        # 底部分割线
        self.create_line(0, h - 1, w, h - 1, fill=BORDER, width=1)


class TransportProgressCanvas(tk.Canvas):
    """底部走带栏：平滑进度条 + 运行中 10 段频谱电平条。"""

    def __init__(self, parent):
        super().__init__(parent, height=_s(18), bg=BG_RAIL, bd=0, highlightthickness=0)
        self._value = 0.0
        self._running = False
        self._phase = 0.0
        self._after_id = None
        self.bind("<Configure>", lambda _e: self.redraw())

    def set_value(self, value: float) -> None:
        self._value = max(0.0, min(100.0, float(value)))
        self.redraw()

    def set_running(self, running: bool) -> None:
        self._running = running
        if running and self._after_id is None:
            self._tick()
        elif not running and self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None
            self.redraw()

    def stop(self) -> None:
        self._running = False
        if self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None

    def _tick(self):
        if not self._running:
            self._after_id = None
            return
        self._phase += 0.38
        self.redraw()
        try:
            self._after_id = self.after(75, self._tick)
        except tk.TclError:
            self._after_id = None

    def redraw(self):
        self.delete("all")
        w = max(self.winfo_width(), _s(160))
        h = max(self.winfo_height(), _s(18))

        # 右侧预留等比空间给迷你频谱电平条
        pitch = _s(5)
        bar_thick = _s(3)
        meter_w = pitch * 10
        gap = _s(14)
        bar_w = max(_s(60), w - meter_w - gap)
        y0, y1 = _s(6), _s(12)
        self.create_rectangle(0, y0, bar_w, y1, fill="#1A2232", outline="")
        fill_w = int(bar_w * (self._value / 100.0))
        if fill_w > 0:
            col = EMERALD if (self._value >= 100.0 and not self._running) else ACCENT
            self.create_rectangle(0, y0, fill_w, y1, fill=col, outline="")
            if fill_w > _s(4):
                cap = "#6EE7B7" if col == EMERALD else ACCENT_GLOW
                self.create_rectangle(fill_w - _s(3), y0, fill_w, y1, fill=cap, outline="")

        # 10 段实时声学电平
        mx = bar_w + gap
        min_bh = _s(3)
        margin_b = _s(2)
        for idx in range(10):
            if self._running:
                lv = 0.25 + 0.75 * abs(math.sin(self._phase + idx * 0.65))
                bh = max(min_bh, int(lv * (h - _s(4))))
                c = ACCENT_HOVER if idx < 7 else AMBER
            else:
                bh = min_bh
                c = "#232D40"
            bx = mx + idx * pitch
            self.create_rectangle(bx, h - margin_b - bh, bx + bar_thick, h - margin_b, fill=c, outline="")


# ══════════════════════════════════════════════════════════════════════════
#  暗色台本规范指南弹窗
# ══════════════════════════════════════════════════════════════════════════

def show_script_help_dialog(parent):
    """暗色控制台风格的台本格式与规范指南弹窗。"""
    dlg = tk.Toplevel(parent)
    dlg.title(tr("guide_dlg_window_title"))
    sw, sh = dlg.winfo_screenwidth(), dlg.winfo_screenheight()
    dw = min(_s(720), int(sw * 0.85))
    dh = min(_s(620), int(sh * 0.85))
    dlg.geometry(f"{dw}x{dh}")
    dlg.minsize(min(_s(580), dw), min(_s(480), dh))
    dlg.configure(bg=BG_RAIL)
    dlg.transient(parent)
    dlg.grab_set()
    dlg.update_idletasks()
    px = parent.winfo_x() + max(0, (parent.winfo_width() - dw) // 2)
    py = parent.winfo_y() + max(0, (parent.winfo_height() - dh) // 2)
    dlg.geometry(f"+{px}+{py}")
    _apply_dark_titlebar(dlg)

    head = tk.Frame(dlg, bg=BG_PANEL_ALT, padx=_s(24), pady=_s(16))
    head.pack(fill=tk.X)
    tk.Label(head, text="SPECIFICATION", font=(MONO, 8, "bold"),
             fg=ACCENT_HOVER, bg=BG_PANEL_ALT).pack(anchor="w")
    tk.Label(head, text=tr("guide_dlg_heading"), font=(FONT, 14, "bold"),
             fg=TEXT_PRI, bg=BG_PANEL_ALT).pack(anchor="w", pady=(_s(2), _s(2)))
    tk.Label(head, text=tr("guide_dlg_subheading"), font=(FONT, 9),
             fg=TEXT_SEC, bg=BG_PANEL_ALT).pack(anchor="w")
    tk.Frame(dlg, bg=BORDER, height=1).pack(fill=tk.X)

    body = tk.Frame(dlg, bg=BG_RAIL, padx=_s(24), pady=_s(16))
    body.pack(fill=tk.BOTH, expand=True)

    # 先固定底部按钮区，确保在任意弹窗高度下永不裁切
    foot = tk.Frame(body, bg=BG_RAIL)
    foot.pack(side=tk.BOTTOM, fill=tk.X, pady=(_s(12), 0))

    txt_wrap = tk.Frame(body, bg=BG_CONSOLE, highlightbackground=BORDER, highlightthickness=1)
    txt_wrap.pack(fill=tk.BOTH, expand=True)
    txt_box = tk.Text(txt_wrap, bg=BG_CONSOLE, fg="#CBD5E1", font=(FONT, 10),
                      height=1, relief=tk.FLAT, wrap=tk.WORD, padx=_s(16), pady=_s(14),
                      spacing1=_s(3), spacing3=_s(3), selectbackground=ACCENT_DIM,
                      selectforeground=TEXT_PRI)
    scroll = ttk.Scrollbar(txt_wrap, orient=tk.VERTICAL, command=txt_box.yview,
                           style="Studio.Vertical.TScrollbar")
    txt_box.configure(yscrollcommand=scroll.set)
    scroll.pack(side=tk.RIGHT, fill=tk.Y)
    txt_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    sample_text = I18N.get_guide_text()
    txt_box.insert(tk.END, sample_text)
    txt_box.configure(state=tk.DISABLED)

    def _copy_sample():
        parent.clipboard_clear()
        parent.clipboard_append(sample_text)
        cp_btn.configure(text=tr("guide_btn_copied"), fg=EMERALD)
        dlg.after(1800, lambda: cp_btn.configure(text=tr("guide_btn_copy"), fg=TEXT_PRI))

    cp_btn = tk.Button(foot, text=tr("guide_btn_copy"), font=(FONT, 9),
                       fg=TEXT_PRI, bg=BG_SLOT, activebackground=BG_SLOT_HOVER,
                       activeforeground=TEXT_PRI, relief=tk.FLAT, bd=0,
                       padx=_s(16), pady=_s(8), cursor="hand2", command=_copy_sample)
    cp_btn.pack(side=tk.LEFT)
    tk.Button(foot, text=tr("guide_btn_done"), font=(FONT, 9, "bold"),
              fg=INK_DARK, bg=ACCENT, activebackground=ACCENT_HOVER,
              activeforeground=INK_DARK, relief=tk.FLAT, bd=0,
              padx=_s(22), pady=_s(8), cursor="hand2", command=dlg.destroy).pack(side=tk.RIGHT)


# ══════════════════════════════════════════════════════════════════════════
#  主工作台构建入口
# ══════════════════════════════════════════════════════════════════════════

def build_ui(ctx):
    """创建 FA-ASMR 专业音频控制台前端。ctx 提供业务回调与运行环境信息。"""
    enable_windows_dpi_awareness()
    root = ctx.root
    _init_dpi_scaling(root)

    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    init_w = min(_s(1240), int(sw * 0.92))
    init_h = min(_s(820), int(sh * 0.88))
    min_w = min(_s(980), int(sw * 0.85))
    min_h = min(_s(640), int(sh * 0.80))

    root.title(tr("app_title"))
    root.geometry(f"{init_w}x{init_h}")
    root.minsize(min_w, min_h)
    root.configure(bg=BG_APP)
    _build_studio_icon(root, bool(ctx.pil))

    # 全局 ttk 暗色滚动条样式（按 DPI 缩放宽度）
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("Studio.Vertical.TScrollbar",
                    background="#242E42", troughcolor=BG_STAGE,
                    borderwidth=0, arrowsize=0, width=_s(8), relief=tk.FLAT)
    style.map("Studio.Vertical.TScrollbar",
              background=[("active", "#334155"), ("pressed", ACCENT)])

    # 线程安全 UI 调度队列
    ui_events: queue.Queue = queue.Queue()
    closing = False

    def _post_ui(callback):
        ui_events.put(callback)

    def _drain_ui_events():
        while True:
            try:
                cb = ui_events.get_nowait()
            except queue.Empty:
                break
            try:
                cb()
            except tk.TclError:
                if not closing:
                    raise
        if not closing:
            root.after(30, _drain_ui_events)

    root.after(30, _drain_ui_events)

    # ── 状态变量与设置持久化（与旧版 JSON 结构 100% 兼容） ──────────
    lang_var = tk.StringVar(value=I18N.current_lang)
    mode_var = tk.StringVar(value="single")
    dev_var = tk.StringVar(value="cuda" if ctx.cuda_ok else "cpu")
    end_extend_var = tk.StringVar(value="0.8")
    ono_gate_var = tk.BooleanVar(value=True)
    multi_order_var = tk.BooleanVar(value=False)
    multi_persp_var = tk.BooleanVar(value=False)

    _var_meta = [
        (lang_var, "lang", str),
        (mode_var, "align_mode", str),
        (dev_var, "dev", str),
        (end_extend_var, "end_extend", str),
        (ono_gate_var, "ono_gate", lambda v: v in (True, "true", "True", 1, "1")),
        (multi_order_var, "multi_order", lambda v: v in (True, "true", "True", 1, "1")),
        (multi_persp_var, "multi_persp", lambda v: v in (True, "true", "True", 1, "1")),
    ]

    def _save_settings():
        try:
            cur = mode_var.get()
            multi_order_var.set(cur in ("multi_speaker", "multi_persp"))
            multi_persp_var.set(cur == "multi_persp")
            with open(ctx.settings_file, "w", encoding="utf-8") as f:
                json.dump({k: v.get() for v, k, _ in _var_meta}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _load_settings():
        if not os.path.isfile(ctx.settings_file):
            init_lang = detect_system_lang()
            I18N.set_lang(init_lang)
            lang_var.set(init_lang)
            return
        try:
            with open(ctx.settings_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for var, key, conv in _var_meta:
                if key in data:
                    var.set(conv(data[key]))
            if not ctx.cuda_ok:
                dev_var.set("cpu")
            cur_lang = data["lang"] if data.get("lang") in LANG_MAP else detect_system_lang()
            I18N.set_lang(cur_lang)
            lang_var.set(cur_lang)
            if "align_mode" not in data:
                mode_var.set("multi_persp" if data.get("multi_persp") else
                             "multi_speaker" if data.get("multi_order") else "single")
        except Exception:
            pass

    _load_settings()

    # ══════════════════════════════════════════════════════════════════════
    #  主双栏容器：Left Control Rail (284px) | Center Studio Stage (Flex)
    # ══════════════════════════════════════════════════════════════════════
    rail_w = _s(284)
    stage_min_w = max(_s(480), min_w - rail_w - 2)
    shell = tk.Frame(root, bg=BG_APP)
    shell.pack(fill=tk.BOTH, expand=True)
    shell.grid_columnconfigure(0, weight=0, minsize=rail_w)
    shell.grid_columnconfigure(1, weight=0, minsize=1)
    shell.grid_columnconfigure(2, weight=1, minsize=stage_min_w)
    shell.grid_rowconfigure(0, weight=1)

    rail = tk.Frame(shell, bg=BG_RAIL, width=rail_w)
    rail.grid(row=0, column=0, sticky="nsew")
    rail.pack_propagate(False)

    tk.Frame(shell, bg=BORDER, width=1).grid(row=0, column=1, sticky="ns")

    stage = tk.Frame(shell, bg=BG_STAGE)
    stage.grid(row=0, column=2, sticky="nsew")

    # ── 1. 左侧控制导轨（顶：示波器 | 底：固定系统区 | 中：自适应可滚参数区）──
    scope_header = OscilloscopeHeader(rail, cuda_ok=bool(ctx.cuda_ok))
    scope_header.pack(side=tk.TOP, fill=tk.X)

    # 1.3 底部系统区（优先 pack(BOTTOM)，确保任意窗口高度下语言切换与指南永不裁切）
    rail_footer = tk.Frame(rail, bg=BG_RAIL, padx=_s(14), pady=_s(10))
    rail_footer.pack(side=tk.BOTTOM, fill=tk.X)
    tk.Frame(rail_footer, bg=BORDER, height=1).pack(fill=tk.X, pady=(0, _s(8)))
    sec_sys_lbl = tk.Label(rail_footer, text=_etr("rail_section_sys"),
                           font=(FONT, 8, "bold"), fg=TEXT_MUTED, bg=BG_RAIL)
    sec_sys_lbl.pack(anchor="w", pady=(0, _s(5)))

    def _on_lang_select(code: str):
        I18N.set_lang(code)
        lang_var.set(code)
        _refresh_all_i18n()
        _save_settings()

    lang_seg = SegmentedControl(
        rail_footer,
        options=[("zh", "中文"), ("ja", "日本語"), ("en", "EN")],
        value=lang_var.get(),
        on_change=_on_lang_select,
    )
    lang_seg.pack(fill=tk.X, pady=(0, _s(6)))

    help_btn = tk.Button(rail_footer, text=f"?  {tr('btn_show_guide')}", font=(FONT, 8, "bold"),
                         fg=TEXT_SEC, bg=BG_PANEL, activebackground=BG_SLOT_HOVER,
                         activeforeground=TEXT_PRI, relief=tk.FLAT, bd=0,
                         highlightbackground=BORDER, highlightthickness=1,
                         pady=_s(6), cursor="hand2", command=lambda: show_script_help_dialog(root))
    help_btn.pack(fill=tk.X)

    # 中部参数导轨：默认高度无需滚动，缩至最小尺寸时自动允许滚轮滚动
    rail_scroll_cv = tk.Canvas(rail, bg=BG_RAIL, bd=0, highlightthickness=0)
    rail_scroll_cv.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    rail_body = tk.Frame(rail_scroll_cv, bg=BG_RAIL, padx=_s(14), pady=_s(10))
    rail_win_id = rail_scroll_cv.create_window((0, 0), window=rail_body, anchor="nw")

    def _sync_rail_scroll(_e=None):
        rail_scroll_cv.configure(scrollregion=rail_scroll_cv.bbox("all"))

    rail_scroll_cv.bind("<Configure>", lambda e: rail_scroll_cv.itemconfig(rail_win_id, width=e.width))
    rail_body.bind("<Configure>", _sync_rail_scroll)

    def _on_rail_wheel(event):
        if rail_body.winfo_reqheight() > rail_scroll_cv.winfo_height():
            rail_scroll_cv.yview_scroll(int(-1 * (event.delta / 120)), "units")

    rail_scroll_cv.bind("<Enter>", lambda _e: root.bind_all("<MouseWheel>", _on_rail_wheel))
    rail_scroll_cv.bind("<Leave>", lambda _e: root.unbind_all("<MouseWheel>"))

    # 1.1 引擎模式选择组
    sec_engine_lbl = tk.Label(rail_body, text=_etr("rail_section_engine"),
                              font=(FONT, 8, "bold"), fg=TEXT_MUTED, bg=BG_RAIL)
    sec_engine_lbl.pack(anchor="w", pady=(0, _s(6)))

    mode_tiles: dict[str, dict] = {}
    _MODE_CODES = {"single": "SEQ", "multi_speaker": "SPK", "multi_persp": "POV"}
    _MODE_ACCENTS = {"single": ACCENT_HOVER, "multi_speaker": ACCENT_HOVER, "multi_persp": AMBER}
    _MODE_DIMS = {"single": ACCENT_DIM, "multi_speaker": ACCENT_DIM, "multi_persp": AMBER_DIM}

    def _select_mode(key: str):
        if ctx.is_running():
            return
        mode_var.set(key)
        _on_mode_change()

    for m_key in ("single", "multi_speaker", "multi_persp"):
        tile = tk.Frame(rail_body, bg=BG_PANEL, highlightbackground=BORDER_SUBTLE,
                        highlightthickness=1, cursor="hand2")
        tile.pack(fill=tk.X, pady=(0, _s(5)))

        bar = tk.Frame(tile, bg=BORDER, width=_s(3))
        bar.pack(side=tk.LEFT, fill=tk.Y)

        inner = tk.Frame(tile, bg=BG_PANEL, padx=_s(9), pady=_s(6))
        inner.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        row_top = tk.Frame(inner, bg=BG_PANEL)
        row_top.pack(fill=tk.X)

        tag_lbl = tk.Label(row_top, text=_MODE_CODES[m_key], font=(MONO, 7, "bold"),
                           fg=TEXT_MUTED, bg=BG_SLOT, padx=_s(5), pady=_s(1))
        tag_lbl.pack(side=tk.RIGHT)

        title_lbl = tk.Label(row_top, text=tr(f"mode_{m_key}_title"),
                             font=(FONT, 9, "bold"), fg=TEXT_PRI, bg=BG_PANEL)
        title_lbl.pack(side=tk.LEFT)

        sub_lbl = tk.Label(inner, text=tr(f"mode_{m_key}_sub"), font=(FONT, 8),
                           fg=TEXT_MUTED, bg=BG_PANEL, anchor="w", justify=tk.LEFT, wraplength=_s(216))
        sub_lbl.pack(fill=tk.X, pady=(_s(2), 0))

        mode_tiles[m_key] = {
            "tile": tile, "bar": bar, "inner": inner,
            "row_top": row_top, "tag": tag_lbl, "title": title_lbl, "sub": sub_lbl,
        }

        def _bind_tile(w, k=m_key):
            w.bind("<Button-1>", lambda _e: _select_mode(k))
            w.bind("<Enter>", lambda _e: _paint_mode_tiles(hover_key=k))
            w.bind("<Leave>", lambda _e: _paint_mode_tiles())

        for w in (tile, bar, inner, row_top, tag_lbl, title_lbl, sub_lbl):
            _bind_tile(w)

    # 模式说明微面板
    mode_desc_box = tk.Frame(rail_body, bg=BG_APP, highlightbackground=BORDER_SUBTLE,
                             highlightthickness=1, padx=_s(9), pady=_s(6))
    mode_desc_box.pack(fill=tk.X, pady=(_s(1), _s(10)))
    mode_desc_var = tk.StringVar(value=tr(f"mode_{mode_var.get()}_desc"))
    mode_desc_lbl = tk.Label(mode_desc_box, textvariable=mode_desc_var, font=(FONT, 8),
                             fg=TEXT_SEC, bg=BG_APP, anchor="w", justify=tk.LEFT, wraplength=_s(230))
    mode_desc_lbl.pack(fill=tk.X)

    def _paint_mode_tiles(hover_key: str | None = None):
        cur = mode_var.get()
        for k, refs in mode_tiles.items():
            sel = k == cur
            hov = (k == hover_key) and not sel
            ac = _MODE_ACCENTS[k]
            dim = _MODE_DIMS[k]
            bg_c = "#17202D" if sel else (BG_SLOT if hov else BG_PANEL)
            refs["tile"].configure(bg=bg_c, highlightbackground=ac if sel else (BORDER_STRONG if hov else BORDER_SUBTLE))
            refs["bar"].configure(bg=ac if sel else (BORDER_STRONG if hov else BORDER_SUBTLE))
            refs["inner"].configure(bg=bg_c)
            refs["row_top"].configure(bg=bg_c)
            refs["title"].configure(bg=bg_c, fg=TEXT_PRI if (sel or hov) else TEXT_SEC)
            refs["sub"].configure(bg=bg_c, fg=TEXT_SEC if sel else TEXT_MUTED)
            refs["tag"].configure(bg=dim if sel else BG_SLOT, fg=ac if sel else TEXT_MUTED)

    # 1.2 运行参数区
    tk.Frame(rail_body, bg=BORDER, height=1).pack(fill=tk.X, pady=(0, _s(8)))
    sec_params_lbl = tk.Label(rail_body, text=_etr("rail_section_params"),
                              font=(FONT, 8, "bold"), fg=TEXT_MUTED, bg=BG_RAIL)
    sec_params_lbl.pack(anchor="w", pady=(0, _s(7)))

    # 驻留延长（步进器 + 快捷预设）
    ext_head = tk.Frame(rail_body, bg=BG_RAIL)
    ext_head.pack(fill=tk.X, pady=(0, _s(4)))
    ext_title_lbl = tk.Label(ext_head, text=tr("label_end_extend"), font=(FONT, 8),
                             fg=TEXT_SEC, bg=BG_RAIL)
    ext_title_lbl.pack(side=tk.LEFT)
    ext_unit_lbl = tk.Label(ext_head, text=tr("label_sec"), font=(FONT, 8),
                            fg=TEXT_MUTED, bg=BG_RAIL)
    ext_unit_lbl.pack(side=tk.RIGHT)

    ext_ctrl = tk.Frame(rail_body, bg=BG_APP, highlightbackground=BORDER, highlightthickness=1)
    ext_ctrl.pack(fill=tk.X, pady=(0, _s(5)))

    def _step_extend(delta: float):
        if ctx.is_running():
            return
        try:
            v = float(end_extend_var.get() or 0)
        except ValueError:
            v = 0.0
        v = max(0.0, round(v + delta, 1))
        end_extend_var.set(f"{v:.1f}")
        _refresh_preset_chips()
        _save_settings()

    btn_ext_minus = tk.Button(ext_ctrl, text="−", font=(MONO, 10, "bold"), fg=TEXT_SEC,
                              bg=BG_SLOT, activebackground=BG_SLOT_HOVER, activeforeground=TEXT_PRI,
                              relief=tk.FLAT, bd=0, width=3, cursor="hand2",
                              command=lambda: _step_extend(-0.1))
    btn_ext_minus.pack(side=tk.LEFT, fill=tk.Y)

    ext_entry = tk.Entry(ext_ctrl, textvariable=end_extend_var, font=(MONO, 10, "bold"),
                         bg=BG_APP, fg=ACCENT_HOVER, insertbackground=ACCENT_HOVER,
                         relief=tk.FLAT, bd=0, justify="center")
    ext_entry.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, ipady=_s(4))
    ext_entry.bind("<FocusOut>", lambda _e: (_refresh_preset_chips(), _save_settings()))
    ext_entry.bind("<Return>", lambda _e: (_refresh_preset_chips(), _save_settings()))

    btn_ext_plus = tk.Button(ext_ctrl, text="＋", font=(MONO, 10, "bold"), fg=TEXT_SEC,
                             bg=BG_SLOT, activebackground=BG_SLOT_HOVER, activeforeground=TEXT_PRI,
                             relief=tk.FLAT, bd=0, width=3, cursor="hand2",
                             command=lambda: _step_extend(0.1))
    btn_ext_plus.pack(side=tk.RIGHT, fill=tk.Y)

    # 快捷预设芯片
    preset_row = tk.Frame(rail_body, bg=BG_RAIL)
    preset_row.pack(fill=tk.X, pady=(0, _s(10)))
    preset_btns: dict[str, tk.Label] = {}

    def _set_preset(val_s: str):
        if ctx.is_running():
            return
        end_extend_var.set(val_s)
        _refresh_preset_chips()
        _save_settings()

    def _refresh_preset_chips():
        try:
            cur_v = f"{float(end_extend_var.get()):.1f}"
        except ValueError:
            cur_v = ""
        for val_s, lbl in preset_btns.items():
            active = val_s == cur_v
            lbl.configure(bg=ACCENT_DIM if active else BG_PANEL,
                          fg=ACCENT_HOVER if active else TEXT_MUTED,
                          highlightbackground=ACCENT if active else BORDER_SUBTLE)

    for idx, pv in enumerate(("0.0", "0.5", "0.8", "1.5")):
        chip = tk.Label(preset_row, text=f"{pv}s", font=(MONO, 8), pady=_s(2),
                        bg=BG_PANEL, fg=TEXT_MUTED, highlightbackground=BORDER_SUBTLE,
                        highlightthickness=1, cursor="hand2")
        chip.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0 if idx == 0 else _s(4), 0))
        chip.bind("<Button-1>", lambda _e, v=pv: _set_preset(v))
        preset_btns[pv] = chip
    _refresh_preset_chips()

    # Ono-Gate 自定义超采样滑块开关
    ono_row = tk.Frame(rail_body, bg=BG_RAIL)
    ono_row.pack(fill=tk.X, pady=(0, _s(10)))
    ono_lbl = tk.Label(ono_row, text=tr("check_ono_gate"), font=(FONT, 8),
                       fg=TEXT_PRI, bg=BG_RAIL, anchor="w", wraplength=_s(192), justify=tk.LEFT)
    ono_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
    ono_switch = ToggleSwitch(ono_row, ono_gate_var, command=_save_settings,
                              bg=BG_RAIL, pil_ok=bool(ctx.pil))
    ono_switch.pack(side=tk.RIGHT, padx=(_s(6), 0))

    # 计算设备分段切换
    dev_lbl = tk.Label(rail_body, text=tr("label_device"), font=(FONT, 8),
                       fg=TEXT_SEC, bg=BG_RAIL)
    dev_lbl.pack(anchor="w", pady=(0, _s(4)))
    dev_seg = SegmentedControl(
        rail_body,
        options=[("cuda", "CUDA GPU"), ("cpu", "CPU")],
        value=dev_var.get(),
        on_change=lambda k: (dev_var.set(k), _save_settings()),
    )
    if not ctx.cuda_ok:
        dev_seg.set_disabled_keys(["cuda"])
    dev_seg.pack(fill=tk.X, pady=(0, _s(4)))

    # ══════════════════════════════════════════════════════════════════════
    #  2. 中央工作台（顶栏 + 未配对暂存坞 + 轨道矩阵 + 终端抽屉 + 走带栏）
    # ══════════════════════════════════════════════════════════════════════

    # 2.1 顶部工具栏（右侧按钮优先 pack，窄屏永不裁切）
    topbar = tk.Frame(stage, bg=BG_PANEL_ALT, height=_s(52), padx=_s(18))
    topbar.pack(fill=tk.X)
    topbar.pack_propagate(False)

    tb_right = tk.Frame(topbar, bg=BG_PANEL_ALT)
    tb_right.pack(side=tk.RIGHT, fill=tk.Y, pady=_s(10))

    clear_btn = tk.Button(tb_right, text=tr("btn_clear"), font=(FONT, 9),
                          fg=TEXT_SEC, bg=BG_SLOT, activebackground=BG_SLOT_HOVER,
                          activeforeground=TEXT_PRI, relief=tk.FLAT, bd=0,
                          padx=_s(12), pady=_s(4), cursor="hand2")
    clear_btn.pack(side=tk.RIGHT)

    add_btn = tk.Button(tb_right, text=_etr("import_files"), font=(FONT, 9, "bold"),
                        fg=ACCENT_HOVER, bg=ACCENT_DIM, activebackground="#163E58",
                        activeforeground=TEXT_PRI, relief=tk.FLAT, bd=0,
                        highlightbackground="#1E4B68", highlightthickness=1,
                        padx=_s(14), pady=_s(4), cursor="hand2")
    add_btn.pack(side=tk.RIGHT, padx=(0, _s(8)))

    dnd_hint_lbl = tk.Label(tb_right, text=_etr("dnd_hint"), font=(FONT, 8),
                            fg=TEXT_MUTED, bg=BG_PANEL_ALT)
    dnd_hint_lbl.pack(side=tk.RIGHT, padx=(0, _s(14)))

    def _on_topbar_resize(event):
        if event.width < _s(760):
            if dnd_hint_lbl.winfo_manager():
                dnd_hint_lbl.pack_forget()
        else:
            if not dnd_hint_lbl.winfo_manager():
                dnd_hint_lbl.pack(side=tk.RIGHT, padx=(0, _s(14)))

    topbar.bind("<Configure>", _on_topbar_resize)

    tb_left = tk.Frame(topbar, bg=BG_PANEL_ALT)
    tb_left.pack(side=tk.LEFT, fill=tk.Y)
    queue_title_lbl = tk.Label(tb_left, text=tr("queue_title"), font=(FONT, 11, "bold"),
                               fg=TEXT_PRI, bg=BG_PANEL_ALT)
    queue_title_lbl.pack(side=tk.LEFT, pady=_s(12))

    kpi_frame = tk.Frame(tb_left, bg=BG_PANEL_ALT)
    kpi_frame.pack(side=tk.LEFT, padx=(_s(14), 0), pady=_s(12))

    def _make_kpi_pill(color_fg, color_bg):
        box = tk.Frame(kpi_frame, bg=color_bg, highlightbackground=BORDER,
                       highlightthickness=1, padx=_s(8), pady=_s(2))
        box.pack(side=tk.LEFT, padx=(0, _s(6)))
        k_lbl = tk.Label(box, text="", font=(FONT, 8), fg=TEXT_MUTED, bg=color_bg)
        k_lbl.pack(side=tk.LEFT, padx=(0, _s(5)))
        v_lbl = tk.Label(box, text="00", font=(MONO, 9, "bold"), fg=color_fg, bg=color_bg)
        v_lbl.pack(side=tk.LEFT)
        return k_lbl, v_lbl

    kpi_total_k, kpi_total_v = _make_kpi_pill(TEXT_PRI, BG_APP)
    kpi_ready_k, kpi_ready_v = _make_kpi_pill(ACCENT_HOVER, ACCENT_DIM)
    kpi_done_k, kpi_done_v = _make_kpi_pill(EMERALD, EMERALD_DIM)

    tk.Frame(stage, bg=BORDER, height=1).pack(fill=tk.X)

    # 2.2 未配对素材暂存坞（仅在有落单文件时显示）
    dock_frame = tk.Frame(stage, bg="#151311", highlightbackground="#423117", highlightthickness=1)
    dock_accent = tk.Frame(dock_frame, bg=AMBER, width=_s(3))
    dock_accent.pack(side=tk.LEFT, fill=tk.Y)
    dock_content = tk.Frame(dock_frame, bg="#151311", padx=_s(14), pady=_s(8))
    dock_content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    dock_top = tk.Frame(dock_content, bg="#151311")
    dock_top.pack(fill=tk.X, pady=(0, _s(6)))
    dock_title_lbl = tk.Label(dock_top, text=tr("unpaired_tray_title"),
                              font=(FONT, 8, "bold"), fg=AMBER, bg="#151311")
    dock_title_lbl.pack(side=tk.LEFT)
    dock_hint_lbl = tk.Label(dock_top, text=tr("unpaired_tray_hint"),
                             font=(FONT, 8), fg=TEXT_MUTED, bg="#151311")
    dock_hint_lbl.pack(side=tk.LEFT, padx=_s(10))
    dock_pair_btn = tk.Button(dock_top, text=_etr("auto_pair"), font=(FONT, 8, "bold"),
                              fg=AMBER, bg=AMBER_DIM, activebackground="#3C2D14",
                              activeforeground=AMBER_HOVER, relief=tk.FLAT, bd=0,
                              padx=_s(10), pady=_s(2), cursor="hand2")
    dock_pair_btn.pack(side=tk.RIGHT)

    dock_items = tk.Frame(dock_content, bg="#151311")
    dock_items.pack(fill=tk.X)

    # 2.3 中央轨道矩阵主容器（列头 + 滚动视口）
    matrix_outer = tk.Frame(stage, bg=BG_STAGE)
    matrix_outer.pack(fill=tk.BOTH, expand=True)

    col_header = tk.Frame(matrix_outer, bg=BG_PANEL_ALT, height=_s(30))
    col_header.pack_propagate(False)
    # 左侧对齐轨道状态灯条，右侧对齐滚动条槽，中间与每行 grid_body 完全同构
    tk.Frame(col_header, bg=BG_PANEL_ALT, width=_s(3)).pack(side=tk.LEFT, fill=tk.Y)
    tk.Frame(col_header, bg=BG_PANEL_ALT, width=_s(8)).pack(side=tk.RIGHT, fill=tk.Y)
    col_header_grid = tk.Frame(col_header, bg=BG_PANEL_ALT, padx=_s(14))
    col_header_grid.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    col_header_sep = tk.Frame(matrix_outer, bg=BORDER, height=1)

    matrix_scroll_wrap = tk.Frame(matrix_outer, bg=BG_STAGE)
    matrix_scroll_wrap.pack(fill=tk.BOTH, expand=True)

    matrix_canvas = tk.Canvas(matrix_scroll_wrap, bg=BG_STAGE, bd=0, highlightthickness=0)
    matrix_sb = ttk.Scrollbar(matrix_scroll_wrap, orient=tk.VERTICAL,
                              command=matrix_canvas.yview, style="Studio.Vertical.TScrollbar")
    matrix_canvas.configure(yscrollcommand=matrix_sb.set)
    matrix_sb.pack(side=tk.RIGHT, fill=tk.Y)
    matrix_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    matrix_inner = tk.Frame(matrix_canvas, bg=BG_STAGE)
    matrix_win_id = matrix_canvas.create_window((0, 0), window=matrix_inner, anchor="nw")

    def _on_matrix_canvas_resize(event):
        matrix_canvas.itemconfig(matrix_win_id, width=event.width)
        if not _task_store:
            matrix_canvas.itemconfig(matrix_win_id, height=max(event.height, _s(260)))

    matrix_canvas.bind("<Configure>", _on_matrix_canvas_resize)
    matrix_inner.bind("<Configure>", lambda _e: matrix_canvas.configure(scrollregion=matrix_canvas.bbox("all")))

    # 智能作用域滚轮：指针进入轨道矩阵区域时接管滚动，离开时释放
    def _on_matrix_wheel(event):
        if _task_store:
            matrix_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    matrix_scroll_wrap.bind("<Enter>", lambda _e: root.bind_all("<MouseWheel>", _on_matrix_wheel))
    matrix_scroll_wrap.bind("<Leave>", lambda _e: root.unbind_all("<MouseWheel>"))

    # ── 2.4 底部可折叠终端日志抽屉 ─────────────────────────────────────
    console_drawer = tk.Frame(stage, bg=BG_CONSOLE, height=_s(170))
    console_drawer.pack_propagate(False)
    tk.Frame(console_drawer, bg=BORDER, height=1).pack(fill=tk.X)
    console_bar = tk.Frame(console_drawer, bg="#0D1016", padx=_s(16), pady=_s(4))
    console_bar.pack(fill=tk.X)
    console_hdr_lbl = tk.Label(console_bar, text=_etr("console_header"),
                               font=(FONT, 8, "bold"), fg=TEXT_MUTED, bg="#0D1016")
    console_hdr_lbl.pack(side=tk.LEFT)
    log_clear_btn = tk.Button(console_bar, text=tr("btn_clear_log"), font=(FONT, 8),
                              fg=TEXT_MUTED, bg="#161C28", activebackground=BG_SLOT_HOVER,
                              activeforeground=TEXT_PRI, relief=tk.FLAT, bd=0,
                              padx=_s(8), pady=_s(2), cursor="hand2")
    log_clear_btn.pack(side=tk.RIGHT)

    log_body = tk.Frame(console_drawer, bg=BG_CONSOLE)
    log_body.pack(fill=tk.BOTH, expand=True)
    log_text = tk.Text(log_body, bg=BG_CONSOLE, fg="#94A3B8", font=(FONT, 9),
                       relief=tk.FLAT, wrap=tk.WORD, padx=_s(16), pady=_s(8),
                       spacing1=_s(1), spacing3=_s(1),
                       insertbackground=ACCENT_HOVER, selectbackground=ACCENT_DIM)
    log_scroll = ttk.Scrollbar(log_body, orient=tk.VERTICAL, command=log_text.yview,
                               style="Studio.Vertical.TScrollbar")
    log_text.configure(yscrollcommand=log_scroll.set)
    log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    log_text.tag_configure("phase", foreground=ACCENT_HOVER)
    log_text.tag_configure("ok", foreground=EMERALD)
    log_text.tag_configure("warn", foreground=AMBER)
    log_text.tag_configure("err", foreground=CRIMSON)

    log_visible = tk.BooleanVar(value=False)

    def _toggle_console():
        if log_visible.get():
            console_drawer.pack_forget()
            log_visible.set(False)
        else:
            console_drawer.pack(fill=tk.X, before=transport_sep)
            log_visible.set(True)
        _refresh_console_btn()

    def _refresh_console_btn():
        active = log_visible.get()
        sym = "▾" if active else "▸"
        console_toggle_btn.configure(
            text=f">_  {_etr('console_btn')} {sym}",
            bg=ACCENT_DIM if active else BG_SLOT,
            fg=ACCENT_HOVER if active else TEXT_SEC,
        )

    log_clear_btn.configure(command=lambda: log_text.delete("1.0", tk.END))

    def append_log(msg: str):
        def _insert():
            line = msg + "\n"
            tag = None
            if any(k in msg for k in ("失败", "中断", "💔", "ERROR", "Error")):
                tag = "err"
            elif any(k in msg for k in ("⚠", "跳过", "警告")):
                tag = "warn"
            elif any(k in msg for k in ("完成", "✨", "💖", "🎯", "✓")):
                tag = "ok"
            elif msg.lstrip().startswith("["):
                tag = "phase"
            if tag:
                log_text.insert(tk.END, line, tag)
            else:
                log_text.insert(tk.END, line)
            log_text.see(tk.END)
        _post_ui(_insert)

    # ── 2.5 底部走带控制栏 (Studio Transport Deck) ───────────────────────
    transport_sep = tk.Frame(stage, bg=BORDER, height=1)
    transport_sep.pack(fill=tk.X)

    transport = tk.Frame(stage, bg=BG_RAIL, height=_s(74), padx=_s(20), pady=_s(12))
    transport.pack(fill=tk.X)
    transport.pack_propagate(False)

    # 右侧主走带按钮
    start_btn = tk.Button(transport, text=tr("btn_start"), font=(FONT, 10, "bold"),
                          fg=INK_DARK, bg=ACCENT, activebackground=ACCENT_HOVER,
                          activeforeground=INK_DARK, disabledforeground=TEXT_MUTED,
                          relief=tk.FLAT, bd=0, padx=_s(26), pady=_s(10), cursor="hand2")
    start_btn.pack(side=tk.RIGHT, padx=(_s(18), 0))

    # 左侧状态与进度条区域
    trans_left = tk.Frame(transport, bg=BG_RAIL)
    trans_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    trans_top = tk.Frame(trans_left, bg=BG_RAIL)
    trans_top.pack(fill=tk.X, pady=(0, _s(6)))

    led_sz = _s(10)
    led_canvas = tk.Canvas(trans_top, width=led_sz, height=led_sz, bg=BG_RAIL, bd=0, highlightthickness=0)
    led_canvas.pack(side=tk.LEFT, padx=(0, _s(8)), pady=_s(3))
    led_dot_id = led_canvas.create_oval(_s(1), _s(1), led_sz - _s(1), led_sz - _s(1),
                                        fill=TEXT_MUTED, outline="")

    status_var = tk.StringVar(value=tr("status_idle"))
    status_lbl = tk.Label(trans_top, textvariable=status_var, font=(FONT, 9),
                          fg=TEXT_SEC, bg=BG_RAIL, anchor="w")
    status_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)

    progress_pct_var = tk.StringVar(value="0%")
    pct_lbl = tk.Label(trans_top, textvariable=progress_pct_var, font=(MONO, 10, "bold"),
                       fg=ACCENT_HOVER, bg=BG_RAIL)
    pct_lbl.pack(side=tk.RIGHT, padx=(_s(10), 0))

    console_toggle_btn = tk.Button(trans_top, text="", font=(MONO, 8, "bold"),
                                   fg=TEXT_SEC, bg=BG_SLOT, activebackground=BG_SLOT_HOVER,
                                   activeforeground=TEXT_PRI, relief=tk.FLAT, bd=0,
                                   padx=_s(10), pady=_s(2), cursor="hand2", command=_toggle_console)
    console_toggle_btn.pack(side=tk.RIGHT)
    _refresh_console_btn()

    progress_gauge = TransportProgressCanvas(trans_left)
    progress_gauge.pack(fill=tk.X)

    def _set_led(color: str):
        led_canvas.itemconfigure(led_dot_id, fill=color)

    def update_status(msg: str):
        _post_ui(lambda: status_var.set(msg))

    # ══════════════════════════════════════════════════════════════════════
    #  核心数据模型与内部拖拽状态机
    # ══════════════════════════════════════════════════════════════════════
    _task_store: dict[str, dict] = {}
    _unpaired_pool: dict[str, dict] = {}
    _card_id_counter = 0
    _uid_counter = 0
    _slot_registry: list[dict] = []
    _highlighted_slot: dict | None = None

    _drag_info = {
        "active": False,
        "type": None,   # "script" | "audio" | "audio_b"
        "file": None,
        "name": None,
        "cid": None,
        "uid": None,
        "win": None,
        "start_x": 0,
        "start_y": 0,
    }

    def _update_queue_meta():
        total = len(_task_store)
        done = sum(1 for t in _task_store.values() if t.get("status") == "done")
        ready = sum(1 for t in _task_store.values()
                    if t.get("script") and t.get("audio") and t.get("status") != "done")
        kpi_total_v.configure(text=f"{total:02d}")
        kpi_ready_v.configure(text=f"{ready:02d}")
        kpi_done_v.configure(text=f"{done:02d}")

        if not getattr(root, "_fa_running", False):
            start_btn.configure(
                text=f"▸  {tr('btn_start_with_count', count=ready)}" if ready else f"▸  {tr('btn_start')}",
                bg=ACCENT if ready else "#1B2A3A",
                fg=INK_DARK if ready else TEXT_MUTED,
                activebackground=ACCENT_HOVER if ready else "#22364A",
                activeforeground=INK_DARK if ready else TEXT_SEC,
            )

    def _is_persp_visible() -> bool:
        return mode_var.get() == "multi_persp" or any(t.get("audio_b") for t in _task_store.values())

    def _configure_matrix_grid(grid_frame: tk.Frame, show_b: bool):
        grid_frame.grid_columnconfigure(0, weight=0, minsize=_s(34), uniform="c_idx")
        grid_frame.grid_columnconfigure(1, weight=3, minsize=_s(120), uniform="c_slot")
        grid_frame.grid_columnconfigure(2, weight=3, minsize=_s(120), uniform="c_slot")
        grid_frame.grid_columnconfigure(3, weight=3, minsize=_s(120), uniform="c_slot")
        if show_b:
            grid_frame.grid_columnconfigure(4, weight=0, minsize=_s(34), uniform="c_swap")
            grid_frame.grid_columnconfigure(5, weight=3, minsize=_s(120), uniform="c_slot")
            grid_frame.grid_columnconfigure(6, weight=0, minsize=_s(82), uniform="c_act")
        else:
            grid_frame.grid_columnconfigure(4, weight=0, minsize=0, uniform="")
            grid_frame.grid_columnconfigure(5, weight=0, minsize=0, uniform="")
            grid_frame.grid_columnconfigure(6, weight=0, minsize=_s(82), uniform="c_act")
        grid_frame.grid_rowconfigure(0, weight=1)

    def _render_column_headers(show_b: bool):
        for w in col_header_grid.winfo_children():
            w.destroy()
        _configure_matrix_grid(col_header_grid, show_b)

        def _hdr(c, txt, anchor="w", mono=False):
            tk.Label(col_header_grid, text=txt, font=(MONO if mono else FONT, 8, "bold"),
                     fg=TEXT_MUTED, bg=BG_PANEL_ALT, anchor=anchor).grid(
                row=0, column=c, sticky="ew", padx=_s(4))

        _hdr(0, "#", mono=True)
        _hdr(1, tr("col_name"))
        _hdr(2, tr("col_script"))
        _hdr(3, tr("col_persp_a") if show_b else tr("col_audio"))
        if show_b:
            _hdr(4, "⇄", "center", mono=True)
            _hdr(5, tr("col_persp_b"))
        _hdr(6, _etr("col_act"), "e")

    def _render_empty_stage():
        """无任务时绘制专业音频视窗风格的矢量空状态导入区。"""
        col_header.pack_forget()
        col_header_sep.pack_forget()

        empty_cv = tk.Canvas(matrix_inner, bg=BG_STAGE, bd=0, highlightthickness=0, cursor="hand2")
        empty_cv.pack(fill=tk.BOTH, expand=True, padx=_s(24), pady=_s(24))

        hover_state = {"on": False}

        def _draw_empty(_event=None):
            empty_cv.delete("all")
            w = max(empty_cv.winfo_width(), _s(500))
            h = max(empty_cv.winfo_height(), _s(300))
            pad = _s(12)
            hov = hover_state["on"]
            frame_bg = "#111723" if hov else "#0F131C"
            dash_col = ACCENT_HOVER if hov else "#29354B"

            # 背景矩形与技术虚线框
            empty_cv.create_rectangle(pad, pad, w - pad, h - pad, fill=frame_bg, outline="")
            empty_cv.create_rectangle(pad, pad, w - pad, h - pad, outline=dash_col,
                                      width=max(1.0, _sf(1.5)), dash=(_s(6), _s(5)))

            # 四角工业校准角标
            corner = _s(14)
            cw = max(2, _s(2))
            for cx, cy, dx, dy in (
                (pad, pad, corner, corner),
                (w - pad, pad, -corner, corner),
                (pad, h - pad, corner, -corner),
                (w - pad, h - pad, -corner, -corner),
            ):
                empty_cv.create_line(cx, cy, cx + dx, cy, fill=ACCENT_HOVER if hov else BORDER_STRONG, width=cw)
                empty_cv.create_line(cx, cy, cx, cy + dy, fill=ACCENT_HOVER if hov else BORDER_STRONG, width=cw)

            mx, my = w // 2, h // 2 - _s(14)
            # 矢量声波图标（零 Emoji，按 DPI 等比放大）
            bar_heights = [10, 22, 36, 18, 44, 26, 14]
            pitch = _s(9)
            bar_w = _s(4)
            start_x = mx - (len(bar_heights) * pitch) // 2
            wave_cy = my - _s(40)
            for idx, bh in enumerate(bar_heights):
                bx = start_x + idx * pitch
                sbh = _s(bh)
                col = ACCENT_HOVER if (hov or idx in (2, 4)) else "#334460"
                empty_cv.create_rectangle(bx, wave_cy - sbh // 2, bx + bar_w, wave_cy + sbh // 2,
                                          fill=col, outline="")

            prompt = tr("drop_prompt_persp") if mode_var.get() == "multi_persp" else tr("drop_prompt_normal")
            empty_cv.create_text(mx, my + _s(8), text=prompt, fill=TEXT_PRI, font=(FONT, 12, "bold"))
            empty_cv.create_text(mx, my + _s(34), text=_etr("empty_sub"), fill=TEXT_MUTED, font=(FONT, 9))

            # 中心按钮胶囊
            bw, bh_btn = _s(148), _s(34)
            by0 = my + _s(58)
            btn_bg = ACCENT if hov else ACCENT_DIM
            btn_fg = INK_DARK if hov else ACCENT_HOVER
            empty_cv.create_rectangle(mx - bw // 2, by0, mx + bw // 2, by0 + bh_btn,
                                      fill=btn_bg, outline=ACCENT_HOVER, width=1)
            empty_cv.create_text(mx, by0 + bh_btn // 2, text=_etr("import_files"),
                                 fill=btn_fg, font=(FONT, 9, "bold"))

        empty_cv.bind("<Configure>", _draw_empty)
        empty_cv.bind("<Enter>", lambda _e: (hover_state.update(on=True), _draw_empty()))
        empty_cv.bind("<Leave>", lambda _e: (hover_state.update(on=False), _draw_empty()))
        empty_cv.bind("<Button-1>", _browse_files)

        if ctx.has_dnd:
            empty_cv.drop_target_register(ctx.dnd_files)
            empty_cv.dnd_bind("<<Drop>>", _on_external_drop)

        _slot_registry.append({
            "widget": empty_cv, "type": "new_card", "cid": None, "uid": None,
            "rest_border": BORDER, "rest_thick": 0, "in_matrix": True,
        })
        matrix_canvas.update_idletasks()
        matrix_canvas.itemconfig(matrix_win_id, height=max(matrix_canvas.winfo_height(), _s(280)))

    def _bind_drop_targets(*widgets):
        if not ctx.has_dnd:
            return
        for w in widgets:
            try:
                w.drop_target_register(ctx.dnd_files)
                w.dnd_bind("<<Drop>>", _on_external_drop)
            except Exception:
                pass

    def _render_all_cards():
        """渲染未配对素材坞与中央多轨编排矩阵。"""
        nonlocal _highlighted_slot
        _slot_registry.clear()
        _highlighted_slot = None

        # 1. 未配对暂存坞
        for w in dock_items.winfo_children():
            w.destroy()

        if _unpaired_pool:
            dock_frame.pack(fill=tk.X, padx=_s(16), pady=(_s(10), 0), before=matrix_outer)
            _slot_registry.append({
                "widget": dock_frame, "type": "tray", "cid": None, "uid": None,
                "rest_border": "#423117", "rest_thick": 1, "in_matrix": False,
            })
            # 简易自动换行行容器（每行最多放 4 枚芯片）
            row_f = None
            for idx, (uid, item) in enumerate(list(_unpaired_pool.items())):
                if idx % 4 == 0:
                    row_f = tk.Frame(dock_items, bg="#151311")
                    row_f.pack(fill=tk.X, pady=(0, _s(4)))
                    _bind_drop_targets(row_f)
                p_file = item["file"]
                p_type = item["type"]
                p_name = os.path.basename(p_file)
                tag_txt = "TXT" if p_type == "script" else "AUD"
                tag_fg = ACCENT_HOVER if p_type == "script" else AMBER
                tag_bg = ACCENT_DIM if p_type == "script" else AMBER_DIM

                pill = tk.Frame(row_f, bg=BG_SLOT, highlightbackground=BORDER,
                                highlightthickness=1, cursor="hand2", padx=_s(6), pady=_s(4))
                pill.pack(side=tk.LEFT, padx=(0, _s(6)))
                t_tag = tk.Label(pill, text=tag_txt, font=(MONO, 7, "bold"),
                                 fg=tag_fg, bg=tag_bg, padx=_s(4), pady=_s(1))
                t_tag.pack(side=tk.LEFT, padx=(0, _s(6)))
                t_lbl = tk.Label(pill, text=_truncate_filename(p_name, 22),
                                 font=(FONT, 8), fg=TEXT_PRI, bg=BG_SLOT)
                t_lbl.pack(side=tk.LEFT)
                ToolTip(pill, p_file)

                def _start_dock_drag(e, u=uid, p=p_file, tp=p_type, n=p_name):
                    _on_drag_press(e, p_type=tp, p_file=p, p_name=n, cid=None, uid=u)

                for w in (pill, t_tag, t_lbl):
                    w.bind("<ButtonPress-1>", _start_dock_drag)
                    w.bind("<B1-Motion>", _on_drag_motion)
                    w.bind("<ButtonRelease-1>", _on_drag_release)
                _bind_drop_targets(pill, t_tag, t_lbl)
        else:
            dock_frame.pack_forget()

        # 2. 清空轨道视口
        for w in matrix_inner.winfo_children():
            w.destroy()

        if not _task_store:
            _render_empty_stage()
            _update_queue_meta()
            return

        # 恢复自适应高度（空状态曾显式设满视口高度）
        matrix_canvas.itemconfig(matrix_win_id, height=0)

        show_b = _is_persp_visible()
        _render_column_headers(show_b)
        col_header.pack(fill=tk.X, before=matrix_scroll_wrap)
        col_header_sep.pack(fill=tk.X, before=matrix_scroll_wrap)

        def _make_slot_chip(parent, cid, slot_type, file_path, badge_code, empty_text, accent_fg, accent_bg, picker_fn):
            has_file = bool(file_path)
            bg_c = BG_SLOT if has_file else BG_SLOT_EMPTY
            bd_c = BORDER if has_file else "#253046"

            chip = tk.Frame(parent, bg=bg_c, highlightbackground=bd_c,
                            highlightthickness=1, cursor="hand2", padx=_s(8), pady=_s(5))
            _slot_registry.append({
                "widget": chip, "type": slot_type, "cid": cid, "uid": None,
                "rest_border": bd_c, "rest_thick": 1, "in_matrix": True,
            })

            if has_file:
                fname = os.path.basename(file_path)
                tag = tk.Label(chip, text=badge_code, font=(MONO, 7, "bold"),
                               fg=accent_fg, bg=accent_bg, padx=_s(4), pady=_s(1))
                tag.pack(side=tk.LEFT, padx=(0, _s(6)))
                lbl = tk.Label(chip, text=_truncate_filename(fname, 16 if show_b else 24),
                               font=(FONT, 8), fg=TEXT_PRI, bg=bg_c, anchor="w")
                lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
                ToolTip(chip, file_path)

                def _start(e, c=cid, p=file_path, n=fname, st=slot_type):
                    _on_drag_press(e, p_type=st, p_file=p, p_name=n, cid=c, uid=None)

                for w in (chip, tag, lbl):
                    w.bind("<ButtonPress-1>", _start)
                    w.bind("<B1-Motion>", _on_drag_motion)
                    w.bind("<ButtonRelease-1>", _on_drag_release)
                    w.bind("<Button-3>", lambda e, c=cid: _show_card_context_menu(e, c))
                _bind_drop_targets(chip, tag, lbl)
            else:
                ph = tk.Label(chip, text=empty_text, font=(FONT, 8),
                              fg=TEXT_MUTED, bg=bg_c, anchor="center")
                ph.pack(fill=tk.X)
                for w in (chip, ph):
                    w.bind("<Button-1>", lambda _e, c=cid: picker_fn(c))
                    w.bind("<Enter>", lambda _e, f=chip, p=ph: (f.configure(highlightbackground=accent_fg), p.configure(fg=accent_fg)))
                    w.bind("<Leave>", lambda _e, f=chip, p=ph, b=bd_c: (f.configure(highlightbackground=b), p.configure(fg=TEXT_MUTED)))
                _bind_drop_targets(chip, ph)
            return chip

        def _pick_script(cid):
            if ctx.is_running():
                return
            path = filedialog.askopenfilename(
                title=_etr("pick_script_title"),
                filetypes=[(tr("dlg_filter_supported"), "*.txt"), (tr("dlg_filter_all"), "*.*")])
            if path and cid in _task_store:
                _task_store[cid]["script"] = os.path.abspath(path)
                _render_all_cards()

        def _pick_audio(cid):
            if ctx.is_running():
                return
            path = filedialog.askopenfilename(
                title=_etr("pick_audio_title"),
                filetypes=[(tr("dlg_filter_audio"), "*.wav;*.mp3;*.flac;*.m4a;*.ogg;*.aac;*.opus"),
                           (tr("dlg_filter_all"), "*.*")])
            if path and cid in _task_store:
                _task_store[cid]["audio"] = os.path.abspath(path)
                _render_all_cards()

        def _pick_audio_b(cid):
            if ctx.is_running():
                return
            path = filedialog.askopenfilename(
                title=_etr("pick_audio_b_title"),
                filetypes=[(tr("dlg_filter_audio"), "*.wav;*.mp3;*.flac;*.m4a;*.ogg;*.aac;*.opus"),
                           (tr("dlg_filter_all"), "*.*")])
            if path and cid in _task_store:
                _task_store[cid]["audio_b"] = os.path.abspath(path)
                _render_all_cards()

        # 3. 逐条渲染水平轨道条（DAW Arrangement Strips）
        for row_idx, (cid, task) in enumerate(list(_task_store.items())):
            st = task.get("status", "pending")
            conf = task.get("confidence", 100)
            incomplete = not task.get("script") or not task.get("audio")

            if st == "running":
                led_col, badge_fg, badge_bg, badge_txt = ACCENT_HOVER, ACCENT_HOVER, ACCENT_DIM, tr("status_running")
            elif st == "done":
                led_col, badge_fg, badge_bg, badge_txt = EMERALD, EMERALD, EMERALD_DIM, tr("status_done")
            elif st == "failed":
                led_col, badge_fg, badge_bg, badge_txt = CRIMSON, CRIMSON, CRIMSON_DIM, tr("status_failed")
            elif incomplete:
                led_col, badge_fg, badge_bg, badge_txt = AMBER, AMBER, AMBER_DIM, tr("status_warn")
            elif conf and conf < 95:
                led_col, badge_fg, badge_bg, badge_txt = AMBER, AMBER, AMBER_DIM, tr("badge_fuzzy", score=conf)
            else:
                led_col, badge_fg, badge_bg, badge_txt = "#334155", TEXT_SEC, BG_SLOT, tr("status_pending")

            row_bg = BG_PANEL if row_idx % 2 == 0 else BG_PANEL_ALT
            strip_outer = tk.Frame(matrix_inner, bg=row_bg)
            strip_outer.pack(fill=tk.X)

            strip = tk.Frame(strip_outer, bg=row_bg, height=_s(58))
            strip.pack(fill=tk.X)
            strip.pack_propagate(False)

            # 左侧等比状态灯条
            tk.Frame(strip, bg=led_col, width=_s(3)).pack(side=tk.LEFT, fill=tk.Y)

            grid_body = tk.Frame(strip, bg=row_bg, padx=_s(14))
            grid_body.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            _configure_matrix_grid(grid_body, show_b)

            # 列 0：等宽轨号
            idx_lbl = tk.Label(grid_body, text=f"{row_idx + 1:02d}", font=(MONO, 9, "bold"),
                               fg=led_col if st == "running" else TEXT_MUTED, bg=row_bg, anchor="w")
            idx_lbl.grid(row=0, column=0, sticky="ew", padx=_s(4))

            # 列 1：轨道名 + 状态微胶囊
            meta_box = tk.Frame(grid_body, bg=row_bg)
            meta_box.grid(row=0, column=1, sticky="ew", padx=_s(4))
            t_name = task.get("name") or _etr("untitled_track")
            name_lbl = tk.Label(meta_box, text=_truncate_filename(t_name, 18 if show_b else 24),
                                font=(FONT, 9, "bold"), fg=TEXT_PRI, bg=row_bg, anchor="w")
            name_lbl.pack(anchor="w", fill=tk.X)
            ToolTip(name_lbl, t_name)
            st_pill = tk.Label(meta_box, text=badge_txt, font=(FONT, 7, "bold"),
                               fg=badge_fg, bg=badge_bg, padx=_s(6), pady=_s(1))
            st_pill.pack(anchor="w", pady=(_s(3), 0))

            # 列 2：台本槽位
            s_chip = _make_slot_chip(
                grid_body, cid, "script", task.get("script"), "TXT",
                tr("slot_script_empty"), ACCENT_HOVER, ACCENT_DIM, _pick_script,
            )
            s_chip.grid(row=0, column=2, sticky="ew", padx=_s(4))

            # 列 3：主音频槽位
            a_chip = _make_slot_chip(
                grid_body, cid, "audio", task.get("audio"), "A" if show_b else "AUD",
                tr("slot_audio_empty"), EMERALD if not show_b else ACCENT_HOVER,
                EMERALD_DIM if not show_b else ACCENT_DIM, _pick_audio,
            )
            a_chip.grid(row=0, column=3, sticky="ew", padx=_s(4))

            # 列 4 & 5：双视角交换按钮 + 副音频槽位
            if show_b:
                def _swap_ab(_e, c=cid):
                    if ctx.is_running():
                        return
                    t = _task_store.get(c)
                    if t:
                        t["audio"], t["audio_b"] = t.get("audio_b"), t.get("audio")
                        _render_all_cards()

                swap_btn = tk.Label(grid_body, text="⇄", font=(MONO, 10, "bold"),
                                    fg=AMBER, bg=AMBER_DIM, padx=_s(5), pady=_s(4), cursor="hand2")
                swap_btn.grid(row=0, column=4, padx=_s(2))
                swap_btn.bind("<Button-1>", _swap_ab)
                ToolTip(swap_btn, tr("swap_tooltip"))

                b_chip = _make_slot_chip(
                    grid_body, cid, "audio_b", task.get("audio_b"), "B",
                    tr("slot_audio_b_empty"), AMBER, AMBER_DIM, _pick_audio_b,
                )
                b_chip.grid(row=0, column=5, sticky="ew", padx=_s(4))

            # 列 6：内联快捷操作区（打开 / 重试 / 删除）
            act_box = tk.Frame(grid_body, bg=row_bg)
            act_box.grid(row=0, column=6, sticky="e", padx=(_s(4), 0))

            def _make_micro_btn(symbol, tip_key, hover_fg, cmd):
                b = tk.Label(act_box, text=symbol, font=(MONO, 10, "bold"),
                             fg=TEXT_MUTED, bg=row_bg, padx=_s(5), pady=_s(3), cursor="hand2")
                b.pack(side=tk.LEFT, padx=_s(1))
                b.bind("<Button-1>", lambda _e: cmd())
                b.bind("<Enter>", lambda _e: b.configure(fg=hover_fg, bg=BG_SLOT))
                b.bind("<Leave>", lambda _e: b.configure(fg=TEXT_MUTED, bg=row_bg))
                ToolTip(b, _etr(tip_key))
                return b

            def _remove_track(c=cid):
                if ctx.is_running():
                    return
                _task_store.pop(c, None)
                _render_all_cards()

            def _requeue_track(c=cid):
                if ctx.is_running():
                    return
                t = _task_store.get(c)
                if t:
                    t["status"] = "pending"
                    _render_all_cards()
                    update_status(tr("status_requeued", name=t["name"]))

            _make_micro_btn("↗", "tip_open", ACCENT_HOVER, lambda c=cid: _on_card_double_click(c))
            _make_micro_btn("↻", "tip_retry", AMBER, _requeue_track)
            _make_micro_btn("×", "tip_remove", CRIMSON, _remove_track)

            # 行级双击、右键菜单与外部拖放
            for w in (strip, grid_body, idx_lbl, meta_box, name_lbl, st_pill):
                w.bind("<Button-3>", lambda e, c=cid: _show_card_context_menu(e, c))
                w.bind("<Double-1>", lambda _e, c=cid: _on_card_double_click(c))
            _bind_drop_targets(strip_outer, strip, grid_body, idx_lbl, meta_box, name_lbl, st_pill, act_box)

            # 行底微弱分隔线
            tk.Frame(strip_outer, bg=BORDER_SUBTLE, height=1).pack(fill=tk.X)

        # 底部轻量新建轨道投放条
        new_bar = tk.Frame(matrix_inner, bg=BG_SLOT_EMPTY, highlightbackground=BORDER,
                           highlightthickness=1, padx=_s(14), pady=_s(10), cursor="hand2")
        new_bar.pack(fill=tk.X, padx=_s(16), pady=_s(12))
        new_lbl = tk.Label(new_bar, text=_etr("new_track_drop"), font=(FONT, 8),
                           fg=TEXT_MUTED, bg=BG_SLOT_EMPTY)
        new_lbl.pack()
        for w in (new_bar, new_lbl):
            w.bind("<Button-1>", _browse_files)
            w.bind("<Enter>", lambda _e: (new_bar.configure(highlightbackground=ACCENT), new_lbl.configure(fg=ACCENT_HOVER)))
            w.bind("<Leave>", lambda _e: (new_bar.configure(highlightbackground=BORDER), new_lbl.configure(fg=TEXT_MUTED)))
        _bind_drop_targets(new_bar, new_lbl)
        _slot_registry.append({
            "widget": new_bar, "type": "new_card", "cid": None, "uid": None,
            "rest_border": BORDER, "rest_thick": 1, "in_matrix": True,
        })

        _update_queue_meta()

    # ══════════════════════════════════════════════════════════════════════
    #  内部拖拽交互手势（支持槽位互换 / 拖入暂存坞 / 拖建新轨）
    # ══════════════════════════════════════════════════════════════════════

    def _on_drag_press(event, p_type, p_file, p_name, cid, uid):
        if ctx.is_running():
            return
        _drag_info.update(
            active=False, type=p_type, file=p_file, name=p_name,
            cid=cid, uid=uid, start_x=event.x_root, start_y=event.y_root,
        )

    def _on_drag_motion(event):
        if not _drag_info["file"]:
            return
        drag_thresh = _s(4)
        if not _drag_info["active"] and (
            abs(event.x_root - _drag_info["start_x"]) > drag_thresh
            or abs(event.y_root - _drag_info["start_y"]) > drag_thresh
        ):
            _drag_info["active"] = True
            win = tk.Toplevel(root)
            win.wm_overrideredirect(True)
            win.attributes("-topmost", True)
            try:
                win.attributes("-alpha", 0.92)
            except Exception:
                pass
            tag = "TXT" if _drag_info["type"] == "script" else "AUD"
            col = ACCENT_HOVER if tag == "TXT" else AMBER
            box = tk.Frame(win, bg="#0F172A", highlightbackground=col, highlightthickness=1,
                           padx=_s(10), pady=_s(5))
            box.pack()
            tk.Label(box, text=tag, font=(MONO, 7, "bold"), fg=INK_DARK, bg=col,
                     padx=_s(4), pady=_s(1)).pack(side=tk.LEFT, padx=(0, _s(6)))
            tk.Label(box, text=_truncate_filename(_drag_info["name"], 24),
                     font=(FONT, 8, "bold"), fg=TEXT_PRI, bg="#0F172A").pack(side=tk.LEFT)
            _drag_info["win"] = win

        if _drag_info["active"] and _drag_info["win"]:
            off = _s(14)
            _drag_info["win"].geometry(f"+{event.x_root + off}+{event.y_root + off}")
            _update_drag_hover(event.x_root, event.y_root)

    def _is_slot_compatible(src_type: str, tgt_type: str) -> bool:
        if tgt_type in ("tray", "new_card"):
            return True
        is_src_aud = src_type in ("audio", "audio_b")
        is_tgt_aud = tgt_type in ("audio", "audio_b")
        return (src_type == "script" and tgt_type == "script") or (is_src_aud and is_tgt_aud)

    def _restore_slot_border(slot: dict):
        try:
            slot["widget"].configure(
                highlightbackground=slot.get("rest_border", BORDER),
                highlightthickness=slot.get("rest_thick", 1),
            )
        except Exception:
            pass

    def _update_drag_hover(x: int, y: int):
        nonlocal _highlighted_slot
        matched = None
        # 计算轨道矩阵可见视口，避免已滚出顶/底部的槽位在坐标上误抢占暂存坞或顶栏命中
        cv_x0 = matrix_canvas.winfo_rootx()
        cv_y0 = matrix_canvas.winfo_rooty()
        cv_x1 = cv_x0 + matrix_canvas.winfo_width()
        cv_y1 = cv_y0 + matrix_canvas.winfo_height()
        in_matrix_view = (cv_x0 <= x <= cv_x1 and cv_y0 <= y <= cv_y1)

        for item in _slot_registry:
            if item.get("in_matrix") and not in_matrix_view:
                continue
            w = item["widget"]
            if not w.winfo_exists() or not w.winfo_ismapped():
                continue
            rx, ry = w.winfo_rootx(), w.winfo_rooty()
            if rx <= x <= rx + w.winfo_width() and ry <= y <= ry + w.winfo_height():
                matched = item
                break

        if matched != _highlighted_slot:
            if _highlighted_slot:
                _restore_slot_border(_highlighted_slot)
            _highlighted_slot = matched
            if _highlighted_slot:
                ok = _is_slot_compatible(_drag_info["type"], _highlighted_slot["type"])
                try:
                    _highlighted_slot["widget"].configure(
                        highlightbackground=ACCENT_HOVER if ok else CRIMSON,
                        highlightthickness=2,
                    )
                except Exception:
                    pass

    def _on_drag_release(_event):
        nonlocal _highlighted_slot, _uid_counter, _card_id_counter
        if _drag_info["win"]:
            try:
                _drag_info["win"].destroy()
            except Exception:
                pass
            _drag_info["win"] = None

        if _highlighted_slot:
            _restore_slot_border(_highlighted_slot)

        if not _drag_info["active"]:
            _drag_info["file"] = None
            return

        target = _highlighted_slot
        _highlighted_slot = None
        _drag_info["active"] = False
        src_type = _drag_info["type"]
        src_file = _drag_info["file"]
        src_cid = _drag_info["cid"]
        src_uid = _drag_info["uid"]
        _drag_info["file"] = None

        if not target:
            return

        tgt_type = target["type"]
        tgt_cid = target["cid"]

        # 1. 放回未配对暂存坞
        if tgt_type == "tray":
            if src_cid is not None:
                card = _task_store.get(src_cid)
                if card:
                    card[src_type] = None
                    _uid_counter += 1
                    u = f"u_{_uid_counter}"
                    _unpaired_pool[u] = {
                        "uid": u,
                        "type": "script" if src_type == "script" else "audio",
                        "file": src_file,
                        "name": os.path.basename(src_file),
                    }
                    if not card.get("script") and not card.get("audio") and not card.get("audio_b"):
                        _task_store.pop(src_cid, None)
            _render_all_cards()
            return

        # 2. 放入新建轨道区
        if tgt_type == "new_card":
            _card_id_counter += 1
            cid = f"card_{_card_id_counter}"
            is_src_aud = src_type in ("audio", "audio_b")
            raw_stem = os.path.splitext(os.path.basename(src_file))[0]
            clean_name = raw_stem
            if is_src_aud and hasattr(ctx, "_extract_perspective_tag"):
                base_k, _, _ = ctx._extract_perspective_tag(raw_stem)
                if base_k:
                    clean_name = base_k
            _task_store[cid] = {
                "cid": cid, "iid": cid, "name": clean_name,
                "script": None if is_src_aud else src_file,
                "audio": src_file if is_src_aud else None,
                "audio_b": None, "status": "pending", "confidence": 100,
            }
            if src_cid is not None:
                src_card = _task_store.get(src_cid)
                if src_card:
                    src_card[src_type] = None
                    if not src_card.get("script") and not src_card.get("audio") and not src_card.get("audio_b"):
                        _task_store.pop(src_cid, None)
            else:
                _unpaired_pool.pop(src_uid, None)
            _render_all_cards()
            return

        # 3. 放入轨道槽位（填入或互换）
        if not _is_slot_compatible(src_type, tgt_type):
            return
        tgt_card = _task_store.get(tgt_cid)
        if not tgt_card:
            return
        old_tgt = tgt_card.get(tgt_type)
        if not old_tgt:
            tgt_card[tgt_type] = src_file
            if src_cid is not None:
                src_card = _task_store.get(src_cid)
                if src_card:
                    src_card[src_type] = None
                    if not src_card.get("script") and not src_card.get("audio") and not src_card.get("audio_b"):
                        _task_store.pop(src_cid, None)
            else:
                _unpaired_pool.pop(src_uid, None)
        else:
            if src_cid is not None:
                src_card = _task_store.get(src_cid)
                if src_card:
                    src_card[src_type] = old_tgt
                    tgt_card[tgt_type] = src_file
            else:
                tgt_card[tgt_type] = src_file
                _unpaired_pool[src_uid]["file"] = old_tgt
                _unpaired_pool[src_uid]["name"] = os.path.basename(old_tgt)

        tgt_card["confidence"] = 100
        if src_cid and src_cid in _task_store:
            _task_store[src_cid]["confidence"] = 100
        _render_all_cards()

    # ══════════════════════════════════════════════════════════════════════
    #  文件导入、自动配对、菜单与外部 DND
    # ══════════════════════════════════════════════════════════════════════

    def _add_files_to_list(files):
        if not files:
            return
        nonlocal _card_id_counter, _uid_counter
        pool_files = [item["file"] for item in _unpaired_pool.values()]
        incomplete_files = []
        for t in _task_store.values():
            if not t.get("script") or not t.get("audio") or (mode_var.get() == "multi_persp" and not t.get("audio_b")):
                for k in ("script", "audio", "audio_b"):
                    if t.get(k):
                        incomplete_files.append(t[k])

        new_files = [os.path.abspath(f) for f in files if f and str(f).strip()]
        all_candidates = list(dict.fromkeys(pool_files + incomplete_files + new_files))
        pairs, unmatched = ctx.auto_pair_files(all_candidates, mode=mode_var.get())
        added = 0

        existing_auds = {p.get("audio") for p in _task_store.values() if p.get("audio")} | {
            p.get("audio_b") for p in _task_store.values() if p.get("audio_b")
        }
        existing_txts = {p.get("script") for p in _task_store.values() if p.get("script")}

        def _is_incomplete(card: dict) -> bool:
            if not card.get("script") or not card.get("audio"):
                return True
            if mode_var.get() == "multi_persp" and not card.get("audio_b"):
                return True
            return False

        # 若新配对覆盖了已有残缺轨道的台本或音频，原地补齐（并合并/清理冗余残缺项），杜绝幽灵重复轨道
        for pair in list(pairs):
            p_s = pair.get("script")
            p_a = pair.get("audio")
            p_b = pair.get("audio_b")
            hits = [
                cid for cid, t in list(_task_store.items())
                if _is_incomplete(t) and (
                    (p_s and t.get("script") == p_s)
                    or (p_a and t.get("audio") == p_a)
                    or (p_b and t.get("audio_b") == p_b)
                )
            ]
            if not hits:
                continue
            primary = _task_store[hits[0]]
            changed_card = False
            for slot_k, val in (("script", p_s), ("audio", p_a), ("audio_b", p_b)):
                if val and not primary.get(slot_k):
                    primary[slot_k] = val
                    changed_card = True
            if pair.get("name") and not primary.get("name"):
                primary["name"] = pair["name"]
            if changed_card:
                primary["confidence"] = pair.get("confidence", primary.get("confidence", 100))
                added += 1
            for dup_cid in hits[1:]:
                _task_store.pop(dup_cid, None)
            if p_s:
                existing_txts.add(p_s)
            if p_a:
                existing_auds.add(p_a)
            if p_b:
                existing_auds.add(p_b)
            pairs.remove(pair)

        for pair in pairs:
            if pair.get("audio") in existing_auds and pair.get("script") in existing_txts:
                continue
            _card_id_counter += 1
            cid = f"card_{_card_id_counter}"
            _task_store[cid] = {
                "cid": cid, "iid": cid, "name": pair["name"],
                "script": pair.get("script"), "audio": pair.get("audio"),
                "audio_b": pair.get("audio_b"), "status": "pending",
                "confidence": pair.get("confidence", 100),
            }
            for k, s_set in (("audio", existing_auds), ("audio_b", existing_auds), ("script", existing_txts)):
                if pair.get(k):
                    s_set.add(pair[k])
            added += 1

        _unpaired_pool.clear()
        for item in unmatched:
            f_path = item["file"]
            is_txt = item["type"] == "缺音频"
            if (is_txt and f_path in existing_txts) or (not is_txt and f_path in existing_auds):
                continue
            _uid_counter += 1
            uid = f"u_{_uid_counter}"
            _unpaired_pool[uid] = {
                "uid": uid, "type": "script" if is_txt else "audio",
                "file": f_path, "name": item["name"],
            }

        _render_all_cards()
        if added:
            _set_led(ACCENT_HOVER)
            update_status(tr("status_added_files", added=added, total=len(_task_store)))
        else:
            update_status(tr("status_files_already"))

    def _auto_assemble_from_tray():
        if ctx.is_running():
            return
        if _unpaired_pool:
            _add_files_to_list([item["file"] for item in list(_unpaired_pool.values())])

    def _browse_files(_event=None):
        if ctx.is_running():
            return
        files = filedialog.askopenfilenames(
            title=tr("dlg_select_files_title"),
            filetypes=[(tr("dlg_filter_supported"), "*.txt;*.wav;*.mp3;*.flac;*.m4a;*.ogg;*.aac;*.opus;*.wma"),
                       (tr("dlg_filter_all"), "*.*")])
        if files:
            _add_files_to_list(list(files))

    def _clear_tasks():
        if ctx.is_running():
            return
        _task_store.clear()
        _unpaired_pool.clear()
        _render_all_cards()
        progress_gauge.set_value(0)
        progress_pct_var.set("0%")
        _set_led(TEXT_MUTED)
        update_status(tr("status_queue_cleared"))

    add_btn.configure(command=_browse_files)
    clear_btn.configure(command=_clear_tasks)
    dock_pair_btn.configure(command=_auto_assemble_from_tray)

    def _on_external_drop(event):
        if ctx.is_running():
            return
        paths = []
        try:
            for item in root.tk.splitlist(event.data):
                path = str(item).strip().strip('"').strip("'")
                if path:
                    paths.append(os.path.abspath(path))
        except Exception:
            paths = [os.path.abspath(p) for p in event.data.split() if p.strip()]
        _add_files_to_list(paths)

    _bind_drop_targets(stage, matrix_outer, matrix_scroll_wrap, matrix_canvas,
                       matrix_inner, dock_frame, dock_content, dock_items)

    def _show_card_context_menu(event, cid: str):
        item = _task_store.get(cid)
        if not item:
            return
        running = ctx.is_running()
        mut_state = tk.DISABLED if running else tk.NORMAL
        menu = tk.Menu(root, tearoff=0, bg=BG_PANEL, fg=TEXT_PRI,
                       activebackground=ACCENT_DIM, activeforeground=ACCENT_HOVER,
                       disabledforeground=TEXT_MUTED,
                       bd=1, relief=tk.SOLID, font=(FONT, 9))

        def _open_file(path):
            if path and os.path.exists(path):
                os.startfile(os.path.normpath(path))

        def _open_folder():
            target = item.get("audio") or item.get("script")
            if target and os.path.exists(target):
                subprocess.run(["explorer", "/select,", os.path.normpath(target)])

        def _view_output(ext: str):
            audio = item.get("audio")
            if not audio:
                return
            out_path = os.path.splitext(audio)[0] + ext
            if os.path.exists(out_path):
                _open_file(out_path)
            else:
                messagebox.showinfo(tr("msg_no_subtitle_title"),
                                    tr("msg_no_subtitle_text", extension=ext), parent=root)

        def _retry_this():
            if ctx.is_running():
                return
            item["status"] = "pending"
            _render_all_cards()
            update_status(tr("status_requeued", name=item["name"]))

        def _remove_this():
            if ctx.is_running():
                return
            _task_store.pop(cid, None)
            _render_all_cards()

        menu.add_command(label=tr("menu_open_folder"), command=_open_folder)
        menu.add_command(label=tr("menu_open_script"), command=lambda: _open_file(item.get("script")))
        menu.add_command(label=tr("menu_view_lrc"), command=lambda: _view_output(".lrc"))
        menu.add_command(label=tr("menu_view_srt"), command=lambda: _view_output(".srt"))
        menu.add_separator()
        if _is_persp_visible():
            def _swap():
                if ctx.is_running():
                    return
                item["audio"], item["audio_b"] = item.get("audio_b"), item.get("audio")
                _render_all_cards()
            menu.add_command(label=tr("menu_swap_audios"), command=_swap, state=mut_state)
            menu.add_separator()
        menu.add_command(label=tr("menu_retry"), command=_retry_this, state=mut_state)
        menu.add_command(label=tr("btn_remove_card"), command=_remove_this, state=mut_state)
        menu.post(event.x_root, event.y_root)

    def _on_card_double_click(cid: str):
        item = _task_store.get(cid)
        if not item:
            return
        audio = item.get("audio")
        if audio:
            for ext in (".srt", ".lrc"):
                out_path = os.path.splitext(audio)[0] + ext
                if os.path.exists(out_path):
                    os.startfile(os.path.normpath(out_path))
                    return
        target = audio or item.get("script")
        if target and os.path.exists(target):
            subprocess.run(["explorer", "/select,", os.path.normpath(target)])

    # ══════════════════════════════════════════════════════════════════════
    #  模式切换与多语言全量热更新
    # ══════════════════════════════════════════════════════════════════════

    def _on_mode_change():
        cur = mode_var.get()
        _paint_mode_tiles()
        mode_desc_var.set(tr(f"mode_{cur}_desc"))
        _render_all_cards()
        _save_settings()

    def _refresh_all_i18n():
        root.title(tr("app_title"))
        scope_header.redraw()
        sec_engine_lbl.configure(text=_etr("rail_section_engine"))
        sec_params_lbl.configure(text=_etr("rail_section_params"))
        sec_sys_lbl.configure(text=_etr("rail_section_sys"))

        for k, refs in mode_tiles.items():
            refs["title"].configure(text=tr(f"mode_{k}_title"))
            refs["sub"].configure(text=tr(f"mode_{k}_sub"))
        mode_desc_var.set(tr(f"mode_{mode_var.get()}_desc"))

        ext_title_lbl.configure(text=tr("label_end_extend"))
        ext_unit_lbl.configure(text=tr("label_sec"))
        ono_lbl.configure(text=tr("check_ono_gate"))
        dev_lbl.configure(text=tr("label_device"))
        help_btn.configure(text=f"?  {tr('btn_show_guide')}")

        queue_title_lbl.configure(text=tr("queue_title"))
        kpi_total_k.configure(text=_etr("kpi_total"))
        kpi_ready_k.configure(text=_etr("kpi_ready"))
        kpi_done_k.configure(text=_etr("kpi_done"))
        add_btn.configure(text=_etr("import_files"))
        clear_btn.configure(text=tr("btn_clear"))
        dnd_hint_lbl.configure(text=_etr("dnd_hint"))

        dock_title_lbl.configure(text=tr("unpaired_tray_title"))
        dock_hint_lbl.configure(text=tr("unpaired_tray_hint"))
        dock_pair_btn.configure(text=_etr("auto_pair"))

        console_hdr_lbl.configure(text=_etr("console_header"))
        log_clear_btn.configure(text=tr("btn_clear_log"))
        _refresh_console_btn()

        _render_all_cards()
        if not getattr(root, "_fa_running", False) and not _task_store:
            status_var.set(tr("status_idle"))

    # ══════════════════════════════════════════════════════════════════════
    #  对齐执行引擎回调
    # ══════════════════════════════════════════════════════════════════════

    def _set_running(running: bool):
        root._fa_running = running
        scope_header.set_running(running)
        progress_gauge.set_running(running)
        ono_switch.set_disabled(running)
        dev_seg.set_disabled(running)
        state = tk.DISABLED if running else tk.NORMAL
        add_btn.configure(state=state)
        clear_btn.configure(state=state)
        dock_pair_btn.configure(state=state)
        btn_ext_minus.configure(state=state)
        btn_ext_plus.configure(state=state)
        ext_entry.configure(state=state)
        if running:
            _set_led(ACCENT_HOVER)
            start_btn.configure(state=tk.DISABLED, text=f"●  {tr('btn_running')}",
                                bg=ACCENT_DIM, fg=ACCENT_HOVER)
        else:
            start_btn.configure(state=tk.NORMAL)
            _update_queue_meta()

    def _set_task_status(iid: str, status: str):
        def _apply():
            if iid in _task_store:
                _task_store[iid]["status"] = status
                _render_all_cards()
        _post_ui(_apply)

    def _set_progress(value: float, message: str):
        def _apply():
            progress_gauge.set_value(value)
            progress_pct_var.set(f"{value:.0f}%")
            status_var.set(message)
        _post_ui(_apply)

    def _run_all():
        if ctx.is_running():
            return
        _save_settings()
        try:
            end_extend = float(end_extend_var.get() or 0)
            if end_extend < 0:
                raise ValueError
        except ValueError:
            messagebox.showerror(tr("msg_param_error_title"), tr("msg_param_error_text"), parent=root)
            return

        current = mode_var.get()
        valid = [t for t in _task_store.values()
                 if t.get("audio") and t.get("script") and t.get("status") != "done"]
        if not valid:
            messagebox.showinfo(tr("msg_empty_queue_title"), tr("msg_empty_queue_text"), parent=root)
            return

        if current == "multi_persp":
            missing = [t["name"] for t in valid if not t.get("audio_b")]
            if missing:
                preview = "\n".join(missing[:8]) + ("\n…" if len(missing) > 8 else "")
                if not messagebox.askyesno(
                    tr("msg_missing_persp_b_title"),
                    tr("msg_missing_persp_b_text", names=preview), parent=root,
                ):
                    return

        _set_running(True)
        _set_progress(0, tr("status_preparing", count=len(valid)))
        append_log(f"\n[RUN] {tr(f'mode_{current}_title')} · {dev_var.get().upper()} · {len(valid)} track(s)")

        def _worker():
            succeeded, failed = 0, 0
            total = len(valid)
            try:
                for idx, item in enumerate(valid):
                    iid = item["iid"]
                    _set_task_status(iid, "running")
                    append_log(f"[{idx + 1}/{total}] {item['name']}")

                    def _pct(percent, message):
                        overall = (idx * 100 + percent) / total
                        _set_progress(overall, tr("status_progress_step",
                                                  current=idx + 1, total=total, message=message))

                    ok, _output, error = ctx.run_pipeline(
                        script_path=item["script"], audio_path=item["audio"],
                        lora_path=ctx.lora_path, log_callback=append_log, progress_callback=_pct,
                        dev=dev_var.get(), vram_chunk=DEFAULT_VRAM_CHUNK, persist_gpu=True,
                        line_end_extend=end_extend, enable_silence_snap=True,
                        enable_resegment=True, enable_ono_gate=bool(ono_gate_var.get()),
                        enable_multi_order=current in ("multi_speaker", "multi_persp"),
                        audio_path_b=item.get("audio_b") if current == "multi_persp" else None,
                    )
                    _set_task_status(iid, "done" if ok else "failed")
                    if ok:
                        succeeded += 1
                    else:
                        failed += 1
                        append_log(f"处理失败：{error}")
                _post_ui(lambda: _set_led(EMERALD if failed == 0 else AMBER))
                _set_progress(100, tr("status_all_done", succeeded=succeeded, failed=failed))
                append_log(f"✓ 批量任务完成：成功 {succeeded}，失败 {failed}")
            except Exception as err:
                failed += 1
                _post_ui(lambda: _set_led(CRIMSON))
                append_log(f"任务中断：{type(err).__name__}: {err}")
                _set_progress(0, tr("status_interrupted", error=err))
            finally:
                try:
                    ctx.release_gpu()
                finally:
                    _post_ui(lambda: _set_running(False))

        ctx.start_worker(_worker)

    root._fa_set_running = _set_running
    start_btn.configure(command=_run_all)

    def _on_close():
        nonlocal closing
        closing = True
        scope_header.stop()
        progress_gauge.stop()
        _save_settings()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)

    # ── 初始化渲染与居中定位 ──────────────────────────────────────────
    _on_mode_change()
    _refresh_all_i18n()
    root.update_idletasks()
    _apply_dark_titlebar(root)
    w, h = root.winfo_width(), root.winfo_height()
    root.geometry(f"+{max(0, (root.winfo_screenwidth() - w) // 2)}"
                  f"+{max(0, (root.winfo_screenheight() - h) // 2)}")
    return root
