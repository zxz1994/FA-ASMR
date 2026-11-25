# -*- coding: utf-8 -*-
"""FA-ASMR Acoustic Mastering Console — 暗色主题色板、DPI 缩放与自定义控件。

从 fa_asmr_ui.py 抽离的可复用 UI 基础层：
- Obsidian Acoustic Console 色板、字体常量与补充 i18n 文案
- Windows Per-Monitor V2 High-DPI 感知与全局等比缩放 (_s / _sf)
- 暗色标题栏与高清示波器图标绘制
- 自定义控件：ToolTip / ToggleSwitch / SegmentedControl / OscilloscopeHeader / TransportProgressCanvas
- 台本格式规范指南弹窗 (show_script_help_dialog)
"""
from __future__ import annotations

import math
import os
import sys
import tkinter as tk
from tkinter import ttk

from fa_asmr.core.i18n import I18N, tr

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
