# -*- coding: utf-8 -*-
"""FA-ASMR Acoustic Mastering Console — 全新桌面前端界面主壳与控制器。

与强制对齐管线完全解耦，仅通过 build_ui(ctx) 接入业务回调。
架构分层：
- `fa_asmr/ui/widgets.py`：色板排版、High-DPI 等比缩放、自绘示波器/开关/进度条控件、规范指南弹窗
- `fa_asmr/ui/matrix.py`：TaskMatrixController（多轨编排矩阵、未配对暂存坞、内部/外部拖拽、文件配对与右键菜单）
- `fa_asmr_ui.py`：StudioConsoleUI 主控窗口壳（设置持久化、左侧参数导轨、终端日志抽屉、走带栏与批量调度）
"""
from __future__ import annotations

import json
import os
import queue
import sys
import tkinter as tk
from tkinter import messagebox, ttk

from fa_asmr.core.i18n import (
    I18N,
    LANG_MAP,
    detect_system_lang,
    tr,
)
from fa_asmr.ui.matrix import TaskMatrixController
from fa_asmr.ui.widgets import (  # noqa: F401
    ACCENT,
    ACCENT_DIM,
    ACCENT_GLOW,
    ACCENT_HOVER,
    AMBER,
    AMBER_DIM,
    AMBER_HOVER,
    BG_APP,
    BG_CONSOLE,
    BG_PANEL,
    BG_PANEL_ALT,
    BG_RAIL,
    BG_SLOT,
    BG_SLOT_EMPTY,
    BG_SLOT_HOVER,
    BG_STAGE,
    BORDER,
    BORDER_STRONG,
    BORDER_SUBTLE,
    CRIMSON,
    CRIMSON_DIM,
    DEFAULT_VRAM_CHUNK,
    EMERALD,
    EMERALD_DIM,
    FONT,
    INK_DARK,
    MONO,
    TEXT_MUTED,
    TEXT_PRI,
    TEXT_SEC,
    OscilloscopeHeader,
    SegmentedControl,
    ToggleSwitch,
    ToolTip,
    TransportProgressCanvas,
    _apply_dark_titlebar,
    _build_studio_icon,
    _detect_window_dpi,
    _etr,
    _init_dpi_scaling,
    _s,
    _sf,
    _truncate_filename,
    enable_windows_dpi_awareness,
    show_script_help_dialog,
)


class StudioConsoleUI:
    """Obsidian Acoustic Console 主窗口控制器。"""

    def __init__(self, ctx):
        enable_windows_dpi_awareness()
        self.ctx = ctx
        self.root = ctx.root
        _init_dpi_scaling(self.root)

        self.ui_events: queue.Queue = queue.Queue()
        self.closing = False
        self._startup_warnings: list[str] = []
        self._ui_log_sink = None

        # 状态变量与设置持久化（与旧版 JSON 结构 100% 兼容）
        self.lang_var = tk.StringVar(value=I18N.current_lang)
        self.mode_var = tk.StringVar(value="single")
        self.dev_var = tk.StringVar(value="cuda" if ctx.cuda_ok else "cpu")
        self.end_extend_var = tk.StringVar(value="1.2")
        self.ono_gate_var = tk.BooleanVar(value=True)
        self.multi_order_var = tk.BooleanVar(value=False)
        self.multi_persp_var = tk.BooleanVar(value=False)

        self._var_meta = [
            (self.lang_var, "lang", str),
            (self.mode_var, "align_mode", str),
            (self.dev_var, "dev", str),
            (self.end_extend_var, "end_extend", str),
            (self.ono_gate_var, "ono_gate", lambda v: v in (True, "true", "True", 1, "1")),
            (self.multi_order_var, "multi_order", lambda v: v in (True, "true", "True", 1, "1")),
            (self.multi_persp_var, "multi_persp", lambda v: v in (True, "true", "True", 1, "1")),
        ]

        self._mode_codes = {"single": "SEQ", "multi_speaker": "SPK", "multi_persp": "POV"}
        self._mode_accents = {"single": ACCENT_HOVER, "multi_speaker": ACCENT_HOVER, "multi_persp": AMBER}
        self._mode_dims = {"single": ACCENT_DIM, "multi_speaker": ACCENT_DIM, "multi_persp": AMBER_DIM}
        self.mode_tiles: dict[str, dict] = {}
        self.preset_btns: dict[str, tk.Label] = {}
        self.log_visible = tk.BooleanVar(value=False)

    # ── 线程安全 UI 事件泵与设置持久化 ─────────────────────────────────

    def _post_ui(self, callback) -> None:
        self.ui_events.put(callback)

    def _drain_ui_events(self) -> None:
        while True:
            try:
                cb = self.ui_events.get_nowait()
            except queue.Empty:
                break
            try:
                cb()
            except tk.TclError:
                if not self.closing:
                    raise
        if not self.closing:
            self.root.after(30, self._drain_ui_events)

    def _warn_ui(self, msg: str) -> None:
        try:
            print(f"[FA-ASMR UI] ⚠ {msg}", file=sys.stderr)
        except Exception:
            pass
        if self._ui_log_sink is not None:
            self._ui_log_sink(f"⚠ {msg}")
        else:
            self._startup_warnings.append(msg)

    def _save_settings(self) -> None:
        try:
            cur = self.mode_var.get()
            self.multi_order_var.set(cur in ("multi_speaker", "multi_persp"))
            self.multi_persp_var.set(cur == "multi_persp")
            with open(self.ctx.settings_file, "w", encoding="utf-8") as f:
                json.dump({k: v.get() for v, k, _ in self._var_meta}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self._warn_ui(f"保存设置文件失败 ({self.ctx.settings_file}): {type(e).__name__}: {e}")

    def _load_settings(self) -> None:
        if not os.path.isfile(self.ctx.settings_file):
            init_lang = detect_system_lang()
            I18N.set_lang(init_lang)
            self.lang_var.set(init_lang)
            return
        try:
            with open(self.ctx.settings_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError(f"设置文件根节点不是 JSON 对象: {type(data).__name__}")
            for var, key, conv in self._var_meta:
                if key in data:
                    var.set(conv(data[key]))
            if not self.ctx.cuda_ok:
                self.dev_var.set("cpu")
            cur_lang = data["lang"] if data.get("lang") in LANG_MAP else detect_system_lang()
            I18N.set_lang(cur_lang)
            self.lang_var.set(cur_lang)
            if "align_mode" not in data:
                self.mode_var.set(
                    "multi_persp" if data.get("multi_persp") else
                    "multi_speaker" if data.get("multi_order") else "single"
                )
        except Exception as e:
            self._warn_ui(f"读取设置文件失败 ({self.ctx.settings_file}): {type(e).__name__}: {e}，已回退默认设置")

    # ── 窗口与双栏布局构建 ──────────────────────────────────────────────

    def _build_shell(self) -> None:
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        init_w = min(_s(1240), int(sw * 0.92))
        init_h = min(_s(820), int(sh * 0.88))
        min_w = min(_s(980), int(sw * 0.85))
        min_h = min(_s(640), int(sh * 0.80))

        self.root.title(tr("app_title"))
        self.root.geometry(f"{init_w}x{init_h}")
        self.root.minsize(min_w, min_h)
        self.root.configure(bg=BG_APP)
        _build_studio_icon(self.root, bool(self.ctx.pil))

        # 全局 ttk 暗色滚动条样式（按 DPI 缩放宽度）
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(
            "Studio.Vertical.TScrollbar",
            background="#242E42", troughcolor=BG_STAGE,
            borderwidth=0, arrowsize=0, width=_s(8), relief=tk.FLAT,
        )
        style.map(
            "Studio.Vertical.TScrollbar",
            background=[("active", "#334155"), ("pressed", ACCENT)],
        )

        rail_w = _s(284)
        stage_min_w = max(_s(480), min_w - rail_w - 2)
        shell = tk.Frame(self.root, bg=BG_APP)
        shell.pack(fill=tk.BOTH, expand=True)
        shell.grid_columnconfigure(0, weight=0, minsize=rail_w)
        shell.grid_columnconfigure(1, weight=0, minsize=1)
        shell.grid_columnconfigure(2, weight=1, minsize=stage_min_w)
        shell.grid_rowconfigure(0, weight=1)

        self.rail = tk.Frame(shell, bg=BG_RAIL, width=rail_w)
        self.rail.grid(row=0, column=0, sticky="nsew")
        self.rail.pack_propagate(False)

        tk.Frame(shell, bg=BORDER, width=1).grid(row=0, column=1, sticky="ns")

        self.stage = tk.Frame(shell, bg=BG_STAGE)
        self.stage.grid(row=0, column=2, sticky="nsew")

    def _build_left_rail(self) -> None:
        self.scope_header = OscilloscopeHeader(self.rail, cuda_ok=bool(self.ctx.cuda_ok))
        self.scope_header.pack(side=tk.TOP, fill=tk.X)

        # 底部系统区（优先 pack(BOTTOM)，确保任意窗口高度下语言切换与指南永不裁切）
        rail_footer = tk.Frame(self.rail, bg=BG_RAIL, padx=_s(14), pady=_s(10))
        rail_footer.pack(side=tk.BOTTOM, fill=tk.X)
        tk.Frame(rail_footer, bg=BORDER, height=1).pack(fill=tk.X, pady=(0, _s(8)))
        self.sec_sys_lbl = tk.Label(
            rail_footer, text=_etr("rail_section_sys"),
            font=(FONT, 8, "bold"), fg=TEXT_MUTED, bg=BG_RAIL,
        )
        self.sec_sys_lbl.pack(anchor="w", pady=(0, _s(5)))

        self.lang_seg = SegmentedControl(
            rail_footer,
            options=[("zh", "中文"), ("ja", "日本語"), ("en", "EN")],
            value=self.lang_var.get(),
            on_change=self._on_lang_select,
        )
        self.lang_seg.pack(fill=tk.X, pady=(0, _s(6)))

        self.help_btn = tk.Button(
            rail_footer, text=f"?  {tr('btn_show_guide')}", font=(FONT, 8, "bold"),
            fg=TEXT_SEC, bg=BG_PANEL, activebackground=BG_SLOT_HOVER,
            activeforeground=TEXT_PRI, relief=tk.FLAT, bd=0,
            highlightbackground=BORDER, highlightthickness=1,
            pady=_s(6), cursor="hand2", command=lambda: show_script_help_dialog(self.root),
        )
        self.help_btn.pack(fill=tk.X)

        # 中部参数导轨：默认高度无需滚动，缩至最小尺寸时自动允许滚轮滚动
        rail_scroll_cv = tk.Canvas(self.rail, bg=BG_RAIL, bd=0, highlightthickness=0)
        rail_scroll_cv.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        rail_body = tk.Frame(rail_scroll_cv, bg=BG_RAIL, padx=_s(14), pady=_s(10))
        rail_win_id = rail_scroll_cv.create_window((0, 0), window=rail_body, anchor="nw")

        rail_scroll_cv.bind("<Configure>", lambda e: rail_scroll_cv.itemconfig(rail_win_id, width=e.width))
        rail_body.bind("<Configure>", lambda _e=None: rail_scroll_cv.configure(scrollregion=rail_scroll_cv.bbox("all")))

        def _on_rail_wheel(event):
            if rail_body.winfo_reqheight() > rail_scroll_cv.winfo_height():
                rail_scroll_cv.yview_scroll(int(-1 * (event.delta / 120)), "units")

        rail_scroll_cv.bind("<Enter>", lambda _e: self.root.bind_all("<MouseWheel>", _on_rail_wheel))
        rail_scroll_cv.bind("<Leave>", lambda _e: self.root.unbind_all("<MouseWheel>"))

        # 1.1 引擎模式选择组
        self.sec_engine_lbl = tk.Label(
            rail_body, text=_etr("rail_section_engine"),
            font=(FONT, 8, "bold"), fg=TEXT_MUTED, bg=BG_RAIL,
        )
        self.sec_engine_lbl.pack(anchor="w", pady=(0, _s(6)))

        for m_key in ("single", "multi_speaker", "multi_persp"):
            tile = tk.Frame(
                rail_body, bg=BG_PANEL, highlightbackground=BORDER_SUBTLE,
                highlightthickness=1, cursor="hand2",
            )
            tile.pack(fill=tk.X, pady=(0, _s(5)))

            bar = tk.Frame(tile, bg=BORDER, width=_s(3))
            bar.pack(side=tk.LEFT, fill=tk.Y)

            inner = tk.Frame(tile, bg=BG_PANEL, padx=_s(9), pady=_s(6))
            inner.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            row_top = tk.Frame(inner, bg=BG_PANEL)
            row_top.pack(fill=tk.X)

            tag_lbl = tk.Label(
                row_top, text=self._mode_codes[m_key], font=(MONO, 7, "bold"),
                fg=TEXT_MUTED, bg=BG_SLOT, padx=_s(5), pady=_s(1),
            )
            tag_lbl.pack(side=tk.RIGHT)

            title_lbl = tk.Label(
                row_top, text=tr(f"mode_{m_key}_title"),
                font=(FONT, 9, "bold"), fg=TEXT_PRI, bg=BG_PANEL,
            )
            title_lbl.pack(side=tk.LEFT)

            sub_lbl = tk.Label(
                inner, text=tr(f"mode_{m_key}_sub"), font=(FONT, 8),
                fg=TEXT_MUTED, bg=BG_PANEL, anchor="w", justify=tk.LEFT, wraplength=_s(216),
            )
            sub_lbl.pack(fill=tk.X, pady=(_s(2), 0))

            self.mode_tiles[m_key] = {
                "tile": tile, "bar": bar, "inner": inner,
                "row_top": row_top, "tag": tag_lbl, "title": title_lbl, "sub": sub_lbl,
            }
            for w in (tile, bar, inner, row_top, tag_lbl, title_lbl, sub_lbl):
                w.bind("<Button-1>", lambda _e, k=m_key: self._select_mode(k))
                w.bind("<Enter>", lambda _e, k=m_key: self._paint_mode_tiles(hover_key=k))
                w.bind("<Leave>", lambda _e: self._paint_mode_tiles())

        # 模式说明微面板
        mode_desc_box = tk.Frame(
            rail_body, bg=BG_APP, highlightbackground=BORDER_SUBTLE,
            highlightthickness=1, padx=_s(9), pady=_s(6),
        )
        mode_desc_box.pack(fill=tk.X, pady=(_s(1), _s(10)))
        self.mode_desc_var = tk.StringVar(value=tr(f"mode_{self.mode_var.get()}_desc"))
        tk.Label(
            mode_desc_box, textvariable=self.mode_desc_var, font=(FONT, 8),
            fg=TEXT_SEC, bg=BG_APP, anchor="w", justify=tk.LEFT, wraplength=_s(230),
        ).pack(fill=tk.X)

        # 1.2 运行参数区
        tk.Frame(rail_body, bg=BORDER, height=1).pack(fill=tk.X, pady=(0, _s(8)))
        self.sec_params_lbl = tk.Label(
            rail_body, text=_etr("rail_section_params"),
            font=(FONT, 8, "bold"), fg=TEXT_MUTED, bg=BG_RAIL,
        )
        self.sec_params_lbl.pack(anchor="w", pady=(0, _s(7)))

        ext_head = tk.Frame(rail_body, bg=BG_RAIL)
        ext_head.pack(fill=tk.X, pady=(0, _s(4)))
        self.ext_title_lbl = tk.Label(
            ext_head, text=tr("label_end_extend"), font=(FONT, 8), fg=TEXT_SEC, bg=BG_RAIL,
        )
        self.ext_title_lbl.pack(side=tk.LEFT)
        self.ext_unit_lbl = tk.Label(
            ext_head, text=tr("label_sec"), font=(FONT, 8), fg=TEXT_MUTED, bg=BG_RAIL,
        )
        self.ext_unit_lbl.pack(side=tk.RIGHT)

        ext_ctrl = tk.Frame(rail_body, bg=BG_APP, highlightbackground=BORDER, highlightthickness=1)
        ext_ctrl.pack(fill=tk.X, pady=(0, _s(5)))

        self.btn_ext_minus = tk.Button(
            ext_ctrl, text="−", font=(MONO, 10, "bold"), fg=TEXT_SEC,
            bg=BG_SLOT, activebackground=BG_SLOT_HOVER, activeforeground=TEXT_PRI,
            relief=tk.FLAT, bd=0, width=3, cursor="hand2",
            command=lambda: self._step_extend(-0.1),
        )
        self.btn_ext_minus.pack(side=tk.LEFT, fill=tk.Y)

        self.ext_entry = tk.Entry(
            ext_ctrl, textvariable=self.end_extend_var, font=(MONO, 10, "bold"),
            bg=BG_APP, fg=ACCENT_HOVER, insertbackground=ACCENT_HOVER,
            relief=tk.FLAT, bd=0, justify="center",
        )
        self.ext_entry.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, ipady=_s(4))
        self.ext_entry.bind("<FocusOut>", lambda _e: (self._refresh_preset_chips(), self._save_settings()))
        self.ext_entry.bind("<Return>", lambda _e: (self._refresh_preset_chips(), self._save_settings()))

        self.btn_ext_plus = tk.Button(
            ext_ctrl, text="＋", font=(MONO, 10, "bold"), fg=TEXT_SEC,
            bg=BG_SLOT, activebackground=BG_SLOT_HOVER, activeforeground=TEXT_PRI,
            relief=tk.FLAT, bd=0, width=3, cursor="hand2",
            command=lambda: self._step_extend(0.1),
        )
        self.btn_ext_plus.pack(side=tk.RIGHT, fill=tk.Y)

        preset_row = tk.Frame(rail_body, bg=BG_RAIL)
        preset_row.pack(fill=tk.X, pady=(0, _s(10)))
        for idx, pv in enumerate(("0.0", "0.5", "0.8", "1.5")):
            chip = tk.Label(
                preset_row, text=f"{pv}s", font=(MONO, 8), pady=_s(2),
                bg=BG_PANEL, fg=TEXT_MUTED, highlightbackground=BORDER_SUBTLE,
                highlightthickness=1, cursor="hand2",
            )
            chip.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0 if idx == 0 else _s(4), 0))
            chip.bind("<Button-1>", lambda _e, v=pv: self._set_preset(v))
            self.preset_btns[pv] = chip
        self._refresh_preset_chips()

        ono_row = tk.Frame(rail_body, bg=BG_RAIL)
        ono_row.pack(fill=tk.X, pady=(0, _s(10)))
        self.ono_lbl = tk.Label(
            ono_row, text=tr("check_ono_gate"), font=(FONT, 8),
            fg=TEXT_PRI, bg=BG_RAIL, anchor="w", wraplength=_s(192), justify=tk.LEFT,
        )
        self.ono_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.ono_switch = ToggleSwitch(
            ono_row, self.ono_gate_var, command=self._save_settings,
            bg=BG_RAIL, pil_ok=bool(self.ctx.pil),
        )
        self.ono_switch.pack(side=tk.RIGHT, padx=(_s(6), 0))

        self.dev_lbl = tk.Label(
            rail_body, text=tr("label_device"), font=(FONT, 8), fg=TEXT_SEC, bg=BG_RAIL,
        )
        self.dev_lbl.pack(anchor="w", pady=(0, _s(4)))
        self.dev_seg = SegmentedControl(
            rail_body,
            options=[("cuda", "CUDA GPU"), ("cpu", "CPU")],
            value=self.dev_var.get(),
            on_change=lambda k: (self.dev_var.set(k), self._save_settings()),
        )
        if not self.ctx.cuda_ok:
            self.dev_seg.set_disabled_keys(["cuda"])
        self.dev_seg.pack(fill=tk.X, pady=(0, _s(4)))

    def _build_center_stage(self) -> None:
        # 2.1 顶部工具栏
        topbar = tk.Frame(self.stage, bg=BG_PANEL_ALT, height=_s(52), padx=_s(18))
        topbar.pack(fill=tk.X)
        topbar.pack_propagate(False)

        tb_right = tk.Frame(topbar, bg=BG_PANEL_ALT)
        tb_right.pack(side=tk.RIGHT, fill=tk.Y, pady=_s(10))

        self.clear_btn = tk.Button(
            tb_right, text=tr("btn_clear"), font=(FONT, 9),
            fg=TEXT_SEC, bg=BG_SLOT, activebackground=BG_SLOT_HOVER,
            activeforeground=TEXT_PRI, relief=tk.FLAT, bd=0,
            padx=_s(12), pady=_s(4), cursor="hand2",
        )
        self.clear_btn.pack(side=tk.RIGHT)

        self.add_btn = tk.Button(
            tb_right, text=_etr("import_files"), font=(FONT, 9, "bold"),
            fg=ACCENT_HOVER, bg=ACCENT_DIM, activebackground="#163E58",
            activeforeground=TEXT_PRI, relief=tk.FLAT, bd=0,
            highlightbackground="#1E4B68", highlightthickness=1,
            padx=_s(14), pady=_s(4), cursor="hand2",
        )
        self.add_btn.pack(side=tk.RIGHT, padx=(0, _s(8)))

        self.dnd_hint_lbl = tk.Label(
            tb_right, text=_etr("dnd_hint"), font=(FONT, 8), fg=TEXT_MUTED, bg=BG_PANEL_ALT,
        )
        self.dnd_hint_lbl.pack(side=tk.RIGHT, padx=(0, _s(14)))

        def _on_topbar_resize(event):
            if event.width < _s(760):
                if self.dnd_hint_lbl.winfo_manager():
                    self.dnd_hint_lbl.pack_forget()
            elif not self.dnd_hint_lbl.winfo_manager():
                self.dnd_hint_lbl.pack(side=tk.RIGHT, padx=(0, _s(14)))

        topbar.bind("<Configure>", _on_topbar_resize)

        tb_left = tk.Frame(topbar, bg=BG_PANEL_ALT)
        tb_left.pack(side=tk.LEFT, fill=tk.Y)
        self.queue_title_lbl = tk.Label(
            tb_left, text=tr("queue_title"), font=(FONT, 11, "bold"), fg=TEXT_PRI, bg=BG_PANEL_ALT,
        )
        self.queue_title_lbl.pack(side=tk.LEFT, pady=_s(12))

        kpi_frame = tk.Frame(tb_left, bg=BG_PANEL_ALT)
        kpi_frame.pack(side=tk.LEFT, padx=(_s(14), 0), pady=_s(12))

        def _make_kpi_pill(color_fg, color_bg):
            box = tk.Frame(
                kpi_frame, bg=color_bg, highlightbackground=BORDER,
                highlightthickness=1, padx=_s(8), pady=_s(2),
            )
            box.pack(side=tk.LEFT, padx=(0, _s(6)))
            k_lbl = tk.Label(box, text="", font=(FONT, 8), fg=TEXT_MUTED, bg=color_bg)
            k_lbl.pack(side=tk.LEFT, padx=(0, _s(5)))
            v_lbl = tk.Label(box, text="00", font=(MONO, 9, "bold"), fg=color_fg, bg=color_bg)
            v_lbl.pack(side=tk.LEFT)
            return k_lbl, v_lbl

        self.kpi_total_k, self.kpi_total_v = _make_kpi_pill(TEXT_PRI, BG_APP)
        self.kpi_ready_k, self.kpi_ready_v = _make_kpi_pill(ACCENT_HOVER, ACCENT_DIM)
        self.kpi_done_k, self.kpi_done_v = _make_kpi_pill(EMERALD, EMERALD_DIM)

        tk.Frame(self.stage, bg=BORDER, height=1).pack(fill=tk.X)

        # 2.2 未配对素材暂存坞
        self.dock_frame = tk.Frame(self.stage, bg="#151311", highlightbackground="#423117", highlightthickness=1)
        tk.Frame(self.dock_frame, bg=AMBER, width=_s(3)).pack(side=tk.LEFT, fill=tk.Y)
        self.dock_content = tk.Frame(self.dock_frame, bg="#151311", padx=_s(14), pady=_s(8))
        self.dock_content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        dock_top = tk.Frame(self.dock_content, bg="#151311")
        dock_top.pack(fill=tk.X, pady=(0, _s(6)))
        self.dock_title_lbl = tk.Label(
            dock_top, text=tr("unpaired_tray_title"), font=(FONT, 8, "bold"), fg=AMBER, bg="#151311",
        )
        self.dock_title_lbl.pack(side=tk.LEFT)
        self.dock_hint_lbl = tk.Label(
            dock_top, text=tr("unpaired_tray_hint"), font=(FONT, 8), fg=TEXT_MUTED, bg="#151311",
        )
        self.dock_hint_lbl.pack(side=tk.LEFT, padx=_s(10))
        self.dock_pair_btn = tk.Button(
            dock_top, text=_etr("auto_pair"), font=(FONT, 8, "bold"),
            fg=AMBER, bg=AMBER_DIM, activebackground="#3C2D14",
            activeforeground=AMBER_HOVER, relief=tk.FLAT, bd=0,
            padx=_s(10), pady=_s(2), cursor="hand2",
        )
        self.dock_pair_btn.pack(side=tk.RIGHT)

        self.dock_items = tk.Frame(self.dock_content, bg="#151311")
        self.dock_items.pack(fill=tk.X)

        # 2.3 中央轨道矩阵主容器
        self.matrix_outer = tk.Frame(self.stage, bg=BG_STAGE)
        self.matrix_outer.pack(fill=tk.BOTH, expand=True)

        self.col_header = tk.Frame(self.matrix_outer, bg=BG_PANEL_ALT, height=_s(30))
        self.col_header.pack_propagate(False)
        tk.Frame(self.col_header, bg=BG_PANEL_ALT, width=_s(3)).pack(side=tk.LEFT, fill=tk.Y)
        tk.Frame(self.col_header, bg=BG_PANEL_ALT, width=_s(8)).pack(side=tk.RIGHT, fill=tk.Y)
        self.col_header_grid = tk.Frame(self.col_header, bg=BG_PANEL_ALT, padx=_s(14))
        self.col_header_grid.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.col_header_sep = tk.Frame(self.matrix_outer, bg=BORDER, height=1)

        self.matrix_scroll_wrap = tk.Frame(self.matrix_outer, bg=BG_STAGE)
        self.matrix_scroll_wrap.pack(fill=tk.BOTH, expand=True)

        self.matrix_canvas = tk.Canvas(self.matrix_scroll_wrap, bg=BG_STAGE, bd=0, highlightthickness=0)
        matrix_sb = ttk.Scrollbar(
            self.matrix_scroll_wrap, orient=tk.VERTICAL,
            command=self.matrix_canvas.yview, style="Studio.Vertical.TScrollbar",
        )
        self.matrix_canvas.configure(yscrollcommand=matrix_sb.set)
        matrix_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.matrix_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.matrix_inner = tk.Frame(self.matrix_canvas, bg=BG_STAGE)
        self.matrix_win_id = self.matrix_canvas.create_window((0, 0), window=self.matrix_inner, anchor="nw")

        # 2.4 底部可折叠终端日志抽屉
        self.console_drawer = tk.Frame(self.stage, bg=BG_CONSOLE, height=_s(170))
        self.console_drawer.pack_propagate(False)
        tk.Frame(self.console_drawer, bg=BORDER, height=1).pack(fill=tk.X)
        console_bar = tk.Frame(self.console_drawer, bg="#0D1016", padx=_s(16), pady=_s(4))
        console_bar.pack(fill=tk.X)
        self.console_hdr_lbl = tk.Label(
            console_bar, text=_etr("console_header"),
            font=(FONT, 8, "bold"), fg=TEXT_MUTED, bg="#0D1016",
        )
        self.console_hdr_lbl.pack(side=tk.LEFT)
        self.log_clear_btn = tk.Button(
            console_bar, text=tr("btn_clear_log"), font=(FONT, 8),
            fg=TEXT_MUTED, bg="#161C28", activebackground=BG_SLOT_HOVER,
            activeforeground=TEXT_PRI, relief=tk.FLAT, bd=0,
            padx=_s(8), pady=_s(2), cursor="hand2",
        )
        self.log_clear_btn.pack(side=tk.RIGHT)

        log_body = tk.Frame(self.console_drawer, bg=BG_CONSOLE)
        log_body.pack(fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(
            log_body, bg=BG_CONSOLE, fg="#94A3B8", font=(FONT, 9),
            relief=tk.FLAT, wrap=tk.WORD, padx=_s(16), pady=_s(8),
            spacing1=_s(1), spacing3=_s(1),
            insertbackground=ACCENT_HOVER, selectbackground=ACCENT_DIM,
        )
        log_scroll = ttk.Scrollbar(
            log_body, orient=tk.VERTICAL, command=self.log_text.yview,
            style="Studio.Vertical.TScrollbar",
        )
        self.log_text.configure(yscrollcommand=log_scroll.set)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.log_text.tag_configure("phase", foreground=ACCENT_HOVER)
        self.log_text.tag_configure("ok", foreground=EMERALD)
        self.log_text.tag_configure("warn", foreground=AMBER)
        self.log_text.tag_configure("err", foreground=CRIMSON)
        self.log_clear_btn.configure(command=lambda: self.log_text.delete("1.0", tk.END))

        # 2.5 底部走带控制栏
        self.transport_sep = tk.Frame(self.stage, bg=BORDER, height=1)
        self.transport_sep.pack(fill=tk.X)

        transport = tk.Frame(self.stage, bg=BG_RAIL, height=_s(74), padx=_s(20), pady=_s(12))
        transport.pack(fill=tk.X)
        transport.pack_propagate(False)

        self.start_btn = tk.Button(
            transport, text=tr("btn_start"), font=(FONT, 10, "bold"),
            fg=INK_DARK, bg=ACCENT, activebackground=ACCENT_HOVER,
            activeforeground=INK_DARK, disabledforeground=TEXT_MUTED,
            relief=tk.FLAT, bd=0, padx=_s(26), pady=_s(10), cursor="hand2",
        )
        self.start_btn.pack(side=tk.RIGHT, padx=(_s(18), 0))

        trans_left = tk.Frame(transport, bg=BG_RAIL)
        trans_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        trans_top = tk.Frame(trans_left, bg=BG_RAIL)
        trans_top.pack(fill=tk.X, pady=(0, _s(6)))

        led_sz = _s(10)
        self.led_canvas = tk.Canvas(trans_top, width=led_sz, height=led_sz, bg=BG_RAIL, bd=0, highlightthickness=0)
        self.led_canvas.pack(side=tk.LEFT, padx=(0, _s(8)), pady=_s(3))
        self.led_dot_id = self.led_canvas.create_oval(
            _s(1), _s(1), led_sz - _s(1), led_sz - _s(1), fill=TEXT_MUTED, outline="",
        )

        self.status_var = tk.StringVar(value=tr("status_idle"))
        tk.Label(
            trans_top, textvariable=self.status_var, font=(FONT, 9),
            fg=TEXT_SEC, bg=BG_RAIL, anchor="w",
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.progress_pct_var = tk.StringVar(value="0%")
        tk.Label(
            trans_top, textvariable=self.progress_pct_var, font=(MONO, 10, "bold"),
            fg=ACCENT_HOVER, bg=BG_RAIL,
        ).pack(side=tk.RIGHT, padx=(_s(10), 0))

        self.console_toggle_btn = tk.Button(
            trans_top, text="", font=(MONO, 8, "bold"),
            fg=TEXT_SEC, bg=BG_SLOT, activebackground=BG_SLOT_HOVER,
            activeforeground=TEXT_PRI, relief=tk.FLAT, bd=0,
            padx=_s(10), pady=_s(2), cursor="hand2", command=self._toggle_console,
        )
        self.console_toggle_btn.pack(side=tk.RIGHT)
        self._refresh_console_btn()

        self.progress_gauge = TransportProgressCanvas(trans_left)
        self.progress_gauge.pack(fill=tk.X)

        self._ui_log_sink = self.append_log
        for sw_msg in self._startup_warnings:
            self.append_log(f"⚠ {sw_msg}")
        self._startup_warnings.clear()

    def _bind_matrix_and_events(self) -> None:
        self.matrix = TaskMatrixController(
            root=self.root,
            ctx=self.ctx,
            mode_var=self.mode_var,
            dock_frame=self.dock_frame,
            dock_items=self.dock_items,
            matrix_outer=self.matrix_outer,
            col_header=self.col_header,
            col_header_sep=self.col_header_sep,
            col_header_grid=self.col_header_grid,
            matrix_scroll_wrap=self.matrix_scroll_wrap,
            matrix_canvas=self.matrix_canvas,
            matrix_inner=self.matrix_inner,
            matrix_win_id=self.matrix_win_id,
            drop_targets=(
                self.stage, self.matrix_outer, self.matrix_scroll_wrap, self.matrix_canvas,
                self.matrix_inner, self.dock_frame, self.dock_content, self.dock_items,
            ),
            on_queue_changed=self._update_queue_meta,
            on_log=self.append_log,
            on_status=self.update_status,
            on_set_led=self._set_led,
            on_ensure_console=self._ensure_console_open,
        )

        def _on_matrix_canvas_resize(event):
            self.matrix_canvas.itemconfig(self.matrix_win_id, width=event.width)
            if not self.matrix.task_store:
                self.matrix_canvas.itemconfig(self.matrix_win_id, height=max(event.height, _s(260)))

        def _on_matrix_wheel(event):
            if self.matrix.task_store:
                self.matrix_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        self.matrix_canvas.bind("<Configure>", _on_matrix_canvas_resize)
        self.matrix_inner.bind("<Configure>", lambda _e: self.matrix_canvas.configure(scrollregion=self.matrix_canvas.bbox("all")))
        self.matrix_scroll_wrap.bind("<Enter>", lambda _e: self.root.bind_all("<MouseWheel>", _on_matrix_wheel))
        self.matrix_scroll_wrap.bind("<Leave>", lambda _e: self.root.unbind_all("<MouseWheel>"))

        self.add_btn.configure(command=self.matrix.browse_files)
        self.clear_btn.configure(command=self._clear_tasks)
        self.dock_pair_btn.configure(command=self.matrix.auto_assemble_from_tray)
        self.start_btn.configure(command=self._run_all)

        self.root._fa_set_running = self._set_running
        self.root._fa_ui = self
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── 导轨、日志、语言与模式切换交互 ──────────────────────────────────

    def _on_lang_select(self, code: str) -> None:
        I18N.set_lang(code)
        self.lang_var.set(code)
        self._refresh_all_i18n()
        self._save_settings()

    def _select_mode(self, key: str) -> None:
        if self.ctx.is_running():
            return
        self.mode_var.set(key)
        self._on_mode_change()

    def _paint_mode_tiles(self, hover_key: str | None = None) -> None:
        cur = self.mode_var.get()
        for k, refs in self.mode_tiles.items():
            sel = k == cur
            hov = (k == hover_key) and not sel
            ac = self._mode_accents[k]
            dim = self._mode_dims[k]
            bg_c = "#17202D" if sel else (BG_SLOT if hov else BG_PANEL)
            refs["tile"].configure(bg=bg_c, highlightbackground=ac if sel else (BORDER_STRONG if hov else BORDER_SUBTLE))
            refs["bar"].configure(bg=ac if sel else (BORDER_STRONG if hov else BORDER_SUBTLE))
            refs["inner"].configure(bg=bg_c)
            refs["row_top"].configure(bg=bg_c)
            refs["title"].configure(bg=bg_c, fg=TEXT_PRI if (sel or hov) else TEXT_SEC)
            refs["sub"].configure(bg=bg_c, fg=TEXT_SEC if sel else TEXT_MUTED)
            refs["tag"].configure(bg=dim if sel else BG_SLOT, fg=ac if sel else TEXT_MUTED)

    def _step_extend(self, delta: float) -> None:
        if self.ctx.is_running():
            return
        try:
            v = float(self.end_extend_var.get() or 0)
        except ValueError:
            v = 0.0
        v = max(0.0, round(v + delta, 1))
        self.end_extend_var.set(f"{v:.1f}")
        self._refresh_preset_chips()
        self._save_settings()

    def _set_preset(self, val_s: str) -> None:
        if self.ctx.is_running():
            return
        self.end_extend_var.set(val_s)
        self._refresh_preset_chips()
        self._save_settings()

    def _refresh_preset_chips(self) -> None:
        try:
            cur_v = f"{float(self.end_extend_var.get()):.1f}"
        except ValueError:
            cur_v = ""
        for val_s, lbl in self.preset_btns.items():
            active = val_s == cur_v
            lbl.configure(
                bg=ACCENT_DIM if active else BG_PANEL,
                fg=ACCENT_HOVER if active else TEXT_MUTED,
                highlightbackground=ACCENT if active else BORDER_SUBTLE,
            )

    def _toggle_console(self) -> None:
        if self.log_visible.get():
            self.console_drawer.pack_forget()
            self.log_visible.set(False)
        else:
            self.console_drawer.pack(fill=tk.X, before=self.transport_sep)
            self.log_visible.set(True)
        self._refresh_console_btn()

    def _ensure_console_open(self) -> None:
        if not self.log_visible.get():
            self.console_drawer.pack(fill=tk.X, before=self.transport_sep)
            self.log_visible.set(True)
            self._refresh_console_btn()

    def _refresh_console_btn(self) -> None:
        active = self.log_visible.get()
        sym = "▾" if active else "▸"
        self.console_toggle_btn.configure(
            text=f">_  {_etr('console_btn')} {sym}",
            bg=ACCENT_DIM if active else BG_SLOT,
            fg=ACCENT_HOVER if active else TEXT_SEC,
        )

    def append_log(self, msg: str) -> None:
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
                self.log_text.insert(tk.END, line, tag)
            else:
                self.log_text.insert(tk.END, line)
            self.log_text.see(tk.END)
        self._post_ui(_insert)

    def _set_led(self, color: str) -> None:
        self.led_canvas.itemconfigure(self.led_dot_id, fill=color)

    def update_status(self, msg: str) -> None:
        self._post_ui(lambda: self.status_var.set(msg))

    def _update_queue_meta(self) -> None:
        total, ready, done = self.matrix.get_queue_counts()
        self.kpi_total_v.configure(text=f"{total:02d}")
        self.kpi_ready_v.configure(text=f"{ready:02d}")
        self.kpi_done_v.configure(text=f"{done:02d}")

        if not getattr(self.root, "_fa_running", False):
            self.start_btn.configure(
                text=f"▸  {tr('btn_start_with_count', count=ready)}" if ready else f"▸  {tr('btn_start')}",
                bg=ACCENT if ready else "#1B2A3A",
                fg=INK_DARK if ready else TEXT_MUTED,
                activebackground=ACCENT_HOVER if ready else "#22364A",
                activeforeground=INK_DARK if ready else TEXT_SEC,
            )

    def _clear_tasks(self) -> None:
        if self.ctx.is_running():
            return
        self.matrix.clear_all()
        self.progress_gauge.set_value(0)
        self.progress_pct_var.set("0%")
        self._set_led(TEXT_MUTED)
        self.update_status(tr("status_queue_cleared"))

    def _on_mode_change(self) -> None:
        cur = self.mode_var.get()
        self._paint_mode_tiles()
        self.mode_desc_var.set(tr(f"mode_{cur}_desc"))
        self.matrix.render_all_cards()
        self._save_settings()

    def _refresh_all_i18n(self) -> None:
        self.root.title(tr("app_title"))
        self.scope_header.redraw()
        self.sec_engine_lbl.configure(text=_etr("rail_section_engine"))
        self.sec_params_lbl.configure(text=_etr("rail_section_params"))
        self.sec_sys_lbl.configure(text=_etr("rail_section_sys"))

        for k, refs in self.mode_tiles.items():
            refs["title"].configure(text=tr(f"mode_{k}_title"))
            refs["sub"].configure(text=tr(f"mode_{k}_sub"))
        self.mode_desc_var.set(tr(f"mode_{self.mode_var.get()}_desc"))

        self.ext_title_lbl.configure(text=tr("label_end_extend"))
        self.ext_unit_lbl.configure(text=tr("label_sec"))
        self.ono_lbl.configure(text=tr("check_ono_gate"))
        self.dev_lbl.configure(text=tr("label_device"))
        self.help_btn.configure(text=f"?  {tr('btn_show_guide')}")

        self.queue_title_lbl.configure(text=tr("queue_title"))
        self.kpi_total_k.configure(text=_etr("kpi_total"))
        self.kpi_ready_k.configure(text=_etr("kpi_ready"))
        self.kpi_done_k.configure(text=_etr("kpi_done"))
        self.add_btn.configure(text=_etr("import_files"))
        self.clear_btn.configure(text=tr("btn_clear"))
        self.dnd_hint_lbl.configure(text=_etr("dnd_hint"))

        self.dock_title_lbl.configure(text=tr("unpaired_tray_title"))
        self.dock_hint_lbl.configure(text=tr("unpaired_tray_hint"))
        self.dock_pair_btn.configure(text=_etr("auto_pair"))

        self.console_hdr_lbl.configure(text=_etr("console_header"))
        self.log_clear_btn.configure(text=tr("btn_clear_log"))
        self._refresh_console_btn()

        self.matrix.render_all_cards()
        if not getattr(self.root, "_fa_running", False) and not self.matrix.task_store:
            self.status_var.set(tr("status_idle"))

    # ── 对齐执行引擎回调与任务调度 ──────────────────────────────────────

    def _set_running(self, running: bool) -> None:
        self.root._fa_running = running
        self.scope_header.set_running(running)
        self.progress_gauge.set_running(running)
        self.ono_switch.set_disabled(running)
        self.dev_seg.set_disabled(running)
        state = tk.DISABLED if running else tk.NORMAL
        self.add_btn.configure(state=state)
        self.clear_btn.configure(state=state)
        self.dock_pair_btn.configure(state=state)
        self.btn_ext_minus.configure(state=state)
        self.btn_ext_plus.configure(state=state)
        self.ext_entry.configure(state=state)
        if running:
            self._set_led(ACCENT_HOVER)
            self.start_btn.configure(
                state=tk.DISABLED, text=f"●  {tr('btn_running')}",
                bg=ACCENT_DIM, fg=ACCENT_HOVER,
            )
        else:
            self.start_btn.configure(state=tk.NORMAL)
            self._update_queue_meta()

    def _set_task_status(self, iid: str, status: str) -> None:
        self._post_ui(lambda: self.matrix.set_task_status(iid, status))

    def _set_progress(self, value: float, message: str) -> None:
        def _apply():
            self.progress_gauge.set_value(value)
            self.progress_pct_var.set(f"{value:.0f}%")
            self.status_var.set(message)
        self._post_ui(_apply)

    def _run_all(self) -> None:
        if self.ctx.is_running():
            return
        self._save_settings()
        try:
            end_extend = float(self.end_extend_var.get() or 0)
            if end_extend < 0:
                raise ValueError
        except ValueError:
            messagebox.showerror(tr("msg_param_error_title"), tr("msg_param_error_text"), parent=self.root)
            return

        current = self.mode_var.get()
        valid = self.matrix.get_ready_tasks()
        if not valid:
            messagebox.showinfo(tr("msg_empty_queue_title"), tr("msg_empty_queue_text"), parent=self.root)
            return

        if current == "multi_persp":
            missing = [t["name"] for t in valid if not t.get("audio_b")]
            if missing:
                preview = "\n".join(missing[:8]) + ("\n…" if len(missing) > 8 else "")
                if not messagebox.askyesno(
                    tr("msg_missing_persp_b_title"),
                    tr("msg_missing_persp_b_text", names=preview),
                    parent=self.root,
                ):
                    return

        self._set_running(True)
        self._set_progress(0, tr("status_preparing", count=len(valid)))
        self.append_log(f"\n[RUN] {tr(f'mode_{current}_title')} · {self.dev_var.get().upper()} · {len(valid)} track(s)")

        def _worker():
            import traceback as _tb
            succeeded, failed = 0, 0
            failed_details: list[str] = []
            total = len(valid)
            try:
                for idx, item in enumerate(valid):
                    iid = item["iid"]
                    self._set_task_status(iid, "running")
                    self.append_log(f"[{idx + 1}/{total}] {item['name']}")

                    def _pct(percent, message):
                        overall = (idx * 100 + percent) / total
                        self._set_progress(
                            overall,
                            tr("status_progress_step", current=idx + 1, total=total, message=message),
                        )

                    ok, _output, error = self.ctx.run_pipeline(
                        script_path=item["script"], audio_path=item["audio"],
                        lora_path=self.ctx.lora_path, log_callback=self.append_log, progress_callback=_pct,
                        dev=self.dev_var.get(), vram_chunk=DEFAULT_VRAM_CHUNK, persist_gpu=True,
                        line_end_extend=end_extend, enable_silence_snap=True,
                        enable_resegment=True, enable_ono_gate=bool(self.ono_gate_var.get()),
                        enable_multi_order=current in ("multi_speaker", "multi_persp"),
                        audio_path_b=item.get("audio_b") if current == "multi_persp" else None,
                    )
                    self._set_task_status(iid, "done" if ok else "failed")
                    if ok:
                        succeeded += 1
                    else:
                        failed += 1
                        err_text = error or tr("msg_unknown_error")
                        failed_details.append(f"• {item['name']}: {err_text}")
                        self.append_log(f"处理失败：{err_text}")
                        self._post_ui(self._ensure_console_open)
                self._post_ui(lambda: self._set_led(EMERALD if failed == 0 else AMBER))
                self._set_progress(100, tr("status_all_done", succeeded=succeeded, failed=failed))
                self.append_log(f"✓ 批量任务完成：成功 {succeeded}，失败 {failed}")
                if failed > 0:
                    preview = "\n".join(failed_details[:6])
                    if len(failed_details) > 6:
                        preview += "\n" + tr("msg_tasks_failed_more", count=len(failed_details) - 6)
                    self._post_ui(lambda p=preview, s=succeeded, f=failed: (
                        self._ensure_console_open(),
                        messagebox.showerror(
                            tr("msg_tasks_failed_title"),
                            tr("msg_tasks_failed_text", succeeded=s, failed=f, details=p),
                            parent=self.root,
                        ),
                    ))
            except Exception as err:
                failed += 1
                tb_str = _tb.format_exc()
                err_desc = f"{type(err).__name__}: {err}" if str(err) else type(err).__name__
                self._post_ui(lambda: (self._set_led(CRIMSON), self._ensure_console_open()))
                self.append_log(f"任务中断：{err_desc}\n{tb_str}")
                self._set_progress(0, tr("status_interrupted", error=err_desc))
                self._post_ui(lambda d=err_desc: messagebox.showerror(
                    tr("msg_task_interrupted_title"),
                    tr("msg_task_interrupted_text", error=d),
                    parent=self.root,
                ))
            finally:
                try:
                    self.ctx.release_gpu()
                finally:
                    self._post_ui(lambda: self._set_running(False))

        self.ctx.start_worker(_worker)

    def _on_close(self) -> None:
        self.closing = True
        self.scope_header.stop()
        self.progress_gauge.stop()
        self._save_settings()
        self.root.destroy()

    def build(self):
        self.root.after(30, self._drain_ui_events)
        self._load_settings()
        self._build_shell()
        self._build_left_rail()
        self._build_center_stage()
        self._bind_matrix_and_events()

        # 初始化渲染与居中定位
        self._on_mode_change()
        self._refresh_all_i18n()
        self.root.update_idletasks()
        _apply_dark_titlebar(self.root)
        w, h = self.root.winfo_width(), self.root.winfo_height()
        self.root.geometry(
            f"+{max(0, (self.root.winfo_screenwidth() - w) // 2)}"
            f"+{max(0, (self.root.winfo_screenheight() - h) // 2)}"
        )
        return self.root


def build_ui(ctx):
    """创建 FA-ASMR 专业音频控制台前端。ctx 提供业务回调与运行环境信息。"""
    return StudioConsoleUI(ctx).build()
