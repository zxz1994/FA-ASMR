# -*- coding: utf-8 -*-
"""FA-ASMR Acoustic Mastering Console — 多轨编排矩阵与拖拽控制器。

从 fa_asmr_ui.py 抽离的队列数据模型与矩阵交互控制器：
- 任务存储 (_task_store) 与未配对暂存坞 (_unpaired_pool) 管理
- 空状态矢量导入画布、多轨水平条带、视角槽位与快捷操作渲染
- 内部槽位拖拽互换 / 拖入暂存坞 / 拖建新轨状态机
- 外部系统拖放 (DND)、文件选择导入、自动配对与右键菜单
"""
from __future__ import annotations

import os
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Callable, Optional, Sequence

from fa_asmr.core.i18n import tr
from fa_asmr.ui.widgets import (
    ACCENT,
    ACCENT_DIM,
    ACCENT_HOVER,
    AMBER,
    AMBER_DIM,
    BG_PANEL,
    BG_PANEL_ALT,
    BG_SLOT,
    BG_SLOT_EMPTY,
    BG_STAGE,
    BORDER,
    BORDER_STRONG,
    BORDER_SUBTLE,
    CRIMSON,
    CRIMSON_DIM,
    EMERALD,
    EMERALD_DIM,
    FONT,
    INK_DARK,
    MONO,
    TEXT_MUTED,
    TEXT_PRI,
    TEXT_SEC,
    ToolTip,
    _etr,
    _s,
    _sf,
    _truncate_filename,
)


class TaskMatrixController:
    """管理未配对素材坞与中央多轨编排矩阵的数据模型、渲染及交互手势。"""

    def __init__(
        self,
        *,
        root: tk.Misc,
        ctx,
        mode_var: tk.StringVar,
        dock_frame: tk.Frame,
        dock_items: tk.Frame,
        matrix_outer: tk.Frame,
        col_header: tk.Frame,
        col_header_sep: tk.Frame,
        col_header_grid: tk.Frame,
        matrix_scroll_wrap: tk.Frame,
        matrix_canvas: tk.Canvas,
        matrix_inner: tk.Frame,
        matrix_win_id: int,
        drop_targets: Sequence[tk.Misc] = (),
        on_queue_changed: Optional[Callable[[], None]] = None,
        on_log: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
        on_set_led: Optional[Callable[[str], None]] = None,
        on_ensure_console: Optional[Callable[[], None]] = None,
    ):
        self.root = root
        self.ctx = ctx
        self.mode_var = mode_var
        self.dock_frame = dock_frame
        self.dock_items = dock_items
        self.matrix_outer = matrix_outer
        self.col_header = col_header
        self.col_header_sep = col_header_sep
        self.col_header_grid = col_header_grid
        self.matrix_scroll_wrap = matrix_scroll_wrap
        self.matrix_canvas = matrix_canvas
        self.matrix_inner = matrix_inner
        self.matrix_win_id = matrix_win_id

        self._on_queue_changed = on_queue_changed or (lambda: None)
        self._append_log = on_log or (lambda _msg: None)
        self._update_status = on_status or (lambda _msg: None)
        self._set_led = on_set_led or (lambda _color: None)
        self._ensure_console_open = on_ensure_console or (lambda: None)

        self.task_store: dict[str, dict] = {}
        self._unpaired_pool: dict[str, dict] = {}
        self._card_id_counter = 0
        self._uid_counter = 0
        self._slot_registry: list[dict] = []
        self._highlighted_slot: dict | None = None
        self._drag_info: dict = {
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

        if drop_targets:
            self.bind_drop_targets(*drop_targets)

    # ── 对外查询与状态操作 ──────────────────────────────────────────────

    def get_queue_counts(self) -> tuple[int, int, int]:
        total = len(self.task_store)
        done = sum(1 for t in self.task_store.values() if t.get("status") == "done")
        ready = sum(
            1 for t in self.task_store.values()
            if t.get("script") and t.get("audio") and t.get("status") != "done"
        )
        return total, ready, done

    def get_ready_tasks(self) -> list[dict]:
        return [
            t for t in self.task_store.values()
            if t.get("audio") and t.get("script") and t.get("status") != "done"
        ]

    def set_task_status(self, iid: str, status: str) -> None:
        if iid in self.task_store:
            self.task_store[iid]["status"] = status
            self.render_all_cards()

    def is_persp_visible(self) -> bool:
        return self.mode_var.get() == "multi_persp" or any(
            t.get("audio_b") for t in self.task_store.values()
        )

    def clear_all(self) -> None:
        self.task_store.clear()
        self._unpaired_pool.clear()
        self.render_all_cards()

    def bind_drop_targets(self, *widgets) -> None:
        if not self.ctx.has_dnd:
            return
        for w in widgets:
            try:
                w.drop_target_register(self.ctx.dnd_files)
                w.dnd_bind("<<Drop>>", self.on_external_drop)
            except Exception:
                pass

    # ── 矩阵与暂存坞渲染 ────────────────────────────────────────────────

    def _configure_matrix_grid(self, grid_frame: tk.Frame, show_b: bool) -> None:
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

    def _render_column_headers(self, show_b: bool) -> None:
        for w in self.col_header_grid.winfo_children():
            w.destroy()
        self._configure_matrix_grid(self.col_header_grid, show_b)

        def _hdr(c, txt, anchor="w", mono=False):
            tk.Label(
                self.col_header_grid, text=txt, font=(MONO if mono else FONT, 8, "bold"),
                fg=TEXT_MUTED, bg=BG_PANEL_ALT, anchor=anchor,
            ).grid(row=0, column=c, sticky="ew", padx=_s(4))

        _hdr(0, "#", mono=True)
        _hdr(1, tr("col_name"))
        _hdr(2, tr("col_script"))
        _hdr(3, tr("col_persp_a") if show_b else tr("col_audio"))
        if show_b:
            _hdr(4, "⇄", "center", mono=True)
            _hdr(5, tr("col_persp_b"))
        _hdr(6, _etr("col_act"), "e")

    def _render_empty_stage(self) -> None:
        """无任务时绘制专业音频视窗风格的矢量空状态导入区。"""
        self.col_header.pack_forget()
        self.col_header_sep.pack_forget()

        empty_cv = tk.Canvas(self.matrix_inner, bg=BG_STAGE, bd=0, highlightthickness=0, cursor="hand2")
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
            empty_cv.create_rectangle(
                pad, pad, w - pad, h - pad, outline=dash_col,
                width=max(1.0, _sf(1.5)), dash=(_s(6), _s(5)),
            )

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
                empty_cv.create_rectangle(bx, wave_cy - sbh // 2, bx + bar_w, wave_cy + sbh // 2, fill=col, outline="")

            prompt = tr("drop_prompt_persp") if self.mode_var.get() == "multi_persp" else tr("drop_prompt_normal")
            empty_cv.create_text(mx, my + _s(8), text=prompt, fill=TEXT_PRI, font=(FONT, 12, "bold"))
            empty_cv.create_text(mx, my + _s(34), text=_etr("empty_sub"), fill=TEXT_MUTED, font=(FONT, 9))

            # 中心按钮胶囊
            bw, bh_btn = _s(148), _s(34)
            by0 = my + _s(58)
            btn_bg = ACCENT if hov else ACCENT_DIM
            btn_fg = INK_DARK if hov else ACCENT_HOVER
            empty_cv.create_rectangle(
                mx - bw // 2, by0, mx + bw // 2, by0 + bh_btn,
                fill=btn_bg, outline=ACCENT_HOVER, width=1,
            )
            empty_cv.create_text(mx, by0 + bh_btn // 2, text=_etr("import_files"), fill=btn_fg, font=(FONT, 9, "bold"))

        empty_cv.bind("<Configure>", _draw_empty)
        empty_cv.bind("<Enter>", lambda _e: (hover_state.update(on=True), _draw_empty()))
        empty_cv.bind("<Leave>", lambda _e: (hover_state.update(on=False), _draw_empty()))
        empty_cv.bind("<Button-1>", self.browse_files)

        if self.ctx.has_dnd:
            empty_cv.drop_target_register(self.ctx.dnd_files)
            empty_cv.dnd_bind("<<Drop>>", self.on_external_drop)

        self._slot_registry.append({
            "widget": empty_cv, "type": "new_card", "cid": None, "uid": None,
            "rest_border": BORDER, "rest_thick": 0, "in_matrix": True,
        })
        self.matrix_canvas.update_idletasks()
        self.matrix_canvas.itemconfig(self.matrix_win_id, height=max(self.matrix_canvas.winfo_height(), _s(280)))

    def _pick_slot_file(self, cid: str, slot_key: str) -> None:
        if self.ctx.is_running():
            return
        if slot_key == "script":
            path = filedialog.askopenfilename(
                title=_etr("pick_script_title"),
                filetypes=[(tr("dlg_filter_supported"), "*.txt"), (tr("dlg_filter_all"), "*.*")],
            )
        else:
            title_key = "pick_audio_title" if slot_key == "audio" else "pick_audio_b_title"
            path = filedialog.askopenfilename(
                title=_etr(title_key),
                filetypes=[
                    (tr("dlg_filter_audio"), "*.wav;*.mp3;*.flac;*.m4a;*.ogg;*.aac;*.opus;*.wma"),
                    (tr("dlg_filter_all"), "*.*"),
                ],
            )
        if path and cid in self.task_store:
            self.task_store[cid][slot_key] = os.path.abspath(path)
            self.render_all_cards()

    def _make_slot_chip(
        self,
        parent: tk.Frame,
        cid: str,
        slot_type: str,
        file_path: Optional[str],
        badge_code: str,
        empty_text: str,
        accent_fg: str,
        accent_bg: str,
        show_b: bool,
    ) -> tk.Frame:
        has_file = bool(file_path)
        bg_c = BG_SLOT if has_file else BG_SLOT_EMPTY
        bd_c = BORDER if has_file else "#253046"

        chip = tk.Frame(
            parent, bg=bg_c, highlightbackground=bd_c,
            highlightthickness=1, cursor="hand2", padx=_s(8), pady=_s(5),
        )
        self._slot_registry.append({
            "widget": chip, "type": slot_type, "cid": cid, "uid": None,
            "rest_border": bd_c, "rest_thick": 1, "in_matrix": True,
        })

        if has_file:
            fname = os.path.basename(file_path)
            tag = tk.Label(
                chip, text=badge_code, font=(MONO, 7, "bold"),
                fg=accent_fg, bg=accent_bg, padx=_s(4), pady=_s(1),
            )
            tag.pack(side=tk.LEFT, padx=(0, _s(6)))
            lbl = tk.Label(
                chip, text=_truncate_filename(fname, 16 if show_b else 24),
                font=(FONT, 8), fg=TEXT_PRI, bg=bg_c, anchor="w",
            )
            lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
            ToolTip(chip, file_path)

            def _start(e, c=cid, p=file_path, n=fname, st=slot_type):
                self._on_drag_press(e, p_type=st, p_file=p, p_name=n, cid=c, uid=None)

            for w in (chip, tag, lbl):
                w.bind("<ButtonPress-1>", _start)
                w.bind("<B1-Motion>", self._on_drag_motion)
                w.bind("<ButtonRelease-1>", self._on_drag_release)
                w.bind("<Button-3>", lambda e, c=cid: self._show_card_context_menu(e, c))
            self.bind_drop_targets(chip, tag, lbl)
        else:
            ph = tk.Label(chip, text=empty_text, font=(FONT, 8), fg=TEXT_MUTED, bg=bg_c, anchor="center")
            ph.pack(fill=tk.X)
            for w in (chip, ph):
                w.bind("<Button-1>", lambda _e, c=cid, sk=slot_type: self._pick_slot_file(c, sk))
                w.bind("<Enter>", lambda _e, f=chip, p=ph: (f.configure(highlightbackground=accent_fg), p.configure(fg=accent_fg)))
                w.bind("<Leave>", lambda _e, f=chip, p=ph, b=bd_c: (f.configure(highlightbackground=b), p.configure(fg=TEXT_MUTED)))
            self.bind_drop_targets(chip, ph)
        return chip

    def render_all_cards(self) -> None:
        """渲染未配对素材坞与中央多轨编排矩阵。"""
        self._slot_registry.clear()
        self._highlighted_slot = None

        # 1. 未配对暂存坞
        for w in self.dock_items.winfo_children():
            w.destroy()

        if self._unpaired_pool:
            self.dock_frame.pack(fill=tk.X, padx=_s(16), pady=(_s(10), 0), before=self.matrix_outer)
            self._slot_registry.append({
                "widget": self.dock_frame, "type": "tray", "cid": None, "uid": None,
                "rest_border": "#423117", "rest_thick": 1, "in_matrix": False,
            })
            row_f = None
            for idx, (uid, item) in enumerate(list(self._unpaired_pool.items())):
                if idx % 4 == 0:
                    row_f = tk.Frame(self.dock_items, bg="#151311")
                    row_f.pack(fill=tk.X, pady=(0, _s(4)))
                    self.bind_drop_targets(row_f)
                p_file = item["file"]
                p_type = item["type"]
                p_name = os.path.basename(p_file)
                tag_txt = "TXT" if p_type == "script" else "AUD"
                tag_fg = ACCENT_HOVER if p_type == "script" else AMBER
                tag_bg = ACCENT_DIM if p_type == "script" else AMBER_DIM

                pill = tk.Frame(
                    row_f, bg=BG_SLOT, highlightbackground=BORDER,
                    highlightthickness=1, cursor="hand2", padx=_s(6), pady=_s(4),
                )
                pill.pack(side=tk.LEFT, padx=(0, _s(6)))
                t_tag = tk.Label(
                    pill, text=tag_txt, font=(MONO, 7, "bold"),
                    fg=tag_fg, bg=tag_bg, padx=_s(4), pady=_s(1),
                )
                t_tag.pack(side=tk.LEFT, padx=(0, _s(6)))
                t_lbl = tk.Label(
                    pill, text=_truncate_filename(p_name, 22),
                    font=(FONT, 8), fg=TEXT_PRI, bg=BG_SLOT,
                )
                t_lbl.pack(side=tk.LEFT)
                ToolTip(pill, p_file)

                def _start_dock_drag(e, u=uid, p=p_file, tp=p_type, n=p_name):
                    self._on_drag_press(e, p_type=tp, p_file=p, p_name=n, cid=None, uid=u)

                for w in (pill, t_tag, t_lbl):
                    w.bind("<ButtonPress-1>", _start_dock_drag)
                    w.bind("<B1-Motion>", self._on_drag_motion)
                    w.bind("<ButtonRelease-1>", self._on_drag_release)
                self.bind_drop_targets(pill, t_tag, t_lbl)
        else:
            self.dock_frame.pack_forget()

        # 2. 清空轨道视口
        for w in self.matrix_inner.winfo_children():
            w.destroy()

        if not self.task_store:
            self._render_empty_stage()
            self._on_queue_changed()
            return

        # 恢复自适应高度（空状态曾显式设满视口高度）
        self.matrix_canvas.itemconfig(self.matrix_win_id, height=0)

        show_b = self.is_persp_visible()
        self._render_column_headers(show_b)
        self.col_header.pack(fill=tk.X, before=self.matrix_scroll_wrap)
        self.col_header_sep.pack(fill=tk.X, before=self.matrix_scroll_wrap)

        # 3. 逐条渲染水平轨道条（DAW Arrangement Strips）
        for row_idx, (cid, task) in enumerate(list(self.task_store.items())):
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
            strip_outer = tk.Frame(self.matrix_inner, bg=row_bg)
            strip_outer.pack(fill=tk.X)

            strip = tk.Frame(strip_outer, bg=row_bg, height=_s(58))
            strip.pack(fill=tk.X)
            strip.pack_propagate(False)

            # 左侧等比状态灯条
            tk.Frame(strip, bg=led_col, width=_s(3)).pack(side=tk.LEFT, fill=tk.Y)

            grid_body = tk.Frame(strip, bg=row_bg, padx=_s(14))
            grid_body.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            self._configure_matrix_grid(grid_body, show_b)

            # 列 0：等宽轨号
            idx_lbl = tk.Label(
                grid_body, text=f"{row_idx + 1:02d}", font=(MONO, 9, "bold"),
                fg=led_col if st == "running" else TEXT_MUTED, bg=row_bg, anchor="w",
            )
            idx_lbl.grid(row=0, column=0, sticky="ew", padx=_s(4))

            # 列 1：轨道名 + 状态微胶囊
            meta_box = tk.Frame(grid_body, bg=row_bg)
            meta_box.grid(row=0, column=1, sticky="ew", padx=_s(4))
            t_name = task.get("name") or _etr("untitled_track")
            name_lbl = tk.Label(
                meta_box, text=_truncate_filename(t_name, 18 if show_b else 24),
                font=(FONT, 9, "bold"), fg=TEXT_PRI, bg=row_bg, anchor="w",
            )
            name_lbl.pack(anchor="w", fill=tk.X)
            ToolTip(name_lbl, t_name)
            st_pill = tk.Label(
                meta_box, text=badge_txt, font=(FONT, 7, "bold"),
                fg=badge_fg, bg=badge_bg, padx=_s(6), pady=_s(1),
            )
            st_pill.pack(anchor="w", pady=(_s(3), 0))

            # 列 2：台本槽位
            s_chip = self._make_slot_chip(
                grid_body, cid, "script", task.get("script"), "TXT",
                tr("slot_script_empty"), ACCENT_HOVER, ACCENT_DIM, show_b,
            )
            s_chip.grid(row=0, column=2, sticky="ew", padx=_s(4))

            # 列 3：主音频槽位
            a_chip = self._make_slot_chip(
                grid_body, cid, "audio", task.get("audio"), "A" if show_b else "AUD",
                tr("slot_audio_empty"), EMERALD if not show_b else ACCENT_HOVER,
                EMERALD_DIM if not show_b else ACCENT_DIM, show_b,
            )
            a_chip.grid(row=0, column=3, sticky="ew", padx=_s(4))

            # 列 4 & 5：双视角交换按钮 + 副音频槽位
            if show_b:
                def _swap_ab(_e, c=cid):
                    if self.ctx.is_running():
                        return
                    t = self.task_store.get(c)
                    if t:
                        t["audio"], t["audio_b"] = t.get("audio_b"), t.get("audio")
                        self.render_all_cards()

                swap_btn = tk.Label(
                    grid_body, text="⇄", font=(MONO, 10, "bold"),
                    fg=AMBER, bg=AMBER_DIM, padx=_s(5), pady=_s(4), cursor="hand2",
                )
                swap_btn.grid(row=0, column=4, padx=_s(2))
                swap_btn.bind("<Button-1>", _swap_ab)
                ToolTip(swap_btn, tr("swap_tooltip"))

                b_chip = self._make_slot_chip(
                    grid_body, cid, "audio_b", task.get("audio_b"), "B",
                    tr("slot_audio_b_empty"), AMBER, AMBER_DIM, show_b,
                )
                b_chip.grid(row=0, column=5, sticky="ew", padx=_s(4))

            # 列 6：内联快捷操作区（打开 / 重试 / 删除）
            act_box = tk.Frame(grid_body, bg=row_bg)
            act_box.grid(row=0, column=6, sticky="e", padx=(_s(4), 0))

            def _make_micro_btn(symbol, tip_key, hover_fg, cmd):
                b = tk.Label(
                    act_box, text=symbol, font=(MONO, 10, "bold"),
                    fg=TEXT_MUTED, bg=row_bg, padx=_s(5), pady=_s(3), cursor="hand2",
                )
                b.pack(side=tk.LEFT, padx=_s(1))
                b.bind("<Button-1>", lambda _e: cmd())
                b.bind("<Enter>", lambda _e: b.configure(fg=hover_fg, bg=BG_SLOT))
                b.bind("<Leave>", lambda _e: b.configure(fg=TEXT_MUTED, bg=row_bg))
                ToolTip(b, _etr(tip_key))
                return b

            def _remove_track(c=cid):
                if self.ctx.is_running():
                    return
                self.task_store.pop(c, None)
                self.render_all_cards()

            def _requeue_track(c=cid):
                if self.ctx.is_running():
                    return
                t = self.task_store.get(c)
                if t:
                    t["status"] = "pending"
                    self.render_all_cards()
                    self._update_status(tr("status_requeued", name=t["name"]))

            _make_micro_btn("↗", "tip_open", ACCENT_HOVER, lambda c=cid: self._on_card_double_click(c))
            _make_micro_btn("↻", "tip_retry", AMBER, _requeue_track)
            _make_micro_btn("×", "tip_remove", CRIMSON, _remove_track)

            # 行级双击、右键菜单与外部拖放
            for w in (strip, grid_body, idx_lbl, meta_box, name_lbl, st_pill):
                w.bind("<Button-3>", lambda e, c=cid: self._show_card_context_menu(e, c))
                w.bind("<Double-1>", lambda _e, c=cid: self._on_card_double_click(c))
            self.bind_drop_targets(strip_outer, strip, grid_body, idx_lbl, meta_box, name_lbl, st_pill, act_box)

            # 行底微弱分隔线
            tk.Frame(strip_outer, bg=BORDER_SUBTLE, height=1).pack(fill=tk.X)

        # 底部轻量新建轨道投放条
        new_bar = tk.Frame(
            self.matrix_inner, bg=BG_SLOT_EMPTY, highlightbackground=BORDER,
            highlightthickness=1, padx=_s(14), pady=_s(10), cursor="hand2",
        )
        new_bar.pack(fill=tk.X, padx=_s(16), pady=_s(12))
        new_lbl = tk.Label(new_bar, text=_etr("new_track_drop"), font=(FONT, 8), fg=TEXT_MUTED, bg=BG_SLOT_EMPTY)
        new_lbl.pack()
        for w in (new_bar, new_lbl):
            w.bind("<Button-1>", self.browse_files)
            w.bind("<Enter>", lambda _e: (new_bar.configure(highlightbackground=ACCENT), new_lbl.configure(fg=ACCENT_HOVER)))
            w.bind("<Leave>", lambda _e: (new_bar.configure(highlightbackground=BORDER), new_lbl.configure(fg=TEXT_MUTED)))
        self.bind_drop_targets(new_bar, new_lbl)
        self._slot_registry.append({
            "widget": new_bar, "type": "new_card", "cid": None, "uid": None,
            "rest_border": BORDER, "rest_thick": 1, "in_matrix": True,
        })

        self._on_queue_changed()

    # ── 内部拖拽交互手势（槽位互换 / 拖入暂存坞 / 拖建新轨）──────────────

    def _on_drag_press(self, event, p_type: str, p_file: str, p_name: str, cid, uid) -> None:
        if self.ctx.is_running():
            return
        self._drag_info.update(
            active=False, type=p_type, file=p_file, name=p_name,
            cid=cid, uid=uid, start_x=event.x_root, start_y=event.y_root,
        )

    def _on_drag_motion(self, event) -> None:
        if not self._drag_info["file"]:
            return
        drag_thresh = _s(4)
        if not self._drag_info["active"] and (
            abs(event.x_root - self._drag_info["start_x"]) > drag_thresh
            or abs(event.y_root - self._drag_info["start_y"]) > drag_thresh
        ):
            self._drag_info["active"] = True
            win = tk.Toplevel(self.root)
            win.wm_overrideredirect(True)
            win.attributes("-topmost", True)
            try:
                win.attributes("-alpha", 0.92)
            except Exception:
                pass
            tag = "TXT" if self._drag_info["type"] == "script" else "AUD"
            col = ACCENT_HOVER if tag == "TXT" else AMBER
            box = tk.Frame(
                win, bg="#0F172A", highlightbackground=col, highlightthickness=1,
                padx=_s(10), pady=_s(5),
            )
            box.pack()
            tk.Label(box, text=tag, font=(MONO, 7, "bold"), fg=INK_DARK, bg=col, padx=_s(4), pady=_s(1)).pack(
                side=tk.LEFT, padx=(0, _s(6))
            )
            tk.Label(
                box, text=_truncate_filename(self._drag_info["name"], 24),
                font=(FONT, 8, "bold"), fg=TEXT_PRI, bg="#0F172A",
            ).pack(side=tk.LEFT)
            self._drag_info["win"] = win

        if self._drag_info["active"] and self._drag_info["win"]:
            off = _s(14)
            self._drag_info["win"].geometry(f"+{event.x_root + off}+{event.y_root + off}")
            self._update_drag_hover(event.x_root, event.y_root)

    @staticmethod
    def _is_slot_compatible(src_type: str, tgt_type: str) -> bool:
        if tgt_type in ("tray", "new_card"):
            return True
        is_src_aud = src_type in ("audio", "audio_b")
        is_tgt_aud = tgt_type in ("audio", "audio_b")
        return (src_type == "script" and tgt_type == "script") or (is_src_aud and is_tgt_aud)

    @staticmethod
    def _restore_slot_border(slot: dict) -> None:
        try:
            slot["widget"].configure(
                highlightbackground=slot.get("rest_border", BORDER),
                highlightthickness=slot.get("rest_thick", 1),
            )
        except Exception:
            pass

    def _update_drag_hover(self, x: int, y: int) -> None:
        matched = None
        cv_x0 = self.matrix_canvas.winfo_rootx()
        cv_y0 = self.matrix_canvas.winfo_rooty()
        cv_x1 = cv_x0 + self.matrix_canvas.winfo_width()
        cv_y1 = cv_y0 + self.matrix_canvas.winfo_height()
        in_matrix_view = (cv_x0 <= x <= cv_x1 and cv_y0 <= y <= cv_y1)

        for item in self._slot_registry:
            if item.get("in_matrix") and not in_matrix_view:
                continue
            w = item["widget"]
            if not w.winfo_exists() or not w.winfo_ismapped():
                continue
            rx, ry = w.winfo_rootx(), w.winfo_rooty()
            if rx <= x <= rx + w.winfo_width() and ry <= y <= ry + w.winfo_height():
                matched = item
                break

        if matched != self._highlighted_slot:
            if self._highlighted_slot:
                self._restore_slot_border(self._highlighted_slot)
            self._highlighted_slot = matched
            if self._highlighted_slot:
                ok = self._is_slot_compatible(self._drag_info["type"], self._highlighted_slot["type"])
                try:
                    self._highlighted_slot["widget"].configure(
                        highlightbackground=ACCENT_HOVER if ok else CRIMSON,
                        highlightthickness=2,
                    )
                except Exception:
                    pass

    def _on_drag_release(self, _event) -> None:
        if self._drag_info["win"]:
            try:
                self._drag_info["win"].destroy()
            except Exception:
                pass
            self._drag_info["win"] = None

        if self._highlighted_slot:
            self._restore_slot_border(self._highlighted_slot)

        if not self._drag_info["active"]:
            self._drag_info["file"] = None
            return

        target = self._highlighted_slot
        self._highlighted_slot = None
        self._drag_info["active"] = False
        src_type = self._drag_info["type"]
        src_file = self._drag_info["file"]
        src_cid = self._drag_info["cid"]
        src_uid = self._drag_info["uid"]
        self._drag_info["file"] = None

        if not target:
            return

        tgt_type = target["type"]
        tgt_cid = target["cid"]

        # 1. 放回未配对暂存坞
        if tgt_type == "tray":
            if src_cid is not None:
                card = self.task_store.get(src_cid)
                if card:
                    card[src_type] = None
                    self._uid_counter += 1
                    u = f"u_{self._uid_counter}"
                    self._unpaired_pool[u] = {
                        "uid": u,
                        "type": "script" if src_type == "script" else "audio",
                        "file": src_file,
                        "name": os.path.basename(src_file),
                    }
                    if not card.get("script") and not card.get("audio") and not card.get("audio_b"):
                        self.task_store.pop(src_cid, None)
            self.render_all_cards()
            return

        # 2. 放入新建轨道区
        if tgt_type == "new_card":
            self._card_id_counter += 1
            cid = f"card_{self._card_id_counter}"
            is_src_aud = src_type in ("audio", "audio_b")
            raw_stem = os.path.splitext(os.path.basename(src_file))[0]
            clean_name = raw_stem
            if is_src_aud and hasattr(self.ctx, "_extract_perspective_tag"):
                base_k, _, _ = self.ctx._extract_perspective_tag(raw_stem)
                if base_k:
                    clean_name = base_k
            self.task_store[cid] = {
                "cid": cid, "iid": cid, "name": clean_name,
                "script": None if is_src_aud else src_file,
                "audio": src_file if is_src_aud else None,
                "audio_b": None, "status": "pending", "confidence": 100,
            }
            if src_cid is not None:
                src_card = self.task_store.get(src_cid)
                if src_card:
                    src_card[src_type] = None
                    if not src_card.get("script") and not src_card.get("audio") and not src_card.get("audio_b"):
                        self.task_store.pop(src_cid, None)
            else:
                self._unpaired_pool.pop(src_uid, None)
            self.render_all_cards()
            return

        # 3. 放入轨道槽位（填入或互换）
        if not self._is_slot_compatible(src_type, tgt_type):
            return
        tgt_card = self.task_store.get(tgt_cid)
        if not tgt_card:
            return
        old_tgt = tgt_card.get(tgt_type)
        if not old_tgt:
            tgt_card[tgt_type] = src_file
            if src_cid is not None:
                src_card = self.task_store.get(src_cid)
                if src_card:
                    src_card[src_type] = None
                    if not src_card.get("script") and not src_card.get("audio") and not src_card.get("audio_b"):
                        self.task_store.pop(src_cid, None)
            else:
                self._unpaired_pool.pop(src_uid, None)
        else:
            if src_cid is not None:
                src_card = self.task_store.get(src_cid)
                if src_card:
                    src_card[src_type] = old_tgt
                    tgt_card[tgt_type] = src_file
            else:
                tgt_card[tgt_type] = src_file
                self._unpaired_pool[src_uid]["file"] = old_tgt
                self._unpaired_pool[src_uid]["name"] = os.path.basename(old_tgt)

        tgt_card["confidence"] = 100
        if src_cid and src_cid in self.task_store:
            self.task_store[src_cid]["confidence"] = 100
        self.render_all_cards()

    # ── 文件导入、自动配对、菜单与外部 DND ──────────────────────────────

    def add_files_to_list(self, files) -> None:
        if not files:
            return
        pool_files = [item["file"] for item in self._unpaired_pool.values()]
        incomplete_files = []
        for t in self.task_store.values():
            if not t.get("script") or not t.get("audio") or (self.mode_var.get() == "multi_persp" and not t.get("audio_b")):
                for k in ("script", "audio", "audio_b"):
                    if t.get(k):
                        incomplete_files.append(t[k])

        new_files = [os.path.abspath(f) for f in files if f and str(f).strip()]
        supported_exts = {".txt", ".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".opus", ".wma"}
        unsupported = []
        missing_inputs = []
        valid_new = []
        for f in new_files:
            if not os.path.exists(f):
                missing_inputs.append(os.path.basename(f) or f)
                continue
            ext = os.path.splitext(f)[1].lower()
            if os.path.isdir(f) or ext not in supported_exts or f.lower().endswith(".dropped.txt"):
                unsupported.append(os.path.basename(f) or f)
            else:
                valid_new.append(f)
        if missing_inputs:
            self._append_log(f"⚠ 忽略不存在的文件 ({len(missing_inputs)}): {', '.join(missing_inputs[:5])}")
        if unsupported:
            self._append_log(f"⚠ 忽略不支持的格式 ({len(unsupported)}): {', '.join(unsupported[:5])}")

        all_candidates = list(dict.fromkeys(pool_files + incomplete_files + valid_new))
        pairs, unmatched = self.ctx.auto_pair_files(all_candidates, mode=self.mode_var.get())
        added = 0

        existing_auds = {p.get("audio") for p in self.task_store.values() if p.get("audio")} | {
            p.get("audio_b") for p in self.task_store.values() if p.get("audio_b")
        }
        existing_txts = {p.get("script") for p in self.task_store.values() if p.get("script")}

        def _is_incomplete(card: dict) -> bool:
            if not card.get("script") or not card.get("audio"):
                return True
            if self.mode_var.get() == "multi_persp" and not card.get("audio_b"):
                return True
            return False

        # 若新配对覆盖了已有残缺轨道的台本或音频，原地补齐（并合并/清理冗余残缺项），杜绝幽灵重复轨道
        for pair in list(pairs):
            p_s = pair.get("script")
            p_a = pair.get("audio")
            p_b = pair.get("audio_b")
            hits = [
                cid for cid, t in list(self.task_store.items())
                if _is_incomplete(t) and (
                    (p_s and t.get("script") == p_s)
                    or (p_a and t.get("audio") == p_a)
                    or (p_b and t.get("audio_b") == p_b)
                )
            ]
            if not hits:
                continue
            primary = self.task_store[hits[0]]
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
                self.task_store.pop(dup_cid, None)
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
            self._card_id_counter += 1
            cid = f"card_{self._card_id_counter}"
            self.task_store[cid] = {
                "cid": cid, "iid": cid, "name": pair["name"],
                "script": pair.get("script"), "audio": pair.get("audio"),
                "audio_b": pair.get("audio_b"), "status": "pending",
                "confidence": pair.get("confidence", 100),
            }
            for k, s_set in (("audio", existing_auds), ("audio_b", existing_auds), ("script", existing_txts)):
                if pair.get(k):
                    s_set.add(pair[k])
            added += 1

        self._unpaired_pool.clear()
        for item in unmatched:
            f_path = item["file"]
            is_txt = item["type"] == "缺音频"
            if (is_txt and f_path in existing_txts) or (not is_txt and f_path in existing_auds):
                continue
            self._uid_counter += 1
            uid = f"u_{self._uid_counter}"
            self._unpaired_pool[uid] = {
                "uid": uid, "type": "script" if is_txt else "audio",
                "file": f_path, "name": item["name"],
            }

        self.render_all_cards()
        if added:
            self._set_led(ACCENT_HOVER)
            self._update_status(tr("status_added_files", added=added, total=len(self.task_store)))
        elif (unsupported or missing_inputs) and not valid_new:
            self._set_led(AMBER)
            self._ensure_console_open()
            self._update_status(f"未导入：发现 {len(unsupported) + len(missing_inputs)} 个不支持或不存在的文件")
        else:
            self._update_status(tr("status_files_already"))

    def auto_assemble_from_tray(self) -> None:
        if self.ctx.is_running():
            return
        if self._unpaired_pool:
            self.add_files_to_list([item["file"] for item in list(self._unpaired_pool.values())])

    def browse_files(self, _event=None) -> None:
        if self.ctx.is_running():
            return
        files = filedialog.askopenfilenames(
            title=tr("dlg_select_files_title"),
            filetypes=[
                (tr("dlg_filter_supported"), "*.txt;*.wav;*.mp3;*.flac;*.m4a;*.ogg;*.aac;*.opus;*.wma"),
                (tr("dlg_filter_all"), "*.*"),
            ],
        )
        if files:
            self.add_files_to_list(list(files))

    def on_external_drop(self, event) -> None:
        if self.ctx.is_running():
            return
        paths = []
        try:
            for item in self.root.tk.splitlist(event.data):
                path = str(item).strip().strip('"').strip("'")
                if path:
                    paths.append(os.path.abspath(path))
        except Exception as e:
            self._append_log(f"⚠ 拖拽路径解析异常 ({type(e).__name__}: {e})，已回退为空格切分（含空格路径可能受影响）")
            paths = [os.path.abspath(p) for p in event.data.split() if p.strip()]
        self.add_files_to_list(paths)

    def _show_card_context_menu(self, event, cid: str) -> None:
        item = self.task_store.get(cid)
        if not item:
            return
        running = self.ctx.is_running()
        mut_state = tk.DISABLED if running else tk.NORMAL
        menu = tk.Menu(
            self.root, tearoff=0, bg=BG_PANEL, fg=TEXT_PRI,
            activebackground=ACCENT_DIM, activeforeground=ACCENT_HOVER,
            disabledforeground=TEXT_MUTED, bd=1, relief=tk.SOLID, font=(FONT, 9),
        )

        def _open_file(path):
            if not path or not os.path.exists(path):
                messagebox.showwarning(
                    tr("msg_file_missing_title"),
                    tr("msg_file_missing_text", path=path or tr("msg_path_unspecified")),
                    parent=self.root,
                )
                return
            try:
                os.startfile(os.path.normpath(path))
            except Exception as e:
                self._append_log(f"打开文件失败 ({path}): {type(e).__name__}: {e}")
                messagebox.showerror(
                    tr("msg_open_failed_title"),
                    tr("msg_open_file_failed_text", path=path, error=f"{type(e).__name__}: {e}"),
                    parent=self.root,
                )

        def _open_folder():
            target = item.get("audio") or item.get("script")
            if not target or not os.path.exists(target):
                messagebox.showwarning(
                    tr("msg_path_missing_title"),
                    tr("msg_file_missing_text", path=target or tr("msg_path_unspecified")),
                    parent=self.root,
                )
                return
            try:
                subprocess.run(["explorer", "/select,", os.path.normpath(target)], check=False)
            except Exception as e:
                self._append_log(f"打开文件夹失败 ({target}): {type(e).__name__}: {e}")
                messagebox.showerror(
                    tr("msg_open_failed_title"),
                    tr("msg_open_folder_failed_text", path=target, error=f"{type(e).__name__}: {e}"),
                    parent=self.root,
                )

        def _view_output(ext: str):
            audio = item.get("audio")
            if not audio:
                return
            out_path = os.path.splitext(audio)[0] + ext
            if os.path.exists(out_path):
                _open_file(out_path)
            else:
                messagebox.showinfo(
                    tr("msg_no_subtitle_title"),
                    tr("msg_no_subtitle_text", extension=ext), parent=self.root,
                )

        def _retry_this():
            if self.ctx.is_running():
                return
            item["status"] = "pending"
            self.render_all_cards()
            self._update_status(tr("status_requeued", name=item["name"]))

        def _remove_this():
            if self.ctx.is_running():
                return
            self.task_store.pop(cid, None)
            self.render_all_cards()

        menu.add_command(label=tr("menu_open_folder"), command=_open_folder)
        menu.add_command(label=tr("menu_open_script"), command=lambda: _open_file(item.get("script")))
        menu.add_command(label=tr("menu_view_lrc"), command=lambda: _view_output(".lrc"))
        menu.add_command(label=tr("menu_view_srt"), command=lambda: _view_output(".srt"))
        menu.add_separator()
        if self.is_persp_visible():
            def _swap():
                if self.ctx.is_running():
                    return
                item["audio"], item["audio_b"] = item.get("audio_b"), item.get("audio")
                self.render_all_cards()
            menu.add_command(label=tr("menu_swap_audios"), command=_swap, state=mut_state)
            menu.add_separator()
        menu.add_command(label=tr("menu_retry"), command=_retry_this, state=mut_state)
        menu.add_command(label=tr("btn_remove_card"), command=_remove_this, state=mut_state)
        menu.post(event.x_root, event.y_root)

    def _on_card_double_click(self, cid: str) -> None:
        item = self.task_store.get(cid)
        if not item:
            return
        audio = item.get("audio")
        if audio:
            for ext in (".srt", ".lrc"):
                out_path = os.path.splitext(audio)[0] + ext
                if os.path.exists(out_path):
                    try:
                        os.startfile(os.path.normpath(out_path))
                    except Exception as e:
                        self._append_log(f"打开字幕失败 ({out_path}): {type(e).__name__}: {e}")
                        messagebox.showerror(
                            tr("msg_open_failed_title"),
                            tr("msg_open_subtitle_failed_text", path=out_path, error=f"{type(e).__name__}: {e}"),
                            parent=self.root,
                        )
                    return
        target = audio or item.get("script")
        if target and os.path.exists(target):
            try:
                subprocess.run(["explorer", "/select,", os.path.normpath(target)], check=False)
            except Exception as e:
                self._append_log(f"打开文件夹失败 ({target}): {type(e).__name__}: {e}")
                messagebox.showerror(
                    tr("msg_open_failed_title"),
                    tr("msg_locate_file_failed_text", path=target, error=f"{type(e).__name__}: {e}"),
                    parent=self.root,
                )
        else:
            messagebox.showwarning(
                tr("msg_file_missing_title"),
                tr("msg_file_missing_text", path=target or tr("msg_path_unspecified")),
                parent=self.root,
            )
