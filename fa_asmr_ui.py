# -*- coding: utf-8 -*-
"""FA-ASMR 桌面界面。与强制对齐管线解耦，仅负责交互与状态展示。
支持卡片式流式对齐工作台、界面内直接拖拽互换/调配、多语言即时热切换。
"""
from __future__ import annotations

import os
import queue
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from fa_asmr_i18n import (
    I18N,
    tr,
    LANG_NAMES,
    LANG_MAP,
    detect_system_lang,
)

BG = "#f5f2f0"
PANEL = "#fffdfc"
SURFACE = "#f2eeee"
SURFACE_HOVER = "#eae3e5"
SURFACE_ACTIVE = "#e4dade"
TEXT = "#2d282c"
MUTED = "#756d74"
SUBTLE = "#9a9197"
BORDER = "#dfd7da"
BORDER_DASHED = "#e2b8c9"
PINK = "#b9587f"
PINK_HOVER = "#a94770"
PINK_DARK = "#f5e2e9"
PINK_BG = "#fff6f9"
LAVENDER = "#8875aa"
MINT = "#397d64"
AMBER = "#97651f"
RED = "#bd4f61"
INK = "#fffafc"
FONT = "Microsoft YaHei UI"
MONO = "Consolas"

CHUNK_KEYS = ["chunk_standard", "chunk_light", "chunk_fast", "chunk_full"]
CHUNK_VALUES = ["6000", "3000", "12000", "999999"]


def get_chunk_presets() -> list[str]:
    return [tr(k) for k in CHUNK_KEYS]


def get_status_icon(status_key: str) -> str:
    key_map = {
        "pending": "status_pending",
        "running": "status_running",
        "done": "status_done",
        "failed": "status_failed",
        "warn": "status_warn",
    }
    return tr(key_map.get(status_key, "status_pending"))


class ToolTip:
    """轻量悬停提示。"""

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.window = None
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _show(self, _event=None):
        if self.window or not self.text:
            return
        self.window = tk.Toplevel(self.widget)
        self.window.wm_overrideredirect(True)
        self.window.wm_geometry(f"+{self.widget.winfo_rootx()}+{self.widget.winfo_rooty() + 28}")
        tk.Label(self.window, text=self.text, bg=SURFACE_ACTIVE, fg=TEXT,
                 font=(FONT, 8), padx=10, pady=6, justify=tk.LEFT).pack()

    def _hide(self, _event=None):
        if self.window:
            self.window.destroy()
            self.window = None


def show_script_help_dialog(parent):
    """显示台本格式与规范指南弹窗。"""
    dlg = tk.Toplevel(parent)
    dlg.title(tr("guide_dlg_window_title"))
    dlg.geometry("680x600")
    dlg.minsize(560, 480)
    dlg.configure(bg=PANEL)
    dlg.transient(parent)
    dlg.grab_set()
    dlg.update_idletasks()
    dlg.geometry(f"+{parent.winfo_x() + max(0, (parent.winfo_width() - 680) // 2)}"
                 f"+{parent.winfo_y() + max(0, (parent.winfo_height() - 600) // 2)}")

    pad = tk.Frame(dlg, bg=PANEL)
    pad.pack(fill=tk.BOTH, expand=True, padx=26, pady=22)
    tk.Label(pad, text=tr("guide_dlg_heading"), font=(FONT, 16, "bold"),
             fg=TEXT, bg=PANEL).pack(anchor="w")
    tk.Label(pad, text=tr("guide_dlg_subheading"),
             font=(FONT, 9), fg=MUTED, bg=PANEL).pack(anchor="w", pady=(4, 16))
    txt_box = tk.Text(pad, bg="#fffdfc", fg="#1f1a1e", font=(FONT, 10), relief=tk.FLAT,
                      highlightbackground=BORDER, highlightthickness=1,
                      wrap=tk.WORD, padx=16, pady=14, spacing1=3, spacing3=3)
    txt_box.pack(fill=tk.BOTH, expand=True, pady=(0, 14))
    sample_text = I18N.get_guide_text()
    txt_box.insert(tk.END, sample_text)
    txt_box.configure(state=tk.DISABLED)
    row = tk.Frame(pad, bg=PANEL)
    row.pack(fill=tk.X)

    def _copy_sample():
        parent.clipboard_clear()
        parent.clipboard_append(sample_text)
        cp_btn.config(text=tr("guide_btn_copied"), fg=MINT)
        parent.after(1800, lambda: cp_btn.config(text=tr("guide_btn_copy"), fg=TEXT))

    cp_btn = tk.Button(row, text=tr("guide_btn_copy"), font=(FONT, 9), fg=TEXT, bg=SURFACE,
                       activebackground=SURFACE_HOVER, activeforeground=TEXT,
                       relief=tk.FLAT, padx=16, pady=8, cursor="hand2", command=_copy_sample)
    cp_btn.pack(side=tk.LEFT)
    tk.Button(row, text=tr("guide_btn_done"), font=(FONT, 9, "bold"), fg=INK, bg=PINK,
              activebackground=PINK_HOVER, activeforeground=INK, relief=tk.FLAT,
              padx=22, pady=8, cursor="hand2", command=dlg.destroy).pack(side=tk.RIGHT)


def build_ui(ctx):
    """创建完整工作台。ctx 提供业务回调与运行环境信息。"""
    root = ctx.root
    root.title(tr("app_title"))
    root.geometry("1180x840")
    root.minsize(940, 660)
    root.configure(bg=BG)
    if ctx.pil:
        try:
            from PIL import Image, ImageTk
            icon_path = os.path.join(ctx.script_dir, "assets", "mascot_icon.png")
            root._icon_photo = ImageTk.PhotoImage(Image.open(icon_path).convert("RGBA"))
            root.wm_iconphoto(True, root._icon_photo)
        except Exception:
            pass

    def _label(parent, text, size=9, color=TEXT, bold=False, **kwargs):
        return tk.Label(parent, text=text, font=(FONT, size, "bold" if bold else "normal"),
                        fg=color, bg=parent.cget("bg"), **kwargs)

    def _frame(parent, color=PANEL, border=False, **kwargs):
        return tk.Frame(parent, bg=color, highlightbackground=BORDER if border else color,
                        highlightthickness=1 if border else 0, **kwargs)

    def _button(parent, text, command, primary=False, compact=False):
        btn = tk.Button(parent, text=text, command=command,
                        font=(FONT, 9, "bold" if primary else "normal"),
                        fg=INK if primary else TEXT,
                        bg=PINK if primary else SURFACE,
                        activebackground=PINK_HOVER if primary else SURFACE_HOVER,
                        activeforeground=INK if primary else TEXT,
                        disabledforeground=MUTED, relief=tk.FLAT, bd=0,
                        padx=14 if compact else 18, pady=7 if compact else 10,
                        cursor="hand2")
        return btn

    # 全局控件样式。
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TProgressbar", thickness=4, borderwidth=0,
                    troughcolor=SURFACE, background=PINK)
    style.configure("TScrollbar", background=SURFACE, troughcolor=PANEL,
                    borderwidth=0, arrowsize=10, relief=tk.FLAT)
    style.map("TScrollbar", background=[("active", SURFACE_ACTIVE), ("pressed", PINK)])
    style.configure("TCombobox", fieldbackground=SURFACE, background=SURFACE,
                    foreground=TEXT, arrowcolor=MUTED, borderwidth=0,
                    relief=tk.FLAT, padding=6)
    style.map("TCombobox", fieldbackground=[("readonly", SURFACE)],
              selectbackground=[("readonly", SURFACE)], selectforeground=[("readonly", TEXT)],
              foreground=[("readonly", TEXT), ("disabled", MUTED)])
    root.option_add("*TCombobox*Listbox.background", SURFACE)
    root.option_add("*TCombobox*Listbox.foreground", TEXT)
    root.option_add("*TCombobox*Listbox.selectBackground", PINK_DARK)
    root.option_add("*TCombobox*Listbox.selectForeground", PINK)
    root.option_add("*TCombobox*Listbox.relief", "flat")
    root.option_add("*TCombobox*Listbox.borderWidth", "0")

    # 后台工作线程队列调度
    ui_events = queue.Queue()
    closing = False

    def _post_ui(callback):
        ui_events.put(callback)

    def _drain_ui_events():
        while True:
            try:
                callback = ui_events.get_nowait()
            except queue.Empty:
                break
            try:
                callback()
            except tk.TclError:
                if not closing:
                    raise
        if not closing:
            root.after(30, _drain_ui_events)
    root.after(30, _drain_ui_events)

    main = tk.Frame(root, bg=BG)
    main.pack(fill=tk.BOTH, expand=True, padx=24, pady=18)

    # 顶部夜樱横幅
    hero = _frame(main, "#1a1820", border=True, height=142)
    hero.pack(fill=tk.X, pady=(0, 18))
    hero.pack_propagate(False)
    hero_canvas = tk.Canvas(hero, bg="#f5e9ee", bd=0, highlightthickness=0)
    hero_canvas.pack(fill=tk.BOTH, expand=True)
    hero_image_id = hero_canvas.create_image(0, 0, anchor="nw")
    hero_left_id = hero_canvas.create_text(30, 34, anchor="nw", fill="#984467",
                                           text=tr("banner_badge"),
                                           font=(FONT, 8, "bold"))
    hero_title_id = hero_canvas.create_text(28, 57, anchor="nw", fill="#362831",
                                            text=tr("banner_title"),
                                            font=(FONT, 20, "bold"))
    hero_sub_id = hero_canvas.create_text(30, 103, anchor="nw", fill="#66535e",
                                          text=tr("banner_subtitle"),
                                          font=(FONT, 9))
    hero_device_id = hero_canvas.create_text(0, 24, anchor="ne",
                                             fill="#325e4e" if ctx.cuda_ok else "#66535e",
                                             text=tr("device_cuda") if ctx.cuda_ok else tr("device_cpu"),
                                             font=(FONT, 9, "bold"))
    root._hero_photo = None

    def _draw_hero(event=None):
        width = max(hero.winfo_width(), 400)
        height = max(hero.winfo_height(), 140)
        hero_canvas.coords(hero_device_id, width - 24, 24)
        if ctx.pil:
            try:
                from PIL import Image, ImageEnhance, ImageTk
                path = os.path.join(ctx.script_dir, "assets", "asmr_banner.jpg")
                im = Image.open(path).convert("RGB")
                scale = max(width / im.width, height / im.height)
                resized = im.resize((int(im.width * scale) + 1, int(im.height * scale) + 1))
                left = max(0, (resized.width - width) // 2)
                top = max(0, int((resized.height - height) * 0.28))
                im = resized.crop((left, top, left + width, top + height))
                im = ImageEnhance.Brightness(im).enhance(1.10)
                wash = Image.new("RGB", im.size, "#fff8fa")
                im = Image.blend(im, wash, 0.28).convert("RGBA")
                mask = Image.new("L", im.size)
                pixels = mask.load()
                fade_end = int(width * 0.76)
                for x in range(width):
                    alpha = 242 if x < width * 0.48 else int(242 * max(0, (fade_end - x) / (fade_end - width * 0.48)))
                    for y in range(height):
                        pixels[x, y] = alpha
                veil = Image.new("RGBA", im.size, "#fff8fa")
                im = Image.composite(veil, im, mask).convert("RGB")
                root._hero_photo = ImageTk.PhotoImage(im)
                hero_canvas.itemconfigure(hero_image_id, image=root._hero_photo)
            except Exception:
                pass
    hero_canvas.bind("<Configure>", _draw_hero)

    lang_var = tk.StringVar(value=I18N.current_lang)
    lang_display_var = tk.StringVar(value=LANG_MAP.get(I18N.current_lang, "简体中文"))
    mode_var = tk.StringVar(value="single")
    dev_var = tk.StringVar(value="cuda" if ctx.cuda_ok else "cpu")
    vram_chunk_var = tk.StringVar(value="6000")
    chunk_preset_var = tk.StringVar(value=get_chunk_presets()[0])
    end_extend_var = tk.StringVar(value="0.8")
    ono_gate_var = tk.BooleanVar(value=True)
    multi_order_var = tk.BooleanVar(value=False)
    multi_persp_var = tk.BooleanVar(value=False)

    _var_meta = [
        (lang_var, "lang", str),
        (mode_var, "align_mode", str),
        (dev_var, "dev", str),
        (vram_chunk_var, "vram_chunk", str),
        (end_extend_var, "end_extend", str),
        (ono_gate_var, "ono_gate", lambda v: v in (True, "true", "True", 1, "1")),
        (multi_order_var, "multi_order", lambda v: v in (True, "true", "True", 1, "1")),
        (multi_persp_var, "multi_persp", lambda v: v in (True, "true", "True", 1, "1")),
    ]

    def _sync_chunk_preset():
        saved_chunk = vram_chunk_var.get()
        presets = get_chunk_presets()
        idx = CHUNK_VALUES.index(saved_chunk) if saved_chunk in CHUNK_VALUES else 0
        chunk_preset_var.set(presets[idx])

    def _save_settings():
        try:
            current = mode_var.get()
            multi_order_var.set(current in ("multi_speaker", "multi_persp"))
            multi_persp_var.set(current == "multi_persp")
            import json
            with open(ctx.settings_file, "w", encoding="utf-8") as file:
                json.dump({key: var.get() for var, key, _ in _var_meta}, file,
                          ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _load_settings():
        if not os.path.isfile(ctx.settings_file):
            initial_lang = detect_system_lang()
            I18N.set_lang(initial_lang)
            lang_var.set(initial_lang)
            lang_display_var.set(LANG_MAP.get(initial_lang, "简体中文"))
            return
        try:
            import json
            with open(ctx.settings_file, "r", encoding="utf-8") as file:
                data = json.load(file)
            for var, key, convert in _var_meta:
                if key in data:
                    var.set(convert(data[key]))
            if not ctx.cuda_ok:
                dev_var.set("cpu")
            if "lang" in data and data["lang"] in LANG_MAP:
                cur_lang = data["lang"]
            else:
                cur_lang = detect_system_lang()
            I18N.set_lang(cur_lang)
            lang_var.set(cur_lang)
            lang_display_var.set(LANG_MAP.get(cur_lang, "简体中文"))

            if "align_mode" not in data:
                mode_var.set("multi_persp" if data.get("multi_persp") else
                             "multi_speaker" if data.get("multi_order") else "single")
            _sync_chunk_preset()
        except Exception:
            pass
    _load_settings()

    # 三模式选择栏
    mode_bar = _frame(main, BG)
    mode_bar.pack(fill=tk.X, pady=(0, 10))
    mode_title_lbl = _label(mode_bar, tr("mode_title_label"), 10, TEXT, True)
    mode_title_lbl.pack(anchor="w", pady=(0, 8))
    mode_row = _frame(mode_bar, BG)
    mode_row.pack(fill=tk.X)
    mode_buttons = {}

    def _on_mode_change():
        current = mode_var.get()
        for key, button in mode_buttons.items():
            selected = key == current
            button.configure(bg=PINK_DARK if selected else PANEL,
                             fg=PINK if selected else MUTED,
                             activebackground=PINK_DARK if selected else SURFACE,
                             activeforeground=PINK if selected else TEXT)
        mode_desc_var.set(tr(f"mode_{current}_desc"))
        _update_drop_zone_text()
        _render_all_cards()
        _save_settings()

    for index, key in enumerate(("single", "multi_speaker", "multi_persp")):
        btn = tk.Button(mode_row, text=f"0{index + 1}   {tr(f'mode_{key}_title')}", anchor="w",
                        font=(FONT, 10, "bold"), relief=tk.FLAT, bd=0,
                        padx=18, pady=13, cursor="hand2",
                        command=lambda k=key: (mode_var.set(k), _on_mode_change()))
        btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0 if index == 0 else 2, 0))
        mode_buttons[key] = btn

    mode_desc_var = tk.StringVar(value=tr(f"mode_{mode_var.get()}_desc"))
    tk.Label(mode_bar, textvariable=mode_desc_var, font=(FONT, 8), fg=MUTED,
             bg=BG, anchor="w").pack(fill=tk.X, pady=(8, 0))

    # 工作区分栏
    workspace = _frame(main, BG)
    workspace.pack(fill=tk.BOTH, expand=True)
    workspace.grid_columnconfigure(0, weight=1, minsize=520)
    workspace.grid_columnconfigure(1, weight=0, minsize=292)
    workspace.grid_rowconfigure(0, weight=1)
    left = _frame(workspace, BG)
    left.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
    right = _frame(workspace, BG)
    right.grid(row=0, column=1, sticky="nsew")

    # 1. 顶部导入框
    drop_frame = _frame(left, PANEL, border=True, height=105)
    drop_frame.pack(fill=tk.X, pady=(0, 10))
    drop_frame.pack_propagate(False)
    drop_inner = _frame(drop_frame, PANEL)
    drop_inner.place(relx=0.015, rely=0.06, relwidth=0.97, relheight=0.88)
    drop_icon = _label(drop_inner, "＋", 22, PINK)
    drop_icon.pack(pady=(2, 0))
    drop_text_var = tk.StringVar()
    drop_label = tk.Label(drop_inner, textvariable=drop_text_var, font=(FONT, 9),
                          fg=TEXT, bg=PANEL, cursor="hand2", justify=tk.CENTER)
    drop_label.pack()
    drop_hint_lbl = _label(drop_inner, tr("drop_hint"), 8, MUTED)
    drop_hint_lbl.pack(pady=(2, 0))

    def _update_drop_zone_text():
        if mode_var.get() == "multi_persp":
            drop_text_var.set(tr("drop_prompt_persp"))
        else:
            drop_text_var.set(tr("drop_prompt_normal"))

    def _drop_hover(active=False):
        color = SURFACE if active else PANEL
        drop_frame.configure(bg=color, highlightbackground=PINK if active else BORDER)
        for widget in (drop_inner, drop_icon, drop_label):
            widget.configure(bg=color)
        for child in drop_inner.winfo_children():
            child.configure(bg=color)
    drop_frame.bind("<Enter>", lambda e: _drop_hover(True))
    drop_frame.bind("<Leave>", lambda e: _drop_hover(False))
    _update_drop_zone_text()

    # 2. 队列操作顶栏
    queue_head = _frame(left, BG)
    queue_head.pack(fill=tk.X, pady=(2, 6))
    queue_title_lbl = _label(queue_head, tr("queue_title"), 10, TEXT, True)
    queue_title_lbl.pack(side=tk.LEFT)
    queue_meta_var = tk.StringVar(value=tr("queue_empty_meta"))
    tk.Label(queue_head, textvariable=queue_meta_var, font=(FONT, 8), fg=MUTED,
             bg=BG).pack(side=tk.LEFT, padx=8)

    clear_btn = _button(queue_head, tr("btn_clear"), lambda: None, compact=True)
    clear_btn.pack(side=tk.RIGHT)
    add_btn = _button(queue_head, "＋ 添加文件", lambda: None, compact=True)
    add_btn.pack(side=tk.RIGHT, padx=(0, 6))

    # 3. 未配对文件托盘（有落单素材时自动展示）
    tray_frame = _frame(left, "#faf6f7", border=True)
    tray_head = _frame(tray_frame, "#faf6f7")
    tray_head.pack(fill=tk.X, padx=12, pady=(8, 4))
    tray_title_lbl = _label(tray_head, tr("unpaired_tray_title"), 8, LAVENDER, bold=True)
    tray_title_lbl.pack(side=tk.LEFT)
    tray_hint_lbl = _label(tray_head, tr("unpaired_tray_hint"), 7, MUTED)
    tray_hint_lbl.pack(side=tk.LEFT, padx=10)
    tray_assemble_btn = _button(tray_head, "⚡ 智能组装为新任务", lambda: _auto_assemble_from_tray(), compact=True)
    tray_assemble_btn.pack(side=tk.RIGHT)
    tray_items_container = _frame(tray_frame, "#faf6f7")
    tray_items_container.pack(fill=tk.X, padx=12, pady=(0, 8))

    # 4. 可视化任务卡片流（支持内部拖拽互换/调配）
    cards_wrapper = _frame(left, PANEL, border=True)
    cards_wrapper.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

    canvas = tk.Canvas(cards_wrapper, bg=PANEL, bd=0, highlightthickness=0)
    cards_scrollbar = ttk.Scrollbar(cards_wrapper, orient=tk.VERTICAL, command=canvas.yview)
    canvas.configure(yscrollcommand=cards_scrollbar.set)
    cards_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    cards_inner = tk.Frame(canvas, bg=PANEL)
    cards_window_id = canvas.create_window((0, 0), window=cards_inner, anchor="nw")

    def _on_canvas_configure(event):
        canvas.itemconfig(cards_window_id, width=event.width)
    canvas.bind("<Configure>", _on_canvas_configure)

    def _on_inner_configure(_event):
        canvas.configure(scrollregion=canvas.bbox("all"))
    cards_inner.bind("<Configure>", _on_inner_configure)

    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    canvas.bind("<MouseWheel>", _on_mousewheel)
    cards_inner.bind("<MouseWheel>", _on_mousewheel)

    # 状态数据结构
    _task_store = {}      # cid -> dict
    _unpaired_pool = {}    # uid -> dict
    _card_id_counter = 0
    _uid_counter = 0
    _slot_registry = []   # list of dict(widget, type, cid, uid)
    _highlighted_slot = None

    # 内部拖拽状态机
    _drag_info = {
        "active": False,
        "type": None,          # "script" | "audio" | "audio_b"
        "file": None,
        "name": None,
        "cid": None,
        "uid": None,
        "win": None,
        "start_x": 0,
        "start_y": 0,
    }

    def _truncate_filename(name: str, max_len=24) -> str:
        if len(name) <= max_len:
            return name
        return name[:max_len - 6] + "…" + name[-5:]

    def _update_queue_meta():
        count = len(_task_store)
        done = sum(1 for item in _task_store.values() if item.get("status") == "done")
        if count == 0:
            queue_meta_var.set(tr("queue_empty_meta"))
        elif done:
            queue_meta_var.set(tr("queue_meta_done_format", count=count, done=done))
        else:
            queue_meta_var.set(tr("queue_meta_format", count=count))

        if not getattr(root, "_fa_running", False):
            start_btn.configure(text=tr("btn_start_with_count", count=count) if count else tr("btn_start"))

    def _render_all_cards():
        """渲染未配对素材托盘与全部对齐任务卡片。"""
        nonlocal _highlighted_slot
        _slot_registry.clear()
        _highlighted_slot = None

        # 1. 渲染待分配素材托盘
        for child in tray_items_container.winfo_children():
            child.destroy()

        if _unpaired_pool:
            tray_frame.pack(fill=tk.X, pady=(0, 8), before=cards_wrapper)
            # 托盘整体注册为投放目标（可拖出卡片放回托盘）
            _slot_registry.append({"widget": tray_frame, "type": "tray", "cid": None, "uid": None})
            for uid, item in list(_unpaired_pool.items()):
                p_file = item["file"]
                p_type = item["type"]
                p_name = os.path.basename(p_file)
                icon = "📄" if p_type == "script" else "🎵"
                fg_c = PINK if p_type == "script" else LAVENDER

                pill = tk.Frame(tray_items_container, bg=SURFACE, highlightbackground=BORDER,
                                highlightthickness=1, cursor="hand2", padx=8, pady=4)
                pill.pack(side=tk.LEFT, padx=(0, 6), pady=2)
                tk.Label(pill, text=f"{icon}  {_truncate_filename(p_name, 22)}", font=(FONT, 8),
                         fg=TEXT, bg=SURFACE).pack(side=tk.LEFT)
                ToolTip(pill, p_file)

                def _start_tray_drag(e, u=uid, p=p_file, t=p_type, n=p_name):
                    _on_drag_press(e, p_type=t, p_file=p, p_name=n, cid=None, uid=u)

                pill.bind("<ButtonPress-1>", _start_tray_drag)
                pill.bind("<B1-Motion>", _on_drag_motion)
                pill.bind("<ButtonRelease-1>", _on_drag_release)
                for c in pill.winfo_children():
                    c.bind("<ButtonPress-1>", _start_tray_drag)
                    c.bind("<B1-Motion>", _on_drag_motion)
                    c.bind("<ButtonRelease-1>", _on_drag_release)
        else:
            tray_frame.pack_forget()

        # 2. 渲染任务卡片
        for child in cards_inner.winfo_children():
            child.destroy()

        if not _task_store:
            empty_box = tk.Frame(cards_inner, bg=PINK_BG, highlightbackground=BORDER_DASHED,
                                 highlightthickness=2, padx=20, pady=40, cursor="hand2")
            empty_box.pack(fill=tk.BOTH, expand=True, padx=4, pady=10)
            _label(empty_box, "📥", 26, PINK).pack(pady=(0, 4))
            _label(empty_box, "拖拽素材至此新建任务卡片", 10, TEXT, bold=True).pack()
            _label(empty_box, "可将上方待分配素材拖至此区域直接创建任务，并继续拖入其余视角素材配对", 8, MUTED).pack(pady=(2, 0))
            _slot_registry.append({"widget": empty_box, "type": "new_card", "cid": None, "uid": None})
            _update_queue_meta()
            return

        is_persp = mode_var.get() == "multi_persp"

        for cid, task in list(_task_store.items()):
            card = tk.Frame(cards_inner, bg=PANEL, highlightbackground=BORDER,
                            highlightthickness=1, padx=12, pady=10)
            card.pack(fill=tk.X, pady=(0, 8), padx=2)
            card.bind("<MouseWheel>", _on_mousewheel)

            # 顶部信息栏
            top_row = tk.Frame(card, bg=PANEL)
            top_row.pack(fill=tk.X, pady=(0, 8))
            top_row.bind("<MouseWheel>", _on_mousewheel)

            t_name = task.get("name") or "未命名任务"
            tk.Label(top_row, text=t_name, font=(FONT, 9, "bold"), fg=TEXT, bg=PANEL).pack(side=tk.LEFT)

            # 徽章：状态 / 置信度
            st = task.get("status", "pending")
            conf = task.get("confidence")
            badge_text = get_status_icon(st)
            badge_fg = MUTED
            badge_bg = SURFACE
            if st == "done":
                badge_fg = MINT
                badge_bg = "#eaf4ef"
            elif st == "running":
                badge_fg = PINK
                badge_bg = PINK_DARK
            elif st == "failed":
                badge_fg = RED
                badge_bg = "#fbecee"
            elif not task.get("script") or not task.get("audio"):
                badge_text = tr("status_warn")
                badge_fg = AMBER
                badge_bg = "#fbf4e8"
            elif conf and conf < 95:
                badge_text = tr("badge_fuzzy", score=conf)
                badge_fg = AMBER
                badge_bg = "#fbf4e8"

            badge_lbl = tk.Label(top_row, text=f"  {badge_text}  ", font=(FONT, 8, "bold"),
                                 fg=badge_fg, bg=badge_bg, padx=4, pady=2)
            badge_lbl.pack(side=tk.LEFT, padx=10)

            # 右侧删除按钮 ×
            del_lbl = tk.Label(top_row, text=" × ", font=(FONT, 10, "bold"), fg=SUBTLE,
                               bg=PANEL, cursor="hand2")
            del_lbl.pack(side=tk.RIGHT)
            del_lbl.bind("<Enter>", lambda e, w=del_lbl: w.configure(fg=RED))
            del_lbl.bind("<Leave>", lambda e, w=del_lbl: w.configure(fg=SUBTLE))

            def _delete_card(_e, c=cid):
                if ctx.is_running():
                    return
                _task_store.pop(c, None)
                _render_all_cards()
                _update_queue_meta()
            del_lbl.bind("<Button-1>", _delete_card)

            # 下方插槽栏
            slots_row = tk.Frame(card, bg=PANEL)
            slots_row.pack(fill=tk.X)
            slots_row.bind("<MouseWheel>", _on_mousewheel)

            # 1. 台本插槽
            s_file = task.get("script")
            s_frame = tk.Frame(slots_row, bg=SURFACE if s_file else PINK_BG,
                               highlightbackground=BORDER if s_file else BORDER_DASHED,
                               highlightthickness=1, padx=10, pady=6, cursor="hand2")
            s_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
            _slot_registry.append({"widget": s_frame, "type": "script", "cid": cid, "uid": None})

            if s_file:
                s_name = os.path.basename(s_file)
                tk.Label(s_frame, text=f"📄  {_truncate_filename(s_name, 22)}", font=(FONT, 9),
                         fg=TEXT, bg=SURFACE).pack(side=tk.LEFT)
                ToolTip(s_frame, s_file)

                def _start_s_drag(e, c=cid, p=s_file, n=s_name):
                    _on_drag_press(e, p_type="script", p_file=p, p_name=n, cid=c, uid=None)

                s_frame.bind("<ButtonPress-1>", _start_s_drag)
                s_frame.bind("<B1-Motion>", _on_drag_motion)
                s_frame.bind("<ButtonRelease-1>", _on_drag_release)
                for w in s_frame.winfo_children():
                    w.bind("<ButtonPress-1>", _start_s_drag)
                    w.bind("<B1-Motion>", _on_drag_motion)
                    w.bind("<ButtonRelease-1>", _on_drag_release)
            else:
                s_ph = tk.Label(s_frame, text=tr("slot_script_empty"), font=(FONT, 8),
                                fg=PINK, bg=PINK_BG)
                s_ph.pack()

                def _pick_script(_e, c=cid):
                    file = filedialog.askopenfilename(
                        title="选择台本文件",
                        filetypes=[(tr("dlg_filter_supported"), "*.txt"), (tr("dlg_filter_all"), "*.*")])
                    if file:
                        _task_store[c]["script"] = os.path.abspath(file)
                        _render_all_cards()
                        _update_queue_meta()
                s_frame.bind("<Button-1>", _pick_script)
                s_ph.bind("<Button-1>", _pick_script)

            # 连接符号
            tk.Label(slots_row, text=" ── ", font=(MONO, 8), fg="#d0c6cc", bg=PANEL).pack(side=tk.LEFT)

            # 2. 音频插槽 1
            a_file = task.get("audio")
            a_frame = tk.Frame(slots_row, bg=SURFACE if a_file else PINK_BG,
                               highlightbackground=BORDER if a_file else BORDER_DASHED,
                               highlightthickness=1, padx=10, pady=6, cursor="hand2")
            a_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 6 if is_persp else 0))
            _slot_registry.append({"widget": a_frame, "type": "audio", "cid": cid, "uid": None})

            if a_file:
                a_name = os.path.basename(a_file)
                tk.Label(a_frame, text=f"🎵  {_truncate_filename(a_name, 22)}", font=(FONT, 9),
                         fg=TEXT, bg=SURFACE).pack(side=tk.LEFT)
                ToolTip(a_frame, a_file)

                def _start_a_drag(e, c=cid, p=a_file, n=a_name):
                    _on_drag_press(e, p_type="audio", p_file=p, p_name=n, cid=c, uid=None)

                a_frame.bind("<ButtonPress-1>", _start_a_drag)
                a_frame.bind("<B1-Motion>", _on_drag_motion)
                a_frame.bind("<ButtonRelease-1>", _on_drag_release)
                for w in a_frame.winfo_children():
                    w.bind("<ButtonPress-1>", _start_a_drag)
                    w.bind("<B1-Motion>", _on_drag_motion)
                    w.bind("<ButtonRelease-1>", _on_drag_release)
            else:
                a_ph = tk.Label(a_frame, text=tr("slot_audio_empty"), font=(FONT, 8),
                                fg=PINK, bg=PINK_BG)
                a_ph.pack()

                def _pick_audio(_e, c=cid):
                    file = filedialog.askopenfilename(
                        title="选择音频文件",
                        filetypes=[(tr("dlg_filter_audio"), "*.wav;*.mp3;*.flac;*.m4a;*.ogg;*.aac;*.opus"),
                                   (tr("dlg_filter_all"), "*.*")])
                    if file:
                        _task_store[c]["audio"] = os.path.abspath(file)
                        _render_all_cards()
                        _update_queue_meta()
                a_frame.bind("<Button-1>", _pick_audio)
                a_ph.bind("<Button-1>", _pick_audio)

            # 3. 双视角副音频插槽（若为双视角模式或已有副音频）
            if is_persp or task.get("audio_b"):
                # 一键左右交换按钮 ⇄
                swap_lbl = tk.Label(slots_row, text=" ⇄ ", font=(FONT, 9, "bold"), fg=PINK,
                                    bg=SURFACE, padx=4, pady=4, cursor="hand2")
                swap_lbl.pack(side=tk.LEFT, padx=(2, 6))
                ToolTip(swap_lbl, tr("swap_tooltip"))

                def _swap_audios(_e, c=cid):
                    t = _task_store.get(c)
                    if t:
                        t["audio"], t["audio_b"] = t.get("audio_b"), t.get("audio")
                        _render_all_cards()
                swap_lbl.bind("<Button-1>", _swap_audios)

                b_file = task.get("audio_b")
                b_frame = tk.Frame(slots_row, bg=SURFACE if b_file else PINK_BG,
                                   highlightbackground=BORDER if b_file else BORDER_DASHED,
                                   highlightthickness=1, padx=10, pady=6, cursor="hand2")
                b_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
                _slot_registry.append({"widget": b_frame, "type": "audio_b", "cid": cid, "uid": None})

                if b_file:
                    b_name = os.path.basename(b_file)
                    tk.Label(b_frame, text=f"🎧  {_truncate_filename(b_name, 22)}", font=(FONT, 9),
                             fg=TEXT, bg=SURFACE).pack(side=tk.LEFT)
                    ToolTip(b_frame, b_file)

                    def _start_b_drag(e, c=cid, p=b_file, n=b_name):
                        _on_drag_press(e, p_type="audio_b", p_file=p, p_name=n, cid=c, uid=None)

                    b_frame.bind("<ButtonPress-1>", _start_b_drag)
                    b_frame.bind("<B1-Motion>", _on_drag_motion)
                    b_frame.bind("<ButtonRelease-1>", _on_drag_release)
                    for w in b_frame.winfo_children():
                        w.bind("<ButtonPress-1>", _start_b_drag)
                        w.bind("<B1-Motion>", _on_drag_motion)
                        w.bind("<ButtonRelease-1>", _on_drag_release)
                else:
                    b_ph = tk.Label(b_frame, text=tr("slot_audio_b_empty"), font=(FONT, 8),
                                    fg=PINK, bg=PINK_BG)
                    b_ph.pack()

                    def _pick_audio_b(_e, c=cid):
                        file = filedialog.askopenfilename(
                            title="选择副音频文件",
                            filetypes=[(tr("dlg_filter_audio"), "*.wav;*.mp3;*.flac;*.m4a;*.ogg;*.aac;*.opus"),
                                       (tr("dlg_filter_all"), "*.*")])
                        if file:
                            _task_store[c]["audio_b"] = os.path.abspath(file)
                            _render_all_cards()
                            _update_queue_meta()
                    b_frame.bind("<Button-1>", _pick_audio_b)
                    b_ph.bind("<Button-1>", _pick_audio_b)

            # 右键上下文菜单绑定
            def _popup_menu(event, c=cid):
                _show_card_context_menu(event, c)
            card.bind("<Button-3>", _popup_menu)
            top_row.bind("<Button-3>", _popup_menu)
            slots_row.bind("<Button-3>", _popup_menu)

            # 双击快速打开结果
            def _double_open(_e, c=cid):
                _on_card_double_click(c)
            card.bind("<Double-1>", _double_open)
            top_row.bind("<Double-1>", _double_open)

        # 队列已有卡片时，在末尾保留轻量新建槽
        bottom_add_slot = tk.Frame(cards_inner, bg=SURFACE, highlightbackground=BORDER_DASHED,
                                   highlightthickness=1, padx=12, pady=8, cursor="hand2")
        bottom_add_slot.pack(fill=tk.X, padx=2, pady=(4, 10))
        _label(bottom_add_slot, "＋ 拖拽素材至此新建任务卡片", 9, MUTED).pack()
        _slot_registry.append({"widget": bottom_add_slot, "type": "new_card", "cid": None, "uid": None})

        _update_queue_meta()

    # ── 拖拽交互手势 ────────────────────────────────────────────────────

    def _on_drag_press(event, p_type, p_file, p_name, cid, uid):
        if ctx.is_running():
            return
        _drag_info["active"] = False
        _drag_info["type"] = p_type
        _drag_info["file"] = p_file
        _drag_info["name"] = p_name
        _drag_info["cid"] = cid
        _drag_info["uid"] = uid
        _drag_info["start_x"] = event.x_root
        _drag_info["start_y"] = event.y_root

    def _on_drag_motion(event):
        if not _drag_info["file"]:
            return
        dx = abs(event.x_root - _drag_info["start_x"])
        dy = abs(event.y_root - _drag_info["start_y"])
        if not _drag_info["active"] and (dx > 4 or dy > 4):
            _drag_info["active"] = True
            # 创建跟随光标的半透明悬浮指示窗
            win = tk.Toplevel(root)
            win.wm_overrideredirect(True)
            win.attributes("-topmost", True)
            try:
                win.attributes("-alpha", 0.90)
            except Exception:
                pass
            p_icon = "📄" if _drag_info["type"] == "script" else "🎵"
            lbl = tk.Label(win, text=f" {p_icon} {_truncate_filename(_drag_info['name'], 20)} ",
                           font=(FONT, 9, "bold"), bg=PINK_DARK, fg=PINK,
                           bd=1, relief=tk.SOLID, padx=8, pady=4)
            lbl.pack()
            _drag_info["win"] = win

        if _drag_info["active"] and _drag_info["win"]:
            _drag_info["win"].geometry(f"+{event.x_root + 12}+{event.y_root + 12}")
            # 高亮探测当前光标下命中的槽位
            _update_drag_hover(event.x_root, event.y_root)

    def _update_drag_hover(x, y):
        nonlocal _highlighted_slot
        matched = None
        for item in _slot_registry:
            w = item["widget"]
            if not w.winfo_exists() or not w.winfo_ismapped():
                continue
            rx = w.winfo_rootx()
            ry = w.winfo_rooty()
            rw = w.winfo_width()
            rh = w.winfo_height()
            if rx <= x <= rx + rw and ry <= y <= ry + rh:
                matched = item
                break

        if matched != _highlighted_slot:
            if _highlighted_slot:
                try:
                    _highlighted_slot["widget"].configure(highlightbackground=BORDER, highlightthickness=1)
                except Exception:
                    pass
            _highlighted_slot = matched
            if _highlighted_slot:
                try:
                    _highlighted_slot["widget"].configure(highlightbackground=PINK, highlightthickness=2)
                except Exception:
                    pass

    def _on_drag_release(event):
        nonlocal _highlighted_slot
        if _drag_info["win"]:
            try:
                _drag_info["win"].destroy()
            except Exception:
                pass
            _drag_info["win"] = None

        if _highlighted_slot:
            try:
                _highlighted_slot["widget"].configure(highlightbackground=BORDER, highlightthickness=1)
            except Exception:
                pass

        if not _drag_info["active"]:
            _drag_info["file"] = None
            return

        # 查找目标投掷槽位
        target = _highlighted_slot
        _highlighted_slot = None
        _drag_info["active"] = False

        if not target:
            # 盲区丢弃，不做任何修改
            _drag_info["file"] = None
            return

        src_type = _drag_info["type"]
        src_file = _drag_info["file"]
        src_cid = _drag_info["cid"]
        src_uid = _drag_info["uid"]
        tgt_type = target["type"]
        tgt_cid = target["cid"]

        _drag_info["file"] = None

        # 1. 丢入「待分配托盘」 -> 拔出卡片回托盘
        if tgt_type == "tray":
            if src_cid is not None:
                card = _task_store.get(src_cid)
                if card:
                    card[src_type] = None
                    # 加入待分配托盘
                    nonlocal _uid_counter
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

        # 2. 丢入新建任务槽位 -> 自动创建新任务卡片并消耗素材
        if tgt_type == "new_card":
            nonlocal _card_id_counter
            _card_id_counter += 1
            cid = f"card_{_card_id_counter}"
            is_src_audio = src_type in ("audio", "audio_b")
            raw_stem = os.path.splitext(os.path.basename(src_file))[0]
            clean_name = raw_stem
            if is_src_audio and hasattr(ctx, "_extract_perspective_tag"):
                base_k, _, _ = ctx._extract_perspective_tag(raw_stem)
                if base_k:
                    clean_name = base_k

            _task_store[cid] = {
                "cid": cid,
                "iid": cid,
                "name": clean_name,
                "script": src_file if not is_src_audio else None,
                "audio": src_file if is_src_audio else None,
                "audio_b": None,
                "status": "pending",
                "confidence": 100,
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
            _update_queue_meta()
            return

        # 3. 丢入卡片插槽
        # 验证文件类型兼容性（台本进台本槽，音频进音频槽）
        is_src_audio = src_type in ("audio", "audio_b")
        is_tgt_audio = tgt_type in ("audio", "audio_b")
        if (src_type == "script" and tgt_type != "script") or (is_src_audio and not is_tgt_audio):
            # 类型不兼容，拒绝换位
            return

        tgt_card = _task_store.get(tgt_cid)
        if not tgt_card:
            return

        old_tgt_file = tgt_card.get(tgt_type)

        # 目标为空插槽 -> 填入
        if not old_tgt_file:
            tgt_card[tgt_type] = src_file
            if src_cid is not None:
                src_card = _task_store.get(src_cid)
                if src_card:
                    src_card[src_type] = None
                    if not src_card.get("script") and not src_card.get("audio") and not src_card.get("audio_b"):
                        _task_store.pop(src_cid, None)
            else:
                _unpaired_pool.pop(src_uid, None)
        # 目标已有文件 -> 互换位置 (Swap)
        else:
            if src_cid is not None:
                src_card = _task_store.get(src_cid)
                if src_card:
                    src_card[src_type] = old_tgt_file
                    tgt_card[tgt_type] = src_file
            else:
                tgt_card[tgt_type] = src_file
                _unpaired_pool[src_uid]["file"] = old_tgt_file
                _unpaired_pool[src_uid]["name"] = os.path.basename(old_tgt_file)

        # 人工干预后清空模糊标记
        tgt_card["confidence"] = 100
        if src_cid and src_cid in _task_store:
            _task_store[src_cid]["confidence"] = 100

        _render_all_cards()
        _update_queue_meta()

    # ── 文件导入与自动初筛配对 ──────────────────────────────────────────

    def _add_files_to_list(files):
        if not files:
            return
        nonlocal _card_id_counter, _uid_counter

        # 1. 汇集已有素材池文件 + 未配满已有卡片文件 + 新拖入的所有文件进行联合匹配
        pool_files = [item["file"] for item in _unpaired_pool.values()]
        incomplete_files = []
        for t in _task_store.values():
            if not t.get("script") or not t.get("audio") or (mode_var.get() == "multi_persp" and not t.get("audio_b")):
                if t.get("script"):
                    incomplete_files.append(t["script"])
                if t.get("audio"):
                    incomplete_files.append(t["audio"])
                if t.get("audio_b"):
                    incomplete_files.append(t["audio_b"])

        new_files = [os.path.abspath(f) for f in files if f and str(f).strip()]
        all_candidates = list(dict.fromkeys(pool_files + incomplete_files + new_files))

        pairs, unmatched = ctx.auto_pair_files(all_candidates, mode=mode_var.get())
        added = 0

        existing_auds = {p.get("audio") for p in _task_store.values()} | {p.get("audio_b") for p in _task_store.values()}
        existing_txts = {p.get("script") for p in _task_store.values()}

        # 2. 检查是否有卡片缺少 audio_b，若本次配对找出了副视角音频，自动补全已有卡片
        if mode_var.get() == "multi_persp":
            for pair in list(pairs):
                # 寻找台本相同且目前缺少 audio_b 的已有任务卡片
                matched_exist = None
                for cid, task in _task_store.items():
                    if task.get("script") == pair.get("script"):
                        matched_exist = task
                        break
                if matched_exist:
                    # 补齐 audio 或 audio_b
                    if not matched_exist.get("audio") and pair.get("audio"):
                        matched_exist["audio"] = pair["audio"]
                        existing_auds.add(pair["audio"])
                        added += 1
                    if not matched_exist.get("audio_b") and pair.get("audio_b"):
                        matched_exist["audio_b"] = pair["audio_b"]
                        existing_auds.add(pair["audio_b"])
                        added += 1
                    pairs.remove(pair)

        # 3. 录入全新配对成功的任务卡片
        for pair in pairs:
            if pair.get("audio") in existing_auds and pair.get("script") in existing_txts:
                continue
            _card_id_counter += 1
            cid = f"card_{_card_id_counter}"
            _task_store[cid] = {
                "cid": cid,
                "iid": cid,
                "name": pair["name"],
                "script": pair.get("script"),
                "audio": pair.get("audio"),
                "audio_b": pair.get("audio_b"),
                "status": "pending",
                "confidence": pair.get("confidence", 100),
            }
            if pair.get("audio"):
                existing_auds.add(pair["audio"])
            if pair.get("audio_b"):
                existing_auds.add(pair["audio_b"])
            if pair.get("script"):
                existing_txts.add(pair["script"])
            added += 1

        # 4. 重新同步未配对素材池：清空旧池，只保留本次仍未配对的文件
        _unpaired_pool.clear()
        for item in unmatched:
            f_path = item["file"]
            is_txt = item["type"] == "缺音频"
            if (is_txt and f_path in existing_txts) or (not is_txt and f_path in existing_auds):
                continue
            _uid_counter += 1
            uid = f"u_{_uid_counter}"
            _unpaired_pool[uid] = {
                "uid": uid,
                "type": "script" if is_txt else "audio",
                "file": f_path,
                "name": item["name"],
            }

        _render_all_cards()
        if added:
            update_status(tr("status_added_files", added=added, total=len(_task_store)))
        else:
            update_status(tr("status_files_already"))

    def _auto_assemble_from_tray():
        if not _unpaired_pool:
            return
        files = [item["file"] for item in list(_unpaired_pool.values())]
        _add_files_to_list(files)

    def _browse_files(_event=None):
        files = filedialog.askopenfilenames(
            title=tr("dlg_select_files_title"),
            filetypes=[(tr("dlg_filter_supported"), "*.txt;*.wav;*.mp3;*.flac;*.m4a;*.ogg;*.aac;*.opus;*.wma"),
                       (tr("dlg_filter_all"), "*.*")])
        if files:
            _add_files_to_list(list(files))

    for widget in (drop_frame, drop_inner, drop_label, drop_icon):
        widget.bind("<Button-1>", _browse_files)
    add_btn.configure(command=_browse_files)

    if ctx.has_dnd:
        def _on_external_drop(event):
            paths = []
            try:
                for item in root.tk.splitlist(event.data):
                    path = str(item).strip().strip('"').strip("'")
                    if path:
                        paths.append(os.path.abspath(path))
            except Exception:
                paths = [os.path.abspath(p) for p in event.data.split() if p.strip()]
            _add_files_to_list(paths)

        for widget in (drop_frame, drop_inner, drop_label, drop_icon, cards_wrapper, canvas):
            widget.drop_target_register(ctx.dnd_files)
            widget.dnd_bind("<<Drop>>", _on_external_drop)

    def _clear_tasks():
        if ctx.is_running():
            return
        _task_store.clear()
        _unpaired_pool.clear()
        _render_all_cards()
        _update_queue_meta()
        progress.configure(value=0)
        progress_pct_var.set("0%")
        update_status(tr("status_queue_cleared"))
    clear_btn.configure(command=_clear_tasks)

    # ── 右键上下文菜单与双击操作 ──────────────────────────────────────

    def _show_card_context_menu(event, cid):
        item = _task_store.get(cid)
        if not item:
            return
        menu = tk.Menu(root, tearoff=0, bg=PANEL, fg=TEXT,
                       activebackground=SURFACE_ACTIVE, activeforeground=PINK, font=(FONT, 9))

        def _open_file(path):
            if path and os.path.exists(path):
                os.startfile(os.path.normpath(path))

        def _open_folder():
            target = item.get("audio") or item.get("script")
            if target and os.path.exists(target):
                import subprocess
                subprocess.run(["explorer", "/select,", os.path.normpath(target)])

        def _view_output(extension):
            audio = item.get("audio")
            if not audio:
                return
            output = os.path.splitext(audio)[0] + extension
            if os.path.exists(output):
                _open_file(output)
            else:
                messagebox.showinfo(tr("msg_no_subtitle_title"),
                                    tr("msg_no_subtitle_text", extension=extension), parent=root)

        def _retry_this():
            item["status"] = "pending"
            _render_all_cards()
            update_status(tr("status_requeued", name=item['name']))

        menu.add_command(label=tr("menu_open_folder"), command=_open_folder)
        menu.add_command(label=tr("menu_open_script"), command=lambda: _open_file(item.get("script")))
        menu.add_command(label=tr("menu_view_lrc"), command=lambda: _view_output(".lrc"))
        menu.add_command(label=tr("menu_view_srt"), command=lambda: _view_output(".srt"))
        menu.add_separator()
        if mode_var.get() == "multi_persp":
            def _swap_ab():
                item["audio"], item["audio_b"] = item.get("audio_b"), item.get("audio")
                _render_all_cards()
            menu.add_command(label=tr("menu_swap_audios"), command=_swap_ab)
            menu.add_separator()
        menu.add_command(label=tr("menu_retry"), command=_retry_this)
        menu.add_command(label=tr("btn_remove_card"), command=lambda: (_task_store.pop(cid, None), _render_all_cards()))
        menu.post(event.x_root, event.y_root)

    def _on_card_double_click(cid):
        item = _task_store.get(cid)
        if not item:
            return
        audio = item.get("audio")
        if audio:
            for extension in (".srt", ".lrc"):
                output = os.path.splitext(audio)[0] + extension
                if os.path.exists(output):
                    os.startfile(os.path.normpath(output))
                    return
        target = audio or item.get("script")
        if target and os.path.exists(target):
            import subprocess
            subprocess.run(["explorer", "/select,", os.path.normpath(target)])

    # ── 日志面板 ────────────────────────────────────────────────────────

    log_panel = _frame(left, BG)
    log_panel.pack(fill=tk.X)
    log_head = _frame(log_panel, BG)
    log_head.pack(fill=tk.X)
    log_toggle = _label(log_head, tr("log_panel_collapsed"), 8, MUTED, cursor="hand2")
    log_toggle.pack(side=tk.LEFT)
    log_clear = _button(log_head, tr("btn_clear_log"), lambda: None, compact=True)
    log_frame = _frame(log_panel, BG, border=True)
    log_text = tk.Text(log_frame, bg=PANEL, fg=MUTED, font=(MONO, 9), relief=tk.FLAT,
                       wrap=tk.WORD, height=5, padx=10, pady=8, insertbackground=PINK,
                       selectbackground=SURFACE_ACTIVE)
    log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=log_text.yview)
    log_text.configure(yscrollcommand=log_scroll.set)
    log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    log_visible = tk.BooleanVar(value=False)

    def _toggle_log():
        if log_visible.get():
            log_frame.pack_forget()
            log_clear.pack_forget()
            log_toggle.configure(text=tr("log_panel_collapsed"))
            log_visible.set(False)
        else:
            log_clear.pack(side=tk.RIGHT)
            log_frame.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
            log_toggle.configure(text=tr("log_panel_expanded"))
            log_visible.set(True)
    log_toggle.bind("<Button-1>", lambda e: _toggle_log())
    log_clear.configure(command=lambda: log_text.delete("1.0", tk.END))

    def append_log(msg):
        _post_ui(lambda: (log_text.insert(tk.END, msg + "\n"), log_text.see(tk.END)))

    # ── 右侧运行配置面板（方案 A） ─────────────────────────────────────

    settings = _frame(right, PANEL, border=True)
    settings.pack(fill=tk.X)
    settings_pad = _frame(settings, PANEL)
    settings_pad.pack(fill=tk.X, padx=18, pady=16)
    settings_title_lbl = _label(settings_pad, tr("settings_title"), 11, TEXT, True)
    settings_title_lbl.pack(anchor="w")
    settings_sub_lbl = _label(settings_pad, tr("settings_sub"), 8, MUTED)
    settings_sub_lbl.pack(anchor="w", pady=(3, 16))

    # 1. 界面语言选择
    lang_field_lbl = _label(settings_pad, tr("label_lang"), 8, MUTED)
    lang_field_lbl.pack(anchor="w", pady=(0, 6))
    lang_combo = ttk.Combobox(settings_pad, textvariable=lang_display_var,
                              values=[name for _, name in LANG_NAMES],
                              state="readonly", width=25)
    lang_combo.pack(fill=tk.X, pady=(0, 14))

    # 2. 显存分块预设
    chunk_field_lbl = _label(settings_pad, tr("label_chunk"), 8, MUTED)
    chunk_field_lbl.pack(anchor="w", pady=(0, 6))
    chunk_combo = ttk.Combobox(settings_pad, textvariable=chunk_preset_var,
                               values=get_chunk_presets(), state="readonly", width=25)
    chunk_combo.pack(fill=tk.X, pady=(0, 14))

    def _on_chunk_preset(_event=None):
        idx = chunk_combo.current()
        if 0 <= idx < len(CHUNK_VALUES):
            vram_chunk_var.set(CHUNK_VALUES[idx])
            _save_settings()
    chunk_combo.bind("<<ComboboxSelected>>", _on_chunk_preset)

    # 3. 字幕驻留延长
    extend_field_lbl = _label(settings_pad, tr("label_end_extend"), 8, MUTED)
    extend_field_lbl.pack(anchor="w", pady=(0, 6))
    extend_row = _frame(settings_pad, SURFACE)
    extend_row.pack(fill=tk.X, pady=(0, 14))
    ext_ent = tk.Entry(extend_row, textvariable=end_extend_var, font=(MONO, 10), width=7,
                       relief=tk.FLAT, bg=SURFACE, fg=TEXT, insertbackground=PINK,
                       highlightthickness=0, justify="left")
    ext_ent.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=7, padx=(9, 2))
    sec_lbl = _label(extend_row, tr("label_sec"), 9, MUTED)
    sec_lbl.pack(side=tk.RIGHT, padx=(0, 10))
    ext_ent.bind("<FocusOut>", lambda e: _save_settings())

    # 4. Ono-Gate 拟声词剔除
    ono_check = tk.Checkbutton(settings_pad, text=tr("check_ono_gate"), variable=ono_gate_var,
                               font=(FONT, 9), fg=TEXT, bg=PANEL, selectcolor=SURFACE,
                               activebackground=PANEL, activeforeground=PINK,
                               highlightthickness=0, bd=0, anchor="w", cursor="hand2",
                               command=_save_settings)
    ono_check.pack(fill=tk.X, pady=(0, 12))

    # 5. 高级设置
    adv_toggle = _button(settings_pad, tr("adv_expand"), lambda: None, compact=True)
    adv_toggle.pack(fill=tk.X, pady=(0, 10))
    adv_panel = _frame(settings_pad, PANEL)
    adv_show = tk.BooleanVar(value=False)

    def _toggle_adv():
        if adv_show.get():
            adv_panel.pack_forget()
            adv_toggle.configure(text=tr("adv_expand"))
            adv_show.set(False)
        else:
            adv_panel.pack(fill=tk.X, pady=(0, 10), before=start_btn)
            adv_toggle.configure(text=tr("adv_collapse"))
            adv_show.set(True)
    adv_toggle.configure(command=_toggle_adv)

    dev_field_lbl = _label(adv_panel, tr("label_device"), 8, MUTED)
    dev_field_lbl.pack(anchor="w", pady=(0, 5))
    dev_cb = ttk.Combobox(adv_panel, textvariable=dev_var,
                          values=["cuda", "cpu"] if ctx.cuda_ok else ["cpu"], state="readonly")
    dev_cb.pack(fill=tk.X, pady=(0, 10))
    dev_cb.bind("<<ComboboxSelected>>", lambda e: _save_settings())
    custom_chunk_lbl = None
    if ctx.cuda_ok:
        custom_chunk_lbl = _label(adv_panel, tr("label_custom_chunk"), 8, MUTED)
        custom_chunk_lbl.pack(anchor="w", pady=(0, 5))
        custom = tk.Entry(adv_panel, textvariable=vram_chunk_var, font=(MONO, 10), relief=tk.FLAT,
                          bg=SURFACE, fg=TEXT, insertbackground=PINK)
        custom.pack(fill=tk.X, ipady=6)
        custom.bind("<FocusOut>", lambda e: (_sync_chunk_preset(), _save_settings()))

    start_btn = _button(settings_pad, tr("btn_start"), lambda: None, primary=True)
    start_btn.pack(fill=tk.X, pady=(4, 12))
    help_btn = _button(settings_pad, tr("btn_show_guide"), lambda: show_script_help_dialog(root), compact=True)
    help_btn.pack(fill=tk.X)

    # 执行状态面板
    run_state = _frame(right, PANEL, border=True)
    run_state.pack(fill=tk.X, pady=(12, 0))
    state_pad = _frame(run_state, PANEL)
    state_pad.pack(fill=tk.X, padx=16, pady=13)
    state_top = _frame(state_pad, PANEL)
    state_top.pack(fill=tk.X, pady=(0, 8))
    state_title_lbl = _label(state_top, tr("state_title"), 8, MUTED)
    state_title_lbl.pack(side=tk.LEFT)
    progress_pct_var = tk.StringVar(value="0%")
    tk.Label(state_top, textvariable=progress_pct_var, font=(MONO, 9, "bold"),
             fg=PINK, bg=PANEL).pack(side=tk.RIGHT)
    progress = ttk.Progressbar(state_pad, mode="determinate", maximum=100)
    progress.pack(fill=tk.X, pady=(0, 8))
    status_var = tk.StringVar(value=tr("status_idle"))
    status_label = tk.Label(state_pad, textvariable=status_var, font=(FONT, 8),
                            fg=MUTED, bg=PANEL, anchor="w", justify=tk.LEFT,
                            wraplength=250)
    status_label.pack(fill=tk.X)

    def update_status(msg):
        _post_ui(lambda: status_var.set(msg))

    def _refresh_all_i18n():
        """即时热切换全界面多语言文案。"""
        root.title(tr("app_title"))
        hero_canvas.itemconfigure(hero_left_id, text=tr("banner_badge"))
        hero_canvas.itemconfigure(hero_title_id, text=tr("banner_title"))
        hero_canvas.itemconfigure(hero_sub_id, text=tr("banner_subtitle"))
        hero_canvas.itemconfigure(hero_device_id, text=tr("device_cuda") if ctx.cuda_ok else tr("device_cpu"))

        mode_title_lbl.configure(text=tr("mode_title_label"))
        for index, key in enumerate(("single", "multi_speaker", "multi_persp")):
            if key in mode_buttons:
                mode_buttons[key].configure(text=f"0{index + 1}   {tr(f'mode_{key}_title')}")
        mode_desc_var.set(tr(f"mode_{mode_var.get()}_desc"))

        _update_drop_zone_text()
        drop_hint_lbl.configure(text=tr("drop_hint"))

        queue_title_lbl.configure(text=tr("queue_title"))
        clear_btn.configure(text=tr("btn_clear"))
        tray_title_lbl.configure(text=tr("unpaired_tray_title"))
        tray_hint_lbl.configure(text=tr("unpaired_tray_hint"))

        _render_all_cards()

        settings_title_lbl.configure(text=tr("settings_title"))
        settings_sub_lbl.configure(text=tr("settings_sub"))
        lang_field_lbl.configure(text=tr("label_lang"))
        chunk_field_lbl.configure(text=tr("label_chunk"))
        chunk_combo.configure(values=get_chunk_presets())
        _sync_chunk_preset()
        extend_field_lbl.configure(text=tr("label_end_extend"))
        sec_lbl.configure(text=tr("label_sec"))
        ono_check.configure(text=tr("check_ono_gate"))
        adv_toggle.configure(text=tr("adv_collapse") if adv_show.get() else tr("adv_expand"))
        dev_field_lbl.configure(text=tr("label_device"))
        if custom_chunk_lbl is not None:
            custom_chunk_lbl.configure(text=tr("label_custom_chunk"))
        help_btn.configure(text=tr("btn_show_guide"))

        log_toggle.configure(text=tr("log_panel_expanded") if log_visible.get() else tr("log_panel_collapsed"))
        log_clear.configure(text=tr("btn_clear_log"))

        state_title_lbl.configure(text=tr("state_title"))
        if not getattr(root, "_fa_running", False) and len(_task_store) == 0:
            status_var.set(tr("status_idle"))

    def _on_lang_change(_event=None):
        idx = lang_combo.current()
        if 0 <= idx < len(LANG_NAMES):
            code, name = LANG_NAMES[idx]
            I18N.set_lang(code)
            lang_var.set(code)
            lang_display_var.set(name)
            _refresh_all_i18n()
            _save_settings()
    lang_combo.bind("<<ComboboxSelected>>", _on_lang_change)

    # ── 对齐执行链路 ────────────────────────────────────────────────────

    def _set_running(running):
        root._fa_running = running
        pending = sum(1 for item in _task_store.values()
                      if item.get("audio") and item.get("script") and item.get("status") != "done")
        btn_text = tr("btn_running") if running else (
            tr("btn_start_with_count", count=pending) if pending else tr("btn_start")
        )
        start_btn.configure(state=tk.DISABLED if running else tk.NORMAL,
                            text=btn_text,
                            bg=SURFACE if running else PINK,
                            fg=MUTED if running else INK)
        state = tk.DISABLED if running else tk.NORMAL
        clear_btn.configure(state=state)
        add_btn.configure(state=state)

    def _set_task_status(iid, status):
        def _apply():
            if iid in _task_store:
                _task_store[iid]["status"] = status
                _render_all_cards()
        _post_ui(_apply)

    def _set_progress(value, message):
        def _apply():
            progress.configure(value=value)
            progress_pct_var.set(f"{value:.0f}%")
            status_var.set(message)
        _post_ui(_apply)

    def _run_all():
        if ctx.is_running():
            return
        _save_settings()
        try:
            end_extend = float(end_extend_var.get() or 0)
            chunk = int(vram_chunk_var.get() or 6000)
            if end_extend < 0 or chunk <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror(tr("msg_param_error_title"), tr("msg_param_error_text"), parent=root)
            return
        current = mode_var.get()
        items = list(_task_store.values())
        valid = [item for item in items if item.get("audio") and item.get("script")
                 and item.get("status") != "done"]
        if not valid:
            messagebox.showinfo(tr("msg_empty_queue_title"), tr("msg_empty_queue_text"), parent=root)
            return
        if current == "multi_persp":
            missing = [item["name"] for item in valid if not item.get("audio_b")]
            if missing:
                missing_preview = "\n".join(missing[:8]) + ("\n…" if len(missing) > 8 else "")
                if not messagebox.askyesno(
                        tr("msg_missing_persp_b_title"),
                        tr("msg_missing_persp_b_text", names=missing_preview), parent=root):
                    return

        _set_running(True)
        _set_progress(0, tr("status_preparing", count=len(valid)))
        append_log(f"\n开始批量处理 · {tr(f'mode_{current}_title')} · {dev_var.get().upper()}")

        def _worker():
            succeeded = 0
            failed = 0
            total = len(valid)
            try:
                for index, item in enumerate(valid):
                    iid = item["iid"]
                    _set_task_status(iid, "running")
                    append_log(f"[{index + 1}/{total}] {item['name']}")

                    def _pct(percent, message):
                        overall = (index * 100 + percent) / total
                        _set_progress(overall, tr("status_progress_step", current=index + 1, total=total, message=message))

                    ok, _output, error = ctx.run_pipeline(
                        script_path=item["script"], audio_path=item["audio"],
                        lora_path=ctx.lora_path, log_callback=append_log, progress_callback=_pct,
                        dev=dev_var.get(), vram_chunk=chunk, persist_gpu=True,
                        line_end_extend=end_extend, enable_silence_snap=True,
                        enable_resegment=True, enable_ono_gate=bool(ono_gate_var.get()),
                        enable_multi_order=current in ("multi_speaker", "multi_persp"),
                        audio_path_b=item.get("audio_b") if current == "multi_persp" else None)
                    _set_task_status(iid, "done" if ok else "failed")
                    if ok:
                        succeeded += 1
                    else:
                        failed += 1
                        append_log(f"处理失败：{error}")
                _set_progress(100, tr("status_all_done", succeeded=succeeded, failed=failed))
                append_log(f"批量任务完成：成功 {succeeded}，失败 {failed}")
            except Exception as error:
                failed += 1
                append_log(f"任务中断：{type(error).__name__}: {error}")
                _set_progress(0, tr("status_interrupted", error=error))
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
        _save_settings()
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", _on_close)

    _on_mode_change()
    _render_all_cards()
    _refresh_all_i18n()
    root.update_idletasks()
    width, height = root.winfo_width(), root.winfo_height()
    root.geometry(f"+{max(0, (root.winfo_screenwidth() - width) // 2)}"
                  f"+{max(0, (root.winfo_screenheight() - height) // 2)}")
    return root
