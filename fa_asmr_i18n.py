# -*- coding: utf-8 -*-
"""FA-ASMR 国际化与多语言模块。提供中/日/英界面文本及动态切换支持。"""
from __future__ import annotations

import locale
import sys
from typing import Dict, Any, List, Tuple

LANG_NAMES: List[Tuple[str, str]] = [
    ("zh", "简体中文"),
    ("ja", "日本語"),
    ("en", "English"),
]

LANG_MAP = dict(LANG_NAMES)
DEFAULT_LANG = "zh"


def detect_system_lang() -> str:
    """根据系统区域与首选语言检测匹配语言代码。"""
    try:
        if sys.platform == "win32":
            import ctypes
            # GetUserDefaultUILanguage() 返回 LANGID（如 0x0804 为 zh-CN，0x0411 为 ja-JP，0x0409 为 en-US）
            lang_id = ctypes.windll.kernel32.GetUserDefaultUILanguage()
            primary = lang_id & 0x3FF
            if primary == 0x11:  # LANG_JAPANESE
                return "ja"
            elif primary == 0x04:  # LANG_CHINESE
                return "zh"
            elif primary == 0x09:  # LANG_ENGLISH
                return "en"
    except Exception:
        pass

    try:
        loc, _ = locale.getdefaultlocale()
        if loc:
            loc = loc.lower()
            if loc.startswith("ja"):
                return "ja"
            elif loc.startswith("zh"):
                return "zh"
            elif loc.startswith("en"):
                return "en"
    except Exception:
        pass

    return DEFAULT_LANG


# ══════════════════════════════════════════════════════════════════════════
# 多语言文本词典
# ══════════════════════════════════════════════════════════════════════════

TRANSLATIONS: Dict[str, Dict[str, str]] = {
    # ── 窗口与 Banner ──
    "app_title": {
        "zh": "FA-ASMR · 音声对齐工作台",
        "ja": "FA-ASMR · 音声アライメントワークステーション",
        "en": "FA-ASMR · Forced Alignment Studio",
    },
    "banner_badge": {
        "zh": "FA-ASMR V2",
        "ja": "FA-ASMR V2",
        "en": "FA-ASMR V2",
    },
    "banner_title": {
        "zh": "音声物理声学强制对齐工具",
        "ja": "音声音響物理強制アライメントツール",
        "en": "Acoustic Forced Alignment Tool",
    },
    "banner_subtitle": {
        "zh": "音声强制对齐工作台   ·   精确生成 LRC 与 SRT 字幕",
        "ja": "音声強制アライメント   ·   高精度 LRC / SRT 字幕生成",
        "en": "Forced Alignment Studio   ·   Precision LRC & SRT Subtitles",
    },
    "device_cuda": {
        "zh": "●  CUDA 加速",
        "ja": "●  CUDA 加速",
        "en": "●  CUDA Accel",
    },
    "device_cpu": {
        "zh": "●  CPU 模式",
        "ja": "●  CPU モード",
        "en": "●  CPU Mode",
    },

    # ── 模式分段选择 ──
    "mode_title_label": {
        "zh": "对齐模式",
        "ja": "アライメントモード",
        "en": "Alignment Mode",
    },
    "mode_single_title": {
        "zh": "顺序对齐",
        "ja": "シーケンシャル",
        "en": "Sequential",
    },
    "mode_single_sub": {
        "zh": "音频发音与台本顺序完全相同",
        "ja": "音声の発音と台本の順序が完全一致",
        "en": "Audio strictly follows script order",
    },
    "mode_single_desc": {
        "zh": "整篇音频发音与台本全局一一对应、单向推进。无角色标记或带有 〖角色名〗 均可（输出自动保留标签），对齐速度最快、稳定性与抗噪最高。",
        "ja": "音声の発音順序と台本のテキスト順序が完全に一致し、単方向に進みます。話者タグなし、または〖話者名〗付きのいずれにも対応（字幕にタグを自動保持）。最速かつ最も安定したモードです。",
        "en": "Audio pronunciation strictly follows the script sequentially. Supports untagged scripts or bracketed [Speaker Name] tags (retained in output). Fastest and most stable mode.",
    },
    "mode_multi_speaker_title": {
        "zh": "多人分角色",
        "ja": "複数話者",
        "en": "Multi-Speaker",
    },
    "mode_multi_speaker_sub": {
        "zh": "单角色顺序不变 · 角色间可交错/乱序/同时说话",
        "ja": "話者内順序維持 · 話者間の交錯・乱序・重複発言に対応",
        "en": "Per-speaker order preserved · Interleaved/overlapping speech",
    },
    "mode_multi_speaker_desc": {
        "zh": "各角色自身台词顺序不变，角色间允许交错、抢话或同时重叠。台本需用 〖角色名〗 标注，引擎按角色分流对齐并互斥合并。",
        "ja": "各話者の台詞順序は維持され、話者間の会話の割り込み、順序入れ替わり、同時発話（重なり）を許容します。台本には〖話者名〗が必要です。エンジンが話者ごとに分離・並行対齐し排他統合します。",
        "en": "Each speaker's lines proceed sequentially, while allowing cross-speaker interruptions, overlapping, or out-of-order speech. Scripts require [Speaker Name] tags for multi-stream alignment and merging.",
    },
    "mode_multi_persp_title": {
        "zh": "双视角联合",
        "ja": "デュアル視点",
        "en": "Dual-Perspective",
    },
    "mode_multi_persp_sub": {
        "zh": "两份不同机位/视角音频 · 动态整合两边最优结果",
        "ja": "2つの異なる視点/マイク音声 · 両視点の最適結果を動的統合",
        "en": "Two perspective audios · Joint optimal alignment",
    },
    "mode_multi_persp_desc": {
        "zh": "针对同一场景的两份不同机位/视角音频（视角 A 与 视角 B）。自动测算时间差并联合求解，同时生成两份高精度 LRC 与 SRT 字幕。",
        "ja": "同一シーンの2つの異なるマイク/視点音声（視点 A と 視点 B）を対象とします。時間差（オフセット）を自動測定し共同求解、両視点の高精度 LRC / SRT 字幕を同時出力します。",
        "en": "Designed for two recordings (Perspective A & B) of the same scene. Automatically estimates time offset and performs joint acoustic alignment, exporting synchronized LRC and SRT subtitles for both.",
    },

    # ── 拖放导入区 ──
    "drop_prompt_normal": {
        "zh": "拖入台本与音频，开始新的对齐",
        "ja": "台本と音声ファイルをドロップして開始",
        "en": "Drop script & audio files here to start",
    },
    "drop_prompt_persp": {
        "zh": "拖入台本与 A / B 双视角音频",
        "ja": "台本と A / B デュアル視点音声をドロップ",
        "en": "Drop script and Perspective A / B audios",
    },
    "drop_hint": {
        "zh": "点击选择文件  ·  支持拖放  ·  自动按文件名配对",
        "ja": "クリックして選択  ·  ドラッグ＆ドロップ対応  ·  自動ファイル名照合",
        "en": "Click to select  ·  Drag & Drop supported  ·  Auto-paired by name",
    },

    # ── 任务队列 ──
    "queue_title": {
        "zh": "任务队列",
        "ja": "タスクキュー",
        "en": "Task Queue",
    },
    "queue_empty_meta": {
        "zh": "0 项",
        "ja": "0 件",
        "en": "0 items",
    },
    "queue_meta_format": {
        "zh": "{count} 项",
        "ja": "{count} 件",
        "en": "{count} items",
    },
    "queue_meta_done_format": {
        "zh": "{count} 项 · {done} 已完成",
        "ja": "{count} 件 · {done} 件完了",
        "en": "{count} items · {done} done",
    },
    "btn_clear": {
        "zh": "清空",
        "ja": "クリア",
        "en": "Clear",
    },
    "btn_remove_selected": {
        "zh": "移除选中",
        "ja": "選択削除",
        "en": "Remove",
    },

    # ── 队列表头 ──
    "col_name": {
        "zh": "任务",
        "ja": "タスク",
        "en": "Task",
    },
    "col_script": {
        "zh": "台本",
        "ja": "台本",
        "en": "Script",
    },
    "col_audio": {
        "zh": "音频",
        "ja": "音声",
        "en": "Audio",
    },
    "col_persp_a": {
        "zh": "视角 A",
        "ja": "視点 A",
        "en": "Perspective A",
    },
    "col_persp_b": {
        "zh": "视角 B",
        "ja": "視点 B",
        "en": "Perspective B",
    },
    "col_status": {
        "zh": "状态",
        "ja": "状態",
        "en": "Status",
    },

    # ── 状态徽章 ──
    "unpaired_tray_title": {
        "zh": "待分配素材池 (未配对文件)",
        "ja": "未割り当て素材プール",
        "en": "Unassigned Materials Pool",
    },
    "unpaired_tray_hint": {
        "zh": "按住素材拖拽至下方卡片中进行配对或调换",
        "ja": "下のカードにドラッグしてペアリングまたは入れ替え",
        "en": "Drag items to cards below to pair or swap",
    },
    "slot_script_empty": {
        "zh": "＋ 拖入台本",
        "ja": "＋ 台本をドロップ",
        "en": "＋ Drop Script",
    },
    "slot_audio_empty": {
        "zh": "＋ 拖入音频",
        "ja": "＋ 音声をドロップ",
        "en": "＋ Drop Audio",
    },
    "slot_audio_b_empty": {
        "zh": "＋ 拖入副音频",
        "ja": "＋ 副音声をドロップ",
        "en": "＋ Drop Sub Audio",
    },
    "badge_exact": {
        "zh": "✓ 精确匹配",
        "ja": "✓ 完全一致",
        "en": "✓ Exact Match",
    },
    "badge_fuzzy": {
        "zh": "● 建议匹配 {score}%",
        "ja": "● 推定一致 {score}%",
        "en": "● Fuzzy {score}%",
    },
    "swap_tooltip": {
        "zh": "点击交换主副音频",
        "ja": "主音声と副音声を入れ替え",
        "en": "Swap Main and Sub Audio",
    },
    "btn_remove_card": {
        "zh": "移除此卡片",
        "ja": "カードを削除",
        "en": "Remove Card",
    },
    "menu_swap_audios": {
        "zh": "交换主音频与副音频",
        "ja": "主音声と副音声を入れ替える",
        "en": "Swap Main & Sub Audio",
    },
    "status_pending": {
        "zh": "○  等待处理",
        "ja": "○  待機中",
        "en": "○  Waiting",
    },
    "status_running": {
        "zh": "●  正在对齐",
        "ja": "●  アライメント中",
        "en": "●  Aligning",
    },
    "status_done": {
        "zh": "✓  已完成",
        "ja": "✓  完了",
        "en": "✓  Done",
    },
    "status_failed": {
        "zh": "×  处理失败",
        "ja": "×  失敗",
        "en": "×  Failed",
    },
    "status_warn": {
        "zh": "!  缺少文件",
        "ja": "!  ファイル不足",
        "en": "!  Missing File",
    },
    "missing_audio": {
        "zh": "缺少音频",
        "ja": "音声不足",
        "en": "Missing audio",
    },
    "missing_script": {
        "zh": "缺少台本",
        "ja": "台本不足",
        "en": "Missing script",
    },

    # ── 日志面板 ──
    "log_panel_collapsed": {
        "zh": "运行日志  ＋",
        "ja": "実行ログ  ＋",
        "en": "Execution Log  ＋",
    },
    "log_panel_expanded": {
        "zh": "运行日志  −",
        "ja": "実行ログ  −",
        "en": "Execution Log  −",
    },
    "btn_clear_log": {
        "zh": "清空日志",
        "ja": "ログ消去",
        "en": "Clear Log",
    },

    # ── 运行配置面板 ──
    "settings_title": {
        "zh": "运行配置",
        "ja": "実行設定",
        "en": "Configuration",
    },
    "settings_sub": {
        "zh": "常用设置会在此设备上自动保存",
        "ja": "設定は本デバイスに自動保存されます",
        "en": "Settings are saved automatically",
    },
    "label_lang": {
        "zh": "界面语言",
        "ja": "表示言語",
        "en": "Language",
    },
    "label_chunk": {
        "zh": "显存分块",
        "ja": "VRAM 分割サイズ",
        "en": "VRAM Chunking",
    },
    "chunk_standard": {
        "zh": "标准 · 6–8 GB（6000 帧）",
        "ja": "標準 · 6–8 GB（6000 フレーム）",
        "en": "Standard · 6–8 GB (6000 frames)",
    },
    "chunk_light": {
        "zh": "轻量 · 4 GB（3000 帧）",
        "ja": "軽量 · 4 GB（3000 フレーム）",
        "en": "Light · 4 GB (3000 frames)",
    },
    "chunk_fast": {
        "zh": "极速 · 12 GB+（12000 帧）",
        "ja": "高速 · 12 GB+（12000 フレーム）",
        "en": "Fast · 12 GB+ (12000 frames)",
    },
    "chunk_full": {
        "zh": "全长 · 24 GB+（单块）",
        "ja": "全編 · 24 GB+（単一ブロック）",
        "en": "Full · 24 GB+ (Single chunk)",
    },
    "label_end_extend": {
        "zh": "字幕驻留延长",
        "ja": "字幕表示の延長",
        "en": "Subtitle Dwell Extension",
    },
    "label_sec": {
        "zh": "秒",
        "ja": "秒",
        "en": "sec",
    },
    "check_ono_gate": {
        "zh": "剔除脱落拟声词 (Ono-Gate)",
        "ja": "脱落擬音語の除去 (Ono-Gate)",
        "en": "Filter Dropped Onomatopoeia (Ono-Gate)",
    },
    "adv_expand": {
        "zh": "高级设置  ＋",
        "ja": "詳細設定  ＋",
        "en": "Advanced Settings  ＋",
    },
    "adv_collapse": {
        "zh": "高级设置  −",
        "ja": "詳細設定  −",
        "en": "Advanced Settings  −",
    },
    "label_device": {
        "zh": "计算设备",
        "ja": "計算デバイス",
        "en": "Compute Device",
    },
    "label_custom_chunk": {
        "zh": "自定义切片帧数",
        "ja": "カスタム分割フレーム数",
        "en": "Custom Chunk Frames",
    },
    "btn_start": {
        "zh": "开始对齐   →",
        "ja": "アライメント開始   →",
        "en": "Start Alignment   →",
    },
    "btn_start_with_count": {
        "zh": "开始对齐   →   {count}",
        "ja": "アライメント開始   →   {count}",
        "en": "Start Alignment   →   {count}",
    },
    "btn_running": {
        "zh": "正在对齐…",
        "ja": "アライメント処理中…",
        "en": "Aligning…",
    },
    "btn_show_guide": {
        "zh": "查看台本格式指南",
        "ja": "台本フォーマット規約を見る",
        "en": "View Script Format Guide",
    },

    # ── 状态展示 ──
    "state_title": {
        "zh": "当前状态",
        "ja": "現在の状態",
        "en": "Current Status",
    },
    "status_idle": {
        "zh": "等待导入台本与音频",
        "ja": "台本と音声のインポート待機中",
        "en": "Waiting for script & audio files",
    },
    "status_added_files": {
        "zh": "已添加 {added} 项 · 队列共 {total} 项",
        "ja": "{added} 件追加 · キュー合計 {total} 件",
        "en": "Added {added} items · Total {total} in queue",
    },
    "status_files_already": {
        "zh": "这些文件已在任务队列中",
        "ja": "これらのファイルは既に登録されています",
        "en": "Files are already in task queue",
    },
    "status_queue_cleared": {
        "zh": "队列已清空 · 等待导入文件",
        "ja": "キューを消去しました · ファイル待機中",
        "en": "Queue cleared · Waiting for files",
    },
    "status_queue_remaining": {
        "zh": "队列剩余 {count} 项",
        "ja": "キュー残り {count} 件",
        "en": "{count} items remaining in queue",
    },
    "status_updated_audio_b": {
        "zh": "已更新视角 B 音频",
        "ja": "視点 B 音声を更新しました",
        "en": "Updated Perspective B audio",
    },
    "status_requeued": {
        "zh": "已重新加入待处理：{name}",
        "ja": "再待機に追加：{name}",
        "en": "Re-queued: {name}",
    },
    "status_preparing": {
        "zh": "准备处理 {count} 项任务",
        "ja": "{count} 件のタスク処理を準備中",
        "en": "Preparing to process {count} tasks",
    },
    "status_progress_step": {
        "zh": "{current}/{total} · {message}",
        "ja": "{current}/{total} · {message}",
        "en": "{current}/{total} · {message}",
    },
    "status_all_done": {
        "zh": "处理完成 · 成功 {succeeded} · 失败 {failed}",
        "ja": "処理完了 · 成功 {succeeded} · 失敗 {failed}",
        "en": "Completed · Succeeded {succeeded} · Failed {failed}",
    },
    "status_interrupted": {
        "zh": "任务中断 · {error}",
        "ja": "タスク中断 · {error}",
        "en": "Interrupted · {error}",
    },

    # ── 右键上下文菜单 ──
    "menu_open_folder": {
        "zh": "打开文件所在文件夹",
        "ja": "ファイルの保存先フォルダを開く",
        "en": "Open Containing Folder",
    },
    "menu_open_script": {
        "zh": "打开台本",
        "ja": "台本ファイルを開く",
        "en": "Open Script File",
    },
    "menu_view_lrc": {
        "zh": "查看 LRC 字幕",
        "ja": "LRC 字幕を確認",
        "en": "View LRC Subtitle",
    },
    "menu_view_srt": {
        "zh": "查看 SRT 字幕",
        "ja": "SRT 字幕を確認",
        "en": "View SRT Subtitle",
    },
    "menu_pick_audio_b": {
        "zh": "指定 / 更换视角 B 音频…",
        "ja": "視点 B 音声を指定・変更…",
        "en": "Specify / Change Perspective B Audio…",
    },
    "menu_retry": {
        "zh": "重新对齐此项",
        "ja": "このタスクを再実行",
        "en": "Re-align this task",
    },
    "menu_remove": {
        "zh": "从队列移除",
        "ja": "キューから削除",
        "en": "Remove from queue",
    },

    # ── 弹窗对话框 ──
    "dlg_select_files_title": {
        "zh": "选择台本与音频文件",
        "ja": "台本と音声ファイルを選択",
        "en": "Select Script and Audio Files",
    },
    "dlg_filter_supported": {
        "zh": "支持的台本与音频",
        "ja": "対応する台本および音声ファイル",
        "en": "Supported Scripts & Audio Files",
    },
    "dlg_filter_all": {
        "zh": "所有文件",
        "ja": "すべてのファイル",
        "en": "All Files",
    },
    "dlg_pick_b_title": {
        "zh": "为 {name} 指定视角 B 音频",
        "ja": "{name} の視点 B 音声を選択",
        "en": "Select Perspective B audio for {name}",
    },
    "dlg_filter_audio": {
        "zh": "音频文件",
        "ja": "音声ファイル",
        "en": "Audio Files",
    },
    "msg_no_subtitle_title": {
        "zh": "字幕尚未生成",
        "ja": "字幕がまだ生成されていません",
        "en": "Subtitles Not Generated",
    },
    "msg_no_subtitle_text": {
        "zh": "该任务还没有 {extension} 文件。",
        "ja": "このタスクにはまだ {extension} ファイルがありません。",
        "en": "This task does not have a {extension} file yet.",
    },
    "msg_param_error_title": {
        "zh": "参数格式有误",
        "ja": "パラメータの形式が不正です",
        "en": "Invalid Parameters",
    },
    "msg_param_error_text": {
        "zh": "驻留延长需为非负数字，切片帧数需为正整数。",
        "ja": "表示延長は0以上の数値、分割フレーム数は正の整数で指定してください。",
        "en": "Dwell extension must be non-negative, and chunk frames must be a positive integer.",
    },
    "msg_empty_queue_title": {
        "zh": "任务队列为空",
        "ja": "タスクキューが空です",
        "en": "Task Queue Empty",
    },
    "msg_empty_queue_text": {
        "zh": "请先添加同时包含台本和音频的有效任务。",
        "ja": "台本と音声の両方が揃ったタスクを追加してください。",
        "en": "Please add valid tasks with paired script and audio files.",
    },
    "msg_missing_persp_b_title": {
        "zh": "部分任务缺少视角 B",
        "ja": "一部のタスクに視点 B 音声がありません",
        "en": "Perspective B Missing for Some Tasks",
    },
    "msg_missing_persp_b_text": {
        "zh": "以下任务将按单音频处理：\n\n{names}\n\n继续执行吗？",
        "ja": "以下のタスクは単一音声として処理されます：\n\n{names}\n\n続行しますか？",
        "en": "The following tasks will be processed as single-audio:\n\n{names}\n\nDo you want to proceed?",
    },

    # ── 台本指南对话框 ──
    "guide_dlg_window_title": {
        "zh": "台本格式指南 · FA-ASMR",
        "ja": "台本フォーマット規約 · FA-ASMR",
        "en": "Script Format Guide · FA-ASMR",
    },
    "guide_dlg_heading": {
        "zh": "台本格式指南",
        "ja": "台本フォーマット規約",
        "en": "Script Format Guide",
    },
    "guide_dlg_subheading": {
        "zh": "FA-ASMR 台本格式与匹配规范",
        "ja": "FA-ASMR 台本フォーマットと命名照合仕様",
        "en": "FA-ASMR Script Formatting & Pairing Specifications",
    },
    "guide_btn_copy": {
        "zh": "复制规范文本",
        "ja": "規約テキストをコピー",
        "en": "Copy Guide Text",
    },
    "guide_btn_copied": {
        "zh": "✓  已复制",
        "ja": "✓  コピー完了",
        "en": "✓  Copied",
    },
    "guide_btn_done": {
        "zh": "完成",
        "ja": "閉じる",
        "en": "Done",
    },
}

# ══════════════════════════════════════════════════════════════════════════
# 台本格式指南长文本
# ══════════════════════════════════════════════════════════════════════════

GUIDE_TEXTS: Dict[str, str] = {
    "zh": """══════════════════════════════════════════════════════
               FA-ASMR 台本格式指南
══════════════════════════════════════════════════════

【一、文件要求与命名配对】
• 文本编码：必须保存为 UTF-8 编码纯文本（.txt 文件）。
• 同名匹配：台本文件名需与音频文件名一致（如 01.txt 与 01.flac/wav），直接拖入自动配对。
• 文件名角色声明：支持在台本文件名尾部附加角色标签，如：
  01_话名_〖澄华〗〖天音〗.txt
  程序会自动剥除说话人后缀并与音频精准配对。

══════════════════════════════════════════════════════
【二、台本核心处理要求】
══════════════════════════════════════════════════════
• 必须清除非发音部分：
  提前删去台本中所有不发音的内容，包括括号内的动作提示、背景设定、场景号、SE 音效等。
  （例如：（SE：雨音）、（深呼吸）、※此处叹气 等不发音文字，必须清除）

══════════════════════════════════════════════════════
【三、不同模式的台本规范】
══════════════════════════════════════════════════════
1. 顺序对齐模式 (发音与台本全局严格按序对应)：
   • 核心假设为整篇音频的发音顺序与台本文字顺序严格一致、单向推进。
   • 纯台词文本或带有 〖角色名〗 标记均可，输出字幕会自动保留说话人标签。

2. 多人分角色模式 (单角色顺序不变 · 角色间可交错/同时说话)：
   • 适用于多角色对话中存在抢话、插话、颠倒乱序或时间轴交错重叠的场景。
   • 台本必须使用 〖角色名〗 标记各个角色（如 〖美雨〗、〖玲奈〗），全篇同一角色名称保持一致。
   • 各角色内部说的话顺序保持单向推进即可，角色之间的交错抢话由引擎自动分流并互斥合并。

3. 双视角联合模式：
   • 针对同一场景的 A / B 两份不同机位或视角音频（如 01A.wav 与 01B.wav），两份音频共用同一份台本即可。

══════════════════════════════════════════════════════
【四、输出产物】
══════════════════════════════════════════════════════
• 对齐完成后在音频同级目录下输出同名标准 .lrc 与 .srt 字幕文件。
• 时间戳采用标准格式 [MM:SS.CC]，紧随台词之后的独立空时间戳行表示发音结束时间（下轴点）。""",

    "ja": """══════════════════════════════════════════════════════
               FA-ASMR 台本フォーマット規約
══════════════════════════════════════════════════════

【1. ファイル要件と自動ペアリング】
• 文字コード：必ず UTF-8 形式のプレーンテキスト（.txt）で保存してください。
• 同名ペアリング：台本と音声のファイル名を一致させると（例: 01.txt と 01.wav/flac）、ドラッグ＆ドロップで自動照合されます。
• ファイル名での話者指定：台本ファイル名の末尾に話者タグを付与可能（例: 01_第1話_〖澄華〗〖天音〗.txt）。
  ツールが話者タグを自動認識・除外して音声と正確に照合します。

══════════════════════════════════════════════════════
【2. 台本クレンジングの重要事項】
══════════════════════════════════════════════════════
• 非発話テキストの事前除去：
  台本内の「発音されない文字」はあらかじめすべて削除してください。
  （例：（SE：雨の音）、（深呼吸）、※ここでため息、情景描写などはすべて削除必須です）

══════════════════════════════════════════════════════
【3. 各モードの台本仕様】
══════════════════════════════════════════════════════
1. シーケンシャルモード（音声と台本が完全に単方向順序一致）：
   • 音声の発声順序と台本テキストの出現順序が完全に一致する標準モードです。
   • タグなしのプレーン台詞、または 〖話者名〗 タグ付きのいずれにも対応（字幕出力時に話者タグを自動保持）。

2. 複数話者モード（話者内順序維持 · 話者間の会話重複・交錯対応）：
   • 会話の割り込み、言い淀み、重なり、順序前後がある複数キャラクター作品向け。
   • 台本各行で 〖話者名〗（例: 〖美雨〗、〖玲奈〗）を明記してください。
   • 同一話者の台詞順序が維持されていれば、話者間の時間交錯はエンジンが自動的に分離対齐し統合します。

3. デュアル視点モード：
   • 同一シーンを異なる2つのマイク/視点で収録した音声（例: 01A.wav と 01B.wav）向け。台本は1つで共有可能です。

══════════════════════════════════════════════════════
【4. 出力生成物】
══════════════════════════════════════════════════════
• アライメント完了後、音声ファイルと同階層に同名の標準 .lrc および .srt 字幕が出力されます。
• タイムスタンプ形式は [MM:SS.CC]、台詞直後の独立タイムスタンプ行は発声終了時刻を表します。""",

    "en": """══════════════════════════════════════════════════════
               FA-ASMR Script Format Guide
══════════════════════════════════════════════════════

[1. File Requirements & Pairing]
• Encoding: Must be saved as plain UTF-8 text (.txt).
• Matching: Script and audio files with the same stem (e.g. 01.txt and 01.wav/flac) are automatically paired when dragged in.
• Speaker tag suffix: Scripts can specify speakers at the end of the filename (e.g. 01_Track_〖Kasumi〗〖Amane〗.txt).
  The program automatically strips speaker brackets to match with the corresponding audio.

══════════════════════════════════════════════════════
[2. Script Preprocessing Essentials]
══════════════════════════════════════════════════════
• Remove all non-spoken texts:
  Please delete all non-verbal contents before alignment, such as parenthetical stage directions, sound effects (SE), and scene notes.
  (e.g., (SE: Rain), (Deep breath), *sighs here* must be strictly removed).

══════════════════════════════════════════════════════
[3. Mode Specifications]
══════════════════════════════════════════════════════
1. Sequential Mode (Strict monotonic order):
   • Audio strictly matches script text monotonically.
   • Plain text or lines tagged with 〖Speaker Name〗 are both supported (tags preserved in subtitles). Fastest and most accurate.

2. Multi-Speaker Mode (Monotonic per speaker · Interleaved/overlapping speech):
   • Ideal for multiple character dialogues with interruptions, crosstalk, or overlapping voices.
   • Script must specify speakers using 〖Speaker Name〗 tags (e.g., 〖Miu〗, 〖Reina〗).
   • As long as each character speaks sequentially, interleaved dialogue is aligned and merged automatically.

3. Dual-Perspective Mode:
   • For dual-angle or binaural multi-mic recordings of the same scene (e.g. 01A.wav and 01B.wav). Shares one script.

══════════════════════════════════════════════════════
[4. Output Products]
══════════════════════════════════════════════════════
• Upon completion, standard .lrc and .srt subtitle files are generated in the same directory as the audio.
• Timestamps follow the [MM:SS.CC] format, with standalone trailing timestamps indicating speech end times.""",
}


class I18nManager:
    """全局语言状态管理器。"""

    def __init__(self, default_lang: str = "zh"):
        self._current_lang = default_lang if default_lang in LANG_MAP else DEFAULT_LANG

    @property
    def current_lang(self) -> str:
        return self._current_lang

    def set_lang(self, lang: str) -> bool:
        if lang in LANG_MAP:
            self._current_lang = lang
            return True
        return False

    def tr(self, key: str, **kwargs) -> str:
        """获取翻译文本。若当前语言无对应键，回退到简体中文或键名本身。"""
        entry = TRANSLATIONS.get(key)
        if not entry:
            return key
        text = entry.get(self._current_lang)
        if text is None:
            text = entry.get(DEFAULT_LANG, key)
        if kwargs:
            try:
                return text.format(**kwargs)
            except Exception:
                return text
        return text

    def get_guide_text(self) -> str:
        return GUIDE_TEXTS.get(self._current_lang, GUIDE_TEXTS[DEFAULT_LANG])


# 全局单例
I18N = I18nManager()
tr = I18N.tr
