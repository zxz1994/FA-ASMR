# -*- coding: utf-8 -*-
"""模块化架构、fa_asmr 包布局、循环依赖消除、根入口约束及 UI 冒烟回归测试。"""

import ast
import os
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

EXPECTED_ROOT_PY = {
    "FA-ASMR_GUI.py",
    "fa_asmr_cli.py",
    "fa_asmr_spatial_sep.py",
    "lrc_end_extend.py",
    "build_in_ci.py",
    "prepare_embedded_torch.py",
    "_collect_stdlib.py",
}


def _ast_imported_modules(py_path: str) -> set[str]:
    with open(py_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=py_path)
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
            for alias in node.names:
                if alias.name != "*":
                    mods.add(f"{node.module}.{alias.name}")
    return mods


def _collect_fa_asmr_modules() -> dict[str, str]:
    modules: dict[str, str] = {}
    pkg_root = os.path.join(REPO_ROOT, "fa_asmr")
    for root, _, files in os.walk(pkg_root):
        for f in files:
            if not f.endswith(".py"):
                continue
            fp = os.path.join(root, f)
            rel = os.path.relpath(fp, REPO_ROOT).replace(os.sep, "/")
            mod = rel[:-3].replace("/", ".")
            if mod.endswith(".__init__"):
                mod = mod[:-9]
            modules[mod] = fp
    return modules


class TestModularArchitecture(unittest.TestCase):
    def test_root_directory_contains_only_external_and_ci_entrypoints(self):
        root_py = {
            f for f in os.listdir(REPO_ROOT)
            if f.endswith(".py") and os.path.isfile(os.path.join(REPO_ROOT, f))
        }
        self.assertEqual(root_py, EXPECTED_ROOT_PY)

    def test_clause_and_ono_filter_have_no_circular_import(self):
        import fa_asmr.text.clause as clause
        import fa_asmr.text.ono_base as ono_base
        import fa_asmr.text.ono_filter as ono_filter

        clause_imports = _ast_imported_modules(
            os.path.join(REPO_ROOT, "fa_asmr", "text", "clause.py")
        )
        self.assertFalse(
            any(m == "ono_filter" or m.endswith(".ono_filter") for m in clause_imports)
        )
        self.assertIs(ono_filter.is_clause_boundary, clause.is_clause_boundary)
        self.assertIs(ono_filter.is_easy_ono, ono_base.is_easy_ono)

    def test_no_two_node_import_cycles_across_root_and_package(self):
        py_files: dict[str, str] = {}
        for f in os.listdir(REPO_ROOT):
            fp = os.path.join(REPO_ROOT, f)
            if f.endswith(".py") and os.path.isfile(fp):
                py_files[os.path.splitext(f)[0]] = fp
        py_files.update(_collect_fa_asmr_modules())

        mod_names = set(py_files.keys())
        edges: dict[str, set[str]] = defaultdict(set)
        for src_mod, fp in py_files.items():
            for imp in _ast_imported_modules(fp):
                if imp in mod_names and imp != src_mod:
                    edges[src_mod].add(imp)

        cycles = []
        for a, bs in edges.items():
            for b in bs:
                if a in edges.get(b, set()) and a < b:
                    cycles.append((a, b))
        self.assertEqual(cycles, [], f"检测到双向循环导入: {cycles}")

    def test_cli_does_not_import_heavy_align_multispeaker(self):
        cli_imports = _ast_imported_modules(os.path.join(REPO_ROOT, "fa_asmr_cli.py"))
        self.assertNotIn("align_multispeaker", cli_imports)
        self.assertFalse(any(m.startswith("fa_asmr.aligners") for m in cli_imports))

    def test_package_root_relative_paths_resolve_to_repo_root(self):
        from fa_asmr.core import advanced_config
        from fa_asmr.pipeline import ruby, runner

        self.assertEqual(os.path.normcase(ruby.SCRIPT_DIR), os.path.normcase(REPO_ROOT))
        self.assertTrue(os.path.isdir(os.path.join(ruby.SCRIPT_DIR, "furigana")))
        self.assertEqual(os.path.normcase(runner.SCRIPT_DIR), os.path.normcase(REPO_ROOT))
        self.assertEqual(
            os.path.normcase(advanced_config.config_path()),
            os.path.normcase(os.path.join(REPO_ROOT, "advanced_config.json")),
        )

        try:
            from fa_asmr.aligners import align, align_model
        except ImportError:
            return
        self.assertEqual(
            os.path.normcase(str(align.ROOT_DIR)),
            os.path.normcase(REPO_ROOT),
        )
        self.assertEqual(
            os.path.normcase(str(align.DEFAULT_LORA_PATH)),
            os.path.normcase(os.path.join(REPO_ROOT, "lora", align.DEFAULT_LORA_NAME)),
        )
        # align_model 使用三层上跳解析到 REPO_ROOT
        am_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(align_model.__file__))))
        self.assertEqual(os.path.normcase(am_root), os.path.normcase(REPO_ROOT))

    def test_all_fa_asmr_modules_registered_in_spec_hiddenimports(self):
        spec_path = os.path.join(REPO_ROOT, "FA-ASMR.spec")
        with open(spec_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=spec_path)
        hidden: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "hiddenimports":
                        if isinstance(node.value, ast.List):
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                    hidden.add(elt.value)
        expected = set(_collect_fa_asmr_modules().keys()) | {
            "lrc_end_extend",
            "fa_asmr_cli",
            "fa_asmr_spatial_sep",
        }
        missing = sorted(expected - hidden)
        self.assertEqual(missing, [], f"缺失于 FA-ASMR.spec hiddenimports: {missing}")

    def test_backward_compatible_reexports_on_facades(self):
        from fa_asmr import ui as ui_pkg
        import fa_asmr.aligners.multiperspective as amp
        import fa_asmr.aligners.multispeaker as ams
        import fa_asmr.pipeline.runner as pipe
        import fa_asmr.ui.console as ui
        from fa_asmr.aligners import (
            multipersp_acoustic,
            multipersp_pipeline,
            multipersp_segments,
            multispeaker_acoustic,
            multispeaker_decoder,
            multispeaker_post,
            multispeaker_schedule,
        )
        from fa_asmr.pipeline import pairing, ruby, subtitles
        from fa_asmr.ui import matrix, widgets

        gui_expected = [
            "AUDIO_EXTS",
            "_SPEAKER_BRACKET_RE",
            "_SPEAKER_SEP",
            "_iter_speaker_marks",
            "_split_speaker_segments",
            "AutoRuby",
            "read_text_auto_encoding",
            "_fmt_lrc_time",
            "_lrc_time_to_cs",
            "build_standard_lrc",
            "_fmt_line",
            "build_standard_srt",
            "_PUNCT_CHARS",
            "_strip_orphan_punct",
            "_INLINE_RUBY_RE",
            "_strip_ruby_for_src",
            "_attach_source_spans",
            "_release_gpu",
            "run_pipeline",
            "_parse_speaker_suffix",
            "_PERSP_PATTERNS",
            "_extract_perspective_tag",
            "_clean_stem_for_matching",
            "auto_pair_files",
            "load_ruby_map",
            "resolve_ruby_map_for_script",
            "apply_ruby_overrides_to_line",
            "annotate_lines_with_overrides",
            "extract_ruby_glossary",
            "merge_ruby_glossaries",
            "auto_normalize_speaker_headers",
            "build_independent_multispeaker_srt",
            "split_unison_prefix_suffix_rl",
        ]
        for sym in gui_expected:
            self.assertTrue(hasattr(pipe, sym), f"fa_asmr.pipeline.runner 缺少导出: {sym}")

        self.assertIs(pipe.AutoRuby, ruby.AutoRuby)
        self.assertIs(pipe.auto_pair_files, pairing.auto_pair_files)
        self.assertIs(pipe.build_standard_lrc, subtitles.build_standard_lrc)
        self.assertIs(pipe.build_standard_srt, subtitles.build_standard_srt)

        self.assertIs(ui.enable_windows_dpi_awareness, widgets.enable_windows_dpi_awareness)
        self.assertIs(ui.TaskMatrixController, matrix.TaskMatrixController)
        self.assertIs(ui_pkg.build_ui, ui.build_ui)
        self.assertIs(ui_pkg.enable_windows_dpi_awareness, ui.enable_windows_dpi_awareness)

        self.assertIs(ams.MultiSpeakerCTCAligner, multispeaker_decoder.MultiSpeakerCTCAligner)
        self.assertIs(ams.extract_acoustic_context, multispeaker_acoustic.extract_acoustic_context)
        self.assertIs(ams.align_multispeaker_or_reordered, multispeaker_schedule.align_multispeaker_or_reordered)
        self.assertIs(ams.post_process_and_merge_streams, multispeaker_post.post_process_and_merge_streams)
        self.assertIs(ams.auto_normalize_speaker_headers, pairing.auto_normalize_speaker_headers)
        self.assertIs(ams.build_independent_multispeaker_srt, subtitles.build_independent_multispeaker_srt)
        self.assertIs(ams.split_unison_prefix_suffix_rl, subtitles.split_unison_prefix_suffix_rl)
        self.assertIs(ams._strip_orphan_punct, subtitles._strip_orphan_punct)

        self.assertIs(amp.detect_clean_audio_segments, multipersp_acoustic.detect_clean_audio_segments)
        self.assertIs(amp.segment_and_assign_home, multipersp_segments.segment_and_assign_home)
        self.assertIs(amp.run_multiperspective_pipeline, multipersp_pipeline.run_multiperspective_pipeline)
        self.assertIs(amp.align_multiperspective_joint, multipersp_pipeline.align_multiperspective_joint)

    def test_ui_smoke_build_add_mode_lang_and_close(self):
        import tkinter as tk
        from fa_asmr.pipeline import pairing
        import fa_asmr.ui.console as ui

        try:
            root = tk.Tk()
        except tk.TclError as e:
            self.skipTest(f"当前环境无可用 Tk 显示: {e}")
            return

        root.withdraw()
        with tempfile.TemporaryDirectory() as td:
            txt_p = os.path.join(td, "01_test.txt")
            wav_p = os.path.join(td, "01_test.wav")
            with open(txt_p, "w", encoding="utf-8") as f:
                f.write("こんにちは\n")
            open(wav_p, "wb").close()
            settings_p = os.path.join(td, "settings.json")
            ctx = SimpleNamespace(
                root=root,
                script_dir=REPO_ROOT,
                settings_file=settings_p,
                cuda_ok=False,
                has_dnd=False,
                dnd_files=None,
                pil=False,
                auto_pair_files=pairing.auto_pair_files,
                run_pipeline=lambda **kw: (True, None, None),
                lora_path="",
                release_gpu=lambda: None,
                is_running=lambda: bool(getattr(root, "_fa_running", False)),
                start_worker=lambda target: target(),
            )
            try:
                built_root = ui.build_ui(ctx)
                self.assertIs(built_root, root)
                self.assertTrue(callable(getattr(root, "_fa_set_running", None)))
                console = root._fa_ui

                console.matrix.add_files_to_list([txt_p, wav_p])
                self.assertEqual(console.matrix.get_queue_counts(), (1, 1, 0))

                for m in ("multi_speaker", "multi_persp", "single"):
                    console._select_mode(m)
                    self.assertEqual(console.mode_var.get(), m)

                for lang in ("ja", "en", "zh"):
                    console._on_lang_select(lang)
                    self.assertEqual(console.lang_var.get(), lang)

                console._on_close()
                self.assertTrue(os.path.isfile(settings_p))
            finally:
                try:
                    if root.winfo_exists():
                        root.destroy()
                except tk.TclError:
                    pass


if __name__ == "__main__":
    unittest.main()
