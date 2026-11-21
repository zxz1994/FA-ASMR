# -*- coding: utf-8 -*-
"""错误处理与防静默失败回归测试。

覆盖：
- advanced_config.py 配置语法/类型/未知键告警（走 stderr，不污染 stdout）
- fa_asmr_cli.py 参数校验、--json-out 独立落盘、--verify-script 空台本与结构化异常捕获
- fa_asmr_pipeline.py 坏 UTF-8 BOM 识别、缺失文件/非法参数/显式缺失 LoRA/空台本前置报错
- furigana/fa_asmr_converter.py _get_heteronyms 失败不污染缓存、_ensure_pykakasi 抛异常而非 sys.exit、fugashi 退化告警
- align_model.py _resolve_device / LoraAdapter 失败不改脏模型且抛错 / compute_emission 空波形校验
"""
from __future__ import annotations

import builtins
import contextlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import advanced_config as adv
import fa_asmr_cli as cli
import fa_asmr_pipeline as pipe


class TestAdvancedConfigWarnings(unittest.TestCase):
    def tearDown(self):
        adv.reset_cache()

    def test_invalid_json_emits_stderr_warning_and_records_last_warnings(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = os.path.join(td, "advanced_config.json")
            with open(cfg_path, "w", encoding="utf-8") as f:
                f.write("{invalid json")
            adv.reset_cache()
            buf_out, buf_err = io.StringIO(), io.StringIO()
            with mock.patch.object(adv, "config_path", return_value=cfg_path), \
                    contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                data = adv.load()
            self.assertEqual(data, adv.DEFAULTS)
            self.assertEqual(buf_out.getvalue(), "")
            self.assertIn("解析", buf_err.getvalue())
            self.assertTrue(adv.LAST_WARNINGS)

    def test_unknown_key_and_invalid_field_types_warn(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = os.path.join(td, "advanced_config.json")
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump({
                    "cpu_threads": True,
                    "quantize_int8": "yes",
                    "cpu_chunk_cap": 10,
                    "typo_option": 123,
                }, f)
            adv.reset_cache()
            buf_out, buf_err = io.StringIO(), io.StringIO()
            with mock.patch.object(adv, "config_path", return_value=cfg_path), \
                    contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                data = adv.load()
            self.assertEqual(data, adv.DEFAULTS)
            self.assertEqual(buf_out.getvalue(), "")
            self.assertGreaterEqual(len(adv.LAST_WARNINGS), 4)
            self.assertIn("typo_option", buf_err.getvalue())


class TestCliErrorReporting(unittest.TestCase):
    def _run_cli(self, argv):
        buf_out, buf_err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            code = cli.main(argv)
        return code, buf_out.getvalue(), buf_err.getvalue()

    def test_json_out_written_on_error_even_without_json_flag(self):
        with tempfile.TemporaryDirectory() as td:
            jout = os.path.join(td, "sub", "err.json")
            code, out, err = self._run_cli(["--audio", "only_audio.wav", "--json-out", jout])
            self.assertEqual(code, 2)
            self.assertEqual(out, "")
            self.assertIn("[ERROR]", err)
            self.assertTrue(os.path.isfile(jout))
            with open(jout, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertFalse(payload["ok"])
            self.assertIn("--script", payload["error"])

    def test_invalid_numeric_and_missing_custom_lora_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            aud = os.path.join(td, "01.wav")
            txt = os.path.join(td, "01.txt")
            open(aud, "wb").close()
            with open(txt, "w", encoding="utf-8") as f:
                f.write("テスト\n")

            code, out, _ = self._run_cli([
                "--audio", aud, "--script", txt, "--extend", "-0.5", "--json",
            ])
            self.assertEqual(code, 2)
            self.assertIn("负数", json.loads(out)["error"])

            code, out, _ = self._run_cli([
                "--audio", aud, "--script", txt, "--vram-chunk", "0", "--json",
            ])
            self.assertEqual(code, 2)
            self.assertIn("vram-chunk", json.loads(out)["error"])

            code, out, _ = self._run_cli([
                "--audio", aud, "--script", txt,
                "--lora", os.path.join(td, "missing_lora.pt"), "--json",
            ])
            self.assertEqual(code, 2)
            self.assertIn("LoRA", json.loads(out)["error"])

    def test_same_audio_ab_and_output_collisions_and_unsupported_batch_file(self):
        with tempfile.TemporaryDirectory() as td:
            aud = os.path.join(td, "01.wav")
            txt = os.path.join(td, "01.txt")
            pdf = os.path.join(td, "readme.pdf")
            not_a_dir = os.path.join(td, "existing_file.txt")
            open(aud, "wb").close()
            open(pdf, "wb").close()
            with open(txt, "w", encoding="utf-8") as f:
                f.write("おはよう\n")
            with open(not_a_dir, "w", encoding="utf-8") as f:
                f.write("x")

            # 1. --audio 与 --audio-b 同一文件
            code, out, _ = self._run_cli([
                "--audio", aud, "--audio-b", aud, "--script", txt, "--json",
            ])
            self.assertEqual(code, 2)
            self.assertIn("同一个文件", json.loads(out)["error"])

            # 2. --output-dir 指向已存在的普通文件
            code, out, _ = self._run_cli([
                "--audio", aud, "--script", txt, "--output-dir", not_a_dir, "--json",
            ])
            self.assertEqual(code, 2)
            self.assertIn("普通文件", json.loads(out)["error"])

            # 3. 不同目录同名双视角 + --output-dir 路径冲突
            sub_a = os.path.join(td, "A")
            sub_b = os.path.join(td, "B")
            out_d = os.path.join(td, "out")
            os.makedirs(sub_a)
            os.makedirs(sub_b)
            aud_a = os.path.join(sub_a, "track01.wav")
            aud_b = os.path.join(sub_b, "track01.wav")
            open(aud_a, "wb").close()
            open(aud_b, "wb").close()
            code, out, _ = self._run_cli([
                "--audio", aud_a, "--audio-b", aud_b, "--script", txt,
                "--output-dir", out_d, "--json",
            ])
            self.assertEqual(code, 2)
            self.assertIn("互相覆盖", json.loads(out)["error"])

            # 4. --batch 显式传入不支持文件（如 .pdf）应出现在 unmatched 中而不是静默消失
            code, out, _ = self._run_cli([
                "--batch", aud, txt, pdf, "--dry-run", "--json",
            ])
            self.assertEqual(code, 0)
            payload = json.loads(out)
            self.assertEqual(payload["matched_count"], 1)
            self.assertTrue(any(u["type"] == "不支持的格式" and u["file"].endswith("readme.pdf") for u in payload["unmatched"]))

            # 5. _resolve_tasks 读取台本失败应返回结构化 JSON 错误而不是裸栈崩溃
            with mock.patch.object(cli, "_detect_task_mode", side_effect=PermissionError("access denied")):
                code, out, _ = self._run_cli(["--audio", aud, "--script", txt, "--dry-run", "--json"])
            self.assertEqual(code, 2)
            payload = json.loads(out)
            self.assertFalse(payload["ok"])
            self.assertIn("PermissionError", payload["error"])

    def test_dry_run_surfaces_empty_and_corrupted_encoding_warnings(self):
        with tempfile.TemporaryDirectory() as td:
            aud = os.path.join(td, "01.wav")
            txt = os.path.join(td, "01.txt")
            open(aud, "wb").close()
            with open(txt, "wb") as f:
                f.write(b"\xef\xbb\xbf\xff\xfe\xfd")
            code, out, _ = self._run_cli(["--audio", aud, "--script", txt, "--dry-run", "--json"])
            self.assertEqual(code, 0)
            payload = json.loads(out)
            si = payload["tasks"][0]["script_info"]
            self.assertEqual(si["encoding"], "utf-8-sig-fallback")
            self.assertTrue(any("编码异常" in w for w in si["warnings"]))

    def test_verify_script_runtime_error_returns_structured_json(self):
        with tempfile.TemporaryDirectory() as td:
            txt = os.path.join(td, "01.txt")
            jout = os.path.join(td, "verify.json")
            with open(txt, "w", encoding="utf-8") as f:
                f.write("おはよう\n")
            with mock.patch.object(pipe, "_ensure_runtime_ready", side_effect=RuntimeError("boom-runtime")):
                code, out, _ = self._run_cli(["--verify-script", txt, "--json", "--json-out", jout])
            self.assertEqual(code, 1)
            payload = json.loads(out)
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["error_type"], "RuntimeError")
            self.assertIn("boom-runtime", payload["error"])
            self.assertTrue(os.path.isfile(jout))


class TestPipelineValidation(unittest.TestCase):
    def test_corrupted_utf8_bom_flagged_as_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "bad_bom.txt")
            with open(p, "wb") as f:
                f.write(b"\xef\xbb\xbf\xff\xfe")
            text, enc = pipe.read_text_auto_encoding(p)
            self.assertEqual(enc, "utf-8-sig-fallback")
            self.assertIn("\ufffd", text)

    def test_run_pipeline_rejects_missing_and_invalid_inputs_early(self):
        with tempfile.TemporaryDirectory() as td:
            aud = os.path.join(td, "01.wav")
            txt = os.path.join(td, "01.txt")
            open(aud, "wb").close()
            with open(txt, "w", encoding="utf-8") as f:
                f.write("   \n\n")

            fake_align = types.SimpleNamespace(
                DEFAULT_LORA_PATH="",
                clear_model_cache=lambda: None,
            )
            fake_hn = types.SimpleNamespace()

            with mock.patch.object(pipe, "_ensure_runtime_ready"), \
                    mock.patch.dict(sys.modules, {"align": fake_align, "haruraw2norm": fake_hn}):
                # 1. 缺失台本
                summary = {}
                ok, _, err = pipe.run_pipeline(
                    os.path.join(td, "no_such.txt"), aud, None,
                    log_callback=None, progress_callback=None,
                    summary_out=summary, echo_stdout=False,
                )
                self.assertFalse(ok)
                self.assertEqual(summary.get("error_type"), "FileNotFoundError")
                self.assertIn("台本文件不存在", err)

                # 2. 负数行尾延长
                summary = {}
                ok, _, err = pipe.run_pipeline(
                    txt, aud, None,
                    log_callback=None, progress_callback=None,
                    line_end_extend=-1.0,
                    summary_out=summary, echo_stdout=False,
                )
                self.assertFalse(ok)
                self.assertEqual(summary.get("error_type"), "ValueError")
                self.assertIn("负数", err)

                # 3. 空台本（通过 mock AutoRuby 避免加载重型模型）
                class _DummyRuby:
                    _ann = None
                    def annotate(self, s):
                        return s
                    def release_gpu(self):
                        pass

                summary = {}
                with mock.patch.object(pipe, "AutoRuby", _DummyRuby):
                    ok, _, err = pipe.run_pipeline(
                        txt, aud, None,
                        log_callback=None, progress_callback=None,
                        summary_out=summary, echo_stdout=False,
                    )
                self.assertFalse(ok)
                self.assertEqual(summary.get("error_type"), "ValueError")
                self.assertIn("有效文本行", err)

                # 4. 双视角 audio_path 与 audio_path_b 指向同一文件
                summary = {}
                ok, _, err = pipe.run_pipeline(
                    txt, aud, None,
                    log_callback=None, progress_callback=None,
                    audio_path_b=aud,
                    summary_out=summary, echo_stdout=False,
                )
                self.assertFalse(ok)
                self.assertEqual(summary.get("error_type"), "ValueError")
                self.assertIn("同一个音频文件", err)

                # 5. output_dir 指向已存在的普通文件
                summary = {}
                ok, _, err = pipe.run_pipeline(
                    txt, aud, None,
                    log_callback=None, progress_callback=None,
                    output_dir=txt,
                    summary_out=summary, echo_stdout=False,
                )
                self.assertFalse(ok)
                self.assertEqual(summary.get("error_type"), "ValueError")
                self.assertIn("普通文件", err)


class TestFuriganaAndModelErrors(unittest.TestCase):
    def test_heteronyms_failure_does_not_poison_cache(self):
        furigana_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "furigana")
        if furigana_dir not in sys.path:
            sys.path.insert(0, furigana_dir)
        from fa_asmr_converter import RubyAnnotator

        ann = object.__new__(RubyAnnotator)
        ann._heteronyms = None

        real_import = builtins.__import__
        def _fail_yomikata(name, *args, **kwargs):
            if name == "yomikata" or name.startswith("yomikata."):
                raise ImportError("mock missing yomikata")
            return real_import(name, *args, **kwargs)

        with mock.patch.object(builtins, "__import__", side_effect=_fail_yomikata):
            with self.assertRaises(RuntimeError):
                ann._get_heteronyms()
            # 关键：第二次调用仍应抛 RuntimeError，绝不能返回空 set 静默跳过 dBert
            self.assertIsNone(ann._heteronyms)
            with self.assertRaises(RuntimeError):
                ann._get_heteronyms()

    def test_missing_pykakasi_raises_runtime_error_not_sysexit(self):
        furigana_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "furigana")
        if furigana_dir not in sys.path:
            sys.path.insert(0, furigana_dir)
        from fa_asmr_converter import RubyAnnotator

        ann = object.__new__(RubyAnnotator)
        ann._pykakasi = None
        real_import = builtins.__import__

        def _fail_pykakasi(name, *args, **kwargs):
            if name == "pykakasi":
                raise ImportError("mock missing pykakasi")
            return real_import(name, *args, **kwargs)

        with mock.patch.object(builtins, "__import__", side_effect=_fail_pykakasi):
            with self.assertRaises(RuntimeError):
                ann._ensure_pykakasi()

    def test_fugashi_init_failure_logs_warning_even_when_not_verbose(self):
        furigana_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "furigana")
        if furigana_dir not in sys.path:
            sys.path.insert(0, furigana_dir)
        from fa_asmr_converter import RubyAnnotator

        ann = object.__new__(RubyAnnotator)
        ann.verbose = False
        ann._tagger = None
        ann.fugashi_error = None
        ann._pykakasi = object()
        real_import = builtins.__import__

        def _fail_fugashi(name, *args, **kwargs):
            if name == "fugashi":
                raise RuntimeError("unidic broken")
            return real_import(name, *args, **kwargs)

        buf_out, buf_err = io.StringIO(), io.StringIO()
        with mock.patch.object(builtins, "__import__", side_effect=_fail_fugashi), \
                contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            ann._init_fugashi()
        self.assertIsNone(ann._tagger)
        self.assertIn("unidic broken", ann.fugashi_error)
        self.assertEqual(buf_out.getvalue(), "")
        self.assertIn("降级为 pykakasi", buf_err.getvalue())

    def test_align_model_device_and_lora_fail_fast(self):
        try:
            import torch
            import torch.nn as nn
            import align_model as am
        except ImportError:
            self.skipTest("当前环境未安装 torch/torchaudio")

        # 1. 非法设备名必须报错
        with self.assertRaises(ValueError):
            am._resolve_device("gpu")

        # 2. 指定 cuda 但无 CUDA 时必须报错
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                am._resolve_device("cuda")

        # 3. 损坏/缺字段的 LoRA 文件必须抛 RuntimeError，且不能提前改脏 w2v2 Linear 层
        class _DummyW2V2(nn.Module):
            def __init__(self):
                super().__init__()
                self.k_proj = nn.Linear(4, 4)
                self.aux = nn.Linear(4, 2)

        class _DummyBundleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = _DummyW2V2()

        with tempfile.TemporaryDirectory() as td:
            bad_lora = os.path.join(td, "bad.pt")
            torch.save({"only_garbage": 1}, bad_lora)
            m = _DummyBundleModel()
            adapter = am.LoraAdapter(m, lora_path=bad_lora)
            with self.assertRaises(RuntimeError):
                adapter.apply()
            # 确认未提前把 k_proj 替换成随机 SimpleLoraLinear
            self.assertIsInstance(m.model.k_proj, nn.Linear)
            self.assertNotIsInstance(m.model.k_proj, am.SimpleLoraLinear)

        # 4. 空波形必须提前报 ValueError
        with self.assertRaises(ValueError):
            am.compute_emission(torch.zeros((1, 0)), 16000, device="cpu")


class TestBootstrapAndUtilitiesErrors(unittest.TestCase):
    def test_torch_bootstrap_raises_runtime_error_when_headless_and_missing_torch(self):
        import torch_bootstrap as tb

        prev_done = tb.TORCH_DONE
        try:
            tb.TORCH_DONE = False
            real_import = builtins.__import__

            def _fail_torch(name, *args, **kwargs):
                if name in ("torch", "torchaudio"):
                    raise ImportError("mock missing torch")
                return real_import(name, *args, **kwargs)

            with mock.patch.object(builtins, "__import__", side_effect=_fail_torch), \
                    mock.patch.object(tb, "_write_bootstrap_log"), \
                    mock.patch.object(tb.subprocess, "run", side_effect=FileNotFoundError("no python")):
                with self.assertRaises(RuntimeError) as cm:
                    tb.ensure_torch(None)
                self.assertIn("未找到 Python", str(cm.exception))
        finally:
            tb.TORCH_DONE = prev_done

    def test_lrc_end_extend_rejects_negative_extend(self):
        import lrc_end_extend as ext

        buf_err = io.StringIO()
        with mock.patch.object(sys, "argv", ["lrc_end_extend.py", "dummy.lrc", "-e", "-1.0"]), \
                contextlib.redirect_stderr(buf_err):
            with self.assertRaises(SystemExit) as cm:
                ext.main()
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("负数", buf_err.getvalue())

    def test_i18n_logs_warning_on_missing_key_or_bad_format(self):
        import fa_asmr_i18n as i18n

        with self.assertLogs(i18n.logger, level="WARNING") as cm:
            self.assertEqual(i18n.tr("__nonexistent_key__"), "__nonexistent_key__")
            i18n.tr("status_all_done", wrong_kw=1)
        joined = "\n".join(cm.output)
        self.assertIn("__nonexistent_key__", joined)
        self.assertIn("模板格式化失败", joined)


if __name__ == "__main__":
    unittest.main()
