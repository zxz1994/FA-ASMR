# -*- coding: utf-8 -*-
"""fa_asmr_cli.py 参数校验、批量配对预检 (--dry-run) 与 JSON 输出回归测试。"""
from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest

import fa_asmr_cli as cli


class TestCli(unittest.TestCase):
    def _run_cli(self, argv):
        buf_out = io.StringIO()
        buf_err = io.StringIO()
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            code = cli.main(argv)
        return code, buf_out.getvalue(), buf_err.getvalue()

    def test_invalid_argument_combinations(self):
        code, _, _ = self._run_cli([])
        self.assertEqual(code, 2)

        code, out, _ = self._run_cli(["--audio", "a.wav", "--json"])
        self.assertEqual(code, 2)
        payload = json.loads(out)
        self.assertFalse(payload["ok"])

        with tempfile.TemporaryDirectory() as td:
            aud = os.path.join(td, "01.wav")
            txt = os.path.join(td, "01.txt")
            open(aud, "wb").close()
            with open(txt, "w", encoding="utf-8") as f:
                f.write("テスト\n")
            code, out, _ = self._run_cli([
                "--audio", aud, "--script", txt, "--mode", "multi_persp", "--json",
            ])
            self.assertEqual(code, 2)
            self.assertIn("audio-b", json.loads(out)["error"])

    def test_batch_dry_run_auto_detects_single_and_multi_speaker(self):
        with tempfile.TemporaryDirectory() as td:
            aud1 = os.path.join(td, "01_multi.wav")
            txt1 = os.path.join(td, "01_multi.txt")
            aud2 = os.path.join(td, "02_single.wav")
            txt2 = os.path.join(td, "02_single.txt")
            extra = os.path.join(td, "03_orphan.wav")
            json_out = os.path.join(td, "summary.json")

            for p in (aud1, aud2, extra):
                open(p, "wb").close()
            # CP932 + 独立 【角色名】 高频标题行（>=2 角色各出现 >=2 次）
            with open(txt1, "wb") as f:
                f.write(
                    "【澄華】\nおはよう！\n【天音】\nまたね\n【澄華】\nうん\n【天音】\nバイバイ\n".encode("cp932")
                )
            with open(txt2, "w", encoding="utf-8") as f:
                f.write("ひとりごとだよ\n")

            code, out, _ = self._run_cli([
                "--batch", td,
                "--dry-run",
                "--json",
                "--json-out", json_out,
            ])
            self.assertEqual(code, 0)
            payload = json.loads(out)
            self.assertTrue(payload["ok"])
            self.assertTrue(payload["dry_run"])
            self.assertEqual(payload["matched_count"], 2)
            self.assertEqual(payload["unmatched_count"], 1)

            by_name = {t["name"]: t for t in payload["tasks"]}
            self.assertEqual(by_name["01_multi"]["mode"], "multi_speaker")
            self.assertEqual(by_name["01_multi"]["script_info"]["encoding"], "cp932")
            self.assertEqual(by_name["01_multi"]["script_info"]["detected_speakers"], ["澄華", "天音"])
            self.assertEqual(by_name["02_single"]["mode"], "single")

            # 校验 --json-out 落盘一致
            self.assertTrue(os.path.isfile(json_out))
            with open(json_out, "r", encoding="utf-8") as f:
                saved = json.load(f)
            self.assertEqual(saved, payload)

    def test_multi_persp_auto_dry_run(self):
        with tempfile.TemporaryDirectory() as td:
            for name in ("01_A.wav", "01_B.wav"):
                open(os.path.join(td, name), "wb").close()
            with open(os.path.join(td, "01.txt"), "w", encoding="utf-8") as f:
                f.write("〖A〗こんにちは\n〖B〗こんばんは\n")

            # 不显式传 --mode，验证 --mode auto 自动识别双视角配对
            code, out, _ = self._run_cli([
                "--batch", td,
                "--dry-run",
                "--json",
            ])
            self.assertEqual(code, 0)
            payload = json.loads(out)
            self.assertEqual(payload["matched_count"], 1)
            self.assertEqual(payload["tasks"][0]["mode"], "multi_persp")
            self.assertTrue(payload["tasks"][0]["audio"].endswith("01_A.wav"))
            self.assertTrue(payload["tasks"][0]["audio_b"].endswith("01_B.wav"))

    def test_verify_script_missing_file(self):
        code, out, _ = self._run_cli(["--verify-script", "nonexistent_script_xyz.txt", "--json"])
        self.assertEqual(code, 2)
        self.assertFalse(json.loads(out)["ok"])


if __name__ == "__main__":
    unittest.main()
