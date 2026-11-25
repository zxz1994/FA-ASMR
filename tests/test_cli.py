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

    def test_export_ruby_and_ruby_map_validation_errors(self):
        # --export-ruby 单独使用报错
        code, out, _ = self._run_cli(["--export-ruby", "--json"])
        self.assertEqual(code, 2)
        self.assertIn("--verify-script", json.loads(out)["error"])

        # --ruby-map 指向不存在文件报错
        with tempfile.TemporaryDirectory() as td:
            aud = os.path.join(td, "01.wav")
            txt = os.path.join(td, "01.txt")
            open(aud, "wb").close()
            with open(txt, "w", encoding="utf-8") as f:
                f.write("テスト\n")
            code, out, _ = self._run_cli([
                "--audio", aud, "--script", txt, "--ruby-map", os.path.join(td, "missing.json"), "--json",
            ])
            self.assertEqual(code, 2)
            self.assertFalse(json.loads(out)["ok"])


class TestRubyGlossaryAndOverrides(unittest.TestCase):
    def test_load_ruby_map_formats_and_errors(self):
        from fa_asmr.pipeline.runner import load_ruby_map

        # 扁平格式
        m1 = load_ruby_map({"陽菜羽": "ひなは", "コソ練": "こそれん"})
        self.assertEqual(m1["global"], {"陽菜羽": "ひなは", "コソ練": "こそれん"})
        self.assertEqual(m1["by_line"], {})

        # 分层格式
        m2 = load_ruby_map({"global": {"私": "あたし"}, "by_line": {"12": {"本気": "まじ"}}})
        self.assertEqual(m2["global"], {"私": "あたし"})
        self.assertEqual(m2["by_line"], {12: {"本気": "まじ"}})

        # JSON 文件加载
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "fixes.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump({"槍峰": "やりみね"}, f, ensure_ascii=False)
            self.assertEqual(load_ruby_map(p)["global"], {"槍峰": "やりみね"})

        # 非法输入必须显式抛错
        with self.assertRaises(ValueError):
            load_ruby_map({"私": ""})
        with self.assertRaises(ValueError):
            load_ruby_map({"私": "あた{し}"})
        with self.assertRaises(ValueError):
            load_ruby_map({"global": {}, "by_line": {"0": {"私": "あたし"}}})

    def test_annotate_lines_with_overrides_beats_yomikata_and_supports_by_line(self):
        from fa_asmr.pipeline.runner import annotate_lines_with_overrides

        class _FakeRuby:
            # 模拟 AutoRuby：已有 {s|r} 原样保留，裸文本正常注音，并模拟 yomikata 把 {私|あたし} 改回 {私|わたし}
            def annotate(self, text: str) -> str:
                import re as _re
                parts = []
                pos = 0
                for m in _re.finditer(r"\{[^|{}]+\|[^|{}]*\}", text):
                    seg = text[pos:m.start()]
                    seg = seg.replace("本気", "{本気|ほんき}").replace("練", "{練|れん}")
                    parts.append(seg)
                    tok = m.group(0)
                    parts.append("{私|わたし}" if tok == "{私|あたし}" else tok)
                    pos = m.end()
                tail = text[pos:].replace("本気", "{本気|ほんき}").replace("練", "{練|れん}")
                parts.append(tail)
                return "".join(parts)

        lines = [
            "私とコソ練しよう",
            "本気で好きだよ",
            "本気じゃないよ",
        ]
        rmap = {
            "global": {"私": "あたし", "コソ練": "こそれん", "練": "ねり"},
            "by_line": {"10": {"本気": "まじ"}},
        }
        annotated, hits = annotate_lines_with_overrides(
            _FakeRuby(), lines, src_line_nos=[5, 10, 15], ruby_map=rmap
        )
        # 1) 权威后置覆写胜过 yomikata；最长词优先让 コソ練 不被 練 截断
        self.assertEqual(annotated[0], "{私|あたし}と{コソ練|こそれん}しよう")
        # 2) by_line 只在第 10 行生效，第 15 行保持 AutoRuby 的 ほんき
        self.assertEqual(annotated[1], "{本気|まじ}で好きだよ")
        self.assertEqual(annotated[2], "{本気|ほんき}じゃないよ")
        self.assertGreaterEqual(hits, 3)

    def test_extract_ruby_glossary_sorts_polyphones_first_with_context(self):
        from fa_asmr.pipeline.runner import extract_ruby_glossary

        annotated = [
            "{私|わたし}は{今日|きょう}、{開|ひら}くよ。",
            "{私|あたし}も{今日|きょう}、{今日|きょう}の{開|あ}く{扉|とびら}。",
        ]
        glossary = extract_ruby_glossary(annotated, src_line_nos=[3, 9])
        # 多音词（私、開）应排在前面
        self.assertTrue(glossary[0]["is_polyphone"])
        self.assertTrue(glossary[1]["is_polyphone"])
        by_surf = {g["surface"]: g for g in glossary}
        self.assertEqual(by_surf["私"]["readings"], {"あたし": 1, "わたし": 1})
        self.assertEqual({ex["line"] for ex in by_surf["私"]["examples"]}, {3, 9})
        self.assertEqual({ex["reading"] for ex in by_surf["私"]["examples"]}, {"わたし", "あたし"})
        self.assertFalse(by_surf["今日"]["is_polyphone"])
        self.assertEqual(by_surf["今日"]["count"], 3)
        self.assertEqual(by_surf["私"]["lines_by_reading"], {"あたし": [9], "わたし": [3]})
        self.assertEqual(by_surf["今日"]["lines_by_reading"], {"きょう": [3, 9]})

    def test_single_kanji_override_does_not_split_compound_and_supports_by_file(self):
        from fa_asmr.pipeline.runner import annotate_lines_with_overrides, resolve_ruby_map_for_script

        class _CompoundFakeRuby:
            def annotate(self, text: str) -> str:
                # 模拟 UniDic：若裸文本未被劈开，「途中」整体识别为 {途中|とちゅー}，独立的「中」识别为 {中|ちゅー}
                import re as _re
                parts = []
                pos = 0
                for m in _re.finditer(r"\{[^|{}]+\|[^|{}]*\}", text):
                    seg = text[pos:m.start()]
                    seg = _re.sub(r"途中|中", lambda pm: "{途中|とちゅー}" if pm.group(0) == "途中" else "{中|ちゅー}", seg)
                    parts.append(seg)
                    parts.append(m.group(0))
                    pos = m.end()
                tail = _re.sub(r"途中|中", lambda pm: "{途中|とちゅー}" if pm.group(0) == "途中" else "{中|ちゅー}", text[pos:])
                parts.append(tail)
                return "".join(parts)

        rmap = {
            "global": {"中": "なか"},
            "by_file": {
                "04": {"by_line": {"10": {"中": "ちゅー"}}},
            },
        }
        # 1) 第 5 行：单字 global {"中": "なか"} 不得把「途中」劈成「途{中|なか}」，但独立「中」应被改为 {中|なか}
        ann_04, _ = annotate_lines_with_overrides(
            _CompoundFakeRuby(),
            ["途中で口の中に"],
            src_line_nos=[5],
            ruby_map=rmap,
            script_path=r"E:\work\04_track.txt",
        )
        self.assertEqual(ann_04[0], "{途中|とちゅー}で口の{中|なか}に")

        # 2) by_file 作用域仅对匹配的 04_track.txt 第 10 行生效，对 05_track.txt 第 10 行不生效
        resolved_04 = resolve_ruby_map_for_script(rmap, r"E:\work\04_track.txt")
        resolved_05 = resolve_ruby_map_for_script(rmap, r"E:\work\05_track.txt")
        self.assertEqual(resolved_04["by_line"].get(10), {"中": "ちゅー"})
        self.assertIsNone(resolved_05["by_line"].get(10))

    def test_extract_ruby_glossary_collocation_dedup_and_priority_tiering(self):
        from fa_asmr.pipeline.runner import extract_ruby_glossary, merge_ruby_glossaries

        # 复现真实盲区：前两条全是「母親」，第 3 条才出现「お母様」（全部误标成 はは）；
        # 按局部搭配去重后，「お母様」必须同样保留在 examples 与 collocations 中，且标为 high(affix_split)！
        ann_t01 = [
            "{失礼|しつれー}な{母|はは}{親|おや}だ",
            "また{母|はは}{親|おや}が来た",
            "お{母|はは}{様|さま}、ごめんなさい",
        ]
        gl_t01 = extract_ruby_glossary(ann_t01, src_line_nos=[4, 8, 20], speaker_names=["京香"], file_label="01")
        by_surf = {g["surface"]: g for g in gl_t01}
        self.assertEqual(by_surf["母"]["priority"], "high")
        self.assertIn("affix_split", by_surf["母"]["priority_reasons"])
        self.assertIn("お母様", by_surf["母"]["collocations"])
        self.assertIn("母親", by_surf["母"]["collocations"])
        self.assertEqual({ex["line"] for ex in by_surf["母"]["examples"]}, {4, 20})
        self.assertEqual(by_surf["失礼"]["priority"], "normal")

        ann_t02 = ["{京香|きょうか}は{失礼|しつれー}ではない"]
        gl_t02 = extract_ruby_glossary(ann_t02, src_line_nos=[3], speaker_names=["京香"], file_label="02")
        merged = merge_ruby_glossaries([("01", gl_t01), ("02", gl_t02)], speaker_names=["京香"])
        m_by_surf = {g["surface"]: g for g in merged}
        self.assertEqual(m_by_surf["京香"]["priority"], "high")
        self.assertIn("speaker_name", m_by_surf["京香"]["priority_reasons"])
        self.assertEqual(m_by_surf["失礼"]["count"], 2)
        self.assertEqual(m_by_surf["失礼"]["lines_by_file_reading"], {"01": {"しつれー": [4]}, "02": {"しつれー": [3]}})

    def test_batch_only_filter_dry_run(self):
        with tempfile.TemporaryDirectory() as td:
            for name in ("01_a", "05_b", "15_c"):
                open(os.path.join(td, f"{name}.wav"), "wb").close()
                with open(os.path.join(td, f"{name}.txt"), "w", encoding="utf-8") as f:
                    f.write("こんにちは\n")

            buf_out, buf_err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                code = cli.main(["--batch", td, "--only", "5", "--dry-run", "--json"])
            self.assertEqual(code, 0)
            payload = json.loads(buf_out.getvalue())
            # 数字过滤 "5" 仅精确匹配 05_b，绝不误伤 15_c
            self.assertEqual([t["name"] for t in payload["tasks"]], ["05_b"])

            # 无匹配时返回明确错误码 2
            buf_out2, buf_err2 = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(buf_out2), contextlib.redirect_stderr(buf_err2):
                code2 = cli.main(["--batch", td, "--only", "99", "--dry-run", "--json"])
            self.assertEqual(code2, 2)
            self.assertIn("--only", json.loads(buf_out2.getvalue())["error"])

    def test_manifest_mixed_modes_and_embedded_track_only_filter(self):
        with tempfile.TemporaryDirectory() as td:
            # 模拟混合目录：双视角轨 T1 + 单人轨 T2（含 U+2017 双下划线）+ 需跳过的特典差分轨
            for fn in (
                "トラック1_挨拶_主人公視点.wav",
                "トラック1_挨拶_夫視点.wav",
                "トラック2\u2017単体トラック.wav",
                "トラック1_挨拶_集中Ver.wav",
            ):
                open(os.path.join(td, fn), "wb").close()
            with open(os.path.join(td, "t1.txt"), "w", encoding="utf-8") as f:
                f.write("〖凛咲〗おはよう\n〖文乃〗おはようございます\n")
            with open(os.path.join(td, "t2.txt"), "w", encoding="utf-8") as f:
                f.write("やっと土日だ\n")

            manifest_p = os.path.join(td, "pairs.json")
            manifest_data = {
                "base_dir": td,
                "tasks": [
                    {
                        "name": "T1_dual",
                        "script": "t1.txt",
                        "audio": "トラック1_挨拶_主人公視点.wav",
                        "audio_b": "トラック1_挨拶_夫視点.wav",
                    },
                    {
                        "name": "トラック2\u2017単体トラック",
                        "script": "t2.txt",
                        "audio": "トラック2\u2017単体トラック.wav",
                    },
                ],
            }
            with open(manifest_p, "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, ensure_ascii=False)

            buf_out, buf_err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                code = cli.main(["--manifest", manifest_p, "--dry-run", "--json"])
            self.assertEqual(code, 0)
            payload = json.loads(buf_out.getvalue())
            self.assertEqual(payload["mode"], "mixed")
            self.assertEqual(payload["matched_count"], 2)
            self.assertEqual([t["mode"] for t in payload["tasks"]], ["multi_persp", "single"])

            # 验证 --only 2 能识别内嵌「トラック2‗...」轨号
            buf_only, _ = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(buf_only), contextlib.redirect_stderr(io.StringIO()):
                code_only = cli.main(["--manifest", manifest_p, "--only", "2", "--dry-run", "--json"])
            self.assertEqual(code_only, 0)
            p_only = json.loads(buf_only.getvalue())
            self.assertEqual([t["name"] for t in p_only["tasks"]], ["トラック2\u2017単体トラック"])

            # 验证 --manifest 与 --batch 互斥拦截
            buf_mut, _ = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(buf_mut), contextlib.redirect_stderr(io.StringIO()):
                code_mut = cli.main(["--manifest", manifest_p, "--batch", td, "--dry-run", "--json"])
            self.assertEqual(code_mut, 2)


if __name__ == "__main__":
    unittest.main()
