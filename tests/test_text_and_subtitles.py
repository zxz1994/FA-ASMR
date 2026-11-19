# -*- coding: utf-8 -*-
"""文本编码、说话人切分、原文切片还原、LRC/SRT 生成与文件配对回归测试。"""
from __future__ import annotations

import os
import tempfile
import unittest

from clause import F
import fa_asmr_pipeline as pipe
import lrc_end_extend as lrc_ext


class TestEncodingAndSpeakerMarks(unittest.TestCase):
    def test_read_text_auto_encoding_variants(self):
        sample_jp = "おはよう、今日はいい天気ですね。"
        sample_cn = "你好，测试编码识别。"
        cases = [
            ("utf8.txt", sample_jp.encode("utf-8"), "utf-8", sample_jp),
            ("utf8bom.txt", b"\xef\xbb\xbf" + sample_jp.encode("utf-8"), "utf-8-sig", sample_jp),
            ("cp932.txt", sample_jp.encode("cp932"), "cp932", sample_jp),
            ("gb18030.txt", sample_cn.encode("gb18030"), "gb18030", sample_cn),
            ("utf16.txt", sample_jp.encode("utf-16"), "utf-16", sample_jp),
        ]
        with tempfile.TemporaryDirectory() as td:
            for fname, raw, expected_enc, expected_text in cases:
                p = os.path.join(td, fname)
                with open(p, "wb") as f:
                    f.write(raw)
                text, enc = pipe.read_text_auto_encoding(p)
                self.assertEqual(enc, expected_enc, f"{fname} 编码识别错误")
                self.assertEqual(text, expected_text, f"{fname} 内容解码错误")

    def test_speaker_marks_and_bare_circled_number(self):
        # 裸圈号 ① 不应被误判为说话人切换
        segs, cur = pipe._split_speaker_segments("それは①番だよ", ())
        self.assertEqual(segs, [((), "それは①番だよ")])
        self.assertEqual(cur, ())

        # 〖角色〗并列与续行继承
        segs1, cur = pipe._split_speaker_segments("〖澄华〗〖天音〗せーの！", ())
        self.assertEqual(segs1, [(("澄华", "天音"), "せーの！")])
        self.assertEqual(cur, ("澄华", "天音"))

        segs2, cur = pipe._split_speaker_segments("ありがとう〖澄华〗またね", cur)
        self.assertEqual(segs2, [(("澄华", "天音"), "ありがとう"), (("澄华",), "またね")])
        self.assertEqual(cur, ("澄华",))


class TestSourceSpanMapping(unittest.TestCase):
    def test_attach_source_spans_preserves_dakuten_and_nfkc_and_ruby(self):
        # 1) 独立浊点 / NFD 组合字符不引发后半行雪崩
        src = "あ゛っ…気持ちいい！"
        toks = [
            {"orig": "あ", "type": 3, "pron": "a"},
            {"orig": " ", "type": 0, "pron": ""},
            {"orig": "っ", "type": 3, "pron": ""},
            {"orig": "…", "type": 0, "pron": ""},
            {"orig": "気", "type": 1, "pron": "k"},
            {"orig": "持", "type": 1, "pron": "m"},
            {"orig": "ち", "type": 3, "pron": "t"},
            {"orig": "い", "type": 3, "pron": "i"},
            {"orig": "い", "type": 3, "pron": "i"},
            {"orig": "!", "type": 0, "pron": ""},
        ]
        pipe._attach_source_spans(toks, src, src)
        reconstructed = "".join(t.get(F.SRC, "") for t in toks)
        self.assertEqual(reconstructed, src)

        # 2) NFKC 一拆多（℃ -> °C）不翻倍重复
        src_deg = "体温37℃だよ"
        toks_deg = [
            {"orig": c, "type": 3, "pron": "a"}
            for c in "体温37°Cだよ"
        ]
        pipe._attach_source_spans(toks_deg, src_deg, src_deg)
        self.assertEqual("".join(t.get(F.SRC, "") for t in toks_deg), src_deg)

        # 3) 原文自带 {漢字|かな} 折叠成表层字，不撕裂花括号
        src_ruby = "{今日|きょう}は晴れ"
        toks_ruby = [
            {"orig": "今日", "type": 1, "pron": "k"},
            {"orig": "は", "type": 3, "pron": "w"},
            {"orig": "晴", "type": 1, "pron": "h"},
            {"orig": "れ", "type": 3, "pron": "r"},
        ]
        pipe._attach_source_spans(toks_ruby, src_ruby, src_ruby)
        self.assertEqual("".join(t.get(F.SRC, "") for t in toks_ruby), "今日は晴れ")


class TestLrcFormattingAndExtend(unittest.TestCase):
    def test_build_standard_lrc_dwell_overlap_and_100min(self):
        rl = [
            # 第一句 120 分钟后、仅 0.2s，应补足 1.0s 最短驻留（因为下一句在 120:05.00）
            {"orig": "、", "type": 0, "pron": ""},
            {"orig": "おはよう。", "type": 3, "pron": "o", "start": "[120:01:00]", "end": "[120:01:20]", F.SPK: ("A",)},
            {"orig": "\n", "type": 0, "pron": ""},
            # 被拟声守门删除的行，完全跳过
            {"orig": "ちゅ", "type": 3, "pron": "c", "start": "[120:03:00]", "end": "[120:03:50]", F.ONO_DROPPED: True},
            {"orig": "\n", "type": 0, "pron": ""},
            # 第二句持续较长（05.00 -> 10.00）
            {"orig": "ずっと一緒だよ", "type": 3, "pron": "z", "start": "[120:05:00]", "end": "[120:10:00]", F.SPK: ("A",)},
            {"orig": "\n", "type": 0, "pron": ""},
            # 嵌套短插话（06.00 -> 07.00），不应在 07.00 提前清屏，而应延到 10.00
            {"orig": "うん！", "type": 3, "pron": "u", "start": "[120:06:00]", "end": "[120:07:00]", F.SPK: ("B",)},
            {"orig": "\n", "type": 0, "pron": ""},
        ]
        lrc = pipe.build_standard_lrc(rl)
        lines = lrc.splitlines()
        # 行首孤立顿号被剔除，句末句号保留，分钟位 3 位正常输出
        self.assertEqual(lines[0], "[120:01.00]A：おはよう。")
        # 补足 1.0s 驻留
        self.assertEqual(lines[1], "[120:02.00]")
        # 嵌套插话结束后等到全局活跃最大结束时间 120:10.00 才清屏
        self.assertEqual(lines[2], "[120:05.00]A：ずっと一緒だよ")
        self.assertEqual(lines[3], "[120:06.00]B：うん！")
        self.assertEqual(lines[4], "[120:10.00]")

    def test_lrc_end_extend_parse_and_dedup(self):
        # 1 位与 2 位小数解析一致（.5 = 0.50s，.05 = 0.05s）
        self.assertAlmostEqual(lrc_ext.parse_ts("[00:01.5]"), 1.50)
        self.assertAlmostEqual(lrc_ext.parse_ts("[00:01.05]"), 1.05)
        self.assertAlmostEqual(lrc_ext.parse_ts("[120:00.00]"), 7200.0)

        # 延长后贴到下一句起点时，应剔除空白清屏行避免重复时间戳死行
        src = "\n".join([
            "[00:01.00]一行目",
            "[00:02.50]",
            "[00:03.00]二行目",
            "[00:04.00]",
        ])
        out, changed = lrc_ext.extend_lrc(src, 1.0)
        self.assertEqual(changed, 2)
        self.assertEqual(out.splitlines(), [
            "[00:01.00]一行目",
            "[00:03.00]二行目",
            "[00:05.00]",
        ])


class TestAutoPairFiles(unittest.TestCase):
    def test_single_and_speaker_suffix_pairing(self):
        with tempfile.TemporaryDirectory() as td:
            files = [
                os.path.join(td, "01_intro.wav"),
                os.path.join(td, "01_intro〖澄华〗〖天音〗.txt"),
                os.path.join(td, "02_only_audio.flac"),
            ]
            pairs, unmatched = pipe.auto_pair_files(files, mode="multi_speaker")
            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0]["name"], "01_intro")
            self.assertEqual(pairs[0]["speaker_marks"], ["澄华", "天音"])
            self.assertEqual(len(unmatched), 1)
            self.assertEqual(unmatched[0]["type"], "缺台本")

    def test_multi_perspective_japanese_and_ab_pairing_no_duplicate(self):
        with tempfile.TemporaryDirectory() as td:
            files = [
                os.path.join(td, "トラック7_テスト‗主人公視点.wav"),
                os.path.join(td, "トラック7_テスト‗夫視点.wav"),
                os.path.join(td, "トラック7_テスト.txt"),
                os.path.join(td, "01_scene_A.mp3"),
                os.path.join(td, "01_scene_B.mp3"),
                os.path.join(td, "01_scene.txt"),
            ]
            pairs, unmatched = pipe.auto_pair_files(files, mode="multi_persp")
            self.assertEqual(unmatched, [])
            self.assertEqual(len(pairs), 2)
            by_name = {p["name"]: p for p in pairs}
            self.assertIn("トラック7_テスト", by_name)
            self.assertIn("主人公視点", by_name["トラック7_テスト"]["audio"])
            self.assertIn("夫視点", by_name["トラック7_テスト"]["audio_b"])
            self.assertIn("01_scene", by_name)
            self.assertTrue(by_name["01_scene"]["audio"].endswith("01_scene_A.mp3"))
            self.assertTrue(by_name["01_scene"]["audio_b"].endswith("01_scene_B.mp3"))


if __name__ == "__main__":
    unittest.main()
