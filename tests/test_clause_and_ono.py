# -*- coding: utf-8 -*-
"""clause.py / ono_block_corpus.py / ono_filter.py 回归测试。"""
from __future__ import annotations

import unittest

import clause
from clause import F
import ono_block_corpus
from ono_filter import is_easy_ono


class TestClauseBoundaries(unittest.TestCase):
    def test_newline_boundary_vs_punct_cut(self):
        nl = {"orig": "\n", "type": 0, "pron": ""}
        self.assertTrue(clause.is_clause_boundary(nl))
        self.assertFalse(clause.is_punct_cut_point(nl))

    def test_regular_punct_is_both_boundary_and_cut(self):
        comma = {"orig": "、", "type": 0, "pron": ""}
        period = {"orig": "。", "type": 0, "pron": ""}
        for item in (comma, period):
            self.assertTrue(clause.is_clause_boundary(item))
            self.assertTrue(clause.is_punct_cut_point(item))

    def test_empty_pron_sokuon_not_cut_point(self):
        # 无读音文本促音 っ（如 って）绝不能被当成切点
        sokuon = {"orig": "っ", "type": 3, "pron": ""}
        self.assertFalse(clause.is_clause_boundary(sokuon))
        self.assertFalse(clause.is_punct_cut_point(sokuon))

    def test_open_brackets_prolong_and_voicing_not_boundaries(self):
        # 前导开引号/括号、词内延音符、独立浊点符不构成子句切分边界
        for item in (
            {"orig": "「", "type": 0, "pron": ""},
            {"orig": "『", "type": 0, "pron": ""},
            {"orig": "（", "type": 0, "pron": ""},
            {"orig": "～", "type": 0, "pron": ""},
            {"orig": "ー", "type": 0, "pron": ""},
            {"orig": "゛", "type": 0, "pron": ""},
            {"orig": " ", "type": 0, "pron": "", F.SRC: "゛"},
        ):
            self.assertFalse(clause.is_clause_boundary(item), f"不应切分: {item}")
            self.assertFalse(clause.is_punct_cut_point(item), f"不应为标点切点: {item}")

    def test_no_clause_cut_flag_suppresses_split(self):
        comma = {"orig": "、", "type": 0, "pron": "", F.NO_CLAUSE_CUT: True}
        self.assertFalse(clause.is_clause_boundary(comma))
        self.assertFalse(clause.is_punct_cut_point(comma))

    def test_mark_short_comma_bridges_semantic_and_stutter(self):
        # 1) 同行短实词顿号桥接：ダメ、です -> 桥接
        rl = [
            {"orig": "ダ", "type": 4, "pron": "da"},
            {"orig": "メ", "type": 4, "pron": "me"},
            {"orig": "、", "type": 0, "pron": ""},
            {"orig": "で", "type": 3, "pron": "de"},
            {"orig": "す", "type": 3, "pron": "su"},
            {"orig": "\n", "type": 0, "pron": ""},
        ]
        bridged = clause.mark_short_comma_bridges(rl)
        self.assertEqual(bridged, 1)
        self.assertTrue(rl[2].get(F.NO_CLAUSE_CUT))
        clauses = clause.clauses_with_surface(rl)
        self.assertEqual([surf for _, surf in clauses], ["ダメです"])

    def test_mark_short_comma_bridges_respects_speaker_change(self):
        rl = [
            {"orig": "ダ", "type": 4, "pron": "da", F.SPK: ("A",)},
            {"orig": "メ", "type": 4, "pron": "me", F.SPK: ("A",)},
            {"orig": "、", "type": 0, "pron": ""},
            {"orig": "で", "type": 3, "pron": "de", F.SPK: ("B",)},
            {"orig": "す", "type": 3, "pron": "su", F.SPK: ("B",)},
            {"orig": "\n", "type": 0, "pron": ""},
        ]
        bridged = clause.mark_short_comma_bridges(rl)
        self.assertEqual(bridged, 0)
        self.assertFalse(rl[2].get(F.NO_CLAUSE_CUT, False))


class TestOnoCorpusAndFilter(unittest.TestCase):
    def test_elongated_countdown_and_real_words_blocked(self):
        for word in ("い～～ち", "に～～い", "さ～ん", "よ～ん", "ご～お", "ろ～く", "ぎゅ～～～", "おちんちんっ"):
            self.assertTrue(ono_block_corpus.is_blocked(word), f"实词应被黑名单保护: {word}")

    def test_easy_ono_distinguishes_noise_vs_real_words(self):
        for noise in ("ちゅ", "んっ", "あぁ", "じゅる"):
            self.assertTrue(is_easy_ono(noise), f"纯拟声/喘息应识别为 easy_ono: {noise}")
        for real_word in ("ありがとう", "だめです", "おちんちん", "い～～ち", "ろ～く"):
            self.assertFalse(is_easy_ono(real_word), f"实词不应被误判为 easy_ono: {real_word}")


class TestMergeShortForbiddenSpans(unittest.TestCase):
    def test_blocks_merge_across_intervening_speaker_but_allows_continuous_overlap(self):
        import merge_short as ms

        # 同角色两句之间停顿 1.0s (4.0 -> 5.0)，期间他方角色发言 (4.1 -> 4.9) -> 禁止合并
        rl_gap = [
            {"orig": "ど", "type": 3, "pron": "do", "start": "[00:03:00]", "end": "[00:04:00]", F.KANA_N: 2},
            {"orig": "ち", "type": 3, "pron": "ti", "start": "[00:03:50]", "end": "[00:04:00]", F.KANA_N: 2},
            {"orig": "？", "type": 0, "pron": ""},
            {"orig": "\n", "type": 0, "pron": ""},
            {"orig": "は", "type": 3, "pron": "ha", "start": "[00:05:00]", "end": "[00:05:60]", F.KANA_N: 2},
            {"orig": "い", "type": 3, "pron": "i", "start": "[00:05:30]", "end": "[00:05:60]", F.KANA_N: 2},
            {"orig": "。", "type": 0, "pron": ""},
            {"orig": "\n", "type": 0, "pron": ""},
        ]
        # 无 forbidden_spans 时 1.0s <= 1.5s 会合并成 1 行
        merged_plain = ms.merge_short_lines(rl_gap, max_chars=20, max_gap=1.5)
        self.assertEqual(sum(1 for it in merged_plain if it.get("orig") == "\n"), 1)

        # 传入他方角色在缝隙内的发言区间 -> 保持 2 行不跨角色合并
        merged_guarded = ms.merge_short_lines(
            rl_gap, max_chars=20, max_gap=1.5, forbidden_spans=[(4.1, 4.9)]
        )
        self.assertEqual(sum(1 for it in merged_guarded if it.get("orig") == "\n"), 2)

        # 连续独白（停顿仅 0.1s）即便他方在背景持续叠声，也不应误阻断合并
        rl_cont = [
            {"orig": "ど", "type": 3, "pron": "do", "start": "[00:03:00]", "end": "[00:04:00]", F.KANA_N: 2},
            {"orig": "？", "type": 0, "pron": ""},
            {"orig": "\n", "type": 0, "pron": ""},
            {"orig": "は", "type": 3, "pron": "ha", "start": "[00:04:10]", "end": "[00:04:60]", F.KANA_N: 2},
            {"orig": "。", "type": 0, "pron": ""},
            {"orig": "\n", "type": 0, "pron": ""},
        ]
        merged_cont = ms.merge_short_lines(
            rl_cont, max_chars=20, max_gap=1.5, forbidden_spans=[(2.5, 5.5)]
        )
        self.assertEqual(sum(1 for it in merged_cont if it.get("orig") == "\n"), 1)


class TestUnisonSpeakerBoundaryAndBlocklist(unittest.TestCase):
    def test_ieei_in_ono_block_corpus(self):
        for w in ("イエーイ", "イエーイ！", "イェーイ", "イエイ", "ねー", "ねーー", "ネー", "ハーイ", "まー", "そー", "どー", "もー", "おーい"):
            self.assertTrue(ono_block_corpus.is_blocked(w), f"expected blocked: {w}")
        for w in ("ねー", "ネー", "ハーイ", "イェーイ", "まー", "そー", "おーい"):
            self.assertFalse(is_easy_ono(w), f"expected non-easy-ono: {w}")
        for w in ("くぷ", "くぽ", "ぐぷ", "くぷっ", "けほっ"):
            self.assertTrue(is_easy_ono(w), f"expected easy-ono: {w}")
        # 确保片假名转平假名不会误伤片假名喘息
        for w in ("アッ", "ハァ", "フゥ", "チュッ", "ジュポ", "クプ"):
            self.assertTrue(is_easy_ono(w), f"expected katakana easy-ono: {w}")

    def test_merge_short_does_not_merge_unison_with_single_speaker(self):
        import merge_short as ms

        rl = [
            {"orig": "せーの！", "type": 3, "pron": "seeno", "start": "[00:10:00]", "end": "[00:11:00]", F.KANA_N: 3, F.SPK: ("ゆりか", "つかさ")},
            {"orig": "\n", "type": 0, "pron": ""},
            {"orig": "こんにちは。", "type": 3, "pron": "konnichiha", "start": "[00:11:10]", "end": "[00:12:00]", F.KANA_N: 5, F.SPK: ("ゆりか",)},
            {"orig": "\n", "type": 0, "pron": ""},
        ]
        merged = ms.merge_short_lines(rl, max_chars=20, max_gap=1.5)
        self.assertEqual(sum(1 for it in merged if it.get("orig") == "\n"), 2)

    def test_post_process_keeps_unison_and_single_separate_in_same_stream(self):
        import align_multispeaker as ams

        # Stream 0 contains unison ("ゆりか", "つかさ") followed immediately (0.15s gap) by solo ("ゆりか",)
        # Stream 1 contains ("つかさ",) so post_process_and_merge_streams runs multi-stream consolidation
        rl = [
            {"orig": "お届けしまーす", "type": 3, "pron": "otodokeshimasu", "start": "[00:01:00]", "end": "[00:02:00]", F.KANA_N: 7, F.SPK: ("ゆりか", "つかさ")},
            {"orig": "！", "type": 0, "pron": "", F.SPK: ("ゆりか", "つかさ")},
            {"orig": "\n", "type": 0, "pron": ""},
            {"orig": "って感じ", "type": 3, "pron": "ttekanji", "start": "[00:02:15]", "end": "[00:02:70]", F.KANA_N: 4, F.SPK: ("ゆりか",)},
            {"orig": "。", "type": 0, "pron": "", F.SPK: ("ゆりか",)},
            {"orig": "\n", "type": 0, "pron": ""},
            {"orig": "決まったね", "type": 3, "pron": "kimattane", "start": "[00:03:00]", "end": "[00:03:80]", F.KANA_N: 5, F.SPK: ("つかさ",)},
            {"orig": "。", "type": 0, "pron": "", F.SPK: ("つかさ",)},
            {"orig": "\n", "type": 0, "pron": ""},
        ]
        ar = [
            {"original_start": 1.00, "original_end": 2.00, "score": -0.5},
            {"original_start": 2.15, "original_end": 2.70, "score": -0.5},
            {"original_start": 3.00, "original_end": 3.80, "score": -0.5},
        ]
        tok_map = {0: 0, 1: 3, 2: 6}
        _, indep_rl = ams.post_process_and_merge_streams(
            result_list=rl,
            alignment_results=ar,
            token_to_index_map=tok_map,
            stream_clause_lists=[[0, 1], [2]],
            enable_ono_gate=False,
            enable_resegment=False,
            enable_merge_short=True,
            merge_short_chars=20,
            merge_short_gap=1.5,
            merge_ono_chars=40,
            merge_ono_gap=2.5,
            return_independent_rl=True,
        )
        srt = ams.build_independent_multispeaker_srt(indep_rl, line_end_extend=0.0)
        self.assertIn("【ゆりか、つかさ】お届けしまーす！", srt)
        self.assertIn("【ゆりか】って感じ。", srt)
        self.assertNotIn("【ゆりか、つかさ】お届けしまーす！って感じ。", srt)

    def test_merge_short_visible_chars_without_clause_head_does_not_multiply(self):
        """关闭 Ono-Gate 未运行 mark_ono_candidates（无 F.CLAUSE_HEAD）时，同子句各 token 的 F.KANA_N 绝不能按 token 数平方累加。"""
        import merge_short as ms

        # 两个 5 字子句（合起来 10 字 <= max_chars=20），每个 token 都带整子句 F.KANA_N=5，且故意不设 F.CLAUSE_HEAD
        rl = [
            {"orig": "あ", "type": 3, "pron": "a", "start": "[00:01:00]", "end": "[00:01:10]", F.KANA_N: 5},
            {"orig": "り", "type": 3, "pron": "ri", "start": "[00:01:10]", "end": "[00:01:20]", F.KANA_N: 5},
            {"orig": "が", "type": 3, "pron": "ga", "start": "[00:01:20]", "end": "[00:01:30]", F.KANA_N: 5},
            {"orig": "と", "type": 3, "pron": "to", "start": "[00:01:30]", "end": "[00:01:40]", F.KANA_N: 5},
            {"orig": "う", "type": 3, "pron": "u", "start": "[00:01:40]", "end": "[00:01:50]", F.KANA_N: 5},
            {"orig": "、", "type": 0, "pron": ""},
            {"orig": "ご", "type": 3, "pron": "go", "start": "[00:01:60]", "end": "[00:01:70]", F.KANA_N: 5},
            {"orig": "ざ", "type": 3, "pron": "za", "start": "[00:01:70]", "end": "[00:01:80]", F.KANA_N: 5},
            {"orig": "い", "type": 3, "pron": "i", "start": "[00:01:80]", "end": "[00:01:90]", F.KANA_N: 5},
            {"orig": "ま", "type": 3, "pron": "ma", "start": "[00:01:90]", "end": "[00:02:00]", F.KANA_N: 5},
            {"orig": "す", "type": 3, "pron": "su", "start": "[00:02:00]", "end": "[00:02:10]", F.KANA_N: 5},
            {"orig": "。", "type": 0, "pron": ""},
            {"orig": "\n", "type": 0, "pron": ""},
        ]
        self.assertEqual(ms._visible_chars(rl, 0, 6), 5)
        self.assertEqual(ms._visible_chars(rl, 0, len(rl)), 10)
        merged = ms.merge_short_lines(rl, max_chars=20, max_gap=1.5)
        self.assertEqual(sum(1 for it in merged if it.get("orig") == "\n"), 1)

    def test_buchu_is_easy_ono_and_inline_leak_preserves_suru_shi(self):
        import ono_filter as of
        from furigana.fa_asmr_converter import RubyAnnotator
        from inline_leak_clean import InlineLeakCleaner

        # 1) ぶちゅっ / ぶっちゅぅう 属于易错接吻拟声，不应进实词黑名单
        self.assertTrue(of.is_easy_ono("ぶちゅっ"))
        self.assertTrue(of.is_easy_ono("ぶっちゅぅう～～～"))
        self.assertTrue(of.is_easy_ono("ぷちゅっ"))

        # 2) ご奉仕して / いい年して 中的「し」是合法する动词连用形，绝不能当内联振假名泄露删掉
        ra = RubyAnnotator(verbose=False)
        if ra._tagger is not None:
            cleaner = InlineLeakCleaner(ra)
            self.assertEqual(cleaner.clean("こうやって、お耳にご奉仕して…♡"), "こうやって、お耳にご奉仕して…♡")
            self.assertEqual(cleaner.clean("いい年して、ＪＫ制服を着た、"), "いい年して、ＪＫ制服を着た、")


if __name__ == "__main__":
    unittest.main()
