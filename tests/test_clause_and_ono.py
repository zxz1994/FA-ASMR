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


if __name__ == "__main__":
    unittest.main()
