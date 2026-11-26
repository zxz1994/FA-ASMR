# -*- coding: utf-8 -*-
"""clause.py / ono_block_corpus.py / ono_filter.py 回归测试。"""
from __future__ import annotations

import unittest

import fa_asmr.text.clause as clause
from fa_asmr.text.clause import F
import fa_asmr.text.ono_block_corpus as ono_block_corpus
from fa_asmr.text.ono_filter import is_easy_ono


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
    def test_blacklisted_words_keep_their_identity_when_stretched(self):
        for word in ("まぁ", "もぉっ", "もおっ", "さぁん", "にぃい", "いぃえ",
                     "ハァ～ィ", "おぉ～い", "イェェ～イ", "こぉれ"):
            with self.subTest(word=word):
                self.assertTrue(ono_block_corpus.is_blocked(word))
                self.assertFalse(is_easy_ono(word))
        for sound in ("はぁぁ～", "ほぉぅぅっ", "んほぉぅぅっ", "っごおっ", "きゅっ",
                      "ちゅぅぅっ", "じゅぅぅっ", "きゅぅぅっ", "ふぅぅ", "んっ"):
            with self.subTest(sound=sound):
                self.assertFalse(ono_block_corpus.is_blocked(sound))
                self.assertTrue(is_easy_ono(sound))

    def test_stretched_reply_and_call_survive_the_ono_gate(self):
        for text in ("はぁ～い", "ハァ～イ", "はあい", "はぁい～", "はぁ～いぃ", "ね～", "ねぇっ", "っ……ねぇっ……", "わ"):
            with self.subTest(text=text):
                self.assertFalse(is_easy_ono(text))
        for breath in ("はぁ～", "はぁっ", "んっ", "すぅぅ", "ねりゅっ", "わっ"):
            with self.subTest(breath=breath):
                self.assertTrue(is_easy_ono(breath))

    def test_same_spoken_calls_in_alignment_and_ono_deletion(self):
        from fa_asmr.aligners.multipersp_acoustic import _is_pure_sfx_clause
        from fa_asmr.pipeline.runner import _tokenize_for_alignment
        from fa_asmr.text import haruraw2norm
        from fa_asmr.text.ono_filter import mark_ono_candidates, decide_ono_deletions

        lines = ["はぁ～い", "はぁい～", "ね～", "ねぇっ", "わ。すご。", "はぁ～", "んっ"]
        data = _tokenize_for_alignment(haruraw2norm, "\n".join(lines), lines,
                                       [("声",)] * len(lines), list(range(1, 8)), "ono-test")
        self.assertEqual(data["ono_clause_flags"], [0, 0, 0, 0, 0, 0, 1, 1])
        token_map = data["token_to_index_map"]
        items = data["result_list"]
        # Force an implausibly long span so a misclassified call would be marked.
        alignment = []
        for k, idx in token_map.items():
            items[idx].update(start=k * 5.0, end=k * 5.0 + 4.0)
            alignment.append(dict(original_start=k * 5.0, original_end=k * 5.0 + 4.0))
        mark_ono_candidates(items, alignment, token_map)
        decide_ono_deletions(items)
        for i, (surface, tokens) in enumerate(zip(data["clause_surfaces"], data["alignment_tokens"])):
            self.assertEqual(_is_pure_sfx_clause(surface, tokens, dict.fromkeys("haiwnesugq", 1)), i >= 6)
        spoken = [row for row in items if row.get("pron") and not row.get(F.CLAUSE_ONO)]
        self.assertEqual(len(spoken), sum(map(len, data["alignment_tokens"][:6])))
        self.assertTrue(all(not row.get(F.ONO_DROPPED) for row in spoken))

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
        import fa_asmr.text.merge_short as ms

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
        import fa_asmr.text.merge_short as ms

        rl = [
            {"orig": "せーの！", "type": 3, "pron": "seeno", "start": "[00:10:00]", "end": "[00:11:00]", F.KANA_N: 3, F.SPK: ("ゆりか", "つかさ")},
            {"orig": "\n", "type": 0, "pron": ""},
            {"orig": "こんにちは。", "type": 3, "pron": "konnichiha", "start": "[00:11:10]", "end": "[00:12:00]", F.KANA_N: 5, F.SPK: ("ゆりか",)},
            {"orig": "\n", "type": 0, "pron": ""},
        ]
        merged = ms.merge_short_lines(rl, max_chars=20, max_gap=1.5)
        self.assertEqual(sum(1 for it in merged if it.get("orig") == "\n"), 2)

    def test_post_process_keeps_unison_and_single_separate_in_same_stream(self):
        import fa_asmr.aligners.multispeaker as ams

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
        import fa_asmr.text.merge_short as ms

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
        import fa_asmr.text.ono_filter as of
        from furigana.fa_asmr_converter import RubyAnnotator
        from fa_asmr.text.inline_leak_clean import InlineLeakCleaner

        # 1) ぶちゅっ / ぶっちゅぅう 属于易错接吻拟声，不应进实词黑名单
        self.assertTrue(of.is_easy_ono("ぶちゅっ"))
        self.assertTrue(of.is_easy_ono("ぶっちゅぅう～～～"))
        self.assertTrue(of.is_easy_ono("ぷちゅっ"))

        # 2) ご奉仕して / いい年して 及真实基准台本中抓出的 6 类合法实词，绝不能被误删假名
        ra = RubyAnnotator(verbose=False)
        if ra._tagger is not None:
            cleaner = InlineLeakCleaner(ra)
            for clean_line in (
                "こうやって、お耳にご奉仕して…♡",
                "いい年して、ＪＫ制服を着た、",
                "指でっ、押したりっ……擦ったりっ……",
                "今まででっ、一番っ……ヤバいっ……",
                "手のひらでっ、押されるのもぉっ……",
                "行ってきますの、チューって感じじゃなくて……",
                "美人で……清楚そうな、奥さんなのに……",
                "乳首っ、強いいぃっ、振動っ！",
            ):
                with self.subTest(clean_line=clean_line):
                    self.assertEqual(cleaner.clean(clean_line), clean_line)
            # 3) 真实内联振假名泄露正例仍正常修复
            for bad, good in (
                ("沖縄きなわに行く", "沖縄に行く"),
                ("弱点くてんを攻める", "弱点を攻める"),
                ("気持ちちいい", "気持ちいい"),
                ("大好ききだ", "大好きだ"),
                ("悪るい子", "悪い子"),
                ("次ぎはこれ", "次はこれ"),
            ):
                with self.subTest(leak=bad):
                    self.assertEqual(cleaner.clean(bad), good)

    def test_simultaneous_identical_text_merges_speakers_in_lrc_and_srt(self):
        import fa_asmr.aligners.multispeaker as am
        import fa_asmr.pipeline.runner as pipe

        def _mk_line(spk, text, st, en):
            return [
                {"orig": text, "type": 1, "pron": "daisuki", "start": am._format_time(st), "end": am._format_time(en), F.SPK: (spk,)},
                {"orig": "\n", "type": 0, "pron": "", F.INSERTED: True},
            ]

        # 1) 1对1同步齐声（dt=0.14s, 高度重叠）-> 合并为 【押名A、押名B】大好きです。 先輩。
        # 2) 错开轮唱相同台词（dt=1.00s > 0.45s）-> 不合并
        # 3) 1对多同步齐声（方案 A：保留较细分段，押名B 被切为 2 段 vs 押名A 整句）-> 输出 2 条 【押名B、押名A】
        # 4) 同步不同台词 -> 不合并说话人
        rl = (
            _mk_line("押名A", "大好きです。 先輩。", 10.00, 12.80)
            + _mk_line("押名B", "大好きです。 先輩。", 10.14, 13.00)
            + _mk_line("押名A", "おかえり。", 20.00, 21.50)
            + _mk_line("押名B", "おかえり。", 21.00, 22.50)
            + _mk_line("押名B", "好きな子と、", 50.10, 51.10)
            + _mk_line("押名A", "好きな子と、痙攣するほどの快感に、溺れましょうよ。", 50.10, 57.06)
            + _mk_line("押名B", "痙攣するほどの快感に、溺れましょうよ。", 52.66, 57.04)
            + _mk_line("押名A", "おはよう。", 70.00, 71.20)
            + _mk_line("押名B", "こんにちは。", 70.10, 71.50)
        )
        srt_text = am.build_independent_multispeaker_srt(rl, line_end_extend=0.0)
        # 1对1 齐声合并
        self.assertIn("【押名A、押名B】大好きです。 先輩。", srt_text)
        self.assertNotIn("【押名A】大好きです。", srt_text)
        self.assertNotIn("【押名B】大好きです。", srt_text)
        # 错开轮唱不合并
        self.assertIn("【押名A】おかえり。", srt_text)
        self.assertIn("【押名B】おかえり。", srt_text)
        # 1对多 齐声按方案 A 保留较细分段，且不再保留整句冗余条
        self.assertIn("【押名B、押名A】好きな子と、", srt_text)
        self.assertIn("【押名B、押名A】痙攣するほどの快感に、溺れましょうよ。", srt_text)
        self.assertNotIn("【押名A】好きな子と、痙攣するほどの快感に、溺れましょうよ。", srt_text)
        # 文本不同不合并
        self.assertIn("【押名A】おはよう。", srt_text)
        self.assertIn("【押名B】こんにちは。", srt_text)

        # 验证 LRC 组装层（build_standard_lrc）统一处理：
        #   - 行内并发 1对1 齐声
        #   - 行内并发首段 + 续行 1对多 齐声（方案 A 保留较细分段）
        rl_lrc = [
            # 行内并发 1对1
            {"orig": "大好きです。 先輩。", "type": 1, "pron": "a", "start": am._format_time(10.00), "end": am._format_time(12.80), F.SPK: ("押名A",)},
            {"orig": "大好きです。 先輩。", "type": 1, "pron": "a", "start": am._format_time(10.14), "end": am._format_time(13.00), F.SPK: ("　押名B",)},
            {"orig": "\n", "type": 0, "pron": ""},
            # 行内并发首段 + 续行 1对多
            {"orig": "好きな子と、", "type": 1, "pron": "a", "start": am._format_time(50.10), "end": am._format_time(51.10), F.SPK: ("押名B",)},
            {"orig": "好きな子と、痙攣するほどの快感に、溺れましょうよ。", "type": 1, "pron": "a", "start": am._format_time(50.10), "end": am._format_time(57.06), F.SPK: ("　押名A",)},
            {"orig": "\n", "type": 0, "pron": ""},
            {"orig": "痙攣するほどの快感に、溺れましょうよ。", "type": 1, "pron": "a", "start": am._format_time(52.66), "end": am._format_time(57.04), F.SPK: ("押名B",)},
            {"orig": "\n", "type": 0, "pron": ""},
        ]
        lrc_text = pipe.build_standard_lrc(rl_lrc)
        self.assertIn("[00:10.00]押名A、押名B：大好きです。 先輩。", lrc_text)
        self.assertNotIn("　押名B：大好きです。", lrc_text)
        self.assertIn("[00:50.10]押名B、押名A：好きな子と、", lrc_text)
        self.assertIn("[00:52.66]押名B、押名A：痙攣するほどの快感に、溺れましょうよ。", lrc_text)
        self.assertNotIn("　押名A：好きな子と、痙攣するほどの快感に、", lrc_text)

    def test_token_level_prefix_suffix_unison_split_in_srt_and_lrc(self):
        import fa_asmr.aligners.multispeaker as am
        import fa_asmr.pipeline.runner as pipe

        # 复现「单人引子前缀 + 带引号同步齐声后缀」：
        #   押名B: はい。(5.10..5.50) + 「(无时) + どーぞ。先輩。(6.20..8.94) + 」(无时)
        #   押名A: どーぞ。先輩。(6.20..8.86)
        # 组装层应直接按逐 token 时间拆成：
        #   1) 押名B 单人引子 はい。 (5.10 起，不越过 6.20)
        #   2) 押名B、押名A 齐声 どーぞ。先輩。 (6.20..8.94，并优先去掉仅单侧带有的外层引号)
        rl_srt = [
            {"orig": "はい", "type": 1, "pron": "hai", "start": am._format_time(5.10), "end": am._format_time(5.50), F.SPK: ("押名B",)},
            {"orig": "。", "type": 0, "pron": "", F.SPK: ("押名B",)},
            {"orig": "「", "type": 0, "pron": "", F.SPK: ("押名B",)},
            {"orig": "どーぞ", "type": 1, "pron": "doozo", "start": am._format_time(6.20), "end": am._format_time(7.40), F.SPK: ("押名B",)},
            {"orig": "。", "type": 0, "pron": "", F.SPK: ("押名B",)},
            {"orig": "先輩", "type": 1, "pron": "senpai", "start": am._format_time(7.50), "end": am._format_time(8.94), F.SPK: ("押名B",)},
            {"orig": "。」", "type": 0, "pron": "", F.SPK: ("押名B",)},
            {"orig": "\n", "type": 0, "pron": "", F.INSERTED: True},
            {"orig": "どーぞ", "type": 1, "pron": "doozo", "start": am._format_time(6.20), "end": am._format_time(7.35), F.SPK: ("押名A",)},
            {"orig": "。", "type": 0, "pron": "", F.SPK: ("押名A",)},
            {"orig": "先輩", "type": 1, "pron": "senpai", "start": am._format_time(7.45), "end": am._format_time(8.86), F.SPK: ("押名A",)},
            {"orig": "。", "type": 0, "pron": "", F.SPK: ("押名A",)},
            {"orig": "\n", "type": 0, "pron": "", F.INSERTED: True},
        ]
        srt_text = am.build_independent_multispeaker_srt(rl_srt, line_end_extend=0.0)
        self.assertIn("00:00:05,100 --> 00:00:06,100\n【押名B】はい。", srt_text)
        self.assertTrue(
            "00:00:06,200 --> 00:00:08,940\n【押名A、押名B】どーぞ。先輩。" in srt_text
            or "00:00:06,200 --> 00:00:08,940\n【押名B、押名A】どーぞ。先輩。" in srt_text
        )
        self.assertNotIn("【押名A】どーぞ。先輩。", srt_text)

        # 验证 LRC 行内并发拼行场景同样按逐 token 时间精准拆合并输出
        rl_lrc = [
            {"orig": "はい", "type": 1, "pron": "hai", "start": am._format_time(5.10), "end": am._format_time(5.50), F.SPK: ("押名B",)},
            {"orig": "。", "type": 0, "pron": "", F.SPK: ("押名B",)},
            {"orig": "「", "type": 0, "pron": "", F.SPK: ("押名B",)},
            {"orig": "どーぞ", "type": 1, "pron": "doozo", "start": am._format_time(6.20), "end": am._format_time(7.40), F.SPK: ("押名B",)},
            {"orig": "。", "type": 0, "pron": "", F.SPK: ("押名B",)},
            {"orig": "先輩", "type": 1, "pron": "senpai", "start": am._format_time(7.50), "end": am._format_time(8.94), F.SPK: ("押名B",)},
            {"orig": "。」", "type": 0, "pron": "", F.SPK: ("押名B",)},
            {"orig": "どーぞ", "type": 1, "pron": "doozo", "start": am._format_time(6.20), "end": am._format_time(7.35), F.SPK: ("　押名A",)},
            {"orig": "。", "type": 0, "pron": "", F.SPK: ("　押名A",)},
            {"orig": "先輩", "type": 1, "pron": "senpai", "start": am._format_time(7.45), "end": am._format_time(8.86), F.SPK: ("　押名A",)},
            {"orig": "。", "type": 0, "pron": "", F.SPK: ("　押名A",)},
            {"orig": "\n", "type": 0, "pron": "", F.INSERTED: True},
        ]
        lrc_text = pipe.build_standard_lrc(rl_lrc)
        self.assertIn("[00:05.10]押名B：はい。", lrc_text)
        self.assertTrue(
            "[00:06.20]押名A、押名B：どーぞ。先輩。" in lrc_text
            or "[00:06.20]押名B、押名A：どーぞ。先輩。" in lrc_text
        )
        self.assertNotIn("　押名A：", lrc_text)

    def test_viscous_water_ono_stems_detected(self):
        import fa_asmr.text.ono_filter as of

        for w in ("れぢゅっ", "ねぢゅりゅりゅっ", "ぢゅぐぶぶぶぶっ", "ぬりゅぅぅぅぅっ"):
            self.assertTrue(of.is_easy_ono(w), f"Expected easy ono: {w}")

    def test_multipersp_dynamic_home_order_overrides_static_pref(self):
        from fa_asmr.aligners.multipersp_pipeline import project_result_list_to_perspective
        import fa_asmr.aligners.multispeaker as am

        # 整轨静态偏好愛蘭偏A (pref < 0)，但当前段明灯为实时主场 (_mp_home=True)、愛蘭为客场 (_mp_home=False)
        rl = [
            {"orig": "ありがとぉ", "type": 1, "pron": "arigatoo", "start": am._format_time(3.20), "end": am._format_time(5.30), F.SPK: ("愛蘭",), "_mp_home": False},
            {"orig": "キモチよかったぁ", "type": 1, "pron": "kimochiyokattaa", "start": am._format_time(3.50), "end": am._format_time(6.70), F.SPK: ("　明灯",), "_mp_home": True},
            {"orig": "\n", "type": 0, "pron": "", F.INSERTED: True},
        ]
        proj = project_result_list_to_perspective(
            rl, 0.0, 100.0, perspective_side="A", spk_persp_pref={"愛蘭": -2.0, "明灯": 2.0}
        )
        # 实时主场明灯必须排在客场愛蘭之前
        self.assertEqual(proj[0]["orig"], "キモチよかったぁ")
        self.assertEqual(proj[0][F.SPK][0], "明灯")
        self.assertEqual(proj[1]["orig"], "ありがとぉ")
        self.assertEqual(proj[1][F.SPK][0], "　愛蘭")

    def test_slow_asmr_unfinished_prefix_merges_without_intervening_speaker(self):
        import fa_asmr.aligners.multispeaker as ams

        # 明灯两句之间停顿 1.30s（慢语速喘息），前句仅 6 字且以省略号收尾、中间无别人插话 -> 应合并为 1 句
        rl = [
            {"orig": "それまで", "type": 3, "pron": "soremade", "start": "[00:01:00]", "end": "[00:02:00]", F.KANA_N: 4, F.SPK: ("明灯",)},
            {"orig": "……", "type": 0, "pron": "", F.SPK: ("明灯",)},
            {"orig": "\n", "type": 0, "pron": ""},
            {"orig": "ガマン", "type": 3, "pron": "gaman", "start": "[00:03:30]", "end": "[00:04:10]", F.KANA_N: 3, F.SPK: ("明灯",)},
            {"orig": "……", "type": 0, "pron": "", F.SPK: ("明灯",)},
            {"orig": "\n", "type": 0, "pron": ""},
            {"orig": "なるほど", "type": 3, "pron": "naruhodo", "start": "[00:10:00]", "end": "[00:11:00]", F.KANA_N: 4, F.SPK: ("愛蘭",)},
            {"orig": "。", "type": 0, "pron": "", F.SPK: ("愛蘭",)},
            {"orig": "\n", "type": 0, "pron": ""},
        ]
        ar = [
            {"original_start": 1.00, "original_end": 2.00, "score": -0.5},
            {"original_start": 3.30, "original_end": 4.10, "score": -0.5},
            {"original_start": 10.00, "original_end": 11.00, "score": -0.5},
        ]
        tok_map = {0: 0, 1: 3, 2: 6}
        _, indep_rl = ams.post_process_and_merge_streams(
            result_list=rl,
            alignment_results=ar,
            token_to_index_map=tok_map,
            stream_clause_lists=[[0, 1], [2]],
            enable_ono_gate=False,
            enable_resegment=False,
            enable_merge_short=False,
            merge_short_chars=20,
            merge_short_gap=1.5,
            merge_ono_chars=40,
            merge_ono_gap=2.5,
            return_independent_rl=True,
        )
        srt = ams.build_independent_multispeaker_srt(indep_rl, line_end_extend=0.0)
        self.assertIn("【明灯】それまで……ガマン……", srt)

    def test_ono_stretched_blocklist_and_real_words_not_misclassified(self):
        import fa_asmr.text.ono_block_corpus as obc
        import fa_asmr.text.ono_filter as of

        # 1) 黑名单词根的拖长音 / 促音 / 小元音变体全部受保护，不被误判为拟声
        for w in (
            "いぃい", "いぃいっ", "いぃぃぃ", "もぉういい", "わぁーい", "わーーい", "わぁぁぁい",
            "うわぁぁぁい", "おにぃぃ", "おにぃー", "おっーい",
            "ないっ", "たいっ", "いうっ", "ほうっ", "ここっ", "ごごっ", "ふんっ",
            "ぱんっ", "りんっ", "やんっ", "つんっ", "びゅっ", "ぴゅっ", "ずぶっ", "ずぼっ", "しこっ",
            "さぁっー", "しぃっー", "みぃっー", "ミアっ",
        ):
            with self.subTest(blocked_variant=w):
                self.assertTrue(obc.is_blocked(w), f"Expected blocked: {w}")
                self.assertFalse(of.is_easy_ono(w), f"Expected NOT easy_ono: {w}")

        # 2) 常见纯假名实词（辅音+异行元音/ん、叠字+促音/长音、前导っ/ん+实词、动词接龙）不被误判为拟声
        for w in (
            "ふあん", "ふおん", "におい", "しあい", "きあい", "であい", "よいん",
            "まおう", "かおう", "せおう", "ていう", "という", "そいう", "わいん", "らいん",
            "めいん", "むおん", "ごえん", "きおん", "まうえ", "のうえ", "ぼいん",
            "ままー", "ままっ", "ぱぱー", "ぱぱっ", "みみっ", "みみー", "ももっ", "ちちっ", "ははっ",
            "あがる", "あがく", "あがれ", "あふれ", "あひる", "もぐる", "あはれ",
            "っまって", "っいたい", "っうそ", "っだれ", "っだめ", "っいく", "っいぐ",
            "んまって", "んいたい", "んうそ", "んすごい",
        ):
            with self.subTest(real_word=w):
                self.assertFalse(of.is_easy_ono(w), f"Real word misclassified as easy_ono: {w}")

        # 3) 正例拟声与波浪号统一（～ U+FF5E 与 〜 U+301C）保持 100% 识别
        for s in (
            "は～", "は〜", "はぁ～", "はぁ〜", "はぁっ", "ほぉぅぅっ", "んほぉぅぅっ",
            "っごおっ", "っあっ", "んあっ", "んふぅ", "んちゅ", "じゅぅぅっ", "きゅぅぅっ",
            "じゅうっ", "じゅうう", "んじゅうぅぅ",
            "れぢゅっ", "ねぢゅりゅりゅっ", "ぢゅぐぶぶぶぶっ", "ぬりゅぅぅぅぅっ",
            "お“っ", "お”っ", "お“", "お”ぉ", "はぷ", "ふちゅる", "ねぶぢゅるるっ",
            "ねぶりゅりゅっ", "すじゅ", "おごっ", "ふぶ", "うぅぅん",
            "ぶっぽ", "ぶぽっ", "ぶぱっ", "くちゃ", "くちゃくちゃ", "がぼっ",
            "ずずっ", "ずぞっ", "ずずぅう", "んぷ", "んれ", "ちゅぐ", "どちゅどちゅ",
            "はひっ", "はへぇっ", "ひうっ", "ふわぁ", "ふわぁっ", "ぐおっ",
            "おふっ", "くふ", "くふふ", "へへ", "くくっ", "ごきゅ", "ごきゅっ",
            "ごおおっ", "んごぉっ", "ほぉぉぉぉぉぉぉっ",
            "もぉ゛ッ", "ぬぉ゛ッ", "のぉ゛ッ", "あんっ", "あぁんっ", "ああ", "あぁ", "おぉっ",
        ):
            with self.subTest(true_ono=s):
                self.assertTrue(of.is_easy_ono(s), f"True ono missed: {s}")

        # 4) NAS 大规模实测抓出的高频实词/混写/截断词 Bug 防回归
        for w in (
            "イっ", "イ", "れ", "ふ", "げ", "んじゃ", "いいん",
            "うん", "うんっ", "うんうん", "うーん", "うーんっ", "う～ん", "う~ん", "う〜ん", "う～～ん", "ううん", "ううんっ",
            "ええ", "えー", "えっ", "へー", "ねっ", "おごる",
            "あーあ", "あ～あ", "あ~あ", "あれれ", "えい", "えいっ", "あーん", "あ～ん", "あ~ん", "わぁ", "わー", "おー",
            "んうん", "んいい", "んえっ", "っうん", "っいい",
            "やっ", "やぁ", "やぁっ", "なっ", "なぁ", "もっ", "でっ", "まっ",
            "てっ", "てぇ", "ってぇ", "にっ", "さっ", "きっ", "たっ", "のっ", "のぉ",
            "しっ", "めっ", "よぉ", "よっ", "がっ", "るっ", "ろっ",
            "じゅう", "ごお", "ご～お", "ふわふわ", "ふぐ", "ふぐっ", "ほご",
        ):
            with self.subTest(nas_real=w):
                self.assertFalse(of.is_easy_ono(w), f"NAS real word misclassified as easy_ono: {w}")

    def test_real_tokenization_entry_consistency_for_ono_and_interjections(self):
        """端到端校验正式分词入口（NFKC ASCII ~、独立浊点 ゛/ﾞ/“、R5 前导ん、超长单辅音尾音与 _is_pure_sfx_clause 一致性）。"""
        from fa_asmr.aligners.multipersp_acoustic import _is_pure_sfx_clause
        from fa_asmr.pipeline.runner import _tokenize_for_alignment
        from fa_asmr.text import haruraw2norm
        from fa_asmr.text.ono_filter import mark_ono_candidates, decide_ono_deletions

        protected = ["いいん", "じゅうん", "う～ん", "う〜ん", "うーんっ", "う～～ん", "あ～ん", "あ～あ", "んうん", "んいい", "んえっ", "えっ", "えっっ", "ねっっ", "うん“うん”", "のぉ”"]
        expected_ono = ["もぉ゛ッ", "もぉ\u3099ッ", "ぬぉ゛ッ", "のぉ゛ッ", "お“っ", "ほぉぉぉぉぉぉぉっ", "ほ" + "ぉ" * 31 + "っ", "ごおおっ", "んごぉっ"]
        lines = protected + expected_ono
        data = _tokenize_for_alignment(
            haruraw2norm, "\n".join(lines), lines,
            [("声",)] * len(lines), list(range(1, len(lines) + 1)), "e2e-ono",
        )
        n_prot = len(protected)
        self.assertEqual(data["ono_clause_flags"][:n_prot], [0] * n_prot,
                         f"Protected lines flagged as ono: {list(zip(protected, data['clause_surfaces'][:n_prot], data['ono_clause_flags'][:n_prot]))}")
        self.assertEqual(data["ono_clause_flags"][n_prot:], [1] * len(expected_ono),
                         f"Expected ono lines missed: {list(zip(expected_ono, data['clause_surfaces'][n_prot:], data['ono_clause_flags'][n_prot:]))}")
        # 校验顿号桥接与浊点识别口径一致：もぉ゛ッ、だめ 不得被误桥接成 1 个子句
        bridge_data = _tokenize_for_alignment(
            haruraw2norm, "もぉ゛ッ、だめ", ["もぉ゛ッ、だめ"],
            [("声",)], [1], "e2e-bridge",
        )
        self.assertEqual(bridge_data["ono_clause_flags"], [1, 0])
        # 校验えっ/えっっ/ねっっ/う~ん在 _is_pure_sfx_clause 中保持非 SFX
        self.assertFalse(_is_pure_sfx_clause("えっ", ["e"], {"e": 1}))
        self.assertFalse(_is_pure_sfx_clause("えっっ", ["e"], {"e": 1}))
        self.assertFalse(_is_pure_sfx_clause("ねっっ", ["n", "e"], {"n": 1, "e": 2}))
        self.assertFalse(_is_pure_sfx_clause("う~ん", ["u", "N"], {"u": 1, "N": 2}))
        # 注入拉伸时间轴确认所有受保护行在 ono_filter 链路中绝不被删除
        items = data["result_list"]
        token_map = data["token_to_index_map"]
        alignment = []
        for k, idx in token_map.items():
            items[idx].update(start=k * 5.0, end=k * 5.0 + 4.0)
            alignment.append(dict(original_start=k * 5.0, original_end=k * 5.0 + 4.0))
        mark_ono_candidates(items, alignment, token_map)
        decide_ono_deletions(items)
        # 统计前 n_prot 行对应发音 token，确认全部未被标为 ONO_DROPPED
        line_idx = 0
        for row in items:
            if row.get("orig") == "\n":
                line_idx += 1
                continue
            if line_idx < n_prot and row.get("pron"):
                self.assertFalse(row.get(F.ONO_DROPPED), f"Protected line {lines[line_idx]} dropped!")


if __name__ == "__main__":
    unittest.main()
