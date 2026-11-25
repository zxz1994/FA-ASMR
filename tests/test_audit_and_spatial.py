# -*- coding: utf-8 -*-
"""fa_asmr.pipeline.audit 与 fa_asmr_spatial_sep.py 单元测试。"""
from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest

import numpy as np

import fa_asmr.pipeline.audit as audit
import fa_asmr_spatial_sep as ssep


class TestAlignmentAudit(unittest.TestCase):
    def test_flags_low_score_compressed_skipped_and_unaligned_speech_gaps(self):
        # 构造 5 句对齐结果：
        #   ci0: 正常实词 (0.5..2.0s, sc=-1.0)
        #   ci1: 被跳过实词 (skipped=True -> skipped_semantic)
        #   ci2: 极低分实词+压缩 (2.5..2.65s, 6 音素, sc=-4.2 -> low_score + compressed)
        #   ci3: 易错拟声 (ono=1, 即使低分也不应误报)
        #   ci4: 正常实词 (10.0..12.0s, sc=-1.1)，中间 2.65..10.0s 空档在 5.0..6.5s 有密集 CTC 音素活跃岛！
        alignment_tokens = [
            ["ko", "n", "ni", "chi", "wa"],
            ["a", "ri", "ga", "to"],
            ["ya", "ri", "mi", "ne", "ku", "n"],
            ["a", "a"],
            ["o", "ya", "su", "mi"],
        ]
        alignment_results = [
            # ci0 (5 tokens)
            {"original_start": 0.50, "original_end": 0.80, "score": -0.9},
            {"original_start": 0.80, "original_end": 1.10, "score": -1.0},
            {"original_start": 1.10, "original_end": 1.40, "score": -1.1},
            {"original_start": 1.40, "original_end": 1.70, "score": -1.0},
            {"original_start": 1.70, "original_end": 2.00, "score": -1.0},
            # ci1 (4 tokens, skipped)
            {"skipped": True, "score": -10.0},
            {"skipped": True, "score": -10.0},
            {"skipped": True, "score": -10.0},
            {"skipped": True, "score": -10.0},
            # ci2 (6 tokens, low score + compressed 0.12s)
            {"original_start": 2.50, "original_end": 2.52, "score": -4.2},
            {"original_start": 2.52, "original_end": 2.54, "score": -4.3},
            {"original_start": 2.54, "original_end": 2.56, "score": -4.1},
            {"original_start": 2.56, "original_end": 2.58, "score": -4.0},
            {"original_start": 2.58, "original_end": 2.60, "score": -4.4},
            {"original_start": 2.60, "original_end": 2.62, "score": -4.2},
            # ci3 (2 tokens, ono)
            {"original_start": 2.80, "original_end": 2.90, "score": -5.0},
            {"original_start": 2.90, "original_end": 3.00, "score": -5.0},
            # ci4 (4 tokens)
            {"original_start": 10.00, "original_end": 10.50, "score": -1.1},
            {"original_start": 10.50, "original_end": 11.00, "score": -1.0},
            {"original_start": 11.00, "original_end": 11.50, "score": -1.2},
            {"original_start": 11.50, "original_end": 12.00, "score": -1.1},
        ]
        # 14s = 700 frames (0.02s/frame); 5.0..6.5s (frames 250..325) 每隔 3 帧一个活跃音素尖峰 (~25 帧)
        nb = np.zeros(700, dtype=bool)
        nb[250:325:3] = True
        # 8.0s 处放 2 帧孤立噪声，不应误报
        nb[400:402] = True

        rep = audit.build_alignment_audit(
            alignment_tokens=alignment_tokens,
            alignment_results=alignment_results,
            ono_clause_flags=[0, 0, 0, 1, 0],
            clause_surfaces=["こんにちは", "ありがとう", "槍峰くん", "ああ", "おやすみ"],
            clause_speakers=[("A",), ("A",), ("B",), ("B",), ("A",)],
            clause_line_nos=[1, 3, 5, 6, 9],
            clause_annotated=["こんにちは", "ありがとう", "{槍|そー}{峰|みね}くん", "ああ", "おやすみ"],
            diagnostics={"ctc_nb_active": nb, "frame_sec": 0.02, "duration_sec": 14.0},
        )
        self.assertEqual(rep["suspect_clauses_count"], 2)
        by_ci = {s["clause_index"]: s for s in rep["suspect_clauses"]}
        self.assertIn("skipped_semantic", by_ci[1]["reasons"])
        self.assertIn("low_score", by_ci[2]["reasons"])
        self.assertIn("compressed", by_ci[2]["reasons"])
        self.assertEqual(by_ci[2]["line"], 5)
        self.assertEqual(len(by_ci[2]["ruby_terms"]), 2)

        # 未覆盖语音岛应精确检出 5.0..6.5s 这 1 处（忽略 8.0s 处 2 帧噪声）
        self.assertEqual(rep["unaligned_speech_gaps_count"], 1)
        gap = rep["unaligned_speech_gaps"][0]
        self.assertGreaterEqual(gap["start"], 4.9)
        self.assertLessEqual(gap["end"], 6.6)
        self.assertEqual(gap["next_line"]["line"], 9)

    def test_short_four_five_phone_clauses_use_stricter_threshold(self):
        # 5 音素短语气词 -3.8 不应误报，但 -6.0 极端异常仍应报出
        alignment_tokens = [
            ["ko", "n", "ni", "chi", "wa"],
            ["de", "mo", "o", "o"],       # 有效音素 ~4 (de+mo+oo=5 -> folded 4-5)
            ["ko", "ri", "ya", "a"],      # 有效音素 ~5，极端 -6.0
        ]
        alignment_results = [
            {"original_start": 0.5, "original_end": 0.8, "score": -1.0},
            {"original_start": 0.8, "original_end": 1.1, "score": -1.0},
            {"original_start": 1.1, "original_end": 1.4, "score": -1.0},
            {"original_start": 1.4, "original_end": 1.7, "score": -1.0},
            {"original_start": 1.7, "original_end": 2.0, "score": -1.0},
            # ci1: mild -3.8
            {"original_start": 2.5, "original_end": 2.7, "score": -3.8},
            {"original_start": 2.7, "original_end": 2.9, "score": -3.8},
            {"original_start": 2.9, "original_end": 3.0, "score": -3.8},
            {"original_start": 3.0, "original_end": 3.1, "score": -3.8},
            # ci2: extreme -6.0
            {"original_start": 4.0, "original_end": 4.1, "score": -6.0},
            {"original_start": 4.1, "original_end": 4.2, "score": -6.0},
            {"original_start": 4.2, "original_end": 4.3, "score": -6.0},
            {"original_start": 4.3, "original_end": 4.4, "score": -6.0},
        ]
        rep = audit.build_alignment_audit(
            alignment_tokens=alignment_tokens,
            alignment_results=alignment_results,
            ono_clause_flags=[0, 0, 0],
            clause_surfaces=["こんにちは", "でもぉ", "こりゃぁ"],
            clause_speakers=[("A",), ("A",), ("A",)],
            clause_line_nos=[1, 2, 3],
        )
        self.assertEqual(rep["suspect_clauses_count"], 1)
        self.assertEqual(rep["suspect_clauses"][0]["clause_index"], 2)

    def test_multiperspective_joint_audit_suppresses_away_skips_and_peer_bleed(self):
        # 3 句：
        #   ci0: A 清晰 (-1.0, 1..2.5s)，B 跳过（客视角正常跳过 -> 不应报 skipped_semantic！）
        #   ci1: A 跳过，B 清晰 (-1.1, 3..4.5s)（客视角正常跳过 -> 不应报 skipped_semantic！）
        #   ci2: A 与 B 两侧均跳过 -> 应报 skipped_semantic (perspective='BOTH')
        alignment_tokens = [
            ["ko", "n", "ni", "chi", "wa"],
            ["o", "ya", "su", "mi", "na"],
            ["a", "ri", "ga", "to", "ne"],
        ]
        res_a = [
            # ci0 aligned on A
            {"original_start": 1.0, "original_end": 1.3, "score": -1.0},
            {"original_start": 1.3, "original_end": 1.6, "score": -1.0},
            {"original_start": 1.6, "original_end": 1.9, "score": -1.0},
            {"original_start": 1.9, "original_end": 2.2, "score": -1.0},
            {"original_start": 2.2, "original_end": 2.5, "score": -1.0},
            # ci1 skipped on A
            *[{"skipped": True, "score": -10.0} for _ in range(5)],
            # ci2 skipped on A
            *[{"skipped": True, "score": -10.0} for _ in range(5)],
        ]
        res_b = [
            # ci0 skipped on B
            *[{"skipped": True, "score": -10.0} for _ in range(5)],
            # ci1 aligned on B
            {"original_start": 3.0, "original_end": 3.3, "score": -1.1},
            {"original_start": 3.3, "original_end": 3.6, "score": -1.1},
            {"original_start": 3.6, "original_end": 3.9, "score": -1.1},
            {"original_start": 3.9, "original_end": 4.2, "score": -1.1},
            {"original_start": 4.2, "original_end": 4.5, "score": -1.1},
            # ci2 skipped on B
            *[{"skipped": True, "score": -10.0} for _ in range(5)],
        ]
        # 12s = 600 frames:
        #   - 1.0..2.5s 在 B 上也有隔墙闷音串扰（应被 A 的 ci0 覆盖抑制，不报空档）
        #   - 7.0..8.6s 在 A 上有真正缺句语音（两侧均未覆盖 -> 应报 1 处 perspective='A' 空档）
        nb_a = np.zeros(600, dtype=bool)
        nb_a[350:430:3] = True
        nb_b = np.zeros(600, dtype=bool)
        nb_b[50:125:3] = True

        rep = audit.build_multiperspective_audit(
            alignment_tokens=alignment_tokens,
            alignment_results_a=res_a,
            alignment_results_b=res_b,
            ono_clause_flags=[0, 0, 0],
            clause_surfaces=["こんにちは", "おやすみな", "ありがとね"],
            clause_speakers=[("A",), ("B",), ("A",)],
            clause_line_nos=[1, 2, 3],
            diagnostics_a={"ctc_nb_active": nb_a, "frame_sec": 0.02, "duration_sec": 12.0},
            diagnostics_b={"ctc_nb_active": nb_b, "frame_sec": 0.02, "duration_sec": 12.0},
            delta_sec=0.0,
        )
        self.assertTrue(rep["is_multiperspective"])
        self.assertEqual(rep["suspect_clauses_count"], 1)
        self.assertEqual(rep["suspect_clauses"][0]["clause_index"], 2)
        self.assertEqual(rep["suspect_clauses"][0]["perspective"], "BOTH")
        self.assertEqual(rep["unaligned_speech_gaps_count"], 1)
        self.assertEqual(rep["unaligned_speech_gaps"][0]["perspective"], "A")
        self.assertGreaterEqual(rep["unaligned_speech_gaps"][0]["start"], 6.9)


class TestSpatialSeparation(unittest.TestCase):
    def test_separates_left_center_right_three_sources(self):
        sr = 16000
        t = np.arange(sr * 2, dtype=np.float32) / float(sr)
        # 三个正交频率、分居左/中/右（含显著 ILD）
        s_left = 0.6 * np.sin(2.0 * np.pi * 400.0 * t)
        s_center = 0.6 * np.sin(2.0 * np.pi * 900.0 * t)
        s_right = 0.6 * np.sin(2.0 * np.pi * 1500.0 * t)
        # 立体声混合：左源 L>>R，中源 L==R，右源 R>>L
        L = s_left + 0.7 * s_center + 0.08 * s_right
        R = 0.08 * s_left + 0.7 * s_center + s_right
        wav_lr = np.stack([L, R], axis=0).astype(np.float32)

        sep, meta = ssep.separate_stereo_spatial(wav_lr, sr=sr, n_sources=3)
        self.assertEqual(sep.shape, (3, len(t)))
        self.assertEqual([m["position_tag"] for m in meta], ["L", "C", "R"])

        # 验证三路各自频带能量主导对应声源（400Hz -> K1, 900Hz -> K2, 1500Hz -> K3）
        freqs = np.fft.rfftfreq(len(t), 1.0 / sr)
        target_hz = [400.0, 900.0, 1500.0]
        for k, hz in enumerate(target_hz):
            spec = np.abs(np.fft.rfft(sep[k])) ** 2
            e_own = float(np.sum(spec[(freqs >= hz - 40) & (freqs <= hz + 40)]))
            e_others = sum(
                float(np.sum(spec[(freqs >= ohz - 40) & (freqs <= ohz + 40)]))
                for j, ohz in enumerate(target_hz) if j != k
            )
            self.assertGreater(e_own, e_others * 5.0, f"K{k+1} ({hz}Hz) isolation too weak")

    def test_cli_single_clip_end_to_end(self):
        import soundfile as sf

        sr = 16000
        t = np.arange(sr * 2, dtype=np.float32) / float(sr)
        L = 0.5 * np.sin(2.0 * np.pi * 450.0 * t) + 0.05 * np.sin(2.0 * np.pi * 1200.0 * t)
        R = 0.05 * np.sin(2.0 * np.pi * 450.0 * t) + 0.5 * np.sin(2.0 * np.pi * 1200.0 * t)
        with tempfile.TemporaryDirectory() as td:
            aud = os.path.join(td, "mix.wav")
            out_dir = os.path.join(td, "out")
            sf.write(aud, np.stack([L, R], axis=1), sr)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = ssep.main(["--audio", aud, "--out-dir", out_dir, "--sources", "2", "--json"])
            self.assertEqual(code, 0)
            payload = json.loads(buf.getvalue())
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["segments_count"], 1)
            self.assertEqual(len(payload["segments"][0]["channels"]), 2)
            for ch in payload["segments"][0]["channels"]:
                self.assertTrue(os.path.isfile(ch["wav"]))

    def test_stereo_spread_detector_distinguishes_solo_vs_overlap(self):
        sr = 16000
        t = np.arange(sr * 2, dtype=np.float32) / float(sr)
        # 居中单声源（左右完全一致）-> False
        solo = 0.5 * np.sin(2.0 * np.pi * 500.0 * t) + 0.3 * np.sin(2.0 * np.pi * 1100.0 * t)
        wav_solo = np.stack([solo, solo], axis=0)
        is_sp_solo, info_solo = ssep.has_stereo_spatial_spread(wav_solo, sr=sr)
        self.assertFalse(is_sp_solo)
        self.assertLess(info_solo["ild_span_db"], 1.0)

        # 左右双声源重叠 -> True
        L = 0.6 * np.sin(2.0 * np.pi * 400.0 * t) + 0.06 * np.sin(2.0 * np.pi * 1300.0 * t)
        R = 0.06 * np.sin(2.0 * np.pi * 400.0 * t) + 0.6 * np.sin(2.0 * np.pi * 1300.0 * t)
        wav_ov = np.stack([L, R], axis=0)
        is_sp_ov, info_ov = ssep.has_stereo_spatial_spread(wav_ov, sr=sr)
        self.assertTrue(is_sp_ov)
        self.assertGreater(info_ov["ild_span_db"], 15.0)

    def test_audit_clips_merge_and_auto_separate_linkage(self):
        import soundfile as sf

        sr = 16000
        n = sr * 12  # 12s: 0..5s 左右重叠, 6..12s 居中单声源
        t = np.arange(n, dtype=np.float32) / float(sr)
        L = np.zeros(n, dtype=np.float32)
        R = np.zeros(n, dtype=np.float32)
        ov_mask = (t >= 0.5) & (t <= 5.0)
        L[ov_mask] = 0.6 * np.sin(2.0 * np.pi * 420.0 * t[ov_mask]) + 0.05 * np.sin(2.0 * np.pi * 1350.0 * t[ov_mask])
        R[ov_mask] = 0.05 * np.sin(2.0 * np.pi * 420.0 * t[ov_mask]) + 0.6 * np.sin(2.0 * np.pi * 1350.0 * t[ov_mask])
        solo_mask = (t >= 7.0) & (t <= 11.0)
        solo_sig = 0.5 * np.sin(2.0 * np.pi * 600.0 * t[solo_mask]) + 0.25 * np.sin(2.0 * np.pi * 1200.0 * t[solo_mask])
        L[solo_mask] = solo_sig
        R[solo_mask] = solo_sig

        report = {
            "suspect_clauses": [
                # 相邻两条（间隔 0.4s < 1.5s）应合并为 1 个审听窗并触发立体声分离
                {"clause_index": 0, "line": 10, "speaker": "A", "start": 1.2, "end": 2.4, "reasons": ["low_score"]},
                {"clause_index": 1, "line": 10, "speaker": "B", "start": 2.8, "end": 4.0, "reasons": ["low_score"]},
            ],
            "unaligned_speech_gaps": [
                # 居中单声源空档：应切原片段，但跳过空间分离
                {"start": 8.0, "end": 10.0, "duration": 2.0},
            ],
        }

        with tempfile.TemporaryDirectory() as td:
            aud = os.path.join(td, "track.wav")
            clips_dir = os.path.join(td, "clips")
            sf.write(aud, np.stack([L, R], axis=1), sr)

            n_clips = audit.export_audit_clips(
                aud, report, clips_dir, pad_sec=0.3, enable_separate=True, sep_sources=2
            )
            # 2 条相邻 suspect 合并成 1 个切片 + 1 个 gap 切片 = 共 2 个切片
            self.assertEqual(n_clips, 2)
            self.assertEqual(report["separated_clips_count"], 1)
            self.assertEqual(len(report["review_clips"]), 2)

            # 两条 suspect 指向同一个合并切片，且均已回填 2 路 separated_wavs
            s0, s1 = report["suspect_clauses"]
            self.assertEqual(s0["clip_wav"], s1["clip_wav"])
            self.assertTrue(os.path.isfile(s0["clip_wav"]))
            self.assertEqual(len(s0["separated_wavs"]), 2)
            self.assertTrue(all(os.path.isfile(c["wav"]) for c in s0["separated_wavs"]))

            # 居中单声源 gap 跳过分离
            g0 = report["unaligned_speech_gaps"][0]
            self.assertTrue(os.path.isfile(g0["clip_wav"]))
            self.assertFalse(g0["stereo_spread"]["separated"])
            self.assertNotIn("separated_wavs", g0)

            # 反向联动：fa_asmr_spatial_sep.py --from-audit <audit.json> --only-stereo-spread
            audit_json = os.path.join(td, "audit.json")
            with open(audit_json, "w", encoding="utf-8") as f:
                json.dump({"ok": True, "tasks": [{"name": "track", "audio": aud, "audit": report}]}, f, ensure_ascii=False)
            out_sep = os.path.join(td, "from_audit_sep")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = ssep.main([
                    "--from-audit", audit_json,
                    "--out-dir", out_sep,
                    "--sources", "2",
                    "--pad-sec", "0.3",
                    "--only-stereo-spread",
                    "--json",
                ])
            self.assertEqual(code, 0)
            res = json.loads(buf.getvalue())
            self.assertTrue(res["ok"])
            self.assertEqual(res["segments_count"], 2)
            self.assertEqual(res["separated_count"], 1)
            by_kind = {seg["kind"]: seg for seg in res["segments"]}
            self.assertTrue(by_kind["suspect"]["separated"])
            self.assertEqual(len(by_kind["suspect"]["channels"]), 2)
            self.assertFalse(by_kind["gap"]["separated"])
            self.assertEqual(by_kind["gap"]["channels"], [])

    def test_isolated_speaker_jump_and_priority_only_clipping(self):
        import soundfile as sf

        # 构造多人轨：
        #   - A 在 1.0..3.0s 正常说话，然后在 60.0..62.0s 正常说话
        #   - B 在 10.0..14.0s 说话（期间 A 的短称呼 ci2 孤立重叠前吸到 11.0..11.8s -> 形态 A isolated_speaker_jump！）
        #   - B 在 30.0..33.0s 与 35.0..38.0s 连续说话（中间 A 的短称呼 ci5 楔入 33.5..34.3s -> 形态 B isolated_speaker_jump！）
        #   - B 在 39.0..41.0s 出现一条轻度低分句 ci7 (-3.6, normal priority)
        #   - A/B 在 62.5..65.5s 正常持续叠声（彼此相邻 <2s -> 不应误报 isolated_speaker_jump！）
        alignment_tokens = [
            ["ko", "n", "ni", "chi", "wa"],   # ci0 A 1..3
            ["o", "ha", "yo", "u", "ne"],     # ci1 B 10..14
            ["go", "shu", "ji", "n"],         # ci2 A 11..11.8 (Case A jump)
            ["ma", "ta", "ne", "yo", "o"],    # ci3 B 30..33
            ["so", "u", "da", "yo", "o"],     # ci4 B 35..38
            ["ne", "e", "sa", "ma"],          # ci5 A 33.5..34.3 (Case B jump)
            ["ki", "mo", "chi", "i", "i"],    # ci6 B 39..41 (mild low_score -3.6 -> normal)
            ["da", "i", "su", "ki", "da"],    # ci7 A 60..62
            ["su", "ki", "da", "yo", "o"],    # ci8 A 62.5..64.0 (normal overlap)
            ["wa", "ta", "shi", "mo", "o"],   # ci9 B 63.0..64.5 (normal overlap)
            ["a", "ri", "ga", "to", "o"],     # ci10 B 65.0..66.5
        ]

        def _mk(n, st, en, sc):
            step = (en - st) / float(n)
            return [{"original_start": st + i * step, "original_end": st + (i + 1) * step, "score": sc} for i in range(n)]

        alignment_results = (
            _mk(5, 1.0, 3.0, -1.0)
            + _mk(5, 10.0, 14.0, -1.0)
            + _mk(4, 11.0, 11.8, -1.2)
            + _mk(5, 30.0, 33.0, -1.0)
            + _mk(5, 35.0, 38.0, -1.0)
            + _mk(4, 33.5, 34.3, -1.2)
            + _mk(5, 39.0, 41.0, -3.6)
            + _mk(5, 60.0, 62.0, -1.0)
            + _mk(5, 62.5, 64.0, -1.0)
            + _mk(5, 63.0, 64.5, -1.0)
            + _mk(5, 65.0, 66.5, -1.0)
        )
        rep = audit.build_alignment_audit(
            alignment_tokens=alignment_tokens,
            alignment_results=alignment_results,
            ono_clause_flags=[0] * len(alignment_tokens),
            clause_surfaces=["こんにちは", "おはようね", "ご主人", "またねよ", "そうだよ", "ねえさま", "きもちいい", "だいすきだ", "すきだよ", "わたしも", "ありがとう"],
            clause_speakers=[("A",), ("B",), ("A",), ("B",), ("B",), ("A",), ("B",), ("A",), ("A",), ("B",), ("B",)],
            clause_line_nos=list(range(1, 12)),
        )
        by_ci = {s["clause_index"]: s for s in rep["suspect_clauses"]}
        self.assertIn(2, by_ci)
        self.assertIn("isolated_speaker_jump", by_ci[2]["reasons"])
        self.assertEqual(by_ci[2]["priority"], "high")
        self.assertIn(5, by_ci)
        self.assertIn("isolated_speaker_jump", by_ci[5]["reasons"])
        self.assertEqual(by_ci[5]["priority"], "high")
        # 轻度低分 ci6 标为 normal；正常持续叠声 ci8/ci9 不误报
        self.assertIn(6, by_ci)
        self.assertEqual(by_ci[6]["priority"], "normal")
        self.assertNotIn(8, by_ci)
        self.assertNotIn(9, by_ci)
        self.assertEqual(rep["priority_review_count"], 2)

        # 验证 priority_only=True 仅导出 2 个高危 WAV 并回填到 priority_review，不切轻度低分 ci6
        with tempfile.TemporaryDirectory() as td:
            aud = os.path.join(td, "mix.wav")
            clips_dir = os.path.join(td, "clips")
            sf.write(aud, np.zeros((16000 * 70, 2), dtype=np.float32), 16000)
            n_clips = audit.export_audit_clips(aud, rep, clips_dir, pad_sec=0.2, priority_only=True)
            self.assertEqual(n_clips, 2)
            self.assertTrue(all("clip_wav" in pr and os.path.isfile(pr["clip_wav"]) for pr in rep["priority_review"]))
            self.assertNotIn("clip_wav", by_ci[6])

    def test_internal_gap_stretch_and_cross_speaker_engulf_high_priority(self):
        # 复现 03 #28 形态：
        #   ci0 (A): 句首未录词前吸到 10.0..10.5s，后半段正常落在 18.0..20.0s（均分高至 -1.4，但内部裂口 7.5s 且吞没 B 的 ci1/ci2）
        #   ci1 (B): 12.0..14.0s 正常独白
        #   ci2 (B): 15.0..17.0s 正常独白
        #   ci3 (A) 与 ci4 (B): 正常双人同步叠声（A 30.0..32.5s 包夹 B 30.5..31.8s，但无断层且正常语速 -> 不应误报！）
        alignment_tokens = [
            ["de", "su", "ka", "ra", "a", "n", "shi", "n", "shi", "te"],  # ci0 A
            ["o", "n", "na", "no", "ko"],                                 # ci1 B
            ["ma", "ru", "da", "shi", "da"],                              # ci2 B
            ["ki", "mo", "chi", "i", "i", "ne", "yo", "ne", "e", "e"],    # ci3 A (normal overlap)
            ["su", "ki", "da", "yo"],                                     # ci4 B (normal overlap)
        ]

        def _mk(n, st, en, sc):
            step = (en - st) / float(n)
            return [{"original_start": st + i * step, "original_end": st + (i + 1) * step, "score": sc} for i in range(n)]

        alignment_results = (
            _mk(4, 10.0, 10.5, -2.2) + _mk(6, 18.0, 20.0, -0.9)  # ci0: gap=7.5s, mean_score=-1.42
            + _mk(5, 12.0, 14.0, -3.4)                           # ci1: B
            + _mk(5, 15.0, 17.0, -3.3)                           # ci2: B
            + _mk(10, 30.0, 32.5, -1.0)                          # ci3: A normal overlap
            + _mk(4, 30.5, 31.8, -1.0)                           # ci4: B normal overlap
        )
        rep = audit.build_alignment_audit(
            alignment_tokens=alignment_tokens,
            alignment_results=alignment_results,
            ono_clause_flags=[0, 0, 0, 0, 0],
            clause_surfaces=["ですから安心して", "おんなのこ", "まるだしだ", "きもちいいねよね", "すきだよ"],
            clause_speakers=[("A",), ("B",), ("B",), ("A",), ("B",)],
            clause_line_nos=[10, 11, 12, 20, 21],
        )
        by_ci = {s["clause_index"]: s for s in rep["suspect_clauses"]}
        self.assertIn(0, by_ci)
        self.assertIn("internal_gap_stretch", by_ci[0]["reasons"])
        self.assertIn("cross_speaker_engulf", by_ci[0]["reasons"])
        self.assertEqual(by_ci[0]["priority"], "high")
        self.assertAlmostEqual(by_ci[0]["internal_gap_info"]["gap_sec"], 7.5, places=1)
        self.assertEqual(by_ci[0]["engulf_info"]["engulfed_count"], 2)
        self.assertEqual(by_ci[0]["engulf_info"]["engulfed_lines"], [11, 12])
        # 正常同步叠声不应误报
        self.assertNotIn(3, by_ci)
        self.assertNotIn(4, by_ci)
        pr_cis = [p["clause_index"] for p in rep["priority_review"] if p["kind"] == "suspect"]
        self.assertIn(0, pr_cis)

    def test_compact_clips_outputs_16k_mono_pcm16(self):
        import soundfile as sf

        sr = 48000
        t = np.arange(sr * 4, dtype=np.float32) / float(sr)
        sig = 0.4 * np.sin(2.0 * np.pi * 500.0 * t)
        report = {
            "suspect_clauses": [
                {"clause_index": 0, "line": 1, "speaker": "", "start": 1.0, "end": 2.5, "reasons": ["internal_gap_stretch"], "priority": "high"},
            ],
            "unaligned_speech_gaps": [],
        }
        with tempfile.TemporaryDirectory() as td:
            aud = os.path.join(td, "hi_res.wav")
            clips_dir = os.path.join(td, "clips")
            sf.write(aud, np.stack([sig, sig], axis=1), sr, subtype="PCM_24")
            n_clips = audit.export_audit_clips(
                aud, report, clips_dir, pad_sec=0.2, priority_only=True, compact_clips=True
            )
            self.assertEqual(n_clips, 1)
            clip_p = report["suspect_clauses"][0]["clip_wav"]
            info = sf.info(clip_p)
            self.assertEqual(info.samplerate, 16000)
            self.assertEqual(info.channels, 1)
            self.assertEqual(info.subtype, "PCM_16")
            self.assertTrue(report["review_clips"][0]["compact"])

    def test_solo_speaker_overlap_distinguishes_pure_solo_vs_true_stereo_overlap(self):
        sr = 16000
        dur = 12
        n = sr * dur
        rng = np.random.default_rng(42)
        L = np.zeros(n, dtype=np.float32)
        R = np.zeros(n, dtype=np.float32)

        # 1.0..3.5s: 纯净单声源（带随机调制信号，左右声道严格按固定增益线性相关 -> msc~1.0, cres~0.0）
        s_solo = rng.standard_normal(int(2.5 * sr)).astype(np.float32)
        L[int(1.0 * sr):int(3.5 * sr)] = 0.6 * s_solo
        R[int(1.0 * sr):int(3.5 * sr)] = 0.4 * s_solo

        # 5.0..7.5s: 双人立体声真叠声（两路独立非相关信号，即使一侧大出 >12dB 也不应误报！）
        s_a = rng.standard_normal(int(2.5 * sr)).astype(np.float32)
        s_b = rng.standard_normal(int(2.5 * sr)).astype(np.float32)
        L[int(5.0 * sr):int(7.5 * sr)] = 0.9 * s_a + 0.05 * s_b
        R[int(5.0 * sr):int(7.5 * sr)] = 0.05 * s_a + 0.15 * s_b  # ILD > 12dB, 但两路独立 -> msc 低, cres 高

        # 8.5..10.5s: 单声源参考基线段
        s_ref = rng.standard_normal(int(2.0 * sr)).astype(np.float32)
        L[int(8.5 * sr):int(10.5 * sr)] = 0.5 * s_ref
        R[int(8.5 * sr):int(10.5 * sr)] = 0.5 * s_ref

        wav_stereo = np.stack([L, R], axis=0)

        def _mk(n_tok, st, en, sc):
            step = (en - st) / float(n_tok)
            return [{"original_start": st + i * step, "original_end": st + (i + 1) * step, "score": sc} for i in range(n_tok)]

        alignment_tokens = [
            ["ko", "n", "ni", "chi", "wa"],  # ci0 A 1.0..3.4 (-0.9)
            ["o", "ha", "yo", "u", "ne"],    # ci1 B 1.2..2.8 (-2.1, 与 ci0 在纯单声源段重叠 -> strict_solo high!)
            ["su", "ki", "da", "yo", "o"],   # ci2 A 5.0..7.4 (-1.0)
            ["wa", "ta", "shi", "mo", "o"],  # ci3 B 5.2..7.2 (-1.1, 真叠声 ILD>12dB -> 不应误报!)
            ["a", "ri", "ga", "to", "o"],    # ci4 A 8.5..10.5 (-1.0)
        ]
        alignment_results = (
            _mk(5, 1.0, 3.4, -0.9)
            + _mk(5, 1.2, 2.8, -2.1)
            + _mk(5, 5.0, 7.4, -1.0)
            + _mk(5, 5.2, 7.2, -1.1)
            + _mk(5, 8.5, 10.5, -1.0)
        )
        rep = audit.build_alignment_audit(
            alignment_tokens=alignment_tokens,
            alignment_results=alignment_results,
            ono_clause_flags=[0, 0, 0, 0, 0],
            clause_surfaces=["こんにちは", "おはようね", "すきだよ", "わたしも", "ありがとう"],
            clause_speakers=[("A",), ("B",), ("A",), ("B",), ("A",)],
            clause_line_nos=[1, 2, 3, 4, 5],
            diagnostics={"stereo_wav_16k": wav_stereo, "frame_sec": 0.02, "duration_sec": float(dur)},
        )
        by_ci = {s["clause_index"]: s for s in rep["suspect_clauses"]}
        self.assertIn(1, by_ci)
        self.assertIn("solo_speaker_overlap", by_ci[1]["reasons"])
        self.assertTrue(by_ci[1]["overlap_info"]["strict_solo"])
        self.assertEqual(by_ci[1]["priority"], "high")
        self.assertNotIn(2, by_ci)
        self.assertNotIn(3, by_ci)
        self.assertEqual(rep["priority_review_count"], 1)

    def test_neutral_ascii_clip_filenames_avoid_sensitive_stem_leak(self):
        import soundfile as sf

        sr = 16000
        sig = np.zeros((sr * 4, 2), dtype=np.float32)
        report = {
            "suspect_clauses": [
                {"clause_index": 0, "line": 12, "speaker": "凛咲", "start": 1.0, "end": 2.2, "reasons": ["skipped_semantic"], "priority": "high"},
            ],
            "unaligned_speech_gaps": [],
        }
        with tempfile.TemporaryDirectory() as td:
            sensitive_stem = "トラック2\u2017お隣のセックスを盗み聞きＨ"
            aud = os.path.join(td, f"{sensitive_stem}.wav")
            clips_dir = os.path.join(td, "clips")
            sf.write(aud, sig, sr)
            n_clips = audit.export_audit_clips(aud, report, clips_dir, pad_sec=0.2, priority_only=True)
            self.assertEqual(n_clips, 1)
            clip_p = report["suspect_clauses"][0]["clip_wav"]
            clip_bname = os.path.basename(clip_p)
            self.assertTrue(clip_bname.isascii(), f"Expected pure ASCII clip filename, got: {clip_bname}")
            self.assertTrue(clip_bname.startswith("t02_"))
            self.assertEqual(report["review_clips"][0]["source_stem"], sensitive_stem)

    def test_priority_review_includes_neighbor_context_and_caps_merged_window_duration(self):
        # 1) 验证 priority_review 自动携带前 2 句 / 后 2 句上下文
        alignment_tokens = [
            ["ko", "n", "ni", "chi", "wa"],   # L1 1..2s
            ["o", "ha", "yo", "u", "ne"],     # L2 2.5..4s
            ["su", "ki", "da", "yo", "o"],    # L3 skipped -> high priority suspect
            ["ma", "ta", "ne", "yo", "o"],    # L4 6..7.5s
            ["a", "ri", "ga", "to", "o"],     # L5 8..9.5s
        ]

        def _mk(n, st, en, sc):
            step = (en - st) / float(n)
            return [{"original_start": st + i * step, "original_end": st + (i + 1) * step, "score": sc} for i in range(n)]

        alignment_results = (
            _mk(5, 1.0, 2.0, -1.0)
            + _mk(5, 2.5, 4.0, -1.0)
            + [{"skipped": True, "score": -10.0} for _ in range(5)]
            + _mk(5, 6.0, 7.5, -1.0)
            + _mk(5, 8.0, 9.5, -1.0)
        )
        rep = audit.build_alignment_audit(
            alignment_tokens=alignment_tokens,
            alignment_results=alignment_results,
            ono_clause_flags=[0, 0, 0, 0, 0],
            clause_surfaces=["こんにちは", "おはようね", "すきだよ", "またねよ", "ありがとう"],
            clause_speakers=[("凛咲",), ("文乃",), ("凛咲",), ("文乃",), ("凛咲",)],
            clause_line_nos=[1, 2, 3, 4, 5],
        )
        self.assertEqual(len(rep["priority_review"]), 1)
        pr = rep["priority_review"][0]
        self.assertEqual(len(pr.get("context_before", [])), 2)
        self.assertEqual(len(pr.get("context_after", [])), 2)
        self.assertIn("こんにちは", pr["context_before"][0])
        self.assertIn("おはようね", pr["context_before"][1])
        self.assertIn("またねよ", pr["context_after"][0])
        self.assertIn("ありがとう", pr["context_after"][1])

        # 2) 验证 _merge_interval_items 传入 max_window_sec=12.0 时自动断开，不拼成 25s 超长切片
        chain = [
            {"start": 0.0, "end": 4.0},
            {"start": 5.0, "end": 9.0},
            {"start": 10.0, "end": 14.0},
            {"start": 15.0, "end": 19.0},
            {"start": 20.0, "end": 24.0},
        ]
        wins = ssep._merge_interval_items(chain, merge_gap_sec=1.5, max_window_sec=12.0)
        self.assertGreaterEqual(len(wins), 3)
        self.assertTrue(all((w["end"] - w["start"]) <= 12.0 for w in wins))


if __name__ == "__main__":
    unittest.main()
