# -*- coding: utf-8 -*-
"""双视角体检时钟映射（_peer_shift）与区间并集覆盖率回归测试。"""
import unittest

import numpy as np

from fa_asmr.pipeline.audit import build_multiperspective_audit


class TestPerspectiveGapAudit(unittest.TestCase):
    @staticmethod
    def gaps(a_time, b_time, segments, *, active_side="A", peer_times=(), length=2.0):
        gap_start = a_time if active_side == "A" else b_time
        activity = np.zeros(7500, dtype=bool)
        activity[round(gap_start / 0.02):round((gap_start + length) / 0.02):3] = True
        peer = []
        for start, end in peer_times:
            for i in range(5):
                step = (end - start) / 5
                peer.append(dict(original_start=start + i * step,
                                 original_end=start + (i + 1) * step, score=-1.0))
        skipped = [dict(skipped=True, score=-10.0)] * len(peer)
        n = len(peer_times)
        diag = dict(ctc_nb_active=activity, frame_sec=0.02, duration_sec=150)
        rep = build_multiperspective_audit(
            [["ka"] * 5 for _ in range(n)], skipped if active_side == "A" else peer,
            peer if active_side == "A" else skipped, [0] * n, ["spoken"] * n,
            list(range(n)), list(range(n)), diagnostics_a=diag if active_side == "A" else {},
            diagnostics_b=diag if active_side == "B" else {}, sim_segs=segments, delta_sec=0.0,
        )
        return len(rep["unaligned_speech_gaps"])

    def test_offsets_and_wrong_scene_in_both_directions(self):
        for a, b in ((60, 50), (50, 60), (50, 50)):
            segs = [dict(A_st=a - 5, A_en=a + 10, B_st=b - 5, B_en=b + 10, dt_AB=a - b)]
            for side, peer in (("A", b), ("B", a)):
                with self.subTest(a=a, b=b, side=side):
                    self.assertEqual(self.gaps(a, b, segs, active_side=side,
                                               peer_times=[(peer, peer + 2)]), 0)
                    self.assertEqual(self.gaps(a, b, segs, active_side=side,
                                               peer_times=[(peer + 20, peer + 22)]), 1)

    def test_coincident_speech_without_local_sync_is_retained(self):
        for segments in ([], [dict(A_st=10, A_en=25, B_st=10, B_en=25, dt_AB=0)],
                         [dict(A_st=50, A_en=75, dt_AB=0)]):
            self.assertEqual(self.gaps(60, 60, segments, peer_times=[(60, 62)]), 1)

    def test_ambiguous_or_cross_boundary_mapping_keeps_the_gap(self):
        first = dict(A_st=50, A_en=61, B_st=30, B_en=41, dt_AB=20)
        second = dict(A_st=61, A_en=70, B_st=61, B_en=70, dt_AB=0)
        self.assertEqual(self.gaps(60, 60, [first, second], peer_times=[(60, 62)]), 1)
        first.update(A_en=66, B_en=46)
        second.update(A_st=59, B_st=59)
        self.assertEqual(self.gaps(60, 60, [first, second], peer_times=[(40, 42)]), 1)
        self.assertEqual(self.gaps(63, 63, [second], peer_times=[(63, 65)]), 0)
        # 目标侧多对一映射歧义（A[50,70] 与 A[80,100] 都映射到 B[30,50]）不得吞掉告警
        m1 = dict(A_st=50, A_en=70, B_st=30, B_en=50, dt_AB=20)
        m2 = dict(A_st=80, A_en=100, B_st=30, B_en=50, dt_AB=50)
        self.assertEqual(self.gaps(60, 40, [m1, m2], active_side="A", peer_times=[(40, 42)]), 1)
        # 竞争段仅缺一个目标侧边界（B_en 缺失）仍指向同一目标窗口时，同样不得绕过多对一歧义检查
        m2_partial = dict(A_st=80, A_en=100, B_st=39, dt_AB=41)
        self.assertEqual(self.gaps(60, 40, [m1, m2_partial], active_side="A", peer_times=[(40, 42)]), 1)
        # 无关的历史段缺字段不应破坏有效段映射；相交段含非法或溢出 dt_AB 应安全降级为 None（不抛异常、保留告警）
        unrelated_bad = dict(A_st=0, A_en=10, B_st=0, dt_AB=0)
        self.assertEqual(self.gaps(60, 40, [unrelated_bad, m1], active_side="A", peer_times=[(40, 42)]), 0)
        bad_dt = dict(A_st=50, A_en=70, B_st=30, B_en=50, dt_AB="bad")
        self.assertEqual(self.gaps(60, 40, [bad_dt], active_side="A", peer_times=[(40, 42)]), 1)
        overflow_dt = dict(A_st=50, A_en=70, B_st=30, B_en=50, dt_AB=10**400)
        self.assertEqual(self.gaps(60, 40, [overflow_dt], active_side="A", peer_times=[(40, 42)]), 1)

    def test_concurrent_speakers_cover_the_same_seconds(self):
        segs = [dict(A_st=50, A_en=80, B_st=50, B_en=80, dt_AB=0)]
        self.assertEqual(self.gaps(60, 60, segs, length=10, peer_times=[(62, 63)] * 2), 1)
        self.assertEqual(self.gaps(60, 60, segs, length=10, peer_times=[(62, 63), (65, 67)]), 0)

    def test_two_uncovered_views_deduplicate_only_with_matching_clocks(self):
        def diagnostics(start):
            active = np.zeros(2000, dtype=bool)
            active[start:start + 90:3] = True
            return dict(ctc_nb_active=active, frame_sec=0.02, duration_sec=40)

        segment = dict(A_st=15, A_en=25, B_st=5, B_en=15, dt_AB=10)
        for segments, expected in (([segment], ["A"]), ([], ["B", "A"])):
            report = build_multiperspective_audit(
                [], [], [], [], [], [], [], diagnostics_a=diagnostics(1000),
                diagnostics_b=diagnostics(500), sim_segs=segments,
            )
            self.assertEqual([g["perspective"] for g in report["unaligned_speech_gaps"]], expected)


if __name__ == "__main__":
    unittest.main()
