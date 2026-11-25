# -*- coding: utf-8 -*-
"""tools/compact_subs.py 原地修轴 (patch)、跨视角共句去重 (--dedup-pov)、锚点防错与质量体检单元测试。"""

import contextlib
import io
import json
import os
import pathlib
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.join(REPO_ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import compact_subs as cs  # noqa: E402


class TestCompactSubs(unittest.TestCase):
    def _run_silent(self, argv):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return cs.main(argv)

    def _write_sample_pair(self, folder: str, stem: str, cues):
        srt_blocks = []
        cues_sec = []
        for idx, (st, en, spk, body) in enumerate(cues, 1):
            ts = f"{cs.format_srt_time(st)} --> {cs.format_srt_time(en)}"
            full = f"【{spk}】{body}" if spk else body
            srt_blocks.append(f"{idx}\n{ts}\n{full}\n")
            cues_sec.append((st, en, spk, body))
        pathlib.Path(folder, f"{stem}.srt").write_text("\n".join(srt_blocks), encoding="utf-8")
        pathlib.Path(folder, f"{stem}.lrc").write_text(cs.render_lrc_from_cues(cues_sec), encoding="utf-8")

    def test_patch_atomic_delete_update_replace_insert_and_chronological_sort(self):
        with tempfile.TemporaryDirectory() as td:
            self._write_sample_pair(
                td,
                "02_track",
                [
                    (1.0, 2.5, "凛咲", "ねむ……"),
                    (3.0, 9.8, "凛咲", "やば……"),          # #2 尾音被拉长，应缩至 4.5s
                    (11.0, 11.6, "凛咲", "未収録のセリフ"),   # #3 孤立删戏句，应直接删除
                    (15.0, 17.0, "凛咲", "っうぅｘ～～っ"),   # #4 含 PDF 错字全角 ｘ
                ],
            )
            ops_p = os.path.join(td, "ops.json")
            ops = {
                "02_track": [
                    {"op": "update", "cue": 2, "end": 4.5, "expect": "やば"},
                    {"op": "delete", "cue": 3, "expect": "未収録"},
                    {"op": "replace", "cue": 4, "old": "ｘ", "new": ""},
                    {"op": "insert", "start": 6.0, "end": 7.4, "speaker": "文乃", "text": "はいっ……"},
                ]
            }
            pathlib.Path(ops_p).write_text(json.dumps(ops, ensure_ascii=False), encoding="utf-8")

            code = self._run_silent(["patch", "--sub-dir", td, "--ops-json", ops_p])
            self.assertEqual(code, 0)

            patched = cs.parse_srt(pathlib.Path(td, "02_track.srt"))
            self.assertEqual(len(patched), 4)
            # 插入的 6.0..7.4s 条目应自动排在第 3 条，原第 4 条排在第 4 条且去掉了 ｘ
            self.assertEqual([c[0] for c in patched], ["1", "2", "3", "4"])
            self.assertIn("00:00:04,500", patched[1][1])
            self.assertEqual(patched[2][2], "【文乃】はいっ……")
            self.assertEqual(patched[3][2], "【凛咲】っうぅ～～っ")

            # 验证同步重生成的 LRC 可无损通过后续翻译 build + verify
            cn_dir = os.path.join(td, "cn")
            zh_p = os.path.join(td, "zh.json")
            pathlib.Path(zh_p).write_text(
                json.dumps({"02_track": ["1|好困……", "2|糟了……", "3|好的……", "4|呜呜——"]}, ensure_ascii=False),
                encoding="utf-8",
            )
            self.assertEqual(self._run_silent(["build", "--jp-dir", td, "--cn-dir", cn_dir, "--in-json", zh_p, "--require-index"]), 0)
            self.assertEqual(self._run_silent(["verify", "--jp-dir", td, "--cn-dir", cn_dir, "--strict"]), 0)

    def test_dedup_pov_export_and_shared_build_and_bom_strip(self):
        with tempfile.TemporaryDirectory() as td:
            jp_dir = os.path.join(td, "jp")
            cn_dir = os.path.join(td, "cn")
            os.makedirs(jp_dir)
            self._write_sample_pair(
                jp_dir,
                "トラック1_挨拶_主人公視点",
                [
                    (1.0, 2.5, "凛咲", "ねむ……"),
                    (3.0, 5.0, "文乃", "おはようございますっ……！"),
                    (6.0, 8.0, "凛咲", "行ってきます……！"),
                ],
            )
            self._write_sample_pair(
                jp_dir,
                "トラック1_挨拶_夫視点",
                [
                    (1.0, 2.5, "文乃", "忘れ物はないですか？"),
                    (3.0, 5.0, "文乃", "おはようございますっ……！"),
                    (6.0, 8.0, "凛咲", "行ってきます……！"),
                ],
            )
            exp_p = os.path.join(td, "compact_dedup.json")
            self.assertEqual(self._run_silent(["export", "--jp-dir", jp_dir, "--out", exp_p, "--dedup-pov"]), 0)
            dedup = json.loads(pathlib.Path(exp_p).read_text(encoding="utf-8"))
            self.assertEqual(dedup["format"], "dedup_v1")
            self.assertEqual(dedup["stats"]["shared_unique"], 2)
            self.assertEqual(dedup["stats"]["saved_duplicate_cues"], 2)

            # 故意注入 \ufeff BOM 测试自动剥离，并通过 @S0001 / @S0002 展开共句
            zh_payload = {
                "shared": ["\ufeffS0001|早上好……！", "S0002|我们走了……！"],
                "tracks": {
                    "トラック1_挨拶_主人公視点": ["\ufeff1|好困……", "2|@S0001", "3|@S0002"],
                    "トラック1_挨拶_夫視点": ["1|有没有忘带东西？", "2|@S0001", "3|@S0002"],
                },
            }
            zh_p = os.path.join(td, "zh_dedup.json")
            pathlib.Path(zh_p).write_text(json.dumps(zh_payload, ensure_ascii=False), encoding="utf-8")
            self.assertEqual(self._run_silent(["build", "--jp-dir", jp_dir, "--cn-dir", cn_dir, "--in-json", zh_p, "--require-index", "--strict"]), 0)
            self.assertEqual(self._run_silent(["verify", "--jp-dir", jp_dir, "--cn-dir", cn_dir, "--strict"]), 0)

    def test_require_index_blocks_offset_canceling_drift_and_verify_catches_lints(self):
        with tempfile.TemporaryDirectory() as td:
            jp_dir = os.path.join(td, "jp")
            cn_dir = os.path.join(td, "cn")
            os.makedirs(jp_dir)
            self._write_sample_pair(
                jp_dir,
                "トラック1_A",
                [
                    (1.0, 2.0, "凛咲", "おはようございます"),
                    (3.0, 4.0, "凛咲", "大丈夫ですか"),
                    (5.0, 6.0, "凛咲", "ちゅぱっ……れろっ……"),
                ],
            )
            self._write_sample_pair(
                jp_dir,
                "トラック1_B",
                [
                    (1.0, 2.0, "凛咲", "おはようございます"),
                ],
            )

            # 1) 抵消型错位：漏了第 2 条但把第 3 条拆成两行，总条数仍为 3，但锚点 1, 3, 3 必须立即报错拦截
            bad_zh_p = os.path.join(td, "bad_drift.json")
            pathlib.Path(bad_zh_p).write_text(
                json.dumps({"トラック1_A": ["1|早上好", "3|啾啪……", "3|舔舔……"]}, ensure_ascii=False),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as ctx:
                cs.main(["build", "--jp-dir", jp_dir, "--cn-dir", cn_dir, "--in-json", bad_zh_p, "--require-index"])
            self.assertIn("行号锚点错位", str(ctx.exception))

            # 2) 验证质量巡检能检出：未翻译日文词(大丈夫)、禁用拟声词(滋溜) 与 跨视角同句异译(早上好 vs 早啊)
            warn_zh_p = os.path.join(td, "warn.json")
            pathlib.Path(warn_zh_p).write_text(
                json.dumps(
                    {
                        "トラック1_A": ["1|早上好", "2|大丈夫吗", "3|啾啪……滋溜……"],
                        "トラック1_B": ["1|早啊"],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            self.assertEqual(self._run_silent(["build", "--jp-dir", jp_dir, "--cn-dir", cn_dir, "--in-json", warn_zh_p]), 0)
            buf_out, buf_err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                v_code = cs.main(["verify", "--jp-dir", jp_dir, "--cn-dir", cn_dir, "--strict"])
            self.assertEqual(v_code, 1)
            err_txt = buf_err.getvalue()
            self.assertIn("大丈夫", err_txt)
            self.assertIn("滋溜", err_txt)
            self.assertIn("跨视角同句异译", err_txt)


if __name__ == "__main__":
    unittest.main()
