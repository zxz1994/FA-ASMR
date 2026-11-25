[English below](#-which-file-should-i-download)

## 📦 下载哪个？

| 你的电脑 | 下载文件 | 大小 |
| :--- | :--- | :--- |
| 没有 NVIDIA 显卡，或不确定 | `FA-ASMR-v2.6.0-Windows-CPU.zip` | 约 1.5 GB |
| NVIDIA 显卡，显存 4 GB 以上（推荐） | `FA-ASMR-v2.6.0-Windows-GPU.zip.001` / `.002` / `.003` **三个分卷全部下载**，再加上 `merge_split_gpu_zip.bat` | 合计约 3.8 GB |

两个版本都已内置 MMS-FA 基础模型、yomikata dBert 注音权重和 ASMR LoRA，**解压即用，不需要安装 Python 或 PyTorch，全程离线**。

## 🚀 使用步骤

**CPU 版**
1. 解压 `FA-ASMR-v2.6.0-Windows-CPU.zip`
2. 双击 `运行FA-ASMR.bat`

**GPU 版**
1. 把三个分卷和 `merge_split_gpu_zip.bat` 放到**同一个文件夹**
2. 双击 `merge_split_gpu_zip.bat`，合并出 `FA-ASMR-v2.6.0-Windows-GPU.zip`
3. 解压后双击 `运行FA-ASMR.bat`

> 已安装 7-Zip 的话，也可以直接右键 `.001` 分卷解压，跳过合并这一步。

**开始打轴**：把音频（wav / mp3 / flac 等）和同名的日文台本 `.txt` 一起拖进窗口，程序会自动配对，点击开始即可。字幕（`.lrc` + `.srt`）输出在音频所在目录。台本格式可以点左下角的「?」查看说明。

## ✨ v2.6.0 更新内容

- **异构双视角主客场判定全面升级（`multi_persp`）**：
  - 修复了 `raw_cl` 跨空合并 `SHARED` 同步点的底层 Bug（加入连续非共享句距与时间跨度约束），彻底避免片头/场景切换的主场戏被后半段吞并错判。
  - 补齐同步段清晰度分差判据（`SPLIT_SYNC`），优化主导权翻转检测与弱句区（`SUBTHRESH`）的加权防抖归属。
- **客视角 4 维正交物理拦截与整句补全**：
  - 引入 **`SimSeg` 同步漂移软惩罚与硬截断**、**跨视角音素同源度 `ph_sim` 兜底**、**低漂移补全锚点严控** 与 **定宽对称滑窗强锚点密度门控**，在异构剪辑 / 隔间双视角下大幅消除不在场角色的假阳性幽灵串入。
  - 新增按台本行（`line_group`）为单位的**客视角整句投影补全与孤立残肢清理**，解决以往客视角台词被前景喘息遮挡后吐出半截残词的问题。
- **多角色字幕组装与拟声词过滤优化**：
  - 实时透传每句主客归属（`_mp_home`），双人同屏时优先按当前段主场居左排序，收紧 LRC 拼行策略避免短暂插话打乱长句。
  - 新增 ASMR 慢语速未完结短语无打断自动合并（如 `それまで……ガマン……`），大幅减少冗余碎句与高潮段 LRC 闪屏；补全黏腻水声词干（`れぢゅ` / `ねぢゅ` / `ぢゅぐ` / `ぬりゅ`）。
- **模块化架构重构与 CLI / 翻译工具链增强**：
  - 代码库正式重构为清晰的 `fa_asmr/{core,text,aligners,pipeline,ui}` 模块化包架构，新增显式配对清单 `--manifest`、`--audit-clips-compact` 瘦身切片，以及 `tools/compact_subs.py` 的跨视角去重（`--dedup-pov`）与原地修轴（`patch`）能力。

免安装版同样支持命令行调用：`FA-ASMR.exe --cli ... --json-out summary.json`（免安装版没有控制台窗口，结构化结果请通过 `--json-out` 读取）。完整命令见 [README](https://github.com/zxz1994/FA-ASMR#readme) 和 [AGENTS.md](https://github.com/zxz1994/FA-ASMR/blob/main/AGENTS.md)。

## 🐞 遇到问题

- 程序异常退出时，会在 exe 同级目录生成 `FA-ASMR_crash.log`
- 怀疑安装不完整时，可以在命令行运行 `FA-ASMR.exe --self-test`，结果会写入 `FA-ASMR_selftest.log`

提 Issue 时附上这两个文件，能更快定位问题。

---

## 📦 Which file should I download?

| Your PC | Download | Size |
| :--- | :--- | :--- |
| No NVIDIA GPU, or not sure | `FA-ASMR-v2.6.0-Windows-CPU.zip` | ~1.5 GB |
| NVIDIA GPU with 4 GB+ VRAM (recommended) | **All three parts** `FA-ASMR-v2.6.0-Windows-GPU.zip.001` / `.002` / `.003`, plus `merge_split_gpu_zip.bat` | ~3.8 GB total |

Both builds ship with the MMS-FA base model, yomikata dBert reading weights and the ASMR LoRA built in. **Just extract and run: no Python or PyTorch install, fully offline.**

## 🚀 Getting started

**CPU build**
1. Extract `FA-ASMR-v2.6.0-Windows-CPU.zip`
2. Double-click `运行FA-ASMR.bat` (the "Run FA-ASMR" launcher)

**GPU build**
1. Put all three parts and `merge_split_gpu_zip.bat` in the **same folder**
2. Double-click `merge_split_gpu_zip.bat` to produce `FA-ASMR-v2.6.0-Windows-GPU.zip` (its console messages are in Chinese; just wait for it to finish)
3. Extract it, then double-click `运行FA-ASMR.bat`

> If you have 7-Zip, you can skip the merge step and extract directly from the `.001` part.

**To start aligning:** Drag your audio (wav / mp3 / flac, etc.) and the Japanese script `.txt` with the same name into the window. Files pair automatically; then press Start. Subtitles (`.lrc` + `.srt`) are written next to the audio. Click the "?" button at the bottom-left for the script format guide. The UI language can be switched to English in the app.

## ✨ What's New in v2.6.0

- **Heterogeneous Dual-POV Home/Away Segmentation Overhaul (`multi_persp`)**:
  - Fixed the non-local `raw_cl` clustering bug that merged distant sync points across single-speaker scenes into giant `SHARED` blocks, preventing intro or scene-switch home segments from being swallowed by the opposite perspective.
  - Added acoustic clarity margin checks (`SPLIT_SYNC`) for synced segments, improved dominance-flip detection, and stabilized `SUBTHRESH` assignment against small noisy chunks.
- **4D Physical Ghost Filtering & Line-Level Away Completion**:
  - Added **`SimSeg` timing drift penalty/cutoff**, **cross-POV phoneme similarity (`ph_sim`) verification**, **low-drift rescue anchor gating**, and **fixed-width symmetric strong-anchor density filtering**, cleanly eliminating off-screen ghost bleed-through in partitioned / heterogeneous dual-POV tracks.
  - Added script-line (`line_group`) away-side completion and orphan fragment pruning so partially masked off-mic lines are recovered as complete sentences instead of broken fragments.
- **Multi-Speaker Subtitle Assembly & Vocalization Filtering**:
  - Propagates per-line live home/away ownership (`_mp_home`) so the active home speaker always leads on shared lines, and tightens LRC simultaneous line merging.
  - Merges short unfinished slow-ASMR phrases without intervening speakers, reducing fragmented cues and LRC flicker; added viscous water/kissing stems (`れぢゅ`, `ねぢゅ`, `ぢゅぐ`, `ぬりゅ`) to the onomatopoeia gate.
- **Modular Package Refactor & CLI / Translation Tooling**:
  - Reorganized the codebase into a clean `fa_asmr/{core,text,aligners,pipeline,ui}` package hierarchy; added `--manifest` explicit pairing, `--audit-clips-compact`, and `tools/compact_subs.py` dual-POV deduplication (`--dedup-pov`) plus atomic in-place subtitle patching (`patch`).

The portable build also works from the command line: `FA-ASMR.exe --cli ... --json-out summary.json`. The portable build has no console window, so read structured results via `--json-out`. See the [README](https://github.com/zxz1994/FA-ASMR/blob/main/README_EN.md) and [AGENTS.md](https://github.com/zxz1994/FA-ASMR/blob/main/AGENTS.md) for all commands.

## 🐞 Troubleshooting

- If the app exits unexpectedly, it writes `FA-ASMR_crash.log` next to the exe.
- If you suspect an incomplete install, run `FA-ASMR.exe --self-test` from a command prompt; results go to `FA-ASMR_selftest.log`.

Attaching both files to an Issue makes problems much faster to track down.
