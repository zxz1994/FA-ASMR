# FA-ASMR (`v2.5.0`)

**English** | [简体中文](README.md) | [Agent Guide (AGENTS.md)](AGENTS.md)

> **High-Precision Pure-Acoustic Forced Alignment Engine for Japanese ASMR & Voice Dramas**
>
> Feed an official Japanese script and voice audio to generate millisecond-accurate `.lrc` and `.srt` subtitles completely offline. Ships with both a dark desktop studio UI and a headless CLI + [`AGENTS.md`](AGENTS.md) workflow for seamless integration with external AI Agents.

---

## 1. Core Capabilities

In Japanese ASMR and voice drama subtitling where official scripts are available, general-purpose ASR models like Whisper frequently drop whispered dialogue, hallucinate proper nouns, and merge utterances into long blocks. **FA-ASMR** takes a pure physical acoustic forced-alignment approach (Meta MMS-FA + custom ASMR LoRA `fa_asmr_v9e10.pt`) engineered specifically for binaural voice recordings:

* **100% Script Fidelity with Direct Anchor Slicing**: Aligns directly against the official script and restores exact original characters via `SequenceMatcher` slicing (preserving standalone dakuten/handakuten, NFKC symbols, pre-existing `{漢字|かな}` ruby annotations, leading opening quotes, and silent sokuon).
* **Binaural Better-Ear Noise Resilience**: Dynamically evaluates left/right channel SNR and noise floors on stereo binaural recordings to suppress mic wind, cloth rustling, and single-ear interference.
* **Three Specialized Alignment Modes**:
  * **01 Single-Track Sequential (`single`)**: Standard single-speaker dramas using a Numba JIT multi-threaded global CTC Viterbi decoder with silence snapping.
  * **02 Multi-Speaker / Reordered (`multi_speaker`)**: Handles multi-VA split-track mixes, interleaved lines, local script reordering, and overlapping speech/breaths via per-speaker stream alignment and stereo STFT coherence solo masking.
  * **03 Multi-Perspective Joint Alignment (`multi_persp`)**: Built for dual-POV recordings where **both perspectives contain active dialogue with complementary clarity** across speakers or segments; calibrates inter-track time offset $\Delta t$, selects the clearer acoustic side per segment, and exports independent `.lrc` and `.srt` timelines for both POV A and POV B. *(Note: if the second track is merely a wall-muffled / low-pass version with no complementary clarity, align the clear primary track with Mode 01/02 instead.)*
* **Built-In Onomatopoeia Gate & Smart Segmentation (`ono_filter` / `resegment` / `merge_short`)**:
  * Filters timing-breaking pure kissing/licking/swallowing/breath-only lines while protecting mid-sentence vocalizations and elongated real words (e.g., `ろぉ～く`, `い～～ち`);
  * Automatically segments on acoustic pauses and punctuation, bridges short comma-separated words, and merges overly fragmented clauses so output subtitles are immediately watchable and translation-ready.
* **Obsidian Acoustic Console Desktop UI (100% Standalone — No AI Agent Required)**: Dark studio console frontend featuring Windows High-DPI awareness, drag-and-drop track pairing/swapping, instant 3-mode switching, and live Chinese / Japanese / English UI hot-switching. Regular users can simply launch the app and align tracks completely offline with zero AI Agent or LLM setup.
* **Offline Precision Audit Loop (New in `v2.5.0`)**:
  * **Single-Pass Tiered Ruby Glossary & Overrides (`--export-ruby` / `--ruby-map`)**: Loads the model once per directory to export a collocation-deduped glossary with high-risk polyphones, affix-split single kanji (e.g., `お母様`), and speaker names prioritized at the top; supports `global` and `by_file -> by_line` overrides without splitting compounds;
  * **Post-Alignment Acoustic Audit Radar (`fa_asmr_audit.py` / `--audit`)**: Zero extra inference overhead; flags unaligned active speech gaps (missing lines / VA ad-libs), unrecorded/cut lines, and multi-speaker isolated short-word early pulls (`isolated_speaker_jump`), aggregated into `priority_review` with selective high-priority WAV clipping;
  * **Zero-Dependency Binaural Spatial Separation (`fa_asmr_spatial_sep.py` / `--audit-separate`)**: Pure `numpy`/`scipy` ILD+IPD soft-masking splits genuine multi-speaker overlaps into clean L/C/R WAV stems at native sample rate (automatically skipped on single-speaker tracks; also works standalone for scriptless multi-VA recordings).
* **Auto Encoding Detection & Low VRAM Footprint**: Automatically detects UTF-8, UTF-8-BOM, Shift-JIS (CP932), GB18030, and UTF-16 without manual transcoding; runs CUDA FP16 with a fixed optimal `3000`-frame chunk (`~3.74 GB` peak VRAM, 4 GB+ GPUs supported) and CPU INT8 dynamic quantization.

---

## 2. Quick Start

### 2.1 Pre-Built Windows Releases (Recommended)

No Python or PyTorch setup required—includes the 1.26 GB MMS-FA base model, yomikata dBert weights, and ASMR LoRA adapter out of the box:

* **Windows CPU Edition** (all PCs): Download `FA-ASMR-v2.5.0-Windows-CPU.zip`, extract, and double-click `运行FA-ASMR.bat`.
* **Windows CUDA GPU Edition** (NVIDIA GPUs, 4 GB+ VRAM): Download all split volumes (`FA-ASMR-v2.5.0-Windows-GPU.zip.001`, etc.) and `merge_split_gpu_zip.bat` into the same folder, run the merge script, extract, and launch `运行FA-ASMR.bat`.

> Download: [GitHub Releases (v2.5.0)](https://github.com/zxz1994/FA-ASMR/releases/latest)

### 2.2 Run from Source (Developers / CLI / AI Agents)

Requires Python 3.10–3.12 (64-bit).

```bash
# 1. Clone the repository
git clone https://github.com/zxz1994/FA-ASMR.git
cd FA-ASMR

# 2. Install PyTorch 2.6.0 (choose CUDA or CPU)
# CUDA 12.4:
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# Or CPU:
# pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu

# 3. Install core dependencies
pip install -r requirements.txt

# 4. Install yomikata without extra dev dependencies
pip install --no-deps yomikata
```

**Model Weights**:

| Weight File | Path | Notes |
| :--- | :--- | :--- |
| **MMS-FA Base Model** | `models/hub/checkpoints/model.pt` | ~1.2 GB; auto-downloaded on first run or placed manually for offline use |
| **ASMR LoRA Adapter** | `lora/fa_asmr_v9e10.pt` | **Bundled in repo** (~12.2 MB), fine-tuned for whispered Japanese ASMR |

**Launch Desktop App**: double-click `run.bat` on Windows or run `python FA-ASMR_GUI.py`.

---

## 3. AI Agent & CLI Automation (Optional End-to-End Workflow)

**FA-ASMR runs 100% standalone without any AI Agent, LLM API keys, or network access**—both the desktop GUI and CLI work completely offline out of the box.

When you want full end-to-end automation (*raw PDF/DOCX extraction → ruby audit → forced alignment & acoustic verification → localized subtitle translation*), pair this repo with your preferred **AI Agent** (Antigravity, Claude Code, Codex, Cursor, Gemini CLI, etc.). The repository ships with a dedicated [`AGENTS.md`](AGENTS.md) operating guide, headless CLI ([`fa_asmr_cli.py`](fa_asmr_cli.py)), and compact translation helper ([`tools/compact_subs.py`](tools/compact_subs.py)).

### 3.1 Handing the Workflow to an AI Agent

Point any terminal-capable (and multimodal audio-capable) AI Agent to this repository and your target work folder with a prompt like:

> "Read `E:\FA-ASMR\AGENTS.md` and process `./RJ01234567` end-to-end: clean the script, audit the batch ruby glossary, align and verify high-priority audit clips with `fa_asmr_cli.py` until the Japanese timelines are finalized, then research the work's background and translate in compact format."

Following [`AGENTS.md`](AGENTS.md), the Agent runs a streamlined, zero-rework loop (**always finalizing Japanese timelines before starting translation**):
1. **Script Extraction & Cleanup**: Extracts vertical/multi-tier/color-coded PDFs, strips stage/SE notes, and normalizes `〖Speaker〗` markers;
2. **Single-Pass Batch Ruby Review**: Runs `--batch --export-ruby` to export one collocation-deduped, risk-tiered glossary across all tracks and writes only verified misreadings into `ruby_fixes.json` (`global` + `by_file -> by_line`);
3. **Alignment + Tiered Audit + Multimodal Listening**: Runs `--batch --audit-separate` (auto-skipping spatial separation on single-speaker tracks while splitting true overlaps on multi-speaker tracks), listens to `priority_review` WAV clips (gaps, cut lines, `isolated_speaker_jump` early pulls), and uses `--only` to partially re-align modified tracks;
4. **Background Research & Compact Translation**: Builds a concise character/glossary card (e.g., via DLsite RJ code), strips timestamps with `tools/compact_subs.py export` (~35%–40% input token savings), translates line-by-line, and rebuilds/verifies `.srt` + `.lrc` with 1:1 timestamp parity and zero kana leaks.

### 3.2 `fa_asmr_cli.py` & Companion CLI Usage

Default `--mode auto` automatically selects `multi_persp` when `--audio-b` or dual-POV pairs are detected, `multi_speaker` when ≥2 speakers are present in the script, and `single` otherwise:

```bash
# 1. Fast folder pairing & encoding pre-flight check (no model loaded)
python fa_asmr_cli.py --batch ./work_dir --dry-run --json

# 2. Single-pass merged & risk-tiered ruby glossary across all scripts in a folder
python fa_asmr_cli.py --batch ./work_dir --export-ruby --json-out ./ruby_batch.json

# 3. Batch alignment + acoustic audit + selective priority WAV clips (reuses VRAM; auto-separates multi-speaker overlaps)
python fa_asmr_cli.py --batch ./work_dir --ruby-map ./ruby_fixes.json --audit-separate --audit-out ./audit_batch.json --audit-clips-dir ./audit_clips --json

# 4. Partial rerun for modified tracks only (still reuses a single model load)
python fa_asmr_cli.py --batch ./work_dir --only 05,07,13 --ruby-map ./ruby_fixes.json --audit-separate --audit-out ./audit_realign.json --audit-clips-dir ./audit_clips_realign --json

# 5. Single-track / dual-perspective joint alignment
python fa_asmr_cli.py --audio ./01.flac --script ./01.txt --extend 1.0 --json
python fa_asmr_cli.py --audio ./01_A.wav --audio-b ./01_B.wav --script ./01.txt --audit-separate --json

# 6. Standalone zero-dependency binaural spatial separation (scriptless overlap scanning)
python fa_asmr_spatial_sep.py --audio ./02.flac --auto-overlaps --sources 3 --out-dir ./sep_out --json

# 7. Compact translation export / rebuild / 1:1 parity verification
python tools/compact_subs.py export --jp-dir ./subs_jp --out ./compact_jp.json
python tools/compact_subs.py build  --jp-dir ./subs_jp --cn-dir ./subs_cn --in-json ./zh_batch.json
python tools/compact_subs.py verify --jp-dir ./subs_jp --cn-dir ./subs_cn
```

> **Pre-built `.exe` users**: Headless CLI is also available via `FA-ASMR.exe --cli ... --json-out summary.json` (use `--json-out` in windowed builds where console stdout is suppressed).

---

## 4. Desktop Usage & Script Conventions

* **Supported Audio Formats**: `.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, `.aac`, `.opus`, `.wma`.
* **Smart File Pairing**:
  * Matching stems (`01_track.wav` + `01_track.txt` or `01_track〖Alice〗〖Bob〗.txt`);
  * Dual-POV suffixes (`01_A.wav` / `01_B.wav`, `Track7‗主人公視点.wav` / `Track7‗夫視点.wav`) auto-group with the shared script.
* **Script Guidelines**:
  * Any standard encoding (UTF-8 / UTF-8-BOM / Shift-JIS(CP932) / GB18030 / UTF-16) is read automatically;
  * Strip non-spoken stage/SE notes; convert inline `漢字（かんじ）` readings to `{漢字|かんじ}` or remove the parenthesized reading;
  * Mark speakers with `〖Name〗` (standalone line or inline, simultaneous speech as `〖A〗〖B〗せーの！`) or recurring standalone `【Name】` header lines.
* **Outputs**: Writes `.lrc` (with `Name：` prefixes) and `.srt` (with per-speaker independent timelines and `【Name】` prefixes) beside each audio file.
* **Advanced Hardware Config (`advanced_config.json`)**: Optional read-only tuning for `cpu_threads`, `quantize_int8`, and `cpu_chunk_cap` (default `2000`).

---

## 5. Architecture & Regression Tests

```text
FA-ASMR/
├── AGENTS.md                   # End-to-end AI Agent operating & maintenance guide
├── fa_asmr_cli.py              # Headless CLI (--mode auto / --export-ruby / --ruby-map / --audit-separate / --only / --json)
├── fa_asmr_pipeline.py         # Non-GUI core pipeline, encoding detection, tiered ruby glossary, LRC/SRT & auto-pairing
├── fa_asmr_audit.py            # Post-alignment acoustic audit, dual-POV joint audit, speaker-jump & priority WAV clipping
├── fa_asmr_spatial_sep.py      # Zero-dependency binaural ILD+IPD spatial separator (standalone & audit-linked)
├── tools/compact_subs.py       # Compact translation export / .srt+.lrc rebuild / 1:1 parity verifier
├── FA-ASMR_GUI.py              # Desktop launcher, --cli router & --self-test verification
├── fa_asmr_ui.py               # Obsidian Acoustic Console dark studio UI
├── fa_asmr_i18n.py             # ZH / JA / EN hot-switchable localization
├── align.py                    # Single-track acoustic alignment engine
├── align_multispeaker.py       # Multi-speaker / reordered / STFT solo-mask engine
├── align_multiperspective.py   # Dual-perspective joint calibration & projection engine
├── align_trellis.py            # Global constrained CTC Viterbi decoder (Numba JIT)
├── align_acoustic.py           # Frame RMS, noise floor & binaural Better-Ear mixture
├── align_model.py              # MMS-FA loader, LoRA injection, FP16 / INT8 inference
├── align_post.py               # Silence snapping, timestamp mapping & monotonicity
├── clause.py                   # Shared data contract (F), clause boundaries & comma bridging
├── haruraw2norm.py             # Japanese kanji/kana/romaji to MMS-FA phoneme converter
├── inline_leak_clean.py        # Inline furigana leak cleaner
├── ono_base.py / ono_filter.py / ono_block_corpus.py  # Multi-stage ONO gate & real-word protection
├── resegment.py / merge_short.py / lrc_end_extend.py  # Resegmentation, short-line merge & tail extend
├── furigana/                   # 3-tier Japanese furigana annotator & heteronym overrides
├── lora/fa_asmr_v9e10.pt       # Bundled ASMR-specialized LoRA weights
├── tests/                      # Zero-dependency regression suite (stdlib unittest)
└── build_in_ci.py / FA-ASMR.spec  # GitHub Actions Windows CPU & CUDA release packaging
```

Run the regression test suite (zero external test dependencies):

```bash
python -m unittest discover -s tests -v
```

---

## 6. License & Acknowledgments

Released under the **MIT License**.
* Acoustic alignment foundation based on [Meta MMS-FA](https://pytorch.org/audio/stable/tutorials/forced_alignment_for_multilingual_data_tutorial.html)
* Japanese tokenization & furigana powered by [Fugashi (UniDic)](https://github.com/polm/fugashi), [yomikata](https://github.com/passaglia/yomikata), and [PyKakasi](https://github.com/miurahr/pykakasi)
