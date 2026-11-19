# FA-ASMR (`v2.4.0`)

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
* **Obsidian Acoustic Console Desktop UI**: Dark studio console frontend featuring Windows High-DPI awareness, drag-and-drop track pairing/swapping, instant 3-mode switching, and live Chinese / Japanese / English UI hot-switching.
* **Auto Encoding Detection & Low VRAM Footprint**: Automatically detects UTF-8, UTF-8-BOM, Shift-JIS (CP932), GB18030, and UTF-16 without manual transcoding; runs CUDA FP16 with a fixed optimal `3000`-frame chunk (`~3.74 GB` peak VRAM, 4 GB+ GPUs supported) and CPU INT8 dynamic quantization.

---

## 2. Quick Start

### 2.1 Pre-Built Windows Releases (Recommended)

No Python or PyTorch setup required—includes the 1.26 GB MMS-FA base model, yomikata dBert weights, and ASMR LoRA adapter out of the box:

* **Windows CPU Edition** (all PCs): Download `FA-ASMR-v2.4.0-Windows-CPU.zip`, extract, and double-click `运行FA-ASMR.bat`.
* **Windows CUDA GPU Edition** (NVIDIA GPUs, 4 GB+ VRAM): Download all split volumes (`FA-ASMR-v2.4.0-Windows-GPU.zip.001`, etc.) and `merge_split_gpu_zip.bat` into the same folder, run the merge script, extract, and launch `运行FA-ASMR.bat`.

> Download: [GitHub Releases (v2.4.0)](https://github.com/zxz1994/FA-ASMR/releases/latest)

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

## 3. AI Agent & CLI Automation

**FA-ASMR intentionally embeds zero LLM API keys or network calls**, keeping the acoustic core 100% offline and deterministic. For full end-to-end automation (*raw PDF/DOCX script extraction → forced alignment → DLsite background research & localized subtitle translation*), pair this repo with your preferred **AI Agent** (Antigravity, Claude Code, Codex, Cursor, Gemini CLI, etc.).

The repository ships with a dedicated [`AGENTS.md`](AGENTS.md) operating guide and a headless CLI ([`fa_asmr_cli.py`](fa_asmr_cli.py)).

### 3.1 Handing the Workflow to an AI Agent

Point any terminal-capable AI Agent to this repository and your target work folder with a prompt like:

> "Read `E:\FA-ASMR\AGENTS.md` and process `./RJ01234567` end-to-end: extract and clean the script, align with `fa_asmr_cli.py`, research the work's background/characters, and translate the aligned subtitles."

Following [`AGENTS.md`](AGENTS.md), the Agent will:
1. Extract and clean raw vertical/color-coded PDFs or text scripts and standardize `〖Speaker〗` markers;
2. Semantically select single / multi-speaker / true multi-perspective mode, run `--dry-run` / `--verify-script` pre-checks, and execute `fa_asmr_cli.py --json`;
3. Look up the work's background and character personas (e.g., via DLsite RJ code) and translate line-by-line directly on top of the engine's accurate timelines.

### 3.2 `fa_asmr_cli.py` Usage

Default `--mode auto` automatically selects `multi_persp` when `--audio-b` or dual-POV pairs are detected, `multi_speaker` when ≥2 speakers are present in the script, and `single` otherwise:

```bash
# 1. Fast folder pairing & encoding pre-flight check (no model loaded)
python fa_asmr_cli.py --batch ./work_dir --dry-run --json

# 2. Deep single-script verification (runs AutoRuby, speaker normalization & clause stats)
python fa_asmr_cli.py --verify-script ./01.txt --json

# 3. Single-track alignment (JSON summary on stdout; progress logs on stderr)
python fa_asmr_cli.py --audio ./01.flac --script ./01.txt --extend 1.0 --json

# 4. Dual-perspective joint alignment
python fa_asmr_cli.py --audio ./01_A.wav --audio-b ./01_B.wav --script ./01.txt --json

# 5. Batch directory alignment (reuses GPU memory across tracks, releases on exit)
python fa_asmr_cli.py --batch ./work_dir --json
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
├── fa_asmr_cli.py              # Headless CLI (--mode auto / --dry-run / --verify-script / --json)
├── fa_asmr_pipeline.py         # Non-GUI core pipeline, encoding detection, LRC/SRT & auto-pairing
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
