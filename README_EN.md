# 🌸 FA-ASMR V2 (Forced Alignment for Japanese ASMR & Voice Dramas)

[ English ](README_EN.md) | [ 简体中文 ](README.md)

> **High-precision acoustic forced alignment engine designed specifically for Japanese ASMR, drama CDs, and anime voice acting.**
> 
> Input: Japanese script (`.txt`) + Vocal audio (`.wav`, `.mp3`, `.flac`, etc.)  
> Output: Millisecond-accurate standard `.lrc` subtitle/lyrics file with precise start/end timestamps.

---

## 1. Core Advantages (vs. Whisper/ASR Approaches)

When subtitle makers and translators already possess official scripts (as is standard for DLsite/DMM voice dramas), using general-purpose ASR models like Whisper introduces critical issues: missed whispers, misheard character names, and bloated 10–20 second monolithic subtitle blocks. 

FA-ASMR V2 utilizes a **100% pure acoustic forced alignment architecture** (Meta MMS-FA + ASMR-specialized LoRA + Constrained CTC Trellis dynamic programming). Its key advantages include:

- **100% Script Fidelity (Zero Hallucination)**: Directly aligns the official script to acoustic features. Eliminates ASR transcription errors (e.g., mishearing names or erotic voice drama terminology) and prevents near-ear whispers from being discarded as silence. Translators can work directly from the original script without audio proofreading.
- **Sub-clause Granularity & Natural Pauses**: Intelligently segments subtitles based on acoustic pauses and Japanese punctuation into natural reading units (typically 2–4 seconds per line). Automatically inserts empty timestamp lines upon vocal cessation, eliminating unnatural text hanging on screen.
- **Binaural Better-Ear Acoustic Fusion**: Specifically optimized for 3Dio and dummy-head stereo recordings. Evaluates log-odds ratios across stereo channels on a frame-by-frame basis to dynamically select the cleanest ear, suppressing near-ear breath pop, mouth friction, and unilateral background noise.
- **Native Speaker Tag Support**: Automatically detects, strips for alignment, and restores `〖Character Name〗` speaker tags (U+3016 / U+3017), facilitating multi-role colored subtitles.
- **High Performance & Low VRAM Usage**: The Viterbi decoding core is accelerated with Numba JIT. VRAM consumption is a fraction of large ASR models, aligning a 1-hour drama CD in just a few minutes.

### Limitations & Acting Onomatopoeia Filter (Ono Filter)

- **Monotonic Single-Chain Constraint**: The alignment engine assumes chronological monotonic progress. If a drama CD features simultaneous overlapping dialogue (two voice actors speaking at the exact same moment) or if the script lines are scrambled out of chronological order, the timeline may drift. Scripts should be ordered chronologically before alignment.
- **Acting Onomatopoeia Filter (Ono Filter)**:
  1. **Physical Action Sounds**: Scripts often write sound effects with kana (e.g., `ちゅ` for kissing, `じゅる` for sucking, `れるぅ` for licking). However, voice actors produce actual physical mouth/body sounds rather than pronouncing kana phonemes. Acoustic models cannot match non-speech audio to phonetic kana, frequently causing severe timeline stretching or tearing.
  2. **Spontaneous Moans & Panting**: Kana sequences like `んっ`, `ふぁ…`, or `あぁ…` represent emotional performance directions. Voice actors deliver these with spontaneous pacing, pitch, and duration that rarely match the literal kana count.  
  Forcing alignment on these tokens will cause distortions. The built-in Ono Filter detects these acting markers and automatically drops corrupted or squeezed onomatopoeia lines while strictly preserving genuine speech dialogue and adjacent lines (can be toggled in GUI settings).

---

## 2. Quick Start

### 2.0 Prebuilt Portable Release (Recommended for End Users)

No Python or PyTorch installation required. Unzip and run — the 1.26 GB acoustic model and LoRA adapter are bundled inside:

- **CPU edition** (works on any PC): download `FA-ASMR-v2.0.0-Windows-CPU.zip`, extract, then double-click `运行FA-ASMR.bat`;
- **CUDA GPU edition** (fast on NVIDIA GPUs): download all split parts plus `merge_split_gpu_zip.bat` into one folder, then double-click the batch file to merge and extract automatically.

Download: [GitHub Releases](https://github.com/zxz1994/FA-ASMR/releases/latest)

> The section below covers running from source, intended for developers or users who need to modify the code.

### 2.1 Environment Setup

- **Python**: Recommended Python 3.10 – 3.12 (64-bit)
- **Hardware**: NVIDIA GPU (CUDA supported) recommended for fast inference; CPU mode (multi-threading + dynamic quantization) is also supported.

```bash
# 1. Clone repository
git clone https://github.com/zxz1994/FA-ASMR.git
cd FA-ASMR

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install the furigana model separately
#    (its setup pulls in many unrelated dev dependencies, hence excluded from requirements)
pip install --no-deps yomikata
```

> **PyTorch Recommendation**:
> PyTorch 2.6.0 is recommended:
> - **CUDA**: `pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124`
> - **CPU**: `pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu`

### 2.2 Model Weights

| Model | Target Path | Description |
|---|---|---|
| **MMS-FA Base Model** (`model.pt`) | `models/hub/checkpoints/model.pt` | ~1.26 GB. Automatically downloaded on first run via `torchaudio.pipelines.MMS_FA`, or can be manually placed for complete offline usage. |
| **ASMR LoRA Adapter** (`fa_asmr_v9e10.pt`) | `lora/fa_asmr_v9e10.pt` | **Included in repository** (~12.2 MB). Fine-tuned specifically on Japanese ASMR acoustics. |

### 2.3 Launching the Application

- **Windows**: Double-click `run.bat`
- **Command Line**:
  ```bash
  python FA-ASMR_GUI.py
  ```

---

## 3. Usage & Specifications

### 3.1 GUI Workflow

1. **Drag-and-Drop**: Drag multiple scripts (`.txt`) and audio files directly into the window. Pairs are matched automatically by filename.
2. **Audio Formats Supported**: `.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, `.aac`, `.opus`, `.wma`.
3. **Key Options**:
   - **Device**: Select `cuda` (GPU) or `cpu`.
   - **VRAM Chunk**: Lower to 3000–6000 frames if running on lower VRAM GPUs to prevent OOM.
   - **Line End Extension**: Adds a small natural delay to sentence end timestamps for more comfortable reading.
   - **Ono Gate**: Enable to prune stretched/distorted acting onomatopoeia.

### 3.2 Script Format & Pairing Conventions

- **Pairing Rules**:
  - **Identical Stems**: `01_track.wav` pairs with `01_track.txt`.
  - **Speaker Suffix**: Filenames ending with speaker tags (e.g., `01_track_〖Alice〗〖Bob〗.txt`) automatically strip the tag to match `01_track.wav` and register the speaker metadata.
- **Content Guidelines**:
  - Must be UTF-8 encoded text matching the recording's chronological vocal order.
  - Remove non-vocal stage directions, track headers, and scene descriptions inside brackets.
  - In-line speaker tags like `〖Name〗` anywhere in the line will be preserved in the final LRC output.

### 3.3 Output Format

The output `.lrc` file is written to the same directory as the audio file. Timestamps follow standard centisecond precision `[MM:SS.CC]`. Each dialogue line is followed by an empty timestamp indicating the exact vocal ending time (despawn point):

```text
[00:05.58]わぁ...! ねぇ、見てみてっ!
[00:08.62]
[00:12.30]すっごくいい景色!
[00:15.00]
```

### 3.4 Advanced Configuration (`advanced_config.json`)

Optional low-level system settings (read-only by GUI; edit manually if needed):
- `cpu_threads`: CPU inference forward threads (`null` = auto-detect physical cores).
- `quantize_int8`: Dynamic INT8 quantization for CPU mode.
- `cpu_chunk_cap`: Maximum frame chunk cap in CPU mode (default: `2000`).

---

## 4. Architecture & Pipeline

Repository module topology:

```text
FA-ASMR/
├── assets/                     # GUI banners and mascot icons
├── furigana/                   # Japanese furigana / phoneme conversion
│   ├── fa_asmr_converter.py    # 3-tier furigana generator & Unicode radical fixer
│   └── overrides.json          # Heteronym & reading override dictionary
├── lora/                       # Specialized model adapters
│   └── fa_asmr_v9e10.pt        # Production-grade ASMR LoRA weights
├── advanced_config.json        # Advanced system settings
├── advanced_config.py          # Configuration loader
├── align.py                    # Alignment pipeline orchestrator
├── align_acoustic.py           # RMS, noise floor, and binaural Better-Ear fusion
├── align_model.py              # MMS-FA loader, LoRA injection & dynamic quantization
├── align_post.py               # Post-processing, silence snap & timestamp monotonicity
├── align_trellis.py            # Constrained CTC Trellis Viterbi aligner (Numba JIT)
├── align_utils.py              # Normalization, prolonged sound folding, time formatting
├── clause.py                   # Data contract constants (F) & clause boundaries
├── FA-ASMR_GUI.py              # Tkinter GUI interface & multi-threaded batch runner
├── fa_asmr_settings.json       # GUI settings persistence
├── haruraw2norm.py             # Kanji/Kana/Romaji to MMS-FA phoneme mapping
├── inline_leak_clean.py        # In-line furigana leak cleaner
├── lrc_end_extend.py           # Line-end timestamp extender
├── merge_short.py              # Clause-based short line merger
├── np_compat.py                # NumPy 2.x compatibility shim
├── ono_base.py                 # Core onomatopoeia pattern classifier
├── ono_block_corpus.py         # Substantive word blocklist
├── ono_filter.py               # Acting onomatopoeia gatekeeper
├── resegment.py                # Acoustic gap punctuation re-segmenter
├── torch_bootstrap.py          # PyTorch environment bootstrap & CUDA fallback
├── requirements.txt            # Python dependencies
└── run.bat                     # Windows quick launcher
```

### Data Pipeline Overview

```text
[1/4] Script Ingestion -> Speaker Extraction -> Furigana Leak Cleaning -> 3-Tier Furigana Annotation
      ↓
[2/4] Phoneme Normalization & MMS-FA Vocabulary Mapping -> Blank Masks & Grouping
      ↓
[3/4] Stereo Acoustic Analysis -> Better-Ear Fusion -> Energy Decoupling -> Trellis CTC Decoding
      ↓
[4/4] Ono Marking -> Punctuation Resegment -> Short Line Merge -> Ono Pruning -> Standard LRC Export
```

---

## 5. License & Acknowledgements

This project is licensed under the MIT License.
- Forced alignment principles inspired by Meta MMS-FA (Massively Multilingual Speech).
- Japanese morphological analysis & furigana integrate Fugashi (UniDic), yomikata, and PyKakasi.
