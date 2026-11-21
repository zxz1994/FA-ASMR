# FA-ASMR (`v2.5.0`)

[English](README_EN.md) | **简体中文** | [Agent 操作指南 (AGENTS.md)](AGENTS.md)

> **专为日语 ASMR / 二次元音声打造的高精度纯声学强制对齐（打轴）引擎**
>
> 输入「官方日文台本 + 人声音频」，离线输出毫秒级精确对齐的 `.lrc` 与 `.srt` 双格式字幕；同时提供无界面 CLI 与标准化 [AGENTS.md](AGENTS.md)，可直接接入外部 AI Agent 实现从原始 PDF 台本到成品双语字幕的全流程自动化。

---

## 一、核心特性

在拥有官方台本的日语 ASMR / 音声字幕制作中，传统 Whisper 等通用 ASR 盲听转录常出现弱音吞词、专有名词错写以及长句粘连。**FA-ASMR** 采用纯声学物理强制对齐路线（Meta MMS-FA + ASMR 专用 LoRA `fa_asmr_v9e10.pt`），面向真实音声制作痛点做了全链路特化：

* **100% 原文保真与双序列切片还原**：直接依据官方台本定位声学时间轴，基于 `SequenceMatcher` 锚点切片精准还原原文字符（兼容独立浊音符、NFKC 特殊符号、已有 `{漢字|かな}` 标注、前导开引号与无读音促音），告别盲听错别字。
* **双耳人头麦抗噪 (Better-Ear)**：针对双声道 ASMR 录音，动态评估左右声道声学质量与底噪，抑制近耳风噪、摩擦声与单侧干扰。
* **三种专业对齐模式**：
  * **01 常规单向对齐 (`single`)**：标准单声优音声，全局 CTC Viterbi 解码（Numba JIT 多线程加速）+ 静音吸附。
  * **02 多人分轨 / 乱序重排 (`multi_speaker`)**：面向多声优分轨混音、角色插话交错、台词与喘息重叠场景；按角色拆流独立定位并结合双声道 STFT 相干性独白遮罩修复抢词冲突。
  * **03 多视角联合对齐 (`multi_persp`)**：面向**两轨视角各自都有角色说话且清晰度互补**的双机位作品；自动校准双轨物理时间差 $\Delta t$，按清晰侧择优打轴并分别投影输出视角 A / B 各自独立坐标系的 `.lrc` 与 `.srt`。（注：若另一轨仅为隔墙闷音处理而无清晰度互补，直接使用清晰主轨跑模式 01/02 即可。）
* **内置拟声词守门与智能断句 (`ono_filter` / `resegment` / `merge_short`)**：
  * 自动识别亲吻、舔舐、吞咽及自由发挥喘息等易致时间轴撕裂的纯拟声行并安全过滤，同时保护句中夹心拟声与延音实词（如 `ろぉ～く`、`い～～ち`）；
  * 结合声学停顿与标点自动切分短句、桥接短实词顿号并合并过碎片段，默认输出即可直接观看或翻译。
* **Obsidian Acoustic Console 暗色工作台**：全新暗色专业音频控制台桌面端（支持 Windows 高分屏 DPI 感知、拖拽配对与互换、单轨/多轨/双视角切换，以及简体中文 / 日本語 / English 即时热切换）。
* **多编码自动嗅探与低硬件门槛**：原生自动识别 UTF-8 / UTF-8-BOM / Shift-JIS (CP932) / GB18030 / UTF-16，无需手动转码；CUDA 默认 FP16 混合精度 + 固定最优 `3000` 帧分块（峰值显存约 `3.74 GB`，4 GB+ 显存即可运行），无显卡设备支持 CPU INT8 动态量化。

---

## 二、快速开始

### 2.1 免安装预编译版（推荐普通用户）

无需安装 Python 与 PyTorch，解压即用，已完整内置 1.26 GB MMS-FA 基础模型、yomikata dBert 权重与 ASMR LoRA 适配器：

* **Windows CPU 版**（兼容所有电脑）：下载 `FA-ASMR-v2.5.0-Windows-CPU.zip`，解压后双击 `运行FA-ASMR.bat`；
* **Windows CUDA GPU 版**（推荐 N 卡用户，支持 4 GB+ 显存）：下载全部分卷（`FA-ASMR-v2.5.0-Windows-GPU.zip.001` 等）与 `merge_split_gpu_zip.bat` 至同一目录，双击合并脚本后解压运行 `运行FA-ASMR.bat`。

> 前往下载：[GitHub Releases (v2.5.0)](https://github.com/zxz1994/FA-ASMR/releases/latest)

### 2.2 源码运行（开发者 / CLI / Agent 环境）

环境要求：Python 3.10 ~ 3.12 (64-bit)，推荐 NVIDIA GPU (CUDA) 或多核 CPU。

```bash
# 1. 克隆仓库
git clone https://github.com/zxz1994/FA-ASMR.git
cd FA-ASMR

# 2. 安装 PyTorch 2.6.0（按硬件二选一）
# CUDA 12.4：
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# 或 CPU 版：
# pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu

# 3. 安装核心依赖
pip install -r requirements.txt

# 4. 单独安装注音模型（避免拉入冗余开发依赖）
pip install --no-deps yomikata
```

**模型权重说明**：

| 模型文件 | 路径 | 说明 |
| :--- | :--- | :--- |
| **MMS-FA 基础模型** | `models/hub/checkpoints/model.pt` | 约 1.2 GB，首次运行自动下载；也可预先放置实现纯离线运行 |
| **ASMR 专用 LoRA** | `lora/fa_asmr_v9e10.pt` | **仓库已内置**（约 12.2 MB），针对日语 ASMR 耳语/弱音调优 |

**启动桌面端**：
* Windows 直接双击 `run.bat`，或命令行执行 `python FA-ASMR_GUI.py`。

---

## 三、AI Agent 与命令行自动化（端到端工作流）

**FA-ASMR 不内嵌任何大模型 API Key 或在线接口**，保持 100% 纯离线物理对齐；需要“原始 PDF/DOCX 台本清洗 → 强制打轴 → 结合 DLsite 作品背景精翻中文/双语字幕”全自动闭环时，推荐直接配合你手头的 **AI Agent**（Antigravity、Claude Code、Codex、Cursor、Gemini CLI 等）驱动本仓库。

仓库根目录已内置完整的 [AGENTS.md](AGENTS.md) 操作手册与无界面命令行工具 [fa_asmr_cli.py](fa_asmr_cli.py)。

### 3.1 直接让 Agent 接管全流程

把本仓库路径和你的作品目录交给任意支持终端的 AI Agent，只需发一句：

> “请阅读 `E:\FA-ASMR\AGENTS.md`，按里面的工作流处理 `./RJ01234567`：提取清洗台本、调用 `fa_asmr_cli.py` 完成对齐，并调研作品背景后翻译输出中文字幕。”

Agent 会自动按 [AGENTS.md](AGENTS.md) 执行：
1. **前置台本提取与清洗**：解析竖排 / 分栏 / 颜色分角色 PDF，剥除动作与 SE 提示，统一 `〖角色名〗` 标注；
2. **模式判定与 CLI 对齐**：根据剧情语义判定单人 / 多人 / 真双视角，先跑 `--dry-run` / `--verify-script` 校验，再调用 `fa_asmr_cli.py --json` 批量生成精确日轴；
3. **背景调研与逐行翻译**：根据 RJ 编号查清作品设定与角色口吻，直接沿用引擎产出的精准时间轴逐行翻译为自然流畅的中文/双语字幕。

### 3.2 `fa_asmr_cli.py` 常用命令

默认 `--mode auto`（有 `--audio-b` 或双视角文件对自动走 `multi_persp`，台本 ≥2 角色自动走 `multi_speaker`，否则走 `single`）：

```bash
# 1. 整目录配对预检（不加载模型，秒级检查配对、编码、角色与自动判定的模式）
python fa_asmr_cli.py --batch ./作品目录 --dry-run --json

# 2. 单台本深度校验（真实运行 AutoRuby 注音、角色标题归一化、顿号桥接与子句统计）
python fa_asmr_cli.py --verify-script ./01.txt --json

# 3. 单条音频对齐（自动判模式；JSON 摘要走 stdout，过程日志自动走 stderr）
python fa_asmr_cli.py --audio ./01.flac --script ./01.txt --extend 1.0 --json

# 4. 双视角联合对齐
python fa_asmr_cli.py --audio ./01_A.wav --audio-b ./01_B.wav --script ./01.txt --json

# 5. 整目录批量对齐（自动复用显存，结束后释放）
python fa_asmr_cli.py --batch ./作品目录 --json
```

> **免安装冻结版用户**：同样支持命令行调用 `FA-ASMR.exe --cli ... --json-out summary.json`（因冻结版为无控制台窗口模式，建议配合 `--json-out` 读取结构化结果）。

---

## 四、桌面端使用与台本规范

### 4.1 音频支持与配对规则

* **支持音频格式**：`.wav`、`.mp3`、`.flac`、`.m4a`、`.ogg`、`.aac`、`.opus`、`.wma`。
* **智能文件配对**：
  * **基础同名**：`01_话名.wav` 匹配 `01_话名.txt`（支持在台本文件名尾部附加角色后缀 `01_话名〖澄華〗〖天音〗.txt`）；
  * **双视角配对**：`01_A.wav` / `01_B.wav` 或 `トラック7‗主人公視点.wav` / `トラック7‗夫視点.wav` 自动归并匹配同基名台本；
  * 拖拽导入后，可在中央轨道矩阵内直接拖放互换或调整配对。

### 4.2 台本格式建议

* **编码**：UTF-8 / UTF-8-BOM / Shift-JIS (CP932) / GB18030 / UTF-16 全自动识别，无需手动转码。
* **非发音内容**：建议提前移除动作/演技指导、SE/BGM 提示与场次标题；内联注音 `漢字（かんじ）` 建议剥除或写为 `{漢字|かんじ}`。
* **说话人标记（多人 / 双视角模式）**：
  * 推荐统一使用 `〖角色名〗`（独立行或行首/行内均可），多人同时说话写为 `〖澄華〗〖天音〗せーの！`；
  * 原生兼容高频复现的独立 `【角色名】` 标题行。

### 4.3 字幕输出与进阶配置

* 对齐完成后，在音频同级目录生成同名标准 **`.lrc`**（带精确下轴空白行与 `角色名：` 前缀）与 **`.srt`**（支持多角色独立时间流与 `【角色名】` 前缀）。
* 进阶硬件参数可在只读文件 `advanced_config.json` 中调整（保存后重启生效）：
  * `cpu_threads`：CPU 推理线程数（默认 `null` 自动取物理核心数）；
  * `quantize_int8`：CPU INT8 动态量化（默认 `null`：CPU 开启 / GPU 关闭）；
  * `cpu_chunk_cap`：CPU 单块最大帧数上限（默认 `2000`）。

---

## 五、项目结构与回归测试

```text
FA-ASMR/
├── AGENTS.md                   # 外部 AI Agent 端到端操作与维护指南
├── fa_asmr_cli.py              # 无界面 CLI（--mode auto / --dry-run / --verify-script / --json）
├── fa_asmr_pipeline.py         # 无界面核心对齐管线、编码嗅探、LRC/SRT 组装与文件配对
├── FA-ASMR_GUI.py              # 桌面端启动器、--cli 路由与 --self-test 打包自检
├── fa_asmr_ui.py               # Obsidian Acoustic Console 暗色音频控制台前端
├── fa_asmr_i18n.py             # 中 / 日 / 英三语热切换词典
├── align.py                    # 单向对齐主引擎入口
├── align_multispeaker.py       # 多人分轨 / 乱序重排 / STFT 独白遮罩引擎
├── align_multiperspective.py   # 多视角联合校准与双坐标投影引擎
├── align_trellis.py            # 全局约束 CTC Viterbi 解码器（Numba JIT）
├── align_acoustic.py           # 逐帧 RMS、底噪估计与双声道 Better-Ear 融合
├── align_model.py              # MMS-FA 加载、LoRA 注入、FP16 / INT8 推理
├── align_post.py               # 静音吸附、Token 时间戳回映射与单调化
├── align_utils.py              # 基础文本规范化与时间戳正则
├── clause.py                   # 跨模块数据契约常量 (F)、子句边界与短句顿号桥接
├── haruraw2norm.py             # 汉字 / 假名 / 罗马音到 MMS-FA 音素转换
├── inline_leak_clean.py        # 内联振假名泄露自动清洗
├── ono_base.py / ono_filter.py / ono_block_corpus.py  # 拟声词多阶段守门与实词保护
├── resegment.py / merge_short.py / lrc_end_extend.py  # 标点重分段、短句合并与尾端延展
├── furigana/                   # 三层日语振假名注音引擎与异读覆盖表
├── lora/fa_asmr_v9e10.pt       # 内置 ASMR 专用 LoRA 适配器权重
├── tests/                      # 零依赖回归测试套件（标准库 unittest）
└── build_in_ci.py / FA-ASMR.spec  # GitHub Actions CPU & CUDA 双变体打包脚本
```

运行零依赖回归测试（标准库 `unittest`，无需安装额外测试框架）：

```bash
python -m unittest discover -s tests -v
```

---

## 六、开源协议与鸣谢

本项目基于 **MIT License** 开源。
* 声学强制对齐基础模型基于 [Meta MMS-FA](https://pytorch.org/audio/stable/tutorials/forced_alignment_for_multilingual_data_tutorial.html)
* 日语分词与振假名注音集成了 [Fugashi (UniDic)](https://github.com/polm/fugashi)、[yomikata](https://github.com/passaglia/yomikata) 与 [PyKakasi](https://github.com/miurahr/pykakasi)
