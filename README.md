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
* **Obsidian Acoustic Console 暗色工作台（无需 Agent，开箱即用）**：全新暗色专业音频控制台桌面端（支持 Windows 高分屏 DPI 感知、拖拽配对与互换、单轨/多轨/双视角切换，以及简体中文 / 日本語 / English 即时热切换），普通用户双击启动即可纯离线完成全部打轴，完全不依赖任何 AI Agent。
* **纯离线高精度闭环（`v2.5.0` 新增）**：
  * **整目录分级注音词表与精准覆写（`--export-ruby` / `--ruby-map`）**：一次加载模型合并导出去重词表，按局部搭配去重采样上下文，自动将多音字、单字接辞切碎（如 `お母様`）、角色名提权置顶；支持 `global` 与 `by_file -> by_line` 覆写且单字绝不劈裂复合词；
  * **声学体检雷达（`fa_asmr_audit.py` / `--audit`）**：零额外推理开销，自动标出未覆盖活跃语音空档（台本缺句/即兴加词）、删戏未录句与多人轨短词跨区前吸（`isolated_speaker_jump`），顶层聚合 `priority_review` 并默认仅切高危审听 WAV；
  * **零依赖双耳立体声空间分离（`fa_asmr_spatial_sep.py` / `--audit-separate`）**：纯 `numpy/scipy` 基于 ILD+IPD 软掩码将多人真叠声段无损拆出左/中/右干声 WAV（单人轨自动跳过拆轨，亦支持无台本独立扫描拆轨）。
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

## 三、AI Agent 与命令行自动化（可选端到端工作流）

**FA-ASMR 本身完全独立运行，不依赖任何 AI Agent，也不内嵌任何大模型 API Key 或在线接口**——普通用户直接打开桌面端或命令行即可 100% 纯离线打轴。

如果你希望把「原始 PDF/DOCX 台本清洗 → 注音审校 → 强制打轴与声学体检 → 多模态听审纠错 → 结合作品背景精翻中文/双语字幕」串成全自动闭环，可以直接用你手头的 **AI Agent**（Antigravity、Claude Code、Codex、Cursor、Gemini CLI 等）驱动本仓库。仓库已内置完整的 [AGENTS.md](AGENTS.md) 操作手册、无界面命令行 [fa_asmr_cli.py](fa_asmr_cli.py) 与紧凑翻译辅助脚本 [tools/compact_subs.py](tools/compact_subs.py)。

### 3.1 直接让 Agent 接管端到端闭环

把本仓库路径和你的作品目录交给任意支持终端（及多模态音频读取）的 AI Agent，只需发一句：

> “请阅读 `AGENTS.md`，按里面的工作流处理：提取清洗台本、先审校注音词表再调用 `fa_asmr_cli.py` 完成对齐与声学体检听审定稿日轴，最后调研作品背景并用紧凑格式翻译输出中文字幕。”

Agent 会自动按 [AGENTS.md](AGENTS.md) 执行精简闭环（严格遵循**先定稿日轴、再开翻中文**，杜绝返工）：
1. **前置台本提取与清洗**：解析竖排 / 分栏 / 颜色分角色 PDF，剥除动作与 SE 提示，统一 `〖角色名〗` 标注；
2. **批量分级注音审校**：执行 `--batch --export-ruby` 一次加载模型导出整部作品去重词表（按局部搭配去重、高危多音字/单字接辞切碎/角色名置顶），仅对确有错读的词写入 `ruby_fixes.json`（无需改动原台本）；
3. **对齐 + 分级声学体检 + 多模态定向听审**：运行 `--batch --audit-separate` 生成日轴与 `priority_review` 高危审听切片（单人轨自动跳过空间拆轨，多人轨真叠声自动拆出左/中/右干声）；支持原生音频的 Agent（如 Antigravity / Gemini 系）直接听审高危 WAV 纠错并用 `--only` 局部重跑定稿日轴；**若使用不支持原生读音频的 Agent（如 Claude Code / Codex / Cursor）**，则由 Agent 列出这几段短 WAV 交由用户快速听审确认或辅助局部 ASR 后再继续，其余环节均不受影响；
4. **背景调研与紧凑格式精翻**：根据 RJ 编号查清作品设定与角色口吻，通过 `tools/compact_subs.py` 剥离时间戳导出 `行号|角色|原文`（省约 35%–40% 翻译输入 Token），译完自动回填生成 `.srt` + `.lrc` 并校验 1:1 时间轴一致性与 0 假名残留。

### 3.2 `fa_asmr_cli.py` 与配套工具常用命令

默认 `--mode auto`（有 `--audio-b` 或双视角文件对自动走 `multi_persp`，台本 ≥2 角色自动走 `multi_speaker`，否则走 `single`）：

```bash
# 1. 整目录配对预检（不加载模型，秒级检查配对、编码、角色与自动判定的模式）
python fa_asmr_cli.py --batch ./作品目录 --dry-run --json

# 2. 整目录一次加载模型，合并导出分级去重注音词表（高危多音/单字切碎/角色名排最前）
python fa_asmr_cli.py --batch ./作品目录 --export-ruby --json-out ./ruby_batch.json

# 3. 整目录批量对齐 + 声学体检 + 高危切片（复用显存；多人轨自动空间分离，单人轨自动跳过拆轨）
python fa_asmr_cli.py --batch ./作品目录 --ruby-map ./ruby_fixes.json --audit-separate --audit-out ./audit_batch.json --audit-clips-dir ./audit_clips --json

# 4. 听审修正少数几轨台本后，局部重跑（仍复用单次显存加载）
python fa_asmr_cli.py --batch ./作品目录 --only 05,07,13 --ruby-map ./ruby_fixes.json --audit-separate --audit-out ./audit_realign.json --audit-clips-dir ./audit_clips_realign --json

# 5. 单条对齐 / 双视角联合对齐
python fa_asmr_cli.py --audio ./01.flac --script ./01.txt --extend 1.0 --json
python fa_asmr_cli.py --audio ./01_A.wav --audio-b ./01_B.wav --script ./01.txt --audit-separate --json

# 6. 独立零依赖立体声空间分离（事后补拆或无台本多人人头麦自动扫描叠声）
python fa_asmr_spatial_sep.py --audio ./02.flac --auto-overlaps --sources 3 --out-dir ./sep_out --json

# 7. 紧凑格式导出 / 回填 / 终检（配合翻译阶段省 Token）
python tools/compact_subs.py export --jp-dir ./字幕（日文） --out ./compact_jp.json
python tools/compact_subs.py build  --jp-dir ./字幕（日文） --cn-dir ./字幕（中文） --in-json ./zh_batch.json
python tools/compact_subs.py verify --jp-dir ./字幕（日文） --cn-dir ./字幕（中文）
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
├── fa_asmr_cli.py              # 无界面 CLI（--mode auto / --export-ruby / --ruby-map / --audit-separate / --only / --json）
├── fa_asmr_pipeline.py         # 无界面核心对齐管线、编码嗅探、分级注音词表与覆写、LRC/SRT 组装与配对
├── fa_asmr_audit.py            # 对齐后声学体检雷达、双视角联合体检、短词跳变检测与高危切片导出
├── fa_asmr_spatial_sep.py      # 零依赖双耳立体声 ILD+IPD 空间分离工具（支持独立拆轨与体检联动）
├── tools/compact_subs.py       # 紧凑翻译导出 / 回填 .srt+.lrc / 1:1 一致性终检脚本
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
