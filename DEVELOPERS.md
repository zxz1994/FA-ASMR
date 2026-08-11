# FA-ASMR 开发者源码包（FA-ASMR-Source）

本目录是 **FA-ASMR 强制对齐工具**的干净源码，面向想要继续开发的开发者。
从 `FA-ASMR-Release/` 整理而来：**去掉了构建产物、内置 torch 依赖包、日志与临时脚本**，
只保留可运行 / 可维护的源码与小型资产。

---

## 1. 目录结构

```text
FA-ASMR-Source/
├── README.md                 # 项目总览（功能、流程、实测数据、许可）
├── DEVELOPERS.md             # 本文件：开发环境搭建与运行
├── 使用说明.txt              # 面向最终用户的使用说明（含 GUI 高级设置解释）
├── requirements.txt          # Python 依赖
├── run.bat                   # 源码版启动器（python FA-ASMR_GUI.py）
│
├── FA-ASMR_GUI.py            # GUI 主程序（拖放批量对齐）+ CLI 模式
├── align.py                  # 强制对齐主入口（align_audio_with_text）
├── align_model.py            # MMS_FA 模型缓存 / 自研最小 LoRA 适配器 / emission 计算
├── align_utils.py            # token 工具 / span 打分 / 重复行检测
├── align_post.py             # 后处理（auto_repeat 拟声循环行修复等）
├── haruraw2norm.py           # 注音文本 → 罗马音/假名 → MMS_FA token
├── norm2lrc.py               # LRC / 时间轴输出
├── torch_bootstrap.py        # 内置 torch 加载引导（CPU/GPU 自包含，无需另行安装）
├── advanced_config.py        # 底层系统配置加载器（cpu_threads / INT8 / 分块）
├── advanced_config.json      # 底层系统配置（手编，不进 GUI）
├── fa_asmr_settings.json     # GUI 用户设置（默认模板）
├── ono_table.json            # 小书档案（OnoWave 波形计数等）
│
├── furigana/
│   └── fa_asmr_converter.py  # 自动注音引擎（UniDic + yomikata 异读消歧 + 部首修复）
├── lora/
│   └── fa_asmr_e30.pt        # 最优 LoRA 权重（E30，eval 最优，已附带）
├── tasks/
│   └── fa_lora_checkpoints/
│       └── best_model.pt     # 早期检查点（备用）
├── models/
│   └── hub/checkpoints/      # ★ 空占位，需手动放入 model.pt（见第 3 节）
├── custom_hooks/
│   └── hook-nltk.py          # PyInstaller 打包 hook
└── build/                    # ★ 打包/安装工具（可选，普通开发用不到）
    ├── _build_exe.py         # PyInstaller 打包脚本（onedir + 内置 embedded_torch）
    ├── _build_exe.bat
    ├── FA-ASMR.spec
    └── pyi_rth_fix_scipy.py  # 运行时 hook
```

---

## 2. 环境搭建

### 2.1 Python
需要 **Python 3.10+**（推荐 3.10–3.12）。

### 2.2 安装依赖
```shell
pip install -r requirements.txt
```
> 说明：`peft` / `transformers` 在 `requirements.txt` 中列出，但 `align_model.py`
> 的 LoRA 是**自研最小实现、不依赖 peft 库**（打包时也会排除这二者以控制体积）。
> `transformers` 仅在你启用 yomikata 的 BERT 异读后端时才可能被间接用到。
> 如果只想跑基础对齐，可只装核心：
> `torch torchaudio numpy soundfile scipy fugashi sudachipy sudachidict_core pykakasi janome jaconv yomikata tkinterdnd2`

### 2.3 准备 MMS_FA 权重（必须）
对齐核心依赖 **MMS_FA** 的 `model.pt`（约 1.26GB）。该文件**不随源码分发**，
需自行准备并放到：
```text
FA-ASMR-Source/models/hub/checkpoints/model.pt
```
（来源：FA-Kara / MMS_FA 官方权重。）

### 2.4 LoRA 权重（已附带）
最优权重 `lora/fa_asmr_e30.pt` 已在本包内，GUI 默认加载，无需额外下载。

---

## 3. 运行

### GUI（推荐）
```shell
cd FA-ASMR-Source
python FA-ASMR_GUI.py
# 或直接双击 run.bat
```
- 首次缺失 PyTorch 时，GUI 会引导联网安装（CPU 版），否则回退 CPU 路径。
- 拖入「台本 `.txt`（可含 `{漢字|かな}` 注音）+ 音频 `.wav/.mp3`」，按文件名自动配对。
- 详见 `使用说明.txt` 与 `README.md` 的「使用前提」（台本需提前清理非发音内容、不支持乱序）。

### CLI
```shell
python FA-ASMR_GUI.py --cli --script 台本.txt --audio 音频.wav [--lora lora/fa_asmr_e30.pt]
```

---

## 4. 本包【已排除】的内容（重要）

| 排除项 | 原因 |
|--------|------|
| `dist/` | PyInstaller 构建产物（exe + _internal，数千文件） |
| `embedded_torch/` | 内置 torch 依赖包（CPU 版 ~0.36GB / GPU 版内含 CUDA torch）。开发者用 `pip install torch` 即可，**无需此目录**。 |
| `models/hub/checkpoints/model.pt` | 1.26GB 权重，不随源码分发，见第 2.3 节手动放置。 |
| `*_log.txt` / `*_err.txt` / `build_pid.txt` / `cli_test_*.txt` | 构建/测试日志 |
| `_chk_emb.py` / `_collect_stdlib.py` / `_copy_doc.py` / `_finalize_dist.py` / `_install_emb.py` / `_stdlib_imports.txt` | 构建 `embedded_torch` 用的临时辅助脚本，随 `embedded_torch` 一并排除 |

> `build/` 子目录保留是为了让想**重新打包 exe** 的开发者有入口；普通开发改代码、跑 GUI 不需要它。

---

## 5. 关键模块职责（改哪看哪）

| 文件 | 改这里为了… |
|------|------------|
| `FA-ASMR_GUI.py` | 界面、拖放、批量流程、高级设置面板、CLI 入口 |
| `align.py` | 对齐主算法（CTC 强制单调覆盖、auto_repeat、信道择优、Refine） |
| `align_model.py` | 模型加载 / 自研 LoRA 注入 / emission 计算（含 INT8、分块、CPU 线程旋钮） |
| `align_utils.py` | token 解析、span 打分、重复行检测、波形计数 |
| `align_post.py` | 后处理（拟声循环行展开修复等） |
| `haruraw2norm.py` | 注音文本 → 罗马音/假名 → MMS_FA token 的归一化 |
| `furigana/fa_asmr_converter.py` | 自动注音（UniDic + yomikata 异读 + 部首字符修复） |
| `torch_bootstrap.py` | 内置 torch 加载引导（CPU/GPU 自包含） |
| `advanced_config.py` + `.json` | CPU 性能旋钮（线程 / INT8 / 分块），手编不进 GUI |

---

## 6. 已知限制（接手前必读）

- 对齐**起点可信、终点（时长）偏紧**：CTC 会把孤立/弱发射 token 压短，长喘息/拟声行尤其明显。
- **不支持乱序台本**（台本顺序必须 = 音频时间顺序）。
- 约一成行误差 > 1s，集中在台本拟声/喘息行（token 密度与真实声学不匹配），属 CTC 帧分配的
  结构性局限，目前无后处理根治方案；曾验证无效的实验路径（`two_pass` / `region_align` /
  `anchor_realign`）已被移除或默认关闭。
- 精度已触底（中位 ≈ 110ms），LoRA 的作用是抑制尾部大幅错误而非提升常规精度，
  E30 为过拟合拐点。

---

## 7. 二次打包（可选）

若要从本源码重新构建 exe：
```shell
cd FA-ASMR-Source/build
# 需先在 FA-ASMR-Source/ 准备好 embedded_torch/（从原 Release 复制）或依赖系统 torch
python _build_exe.py
```
打包脚本默认输出到临时目录再合并覆盖，避免批量删除触发安全拦截。
更完整的打包说明见原 `FA-ASMR-Release/_build_exe.py` 顶部注释。

---

## 8. LRC 行尾延长（后处理工具）

GUI 的「行结束延长」开关有时在最终 `.lrc` 上不生效。更可靠的做法是**直接对生成的
`.lrc` 做后处理**：`lrc_end_extend.py` 把每个句末空 `[mm:ss.xx]` 结束标记往后推 N 秒，
封顶到下一句起点（不越界），与 GUI 语义一致、但独立于对齐管线、必然生效。

```shell
cd FA-ASMR-Source
python lrc_end_extend.py 你的文件.lrc -e 1.0
#   -e 延长秒数(默认 1.0, 推荐 0.5~1.5)
#   省略 -o 时原地修改, 自动备份 你的文件.lrc.bak
python lrc_end_extend.py 输入.lrc -o 输出.lrc -e 0.8
```

LRC 格式约定（本工具依赖）：`[start]文本` 一行 + 紧跟的空 `[end]` 行（结束标记），
由 `norm2lrc.build_standard_lrc` 产出。

---

## 许可

本工具是 [moriwx/FA-Kara](https://github.com/moriwx/FA-Kara) 的重度魔改分叉，沿用其 **MIT License**
（Copyright (c) 2025 moriwx）。详见 `README.md` 顶部声明。
