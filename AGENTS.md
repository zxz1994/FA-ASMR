# FA-ASMR Agent 操作指南 (`AGENTS.md`)

本文件面向接入本仓库的外部 AI Agent（Antigravity、Claude Code、Codex、Cursor、Gemini CLI 等），说明如何将 **FA-ASMR** 作为确定性物理声学强制对齐引擎编排进完整的日语 ASMR 字幕制作流水线，以及维护本仓库时的工程红线。

---

## 一、职责分工（物理层 vs 语义层）

- **FA-ASMR 负责（确定性物理层）**：根据日文台本与人声音频，纯离线输出毫秒级精确对齐的 `.lrc` 与 `.srt` 原文日轴。
  - 引擎已原生内置自动编码识别（UTF-8 / UTF-8-BOM / Shift-JIS(CP932) / GB18030 / UTF-16 均可直接读取，**无需手动转码**）；
  - 已内置 `ono_filter` 拟声守门、标点重分段与短句合并，产出的日轴已完成碎句整理与孤立杂音过滤；
  - 提供纯离线精度闭环工具：**去重注音词表导出与覆写**（`--export-ruby` / `--ruby-map`）、**对齐后声学体检雷达**（`fa_asmr/pipeline/audit.py` / `--audit`）与**零依赖立体声空间分离**（`fa_asmr_spatial_sep.py` / `--audit-separate`）；
  - 底层固定最优 `3000` 帧分块 + CUDA FP16 推理（峰值显存 ~3.74 GB，4 GB+ 显存即可运行）。
- **Agent 负责（强语境语义层）**：
  1. **前置**：从官方 PDF / DOCX / TXT 提取并清洗纯发音台本（含竖排/分栏/颜色分角色 PDF 解析）、规范说话人标注与文件名配对；
  2. **注音审校与对齐**：导出 `ruby_glossary` 审校专有名词/多音字读音并注入 `--ruby-map`；按剧情与人声清晰度语义判定对齐模式（尤其是区分真双视角 vs 仅隔墙闷音处理），执行对齐并开启 `--audit-separate`；
  3. **体检复核与补全**：结合体检报告的 `suspect_clauses` 与 `unaligned_speech_gaps`，听审切出的原生采样率 WAV（含自动空间分离出的左/中/右干声），补全漏句/即兴加词或删去台本里未录制的删戏句；
  4. **后置**：基于定稿日轴做逐行语境化翻译，输出自然流畅的中文/双语字幕（保留原时间戳与说话人前缀）。

---

## 二、标准端到端工作流

### 阶段 1：原始台本 / PDF 提取与清洗（Agent 执行）

#### 1.1 竖排 / 分栏 / 颜色区分角色的 PDF 提取规范（`PyMuPDF` / `fitz`）

许多官方日文 ASMR 台本采用**竖排（縦書き）**、**上下多栏**或**纯字体颜色区分声优**，提取前必须先检查 `page.get_text("dict")`：

1. **排版方向与列聚类**：
   - 竖排 PDF 的列按 **从右到左（`x` 降序）** 排列，列内字符按 **从上到下（`y` 升序）** 排列；
   - 按 `x_center` 聚类分组（常用容差 `5.0~6.0 pt`），再按 `-x` 排列、列内按 `+y` 排字；
   - 上下分栏页面先按 `y` 切分栏层，再在各层内从右向左排。
2. **颜色识别说话人**：
   - 检查 `span["color"]`：不少多人台本不印角色名，仅用 RGB 颜色区分角色与 SE（如黑色=角色 A、红色=角色 B、蓝色=SE/动作指示）；
   - 说话人切换时输出独立标记行或行首标记：
     ```text
     〖角色名〗
     台词内容……
     ```
3. **页码/页眉页脚联合过滤（防止误切底行正文）**：
   - 严禁仅靠固定 `y` 阈值硬切（竖排正文列常延伸至 `y = 530..555`）；
   - 须联合 **位置 + 字号 + 文本形态**（如 `y > 540` 且 `font_size < 10.5` 且匹配纯数字 / `- 12 -`）过滤页码。
4. **底边跨栏续行自动拼接**：
   - 当竖排长句触底（`prev_col.y1 > 540`）且下一列从顶格续写（`curr_col.y0 < 80`）、**同一说话人**且前列末尾无终结标点（`。！？♡♪…`）时，将两列拼回同一行，避免把词句从中间劈断。
5. **竖排标点与内联注音规范化**：
   - 竖排符号转横排：`︙`/`⋮` → `…`，`︵︶︷︸︹︺︻︼︽︾︿﹀﹁﹂﹃﹄` → `（）｛｝〔〕【】《》〈〉「」『』`；
   - **内联注音处理**：若原台本写为 `漢字（かんじ）`，必须剥除 `（かんじ）` 或转为 `{漢字|かんじ}`（FA-ASMR 会消费 `{漢字|かんじ}` 的读音做对齐并在显示字幕中自动还原为 `漢字`；若保留圆括号则会被当成正文重复对齐）；
   - **保留延音实词**：如 `ろぉ～く`、`ご～お`、`よ～ん`、`さ～ん♡`、`に～～い♡`、`い～～ち♡` 等直接保留原文，`ono_block_corpus.py` 已内置词内延音折叠保护。

#### 1.2 台本清洗、说话人与配对规则

1. **剔除非发音元数据**：删除封面标题、音轨编号（`Track 01` / `トラック1`）、动作与演技指导（`※耳元で`、`（右耳に近づいて）`）、SE/BGM 提示（`（SE：ドアが開く音）`）与声优注释；严禁删改实际发音台词。
2. **说话人标注（多人 / 双视角模式）**：
   - 统一使用 `〖角色名〗`（Unicode `U+3016` / `U+3017`，独立行或行内行首均可），亦原生兼容高频复现的独立 `【角色名】` 标题行；
   - 多人并发同时说话写成连续标记 `〖澄華〗〖天音〗せーの！`；
   - 避免仅靠裸圈号 `①` 区分角色（会被视为正文数字而非角色切换）。
   - 字幕产出格式：`.srt` 自动添加 `【角色名】` 前缀，`.lrc` 添加 `角色名：` 前缀。
3. **文件命名与配对**：
   - 已有 `.txt` 台本支持 UTF-8 / Shift-JIS(CP932) / GB18030 / UTF-16 自动嗅探，**直接读取即可，无需转码**；
   - 常规/多人模式：台本与音频同基名（如 `01_track.flac` + `01_track.txt`，或带角色后缀 `01_track〖澄華〗〖天音〗.txt`）；
   - 双视角模式：两轨视角音频使用统一基名 + 视角后缀（如 `01_A.wav` / `01_B.wav` 或 `トラック7‗主人公視点.wav` / `トラック7‗夫視点.wav`），搭配同基名台本。

---

### 阶段 2：模式选择、校验与 CLI 声学对齐

统一使用仓库内置的无界面命令行入口 `fa_asmr_cli.py`（默认 `--mode auto`，自动根据 `--audio-b` 与台本角色数判定模式）。

#### 1. 四种模式说明与多视角语义判据

| `--mode` | 行为与适用条件 |
| :--- | :--- |
| `auto`（默认） | 有 `--audio-b` 或批量目录检测到双视角后缀 → `multi_persp`；台本检测到 ≥2 位角色 → `multi_speaker`；否则 → `single` |
| `single` | 单声优、台本顺序与音频一致（`fa_asmr/aligners/align.py`） |
| `multi_speaker` | 多声优分轨混音、角色交错/颠倒、台词与喘息重叠（`fa_asmr/aligners/multispeaker.py`） |
| `multi_persp` | 仅用于**真正的互补清晰度双视角**，两轨联合择优对齐并分别投影导出两套独立坐标系字幕（`fa_asmr/aligners/multiperspective.py`） |

> **⚠️ 多视角（`multi_persp`）必须由 Agent 结合剧情/台本语义判断，切勿仅凭文件名含 `視点` / `A/B` 盲开：**
> - **适用 `multi_persp`**：两轨视角**各自都有角色说话，且不同角色或段落的清晰度在两轨之间形成互补**（例如：视角 A 中角色甲贴耳清晰、角色乙发远模糊，而视角 B 中角色乙清晰、角色甲发远；或双机位交替移动导致两轨各有一部分更清晰）。此时两轨联合择优才能显著提升整体打轴精度。
> - **无需 `multi_persp`**：如果另一轨**仅仅是隔墙处理（闷音/低通滤波/整体远距衰减）**，不存在“另一轨反而有某段更清晰人声”的互补关系，则完全没必要跑 `multi_persp`——直接用清晰主轨跑 `single` 或 `multi_speaker` 即可（若隔墙轨也需要字幕，复用清晰主轨的对齐结果即可）。
> - **无法读取音频时**：不支持原生音频的 Agent 无法完成上述判断，**须向用户确认**另一轨属于「互补清晰」还是「仅隔墙闷音」，并请用户指明清晰主轨；在用户答复前不得凭文件名或台本自行选定模式。

#### 2. 预检与注音词表审校（推荐在对齐前先做）

AutoRuby（`fugashi` + `yomikata`）对角色人名、二次元复合词、口语汉字（如 `描→か`、`良い→いい`、`この中→なか`、`出た→で`、`何→なに`、`方→ほー`、`お人形→にんぎょー`）偶有误读。**不要让 LLM 重写整份台本注音**，而是导出紧凑的去重词表审校后通过 `--ruby-map` 精准覆写：

```bash
# 轻量整目录配对预检（不加载模型，秒级检查配对、编码、角色与自动判定的模式）
python fa_asmr_cli.py --batch ./作品目录 --dry-run --json

# 整目录一次加载模型，合并导出去重分级注音词表（高危多音/单字切碎/角色名排最前，按局部搭配去重采样上下文）
python fa_asmr_cli.py --batch ./作品目录 --export-ruby --json-out ./ruby_batch.json

# 单台本深度校验 + 导出去重注音词表（多音词与高危项排最前，附 1-based 原文件行号与局部搭配上下文）
python fa_asmr_cli.py --verify-script ./01.txt --export-ruby --json-out ./ruby_01.json
```

审校 `ruby_batch.json` 中的 `ruby_glossary`（**优先精审 `priority == "high"` 与 `is_polyphone == true` 项，注意核对 `collocations` 与 `examples` 局部搭配**）后，仅把**确有错读**的词写成 `ruby_fixes.json`（无需修改原 `.txt`）：

```json
{
  "global": {
    "陽菜羽": "ひなは",
    "槍峰": "やりみね",
    "同級生": "どーきゅーせい",
    "良い": "いい"
  },
  "by_file": {
    "05": {
      "by_line": {
        "30": { "中": "なか" },
        "48": { "何": "なに" }
      }
    }
  }
}
```

> **注音覆写编写注意**：
> - `global` 采用最长优先匹配，适合**人名与多字词**；
> - 容易嵌在复合词里的**单汉字**（如 `中` 可能出现在 `集中`、`出` 可能出现在 `出て/出して`、`何` 可能出现在 `何度`）优先写入 `by_file -> by_line`（键为原 `.txt` 的 1-based 行号），且单汉字覆写仅在分词后作用于独立 `{单字|...}`，不会劈裂复合词；
> - **`by_line` 绑定的是台本的物理行号**：第 4 步复核中一旦补句或删句，该行之后的行号会整体偏移，必须同步平移该轨的 `by_line` 键。词条落到错误行时**不会报错**——要么静默失效，要么把读音套到新落点行的同名单字上；
> - `ruby_map` 同时在注音前与注音后执行权威覆写，优先级高于 `yomikata` 模型。

#### 3. 执行声学对齐、声学体检与立体声空间分离联动

```bash
# 推荐标准命令：整目录批量对齐（复用显存 + 声学体检 + 默认仅切 priority_review 高危纯 ASCII 中性名 WAV + 16k 单声道瘦身 + 多人轨自动空间分离）
# 日文字幕统一输出到 ./字幕（日文），阶段 3 的 compact_subs.py 直接从该目录读取
python fa_asmr_cli.py --batch ./作品目录 --ruby-map ./ruby_fixes.json --output-dir ./字幕（日文） --audit-separate --audit-clips-compact --audit-out ./audit_batch.json --audit-clips-dir ./audit_clips --json

# 混合目录显式配对清单模式（含单轨+双视角混放、需跳过特典差分轨如 集中Ver、或特殊分隔符命名时，用 --manifest 替代 --batch，同样支持 --dry-run / --export-ruby / --only）
python fa_asmr_cli.py --manifest ./pairs.json --ruby-map ./ruby_fixes.json --output-dir ./字幕（日文） --audit-separate --audit-clips-compact --audit-out ./audit_batch.json --audit-clips-dir ./audit_clips --json

# 仅在发生“重大结构性错位”修完台本后，才用 --only 局部重跑整轨对齐（常规小瑕疵优先走第 4 步 compact_subs.py patch 原地修轴，严禁盲目重跑整轨）
python fa_asmr_cli.py --batch ./作品目录 --only 06 --ruby-map ./ruby_fixes.json --output-dir ./字幕（日文） --audit-separate --audit-clips-compact --audit-out ./audit_realign.json --audit-clips-dir ./audit_clips_realign --json

# 单条对齐（含体检与切片）
python fa_asmr_cli.py --audio ./01.flac --script ./01.txt --ruby-map ./ruby_fixes.json --audit-separate --audit-out ./audit_01.json --audit-clips-dir ./audit_clips --json

# 双视角联合对齐 + 双视角联合体检（仅两侧均跳过才报未录，自动排除对侧闷音串扰）
python fa_asmr_cli.py --audio ./01_A.wav --audio-b ./01_B.wav --script ./01.txt --ruby-map ./ruby_fixes.json --audit-separate --audit-out ./audit_01.json --json
```

常用可选参数：
- `--manifest <JSON>`：传入显式配对清单（支持 `[{script, audio, audio_b?, mode?, name?}]` 或 `{base_dir?, tasks: [...]}`），彻底解决含特典差分轨或异构命名的混合目录编排，无需手写临时 Python 脚本；
- `--device auto|cuda|cpu`：指定推理设备（默认 `auto`）；
- `--line-end-extend / --extend <秒>`：字幕行尾自然延长秒数（默认 `1.2`，补偿 CTC 尾音尖峰偏短、提升 ASMR 阅读停留感；传 `0` 关闭）；
- `--output-dir <目录>`：重定向字幕输出目录（默认输出到各音频同级目录）。**走完整工作流时请固定为 `./字幕（日文）`**：阶段 3 的 `compact_subs.py --jp-dir` 只读取单一目录下的 `.srt` / `.lrc`，而默认输出会散落在各音频旁；
- `--only <列表>`：批量或清单模式下按逗号分隔轨号或文件名关键词筛选（如 `--only 05,07,13`，原生兼容 `トラック5` / `Track 05` 等内嵌轨号）；
- `--ruby-map <JSON>`：注入外部权威注音覆写字典（支持 `--verify-script` / 单条 / `--batch` / `--manifest`）；
- `--audit` / `--audit-out <JSON>`：零额外推理开销，输出声学体检报告（含顶层高危必审汇总 `priority_review`、`suspect_clauses` 与 `unaligned_speech_gaps`）；
- `--audit-clips-dir <目录>`：合并相邻异常窗并切出**纯 ASCII 中性命名**审听 WAV（如 `t02_8f31c9_suspect_01_L10_0.9s_4.3s.wav`，在 `audit.json` 的 `review_clips` 中保留 `source_stem`/`source_audio` 映射，防止原文件名敏感词触发多模态安全拦截）；默认仅切 `priority_review` 高危必审项与缺句空档（加 `--audit-clips-all` 可切全部轻度低分项）；
- `--audit-clips-compact`：将审听切片转为 16kHz 单声道 PCM16（空间分离仍先在原生立体声上执行，确保分离精度不变），体积缩小约 80%–90%，大幅降低多模态听审开销；
- `--audit-separate` / `--audit-sep-sources 0|2|3|4`：正向联动立体声空间分离（`0` 自动判定：**单人轨保留完整体检与原声切片，仅自动跳过空间拆轨**；多人轨按说话人数自动选 2 或 3 路）；利用人头麦双耳相干度 `MSC` + 跨声道残差 `cres` 仅对真叠声段拆出左/中/右干声 WAV（写入 `separated_wavs`）；
- `--no-ono-gate`：临时关闭内置拟声守门（保留全部拟声词原始时间戳供 Agent 自行精修）；
- `--json-out <文件路径>`：将结构化 JSON 摘要写入文件（适用于打包后无控制台 `console=False` 的 `FA-ASMR.exe --cli ...` 场景）。

#### 4. 体检复核与日轴定稿（默认原地修轴，无重大错误严禁全局重跑）

> **⚠️ 严控全局重跑铁律（L1 原地修轴优先 vs L3 重大错误重跑）**：
> - CTC + Trellis 对齐具备强局部锚点收敛性，孤立 1~2 句未录删戏、单句首尾拉伸、错别字或短句跳位**不会破坏周围正常句子的时间轴**；同理，若目录内已有质量良好的新版日轴，亦无需推倒重跑 Stage 1–2 全局对齐。
> - **默认走 L1 原地同步修轴（覆盖 80%+ 场景，零重跑、零 `by_line` 偏移）**：使用 `python tools/compact_subs.py patch` 直接在定稿 `.srt` + `.lrc` 上原子执行删句、插句、改起止时间或替换错字（自动重排 SRT 序号、按时间排序并确定性同步重建 `.lrc`）。
> - **仅 L3 重大结构性崩坏才允许改 `.txt` 跑 `--only` 整轨重对齐**：仅当出现**多处整块删戏/大段漏词引发跨十秒级连锁拉扯吞没**、**多人轨角色标记漏写/写反导致大面积串人**、或**高频核心词注音大面积全错导致整体滑移**时，才回退修改 `.txt`（同步平移受影响的 `by_line` 行号）并执行 `--only` 重跑。

读取 `audit_batch.json` 后，Agent **必须聚焦各轨顶层的 `priority_review` 列表，并遵循「子 Agent 音频隔离」铁律**完成定向实听复核（严禁跳过听审仅凭文本猜测）：

> **⚠️ 音频防上下文爆炸与极速听审铁律**：
> - **主 Agent 严禁直接 `read_file` `.wav` 音频**（多段音频会永久驻留主会话上下文导致极速膨胀卡慢）；
> - **快模型 + 精准背景 + 严限切片范围**：将 `priority_review` 中的 `clip_wav` 按 **每批 3–5 段** 分组派发给只读子 Agent（`research`，**默认用 `Model="flash"`**，仅对 `flash` 明确反馈叠声过重无法辨认的 1~2 段再升 `inherit` 复听），并在 Prompt 中必须做到：
>   1. **附 1~2 句场景/角色/视角背景**（角色名、音色特征、哪个视角贴麦、是否有隔墙背景音）+ 直接透传 `priority_review` 自带的 **`context_before` / `context_after` 前后邻句上下文**，让子 Agent 带答案快速核对而非盲听猜词；
>   2. **严限切片位置与工具边界**：明确规定子 Agent **只能 `read_file` 列表里指定的 WAV 路径**，**严禁扫目录（`listdir`/`glob`）、严禁读其他轨道/台本、严禁写 Python 跑 RMS/波形脚本**；
>   3. **先读原声、按需补读分离轨**：首轮仅并行 `read_file` 原声 `clip_wav`（合并切片已默认限 12 秒上限），仅当原声确实被叠声盖住听不清时才补读对应的 `_K1/_K2` 分离轨，直接回传极简文字结论（是否录制、实际台词、精确起止秒数）。音频 Token 随子 Agent 结束自动释放，主上下文保持 0 音频污染。

> **无原生音频模态的 Agent 降级指引（如 Claude Code / Codex / Cursor 等）**：
> - 前置清洗、注音词表审校、对齐体检与后置翻译均不依赖音频模态，可 100% 正常执行；
> - 到本步骤时，若当前模型/客户端不支持直接读取 `.wav` 音频，**切勿凭文本盲猜删改**——请直接把 `priority_review` 对应的少量短 WAV 路径与上下文列给用户人工听审确认（通常仅 3~8 段各 3~5 秒），或仅对这些局部切片调用本地 ASR 辅助核验后再由 Agent 执行原地修轴（或重大错位 `--only` 重跑）。

1. **L1 常规局部问题 → 直接 `compact_subs.py patch` 原地同步修 `.srt` + `.lrc`（免重跑）**：
   - **错印符号/原台本笔误**：`python tools/compact_subs.py patch --srt "./字幕（日文）/05.srt" --cue 134 --replace "ｘ="`；
   - **孤立 1~2 句未录删戏句（确认相邻句时间戳本身准确）**：`python tools/compact_subs.py patch --srt "./字幕（日文）/01.srt" --delete 114,115`；
   - **单句起止拉伸（`internal_gap_stretch`）或短称呼跳位（`isolated_speaker_jump`）**：听审确认真实起止秒数后，`python tools/compact_subs.py patch --srt "./字幕（日文）/02.srt" --cue 65 --end 411.5`（支持浮点秒数或 `HH:MM:SS,mmm`，改时间后自动按时间重排）；
   - **单句简短即兴加词（`unaligned_speech_gaps`）**：`python tools/compact_subs.py patch --srt "./字幕（日文）/03.srt" --start 192.4 --end 194.1 --speaker "文乃" --insert-text "はいっ……"`；
   - 多轨多处修补也可一次性写入 `patch_ops.json`，执行 `python tools/compact_subs.py patch --sub-dir ./字幕（日文） --ops-json ./patch_ops.json`（同一轨内所有 `cue` 均按修改前原序号定位，互不干扰偏移）。
2. **L3 重大结构性错位 → 改 `.txt` + 平移 `by_line` + `--only` 局部重跑**：
   - 仅当多处连续删戏/漏本导致大范围连锁错位、手工修轴成本高于重跑时：修改清洗台本 `.txt`（若有漏查错读汉字一并补入 `ruby_fixes.json`）；
   - 若该轨在 `ruby_fixes.json` 含有改动点之后的 `by_line` 词条，同步平移物理行号（第 N 行前补 1 行则 ≥N 键 +1；删第 N 行则 >N 键 −1）；
   - 用 `--batch`（或 `--manifest`）配合 `--only <变动轨号>` 重跑该轨，若剩余少量边缘拉伸再用 `compact_subs.py patch` 收尾。
   - 至此确保**日文终轴 100% 定稿后再进入阶段 3 翻译**（避免翻完再返工）。
3. **独立调用立体声空间分离工具（`fa_asmr_spatial_sep.py`）**：
   - 适用于事后基于既有 `audit.json` 补做分离，或**无台本多人人头麦**直接拆声听写（零额外模型权重、原生采样率无损保高频气音）：
   ```bash
   # 事后直接读取体检报告，仅对真重叠异常段做空间分离
   python fa_asmr_spatial_sep.py --from-audit ./audit_01.json --out-dir ./sep_clips --only-stereo-spread --sources 3 --json

   # 手动分离指定重叠时间段（例如 3 人左/中/右）
   python fa_asmr_spatial_sep.py --audio ./02.flac --start 1267 --end 1275 --sources 3 --out-dir ./sep_out --json

   # 无台本全轨自动扫描真叠声区间并批量拆轨
   python fa_asmr_spatial_sep.py --audio ./02.flac --auto-overlaps --sources 3 --out-dir ./sep_overlaps --json
   ```

---

### 阶段 3：后置翻译（Agent 执行）

由于 FA-ASMR 引擎已内置 `ono_filter` 拟声守门（含句中夹心拟声保护）、标点驱动重分段（`resegment`）、短句智能合并（`merge_short`）与短实词顿号桥接，产出的 `.srt` / `.lrc` 日轴默认已完成碎句整理与孤立杂音过滤，**不再需要做旧版的“人工顺接 + 批量剔除杂音”流程**。

#### 3.1 翻译前先调研作品背景（推荐直连 DLsite）

动笔翻译前，Agent 应先查清作品背景与角色设定，避免脱离语境产生称呼错乱、性格割裂或专有名词前后不统一：

1. **识别 RJ 编号并直连 DLsite 查询**：
   - 从作品文件夹名或文件名提取 RJ 编号（如 `RJ01435444`），直接查询对应 DLsite 作品页（如 `https://www.dlsite.com/maniax/work/=/product_id/{RJ编号}.html`，全年龄作可在 `home` 分区或通过作品名检索）；
   - 同时结合本地目录内自带的说明书、台本封面、人物介绍与音轨列表互证（若无网络或页面受限，则以本地资料与前后文推断为准）。
2. **先整理紧凑的《角色设定与术语卡》（约 300~500 字）再开翻**：
   - **剧情与场景**：明确世界观、角色关系、氛围基调以及当前音轨在全剧中的剧情承接；
   - **角色人设与口吻**：锁定每位角色的性格与语域（如温柔年上、傲娇、病娇、敬语系、辣妹系、口癖与自称），以及对听众/主角的固定称呼；
   - **跨轨统一译名**：固定角色名、昵称、专有名词与关键设定词译法，确保同一作品多条音轨风格和译名始终一致；
   - **强制断点（≥ 3 轨长作品）**：在日轴全部定稿后，将《角色设定与术语卡》落盘并**提示用户执行 `/compact`**（或新开会话），甩掉前半程的体检日志与中间历史后再进入翻译阶段。

#### 3.2 紧凑格式逐行语境化翻译规范

1. **用紧凑格式喂给翻译、带锚点防错回填与双视角去重（省 ~35%–55% Token）**：
   - 使用仓库自带的 `tools/compact_subs.py` 剥离 SRT 时间戳与空行：
     - **含双视角作品推荐加 `--dedup-pov`**：`python tools/compact_subs.py export --jp-dir ./字幕（日文） --out ./compact_jp.json --dedup-pov`，自动提取跨视角/跨轨重复台词为 `S0001|...` 共句表，仅需翻译一次即可通过 `@S0001` 自动复用到各视角，既省输出 Token 又 100% 保证跨视角同句同译；
     - **携带 `idx|` 行号锚点回填**：翻译列表统一采用 `"1|译文"` 格式并配合 `build --require-index`（自动剥除 BOM、逐行核验序号锚点，彻底杜绝“一处合行+一处拆行”导致的隐性串行错位）；
     - **深度终检与双语抽检**：`python tools/compact_subs.py verify --jp-dir ./字幕（日文） --cn-dir ./字幕（中文） --spot-check 3`，除校验 1:1 行数/时间戳/说话人/0 假名外，还会自动巡检日文新字体汉字与未翻汉字词残留、规范禁用拟声词、长度失衡以及跨视角同句异译，并输出均匀抽样的中日双语对照供快速终审；
2. **直接信任并沿用定稿日轴的时间轴与分行结构**：逐行一一对应翻译，不随意合并、拆分或改写时间戳；
3. **角色人设与听觉沉浸感优先（严禁跨句倒装）**：贴合已查明的角色口吻与场景情绪翻出自然口语化的中文/双语字幕（避免生硬直译腔），保留原有说话人前缀（SRT 为 `【角色名】`，LRC 为 `角色名：`）与情绪符号（`♡`、`……` 等）；**严禁为了凑书面中文语序而把后句情绪核心词/谓语跨停顿倒装搬到前句**，宁可用自然口语顺承紧贴日语发声推进，确保听觉与字幕实时同步；
4. **仅处理极少数漏网异常**：只有当某行明显属于拟声守门遗漏的纯非发音说明或异常残留时，才结合官方台本做最小化人工微调（严禁写正则脚本批量删改）。

---

## 三、仓库维护与二次开发红线

如果你被要求修改本仓库自身的代码，请遵守以下边界：

1. **核心声学算法区默认冻结（无明确需求勿大改/大拆）**：
   - `fa_asmr/aligners/`（`align.py`、`align_trellis.py`、`align_acoustic.py`、`multispeaker*.py`、`multipersp*.py` 等）包含大量实测标定的声学阈值与 Numba JIT 边界约束；
   - 仅做最小增量修复，不要为了“代码整洁”主动重构算法核。
2. **清晰的分层入口**：
   - **根目录外部/CI 入口（仅 7 个 `.py`）**：`FA-ASMR_GUI.py`（桌面启动器）、`fa_asmr_cli.py`（无界面 CLI）、`fa_asmr_spatial_sep.py`（独立立体声分离工具）、`lrc_end_extend.py`（字幕尾端延展工具），以及打包/构建脚本 `build_in_ci.py`、`prepare_embedded_torch.py`、`_collect_stdlib.py`；
   - `fa_asmr/core/`：`np_compat.py`（NumPy/SciPy 兼容垫片）、`torch_bootstrap.py`（内置 PyTorch 引导）、`advanced_config.py`（进阶硬件配置）、`i18n.py`（三语词典）；
   - `fa_asmr/text/`：`clause.py`、`ono_block_corpus.py`、`ono_base.py`、`ono_filter.py`、`resegment.py`、`merge_short.py`、`haruraw2norm.py`、`inline_leak_clean.py`；
   - `fa_asmr/aligners/`：`align.py`、`align_model.py`、`align_acoustic.py`、`align_trellis.py`、`align_post.py`、`align_utils.py`，多人引擎（`multispeaker.py` + `multispeaker_*.py`）与双视角引擎（`multiperspective.py` + `multipersp_*.py`）；
   - `fa_asmr/pipeline/`：`runner.py`（核心流水线总控）、`pairing.py`（编码嗅探、说话人切分、批量配对）、`ruby.py`（`AutoRuby` 单例、`ruby-map` 覆写与词表提取）、`subtitles.py`（LRC/SRT 组装与齐声合并）、`audit.py`（对齐后声学体检雷达）；
   - `fa_asmr/ui/`：`console.py`（暗色音频控制台主界面）、`widgets.py`（主题控件）、`matrix.py`（多轨编排矩阵）；
   - 新增模块若需随 exe 打包，须同步加入 `FA-ASMR.spec` 的 `hiddenimports`。
3. **改动前后必跑回归测试**：
   ```bash
   python -m unittest discover -s tests -v
   ```
   每修复一个真实台本解析、子句切分、拟声守门或字幕格式 Bug，都应先在 `tests/` 补充对应的最小复现用例，再提交修复。
