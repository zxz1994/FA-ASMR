# -*- coding: utf-8 -*-
"""
fa-asmr 歌词自动注音转换工具 v2
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
主引擎: fugashi + UniDic (词级上下文感知分词注音)
异读层: yomikata (dBert, 模型驱动异读词消歧, 覆盖 UniDic 的异读读音)
回退:   pykakasi (逐字字典)
输出格式: {汉字|假名}  (haruraw2norm.py 兼容)

注: 不维护任何手写异读词典/正则。异读词读音由 yomikata 模型决定,
    其余由 UniDic 引擎决定, 取不到时回退 pykakasi。
    仅保留一个极小的「已验证修正」表, 收人工确认过的 UniDic 字典硬性数据错
    (如 耳穴/耳舐め), 不手写异读词、不做正则。如需纠音, 应修引擎/数据。
"""

import sys
import os
import glob
import json
import contextlib
import unicodedata
from collections import defaultdict, deque


# ─────────────────────────────────────────────────────────────────────────
# yomikata 源码编码缺陷的运行时兜底补丁
#
# 上游 yomikata 0.0.4 有若干 open() 没指定 encoding:
#   yomikata/config/config.py  读 heteronyms.json
#   yomikata/utils.py          读写 label_encoder.json 等
# 这些 JSON 都是 UTF-8。在中文 Windows 上 locale 编码为 cp936，
# json.load() 会抛 UnicodeDecodeError，导致 dBert 加载失败、注音终止。
#
# 本地开发机曾手工改过 site-packages 里的这两个文件，但那是环境级修改，
# 既不能随源码分发，也不会体现在打包产物里 —— 换台机器就复现崩溃。
# 因此在应用侧统一打补丁：接管内置 open()，仅对 yomikata 目录下的
# 文本读写补上 encoding="utf-8"。二进制模式不动。
# ─────────────────────────────────────────────────────────────────────────
_YOMIKATA_ENC_PATCHED = False


@contextlib.contextmanager
def _patch_yomikata_encoding():
    """在 with 块内，把 yomikata 目录下的文本读写默认编码改为 UTF-8。

    全局替换 builtins.open 有性能与影响面代价，故仅覆盖 yomikata 导入与
    dBert 构造这一段（JSON 读取发生在构造期的 load() 中），退出即还原。
    嵌套使用时内层复用外层补丁，不做重复替换。
    """
    global _YOMIKATA_ENC_PATCHED
    if _YOMIKATA_ENC_PATCHED:
        yield
        return

    import builtins

    _real_open = builtins.open

    def _utf8_open(file, mode="r", buffering=-1, encoding=None,
                   errors=None, newline=None, closefd=True, opener=None):
        # 未显式指定编码、且为文本模式时，才介入
        if encoding is None and "b" not in mode:
            try:
                path = os.fspath(file)
                if isinstance(path, (str, bytes)):
                    p = path if isinstance(path, str) else path.decode(
                        "utf-8", "replace")
                    norm = p.replace("\\", "/")
                    if ("/yomikata/" in norm
                            or norm.rstrip("/").endswith("/yomikata")):
                        encoding = "utf-8"
            except TypeError:
                pass
        return _real_open(file, mode, buffering, encoding,
                          errors, newline, closefd, opener)

    builtins.open = _utf8_open
    _YOMIKATA_ENC_PATCHED = True
    try:
        yield
    finally:
        builtins.open = _real_open
        _YOMIKATA_ENC_PATCHED = False


def _build_radical_map() -> "dict[str, str]":
    """
    自动构建「部首字符 → 普通汉字」映射 (不手写词典, 导入时生成一次)。

    背景: 台本里混入了长得像汉字但码位不同的「部首字符」, UniDic/pykakasi
          均无法识别, 会当成未知字整行读错。分两个区处理:

      1. 康熙部首 U+2F00-2FD5 (如 ⼈ U+2F08): Unicode 官方有 NFKC 映射,
         normalize('NFKC') 即可还原 → 人。此处仅用它来建「部首名 → 汉字」索引。
      2. CJK部首补充 U+2E80-2EF3 (如 ⻑ U+2ED1、⻲ U+2EF2): NFKC **不处理**,
         必须自建映射。做法是用 Unicode 字符名与康熙部首对齐:
             'CJK RADICAL LONG ONE'          → 剥修饰词 → 'LONG'   → 長
             'CJK RADICAL J-SIMPLIFIED TURTLE' → 剥修饰词 → 'TURTLE' → 龜
    """
    # 1) 借 NFKC 建立「康熙部首名 → 普通汉字」索引
    kangxi: "dict[str, str]" = {}
    for cp in range(0x2F00, 0x2FD6):
        ch = chr(cp)
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        if not name.startswith('KANGXI RADICAL '):
            continue
        target = unicodedata.normalize('NFKC', ch)
        if len(target) == 1 and target != ch:
            kangxi[name[len('KANGXI RADICAL '):]] = target

    # 2) 部首补充区: 剥掉字形修饰词后按名字匹配康熙部首
    #    (修饰词表示同一部首的异体/简化/变体写法, 语义上是同一个字)
    prefixes = ('J-SIMPLIFIED ', 'C-SIMPLIFIED ', 'SIMPLIFIED ', 'KANGXI ')
    suffixes = (' ONE', ' TWO', ' THREE', ' FOUR', ' FIVE', ' SIX')
    mapping: "dict[str, str]" = {}
    for cp in range(0x2E80, 0x2EF4):
        ch = chr(cp)
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        if not name.startswith('CJK RADICAL '):
            continue
        key = name[len('CJK RADICAL '):]
        for p in prefixes:
            if key.startswith(p):
                key = key[len(p):]
                break
        for s in suffixes:
            if key.endswith(s):
                key = key[:-len(s)]
                break
        target = kangxi.get(key)
        if target:
            mapping[ch] = target

    # 3) 日本简化字形修正: 名字带 J-SIMPLIFIED 的部首本身即日本字形,
    #    但上一步按部首名匹配会落到康熙繁体 (旧字体), UniDic 读不出。
    #    依据 Unicode 名的 J-SIMPLIFIED 客观标记, 改指日本新字体。
    #    (⻲→龜 会让 UniDic 输出 None, 改 亀 后「亀頭」正常读作 きとう)
    shinjitai = {'齊': '斉', '齒': '歯', '龍': '竜', '龜': '亀'}
    for cp in range(0x2E80, 0x2EF4):
        ch = chr(cp)
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        if 'J-SIMPLIFIED' in name and ch in mapping:
            mapping[ch] = shinjitai.get(mapping[ch], mapping[ch])
    return mapping


# 部首补充区映射 (康熙部首区交给 NFKC, 无需入表)
_RADICAL_MAP = _build_radical_map()


# MMS_FA 词表外的日语系字符处理表 (见 RubyAnnotator._strip_unmappable_kana)
#   · 合字假名 ヷヸヹヺ (VA/VI/VE/VO): 转写为等价的 ヴ+小写元音, 保住读音
#   · 迭字符 ゝゞヽヾ: 语义是"重复前一字符", 无独立读音 → 删除 (映射为空串)
#   · 未分配码位 U+3040/U+3097/U+3098: 无读音 → 删除
_UNMAPPABLE_KANA = {
    '\u30F7': '\u30F4\u30A1',   # ヷ → ヴァ (va)
    '\u30F8': '\u30F4\u30A3',   # ヸ → ヴィ (vi)
    '\u30F9': '\u30F4\u30A7',   # ヹ → ヴェ (ve)
    '\u30FA': '\u30F4\u30A9',   # ヺ → ヴォ (vo)
    '\u309D': '',               # ゝ 平假名迭字符
    '\u309E': '',               # ゞ 平假名浊音迭字符
    '\u30FD': '',               # ヽ 片假名迭字符
    '\u30FE': '',               # ヾ 片假名浊音迭字符
    '\u3040': '',               # 未分配
    '\u3097': '',               # 未分配
    '\u3098': '',               # 未分配
}





class RubyAnnotator:
    """
    日文汉字注音器 v2
    ────────────────
    主引擎: fugashi + UniDic (词级分词 + 读音)
    回退:   pykakasi (逐字)
    预清洗: NFKC 规范化 + 浊音假名修正 + 波浪号转换
    已验证修正: 极小白名单, 仅覆盖人工确认过的 UniDic 字典数据错 (非异读/非正则)
    """

    def __init__(self, verbose: bool = False, strip_inline_readings: bool = False):
        self.verbose = verbose
        self.strip_inline_readings = strip_inline_readings
        self._tagger = None
        self._pykakasi = None
        self._yomikata = None          # 懒加载 yomikata dBert
        self._yomikata_loaded = False
        self._heteronyms = None        # 懒加载异读词表 (surface 集合)
        self._init_fugashi()

    # ── 初始化 ──────────────────────────────────

    def _init_fugashi(self) -> None:
        """尝试加载 fugashi, 失败则标记为 None"""
        try:
            import fugashi
            self._tagger = fugashi.Tagger()
            if self.verbose:
                print("[RubyAnnotator] 主引擎: fugashi + UniDic")
        except Exception:
            self._tagger = None
            if self.verbose:
                print("[RubyAnnotator] fugashi 未安装, 回退到 pykakasi")
        self._ensure_pykakasi()

    def _ensure_pykakasi(self) -> None:
        """确保 pykakasi 可用"""
        if self._pykakasi is not None:
            return
        try:
            import pykakasi
            self._pykakasi = pykakasi.kakasi()
        except ImportError:
            print("【错误】pykakasi 未安装。请运行: pip install pykakasi")
            sys.exit(1)

    @property
    def has_fugashi(self) -> bool:
        return self._tagger is not None

    # ── 文本规范化 ──────────────────────────────

    @staticmethod
    def normalize(text: str) -> str:
        """
        预处理: 解决 G2P 引擎 (pykakasi/fugashi/UniDic) 会报错或错读的输入。
        纯字符串扫描, 不依赖正则。
        """
        # 0. 部首字符 → 普通汉字 (⻑→長、⻲→龜)
        #    CJK部首补充区 U+2E80-2EF3 无 NFKC 映射, 必须先手动替换;
        #    康熙部首区 U+2F00-2FD5 (⼈→人) 由下一步 NFKC 自动处理。
        text = RubyAnnotator._normalize_radicals(text)

        # 1. NFKC 规范化 → 合并 compatibility chars (康熙部首、CJK兼容汉字、全角英数等)
        text = unicodedata.normalize('NFKC', text)

        # 2. 剥离残留的非标准浊点 (U+3099/U+309A combining)
        text = text.translate(str.maketrans('', '', '\u3099\u309A'))

        # 2.5 NFKC 将 U+309B(spacing) 映射为 U+0020+U+3099, 剥离后剩孤儿空格
        #     → 删除『假名 + 空格 + (假名/汉字/{)』之间的孤儿空格
        text = RubyAnnotator._strip_orphan_spaces(text)

        # 3. 波浪号 〜 → 长音符 ー (假名后) 或删除 (孤立)
        text = RubyAnnotator._normalize_wavedash(text)

        # 4. 连续促音截断 (CTC 数字约束: max 3)
        text = RubyAnnotator._truncate_sokuon(text)

        # 5. 剥离 MMS_FA 词表外的日语系字符 (迭代符/合字/未分配码位)
        #    这些字符 NFKC 不处理, 也不是 kana 区间内的可转写假名, 会原样漏进
        #    romaji token 列表 (如 ゞ U+309E), 而令牌表仅含 a-z 与 ' * - ,
        #    下游 tokenizer 直接 KeyError / 产出未知音素污染对齐标签。
        text = RubyAnnotator._strip_unmappable_kana(text)

        return text.strip()

    @staticmethod
    def _strip_unmappable_kana(text: str) -> str:
        """剥离无法转写为罗马音、且不在 MMS_FA 词表内的日语系字符。

        这些字符 NFKC 不处理, 也不属于可转写的假名区间, 会原样漏进 romaji
        token 列表; 而 MMS_FA 字符表仅含 a-z 与 ' * - , 下游会产出未知音素
        污染对齐标签。

        故意不使用 NFD 分解: NFD 会把规范浊音假名 (が→か+U+3099) 一并拆开,
        后续剥离组合浊点将丢失浊音, 破坏正常读音。此处只做显式映射:
          · 合字假名 ヷヸヹヺ (VA/VI/VE/VO) → 等价的 ヴ+小写元音
          · 迭字符 ゝゞヽヾ 与未分配码位 U+3040/U+3097/U+3098 → 删除
        """
        if not any(ch in _UNMAPPABLE_KANA for ch in text):
            return text
        return ''.join(_UNMAPPABLE_KANA.get(ch, ch) for ch in text)

    @staticmethod
    def _normalize_radicals(text: str) -> str:
        """CJK部首补充区 (U+2E80-2EF3) 部首字符 → 普通汉字"""
        if not _RADICAL_MAP:
            return text
        return ''.join(_RADICAL_MAP.get(ch, ch) for ch in text)

    @staticmethod
    def _strip_orphan_spaces(text: str) -> str:
        """删除『假名 + 空格 + (假名/汉字/{)』之间的孤儿空格"""
        out: "list[str]" = []
        n = len(text)
        for i, ch in enumerate(text):
            if ch == ' ' and i > 0 and i + 1 < n:
                prev, nxt = text[i - 1], text[i + 1]
                if (RubyAnnotator._is_kana(prev)
                        and (RubyAnnotator._is_kana(nxt)
                             or RubyAnnotator._is_kanji(nxt)
                             or nxt == '{')):
                    continue  # 删除孤儿空格
            out.append(ch)
        return ''.join(out)

    @staticmethod
    def _normalize_wavedash(text: str) -> str:
        """保留所有 〜 / ～ / ~ 等波浪号与长音符，不再截断或删除"""
        return text

    @staticmethod
    def _truncate_sokuon(text: str) -> str:
        """连续促音 (っ/ッ) 超过 3 个则截断 (CTC 数字约束)"""
        out: "list[str]" = []
        run = 0
        for ch in text:
            if ch == '\u3063' or ch == '\u30C3':
                if run < 3:
                    out.append(ch)
                    run += 1
                # 超过 3 个的促音丢弃
            else:
                out.append(ch)
                run = 0
        return ''.join(out)

    # ── 已验证修正 (解耦独立配置文件: overrides.json) ──
    # 用户可在 furigana/overrides.json 中维护人名、生造词、专有注音白名单。
    #
    # ⚠ 下方两个内置字典是**兜底**，仅当 overrides.json 缺失该键时才生效：
    #     最终值 = 内置值 → 被 json 的 update() 覆盖
    #   因此改这里对已存在于 json 的词**不会有任何效果**。要改注音请直接改
    #   overrides.json（内置仅保留 json 中没有的键）。
    @staticmethod
    def _load_overrides():
        base_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(base_dir, "overrides.json")
        # 仅保留 json 中不存在的兜底项（原先这里还有 6 个词 + 1 条短语与 json
        # 完全同值，属重复维护：改了不生效，容易误导）。
        words = {
            "搾": "しぼ",
        }
        phrases = {}
        if os.path.isfile(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "words" in data and isinstance(data["words"], dict):
                    words.update(data["words"])
                if "phrases" in data and isinstance(data["phrases"], dict):
                    phrases.update(data["phrases"])
            except Exception as e:
                print(f"[RubyAnnotator] 读取 overrides.json 失败: {e}")
        return words, phrases

    _VERIFIED_OVERRIDES, _VERIFIED_PHRASE_OVERRIDES = _load_overrides()

    # ── 代词默认读法覆盖 (词级, 安全不误伤复合词) ──
    _PRONOUN_OVERRIDES = {
        "私": "ワタシ",
    }

    @classmethod
    def _apply_verified_overrides(cls, text: str) -> str:
        """在注音前, 把人工确认过的 UniDic 字典数据错整体预注音, 由 _annotate_mixed 原样保留。"""
        out = []
        for part in cls._split_ruby(text):
            if cls._is_ruby_span(part):
                out.append(part)
                continue
            for surf, read in cls._VERIFIED_OVERRIDES.items():
                if surf in part:
                    part = part.replace(surf, "{%s|%s}" % (surf, read))
            for phrase, ruby in cls._VERIFIED_PHRASE_OVERRIDES.items():
                if phrase in part:
                    part = part.replace(phrase, ruby)
            out.append(part)
        return "".join(out)

    # ── 注音核心 ────────────────────────────────

    def annotate(self, text: str, use_yomikata: bool = True) -> str:
        """
        将日文文本转换为 {漢字|かな} 格式
        对已有注音的行直接返回原文
        """
        if not text or not text.strip():
            return text

        # 必须先 normalize 再判断是否需要注音:
        # 部首字符 (⼈ U+2F08 / ⻑ U+2ED1) 不在 _has_kanji 的 CJK 区间内,
        # 若先判断会被当作「纯假名行」跳过注音, 归一后的裸汉字将漏注。
        text = self.normalize(text)

        # Some subtitle/script sources store furigana as unmarked trailing kana
        # (私たし, 身体んたい, 沖縄きなわ).  Feeding both the kanji reading and
        # that suffix to CTC duplicates phonemes.  The cleanup is opt-in until a
        # caller has identified this source format.
        if self.strip_inline_readings and self._tagger:
            text = self._strip_inline_reading_suffixes(text)

        # 纯假名/英文/标点行无需注音 (normalize 已完成清洗: 部首/NFKC/浊点/〜/
        # 全角/超长促音), 否则残留字符会产出 MMS_FA vocab 外非法 token。
        if not self._needs_ruby(text):
            return text

        # 已验证修正: 覆盖人工确认过的 UniDic 字典数据错 (整词/词组级)
        text = self._apply_verified_overrides(text)

        # 按已有 ruby 标记分段, 只对非 ruby 片段注音
        result = self._annotate_mixed(text)
        # 孤立的 ASR word timestamp 没有可供 dBert 消歧的句子上下文。
        # 调用方可关闭该层；默认值保持现有台本流程的行为。
        return self._apply_yomikata(text, result) if use_yomikata else result

    def _strip_inline_reading_suffixes(self, text: str) -> str:
        """Remove unmarked kana suffixes that repeat a kanji token's reading.

        At least two kana must match the tail of the reading.  This deliberately
        avoids one-kana guesses such as ``物の`` where ``の`` is a real particle.
        Existing ``{漢字|かな}`` spans are handled before this method by the
        mixed annotator and are left untouched here.
        """
        if not text or "{" in text:
            return text
        replacements = []
        pos = 0
        for word in self._tagger(text):
            surface = word.surface
            start = text.find(surface, pos) if surface else pos
            if start < 0:
                start = pos
            end = start + len(surface)
            pos = end
            if not surface or not self._has_kanji(surface) or end >= len(text):
                continue
            if surface in self._VERIFIED_OVERRIDES:
                reading = self._VERIFIED_OVERRIDES[surface]
            elif surface in self._PRONOUN_OVERRIDES:
                reading = self._kata_to_hira(self._PRONOUN_OVERRIDES[surface])
            else:
                reading_raw = self._get_fugashi_reading(word)
                if reading_raw is None:
                    continue
                reading = self._kata_to_hira(reading_raw)

            # A token may have a more appropriate pronunciation for this source
            # in its kana field or pykakasi.  Keep each candidate only for
            # matching; use the candidate that actually matched in the ruby.
            match_readings = [reading]
            if surface not in self._VERIFIED_OVERRIDES and surface not in self._PRONOUN_OVERRIDES:
                kana = getattr(getattr(word, "feature", None), "kana", None)
                if kana:
                    kana = self._kata_to_hira(kana)
                    if kana not in match_readings:
                        match_readings.append(kana)
                self._ensure_pykakasi()
                fallback = "".join(item.get("hira", "") for item in self._pykakasi.convert(surface))
                if fallback and fallback not in match_readings:
                    match_readings.append(fallback)
            j = end
            while j < len(text) and self._is_kana(text[j]):
                j += 1
            run = text[end:j]
            if len(run) < 2:
                continue
            matched = 0
            matched_reading = None
            # 只认「整个假名串 == 读音尾」为未标记读音后缀。原实现拿 run 的
            # 前缀逐个 size 去凑读音尾，会把真实助词/名词误当读音吞掉：
            #   女の子のこと -> run=のこと，size=2 的 のこ 恰是 おんなのこ 的后缀
            #   -> のこと 被吞成 と。放宽前缀匹配是这类假阳性的根源。
            matched_reading = next((item for item in match_readings if item.endswith(run)), None)
            if matched_reading is not None:
                matched = len(run)
            if matched:
                replacements.append(
                    (start, end + matched, self._format_ruby(surface, matched_reading)))
        if not replacements:
            return text
        out = []
        cursor = 0
        for start, end, replacement in replacements:
            if start < cursor:
                continue
            out.append(text[cursor:start])
            out.append(replacement)
            cursor = end
        out.append(text[cursor:])
        return "".join(out)

    def _needs_ruby(self, text: str) -> bool:
        """文本是否含有需要注音的汉字 (已有 {kanji|kana} 的片段由 _annotate_mixed 保留)"""
        return self._has_kanji(text)

    def _annotate_mixed(self, text: str) -> str:
        """
        将文本按已有 {kanji|kana} 分段:
         - 已有注音的片段 => 原样保留
         - 无注音的片段   => 送入引擎 (fugashi/pykakasi)
        """
        parts = self._split_ruby(text)
        engine = self._annotate_fugashi if self._tagger else self._annotate_pykakasi

        result: "list[str]" = []
        for part in parts:
            if self._is_ruby_span(part):
                result.append(part)        # 已有注音 → 保留
            elif part:
                result.append(engine(part))  # 非注音片段 → 注音
        return ''.join(result)

    # ── yomikata 异读词消歧层 (模型驱动, 不手写词典) ──────────

    def _apply_yomikata(self, text: str, ruby: str) -> str:
        """用 yomikata (dBert) 对异读词做上下文消歧, 仅覆盖对应 {surface|...} 片段的读音。
        其余词完全保留 UniDic/pykakasi 的注音, 不污染。"""
        hetero = self._get_heteronyms()
        plain = self._strip_ruby(text)
        # 无任意异读词则跳过模型调用
        if not any(w in plain for w in hetero):
            return ruby
        # Loading dBert is expensive.  Do it only after the cheap lexicon test
        # proves that this text actually contains an ambiguous surface.
        yk = self._get_yomikata()
        try:
            yomi = yk.furigana(plain)
        except Exception as e:
            raise RuntimeError(
                "yomikata dBert 推理失败；为避免静默回退到 UniDic，已终止注音"
            ) from e
        pairs = self._extract_yomi_pairs(yomi)
        if not pairs:
            return ruby
        return self._merge_yomi(ruby, pairs)

    def _get_yomikata(self):
        """懒加载 yomikata dBert；加载失败时明确终止，禁止静默回退。"""
        if self._yomikata_loaded:
            if self._yomikata is None:
                raise RuntimeError("yomikata dBert 此前加载失败")
            return self._yomikata
        self._yomikata_loaded = True
        try:
            # NumPy 2.x removed aliases still imported by the current
            # yomikata/transformers dependency chain.
            # 原为三处重复副本之一，已统一到 np_compat（幂等，可安全叠加调用）。
            # np_compat 位于项目根目录（本文件在 furigana/ 子目录），
            # 正常由 GUI 启动时加入 sys.path；此处补一次回退，
            # 保证本模块被单独导入时也能找到。
            try:
                import np_compat
            except ImportError:
                import os as _os
                import sys as _sys
                _sys.path.insert(
                    0, _os.path.dirname(_os.path.dirname(
                        _os.path.abspath(__file__))))
                import np_compat
            np_compat.apply()
            # yomikata hardcodes rotating logs under site-packages/logs.
            # Redirect those handlers to the writable application directory
            # so importing dBert never depends on package-directory access.
            import logging.config
            logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
            os.makedirs(logs_dir, exist_ok=True)
            original_dict_config = logging.config.dictConfig

            def _app_log_dict_config(config):
                local_config = dict(config)
                local_config["handlers"] = {
                    name: dict(handler)
                    for name, handler in config.get("handlers", {}).items()
                }
                for name, handler in local_config.get("handlers", {}).items():
                    if "filename" in handler:
                        handler["filename"] = os.path.join(
                            logs_dir, os.path.basename(str(handler["filename"])))
                original_dict_config(local_config)

            logging.config.dictConfig = _app_log_dict_config
            try:
                # 补丁必须覆盖 dBert() 构造：权重 JSON 是在构造期的 load() 里读的
                with _patch_yomikata_encoding():
                    from yomikata.dbert import dBert
                    self._yomikata = dBert()
            finally:
                logging.config.dictConfig = original_dict_config
        except Exception as e:
            self._yomikata = None
            raise RuntimeError(
                "yomikata dBert 加载失败；为避免静默回退到 UniDic，已终止注音"
            ) from e
        return self._yomikata

    def release_gpu(self):
        """释放 yomikata dBert (常驻 CUDA) 与 CUDA 缓存。一条音频注完台本后调用。"""
        self._yomikata = None
        self._yomikata_loaded = False
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _get_heteronyms(self):
        """懒加载 yomikata 自带异读词表 (surface 集合), 仅用于是否调用模型的快速判断。"""
        if self._heteronyms is None:
            self._heteronyms = set()
            try:
                import json
                import yomikata
                p = os.path.join(os.path.dirname(yomikata.__file__), 'config', 'heteronyms.json')
                with open(p, encoding='utf-8') as f:
                    self._heteronyms = set(json.load(f).keys())
            except Exception as e:
                raise RuntimeError(
                    "无法加载 yomikata 异读词表；为避免静默跳过 dBert，已终止注音"
                ) from e
        return self._heteronyms

    @staticmethod
    def _strip_ruby(s: str) -> str:
        """去掉文本中的 {汉字|假名}, 只保留汉字表面形 (供 yomikata 吃纯文本)。手动扫描。"""
        out = []
        i, n = 0, len(s)
        while i < n:
            c = s[i]
            if c == '{':
                j = s.find('}', i)
                if j == -1:
                    out.append(c)
                    i += 1
                    continue
                inner = s[i + 1:j]
                surf = inner.split('|')[0] if '|' in inner else inner
                out.append(surf)
                i = j + 1
            else:
                out.append(c)
                i += 1
        return ''.join(out)

    @staticmethod
    def _extract_yomi_pairs(yomi: str):
        """从 yomikata 输出 (如 {表/おもて} 或 {表:おもて}) 提取有序 (surface, reading)。
        跳过无读音的畸形输出 (如 {一日}, <OTHER> 弃权)。"""
        pairs = []
        i, n = 0, len(yomi)
        while i < n:
            if yomi[i] == '{':
                j = yomi.find('}', i)
                if j == -1:
                    break
                inner = yomi[i + 1:j]
                if '/' in inner:
                    surf, read = inner.split('/', 1)
                elif ':' in inner:
                    surf, read = inner.split(':', 1)
                else:
                    surf, read = inner, ''
                if surf and read:
                    pairs.append((surf, read))
                i = j + 1
            else:
                i += 1
        return pairs

    # 平假名 → 所属段 (用于把 ー 展开为等价的长音假名)
    _VOWEL_ROWS = (
        ('あぁかがさざただなはばぱまゃやらわ', 'あ'),
        ('いぃきぎしじちぢにひびぴみり', 'い'),
        ('うぅくぐすずつづぬふぶぷむゅゆるゔ', 'う'),
        ('えぇけげせぜてでねへべぺめれ', 'え'),
        ('おぉこごそぞとどのほぼぽもょよろを', 'お'),
    )
    _LONG_CHAR = {'あ': 'あ', 'い': 'い', 'う': 'う', 'え': 'い', 'お': 'う'}

    @classmethod
    def _expand_macron(cls, s: str) -> str:
        """把发音形长音 ー 展开为等价的长音假名 (きょー → きょう)。

        仅用于比较两种写法的读音是否等价, 不用于产出注音。
        """
        vowel = {}
        for chars, v in cls._VOWEL_ROWS:
            for c in chars:
                vowel[c] = v
        out = []
        i, n = 0, len(s)
        while i < n:
            c = s[i]
            if i + 1 < n and s[i + 1] in 'ゃゅょぁぃぅぇぉ':
                out.append(s[i:i + 2])
                i += 2
                continue
            if c == 'ー' and out:
                v = vowel.get(out[-1][-1])
                out.append(cls._LONG_CHAR.get(v, 'ー'))
            else:
                out.append(c)
            i += 1
        return ''.join(out)

    @classmethod
    def _merge_yomi(cls, ruby: str, pairs):
        """按 surface 把 yomikata 的读音覆盖进 UniDic 的 {surface|...} 片段。
        用 per-surface 队列, 同词多次出现依次消费, 未在 ruby 中出现则忽略 (不污染)。

        长音写法保护: dBert 输出的是「假名形」(今日→きょう), 而 UniDic 输出
        「发音形」(今日→きょー)。二者只是书写差异, 但下游 sylla_split 会把 ー
        并入前一音节 (きょー→[kyooooo] 1 个 token), 而假名形会拆出独立的 う
        (きょう→[kyo, u] 2 个 token), 凭空多一个虚假音素, 拉伸强制对齐。
        故当两者仅相差长音写法时, 保留 UniDic 的发音形; 只有 dBert 给出的是
        真正的异读消歧 (如 身体 しんたい→からだ) 才采用 dBert 的读音。
        """
        q = defaultdict(deque)
        for surf, read in pairs:
            q[surf].append(read)
        out = []
        i, n = 0, len(ruby)
        while i < n:
            if ruby[i] == '{':
                j = ruby.find('}', i)
                if j == -1:
                    out.append(ruby[i:])
                    break
                inner = ruby[i + 1:j]
                if '|' in inner:
                    surf, uni_read = inner.split('|', 1)
                    dq = q.get(surf)
                    if dq:
                        yomi_read = dq.popleft()
                        # 仅书写差异 → 保留 UniDic 发音形
                        if cls._expand_macron(uni_read) == yomi_read:
                            out.append(ruby[i:j + 1])
                        else:
                            out.append('{%s|%s}' % (surf, yomi_read))
                    else:
                        out.append(ruby[i:j + 1])
                else:
                    out.append(ruby[i:j + 1])
                i = j + 1
            else:
                out.append(ruby[i])
                i += 1
        return ''.join(out)

    @staticmethod
    def _is_ruby_span(part: str) -> bool:
        """判断片段是否为完好的 {kanji|kana} 注音片段"""
        return bool(part) and part[0] == '{' and part[-1] == '}' and '|' in part[1:-1]

    @staticmethod
    def _split_ruby(text: str) -> "list[str]":
        """
        将文本切成 [非注音片段, 注音片段{...|...}, 非注音片段, ...]。
        不用正则: 手动扫描花括号配对。
        """
        parts: "list[str]" = []
        buf: "list[str]" = []
        i, n = 0, len(text)
        while i < n:
            c = text[i]
            if c == '{':
                end = i + 1
                depth = 1
                while end < n and depth > 0:
                    if text[end] == '{':
                        depth += 1
                    elif text[end] == '}':
                        depth -= 1
                    end += 1
                span = text[i:end]
                if depth == 0 and '|' in span[1:-1]:
                    if buf:
                        parts.append(''.join(buf))
                        buf = []
                    parts.append(span)
                    i = end
                    continue
                buf.append(c)
                i += 1
            else:
                buf.append(c)
                i += 1
        if buf:
            parts.append(''.join(buf))
        return parts

    # ── fugashi 注音 ────────────────────────────

    def _annotate_fugashi(self, text: str) -> str:
        """
        使用 fugashi 分词 → 词条读音 → {kanji|kana} 输出
        """
        words = list(self._tagger(text))
        if not words:
            return self._annotate_pykakasi(text)

        output_parts: "list[str]" = []
        pos = 0

        for word in words:
            surface = word.surface
            start = text.find(surface, pos) if surface else pos

            # 输出词间非词条字符 (标点、空格等)
            if start > pos:
                output_parts.append(text[pos:start])

            # 获取读音
            reading = self._get_fugashi_reading(word)
            # 代词默认读法覆盖(词级, 不误伤复合词: 私生活/私事 等 surface≠'私' 不命中)
            if reading is not None and surface in self._PRONOUN_OVERRIDES:
                reading = self._PRONOUN_OVERRIDES[surface]
            if reading is None:
                # fugashi 无法读取 → 回退 pykakasi
                output_parts.append(self._annotate_pykakasi(surface))
            else:
                # 转换片假名读音 → 平假名
                reading_hira = self._kata_to_hira(reading)
                # 修正 UniDic 已知的不稳定读音 (仅精确匹配 surface, 不影响复合词)
                reading_hira = self._apply_reading_fix(surface, reading_hira)
                output_parts.append(self._format_ruby(surface, reading_hira))

            pos = start + len(surface)

        # 尾部剩余
        if pos < len(text):
            output_parts.append(text[pos:])

        return ''.join(output_parts)

    # 四つ仮名: pron(発音形) 会把 ヅ/ヂ 表音化为 ズ/ジ, 需借 kana(仮名形) 还原正字法
    _YOTSUGANA_RESTORE = {('ズ', 'ヅ'), ('ジ', 'ヂ')}

    @classmethod
    def _restore_yotsugana(cls, pron: str, kana: str) -> str:
        """
        UniDic 的 pron 是「発音形」(妖魔→ヨーマ, 続け→ツズケ),
        kana 是「仮名形」(妖魔→ヨウマ, 続け→ツヅケ)。

        pron 把长音写作 ー, 下游 sylla_split 会将 ー 并入前一音节
        (よーま→[よー,ま]), 比 kana 的 ようま→[よ,う,ま] 少一个虚假 token,
        对强制对齐更有利 → 长音必须保留 pron 的写法。

        但 pron 同时把 ヅ/ヂ 合并成了 ズ/ジ (四つ仮名), 属正字法丢失
        (続け→ツズケ, 鼻血→ハナジ) → 逐位对照 kana 还原。

        仅在两者等长时逐位比较, 且只还原四つ仮名;
        其余差异 (助词 ハ→ワ 等发音层修正) 一律保持 pron。
        """
        if not kana or len(pron) != len(kana):
            return pron
        chars = list(pron)
        for i, (p, k) in enumerate(zip(pron, kana)):
            if p != k and (p, k) in cls._YOTSUGANA_RESTORE:
                chars[i] = k
        return ''.join(chars)

    def _get_fugashi_reading(self, word) -> "str | None":
        """从 fugashi 词条提取读音 (片假名)
        优先级: pron(表层读法) > lForm(词条读法) > kana"""
        try:
            feat = word.feature
            # pron=表层发音 (开い→ヒライ), lForm=词典形 (开い→ヒラク)
            pron = getattr(feat, 'pron', None)
            kana = getattr(feat, 'kana', None)
            if pron:
                return self._restore_yotsugana(pron, kana or '')
            lform = getattr(feat, 'lForm', None)
            if lform:
                return lform
            if kana:
                return kana
        except Exception:
            pass

        # 回退: 按索引提取 (兼容旧格式)
        #   str(word.feature) 是 UnidicFeatures29 的 repr, 未取到的字段渲染为
        #   裸 None (如 " pron=None"), 经 split('=')[-1] 会得到字符串 'None'。
        #   若原样返回, 调用方的 `reading is None` 判断失效, 下游按字面拆成
        #   N-o-n-e 四个音素污染对齐标签。必须显式识别并继续尝试其它索引。
        try:
            feats = str(word.feature).split(',')
            for idx in (9, 7, 6):
                if len(feats) <= idx or not feats[idx]:
                    continue
                val = feats[idx].split('=')[-1].strip("'\"")
                if not val or val == '*' or val == 'None':
                    continue
                return val
        except Exception:
            pass

        return None

    # ── UniDic 读音修正表 ────────────────────────
    # 背景: UniDic 对同一表记在不同句子里会给出不一致的读音 (词格判定不稳定)。
    #   实测例: 「舌」单独→シタ, 「舌、」→ゼツ; 「耳」单独→ジ, 「耳、」→ミミ。
    #   这类错误无法靠引擎自洽修复 (pron/kana/lForm/pronBase 四个字段都不稳定),
    #   故在此维护一张「已验证正确读音」表。
    #
    # 数据来源: 4 套独立词典 (UniDic / OpenJTalk-NAIST / IPAdic / Sudachi / pykakasi)
    #   交叉比对 + 语料人工复核。分歧词共 326 个, 此处收录已确认的高频项。
    #
    # 生效范围: 仅当 fugashi 分词结果的 surface 与表中键**完全相等**时替换读音。
    #   复合词 (如 舌先/耳穴/身体中) 的 surface 不等于单字键, 天然不受影响。
    #   → 不会出现 overrides「words」那种字符串替换破坏复合词的问题。
    _READING_FIX = {
        # 人称/敬称
        "兄": "あに", "様": "さま", "達": "たち", "姉": "あね", "君": "くん",
        "者": "しゃ", "他": "た", "方": "かた",
        # 身体
        "身体": "からだ", "体": "からだ", "声": "こえ", "音": "おと",
        "間": "あいだ", "時": "とき", "額": "ひたい",
        # 动词/形容词活用
        "続": "つづ", "出": "だ", "分": "ぶん", "注": "そそ", "着": "き",
        "犯": "おか", "搾": "しぼ", "捨": "す", "越": "ご", "願": "ねが",
        "動": "うご", "来": "こ", "汚": "よご", "拭": "ふ", "向": "む",
        "待": "ま", "垂": "た", "苦": "くる", "驚": "おどろ", "空": "あ",
        "吸い付": "すいつ", "片付": "かたづ", "美味": "おい",
        # 名词/接尾
        "何": "なん", "中": "ちゅー", "的": "てき", "愛": "あい", "生": "なま",
        "一": "いっ", "本": "ぽん", "発": "ぱつ", "聖": "せー", "奏": "かなで",
        "熱": "あつ", "明日": "あす", "祭": "さい", "国": "こく", "竿": "さお",
        "力": "りょく", "無": "む", "係": "かかり", "骨": "こつ", "年": "ねん",
        "日": "にち", "章": "しょー", "臭": "しゅー", "癒": "いや",
        # 复合/熟字
        "人妻": "ひとづま", "金玉": "きんたま", "中出": "なかだ",
        "十分": "じゅうぶん", "間近": "まぢか",
        # 单字名词: UniDic 在句中出现时会误取音读 (实测: 舌、→ゼツ; 耳→ジ)
        "舌": "した", "耳": "みみ", "鏡": "かがみ", "汁": "しる",
        "物": "もの", "風": "ふう", "鼻": "はな", "眉": "まゆ",
        "爪": "つめ", "肌": "はだ", "髪": "かみ", "首": "くび",
        "腕": "うで", "足": "あし", "手": "て", "目": "め",
        "顔": "かお", "指": "ゆび", "背": "せ", "腹": "なか",
        "胸": "むね", "腰": "こし", "尻": "しり", "喉": "のど",
        # 熟字训: UniDic 会误按音读拆读
        "数多": "あまた", "不足": "ふそく", "身近": "みぢか",
    }

    def _apply_reading_fix(self, surface: str, reading_hira: str) -> str:
        """按表修正已知的 UniDic 读音错误 (仅精确匹配 surface)"""
        fixed = self._READING_FIX.get(surface)
        return fixed if fixed else reading_hira

    # ── pykakasi 注音 (回退) ─────────────────────

    def _annotate_pykakasi(self, text: str) -> str:
        """逐字查字典注音 (回退方案)"""
        self._ensure_pykakasi()
        result = self._pykakasi.convert(text)
        output: "list[str]" = []

        for item in result:
            orig = item['orig']
            hira = item['hira']

            # 判是否包含汉字
            if not self._has_kanji(orig):
                output.append(orig)
                continue

            # pykakasi 也读不出读音 (如非日文汉字 绪/甞 等字典外字):
            # 不得产出 {X|} 空读音结构。下游 _parse_ja_tokens 会对空读音
            # raise ValueError, 使整行注音中断、该样本被静默丢弃。
            # 此时保持原样输出, 由下游按无读音字符跳过。
            if not hira:
                output.append(orig)
                continue

            # 送假名分离
            pref, body, suff = self._split_okurigana(orig, hira)
            if suff:
                output.append(f"{pref}{{{body}|{suff[0]}}}{suff[1]}" if len(suff) == 2 and body
                              else f"{pref}{{{body}|{hira}}}")
            elif body:
                output.append(f"{pref}{{{body}|{hira}}}")
            else:
                output.append(orig)

        return ''.join(output)

    # ── 工具函数 ────────────────────────────────

    @staticmethod
    def _has_kanji(text: str) -> bool:
        """判断是否包含日文汉字"""
        return RubyAnnotator._is_kanji(text)

    @staticmethod
    def _is_kana(ch: str) -> bool:
        o = ord(ch)
        return 0x3041 <= o <= 0x3096 or 0x30A1 <= o <= 0x30F6

    @staticmethod
    def _is_kanji(text: str) -> bool:
        return any(0x4E00 <= ord(c) <= 0x9FFF or 0x3400 <= ord(c) <= 0x4DBF for c in text)

    @staticmethod
    def _kata_to_hira(text: str) -> str:
        """片假名 → 平假名"""
        result: "list[str]" = []
        for c in text:
            code = ord(c)
            if 0x30A1 <= code <= 0x30F6:
                result.append(chr(code - 0x60))
            else:
                result.append(c)
        return ''.join(result)

    @staticmethod
    def _format_ruby(surface: str, reading: str) -> str:
        """
        格式化 {kanji|kana}，自动处理送假名分离
        例: ("食べる", "たべる") → "{食|た}べる"
        """
        if not surface or not reading:
            return surface
        if not RubyAnnotator._has_kanji(surface):
            return surface

        # 去掉头尾相同假名
        pfx_len = 0
        while (pfx_len < len(surface) and pfx_len < len(reading)
               and surface[pfx_len] == reading[pfx_len]
               and not RubyAnnotator._has_kanji(surface[pfx_len])):
            pfx_len += 1

        prefix = surface[:pfx_len]
        kanji_body = surface[pfx_len:]
        reading_body = reading[pfx_len:]

        sfx_len = 0
        while (sfx_len < len(kanji_body) and sfx_len < len(reading_body)
               and kanji_body[-(sfx_len + 1)] == reading_body[-(sfx_len + 1)]):
            sfx_len += 1

        if sfx_len > 0:
            core_kanji = kanji_body[:-sfx_len]
            core_reading = reading_body[:-sfx_len]
            suffix = kanji_body[-sfx_len:]
            if core_kanji:
                return f"{prefix}{{{core_kanji}|{core_reading}}}{suffix}"
            return surface

        if kanji_body:
            return f"{prefix}{{{kanji_body}|{reading_body}}}"
        return surface

    @staticmethod
    def _split_okurigana(orig: str, hira: str) -> "tuple[str, str, tuple[str, str] | None]":
        """pykakasi 用: 剥离头尾相同假名 → (prefix, kanji_body, (reading_body, suffix))"""
        pfx = 0
        while (pfx < len(orig) and pfx < len(hira)
               and orig[pfx] == hira[pfx]
               and not RubyAnnotator._has_kanji(orig[pfx])):
            pfx += 1
        prefix = orig[:pfx]
        o_body = orig[pfx:]
        h_body = hira[pfx:]

        sfx = 0
        while (sfx < len(o_body) and sfx < len(h_body)
               and o_body[-(sfx + 1)] == h_body[-(sfx + 1)]):
            sfx += 1

        if sfx > 0:
            return prefix, o_body[:-sfx], (h_body[:-sfx], o_body[-sfx:])
        return prefix, o_body, None

    # ── 文件处理 ────────────────────────────────

    def process_file(self, file_path: str) -> bool:
        """处理单个文件, 覆盖写入. 返回是否成功."""
        content = self._read_file(file_path)
        if content is None:
            return False

        lines = content.splitlines(keepends=True)
        out_lines: "list[str]" = []

        for line in lines:
            ending = ''
            if line.endswith('\r\n'):
                core = line[:-2]; ending = '\r\n'
            elif line.endswith('\n'):
                core = line[:-1]; ending = '\n'
            else:
                core = line

            out_lines.append(self.annotate(core) + ending)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(''.join(out_lines))

        print(f"  [OK] {os.path.basename(file_path)}")
        return True

    @staticmethod
    def _read_file(file_path: str) -> "str | None":
        """自动检测编码读取文件"""
        for enc in ['utf-8', 'utf-8-sig', 'shift_jis', 'cp932', 'gbk']:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    return f.read()
            except (UnicodeDecodeError, UnicodeError):
                continue
        print(f"  [跳过] 无法识别编码: {os.path.basename(file_path)}")
        return None


# ── 批量入口 ──────────────────────────────────────

def main() -> None:
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    converter = RubyAnnotator(verbose=True)
    current_dir = os.getcwd()
    txt_files = glob.glob(os.path.join(current_dir, "*.txt"))

    if not txt_files:
        print("当前目录下未找到 .txt 文件。")
        return

    engine_name = "fugashi + UniDic" if converter.has_fugashi else "pykakasi (请安装 fugashi + unidic-lite)"
    print(f"引擎: {engine_name}")
    print(f"检测到 {len(txt_files)} 个 .txt 文件\n")

    for fp in txt_files:
        converter.process_file(fp)

    print(f"\n完成! 共处理 {len(txt_files)} 个文件。")


if __name__ == "__main__":
    main()
