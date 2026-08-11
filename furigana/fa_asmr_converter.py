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
import unicodedata
from collections import defaultdict, deque


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


class RubyAnnotator:
    """
    日文汉字注音器 v2
    ────────────────
    主引擎: fugashi + UniDic (词级分词 + 读音)
    回退:   pykakasi (逐字)
    预清洗: NFKC 规范化 + 浊音假名修正 + 波浪号转换
    已验证修正: 极小白名单, 仅覆盖人工确认过的 UniDic 字典数据错 (非异读/非正则)
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
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

        return text.strip()

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
        """〜 (U+301C / U+FF5E) → ー (假名后) 或删除 (孤立)"""
        out: "list[str]" = []
        for i, ch in enumerate(text):
            if ch == '\u301C' or ch == '\uFF5E':
                if (i > 0 and out and out[-1] != '\u30FC'
                        and RubyAnnotator._is_kana(text[i - 1])):
                    out.append('\u30FC')  # ー
                # 孤立 〜 直接丢弃
            else:
                out.append(ch)
        return ''.join(out)

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

    # ── 已验证修正 (人工确认过的 UniDic 字典数据错, 极小白名单) ──
    # 注意: 这不是异读词词典, 也不是正则, 而是引擎确实读错的确定项。
    # 异读词由 yomikata 模型层处理, 此处只放引擎的硬性数据错。
    # 按整词/词组级别覆盖, 避免误伤同字异读 (如 耳: みみ/じ 并存)。
    _VERIFIED_OVERRIDES = {
        # surface -> 正确读音 (UniDic 把整词错配成别的意思时, 按整词覆盖)
        "耳穴": "みみあな",          # UniDic 错配成「蚯蚓(ミミズ)」
        "搾": "しぼ",                # UniDic 把 搾 错读成 しめ(混淆 絞), 正确为 しぼ(搾る=しぼる)
        "雑魚": "ざこ",              # ASMR 语境均为「弱小者/小角色」=ざこ; UniDic 给 じゃこ(小白鱼) 几乎不出现
        "騎乗位": "きじょうい",      # UniDic 把 位 在 い/くらい 间摇摆(句尾/後接を・が时误读くらい); 騎乗位固定=骑乘位, 恒读 きじょうい
        "迸光": "ほうこう",          # 生造词(招式名), UniDic/yomikata 均无此条目返回 None; 迸(ホウ)+光(コウ) 音读
        "金玉": "きんたま",          # ASMR 语境为俗語「睾丸」=きんたま; UniDic 按文语「金银财宝」错读 きんぎょく
    }
    _VERIFIED_PHRASE_OVERRIDES = {
        # 字面词组 -> 预注音 (解决 耳 在 じ/みみ 并存, 不能单字覆盖的情况)
        "耳舐め": "{耳|みみ}{舐|な}め",   # UniDic 把 耳 当接頭辞(ジ) 误读, 应为 みみなめ
    }

    # ── 代词默认读法覆盖 (词级, 安全不误伤复合词) ──
    # 仅当 fugashi 把孤立汉字切成独立代词 token 时覆盖;
    # 汉字连续出现时 fugashi 会先切成复合词(如 私生活→整词, surface='私生活'≠'私'), 不会被命中。
    # "默认 watashi" 的发音先验: 自然口语绝大多数第一人称说 わたし, 仅正式角色说 わたくし。
    _PRONOUN_OVERRIDES = {
        "私": "ワタシ",   # UniDic 默认 ワタクシ; 仅お嬢様/メイド等正式角色才说 わたくし
    }

    @classmethod
    def _apply_verified_overrides(cls, text: str) -> str:
        """在注音前, 把人工确认过的 UniDic 字典数据错整体预注音, 由 _annotate_mixed 原样保留。"""
        for surf, read in cls._VERIFIED_OVERRIDES.items():
            if surf in text:
                text = text.replace(surf, "{%s|%s}" % (surf, read))
        for phrase, ruby in cls._VERIFIED_PHRASE_OVERRIDES.items():
            if phrase in text:
                text = text.replace(phrase, ruby)
        return text

    # ── 注音核心 ────────────────────────────────

    def annotate(self, text: str) -> str:
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

        # 纯假名/英文/标点行无需注音 (normalize 已完成清洗: 部首/NFKC/浊点/〜/
        # 全角/超长促音), 否则残留字符会产出 MMS_FA vocab 外非法 token。
        if not self._needs_ruby(text):
            return text

        # 已验证修正: 覆盖人工确认过的 UniDic 字典数据错 (整词/词组级)
        text = self._apply_verified_overrides(text)

        # 按已有 ruby 标记分段, 只对非 ruby 片段注音
        result = self._annotate_mixed(text)
        # yomikata 异读词消歧: 仅覆盖异读词的读音, 不影响其他词
        return self._apply_yomikata(text, result)

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
        yk = self._get_yomikata()
        if yk is None:
            return ruby
        hetero = self._get_heteronyms()
        plain = self._strip_ruby(text)
        # 无任意异读词则跳过模型调用
        if not any(w in plain for w in hetero):
            return ruby
        try:
            yomi = yk.furigana(plain)
        except Exception:
            return ruby
        pairs = self._extract_yomi_pairs(yomi)
        if not pairs:
            return ruby
        return self._merge_yomi(ruby, pairs)

    def _get_yomikata(self):
        """懒加载 yomikata dBert (GPU 可用时自动走 CUDA)。失败则标记 None 并跳过。"""
        if self._yomikata_loaded:
            return self._yomikata
        self._yomikata_loaded = True
        try:
            from yomikata.dbert import dBert
            self._yomikata = dBert()
        except Exception as e:
            if self.verbose:
                print("[RubyAnnotator] yomikata 不可用, 跳过异读层: %s" % e)
            self._yomikata = None
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
            except Exception:
                self._heteronyms = set()
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

    @staticmethod
    def _merge_yomi(ruby: str, pairs):
        """按 surface 把 yomikata 的读音覆盖进 UniDic 的 {surface|...} 片段。
        用 per-surface 队列, 同词多次出现依次消费, 未在 ruby 中出现则忽略 (不污染)。"""
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
                    surf = inner.split('|')[0]
                    dq = q.get(surf)
                    if dq:
                        out.append('{%s|%s}' % (surf, dq.popleft()))
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
        try:
            feats = str(word.feature).split(',')
            for idx in (9, 7, 6):
                if len(feats) > idx and feats[idx] and feats[idx] != '*':
                    val = feats[idx].split('=')[-1].strip("'\"")
                    return val
        except Exception:
            pass

        return None

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
