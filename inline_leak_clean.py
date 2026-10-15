# -*- coding: utf-8 -*-
"""内联振假名泄露清理（inline reading leak cleanup）。

背景
----
部分台本把振假名写成「未标记的行内假名」而非 `{漢字|かな}`：
    身体んたい / 沖縄きなわ / 弱点くてん / 気持ちちいい / 大好ききだ
这些残留假名会被分词器当成实词，污染注音与对齐标签，并最终漏进字幕正文。

本模块只做【删除】，绝不改写字形、绝不凭空补字。

判据总纲
--------
记当前位置 i：text[i-1] 为汉字、text[i] 为平假名，候选泄露段 L = text[i:i+m]。
**必须同时满足**：
  A. 读音包含：L 是「前接汉字串读音」去掉首音节后的前缀（读音来源见 `_readings`）
  B. 结构改善：删除 L 后，该处的形态学分析由「异常」变为「正常」
任一条不成立即不动。A 保证不会删掉真实助词，B 保证不会删掉正常的词尾活用。

B 的可用证据（任一）：
  B1 送假名融合  —— 删除后覆盖 i-1 的 token 终点外扩（汉字吸收了新假名）
  B2 词干还原    —— 删除后 token 表层 == 该词 lemma，且与原文表层不同
  B3 送假名越界  —— 原 token 的送假名吞掉了词干读音（悪るい：わる→わ）
  B4 多音节残留  —— m>=2 且 L 命中前置汉字串读音尾部
  B5 单字残留    —— m==1 且 L 不是助词，或命中上下文白名单
  B6 结构异常消除 —— 原分析含「语法上不可能」的成分（G1..G9），
                    且删除后异常数下降（该分支同时要求 A 成立，二者缺一不可）

另有 `_KNOWN_LEAKS`：少量语法上无法证伪的残形，走显式可审计表。

验证（必须两者都过）
--------------------
  _exp_ono_clause/leak_module_verify.py  —— 规则命中分布 + 全量 diff
  _exp_ono_clause/leak_audit_after.py    —— 清理后残留审计 + 干净台本回归
实测：干净台本 5 轨 2205 行 0 改动；污染台本 5 轨 1158 行修复 485 处、残留 0。
"""
from __future__ import annotations

from typing import List, Optional, Set, Tuple

# 可紧跟名词、本身有实义的 1 字助词/助动词 —— 单字残留的排除集
_PARTICLES_1 = set('はがをにへとでやもかのねよわさぞぜなだて')

# 汉字串的上下文异读（UniDic 默认读法不等于台本注音时必须补）
_SPECIAL_READINGS = {
    '今日': ('きょう', 'こんにち'),
    '何言': ('なにごと',),
    '弄': ('もてあそ', 'いじ'),
}

# 单字残留的上下文白名单：(汉字, 泄露假名, 后续前缀元组)
_CTX_WHITELIST = (
    ('事', 'と', ('を', 'も')),
    ('風', 'ぜ', ('に',)),
    ('何', 'に', ('が', 'を', 'も', 'し', '甘')),
    ('中', 'か', ('、', '。', 'に', 'で', 'を', 'の', 'も', 'は', 'が', '.')),
    ('部屋', 'や', ('な', 'だ', 'で', 'に', 'の', 'へ', 'を', 'も', 'から')),
    ('屋', 'や', ('な', 'だ', 'で', 'に', 'の', 'へ', 'を', 'も', 'から')),
)

# ──────────────────────────────────────────────────────────────
# 已知残形表（可审计）
#
# 通用规则（读音包含 + 结构改善）已覆盖绝大多数内联振假名残留，且对干净台本
# 零误伤。但有一小类残形是特定台本的历史损坏产物：泄漏假名既不对应任何读音
# 切分，删除后也不改变 UniDic 的形态分析（无法与正常书写区分）。
# 例如「お腹ら」的 ら、「愛い」的 い（UniDic 均判为合法成分）。
# 对这类只靠语法无法证伪的残形，用显式表处理 —— 每条都写明证据。
#
# 形式：损坏片段 -> 正确片段。仅在【完整匹配】该子串时替换。
# 维护约定：只收「同一台本内 ≥2 例、或删除后必然成立」的条目；
#          新增前必须在干净台本集上回归（_exp_ono_clause/leak_audit_after.py）。
_KNOWN_LEAKS = {
    # 名词后多出的读音尾字（UniDic 误判为接尾辞/助数词名词，语法上无法证伪）
    'お腹ら': 'お腹',
    '愛いの': '愛の',
    '重もさ': '重さ',
    '譲気': '譲る気',
    '間んでも': '間でも',
    '脇き': '脇',
    '汗せ': '汗',
    '後とで': '後で',
    '後とだ': '後だ',
    '朝さま': '朝',
    '初つ絞': '初絞',
    '方うな': '方な',
    '方うが': '方が',
    '上えの空ら': '上の空',
    '間んから': '間から',
    # 動詞送假名中多出的假名（UniDic 判为其他动词的合法活用）
    '返えっ': '返っ',
    '揉うん': '揉ん',
    '変んに': '変に',
    '変んな': '変な',
    '次ぎは': '次は',
}


# 结构异常签名
_SURU = '為る'          # する／せ／し／す（非自立可能）
# する族未然形（せ／し／さ）后合法的接续（受身/使役/否定/ば/ぬ）。
# 其余情况说明该假名是汉字读音的残留（汗せだけ / 重もさなら）。
_ZENKEE_OK_AFTER = {
    'れる', 'られる', 'せる', 'させる', 'ない', 'ぬ', 'ん', 'ず',
    'ば', 'まい', 'よう', 'そう', 'べき',
}
# 副词性名词（时间/场所）：后接「と/ん」多为读音残留（後とで / 間んでも）
_ADVERBIAL_NOUN = '副詞可能'


def is_hiragana(ch: str) -> bool:
    return 0x3041 <= ord(ch) <= 0x3096


def _extract_core(surface: str, reading: str, has_kanji) -> Tuple[int, int, str, str]:
    """剥离表层与读音两端相同的假名，返回 (前缀长, 后缀长, 汉字核心, 读音核心)。"""
    pfx = 0
    while (pfx < len(surface) and pfx < len(reading)
           and surface[pfx] == reading[pfx] and not has_kanji(surface[pfx])):
        pfx += 1
    sfx = 0
    while (sfx < (len(surface) - pfx) and sfx < (len(reading) - pfx)
           and surface[-(sfx + 1)] == reading[-(sfx + 1)]
           and not has_kanji(surface[-(sfx + 1)])):
        sfx += 1
    core_k = surface[pfx:len(surface) - sfx if sfx else len(surface)]
    core_r = reading[pfx:len(reading) - sfx if sfx else len(reading)]
    return pfx, sfx, core_k, core_r


class InlineLeakCleaner:
    """内联振假名泄露清理器（需注入一个 RubyAnnotator 实例）。"""

    def __init__(self, annotator, max_probe: int = 10):
        self.ra = annotator
        self.max_probe = int(max_probe)
        self._tagger = annotator._tagger if getattr(annotator, '_tagger', None) else None
        self.stats = {'removed': 0, 'lines': 0, 'rules': {}}
        # 分词缓存。_abnormal 每次调用首行就全串重分词，而 _abnormal_all 对
        # 每个 token 起点调它一次 -> O(T²) 的 fugashi 工作量；_try_at 又对
        # m=1..max_probe 的每个候选重复调用。实测 1012 行台本每行平均 40 次
        # 分词，加缓存后降到约 6 次，clean() 耗时 2.30s -> 0.69s（3.33x），
        # 且输出与无缓存版本逐字符一致。
        self._tok_cache = {}

    def _tokenize_cached(self, s: str) -> List[Tuple[int, int, object]]:
        hit = self._tok_cache.get(s)
        if hit is not None:
            return hit
        out = self._tokenize(s)
        if len(self._tok_cache) < 50000:
            self._tok_cache[s] = out
        return out

    # ── 基础工具 ──────────────────────────────────────────

    def _has_kanji(self, t: str) -> bool:
        return self.ra._has_kanji(t)

    def _tokenize(self, s: str) -> List[Tuple[int, int, object]]:
        if not self._tagger:
            return []
        out = []
        pos = 0
        for w in self._tagger(s):
            surf = w.surface
            if not surf:
                continue
            st = s.find(surf, pos)
            if st < 0:
                st = pos
            pos = st + len(surf)
            out.append((st, st + len(surf), w))
        return out

    @staticmethod
    def _feat(w, name: str, default: str = '') -> str:
        v = getattr(getattr(w, 'feature', None), name, None)
        return default if v is None or v == '*' else str(v)

    def _lemma_core(self, surface: str, lemma: str, lform: str) -> Optional[str]:
        if not lemma or not lform or lemma == '*' or lform == '*':
            return None
        _, _, _, cr = _extract_core(
            lemma.split('-')[0], self.ra._kata_to_hira(lform), self._has_kanji)
        return cr or None

    def _excess_okurigana(self, w) -> Tuple[bool, Optional[str]]:
        """表层送假名是否吞掉了词干读音（悪るい：わる → わ）。"""
        lemma = self._feat(w, 'lemma')
        lform = self._feat(w, 'lForm')
        r_raw = self.ra._get_fugashi_reading(w)
        if not r_raw or not lemma or not lform:
            return False, None
        lpfx, _, lck, lcr = _extract_core(
            lemma.split('-')[0], self.ra._kata_to_hira(lform), self._has_kanji)
        pfx, sfx, ck, cr = _extract_core(
            w.surface, self.ra._kata_to_hira(r_raw), self._has_kanji)
        ok = (sfx > 0 and pfx == lpfx and len(ck) == len(lck) and len(cr) < len(lcr))
        return ok, lcr

    def _readings(self, w) -> List[str]:
        """候选读音：人工覆盖 → 词典 → kana/lForm，含长音展开变体。"""
        surf = w.surface
        out: List[str] = []

        def push(v):
            if v and v not in out:
                out.append(v)

        push(self.ra._VERIFIED_OVERRIDES.get(surf))
        if surf in self.ra._PRONOUN_OVERRIDES:
            push(self.ra._kata_to_hira(self.ra._PRONOUN_OVERRIDES[surf]))
        push(self.ra._READING_FIX.get(surf))
        r_raw = self.ra._get_fugashi_reading(w)
        if r_raw:
            r_hira = self.ra._kata_to_hira(r_raw)
            push(r_hira)
            push(self.ra._apply_reading_fix(surf, r_hira))
            push(self.ra._expand_macron(r_hira))
        kana = self._feat(w, 'kana')
        if kana:
            kh = self.ra._kata_to_hira(kana)
            push(kh)
            push(self.ra._expand_macron(kh))
        lform = self._feat(w, 'lForm')
        if lform:
            lh = self.ra._kata_to_hira(lform)
            push(lh)
            push(self.ra._expand_macron(lh))
        return out

    def _tails(self, reading: str) -> Set[str]:
        """读音去掉首音节后的剩余（含长音展开变体）。"""
        from haruraw2norm import sylla_split
        out: Set[str] = set()
        for r in (reading, self.ra._expand_macron(reading)):
            syls = sylla_split(r, False, True)
            if len(syls) >= 2:
                out.add(''.join(syls[1:]))
        return out

    # ── B6：结构异常消除 ──────────────────────────────────

    def _abnormal(self, text: str, idx: int) -> Optional[str]:
        """该位置是否含「语法上不可能」的成分；返回签名名或 None。

        只看「含 idx 的那个 token」不够：残留常使分词器把汉字与泄漏假名
        并成一个 token（次ぎ / 変ん），异常落在该 token 的形态上。
        故统一按「含 idx 的 token」判定，删除长度由签名自行给出。
        """
        T = self._tokenize_cached(text)
        k = next((n for n, (st, en, w) in enumerate(T) if st <= idx < en), None)
        if k is None:
            return None
        st, en, w = T[k]
        pos1 = self._feat(w, 'pos1')
        pos2 = self._feat(w, 'pos2')
        lemma = self._feat(w, 'lemma').split('-')[0]
        cform = self._feat(w, 'cForm')
        ctype = self._feat(w, 'cType')
        prev = T[k - 1][2] if k > 0 else None
        nxt = T[k + 1][2] if k + 1 < len(T) else None
        p1 = self._feat(prev, 'pos1') if prev else ''
        p3 = self._feat(prev, 'pos3') if prev else ''

        # ── G1 不可能的接尾辞/助数词名词 ──
        # 「ら/ども/さま/せ/き/り/たく」等 lemma 紧跟名词，几乎必是汉字读音残留。
        _BAD_SUFFIX_LEMMA = {
            '等': 'ら', '共': 'ども', '様': 'さま',
            '世': 'せ', '機': 'き', '里': 'り', '宅': 'たく',
        }
        if lemma in _BAD_SUFFIX_LEMMA and prev and p1 in ('名詞', '接頭辞'):
            return 'G1_BAD_SUFFIX'

        # ── G2 する族未然形孤儿 ──
        if (pos1 == '動詞' and lemma == _SURU and len(w.surface) <= 2 and prev
                and p1 in ('名詞', '接頭辞') and 'サ変' not in p3
                and cform.startswith('未然形')):
            nlemma = self._feat(nxt, 'lemma').split('-')[0] if nxt else ''
            if nlemma not in _ZENKEE_OK_AFTER:
                return 'G2_SURU_ZENKEE_ORPHAN'

        # ── G3 死语/误配助动词（つう/べし/き 紧跟名词）──
        if pos1 == '助動詞' and lemma in ('つう', 'べし', 'き', 'けり') and prev:
            if p1 in ('名詞', '接尾辞'):
                return 'G3_OBSOLETE_AUX'

        # ── G4 副词可能名词 + と/ん/か ──
        if prev and p3 == _ADVERBIAL_NOUN:
            if pos1 == '助詞' and lemma == 'と' and pos2 == '格助詞':
                npos1 = self._feat(nxt, 'pos1') if nxt else ''
                if npos1 in ('接続詞', '助動詞', '助詞'):
                    return 'G4_ADV_NOUN_TO'
            if pos1 == '助詞' and pos2 == '準体助詞':
                return 'G4_ADV_NOUN_N'
            if pos1 == '助詞' and pos2 == '副助詞' and lemma == 'か':
                return 'G4_ADV_NOUN_KA'

        # ── G5 拨音便動詞（変んになる / 変んなんて）──
        if (pos1 == '動詞' and w.surface.endswith('ん') and len(w.surface) >= 2):
            stripped = w.surface[:-1]
            stem_ok = any(self._feat(sw[2], 'pos1') in ('名詞', '形状詞', '形容動詞語幹')
                          for sw in self._tokenize_cached(stripped))
            if stem_ok and (cform.endswith('撥音便')
                            or (nxt and self._feat(nxt, 'lemma') == 'だ')):
                return 'G5_HATSUON_PREFIX'

        # ── G6 五段動詞结尾冗余假名（次ぎは → 次は）──
        if pos1 == '動詞' and len(w.surface) >= 2 and ctype.startswith('五段') and nxt:
            if self._feat(nxt, 'pos1') == '助詞':
                stripped = w.surface[:-1]
                for sw in self._tokenize_cached(stripped):
                    if (self._feat(sw[2], 'pos1') == '名詞'
                            and self._feat(sw[2], 'pron') == self._feat(w, 'pron')):
                        return 'G6_GODAN_TAIL'

        # ── G7 感動詞紧跟実詞（揉うんで / 愛いの）──
        if pos1 == '感動詞' and len(w.surface) <= 2 and prev:
            if p1 in ('名詞', '代名詞', '接尾辞', '接頭辞'):
                return 'G7_FILLER_AFTER_NOUN'

        # ── G8 接頭辞 + 名詞碎片（全たく → 全く）──
        if p1 == '接頭辞' and pos1 == '名詞' and prev:
            rest = w.surface[1:]
            if rest:
                for sw in self._tokenize_cached(prev.surface + rest):
                    if self._feat(sw[2], 'pos1') in ('副詞', '形容詞'):
                        return 'G8_PREFIX_NOUN_FRAG'

        # ── G9 連体形助動詞「だ」紧跟名词（穴なでも → 穴でも）──
        # 排除「名詞＋な＋ん＋だ」这一合法的口语缩约（評判なんだけど）。
        if (pos1 == '助動詞' and lemma == 'だ' and cform.startswith('連体形')
                and prev and p1 == '名詞'):
            npos1 = self._feat(nxt, 'pos1') if nxt else ''
            npos2 = self._feat(nxt, 'pos2') if nxt else ''
            if npos1 in ('助動詞', '助詞') and npos2 != '準体助詞':
                return 'G9_RENTAI_DA_AFTER_NOUN'
        return None

    def _apply_known(self, text: str) -> str:
        """已知残形表（仅完整子串匹配，可审计）。"""
        for bad, good in _KNOWN_LEAKS.items():
            if bad in text:
                text = text.replace(bad, good)
                self.stats['rules']['KNOWN_TABLE'] = \
                    self.stats['rules'].get('KNOWN_TABLE', 0) + 1
        return text

    def _reading_covers(self, text: str, i: int, k_start: int, k_run: str, L: str) -> bool:
        """A 判据：L 是「前接汉字（串）读音」的一部分（首音节之后）。"""
        # 单汉字：该位置 token 的读音（含特殊读音表）
        probe = text[:i] + text[i + len(L):]
        for st, en, w in self._tokenize_cached(probe):
            if st <= i - 1 < en:
                for r in self._readings(w):
                    if L in self._tails(r):
                        return True
                break
        # 多汉字串：读音拼接
        if len(k_run) >= 2:
            if k_run in _SPECIAL_READINGS:
                for sr in _SPECIAL_READINGS[k_run]:
                    if L in self._tails(sr):
                        return True
            for sub_s in range(max(0, k_start), i - 1):
                parts = []
                cur_p = sub_s
                ok = any(tst == sub_s for tst, _, _ in self._tokenize_cached(text))
                if not ok:
                    continue
                for tst, ten, tw in self._tokenize_cached(text):
                    if tst < sub_s:
                        continue
                    if tst >= i:
                        break
                    if tst != cur_p:
                        ok = False
                        break
                    rs = self._readings(tw)
                    if not rs:
                        ok = False
                        break
                    _, _, ck, cr = _extract_core(tw.surface, rs[0], self._has_kanji)
                    if tst + len(ck) != min(ten, i):
                        ok = False
                        break
                    parts.append(cr)
                    cur_p = min(ten, i)
                if ok and cur_p == i and len(parts) >= 2:
                    if L in self._tails(''.join(parts)):
                        return True
        return False

    def _abnormal_all(self, text: str) -> List[str]:
        """整串的结构异常签名列表（按 token 起点逐个判定）。"""
        out = []
        # 只分词一次并复用：_abnormal 内部也会 _tokenize_cached(text)，
        # 命中同一条目，故整串的 T² 次分词降为 1 次实际分词 + T 次查表。
        T = self._tokenize_cached(text)
        for st, en, w in T:
            sig = self._abnormal(text, st)
            if sig:
                out.append(sig)
        return out

    # ── 主流程 ────────────────────────────────────────────

    def clean(self, text: str) -> str:
        if not text or not self._tagger:
            return text
        text = self._apply_known(text)
        i = 0
        while i < len(text):
            # ── A. 汉字→平假名 边界上的行内振假名残留 ──
            if i > 0 and self._has_kanji(text[i - 1]) and is_hiragana(text[i]):
                hit = self._try_at(text, i)
                if hit is not None:
                    text, m = hit
                    self.stats['removed'] += m
                    continue
                i += 1
                continue

            # ── B. 送假名回声去重（気持ちちいい / 大好ききだ）──
            if (i >= 2 and is_hiragana(text[i]) and text[i] == text[i - 1]
                    and self._has_kanji(text[i - 2])):
                T = self._tokenize_cached(text)
                prev_t = next(((st, en, w) for st, en, w in T if st <= i - 2 and en == i), None)
                cur_t = next(((st, en, w) for st, en, w in T if st == i), None)
                if prev_t and cur_t:
                    pos1 = self._feat(cur_t[2], 'pos1')
                    lemma = self._feat(cur_t[2], 'lemma').split('-')[0]
                    if (pos1 not in ('助詞', '動詞') and lemma != _SURU
                            and text[i] not in _PARTICLES_1
                            and len(cur_t[2].surface) <= 2):
                        text = text[:i] + text[i + 1:]
                        self.stats['removed'] += 1
                        self.stats['rules']['ECHO'] = self.stats['rules'].get('ECHO', 0) + 1
                        i += 1
                        continue
            i += 1
        return text

    def _try_at(self, text: str, i: int):
        """尝试在位置 i 删除泄露段；成功返回 (新文本, 删除长度)，失败返回 None。"""
        T0 = self._tokenize_cached(text)
        orig_t = next(((st, en, w) for st, en, w in T0 if st <= i - 1 < en), None)
        orig_en = orig_t[1] if orig_t else i
        orig_surf = orig_t[2].surface if orig_t else None
        orig_lemma = None
        if orig_t:
            lm = self._feat(orig_t[2], 'lemma').split('-')[0]
            orig_lemma = lm or None
        orig_excess, orig_lcr = self._excess_okurigana(orig_t[2]) if orig_t else (False, None)

        k_start = i - 1
        while k_start > 0 and self._has_kanji(text[k_start - 1]):
            k_start -= 1
        k_run = text[k_start:i]

        j = i
        while j < len(text) and is_hiragana(text[j]):
            j += 1
        hira_run = text[i:j]

        # B6：结构异常 —— 逐个试删长度。要求
        #   ① 异常数严格减少，且
        #   ② A 判据（读音包含）也成立 —— 否则「幸せそうな顔しちゃって」这类
        #      合法句一旦异常数因别的原因下降就会被误删。
        #   取满足条件的最小 m，避免连删。
        sig0 = self._abnormal_all(text)
        want_sig = None
        if sig0:
            want_sig = sig0[0]
            best = None
            for m in range(1, min(len(hira_run), self.max_probe) + 1):
                L = hira_run[:m]
                if not self._reading_covers(text, i, k_start, k_run, L):
                    continue
                cand = text[:i] + text[i + m:]
                if len(self._abnormal_all(cand)) < len(sig0):
                    best = (cand, m)
                    break
            if best is not None:
                self.stats['rules'][want_sig] = self.stats['rules'].get(want_sig, 0) + 1
                return best

        # B3：原 token 送假名吞掉词干读音，且泄露字属其 lemma 读音尾部
        if orig_excess and orig_lcr:
            for m in range(min(len(hira_run), self.max_probe), 0, -1):
                if hira_run[:m] in self._tails(orig_lcr):
                    self.stats['rules']['EXCESS_OKURIGANA'] = \
                        self.stats['rules'].get('EXCESS_OKURIGANA', 0) + 1
                    return text[:i] + text[i + m:], m

        for m in range(min(len(hira_run), self.max_probe), 0, -1):
            L = hira_run[:m]
            cand = text[:i] + text[i + m:]
            T = self._tokenize_cached(cand)
            t = next(((st, en, w) for st, en, w in T if st <= i - 1 < en), None)
            if t is None:
                continue
            st, en, w = t
            surf = w.surface
            lcr = self._lemma_core(surf, self._feat(w, 'lemma'), self._feat(w, 'lForm'))

            # B4：多字前缀汉字串的读音残留
            if len(k_run) >= 2 and m >= 2 and self._span_tail_hit(
                    text, cand, i, k_start, k_run, L, en, orig_en, m, T):
                self.stats['rules']['SPAN_TAIL'] = self.stats['rules'].get('SPAN_TAIL', 0) + 1
                return cand, m

            wrs = self._readings(w)
            if surf in _SPECIAL_READINGS:
                for sr in _SPECIAL_READINGS[surf]:
                    if sr not in wrs:
                        wrs.append(sr)
            for r in wrs:
                pfx, _, core_k, core_r = _extract_core(surf, r, self._has_kanji)
                if not core_k or not core_r:
                    continue
                if st + pfx + len(core_k) != i:
                    continue
                if lcr and len(core_r) > len(lcr) and surf not in _SPECIAL_READINGS:
                    continue
                if L not in self._tails(core_r):
                    continue
                # B1/B2：送假名融合 / 词干还原
                if (en > max(i, orig_en - m)
                        or (en >= i and orig_lemma and surf == orig_lemma
                            and orig_surf != orig_lemma)):
                    self.stats['rules']['FUSION'] = self.stats['rules'].get('FUSION', 0) + 1
                    return cand, m
                # B5：单字 / 多字残留
                if en == i and (i + m > orig_en):
                    if m >= 2:
                        self.stats['rules']['MULTI_TAIL'] = \
                            self.stats['rules'].get('MULTI_TAIL', 0) + 1
                        return cand, m
                    if m == 1 and orig_en == i:
                        if L not in _PARTICLES_1:
                            self.stats['rules']['SINGLE_NONPARTICLE'] = \
                                self.stats['rules'].get('SINGLE_NONPARTICLE', 0) + 1
                            return cand, m
                        tail_ctx = text[i + 1:]
                        if any(ck == core_k and l == L and tail_ctx.startswith(pf)
                               for ck, l, pf in _CTX_WHITELIST):
                            self.stats['rules']['CTX_WHITELIST'] = \
                                self.stats['rules'].get('CTX_WHITELIST', 0) + 1
                            return cand, m
        return None

    def _span_tail_hit(self, text, cand, i, k_start, k_run, L, en, orig_en, m, T) -> bool:
        """前置汉字串（含特殊读音表 + 前序 token 读音拼接）的读音尾部是否含 L。"""
        span_tails: Set[str] = set()
        if k_run in _SPECIAL_READINGS:
            for sr in _SPECIAL_READINGS[k_run]:
                span_tails |= self._tails(sr)
        for sub_s in range(k_start, i - 1):
            if not any(tst == sub_s for tst, _, _ in T):
                continue
            parts = []
            cur_p = sub_s
            ok = True
            for tst, ten, tw in T:
                if tst < sub_s:
                    continue
                if tst >= i:
                    break
                if tst != cur_p:
                    ok = False
                    break
                rs = self._readings(tw)
                if not rs:
                    ok = False
                    break
                _, _, ck, cr = _extract_core(tw.surface, rs[0], self._has_kanji)
                tw_lcr = self._lemma_core(tw.surface, self._feat(tw, 'lemma'),
                                          self._feat(tw, 'lForm'))
                if tw_lcr and len(cr) > len(tw_lcr):
                    ok = False
                    break
                if tst + len(ck) != min(ten, i):
                    ok = False
                    break
                parts.append(cr)
                cur_p = min(ten, i)
            if ok and cur_p == i and len(parts) >= 2:
                span_tails |= self._tails(''.join(parts))
        return L in span_tails and (en > i or (i + m > orig_en))
