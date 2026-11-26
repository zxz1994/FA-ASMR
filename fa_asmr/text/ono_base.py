# -*- coding: utf-8 -*-
"""易错拟声判定：发声词干、音素组合、字形变换与黑名单。

只区分易错与非易错，不将所有拟声词都视作易错。
亲吻、舔耳、含混、气息与笑声词干允许口腔音续接；
续接音节只在这类词干之后使用，不作为任意单词的起手词根。
吞咽、咳嗽等词干仍可重复或与其他完整词干拼接。

片假名、小写元音、延长/促音/浊点及重复由变换处理，
保留原有鼻音起首、促音起首和短气息音的宽松判据。
黑名单优先拦截实词与明确排除的动作/状态拟声。
"""

import re
import sys

# ══ ① 单音素 ══
_SINGLE = frozenset("""
    ん っ ッ ー ～ ゛ ゜ ﾞ ﾟ
    ぁ ぃ ぅ ぇ ぉ ゃ ゅ ょ
    あ い う え お
""".split())

# ══ ② 词干 ══
_COMBO = frozenset("""
    あが あぐ あっく あは あひ あへ あふ あむ うぐ うふ
    うぷ うぷぷ うへ うほ えへ おご おふ おへ おほ がぼ きゃ きゃは ぎゃ
    きゅ くく くす くちゃ くちゅ くふ くぷ くぽ くん ぐお ぐちゃ ぐちゅ ぐぷ ぐっぽ ぐぼ ぐぽ
    けほ げふ げぷ げほ こほん こぽ こぽっ ごきゅ ごく ごっくん ごぽ
    ごぽっ じゅ じゅぱ じゅぶ じゅぷ じゅぼ じゅぽ じゅる すぅ
    すうすう すじゅ すふ すん ずず ずぞ ぞりゅ ちゅ ちゅぐ ちゅぱ ちゅぴ ちゅぶ ちゅぷ
    ちゅぼ ちゅぽ ちゅる ぢゅ ぢゅぐ ぢゅぶ ぢゅぷ ぢゅぼ ぢゅぽ ぢゅる
    どちゅ れちゅ れぢゅ でへ
    にゅ にゅちゅ にゅぷ にゅぽ にゅる にゅるる ぬちゃ ぬちゅ ぬりゅ
    ねちゅ ねぢゅ ねぶ ねりゅ ねれ ねろ はぁ
    はあはあ はひ はふ はぷ はへ はむ ひう ひゃ ひゅ ぴちゅ ふぁ ふぃ ふぅ
    ふうふう ふぇ ふぉ ふちゅ ふは ふぶ ふふ ふふぅ ふーふー ふわぁ ぶちゅ ぶぱ ぶぽ ぶっぽ ぷちゅ ぷは べろ
    へへ ぺちゃ ぺろ むぐ むちゅ むにゅ むふ もぐ れう
    れぅ れぇ れぇろ れら れり れりゅ れる れれ れろ
    れろぉ んき んぎ んく んぐ んじ んじゅ んちゅ んぢ
    んは んば んぱ んび んぴ んふ んぶ んぷ んべ んほ
    んみ んむ んり んれ んれろ ゲェ
""".split())

# 片假名对称
_H2K = {chr(_o): chr(_o + 0x60) for _o in range(0x3041, 0x3097)}
_K2H = {v: k for k, v in _H2K.items()}
_COMBO = _COMBO | frozenset(
    "".join(_H2K.get(_c, _c) for _c in _b) for _b in _COMBO)



# ══ ③ 变换 ══
_VEQ = {"ぁ": "あ", "ぃ": "い", "ぅ": "う", "ぇ": "え", "ぉ": "お"}
_SMALL = frozenset("ぁぃぅぇぉ")
_TRANSPARENT = frozenset("っッー～〜~゛゜ﾞﾟ\u3099\u309a")
_TAIL_EXT = frozenset("っッー～〜~゛ﾞ゜ﾟ\u3099\u309aんンあいうえおぁぃぅぇぉァィゥェォ")
# 续接音节仅在口腔词干之后使用，不参与独立起手匹配。
_MOUTH_TAIL = tuple(sorted(
    set("むろるれぷぽぶぼぱこくふあいうえおんゃゅょ")
    | {c + v for c in "ぎぢびぴみり" for v in "ゃゅょ"},
    key=len, reverse=True,
))
# 有清晰词形的吞咽/清嗓音、尖叫及易构成实词的2字词干，只接记号、元音或完整词干，不开启口腔音自由续接。
_CLOSED_STEMS = frozenset("""
    ごきゅ ごく ごっくん ごぽ ごぽっ こぽ こぽっ こほん
    けほ げふ げぷ げほ ゲェ ぎゃ
    あが あぐ あは あひ あふ あむ おご おふ がぼ くく くす くふ くん ぐお
    すふ すん はひ はへ はむ ひう ふわぁ へへ もぐ んじ
""".split())
_CONS = frozenset(
    "かきくけこがぎぐげごさしすせそざじずぜぞたちつてとだぢづでど"
    "なにぬねのはひふへほばびぶべぼぱぴぷぺぽまみむめもやゆよらりるれろわ")
# 普通单辅音+促音/延长/小元音仅限真实气息/呻吟/喉音辅音，
# 避免把 やっ/もっ/まっ/てっ/でっ/なっ/なぁ/にっ/さっ/めっ/よぉ/のぉ/てぇ 等实词截断与助词拖音误判为拟声；
# 若串内含独立浊点 ゛/ﾞ（如 もぉ゛ッ / ぬぉ゛ッ / のぉ゛ッ），仍允许全量 _CONS 匹配。
_BREATH_CONS = frozenset("はひふへほばびぶべぼぱぴぷぺぽくぐすずむげごれわ")
_VOW = frozenset("あいうえおぁぃぅぇぉ")
_CONS_VOWEL = {kana: vowel for vowel, row in (
    ("あ", "あかがさざただなはばぱまやゃらわ"),
    ("い", "いきぎしじちぢにひびぴみり"),
    ("う", "うくぐすずつづぬふぶぷむゆゅるゔ"),
    ("え", "えけげせぜてでねへべぺめれ"),
    ("お", "おこごそぞとどのほぼぽもよょろを"),
) for kana in row}
_PURE_MARKS = frozenset("っー～〜~゛゜ﾞﾟ\u3099\u309a")

# 音节形状本身不证明实际发声方式；续接只由已识别的口腔词干开启。

_KANA_RE = re.compile(r"^[\u3041-\u309F\u30A0-\u30FFー～〜~゛ﾞ゜ﾟ\u3099\u309a]+$")

# 黑名单不可用时的降级告警只打一次（见 is_ono）
_BLOCK_IMPORT_WARNED = False


# 同人台本常用弯引号 “ / ” 紧跟假名后替代低沉浊点 ゛（如 お“っ / お”っ / お“ぉ）
_PSEUDO_DAKUTEN_RE = re.compile(r"(?<=[\u3041-\u309F\u30A0-\u30FF])[“”]")


def _normalize_pseudo_dakuten(s: str) -> str:
    return _PSEUDO_DAKUTEN_RE.sub("゛", s) if ("“" in s or "”" in s) else s


def _norm(s: str) -> str:
    """R1 片假名→平假名 → R2 小写→大写（词尾小写保留作尾记号）→ R4 重复压缩。"""
    s = _normalize_pseudo_dakuten(s)
    t = "".join(_K2H.get(c, c) for c in s)
    out = []
    for i, c in enumerate(t):
        if c in _SMALL and i == len(t) - 1:
            out.append(c)          # 词尾小写保留（くぅ 的 ぅ）
        else:
            out.append(_VEQ.get(c, c))
    t = "".join(out)
    t = re.sub(r"(.)\1+", r"\1", t)
    t = re.sub(r"(..)\1+", r"\1", t)
    return t


def _strip_marks(s: str) -> str:
    """去掉所有透记号（R3）。"""
    return "".join(c for c in s if c not in _TRANSPARENT)


# 剔除经重复压缩后缩成单字的伪词干（如 れれ→れ、ふふ→ふ），防止裸单字 れ/ふ 误判为拟声
_COMBO_NORM = frozenset(nw for w in _COMBO for nw in (_norm(w),) if len(nw) >= 2)
_COMBO_UNMARKED = frozenset(uw for w in _COMBO for uw in (_strip_marks(_norm(w)),) if len(uw) >= 2)


def _segment_form(s: str) -> str:
    """统一字形并去透明记号，但保留词干音节数，避免把ふふ缩成裸ふ。"""
    s = _normalize_pseudo_dakuten(s)
    hira = "".join(_K2H.get(c, c) for c in s)
    return "".join(_VEQ.get(c, c) for c in _strip_marks(hira))


_SEG_STEMS = tuple(sorted({_segment_form(w) for w in _COMBO}, key=len, reverse=True))
_CLOSED_STEMS = frozenset(_segment_form(w) for w in _CLOSED_STEMS)


def _is_cons_tail(s: str) -> bool:
    """呼气/呻吟的最简形式（不做叠字压缩，避免把 ままー/ぱぱっ/みみっ 误压成单辅音）。

    长度 2：
      ① 元音重复       ああ うう おお（いい/ええ 是实词，不收）
      ② 辅音 + 促音/长音/波浪/小写元音  はっ ふー くぅ ほぉ わっ
      —— 不含「辅音 + 大写元音」（ここ / もう / いい 是实词，不收）
    长度 >= 3：
      单辅音 + 纯延长记号 / 小写元音 / 本行元音（はぁぁ～ / ふぅぅ / ほぉぅぅっ / ごおっ / ほぉぉぉぉぉぉぉっ），
      严禁接异行大写元音或裸「ん」（避免误杀 ふあん / におい / しあい / ていう / まおう / わいん）。
      总长度上限统一由 is_ono 顶层（<= 60）管控，不在本函数设额外长度断层。
    """
    if not s:
        return False
    h = "".join(_K2H.get(c, c) for c in s)
    has_standalone_dakuten = any(c in "゛ﾞ\u3099" for c in h)
    valid_cons = _CONS if has_standalone_dakuten else _BREATH_CONS
    if len(h) == 2:
        # 元音重复（ああ うう おお）—— 但 いい/ええ 是实词
        if h[0] in "あうお" and h[0] == h[1]:
            return True
        # 气息辅音 + 促音/长音/波浪/小写元音（はっ ふー くぅ ほぉ わっ）
        return h[0] in valid_cons and h[1] in "っー～〜~ぁぃぅぇぉ"
    if h[0] not in valid_cons:
        return False
    v0 = _CONS_VOWEL.get(h[0], "")
    # 允许纯延长/促音/浊点、小写元音（ほぉぅぅっ）、以及首辅音的本行大写元音（ごおっ / はあぁ）
    allowed = _PURE_MARKS | _SMALL | ({v0} if v0 else set())
    if not all(c in allowed for c in h[1:]):
        return False
    # 至少包含 1 个延长/促音/小写元音特征（防止纯大写元音串如 はああ 以外的异常），本行重复元音 >=2 也可
    return any(c in _PURE_MARKS or c in _SMALL for c in h[1:]) or (v0 and h[1:].count(v0) >= 2)


def _seg(s: str, memo: dict) -> bool:
    """合法词干拼接；口腔音可续接，任意未知尾串仍须完整验证。"""
    s = _segment_form(s)

    def parse(pos: int, mouth: bool) -> bool:
        if pos == len(s):
            return True
        key = (s, pos, mouth)
        if key in memo:
            return memo[key]
        memo[key] = False
        for stem in _SEG_STEMS:
            if s.startswith(stem, pos):
                if parse(pos + len(stem), stem not in _CLOSED_STEMS):
                    memo[key] = True
                    return True
        if s[pos] in _SINGLE and parse(pos + 1, mouth):
            memo[key] = True
            return True
        if mouth:
            for syllable in _MOUTH_TAIL:
                if s.startswith(syllable, pos) and parse(pos + len(syllable), True):
                    memo[key] = True
                    return True
        return False

    return parse(0, False)


def is_ono(text: str) -> bool:
    """该串是否为「CV 自由发挥类拟声」。

    判定顺序（黑名单优先）：
      1. 黑名单命中 -> 否
      2. R5 ん 起首 / R6 促音起首 / R7 纯元音记号 -> 是
      3. 归一后命中词干 / 去透记号后命中词干 -> 是
      4. 单辅音 + 纯延长记号 -> 是
      5. 切分 -> 是
    """
    if not text or len(text) > 60:
        return False
    global _BLOCK_IMPORT_WARNED
    is_blocked_fn = None
    try:
        from fa_asmr.text.ono_block_corpus import is_blocked as is_blocked_fn
        if is_blocked_fn(text):
            return False
    except ImportError as e:
        # 黑名单不可用时跳过检查 —— 实词会被误判为拟声，进而可能被当呻吟删掉。
        # 这是「宁缺毋滥」的反面，必须留痕，不能静默。
        if not _BLOCK_IMPORT_WARNED:
            _BLOCK_IMPORT_WARNED = True
            print(
                f"  ⚠ [ono_base] 拟声黑名单不可用（{e}），已跳过实词拦截 —— "
                "存在把实词误判为拟声（进而误删）的风险。",
                file=sys.stderr,
            )

    # 先判 cons_tail（直接基于原串，不做叠字压缩，保留小写元音并避免 ままー/みみっ 误杀）
    text = _normalize_pseudo_dakuten(text)
    if _is_cons_tail(text):
        return True
    hira_raw = "".join(_K2H.get(c, c) for c in text)
    n = _norm(text)
    # R5 ん 起首 / R6 促音起首：剥离前导喘息后验证剩余内容，严禁一刀切秒杀 っまって / っだめ / んいたい / んうそ 等实词；
    # 若剩余部分本身命中黑名单（如 んうん / んいい / んえっ / っうん / っいい），直接拦截，防止落入 _seg 被拆成单字放行。
    _ACTION_GLIDE = frozenset("しきち")
    if hira_raw.startswith("ん") and _KANA_RE.match(hira_raw):
        if len(n) > 1 and n[1] in _ACTION_GLIDE:
            pass
        else:
            rest = hira_raw.lstrip("んっー～〜~゛゜ﾞﾟ")
            if not rest:
                return True
            if is_blocked_fn is not None and is_blocked_fn(rest):
                return False
            if _is_cons_tail(rest) or (rest != hira_raw and is_ono(rest)):
                return True
    if hira_raw.startswith("っ") and _KANA_RE.match(hira_raw):
        rest = hira_raw.lstrip("っー～〜~゛゜ﾞﾟ")
        if not rest:
            return True
        if is_blocked_fn is not None and is_blocked_fn(rest):
            return False
        if _is_cons_tail(rest) or (rest != hira_raw and is_ono(rest)):
            return True
    # R7 纯元音 + 记号 —— あ/う/お 开头，长度 ≥ 2（'い' 单字不算）
    if (len(n) >= 2 and n[0] in "あうお"
            and all(c in _VOW or c in _TRANSPARENT for c in n)):
        return True
    if n in _COMBO_NORM or _strip_marks(n) in _COMBO_UNMARKED:
        return True
    if _seg(text, {}):
        return True
    if n != text and _seg(n, {}):
        return True
    # R3 内嵌透记号：去掉后若能切分，算命中（はーふう → はふう）
    stripped = _strip_marks(n)
    if stripped != n and _seg(stripped, {}):
        return True
    return False


# 汉字 / 英文 / 数字（含全角）
_KANJI_RE = re.compile(r"[一-鿿㐀-䶿豈-﫿々]")
_LATIN_RE = re.compile(r"[A-Za-zＡ-Ｚａ-ｚ]")
_DIGIT_RE = re.compile(r"[0-9０-９]")


def _has_kanji(text: str) -> bool:
    return bool(_KANJI_RE.search(text or ""))


def is_easy_ono(surface: str) -> bool:
    """该子句是否为【易错拟声行】—— 全项目唯一的二分判据。

    只有两种：易错拟声行 / 非易错拟声行。
    **没有命中易错拟声判定的，一律是非易错拟声行** —— 不再细分原因。

    判定顺序：
      1. 含汉字/英文/数字  -> False
      2. 不是拟声         -> False（ono_base 内部已含黑名单：实词/动作音/喷射音…）
      其余 -> True（易错拟声行）
    """
    if not surface:
        return False
    # 1. 汉字/英文/数字 —— 直接排除
    if _has_kanji(surface) or _LATIN_RE.search(surface) or _DIGIT_RE.search(surface):
        return False
    # 2. 不是拟声（is_ono 内部已查黑名单，此处不再重复查）
    return is_ono(surface)
