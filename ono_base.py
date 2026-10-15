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

# ══ ① 单音素 ══
_SINGLE = frozenset("""
    ん っ ッ ー ～ ゛ ゜ ﾞ ﾟ
    ぁ ぃ ぅ ぇ ぉ ゃ ゅ ょ
    あ い う え お
""".split())

# ══ ② 词干 ══
_COMBO = frozenset("""
    あが あぐ あっく あは あひ あへ あふ あむ うぐ うふ
    うぷ うぷぷ うへ うほ えへ おへ おほ きゃ きゃは ぎゃ
    きゅ くす くちゅ くん ぐちゃ ぐちゅ ぐっぽ ぐぼ ぐぽ
    げふ げぷ げほ こほん こぽ こぽっ ごく ごっくん ごぽ
    ごぽっ じゅ じゅぱ じゅぶ じゅぷ じゅぼ じゅぽ じゅる すぅ
    すうすう すふ すん ぞりゅ ちゅ ちゅぱ ちゅぴ ちゅぶ ちゅぷ
    ちゅぼ ちゅぽ ちゅる ぢゅ ぢゅぶ ぢゅぷ ぢゅぼ ぢゅぽ ぢゅる
    れちゅ でへ
    にゅ にゅちゅ にゅぷ にゅぽ にゅる にゅるる ぬちゃ ぬちゅ はぁ
    はあはあ はふ はむ ひゃ ひゅ ぴちゅ ふぁ ふぃ ふぅ
    ふうふう ふぇ ふぉ ふは ふふ ふふぅ ふーふー ぷは べろ
    ぺちゃ ぺろ むぐ むちゅ むにゅ むふ もぐ れう
    れぅ れぇ れぇろ れら れり れりゅ れる れれ れろ
    れろぉ んき んぎ んく んぐ んじ んじゅ んちゅ んぢ
    んは んば んぱ んび んぴ んふ んぶ んべ んほ
    んみ んむ んり んれろ ゲ ゲェ
""".split())

# 片假名对称
_H2K = {chr(_o): chr(_o + 0x60) for _o in range(0x3041, 0x3097)}
_K2H = {v: k for k, v in _H2K.items()}
_COMBO = _COMBO | frozenset(
    "".join(_H2K.get(_c, _c) for _c in _b) for _b in _COMBO)



# ══ ③ 变换 ══
_VEQ = {"ぁ": "あ", "ぃ": "い", "ぅ": "う", "ぇ": "え", "ぉ": "お"}
_SMALL = frozenset("ぁぃぅぇぉ")
_TRANSPARENT = frozenset("っッー～〜~゛゜ﾞﾟ")
_TAIL_EXT = frozenset("っッー～〜~゛ﾞ゜ﾟんンあいうえおぁぃぅぇぉァィゥェォ")
# 续接音节仅在口腔词干之后使用，不参与独立起手匹配。
_MOUTH_TAIL = tuple(sorted(
    set("むろるぷぽぶぼぱこくふあいうえおんゃゅょ")
    | {c + v for c in "ぎぢびぴみり" for v in "ゃゅょ"},
    key=len, reverse=True,
))
# 有清晰词形的吞咽/清嗓音及尖叫，只接记号、元音或完整词干。
_CLOSED_STEMS = frozenset("""
    ごく ごっくん ごぽ ごぽっ こぽ こぽっ こほん
    げふ げぷ げほ ゲ ゲェ ぎゃ
""".split())
_CONS = frozenset(
    "かきくけこがぎぐげごさしすせそざじずぜぞたちつてとだぢづでど"
    "なにぬねのはひふへほばびぶべぼぱぴぷぺぽまみむめもやゆよらりるれろわ")
_VOW = frozenset("あいうえおぁぃぅぇぉ")

# 音节形状本身不证明实际发声方式；续接只由已识别的口腔词干开启。

_KANA_RE = re.compile(r"^[\u3041-\u309F\u30A0-\u30FFー～゛ﾞ゜ﾟ]+$")

# 黑名单不可用时的降级告警只打一次（见 is_ono）
_BLOCK_IMPORT_WARNED = False


def _norm(s: str) -> str:
    """R1 片假名→平假名 → R2 小写→大写（词尾小写保留作尾记号）→ R4 重复压缩。"""
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


_COMBO_NORM = frozenset(_norm(w) for w in _COMBO)
_COMBO_UNMARKED = frozenset(_strip_marks(_norm(w)) for w in _COMBO)


def _segment_form(s: str) -> str:
    """统一字形并去透明记号，但保留词干音节数，避免把ふふ缩成裸ふ。"""
    hira = "".join(_K2H.get(c, c) for c in s)
    return "".join(_VEQ.get(c, c) for c in _strip_marks(hira))


_SEG_STEMS = tuple(sorted({_segment_form(w) for w in _COMBO}, key=len, reverse=True))
_CLOSED_STEMS = frozenset(_segment_form(w) for w in _CLOSED_STEMS)


def _is_cons_tail(s: str) -> bool:
    """呼气/呻吟的最简形式。

    长度 2：
      ① 元音重复       ああ いい うう ええ おお
      ② 辅音 + 促音/长音  はっ ふっ はー ふー
      —— 不含「辅音 + 元音」（ここ / もう / いい 是实词，不收）
    长度 3~4：
      单辅音 + 纯延长记号（はっぁ / ふーっ / くぅぅ / ほぉぉ）
    """
    if not s or len(s) > 4:
        return False
    if len(s) == 2:
        # 元音重复（ああ うう おお）—— 但 いい/ええ 是实词
        if s[0] in "あうお" and s[0] == s[1]:
            return True
        # 辅音 + 促音/长音/波浪/小写元音（はっ ふー くぅ ほぉ）
        return s[0] in _CONS and s[1] in "っッー～ぁぃぅぇぉ"
    return s[0] in _CONS and all(c in _TAIL_EXT for c in s[1:])


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
    try:
        from ono_block_corpus import is_blocked
        if is_blocked(text):
            return False
    except ImportError as e:
        # 黑名单不可用时跳过检查 —— 实词会被误判为拟声，进而可能被当呻吟删掉。
        # 这是「宁缺毋滥」的反面，必须留痕，不能静默。
        if not _BLOCK_IMPORT_WARNED:
            _BLOCK_IMPORT_WARNED = True
            print(f"  ⚠ [ono_base] 拟声黑名单不可用（{e}），已跳过实词拦截 —— "
                  "存在把实词误判为拟声（进而误删）的风险。")

    # 先判 cons_tail（归一会把 くぅ 变成 くう，丢失小写信息）
    if _is_cons_tail(text):
        return True
    n = _norm(text)
    # R5 ん 不能起首日语词
    # 但 ん+动作音拗音（んしょ/んきゅ/んちゅ）是动作音，不收；
    # ん+其他拗音（んぢぁ/んじぇ/んびぉ）是呻吟，收。
    _ACTION_GLIDE = frozenset("しきち")
    if (n.startswith("ん") and _KANA_RE.match(n)
            and not (len(n) > 1 and n[1] in _ACTION_GLIDE)):
        return True
    # R6 促音不能起首
    if n[:1] == "っ":
        return True
    # R7 纯元音 + 记号 —— あ/う/お 开头，长度 ≥ 2（'い' 单字不算）
    if (len(n) >= 2 and n[0] in "あうお"
            and all(c in _VOW or c in _TRANSPARENT or c == "～" for c in n)):
        return True
    if n in _COMBO_NORM or _strip_marks(n) in _COMBO_UNMARKED:
        return True
    if _is_cons_tail(n):
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
