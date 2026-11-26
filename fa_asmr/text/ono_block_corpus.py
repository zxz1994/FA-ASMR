# -*- coding: utf-8 -*-
"""真词黑名单 —— 词根表 + 含词根即拦。

设计：
  ① 词根表 —— 每个实词只留词根，不枚举变体
     例：いぐ（含 いぐぅぅ / いぐっ / いぐ～～ 全部覆盖）
         おちんちん（含 おちんちんっ / おちんちんに / おちんちんイグっ）
         ぎゅ（含 ぎゅっ / ぎゅ～～～ / ぎゅぅぅ）
  ② 匹配规则 —— 含词根即拦
     is_blocked(text) = any(root in text for root in _ROOTS)
     这样词根表只需几十条，自动覆盖所有变体。

从 603 条变体枚举提炼而来。
"""

import re

# 词根表（按类别）
_ROOTS = frozenset("""
    あそこ あたし あちら あなた あの あら ありがと ある
    あれ あんた あんな いい いいこ いいぞ いいの いいよ
    いいわよ いう いえ いえい いえいえ いえす いえーい いぇい いぇーい いく いくよ
    いぐ いち いっぱい いつも いや いる うち うれし
    うわ うわーい うわーん うんうん うんちする ええ えー えい えいっ えっ えっと おう おうどう おーい おく
    おしっこ おちんちん おちんぽ おっぱい おにい おにー おぼッ おまんこ
    おや おれ おれの かおが かおに かっこいい かのじょ から
    かれ かわいい きた きっと きみ きもち きゅう ぎゅ
    ください くにゅ くる くるくる けど こう こうして こうどう こー
    ここ ここから ここで ここに ここまで こしょ こしょこしょ こす
    こすこす こちら こと この このまま これ これが これは
    こんな こんなに ご ごご ごし ごぜん ごはん
    ごはんっ ごめん ごろ さあ さぁ さわ さわさわ さん しぃ
    しこ しこしこ しこしこっ しこっ しごと した して しまう
    しゅき じぶん じゃない すき すきすき すこし すご
    すみません すり すりすり する ず ずこ ずこずこ ずっと
    ずぶ ずぷ ずぼ ずぼずぼ ずる ずるずる せんせ そう そー
    そうですか そうどう そうなの そこ そちら そっか その それ
    それじゃあ それでは それに そんな たい たち たのし ため
    だいすき だから だけ だって だめ ちゃう ちゃった ちょ
    ちょっと ちらちら ちろ ちんちん ちんぽ つもり つん です
    ですから でも とき ところ ところで とても とにかく ともかく
    とりあえず とんでも どう どうして どうぞ どうどう どうも どー どこ
    どちら どれ どんな ない なく など なな なに
    なる にぃ にい ぬぎ ねえ ねー ねっ ので のに はい
    はいこれ はいどうぞ はいはい はず はずかし はち ぱん ひと
    びゅ びゅる ぴゅ ぴゅる ふあん ふおん ふり ふん
    へー へこ ほう ほら ほんと ほーら ぼく
    ぼくの まあ まー ました ます まずは ません また まだ
    まったく まで まんこ みぃ みる みんな むき むぎゅ
    もう もー もういい もうすぐ もうだめ もこ もしかして もちろん もっと
    もの もみ もみもみ やい やだ やっぱり やば やめ
    やん ゆうか ゆうき ようこそ ような ように ようやく よく
    よし よしよし よちよち より よん りん ろく わかった
    わけ わたくし わたくしの わたし わたしの わぁ わー わーい あれれ んじゃ んしょ アイカ
    アリス イキます イク イグ イッ イエーイ エリス オッケー カリ
    カンナ キス クリ ザーメン シオン セックス ダーリン チンカス
    ナツキ フェラ ベロニカ ボク ミア メリル ルビィ ルーシア
    ワタクシ ワタシ
""".split())


_TAIL = frozenset("っッー～〜~゛゜ﾞﾟんンぁぃぅぇぉァィゥェォ")
_PROLONG_RE = re.compile(r"[ー～〜~]+")
_K2H = {chr(_o + 0x60): chr(_o) for _o in range(0x3041, 0x3097)}
_SMALL_VOWELS = dict(zip("ぁぃぅぇぉ", "あいうえお"))
_VOWEL = {kana: vowel for vowel, row in (
    ("あ", "あかがさざただなはばぱまやゃらわ"),
    ("い", "いきぎしじちぢにひびぴみり"),
    ("う", "うくぐすずつづぬふぶぷむゆゅるゔ"),
    ("え", "えけげせぜてでねへべぺめれ"),
    ("お", "おこごそぞとどのほぼぽもよょろを"),
) for kana in row}


def _stretch_key(hira: str) -> str:
    """Fold only the preceding kana's own vowel. Keep length on single-mora words."""
    letters = []
    vowel = ""
    stretched = False
    for c in hira:
        full = _SMALL_VOWELS.get(c, c)
        if vowel and (c in "ー～〜~" or full == vowel):
            stretched = True
            continue
        letters.append(c)
        vowel = _VOWEL.get(full, "")
    if stretched and len(letters) == 1:
        letters.append("ー")
    return "".join(letters)


# Match complete words. Shortening a substring inside a breath/noise can expose
# a different word. A lone vowel or consonant also supplies too little evidence.
_HIRA_TABLE = str.maketrans(_K2H)
_ROOTS_HIRA = frozenset(_r.translate(_HIRA_TABLE) for _r in _ROOTS)
# 精确匹配的短应答/口语感叹：对所有延长符（ー/～/〜/ASCII ~）归一折叠并剥离尾促音后精确比对，
# 不参与本行元音折叠，避免误伤长呻吟（うぅぅん / あんっ / あぁ / おぉ）及湿吻吸吮音（じゅうっ / んじゅうぅぅ）。
_EXACT_SPOKEN = frozenset((
    "うん", "うんうん", "うーん", "ううん",
    "あーあ", "あーん", "おー",
))
# 严格数词精确匹配（不剥离尾促音，不参与子串/本行元音折叠）：
# 保护「じゅう」「じゅうん」「ごお」「ご～お」等数数/实词，但不误伤湿吻/喉音「じゅうっ」「じゅうう」「んじゅうぅぅ」「ごおおっ」「っごおっ」「んごぉっ」。
_EXACT_COUNT = frozenset(("じゅう", "じゅうん", "ごお"))
# 兼容旧名称
_SHORT_YES = _EXACT_SPOKEN
# Counting words whose single-mora stretched form collides with pure kissing/sucking
# sounds (じゅぅぅっ / きゅぅぅっ) must not collapse to じゅー / きゅー.
_NO_STRETCH_FOLD = frozenset(("きゅう", "じゅう", "ごお"))
_STRETCHED_WORDS = set()
for _rh in _ROOTS_HIRA:
    if len(_rh) >= 2:
        _STRETCHED_WORDS.add(_rh)
        if _rh not in _NO_STRETCH_FOLD:
            _sk = _stretch_key(_rh)
            if len(_sk) >= 2:
                _STRETCHED_WORDS.add(_sk)
_CALLS = {_stretch_key(word) for word in ("はい", "はぃ", "ねえ", "まあ", "もー", "もう", "もぅ", "そー", "どー", "おーい", "いえ", "ええ", "えー", "へー")}
_SPOKEN_EXACT_INTERJECTIONS = _EXACT_SPOKEN | frozenset(("えっ", "ねっ"))
_EDGES = "…。、！？!?♡♪"


def _norm_exact_spoken(hira_edges_stripped: str) -> str:
    """将所有全/半角延长波浪符（ー/～/〜/~）统一折叠为单个「ー」并去尾促音，供精确短应答比对。"""
    return _PROLONG_RE.sub("ー", hira_edges_stripped.rstrip("っ"))


def _spoken_key(hira: str) -> str:
    return _stretch_key(hira.strip("っ" + _EDGES))


def is_spoken_interjection(text: str) -> bool:
    """识别应答、呼唤和独立的「わ」感叹词，保留区分喘息所需的尾音。"""
    hira = text.translate(_HIRA_TABLE)
    se = hira.strip(_EDGES)
    se_sokuon = re.sub(r"っ+$", "っ", se)
    return (
        se == "わ"
        or se in _SPOKEN_EXACT_INTERJECTIONS
        or se_sokuon in _SPOKEN_EXACT_INTERJECTIONS
        or _norm_exact_spoken(se) in _EXACT_SPOKEN
        or _spoken_key(hira) in _CALLS
    )


def is_blocked(text: str) -> bool:
    """该串是否含实词词根。

    匹配规则：
      1. 独立片假名「イ」、呼唤/应答语、归一后的精确短应答（兼容全/半角波浪号与尾促音）及独立「わ」直接拦；
      2. 严格数词精确匹配（じゅう / ごお / ご～お，不误伤 じゅうっ / ごおおっ / っごおっ）；
      3. 去掉句末边缘符号后的核心串（含/不含尾随促音与延长符）及其本行元音折叠键
         命中任意 len >= 2 的黑名单词根（含平假名归一与折叠键）即拦；
      4. 长度 == 2 的词根：【整串相等】或【词根 + 纯尾记号】即拦；
      5. 长度 >= 3 的词根：【含它】即拦（同时覆盖剥除长音符与折叠后形式）。
    """
    # 独立片假名「イ」（高频「イ、イク！」截断词）精确拦截，不影响平假名喘息「い」
    if text.strip(_EDGES) == "イ":
        return True
    hira = text.translate(_HIRA_TABLE)
    stripped_edges = hira.strip(_EDGES)
    if (
        stripped_edges == "わ"
        or _norm_exact_spoken(stripped_edges) in _EXACT_SPOKEN
        or _spoken_key(hira) in _CALLS
    ):
        return True
    # 严格数词保护（不剥离尾促音，允许词内延长符如 ご～お -> ごお，但不影响 ごおおっ / じゅうっ / じゅうう）
    if stripped_edges in _EXACT_COUNT or _PROLONG_RE.sub("", stripped_edges) in _EXACT_COUNT:
        return True
    if stripped_edges in _STRETCHED_WORDS:
        return True
    core = stripped_edges.rstrip("っー～〜~")
    if core in _STRETCHED_WORDS:
        return True
    key = _stretch_key(core)
    if key in _STRETCHED_WORDS:
        return True
    # 词内含促音/延长符（如 わぁーい / もぉういい / いっえいえ）时，仅对 len >= 3 的长词根做子串保护
    core_no_sokuon = core.replace("っ", "")
    key_no_sokuon = _stretch_key(core_no_sokuon)
    cands_list = []
    for base in (hira, stripped_edges, core, key):
        if base and base not in cands_list:
            cands_list.append(base)
        compact = _PROLONG_RE.sub("", base)
        if compact and compact not in cands_list:
            cands_list.append(compact)
    for cand in cands_list:
        if not cand:
            continue
        for r in _ROOTS_HIRA:
            if len(r) == 1:
                # 单字词根（如 ご / ず）仅精确匹配去边缘标点后的原串，
                # 严禁匹配经本行元音折叠后再剥除「ー」退化成的单字（避免 ごおおっ/ごぉっ -> ごー -> ご 误杀）
                if stripped_edges == r:
                    return True
            elif len(r) == 2:
                if cand == r:
                    return True
                if cand.startswith(r):
                    rest = cand[2:]
                    if rest and all(c in _TAIL for c in rest):
                        return True
            elif r in cand:
                return True
    # 长黑名词根（去内部促音后 len >= 3）在拖长折叠后完整命中（如 もぉういい / おっーい / きっゅう），
    # 不剥除前导 っ，避免误伤 っごおっ 等前导促音喉咙喘息
    if (
        not stripped_edges.startswith("っ")
        and len(core_no_sokuon) >= 3
        and (core_no_sokuon in _STRETCHED_WORDS or key_no_sokuon in _STRETCHED_WORDS)
    ):
        return True
    return False


# 兼容旧接口
_ONO_BLOCK_CORPUS = _ROOTS
