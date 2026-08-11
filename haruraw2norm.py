from janome.tokenizer import Tokenizer  # noqa: F401  （仅导入模块；构造器在首次使用时惰性创建）
import pykakasi
import re
import logging

logger = logging.getLogger(__name__)

# nltk / pyphen / pypinyin 仅用于中英文音素转换，日文场景不需要
# 条件导入避免 FA-ASMR 打包时引入不必要依赖
try:
    import nltk
    from nltk.corpus import cmudict
except ImportError:
    nltk = None
    cmudict = None
try:
    import pyphen
except ImportError:
    pyphen = None
try:
    import pypinyin
    from pypinyin import pinyin, Style
except ImportError:
    pypinyin = None

# 惰性加载重型依赖：import 本模块时不再立即构造 pykakasi / janome / 下载 nltk 词典，
# 仅在首次实际使用时初始化并缓存，显著加快打包后的启动速度。
_kks = None
_tokenizer = None
_cmu_dict = None
_eng_dic = None


def _get_kks():
    global _kks
    if _kks is None:
        _kks = pykakasi.kakasi()
    return _kks


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = Tokenizer()
    return _tokenizer


def _get_cmu_dict():
    global _cmu_dict
    if _cmu_dict is None and cmudict is not None:
        try:
            _cmu_dict = cmudict.dict()
        except LookupError:
            try:
                nltk.download('cmudict')
                _cmu_dict = cmudict.dict()
            except Exception:
                _cmu_dict = None
    return _cmu_dict


def _get_eng_dic():
    global _eng_dic
    if _eng_dic is None and pyphen is not None:
        _eng_dic = pyphen.Pyphen(lang='en_US')
    return _eng_dic


tail_pron = '' # 'h'

# pykakasi 对非常规假名组合可能返回空（外来语、语气延长等），手动兜底
KANA_FALLBACK = {
    # 外来语：ふ行 + 小写元音
    'ふぁ': 'fa', 'ふぃ': 'fi', 'ふぇ': 'fe', 'ふぉ': 'fo',
    'ファ': 'fa', 'フィ': 'fi', 'フェ': 'fe', 'フォ': 'fo',
    # う + 小写元音
    'うぃ': 'wi', 'うぇ': 'we', 'うぉ': 'wo',
    'ウィ': 'wi', 'ウェ': 'we', 'ウォ': 'wo',
    # て/で + 小写元音
    'てぃ': 'ti', 'でぃ': 'di', 'とぅ': 'tu', 'どぅ': 'du',
    'ティ': 'ti', 'ディ': 'di', 'トゥ': 'tu', 'ドゥ': 'du',
    # つ + 小写元音
    'つぁ': 'tsa', 'つぃ': 'tsi', 'つぇ': 'tse', 'つぉ': 'tso',
    'ツァ': 'tsa', 'ツィ': 'tsi', 'ツェ': 'tse', 'ツォ': 'tso',
    # く/ぐ + 小写元音
    'ぐぁ': 'gwa', 'くぁ': 'kwa', 'くぃ': 'kwi', 'くぇ': 'kwe', 'くぉ': 'kwo',
    'グァ': 'gwa', 'クァ': 'kwa', 'クィ': 'kwi', 'クェ': 'kwe', 'クォ': 'kwo',
    # ヴ行（う゛/ヴ + 小写元音）
    'ヴぁ': 'va', 'ヴぃ': 'vi', 'ヴぇ': 've', 'ヴぉ': 'vo',
    'う゛ぁ': 'va', 'う゛ぃ': 'vi', 'う゛ぇ': 've', 'う゛ぉ': 'vo',
    # 小写元音延长/语气（浊音+小写元音为主，pykakasi 常返回空）
    'ふぅ': 'fuu', 'すぃ': 'si', 'ずぃ': 'zi',
    'てぇ': 'tee', 'でぇ': 'dee', 'せぇ': 'see',
    'ほぅ': 'hoo', 'へぇ': 'hee',
    'ざぁ': 'zaa', 'じゃぁ': 'jaa', 'だぁ': 'daa',
    'ばぁ': 'baa', 'ぱぁ': 'paa', 'がぁ': 'gaa',
    'なぁ': 'naa', 'まぁ': 'maa', 'らぁ': 'raa', 'わぁ': 'waa',
    'かぁ': 'kaa', 'さぁ': 'saa', 'たぁ': 'taa', 'はぁ': 'haa',
    'あぁ': 'aa', 'やぁ': 'yaa', 'んぁ': 'na',
    'じぃ': 'jii', 'りぃ': 'rii', 'にぃ': 'nii', 'うぃ': 'ui',
    'くぅ': 'kuu', 'ぐぅ': 'guu', 'すぅ': 'suu', 'つぅ': 'tsuu',
    'ぬぅ': 'nuu', 'むぅ': 'muu', 'ゆぅ': 'yuu', 'るぅ': 'ruu',
    'うぅ': 'uu', 'ぶぅ': 'buu', 'ぷぅ': 'puu', 'ずぅ': 'zuu',
    'づぅ': 'zuu', 'んぅ': 'nu',
    'けぇ': 'kee', 'げぇ': 'gee', 'ぜぇ': 'zee',
    'べぇ': 'bee', 'ぺぇ': 'pee', 'ねぇ': 'nee',
    'めぇ': 'mee', 'れぇ': 'ree', 'えぇ': 'ee',
    'ほぉ': 'hoo', 'こぉ': 'koo', 'ごぉ': 'goo', 'そぉ': 'soo',
    'ぞぉ': 'zoo', 'とぉ': 'too', 'どぉ': 'doo',
    'ぼぉ': 'boo', 'ぽぉ': 'poo', 'のぉ': 'noo',
    'もぉ': 'moo', 'よぉ': 'yoo', 'ろぉ': 'roo',
    'おぉ': 'oo', 'をぉ': 'woo', 'んぉ': 'no',
    'フゥ': 'fuu', 'スィ': 'si', 'ズィ': 'zi',
    'テェ': 'tee', 'デェ': 'dee', 'セェ': 'see',
    'ホゥ': 'hoo', 'ヘェ': 'hee',
    'ザァ': 'zaa', 'ジャァ': 'jaa', 'ダァ': 'daa', 'バァ': 'baa', 'パァ': 'paa',
    'ガァ': 'gaa', 'ナァ': 'naa', 'マァ': 'maa', 'ラァ': 'raa', 'ワァ': 'waa',
    'カァ': 'kaa', 'サァ': 'saa', 'タァ': 'taa', 'ハァ': 'haa',
    'アァ': 'aa', 'ヤァ': 'yaa', 'ンァ': 'na',
    'ジィ': 'jii', 'リィ': 'rii', 'ニィ': 'nii',
    'クゥ': 'kuu', 'グゥ': 'guu', 'スゥ': 'suu', 'ツゥ': 'tsuu',
    'ヌゥ': 'nuu', 'ムゥ': 'muu', 'ュゥ': 'yuu', 'ルゥ': 'ruu',
    'ウゥ': 'uu', 'ブゥ': 'buu', 'プゥ': 'puu', 'ズゥ': 'zuu',
    'ヶェ': 'kee', 'ゲェ': 'gee', 'ゼェ': 'zee',
    'ベェ': 'bee', 'ペェ': 'pee', 'ネェ': 'nee',
    'メェ': 'mee', 'レェ': 'ree', 'エェ': 'ee',
    'ホォ': 'hoo', 'コォ': 'koo', 'ゴォ': 'goo', 'ソォ': 'soo',
    'ゾォ': 'zoo', 'トォ': 'too', 'ドォ': 'doo',
    'ボォ': 'boo', 'ポォ': 'poo', 'ノォ': 'noo',
    'モォ': 'moo', 'ヨォ': 'yoo', 'ロォ': 'roo',
    'オォ': 'oo', 'ヲォ': 'woo', 'ンォ': 'no',
    # 其他
    'いぇ': 'ye', 'イェ': 'ye',
    'じぇ': 'je', 'ジェ': 'je',
    'ちぇ': 'che', 'チェ': 'che',
    'しぇ': 'she', 'シェ': 'she',
    'ぐぃ': 'gwi', 'ぐぇ': 'gwe', 'グィ': 'gwi', 'グェ': 'gwe',
}

def safe_kks_convert(kana_str):
    """pykakasi 转换假名→罗马音，KANA_FALLBACK 兜底"""
    kk = _get_kks().convert(kana_str)
    if kk:
        return kk[0]['hepburn']
    return KANA_FALLBACK.get(kana_str, tail_pron)

phoneme_map = {
    'AA': 'a', 'AE': 'a', 'AH': 'a', 'AO': 'o', 'AW': 'au', 'AY': 'ai',
    'B': 'b', 'CH': 'ch', 'D': 'd', 'DH': 'z', 'EH': 'e', 'ER': 'a',
    'EY': 'ei', 'F': 'f', 'G': 'g', 'HH': 'h', 'IH': 'i', 'IY': 'i',
    'JH': 'j', 'K': 'k', 'L': 'r', 'M': 'm', 'N': 'n', 'NG': 'ng',
    'OW': 'o', 'OY': 'oi', 'P': 'p', 'R': 'r', 'S': 's', 'SH': 'sh',
    'T': 't', 'TH': 's', 'UH': 'u', 'UW': 'u', 'V': 'v', 'W': 'w',
    'Y': 'y', 'Z': 'z', 'ZH': 'j'
}
PINYIN_TO_PHONETIC = {
    # 声母
    'b': 'b', 'p': 'p', 'm': 'm', 'f': 'f',
    'd': 'd', 't': 't', 'n': 'n', 'l': 'r',
    'g': 'g', 'k': 'k', 'h': 'h',
    'j': 'j', 'q': 'ch', 'x': 'sh',
    'zh': 'j', 'ch': 'ch', 'sh': 'sh', 'r': 'r',
    'z': 'z', 'c': 'ts', 's': 's',
    'y': 'y', 'w': 'w',
    
    # 韵母
    'a': 'a', 'o': 'o', 'e': 'e', 'i': 'i', 'u': 'u', 'ü': 'yu',
    'ai': 'ai', 'ei': 'ei', 'ao': 'ao', 'ou': 'ou',
    'an': 'an', 'en': 'en', 'ang': 'ang', 'eng': 'eng', 'ong': 'ong',
    'ia': 'ya', 'ie': 'ye', 'iao': 'yao', 'iu': 'yu', 
    'ian': 'yan', 'in': 'in', 'iang': 'yang', 'ing': 'ing', 'iong': 'yong',
    'ua': 'wa', 'uo': 'wo', 'uai': 'wai', 'ui': 'wei',
    'uan': 'wan', 'un': 'wen', 'uang': 'wang', 'ueng': 'weng',
    'üe': 'yue', 'üan': 'yuan', 'ün': 'yun',
    
    'er': 'a', 'io': 'yo', 'o': 'wo', 'e': 'e'
}
# 注：cmu_dict / eng_dic 已改为惰性加载（见 _get_cmu_dict / _get_eng_dic），
# 不再在模块导入时构造，避免打包后启动拖慢。

newnums = ['①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧', '⑨', '⑩',
           '⑪', '⑫', '⑬', '⑭', '⑮', '⑯', '⑰', '⑱', '⑲', '⑳',
           '㉑', '㉒', '㉓', '㉔', '㉕', '㉖', '㉗', '㉘', '㉙', '㉚']

# 有实际日文读音的符号 → 罗马音（type 5 直接注音）
SYMBOL_TO_PRONS = {
    '％': 'paasento',
    '√': 'ruuto',
    '×': 'kakeru',
    '÷': 'waru',
    '＋': 'purasu',
    '－': 'mainasu',
    '＝': 'ikooru',
    '℃': 'do',
    '°': 'do',
    '∞': 'mugendai',
    '＆': 'ando',
    '＠': 'atto',
    '＃': 'shaapu',
    '＄': 'doru',
}

# 无声分隔符（中圆点/片假名连字点）：落在片假名区但无读音，MMS_FA 词典外。
# 在 process_haruhi_line 的字符分类里直接跳过，避免被 pykakasi 读成 '・' 当 pron
# 喂给对齐器触发 KeyError（与 align.py 喂 MMS_FA 前过滤 vocab 外字符的行为一致）。
SEPARATOR_KANA = ('・', '゠')

def normalize_numbers(text):
    """将字符串中的各种数字字符转换为半角阿拉伯数字"""
    # 全角数字转半角
    fullwidth_to_half = str.maketrans('０１２３４５６７８９', '0123456789')
    text = text.translate(fullwidth_to_half)
    
    # 特殊数字转换映射表
    number_map = {
        # 分数
        '½': '0.5', '⅓': '0.333', '⅔': '0.666', '¼': '0.25', '¾': '0.75',
        '⅕': '0.2', '⅖': '0.4', '⅗': '0.6', '⅘': '0.8', '⅙': '0.166',
        '⅚': '0.833', '⅛': '0.125', '⅜': '0.375', '⅝': '0.625', '⅞': '0.875',
        # 罗马数字
        'Ⅰ': '1', 'Ⅱ': '2', 'Ⅲ': '3', 'Ⅳ': '4', 'Ⅴ': '5',
        'Ⅵ': '6', 'Ⅶ': '7', 'Ⅷ': '8', 'Ⅸ': '9', 'Ⅹ': '10',
        'Ⅺ': '11', 'Ⅻ': '12', 'Ⅼ': '50', 'Ⅽ': '100', 'Ⅾ': '500', 'Ⅿ': '1000',
        # 汉字数字
        '零': '0', '一': '1', '二': '2', '三': '3', '四': '4', '五': '5',
        '六': '6', '七': '7', '八': '8', '九': '9', '十': '10', '百': '100',
        '千': '1000', '万': '10000', '亿': '100000000',
        # 上标/下标
        '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4', '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9',
        '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4', '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9'
    }
    
    # 构建转换表
    trans_table = str.maketrans(number_map)
    return text.translate(trans_table)

def number_to_english(number_str):
    """将数字字符串转换为英文单词"""
    try:
        if '.' in number_str:
            num = float(number_str)
        else:
            num = int(number_str)
    except ValueError:
        logger.warning('Unable to process number "%s"...', number_str)
        return tail_pron
    
    ones = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
            "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
            "seventeen", "eighteen", "nineteen"]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    
    if isinstance(num, float):
        integer_part = int(num)
        decimal_part = round(num - integer_part, 3)
        integer_words = number_to_english(str(integer_part)) if integer_part > 0 else ""
        
        # 处理小数部分
        decimal_str = f"{decimal_part:.3f}"[2:] # 获取小数点后三位
        decimal_str = decimal_str.rstrip('0') or '0'  # 去除尾部无效零
        decimal_words = " point"
        for digit in decimal_str:
            if digit == '0' and not decimal_words.endswith(' zero'):
                decimal_words += " zero"
            elif digit != '0':
                decimal_words += " " + ones[int(digit)]
        return (integer_words + decimal_words).strip()
    
    if num < 0:
        return "minus " + number_to_english(str(abs(num)))
    
    if num < 20:
        return ones[num]
    
    if num < 100:
        return tens[num // 10] + ((" " + ones[num % 10]) if num % 10 != 0 else "")
    
    if num < 1000:
        return ones[num // 100] + " hundred" + (" and " + number_to_english(str(num % 100)) if num % 100 != 0 else "")
    
    # 处理1000及以上
    scales = [
        (10**12, "trillion"),
        (10**9, "billion"),
        (10**6, "million"),
        (10**3, "thousand")
    ]
    
    for scale_value, scale_name in scales:
        if num >= scale_value:
            return number_to_english(str(num // scale_value)) + " " + scale_name + (" " + number_to_english(str(num % scale_value)) if num % scale_value != 0 else "")
    
    logger.warning('Unable to process number "%s"...', number_str)
    return tail_pron

def number_to_japanese(number_str):
    """将数字字符串转换为日语罗马音（≤3位语义数值读，≥4位逐位读）"""
    digit_map = {'0': 'zero', '1': 'ichi', '2': 'ni', '3': 'san',
                 '4': 'yon', '5': 'go', '6': 'roku', '7': 'nana',
                 '8': 'hachi', '9': 'kyuu'}
    try:
        if '.' in number_str:
            int_part, dec_part = number_str.split('.', 1)
            int_str = number_to_japanese(int_part.lstrip('0') or '0')
            dec_str = 'ten' + ''.join(digit_map.get(d, '') for d in dec_part)
            return int_str + dec_str
        num = int(number_str)
    except ValueError:
        logger.warning('Unable to process number "%s"...', number_str)
        return tail_pron

    if num < 0:
        return 'mainasu' + number_to_japanese(str(abs(num)))

    digits_str = str(num)
    if len(digits_str) >= 4:
        # 逐位读
        return ''.join(digit_map[d] for d in digits_str)

    # ≤3位：语义数值读
    n = num
    if n == 0:
        return 'zero'

    hundreds_ja = ['', 'hyaku', 'nihyaku', 'sanbyaku', 'yonhyaku', 'gohyaku',
                   'roppyaku', 'nanahyaku', 'happyaku', 'kyuuhyaku']
    tens_ja = ['', '', 'ni', 'san', 'yon', 'go', 'roku', 'nana', 'hachi', 'kyuu']
    ones_ja = ['', 'ichi', 'ni', 'san', 'yon', 'go', 'roku', 'nana', 'hachi', 'kyuu']

    result = ''

    if n >= 100:
        result += hundreds_ja[n // 100]
        n %= 100

    if n >= 20:
        result += tens_ja[n // 10] + 'juu'
        n %= 10
    elif n >= 10:
        result += 'juu'
        n %= 10

    if n > 0:
        result += ones_ja[n]

    return result

def is_english(text):
    return bool(re.match(r'^[a-zA-Z]+$', text))

def is_english_punctuation(char):
    return char == "'" # in string.punctuation

def is_kanji(char):
    return ('\u4E00' <= char <= '\u9FFF' or '\u3400' <= char <= '\u4DBF' or
            '\uF900' <= char <= '\uFAFF' or char == '\u3005')

def is_hiragana(char):
    return '\u3040' <= char <= '\u309F'

def is_katakana(char):
    return '\u30A0' <= char <= '\u30FF'

def is_kana(text):
    """判断字符串是否全为假名（支持多字符如 'きゃ'）"""
    for i in text:
        if not (is_hiragana(i) or is_katakana(i) or i in ['・', '゠', 'ー']):
            return False
    return True

def is_number(char):
    # 目前支持判断单个字符
    if char in newnums:
        return False
    return char.isdigit() # isnumeric

def get_norm_ruby(item):
    # 1:英文, 2:注音结构, 3:东亚文字（假名、中文汉字）, 4:数字, 5:隐式辅助注音
    if item['type'] == 2:
        return item['ruby']
    if item['type'] in (3, 5):
        return item['orig'].lower() if is_english(item['orig']) else item['orig']
    if item['type'] == 1:
        return ''.join([char for char in item['orig'].strip() if not is_english_punctuation(char)]).lower()
    return tail_pron

def get_norm_surface(item):
    if item['type'] in (1,2,3,4,5):
        return item['orig']
    return ''

def min_error_split(target_list, s):
    n = len(s)
    m = len(target_list)
    
    # 初始化 DP 表
    # dp[i][k] 表示处理到字符串位置 i 时，已匹配 k 个目标项的最小错误数
    dp = [[float('inf')] * (m + 1) for _ in range(n + 1)]
    # 记录回溯路径
    backtrack = [[None] * (m + 1) for _ in range(n + 1)]
    
    # 初始状态：空字符串匹配 0 个目标项
    dp[0][0] = 0
    
    # 动态规划填表
    for i in range(n + 1):
        for k in range(m + 1):
            if dp[i][k] == float('inf'):
                continue
                
            # 尝试匹配下一个目标项
            if k < m:
                target = target_list[k]
                # 处理空字符串目标项
                if target == "":
                    # 不消耗任何字符
                    if dp[i][k] < dp[i][k + 1]:
                        dp[i][k + 1] = dp[i][k]
                        backtrack[i][k + 1] = (i, k, "")
                else:
                    # 尝试所有可能的子串
                    for j in range(i + 1, n + 1):
                        segment = s[i:j]
                        # 计算错误成本（0~1 匹配~不匹配）
                        if segment == target:
                            cost = 0
                        elif target == tail_pron:
                            cost = min(len(segment)*0.1, 1)
                        elif segment=='wa' and target=='ha' or segment=='e' and target=='ha':
                            # 此处可添加当て字
                            cost = 0.1
                        else:
                            cost = 1
                        new_cost = dp[i][k] + cost
                        if new_cost < dp[j][k + 1]:
                            dp[j][k + 1] = new_cost
                            backtrack[j][k + 1] = (i, k, segment)
    
    # 回溯找到最佳分割
    if dp[n][m] == float('inf'):
        return None  # 无有效分割
    
    # 从终点回溯
    result = []
    i, k = n, m
    while k > 0:
        prev_i, prev_k, segment = backtrack[i][k]
        result.append(segment)
        i, k = prev_i, prev_k
    
    # 反转结果（因为是从后往前回溯）
    return result[::-1]

def sylla_split(kana_str, sokuon_split=False, hatsuon_split=True):
    kana_list = []
    i = 0
    n = len(kana_str)
    small_kana = ['ゃ', 'ゅ', 'ょ', 'ぁ', 'ぃ', 'ぅ', 'ぇ', 'ぉ', 'ー',
                  'ャ', 'ュ', 'ョ', 'ァ', 'ィ', 'ゥ', 'ェ', 'ォ']
    if not sokuon_split: small_kana += ['っ', 'ッ']
    if not hatsuon_split: small_kana += ['ん', 'ン']
    while i < n:
        current_char = kana_str[i]
        if current_char in small_kana:
            # 仅当前一字符是真假名时才拼合(拗音 きゃ/ュ 等);
            # 标点/数字/空格后的小假名独立成 token, 避免 "、っ" 这类
            # 把促音拼进标点 token 导致空 pron (且丢失 geminate 借音)。
            if i > 0 and (is_hiragana(kana_str[i - 1]) or is_katakana(kana_str[i - 1])):
                kana_list[-1] += current_char
            else:
                kana_list.append(current_char)
            i += 1
        else:
            kana_list.append(current_char)
            i += 1
    return kana_list

def convert_phoneme(ph):
    """去除音素中的重音标记并映射为罗马音"""
    base_ph = ph.rstrip('012') # 移除数字重音标记
    return phoneme_map.get(base_ph, '')

def split_into_syllables_en(phonemes):
    """将英语音素序列拆分为音节"""
    vowels = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 
              'IH', 'IY', 'OW', 'OY', 'UH', 'UW']
    vowel_positions = []
    # 识别元音位置
    for i, ph in enumerate(phonemes):
        base_ph = ph.rstrip('012')
        if base_ph in vowels:
            vowel_positions.append(i)
    if not vowel_positions: return [phonemes]
    
    syllables = []
    prev_vowel_idx = -1

    # 按元音位置拆分音节
    for i, vowel_idx in enumerate(vowel_positions):
        if i == 0:
            # 首个音节
            onset = phonemes[:vowel_idx]
            vowel = [phonemes[vowel_idx]]
            syllables.append(onset + vowel)
            prev_vowel_idx = vowel_idx
        else:
            # 获取两个元音之间的辅音序列
            consonants = phonemes[prev_vowel_idx + 1 : vowel_idx]
            if consonants:
                onset_start = 0
                # 最大节首辅音原则
                if len(consonants) > 1:
                    # 将第一个辅音分配给前一个音节
                    syllables[-1].append(consonants[0])
                    onset_start = 1
                # 剩余辅音分配给后一个音节
                onset = consonants[onset_start:]
                vowel = [phonemes[vowel_idx]]
                syllables.append(onset + vowel)
            else:
                # 没有辅音，直接开始新音节
                syllables.append([phonemes[vowel_idx]])
            prev_vowel_idx = vowel_idx
    
    # 添加尾部剩余辅音到最后一个音节
    if prev_vowel_idx < len(phonemes) - 1:
        trailing = phonemes[prev_vowel_idx + 1:]
        syllables[-1].extend(trailing)
    
    return syllables

def align_syllables_en(a, b):
    """简单对齐表面音节和发音音节"""
    if len(a) > len(b):
        long_list, short_list = a, b
        long_to_short = True
    elif len(b) > len(a):
        long_list, short_list = b, a
        long_to_short = False
    else:
        return list(zip(a, b))
    
    logger.info("Syllable mismatch for '%s': auto-aligned %d→%d segments", ''.join(a), len(long_list), len(short_list))
    n_segments = len(short_list)
    total_elements = len(long_list)
    
    # 计算每段应包含的元素数
    base_size = total_elements // n_segments
    extra = total_elements % n_segments
    
    merged_list = []
    start = 0
    for i in range(n_segments):
        seg_size = base_size + (1 if i >= n_segments-extra else 0) # 后 extra 段多一个元素
        segment = long_list[start:start+seg_size]
        merged = ''.join(segment)
        merged_list.append(merged)
        start += seg_size

    if long_to_short:
        return list(zip(merged_list, short_list))
    else:
        return list(zip(short_list, merged_list))

def process_english_word(word, surf=True):
    if _get_cmu_dict() is None or _get_eng_dic() is None:
        return [(word, word)]
    if word=='a':
        return [('a', 'ei')]
    elif word=='A':
        return [('A', 'ei')]

    hyphenated = _get_eng_dic().inserted(word)
    surface_syllables = hyphenated.split('-')

    word_lower = word.lower()
    # if word_lower=='you':
    #     return [(word, 'iyu')]
    if word_lower not in _get_cmu_dict():
        logger.warning("Word '%s' not in the dictionary...", word)
        direct_syllables = [i.replace("'", '').lower() for i in surface_syllables]
        return list(zip(surface_syllables, direct_syllables))
    
    phonemes = _get_cmu_dict()[word_lower][0]
    syllables_phonemes = split_into_syllables_en(phonemes)
    syllables_romaji = []
    for syl in syllables_phonemes:
        romaji = ''.join(convert_phoneme(p) for p in syl) # p.rstrip('012').lower()
        syllables_romaji.append(romaji)
    
    if not surf:
        return ''.join(syllables_romaji)

    # 对齐表面音节和发音音节
    return align_syllables_en(surface_syllables, syllables_romaji)

def convert_pinyin_to_phonetic(pinyin_str):
    """将拼音转换为表音拼写，赞美AI"""
    # 移除音调数字
    pinyin_str = re.sub(r'[1-5]', '', pinyin_str)
    
    # 特殊整体音节处理
    special_cases = {
        'zhi': 'jru', 'chi': 'chu', 'shi': 'shu', 'ri': 'ru',
        'zi': 'zu', 'ci': 'tsu', 'si': 'su',
        'yi': 'i', 'wu': 'u', 'yu': 'yu',
        'ye': 'ye', 'yue': 'yue', 'yuan': 'yuen',
        'yin': 'in', 'yun': 'yun', 'ying': 'ing'
    }
    
    if pinyin_str in special_cases:
        return special_cases[pinyin_str]
    
    # 分离声母和韵母
    initials = ['b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h', 
                'j', 'q', 'x', 'zh', 'ch', 'sh', 'r', 'z', 'c', 's', 'y', 'w']
    
    # 找出最长的匹配声母
    found_initial = ""
    for initial in sorted(initials, key=len, reverse=True):
        if pinyin_str.startswith(initial):
            found_initial = initial
            break
    
    # 获取韵母部分
    final = pinyin_str[len(found_initial):] if found_initial else pinyin_str
    
    # 转换声母和韵母
    phonetic_initial = PINYIN_TO_PHONETIC.get(found_initial, found_initial)
    phonetic_final = PINYIN_TO_PHONETIC.get(final, final)
    
    # 特殊规则处理
    # 1. i/u/ü 开头的韵母前没有声母时，添加y/w
    if not found_initial:
        if final.startswith('i'):
            phonetic_final = 'y' + phonetic_final
        elif final.startswith('u'):
            phonetic_final = 'w' + phonetic_final
        elif final.startswith('ü'):
            phonetic_final = 'yu' + phonetic_final[2:] if len(phonetic_final) > 2 else 'yu'
    
    # 2. j/q/x 后的 ü 去掉两点
    if found_initial in ['j', 'q', 'x'] and final.startswith('ü'):
        phonetic_final = 'u' + phonetic_final[2:] if len(phonetic_final) > 2 else 'u'
    
    # 组合声母和韵母
    return phonetic_initial + phonetic_final

def hanzi_to_phonetic(text, heteronym=False):
    """将汉字文本转换为表音拼写，可以结合语义处理句子但目前只用来处理单字"""
    if pypinyin is None:
        return [text]  # 中日文混用场景无 pypinyin 时直出原文
    # 获取拼音
    pinyin_list = pinyin(text, style=Style.TONE3, heteronym=heteronym)
    
    result = []
    for item in pinyin_list:
        if not item: # 非汉字字符
            result.append("")
            continue
            
        # 处理多音字
        pronunciations = []
        for py in item:
            phonetic_form = convert_pinyin_to_phonetic(py)
            pronunciations.append(phonetic_form)
        
        # 去重并选择第一个发音
        unique_prons = list(dict.fromkeys(pronunciations))
        result.append(unique_prons[0])
    return result

# 日语语境下的字母表音（アルファベットのカタカナ読み）
_ALPHA_JA_PRON = {
    'a': 'ee',   'b': 'bii',  'c': 'shii',     'd': 'dii',     'e': 'ii',
    'f': 'efu',  'g': 'jii',  'h': 'eichi',    'i': 'ai',      'j': 'jee',
    'k': 'kee',  'l': 'eru',  'm': 'emu',      'n': 'enu',     'o': 'oo',
    'p': 'pii',  'q': 'kyuu', 'r': 'aaru',     's': 'esu',     't': 'tii',
    'u': 'yuu',  'v': 'bui',  'w': 'daburyuu', 'x': 'ekkusu',
    'y': 'wai',  'z': 'zetto',
}


def _is_ja_initialism(word):
    """判断日文行内的拉丁串是否应按日语字母音逐字读（ＪＫ→ジェーケー）。

    日语台本里的大写拉丁串绝大多数是字母缩写，声优按カタカナ英語的字母音朗读
    （ＪＫ→ジェーケー、ＳＮＳ→エスエヌエス、ドＳ→ドエス）。按英语词发音会得到
    严重偏短的 romaji（jk 仅 2 字符 vs 实际 4 拍），污染 CTC 对齐目标长度。

    判据：纯大写 ASCII 字母串；长度 >=3 且被 cmudict 收录者视为真外来词
    （SEX/MAX/HEY），仍走英语发音；其余一律逐字母读。
    """
    if not (word and word.isascii() and word.isalpha() and word.isupper()):
        return False
    if len(word) >= 3:
        cmu = _get_cmu_dict()
        if cmu is not None and word.lower() in cmu:
            return False
    return True


# 全角拉丁 → 半角，统一判定（含 ｗ 网络梗与 ＪＫ 等全角缩写）
_FW2HW = str.maketrans(
    'ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ'
    'ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ',
    'abcdefghijklmnopqrstuvwxyz'
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ')


def _is_warai(token):
    """网络梗 w（草/笑）：token 纯由 w/ｗ 组成（可连写 www）时跳过不发音。

    真实含 w 的英文词（Win/Web/Wow…）还含其它字母，不会被误杀。
    """
    norm = token.translate(_FW2HW).lower()
    return len(norm) > 0 and set(norm) == {'w'}


def _initialism_to_ja(word):
    """ＪＫ → [('J', 'jee'), ('K', 'kee')]，每字母独立成 token 便于对齐。"""
    return [(ch, _ALPHA_JA_PRON[ch.lower()]) for ch in word]


def haruhi_eng_pron(result_list, ja_context=False):
    """英语分音节注音、数字注音：把 type=1(英文)/type=4(数字) 元素补上 pron。

    ja_context=True（日文行）时，大写字母缩写按日语字母音展开而非英语发音。
    """
    new_list = []
    for item in result_list:
        if item.get('type') == 1:
            new_elements = get_norm_surface(item)
            if ja_context:
                hw = new_elements.translate(_FW2HW)
                if _is_warai(hw):
                    continue  # 网络梗 w（草/笑）不发音，跳过该 token
                if _is_ja_initialism(hw):
                    pairs = _initialism_to_ja(hw)
                else:
                    pairs = process_english_word(hw)
            else:
                pairs = process_english_word(new_elements)
            new_list.extend([{'orig': char, 'type': 1, 'pron': pron}
                             for char, pron in pairs])
        elif item.get('type') == 4:
            norm_surface = normalize_numbers(get_norm_surface(item))
            pron = number_to_japanese(norm_surface)
            new_list.append({'orig': get_norm_surface(item), 'type': 4, 'pron': pron})
        else:
            new_list.append(item)
    return new_list


def _parse_ja_tokens(line, lang, sokuon_split, hatsuon_split):
    """解析日文/日英混排行：{振假名}、[隐式注音]、普通字符。"""
    tokens = re.split(r'(\{.*?\}|\[.*?\])', line)
    result = []
    for token in tokens:
        if not token:
            continue
        # 处理振假名结构 {漢字|假名}
        if token.startswith('{') and token.endswith('}'):
            content = token[1:-1]
            parts = content.split('|')
            if len(parts) != 2:
                raise ValueError(f"注音格式错误：{token}")
            kanji, ruby_text = parts
            ruby_text = sylla_split(ruby_text, sokuon_split, hatsuon_split)
            if len(ruby_text) < 1:
                raise ValueError("振假名为空")
            result.append({'orig': kanji, 'type': 2, 'ruby': ruby_text[0]})
            if len(ruby_text) >= 2:
                for i in range(1, len(ruby_text)):
                    result.append({
                        'orig': ruby_text[i],  # 用假名占位，保证表面字符数=音节数
                        'type': 2,
                        'ruby': ruby_text[i]
                    })
        # 隐式注音 [字符|romaji]
        elif token.startswith('[') and token.endswith(']'):
            content = token[1:-1]
            parts = content.split('|')
            if len(parts) != 2:
                raise ValueError(f"注音格式错误：{token}")
            kanji, ruby_text = parts
            if not is_english(ruby_text):
                raise ValueError(f"辅助注音格式错误：{token}")
            result.append({'orig': kanji, 'type': 5, 'pron': ruby_text})
        # 处理普通字符
        else:
            token = sylla_split(token, sokuon_split, hatsuon_split)
            # sylla_split 可能把 ・/゠ 与小假名(っ/ゃ)拼成多字符元素(如 ・っ),
            # 导致下方逐元素遍历时 SEPARATOR_KANA 字符级跳过失效。
            # 先展平成单字符并剔除分隔中点, 所有分支(ja/jaen/zhen)统一生效。
            token = [c for c in ''.join(token) if c not in SEPARATOR_KANA]
            if lang == 'ja':
                for char in token:
                    if char in SEPARATOR_KANA:
                        continue
                    if is_kana(char) or is_english(char):
                        result.append({'orig': char, 'type': 3})
                    # elif is_number(char):
                    #     result.append({'orig': char, 'type': 4})
                    else:
                        result.append({'orig': char, 'type': 0})
            elif lang == 'jaen':
                for char in token:
                    if char in SEPARATOR_KANA:
                        continue
                    if is_kana(char):
                        result.append({'orig': char, 'type': 3})
                    elif is_english(char) or is_english_punctuation(char):
                        if result and result[-1].get('type') == 1:
                            result[-1]['orig'] += char
                        elif is_english(char):
                            result.append({'orig': char, 'type': 1})
                        else:
                            result.append({'orig': char, 'type': 0})
                    elif is_number(char):
                        if result and result[-1].get('type') == 4:
                            result[-1]['orig'] += char
                        else:
                            result.append({'orig': char, 'type': 4})
                    elif char in SYMBOL_TO_PRONS:
                        result.append({'orig': char, 'type': 5, 'pron': SYMBOL_TO_PRONS[char]})
                    else:
                        result.append({'orig': char, 'type': 0})
    return result


def _parse_zhen_tokens(line):
    """解析中文行：[隐式注音]、汉字、英文、数字等。"""
    tokens = re.split(r'(\[.*?\])', line)
    result = []
    for token in tokens:
        if not token:
            continue
        if token.startswith('[') and token.endswith(']'):
            content = token[1:-1]
            parts = content.split('|')
            if len(parts) != 2:
                raise ValueError(f"注音格式错误：{token}")
            kanji, ruby_text = parts
            if not is_english(ruby_text):
                raise ValueError(f"辅助注音格式错误：{token}")
            result.append({'orig': kanji, 'type': 5, 'pron': ruby_text})
        else:
            for char in token:
                if char in SEPARATOR_KANA:
                    continue
                if is_kanji(char):
                    result.append({'orig': char, 'type': 3, 'pron': hanzi_to_phonetic(char)[0]})
                elif is_english(char) or is_english_punctuation(char):
                    if result and result[-1].get('type') == 1:
                        result[-1]['orig'] += char
                    elif is_english(char):
                        result.append({'orig': char, 'type': 1})
                    else:
                        result.append({'orig': char, 'type': 0})
                elif is_number(char):
                    if result and result[-1].get('type') == 4:
                        result[-1]['orig'] += char
                    else:
                        result.append({'orig': char, 'type': 4})
                elif char in SYMBOL_TO_PRONS:
                    result.append({'orig': char, 'type': 5, 'pron': SYMBOL_TO_PRONS[char]})
                else:
                    result.append({'orig': char, 'type': 0})
    return result


def _annotate_pron(result):
    """为 type=0/2/3 元素标注单字罗马音（含促音借音处理）。原地修改并返回。"""
    postpron = None
    for i in range(len(result) - 1, -1, -1):
        if result[i].get('type') in (0, 2, 3):
            ruby_now = get_norm_ruby(result[i])
            if result[i].get('type') != 0 and ruby_now and ruby_now[-1] in ('っ', 'ッ'):
                # 统计末尾连续促音（含触发字符本身），句末/无借音时一并丢弃，
                # 中段有借音时按个数双写后字辅音（stutter 链 グッッ 等也能正确处理）
                run = len(ruby_now) - len(ruby_now.rstrip('っッ'))
                base = ruby_now[:-run] if run < len(ruby_now) else ''
                if postpron and postpron[0].isalpha():
                    # 有后字可借辅音：geminate 双写（中段促音）
                    cons = postpron[0]
                    if cons == 'c':
                        cons = 't'
                    prefix = safe_kks_convert(base) if base else ''
                    pron = prefix + cons * run
                else:
                    # 段末 / 后字无可借辅音：句末促音无声门闭塞对应的独立音素，
                    # 丢弃所有末尾促音仅保留前缀音节（与接符号分支行为一致），
                    # 避免 pykakasi 把孤立 っ 读成幻影 'tsu' 污染 CTC 对齐目标。
                    pron = safe_kks_convert(base) if base else ''
            else:
                pron = safe_kks_convert(ruby_now)
            result[i]['pron'] = pron
        postpron = result[i].get('pron', tail_pron)
    return result


def _fix_ha_he(result):
    """通用读音修正：借助 Janome+pykakasi 修正助词 は→wa、へ→e。原地修改并返回。"""
    line_pron_list = [item['pron'] for item in result]
    line_surface = ''.join([get_norm_surface(i) for i in result])
    # 纯假名/呻吟句无汉字，Janome 可能误判助词（如把感叹词的 は 读成 wa），直接跳过
    if any(is_kanji(c) for c in line_surface):
        try:
            token_phonetic = ''.join([token.phonetic for token in _get_tokenizer().tokenize(line_surface)])
        except Exception:
            token_phonetic = line_surface
        line_kks = _get_kks().convert(token_phonetic)
        line_roma = ''.join([i['hepburn'] for i in line_kks]) if line_kks else ''
        line_roma_proc = min_error_split(line_pron_list, line_roma)
        if line_roma_proc is not None:
            for i in range(len(result)):
                if result[i]['type'] in (2, 3):
                    try:
                        if result[i]['orig'] == 'は' and line_roma_proc[i] == 'wa':
                            result[i]['pron'] = 'wa'
                        elif result[i]['orig'] == 'へ' and line_roma_proc[i] == 'e':
                            result[i]['pron'] = 'e'
                    except Exception:
                        logger.warning('Ignored errors when trying to correct ha and he...')
    return result


def _rescue_isolated_sokuon(result):
    """兜底：整行无任何可发音 token 却含促音时，给首个促音一个 'n' 占位音。

    ASMR 台本常见 「っ…♡」「…っ！」 这类只有促音+标点的喘息行：促音在
    _annotate_pron 里因无后字可借辅音被丢成空串，导致整行 romaji 为空、
    被下游当作不可对齐样本整条丢弃。但音频里这类片段确有发声（吸气/顿挫），
    故补一个鼻音 'n' 作为对齐锚点。

    仅在整行零发音时触发，正常句中/句末促音的既有行为不受影响。
    """
    if any(item.get('pron') for item in result):
        return result
    for item in result:
        if item.get('orig') in ('っ', 'ッ'):
            item['pron'] = 'n'
            break
    return result


def process_haruhi_line(line, lang='jaen', sokuon_split=False, hatsuon_split=True):
    """解析一行注音文本为元素列表。

    lang: 'ja' / 'jaen' / 'zhen' / 'auto'
    """
    if lang == 'auto':  # 后续考虑 langdetect
        for char in line:
            if is_hiragana(char) or is_katakana(char):
                lang = 'jaen'
                break
        else:
            lang = 'zhen'

    if lang in ['ja', 'jaen']:
        result = _parse_ja_tokens(line, lang, sokuon_split, hatsuon_split)
        result = haruhi_eng_pron(result, ja_context=True)
        result = _annotate_pron(result)
        result = _fix_ha_he(result)
        result = _rescue_isolated_sokuon(result)
    elif lang == 'zhen':
        result = _parse_zhen_tokens(line)
        # 数字先给英语吧
        result = haruhi_eng_pron(result)
    else:
        raise ValueError(f"不支持的 lang：{lang}")
    return result



if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s:%(name)s:%(message)s')
    input_string = "{阻|はば}むも[の|n]は{無|な}い {身|み}{勝|かっ}{手|て}に More love, more jump!"
    parsed = process_haruhi_line(input_string)
    print(parsed)