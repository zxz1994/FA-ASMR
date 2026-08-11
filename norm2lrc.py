import numpy as np
import re
import unicodedata
import logging

logger = logging.getLogger(__name__)

def parse_time_to_hundredths(time_str):
    match = re.match(r'\[(\d{2}):(\d{2}):(\d{2})\]', time_str)
    minutes, seconds, hundredths = int(match.group(1)), int(match.group(2)), int(match.group(3))
    return minutes * 6000 + seconds * 100 + hundredths

def format_hundredths_to_time_str(total_hundredths):
    minutes = total_hundredths // 6000
    remaining = total_hundredths % 6000
    seconds = remaining // 100
    hundredths = remaining % 100
    return f"[{minutes:02d}:{seconds:02d}:{hundredths:02d}]"

def calculate_length(surface):
    "结合全半角计算字符串的长度"
    length = 0.0
    for char in surface:
        if unicodedata.east_asian_width(char) in ('F', 'W', 'A'):
            length += 1
        else:
            length += 0.5
    return length

def countdown_str_forward(starttime, bpm=60, num=4, symbol='●'):
    t = 6000 / bpm
    if isinstance(starttime, str):
        starttime = parse_time_to_hundredths(starttime)
    result = format_hundredths_to_time_str(starttime)
    for i in range(1, num+1):
        result = format_hundredths_to_time_str(round(max(starttime-i*t,0))) + symbol + result
    return result

def non_silent_head_adjust(result_list, non_silent_ranges):
    '保证乐句完全位于同一个非静音区间'
    if not non_silent_ranges:
        return result_list
    else:
        i = si = 0
        sentences_list = []
        st = None
        while i < len(result_list):
            if result_list[i].get('type') == 0:
                if st:
                    sentences_list.append((si, i-1, st, result_list[i-1].get('end')))
                    st = None
            elif not st:
                si = i
                st = result_list[i].get('start')
            i += 1
        for inds, inde, sst, sen in sentences_list:
            sst = parse_time_to_hundredths(sst)
            sen = parse_time_to_hundredths(sen)
            interval_covered = False
            for ns_start, ns_end in non_silent_ranges:
                if int(ns_start * 100) > sst:
                    break
                # 检查非静音段是否覆盖整个区间
                if int(ns_start * 100) <= sst and int(np.ceil(ns_end * 100)) >= sen:
                    interval_covered = True
                    break
            if not interval_covered:
                end_covered = False
                for i in range(len(non_silent_ranges)):
                    ns_start = int(non_silent_ranges[i][0] * 100)
                    ns_end = int(np.ceil(non_silent_ranges[i][1] * 100))
                    if ns_start > sen:
                        break
                    if ns_start <= sen:
                        if ns_end >= sen:
                            end_covered = True
                            adjust_target = ns_start
                            break
                    elif ns_start <= parse_time_to_hundredths(result_list[inde].get('start')) <= ns_end:
                        end_covered = True
                        adjust_target = ns_start
                        break
                if not end_covered:
                    logger.warning('Errors ignored while trying to correct end sounds...')
                    break
                else:
                    adjust_target = min(parse_time_to_hundredths(result_list[inds]['end']), adjust_target)
                    result_list[inds]['start'] = format_hundredths_to_time_str(adjust_target)
        return result_list

def split_long_segments(elements, max_length=20):
    """
    处理元素列表，确保每两个换行符之间的长度不超过 max_length；
    如果超过，则寻找最合适的空格替换为换行符。原地修改 elements。
    """
    n = len(elements)

    def _seg_len(lo, hi):
        return sum(calculate_length(elements[k].get('orig', '')) for k in range(lo, hi))

    # 1) 按已有 '\n' 将元素切分为若干逻辑段 [lo, hi)
    segments = []
    seg_start = 0
    for i in range(n + 1):
        is_boundary = (i == n) or (
            elements[i].get('type') == 0 and elements[i].get('orig') == '\n'
        )
        if is_boundary:
            if i > seg_start:
                segments.append((seg_start, i))
            seg_start = i + 1

    # 2) 对每段：总长超限时，在空格处插入换行切成若干子段
    for lo, hi in segments:
        total = _seg_len(lo, hi)
        if total <= max_length:
            continue

        # 收集段内空格位置及其到段首的累计长度
        spaces = []  # (index, length_before)
        acc = 0.0
        for k in range(lo, hi):
            if elements[k].get('orig') in (' ', '　') and elements[k].get('type') == 0:
                spaces.append((k, acc))
            acc += calculate_length(elements[k].get('orig', ''))

        if not spaces:
            continue  # 无空格可切，保持原样

        n_cuts = int(total // max_length) + 1
        target = total / n_cuts

        # 从前往后，每接近一个目标长度就在最近的空格处切一刀
        chosen = []
        cursor = target
        for pos, length_before in spaces:
            if length_before >= cursor:
                chosen.append(pos)
                cursor += target
                if len(chosen) == n_cuts - 1:
                    break
        # 兜底：空格太少，用靠前的空格补齐
        if len(chosen) < n_cuts - 1:
            chosen = [p for p, _ in spaces[: n_cuts - 1]]

        for pos in chosen:
            elements[pos]['orig'] = '\n'

def process_main(result_list, tag_offset=-150, bpm=60, beats_per_bar=3):
    result = []
    current_line = ""
    last_end = None
    last_end_time = None

    i = 0
    while i < len(result_list):
        item = result_list[i]

        if ('start' in item and current_line == "" and item['type'] in [1, 2, 3, 4, 5]):
            current_start_time = parse_time_to_hundredths(item['start'])

            if bpm>0 and ((last_end_time and current_start_time - last_end_time > 6000/bpm*beats_per_bar+400) or
                (last_end_time is None and current_start_time > 6000/bpm*beats_per_bar+100)):
                current_line += countdown_str_forward(current_start_time, bpm, beats_per_bar)

        if item['type'] in [1, 3, 4, 5] and 'start' in item or item['type'] == 0 and item['orig']!='\n' and 'start' in item:
            current_line += f"{item['start']}{item['orig']}"
            last_end = item['end']
        elif item['type'] == 2:
            if item['orig'] != '' and 'start' in item:
                current_line += f"{item['start']}{item['orig']}"
            if 'end' in item:
                last_end = item['end']
        elif item['type'] == 0 and item['orig']!='\n' and 'start' not in item:
            if last_end and item['orig'] in (' ', '　'):
                current_line += last_end+item['orig']
                last_end = None
            else:
                current_line += item['orig']
        elif item['type'] == 0 and item['orig']=='\n' and 'start' not in item:
            if 'start' in item:
                current_line += item['start']+item['orig']
                result.append(current_line)
                last_end_time = parse_time_to_hundredths(last_end)
                current_line = ""
                last_end = None
            elif last_end:
                current_line += last_end+item['orig']
                result.append(current_line)
                last_end_time = parse_time_to_hundredths(last_end)
                current_line = ""
                last_end = None
            else:
                current_line += item['orig']
            
        i += 1

    if last_end:
        current_line += last_end
    result.append(current_line)
    if item['orig']!='\n':
        result.append("\n")
    result.append("\n@Offset="+str(tag_offset))
    return "".join(result)

def process_ruby(result_list):
    ruby_annotations = []
    i = 0

    while i < len(result_list):
        item = result_list[i]

        if item['type'] == 2 and item['orig'] != '' and 'start' in item:
            ruby1 = item['orig']
            ruby2 = item['ruby']
            ruby3 = item['start']
            ruby4 = ''

            first_start_time = parse_time_to_hundredths(item['start'])

            j = i + 1
            while j < len(result_list) and result_list[j]['type'] == 2 and result_list[j]['orig'] == '' and 'start' in result_list[j]:
                current_item = result_list[j]
                current_start_time = parse_time_to_hundredths(current_item['start'])
                time_diff = current_start_time - first_start_time
                time_diff_str = format_hundredths_to_time_str(time_diff)
                ruby2 += f"{time_diff_str}{current_item['ruby']}"
                j += 1

            for k in range(len(ruby_annotations) - 1, -1, -1):
                if ruby_annotations[k]['ruby1'] == ruby1:
                    ruby_annotations[k]['ruby4'] = item['start']
                    break

            ruby_annotations.append({'ruby1': ruby1, 'ruby2': ruby2, 'ruby3': ruby3, 'ruby4': ruby4})
            i = j
        else:
            i += 1

    result = []
    for idx, annotation in enumerate(ruby_annotations, 1):
        result.append(f"@Ruby{idx}={annotation['ruby1']},{annotation['ruby2']},{annotation['ruby3']},{annotation['ruby4']}")

    return "\n".join(result)

def process_rlf(result_list):
    current_line = ""
    last_end = None

    i = 0
    while i < len(result_list):
        item = result_list[i]

        if item['type'] in [1, 3, 4, 5] and 'start' in item:
            current_line += f"[1|{item['start'][1:-1]}]{item['orig']}"
            last_end = item['end']
        elif item['type'] == 2 and 'start' in item: # 不考虑加号
            if item['orig'] == '':
                raise ValueError("空字符有注音，rlf生成失败！")
            kana_cnt = 1
            kanji_surf = item['orig']
            struc_str = f"{item['start'][1:-1]}]{item['ruby']}"
            while result_list[i+1]['type'] == 2 and result_list[i+1]['orig'] == '' and 'start' in result_list[i+1]:
                i += 1
                kana_cnt += 1
                struc_str += f"[{result_list[i]['start'][1:-1]}]{result_list[i]['ruby']}"
            kana_cnt = 9 if kana_cnt>9 else kana_cnt
            current_line += '{'+kanji_surf+'|['+str(kana_cnt)+'|'+struc_str+'}'
            last_end = result_list[i].get('end', last_end)
        elif item['type'] == 0 and 'start' in item:
            current_line += f"[10|{item['start'][1:-1]}]{item['orig']}"
            last_end = None
        elif item['type'] == 0 and 'start' not in item:
            if last_end and item['orig'] in ('\n', '', ' ', '　'):
                current_line += f"[10|{last_end[1:-1]}]{item['orig']}"
                last_end = None
            else:
                current_line += item['orig']

        i += 1

    if current_line and last_end:
        current_line += last_end
    return current_line