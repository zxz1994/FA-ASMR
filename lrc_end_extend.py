"""lrc_end_extend.py — 在最终产物 .lrc 上做「行尾延长」后处理。

FA-ASMR 产出的标准 LRC 格式:
    [mm:ss.xx]文本        <- 一句的起点 + 内容
    [mm:ss.xx]            <- 紧跟的空行 = 该句的结束标记 (group_end)

本脚本把每个「结束标记」往后推 extend 秒, 但不超过下一句起点
(下一句起点 = 下一个非空 [mm:ss.xx] 行的时间), 与 GUI 高级设置
"行结束延长" 的语义一致, 但直接在已生成的 .lrc 上改, 不依赖对齐管线。

用法:
    python lrc_end_extend.py 输入.lrc [-o 输出.lrc] [-e 延长秒数]

    -e 默认 1.0 秒 (GUI 推荐 0.5~1.5)。
    -o 省略时原地修改, 并先备份为 输入.lrc.bak (已存在 .bak 则不覆盖)。
"""
import os
import re
import sys
import argparse

# 匹配 [mm:ss.xx] 或 [mm:ss:hh] (xx/hh 为百分秒, 2 位)
_TS = re.compile(r"\[(\d{1,2}):(\d{2})(?:[.:](\d{1,2}))?\]")


def parse_ts(s):
    """' [mm:ss.xx]' -> 秒(float), 失败返回 None"""
    m = _TS.search(s)
    if not m:
        return None
    mm = int(m.group(1))
    ss = int(m.group(2))
    cc = int(m.group(3) or 0)
    return mm * 60 + ss + cc / 100.0


def fmt_ts(sec):
    sec = max(0.0, sec)
    mm = int(sec // 60)
    rem = sec - mm * 60
    ss = int(rem)
    cc = int(round((rem - ss) * 100))
    if cc >= 100:
        ss += 1
        cc -= 100
    if ss >= 60:
        mm += 1
        ss -= 60
    return f"[{mm:02d}:{ss:02d}.{cc:02d}]"


def extend_lrc(text, extend):
    """返回 (新文本, 延长行数)。

    只把「空白行」(空文本的时间戳行, 即每句的结束标记) 往后推 extend 秒,
    封顶到下一句起点; **文本行本身的时间戳绝不改动**。
    """
    lines = text.split("\n")
    parsed = []  # (is_ts, t, rest, raw, m)
    for ln in lines:
        m = _TS.search(ln)
        if not m:
            parsed.append((False, None, None, ln, None))
            continue
        rest = ln[m.end():].strip()
        parsed.append((True, parse_ts(ln), rest, ln, m))

    n = len(parsed)
    changed = 0
    for i in range(n):
        is_ts, t, rest, raw, m = parsed[i]
        if not is_ts or rest != "":
            continue  # 只处理「空白标记行」; 文本行 / 非时间戳行跳过
        # 本空白行之后下一个「文本行」的时间 = 封顶上限 (不越过下一句)
        cap = None
        for k in range(i + 1, n):
            if parsed[k][0] and parsed[k][2] != "":
                cap = parsed[k][1]
                break
        new_t = t + extend
        if cap is not None:
            new_t = min(new_t, cap)
        if new_t > t + 1e-6:
            new_ts = fmt_ts(new_t)
            parsed[i] = (True, new_t, rest, new_ts + raw[m.end():], m)
            changed += 1

    # 安全断言: 任何文本行都应保持原样 (空行延长不应影响文本行时间戳)
    for i in range(n):
        is_ts, t, rest, raw, m = parsed[i]
        if is_ts and rest != "":
            assert raw == lines[i], f"文本行被意外修改: line {i}"

    return "\n".join(p[3] for p in parsed), changed


def main():
    ap = argparse.ArgumentParser(description="在最终 .lrc 上做行尾延长")
    ap.add_argument("input", help="输入 .lrc 路径")
    ap.add_argument("-o", "--output", help="输出路径 (省略=原地, 先备份 .bak)")
    ap.add_argument("-e", "--extend", type=float, default=1.0, help="延长秒数 (默认 1.0)")
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        print(f"[错误] 找不到文件: {args.input}")
        sys.exit(1)

    with open(args.input, encoding="utf-8") as f:
        text = f.read()

    out, n = extend_lrc(text, args.extend)

    outp = args.output or args.input
    if outp == args.input:
        bak = args.input + ".bak"
        if not os.path.exists(bak):
            os.replace(args.input, bak)
            print(f"[备份] {bak}")
    with open(outp, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"[完成] 延长 {n} 行 × {args.extend}s → {outp}")


if __name__ == "__main__":
    main()
