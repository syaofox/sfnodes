"""长文本分句与分段打包纯逻辑（AuK 长语音节点用；无 torch/ComfyUI 依赖，可直测）。

`split_sentences`：按句末标点（。！？!?… 与换行）切句，保留标点与后置引号/括号；
`split_clauses`：次级切分（，,、；;：:），用于超长句；
`pack_chunks`：按传入的时长估计函数贪心打包为段落（留 margin 余量），单段超限时
先切分句、再按字符二分硬切，保证每段估计时长 ≤ 上限。
"""

import re

# 句末标点 + 可能跟随的收尾引号/括号（保留在句内）。
# 英文 .!? 需后接空白/行尾才算句末（避免切开 3.14、Mr.Smith 这类写法）。
_SENTENCE_END_RE = re.compile(r"(?:[。！？…]+|[.!?]+(?=\s|$))[”’\"』」）)\]]*")
_CLAUSE_END_RE = re.compile(r"[，,、；;：:]+[”’\"』」）)\]]*")


def _is_cjk(char):
    return "\u3400" <= char <= "\u9fff" or char in "，。！？；：、“”‘’（）《》〈〉【】…—～·"


def _join_parts(parts):
    """拼接句子片段：两侧都不是 CJK 时补一个空格（英文句间），其余直接相连。"""
    text = ""
    for part in parts:
        if not part:
            continue
        if text and not _is_cjk(text[-1]) and not _is_cjk(part[0]):
            text += " "
        text += part
    return text


def _split_by_pattern(text, pattern):
    units = []
    start = 0
    for match in pattern.finditer(text):
        piece = text[start:match.end()].strip()
        if piece:
            units.append(piece)
        start = match.end()
    tail = text[start:].strip()
    if tail:
        units.append(tail)
    return units


def split_sentences(text):
    """按行与句末标点切句（保留标点）；空行/空白片段丢弃。"""
    units = []
    for line in str(text or "").splitlines():
        line = line.strip()
        if line:
            units.extend(_split_by_pattern(line, _SENTENCE_END_RE))
    return units


def split_clauses(text):
    """按逗号/顿号/分号/冒号切分（保留标点）。"""
    return _split_by_pattern(str(text or ""), _CLAUSE_END_RE)


def _split_oversize(unit, estimate, limit):
    """把超过 limit 的单元切成多块：先按分句，再按字符二分。"""
    if estimate(unit) <= limit:
        return [unit]
    pieces = []
    for clause in split_clauses(unit):
        if estimate(clause) <= limit:
            pieces.append(clause)
            continue
        remaining = clause
        while remaining:
            if estimate(remaining) <= limit:
                pieces.append(remaining)
                break
            low, high = 1, len(remaining)
            while low < high:
                mid = (low + high + 1) // 2
                if estimate(remaining[:mid]) <= limit:
                    low = mid
                else:
                    high = mid - 1
            pieces.append(remaining[:low])
            remaining = remaining[low:]
    # 硬切可能切出纯标点片段（如 "。"）：并入前一片，避免孤标点成段
    merged = []
    for piece in pieces:
        if merged and not re.search(r"[\w\u3400-\u9fff]", piece):
            merged[-1] += piece
        else:
            merged.append(piece)
    return merged


def pack_chunks(text, estimate, max_seconds, margin=0.08):
    """长文本 → 段落列表 `[{"text", "seconds"}]`（贪心打包，留 margin 比例余量）。

    estimate(text) 返回估计秒数；max_seconds 为单段生成时长上限。
    """
    limit = max(0.1, float(max_seconds) * (1.0 - max(0.0, float(margin))))
    units = []
    for sentence in split_sentences(text):
        units.extend(_split_oversize(sentence, estimate, limit))

    chunks = []
    current = []
    for unit in units:
        if current and estimate(_join_parts(current + [unit])) > limit:
            chunks.append(_join_parts(current))
            current = [unit]
        else:
            current.append(unit)
    if current:
        chunks.append(_join_parts(current))
    return [{"text": chunk, "seconds": estimate(chunk)} for chunk in chunks]
