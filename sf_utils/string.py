

def split_text(text, delimiter, filter_empty=False):
    """按分隔符切分文本，处理转义、空分隔符守卫与空项过滤。

    - delimiter is None/"" 时退化为单元素列表（不崩，原版 split("") 会 ValueError）
    - 将字面 "\\n"/"\\t" 转为真实换行/制表（对齐 SFTextConcatenate 与原 LongTextToList）
    - filter_empty 为 True 时过滤掉去空白后为空的项（对齐 SFPromptList skip_empty）
    - 无 ComfyUI 依赖，可 .mjs 镜像测试
    """
    if delimiter is None:
        delimiter = ""
    # 原版仅处理 \\n，此处追加 \\t 对齐 SFTextConcatenate 的双转义约定
    delimiter = delimiter.replace("\\n", "\n").replace("\\t", "\t")
    if delimiter == "":
        t = text or ""
        parts = [t] if t else []
    else:
        parts = (text or "").split(delimiter)
    if filter_empty:
        parts = [p for p in parts if p.strip()]
    return parts


def affix_list(items, prefix="", suffix="", filter_empty=True):
    """为列表每项添加前后缀，支持转义与空项过滤。

    - prefix/suffix 中字面 "\\n"/"\\t" 转为换行/制表（对齐 split_text/ SFTextConcatenate）
    - filter_empty 为 True 时过滤掉去空白后为空的原项（非附加后）
    - items 为 None/非列表时按单项处理；无 ComfyUI 依赖
    """
    if items is None:
        return []
    if isinstance(items, (list, tuple)):
        # INPUT_IS_LIST 场景：可能为单层列表或包裹一层列表
        if len(items) == 1 and isinstance(items[0], (list, tuple)):
            items = items[0]
        # 保持原列表拷贝，转 str
        raw = list(items)
    else:
        raw = [items]
    prefix = (prefix or "").replace("\\n", "\n").replace("\\t", "\t")
    suffix = (suffix or "").replace("\\n", "\n").replace("\\t", "\t")
    out = []
    for it in raw:
        s = "" if it is None else str(it)
        if filter_empty and not s.strip():
            continue
        out.append(f"{prefix}{s}{suffix}")
    return out


def pad_number_text(text, digits):
    """纯整数文本左侧补零到 digits 位，其余原样返回。

    - 仅匹配可选正负号 + 纯数字的整数文本（如 "7"、"-5"、"+3"）；浮点/非数字不补
    - 符号保留在补零结果之外（"-5", digits=2 -> "-05"）
    - digits <= 0 或位数已足够时原样返回；无 ComfyUI 依赖
    """
    if digits is None or digits <= 0:
        return text
    body = text[1:] if text[:1] in ("+", "-") else text
    if not body.isdigit():
        return text
    if len(body) >= digits:
        return text
    sign = text[:1] if body != text else ""
    return sign + body.zfill(digits)


# 判断是否包含中文
def has_chinese_character(string):
    """
    判断字符串中是否包含中文字符
    :param string: 待判断的字符串
    :return: True如果字符串中至少包含一个中文字符，否则返回False
    """
    for char in string:
        if '\u4e00' <= char <= '\u9fff':
            return True
    return False

