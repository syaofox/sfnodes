

def split_text(text, delimiter):
    """按分隔符切分文本，处理转义并守卫空分隔符/None。

    - delimiter is None/"" 时退化为单元素列表（不崩，原版 split("") 会 ValueError）
    - 将字面 "\\n"/"\\t" 转为真实换行/制表（对齐 SFTextConcatenate 与原 LongTextToList）
    - 无 ComfyUI 依赖，可 .mjs 镜像测试
    """
    if delimiter is None:
        delimiter = ""
    # 原版仅处理 \\n，此处追加 \\t 对齐 SFTextConcatenate 的双转义约定
    delimiter = delimiter.replace("\\n", "\n").replace("\\t", "\t")
    if delimiter == "":
        t = text or ""
        return [t] if t else []
    return (text or "").split(delimiter)


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

