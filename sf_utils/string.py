

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

