import base64

from comfy.comfy_types.node_typing import IO

from ...sf_utils.spreadsheet import build_outputs, decode_separator, parse_table

_CATEGORY = "sfnodes/text"

_OFFICE_MAGIC = (b"PK\x03\x04", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1")


def _decode_text_input(data):
    """明文直接使用；strict base64 且解出合法 UTF-8 文本时用解码文本。

    解出 Office 二进制魔数（zip / CFB，含 .xlsx/.ods/.xls）时抛明确错误——
    本节点仅支持文本表格，避免把 base64 当 CSV 静默解析出垃圾数据。
    """
    raw = data or ""
    stripped = raw.strip()
    if not stripped:
        return ""
    try:
        decoded = base64.b64decode(stripped, validate=True)
    except Exception:
        return raw
    if decoded.startswith(_OFFICE_MAGIC):
        raise ValueError(
            "SFSpreadsheetOutputList 仅支持文本表格（CSV/TSV/Markdown），不支持 Excel/ODS 文件；"
            "请在表格软件中另存为 CSV 后重新加载"
        )
    try:
        text = decoded.decode("utf-8")
    except UnicodeDecodeError:
        return raw
    if "\x00" in text:
        return raw
    return text


class SFSpreadsheetOutputList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rows_and_cols": (IO.STRING, {
                    "multiline": False,
                    "default": "",
                    "tooltip": "选择器列表（按 separator 分隔，留空=全选）：表头名 / 列名（A、B…ZZZZ）或行号（1…65536）；"
                               "表格行号 1 起、列名 A 起，而 OutputList 逐项输出 0 起（select_nth 同 0 起）",
                }),
                "separator": (IO.STRING, {
                    "multiline": False,
                    "default": ",",
                    "tooltip": "选择器与文本表格的分隔符；支持转义：\\t 制表符、\\n 换行、\\\\ 反斜杠",
                }),
                "is_topdown": ("BOOLEAN", {
                    "default": True,
                    "label_on": "top-down",
                    "label_off": "left-to-right",
                    "tooltip": "迭代方向：top-down 按行（上→下），left-to-right 按列（左→右）",
                }),
                "num_headers": ("INT", {
                    "default": 1, "min": 0, "max": 65536,
                    "tooltip": "前 x 行（top-down）或列（left-to-right）作为表头并从列表跳过；表头同时用作行/列名匹配",
                }),
                "select_nth": ("INT", {
                    "default": -1, "min": -1, "max": 65536,
                    "tooltip": "仅取第 n 项（0 起），-1=全部；可配合 PrimitiveInt 的 increment 逐次生成",
                }),
                "string_or_base64": (IO.STRING, {
                    "multiline": True,
                    "default": "",
                    "tooltip": "CSV/TSV/Markdown 文本，或该文本的 base64；不支持 Excel/ODS 二进制文件",
                }),
            },
        }

    RETURN_TYPES = ("INT", "DICT", "ARRAY", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("count", "values_dict", "values_list", "item_a", "item_b", "item_c", "item_d")
    OUTPUT_IS_LIST = (False, True, True, True, True, True, True)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "从文本表格（CSV/TSV/Markdown）创建多个输出列表：按选择器（表头/列名/行号）与方向逐行或逐列输出" \
                  "字典/列表/前四项；复刻 geroldmeisinger/ComfyUI-outputlists-combiner SpreadsheetOutputList（仅文本格式）"

    def execute(self, rows_and_cols="", separator=",", is_topdown=True, num_headers=1, select_nth=-1, string_or_base64=""):
        empty = (0, [], [], [], [], [], [])
        text = _decode_text_input(string_or_base64).strip()
        if not text:
            return empty
        separator = decode_separator(separator)
        matrix = parse_table(text, separator)
        if not matrix:
            print("[SFSpreadsheetOutputList] 表格解析失败或为空，返回空输出")
            return empty
        return build_outputs(matrix, rows_and_cols, separator, bool(is_topdown), int(num_headers), int(select_nth))
