import csv
import os
import re

from ...sf_utils.translation import translators
from comfy.comfy_types.node_typing import IO

_CATEGORY = "sfnodes/text"


def load_csv_data(filename):
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                data.append({"label": row[0], "value": row[1]})
    return data


class Text_Translation:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trans_switch": (
                    "BOOLEAN",
                    {"default": False, "label_on": "on", "label_off": "off"},
                ),
                "translator": (
                    [
                        "Niutrans",
                        "MyMemory",
                        "Alibaba",
                        "Baidu",
                        "ModernMt",
                        "VolcEngine",
                        "Iciba",
                        "Iflytek",
                        "Google",
                        "Bing",
                        "Lingvanex",
                        "Yandex",
                        "Itranslate",
                        "SysTran",
                        "Argos",
                        "Apertium",
                        "Reverso",
                        "Deepl",
                        "CloudTranslation",
                        "QQTranSmart",
                        "TranslateCom",
                        "Sogou",
                        "Tilde",
                        "Caiyun",
                        "QQFanyi",
                        "TranslateMe",
                        "Papago",
                        "Mirai",
                        "Youdao",
                        "Iflyrec",
                        "Hujiang",
                        "Yeekit",
                        "LanguageWire",
                        "Elia",
                        "Judic",
                        "Mglip",
                        "Utibet",
                    ],
                    {"default": "google"},
                ),
                "trans_text": ("STRING", {"multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)

    FUNCTION = "func"

    # OUTPUT_NODE = False

    CATEGORY = _CATEGORY

    def func(self, trans_switch, translator, trans_text):
        output_text = ""
        if trans_switch:
            output_text = translators(text=trans_text, translator=translator.lower())
        else:
            output_text = trans_text
        return (output_text,)


_MAX_STRING_SLOTS = 20


class StringConcatenate:
    """字符串拼接节点，支持 1～N 个 string 槽位，由前端 JS 动态控制可见数量。"""

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(1, _MAX_STRING_SLOTS + 1):
            optional[f"string_{i}"] = (IO.STRING, {"multiline": True, "default": ""})
        optional["text_in"] = ("STRING", {"forceInput": True})
        return {
            "required": {
                "trans_switch": (
                    "BOOLEAN",
                    {"default": False, "label_on": "on", "label_off": "off"},
                ),
                "delimiter": (IO.STRING, {"multiline": False, "default": ","}),
            },
            "optional": optional,
        }

    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("combined",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY

    def execute(self, trans_switch, delimiter, **kwargs):
        strings = [
            kwargs.get(f"string_{i}", "") or "" for i in range(1, _MAX_STRING_SLOTS + 1)
        ]
        if trans_switch:
            strings = [translators(text=s) for s in strings]
        strings = [s for s in strings if s and s.strip()] # type: ignore
        text_in = kwargs.get("text_in") or ""
        if text_in:
            strings.insert(0, text_in)
        return (delimiter.join(strings),)


class TextCombine:
    """合并 text_in 和 text，用 delimiter 连接。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (IO.STRING, {"multiline": True, "default": ""}),
                "delimiter": (IO.STRING, {"multiline": False, "default": ","}),
                # text 在 text_in 的前置 / 后置
                "position": (["后置", "前置"], {"default": "后置"}),
            },
            "optional": {
                "text_in": (IO.STRING, {"forceInput": True}),
            },
        }

    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("combined",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY

    def execute(self, text, delimiter, position, text_in=None):
        """
        position:
            - "后置": text 追加在 text_in 之后（兼容旧行为）
            - "前置": text 放在 text_in 之前
        """
        parts = []

        text_clean = text.strip() if text and text.strip() else ""
        text_in_clean = text_in.strip() if text_in and text_in.strip() else ""

        if position == "前置":
            if text_clean:
                parts.append(text_clean)
            if text_in_clean:
                parts.append(text_in_clean)
        else:  # "后置" 或其他非法值都按旧行为处理
            if text_in_clean:
                parts.append(text_in_clean)
            if text_clean:
                parts.append(text_clean)

        return (delimiter.join(parts),)


class AnimeCharSelect:
    @classmethod
    def INPUT_TYPES(cls):
        # 获取当前脚本所在目录，构建数据文件的绝对路径
        current_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        data_file = os.path.join(current_dir, "data", "anime_char", "characters.csv")

        # 读取CSV文件
        cls.character_options = []
        cls.character_options = load_csv_data(data_file)

        return {
            "required": {
                "character": (
                    [option["label"] for option in cls.character_options],
                    {
                        "default": cls.character_options[0]["label"]
                        if cls.character_options
                        else ""
                    },
                )
            },
            "optional": {
                "text_in": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "filename")
    FUNCTION = "func"
    CATEGORY = _CATEGORY

    def func(self, character, text_in=""):
        # 根据显示名找到对应的第二列值
        selected_value = ""
        for option in self.character_options:
            if option["label"] == character:
                selected_value = option["value"]
                break

        # 对输出值中的括号进行转义处理
        escaped_character = selected_value.replace("(", r"\(").replace(")", r"\)")
        # 处理特殊字符，让他可以是合法文件名
        filename = re.sub(r'[<>:"/\\|?*]', "", selected_value)
        # 返回转义后的角色名（即第二列的内容）
        if text_in:
            prompt = f"{escaped_character},{text_in}"
        else:
            prompt = f"{escaped_character}"
        return (
            prompt,
            filename,
        )


class TextToFilename:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY

    def execute(self, text):
        filename = re.sub(r'[<>:"/\\|?*]', "", text)
        return (filename,)
