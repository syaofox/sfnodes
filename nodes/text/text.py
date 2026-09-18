import csv
import os
import re

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


class TextCombine:
    """合并 text_in 和 text，用 delimiter 连接。保留原始值，不清空空字符。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (IO.STRING, {"multiline": True, "default": "", "tooltip": "要合并的文本"}),
                "delimiter": (IO.STRING, {"multiline": False, "default": "", "tooltip": "合并文本的分隔符"}),
                "position": (["后置", "前置"], {"default": "后置", "tooltip": "text 放在 text_in 之前还是之后"}),
            },
            "optional": {
                "text_in": (IO.STRING, {"forceInput": True, "tooltip": "从上游节点输入的文本"}),
            },
        }

    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("combined",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "合并两段文本，中间用分隔符连接"

    def execute(self, text, delimiter, position, text_in=None):
        text = text or ""
        if text_in is None:
            return (text,)

        if position == "前置":
            return (delimiter.join([text, text_in]),)
        else:
            return (delimiter.join([text_in, text]),)


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
                        else "",
                        "tooltip": "选择动漫角色",
                    },
                )
            },
            "optional": {
                "text_in": ("STRING", {"forceInput": True, "tooltip": "额外的提示文本，将附加在角色名之后"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "filename")
    FUNCTION = "func"
    CATEGORY = _CATEGORY
    DESCRIPTION = "从 CSV 数据中选择动漫角色，返回角色提示词和文件名"

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
                "text": ("STRING", {"forceInput": True, "tooltip": "要转换为文件名的文本"}),
                "allow_path": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "on",
                        "label_off": "off",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "将文本转换为合法文件名，替换非法字符；开启 allow_path 时保留 / 与 \\，可输出带目录的完整路径"

    def execute(self, text, allow_path):
        if allow_path:
            filename = re.sub(r'[<>:"|?*]', "", text)
        else:
            filename = re.sub(r'[<>:"/\\|?*]', "", text)
        return (filename,)
