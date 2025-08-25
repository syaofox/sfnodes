import csv
from fileinput import filename
from math import fabs
import os
import re

from .utils.translation import  translators
from comfy.comfy_types.node_typing import IO

_CATEGORY = "sfnodes/Text"

def load_csv_data(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
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
                "trans_switch": ("BOOLEAN", {"default": True, "label_on": "on", "label_off": "off"}),
                "translator": (["Niutrans","MyMemory","Alibaba","Baidu","ModernMt","VolcEngine","Iciba","Iflytek","Google","Bing","Lingvanex","Yandex","Itranslate","SysTran","Argos","Apertium","Reverso","Deepl","CloudTranslation","QQTranSmart","TranslateCom","Sogou","Tilde","Caiyun","QQFanyi","TranslateMe","Papago","Mirai","Youdao","Iflyrec","Hujiang","Yeekit","LanguageWire","Elia","Judic","Mglip","Utibet"], {"default": "google"}),
                "trans_text": ("STRING",  {"multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)

    FUNCTION = "func"

    #OUTPUT_NODE = False

    CATEGORY = _CATEGORY

    def func(self, trans_switch, translator, trans_text):
        output_text = ""
        if trans_switch:
            output_text  = translators(text = trans_text, translator = translator.lower())
        else:
            output_text = trans_text
        return (output_text,)


class TextString:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trans_switch": ("BOOLEAN", {"default": False, "label_on": "on", "label_off": "off"}),
                "text": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("string",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY

    def execute(self, trans_switch, text):
        if trans_switch:
            text = translators(text = text)
        return (text,)

class StringConcatenate():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trans_switch": ("BOOLEAN", {"default": True, "label_on": "on", "label_off": "off"}),
                "string_a": (IO.STRING, {"multiline": True}),
                "string_b": (IO.STRING, {"multiline": True}),
                "string_c": (IO.STRING, {"multiline": True}),                
                "delimiter": (IO.STRING, {"multiline": False, "default": ","})
            },
             "optional": {
                "text_in": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = (IO.STRING,IO.STRING,IO.STRING,IO.STRING)
    RETURN_NAMES = ("combined","string_a","string_b","string_c")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY

    def execute(self, trans_switch,string_a, string_b, string_c, delimiter,text_in=''): 
        if trans_switch:
            string_a = translators(text = string_a)
            string_b = translators(text = string_b)
            string_c = translators(text = string_c)
        
        strings = [string_a, string_b, string_c]
        strings = [s for s in strings if s and s.strip()] # type: ignore 
        if text_in:
            strings.append(text_in)
        return delimiter.join(strings),string_a,string_b,string_c



class StringConcatenateLong():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trans_switch": ("BOOLEAN", {"default": True, "label_on": "on", "label_off": "off"}),
                "string_a": (IO.STRING, {"multiline": True}),
                "string_b": (IO.STRING, {"multiline": True}),
                "string_c": (IO.STRING, {"multiline": True}),
                "string_d": (IO.STRING, {"multiline": True}),
                "string_e": (IO.STRING, {"multiline": True}),
                "string_f": (IO.STRING, {"multiline": True}),
                "string_g": (IO.STRING, {"multiline": True}),
                "delimiter": (IO.STRING, {"multiline": False, "default": ","})
            },
             "optional": {
                "text_in": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = (IO.STRING,IO.STRING,IO.STRING,IO.STRING,IO.STRING,IO.STRING,IO.STRING,IO.STRING)
    RETURN_NAMES = ("combined","string_a","string_b","string_c","string_d","string_e","string_f","string_g")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY

    def execute(self, trans_switch,string_a, string_b, string_c, string_d, string_e, string_f, string_g, delimiter,text_in=''):       
        if trans_switch:
            string_a = translators(text = string_a)
            string_b = translators(text = string_b)
            string_c = translators(text = string_c)
            string_d = translators(text = string_d)
            string_e = translators(text = string_e)
            string_f = translators(text = string_f)
            string_g = translators(text = string_g)
        
        strings = [string_a, string_b, string_c, string_d, string_e, string_f, string_g]
        strings = [s for s in strings if s and s.strip()] # type: ignore 
        if text_in:
            strings.append(text_in)
        return delimiter.join(strings),string_a,string_b,string_c,string_d,string_e,string_f,string_g



class AnimeCharSelect:
    @classmethod
    def INPUT_TYPES(cls):
        # 获取当前脚本所在目录，构建数据文件的绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_file = os.path.join(current_dir, 'data', 'characters.csv')
        
        # 读取CSV文件
        cls.character_options = []
        cls.character_options = load_csv_data(data_file)        
        
        return {
            "required": {
                "character": ([option["label"] for option in cls.character_options], {"default": cls.character_options[0]["label"] if cls.character_options else ""})
            },
             "optional": {
                "text_in": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING","STRING")
    RETURN_NAMES = ("prompt","filename")
    FUNCTION = "func"
    CATEGORY = _CATEGORY

    def func(self, character,text_in=''):
        # 根据显示名找到对应的第二列值
        selected_value = ""
        for option in self.character_options:
            if option["label"] == character:
                selected_value = option["value"]
                break
        
        # 对输出值中的括号进行转义处理
        escaped_character = selected_value.replace('(', '\(').replace(')', '\)')
        # 处理特殊字符，让他可以是合法文件名
        filename = re.sub(r'[<>:"/\\|?*]', '', selected_value)
        # 返回转义后的角色名（即第二列的内容）
        if text_in:
            prompt = f"{escaped_character},{text_in}"
        else:
            prompt = f"{escaped_character}"
        return (prompt,filename,)


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
        filename = re.sub(r'[<>:"/\\|?*]', '', text)
        return (filename,)



class BaseTags:
    @classmethod
    def INPUT_TYPES(cls):
        # 为每个标签创建一个布尔复选框和强度调节滑块
        inputs = {}
        for option in cls.tag_options:
            label = option["label"]
            inputs[label] = ("BOOLEAN", {"default": False})
            inputs[f"{label}_weight"] = ("FLOAT", {"default": 1.0, "min": 0, "max": 2, "step": 0.05, "display": "number"})
        
        return {
            "required": {
                **inputs,
                "delimiter": ("STRING", {"default": ",", "multiline": False})
            },
            "optional": {
                "text": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tags",)
    FUNCTION = "func"
    CATEGORY = _CATEGORY
    
    def func(self, **kwargs):
        # 获取分隔符（最后一个参数）
        delimiter = kwargs.pop("delimiter", ",")
        
        # 收集所有选中的标签
        selected_values = []
        for option in self.tag_options:
            label = option["label"]
            if kwargs.get(label, False):
                # 获取对应的强度值
                weight = kwargs.get(f"{label}_weight", 0.0)
                #_替换为,
                value = option["value"].replace("_",",")
                
                # 根据强度值格式化输出
                if weight == 1.0:
                    selected_values.append(value)
                else:
                    #保留小数点后面两位
                    selected_values.append(f"({value}:{weight:.2f})")
        
        # 使用指定分隔符连接选中的标签值
        result = delimiter.join(selected_values)
        # 从kwargs中获取text参数
        text = kwargs.get("text", "")
        # 合并text和result
        if text:
            result = f"{text},{result}"
        return (result,)

class NsfwTags(BaseTags):
    @classmethod
    def INPUT_TYPES(cls):
        # 获取当前脚本所在目录，构建数据文件的绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_file = os.path.join(current_dir, 'data', 'nsfwtags.csv')
        
        # 读取CSV文件
        cls.tag_options = load_csv_data(data_file)
        
        # 调用父类方法生成输入配置
        return super().INPUT_TYPES()


class ExpressionTags(BaseTags):
    @classmethod
    def INPUT_TYPES(cls):
        # 获取当前脚本所在目录，构建数据文件的绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_file = os.path.join(current_dir, 'data','nsfw', 'expression.csv')
        
        # 读取CSV文件
        cls.tag_options = load_csv_data(data_file)
        
        # 调用父类方法生成输入配置
        return super().INPUT_TYPES()

class ForeplayTags(BaseTags):
    @classmethod
    def INPUT_TYPES(cls):
        # 获取当前脚本所在目录，构建数据文件的绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_file = os.path.join(current_dir, 'data','nsfw', 'foreplay.csv')
        
        # 读取CSV文件
        cls.tag_options = load_csv_data(data_file)
        
        # 调用父类方法生成输入配置
        return super().INPUT_TYPES()

class PositionsTags(BaseTags):
    @classmethod
    def INPUT_TYPES(cls):
        # 获取当前脚本所在目录，构建数据文件的绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_file = os.path.join(current_dir, 'data','nsfw', 'positions.csv')
        
        # 读取CSV文件
        cls.tag_options = load_csv_data(data_file)
        
        # 调用父类方法生成输入配置
        return super().INPUT_TYPES()
