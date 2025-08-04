import csv
from fileinput import filename
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
            }
        }

    RETURN_TYPES = (IO.STRING,IO.STRING,IO.STRING,IO.STRING)
    RETURN_NAMES = ("combined","string_a","string_b","string_c")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY

    def execute(self, trans_switch,string_a, string_b, string_c, delimiter):       
        if trans_switch:
            string_a = translators(text = string_a)
            string_b = translators(text = string_b)
            string_c = translators(text = string_c)
        
        strings = [string_a, string_b, string_c]
        strings = [s for s in strings if s and s.strip()] # type: ignore 
        return delimiter.join(strings),string_a,string_b,string_c,



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
            }
        }

    RETURN_TYPES = (IO.STRING,IO.STRING,IO.STRING,IO.STRING,IO.STRING,IO.STRING,IO.STRING,IO.STRING)
    RETURN_NAMES = ("combined","string_a","string_b","string_c","string_d","string_e","string_f","string_g")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY

    def execute(self, trans_switch,string_a, string_b, string_c, string_d, string_e, string_f, string_g, delimiter):       
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
            }
        }

    RETURN_TYPES = ("STRING","STRING")
    RETURN_NAMES = ("prompt","filename")
    FUNCTION = "func"
    CATEGORY = _CATEGORY

    def func(self, character):
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
        return (escaped_character,filename,)


class TextToFilename:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY

    def execute(self, text):
        filename = re.sub(r'[<>:"/\\|?*]', '', text)
        return (filename,)



class NsfwTags:
    @classmethod
    def INPUT_TYPES(cls):
        # 获取当前脚本所在目录，构建数据文件的绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_file = os.path.join(current_dir, 'data', 'nsfwtags.csv')
        
        # 读取CSV文件
        cls.tag_options = []
        cls.tag_options = load_csv_data(data_file)
                
        return {
            "required": {
                "nsfw_tag": ([option["label"] for option in cls.tag_options], {"default": cls.tag_options[0]["label"] if cls.tag_options else ""})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tag",)
    FUNCTION = "func"
    CATEGORY = _CATEGORY

    def func(self, nsfw_tag):
        # 根据显示名找到对应的英文标签值
        selected_value = ""
        for option in self.tag_options:
            if option["label"] == nsfw_tag:
                selected_value = option["value"]
                break
        
        # 返回英文标签
        return (selected_value,)