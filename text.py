
from .utils.translation import  translators
from comfy.comfy_types.node_typing import IO

_CATEGORY = "sfnodes/Text"


class Text_Translation:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trans_switch": ("BOOLEAN", {"default": True, "label_on": "on", "label_off": "off"}),
                "trans_text": ("STRING",  {"multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)

    FUNCTION = "func"

    #OUTPUT_NODE = False

    CATEGORY = _CATEGORY

    def func(self, trans_switch, trans_text):
        output_text = ""
        if trans_switch:
            output_text  = translators(text = trans_text)
        else:
            output_text = trans_text
        return (output_text,)



class StringConcatenate():
    @classmethod
    def INPUT_TYPES(s):
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
        strings = [s for s in strings if s and s.strip()]
        return delimiter.join(strings),string_a,string_b,string_c,



class StringConcatenateLong():
    @classmethod
    def INPUT_TYPES(s):
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
        strings = [s for s in strings if s and s.strip()]
        return delimiter.join(strings),string_a,string_b,string_c,string_d,string_e,string_f,string_g
