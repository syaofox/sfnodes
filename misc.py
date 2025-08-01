from comfy.comfy_types.node_typing import IO

_CATEGORY = "sfnodes/misc"


class DisplayAny:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": (("*", {})),
                "mode": (["raw value", "tensor shape"],),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    OUTPUT_NODE = True

    CATEGORY = _CATEGORY

    def execute(self, input, mode):
        if mode == "tensor shape":
            text = []

            def tensorShape(tensor):
                if isinstance(tensor, dict):
                    for k in tensor:
                        tensorShape(tensor[k])
                elif isinstance(tensor, list):
                    for i in range(len(tensor)):
                        tensorShape(tensor[i])
                elif hasattr(tensor, "shape"):
                    text.append(list(tensor.shape))  # type: ignore

            tensorShape(input)
            input = text

        text = str(input)

        return {"ui": {"text": text}, "result": (text,)}


class StringConcatenate():
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
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

    RETURN_TYPES = (IO.STRING,)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY

    def execute(self, string_a, string_b, string_c, string_d, string_e, string_f, string_g, delimiter):       
        
        strings = [string_a, string_b, string_c, string_d, string_e, string_f, string_g]
        strings = [s for s in strings if s and s.strip()]
        return delimiter.join(strings),
