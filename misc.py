
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


