
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



class Bus:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "bus": ("BUS",{"default": None}),
                "model": ("MODEL",{"default": None}),
                "positive": ("CONDITIONING", {"default": None}),
                "negative": ("CONDITIONING", {"default": None}),
                "latent": ("LATENT", {"default": None}),
                "clip": ("CLIP", {"default": None}),
                "vae": ("VAE", {"default": None}),                
            }
        }

    RETURN_TYPES = ("BUS", "MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "CLIP", "VAE")
    RETURN_NAMES = (
        "bus",
        "model",
        "positive",
        "negative",
        "latent",
        "clip",
        "vae",
    )
   
    FUNCTION = "run"
    CATEGORY = _CATEGORY
    DESCRIPTION = "将输入的模型、条件、潜在表示、CLIP和VAE添加到总线上。"

    def run(self, bus=None, model=None, positive=None, negative=None, latent=None, clip=None, vae=None):

        if bus is None:          
            bus = {
                "model": model,
                "positive": positive,
                "negative": negative,
                "latent": latent,
                "clip": clip,
                "vae": vae,
            }
       
        if model is not None:
            bus["model"] = model
        if positive is not None:
            bus["positive"] = positive
        if negative is not None:
            bus["negative"] = negative
        if latent is not None:
            bus["latent"] = latent
        if clip is not None:
            bus["clip"] = clip
        if vae is not None: 
            bus["vae"] = vae
        return (bus, bus["model"], bus["positive"], bus["negative"], bus["latent"], bus["clip"], bus["vae"])
