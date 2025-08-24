import torch
import comfy.model_management
from nodes import MAX_RESOLUTION


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
    def VALIDATE_INPUTS(cls, input_types):
        return True
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
                "image1": ("IMAGE", {"default": None}),
                "image2": ("IMAGE", {"default": None}),
                "image3": ("IMAGE", {"default": None}),
                "text1": ("STRING", {"default": None, "forceInput": True}),
                "text2": ("STRING", {"default": None, "forceInput": True}),
                "text3": ("STRING", {"default": None, "forceInput": True}),
            }
        }

    RETURN_TYPES = ("BUS", "MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "CLIP", "VAE", "IMAGE", "IMAGE", "IMAGE","STRING","STRING", "STRING", )
    RETURN_NAMES = (
        "bus",
        "model",
        "positive",
        "negative",
        "latent",
        "clip",
        "vae",
        "image1",
        "image2",
        "image3",
        "text1",
        "text2",
        "text3",


    )
   
    FUNCTION = "run"
    CATEGORY = _CATEGORY
    DESCRIPTION = "将输入的模型、条件、潜在表示、CLIP和VAE添加到总线上。"

    def run(self, bus=None, model=None, positive=None, negative=None, latent=None, clip=None, vae=None, image1=None, image2=None, image3=None, text1=None, text2=None, text3=None):



        if bus is None:          
            bus = {
                "model": model,
                "positive": positive,
                "negative": negative,
                "latent": latent,
                "clip": clip,
                "vae": vae,
                "image1": image1,
                "image2": image2,
                "image3": image3,
                "text1": text1,
                "text2": text2,
                "text3": text3,



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
        if image1 is not None:            
            bus["image1"] = image1
        if image2 is not None:
            bus["image2"] = image2
        if image3 is not None:
            bus["image3"] = image3
        if text1 is not None:
            bus["text1"] = text1
        if text2 is not None:
            bus["text2"] = text2
        if text3 is not None:
            bus["text3"] = text3

        return (bus, bus["model"], bus["positive"], bus["negative"], bus["latent"], bus["clip"], bus["vae"], bus["image1"], bus["image2"], bus["image3"], bus["text1"], bus["text2"], bus["text3"])


class RemoveLatentMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples": ("LATENT",),}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "execute"

    CATEGORY = _CATEGORY

    def execute(self, samples):
        s = samples.copy()
        if "noise_mask" in s:
            del s["noise_mask"]

        return (s,)


class SDXLEmptyLatentSizePicker:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "resolution": (["704x1408 (0.5)","704x1344 (0.52)","768x1344 (0.57)","768x1280 (0.6)","832x1216 (0.68)","832x1152 (0.72)","896x1152 (0.78)","896x1088 (0.82)","960x1088 (0.88)","960x1024 (0.94)","1024x1024 (1.0)","1024x960 (1.07)","1088x960 (1.13)","1088x896 (1.21)","1152x896 (1.29)","1152x832 (1.38)","1216x832 (1.46)","1280x768 (1.67)","1344x768 (1.75)","1344x704 (1.91)","1408x704 (2.0)","1472x704 (2.09)","1536x640 (2.4)","1600x640 (2.5)","1664x576 (2.89)","1728x576 (3.0)",], {"default": "1024x1024 (1.0)"}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            "width_override": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
            "height_override": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
            }}

    RETURN_TYPES = ("LATENT","INT","INT",)
    RETURN_NAMES = ("LATENT","width","height",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY

    def execute(self, resolution, batch_size, width_override=0, height_override=0):
        width, height = resolution.split(" ")[0].split("x")
        width = width_override if width_override > 0 else int(width)
        height = height_override if height_override > 0 else int(height)

        latent = torch.zeros([batch_size, 4, height // 8, width // 8], device=self.device)

        return ({"samples":latent}, width, height,)
