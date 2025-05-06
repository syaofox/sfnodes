from .inpaint_cropandstitch import InpaintCrop
from .inpaint_cropandstitch import InpaintStitch
from .face_morph import FaceMorph


WEB_DIRECTORY = "js"

NODE_CLASS_MAPPINGS = {
    "InpaintCrop": InpaintCrop,
    "InpaintStitch": InpaintStitch,
    "FaceMorph": FaceMorph,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InpaintCrop": "SF Inpaint Crop",
    "InpaintStitch": "SF Inpaint Stitch",
    "FaceMorph": "SF Face Morph",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
