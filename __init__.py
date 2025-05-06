from .inpaint_cropandstitch import InpaintCrop
from .inpaint_cropandstitch import InpaintStitch
from .face_morph import FaceMorph
from .face_shape import FaceShaperMatch

WEB_DIRECTORY = "js"

NODE_CLASS_MAPPINGS = {
    "InpaintCrop": InpaintCrop,
    "InpaintStitch": InpaintStitch,
    "FaceMorph": FaceMorph,
    "FaceShaperMatch": FaceShaperMatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InpaintCrop": "SF Inpaint Crop",
    "InpaintStitch": "SF Inpaint Stitch",
    "FaceMorph": "SF Face Morph",
    "FaceShaperMatch": "SF Face Shaper Match",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
