from .inpaint_cropandstitch import InpaintCrop
from .inpaint_cropandstitch import InpaintStitch
from .face_morph import FaceMorph
from .face_analysis import OccluderLoader
from .face_analysis import GeneratePreciseFaceMask
from .face_analysis import AlignImageByFace
from .face_analysis import FaceCutout
from .face_analysis import FacePaste
from .face_analysis import ExtractBoundingBox
from .face_analysis import FaceAnalysisModels



WEB_DIRECTORY = "js"






NODE_CLASS_MAPPINGS = {
    "InpaintCrop": InpaintCrop,
    "InpaintStitch": InpaintStitch,
    "FaceMorph": FaceMorph,
    'OccluderLoader': OccluderLoader,
    'GeneratePreciseFaceMask': GeneratePreciseFaceMask,
    'AlignImageByFace': AlignImageByFace,
    'FaceCutout': FaceCutout,
    'FacePaste': FacePaste,
    'ExtractBoundingBox': ExtractBoundingBox,
    'FaceAnalysisModels': FaceAnalysisModels,

    
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InpaintCrop": "SF Inpaint Crop",
    "InpaintStitch": "SF Inpaint Stitch",
    "FaceMorph": "SF Face Morph",
    'OccluderLoader': 'SF Occluder Loader',
    'GeneratePreciseFaceMask': 'SF Generate PreciseFaceMask',
    'AlignImageByFace': 'SF Align Image By Face',
    'FaceCutout': 'SF Face Cutout',
    'FacePaste': 'SF Face Paste',
    'ExtractBoundingBox': 'SF Extract Bounding Box',
    'FaceAnalysisModels': 'SF Face Analysis Models',
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
