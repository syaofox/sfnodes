from .inpaint_cropandstitch import InpaintCrop
from .inpaint_cropandstitch import InpaintStitch
from .face_morph import FaceMorph
from .face_occluder import OccluderLoader
from .face_occluder import GeneratePreciseFaceMask
from .face_analysis import AlignImageByFace
from .face_analysis import FaceCutout
from .face_analysis import FacePaste
from .face_analysis import ExtractBoundingBox
from .face_analysis import FaceAnalysisModels



WEB_DIRECTORY = "js"


NODE_CLASS_MAPPINGS = {
    # 局部修复节点
    "InpaintCrop": InpaintCrop,
    "InpaintStitch": InpaintStitch,

    # 人脸变形节点
    "FaceMorph": FaceMorph,
    
    # 人脸遮挡节点
    'OccluderLoader': OccluderLoader,
    'GeneratePreciseFaceMask': GeneratePreciseFaceMask,

    # 人脸分析节点
    'AlignImageByFace': AlignImageByFace,
    'FaceCutout': FaceCutout,
    'FacePaste': FacePaste,
    'ExtractBoundingBox': ExtractBoundingBox,
    'FaceAnalysisModels': FaceAnalysisModels,

    
}

NODE_DISPLAY_NAME_MAPPINGS = { 
    # 局部修复节点
    "InpaintCrop": "SF Inpaint Crop",
    "InpaintStitch": "SF Inpaint Stitch",

    # 人脸变形节点
    "FaceMorph": "SF Face Morph",

    # 人脸遮挡节点
    'OccluderLoader': 'SF Occluder Loader',
    'GeneratePreciseFaceMask': 'SF Generate PreciseFaceMask',

    # 人脸分析节点
    'AlignImageByFace': 'SF Align Image By Face',
    'FaceCutout': 'SF Face Cutout',
    'FacePaste': 'SF Face Paste',
    'ExtractBoundingBox': 'SF Extract Bounding Box',
    'FaceAnalysisModels': 'SF Face Analysis Models',
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
