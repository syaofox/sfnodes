from .inpaint_cropandstitch import InpaintCrop
from .inpaint_cropandstitch import InpaintStitch
from .face_morph import FaceMorph
from .face_occluder import OccluderLoader, GeneratePreciseFaceMask
from .face_analysis import AlignImageByFace, FaceCutout, FacePaste, ExtractBoundingBox, FaceAnalysisModels
from .face_region import BiSeNetLoader, RegionSelector, GenerateRegionFaceMask
from .files import LoadImagesFromFolder

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

    # 人脸区域节点
    'BiSeNetLoader': BiSeNetLoader,
    'RegionSelector': RegionSelector,
    'GenerateRegionFaceMask': GenerateRegionFaceMask,

    # 人脸分析节点
    'AlignImageByFace': AlignImageByFace,
    'FaceCutout': FaceCutout,
    'FacePaste': FacePaste,
    'ExtractBoundingBox': ExtractBoundingBox,
    'FaceAnalysisModels': FaceAnalysisModels,

    # 文件节点
    'LoadImagesFromFolder': LoadImagesFromFolder,

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

    # 人脸区域节点
    'BiSeNetLoader': 'SF BiSeNet Loader',
    'RegionSelector': 'SF Region Selector',
    'GenerateRegionFaceMask': 'SF Generate Region Face Mask',

    # 人脸分析节点
    'AlignImageByFace': 'SF Align Image By Face',
    'FaceCutout': 'SF Face Cutout',
    'FacePaste': 'SF Face Paste',
    'ExtractBoundingBox': 'SF Extract Bounding Box',
    'FaceAnalysisModels': 'SF Face Analysis Models',

    # 文件节点
    'LoadImagesFromFolder': 'SF Load Images From Folder',
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
