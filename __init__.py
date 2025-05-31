from .inpaint_cropandstitch import InpaintCrop
from .inpaint_cropandstitch import InpaintStitch
from .face_morph import FaceMorph
from .face_occluder import OccluderLoader, GeneratePreciseFaceMask
from .face_analysis import AlignImageByFace, FaceCutout, FacePaste, ExtractBoundingBox, FaceAnalysisModels, FaceEmbedDistance,FaceWarp
from .face_region import BiSeNetLoader, RegionSelector, GenerateRegionFaceMask
from .files import LoadImagesFromFolder, LoadImageFromPath, SelectFace, LoadImages
from .image_scale import GetImageSize, ImageScalerForSDModels, ImageScalerByPixels, ImageScaleBySpecifiedSide, ComputeImageScaleRatio, ImageRotate, TrimImageBorders, AddImageBorder
from .masks import OutlineMask, CreateBlurredEdgeMask, MaskChange, Depth2Mask, MaskScaleBy, MaskScale,MaskPaintArea,MaskAdjustGrayscale
from .image_processing import ColorAdjustment, ColorTint, ColorBlockEffect, FlatteningEffect
from .ipadapter import IPAdapterMSLayerWeights, IPAdapterMSTiled
from .person_mask import PersonSegmenterLoader, PersonMaskGenerator

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
    'FaceEmbedDistance': FaceEmbedDistance,
    'FaceWarp': FaceWarp,

    # 文件节点
    'LoadImagesFromFolder': LoadImagesFromFolder,
    'LoadImageFromPath': LoadImageFromPath,
    'SelectFace': SelectFace,
    'LoadImages': LoadImages,

    # 图片缩放节点
    'GetImageSize': GetImageSize,
    'ImageScalerForSDModels': ImageScalerForSDModels,
    'ImageScalerByPixels': ImageScalerByPixels,
    'ImageScaleBySpecifiedSide': ImageScaleBySpecifiedSide,
    'ComputeImageScaleRatio': ComputeImageScaleRatio,
    'ImageRotate': ImageRotate,
    'TrimImageBorders': TrimImageBorders,
    'AddImageBorder': AddImageBorder,

    # 遮罩节点
    'OutlineMask': OutlineMask,
    'CreateBlurredEdgeMask': CreateBlurredEdgeMask,
    'MaskChange': MaskChange,
    'Depth2Mask': Depth2Mask,
    'MaskScaleBy': MaskScaleBy,
    'MaskScale': MaskScale,
    'MaskPaintArea': MaskPaintArea,
    'MaskAdjustGrayscale': MaskAdjustGrayscale,

    # 图片处理节点
    'ColorAdjustment': ColorAdjustment,
    'ColorTint': ColorTint,
    'ColorBlockEffect': ColorBlockEffect,
    'FlatteningEffect': FlatteningEffect,

    # IPAdapter节点
    'IPAdapterMSLayerWeights': IPAdapterMSLayerWeights,
    'IPAdapterMSTiled': IPAdapterMSTiled,

    # 人像分割节点
    'PersonSegmenterLoader': PersonSegmenterLoader,
    'PersonMaskGenerator': PersonMaskGenerator,
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
    'FaceEmbedDistance': 'SF Face Embed Distance',
    'FaceWarp': 'SF Face Warp',
    
    # 文件节点
    'LoadImagesFromFolder': 'SF Load Images From Folder',
    'LoadImageFromPath': 'SF Load Image From Path',
    'SelectFace': 'SF Select Face',
    'LoadImages': 'SF Load Images',

    # 图片缩放节点
    'GetImageSize': 'SF Get Image Size',
    'ImageScalerForSDModels': 'SF Image Scaler For SD Models',
    'ImageScalerByPixels': 'SF Image Scaler By Pixels',
    'ImageScaleBySpecifiedSide': 'SF Image Scale By Specified Side',
    'ComputeImageScaleRatio': 'SF Compute Image Scale Ratio',
    'ImageRotate': 'SF Image Rotate',
    'TrimImageBorders': 'SF Trim Image Borders',
    'AddImageBorder': 'SF Add Image Border',

    # 遮罩节点
    'OutlineMask': 'SF Outline Mask',
    'CreateBlurredEdgeMask': 'SF Create Blurred Edge Mask',
    'MaskChange': 'SF Mask Change',
    'Depth2Mask': 'SF Depth2Mask',
    'MaskScaleBy': 'SF Mask Scale By',
    'MaskScale': 'SF Mask Scale',
    'MaskPaintArea': 'SF Mask Paint Area',
    'MaskAdjustGrayscale': 'SF Mask Adjust Grayscale',
    
    # 图片处理节点
    'ColorAdjustment': 'SF Color Adjustment',
    'ColorTint': 'SF Color Tint',
    'ColorBlockEffect': 'SF Color Block Effect',
    'FlatteningEffect': 'SF Flattening Effect',

    # IPAdapter节点
    'IPAdapterMSLayerWeights': 'SF IPAdapter MS Layer Weights',
    'IPAdapterMSTiled': 'SF IPAdapter MS Tiled',

    # 人像分割节点
    'PersonSegmenterLoader': 'SF Person Segmenter Loader',
    'PersonMaskGenerator': 'SF Person Mask Generator',
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
