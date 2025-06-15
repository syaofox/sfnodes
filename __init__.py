from .inpaint_cropandstitch import InpaintCrop, InpaintStitch, InpaintExtendOutpaint
from .face_morph import FaceMorph
from .face_occluder import OccluderLoader, GeneratePreciseFaceMask
from .face_cutandpaste import FaceCutout, FacePaste, ExtractBoundingBox
from .face_analysis import FaceAnalysisModels, FaceEmbedDistance
from .face_warp import FaceWarp
from .face_alginandrotate import (
    AlignImageByFace,
    RestoreRotatedImage,
    ExtractRotationInfo,
)
from .face_region import BiSeNetLoader, RegionSelector, GenerateRegionFaceMask
from .files import LoadImagesFromFolder, LoadImageFromPath, SelectFace, LoadImages
from .image_scale import (
    GetImageSize,
    ImageScalerForSDModels,
    ImageScalerByPixels,
    ImageScaleBySpecifiedSide,
    ComputeImageScaleRatio,
    ImageRotate,
    TrimImageBorders,
    AddImageBorder,
)
from .masks import (
    MaskParams,
    OutlineMask,
    CreateBlurredEdgeMask,
    MaskChange,
    Depth2Mask,
    MaskScaleBy,
    MaskScale,
    MaskPaintArea,
    MaskAdjustGrayscale,
    PreviewMask,
    MaskedFill,
    FillWithReferenceColor,
)
from .image_processing import (
    ColorAdjustment,
    ColorTint,
    ColorBlockEffect,
    FlatteningEffect,
    ImageColorMatch,
)
from .ipadapter import IPAdapterMSLayerWeights, IPAdapterMSTiled
from .person_mask import PersonSegmenterLoader, PersonMaskGenerator
from .adv_clip import (
    AdvancedCLIPTextEncode,
    AddCLIPSDXLParams,
    AddCLIPSDXLRParams,
    AdvancedCLIPTextEncodeSDXL,
)
from .misc import DisplayAny
from .inpaint_cutandpaste import InpaintCutOut, InpaintPaste, ExtractCutInfo

WEB_DIRECTORY = "js"


NODE_CLASS_MAPPINGS = {
    # 局部修复节点
    "SFInpaintCrop": InpaintCrop,
    "SFInpaintStitch": InpaintStitch,
    "SFInpaintExtendOutpaint": InpaintExtendOutpaint,
    # 人脸变形节点
    "SFFaceMorph": FaceMorph,
    # 人脸遮挡节点
    "SFOccluderLoader": OccluderLoader,
    "SFGeneratePreciseFaceMask": GeneratePreciseFaceMask,
    # 人脸区域节点
    "SFBiSeNetLoader": BiSeNetLoader,
    "SFRegionSelector": RegionSelector,
    "SFGenerateRegionFaceMask": GenerateRegionFaceMask,
    # 人脸分析节点
    "SFAlignImageByFace": AlignImageByFace,
    "SFRestoreRotatedImage": RestoreRotatedImage,
    "SFExtractRotationInfo": ExtractRotationInfo,
    "SFFaceCutout": FaceCutout,
    "SFFacePaste": FacePaste,
    "SFExtractBoundingBox": ExtractBoundingBox,
    "SFFaceAnalysisModels": FaceAnalysisModels,
    "SFFaceEmbedDistance": FaceEmbedDistance,
    "SFFaceWarp": FaceWarp,
    # 文件节点
    "SFLoadImagesFromFolder": LoadImagesFromFolder,
    "SFLoadImageFromPath": LoadImageFromPath,
    "SFSelectFace": SelectFace,
    "SFLoadImages": LoadImages,
    # 图片缩放节点
    "SFGetImageSize": GetImageSize,
    "SFImageScalerForSDModels": ImageScalerForSDModels,
    "SFImageScalerByPixels": ImageScalerByPixels,
    "SFImageScaleBySpecifiedSide": ImageScaleBySpecifiedSide,
    "SFComputeImageScaleRatio": ComputeImageScaleRatio,
    "SFImageRotate": ImageRotate,
    "SFTrimImageBorders": TrimImageBorders,
    "SFAddImageBorder": AddImageBorder,
    # 遮罩节点
    "SFMaskParams": MaskParams,
    "SFOutlineMask": OutlineMask,
    "SFCreateBlurredEdgeMask": CreateBlurredEdgeMask,
    "SFMaskChange": MaskChange,
    "SFDepth2Mask": Depth2Mask,
    "SFMaskScaleBy": MaskScaleBy,
    "SFMaskScale": MaskScale,
    "SFMaskPaintArea": MaskPaintArea,
    "SFMaskAdjustGrayscale": MaskAdjustGrayscale,
    "SFPreviewMask": PreviewMask,
    "SFMaskedFill": MaskedFill,
    "SFFillWithReferenceColor": FillWithReferenceColor,
    # 图片处理节点
    "SFColorAdjustment": ColorAdjustment,
    "SFColorTint": ColorTint,
    "SFColorBlockEffect": ColorBlockEffect,
    "SFFlatteningEffect": FlatteningEffect,
    "SFImageColorMatch": ImageColorMatch,
    # IPAdapter节点
    "SFIPAdapterMSLayerWeights": IPAdapterMSLayerWeights,
    "SFIPAdapterMSTiled": IPAdapterMSTiled,
    # 人像分割节点
    "SFPersonSegmenterLoader": PersonSegmenterLoader,
    "SFPersonMaskGenerator": PersonMaskGenerator,
    # 显示节点
    "SFDisplayAny": DisplayAny,
    # 高级CLIP节点
    "SFAdvancedCLIPTextEncode": AdvancedCLIPTextEncode,
    "SFAddCLIPSDXLParams": AddCLIPSDXLParams,
    "SFAddCLIPSDXLRParams": AddCLIPSDXLRParams,
    "SFAdvancedCLIPTextEncodeSDXL": AdvancedCLIPTextEncodeSDXL,

    # 局部修复节点
    "SFInpaintCutOut": InpaintCutOut,
    "SFInpaintPaste": InpaintPaste,
    "SFExtractCutInfo": ExtractCutInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # 局部修复节点
    "SFInpaintCrop": "SF Inpaint Crop",
    "SFInpaintStitch": "SF Inpaint Stitch",
    "SFInpaintExtendOutpaint": "SF Inpaint Extend Outpaint",
    # 人脸变形节点
    "SFFaceMorph": "SF Face Morph",
    # 人脸遮挡节点
    "SFOccluderLoader": "SF Occluder Loader",
    "SFGeneratePreciseFaceMask": "SF Generate PreciseFaceMask",
    # 人脸区域节点
    "BiSeNetLoader": "SF BiSeNet Loader",
    "SFRegionSelector": "SF Region Selector",
    "SFGenerateRegionFaceMask": "SF Generate Region Face Mask",
    # 人脸分析节点
    "SFAlignImageByFace": "SF Align Image By Face",
    "SFRestoreRotatedImage": "SF Restore Rotated Image",
    "SFExtractRotationInfo": "SF Extract Rotation Info",
    "SFFaceCutout": "SF Face Cutout",
    "SFFacePaste": "SF Face Paste",
    "SFExtractBoundingBox": "SF Extract Bounding Box",
    "SFFaceAnalysisModels": "SF Face Analysis Models",
    "SFFaceEmbedDistance": "SF Face Embed Distance",
    "SFFaceWarp": "SF Face Warp",
    # 文件节点
    "SFLoadImagesFromFolder": "SF Load Images From Folder",
    "SFLoadImageFromPath": "SF Load Image From Path",
    "SFSelectFace": "SF Select Face",
    "SFLoadImages": "SF Load Images",
    # 图片缩放节点
    "SFGetImageSize": "SF Get Image Size",
    "SFImageScalerForSDModels": "SF Image Scaler For SD Models",
    "SFImageScalerByPixels": "SF Image Scaler By Pixels",
    "SFImageScaleBySpecifiedSide": "SF Image Scale By Specified Side",
    "SFComputeImageScaleRatio": "SF Compute Image Scale Ratio",
    "SFImageRotate": "SF Image Rotate",
    "SFTrimImageBorders": "SF Trim Image Borders",
    "SFAddImageBorder": "SF Add Image Border",
    # 遮罩节点
    "SFMaskParams": "SF Mask Params",
    "SFOutlineMask": "SF Outline Mask",
    "SFCreateBlurredEdgeMask": "SF Create Blurred Edge Mask",
    "SFMaskChange": "SF Mask Change",
    "SFDepth2Mask": "SF Depth2Mask",
    "SFMaskScaleBy": "SF Mask Scale By",
    "SFMaskScale": "SF Mask Scale",
    "SFMaskPaintArea": "SF Mask Paint Area",
    "SFMaskAdjustGrayscale": "SF Mask Adjust Grayscale",
    "SFPreviewMask": "SF Preview Mask",
    "SFMaskedFill": "SF Masked Fill",
    "SFFillWithReferenceColor": "SF Fill With Reference Color",
    # 图片处理节点
    "SFColorAdjustment": "SF Color Adjustment",
    "SFColorTint": "SF Color Tint",
    "SFColorBlockEffect": "SF Color Block Effect",
    "SFFlatteningEffect": "SF Flattening Effect",
    "SFImageColorMatch": "SF Image Color Match",
    # IPAdapter节点
    "SFIPAdapterMSLayerWeights": "SF IPAdapter MS Layer Weights",
    "SFIPAdapterMSTiled": "SF IPAdapter MS Tiled",
    # 人像分割节点
    "SFPersonSegmenterLoader": "SF Person Segmenter Loader",
    "SFPersonMaskGenerator": "SF Person Mask Generator",
    # 显示节点
    "SFDisplayAny": "SF Display Any",
    # 高级CLIP节点
    "SFAdvancedCLIPTextEncode": "SF Advanced CLIP Text Encode",
    "SFAddCLIPSDXLParams": "SF Add CLIP SDXL Params",
    "SFAddCLIPSDXLRParams": "SF Add CLIP SDXLR Params",
    "SFAdvancedCLIPTextEncodeSDXL": "SF Advanced CLIP Text Encode SDXL",
    # 局部修复节点
    "SFInpaintCutOut": "SF Inpaint Cut Out",
    "SFInpaintPaste": "SF Inpaint Paste",
    "SFExtractCutInfo": "SF Extract Cut Info",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
