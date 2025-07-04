from .inpaint_cropandstitch import InpaintCrop, InpaintStitch, InpaintExtendOutpaint
from .face_morph import FaceMorph, FaceReshape
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
    ScaleImageToSquare
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
    MaskCrop,
)
from .image_processing import (
    ColorAdjustment,
    ColorTint,
    ColorBlockEffect,
    FlatteningEffect,
    ImageColorMatch,
)
from .ipadapter import IPAdapterMSLayerWeights, IPAdapterMSTiled,IPAdapterEmbedsMS,IPAdapterEmbedsMSBatch
from .person_mask import PersonSegmenterLoader, PersonMaskGenerator
from .adv_clip import (
    AdvancedCLIPTextEncode,
    AddCLIPSDXLParams,
    AddCLIPSDXLRParams,
    AdvancedCLIPTextEncodeSDXL,
)
from .misc import DisplayAny
from .inpaint_cutandpaste import InpaintCutOut, InpaintPaste, ExtractCutInfo
from .prompt_injection import PromptInjection, PromptInjectionIdx, SimplePromptInjection, AdvancedPromptInjection
from .tag import Tagger, SaveTags, FluxCLIPTextEncode, CaptionAnalyzer
from .hyperlora import HyperLoRALoadCharLoRANode

WEB_DIRECTORY = "js"


NODE_CLASS_MAPPINGS = {
    # 局部修复节点
    "SFInpaintCrop": InpaintCrop,
    "SFInpaintStitch": InpaintStitch,
    "SFInpaintExtendOutpaint": InpaintExtendOutpaint,
    # 人脸变形节点
    "SFFaceMorph": FaceMorph,
    "SFFaceReshape": FaceReshape,
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
    "SFScaleImageToSquare": ScaleImageToSquare,
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
    "SFMaskCrop": MaskCrop,
    # 图片处理节点
    "SFColorAdjustment": ColorAdjustment,
    "SFColorTint": ColorTint,
    "SFColorBlockEffect": ColorBlockEffect,
    "SFFlatteningEffect": FlatteningEffect,
    "SFImageColorMatch": ImageColorMatch,
    # IPAdapter节点
    "SFIPAdapterMSLayerWeights": IPAdapterMSLayerWeights,
    "SFIPAdapterMSTiled": IPAdapterMSTiled,
    "SFIPAdapterEmbedsMS": IPAdapterEmbedsMS,
    "SFIPAdapterEmbedsMSBatch": IPAdapterEmbedsMSBatch,
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
    # 提示注入节点
    "SFPromptInjection": PromptInjection,
    "SFPromptInjectionIdx": PromptInjectionIdx,
    "SFSimplePromptInjection": SimplePromptInjection,
    "SFAdvancedPromptInjection": AdvancedPromptInjection,
    # 标签节点
    "SFTagger": Tagger,
    "SFSaveTags": SaveTags,
    "SFFlux_CLIPTextEncode": FluxCLIPTextEncode,
    "SFCaption_Analyzer": CaptionAnalyzer,

    # HyperLoRA节点
    "SFHyperLoRALoadCharLoRANode": HyperLoRALoadCharLoRANode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # 局部修复节点
    "SFInpaintCrop": "SF Inpaint Crop",
    "SFInpaintStitch": "SF Inpaint Stitch",
    "SFInpaintExtendOutpaint": "SF Inpaint Extend Outpaint",
    # 人脸变形节点
    "SFFaceMorph": "SF Face Morph",
    "SFFaceReshape": "SF Face Reshape",
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
    "SFScaleImageToSquare": "SF Scale Image To Square",
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
    "SFMaskCrop": "SF Mask Crop",
    # 图片处理节点
    "SFColorAdjustment": "SF Color Adjustment",
    "SFColorTint": "SF Color Tint",
    "SFColorBlockEffect": "SF Color Block Effect",
    "SFFlatteningEffect": "SF Flattening Effect",
    "SFImageColorMatch": "SF Image Color Match",
    # IPAdapter节点
    "SFIPAdapterMSLayerWeights": "SF IPAdapter MS Layer Weights",
    "SFIPAdapterMSTiled": "SF IPAdapter MS Tiled",
    "SFIPAdapterEmbedsMS": "SF IPAdapter Embeds MS",
    "SFIPAdapterEmbedsMSBatch": "SF IPAdapter Embeds MS Batch",
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
    # 提示注入节点
    "SFPromptInjection": "SF Prompt Injection",
    "SFPromptInjectionIdx": "SF Prompt Injection Idx",
    "SFSimplePromptInjection": "SF Simple Prompt Injection",
    "SFAdvancedPromptInjection": "SF Advanced Prompt Injection",
    # 标签节点
    "SFTagger": "SF Tagger",
    "SFSaveTags": "SF Save Tags",
    "SFFlux_CLIPTextEncode": "SF Flux CLIP Text Encode",
    "SFCaption_Analyzer": "SF Caption Analyzer",

    # HyperLoRA节点
    "SFHyperLoRALoadCharLoRANode": "SF HyperLoRA Load Char LoRA",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
