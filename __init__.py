from .inpaint_cropandstitch import InpaintCrop, InpaintStitch, InpaintExtendOutpaint
from .face_occluder import OccluderLoader, GeneratePreciseFaceMask
from .face_cutandpaste import FaceCutout, FacePaste, ExtractBoundingBox
from .face_analysis import FaceAnalysisModels, FaceEmbedDistance, FaceSegmentation

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
    ScaleImageToSquare,
    SFLoadImage,
    SFLoadImageSubfolder,
    SFLoadImageSubfolderSortedByMtime,
    ImageResizePlus,
    ApexSmartResize
)
from .masks import (
    MaskParams,
    MaskParamsEdges,
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
    ImageMaskToTransparency,
    FillWithReferenceColor,
    MaskCrop,
    MaskFillPercentArea,
    MaskFillColor
)
from .image_processing import (
    ColorAdjustment,
    ColorTint,
    ColorBlockEffect,
    FlatteningEffect,
    ImageColorMatch,
)
from .imitation_hue import ImitationHueNode
from .ipadapter import IPAdapterMSLayerWeights, IPAdapterMSTiled,IPAdapterEmbedsMS,IPAdapterEmbedsMSBatch,IPAdapterStyleCompositionTiled
from .person_mask import PersonSegmenterLoader, PersonMaskGenerator
from .adv_clip import (
    AdvancedCLIPTextEncode,
    AddCLIPSDXLParams,
    AddCLIPSDXLRParams,
    AdvancedCLIPTextEncodeSDXL,
)
from .misc import DisplayAny, Bus, RemoveLatentMask,SDXLEmptyLatentSizePicker
from .empty_latent_ratio import EmptyLatentByAspectRatio

from .inpaint_cutandpaste import InpaintCutOut, InpaintPaste, ExtractCutInfo
from .prompt_injection import PromptInjection, PromptInjectionIdx, SimplePromptInjection, AdvancedPromptInjection
from .tag import Tagger, SaveTags, FluxCLIPTextEncode, CaptionAnalyzer
from .hyperlora import HyperLoRALoadCharLoRANode,HyperLoRASaveCharLoRANode
from .multi_lora import MultiLoraLoader, MultiLoraLoaderModelOnly
from .image_compare import ImageCompare
from .text import Text_Translation,StringConcatenate,TextCombine,TextString,AnimeCharSelect,TextToFilename,NsfwTags,ExpressionTags,ForeplayTags,PositionsTags
from .simple_math import SimpleMathFloat,SimpleMathPercent,SimpleMathInt,SimpleMathSlider,SimpleMathSliderLowRes,SimpleMathBoolean,SimpleMath,SimpleMathDual,SimpleMathCondition,SimpleCondition,SimpleComparison,ConsoleDebug,DebugTensorShape,BatchCount,Float
from .text_dropdown import SFTextDropdown

from .wan22_prompt_selector import Wan22PromptSelector
from .qwen import TextEncodeQwenImageEdit, TextEncodeQwenImageEditPlus
from .flux_resolution import FluxResolutionNode

WEB_DIRECTORY = "js"


NODE_CLASS_MAPPINGS = {
    # 局部修复节点
    "SFInpaintCrop": InpaintCrop,
    "SFInpaintStitch": InpaintStitch,
    "SFInpaintExtendOutpaint": InpaintExtendOutpaint,
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
    "SFFaceSegmentation": FaceSegmentation,
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
    "SFLoadImage": SFLoadImage,
    "SFLoadImageSubfolder": SFLoadImageSubfolder,
    "SFLoadImageSubfolderSortedByMtime": SFLoadImageSubfolderSortedByMtime,
    "SFImageResizePlus": ImageResizePlus,
    "SFSmartResize": ApexSmartResize,
    # 遮罩节点
    "SFMaskParams": MaskParams,
    "SFMaskParamsEdges": MaskParamsEdges,
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
    "SFImageMaskToTransparency": ImageMaskToTransparency,
    "SFFillWithReferenceColor": FillWithReferenceColor,
    "SFMaskCrop": MaskCrop,
    "SFMaskFillPercentArea": MaskFillPercentArea,
    "SFMaskFillColor": MaskFillColor,

    # 图片处理节点
    "SFColorAdjustment": ColorAdjustment,
    "SFColorTint": ColorTint,
    "SFColorBlockEffect": ColorBlockEffect,
    "SFFlatteningEffect": FlatteningEffect,
    "SFImageColorMatch": ImageColorMatch,
    "SFImitationHue": ImitationHueNode,
    # IPAdapter节点
    "SFIPAdapterMSLayerWeights": IPAdapterMSLayerWeights,
    "SFIPAdapterMSTiled": IPAdapterMSTiled,
    "SFIPAdapterEmbedsMS": IPAdapterEmbedsMS,
    "SFIPAdapterEmbedsMSBatch": IPAdapterEmbedsMSBatch,
    "SFIPAdapterStyleCompositionTiled": IPAdapterStyleCompositionTiled,
    # 人像分割节点
    "SFPersonSegmenterLoader": PersonSegmenterLoader,
    "SFPersonMaskGenerator": PersonMaskGenerator,
    # 显示节点
    "SFDisplayAny": DisplayAny,
    "SFBus": Bus,
    "SFRemoveLatentMask": RemoveLatentMask,
    "SFSDXLEmptyLatentSizePicker": SDXLEmptyLatentSizePicker,
    "SFEmptyLatentByAspectRatio": EmptyLatentByAspectRatio,

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
    "SFHyperLoRASaveCharLoRANode": HyperLoRASaveCharLoRANode,
    # 多LoRA节点
    "SFMultiLoraLoader": MultiLoraLoader,
    "SFMultiLoraLoaderModelOnly": MultiLoraLoaderModelOnly,
    # 图片对比节点
    "SFImageCompare": ImageCompare,
    # 文本节点
    "SFTextTranslation": Text_Translation,
    "SFStringConcatenate": StringConcatenate,
    "SFTextCombine": TextCombine,
    "SFTextString": TextString,
    "SFAnimeCharSelect": AnimeCharSelect,
    "SFTextToFilename": TextToFilename,
    "SFNsfwTags": NsfwTags,
    "SFExpressionTags": ExpressionTags,
    "SFForeplayTags": ForeplayTags,
    "SFPositionsTags": PositionsTags,
    "SFTextDropdown": SFTextDropdown,
    # 简单数学节点
    "SFSimpleMathFloat": SimpleMathFloat,
    "SFSimpleMathPercent": SimpleMathPercent,
    "SFSimpleMathInt": SimpleMathInt,
    "SFSimpleMathSlider": SimpleMathSlider,
    "SFSimpleMathSliderLowRes": SimpleMathSliderLowRes,
    "SFSimpleMathBoolean": SimpleMathBoolean,
    "SFSimpleMath": SimpleMath,
    "SFSimpleMathDual": SimpleMathDual,
    "SFSimpleMathCondition": SimpleMathCondition,
    "SFSimpleCondition": SimpleCondition,
    "SFSimpleComparison": SimpleComparison,
    "SFConsoleDebug": ConsoleDebug,
    "SFDebugTensorShape": DebugTensorShape,
    "SFBatchCount": BatchCount,
    "SFFloat": Float,
    # 提示词节点
    "SFWan22PromptSelector": Wan22PromptSelector,
    # Qwen节点
    "SFTextEncodeQwenImageEdit": TextEncodeQwenImageEdit,
    "SFTextEncodeQwenImageEditPlus": TextEncodeQwenImageEditPlus,
    # Flux 分辨率节点
    "SFFluxResolution": FluxResolutionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # 局部修复节点
    "SFInpaintCrop": "SF Inpaint Crop",
    "SFInpaintStitch": "SF Inpaint Stitch",
    "SFInpaintExtendOutpaint": "SF Inpaint Extend Outpaint",
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
    "SFFaceSegmentation": "SF Face Segmentation",
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
    "SFLoadImage": "SF Load Image",
    "SFLoadImageSubfolder": "SF Load Image Subfolder",
    "SFLoadImageSubfolderSortedByMtime": "SF Load Image Subfolder Sorted By Mtime",
    "SFImageResizePlus": "SF Image Resize Plus",
    "SFSmartResize": "SF Smart Resize",
    # 遮罩节点
    "SFMaskParams": "SF Mask Params",
    "SFMaskParamsEdges": "SF Mask Params Edges",
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
    "SFImageMaskToTransparency": "SF 图片Mask转透明",
    "SFFillWithReferenceColor": "SF Fill With Reference Color",
    "SFMaskCrop": "SF Mask Crop",
    "SFMaskFillPercentArea": "SF Mask Fill Percent Area",
    "SFMaskFillColor": "SF Mask Fill Color",

    # 图片处理节点
    "SFColorAdjustment": "SF Color Adjustment",
    "SFColorTint": "SF Color Tint",
    "SFColorBlockEffect": "SF Color Block Effect",
    "SFFlatteningEffect": "SF Flattening Effect",
    "SFImageColorMatch": "SF Image Color Match",
    "SFImitationHue": "SF Imitation Hue",
    # IPAdapter节点
    "SFIPAdapterMSLayerWeights": "SF IPAdapter MS Layer Weights",
    "SFIPAdapterMSTiled": "SF IPAdapter MS Tiled",
    "SFIPAdapterEmbedsMS": "SF IPAdapter Embeds MS",
    "SFIPAdapterEmbedsMSBatch": "SF IPAdapter Embeds MS Batch",
    "SFIPAdapterStyleCompositionTiled": "SF IPAdapter Style Composition Tiled",
    # 人像分割节点
    "SFPersonSegmenterLoader": "SF Person Segmenter Loader",
    "SFPersonMaskGenerator": "SF Person Mask Generator",
    # 显示节点
    "SFDisplayAny": "SF Display Any",
    "SFBus": "SF Bus",
    "SFRemoveLatentMask": "SF Remove Latent Mask",
    "SFSDXLEmptyLatentSizePicker": "SF SDXL Empty Latent Size Picker",
    "SFEmptyLatentByAspectRatio": "SF Empty Latent By Aspect Ratio",

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
    "SFHyperLoRASaveCharLoRANode": "SF HyperLoRA Save Char LoRA",
    # 多LoRA节点
    "SFMultiLoraLoader": "SF Multi LoRA Loader",
    "SFMultiLoraLoaderModelOnly": "SF Multi LoRA Loader (Model Only)",
    # 图片对比节点
    "SFImageCompare": "SF Image Compare",
    # 文本节点
    "SFTextTranslation": "SF Text Translation",
    "SFStringConcatenate": "SF String Concatenate",
    "SFTextCombine": "SF Text Combine",
    "SFTextString": "SF Text String",
    "SFAnimeCharSelect": "SF Anime Char Select",
    "SFTextToFilename": "SF Text To Filename",
    "SFNsfwTags": "SF Nsfw Tags",
    "SFExpressionTags": "SF Expression Tags",
    "SFForeplayTags": "SF Foreplay Tags",
    "SFPositionsTags": "SF Positions Tags",
    "SFTextDropdown": "SF Text Dropdown",
    # 简单数学节点
    "SFSimpleMathFloat": "SF Simple Math Float",
    "SFSimpleMathPercent": "SF Simple Math Percent",
    "SFSimpleMathInt": "SF Simple Math Int",
    "SFSimpleMathSlider": "SF Simple Math Slider",
    "SFSimpleMathSliderLowRes": "SF Simple Math Slider Low Res",
    "SFSimpleMathBoolean": "SF Simple Math Boolean",
    "SFSimpleMath": "SF Simple Math",
    "SFSimpleMathDual": "SF Simple Math Dual",
    "SFSimpleMathCondition": "SF Simple Math Condition",
    "SFSimpleCondition": "SF Simple Condition",
    "SFSimpleComparison": "SF Simple Comparison",
    "SFConsoleDebug": "SF Console Debug",
    "SFDebugTensorShape": "SF Debug Tensor Shape",
    "SFBatchCount": "SF Batch Count",
    "SFFloat": "SF Float",
    # 提示词节点
    "SFWan22PromptSelector": "SF Wan2.2 Prompt Selector",
    # Qwen节点
    "SFTextEncodeQwenImageEdit": "SF Text Encode Qwen Image Edit",
    "SFTextEncodeQwenImageEditPlus": "SF Text Encode Qwen Image Edit Plus",
    # Flux 分辨率节点
    "SFFluxResolution": "SF Flux Resolution Calculator",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
