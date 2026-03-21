from .nodes.face.analysis import FaceAnalysisModels, FaceEmbedDistance, FaceSegmentation
from .nodes.face.occluder import OccluderLoader, GeneratePreciseFaceMask
from .nodes.face.cutpaste import FaceCutout, FacePaste, ExtractBoundingBox

from .nodes.face.warp import FaceWarp
from .nodes.face.align import (
    AlignImageByFace,
    RestoreRotatedImage,
    ExtractRotationInfo,
)
from .nodes.face.region import BiSeNetLoader, RegionSelector, GenerateRegionFaceMask
from .nodes.image.files import (
    LoadImagesFromFolder,
    LoadImageFromPath,
    SelectFace,
    LoadImages,
)
from .nodes.image.scale import (
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
    ApexSmartResize,
)
from .nodes.mask.masks import (
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
    MaskFillColor,
)
from .nodes.image.processing import (
    ColorAdjustment,
    ColorTint,
    ColorBlockEffect,
    FlatteningEffect,
    ImageColorMatch,
)
from .nodes.utils.imitation_hue import ImitationHueNode
from .nodes.model.person_mask import PersonSegmenterLoader, PersonMaskGenerator
from .nodes.model.adv_clip import (
    AdvancedCLIPTextEncode,
    AddCLIPSDXLParams,
    AddCLIPSDXLRParams,
    AdvancedCLIPTextEncodeSDXL,
)
from .nodes.utils.misc import (
    DisplayAny,
    Bus,
    RemoveLatentMask,
    SDXLEmptyLatentSizePicker,
)
from .nodes.utils.empty_latent_ratio import EmptyLatentByAspectRatio

from .nodes.inpaint.cutpaste import InpaintCutOut, InpaintPaste, ExtractCutInfo
from .nodes.model.hyperlora import HyperLoRALoadCharLoRANode, HyperLoRASaveCharLoRANode
from .nodes.model.multi_lora import MultiLoraLoader, MultiLoraLoaderModelOnly
from .nodes.image.compare import ImageCompare
from .nodes.text.text import (
    Text_Translation,
    StringConcatenate,
    TextCombine,
    AnimeCharSelect,
    TextToFilename,
)
from .nodes.utils.simple_math import (
    SimpleMathFloat,
    SimpleMathPercent,
    SimpleMathInt,
    SimpleMathSlider,
    SimpleMathSliderLowRes,
    SimpleMathBoolean,
    SimpleMath,
    SimpleMathDual,
    SimpleMathCondition,
    SimpleCondition,
    SimpleComparison,
    ConsoleDebug,
    DebugTensorShape,
    BatchCount,
    Float,
)
from .nodes.text.dropdown import SFTextDropdown

from .nodes.utils.image_edit import TextEncodeQwenImageEdit, TextEncodeQwenImageEditPlus
from .nodes.utils.flux_resolution import FluxResolutionNode

from .nodes.inpaint.cropstitch import InpaintCrop, InpaintStitch, InpaintExtendOutpaint

WEB_DIRECTORY = "web"


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
    "SFAnimeCharSelect": AnimeCharSelect,
    "SFTextToFilename": TextToFilename,
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
    "SFAnimeCharSelect": "SF Anime Char Select",
    "SFTextToFilename": "SF Text To Filename",
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
    # Qwen节点
    "SFTextEncodeQwenImageEdit": "SF Text Encode Qwen Image Edit",
    "SFTextEncodeQwenImageEditPlus": "SF Text Encode Qwen Image Edit Plus",
    # Flux 分辨率节点
    "SFFluxResolution": "SF Flux Resolution Calculator",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
