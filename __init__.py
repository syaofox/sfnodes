from .nodes.face.analysis import FaceAnalysisModels, FaceEmbedDistance, FaceSegmentation
from .nodes.face.occluder import GeneratePreciseFaceMask

from .nodes.face.warp import FaceWarp
from .nodes.face.align import (
    AlignImageByFace,
    RestoreRotatedImage,
    ExtractRotationInfo,
)
from .nodes.face.region import GenerateRegionFaceMask
from .nodes.face.person_mask import SFPersonMask
from .nodes.image.files import (
    LoadImagesFromFolder,
    LoadImageFromPath,
    FaceBankLoader,
    LoadImages,
)
from .nodes.image.browser import SFLoadImageBrowser
from .nodes.image.load_images_path import SFLoadImagesPath
from .nodes.image.batch_index import SFImageBatchIndex
from .nodes.image.scale import (
    GetImageSize,
    ImageScalerForSDModels,
    ImageScalerByPixels,
    ImageScaleBySpecifiedSide,
    ComputeImageScaleRatio,
    ScaleImageToSquare,
    ImageResizePlus,
    ApexSmartResize,
)
from .nodes.image.transform import ImageRotate, TrimImageBorders, AddImageBorder
from .nodes.image.concatenate import (
    ImageConcatenate,
    ImageConcatFromBatch,
)
from .nodes.mask.masks import (
    MaskParams,
    MaskParamsEdges,
    OutlineMask,
    CreateBlurredEdgeMask,
    MaskTransform,
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
from .nodes.mask.conditional_invert import ConditionalInvertMask
from .nodes.image.processing import (
    ColorAdjustment,
    ColorTint,
    ColorBlockEffect,
    FlatteningEffect,
    ImageColorMatch,
)
from .nodes.image.imitation_hue import ImitationHue
from .nodes.image.lut import SFLoadLUT, SFApplyLUT, SFExtractLUT
from .nodes.model.adv_clip import (
    AdvancedCLIPTextEncode,
    AddCLIPSDXLParams,
    AddCLIPSDXLRParams,
    AdvancedCLIPTextEncodeSDXL,
)
from .nodes.utils.misc import (
    DisplayAny,
    RemoveLatentMask,
    SDXLEmptyLatentSizePicker,
)
from .nodes.utils.empty_latent_ratio import EmptyLatentByAspectRatio
from .nodes.utils.seed import SFSeed

from .nodes.inpaint.cutpaste import SFCutout, SFPaste, SFExtractCutInfo
from .nodes.model.hyperlora import HyperLoRALoadCharacter, HyperLoRASaveCharacter
from .nodes.model.lora_loader import LoraLoader
from .nodes.model.lora_loader_model_only import LoraLoaderModelOnly
from .nodes.model.multi_lora import MultiLoraLoader, MultiLoraLoaderModelOnly
from .nodes.model.power_lora_loader import PowerLoraLoader
from .nodes.model.krea2 import TextEncodeKrea2, Krea2SystemPrompt, SFImageInterrogator
from .nodes.model.sage_attention import SFPatchSageAttention
from .nodes.image.compare import ImageCompare
from .nodes.text.text import (
    TextTranslation,

    TextCombine,
    AnimeCharSelect,
    TextToFilename,
)
from .nodes.utils.simple_math import (
    SFNumber,
    SimpleMathSlider,
    SimpleMathSliderLowRes,
    SimpleMathBoolean,
    SimpleMath,

    SimpleMathCondition,
    SimpleComparison,

    BatchCount,
)
from .nodes.text.dropdown import TextDropdown
from .nodes.text.replace import SFTextReplace
from .nodes.text.prompt_list import SFPromptList
from .nodes.text.concatenate import SFTextConcatenate
from .nodes.text.prompt_batcher import SFLoadPromptsFromFolder, SFSaveTextToFiles
from .nodes.text.random_edit_prompt import SFRandomEditPrompt

from .nodes.utils.image_edit import TextEncodeQwenImageEdit, TextEncodeQwenImageEditPlus
from .nodes.utils.flux_resolution import FluxResolution
from .nodes.utils.memory_cleanup import VRAMCleanup, RAMCleanup

from .nodes.inpaint.cropstitch import InpaintCrop, InpaintStitch, InpaintExtendOutpaint

from .nodes.utils.image_orientation import ImageOrientation
from .nodes.utils.workflow_name import SFWorkflowName
from .nodes.utils.path_parse import SFParsePath

from .nodes.logic import (
    AnythingIndexSwitch,
    IsMaskEmpty,
    AnyPack,
    AnyUnpack,
    SFWhileLoopStart,
    SFWhileLoopEnd,
    SFForLoopStart,
    SFForLoopEnd,
    SFBatchAnything,
    SFMathInt,
    SFCompare,
)

WEB_DIRECTORY = "web"


NODE_CLASS_MAPPINGS = {
    # 局部修复节点
    "SFInpaintCrop": InpaintCrop,
    "SFInpaintStitch": InpaintStitch,
    "SFInpaintExtendOutpaint": InpaintExtendOutpaint,
    "SFCutout": SFCutout,
    "SFPaste": SFPaste,
    "SFExtractCutInfo": SFExtractCutInfo,
    # 人脸遮挡节点
    "SFGeneratePreciseFaceMask": GeneratePreciseFaceMask,
    # 人脸区域节点
    "SFGenerateRegionFaceMask": GenerateRegionFaceMask,
    # 人脸分析节点
    "SFAlignImageByFace": AlignImageByFace,
    "SFRestoreRotatedImage": RestoreRotatedImage,
    "SFExtractRotationInfo": ExtractRotationInfo,
    "SFFaceAnalysisModels": FaceAnalysisModels,
    "SFFaceEmbedDistance": FaceEmbedDistance,
    "SFFaceSegmentation": FaceSegmentation,
    "SFFaceWarp": FaceWarp,
    "SFPersonMask": SFPersonMask,
    # 文件节点
    "SFLoadImagesFromFolder": LoadImagesFromFolder,
    "SFLoadImageFromPath": LoadImageFromPath,
    "SFFaceBankLoader": FaceBankLoader,
    "SFLoadImages": LoadImages,
    "SFLoadImageBrowser": SFLoadImageBrowser,
    "SFLoadImagesPath": SFLoadImagesPath,
    "SFImageBatchIndex": SFImageBatchIndex,
    # 图片缩放节点
    "SFGetImageSize": GetImageSize,
    "SFImageScalerForSDModels": ImageScalerForSDModels,
    "SFImageScalerByPixels": ImageScalerByPixels,
    "SFImageScaleBySpecifiedSide": ImageScaleBySpecifiedSide,
    "SFComputeImageScaleRatio": ComputeImageScaleRatio,
    "SFScaleImageToSquare": ScaleImageToSquare,
    "SFImageResizePlus": ImageResizePlus,
    "SFApexSmartResize": ApexSmartResize,
    "SFImageRotate": ImageRotate,
    "SFTrimImageBorders": TrimImageBorders,
    "SFAddImageBorder": AddImageBorder,
    "SFImageConcatenate": ImageConcatenate,
    "SFImageConcatFromBatch": ImageConcatFromBatch,
    # 遮罩节点
    "SFMaskParams": MaskParams,
    "SFMaskParamsEdges": MaskParamsEdges,
    "SFOutlineMask": OutlineMask,
    "SFCreateBlurredEdgeMask": CreateBlurredEdgeMask,
    "SFMaskTransform": MaskTransform,
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
    "SFConditionalInvertMask": ConditionalInvertMask,
    # 图片处理节点
    "SFColorAdjustment": ColorAdjustment,
    "SFColorTint": ColorTint,
    "SFColorBlockEffect": ColorBlockEffect,
    "SFFlatteningEffect": FlatteningEffect,
    "SFImageColorMatch": ImageColorMatch,
    "SFImitationHue": ImitationHue,
    # LUT 节点
    "SFLoadLUT": SFLoadLUT,
    "SFApplyLUT": SFApplyLUT,
    "SFExtractLUT": SFExtractLUT,
    # 显示节点
    "SFDisplayAny": DisplayAny,
    "SFRemoveLatentMask": RemoveLatentMask,
    "SFSDXLEmptyLatentSizePicker": SDXLEmptyLatentSizePicker,
    "SFEmptyLatentByAspectRatio": EmptyLatentByAspectRatio,
    # 高级CLIP节点
    "SFAdvancedCLIPTextEncode": AdvancedCLIPTextEncode,
    "SFAddCLIPSDXLParams": AddCLIPSDXLParams,
    "SFAddCLIPSDXLRParams": AddCLIPSDXLRParams,
    "SFAdvancedCLIPTextEncodeSDXL": AdvancedCLIPTextEncodeSDXL,
    # HyperLoRA节点
    "SFHyperLoRALoadCharacter": HyperLoRALoadCharacter,
    "SFHyperLoRASaveCharacter": HyperLoRASaveCharacter,
    # 多LoRA节点
    "SFMultiLoraLoader": MultiLoraLoader,
    "SFMultiLoraLoaderModelOnly": MultiLoraLoaderModelOnly,
    "SFPowerLoraLoader": PowerLoraLoader,
    "SFLoraLoader": LoraLoader,
    "SFLoraLoaderModelOnly": LoraLoaderModelOnly,
    # 图片对比节点
    "SFImageCompare": ImageCompare,
    # 文本节点
    "SFTextTranslation": TextTranslation,
    "SFTextCombine": TextCombine,
    "SFAnimeCharSelect": AnimeCharSelect,
    "SFTextToFilename": TextToFilename,
    "SFTextDropdown": TextDropdown,
    "SFTextReplace": SFTextReplace,
    "SFPromptList": SFPromptList,
    "SFTextConcatenate": SFTextConcatenate,
    "SFLoadPromptsFromFolder": SFLoadPromptsFromFolder,
    "SFSaveTextToFiles": SFSaveTextToFiles,
    "SFRandomEditPrompt": SFRandomEditPrompt,
    # 简单数学节点
    "SFNumber": SFNumber,
    "SFSimpleMathSlider": SimpleMathSlider,
    "SFSimpleMathSliderLowRes": SimpleMathSliderLowRes,
    "SFSimpleMathBoolean": SimpleMathBoolean,
    "SFSimpleMath": SimpleMath,
    "SFSimpleMathCondition": SimpleMathCondition,
    "SFSimpleComparison": SimpleComparison,
    "SFBatchCount": BatchCount,
    # Qwen节点
    "SFTextEncodeQwenImageEdit": TextEncodeQwenImageEdit,
    "SFTextEncodeQwenImageEditPlus": TextEncodeQwenImageEditPlus,
    # Krea2节点
    "SFTextEncodeKrea2": TextEncodeKrea2,
    "SFKrea2SystemPrompt": Krea2SystemPrompt,
    "SFImageInterrogator": SFImageInterrogator,
    # SageAttention 补丁节点
    "SFPatchSageAttention": SFPatchSageAttention,
    # Flux 分辨率节点
    "SFFluxResolution": FluxResolution,
    # 内存清理节点
    "SFVRAMCleanup": VRAMCleanup,
    "SFRAMCleanup": RAMCleanup,
    # 图像方向节点
    "SFImageOrientation": ImageOrientation,
    # 种子节点
    "SFSeed": SFSeed,
    # 工作流名称节点
    "SFWorkflowName": SFWorkflowName,
    # 路径解析节点
    "SFParsePath": SFParsePath,
    # 逻辑节点
    "SFAnythingIndexSwitch": AnythingIndexSwitch,
    "SFIsMaskEmpty": IsMaskEmpty,
    "SFAnyPack": AnyPack,
    "SFAnyUnpack": AnyUnpack,
    # 循环节点
    "SFWhileLoopStart": SFWhileLoopStart,
    "SFWhileLoopEnd": SFWhileLoopEnd,
    "SFForLoopStart": SFForLoopStart,
    "SFForLoopEnd": SFForLoopEnd,
    "SFBatchAnything": SFBatchAnything,
    "SFMathInt": SFMathInt,
    "SFCompare": SFCompare,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # 局部修复节点
    "SFInpaintCrop": "SF Inpaint Crop",
    "SFInpaintStitch": "SF Inpaint Stitch",
    "SFInpaintExtendOutpaint": "SF Inpaint Extend Outpaint",
    "SFCutout": "SF Cutout",
    "SFPaste": "SF Paste",
    "SFExtractCutInfo": "SF Extract Cut Info",
    # 人脸遮挡节点
    "SFGeneratePreciseFaceMask": "SF Generate Precise Face Mask",
    # 人脸区域节点
    "SFGenerateRegionFaceMask": "SF Generate Region Face Mask",
    # 人脸分析节点
    "SFAlignImageByFace": "SF Align Image By Face",
    "SFRestoreRotatedImage": "SF Restore Rotated Image",
    "SFExtractRotationInfo": "SF Extract Rotation Info",
    "SFFaceAnalysisModels": "SF Face Analysis Models",
    "SFFaceEmbedDistance": "SF Face Embed Distance",
    "SFFaceSegmentation": "SF Face Segmentation",
    "SFFaceWarp": "SF Face Warp",
    "SFPersonMask": "SF Person Mask",
    # 文件节点
    "SFLoadImagesFromFolder": "SF Load Images From Folder",
    "SFLoadImageFromPath": "SF Load Image From Path",
    "SFFaceBankLoader": "SF Face Bank Loader",
    "SFLoadImages": "SF Load Images",
    "SFLoadImageBrowser": "SF Load Image Browser",
    "SFLoadImagesPath": "SF Load Images Path",
    "SFImageBatchIndex": "SF Image Batch Index",
    # 图片缩放节点
    "SFGetImageSize": "SF Get Image Size",
    "SFImageScalerForSDModels": "SF Image Scaler For SD Models",
    "SFImageScalerByPixels": "SF Image Scaler By Pixels",
    "SFImageScaleBySpecifiedSide": "SF Image Scale By Specified Side",
    "SFComputeImageScaleRatio": "SF Compute Image Scale Ratio",
    "SFScaleImageToSquare": "SF Scale Image To Square",
    "SFImageResizePlus": "SF Image Resize Plus",
    "SFApexSmartResize": "SF Apex Smart Resize",
    "SFImageRotate": "SF Image Rotate",
    "SFTrimImageBorders": "SF Trim Image Borders",
    "SFAddImageBorder": "SF Add Image Border",
    "SFImageConcatenate": "SF Image Concatenate",
    "SFImageConcatFromBatch": "SF Image Concat From Batch",
    # 遮罩节点
    "SFMaskParams": "SF Mask Params",
    "SFMaskParamsEdges": "SF Mask Params Edges",
    "SFOutlineMask": "SF Outline Mask",
    "SFCreateBlurredEdgeMask": "SF Create Blurred Edge Mask",
    "SFMaskTransform": "SF Mask Transform",
    "SFDepth2Mask": "SF Depth2Mask",
    "SFMaskScaleBy": "SF Mask Scale By",
    "SFMaskScale": "SF Mask Scale",
    "SFMaskPaintArea": "SF Mask Paint Area",
    "SFMaskAdjustGrayscale": "SF Mask Adjust Grayscale",
    "SFPreviewMask": "SF Preview Mask",
    "SFMaskedFill": "SF Masked Fill",
    "SFImageMaskToTransparency": "SF Image Mask To Transparency",
    "SFFillWithReferenceColor": "SF Fill With Reference Color",
    "SFMaskCrop": "SF Mask Crop",
    "SFMaskFillPercentArea": "SF Mask Fill Percent Area",
    "SFMaskFillColor": "SF Mask Fill Color",
    "SFConditionalInvertMask": "SF Conditional Invert Mask",
    # 图片处理节点
    "SFColorAdjustment": "SF Color Adjustment",
    "SFColorTint": "SF Color Tint",
    "SFColorBlockEffect": "SF Color Block Effect",
    "SFFlatteningEffect": "SF Flattening Effect",
    "SFImageColorMatch": "SF Image Color Match",
    "SFImitationHue": "SF Imitation Hue",
    # LUT 节点
    "SFLoadLUT": "SF Load LUT",
    "SFApplyLUT": "SF Apply LUT",
    "SFExtractLUT": "SF Extract LUT",
    # 显示节点
    "SFDisplayAny": "SF Display Any",
    "SFRemoveLatentMask": "SF Remove Latent Mask",
    "SFSDXLEmptyLatentSizePicker": "SF SDXL Empty Latent Size Picker",
    "SFEmptyLatentByAspectRatio": "SF Empty Latent By Aspect Ratio",
    # 高级CLIP节点
    "SFAdvancedCLIPTextEncode": "SF Advanced CLIP Text Encode",
    "SFAddCLIPSDXLParams": "SF Add CLIP SDXL Params",
    "SFAddCLIPSDXLRParams": "SF Add CLIP SDXLR Params",
    "SFAdvancedCLIPTextEncodeSDXL": "SF Advanced CLIP Text Encode SDXL",
    # HyperLoRA节点
    "SFHyperLoRALoadCharacter": "SF HyperLoRA Load Character",
    "SFHyperLoRASaveCharacter": "SF HyperLoRA Save Character",
    # 多LoRA节点
    "SFMultiLoraLoader": "SF Multi LoRA Loader",
    "SFMultiLoraLoaderModelOnly": "SF Multi LoRA Loader (Model Only)",
    "SFPowerLoraLoader": "SF Power Lora Loader",
    "SFLoraLoader": "SF LoRA Loader",
    "SFLoraLoaderModelOnly": "SF LoRA Loader (Model Only)",
    # 图片对比节点
    "SFImageCompare": "SF Image Compare",
    # 文本节点
    "SFTextTranslation": "SF Text Translation",
    "SFTextCombine": "SF Text Combine",
    "SFAnimeCharSelect": "SF Anime Char Select",
    "SFTextToFilename": "SF Text To Filename",
    "SFTextDropdown": "SF Text Dropdown",
    "SFTextReplace": "SF Text Replace",
    "SFPromptList": "SF Prompt List",
    "SFTextConcatenate": "SF Text Concatenate",
    "SFLoadPromptsFromFolder": "SF Load Prompts From Folder",
    "SFSaveTextToFiles": "SF Save Text To Files",
    "SFRandomEditPrompt": "SF Random Edit Prompt",
    # 简单数学节点
    "SFNumber": "SF Number",
    "SFSimpleMathSlider": "SF Simple Math Slider",
    "SFSimpleMathSliderLowRes": "SF Simple Math Slider Low Res",
    "SFSimpleMathBoolean": "SF Simple Math Boolean",
    "SFSimpleMath": "SF Simple Math",
    "SFSimpleMathCondition": "SF Simple Math Condition",
    "SFSimpleComparison": "SF Simple Comparison",
    "SFBatchCount": "SF Batch Count",
    # Qwen节点
    "SFTextEncodeQwenImageEdit": "SF Text Encode Qwen Image Edit",
    "SFTextEncodeQwenImageEditPlus": "SF Text Encode Qwen Image Edit Plus",
    # Krea2节点
    "SFTextEncodeKrea2": "SF Text Encode (Krea2)",
    "SFKrea2SystemPrompt": "SF Krea2 System Prompt",
    "SFImageInterrogator": "SF Image Interrogator",
    # SageAttention 补丁节点
    "SFPatchSageAttention": "SF Patch Sage Attention",
    # Flux 分辨率节点
    "SFFluxResolution": "SF Flux Resolution Calculator",
    # 内存清理节点
    "SFVRAMCleanup": "SF VRAM Cleanup",
    "SFRAMCleanup": "SF RAM Cleanup",
    # 图像方向节点
    "SFImageOrientation": "SF Image Orientation",
    # 种子节点
    "SFSeed": "SF Seed",
    # 工作流名称节点
    "SFWorkflowName": "SF Workflow Name",
    # 路径解析节点
    "SFParsePath": "SF Parse Path",
    # 逻辑节点
    "SFAnythingIndexSwitch": "SF Anything Index Switch",
    "SFIsMaskEmpty": "SF Is Mask Empty",
    "SFAnyPack": "SF Any Pack",
    "SFAnyUnpack": "SF Any Unpack",
    # 循环节点
    "SFWhileLoopStart": "SF While Loop Start",
    "SFWhileLoopEnd": "SF While Loop End",
    "SFForLoopStart": "SF For Loop Start",
    "SFForLoopEnd": "SF For Loop End",
    "SFBatchAnything": "SF Batch Any",
    "SFMathInt": "SF Math Int",
    "SFCompare": "SF Compare",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
