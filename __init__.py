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
    LoadImages,
)
from .nodes.image.browser import SFLoadImageBrowser
from .nodes.image.load_images_path import SFLoadImagesPath
from .nodes.image.load_image_resize import SFLoadImageResize
from .nodes.image.resize_image import SFImageResize
from .nodes.image.crop import SFImageCrop, SFImageUncrop
from .nodes.image.outpaint import SFImageOutpaint, SFImageOutpaintStitch
from .nodes.image.tile import SFImageTile, SFImageUntile, SFImageTileInfo
from .nodes.image.batch_index import SFImageBatchIndex
from .nodes.image.save_image_exact import SFSaveImageExact
from .nodes.image.scale import (
    GetImageSize,
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
from .nodes.image.blend import SFImageBlend
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
    MaskFill,
    ImageMaskToTransparency,
    FillWithReferenceColor,
    MaskCrop,
    MaskFillPercentArea,
)
from .nodes.mask.conditional_invert import ConditionalInvertMask
from .nodes.image.processing import (
    ColorAdjustment,
    ColorTint,
    ColorBlockEffect,
    FlatteningEffect,
    ImageColorMatch,
)
from .nodes.image.color_match_points import ImageColorMatchByPoints
from .nodes.image.imitation_hue import ImitationHue
from .nodes.image.lut import SFLoadLUT, SFApplyLUT, SFExtractLUT
from .nodes.image.rfmsr_upscale import SFRFMSRUpscale
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
from .nodes.model.lora_selector import LoraSelector
from .nodes.model.lora_preset import SFLoraPreset
from .nodes.model.lora_stack import SFLoraStack
from .nodes.model.load_diffusion_model import SFLoadDiffusionModel
from .nodes.model.lora_plot import SFLoraPlot, SFLoraPlotImageSaver
from .nodes.model.krea2 import TextEncodeKrea2, Krea2SystemPrompt, SFImageInterrogator
from .nodes.model.regional_lora import SFRegionalLoRA
from .nodes.model.sage_attention import SFPatchSageAttention
from .nodes.image.compare import ImageCompare
from .nodes.image.pause_image import SFPauseImage
from .nodes.image.pause_latent import SFPauseLatent
from .nodes.mask.pause_mask import SFPauseMask
from .nodes.latent.klein_tiled_ksampler import SFKleinTiledKSampler
from .nodes.image import preview_routes  # noqa: F401  # 副作用注册 /api/sfnodes/preview/* 路由
from .nodes import workflow_routes  # noqa: F401  # 副作用注册 /api/sfnodes/workflows/* 路由
from .sf_utils import lora_notes  # noqa: F401  # 副作用注册 /api/sfnodes/lora_notes 路由
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
from .nodes.text.dropdown_value import ValueDropdown
from .nodes.text.text_preset import SFTextPreset
from .nodes.text.replace import SFTextReplace
from .nodes.text.prompt_list import SFPromptList
from .nodes.text.prompt_stack import SFPromptStack
from .nodes.text.concatenate import SFTextConcatenate
from .nodes.text.any_to_string import SFAnyToString
from .nodes.text.prompt_batcher import SFLoadPromptsFromFolder, SFSaveTextToFiles
from .nodes.text.regex_extract import SFTextRegexExtract
from .nodes.text.random_edit_prompt import SFRandomEditPrompt
from .nodes.text.multiangle_camera import SFMultiangleCamera
from .nodes.text.prompt_preset import SFPromptPreset, SFUnpackPromptPreset
from .nodes.text.prompt_tags import SFPromptTags
from .nodes.text.pause_text import SFPauseText
from .nodes.text.find_replace import SFTextFindReplace
from .nodes.text.prompt_reader import SFPromptReader
from .nodes.text.styles_selector import SFStylesSelector  # noqa: F401  # 副作用注册 /api/sfnodes/styles 路由
from .nodes.text import prompt_reader_routes  # noqa: F401  # 副作用注册 /api/sfnodes/prompt_reader/extract 路由
from .nodes.text.long_text_to_list import SFLongTextToList
from .nodes.text.text_list_affix import SFTextListAffix

from .nodes.utils.image_edit import TextEncodeQwenImageEdit, TextEncodeQwenImageEditPlus
from .nodes.utils.flux_resolution import FluxResolution
from .nodes.utils.canvas_size import CanvasSizePreset  # noqa: F401  # 副作用注册 /api/sfnodes/canvas_size_presets 路由
from .nodes.utils.memory_cleanup import VRAMCleanup, RAMCleanup

from .nodes.inpaint.cropstitch import InpaintExtendOutpaint
from .nodes.inpaint.inpaint_editor import SFInpaintCrop, SFInpaintStitch

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
    ComboSelector,
)

WEB_DIRECTORY = "web"


NODE_CLASS_MAPPINGS = {
    # 局部修复节点
    "SFInpaintCrop": SFInpaintCrop,
    "SFInpaintStitch": SFInpaintStitch,
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
    "SFLoadImages": LoadImages,
    "SFLoadImageBrowser": SFLoadImageBrowser,
    "SFLoadImagesPath": SFLoadImagesPath,
    "SFLoadImageResize": SFLoadImageResize,
    "SFImageResize": SFImageResize,
    "SFImageCrop": SFImageCrop,
    "SFImageUncrop": SFImageUncrop,
    "SFImageTile": SFImageTile,
    "SFImageUntile": SFImageUntile,
    "SFImageTileInfo": SFImageTileInfo,
    "SFImageOutpaint": SFImageOutpaint,
    "SFImageOutpaintStitch": SFImageOutpaintStitch,
    "SFImageBatchIndex": SFImageBatchIndex,
    "SFSaveImageExact": SFSaveImageExact,
    # 图片缩放节点
    "SFGetImageSize": GetImageSize,
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
    "SFImageBlend": SFImageBlend,
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
    "SFMaskFill": MaskFill,
    "SFImageMaskToTransparency": ImageMaskToTransparency,
    "SFFillWithReferenceColor": FillWithReferenceColor,
    "SFMaskCrop": MaskCrop,
    "SFMaskFillPercentArea": MaskFillPercentArea,
    "SFConditionalInvertMask": ConditionalInvertMask,
    # 图片处理节点
    "SFColorAdjustment": ColorAdjustment,
    "SFColorTint": ColorTint,
    "SFColorBlockEffect": ColorBlockEffect,
    "SFFlatteningEffect": FlatteningEffect,
    "SFImageColorMatch": ImageColorMatch,
    "SFImageColorMatchByPoints": ImageColorMatchByPoints,
    "SFImitationHue": ImitationHue,
    # LUT 节点
    "SFLoadLUT": SFLoadLUT,
    "SFApplyLUT": SFApplyLUT,
    "SFExtractLUT": SFExtractLUT,
    # RFMSR 超分节点
    "SFRFMSRUpscale": SFRFMSRUpscale,
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
    "SFLoraPreset": SFLoraPreset,
    "SFLoraStack": SFLoraStack,
    "SFRegionalLoRA": SFRegionalLoRA,
    "SFLoraPlot": SFLoraPlot,
    "SFLoraPlotImageSaver": SFLoraPlotImageSaver,
    "SFLoraLoader": LoraLoader,
    "SFLoraLoaderModelOnly": LoraLoaderModelOnly,
    "SFLoraSelector": LoraSelector,
    # 扩散模型加载
    "SFLoadDiffusionModel": SFLoadDiffusionModel,
    # 图片对比节点
    "SFImageCompare": ImageCompare,
    "SFPauseImage": SFPauseImage,
    "SFPauseMask": SFPauseMask,
    "SFPauseLatent": SFPauseLatent,
    # 分块采样节点
    "SFKleinTiledKSampler": SFKleinTiledKSampler,
    # 文本节点
    "SFTextTranslation": TextTranslation,
    "SFTextCombine": TextCombine,
    "SFAnimeCharSelect": AnimeCharSelect,
    "SFTextToFilename": TextToFilename,
    "SFValueDropdown": ValueDropdown,
    "SFTextPreset": SFTextPreset,
    "SFTextReplace": SFTextReplace,
    "SFTextRegexExtract": SFTextRegexExtract,
    "SFPromptList": SFPromptList,
    "SFPromptStack": SFPromptStack,
    "SFTextConcatenate": SFTextConcatenate,
    "SFAnyToString": SFAnyToString,
    "SFLoadPromptsFromFolder": SFLoadPromptsFromFolder,
    "SFSaveTextToFiles": SFSaveTextToFiles,
    "SFRandomEditPrompt": SFRandomEditPrompt,
    "SFMultiangleCamera": SFMultiangleCamera,
    "SFPromptPreset": SFPromptPreset,
    "SFUnpackPromptPreset": SFUnpackPromptPreset,
    "SFPromptTags": SFPromptTags,
    "SFPauseText": SFPauseText,
    "SFTextFindReplace": SFTextFindReplace,
    "SFPromptReader": SFPromptReader,
    "SFStylesSelector": SFStylesSelector,
    "SFLongTextToList": SFLongTextToList,
    "SFTextListAffix": SFTextListAffix,
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
    # 画布分辨率预设节点
    "SFCanvasSizePreset": CanvasSizePreset,
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
    "SFComboSelector": ComboSelector,
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
    "SFLoadImages": "SF Load Images",
    "SFLoadImageBrowser": "SF Load Image Browser",
    "SFLoadImagesPath": "SF Load Images Path",
    "SFLoadImageResize": "SF Load Image Resize",
    "SFImageResize": "SF Image Resize",
    "SFImageCrop": "SF Image Crop",
    "SFImageUncrop": "SF Image Uncrop",
    "SFImageTile": "SF Image Tile",
    "SFImageUntile": "SF Image Untile",
    "SFImageTileInfo": "SF Image Tile Info",
    "SFImageOutpaint": "SF Image Outpaint",
    "SFImageOutpaintStitch": "SF Image Outpaint Stitch",
    "SFImageBatchIndex": "SF Image Batch Index",
    "SFSaveImageExact": "SF Save Image Exact",
    # 图片缩放节点
    "SFGetImageSize": "SF Get Image Size",
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
    "SFImageBlend": "SF Image Blend",
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
    "SFMaskFill": "SF Mask Fill",
    "SFImageMaskToTransparency": "SF Image Mask To Transparency",
    "SFFillWithReferenceColor": "SF Fill With Reference Color",
    "SFMaskCrop": "SF Mask Crop",
    "SFMaskFillPercentArea": "SF Mask Fill Percent Area",
    "SFConditionalInvertMask": "SF Conditional Invert Mask",
    # 图片处理节点
    "SFColorAdjustment": "SF Color Adjustment",
    "SFColorTint": "SF Color Tint",
    "SFColorBlockEffect": "SF Color Block Effect",
    "SFFlatteningEffect": "SF Flattening Effect",
    "SFImageColorMatch": "SF Image Color Match",
    "SFImageColorMatchByPoints": "SF Image Color Match By Points",
    "SFImitationHue": "SF Imitation Hue",
    # LUT 节点
    "SFLoadLUT": "SF Load LUT",
    "SFApplyLUT": "SF Apply LUT",
    "SFExtractLUT": "SF Extract LUT",
    # RFMSR 超分节点
    "SFRFMSRUpscale": "SF RFMSR Upscale",
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
    "SFLoraPreset": "SF LoRA Preset",
    "SFLoraStack": "SF LoRA Stack",
    "SFRegionalLoRA": "SF Regional LoRA (Multi-Character)",
    "SFLoraPlot": "SF LoRA Plot",
    "SFLoraPlotImageSaver": "SF LoRA Plot Image Saver",
    "SFLoraLoader": "SF LoRA Loader",
    "SFLoraLoaderModelOnly": "SF LoRA Loader (Model Only)",
    "SFLoraSelector": "SF LoRA Selector",
    # 扩散模型加载
    "SFLoadDiffusionModel": "SF Load Diffusion Model",
    # 图片对比节点
    "SFImageCompare": "SF Image Compare",
    "SFPauseImage": "SF Pause Image",
    "SFPauseMask": "SF Pause Mask",
    "SFPauseLatent": "SF Pause Latent",
    # 分块采样节点
    "SFKleinTiledKSampler": "SF Klein Tiled KSampler",
    # 文本节点
    "SFTextTranslation": "SF Text Translation",
    "SFTextCombine": "SF Text Combine",
    "SFAnimeCharSelect": "SF Anime Char Select",
    "SFTextToFilename": "SF Text To Filename",
    "SFValueDropdown": "SF Value Dropdown",
    "SFTextPreset": "SF Text Preset",
    "SFTextReplace": "SF Text Replace",
    "SFTextRegexExtract": "SF Text Regex Extract",
    "SFPromptList": "SF Prompt List",
    "SFPromptStack": "SF Prompt Stack",
    "SFTextConcatenate": "SF Text Concatenate",
    "SFAnyToString": "SF Any To String",
    "SFLoadPromptsFromFolder": "SF Load Prompts From Folder",
    "SFSaveTextToFiles": "SF Save Text To Files",
    "SFRandomEditPrompt": "SF Random Edit Prompt",
    "SFMultiangleCamera": "SF Multiangle Camera",
    "SFPromptPreset": "SF Prompt Preset",
    "SFUnpackPromptPreset": "SF Unpack Prompt Preset",
    "SFPromptTags": "SF Prompt Tags",
    "SFPauseText": "SF Pause Text",
    "SFTextFindReplace": "SF Text Find Replace",
    "SFPromptReader": "SF Prompt Reader",
    "SFStylesSelector": "SF Styles Selector",
    "SFLongTextToList": "SF Long Text To List",
    "SFTextListAffix": "SF Text List Affix",
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
    # 画布分辨率预设节点
    "SFCanvasSizePreset": "SF Canvas Size Preset",
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
    "SFComboSelector": "SF Combo Selector",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
