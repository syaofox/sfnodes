"""SFLoraPlot / SFLoraPlotImageSaver —— LoRA 批量对比节点(复刻
ComfyUI-LoRAPlotNode,行模型改进)。

行模型改进(相对原插件):原插件固定 10 个 LoRA 下拉 + 一个全局
"strengths" 字符串(笛卡尔积);SFLoraPlot 改为动态行——每行一个
LoRA + 独立强度 + 开/关开关,前端可任意增删/复制/排序行(web/sf_lora_plot.js)。
"同一 LoRA 多个强度"的对比 = 复制行改强度。

执行语义(与原插件一致):每个开启的行从同一基础 model/clip 单独应用该
LoRA(行强度同时驱动 model+clip),输出为三个并行列表(MODEL / CLIP /
metadata)。ComfyUI 对 OUTPUT_IS_LIST 的输出会自动为下游节点逐项执行一次
("每组合跑一遍工作流"),metadata 格式 "{文件名}_{强度}" 供
SFLoraPlotImageSaver 打标注。

行状态(文件名/开关/强度)存在浏览器 node.properties.loraStackState,由
graphToPrompt 钩子注入隐藏 LoraLoaderState 输入——与 SFLoraStack 同构,
状态契约复用 sf_utils/lora_reader.py 的 parse_state(前端
web/sf_lora_stack_core.js 1:1 镜像)。Python 侧因此零新状态代码。
"""
import os

import comfy.sd
import comfy.utils
import folder_paths

from ...sf_utils import lora_plot as P
from ...sf_utils import lora_reader as R
from ...sf_utils.lora_cache import LoraCache
from ...sf_utils.image_convert import tensor2pil, pil2tensor
from ...sf_utils.logger import get_logger

logger = get_logger(__name__)

_CATEGORY = "sfnodes/model"


class SFLoraPlot:
    DESCRIPTION = (
        "批量对比 LoRA:每个开启的行从同一基础模型单独应用一个 LoRA "
        "(行强度同时驱动模型与 CLIP),输出为列表——ComfyUI 会为每一组合"
        "自动跑一遍下游工作流。Add LoRA 添加行;每行有独立开/关开关与"
        "强度;右键行可上移/下移/复制/删除(复制行改强度即可对比同一 "
        "LoRA 的不同强度)。metadata 输出与 SFLoraPlot Image Saver 配对,"
        "给每张图标注 LoRA 名与强度。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "基础扩散模型:每个开启的行都将从它克隆后应用 LoRA。"}),
                "clip": ("CLIP", {"tooltip": "基础 CLIP:每个开启的行都将从它克隆后应用 LoRA。"}),
            },
            "hidden": {"LoraLoaderState": ("STRING", {"default": "{}"})},
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "metadata")
    OUTPUT_IS_LIST = (True, True, True)
    OUTPUT_TOOLTIPS = (
        "每个开启行一个模型(各自应用了行内 LoRA 与强度),按行顺序排列。",
        "每个开启行一个 CLIP,与 model 输出一一对应。",
        '每行一条 "{文件名}_{强度}" 文本,与 model/clip 输出一一对应。',
    )
    FUNCTION = "apply"
    CATEGORY = _CATEGORY

    def __init__(self):
        # 内存模式语义与 SFLoraStack 一致(见 lora_cache.LoraCache):
        # 行状态里 cacheMode 携带 "last"/"all"/"none"。
        self._cache = LoraCache()

    def _get_lora(self, path):
        """缓存读 + 缺省加载。旧 ComfyUI 无 return_metadata 参数时回退
        (否则每一行都 TypeError,调用方吞掉后节点静默交回未触碰的模型)。"""
        cached = self._cache.get(path)
        if cached is not None:
            return cached
        try:
            lora, meta = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
        except TypeError:
            lora, meta = comfy.utils.load_torch_file(path, safe_load=True), None
        self._cache.store(path, (lora, meta))
        return (lora, meta)

    def apply(self, model, clip, LoraLoaderState="{}"):
        state = R.parse_state(LoraLoaderState)
        cache_mode = state.get("cacheMode", "last")
        # 本次 run 实际加载的最近一行(见 lora_cache.note_applied)。
        last_this_run = None

        # 实际应用了 LoRA 的行。缺失/改名文件或加载失败(损坏/不兼容)
        # 跳过该行——输出列表保持并行,metadata 与 model/clip 永远同长。
        used_paths = set()
        applied = 0
        outputs_model = []
        outputs_clip = []
        outputs_meta = []

        for entry in state["loras"]:
            if not entry.get("on"):
                continue
            name = entry["name"]
            try:
                path = folder_paths.get_full_path("loras", name)
            except Exception:
                path = None
            if not path or not os.path.isfile(path):
                logger.warning("[SFLoraPlot] skipped (not found): {}".format(name))
                continue
            sm = float(entry.get("sm", 0.0))
            sc = float(entry.get("sc", sm))
            try:
                lora, meta = self._get_lora(path)
                try:
                    m, c = comfy.sd.load_lora_for_models(
                        model, clip, lora, sm, sc, lora_metadata=meta
                    )
                except TypeError:
                    # 旧 ComfyUI:没有 lora_metadata 参数。去掉重试,让 LoRA
                    # 仍能应用而不是静默什么都不做。
                    m, c = comfy.sd.load_lora_for_models(model, clip, lora, sm, sc)
                used_paths.add(path)
                applied += 1
                outputs_model.append(m)
                outputs_clip.append(c)
                outputs_meta.append(P.build_metadata(name, sm))
                last_this_run = self._cache.note_applied(path, cache_mode, last_this_run)
            except Exception as exc:
                # 加载失败 -> 跳过该行,不出现在任何输出里。
                logger.warning("[SFLoraPlot] failed to apply {}: {}".format(name, exc))

        self._cache.trim(cache_mode, used_paths, last_this_run)

        if not outputs_model:
            raise ValueError(
                "[SFLoraPlot] No LoRA rows were applied. Add rows with the "
                "Add LoRA button, make sure at least one is enabled, and the "
                "file exists in models/loras."
            )
        logger.info("[SFLoraPlot] applied {} LoRA row(s).".format(applied))
        return (outputs_model, outputs_clip, outputs_meta)


class SFLoraPlotImageSaver:
    DESCRIPTION = (
        "给批量对比图片标注文字(与原 LoRAPlotImageSaver 语义一致):在每张"
        "图右上角画半透明背景文字盒,内容为 SFLoraPlot 的 metadata 输出——"
        "LoRA 名与强度。逐帧处理(批次 > 1 时每帧都标注);支持中文 LoRA "
        "名(自动选择系统中文字体)。"
    )

    COLOR_OPTIONS = P.COLOR_OPTIONS

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "待标注的图片(通常来自 SFLoraPlot 下游的采样结果)。"}),
                "metadata": ("STRING", {"tooltip": 'SFLoraPlot 的 metadata 输出,格式 "{文件名}_{强度}"。'}),
                "text_color": (cls.COLOR_OPTIONS, {"default": "white", "tooltip": "文字颜色(命名色或 #RRGGBB)。"}),
                "background_color": (cls.COLOR_OPTIONS, {"default": "black", "tooltip": "文字盒背景颜色。"}),
                "font_size": ("INT", {"default": 24, "min": 8, "max": 128, "step": 1, "tooltip": "文字像素大小。"}),
                "padding": ("INT", {"default": 10, "min": 0, "max": 50, "step": 1, "tooltip": "文字盒内边距(像素)。"}),
                "opacity": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.1, "tooltip": "背景盒不透明度。"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "save_with_overlay"
    CATEGORY = _CATEGORY

    def save_with_overlay(self, images, metadata, text_color="white",
                          background_color="black", font_size=24, padding=10,
                          opacity=0.8):
        # 防御单值(单 tensor / 单字符串)与列表两种形状。
        if not isinstance(images, list):
            images = [images]
        if not isinstance(metadata, list):
            metadata = [metadata]

        # metadata 与图片一一对应;单条 metadata 广播到全部图片。
        if len(metadata) != len(images):
            if len(metadata) == 1:
                metadata = metadata * len(images)
            else:
                raise ValueError(
                    "[SFLoraPlotImageSaver] metadata count ({}) must match "
                    "images count ({}).".format(len(metadata), len(images))
                )

        output_images = []
        for img, meta in zip(images, metadata):
            # 批次 > 1 时逐帧标注(原插件只取第 0 帧,丢帧数据)。
            if img.shape[0] == 1:
                pil_frames = [tensor2pil(img)]
            else:
                pil_frames = [tensor2pil(img[i:i + 1]) for i in range(img.shape[0])]
            lora_name, strength = P.parse_metadata(meta)
            if strength:
                overlay_text = "{}\nStrength: {}".format(lora_name, strength)
            else:
                overlay_text = lora_name
            for frame in pil_frames:
                annotated = P.add_text_overlay(
                    frame, overlay_text, text_color, background_color,
                    font_size, padding, opacity,
                )
                output_images.append(pil2tensor(annotated))

        return (output_images,)
