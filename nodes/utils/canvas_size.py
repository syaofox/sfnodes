"""
Canvas Size Preset 节点：按模型选择官方分辨率预设，输出画布宽高。

分辨率数据来源（2026-08 网络调研，全部为官方出处）：
- Z-Image / Z-Image-Turbo（阿里通义 6B）：官方 Demo 分辨率表（1MP/1.6MP/2.4MP
  三档各 11 种比例）
  https://huggingface.co/spaces/Tongyi-MAI/Z-Image-Turbo/blob/main/app.py
- Qwen-Image / Qwen-Image-2512（通义 20B）：官方 aspect_ratios 表（1328 基准
  7 种比例，其中 16:9/9:16 为官方近似标签，实际约分 52:29）
  https://github.com/QwenLM/Qwen-Image
- Flux.1 (dev/schnell)（BFL 12B）与 SDXL / SD 3.5（Stability）：1MP 基准
  （通用 DiT 训练分辨率表）
- Flux.2 Klein 9B（BFL）：官方规格 1K~2K、宽高 16 整除（无官方具体列表；
  1K 档取通用 DiT 训练分辨率，2K 档按官方 1024~2048 + 16 整除约束推导）
  https://docs.comfy.org/tutorials/flux/flux-2-klein.md
- Krea 2 Turbo/RAW（Krea AI）：官方 --width/--height 1024~2048、补到 16 的
  倍数（Turbo 支持 1k~2k、RAW 训练到 1k）；档位同 Flux.2 Klein
  https://github.com/krea-ai/krea-2
- Wan2.2 T2V/I2V-A14B 与 TI2V-5B：官方 SUPPORTED_SIZES（480p/720p）
  https://github.com/Wan-Video/Wan2.2/blob/main/wan/configs/__init__.py
- HunyuanVideo 1.5（腾讯 8.3B）：官方 720p（ComfyUI 模板 EmptyHunyuanVideo15Latent
  实测 1280x720，可上采样 1080p）；540p 档沿用 HunyuanVideo 1.0 官方档位
  （VAE 8 整除系，960x540 非 16 倍数）
  https://docs.comfy.org/tutorials/video/hunyuan/hunyuan-video-1-5.md
- LTX-2.5（Lightricks 22B）：分辨率 32 整除；0.9MP 档取 LTX 系官方默认
  1216x704（约分 19:11，模板 ResolutionSelector 16:9/0.9MP/32 整除）；
  1K 档取 32 整除的 16:9 组合
  https://github.com/Lightricks/LTX-Video
"""

from aiohttp import web

_CATEGORY = "sfnodes/utils"

# 每档为 (宽高比标签, 宽, 高) 列表。所有数值均为 16 的倍数（DiT patch 整除
# 要求，各模型官方约束一致）。

# Z-Image 官方三档（demo RES_CHOICES，全 16 整除）
_Z_IMAGE_1MP = [
    ("1:1", 1024, 1024), ("9:7", 1152, 896), ("7:9", 896, 1152),
    ("4:3", 1152, 864), ("3:4", 864, 1152), ("3:2", 1248, 832),
    ("2:3", 832, 1248), ("16:9", 1280, 720), ("9:16", 720, 1280),
    ("21:9", 1344, 576), ("9:21", 576, 1344),
]
_Z_IMAGE_16MP = [
    ("1:1", 1280, 1280), ("9:7", 1440, 1120), ("7:9", 1120, 1440),
    ("4:3", 1472, 1104), ("3:4", 1104, 1472), ("3:2", 1536, 1024),
    ("2:3", 1024, 1536), ("16:9", 1536, 864), ("9:16", 864, 1536),
    ("21:9", 1680, 720), ("9:21", 720, 1680),
]
_Z_IMAGE_24MP = [
    ("1:1", 1536, 1536), ("9:7", 1728, 1344), ("7:9", 1344, 1728),
    ("4:3", 1728, 1296), ("3:4", 1296, 1728), ("3:2", 1872, 1248),
    ("2:3", 1248, 1872), ("16:9", 2048, 1152), ("9:16", 1152, 2048),
    ("21:9", 2016, 864), ("9:21", 864, 2016),
]

# Flux.2 Klein / Krea 2 共用档位（官方约束 1K~2K、16 整除）。
# 1K 档 = 通用 DiT 训练分辨率（与 Z-Image 1MP 同数值）。
_DIT_1K = _Z_IMAGE_1MP
# 2K 档 = 2048 边基准，按官方约束推导（比例集与 1K 档一致）。
_DIT_2K = [
    ("1:1", 2048, 2048), ("9:7", 2016, 1568), ("7:9", 1568, 2016),
    ("4:3", 2048, 1536), ("3:4", 1536, 2048), ("3:2", 2016, 1344),
    ("2:3", 1344, 2016), ("16:9", 2048, 1152), ("9:16", 1152, 2048),
    ("21:9", 2016, 864), ("9:21", 864, 2016),
]

# Wan2.2 官方 SUPPORTED_SIZES：T2V/I2V-A14B 各 4 项，TI2V-5B 2 项。
# 注意：Wan2.2 官方尺寸本身并非全部精确 16:9——480p 的 832x480 约分为
# 26:15（近似 16:9）、TI2V-5B 的 1280x704 约分为 20:11；720p 的 1280x720
# 才是精确 16:9。比例标签一律取实际约分值，与数值保持一致。
_WAN_480P = [("26:15", 832, 480), ("15:26", 480, 832)]
_WAN_720P = [("16:9", 1280, 720), ("9:16", 720, 1280)]
_WAN_5B_720P = [("20:11", 1280, 704), ("11:20", 704, 1280)]

# Qwen-Image / Qwen-Image-2512 官方 aspect_ratios 表（1328 基准，全 16 整除）。
# 官方以 16:9/9:16 标注 1664x928 与 928x1664，但实际约分为 52:29（近似
# 16:9），此处沿用官方标签（测试按容差校验）。
_QWEN_OFFICIAL = [
    ("1:1", 1328, 1328), ("16:9", 1664, 928), ("9:16", 928, 1664),
    ("4:3", 1472, 1104), ("3:4", 1104, 1472), ("3:2", 1584, 1056),
    ("2:3", 1056, 1584),
]

# HunyuanVideo 1.5 官方 720p（ComfyUI 官方模板实测 1280x720，可上采样 1080p）；
# 540p 档沿用 HunyuanVideo 1.0 官方档位（VAE 8 整除系，960x540 非 16 倍数）。
_HUNYUAN_720P = [("16:9", 1280, 720), ("9:16", 720, 1280)]
_HUNYUAN_540P = [("16:9", 960, 540), ("9:16", 540, 960)]

# LTX-2.5：分辨率 32 整除。0.9MP 档取 LTX 系官方默认 1216x704（约分 19:11，
# 官方模板 ResolutionSelector 16:9/0.9MP/32 整除 ≈ 该值）；1K 档为 32 整除
# 的 16:9 精确组合。
_LTX_0P9MP = [("19:11", 1216, 704), ("11:19", 704, 1216)]
_LTX_1K = [("1:1", 1024, 1024), ("16:9", 1024, 576), ("9:16", 576, 1024)]

# 模型 -> 档位 -> [(比例, 宽, 高)]。顺序与 MODEL_GROUPS 保持一致；
# 新模型追加到对应分组即可。
PRESETS = {
    "Z-Image (Turbo)": {"1MP": _Z_IMAGE_1MP, "1.6MP": _Z_IMAGE_16MP, "2.4MP": _Z_IMAGE_24MP},
    "Qwen-Image (2512)": {"Official": _QWEN_OFFICIAL},
    "Flux.1 (dev/schnell)": {"1K": _DIT_1K},
    "Krea 2 (Turbo/RAW)": {"1K": _DIT_1K, "2K": _DIT_2K},
    "Flux.2 Klein 9B": {"1K": _DIT_1K, "2K": _DIT_2K},
    "SDXL / SD 3.5": {"1K": _DIT_1K},
    "Wan2.2 T2V": {"480p": _WAN_480P, "720p": _WAN_720P},
    "Wan2.2 I2V": {"480p": _WAN_480P, "720p": _WAN_720P},
    "Wan2.2 TI2V-5B": {"720p": _WAN_5B_720P},
    "HunyuanVideo 1.5": {"720p": _HUNYUAN_720P, "540p": _HUNYUAN_540P},
    "LTX-2.5": {"0.9MP": _LTX_0P9MP, "1K": _LTX_1K},
}

# model 下拉分组（"--x--" 为 ComfyUI combo 原生分组头，只显示不可选）。
MODEL_GROUPS = [
    ("-- Image --", [
        "Z-Image (Turbo)",
        "Qwen-Image (2512)",
        "Flux.1 (dev/schnell)",
        "Krea 2 (Turbo/RAW)",
        "Flux.2 Klein 9B",
        "SDXL / SD 3.5",
    ]),
    ("-- Video --", [
        "Wan2.2 T2V",
        "Wan2.2 I2V",
        "Wan2.2 TI2V-5B",
        "HunyuanVideo 1.5",
        "LTX-2.5",
    ]),
]

MODELS = []
for _header, _items in MODEL_GROUPS:
    MODELS.append(_header)
    MODELS.extend(_items)
DEFAULT_MODEL = "Z-Image (Turbo)"


def _resolution_value(ratio, width, height):
    """选项显示名：'1024x1024 (1:1)'。"""
    return "{}x{} ({})".format(width, height, ratio)


_DEFAULT_ITEM = _Z_IMAGE_1MP[0]  # ("1:1", 1024, 1024)
DEFAULT_RESOLUTION = _resolution_value(_DEFAULT_ITEM[0], _DEFAULT_ITEM[1], _DEFAULT_ITEM[2])


def _build_resolution_values(model):
    """平铺某模型的所有分辨率选项，档位以 '--x--' 分组头融入（ComfyUI combo 原生分组）。"""
    values = []
    for tier, items in PRESETS[model].items():
        values.append("--{}--".format(tier))
        values.extend(_resolution_value(r, w, h) for r, w, h in items)
    return values


RESOLUTION_VALUES = {model: _build_resolution_values(model) for model in PRESETS}
_DEFAULT_VALUES = RESOLUTION_VALUES[DEFAULT_MODEL]


def _parse_resolution(value):
    """解析 '1024x1024 (1:1)' -> (width, height, ratio)。分组头/畸形值回退默认。"""
    if not value or (value.startswith("--") and value.endswith("--")):
        value = DEFAULT_RESOLUTION
    try:
        dims = value.split(" ", 1)[0]
        width, height = dims.split("x")
        left, right = value.find("("), value.find(")")
        if left < 0 or right < 0:
            raise ValueError
        ratio = value[left + 1:right]
        return int(width), int(height), ratio
    except (ValueError, AttributeError):
        return 1024, 1024, "1:1"


class CanvasSizePreset:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    MODELS,
                    {
                        "default": DEFAULT_MODEL,
                        "tooltip": "目标模型。分辨率预设取自各模型官方规格：Z-Image（官方三档）、"
                                   "Flux.2 Klein 9B 与 Krea 2（官方 1K~2K、16 整除）、"
                                   "Wan2.2 T2V/I2V（官方 480p/720p）",
                    },
                ),
                "resolution": (
                    _DEFAULT_VALUES,
                    {
                        "default": DEFAULT_RESOLUTION,
                        "tooltip": "画布分辨率（宽x高 + 比例）。选项按模型分组（--1MP-- 等档位头），"
                                   "切换 model 后选项自动更新",
                    },
                ),
            }
        }

    RETURN_TYPES = ("INT", "INT", "STRING", "FLOAT")
    RETURN_NAMES = ("width", "height", "resolution", "aspect_ratio")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = ("按模型选择官方分辨率预设，输出画布宽高。覆盖 Z-Image / Qwen-Image / "
                   "Flux.1 / Flux.2 Klein 9B / Krea 2 / SDXL·SD 3.5 与 Wan2.2 / HunyuanVideo "
                   "1.5 / LTX-2.5 等生图生视频模型（全部 16 整除或按官方约束，可直接接 "
                   "Empty Latent / Empty SD3 Latent 等节点）")

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # 动态 combo：resolution 选项随 model 切换重建，值可能超出 INPUT_TYPES
        # 静态列表（旧工作流/前端联动期间），一律放行，执行期解析兜底。
        return True

    def execute(self, model, resolution):
        width, height, ratio = _parse_resolution(resolution)
        aspect_ratio = round(width / height, 6)
        return (width, height, resolution, aspect_ratio)


def _register_canvas_size_routes():
    """提供预设数据 API，供前端 web/canvas_size.js 获取（唯一数据源）。"""
    try:
        from server import PromptServer

        ins = getattr(PromptServer, "instance", None)
        if ins is None or not hasattr(ins, "routes"):
            return
        routes = ins.routes

        @routes.get("/api/sfnodes/canvas_size_presets")
        async def _canvas_size_presets(request: web.Request) -> web.Response:
            return web.json_response({"models": MODELS, "values": RESOLUTION_VALUES})

    except Exception:
        pass


_register_canvas_size_routes()
