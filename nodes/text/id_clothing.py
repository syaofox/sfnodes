"""SFIDClothingSelector：证件照服装发型单选器（复刻孤海交互，复用 styles JSON 生态）。

数据层零新增：模板库即 styles 库中 `id_` 前缀的子集
（用户目录 `<user>/sfnodes/styles/id_*.json` + `samples_id_clothing/`（独立命名防与风格库 samples/ 混放），孤海
`标题-提示词` 文件名经一次性脚本转写，见 sf_utils/id_clothing.py），
列表/缩略图路由全复用 `/api/sfnodes/styles*`（本模块 import styles_selector
触发其副作用注册）。
单选语义 + 草稿优先 + IMAGE 模板图输出是本节点独有的三点，
styles_selector.py 的多选 `{prompt}` 拼接逻辑一律不复用。
"""

from .styles_selector import (  # noqa: F401  # 副作用注册 /api/sfnodes/styles* 路由
    _load_styles,
    _parse_selected,
    _style_file_sig,
    _styles_dirs,
    style_library_names,
)

from ...sf_utils import id_clothing as _lib

_CATEGORY = "sfnodes/text"


def _id_libraries():
    """服装库下拉选项（styles 全量子集：id_ 前缀；空库时给 [""] 占位，前端引导用户）。"""
    names = _lib.filter_libraries(style_library_names())
    return names or [""]


def _load_template_tensor(path):
    """模板图落盘 → IMAGE 张量 [1,H,W,3] float32（RGB，原尺寸不缩放）。"""
    from PIL import Image

    from ...sf_utils.image_convert import pil2tensor

    with Image.open(path) as img:
        return pil2tensor(img.convert("RGB"))


def _placeholder_tensor():
    """无选择/无图占位：1x1 黑图（调用方约定的 lean 语义：只影响本节点输出）。"""
    import torch

    return torch.zeros((1, 1, 1, 3), dtype=torch.float32)


class SFIDClothingSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "library": (
                    _id_libraries(),
                    {
                        "default": _id_libraries()[0],
                        "tooltip": "服装模板库：用户目录 <user>/sfnodes/styles/ 下 id_*.json（孤海 7 分类转写：id_服装_女士/男士/女童/男童/老年 + id_发型_女士/男士），同名用户库覆盖内置",
                    },
                ),
            },
            "hidden": {
                "SFIDClothingState": ("STRING", {"default": "[]"}),
                "SFIDClothingPrompt": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("prompt", "template_image")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "证件照服装发型单选器（复刻孤海 IDPhotoClothingSelector 交互）：前端画廊单选模板，输出模板提示词 STRING + 模板图 IMAGE（可作重绘参考图）；模板库复用风格选择器 JSON 生态（user/sfnodes/styles/id_*.json + samples_id_clothing/）；编辑框手改优先输出（切换模板即清空，不写库）"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # library 选项由 styles 目录动态枚举，旧工作流残留值会超出静态列表，
        # 跳过默认 "Value not in list" 校验，execute 内已做未知库降级
        return True

    @classmethod
    def IS_CHANGED(cls, library, **kwargs):
        if not library:
            return 0
        sig = _style_file_sig(library)
        if sig is None:
            return 0
        return (sig[1], sig[2])  # (mtime, size)：模板库文件变化时重跑

    def execute(self, library="", SFIDClothingState="[]", SFIDClothingPrompt=""):
        data = _load_styles(library or "")
        # 单选收敛：styles 状态同形数组只取首个（旧多值数据向前兼容）
        names = _parse_selected(SFIDClothingState)
        selected = names[0] if names else ""
        entry = _lib.find_style(data, selected)
        if entry is None:
            selected = ""
        draft = str(SFIDClothingPrompt) if SFIDClothingPrompt is not None else ""
        prompt = _lib.resolve_prompt(data, selected, draft)
        image = None
        if entry is not None:
            thumb = _lib.thumbnail_of(entry)
            path = _lib.resolve_thumbnail_path(thumb, _styles_dirs())
            if path is not None:
                try:
                    image = _load_template_tensor(path)
                except Exception as e:
                    print(f"[SFIDClothingSelector] 模板图加载失败 {path}: {e}")
                    image = None
        if image is None:
            image = _placeholder_tensor()
        return (prompt, image)
