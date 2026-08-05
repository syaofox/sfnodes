import json
import os
import random
import re
import threading

from aiohttp import web
from comfy.comfy_types.node_typing import IO

_CATEGORY = "sfnodes/text"

_DISABLED = "禁用"
_RANDOM = "随机"

_CATEGORY_KEYS = ("Outfit", "Pose", "Couple Pose", "Environment", "Lighting", "Style", "Camera Angle", "Camera Distance", "Camera Lens")

_presets = {}
_presets_lock = threading.Lock()
_zh_to_preset = {}
_presets_mtime = None


def _presets_path():
    package_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    return os.path.join(package_root, "data", "prompt_presets.json")


def _load_presets():
    """加载预设数据（线程安全；数据文件 mtime 变化时自动重载），并构建中文名 -> (分类, 英文名) 反查索引。"""
    global _presets, _zh_to_preset, _presets_mtime
    path = _presets_path()
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = None
    if _presets and _presets_mtime == mtime:
        return _presets, _zh_to_preset
    with _presets_lock:
        if _presets and _presets_mtime == mtime:
            return _presets, _zh_to_preset
        data = {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                data = loaded
            _presets_mtime = mtime
        except Exception as e:
            print(f"[SFPromptPreset] 加载预设数据失败: {e}")
        _presets = data
        _zh_to_preset = {}
        for category, presets in data.items():
            for name, preset in presets.items():
                zh = preset.get("name_zh") or name
                _zh_to_preset[zh] = (category, name)
    return _presets, _zh_to_preset


def _category_options(category):
    presets, _ = _load_presets()
    names = list(presets.get(category, {}).keys())
    return [_DISABLED, _RANDOM] + [presets[category][n].get("name_zh") or n for n in names]


class SFPromptPreset:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "基础提示词，可包含 [选项A, 选项B] 随机括号，由种子决定选取",
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "随机种子：固定种子可复现随机括号与随机预设的选择结果",
                }),
            },
            "optional": {
                "outfit_preset": (_category_options("Outfit"), {
                    "default": _DISABLED,
                    "tooltip": "服装预设（下拉显示中文，输出保持英文提示词，含成人向内容）",
                }),
                "pose_preset": (_category_options("Pose"), {
                    "default": _DISABLED,
                    "tooltip": "单人动作/姿势预设（下拉显示中文，输出保持英文提示词，含成人向内容）",
                }),
                "couple_preset": (_category_options("Couple Pose"), {
                    "default": _DISABLED,
                    "tooltip": "双人动作预设（下拉显示中文，输出保持英文提示词，含男女/女女成人向内容）",
                }),
                "environment_preset": (_category_options("Environment"), {
                    "default": _DISABLED,
                    "tooltip": "环境预设（下拉显示中文，输出保持英文提示词）",
                }),
                "lighting_preset": (_category_options("Lighting"), {
                    "default": _DISABLED,
                    "tooltip": "灯光预设（下拉显示中文，输出保持英文提示词）",
                }),
                "style_preset": (_category_options("Style"), {
                    "default": _DISABLED,
                    "tooltip": "风格预设（下拉显示中文，输出保持英文提示词）",
                }),
                "camera_angle_preset": (_category_options("Camera Angle"), {
                    "default": _DISABLED,
                    "tooltip": "镜头角度预设，如平视/俯拍/仰拍等（下拉显示中文，输出保持英文提示词）",
                }),
                "camera_distance_preset": (_category_options("Camera Distance"), {
                    "default": _DISABLED,
                    "tooltip": "镜头距离/景别预设，如特写/中景/全景等（下拉显示中文，输出保持英文提示词）",
                }),
                "camera_lens_preset": (_category_options("Camera Lens"), {
                    "default": _DISABLED,
                    "tooltip": "镜头预设（下拉显示中文，输出保持英文提示词）",
                }),
            },
        }

    RETURN_TYPES = (IO.STRING,) * 10
    RETURN_NAMES = ("combined_prompt", "outfit_text", "pose_text", "couple_text", "environment_text", "lighting_text", "style_text", "camera_angle_text", "camera_distance_text", "camera_lens_text")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "按服装、单人动作、双人动作、环境、灯光、风格、镜头角度、镜头距离、镜头九类预设组合提示词：下拉选项为中文，输出保持英文提示词；支持加权随机选择与 [选项A, 选项B] 括号随机"

    @classmethod
    def IS_CHANGED(cls, seed, **kwargs):
        return seed

    def execute(self, input_text, seed=0, outfit_preset=_DISABLED, pose_preset=_DISABLED,
                couple_preset=_DISABLED, environment_preset=_DISABLED, lighting_preset=_DISABLED,
                style_preset=_DISABLED, camera_angle_preset=_DISABLED,
                camera_distance_preset=_DISABLED, camera_lens_preset=_DISABLED):
        seed = seed if seed is not None else 0

        # 单人/双人动作互斥（前端已联动，此处兜底旧工作流）：同时启用时保留 pose
        if pose_preset != _DISABLED and couple_preset != _DISABLED:
            couple_preset = _DISABLED

        if input_text.strip():
            input_text = self._process_random_brackets(input_text, seed)

        outfit_text = self._resolve_preset("Outfit", outfit_preset, seed + 1)
        pose_text = self._resolve_preset("Pose", pose_preset, seed + 2)
        couple_text = self._resolve_preset("Couple Pose", couple_preset, seed + 3)
        env_text = self._resolve_preset("Environment", environment_preset, seed + 4)
        light_text = self._resolve_preset("Lighting", lighting_preset, seed + 5)
        style_text = self._resolve_preset("Style", style_preset, seed + 6)
        angle_text = self._resolve_preset("Camera Angle", camera_angle_preset, seed + 7)
        distance_text = self._resolve_preset("Camera Distance", camera_distance_preset, seed + 8)
        camera_text = self._resolve_preset("Camera Lens", camera_lens_preset, seed + 9)

        parts = [input_text.strip()] if input_text.strip() else []
        preset_parts = []
        for p in (outfit_text, pose_text, couple_text, env_text, light_text, style_text, angle_text, distance_text, camera_text):
            p = p.strip().rstrip(".")
            if p:
                # 片段首字母小写化，与 Krea2 官方小写短语流风格一致（input_text 保持原样）
                preset_parts.append(p[0].lower() + p[1:])
        parts.extend(preset_parts)
        combined = self._clean_prompt(", ".join(parts))

        return (combined, outfit_text, pose_text, couple_text, env_text, light_text, style_text, angle_text, distance_text, camera_text)

    def _resolve_preset(self, category, selection, seed):
        """获取预设英文提示词；仅支持中文名 / 随机 / 禁用。"""
        if selection == _DISABLED:
            return ""
        if selection == _RANDOM:
            return self._random_preset(category, seed)
        presets, zh_to_preset = _load_presets()
        entry = zh_to_preset.get(selection)
        if entry is None:
            return ""
        preset = presets.get(entry[0], {}).get(entry[1])
        return preset.get("prompt", "") if preset else ""

    def _random_preset(self, category, seed):
        """按权重随机选择一个预设，同一种子结果可复现。"""
        presets = _load_presets()[0].get(category, {})
        if not presets:
            return ""
        names = list(presets.keys())
        weights = [presets[n].get("weight", 1.0) for n in names]
        rng = random.Random()
        rng.seed(seed)
        return presets[rng.choices(names, weights=weights, k=1)[0]].get("prompt", "")

    def _process_random_brackets(self, text, seed):
        """处理 [选项A, 选项B] 括号随机，同一种子结果可复现。"""
        rng = random.Random()
        rng.seed(seed)

        def replace(match):
            options = [o.strip() for o in match.group(1).split(",") if o.strip()]
            return rng.choice(options) if options else ""

        return re.sub(r"\[([^\]]+)\]", replace, text)

    @staticmethod
    def _clean_prompt(prompt):
        prompt = ", ".join([part.strip() for part in prompt.split(",") if part.strip()])
        return prompt.replace("., ", ", ")


def _register_prompt_preset_routes():
    """注册描述映射 API：{分类: {中文选项名: 英文 description}}，供前端悬浮卡片展示。"""
    try:
        from server import PromptServer

        ins = getattr(PromptServer, "instance", None)
        if ins is None or not hasattr(ins, "routes"):
            return
        routes = ins.routes

        @routes.get("/api/sfnodes/prompt_presets")
        async def _prompt_presets(request: web.Request) -> web.Response:
            try:
                presets, _ = _load_presets()
                result = {}
                for category, items in presets.items():
                    result[category] = {
                        (item.get("name_zh") or name): item.get("description", "")
                        for name, item in items.items()
                    }
                return web.json_response(result)
            except Exception:
                return web.Response(status=500)

    except Exception:
        pass


_register_prompt_preset_routes()
