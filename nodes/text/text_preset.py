import json

from comfy.comfy_types.node_typing import IO

_CATEGORY = "sfnodes/text"


class SFTextPreset:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (
                    [""],
                    {
                        "default": "",
                        "tooltip": "选择要输出的预设文本；选项由当前工作流中保存的预设动态生成",
                    },
                ),
                "presets_json": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "[]",
                        "display": "hidden",
                        "tooltip": "预设数据载体（JSON 数组 [{name, text}]），随当前工作流保存，请勿手动编辑",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "preset_name")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "工作流绑定的文本预设：预设保存在当前工作流中（保存工作流即保存预设，其他工作流添加此节点为全新空预设），下拉选择输出预设文本，节点上的文本编辑框可直接修改选中预设的内容（输入即保存），点击「⚙ 预设」可新增/编辑/删除预设"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # preset 选项由前端根据 presets_json 动态重建，会超出 INPUT_TYPES 的静态初始列表（[""]），
        # 跳过默认的 "Value not in list" 校验，execute 内已做完整容错
        return True

    def execute(self, preset: str, presets_json: str):
        name = str(preset) if preset is not None else ""
        text = ""
        try:
            data = json.loads(presets_json or "[]")
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and str(item.get("name", "")) == name:
                        text = str(item.get("text", ""))
                        break
        except Exception:
            pass
        return (text, name)
