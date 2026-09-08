import json

from comfy.comfy_types.node_typing import IO

from ...sf_utils import text_presets

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
                        "tooltip": "选择要输出的预设文本；选项由前端从全局预设库（user/sfnodes/text_presets.json）动态生成，跨工作流共享",
                    },
                ),
                "presets_json": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "[]",
                        "display": "hidden",
                        "tooltip": "旧版工作流绑定预设数据载体（JSON 数组 [{name, text}]），仅作向后兼容回退：选中预设不在全局库时按此数据输出，请勿手动编辑",
                    },
                ),
                "text_override": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "",
                        "display": "hidden",
                        "tooltip": "文本草稿（工作流级）：编辑框的临时修改存于此，仅影响本节点输出、不写全局库；切换预设或点「💾 保存到预设」后清空，请勿手动编辑",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "preset_name")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "持久化文本预设：预设统一保存在全局预设库 user/sfnodes/text_presets.json（跨工作流共享），下拉选择输出预设文本；编辑框的修改只作为本节点草稿输出（切换预设即丢弃，不写全局库），点「💾 保存到预设」确认后才写入全局库，点「⚙ 预设」可新增/编辑/删除预设；旧工作流保存在节点内的预设仍可通过回退读取继续输出"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # preset 选项由前端根据全局库 + presets_json 动态重建，会超出 INPUT_TYPES 的静态初始列表（[""]），
        # 跳过默认的 "Value not in list" 校验，execute 内已做完整容错
        return True

    def execute(self, preset: str, presets_json: str = "[]", text_override: str = ""):
        name = str(preset) if preset is not None else ""
        # 草稿优先：编辑框的临时修改只影响本节点输出（切换预设时由前端清空）
        draft = str(text_override) if text_override is not None else ""
        if draft:
            return (draft, name)
        # 全局库优先；未命中回退解析旧工作流的 presets_json（向后兼容）
        text = text_presets.find_text(name)
        if text is None:
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
