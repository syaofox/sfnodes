import json
import os

from comfy.comfy_types.node_typing import IO

_CATEGORY = "sfnodes/Text"


class SFTextDropdown:
    """
    文本下拉选择节点

    - 选中一条文本并输出
    - 文本选项列表保存在 custom_nodes/sfnodes/sfnodes_text_dropdown.json 中
    - 列表在所有 SFTextDropdown 节点之间共享
    """

    @classmethod
    def _get_options_path(cls) -> str:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, "sfnodes_text_dropdown.json")

    @classmethod
    def _load_global_options(cls) -> list[str]:
        path = cls._get_options_path()
        try:
            if not os.path.exists(path):
                return []
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                # 统一为字符串列表
                return [str(x) for x in data]
        except Exception:
            # 文件损坏或解析失败时，忽略并返回空列表
            pass
        return []

    @classmethod
    def _save_global_options(cls, options: list[str]) -> None:
        # 去重并保持顺序
        unique_options = list(dict.fromkeys(str(x) for x in options if str(x).strip()))
        path = cls._get_options_path()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(unique_options, f, ensure_ascii=False, indent=2)
        except Exception:
            # 写入失败时静默忽略，避免打断节点执行
            pass

    @classmethod
    def INPUT_TYPES(cls):
        options = cls._load_global_options()
        default_selected = options[0] if options else ""

        return {
            "required": {
                # 当前选中的文本，由前端 JS 通过下拉框维护
                "selected_text": (
                    IO.STRING,
                    {
                        "multiline": False,
                        "default": default_selected,
                    },
                ),
                # 全部选项列表的 JSON 字符串，由前端 JS 负责同步
                "options_json": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": json.dumps(options, ensure_ascii=False),
                        "display": "hidden",
                    },
                ),
                # 多行输入框，用于添加新项目（整块多行文本为一条）
                "new_item": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "",
                    },
                ),
            }
        }

    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("string",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY

    def execute(self, selected_text: str, options_json: str, new_item: str = ""):
        # 同步并持久化全局列表；new_item 仅作 UI 输入，不参与执行逻辑
        try:
            data = json.loads(options_json)
            if isinstance(data, list):
                type(self)._save_global_options(data)
        except Exception:
            pass

        if not isinstance(selected_text, str):
            selected_text = str(selected_text)

        return (selected_text,)

