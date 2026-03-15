import json
import os

from aiohttp import web
from comfy.comfy_types.node_typing import IO

_CATEGORY = "sfnodes/Text"


def _normalize_item(x):
    """统一为 {alias, content} 结构。"""
    if isinstance(x, dict) and "alias" in x and "content" in x:
        return {"alias": str(x["alias"]).strip(), "content": str(x["content"])}
    if isinstance(x, str):
        s = x.strip()
        first = (s.split("\n")[0] or "").strip() or "未命名"
        return {"alias": first[:64], "content": s}
    return None


class SFTextDropdown:
    """
    文本下拉选择节点（支持别名）

    - 下拉框显示项目别名，输出为对应项目文本内容（可多行）
    - 单行输入框定义别名，多行输入框定义项目内容
    - 选项保存在 custom_nodes/sfnodes/data/text-dropdown/data.json，所有节点共享
    """

    @classmethod
    def _get_options_path(cls) -> str:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, "data", "text-dropdown", "data.json")

    @classmethod
    def _load_global_options(cls) -> list[dict]:
        path = cls._get_options_path()
        try:
            if not os.path.exists(path):
                return []
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                return []
            out = []
            for i, x in enumerate(data):
                item = _normalize_item(x)
                if item and item["alias"]:
                    out.append(item)
            return out
        except Exception:
            pass
        return []

    @classmethod
    def _save_global_options(cls, options: list[dict]) -> None:
        valid = []
        seen = set()
        for x in options:
            item = _normalize_item(x)
            if not item or not item["alias"]:
                continue
            if item["alias"] in seen:
                continue
            seen.add(item["alias"])
            valid.append(item)
        path = cls._get_options_path()
        try:
            d = os.path.dirname(path)
            os.makedirs(d, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(valid, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    @classmethod
    def INPUT_TYPES(cls):
        options = cls._load_global_options()
        default_content = options[0]["content"] if options else ""

        return {
            "required": {
                "selected_text": (
                    IO.STRING,
                    {"multiline": False, "default": default_content},
                ),
                "options_json": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": json.dumps(options, ensure_ascii=False),
                        "display": "hidden",
                    },
                ),
                "alias": (
                    IO.STRING,
                    {"multiline": False, "default": ""},
                ),
                "new_item": (
                    IO.STRING,
                    {"multiline": True, "default": ""},
                ),
            }
        }

    RETURN_TYPES = (IO.STRING, IO.STRING)
    RETURN_NAMES = ("string", "alias")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY

    def execute(
        self,
        selected_text: str,
        options_json: str,
        alias: str = "",
        new_item: str = "",
    ):
        try:
            data = json.loads(options_json)
            if isinstance(data, list):
                type(self)._save_global_options(data)
                opts = [x for x in data if isinstance(x, dict) and "alias" in x and "content" in x]
            else:
                opts = []
        except Exception:
            opts = []

        if not isinstance(selected_text, str):
            selected_text = str(selected_text)
        sel_alias = ""
        for x in opts:
            if str(x.get("content", "")) == selected_text:
                sel_alias = str(x.get("alias", "")).strip()
                break
        return (selected_text, sel_alias)


def _register_text_dropdown_save_route():
    """注册 POST /sfnodes/text_dropdown/save，供前端添加/删除后立即保存 JSON。"""
    try:
        from server import PromptServer

        ins = getattr(PromptServer, "instance", None)
        if ins is None or not hasattr(ins, "routes"):
            return
        routes = ins.routes

        @routes.post("/sfnodes/text_dropdown/save")
        async def _sf_text_dropdown_save(request: web.Request) -> web.Response:
            try:
                body = await request.json()
                raw = body.get("options")
                if not isinstance(raw, list):
                    return web.Response(status=400, text="options array required")
                SFTextDropdown._save_global_options(raw)
                return web.Response(status=200)
            except Exception:
                return web.Response(status=500)

    except Exception:
        pass


_register_text_dropdown_save_route()
