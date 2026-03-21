import json
import os

from aiohttp import web
from comfy.comfy_types.node_typing import IO

_CATEGORY = "sfnodes/text"


class SFTextDropdown:
    @classmethod
    def _get_options_path(cls) -> str:
        base_dir = os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )
            )
        )
        return os.path.join(base_dir, "user", "sfnodes", "text-dropdown.json")

    @classmethod
    def _load_config(cls) -> dict:
        path = cls._get_options_path()
        try:
            if not os.path.exists(path):
                return {"categories": ["default"], "options": []}
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return {"categories": ["default"], "options": []}
            categories = data.get("categories", ["default"])
            if not isinstance(categories, list):
                categories = ["default"]
            options = data.get("options", [])
            if not isinstance(options, list):
                options = []
            return {"categories": categories, "options": options}
        except Exception:
            pass
        return {"categories": ["default"], "options": []}

    @classmethod
    def _save_config(cls, config: dict) -> None:
        path = cls._get_options_path()
        try:
            d = os.path.dirname(path)
            os.makedirs(d, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    @classmethod
    def INPUT_TYPES(cls):
        config = cls._load_config()
        categories = config.get("categories", ["default"])
        options = config.get("options", [])
        default_category = categories[0] if categories else "default"

        default_content = ""
        if options:
            cat_options = [x for x in options if x.get("category") == default_category]
            if cat_options:
                default_content = cat_options[0].get("content", "")

        return {
            "required": {
                "category": (
                    IO.STRING,
                    {"default": default_category},
                ),
                "selected_text": (
                    IO.STRING,
                    {
                        "multiline": False,
                        "default": default_content,
                    },
                ),
                "options_json": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": json.dumps(options, ensure_ascii=False),
                        "display": "hidden",
                    },
                ),
            }
        }

    RETURN_TYPES = (IO.STRING, IO.STRING)
    RETURN_NAMES = ("content", "alias")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY

    def execute(self, category: str, selected_text: str, options_json: str):
        config = self._load_config()
        categories = config.get("categories", ["default"])
        options = config.get("options", [])

        if category not in categories:
            category = categories[0] if categories else "default"

        if not isinstance(selected_text, str):
            selected_text = str(selected_text)

        sel_alias = ""
        for x in options:
            if str(x.get("content", "")) == selected_text:
                sel_alias = str(x.get("alias", "")).strip()
                break
        return (selected_text, sel_alias)


def _register_text_dropdown_routes():
    try:
        from server import PromptServer

        ins = getattr(PromptServer, "instance", None)
        if ins is None or not hasattr(ins, "routes"):
            return
        routes = ins.routes

        @routes.post("/api/sfnodes/text_dropdown/save")
        async def _save(request: web.Request) -> web.Response:
            try:
                body = await request.json()
                config = body.get("config", {})
                if not isinstance(config, dict):
                    return web.Response(status=400, text="config object required")
                SFTextDropdown._save_config(config)
                return web.Response(status=200)
            except Exception:
                return web.Response(status=500)

        @routes.get("/api/sfnodes/text_dropdown/load")
        async def _load(request: web.Request) -> web.Response:
            try:
                config = SFTextDropdown._load_config()
                return web.json_response(config)
            except Exception:
                return web.Response(status=500)

    except Exception:
        pass


_register_text_dropdown_routes()
