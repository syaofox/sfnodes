"""SFCharacterSelect：角色三分镜单选器（脸部特写/半身像/全身像 + 提示词）。

数据层：角色库即 `character_` 前缀 JSON（内置 `data/characters/*.json` +
用户 `<user>/sfnodes/characters/*.json` 同名用户覆盖，styles_selector.py 同款
双源范式）；条目 `{name, prompt?, face?, half?, full?}`，三图为相对路径
（`samples_id_chara/<库>/...` 独立目录防混放）。
列表/原图路由为本域新建 `/api/sfnodes/characters*`（styles 的
normalize 会丢弃 face/half/full 未知字段，复用即语义污染，故独立，
守卫写法与 styles 双路由 1:1：commonpath 钳位 + query quote）。
单选语义 + 草稿优先 + 三路独立 IMAGE 输出；图片→张量复用
id_clothing._load_template_tensor/_placeholder_tensor（同包 import，不拷贝）。
"""

import json
import os
import threading
import urllib.parse

from aiohttp import web

from .id_clothing import _load_template_tensor, _placeholder_tensor
from ...sf_utils import characters as _lib
from ...sf_utils.id_clothing import resolve_thumbnail_path

_CATEGORY = "sfnodes/text"

_characters_cache = {}  # 角色库名 -> (sig, data)
_characters_lock = threading.Lock()


def _package_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _builtin_characters_dir():
    """内置角色库目录（随包分发的只读数据，如 data/characters/example.json）。"""
    return os.path.join(_package_root(), "data", "characters")


def _sf_user_dir():
    """<ComfyUI user dir>/sfnodes —— 本项目用户数据统一目录（与 styles 同约定）。"""
    base = None
    try:
        import folder_paths

        base = folder_paths.get_user_directory()
    except Exception:
        base = None
    if not base:
        base = os.path.join(_package_root(), "user")
    d = os.path.join(base, "sfnodes")
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return d


def _user_characters_dir():
    """用户自定义角色库目录：<user>/sfnodes/characters/*.json，同名覆盖内置。"""
    d = os.path.join(_sf_user_dir(), "characters")
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return d


def _characters_dirs():
    """角色库搜索目录（用户优先）。"""
    return [_user_characters_dir(), _builtin_characters_dir()]


def character_library_names():
    """可用角色库名（json 文件名去扩展名，去重保持先用户后内置的稳定顺序）。"""
    names = []
    seen = set()
    for d in _characters_dirs():
        try:
            entries = sorted(os.listdir(d))
        except OSError:
            continue
        for f in entries:
            if f.endswith(".json"):
                n = f[:-5]
                if n not in seen:
                    seen.add(n)
                    names.append(n)
    return names


def _character_file(name):
    for d in _characters_dirs():
        p = os.path.join(d, name + ".json")
        if os.path.isfile(p):
            return p
    return None


def _character_file_sig(name):
    p = _character_file(name)
    if not p:
        return None
    try:
        st = os.stat(p)
        return (p, st.st_mtime, st.st_size)
    except OSError:
        return None


def _load_characters(name):
    """加载角色库（线程安全；用户目录同名文件覆盖内置；mtime+size 变化自动重载）。

    返回角色条目列表：[{name, prompt?, face?, half?, full?}]。
    """
    sig = _character_file_sig(name)
    if not sig:
        return []
    key = (sig[0], sig[1], sig[2])
    with _characters_lock:
        cached = _characters_cache.get(name)
        if cached and cached[0] == key:
            return cached[1]
        try:
            with open(sig[0], "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                data = []
        except Exception as e:
            print(f"[SFCharacterSelect] 加载角色库 {name} 失败: {e}")
            data = []
        _characters_cache[name] = (key, data)
        return data


def _thumbnail_url(thumb):
    """缩略图 → 前端可直接使用的 URL：http(s) 原样，本地路径转角色图片路由。"""
    if isinstance(thumb, list):
        return _thumbnail_url(thumb[0]) if thumb else None
    if isinstance(thumb, str) and thumb.startswith(("http://", "https://")):
        return thumb
    if isinstance(thumb, str) and thumb:
        return f"/api/sfnodes/characters/image?path={urllib.parse.quote(thumb, safe='/')}"
    return None


def normalize_role_list(data):
    """路由输出形状：前端展示字段（name/prompt/face/half/full，图为可用 URL）。"""
    out = []
    for d in data:
        if not isinstance(d, dict) or not isinstance(d.get("name"), str) or not d["name"].strip():
            continue
        nd = {"name": d["name"]}
        if d.get("prompt"):
            nd["prompt"] = d["prompt"]
        for shot in _lib.SHOTS:
            url = _thumbnail_url(d.get(shot))
            if url:
                nd[shot] = url
        out.append(nd)
    return out


def _character_libraries():
    """角色库下拉选项（character_ 前缀子集；空库时给 [""] 占位，前端引导用户）。"""
    names = _lib.filter_libraries(character_library_names())
    return names or [""]


class SFCharacterSelect:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "library": (
                    _character_libraries(),
                    {
                        "default": _character_libraries()[0],
                        "tooltip": "角色库：内置 data/characters/*.json + 用户 <user>/sfnodes/characters/*.json（同名用户覆盖内置），每角色含脸部特写/半身像/全身像三图",
                    },
                ),
            },
            "hidden": {
                "SFCharacterState": ("STRING", {"default": "[]"}),
                "SFCharacterPrompt": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("prompt", "face", "half", "full")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "角色三分镜单选器：前端画廊单选角色（卡内并排脸部特写/半身像/全身像，无有效选择时自动首选首个角色）；输出角色提示词 STRING + 三路独立 IMAGE（可按需取用）；角色库为 character_ 前缀 JSON（内置 data/characters + 用户 user/sfnodes/characters 同名覆盖）；编辑框手改优先输出（切换角色即清空，不写库）"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # library 选项由角色目录动态枚举，旧工作流残留值会超出静态列表，
        # 跳过默认 "Value not in list" 校验，execute 内已做未知库降级
        return True

    @classmethod
    def IS_CHANGED(cls, library, **kwargs):
        if not library:
            return 0
        sig = _character_file_sig(library)
        if sig is None:
            return 0
        return (sig[1], sig[2])  # (mtime, size)：角色库文件变化时重跑

    def execute(self, library="", SFCharacterState="[]", SFCharacterPrompt=""):
        data = _load_characters(library or "")
        # 单选收敛：状态同形数组只取首个（旧多值数据向前兼容）
        selected = _lib.first_selected(SFCharacterState)
        entry = _lib.find_role(data, selected)
        if entry is None:
            selected = ""
        draft = str(SFCharacterPrompt) if SFCharacterPrompt is not None else ""
        prompt = _lib.resolve_prompt(data, selected, draft)
        images = []
        for shot in _lib.SHOTS:
            image = None
            if entry is not None:
                path = resolve_thumbnail_path(_lib.shot_thumbnail(entry, shot), _characters_dirs())
                if path is not None:
                    try:
                        image = _load_template_tensor(path)
                    except Exception as e:
                        print(f"[SFCharacterSelect] 角色图加载失败 {path}: {e}")
                        image = None
            if image is None:
                image = _placeholder_tensor()
            images.append(image)
        return (prompt, images[0], images[1], images[2])


def _register_characters_routes():
    """注册角色库 API（/api/sfnodes/characters*）：列表 + 原图。模块导入时副作用注册。"""
    try:
        from server import PromptServer

        ins = getattr(PromptServer, "instance", None)
        if ins is None or not hasattr(ins, "routes"):
            return
        routes = ins.routes

        @routes.get("/api/sfnodes/characters")
        async def _characters_list(request: web.Request) -> web.Response:
            name = request.rel_url.query.get("name")
            if not name:
                return web.Response(status=400)
            # 库名仅允许 character_ 前缀（与风格/服装库隔离，防串库读盘）
            if not name.startswith(_lib.LIB_PREFIX):
                return web.Response(status=400)
            data = _load_characters(name)
            return web.json_response(normalize_role_list(data))

        @routes.get("/api/sfnodes/characters/image")
        async def _characters_image(request: web.Request) -> web.Response:
            query = request.rel_url.query
            rel = query.get("path", "")
            if not rel:
                return web.Response(status=400)
            for d in _characters_dirs():
                base = os.path.abspath(d)
                p = os.path.normpath(os.path.join(base, rel.replace("\\", "/")))
                # 解析后必须仍在角色目录内（防路径穿越）
                try:
                    if os.path.commonpath((base, p)) != base:
                        continue
                except Exception:
                    continue
                if os.path.isfile(p):
                    return web.FileResponse(p)
            return web.Response(status=404)

    except Exception:
        pass


_register_characters_routes()
