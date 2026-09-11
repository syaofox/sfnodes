"""SFCharacterSelect：角色多选图库（角色内多选 → images batch 输出）。

数据层：角色库即 `character_` 前缀 JSON（内置 `data/characters/*.json` +
用户 `<user>/sfnodes/characters/*.json` 同名用户覆盖，styles_selector.py 同款
双源范式）；条目新形状 `{name, prompt?, images?: [{label, file, prompt?}]}`，
旧 `{prompt, face/half/full}` 形状双读兼容（见 sf_utils/characters.py）；
图放 `samples_id_chara/<库>/` 独立目录防混放。
列表/原图路由为本域 `/api/sfnodes/characters*`（库名前缀隔离 + commonpath 钳位）。
输出 3 路：拼接 prompt STRING（草稿整体覆盖）+ 角色级 prompt STRING +
images IMAGE（单选 [1,H,W,C]／多选 batch／空选 1×1 占位；尺寸归一到首选图）。
图片→张量/占位/尺寸归一复用 id_clothing 同包与 image_convert（不拷贝）。
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
    """内置角色库目录（随包分发的只读数据，如 data/characters/character_example.json）。"""
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

    返回角色条目列表（新旧两形状原样返回，归一由 sf_utils/characters.py 负责）。
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


def _thumbnail_url(file):
    """分镜图 → 前端可直接使用的 URL：http(s) 原样，本地路径转角色图片路由。"""
    if isinstance(file, list):
        return _thumbnail_url(file[0]) if file else None
    if isinstance(file, str) and file.startswith(("http://", "https://")):
        return file
    if isinstance(file, str) and file:
        return f"/api/sfnodes/characters/image?path={urllib.parse.quote(file, safe='/')}"
    return None


def normalize_role_list(data):
    """路由输出形状：[{name, prompt?, images: [{label, prompt?, url}]}]。

    旧三分镜形状在此归一为 images 数组（label 取中文分镜名），前端只认新形状。
    """
    out = []
    for d in data:
        if not isinstance(d, dict) or not isinstance(d.get("name"), str) or not d["name"].strip():
            continue
        nd = {"name": d["name"]}
        role_prompt = d.get("prompt")
        if role_prompt:
            nd["prompt"] = role_prompt
        images = []
        for item in _lib.image_entries(d):
            url = _thumbnail_url(item["file"])
            if not url:
                continue
            shot = {"label": item["label"], "url": url}
            if item.get("prompt"):
                shot["prompt"] = item["prompt"]
            images.append(shot)
        nd["images"] = images
        out.append(nd)
    return out


def _character_libraries():
    """角色库下拉选项（character_ 前缀子集；空库时给 [""] 占位，前端引导用户）。"""
    names = _lib.filter_libraries(character_library_names())
    return names or [""]


def _batch_tensors(paths):
    """多图路径 → images batch（首选图尺寸归一 lanczos，单张即 [1,H,W,C]）。"""
    import torch

    from ...sf_utils.image_convert import rescale_image

    tensors = [_load_template_tensor(p) for p in paths]
    ref_h, ref_w = tensors[0].shape[1], tensors[0].shape[2]
    normed = []
    for t in tensors:
        if (t.shape[1], t.shape[2]) != (ref_h, ref_w):
            t = rescale_image(t, ref_w, ref_h)
        normed.append(t)
    return torch.cat(normed, dim=0)


class SFCharacterSelect:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "library": (
                    _character_libraries(),
                    {
                        "default": _character_libraries()[0],
                        "tooltip": "角色库：内置 data/characters/*.json + 用户 <user>/sfnodes/characters/*.json（同名用户覆盖内置），每角色不限数量图片",
                    },
                ),
            },
            "hidden": {
                "SFCharacterState": ("STRING", {"default": '{"role": "", "shots": []}'}),
                "SFCharacterPrompt": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("prompt", "role_prompt", "images")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "角色多选图库：单节点多角色切换，角色内勾选不限数量图片，输出拼接提示词 STRING + 角色级提示词 STRING + images batch IMAGE（单选即单张）；尺寸归一到首选图；角色库为 character_ 前缀 JSON（内置 data/characters + 用户 user/sfnodes/characters 同名覆盖）；编辑框手改整体覆盖拼接路（切换角色/改选即清空，不写库）"

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

    def execute(self, library="", SFCharacterState="", SFCharacterPrompt=""):
        data = _load_characters(library or "")
        # 多选收敛：角色有效保留并取交集分镜，否则回落首角首图（旧 ["名"] 数组态迁移首图）
        sel = _lib.coerce_selection(data, SFCharacterState)
        entry = _lib.find_role(data, sel["role"])
        role_prompt = _lib.role_prompt(entry)
        draft = str(SFCharacterPrompt) if SFCharacterPrompt is not None else ""
        prompt = _lib.resolve_prompt(entry, sel["shots"], draft)
        images = None
        if entry is not None and sel["shots"]:
            by_label = {i["label"]: i for i in _lib.image_entries(entry)}
            paths = []
            for label in sel["shots"]:
                item = by_label.get(label)
                if item is None:
                    continue
                path = resolve_thumbnail_path(item["file"], _characters_dirs())
                if path is not None:
                    paths.append(path)
                else:
                    print(f"[SFCharacterSelect] 角色图缺失跳过 {item['file']}")
            if paths:
                try:
                    images = _batch_tensors(paths)
                except Exception as e:
                    print(f"[SFCharacterSelect] batch 拼装失败: {e}")
                    images = None
        if images is None:
            images = _placeholder_tensor()
        return (prompt, role_prompt, images)


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
