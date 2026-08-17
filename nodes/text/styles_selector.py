import json
import os
import threading
import urllib.parse

from aiohttp import web

_CATEGORY = "sfnodes/text"

# Fooocus 官方样式样例图（fooocus_styles 库无本地 samples 缩略图时的远程兜底）
_FOOOCUS_SAMPLES_URL = "https://raw.githubusercontent.com/lllyasviel/Fooocus/main/sdxl_styles/samples/"

_styles_cache = {}  # 样式库名 -> (sig, data)
_styles_lock = threading.Lock()


def _package_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _builtin_styles_dir():
    """内置样式库目录（随包分发的只读数据，如 data/styles/fooocus_styles.json）。"""
    return os.path.join(_package_root(), "data", "styles")


def _sf_user_dir():
    """<ComfyUI user dir>/sfnodes —— 本项目用户数据统一目录（与 lora_routes 同约定）。"""
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


def _user_styles_dir():
    """用户自定义样式库目录：<user>/sfnodes/styles/*.json，同名覆盖内置。"""
    d = os.path.join(_sf_user_dir(), "styles")
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return d


def _styles_dirs():
    """样式库搜索目录（用户优先）。"""
    return [_user_styles_dir(), _builtin_styles_dir()]


def style_library_names():
    """可用样式库名（json 文件名去扩展名，去重保持先用户后内置的稳定顺序）。"""
    names = []
    seen = set()
    for d in _styles_dirs():
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


def _style_file(name):
    for d in _styles_dirs():
        p = os.path.join(d, name + ".json")
        if os.path.isfile(p):
            return p
    return None


def _style_file_sig(name):
    p = _style_file(name)
    if not p:
        return None
    try:
        st = os.stat(p)
        return (p, st.st_mtime, st.st_size)
    except OSError:
        return None


def _load_styles(name):
    """加载样式库（线程安全；用户目录同名文件覆盖内置；mtime+size 变化自动重载）。

    返回样式条目列表：[{name, prompt?, negative_prompt?, name_cn?, thumbnail?}]。
    """
    sig = _style_file_sig(name)
    if not sig:
        return []
    key = (sig[0], sig[1], sig[2])
    with _styles_lock:
        cached = _styles_cache.get(name)
        if cached and cached[0] == key:
            return cached[1]
        try:
            with open(sig[0], "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                data = []
        except Exception as e:
            print(f"[SFStylesSelector] 加载样式库 {name} 失败: {e}")
            data = []
        _styles_cache[name] = (key, data)
        return data


def _thumbnail_url(thumb):
    """缩略图 → 前端可直接使用的 URL：http(s) 原样（远程直链），本地路径转 image 路由。"""
    if isinstance(thumb, list):
        return _thumbnail_url(thumb[0]) if thumb else None
    if isinstance(thumb, str) and thumb.startswith(("http://", "https://")):
        return thumb
    if isinstance(thumb, str) and thumb:
        # path 含空格/特殊字符时前端直用需编码（后端 query 解析自动解码）
        return f"/api/sfnodes/styles/image?path={urllib.parse.quote(thumb, safe='/')}"
    return None


def normalize_style_list(data, styles_name):
    """路由输出形状：前端展示字段（name/name_cn/thumbnail/prompt/negative_prompt）。

    prompt/negative_prompt 供 hover 信息浮窗展示（对齐 Easy-Use v2 previewer）。
    """
    out = []
    for d in data:
        if not isinstance(d, dict) or not isinstance(d.get("name"), str) or not d["name"].strip():
            continue
        nd = {"name": d["name"]}
        if d.get("name_cn"):
            nd["name_cn"] = d["name_cn"]
        nd["thumbnail"] = _thumbnail_url(d.get("thumbnail")) or (
            f"/api/sfnodes/styles/image?name={d['name']}&styles_name={styles_name}"
        )
        if d.get("prompt"):
            nd["prompt"] = d["prompt"]
        if d.get("negative_prompt"):
            nd["negative_prompt"] = d["negative_prompt"]
        out.append(nd)
    return out


def _apply_styles(styles_data, values, positive, negative):
    """复刻 Easy-Use stylesSelector 拼接语义（1:1，含 {prompt} 占位消费与尾逗号怪癖）。

    - 第一个含 {prompt} 占位的样式用用户输入替换占位（has_prompt 只置一次）；
      后续含占位的样式剥离 ", {prompt}" 片段后尾接。
    - 不含占位的样式直接尾接。
    - 用户输入未被任何样式消费时前置拼接（含原版遗留的尾逗号，行为一致）。
    """
    all_styles = {d["name"]: d for d in styles_data if isinstance(d, dict) and d.get("name")}
    positive_prompt, negative_prompt = "", negative
    has_prompt = False
    for val in values:
        style = all_styles.get(val)
        if style is None:
            continue
        p = style.get("prompt")
        if p:
            if "{prompt}" in p and not has_prompt:
                positive_prompt = p.replace("{prompt}", positive)
                has_prompt = True
            elif "{prompt}" in p:
                positive_prompt += ", " + p.replace(", {prompt}", "").replace("{prompt}", "")
            else:
                positive_prompt = p if positive_prompt == "" else positive_prompt + ", " + p
        np_ = style.get("negative_prompt")
        if np_:
            negative_prompt = (negative_prompt + ", " + np_) if negative_prompt else np_
    if not has_prompt and positive:
        positive_prompt = positive + positive_prompt + ", "
    return positive_prompt, negative_prompt


def _parse_selected(state_str):
    """解析前端 SFStylesState（JSON 数组），畸形输入容错为空列表。"""
    if isinstance(state_str, list):
        return [str(v) for v in state_str if str(v)]
    if not isinstance(state_str, str):
        return []
    try:
        v = json.loads(state_str) if state_str else []
    except Exception:
        v = []
    return [str(x) for x in v if str(x)] if isinstance(v, list) else []


class SFStylesSelector:
    @classmethod
    def INPUT_TYPES(cls):
        names = style_library_names()
        if "fooocus_styles" not in names:
            names.insert(0, "fooocus_styles")
        return {
            "required": {
                "styles": (names, {
                    "default": "fooocus_styles",
                    "tooltip": "样式库：内置 data/styles/*.json + 用户 <user>/sfnodes/styles/*.json（同名用户覆盖内置）",
                }),
            },
            "optional": {
                "positive": ("STRING", {
                    "default": "",
                    "forceInput": True,
                    "multiline": True,
                    "tooltip": "基础正向提示词：被样式中的 {prompt} 占位消费（第一个含占位样式），未被消费时前置拼接",
                }),
                "negative": ("STRING", {
                    "default": "",
                    "forceInput": True,
                    "multiline": True,
                    "tooltip": "基础负向提示词：样式负面提示词尾接其后",
                }),
                "select_styles": ("STRING", {
                    "default": "",
                    "forceInput": True,
                    "tooltip": "逗号分隔样式名（接线时优先于前端面板选择，如接 SFValueDropdown 输出）",
                }),
            },
            "hidden": {
                "SFStylesState": ("STRING", {"default": "[]"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "复刻 Easy-Use stylesSelector 的风格提示词选择器：内置 Fooocus 275 风格库，前端多选标签面板（搜索/清空/悬停缩略图预览），用户自定义样式放 <user>/sfnodes/styles/*.json；正负提示词按样式模板拼接（{prompt} 占位消费），支持接线覆盖所选样式"

    @classmethod
    def IS_CHANGED(cls, styles, **kwargs):
        if not styles:
            return 0
        sig = _style_file_sig(styles)
        if sig is None:
            return 0
        return (sig[1], sig[2])  # (mtime, size)：样式库文件变化时重跑

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # styles combo 值可能超出静态选项列表（样式库被删除/改名的旧工作流），
        # 由 _load_styles 安全降级为空列表
        return True

    def execute(self, styles="fooocus_styles", positive="", negative="",
                select_styles=None, SFStylesState="[]"):
        data = _load_styles(styles or "fooocus_styles")
        if select_styles:
            # 接线优先；strip 修掉原版 split 后带空格的样式名匹配不到的 bug
            values = [v.strip() for v in select_styles.split(",") if v.strip()]
        else:
            values = _parse_selected(SFStylesState)
        if not values:
            return (positive, negative)
        return _apply_styles(data, values, positive, negative)


def _register_styles_routes():
    """注册样式库 API（/api/sfnodes/styles*）：列表 + 缩略图。模块导入时副作用注册。"""
    try:
        from server import PromptServer

        ins = getattr(PromptServer, "instance", None)
        if ins is None or not hasattr(ins, "routes"):
            return
        routes = ins.routes

        @routes.get("/api/sfnodes/styles")
        async def _styles_list(request: web.Request) -> web.Response:
            name = request.rel_url.query.get("name")
            if not name:
                return web.Response(status=400)
            data = _load_styles(name)
            return web.json_response(normalize_style_list(data, name))

        @routes.get("/api/sfnodes/styles/image")
        async def _styles_image(request: web.Request) -> web.Response:
            query = request.rel_url.query
            if "path" in query:
                rel = query["path"]
                for d in _styles_dirs():
                    for sub in ("", "samples"):
                        base = os.path.abspath(os.path.join(d, sub))
                        p = os.path.normpath(os.path.join(base, rel))
                        # 解析后必须仍在样式目录内（防路径穿越）
                        if os.path.commonpath((base, p)) == base and os.path.isfile(p):
                            return web.FileResponse(p)
                return web.Response(status=404)
            name = query.get("name")
            if name:
                for d in _styles_dirs():
                    base = os.path.abspath(os.path.join(d, "samples"))
                    p = os.path.normpath(os.path.join(base, name + ".jpg"))
                    # 解析后必须仍在 samples 目录内（防路径穿越）
                    if os.path.commonpath((base, p)) == base and os.path.isfile(p):
                        return web.FileResponse(p)
                if query.get("styles_name") == "fooocus_styles":
                    # 无本地样例图：返回远程 URL 文本，前端按 http 前缀直用
                    return web.Response(text=_FOOOCUS_SAMPLES_URL + name + ".jpg")
                return web.Response(status=404)
            return web.Response(status=400)

    except Exception:
        pass


_register_styles_routes()
