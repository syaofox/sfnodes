"""LLM 客户端（OpenAI 兼容 /chat/completions）：配置读取 + 请求构造/解析 + 同步/异步调用。

供 SFPauseText 翻译按钮（走 /api/sfnodes/translate）与 SFImageInterrogatorAPI
（节点 execute）共用，避免请求构造/响应解析双份实现。默认 DeepSeek deepseek-flash。

凭据来自 ComfyUI Settings（`sfnodes.LLM.*`），后端直接读服务器上的
<user_dir>/default/comfy.settings.json（前端 /settings 写入，见 comfy app_settings）。
多用户安装下 execute 无请求上下文，回落 default 用户；再以环境变量
DEEPSEEK_API_KEY 兜底。旧 id `sfnodes.Translate.*` 作为回退读取（迁移兼容）。

请求构造/解析为纯函数（无网络、可单测）；网络调用分同步（requests，节点 execute
的 worker 线程）与异步（aiohttp，路由协程）两条。
"""

import base64
import hashlib
import io
import json
import os
import threading
from collections import OrderedDict

from .logger import get_logger

logger = get_logger(__name__)

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"

# 服务商预设：base_url/model 默认值。custom 让用户自填（任何 OpenAI 兼容端点）。
PROVIDER_PRESETS = {
    "deepseek": {"base_url": DEFAULT_BASE_URL, "model": DEFAULT_MODEL},
    "custom": {"base_url": "", "model": ""},
}

SETTING_PREFIX = "sfnodes.LLM."
LEGACY_SETTING_PREFIX = "sfnodes.Translate."

_SETTING_SUFFIXES = {
    "provider": "Provider",
    "base_url": "BaseUrl",
    "model": "Model",
    "api_key": "ApiKey",
}


# ── 配置读取 ────────────────────────────────────────────────────────────
def _settings_path():
    """ComfyUI 单用户设置文件路径（多用户安装无请求上下文，只能读 default）。"""
    base = None
    try:
        import folder_paths

        base = folder_paths.get_user_directory()
    except Exception:
        base = None
    if not base:
        base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "user")
    return os.path.join(base, "default", "comfy.settings.json")


# 设置读盘缓存：(path, st_mtime_ns, st_size) → settings dict。仅 stat 命中即返回，
# 文件变更（前端保存设置）自然失效；stat 失败（文件缺失）不缓存，损坏 JSON 随文件键缓存。
_SETTINGS_CACHE = {"key": None, "data": {}}


def read_comfy_settings():
    """读 comfy.settings.json（缺失/损坏返回 {}），按 (path, mtime_ns, size) 缓存。

    每次调用只做一次 stat：文件未变直接返回缓存（无 open/解析开销）；前端保存
    设置会更新 mtime/size，缓存自然失效，改动仍即时生效。返回的 dict 即缓存
    对象，调用方只读、勿修改。
    """
    path = _settings_path()
    try:
        st = os.stat(path)
        key = (path, st.st_mtime_ns, st.st_size)
    except OSError:
        key = None
    if key is not None and _SETTINGS_CACHE["key"] == key:
        return _SETTINGS_CACHE["data"]
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            data = {}
    except Exception:
        data = {}
    if key is not None:
        _SETTINGS_CACHE["key"] = key
        _SETTINGS_CACHE["data"] = data
    return data


def get_llm_config(settings=None):
    """当前 LLM 配置 {provider, base_url, model, api_key}。

    新 id 优先，旧 `sfnodes.Translate.*` 回退；api_key 再以环境变量兜底。
    settings 可显式传入（测试/复用），None 时读 comfy.settings.json。
    """
    if settings is None:
        settings = read_comfy_settings()

    def pick(suffix):
        key = _SETTING_SUFFIXES[suffix]
        for prefix in (SETTING_PREFIX, LEGACY_SETTING_PREFIX):
            v = settings.get(prefix + key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""

    api_key = pick("api_key")
    if not api_key:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    return {
        "provider": pick("provider") or "deepseek",
        "base_url": pick("base_url") or DEFAULT_BASE_URL,
        "model": pick("model") or DEFAULT_MODEL,
        "api_key": api_key,
        "cache_enabled": _read_cache_enabled(settings),
    }


def _read_cache_enabled(settings):
    """读取 LRU 缓存开关（sfnodes.LLM.CacheEnabled，缺省开）。兼容 bool 与字符串。"""
    for prefix in (SETTING_PREFIX, LEGACY_SETTING_PREFIX):
        if prefix + "CacheEnabled" in settings:
            v = settings.get(prefix + "CacheEnabled")
            if isinstance(v, bool):
                return v
            if isinstance(v, str):
                return v.strip().lower() in ("1", "true", "yes", "on")
            return bool(v)
    return True


# ── 请求构造 / 解析（纯函数）────────────────────────────────────────────
def is_deepseek(base_url):
    """DeepSeek 端点（其 `thinking` 非标准扩展字段只对它携带）。"""
    return "deepseek" in (base_url or "").lower()


def build_chat_payload(model, messages, *, temperature=None, max_tokens=None,
                       seed=None, disable_thinking=False):
    """构造 /chat/completions 请求体。

    disable_thinking：DeepSeek V4.1 默认开启思考模式；翻译/反推不需要推理链
    （更慢更贵且 content 可能夹带过程）。仅对支持的端点携带，其他 OpenAI 兼容
    端点会因未知字段 400，故由调用方按 base_url 决定。
    seed：best-effort 复现种子。官方 DeepSeek Chat Completions 未文档化该字段，
    故仅由调用方按需传入（节点 send_seed 开关）；不传则完全不写此键。
    """
    payload = {
        "model": model or DEFAULT_MODEL,
        "messages": messages,
        "stream": False,
    }
    if temperature is not None:
        payload["temperature"] = float(temperature)
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)
    if seed is not None:
        payload["seed"] = int(seed)
    if disable_thinking:
        payload["thinking"] = {"type": "disabled"}
    return payload


def image_content_parts(data_urls, detail="auto"):
    """一张或多张图片 data URL → OpenAI 兼容 image_url content 部分列表。

    data_urls 接受单个字符串或字符串序列（空值跳过）；detail ∈ {low, high, auto}。
    图片只允许出现在 user 消息。
    """
    if isinstance(data_urls, str):
        urls = [data_urls] if data_urls else []
    else:
        urls = [u for u in (data_urls or []) if u]
    parts = []
    for url in urls:
        image_url = {"url": url}
        if detail:
            image_url["detail"] = detail
        parts.append({"type": "image_url", "image_url": image_url})
    return parts


def build_image_content(text, data_url, detail="auto"):
    """user 消息 content 数组：文本 + 一张或多张图片（OpenAI 兼容 image_url 部分）。

    data_url 接受单个 data URL 或 data URL 序列（多图按序排列，供多参考图场景）。
    detail ∈ {low, high, auto}；DeepSeek 亦支持。图片只允许出现在 user 消息。
    """
    return [
        {"type": "text", "text": text if isinstance(text, str) else str(text)},
        *image_content_parts(data_url, detail),
    ]


def parse_chat_response(data):
    """从 /chat/completions 响应中取出正文；结构缺失/为空时抛 ValueError。"""
    if not isinstance(data, dict):
        raise ValueError("响应不是 JSON 对象")
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError(extract_api_error(data) or "响应缺少 choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise ValueError("响应内容为空")
    return content.strip()


def extract_api_error(data):
    """从错误响应里提取可读信息（DeepSeek/OpenAI 均为 {"error": {...}}）。"""
    if not isinstance(data, dict):
        return ""
    err = data.get("error")
    if isinstance(err, dict):
        return str(err.get("message") or err.get("type") or "")
    if isinstance(err, str):
        return err
    return ""


# ── LRU 响应缓存（翻译 / 反推共用）──────────────────────────────────────
# 同参数（模型/端点/messages/温度/max_tokens/seed/thinking）的相同请求直接命中，
# 不再出网。键为 payload 的 sha256（含图片 data URL，故图片不同即不同键；哈希后
# 不驻留图片字节）。仅缓存成功结果。同步调用在 worker 线程、异步在事件循环，
# OrderedDict 非线程安全，统一加锁。
CACHE_CAPACITY = 128


class LruCache:
    """线程安全的定容 LRU：命中提升到队尾，超容淘汰最久未用。"""

    def __init__(self, capacity=CACHE_CAPACITY):
        self._capacity = max(1, int(capacity))
        self._lock = threading.Lock()
        self._data = OrderedDict()

    def get(self, key):
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def set(self, key, value):
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self._capacity:
                self._data.popitem(last=False)

    def clear(self):
        with self._lock:
            self._data.clear()

    def __len__(self):
        with self._lock:
            return len(self._data)


response_cache = LruCache(CACHE_CAPACITY)


def make_cache_key(base_url, payload, extra=None):
    """缓存键：端点 + payload 规范化 JSON 的 sha256 + 可选 extra（节点级 seed 等）。

    extra 用于把"不影响请求体但影响用户期望结果"的参数（如 send_seed 关闭时节点
    的 seed，配合随机化区分缓存）纳入键。
    """
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return (base_url or DEFAULT_BASE_URL, digest, extra)


def _resolve_use_cache(config, use_cache):
    if use_cache is None:
        return bool((config or {}).get("cache_enabled", True))
    return bool(use_cache)


# ── 图片编码（PIL 纯逻辑，可单测）───────────────────────────────────────
def image_to_data_url(image, max_megapixels=None, fmt="JPEG", quality=90):
    """PIL Image → data URL（默认 JPEG）。max_megapixels 为面积上限（只缩小不放大）。"""
    from PIL import Image

    img = image
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    if max_megapixels:
        total = int(float(max_megapixels) * 1024 * 1024)
        w, h = img.size
        if w * h > total and w > 0 and h > 0:
            scale = (total / float(w * h)) ** 0.5
            img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    fmt_up = (fmt or "JPEG").upper()
    if fmt_up in ("JPEG", "JPG"):
        img.save(buf, format="JPEG", quality=int(quality))
        mime = "image/jpeg"
    else:
        img.save(buf, format="PNG")
        mime = "image/png"
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return "data:{};base64,{}".format(mime, b64)


# ── 网络调用 ────────────────────────────────────────────────────────────
def _chat_url(base_url):
    return (base_url or DEFAULT_BASE_URL).rstrip("/") + "/chat/completions"


def _headers(api_key):
    return {"Authorization": "Bearer {}".format(api_key), "Content-Type": "application/json"}


def _require_api_key(config):
    api_key = (config or {}).get("api_key") or ""
    if not api_key:
        raise RuntimeError("未配置 API Key：请在 设置 → SF LLM 中填写。")
    return api_key


def chat_completion_sync(config, messages, *, temperature=None, max_tokens=None,
                         seed=None, timeout=120, use_cache=None, cache_key_extra=None):
    """同步调用（节点 execute 的 worker 线程）。失败抛 RuntimeError/ValueError。

    use_cache=None 时按配置 cache_enabled（设置 sfnodes.LLM.CacheEnabled，缺省开）。
    cache_key_extra 纳入缓存键但不进请求体（如节点 seed）。
    """
    base_url = (config or {}).get("base_url") or DEFAULT_BASE_URL
    model = (config or {}).get("model") or DEFAULT_MODEL
    api_key = _require_api_key(config)
    payload = build_chat_payload(
        model, messages, temperature=temperature, max_tokens=max_tokens,
        seed=seed, disable_thinking=is_deepseek(base_url),
    )
    caching = _resolve_use_cache(config, use_cache)
    key = make_cache_key(base_url, payload, cache_key_extra)
    if caching:
        hit = response_cache.get(key)
        if hit is not None:
            return hit
    text = _do_request_sync(base_url, payload, api_key, timeout)
    if caching:
        response_cache.set(key, text)
    return text


async def chat_completion_async(config, messages, *, temperature=None, max_tokens=None,
                                seed=None, timeout=120, use_cache=None, cache_key_extra=None):
    """异步调用（路由协程）。失败抛 RuntimeError/ValueError/asyncio.TimeoutError。"""
    base_url = (config or {}).get("base_url") or DEFAULT_BASE_URL
    model = (config or {}).get("model") or DEFAULT_MODEL
    api_key = _require_api_key(config)
    payload = build_chat_payload(
        model, messages, temperature=temperature, max_tokens=max_tokens,
        seed=seed, disable_thinking=is_deepseek(base_url),
    )
    caching = _resolve_use_cache(config, use_cache)
    key = make_cache_key(base_url, payload, cache_key_extra)
    if caching:
        hit = response_cache.get(key)
        if hit is not None:
            return hit
    text = await _do_request_async(base_url, payload, api_key, timeout)
    if caching:
        response_cache.set(key, text)
    return text


def _do_request_sync(base_url, payload, api_key, timeout):
    import requests

    resp = requests.post(
        _chat_url(base_url), json=payload, headers=_headers(api_key), timeout=timeout,
    )
    try:
        data = resp.json()
    except Exception:
        data = None
    if resp.status_code != 200:
        msg = extract_api_error(data) if data else ""
        raise RuntimeError(msg or "请求失败（HTTP {}）".format(resp.status_code))
    if data is None:
        raise RuntimeError("响应不是有效 JSON")
    return parse_chat_response(data)


async def _do_request_async(base_url, payload, api_key, timeout):
    import aiohttp

    tout = aiohttp.ClientTimeout(total=timeout)
    data = None
    async with aiohttp.ClientSession(timeout=tout) as session:
        async with session.post(_chat_url(base_url), json=payload, headers=_headers(api_key)) as resp:
            raw = await resp.text()
            try:
                data = json.loads(raw)
            except Exception:
                data = None
            if resp.status != 200:
                msg = extract_api_error(data) if data else ""
                raise RuntimeError(msg or "请求失败（HTTP {}）".format(resp.status))
    if data is None:
        raise RuntimeError("响应不是有效 JSON")
    return parse_chat_response(data)
