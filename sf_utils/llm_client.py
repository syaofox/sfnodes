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
import io
import json
import os

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


def read_comfy_settings():
    """读 comfy.settings.json（缺失/损坏返回 {}）。"""
    try:
        with open(_settings_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


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
    }


# ── 请求构造 / 解析（纯函数）────────────────────────────────────────────
def is_deepseek(base_url):
    """DeepSeek 端点（其 `thinking` 非标准扩展字段只对它携带）。"""
    return "deepseek" in (base_url or "").lower()


def build_chat_payload(model, messages, *, temperature=None, max_tokens=None, disable_thinking=False):
    """构造 /chat/completions 请求体。

    disable_thinking：DeepSeek V4.1 默认开启思考模式；翻译/反推不需要推理链
    （更慢更贵且 content 可能夹带过程）。仅对支持的端点携带，其他 OpenAI 兼容
    端点会因未知字段 400，故由调用方按 base_url 决定。
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
    if disable_thinking:
        payload["thinking"] = {"type": "disabled"}
    return payload


def build_image_content(text, data_url, detail="auto"):
    """user 消息 content 数组：文本 + 图片（OpenAI 兼容 image_url 部分）。

    detail ∈ {low, high, auto}；DeepSeek 亦支持。图片只允许出现在 user 消息。
    """
    image_url = {"url": data_url}
    if detail:
        image_url["detail"] = detail
    return [
        {"type": "text", "text": text if isinstance(text, str) else str(text)},
        {"type": "image_url", "image_url": image_url},
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


def chat_completion_sync(config, messages, *, temperature=None, max_tokens=None, timeout=120):
    """同步调用（节点 execute 的 worker 线程）。失败抛 RuntimeError/ValueError。"""
    import requests

    base_url = (config or {}).get("base_url") or DEFAULT_BASE_URL
    model = (config or {}).get("model") or DEFAULT_MODEL
    api_key = _require_api_key(config)
    payload = build_chat_payload(
        model, messages, temperature=temperature, max_tokens=max_tokens,
        disable_thinking=is_deepseek(base_url),
    )
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


async def chat_completion_async(config, messages, *, temperature=None, max_tokens=None, timeout=120):
    """异步调用（路由协程）。失败抛 RuntimeError/ValueError/asyncio.TimeoutError。"""
    import aiohttp

    base_url = (config or {}).get("base_url") or DEFAULT_BASE_URL
    model = (config or {}).get("model") or DEFAULT_MODEL
    api_key = _require_api_key(config)
    payload = build_chat_payload(
        model, messages, temperature=temperature, max_tokens=max_tokens,
        disable_thinking=is_deepseek(base_url),
    )
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
