from .logger import get_logger
from .string import has_chinese_character

logger = get_logger(__name__)

# ── LLM 翻译（OpenAI 兼容 /chat/completions）─────────────────────────────
# 供 SFPauseText 的「中⇄EN」按钮经 /api/sfnodes/translate 调用。默认服务商
# DeepSeek（deepseek-flash）；此处只放纯函数（请求构造/响应解析/方向判定），
# 网络请求在 nodes/text/translate_routes.py。
DEFAULT_TRANSLATE_BASE_URL = "https://api.deepseek.com"
DEFAULT_TRANSLATE_MODEL = "deepseek-flash"

# 服务商预设：base_url/model 默认值。custom 让用户自填（任何 OpenAI 兼容端点）。
TRANSLATE_PROVIDERS = {
    "deepseek": {"base_url": DEFAULT_TRANSLATE_BASE_URL, "model": DEFAULT_TRANSLATE_MODEL},
    "custom": {"base_url": "", "model": ""},
}

_DIRECTION_TARGET = {"zh2en": "English", "en2zh": "Simplified Chinese"}

_SYSTEM_PROMPT = (
    "You are a professional translation engine. Translate the user's text into "
    "{target}. Output only the translation itself, with no explanations, notes, "
    "quotes, markdown fences or extra text. Preserve formatting, line breaks, "
    "placeholders, tags and special tokens exactly as they appear."
)


def detect_translate_direction(text):
    """按内容判定翻译方向：含中文→zh2en（译成英文），否则→en2zh（译成中文）。

    复用 sf_utils/string.has_chinese_character（基本汉字区）。空文本按 en2zh，
    但路由层会先拒绝空文本，故不会真正发出。
    """
    return "zh2en" if has_chinese_character(text or "") else "en2zh"


def target_language(direction):
    """方向 -> 目标语言英文名（写进系统提示词）。未知方向回退 English。"""
    return _DIRECTION_TARGET.get(direction, "English")


def build_translate_messages(text, direction="zh2en"):
    """构造 OpenAI 兼容 messages（system 约束只输出译文 + user 原文）。"""
    return [
        {"role": "system", "content": _SYSTEM_PROMPT.format(target=target_language(direction))},
        {"role": "user", "content": text if isinstance(text, str) else str(text)},
    ]


def build_translate_payload(text, direction="zh2en", model=DEFAULT_TRANSLATE_MODEL, disable_thinking=True):
    """构造 /chat/completions 请求体。

    disable_thinking：DeepSeek V4.1 默认开启思考模式，翻译任务不需要推理链
    （更慢更贵且 content 可能夹带过程）。仅对支持的端点携带该字段，非 DeepSeek
    的 OpenAI 兼容端点会因未知字段 400，故由调用方按 base_url 决定。
    """
    payload = {
        "model": model or DEFAULT_TRANSLATE_MODEL,
        "messages": build_translate_messages(text, direction),
        "stream": False,
        "temperature": 0.0,
    }
    if disable_thinking:
        payload["thinking"] = {"type": "disabled"}
    return payload


def parse_translate_response(data):
    """从 /chat/completions 响应中取出译文；结构缺失/为空时抛 ValueError。"""
    if not isinstance(data, dict):
        raise ValueError("响应不是 JSON 对象")
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError(extract_api_error(data) or "响应缺少 choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise ValueError("响应译文为空")
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


def translators(
    text: str,
    translator: str = "bing",
    source_language="auto",
    target_language="en",
    timeout: float = 10.0,
):
    if not text:
        return ""
    try:
        import translators

        result = translators.translate_text(
            query_text=text,
            translator=translator,
            from_language=source_language,
            to_language=target_language,
            timeout=timeout,
        )
        return result
    except Exception as e:
        raise Exception(f"Error:  Translation failed , Message : {e}")


def get_translator():
    try:
        import translators

        translators_list = translators.translators_pool
        result = "\n".join(translators_list)
        logger.info(f"Text Translation translator: \n{result}")
        return result

    except Exception as e:
        raise Exception(f"Error:  Translation failed , Message : {e}")
