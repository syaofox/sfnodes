from .llm_client import (
    DEFAULT_BASE_URL as DEFAULT_TRANSLATE_BASE_URL,
    DEFAULT_MODEL as DEFAULT_TRANSLATE_MODEL,
    PROVIDER_PRESETS as TRANSLATE_PROVIDERS,
    build_chat_payload,
    extract_api_error,
    parse_chat_response as parse_translate_response,
)
from .string import has_chinese_character

# ── LLM 翻译（OpenAI 兼容 /chat/completions）─────────────────────────────
# 供 SFPauseText 的「中⇄EN」按钮经 /api/sfnodes/translate 调用；默认服务商
# DeepSeek（deepseek-flash）。通用请求构造/解析/网络调用单源收敛于
# sf_utils/llm_client.py（与 SFImageInterrogatorAPI 共用），此处只留翻译专属的
# 方向判定与提示词；旧函数名以导入别名保留（对外契约不变）。

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
    """构造 /chat/completions 请求体（通用构造单源 llm_client.build_chat_payload）。

    disable_thinking：DeepSeek V4.1 默认开启思考模式，翻译任务不需要推理链
    （更慢更贵且 content 可能夹带过程）。仅对支持的端点携带该字段，非 DeepSeek
    的 OpenAI 兼容端点会因未知字段 400，故由调用方按 base_url 决定。
    """
    return build_chat_payload(
        model or DEFAULT_TRANSLATE_MODEL,
        build_translate_messages(text, direction),
        temperature=0.0,
        disable_thinking=disable_thinking,
    )
