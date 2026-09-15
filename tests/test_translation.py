# SFPauseText 翻译 LLM 纯逻辑测试（python tests/test_translation.py）
# 覆盖 sf_utils/translation.py 的纯函数：方向判定 / 目标语言 / messages /
# 请求体（thinking 开关）/ 响应解析 / 错误提取 / 服务商默认常量。
# 不发网络请求（网络代理在 nodes/text/translate_routes.py，由真实环境验证）。

import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

from sf_utils.translation import (  # noqa: E402
    DEFAULT_TRANSLATE_BASE_URL,
    DEFAULT_TRANSLATE_MODEL,
    TRANSLATE_PROVIDERS,
    build_translate_messages,
    build_translate_payload,
    detect_translate_direction,
    extract_api_error,
    parse_translate_response,
    target_language,
)

failures = []


def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")


def raises(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
        return False
    except Exception:
        return True


# ── 默认常量 / 服务商预设 ──
check("默认 base_url", DEFAULT_TRANSLATE_BASE_URL == "https://api.deepseek.com")
check("默认模型 deepseek-flash", DEFAULT_TRANSLATE_MODEL == "deepseek-flash")
check("deepseek 预设一致", TRANSLATE_PROVIDERS["deepseek"] == {
    "base_url": DEFAULT_TRANSLATE_BASE_URL,
    "model": DEFAULT_TRANSLATE_MODEL,
})
check("custom 预设空", TRANSLATE_PROVIDERS["custom"] == {"base_url": "", "model": ""})

# ── 方向判定 ──
check("全中文 -> zh2en", detect_translate_direction("你好世界") == "zh2en")
check("全英文 -> en2zh", detect_translate_direction("hello world") == "en2zh")
check("中英混合 -> zh2en", detect_translate_direction("hello 世界") == "zh2en")
check("空串 -> en2zh", detect_translate_direction("") == "en2zh")
check("None -> en2zh", detect_translate_direction(None) == "en2zh")
check("标点无汉字 -> en2zh", detect_translate_direction("a, b! 123") == "en2zh")

# ── 目标语言 ──
check("zh2en 目标 English", target_language("zh2en") == "English")
check("en2zh 目标 Simplified Chinese", target_language("en2zh") == "Simplified Chinese")
check("未知方向回退 English", target_language("xx") == "English")

# ── messages ──
msgs = build_translate_messages("你好", "zh2en")
check("messages 两条", isinstance(msgs, list) and len(msgs) == 2)
check("system 角色", msgs[0]["role"] == "system" and "English" in msgs[0]["content"])
check("user 角色原文", msgs[1] == {"role": "user", "content": "你好"})
check("en2zh system 目标中文", "Simplified Chinese" in
      build_translate_messages("hello", "en2zh")[0]["content"])
check("非字符串输入转字符串", build_translate_messages(123, "en2zh")[1]["content"] == "123")

# ── payload ──
p = build_translate_payload("你好", "zh2en")
check("payload 默认模型", p["model"] == DEFAULT_TRANSLATE_MODEL)
check("payload stream False", p["stream"] is False)
check("payload temperature 0", p["temperature"] == 0.0)
check("payload 默认关思考", p["thinking"] == {"type": "disabled"})
p2 = build_translate_payload("hi", "en2zh", model="gpt-4o-mini", disable_thinking=False)
check("payload 自定义模型", p2["model"] == "gpt-4o-mini")
check("payload 不带 thinking 字段", "thinking" not in p2)
check("payload 空模型回退默认", build_translate_payload("hi", "en2zh", model="")["model"] == DEFAULT_TRANSLATE_MODEL)

# ── 响应解析 ──
ok = {"choices": [{"message": {"content": "  Hello world  "}}]}
check("解析并 strip", parse_translate_response(ok) == "Hello world")
check("缺 choices 抛错", raises(parse_translate_response, {}))
check("choices 空抛错", raises(parse_translate_response, {"choices": []}))
check("content 空抛错", raises(parse_translate_response, {"choices": [{"message": {"content": "   "}}]}))
check("content 非字符串抛错", raises(parse_translate_response, {"choices": [{"message": {"content": 5}}]}))
check("非 dict 抛错", raises(parse_translate_response, "nope"))

# ── 错误提取 ──
check("error dict message", extract_api_error({"error": {"message": "bad key"}}) == "bad key")
check("error dict type 兜底", extract_api_error({"error": {"type": "auth_error"}}) == "auth_error")
check("error 字符串", extract_api_error({"error": "boom"}) == "boom")
check("无 error 返回空", extract_api_error({"choices": []}) == "")
check("非 dict 返回空", extract_api_error(None) == "")

print(f"\nFAILURES: {len(failures)}")
if failures:
    sys.exit(1)
print("test_translation: all assertions passed")
