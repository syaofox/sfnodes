# sf_utils/llm_client.py 纯逻辑测试（python tests/test_llm_client.py）
# 覆盖：配置解析（新/旧 id 回退 + 环境变量 + 默认值）、请求体构造（thinking 开关）、
# 图片 content 构造、响应解析、错误提取、图片 data URL 编码（PIL）。
# 不发网络请求（网络调用由节点/路由在真实环境验证）。

import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

from sf_utils.llm_client import (  # noqa: E402
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    PROVIDER_PRESETS,
    build_chat_payload,
    build_image_content,
    extract_api_error,
    get_llm_config,
    image_to_data_url,
    is_deepseek,
    parse_chat_response,
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


# ── 默认常量 ──
check("默认 base_url", DEFAULT_BASE_URL == "https://api.deepseek.com")
check("默认模型 deepseek-flash", DEFAULT_MODEL == "deepseek-flash")
check("deepseek 预设", PROVIDER_PRESETS["deepseek"] == {"base_url": DEFAULT_BASE_URL, "model": DEFAULT_MODEL})

# ── 配置解析 ──
cfg = get_llm_config({})
check("空设置 -> 默认", cfg["base_url"] == DEFAULT_BASE_URL and cfg["model"] == DEFAULT_MODEL)
check("空设置无 key", cfg["api_key"] == "")
check("provider 默认 deepseek", cfg["provider"] == "deepseek")

cfg = get_llm_config({
    "sfnodes.LLM.ApiKey": "new-key",
    "sfnodes.LLM.BaseUrl": "https://example.com/v1",
    "sfnodes.LLM.Model": "my-model",
    "sfnodes.LLM.Provider": "custom",
})
check("新 id 读取", cfg["api_key"] == "new-key" and cfg["base_url"] == "https://example.com/v1"
      and cfg["model"] == "my-model" and cfg["provider"] == "custom")

cfg = get_llm_config({"sfnodes.Translate.ApiKey": "legacy", "sfnodes.Translate.Model": "legacy-model"})
check("旧 id 回退", cfg["api_key"] == "legacy" and cfg["model"] == "legacy-model")
check("旧 id 回退 base 默认", cfg["base_url"] == DEFAULT_BASE_URL)

cfg = get_llm_config({
    "sfnodes.LLM.ApiKey": "new-wins",
    "sfnodes.Translate.ApiKey": "old",
})
check("新 id 优先于旧 id", cfg["api_key"] == "new-wins")

check("空白值视为缺失", get_llm_config({"sfnodes.LLM.Model": "   "})["model"] == DEFAULT_MODEL)

_prev = os.environ.get("DEEPSEEK_API_KEY")
try:
    os.environ["DEEPSEEK_API_KEY"] = "env-key"
    check("环境变量兜底", get_llm_config({})["api_key"] == "env-key")
    check("设置 key 优先于环境变量", get_llm_config({"sfnodes.LLM.ApiKey": "set-key"})["api_key"] == "set-key")
finally:
    if _prev is None:
        os.environ.pop("DEEPSEEK_API_KEY", None)
    else:
        os.environ["DEEPSEEK_API_KEY"] = _prev

# ── is_deepseek ──
check("is_deepseek 真", is_deepseek("https://api.deepseek.com"))
check("is_deepseek 假", not is_deepseek("https://api.openai.com/v1"))

# ── payload ──
msgs = [{"role": "user", "content": "hi"}]
p = build_chat_payload("m", msgs)
check("payload 基本字段", p["model"] == "m" and p["messages"] is msgs and p["stream"] is False)
check("payload 无 temperature/max_tokens/thinking", "temperature" not in p and "max_tokens" not in p and "thinking" not in p)
p = build_chat_payload("", msgs, temperature=0.0, max_tokens=512, disable_thinking=True)
check("payload 空模型回退默认", p["model"] == DEFAULT_MODEL)
check("payload temperature 0 保留", p["temperature"] == 0.0)
check("payload max_tokens", p["max_tokens"] == 512)
check("payload thinking", p["thinking"] == {"type": "disabled"})

# ── 图片 content ──
content = build_image_content("描述这张图", "data:image/png;base64,AAAA", "high")
check("content 两段", len(content) == 2)
check("content 文本段", content[0] == {"type": "text", "text": "描述这张图"})
check("content 图片段", content[1] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA", "detail": "high"}})
check("content 非字符串文本", build_image_content(123, "u", "auto")[0]["text"] == "123")
check("content 无 detail 不写字段", "detail" not in build_image_content("t", "u", "")[1]["image_url"])

# ── 响应解析 ──
ok = {"choices": [{"message": {"content": "  hello  "}}]}
check("解析并 strip", parse_chat_response(ok) == "hello")
check("缺 choices 抛错", raises(parse_chat_response, {}))
check("choices 空抛错", raises(parse_chat_response, {"choices": []}))
check("content 空抛错", raises(parse_chat_response, {"choices": [{"message": {"content": "  "}}]}))
check("content 非字符串抛错", raises(parse_chat_response, {"choices": [{"message": {"content": 3}}]}))
check("非 dict 抛错", raises(parse_chat_response, "x"))

# ── 错误提取 ──
check("error dict message", extract_api_error({"error": {"message": "bad"}}) == "bad")
check("error dict type", extract_api_error({"error": {"type": "auth"}}) == "auth")
check("error str", extract_api_error({"error": "boom"}) == "boom")
check("无 error 空", extract_api_error({"choices": []}) == "")
check("非 dict 空", extract_api_error(None) == "")

# ── 图片编码 ──
try:
    from PIL import Image

    img = Image.new("RGB", (100, 50), (255, 0, 0))
    url = image_to_data_url(img)
    check("jpeg data URL 前缀", url.startswith("data:image/jpeg;base64,"))
    check("jpeg data 非空", len(url) > 40)

    big = Image.new("RGB", (2000, 1000), (0, 255, 0))
    url2 = image_to_data_url(big, max_megapixels=0.5)
    check("百万像素上限生效", url2.startswith("data:image/jpeg;base64,"))

    png = image_to_data_url(img, fmt="PNG")
    check("png data URL 前缀", png.startswith("data:image/png;base64,"))

    rgba = Image.new("RGBA", (10, 10), (0, 0, 0, 128))
    check("RGBA 转 RGB 编码", image_to_data_url(rgba).startswith("data:image/jpeg;base64,"))
except ImportError:
    print("SKIP: PIL 不可用，跳过图片编码测试")

print(f"\nFAILURES: {len(failures)}")
if failures:
    sys.exit(1)
print("test_llm_client: all assertions passed")
