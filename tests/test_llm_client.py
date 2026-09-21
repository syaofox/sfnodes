# sf_utils/llm_client.py 纯逻辑测试（python tests/test_llm_client.py）
# 覆盖：配置解析（新/旧 id 回退 + 环境变量 + 默认值）、请求体构造（thinking 开关）、
# 图片 content 构造、响应解析、错误提取、图片 data URL 编码（PIL）。
# 不发网络请求（网络调用由节点/路由在真实环境验证）。

import json
import os
import sys
import tempfile

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

from sf_utils.llm_client import (  # noqa: E402
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    PROVIDER_PRESETS,
    LruCache,
    build_chat_payload,
    build_image_content,
    chat_completion_async,
    chat_completion_sync,
    extract_api_error,
    get_llm_config,
    image_content_parts,
    image_to_data_url,
    is_deepseek,
    make_cache_key,
    parse_chat_response,
)
import sf_utils.llm_client as L  # noqa: E402

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
check("payload 默认不含 seed", "seed" not in build_chat_payload("m", msgs))
check("payload 含 seed", build_chat_payload("m", msgs, seed=123)["seed"] == 123)
check("payload seed=0 也下发", build_chat_payload("m", msgs, seed=0)["seed"] == 0)

# ── 图片 content ──
content = build_image_content("描述这张图", "data:image/png;base64,AAAA", "high")
check("content 两段", len(content) == 2)
check("content 文本段", content[0] == {"type": "text", "text": "描述这张图"})
check("content 图片段", content[1] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA", "detail": "high"}})
check("content 非字符串文本", build_image_content(123, "u", "auto")[0]["text"] == "123")
check("content 无 detail 不写字段", "detail" not in build_image_content("t", "u", "")[1]["image_url"])
multi = build_image_content("多图", ["data:image/jpeg;base64,AA", "data:image/jpeg;base64,BB"], "low")
check("多图 content 三段", len(multi) == 3 and multi[0]["type"] == "text")
check("多图按序", [p["image_url"]["url"][-2:] for p in multi[1:]] == ["AA", "BB"])
check("多图 detail 一致", all(p["image_url"]["detail"] == "low" for p in multi[1:]))
check("空列表仅文本段", len(build_image_content("t", [], "auto")) == 1)
check("列表内空值跳过", len(image_content_parts(["", "data:x"])) == 1)
check("单串与列表等价", image_content_parts("data:x") == image_content_parts(["data:x"]))

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

# ── cache_enabled 配置读取 ──
check("cache_enabled 默认 True", get_llm_config({})["cache_enabled"] is True)
check("cache_enabled bool False", get_llm_config({"sfnodes.LLM.CacheEnabled": False})["cache_enabled"] is False)
check("cache_enabled 字符串 false", get_llm_config({"sfnodes.LLM.CacheEnabled": "false"})["cache_enabled"] is False)
check("cache_enabled 字符串 true", get_llm_config({"sfnodes.LLM.CacheEnabled": "yes"})["cache_enabled"] is True)

# ── make_cache_key ──
pA = build_chat_payload("m", msgs, temperature=0.0)
pB = build_chat_payload("m", msgs, temperature=0.0)
check("cache key 稳定", make_cache_key("u", pA) == make_cache_key("u", pB))
check("cache key 端点敏感", make_cache_key("u", pA) != make_cache_key("v", pA))
check("cache key payload 敏感", make_cache_key("u", pA) != make_cache_key("u", build_chat_payload("m", msgs, temperature=1.0)))
check("cache key extra 敏感", make_cache_key("u", pA, (1,)) != make_cache_key("u", pA, (2,)))

# ── LruCache ──
c = LruCache(2)
check("空缓存 len 0", len(c) == 0)
c.set("a", 1)
c.set("b", 2)
check("命中提升前", c.get("a") == 1)
c.set("c", 3)   # "b" 最久未用 -> 淘汰
check("超容淘汰最久未用 b", c.get("b") is None)
check("保留 a/c", c.get("a") == 1 and c.get("c") == 3 and len(c) == 2)
c.set("a", 9)
check("覆盖同键", c.get("a") == 9 and len(c) == 2)
c.clear()
check("clear", len(c) == 0)

# ── 缓存集成（打桩网络层，不发请求）──
_sync_calls = {"n": 0}
_async_calls = {"n": 0}
_orig_sync = L._do_request_sync
_orig_async = L._do_request_async


def _fake_sync(base_url, payload, api_key, timeout):
    _sync_calls["n"] += 1
    return f"SYNC-{_sync_calls['n']}"


async def _fake_async(base_url, payload, api_key, timeout):
    _async_calls["n"] += 1
    return f"ASYNC-{_async_calls['n']}"


L._do_request_sync = _fake_sync
L._do_request_async = _fake_async
try:
    L.response_cache.clear()
    cfg = {"base_url": "u", "model": "m", "api_key": "k", "cache_enabled": True}
    msgs = [{"role": "user", "content": "same"}]
    r1 = chat_completion_sync(cfg, msgs, temperature=0.0)
    r2 = chat_completion_sync(cfg, msgs, temperature=0.0)
    check("同参数第二次命中缓存", r1 == r2 and _sync_calls["n"] == 1)
    r3 = chat_completion_sync(cfg, msgs, temperature=0.0, cache_key_extra=(1,))
    check("不同 cache_key_extra 不命中", _sync_calls["n"] == 2 and r3 != r1)
    r4 = chat_completion_sync(cfg, msgs, temperature=0.0, cache_key_extra=(1,))
    check("相同 extra 命中", r4 == r3 and _sync_calls["n"] == 2)
    chat_completion_sync(cfg, msgs, temperature=0.0, seed=7)
    chat_completion_sync(cfg, msgs, temperature=0.0, seed=7)
    check("seed 进缓存键且命中", _sync_calls["n"] == 3 and L.response_cache.get(
        make_cache_key("u", build_chat_payload("m", msgs, temperature=0.0, seed=7))) == "SYNC-3")
    chat_completion_sync(cfg, msgs, temperature=0.0, use_cache=False)
    check("use_cache=False 绕过缓存", _sync_calls["n"] == 4)

    cfg_off = {"base_url": "u", "model": "m", "api_key": "k", "cache_enabled": False}
    chat_completion_sync(cfg_off, msgs, temperature=0.0)
    chat_completion_sync(cfg_off, msgs, temperature=0.0)
    check("cache_enabled=False 不缓存", _sync_calls["n"] == 6)

    # 同步写入 → 异步命中（同进程同缓存）
    import asyncio

    amsgs = [{"role": "user", "content": "async-only"}]
    a1 = asyncio.run(chat_completion_async(cfg, amsgs, temperature=0.0))
    a2 = asyncio.run(chat_completion_async(cfg, amsgs, temperature=0.0))
    check("异步同参数命中缓存", a1 == a2 and _async_calls["n"] == 1)

    bmsgs = [{"role": "user", "content": "cross-sync-async"}]
    before_async = _async_calls["n"]
    chat_completion_sync(cfg, bmsgs, temperature=0.0)
    asyncio.run(chat_completion_async(cfg, bmsgs, temperature=0.0))
    check("异步复用同步写入的缓存", _async_calls["n"] == before_async)
finally:
    L._do_request_sync = _orig_sync
    L._do_request_async = _orig_async
    L.response_cache.clear()

# ── 设置读盘缓存（(path, mtime_ns, size) 命中即返回；文件变更自然失效）──
_tmp_dir = tempfile.mkdtemp(prefix="sf_llm_settings_")
_settings_file = os.path.join(_tmp_dir, "comfy.settings.json")
_orig_settings_path = L._settings_path
L._settings_path = lambda: _settings_file
try:
    check("设置缺失 -> {}", L.read_comfy_settings() == {})

    with open(_settings_file, "w", encoding="utf-8") as fh:
        json.dump({"sfnodes.LLM.ApiKey": "k1"}, fh)
    check("设置首次读取", L.read_comfy_settings().get("sfnodes.LLM.ApiKey") == "k1")

    stat_before = os.stat(_settings_file)
    with open(_settings_file, "w", encoding="utf-8") as fh:
        json.dump({"sfnodes.LLM.ApiKey": "k2"}, fh)
    os.utime(_settings_file, ns=(stat_before.st_atime_ns, stat_before.st_mtime_ns))
    check("同 mtime/size 命中缓存", L.read_comfy_settings().get("sfnodes.LLM.ApiKey") == "k1")

    with open(_settings_file, "w", encoding="utf-8") as fh:
        json.dump({"sfnodes.LLM.ApiKey": "k3"}, fh)
    os.utime(_settings_file, ns=(stat_before.st_atime_ns, stat_before.st_mtime_ns + 10**9))
    check("mtime 变化后重读", L.read_comfy_settings().get("sfnodes.LLM.ApiKey") == "k3")

    with open(_settings_file, "w", encoding="utf-8") as fh:
        fh.write("{broken")
    os.utime(_settings_file, ns=(stat_before.st_atime_ns, stat_before.st_mtime_ns + 2 * 10**9))
    check("损坏 JSON -> {}", L.read_comfy_settings() == {})
finally:
    L._settings_path = _orig_settings_path

print(f"\nFAILURES: {len(failures)}")
if failures:
    sys.exit(1)
print("test_llm_client: all assertions passed")
