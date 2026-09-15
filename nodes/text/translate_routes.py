"""SFPauseText 翻译路由（POST /api/sfnodes/translate）。

前端「中⇄EN」按钮把当前盒子文本提交到此端点，由后端代理调用 LLM 翻译 API
（浏览器直连 DeepSeek 会被 CORS 拦截）。API key / base_url / model 由前端从
ComfyUI Settings（sfnodes.Translate.*）读取后随请求体传入，后端不持久化密钥。

请求体：{"text": "...", "direction": "auto|zh2en|en2zh",
        "api_key": "...", "base_url": "...", "model": "..."}
响应：{"ok": true, "text": "...", "direction": "zh2en"} 或
      {"ok": false, "message": "..."}（恒 200，前端按 ok 分支）。

注册方式沿用 prompt_reader_routes._register_routes 先例：模块导入时副作用注册，
try/except 包裹，环境异常时降级不注册。
"""

import json

from ...sf_utils.translation import (
    DEFAULT_TRANSLATE_BASE_URL,
    DEFAULT_TRANSLATE_MODEL,
    build_translate_payload,
    detect_translate_direction,
    extract_api_error,
    parse_translate_response,
)

# 长文本 + 慢端点，给足总超时（含连接/读取）
_TIMEOUT = 120


def _register_routes():
    try:
        import aiohttp
        from aiohttp import web
        from server import PromptServer

        ins = getattr(PromptServer, "instance", None)
        if ins is None or not hasattr(ins, "routes"):
            print("[sfnodes] PromptServer instance not available, translate route not registered")
            return
        routes = ins.routes

        @routes.post("/api/sfnodes/translate")
        async def api_sf_translate(request):
            try:
                body = await request.json()
            except Exception:
                body = {}
            if not isinstance(body, dict):
                body = {}

            raw_text = body.get("text")
            text = raw_text if isinstance(raw_text, str) else ("" if raw_text is None else str(raw_text))
            if not text.strip():
                return web.json_response({"ok": False, "message": "没有可翻译的文本。"})

            api_key = str(body.get("api_key") or "").strip()
            base_url = str(body.get("base_url") or "").strip() or DEFAULT_TRANSLATE_BASE_URL
            model = str(body.get("model") or "").strip() or DEFAULT_TRANSLATE_MODEL
            direction = str(body.get("direction") or "auto")
            if direction not in ("zh2en", "en2zh"):
                direction = detect_translate_direction(text)

            if not api_key:
                return web.json_response({
                    "ok": False,
                    "message": "未配置 API Key：请在 设置 → SF Translate 中填写。",
                })

            url = base_url.rstrip("/") + "/chat/completions"
            payload = build_translate_payload(
                text,
                direction,
                model=model,
                # thinking 开关仅 DeepSeek 端点识别；其他 OpenAI 兼容端点会因
                # 未知字段 400，故按 base_url 判断
                disable_thinking="deepseek" in base_url.lower(),
            )
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            timeout = aiohttp.ClientTimeout(total=_TIMEOUT)
            data = None
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(url, json=payload, headers=headers) as resp:
                        raw = await resp.text()
                        try:
                            data = json.loads(raw)
                        except Exception:
                            data = None
                        if resp.status != 200:
                            msg = extract_api_error(data) if data else ""
                            return web.json_response({
                                "ok": False,
                                "message": msg or f"翻译请求失败（HTTP {resp.status}）。",
                            })
                if data is None:
                    return web.json_response({"ok": False, "message": "翻译响应不是有效 JSON。"})
                translated = parse_translate_response(data)
            except Exception as e:
                # asyncio.TimeoutError / aiohttp.ClientError / ValueError 均归此
                name = type(e).__name__
                if "Timeout" in name:
                    return web.json_response({"ok": False, "message": "翻译请求超时。"})
                if isinstance(e, ValueError):
                    return web.json_response({"ok": False, "message": str(e)})
                return web.json_response({"ok": False, "message": f"翻译失败：{e}"})

            return web.json_response({"ok": True, "text": translated, "direction": direction})

        print("[sfnodes] translate route registered (/api/sfnodes/translate)")
    except Exception as e:
        print(f"[sfnodes] translate route registration failed: {e}")


_register_routes()
