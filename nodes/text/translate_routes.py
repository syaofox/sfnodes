"""SFPauseText 翻译路由（POST /api/sfnodes/translate）。

前端「中⇄EN」按钮把当前盒子文本提交到此端点，由后端代理调用 LLM 翻译 API
（浏览器直连 DeepSeek 会被 CORS 拦截）。API key / base_url / model 由后端从
ComfyUI Settings（`sfnodes.LLM.*`）读取——密钥不再经浏览器往返。

请求体：{"text": "...", "direction": "auto|zh2en|en2zh"}
响应：{"ok": true, "text": "...", "direction": "zh2en"} 或
      {"ok": false, "message": "..."}（恒 200，前端按 ok 分支）。

注册方式沿用 prompt_reader_routes._register_routes 先例：模块导入时副作用注册，
try/except 包裹，环境异常时降级不注册。
"""

from ...sf_utils.llm_client import chat_completion_async, get_llm_config
from ...sf_utils.translation import build_translate_messages, detect_translate_direction

# 长文本 + 慢端点，给足总超时（含连接/读取）
_TIMEOUT = 120


def _register_routes():
    try:
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

            direction = str(body.get("direction") or "auto")
            if direction not in ("zh2en", "en2zh"):
                direction = detect_translate_direction(text)

            config = get_llm_config()
            if not config.get("api_key"):
                return web.json_response({
                    "ok": False,
                    "message": "未配置 API Key：请在 设置 → SF LLM 中填写。",
                })

            try:
                translated = await chat_completion_async(
                    config,
                    build_translate_messages(text, direction),
                    temperature=0.0,
                    timeout=_TIMEOUT,
                )
            except Exception as e:
                if "Timeout" in type(e).__name__:
                    return web.json_response({"ok": False, "message": "翻译请求超时。"})
                if isinstance(e, (ValueError, RuntimeError)):
                    return web.json_response({"ok": False, "message": str(e)})
                return web.json_response({"ok": False, "message": f"翻译失败：{e}"})

            return web.json_response({"ok": True, "text": translated, "direction": direction})

        print("[sfnodes] translate route registered (/api/sfnodes/translate)")
    except Exception as e:
        print(f"[sfnodes] translate route registration failed: {e}")


_register_routes()
