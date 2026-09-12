"""画布菜单内存清理后端路由（POST /api/sfnodes/memory/ram）。

VRAM 不在此处理——前端直接调 ComfyUI 原生 POST /free（server.py:1192，
队列 flag 有序执行卸载 + soft_empty_cache，与官方释放语义一致，零后端
改动）。浏览器 JS 无法释放服务端进程内存，RAM 清理必须经后端执行，故
本模块仅暴露 RAM 路由，逻辑复用 memory_cleanup.py 的 RAMCleanup
（实例化调用，零复制；retry_times 取 1，菜单点击是交互操作，不宜按
节点默认 3 次 sleep 等待）。
"""

from aiohttp import web

from .memory_cleanup import RAMCleanup


def _register_memory_routes():
    """注册内存清理路由（PromptServer 副作用注册，canvas_size.py 同款）。"""
    try:
        from server import PromptServer

        ins = getattr(PromptServer, "instance", None)
        if ins is None or not hasattr(ins, "routes"):
            return
        routes = ins.routes

        @routes.post("/api/sfnodes/memory/ram")
        async def _memory_ram(request):
            cleaner = RAMCleanup()
            before_usage, before_available = cleaner.get_ram_usage()
            cleaner.clean_ram(True, True, True, 1)
            after_usage, after_available = cleaner.get_ram_usage()
            return web.json_response(
                {
                    "ok": True,
                    "before_usage": round(before_usage, 1),
                    "after_usage": round(after_usage, 1),
                    "freed_mb": round(after_available - before_available),
                }
            )

    except Exception:
        pass


_register_memory_routes()
