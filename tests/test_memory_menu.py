# SF Memory 菜单后端路由测试（Node/Python 直接运行：python3 tests/test_memory_menu.py）
# 覆盖（全部 mock，不碰真实 torch/comfy/aiohttp）：
#   - POST /api/sfnodes/memory/ram 路由已注册
#   - handler 复用 memory_cleanup.RAMCleanup（非内联副本）：clean_ram 以
#     (True, True, True, 1) 调用（菜单交互只跑 1 轮，不按节点默认 3 次等待）
#   - 返回 JSON 形状 {ok, before_usage, after_usage, freed_mb}，
#     freed_mb = after_available - before_available
# mock：comfy/comfy.model_management（桩）、aiohttp.web（json_response 回显
# payload）、server.PromptServer（routes.post 捕获 handler）
import asyncio
import importlib.util
import os
import sys
import types

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

failures = []


def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")


# ── stub comfy（memory_cleanup 顶层 import 用）──
comfy = types.ModuleType("comfy")
comfy_mm = types.ModuleType("comfy.model_management")
comfy_mm.unload_all_models = lambda: None
comfy_mm.cleanup_models_gc = lambda: None
comfy_mm.soft_empty_cache = lambda: None
sys.modules["comfy"] = comfy
sys.modules["comfy.model_management"] = comfy_mm

# ── stub aiohttp.web（json_response 回显 payload 供断言）──
aiohttp = types.ModuleType("aiohttp")


class _JsonResp:
    def __init__(self, payload):
        self.payload = payload


web = types.SimpleNamespace(
    Request=type("Request", (), {}),
    json_response=lambda payload: _JsonResp(payload),
)
aiohttp.web = web
sys.modules["aiohttp"] = aiohttp
sys.modules["aiohttp.web"] = web

# ── stub server.PromptServer（捕获 routes.post 注册）──
posted = {}


def _post(path):
    def deco(fn):
        posted[path] = fn
        return fn

    return deco


routes = types.SimpleNamespace(post=_post)
ins = types.SimpleNamespace(routes=routes)
server = types.ModuleType("server")
server.PromptServer = types.SimpleNamespace(instance=ins)
sys.modules["server"] = server

# ── 包壳（sfnodes 根 __init__ 重依赖，不执行；只建命名空间）──
pkg = types.ModuleType("sfnodes")
pkg.__path__ = [root]
sys.modules["sfnodes"] = pkg
pkg_nodes = types.ModuleType("sfnodes.nodes")
pkg_nodes.__path__ = [os.path.join(root, "nodes")]
sys.modules["sfnodes.nodes"] = pkg_nodes
pkg_utils = types.ModuleType("sfnodes.nodes.utils")
pkg_utils.__path__ = [os.path.join(root, "nodes", "utils")]
sys.modules["sfnodes.nodes.utils"] = pkg_utils
pkg_sf = types.ModuleType("sfnodes.sf_utils")
pkg_sf.__path__ = [os.path.join(root, "sf_utils")]
sys.modules["sfnodes.sf_utils"] = pkg_sf


def _load(modname, path):
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


common = _load("sfnodes.sf_utils.common", os.path.join(root, "sf_utils", "common.py"))
cleanup = _load(
    "sfnodes.nodes.utils.memory_cleanup",
    os.path.join(root, "nodes", "utils", "memory_cleanup.py"),
)
mem_routes = _load(
    "sfnodes.nodes.utils.memory_routes",
    os.path.join(root, "nodes", "utils", "memory_routes.py"),
)

check("路由已注册 POST /api/sfnodes/memory/ram", "/api/sfnodes/memory/ram" in posted)

# ── handler 行为：get_ram_usage 序列 + clean_ram 参数断言 ──
calls = {}
usage_seq = [(60.0, 8000.0), (50.0, 9000.0)]


def _fake_usage(self):
    return usage_seq.pop(0)


def _fake_clean(self, clean_file_cache, clean_processes, clean_buffers, retry_times, **kw):
    calls["clean"] = (clean_file_cache, clean_processes, clean_buffers, retry_times)


cleanup.RAMCleanup.get_ram_usage = _fake_usage
cleanup.RAMCleanup.clean_ram = _fake_clean

handler = posted["/api/sfnodes/memory/ram"]
resp = asyncio.run(handler(object()))
check("返回 ok=True", resp.payload.get("ok") is True)
check("clean_ram 复用且参数 (True,True,True,1)", calls.get("clean") == (True, True, True, 1))
check("before/after 回显", resp.payload.get("before_usage") == 60.0 and resp.payload.get("after_usage") == 50.0)
check("freed_mb = after - before", resp.payload.get("freed_mb") == 1000)

# handler 与节点同源（非内联副本）：路由模块引用的是 memory_cleanup 的类
check("RAMCleanup 同源", mem_routes.RAMCleanup is cleanup.RAMCleanup)

print("OK" if not failures else f"{len(failures)} FAILURES")
sys.exit(1 if failures else 0)
