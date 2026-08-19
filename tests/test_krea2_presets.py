import importlib.util
import os
import sys
import tempfile
import types

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── mock torch / comfy / aiohttp / server / folder_paths ─────────────────
torch = types.ModuleType("torch")
torch.nn = types.SimpleNamespace()
sys.modules["torch"] = torch

comfy = types.ModuleType("comfy")
comfy.utils = types.SimpleNamespace(common_upscale=lambda s, w, h, m, s2: s)
comfy.sd = types.SimpleNamespace()
sys.modules["comfy"] = comfy
sys.modules["comfy.utils"] = comfy.utils

aiohttp = types.ModuleType("aiohttp")
web_mod = types.ModuleType("aiohttp.web")
web_mod.json_response = lambda data, status=200: type("R", (), {"data": data, "status": status})()
web_mod.Response = lambda status=200, text="": type("R", (), {"status": status})()
aiohttp.web = web_mod
sys.modules["aiohttp"] = aiohttp
sys.modules["aiohttp.web"] = web_mod

USER_DIR = tempfile.mkdtemp(prefix="sf_krea2_presets_")
folder_paths = types.ModuleType("folder_paths")
folder_paths.get_user_directory = lambda: USER_DIR
sys.modules["folder_paths"] = folder_paths


class _FakeRoutes:
    def __init__(self):
        self.handlers = {}
    def get(self, path):
        def deco(fn):
            self.handlers[("GET", path)] = fn
            return fn
        return deco
    def post(self, path):
        def deco(fn):
            self.handlers[("POST", path)] = fn
            return fn
        return deco
    def delete(self, path):
        def deco(fn):
            self.handlers[("DELETE", path)] = fn
            return fn
        return deco


server_mod = types.ModuleType("server")
server_mod.PromptServer = type("PS", (), {"instance": type("I", (), {"routes": _FakeRoutes()})()})
sys.modules["server"] = server_mod

# ── 注册 sfnodes 包结构（相对导入 from .logger import 需要）──────────────
pkg = types.ModuleType("sfnodes")
pkg.__path__ = [root]
nodes_pkg = types.ModuleType("sfnodes.nodes")
nodes_pkg.__path__ = [os.path.join(root, "nodes")]
model_pkg = types.ModuleType("sfnodes.nodes.model")
model_pkg.__path__ = [os.path.join(root, "nodes", "model")]
sf_utils_pkg = types.ModuleType("sfnodes.sf_utils")
sf_utils_pkg.__path__ = [os.path.join(root, "sf_utils")]
sys.modules["sfnodes"] = pkg
sys.modules["sfnodes.nodes"] = nodes_pkg
sys.modules["sfnodes.nodes.model"] = model_pkg
sys.modules["sfnodes.sf_utils"] = sf_utils_pkg

# 加载 sf_utils.krea2_presets
spec = importlib.util.spec_from_file_location(
    "sfnodes.sf_utils.krea2_presets", os.path.join(root, "sf_utils", "krea2_presets.py"))
kp = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = kp
spec.loader.exec_module(kp)

fail = 0

def check(name, cond):
    global fail
    print(f"[{'OK' if cond else 'FAIL'}] {name}")
    if not cond:
        fail += 1


# ── merge 纯逻辑 ─────────────────────────────────────────────────────────
BUILTIN = {"default": "D", "a": "A", "b": "B"}
check("merge 无用户存储=内置", kp.merge(BUILTIN, {"overrides": {}, "deleted": []}) == BUILTIN)
check("merge 覆盖内置文本", kp.merge(BUILTIN, {"overrides": {"a": "A2"}, "deleted": []}) ==
      {"default": "D", "a": "A2", "b": "B"})
check("merge 墓碑删除内置", kp.merge(BUILTIN, {"overrides": {}, "deleted": ["b"]}) ==
      {"default": "D", "a": "A"})
check("merge 新增追加末尾", kp.merge(BUILTIN, {"overrides": {"x": "X"}, "deleted": []}) ==
      {"default": "D", "a": "A", "b": "B", "x": "X"})
check("merge 覆盖+删除+新增", kp.merge(BUILTIN,
      {"overrides": {"a": "A2", "z": "Z"}, "deleted": ["default"]}) == {"a": "A2", "b": "B", "z": "Z"})
check("merge 覆盖+墓碑并存=墓碑胜出", kp.merge(BUILTIN,
      {"overrides": {"b": "B2"}, "deleted": ["b"]}) == {"default": "D", "a": "A"})
check("merge 非法 store 兜底", kp.merge(BUILTIN, None) == BUILTIN)
check("merge 空内置", kp.merge({}, {"overrides": {"x": "X"}, "deleted": []}) == {"x": "X"})

# ── 校验 ────────────────────────────────────────────────────────────────
check("_valid_name 空/斜杠/控制符拒", (not kp._valid_name(""), not kp._valid_name("a/b"),
      not kp._valid_name("a\\b"), not kp._valid_name("a\nb"), kp._valid_name(" 正常名 ")))
check("_valid_text 类型/长度", (kp._valid_text("ok"), kp._valid_text(""), not kp._valid_text(123),
      kp._valid_text("x" * 20000), not kp._valid_text("x" * 20001)))

# ── store 读写（临时用户目录） ───────────────────────────────────────────
kp._store_cache.clear()
check("初始空 store", kp.load_store("interrogator") == {"overrides": {}, "deleted": []})
kp.save_store("interrogator", {"overrides": {"u": "U"}, "deleted": ["b"]})
check("save 后读回", kp.load_store("interrogator") == {"overrides": {"u": "U"}, "deleted": ["b"]})
check("save 更新缓存命中", kp.load_store("interrogator")["overrides"] == {"u": "U"})

# ── 路由 CRUD（捕获 handler + 假请求）───────────────────────────────────
# 先清空用户存储（上面 store 读写测试写过数据），保证路由测试从干净状态开始
kp.save_store("interrogator", {"overrides": {}, "deleted": []})
register_fn = kp.register
register_fn("interrogator", BUILTIN)
routes = server_mod.PromptServer.instance.routes


class FakeReq:
    def __init__(self, body=None, query=None):
        self._body = body
        self._query = query or {}
        self.rel_url = types.SimpleNamespace(query=self._query)
    async def json(self):
        if self._body is None:
            raise Exception("no body")
        return self._body


async def call(method, path, body=None, query=None):
    handler = routes.handlers[(method, path)]
    r = handler(FakeReq(body=body, query=query))
    if hasattr(r, "__await__") or hasattr(r, "__aiter__"):
        r = await r
    return r.data if hasattr(r, "data") else r


async def main():
    global fail
    # GET 返回合并 + 元数据
    g = await call("GET", "/api/sfnodes/interrogator_presets")
    check("GET 合并视图", g["presets"] == {"default": "D", "a": "A", "b": "B"})
    check("GET 带内置元数据", set(g["builtin"].keys()) == {"default", "a", "b"})

    # POST 新增
    await call("POST", "/api/sfnodes/interrogator_presets", body={"name": "自定义", "text": "CUSTOM"})
    g = await call("GET", "/api/sfnodes/interrogator_presets")
    check("POST 新增追加", g["presets"].get("自定义") == "CUSTOM")

    # POST 修改内置
    await call("POST", "/api/sfnodes/interrogator_presets", body={"name": "a", "text": "A2"})
    g = await call("GET", "/api/sfnodes/interrogator_presets")
    check("POST 修改内置保持位置", g["presets"] == {"default": "D", "a": "A2", "b": "B", "自定义": "CUSTOM"})

    # POST 复活墓碑
    await call("DELETE", "/api/sfnodes/interrogator_presets", query={"name": "default"})
    g = await call("GET", "/api/sfnodes/interrogator_presets")
    check("DELETE 内置=墓碑", "default" not in g["presets"] and "default" in g["deleted"])
    await call("POST", "/api/sfnodes/interrogator_presets", body={"name": "default", "text": "D2"})
    g = await call("GET", "/api/sfnodes/interrogator_presets")
    check("POST 复活墓碑", g["presets"]["default"] == "D2" and "default" not in g["deleted"])

    # DELETE 用户新增
    await call("DELETE", "/api/sfnodes/interrogator_presets", query={"name": "自定义"})
    g = await call("GET", "/api/sfnodes/interrogator_presets")
    check("DELETE 用户新增移除", "自定义" not in g["presets"])

    # DELETE 不存在 → 404
    r = routes.handlers[("DELETE", "/api/sfnodes/interrogator_presets")](FakeReq(query={"name": "nope"}))
    if hasattr(r, "__await__"):
        r = await r
    check("DELETE 不存在 404", r.status == 404)

    # reset 单个
    await call("POST", "/api/sfnodes/interrogator_presets", body={"name": "a", "text": "A2"})
    await call("POST", "/api/sfnodes/interrogator_presets/reset", body={"name": "a"})
    g = await call("GET", "/api/sfnodes/interrogator_presets")
    check("reset 单个还原内置", g["presets"]["a"] == "A")

    # reset 全部
    await call("POST", "/api/sfnodes/interrogator_presets", body={"name": "zz", "text": "Z"})
    await call("POST", "/api/sfnodes/interrogator_presets/reset", body={"all": True})
    g = await call("GET", "/api/sfnodes/interrogator_presets")
    check("reset 全部清空用户改动", g["presets"] == {"default": "D", "a": "A", "b": "B"})
    check("reset 全部后 deleted 空", g["deleted"] == [])

    # 校验失败
    r = routes.handlers[("POST", "/api/sfnodes/interrogator_presets")](FakeReq(body={"name": "bad/name", "text": "x"}))
    if hasattr(r, "__await__"):
        r = await r
    check("POST 非法名 400", r.status == 400)

    # krea2 kind：保护 "none"
    register_fn("krea2", {"none": "", "default": "D"}, protected=("none",))
    r = routes.handlers[("DELETE", "/api/sfnodes/krea2_presets")](FakeReq(query={"name": "none"}))
    if hasattr(r, "__await__"):
        r = await r
    check("krea2 删除受保护 none 400", r.status == 400)
    g = await call("GET", "/api/sfnodes/krea2_presets")
    check("krea2 none 仍在", "none" in g["presets"])

    print(f"\n{'ALL PASS' if fail == 0 else str(fail) + ' FAILURES'}")
    sys.exit(1 if fail else 0)

import asyncio
asyncio.run(main())