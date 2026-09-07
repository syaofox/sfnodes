# sf_utils/text_presets 后端逻辑测试（Node/Python 直接运行：python tests/test_text_presets.py）
# 覆盖：存储读写/缓存重载/校验/find_text + GET/POST/DELETE 路由（mock aiohttp/server/folder_paths）
import asyncio
import importlib.util
import json
import os
import sys
import tempfile
import types

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# mock aiohttp.web + server.PromptServer（test_prompt_preset.py 范式，补 post/delete）
aiohttp = types.ModuleType("aiohttp")
web_mod = types.ModuleType("aiohttp.web")


class _FakeJsonResponse:
    def __init__(self, data, status=200):
        self.data = data
        self.status = status


web_mod.json_response = lambda data, status=200: _FakeJsonResponse(data, status)
aiohttp.web = web_mod
sys.modules["aiohttp"] = aiohttp
sys.modules["aiohttp.web"] = web_mod


class _FakeRoutes:
    def __init__(self):
        self.handlers = {}

    def _reg(self, method, path):
        def deco(fn):
            self.handlers[(method, path)] = fn
            return fn
        return deco

    def get(self, path):
        return self._reg("GET", path)

    def post(self, path):
        return self._reg("POST", path)

    def delete(self, path):
        return self._reg("DELETE", path)


routes = _FakeRoutes()
server_mod = types.ModuleType("server")
server_mod.PromptServer = type("PS", (), {"instance": type("I", (), {"routes": routes})()})
sys.modules["server"] = server_mod

# mock folder_paths：user 目录指向临时目录
tmp_user = tempfile.mkdtemp(prefix="sf_tp_test_")
fp = types.ModuleType("folder_paths")
fp.get_user_directory = lambda: tmp_user
sys.modules["folder_paths"] = fp

# 包上下文（相对 import `from .logger` 需要）
pkg = types.ModuleType("sfnodes")
pkg.__path__ = [root]
sfu_pkg = types.ModuleType("sfnodes.sf_utils")
sfu_pkg.__path__ = [os.path.join(root, "sf_utils")]
sys.modules["sfnodes"] = pkg
sys.modules["sfnodes.sf_utils"] = sfu_pkg

spec = importlib.util.spec_from_file_location(
    "sfnodes.sf_utils.text_presets",
    os.path.join(root, "sf_utils", "text_presets.py"),
)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

failures = []


def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")


STORE = os.path.join(tmp_user, "sfnodes", "text_presets.json")


class _FakeRequest:
    def __init__(self, body=None, query=None, bad_json=False):
        self._body = body
        self._bad_json = bad_json
        self.rel_url = types.SimpleNamespace(query=query or {})

    async def json(self):
        if self._bad_json:
            raise ValueError("bad json")
        return self._body


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


# 1. 路由注册
check("GET 路由已注册", ("GET", "/api/sfnodes/text_presets") in routes.handlers)
check("POST 路由已注册", ("POST", "/api/sfnodes/text_presets") in routes.handlers)
check("DELETE 路由已注册", ("DELETE", "/api/sfnodes/text_presets") in routes.handlers)

# 2. 空库
r = run(routes.handlers[("GET", "/api/sfnodes/text_presets")](None))
check("空库返回空列表", r.data == {"presets": []} and r.status == 200)

# 3. POST 校验
r = run(routes.handlers[("POST", "/api/sfnodes/text_presets")](_FakeRequest(bad_json=True)))
check("非法 JSON 400", r.status == 400)
r = run(routes.handlers[("POST", "/api/sfnodes/text_presets")](_FakeRequest({"name": "", "text": "x"})))
check("空名 400", r.status == 400)
r = run(routes.handlers[("POST", "/api/sfnodes/text_presets")](_FakeRequest({"name": "a/b", "text": "x"})))
check("路径分隔符名 400", r.status == 400)
r = run(routes.handlers[("POST", "/api/sfnodes/text_presets")](_FakeRequest({"name": "a\\b", "text": "x"})))
check("反斜杠名 400", r.status == 400)
r = run(routes.handlers[("POST", "/api/sfnodes/text_presets")](_FakeRequest({"name": "A\x01", "text": "x"})))
check("控制字符名 400", r.status == 400)
r = run(routes.handlers[("POST", "/api/sfnodes/text_presets")](_FakeRequest({"name": "A", "text": 123})))
check("非字符串 text 400", r.status == 400)
r = run(routes.handlers[("POST", "/api/sfnodes/text_presets")](_FakeRequest({"name": "A" * 201, "text": "x"})))
check("超长名 400", r.status == 400)
r = run(routes.handlers[("POST", "/api/sfnodes/text_presets")](_FakeRequest({"name": "A", "text": "x" * 20001})))
check("超长 text 400", r.status == 400)

# 4. POST 落盘 + GET
r = run(routes.handlers[("POST", "/api/sfnodes/text_presets")](_FakeRequest({"name": "  A  ", "text": "hello"})))
check("POST 成功", r.data == {"ok": True, "name": "A"} and r.status == 200)
check("文件落盘 user/sfnodes", os.path.isfile(STORE))
disk = json.load(open(STORE, encoding="utf-8"))
check("磁盘结构 {presets: [{name,text}]}", disk == {"presets": [{"name": "A", "text": "hello"}]})
r = run(routes.handlers[("POST", "/api/sfnodes/text_presets")](_FakeRequest({"name": "B", "text": ""})))
check("空文本允许保存", r.status == 200)
r = run(routes.handlers[("GET", "/api/sfnodes/text_presets")](None))
check("GET 返回两条", [p["name"] for p in r.data["presets"]] == ["A", "B"])

# 5. 同名覆盖（不追加）
r = run(routes.handlers[("POST", "/api/sfnodes/text_presets")](_FakeRequest({"name": "A", "text": "hello2"})))
check("同名覆盖成功", r.status == 200)
check("同名覆盖不追加", [p["text"] for p in mod.load_presets()] == ["hello2", ""])

# 6. find_text
check("find_text 命中", mod.find_text("A") == "hello2")
check("find_text 空文本命中返回空串", mod.find_text("B") == "")
check("find_text 未命中返回 None", mod.find_text("Nope") is None)
check("find_text 空名返回 None", mod.find_text("") is None)

# 7. DELETE 校验与删除
r = run(routes.handlers[("DELETE", "/api/sfnodes/text_presets")](
    _FakeRequest(query={"name": "Nope"})))
check("删除不存在 404", r.status == 404)
r = run(routes.handlers[("DELETE", "/api/sfnodes/text_presets")](
    _FakeRequest(query={"name": ""})))
check("删除空名 400", r.status == 400)
r = run(routes.handlers[("DELETE", "/api/sfnodes/text_presets")](
    _FakeRequest(query={"name": "B"})))
check("删除成功", r.data == {"deleted": "B"})
check("删除后磁盘同步", [p["name"] for p in mod.load_presets()] == ["A"])

# 8. 文件变化自动重载（mtime+size 缓存）
with open(STORE, "w", encoding="utf-8") as f:
    json.dump({"presets": [{"name": "A", "text": "hello2"}, {"name": "C", "text": "ccc"}]}, f, ensure_ascii=False)
check("外部修改自动重载", [p["name"] for p in mod.load_presets()] == ["A", "C"])

# 9. load_presets 容错
with open(STORE, "w", encoding="utf-8") as f:
    f.write("not json{{{")
mod._cache["sig"] = None
check("非法 JSON 返回空列表", mod.load_presets() == [])
with open(STORE, "w", encoding="utf-8") as f:
    json.dump({"presets": [{"name": "", "text": "x"}, "junk", {"name": "A", "text": "ok"}, {"name": "A", "text": "dup"}]}, f, ensure_ascii=False)
mod._cache["sig"] = None
check("归一化过滤无效条目与重名", mod.load_presets() == [{"name": "A", "text": "ok"}])

# 10. 节点 execute 全局优先 + 回退（加载节点模块验证联动）
spec2 = importlib.util.spec_from_file_location(
    "sfnodes.nodes.text.text_preset",
    os.path.join(root, "nodes", "text", "text_preset.py"),
)
comfy = types.ModuleType("comfy")
node_typing = types.ModuleType("comfy.comfy_types")
node_typing_module = types.ModuleType("comfy.comfy_types.node_typing")


class IO:
    STRING = "STRING"


node_typing_module.IO = IO
comfy.comfy_types = node_typing
comfy.comfy_types.node_typing = node_typing_module
sys.modules["comfy"] = comfy
sys.modules["comfy.comfy_types"] = node_typing
sys.modules["comfy.comfy_types.node_typing"] = node_typing_module

nodes_pkg = types.ModuleType("sfnodes.nodes")
nodes_pkg.__path__ = [os.path.join(root, "nodes")]
text_pkg = types.ModuleType("sfnodes.nodes.text")
text_pkg.__path__ = [os.path.join(root, "nodes", "text")]
sys.modules["sfnodes.nodes"] = nodes_pkg
sys.modules["sfnodes.nodes.text"] = text_pkg

node_mod = importlib.util.module_from_spec(spec2)
sys.modules[spec2.name] = node_mod
spec2.loader.exec_module(node_mod)

node = node_mod.SFTextPreset()
t, n = node.execute("A", '[{"name": "A", "text": "workflow"}]')
check("全局库优先", t == "ok" and n == "A")
t, n = node.execute("WF", '[{"name": "WF", "text": "workflow"}]')
check("未命中回退工作流 presets_json", t == "workflow" and n == "WF")
t, n = node.execute("Nope", "[]")
check("双未命中返回空文本", t == "" and n == "Nope")
t, n = node.execute("Nope", "not json{{{")
check("回退非法 JSON 容错", t == "" and n == "Nope")

print("\nFAILURES:", len(failures))
sys.exit(1 if failures else 0)
