# sf_utils/crop_expand_presets 后端逻辑测试（Node/Python 直接运行：python tests/test_crop_expand_presets.py）
# 覆盖：存储读写/缓存重载/校验/归一化 + GET/POST/DELETE 路由（mock aiohttp/server/folder_paths）
import asyncio
import importlib.util
import json
import os
import sys
import tempfile
import types

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# mock aiohttp.web + server.PromptServer（test_text_presets.py 范式）
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
tmp_user = tempfile.mkdtemp(prefix="sf_cep_test_")
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
    "sfnodes.sf_utils.crop_expand_presets",
    os.path.join(root, "sf_utils", "crop_expand_presets.py"),
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


STORE = os.path.join(tmp_user, "sfnodes", "crop_expand_presets.json")
PATH = "/api/sfnodes/crop_expand_presets"


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
check("GET 路由已注册", ("GET", PATH) in routes.handlers)
check("POST 路由已注册", ("POST", PATH) in routes.handlers)
check("DELETE 路由已注册", ("DELETE", PATH) in routes.handlers)

# 2. 空库
r = run(routes.handlers[("GET", PATH)](None))
check("空库返回空列表", r.data == {"presets": []} and r.status == 200)

# 3. POST 校验
r = run(routes.handlers[("POST", PATH)](_FakeRequest(bad_json=True)))
check("非法 JSON 400", r.status == 400)
r = run(routes.handlers[("POST", PATH)](_FakeRequest({"name": "", "w": 1, "h": 1})))
check("空名 400", r.status == 400)
r = run(routes.handlers[("POST", PATH)](_FakeRequest({"name": "a/b", "w": 1, "h": 1})))
check("路径分隔符名 400", r.status == 400)
r = run(routes.handlers[("POST", PATH)](_FakeRequest({"name": "A\x01", "w": 1, "h": 1})))
check("控制字符名 400", r.status == 400)
r = run(routes.handlers[("POST", PATH)](_FakeRequest({"name": "A" * 201, "w": 1, "h": 1})))
check("超长名 400", r.status == 400)
r = run(routes.handlers[("POST", PATH)](_FakeRequest({"name": "A", "w": 0, "h": 1})))
check("w=0 400", r.status == 400)
r = run(routes.handlers[("POST", PATH)](_FakeRequest({"name": "A", "w": -1, "h": 1})))
check("w<0 400", r.status == 400)
r = run(routes.handlers[("POST", PATH)](_FakeRequest({"name": "A", "w": "2", "h": 1})))
check("字符串 w 400", r.status == 400)
r = run(routes.handlers[("POST", PATH)](_FakeRequest({"name": "A", "w": 10001, "h": 1})))
check("w 超上限 400", r.status == 400)
r = run(routes.handlers[("POST", PATH)](_FakeRequest({"name": "A", "w": 1, "h": True})))
check("bool h 400", r.status == 400)
r = run(routes.handlers[("POST", PATH)](_FakeRequest({"name": "A"})))
check("缺 w/h 400", r.status == 400)

# 4. POST 落盘 + GET
r = run(routes.handlers[("POST", PATH)](_FakeRequest({"name": "  Banner  ", "w": 3, "h": 1})))
check("POST 成功（名去空白）", r.data == {"ok": True, "name": "Banner"} and r.status == 200)
check("文件落盘 user/sfnodes", os.path.isfile(STORE))
disk = json.load(open(STORE, encoding="utf-8"))
check("磁盘结构 {presets: [{name,w,h}]}", disk == {"presets": [{"name": "Banner", "w": 3.0, "h": 1.0}]})
r = run(routes.handlers[("POST", PATH)](_FakeRequest({"name": "Tall", "w": 9, "h": 16})))
check("追加第二条", r.status == 200)
r = run(routes.handlers[("GET", PATH)](None))
check("GET 返回两条保序", [p["name"] for p in r.data["presets"]] == ["Banner", "Tall"])

# 5. 同名覆盖（不追加）
r = run(routes.handlers[("POST", PATH)](_FakeRequest({"name": "Banner", "w": 4, "h": 1})))
check("同名覆盖成功", r.status == 200)
check("同名覆盖不追加", [p["name"] for p in mod.load_presets()] == ["Banner", "Tall"]
      and mod.load_presets()[0] == {"name": "Banner", "w": 4.0, "h": 1.0})

# 6. DELETE 校验与删除
r = run(routes.handlers[("DELETE", PATH)](_FakeRequest(query={"name": "Nope"})))
check("删除不存在 404", r.status == 404)
r = run(routes.handlers[("DELETE", PATH)](_FakeRequest(query={"name": ""})))
check("删除空名 400", r.status == 400)
r = run(routes.handlers[("DELETE", PATH)](_FakeRequest(query={"name": "Tall"})))
check("删除成功", r.data == {"deleted": "Tall"})
check("删除后磁盘同步", [p["name"] for p in mod.load_presets()] == ["Banner"])

# 7. 文件变化自动重载（mtime+size 缓存）
with open(STORE, "w", encoding="utf-8") as f:
    json.dump({"presets": [{"name": "Banner", "w": 4, "h": 1}, {"name": "C", "w": 2, "h": 1}]}, f, ensure_ascii=False)
check("外部修改自动重载", [p["name"] for p in mod.load_presets()] == ["Banner", "C"])

# 8. load_presets 容错与归一化
with open(STORE, "w", encoding="utf-8") as f:
    f.write("not json{{{")
mod._cache["sig"] = None
check("非法 JSON 返回空列表", mod.load_presets() == [])
with open(STORE, "w", encoding="utf-8") as f:
    json.dump({"presets": [
        {"name": "", "w": 1, "h": 1},
        "junk",
        {"name": "A", "w": 3, "h": 1},
        {"name": "A", "w": 9, "h": 9},
        {"name": "B", "w": 0, "h": 1},
        {"name": "C", "w": 1, "h": "x"},
        {"name": "D", "w": 2, "h": 2},
    ]}, f, ensure_ascii=False)
mod._cache["sig"] = None
check("归一化过滤无效与重名", mod.load_presets() == [{"name": "A", "w": 3.0, "h": 1.0}, {"name": "D", "w": 2.0, "h": 2.0}])

# 9. _normalize_presets 接受裸列表
check("裸列表归一化", mod._normalize_presets([{"name": "X", "w": 5, "h": 4}]) == [{"name": "X", "w": 5.0, "h": 4.0}])

# 10. crop_expand.py 触发路由注册（文本断言：import 行存在）
src = open(os.path.join(root, "nodes", "image", "crop_expand.py"), encoding="utf-8").read()
check("crop_expand.py 导入预设模块触发注册", "from ...sf_utils import crop_expand_presets" in src)

print("\nFAILURES:", len(failures))
sys.exit(1 if failures else 0)
