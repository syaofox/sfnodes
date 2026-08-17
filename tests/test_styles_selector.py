# SFStylesSelector 后端逻辑测试（Python 直接运行：python tests/test_styles_selector.py）
# 覆盖：INPUT_TYPES 结构、样式库枚举/用户覆盖内置、{prompt} 占位拼接全分支、
# select_styles 接线优先、归一化、IS_CHANGED、路由 handler 响应形状
import importlib.util
import json
import os
import sys
import tempfile
import types

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── mock aiohttp（本机无运行时依赖，仅需 web.json_response 形状）──
fake_web = types.ModuleType("aiohttp")


class _FakeWeb:
    @staticmethod
    def json_response(payload, **kw):
        return payload

    @staticmethod
    def Response(*args, **kw):
        return types.SimpleNamespace(status=args[0] if args else kw.get("status", 200), text=kw.get("text", None))

    @staticmethod
    def FileResponse(path):
        return types.SimpleNamespace(status=200, path=path)


fake_web.web = _FakeWeb()
sys.modules["aiohttp"] = fake_web

# ── mock server（ComfyUI 运行时提供；捕获路由注册的 handler 供断言）──
handlers = {}


class _FakeRoutes:
    def get(self, path):
        def deco(fn):
            handlers[path] = fn
            return fn

        return deco


fake_server = types.ModuleType("server")
fake_server.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(routes=_FakeRoutes()))
sys.modules["server"] = fake_server

spec = importlib.util.spec_from_file_location(
    "sfnodes.nodes.text.styles_selector",
    os.path.join(root, "nodes", "text", "styles_selector.py"),
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


node = mod.SFStylesSelector()
check("CATEGORY", node.CATEGORY == "sfnodes/text")
check("DESCRIPTION 存在", isinstance(node.DESCRIPTION, str) and len(node.DESCRIPTION) > 0)
check("RETURN_TYPES", node.RETURN_TYPES == ("STRING", "STRING"))
check("RETURN_NAMES", node.RETURN_NAMES == ("positive", "negative"))
check("FUNCTION = execute", node.FUNCTION == "execute")

it = node.INPUT_TYPES()
check("required 含 styles", "styles" in it["required"])
check("styles 含 fooocus_styles 内置库", "fooocus_styles" in it["required"]["styles"][0])
check("styles 默认 fooocus_styles", it["required"]["styles"][1]["default"] == "fooocus_styles")
check("optional 含 positive/negative/select_styles",
      set(it["optional"]) == {"positive", "negative", "select_styles"})
check("positive forceInput", it["optional"]["positive"][1].get("forceInput") is True)
check("negative forceInput", it["optional"]["negative"][1].get("forceInput") is True)
check("select_styles forceInput", it["optional"]["select_styles"][1].get("forceInput") is True)
check("hidden 含 SFStylesState", "SFStylesState" in it["hidden"])

# ── 内置库加载（复制自 Easy-Use 的 fooocus_styles.json）──
data = mod._load_styles("fooocus_styles")
check("fooocus_styles 加载", len(data) == 275)
check("样式条目含 name/prompt/negative_prompt", all(d.get("name") for d in data)
      and all("negative_prompt" in d for d in data))
check("样式条目含 name_cn/thumbnail", all("name_cn" in d for d in data) and all("thumbnail" in d for d in data))

# ── 样式库枚举与用户覆盖内置（monkeypatch 目录为临时目录，事后恢复）──
_orig_user_dir, _orig_builtin_dir = mod._user_styles_dir, mod._builtin_styles_dir
try:
    with tempfile.TemporaryDirectory() as user_dir, tempfile.TemporaryDirectory() as builtin_dir:
        mod._user_styles_dir = lambda: user_dir
        mod._builtin_styles_dir = lambda: builtin_dir

        with open(os.path.join(builtin_dir, "fooocus_styles.json"), "w", encoding="utf-8") as f:
            json.dump([{"name": "Builtin A", "prompt": "builtin {prompt}"}], f)
        with open(os.path.join(builtin_dir, "extra_styles.json"), "w", encoding="utf-8") as f:
            json.dump([], f)
        with open(os.path.join(user_dir, "fooocus_styles.json"), "w", encoding="utf-8") as f:
            json.dump([{"name": "User A", "prompt": "user {prompt}"}], f)
        with open(os.path.join(user_dir, "my_styles.json"), "w", encoding="utf-8") as f:
            json.dump([{"name": "Mine", "prompt": "mine"}], f)

        names = mod.style_library_names()
        check("库名枚举含内置+用户", set(names) == {"fooocus_styles", "extra_styles", "my_styles"})

        # 同名文件：用户目录优先
        u = mod._load_styles("fooocus_styles")
        check("同名库用户覆盖内置", u == [{"name": "User A", "prompt": "user {prompt}"}])
        check("mtime 缓存重载", mod._load_styles("my_styles") == [{"name": "Mine", "prompt": "mine"}])
finally:
    mod._user_styles_dir, mod._builtin_styles_dir = _orig_user_dir, _orig_builtin_dir

# ── 拼接逻辑 _apply_styles 全分支 ──
styles_data = [
    {"name": "WithPrompt", "prompt": "cinematic {prompt}, sharp", "negative_prompt": "blur"},
    {"name": "NoPrompt", "prompt": "masterpiece", "negative_prompt": "text"},
    {"name": "TraitPrompt", "prompt": "moody, {prompt}"},
    {"name": "NegOnly", "negative_prompt": "watermark"},
    {"name": "Unknown"},
]

# 1) 第一个含占位替换用户输入；后续含占位剥离 ", {prompt}" 片段
p, n = mod._apply_styles(styles_data, ["WithPrompt", "TraitPrompt"], "a girl", "base_neg")
check("首个占位替换 + 后续剥离", p == "cinematic a girl, sharp, moody")
check("negative 拼接用户输入之后", n == "base_neg, blur")

# 2) 无占位样式直接尾接（positive 空时不触发前置拼接）
p, _ = mod._apply_styles(styles_data, ["NoPrompt"], "", "")
check("无占位拼接", p == "masterpiece")

# 3) 用户输入未被消费 → 前置拼接（1:1 原版怪癖：无分隔逗号、末尾尾逗号）
p, _ = mod._apply_styles(styles_data, ["NoPrompt"], "a girl", "")
check("未消费前置（原版尾逗号怪癖）", p == "a girlmasterpiece, ")

# 4) negative 为空时直接取样式负面
_, n = mod._apply_styles(styles_data, ["NegOnly", "NoPrompt"], "", "")
check("空 negative 取样式负面", n == "watermark, text")

# 5) 未知样式跳过（不报错、不影响）
p, n = mod._apply_styles(styles_data, ["Unknown", "NoPrompt"], "", "x")
check("未知样式跳过", p == "masterpiece" and n == "x, text")

# ── execute：select_styles 接线优先 + SFStylesState 解析 ──
mod._load_styles("fooocus_styles")  # 预热缓存（用真实内置库）
sel_style = data[1]["name"]
prompt0 = data[1].get("prompt", "")
expect_p = prompt0.replace("{prompt}", "hello") if "{prompt}" in prompt0 else prompt0
r = node.execute(styles="fooocus_styles", positive="hello", negative="",
                 select_styles=sel_style, SFStylesState=json.dumps([]))
check("接线优先于前端状态", r[0] == expect_p)

r = node.execute(styles="fooocus_styles", positive="hello", negative="neg0",
                 select_styles="", SFStylesState=json.dumps([sel_style]))
check("SFStylesState 生效", r[0] == expect_p and r[1].startswith("neg0"))

r = node.execute(styles="fooocus_styles", positive="hello", negative="neg0",
                 select_styles=None, SFStylesState=None)
check("无选择透传", r == ("hello", "neg0"))

# 接线值带空格（原版 split 匹配不到的 bug，本实现 strip 修复）
r = node.execute(styles="fooocus_styles", positive="hello", negative="",
                 select_styles=f" {sel_style} , Fooocus Masterpiece", SFStylesState="[]")
check("接线值空格 strip", r[0].startswith(expect_p))

# 畸形状态容错
r = node.execute(styles="fooocus_styles", positive="a", negative="b",
                 select_styles="", SFStylesState="{bad json")
check("畸形状态容错", r == ("a", "b"))

# 未知库名安全降级
r = node.execute(styles="no_such_library", positive="a", negative="b",
                 select_styles="", SFStylesState="[]")
check("未知库名降级", r == ("a", "b"))

# ── IS_CHANGED / VALIDATE_INPUTS ──
sig = node.IS_CHANGED(styles="fooocus_styles")
check("IS_CHANGED 返回 (mtime, size)", isinstance(sig, tuple) and len(sig) == 2 and isinstance(sig[1], int))
check("IS_CHANGED 未知库返回 0", node.IS_CHANGED(styles="nope") == 0)
check("VALIDATE_INPUTS 恒 True", node.VALIDATE_INPUTS(styles="stale_value") is True)

# ── 归一化 ──
norm = mod.normalize_style_list([
    {"name": "Remote", "thumbnail": "https://x/y.jpg"},
    {"name": "Local", "thumbnail": "samples/a.jpg"},
    {"name": "None"},
    {"name": "Multi", "thumbnail": ["http://a/b.jpg", "local/c.jpg"]},
    {"name": "Zh", "name_cn": "中文", "thumbnail": "https://z/w.jpg", "prompt": "p {prompt}", "negative_prompt": "n"},
    {"name": 123, "prompt": "bad"},
], "fooocus_styles")
check("归一化 http 原样", norm[0]["thumbnail"] == "https://x/y.jpg")
check("归一化本地转路由", norm[1]["thumbnail"] == "/api/sfnodes/styles/image?path=samples/a.jpg")
check("归一化缺省兜底 name 查询", norm[2]["thumbnail"] == "/api/sfnodes/styles/image?name=None&styles_name=fooocus_styles")
check("归一化列表取首项", norm[3]["thumbnail"] == "http://a/b.jpg")
check("归一化 name_cn 保留", norm[4]["name_cn"] == "中文")
check("归一化 prompt/negative_prompt 保留", norm[4]["prompt"] == "p {prompt}" and norm[4]["negative_prompt"] == "n")
check("归一化非 dict 条目跳过", len(norm) == 5)
check("归一化无 name_cn 条目不带该键", all("name_cn" not in d for d in norm[:4]))
check("归一化无 prompt 条目不带该键", all("prompt" not in d for d in norm[:4]))

# ── 路由 handler 响应形状（捕获注册）──
check("路由已注册 /api/sfnodes/styles", "/api/sfnodes/styles" in handlers)
check("路由已注册 /api/sfnodes/styles/image", "/api/sfnodes/styles/image" in handlers)


class _FakeRequest:
    def __init__(self, query):
        self.rel_url = types.SimpleNamespace(query=query)


async def _run(handler, req):
    return await handler(req)


import asyncio

# 列表路由
resp = asyncio.run(_run(handlers["/api/sfnodes/styles"], _FakeRequest({"name": "fooocus_styles"})))
check("列表路由返回全部条目", len(resp) == 275 and resp[0]["name"])
check("列表条目含 prompt/negative_prompt（hover 浮窗；空串条目不携带该键）",
      any("prompt" in d for d in resp) and sum(1 for d in resp if "negative_prompt" in d) == 273)
resp400 = asyncio.run(_run(handlers["/api/sfnodes/styles"], _FakeRequest({})))
check("列表路由缺 name 400", resp400.status == 400)
resp404 = asyncio.run(_run(handlers["/api/sfnodes/styles"], _FakeRequest({"name": "nope"})))
check("列表路由未知库空数组", resp404 == [])

# 图片路由：fooocus 库无本地样例 → 返回远程 URL 文本
resp = asyncio.run(_run(handlers["/api/sfnodes/styles/image"], _FakeRequest({"name": "fooocus_enhance", "styles_name": "fooocus_styles"})))
check("图片路由远程 URL 文本", isinstance(resp.text, str) and resp.text.startswith("https://raw.githubusercontent.com/lllyasviel/Fooocus"))
resp404 = asyncio.run(_run(handlers["/api/sfnodes/styles/image"], _FakeRequest({"name": "x", "styles_name": "other"})))
check("图片路由非 fooocus 无本地图 404", resp404.status == 404)
resp400 = asyncio.run(_run(handlers["/api/sfnodes/styles/image"], _FakeRequest({})))
check("图片路由缺参 400", resp400.status == 400)

# 本地样例文件（用户目录 samples/）与路径穿越防护
with tempfile.TemporaryDirectory() as user_dir, tempfile.TemporaryDirectory() as builtin_dir:
    mod._user_styles_dir = lambda: user_dir
    mod._builtin_styles_dir = lambda: builtin_dir
    os.makedirs(os.path.join(user_dir, "samples"))
    with open(os.path.join(user_dir, "samples", "foo.jpg"), "wb") as f:
        f.write(b"jpgdata")
    resp = asyncio.run(_run(handlers["/api/sfnodes/styles/image"], _FakeRequest({"name": "foo", "styles_name": "other"})))
    check("本地 samples 文件返回 FileResponse", hasattr(resp, "body") or resp.status == 200)
    resp = asyncio.run(_run(handlers["/api/sfnodes/styles/image"], _FakeRequest({"path": "samples/foo.jpg"})))
    check("path 查询返回本地文件", resp is not None)
    resp404 = asyncio.run(_run(handlers["/api/sfnodes/styles/image"], _FakeRequest({"path": "../../etc/passwd"})))
    check("path 穿越防护 404", resp404.status == 404)
    resp404 = asyncio.run(_run(handlers["/api/sfnodes/styles/image"], _FakeRequest({"name": "../../etc/passwd", "styles_name": "x"})))
    check("name 穿越防护 404", resp404.status == 404)

print()
if failures:
    print(f"FAILED: {len(failures)}: {failures}")
    sys.exit(1)
print("ALL PASS")
