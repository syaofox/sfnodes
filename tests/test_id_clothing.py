# SFIDClothingSelector 后端逻辑测试（Python 直接运行：python3 tests/test_id_clothing.py）
# 覆盖：节点元信息、id_ 库过滤、单选收敛、草稿优先、无选择/未知库降级、
# thumbnail 落盘解析（含穿越防护）、孤海文件名映射、IS_CHANGED/VALIDATE
import importlib.util
import json
import os
import sys
import tempfile
import types

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── mock aiohttp（本机无运行时依赖，仅需 web 形状；styles_selector 顶层 import 用）──
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

# ── mock server（ComfyUI 运行时提供；捕获路由注册供断言“零新增路由”）──
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

# ── 包桩（相对 import 解析用；不执行根 __init__ 重依赖）──
for _name in ["sfnodes", "sfnodes.nodes", "sfnodes.nodes.text", "sfnodes.sf_utils"]:
    _m = types.ModuleType(_name)
    _m.__path__ = []
    sys.modules[_name] = _m


def _load(full, path):
    spec = importlib.util.spec_from_file_location(full, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod


pure = _load("sfnodes.sf_utils.id_clothing", os.path.join(root, "sf_utils", "id_clothing.py"))
styles = _load("sfnodes.nodes.text.styles_selector", os.path.join(root, "nodes", "text", "styles_selector.py"))
mod = _load("sfnodes.nodes.text.id_clothing", os.path.join(root, "nodes", "text", "id_clothing.py"))

failures = []


def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")


node = mod.SFIDClothingSelector()
check("CATEGORY", node.CATEGORY == "sfnodes/text")
check("DESCRIPTION 存在", isinstance(node.DESCRIPTION, str) and len(node.DESCRIPTION) > 0)
check("RETURN_TYPES", node.RETURN_TYPES == ("STRING", "IMAGE"))
check("RETURN_NAMES", node.RETURN_NAMES == ("prompt", "template_image"))
check("FUNCTION = execute", node.FUNCTION == "execute")

it = node.INPUT_TYPES()
check("required 含 library", "library" in it["required"])
check("hidden 含双状态", set(it["hidden"]) == {"SFIDClothingState", "SFIDClothingPrompt"})

# ── id_ 库过滤（monkeypatch 节点模块绑定的枚举名，事后恢复）──
_orig_names = mod.style_library_names
try:
    mod.style_library_names = lambda: ["fooocus_styles", "id_服装_女士", "id_发型_男士", "other"]
    libs = mod._id_libraries()
    check("library 只列 id_ 前缀", libs == ["id_服装_女士", "id_发型_男士"])
    mod.style_library_names = lambda: ["fooocus_styles"]
    check("空库占位 ['']", mod._id_libraries() == [""])
finally:
    mod.style_library_names = _orig_names

# ── 纯逻辑：单选收敛/草稿优先/孤海映射 ──
check("首个有效名", pure.first_selected('["A","B"]') == "A")
check("空状态", pure.first_selected("[]") == "")
check("畸形状态容错", pure.first_selected("{bad") == "")
check("单选序列化", pure.serialize_single("A") == '["A"]' and pure.serialize_single("") == "[]")
check("模板原词", pure.resolve_prompt([{"name": "A", "prompt": "pA"}], "A", "") == "pA")
check("草稿优先", pure.resolve_prompt([{"name": "A", "prompt": "pA"}], "A", "hand") == "hand")
check("未选中空串", pure.resolve_prompt([], "A", "") == "")
check("孤海切分", pure.parse_goohai_filename("西装- wear suit") == ("西装", "wear suit"))
check("孤海无提示", pure.parse_goohai_filename("无提示") == ("无提示", ""))
check("孤海条目映射", pure.goohai_entry("西装", "suit", "a.jpg") == {"name": "西装", "prompt": "suit", "thumbnail": "samples/a.jpg"})

# ── thumbnail 落盘解析 ──
with tempfile.TemporaryDirectory() as user_dir:
    os.makedirs(os.path.join(user_dir, "samples"))
    with open(os.path.join(user_dir, "samples", "foo.jpg"), "wb") as f:
        f.write(b"jpgdata")
    p = pure.resolve_thumbnail_path("samples/foo.jpg", [user_dir])
    check("本地 samples 命中", p is not None and p.endswith(os.path.join("samples", "foo.jpg")))
    p = pure.resolve_thumbnail_path("/api/sfnodes/styles/image?path=samples/foo.jpg", [user_dir])
    check("路由 ?path= 形式命中", p is not None)
    check("远程返回 None", pure.resolve_thumbnail_path("https://x/y.jpg", [user_dir]) is None)
    check("穿越防护", pure.resolve_thumbnail_path("../../etc/passwd", [user_dir]) is None)
    check("name 穿越防护", pure.resolve_thumbnail_path("samples/../../etc/passwd", [user_dir]) is None)
    check("缺失文件 None", pure.resolve_thumbnail_path("samples/nope.jpg", [user_dir]) is None)

# ── execute（张量函数打桩为哨兵，避开本机 torch/PIL）──
_FAKE_IMG, _PLACEHOLDER = object(), object()
_orig_load, _orig_ph = mod._load_template_tensor, mod._placeholder_tensor
mod._load_template_tensor = lambda path: _FAKE_IMG
mod._placeholder_tensor = lambda: _PLACEHOLDER
_orig_load_styles, _orig_dirs = mod._load_styles, mod._styles_dirs
try:
    with tempfile.TemporaryDirectory() as user_dir:
        os.makedirs(os.path.join(user_dir, "samples"))
        with open(os.path.join(user_dir, "samples", "suit.jpg"), "wb") as f:
            f.write(b"jpgdata")
        mod._styles_dirs = lambda: [user_dir]
        mod._load_styles = lambda name: [
            {"name": "西装", "prompt": "suit prompt", "thumbnail": "samples/suit.jpg"},
            {"name": "无图", "prompt": "nopic", "thumbnail": "https://x/y.jpg"},
        ] if name == "id_服装_男士" else []

        r = node.execute(library="id_服装_男士", SFIDClothingState='["西装"]', SFIDClothingPrompt="")
        check("选中输出模板词+实图", r == ("suit prompt", _FAKE_IMG))
        r = node.execute(library="id_服装_男士", SFIDClothingState='["西装"]', SFIDClothingPrompt="hand")
        check("草稿优先于模板词", r[0] == "hand" and r[1] is _FAKE_IMG)
        r = node.execute(library="id_服装_男士", SFIDClothingState="[]", SFIDClothingPrompt="")
        check("无选择空串+占位", r == ("", _PLACEHOLDER))
        r = node.execute(library="id_服装_男士", SFIDClothingState='["不存在"]', SFIDClothingPrompt="")
        check("未知条目降级", r == ("", _PLACEHOLDER))
        r = node.execute(library="id_服装_男士", SFIDClothingState="{bad", SFIDClothingPrompt="")
        check("畸形状态容错", r == ("", _PLACEHOLDER))
        r = node.execute(library="id_服装_男士", SFIDClothingState='["无图"]', SFIDClothingPrompt="")
        check("远程缩略图占位但词正常", r == ("nopic", _PLACEHOLDER))
        r = node.execute(library="no_such_library", SFIDClothingState='["西装"]', SFIDClothingPrompt="")
        check("未知库降级", r == ("", _PLACEHOLDER))
finally:
    mod._load_template_tensor, mod._placeholder_tensor = _orig_load, _orig_ph
    mod._load_styles, mod._styles_dirs = _orig_load_styles, _orig_dirs

# ── IS_CHANGED / VALIDATE_INPUTS ──
sig = node.IS_CHANGED(library="fooocus_styles")
check("IS_CHANGED 返回 (mtime, size)", isinstance(sig, tuple) and len(sig) == 2 and isinstance(sig[1], int))
check("IS_CHANGED 未知库返回 0", node.IS_CHANGED(library="nope") == 0)
check("IS_CHANGED 空库返回 0", node.IS_CHANGED(library="") == 0)
check("VALIDATE_INPUTS 恒 True", node.VALIDATE_INPUTS(library="stale_value") is True)

# ── 零新增路由（复用 styles 双路由）──
check("复用列表路由", "/api/sfnodes/styles" in handlers)
check("复用图片路由", "/api/sfnodes/styles/image" in handlers)
check("无 id_clothing 专属路由", not any("id_clothing" in k for k in handlers))

print()
if failures:
    print(f"FAILED: {len(failures)}: {failures}")
    sys.exit(1)
print("ALL PASS")
