# SFCharacterSelect 后端逻辑测试（Python 直接运行：python3 tests/test_character.py）
# 覆盖：节点元信息、character_ 库过滤、单选收敛、草稿优先、三分镜独立输出
# （单分镜缺失仅该路占位）、未知库降级、角色路由形状/前缀隔离/穿越防护
import importlib.util
import json
import os
import sys
import tempfile
import types

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── mock aiohttp（本机无运行时依赖，仅需 web 形状）──
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

# ── mock server（ComfyUI 运行时提供；捕获路由注册供断言）──
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


_load("sfnodes.sf_utils.id_clothing", os.path.join(root, "sf_utils", "id_clothing.py"))
pure = _load("sfnodes.sf_utils.characters", os.path.join(root, "sf_utils", "characters.py"))
_load("sfnodes.nodes.text.styles_selector", os.path.join(root, "nodes", "text", "styles_selector.py"))
_load("sfnodes.nodes.text.id_clothing", os.path.join(root, "nodes", "text", "id_clothing.py"))
mod = _load("sfnodes.nodes.text.character", os.path.join(root, "nodes", "text", "character.py"))

failures = []


def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")


node = mod.SFCharacterSelect()
check("CATEGORY", node.CATEGORY == "sfnodes/text")
check("DESCRIPTION 存在", isinstance(node.DESCRIPTION, str) and len(node.DESCRIPTION) > 0)
check("RETURN_TYPES", node.RETURN_TYPES == ("STRING", "IMAGE", "IMAGE", "IMAGE"))
check("RETURN_NAMES", node.RETURN_NAMES == ("prompt", "face", "half", "full"))
check("FUNCTION = execute", node.FUNCTION == "execute")

it = node.INPUT_TYPES()
check("required 含 library", "library" in it["required"])
check("hidden 含双状态", set(it["hidden"]) == {"SFCharacterState", "SFCharacterPrompt"})

# ── character_ 库过滤 ──
_orig_names = mod.character_library_names
try:
    mod.character_library_names = lambda: ["fooocus_styles", "id_服装_女士", "character_主角", "other"]
    check("library 只列 character_ 前缀", mod._character_libraries() == ["character_主角"])
    mod.character_library_names = lambda: []
    check("空库占位 ['']", mod._character_libraries() == [""])
finally:
    mod.character_library_names = _orig_names

# ── 纯逻辑 ──
check("SHOTS 三分镜", pure.SHOTS == ("face", "half", "full"))
check("首个有效名", pure.first_selected('["A","B"]') == "A")
check("空状态", pure.first_selected("[]") == "")
check("畸形状态容错", pure.first_selected("{bad") == "")
check("单选序列化", pure.serialize_single("A") == '["A"]' and pure.serialize_single("") == "[]")
check("模板原词", pure.resolve_prompt([{"name": "A", "prompt": "pA"}], "A", "") == "pA")
check("草稿优先", pure.resolve_prompt([{"name": "A", "prompt": "pA"}], "A", "hand") == "hand")
check("未选中空串", pure.resolve_prompt([], "A", "") == "")
check("分镜取值", pure.shot_thumbnail({"face": "f.jpg", "half": ["h1.jpg", "h2.jpg"]}, "face") == "f.jpg")
check("分镜数组取首项", pure.shot_thumbnail({"half": ["h1.jpg", "h2.jpg"]}, "half") == "h1.jpg")
check("分镜缺省空串", pure.shot_thumbnail({}, "full") == "")

# ── 双源目录与加载（用户优先/同名覆盖/mtime 缓存）──
_orig_user, _orig_builtin = mod._user_characters_dir, mod._builtin_characters_dir
try:
    with tempfile.TemporaryDirectory() as user_dir, tempfile.TemporaryDirectory() as builtin_dir:
        mod._user_characters_dir = lambda: user_dir
        mod._builtin_characters_dir = lambda: builtin_dir
        with open(os.path.join(builtin_dir, "character_a.json"), "w", encoding="utf-8") as f:
            json.dump([{"name": "Builtin", "prompt": "b"}], f)
        with open(os.path.join(user_dir, "character_a.json"), "w", encoding="utf-8") as f:
            json.dump([{"name": "User", "prompt": "u"}], f)
        with open(os.path.join(user_dir, "character_b.json"), "w", encoding="utf-8") as f:
            json.dump([{"name": "Mine", "prompt": "m", "face": "samples_id_chara/character_b/face_01.jpg"}], f)
        check("库名枚举用户+内置", set(mod.character_library_names()) == {"character_a", "character_b"})
        check("同名库用户覆盖内置", mod._load_characters("character_a") == [{"name": "User", "prompt": "u"}])
        check("未知库空列表", mod._load_characters("nope") == [])
finally:
    mod._user_characters_dir, mod._builtin_characters_dir = _orig_user, _orig_builtin

# ── execute（三张量打桩为哨兵，避开本机 torch/PIL；
# 注意桩打在被测模块命名空间：character.py 已 from .id_clothing import 绑定，
# 打 id_clothing 模块影响不到它，见 §46 同款踩坑）──
_FAKE = {s: object() for s in ("face", "half", "full")}
_PLACEHOLDER = object()
_orig_load, _orig_ph = mod._load_template_tensor, mod._placeholder_tensor
calls = []


def _fake_load(path):
    calls.append(path)
    for shot, obj in _FAKE.items():
        if f"/{shot}_" in path or path.endswith(f"{shot}.jpg"):
            return obj
    return _FAKE["face"]


mod._load_template_tensor = _fake_load
mod._placeholder_tensor = lambda: _PLACEHOLDER
_orig_cls_load, _orig_cls_dirs = mod._load_characters, mod._characters_dirs
try:
    with tempfile.TemporaryDirectory() as user_dir:
        samples = os.path.join(user_dir, "samples_id_chara", "character_b")
        os.makedirs(samples)
        for shot in ("face", "half"):
            with open(os.path.join(samples, f"{shot}_01.jpg"), "wb") as f:
                f.write(b"jpgdata")
        # 注：full 缺文件 → 仅 full 路占位
        mod._characters_dirs = lambda: [user_dir]
        mod._load_characters = lambda name: [
            {"name": "主角", "prompt": "hero prompt",
             "face": "samples_id_chara/character_b/face_01.jpg",
             "half": "samples_id_chara/character_b/half_01.jpg",
             "full": "samples_id_chara/character_b/full_01.jpg"},
        ] if name == "character_b" else []

        r = node.execute(library="character_b", SFCharacterState='["主角"]', SFCharacterPrompt="")
        check("选中输出角色词+三分镜", r[0] == "hero prompt" and r[1] is _FAKE["face"]
              and r[2] is _FAKE["half"] and r[3] is _PLACEHOLDER)
        r = node.execute(library="character_b", SFCharacterState='["主角"]', SFCharacterPrompt="hand")
        check("草稿优先", r[0] == "hand")
        r = node.execute(library="character_b", SFCharacterState="[]", SFCharacterPrompt="")
        check("无选择全占位", r == ("", _PLACEHOLDER, _PLACEHOLDER, _PLACEHOLDER))
        r = node.execute(library="character_b", SFCharacterState='["不存在"]', SFCharacterPrompt="")
        check("未知角色降级", r == ("", _PLACEHOLDER, _PLACEHOLDER, _PLACEHOLDER))
        r = node.execute(library="character_b", SFCharacterState="{bad", SFCharacterPrompt="")
        check("畸形状态容错", r == ("", _PLACEHOLDER, _PLACEHOLDER, _PLACEHOLDER))
        r = node.execute(library="no_such_library", SFCharacterState='["主角"]', SFCharacterPrompt="")
        check("未知库降级", r == ("", _PLACEHOLDER, _PLACEHOLDER, _PLACEHOLDER))
finally:
    mod._load_template_tensor, mod._placeholder_tensor = _orig_load, _orig_ph
    mod._load_characters, mod._characters_dirs = _orig_cls_load, _orig_cls_dirs

# ── IS_CHANGED / VALIDATE_INPUTS ──
check("VALIDATE_INPUTS 恒 True", node.VALIDATE_INPUTS(library="stale_value") is True)
check("IS_CHANGED 空库返回 0", node.IS_CHANGED(library="") == 0)
check("IS_CHANGED 未知库返回 0", node.IS_CHANGED(library="nope") == 0)

# ── 归一化 ──
norm = mod.normalize_role_list([
    {"name": "A", "prompt": "p", "face": "samples_id_chara/x/face.jpg", "half": "https://h/half.jpg"},
    {"name": "B"},
    {"name": 123},
])
check("归一化保留 prompt", norm[0]["prompt"] == "p")
check("归一化本地转路由", norm[0]["face"] == "/api/sfnodes/characters/image?path=samples_id_chara/x/face.jpg")
check("归一化远程原样", norm[0]["half"] == "https://h/half.jpg")
check("归一化缺图不带键", all(k not in norm[1] for k in ("face", "half", "full")))
check("归一化非 name 条目跳过", len(norm) == 2)

# ── 路由 handler（捕获注册）──
check("路由已注册 /api/sfnodes/characters", "/api/sfnodes/characters" in handlers)
check("路由已注册 /api/sfnodes/characters/image", "/api/sfnodes/characters/image" in handlers)


class _FakeRequest:
    def __init__(self, query):
        self.rel_url = types.SimpleNamespace(query=query)


async def _run(handler, req):
    return await handler(req)


import asyncio

with tempfile.TemporaryDirectory() as user_dir, tempfile.TemporaryDirectory() as builtin_dir:
    mod._user_characters_dir = lambda: user_dir
    mod._builtin_characters_dir = lambda: builtin_dir
    with open(os.path.join(user_dir, "character_x.json"), "w", encoding="utf-8") as f:
        json.dump([{"name": "R", "prompt": "p", "face": "samples_id_chara/character_x/face.jpg"}], f)
    os.makedirs(os.path.join(user_dir, "samples_id_chara", "character_x"))
    with open(os.path.join(user_dir, "samples_id_chara", "character_x", "face.jpg"), "wb") as f:
        f.write(b"jpgdata")

    resp = asyncio.run(_run(handlers["/api/sfnodes/characters"], _FakeRequest({"name": "character_x"})))
    check("列表路由返回条目", len(resp) == 1 and resp[0]["name"] == "R")
    check("列表路由缺 name 400", asyncio.run(_run(handlers["/api/sfnodes/characters"], _FakeRequest({}))).status == 400)
    check("列表路由非前缀 400", asyncio.run(_run(handlers["/api/sfnodes/characters"],
                                                         _FakeRequest({"name": "fooocus_styles"}))).status == 400)
    check("列表路由未知库空数组", asyncio.run(_run(handlers["/api/sfnodes/characters"],
                                                           _FakeRequest({"name": "character_nope"}))) == [])
    resp = asyncio.run(_run(handlers["/api/sfnodes/characters/image"],
                              _FakeRequest({"path": "samples_id_chara/character_x/face.jpg"})))
    check("图片路由命中", resp.status == 200)
    check("图片路由穿越 404", asyncio.run(_run(handlers["/api/sfnodes/characters/image"],
                                                   _FakeRequest({"path": "../../etc/passwd"}))).status == 404)
    check("图片路由缺参 400", asyncio.run(_run(handlers["/api/sfnodes/characters/image"],
                                                 _FakeRequest({}))).status == 400)
mod._user_characters_dir, mod._builtin_characters_dir = _orig_user, _orig_builtin

print()
if failures:
    print(f"FAILED: {len(failures)}: {failures}")
    sys.exit(1)
print("ALL PASS")
