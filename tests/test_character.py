# SFCharacterSelect 后端逻辑测试（Python 直接运行：python3 tests/test_character.py）
# 覆盖：节点元信息（3 输出）、character_ 库过滤、新旧双读、单选多选收敛、
# 草稿整体覆盖、角色级 prompt、batch 拼装（首图归一/空选占位）、未知库降级、
# 角色路由形状/前缀隔离/穿越防护
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

# ── mock torch（本机无；batch 拼装只用 zeros/cat，形状可断言）──
fake_torch = types.ModuleType("torch")
fake_torch.float32 = "float32"


class _FakeTensor:
    def __init__(self, shape, tag=""):
        self.shape = tuple(shape)
        self.tag = tag


def _fake_zeros(shape, dtype=None):
    return _FakeTensor(shape, tag="zeros")


def _fake_cat(tensors, dim=0):
    b = sum(t.shape[0] for t in tensors)
    return _FakeTensor((b,) + tensors[0].shape[1:], tag="cat")


fake_torch.zeros = _fake_zeros
fake_torch.cat = _fake_cat
sys.modules["torch"] = fake_torch

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
check("RETURN_TYPES", node.RETURN_TYPES == ("STRING", "STRING", "IMAGE"))
check("RETURN_NAMES", node.RETURN_NAMES == ("prompt", "role_prompt", "images"))
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

# ── 纯逻辑：双读/收敛/拼接 ──
NEW = {"name": "A", "prompt": "R", "images": [
    {"label": "脸", "file": "f.jpg", "prompt": "p1"},
    {"label": "身", "file": "b.jpg"},
]}
OLD = {"name": "B", "prompt": "R2", "face": "f2.jpg", "half": "", "full": "u.jpg"}
check("新形状解析", pure.image_entries(NEW) == [
    {"label": "脸", "file": "f.jpg", "prompt": "p1"},
    {"label": "身", "file": "b.jpg", "prompt": ""}])
check("旧形状映射", pure.image_entries(OLD) == [
    {"label": "脸部特写", "file": "f2.jpg", "prompt": "R2"},
    {"label": "全身像", "file": "u.jpg", "prompt": "R2"}])
check("旧数组态迁移首图", pure.coerce_selection([NEW], '["A"]') == {"role": "A", "shots": ["脸"]})
check("多选交集保序", pure.coerce_selection([NEW], '{"role": "A", "shots": ["身", "脸", "无"]}') == {"role": "A", "shots": ["脸", "身"]})
check("空态回落首角首图", pure.coerce_selection([NEW], "[]") == {"role": "A", "shots": ["脸"]})
check("失效角色回落首图", pure.coerce_selection([NEW], '{"role": "X", "shots": ["身"]}') == {"role": "A", "shots": ["脸"]})
check("空库回落空", pure.coerce_selection([], "[]") == {"role": "", "shots": []})
check("拼接回落角色词", pure.join_prompts(NEW, ["脸", "身"]) == "p1, R")
check("草稿整体覆盖", pure.resolve_prompt(NEW, ["脸"], "hand") == "hand")
check("角色级 prompt", pure.role_prompt(NEW) == "R" and pure.role_prompt(None) == "")

# ── 双源目录与加载 ──
_orig_user, _orig_builtin = mod._user_characters_dir, mod._builtin_characters_dir
try:
    with tempfile.TemporaryDirectory() as user_dir, tempfile.TemporaryDirectory() as builtin_dir:
        mod._user_characters_dir = lambda: user_dir
        mod._builtin_characters_dir = lambda: builtin_dir
        with open(os.path.join(builtin_dir, "character_a.json"), "w", encoding="utf-8") as f:
            json.dump([{"name": "Builtin", "prompt": "b"}], f)
        with open(os.path.join(user_dir, "character_a.json"), "w", encoding="utf-8") as f:
            json.dump([{"name": "User", "prompt": "u"}], f)
        check("同名库用户覆盖内置", mod._load_characters("character_a") == [{"name": "User", "prompt": "u"}])
        check("未知库空列表", mod._load_characters("nope") == [])
finally:
    mod._user_characters_dir, mod._builtin_characters_dir = _orig_user, _orig_builtin

# ── execute（张量/尺寸函数打桩；形状可断言）──
_FACE, _HALF = _FakeTensor((1, 64, 48, 3), "face"), _FakeTensor((1, 128, 96, 3), "half")
_orig_load, _orig_ph = mod._load_template_tensor, mod._placeholder_tensor
mod._load_template_tensor = lambda path: _FACE if "face" in path else _HALF
mod._placeholder_tensor = lambda: _FakeTensor((1, 1, 1, 3), "ph")
rescaled = []


def _fake_rescale(t, w, h):
    rescaled.append((t.tag, w, h))
    return _FakeTensor((1, h, w, 3), t.tag + ">")


# rescale_image 经 sys.modules 预置桩（character._batch_tensors 内惰性 import 落到此桩，
# 真实 image_convert 拉 torch，本机无——此处不断言桩模块本身，只断言调用参数与形状）
sys.modules["sfnodes.sf_utils.image_convert"] = types.SimpleNamespace(rescale_image=_fake_rescale)
_orig_cls_load, _orig_cls_dirs = mod._load_characters, mod._characters_dirs
try:
    with tempfile.TemporaryDirectory() as user_dir:
        samples = os.path.join(user_dir, "samples_id_chara", "character_b")
        os.makedirs(samples)
        for fn in ("face_01.jpg", "half_01.jpg"):
            with open(os.path.join(samples, fn), "wb") as f:
                f.write(b"jpgdata")
        mod._characters_dirs = lambda: [user_dir]
        mod._load_characters = lambda name: [
            {"name": "主角", "prompt": "hero",
             "images": [{"label": "脸", "file": "samples_id_chara/character_b/face_01.jpg", "prompt": "p-face"},
                        {"label": "身", "file": "samples_id_chara/character_b/half_01.jpg"}]},
        ] if name == "character_b" else []

        st = '{"role": "主角", "shots": ["脸", "身"]}'
        r = node.execute(library="character_b", SFCharacterState=st, SFCharacterPrompt="")
        check("拼接+角色词", r[0] == "p-face, hero" and r[1] == "hero")
        check("batch 形状首图归一", r[2].shape == (2, 64, 48, 3) and r[2].tag == "cat")
        check("归一调尺寸到首图", rescaled == [("half", 48, 64)])
        r = node.execute(library="character_b", SFCharacterState=st, SFCharacterPrompt="hand")
        check("草稿覆盖拼接路", r[0] == "hand" and r[1] == "hero")
        n_rescaled = len(rescaled)
        r = node.execute(library="character_b",
                         SFCharacterState='{"role": "主角", "shots": ["脸"]}', SFCharacterPrompt="")
        check("单选单张", r[0] == "p-face" and r[2].shape == (1, 64, 48, 3))
        check("单张同尺寸免归一", len(rescaled) == n_rescaled)
        r = node.execute(library="character_b", SFCharacterState='{"role": "", "shots": []}', SFCharacterPrompt="")
        check("显式空态回落首图", r[0] == "p-face" and r[2].shape == (1, 64, 48, 3))
        r = node.execute(library="character_b", SFCharacterState='["主角"]', SFCharacterPrompt="")
        check("旧数组态迁移首图", r[0] == "p-face" and r[2].shape == (1, 64, 48, 3))
        r = node.execute(library="no_such_library", SFCharacterState=st, SFCharacterPrompt="")
        check("未知库降级", r[0] == "" and r[1] == "" and r[2].shape == (1, 1, 1, 3))
finally:
    mod._load_template_tensor, mod._placeholder_tensor = _orig_load, _orig_ph
    mod._load_characters, mod._characters_dirs = _orig_cls_load, _orig_cls_dirs

# ── IS_CHANGED / VALIDATE_INPUTS ──
check("VALIDATE_INPUTS 恒 True", node.VALIDATE_INPUTS(library="stale_value") is True)
check("IS_CHANGED 空库返回 0", node.IS_CHANGED(library="") == 0)
check("IS_CHANGED 未知库返回 0", node.IS_CHANGED(library="nope") == 0)

# ── 归一化（新形状直通 + 旧形状映射）──
norm = mod.normalize_role_list([
    {"name": "A", "prompt": "p",
     "images": [{"label": "脸", "file": "samples_id_chara/x/face.jpg", "prompt": "pf"},
                {"label": "坏", "file": ""}]},
    {"name": "B", "prompt": "r2", "face": "samples_id_chara/x/f.jpg", "full": "https://h/u.jpg"},
    {"name": "C"},
    {"name": 123},
])
check("归一化 images 直通", norm[0]["images"] == [
    {"label": "脸", "url": "/api/sfnodes/characters/image?path=samples_id_chara/x/face.jpg", "prompt": "pf"}])
check("归一化旧形状映射", norm[1]["images"] == [
    {"label": "脸部特写", "url": "/api/sfnodes/characters/image?path=samples_id_chara/x/f.jpg", "prompt": "r2"},
    {"label": "全身像", "url": "https://h/u.jpg", "prompt": "r2"}])
check("归一化空图 images 空数组", norm[2]["images"] == [])
check("归一化非 name 条目跳过", len(norm) == 3)

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
        json.dump([{"name": "R", "prompt": "p",
                    "images": [{"label": "脸", "file": "samples_id_chara/character_x/face.jpg"}]}], f)
    os.makedirs(os.path.join(user_dir, "samples_id_chara", "character_x"))
    with open(os.path.join(user_dir, "samples_id_chara", "character_x", "face.jpg"), "wb") as f:
        f.write(b"jpgdata")

    resp = asyncio.run(_run(handlers["/api/sfnodes/characters"], _FakeRequest({"name": "character_x"})))
    check("列表路由返回条目", len(resp) == 1 and resp[0]["images"][0]["label"] == "脸")
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
