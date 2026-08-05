import importlib.util
import os
import sys
import types

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

comfy = types.ModuleType("comfy")
node_typing = types.ModuleType("comfy.comfy_types")
node_typing_module = types.ModuleType("comfy.comfy_types.node_typing")
class IO:
    STRING = "STRING"
    INT = "INT"
    FLOAT = "FLOAT"
node_typing_module.IO = IO
comfy.comfy_types = node_typing
comfy.comfy_types.node_typing = node_typing_module
sys.modules["comfy"] = comfy
sys.modules["comfy.comfy_types"] = node_typing
sys.modules["comfy.comfy_types.node_typing"] = node_typing_module

# mock aiohttp.web + server.PromptServer（路由注册验证用）
aiohttp = types.ModuleType("aiohttp")
web_mod = types.ModuleType("aiohttp.web")
class _FakeJsonResponse:
    def __init__(self, data):
        self.data = data
web_mod.json_response = lambda data, status=200: _FakeJsonResponse(data)
web_mod.Response = lambda status=200, text="": type("R", (), {"status": status})()
aiohttp.web = web_mod
sys.modules["aiohttp"] = aiohttp
sys.modules["aiohttp.web"] = web_mod

class _FakeRoutes:
    def __init__(self):
        self.handlers = {}
    def get(self, path):
        def deco(fn):
            self.handlers[path] = fn
            return fn
        return deco

server_mod = types.ModuleType("server")
server_mod.PromptServer = type("PS", (), {"instance": type("I", (), {"routes": _FakeRoutes()})()})
sys.modules["server"] = server_mod

pkg = types.ModuleType("sfnodes")
pkg.__path__ = [root]
nodes_pkg = types.ModuleType("sfnodes.nodes")
nodes_pkg.__path__ = [os.path.join(root, "nodes")]
text_pkg = types.ModuleType("sfnodes.nodes.text")
text_pkg.__path__ = [os.path.join(root, "nodes", "text")]
sys.modules["sfnodes"] = pkg
sys.modules["sfnodes.nodes"] = nodes_pkg
sys.modules["sfnodes.nodes.text"] = text_pkg

spec = importlib.util.spec_from_file_location(
    "sfnodes.nodes.text.prompt_preset",
    os.path.join(root, "nodes", "text", "prompt_preset.py"),
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

# 1. combo options are Chinese
it = mod.SFPromptPreset.INPUT_TYPES()
opt = it["optional"]
check("输入顺序 outfit/pose/couple", list(opt)[:3] == ["outfit_preset", "pose_preset", "couple_preset"])
check("角度/距离/镜头顺序", list(opt)[6:9] == ["camera_angle_preset", "camera_distance_preset", "camera_lens_preset"])
for key in ("outfit_preset", "pose_preset", "couple_preset", "environment_preset", "lighting_preset", "style_preset", "camera_angle_preset", "camera_distance_preset", "camera_lens_preset"):
    opts = opt[key][0]
    check(f"{key} 首项为 禁用", opts[0] == "禁用")
    check(f"{key} 次项为 随机", opts[1] == "随机")
    check(f"{key} 无英文 Disabled/Random", "Disabled" not in opts and "Random" not in opts)
    pure_ascii = [o for o in opts[2:] if o.isascii()]
    check(f"{key} 无纯英文选项", len(pure_ascii) == 0)
    check(f"{key} 默认 禁用", opt[key][1]["default"] == "禁用")
check("outfit 40 项", len(opt["outfit_preset"][0]) == 42)
check("pose 50 项", len(opt["pose_preset"][0]) == 52)
check("couple 32 项", len(opt["couple_preset"][0]) == 34)
check("environment 84 项", len(opt["environment_preset"][0]) == 86)
check("lighting 62 项", len(opt["lighting_preset"][0]) == 64)
check("style 48 项", len(opt["style_preset"][0]) == 50)
check("angle 22 项", len(opt["camera_angle_preset"][0]) == 24)
check("distance 11 项", len(opt["camera_distance_preset"][0]) == 13)
check("camera 43 项", len(opt["camera_lens_preset"][0]) == 45)

# category keys match JSON data keys
import json as _json
_data = _json.load(open(os.path.join(root, "data", "prompt_presets.json"), encoding="utf-8"))
check("分类键与 JSON 一致", list(_data) == list(mod._CATEGORY_KEYS))
check("分类键无 Apex 前缀", all(not k.startswith("Apex") for k in mod._CATEGORY_KEYS))

node = mod.SFPromptPreset()

# 2. specific zh selection -> english prompt (non-empty, no Chinese in output)
combined, outfit, pose, couple, env, light, style, angle, dist, cam = node.execute(
    "test subject", seed=42,
    outfit_preset="旗袍",
    pose_preset="禁用",
    couple_preset="传教士体位",
    environment_preset="现代地铁车厢",
    lighting_preset="自然窗光",
    style_preset="写实摄影风",
    camera_angle_preset="仰拍",
    camera_distance_preset="特写",
    camera_lens_preset="85mm经典人像",
)
check("双人中文反查命中", "missionary" in couple)
check("互斥时 pose 输出为空", pose == "")
check("服装中文反查命中", "qipao" in outfit)

_, _, pose_only, _, env_only, _, _, _, _, _ = node.execute(
    "test subject", seed=42, pose_preset="回眸", environment_preset="现代地铁车厢")
check("姿势中文反查命中", "looking back over the shoulder" in pose_only)
check("角度中文反查命中", "low angle" in angle.lower())
check("距离中文反查命中", "close-up" in dist.lower())
check("环境中文反查命中", "subway train" in env)
check("灯光中文反查命中", "Natural daylight" in light)
check("风格中文反查命中", "Photorealistic" in style)
check("镜头中文反查命中", "85mm" in cam)
check("输出无中文", not any(any('\u4e00' <= c <= '\u9fff' for c in s) for s in (combined, pose, couple, env, light, style, angle, dist, cam)))
_segs = [s.strip().rstrip(".") for s in (outfit, couple, env, light, style, angle, dist, cam)]
check("combined 含全部部分", all(s in combined for s in _segs))
check("拼接顺序 outfit < couple < env", combined.index(_segs[0]) < combined.index(_segs[1]) < combined.index(_segs[2]))
check("拼接顺序 angle < dist < lens", combined.index(_segs[5]) < combined.index(_segs[6]) < combined.index(_segs[7]))
comb_only, _, _, _, env_only_text, _, _, _, _, _ = node.execute("test subject", seed=42, pose_preset="回眸", couple_preset="禁用", environment_preset="现代地铁车厢")
check("拼接顺序 pose < env", comb_only.index(pose_only.strip().rstrip(".")) < comb_only.index(env_only_text.strip().rstrip(".")))
_, _, _, _, _, _, _, _, _, _ = node.execute("test subject", seed=42, pose_preset="回眸", couple_preset="禁用", environment_preset="现代地铁车厢")
comb_op, outfit_op, pose_op, _, _, _, _, _, _, _ = node.execute("test subject", seed=42, outfit_preset="旗袍", pose_preset="回眸")
check("拼接顺序 outfit < pose", comb_op.index(outfit_op.strip().rstrip(".")) < comb_op.index(pose_op.strip().rstrip(".")))

# 2b. NSFW pose resolution
_, _, pose_nsfw, _, _, _, _, _, _, _ = node.execute("x", seed=1, pose_preset="床上自慰")
check("NSFW 姿势命中", "masturbat" in pose_nsfw and "nude" in pose_nsfw)
_, _, _, couple_nsfw, _, _, _, _, _, _ = node.execute("x", seed=1, couple_preset="女女交叉体位")
check("NSFW 女女命中", "scissor" in couple_nsfw and "lesbian" in couple_nsfw)

# 3. english values are ignored
_, _, _, _, env2, _, _, _, _, _ = node.execute("x", seed=1, environment_preset="Ocean Sunrise")
check("英文预设名忽略", env2 == "")
_, _, _, _, env3, _, _, _, _, _ = node.execute("x", seed=1, environment_preset="Disabled")
check("英文 Disabled 忽略", env3 == "")
_, _, _, _, env4, _, _, _, _, _ = node.execute("x", seed=1, environment_preset="Random")
check("英文 Random 忽略", env4 == "")
_, _, _, _, env5, _, _, _, _, _ = node.execute("x", seed=1, environment_preset="None")
check("英文 None 忽略", env5 == "")

# 3b. pose english name ignored
_, _, pose_en, _, _, _, _, _, _, _ = node.execute("x", seed=1, pose_preset="Running")
check("姿势英文名忽略", pose_en == "")

# 4. deterministic weighted random
r1 = node.execute("", seed=100, outfit_preset="随机", pose_preset="随机", couple_preset="禁用", environment_preset="随机", lighting_preset="随机", style_preset="随机", camera_angle_preset="随机", camera_distance_preset="随机", camera_lens_preset="随机")
r2 = node.execute("", seed=100, outfit_preset="随机", pose_preset="随机", couple_preset="禁用", environment_preset="随机", lighting_preset="随机", style_preset="随机", camera_angle_preset="随机", camera_distance_preset="随机", camera_lens_preset="随机")
check("随机预设 seed 确定性", r1 == r2)
r3 = node.execute("", seed=100, outfit_preset="随机", pose_preset="随机", couple_preset="禁用", environment_preset="随机", lighting_preset="随机", style_preset="随机", camera_angle_preset="随机", camera_distance_preset="随机", camera_lens_preset="随机")
check("随机输出非空", all(r3[:3] + r3[4:]))
r4 = node.execute("", seed=200, outfit_preset="随机", pose_preset="随机", couple_preset="禁用", environment_preset="随机", lighting_preset="随机", style_preset="随机", camera_angle_preset="随机", camera_distance_preset="随机", camera_lens_preset="随机")
check("不同 seed 可能不同", r1 != r4 or r3 != r4)

# seed offsets: outfit +1, pose +2, couple +3, environment +4, angle +7, distance +8, lens +9
o_off = node._resolve_preset("Outfit", "随机", 100 + 1)
p_off = node._resolve_preset("Pose", "随机", 100 + 2)
c_off = node._resolve_preset("Couple Pose", "随机", 100 + 3)
e_off = node._resolve_preset("Environment", "随机", 100 + 4)
a_off = node._resolve_preset("Camera Angle", "随机", 100 + 7)
d_off = node._resolve_preset("Camera Distance", "随机", 100 + 8)
l_off = node._resolve_preset("Camera Lens", "随机", 100 + 9)
check("服装随机偏移 seed+1", r1[1] == o_off)
check("姿势随机偏移 seed+2", r1[2] == p_off)
check("环境随机偏移 seed+4", r1[4] == e_off)
check("角度随机偏移 seed+7", r1[7] == a_off)
check("距离随机偏移 seed+8", r1[8] == d_off)
check("镜头随机偏移 seed+9", r1[9] == l_off)
rc = node.execute("", seed=100, outfit_preset="禁用", pose_preset="禁用", couple_preset="随机")
check("双人随机偏移 seed+3", rc[3] == c_off)

# 3c. enriched presets resolve correctly
_, _, _, _, env_new, _, _, _, _, _ = node.execute("x", seed=1, environment_preset="薰衣草田日落")
check("新增环境反查", "lavender" in env_new)
_, _, _, _, env_life, _, _, _, _, _ = node.execute("x", seed=1, environment_preset="樱花大道")
check("樱花大道反查", "cherry blossom" in env_life.lower())
_, _, _, _, _, light_life, _, _, _, _ = node.execute("x", seed=1, lighting_preset="日系柔和窗光")
check("日系窗光反查", "japanese" in light_life.lower() and "window light" in light_life.lower())
_, _, pose_new, _, _, _, _, _, _, _ = node.execute("x", seed=1, pose_preset="仰卧举腿")
check("新增姿势 NSFW 反查", "legs raised" in pose_new and "nude" in pose_new)
_, _, _, couple_new, _, _, _, _, _, _ = node.execute("x", seed=1, couple_preset="反向女上位")
check("新增双人 NSFW 反查", "reverse cowgirl" in couple_new)
_, _, _, _, _, _, _, angle_new, _, _ = node.execute("x", seed=1, camera_angle_preset="荷兰角")
check("角度反查命中", "dutch" in angle_new.lower())
_, _, _, _, _, _, _, angle_new2, _, _ = node.execute("x", seed=1, camera_angle_preset="过肩镜头")
check("过肩反查命中", "over-the-shoulder" in angle_new2.lower())
_, _, _, _, _, _, _, _, dist_new, _ = node.execute("x", seed=1, camera_distance_preset="大远景")
check("大远景反查命中", "extreme long shot" in dist_new.lower())
_, _, _, _, _, _, _, _, dist_new2, _ = node.execute("x", seed=1, camera_distance_preset="牛仔镜头")
check("牛仔镜头反查命中", "cowboy" in dist_new2.lower())

# 5. bracket random
br1 = node.execute("woman in a [flower field, alien landscape, new york street]", seed=7)
br2 = node.execute("woman in a [flower field, alien landscape, new york street]", seed=7)
check("括号随机确定性", br1[0] == br2[0])
check("括号已展开", "[" not in br1[0])
check("括号选项其中之一", any(x in br1[0] for x in ("flower field", "alien landscape", "new york street")))
br3 = node.execute("no brackets here", seed=7)
check("无括号原样", br3[0] == "no brackets here")

# 6. weight distribution sanity (some seed hits various presets)
seen = set()
for s in range(300):
    r = node.execute("", seed=s, camera_lens_preset="随机")
    seen.add(r[9])
check(f"随机多样性 ({len(seen)}/43)", len(seen) > 20)
seen_angle = set()
for s in range(200):
    r = node.execute("", seed=s, camera_angle_preset="随机")
    seen_angle.add(r[7])
check(f"角度随机多样性 ({len(seen_angle)}/22)", len(seen_angle) > 10)
seen_dist = set()
for s in range(150):
    r = node.execute("", seed=s, camera_distance_preset="随机")
    seen_dist.add(r[8])
check(f"距离随机多样性 ({len(seen_dist)}/11)", len(seen_dist) > 5)
seen_outfit = set()
for s in range(200):
    r = node.execute("", seed=s, outfit_preset="随机")
    seen_outfit.add(r[1])
check(f"服装随机多样性 ({len(seen_outfit)}/40)", len(seen_outfit) > 15)
seen_pose = set()
for s in range(200):
    r = node.execute("", seed=s, pose_preset="随机")
    seen_pose.add(r[2])
check(f"姿势随机多样性 ({len(seen_pose)}/50)", len(seen_pose) > 20)
seen_couple = set()
for s in range(200):
    r = node.execute("", seed=s, couple_preset="随机")
    seen_couple.add(r[3])
check(f"双人随机多样性 ({len(seen_couple)}/32)", len(seen_couple) > 15)

# 7. IS_CHANGED
check("IS_CHANGED 返回 seed", mod.SFPromptPreset.IS_CHANGED(seed=123) == 123)

# 热加载：数据文件 mtime 变化后自动重载
_path = mod._presets_path()
_old_mt = os.path.getmtime(_path)
_first = mod._load_presets()[0]
os.utime(_path, (_old_mt + 10, _old_mt + 10))
_second = mod._load_presets()[0]
check("热加载 mtime 变化后重读", _second is not _first)
os.utime(_path, (_old_mt, _old_mt))

# 描述映射 API 路由
import asyncio
_handlers = sys.modules["server"].PromptServer.instance.routes.handlers
_route = _handlers.get("/api/sfnodes/prompt_presets")
check("路由已注册", _route is not None)
_resp = asyncio.run(_route(None))
check("路由返回分类数据", "Pose" in _resp.data and "Environment" in _resp.data)
check("路由返回中文名->描述", _resp.data["Environment"]["现代地铁车厢"].startswith("Contemporary subway"))
check("路由描述为英文", all(
    all(not any('\u4e00' <= c <= '\u9fff' for c in d) for d in cat.values())
    for cat in _resp.data.values()))

# 7b. lens presets free of shot-size wording (orthogonal to Camera Distance)
import re as _re
_lens_pat = _re.compile(r"close-?up|headshot|tight framing|filling (the )?frame|waist-up|chest up|subject from waist|upper torso", _re.I)
_bad = [n for n, v in _data["Camera Lens"].items() if _lens_pat.search(v["prompt"])]
check(f"镜头预设无景别词 ({len(_bad)} 处残留)", len(_bad) == 0)
_cat_pats = [_re.compile(r"close-?up|headshot|tight framing|waist-up|chest up", _re.I) for _ in []]
for _cat in ("Camera Angle", "Camera Distance"):
    _pats = _re.compile(r"close-?up|headshot|tight framing|filling (the )?frame|waist-up|chest up|subject from waist|upper torso", _re.I)
    _bad2 = [n for n, v in _data[_cat].items() if _pats.search(v["prompt"])]
    check(f"{_cat} 保留景别描述（{len(_bad2)}）", len(_bad2) >= 0)
_bad3 = [n for n, v in _data["Camera Distance"].items() if "close-up" not in v["prompt"].lower() and "shot" not in v["prompt"].lower()]
check("Camera Distance 景别完整", len(_bad3) == 0)

# 7c. combined prompt has no "., " stitching artifacts
_, _, _, _, _, _, _, _, _, _ = node.execute("", seed=1, camera_distance_preset="特写", camera_lens_preset="85mm经典人像")
_c1, *_ = node.execute("test subject.", seed=1, camera_distance_preset="特写", camera_lens_preset="85mm经典人像")
check("拼接无 . , 粘连", "., " not in _c1)
_c2, *_ = node.execute("test subject", seed=1, camera_distance_preset="特写", camera_lens_preset="85mm经典人像")
check("拼接无 . , 粘连(无输入句号)", "., " not in _c2)

# 7d. pose/couple mutual exclusion (backend fallback, pose wins)
_, _, p_m, c_m, _, _, _, _, _, _ = node.execute("x", seed=1, pose_preset="回眸", couple_preset="公主抱")
check("互斥兜底 pose 生效", "looking back over the shoulder" in p_m)
check("互斥兜底 couple 忽略", c_m == "")
_, _, p_m2, c_m2, _, _, _, _, _, _ = node.execute("x", seed=1, pose_preset="禁用", couple_preset="公主抱")
check("仅 couple 时生效", c_m2 != "" and "bridal carry" in c_m2 and p_m2 == "")
_, _, p_m3, _, _, _, _, _, _, _ = node.execute("x", seed=1, pose_preset="随机", couple_preset="公主抱")
check("互斥 pose=随机 生效", p_m3 != "")

# 服装 NSFW 反查
_, outfit_nsfw, _, _, _, _, _, _, _, _ = node.execute("x", seed=1, outfit_preset="全透明连衣裙")
check("服装 NSFW 命中", "transparent" in outfit_nsfw and "adult" in outfit_nsfw.lower() or "see-through" in outfit_nsfw)

# 8. registration keys
sys.modules["comfy.comfy_types.node_typing"].IO = IO
import ast
tree = ast.parse(open(os.path.join(root, "__init__.py"), encoding="utf-8").read())
def dict_of(node):
    result = {}
    for k, v in zip(node.keys, node.values):
        result[k.value if isinstance(k, ast.Constant) else ast.unparse(k)] = ast.unparse(v)
    return result
class_mappings = display_mappings = None
for stmt in tree.body:
    if isinstance(stmt, ast.Assign):
        for t in stmt.targets:
            if isinstance(t, ast.Name) and t.id == "NODE_CLASS_MAPPINGS":
                class_mappings = dict_of(stmt.value)
            elif isinstance(t, ast.Name) and t.id == "NODE_DISPLAY_NAME_MAPPINGS":
                display_mappings = dict_of(stmt.value)
check("NODE_CLASS_MAPPINGS 含 SFPromptPreset", "SFPromptPreset" in class_mappings)
check("NODE_DISPLAY_NAME_MAPPINGS 含 SFPromptPreset", "SFPromptPreset" in display_mappings)
check("两字典键一致", set(class_mappings) == set(display_mappings))

print()
print("FAILURES:", len(failures))
sys.exit(1 if failures else 0)
