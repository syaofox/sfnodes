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
check("输入顺序 celebrity/expression/outfit", list(opt)[:3] == ["celebrity_preset", "expression_preset", "outfit_preset"])
check("角度/距离/镜头顺序", list(opt)[8:11] == ["camera_angle_preset", "camera_distance_preset", "camera_lens_preset"])
_celeb_opts = opt["celebrity_preset"][0]
check("celebrity 首项为 禁用", _celeb_opts[0] == "禁用")
check("celebrity 次项为 随机", _celeb_opts[1] == "随机")
check("celebrity 无英文 Disabled/Random", "Disabled" not in _celeb_opts and "Random" not in _celeb_opts)
check("celebrity 含英文名选项", any(o.isascii() and o not in ("禁用", "随机") for o in _celeb_opts))
check("celebrity 实测新增反查", "Tom Hanks" in _celeb_opts and "Denzel Washington" in _celeb_opts)
check("celebrity 含中文名选项", any(not o.isascii() for o in _celeb_opts))
check("celebrity 默认 禁用", opt["celebrity_preset"][1]["default"] == "禁用")
for key in ("outfit_preset", "pose_preset", "couple_preset", "environment_preset", "lighting_preset", "style_preset", "camera_angle_preset", "camera_distance_preset", "camera_lens_preset"):
    opts = opt[key][0]
    check(f"{key} 首项为 禁用", opts[0] == "禁用")
    check(f"{key} 次项为 随机", opts[1] == "随机")
    check(f"{key} 无英文 Disabled/Random", "Disabled" not in opts and "Random" not in opts)
    pure_ascii = [o for o in opts[2:] if o.isascii()]
    check(f"{key} 无纯英文选项", len(pure_ascii) == 0)
    check(f"{key} 默认 禁用", opt[key][1]["default"] == "禁用")
check("celebrity 429 项(含10组随机)", len(opt["celebrity_preset"][0]) == 441)
check("expression 38 项(含6组随机)", len(opt["expression_preset"][0]) == 46)
check("outfit 71 项(含2组随机)", len(opt["outfit_preset"][0]) == 75)
check("pose 76 项(含2组随机)", len(opt["pose_preset"][0]) == 80)
check("couple 47 项(含3组随机)", len(opt["couple_preset"][0]) == 52)
check("environment 112 项(含8组随机)", len(opt["environment_preset"][0]) == 122)
check("lighting 62 项(含7组随机)", len(opt["lighting_preset"][0]) == 71)
check("style 49 项(含2组随机)", len(opt["style_preset"][0]) == 53)
check("angle 26 项(含5组随机)", len(opt["camera_angle_preset"][0]) == 33)
check("distance 11 项(含4组随机)", len(opt["camera_distance_preset"][0]) == 17)
check("camera 43 项(含11组随机)", len(opt["camera_lens_preset"][0]) == 56)

# category keys match JSON data keys
import json as _json
_data = _json.load(open(os.path.join(root, "data", "prompt_presets.json"), encoding="utf-8"))
check("分类键与 JSON 一致", list(_data) == list(mod._CATEGORY_KEYS))
check("分类键无 Apex 前缀", all(not k.startswith("Apex") for k in mod._CATEGORY_KEYS))

# 裸体词正交：Pose/Couple Pose NSFW 动作不内嵌裸体（裸体由 Outfit 分类"全裸"预设控制）
import re as _re_nude
_nude_pat = _re_nude.compile(r"\bnude\b|\bnaked\b|bare skin|unclothed|fully exposed", _re_nude.I)
_pose_nude = [n for n, v in _data["Pose"].items() if _nude_pat.search(v["prompt"]) or _nude_pat.search(v.get("description", ""))]
check(f"Pose 无裸体词 ({_pose_nude})", len(_pose_nude) == 0)
_couple_nude = [n for n, v in _data["Couple Pose"].items() if _nude_pat.search(v["prompt"])]
check(f"Couple 无裸体词 ({_couple_nude})", len(_couple_nude) == 0)
check("Outfit 全裸保留裸体词（职责归属）", "fully nude" in _data["Outfit"]["Fully Nude"]["prompt"])
check("全裸站立已删除（与站立肖像重复）", "Fully Nude Standing" not in _data["Pose"]
      and "全裸站立" not in opt["pose_preset"][0])

# 灯光词正交：Pose/Couple Pose 动作不内嵌灯光（光由 Lighting 分类控制）
_light_pat = _re_nude.compile(r"\blighting\b|\bglow\b|illuminat|shimmer|natural light|soft light|warm light|window light", _re_nude.I)
_pose_light = [n for n, v in _data["Pose"].items() if _light_pat.search(v["prompt"])]
_couple_light = [n for n, v in _data["Couple Pose"].items() if _light_pat.search(v["prompt"])]
check(f"Pose 无灯光词 ({_pose_light})", len(_pose_light) == 0)
check(f"Couple 无灯光词 ({_couple_light})", len(_couple_light) == 0)

node = mod.SFPromptPreset()

# 2. specific zh selection -> english prompt (non-empty, no Chinese in output)
combined, pack = node.execute(
    "test subject", seed=42,
    celebrity_preset="周杰伦",
    expression_preset="禁用",
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
celebrity = pack["Celebrity"]
outfit = pack["Outfit"]
pose = pack["Pose"]
couple = pack["Couple Pose"]
env = pack["Environment"]
light = pack["Lighting"]
style = pack["Style"]
angle = pack["Camera Angle"]
dist = pack["Camera Distance"]
cam = pack["Camera Lens"]
check("双人中文反查命中", "missionary" in couple)
check("互斥时 pose 输出为空", pose == "")
check("服装中文反查命中", "qipao" in outfit)
check("名人中文反查命中", "jay chou" in celebrity.lower() and "taiwanese" in celebrity.lower())
check("名人片段保持专名大小写", "Jay Chou" in combined)

_, pack = node.execute(
    "test subject", seed=42, pose_preset="回眸", environment_preset="现代地铁车厢")
pose_only = pack["Pose"]
env_only = pack["Environment"]
check("姿势中文反查命中", "looking back over the shoulder" in pose_only)
check("角度中文反查命中", "low-angle" in angle.lower())
check("距离中文反查命中", "close-up" in dist.lower())
check("环境中文反查命中", "subway train" in env)
check("灯光中文反查命中", "Natural daylight" in light)
check("风格中文反查命中", "Photorealistic" in style)
check("镜头中文反查命中", "85mm" in cam)
check("输出无中文", not any(any('\u4e00' <= c <= '\u9fff' for c in s) for s in (combined, pose, couple, env, light, style, angle, dist, cam)))
def _seg(s):
    s = s.strip().rstrip(".")
    return s[0].lower() + s[1:] if s else s

_segs = [s.strip().rstrip(".") if s is celebrity else _seg(s) for s in (celebrity, outfit, couple, env, light, style, angle, dist, cam)]
check("combined 含全部部分", all(s in combined for s in _segs))
check("拼接顺序 celebrity < outfit < couple", combined.index(_segs[0]) < combined.index(_segs[1]) < combined.index(_segs[2]))
check("拼接顺序 angle < dist < lens", combined.index(_segs[6]) < combined.index(_segs[7]) < combined.index(_segs[8]))
comb_only, pack = node.execute("test subject", seed=42, pose_preset="回眸", couple_preset="禁用", environment_preset="现代地铁车厢")
env_only_text = pack['Environment']
check("拼接顺序 pose < env", comb_only.index(_seg(pose_only)) < comb_only.index(_seg(env_only_text)))
_, pack = node.execute("test subject", seed=42, pose_preset="回眸", couple_preset="禁用", environment_preset="现代地铁车厢")
comb_op, pack = node.execute("test subject", seed=42, outfit_preset="旗袍", pose_preset="回眸")
outfit_op = pack['Outfit']
pose_op = pack['Pose']
check("拼接顺序 outfit < pose", comb_op.index(_seg(outfit_op)) < comb_op.index(_seg(pose_op)))

# 2b. NSFW pose resolution
_, pack = node.execute("x", seed=1, pose_preset="床上自慰")
pose_nsfw = pack['Pose']
check("NSFW 姿势命中", "masturbat" in pose_nsfw)
_, pack = node.execute("x", seed=1, couple_preset="女女交叉体位")
couple_nsfw = pack['Couple Pose']
check("NSFW 女女命中", "scissor" in couple_nsfw and "lesbian" in couple_nsfw)

# 3. english values are ignored
_, pack = node.execute("x", seed=1, environment_preset="Ocean Sunrise")
env2 = pack['Environment']
check("英文预设名忽略", env2 == "")
_, pack = node.execute("x", seed=1, environment_preset="Disabled")
env3 = pack['Environment']
check("英文 Disabled 忽略", env3 == "")
_, pack = node.execute("x", seed=1, environment_preset="Random")
env4 = pack['Environment']
check("英文 Random 忽略", env4 == "")
_, pack = node.execute("x", seed=1, environment_preset="None")
env5 = pack['Environment']
check("英文 None 忽略", env5 == "")

# 3b. pose english name ignored
_, pack = node.execute("x", seed=1, pose_preset="Running")
pose_en = pack['Pose']
check("姿势英文名忽略", pose_en == "")

# 4. deterministic weighted random
r1 = node.execute("", seed=100, celebrity_preset="随机", expression_preset="随机", outfit_preset="随机", pose_preset="随机", couple_preset="禁用", environment_preset="随机", lighting_preset="随机", style_preset="随机", camera_angle_preset="随机", camera_distance_preset="随机", camera_lens_preset="随机")
r2 = node.execute("", seed=100, celebrity_preset="随机", expression_preset="随机", outfit_preset="随机", pose_preset="随机", couple_preset="禁用", environment_preset="随机", lighting_preset="随机", style_preset="随机", camera_angle_preset="随机", camera_distance_preset="随机", camera_lens_preset="随机")
check("随机预设 seed 确定性", r1 == r2)
r3 = node.execute("", seed=100, celebrity_preset="随机", expression_preset="随机", outfit_preset="随机", pose_preset="随机", couple_preset="禁用", environment_preset="随机", lighting_preset="随机", style_preset="随机", camera_angle_preset="随机", camera_distance_preset="随机", camera_lens_preset="随机")
check("随机输出非空", all(v for k, v in r3[1].items() if k != "Couple Pose"))
r4 = node.execute("", seed=200, celebrity_preset="随机", expression_preset="随机", outfit_preset="随机", pose_preset="随机", couple_preset="禁用", environment_preset="随机", lighting_preset="随机", style_preset="随机", camera_angle_preset="随机", camera_distance_preset="随机", camera_lens_preset="随机")
check("不同 seed 可能不同", r1 != r4 or r3 != r4)

# 组随机：随机·组名 限定在指定 group 内（seed 确定性）
_g1 = node.execute("", seed=55, style_preset="随机·写实")
_g2 = node.execute("", seed=55, style_preset="随机·写实")
check("组随机 seed 确定性", _g1[1]['Style'] == _g2[1]['Style'])
check("组随机命中写实组", _g1[1]['Style'] in {v["prompt"] for v in _data["Style"].values() if v["group"] == "写实"})
_g3 = node.execute("", seed=55, pose_preset="随机·NSFW")
_nsfw_pose = {v["prompt"] for v in _data["Pose"].values() if v["group"] == "NSFW"}
check("组随机不越界(NSFW)", _g3[1]['Pose'] in _nsfw_pose)
_g4 = node.execute("", seed=55, style_preset="随机·非写实")
_nr_style = {v["prompt"] for v in _data["Style"].values() if v["group"] == "非写实"}
check("非写实随机不越界", _g4[1]['Style'] in _nr_style)
check("北欧极简归写实", _data["Style"]["Nordic Minimalism"]["group"] == "写实")

# 分类正交：非 Style 分类无写实风格强制词（风格由 Style 分类控制）
import re as _re_style
_real_pat = _re_style.compile(r"photorealistic|hyperrealistic|ultra-realistic|realistic photo", _re_style.I)
_style_left = {cat: [n for n, v in _data[cat].items() if _real_pat.search(v["prompt"])]
               for cat in _data if cat != "Style"}
_style_left = {k: v for k, v in _style_left.items() if v}
check(f"非 Style 无写实风格词 ({_style_left})", len(_style_left) == 0)
check("名人无写实强制词且保留特征", "photorealistic" not in _data["Celebrity"]["Taylor Swift"]["prompt"].lower()
      and "singer-songwriter" in _data["Celebrity"]["Taylor Swift"]["prompt"]
      and "blonde hair" in _data["Celebrity"]["Taylor Swift"]["prompt"])

# 镜头效果词只允许出现在 Camera Lens（其余分类正交）
_doF_pat = _re_style.compile(r"\bdepth of field\b|\bbokeh\b", _re_style.I)
_dof_left = [f"{cat}/{n}" for cat, items in _data.items() if cat != "Camera Lens"
             for n, v in items.items() if _doF_pat.search(v["prompt"])]
check(f"非镜头分类无景深/散景词 ({_dof_left})", len(_dof_left) == 0)
check("地铁时尚肖像已去镜头词", "bokeh" not in _data["Environment"]["Subway Fashion Portrait"]["prompt"].lower())
check("窗帘柔光已去场景词", "bedroom" not in _data["Lighting"]["Curtain Filtered Light"]["prompt"].lower())
check("组随机选项存在", "随机·写实" in opt["style_preset"][0] and "随机·NSFW" in opt["pose_preset"][0])
check("组随机选项顺序", opt["style_preset"][0][2] == "随机·写实")

# seed offsets: celebrity +1, outfit +2, pose +3, couple +4, environment +5, angle +8, distance +9, lens +10
c_off2 = node._resolve_preset("Celebrity", "随机", 100 + 1)
o_off = node._resolve_preset("Outfit", "随机", 100 + 3)
p_off = node._resolve_preset("Pose", "随机", 100 + 4)
c_off = node._resolve_preset("Couple Pose", "随机", 100 + 5)
e_off = node._resolve_preset("Environment", "随机", 100 + 6)
a_off = node._resolve_preset("Camera Angle", "随机", 100 + 9)
d_off = node._resolve_preset("Camera Distance", "随机", 100 + 10)
l_off = node._resolve_preset("Camera Lens", "随机", 100 + 11)
e_off2 = node._resolve_preset("Expression", "随机", 100 + 2)
check("名人随机偏移 seed+1", r1[1]['Celebrity'] == c_off2)
check("表情随机偏移 seed+2", r1[1]['Expression'] == e_off2)
check("服装随机偏移 seed+3", r1[1]['Outfit'] == o_off)
check("姿势随机偏移 seed+4", r1[1]['Pose'] == p_off)
check("环境随机偏移 seed+6", r1[1]['Environment'] == e_off)
check("角度随机偏移 seed+9", r1[1]['Camera Angle'] == a_off)
check("距离随机偏移 seed+10", r1[1]['Camera Distance'] == d_off)
check("镜头随机偏移 seed+11", r1[1]['Camera Lens'] == l_off)
rc = node.execute("", seed=100, celebrity_preset="禁用", expression_preset="禁用", outfit_preset="禁用", pose_preset="禁用", couple_preset="随机")
check("双人随机偏移 seed+5", rc[1]['Couple Pose'] == c_off)

# 3c. enriched presets resolve correctly
_, pack = node.execute("x", seed=1, environment_preset="薰衣草田日落")
env_new = pack['Environment']
check("新增环境反查", "lavender" in env_new)
_, pack = node.execute("x", seed=1, environment_preset="樱花大道")
env_life = pack['Environment']
check("樱花大道反查", "cherry blossom" in env_life.lower())
_, pack = node.execute("x", seed=1, lighting_preset="日系柔和窗光")
light_life = pack['Lighting']
check("日系窗光反查", "japanese" in light_life.lower() and "window light" in light_life.lower())
_, pack = node.execute("x", seed=1, pose_preset="仰卧举腿")
pose_new = pack['Pose']
check("新增姿势 NSFW 反查", "legs raised" in pose_new)
_, pack = node.execute("x", seed=1, couple_preset="反向女上位")
couple_new = pack['Couple Pose']
check("新增双人 NSFW 反查", "reverse cowgirl" in couple_new)
_, pack = node.execute("x", seed=1, camera_angle_preset="荷兰角")
angle_new = pack['Camera Angle']
check("角度反查命中", "dutch" in angle_new.lower())
_, pack = node.execute("x", seed=1, camera_angle_preset="极限低角特写")
angle_new3 = pack['Camera Angle']
check("Krea2 视角反查", "low-angle" in angle_new3.lower() and "close-up" in angle_new3.lower())
_, pack = node.execute("x", seed=1, camera_angle_preset="过肩镜头")
angle_new2 = pack['Camera Angle']
check("过肩反查命中", "over-the-shoulder" in angle_new2.lower())
_, pack = node.execute("x", seed=1, camera_distance_preset="大远景")
dist_new = pack['Camera Distance']
check("大远景反查命中", "extreme long shot" in dist_new.lower())
_, pack = node.execute("x", seed=1, camera_distance_preset="牛仔镜头")
dist_new2 = pack['Camera Distance']
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
    seen.add(r[1]['Camera Lens'])
check(f"随机多样性 ({len(seen)}/43)", len(seen) > 20)
seen_angle = set()
for s in range(200):
    r = node.execute("", seed=s, camera_angle_preset="随机")
    seen_angle.add(r[1]['Camera Angle'])
check(f"角度随机多样性 ({len(seen_angle)}/22)", len(seen_angle) > 10)
seen_dist = set()
for s in range(150):
    r = node.execute("", seed=s, camera_distance_preset="随机")
    seen_dist.add(r[1]['Camera Distance'])
check(f"距离随机多样性 ({len(seen_dist)}/11)", len(seen_dist) > 5)
seen_outfit = set()
for s in range(200):
    r = node.execute("", seed=s, outfit_preset="随机")
    seen_outfit.add(r[1]['Outfit'])
check(f"服装随机多样性 ({len(seen_outfit)}/56)", len(seen_outfit) > 20)
seen_celeb = set()
for s in range(300):
    r = node.execute("", seed=s, celebrity_preset="随机")
    seen_celeb.add(r[1]['Celebrity'])
check(f"名人随机多样性 ({len(seen_celeb)}/429)", len(seen_celeb) > 50)
seen_pose = set()
for s in range(200):
    r = node.execute("", seed=s, pose_preset="随机")
    seen_pose.add(r[1]['Pose'])
check(f"姿势随机多样性 ({len(seen_pose)}/76)", len(seen_pose) > 25)
seen_couple = set()
for s in range(200):
    r = node.execute("", seed=s, couple_preset="随机")
    seen_couple.add(r[1]['Couple Pose'])
check(f"双人随机多样性 ({len(seen_couple)}/47)", len(seen_couple) > 15)

# 7. IS_CHANGED
check("IS_CHANGED 返回 seed", mod.SFPromptPreset.IS_CHANGED(seed=123) == 123)
check("VALIDATE_INPUTS 接管校验（预设被删旧工作流兼容）", mod.SFPromptPreset.VALIDATE_INPUTS(outfit_preset="全裸站立") is True)

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
check("路由返回中文名->描述", _resp.data["Environment"]["现代地铁车厢"]["description"].startswith("Contemporary subway"))
check("路由描述为英文", all(
    all(not any('\u4e00' <= c <= '\u9fff' for c in d["description"]) for d in cat.values())
    for cat in _resp.data.values()))

# 分组推导
_g = mod._preset_group
check("Celebrity 女演员组", _g("Celebrity", "Scarlett Johansson", {"tags": ["actress", "american"]}) == "女演员")
check("Celebrity 歌手组", _g("Celebrity", "Taylor Swift", {"tags": ["singer", "american"]}) == "歌手")
check("Celebrity 亚洲组", _g("Celebrity", "周杰伦", {"tags": ["singer", "taiwanese"]}) == "亚洲名人")
check("Outfit NSFW", _g("Outfit", "全裸", {"tags": ["nude", "adult"]}) == "NSFW")
check("Pose SFW", _g("Pose", "站立肖像", {"tags": ["standing"]}) == "SFW")
check("Couple SFW", _g("Couple Pose", "拥抱", {"tags": ["two people"]}) == "SFW")
check("Couple 性别细分", _data["Couple Pose"]["Missionary Position"]["group"] == "男女"
      and _data["Couple Pose"]["Lesbian Mutual Caress"]["group"] == "女女"
      and _data["Couple Pose"]["Hugging"]["group"] == "通用")
_couple_groups = [v["group"] for v in _data["Couple Pose"].values()]
check("Couple 分组覆盖", len(_couple_groups) == 47 and set(_couple_groups) == {"通用", "男女", "女女"}
      and _couple_groups.count("男女") == 18 and _couple_groups.count("女女") == 13)
check("Couple 男女随机不越界", node.execute("", seed=9, couple_preset="随机·男女")[1]["Couple Pose"] in
      {v["prompt"] for v in _data["Couple Pose"].values() if v["group"] == "男女"})
_, pack = node.execute("x", seed=1, couple_preset="肛交")
couple_anal = pack['Couple Pose']
check("男女肛交反查", "anal" in couple_anal and "man" in couple_anal)
_, pack = node.execute("x", seed=1, couple_preset="双头龙")
couple_dd = pack['Couple Pose']
check("女女双头龙反查", "double-ended dildo" in couple_dd and "lesbian" in couple_dd)
_, pack = node.execute("x", seed=1, pose_preset="跳蛋自慰")
pose_vib = pack['Pose']
check("跳蛋自慰反查", "vibrator" in pose_vib)
check("Environment 无分组", _g("Environment", "樱花大道", {"tags": []}) is None)
check("路由 group 字段", _resp.data["Celebrity"]["Taylor Swift"]["group"] == "歌手"
      and _resp.data["Outfit"]["全裸"]["group"] == "NSFW"
      and _resp.data["Environment"]["樱花大道"]["group"] == "日系生活"
      and _resp.data["Lighting"]["自然窗光"]["group"] == "自然日光")
check("重音英文名不误判亚洲", _resp.data["Celebrity"]["Beyoncé"]["group"] == "歌手"
      and _resp.data["Celebrity"]["Timothée Chalamet"]["group"] == "男演员")
check("数据驱动 group 全覆盖", all("group" in v for cat in ("Celebrity", "Outfit", "Pose", "Couple Pose") for v in _data[cat].values()))

# Style 数据驱动分组（JSON group 字段）
check("Style 写实（数据字段）", _g("Style", "写实摄影风", _data["Style"]["Photorealistic"]) == "写实")
check("Style 非写实（数据字段）", _g("Style", "动漫手绘风", _data["Style"]["Anime Painterly"]) == "非写实")
check("group 字段优先于推导", _g("Outfit", "全裸", {"tags": ["adult"], "group": "自定义组"}) == "自定义组")
check("路由 Style 分组", _resp.data["Style"]["写实摄影风"]["group"] == "写实"
      and _resp.data["Style"]["动漫手绘风"]["group"] == "非写实")
_all_style = [v["group"] for v in _data["Style"].values()]
check("Style 全部有分组", len(_all_style) == 49 and set(_all_style) == {"写实", "非写实"}
      and _all_style.count("写实") == 23 and _all_style.count("非写实") == 26)
check("时尚影棚单幅人像无拼贴词", "collage" not in _data["Style"]["Fashion Studio Single Portrait"]["prompt"].lower()
      and "single-frame" in _data["Style"]["Fashion Studio Single Portrait"]["prompt"])

# Environment 数据驱动分组
_env_groups = [v["group"] for v in _data["Environment"].values()]
check("Environment 全部有分组", len(_env_groups) == 112 and set(_env_groups) ==
      {"自然风光", "城市街景", "科幻未来", "历史复古", "恐怖暗黑", "室内空间", "日系生活", "私密场所"})
_, pack = node.execute("x", seed=1, environment_preset="都市街头（夜晚）")
check("都市街头（夜晚）反查命中 night", "nighttime" in pack["Environment"].lower())
_, pack = node.execute("x", seed=1, environment_preset="都市街头（白天）")
check("都市街头（白天）反查命中 daytime 无 night", "daytime" in pack["Environment"] and "night" not in pack["Environment"])
check("私密场所分组", _data["Environment"]["Love Hotel Room"]["group"] == "私密场所"
      and _data["Environment"]["Poolside"]["group"] == "室内空间"
      and _data["Environment"]["Onsen"]["group"] == "日系生活")
check("Environment 分组抽查", _data["Environment"]["Modern Subway Train"]["group"] == "室内空间"
      and _data["Environment"]["Space Station Hub"]["group"] == "科幻未来"
      and _data["Environment"]["Cherry Blossom Avenue"]["group"] == "日系生活"
      and _data["Environment"]["Haunted Manor"]["group"] == "恐怖暗黑")

# Camera Angle / Camera Lens 数据驱动分组
_angle_groups = [v["group"] for v in _data["Camera Angle"].values()]
check("Camera Angle 全部有分组", len(_angle_groups) == 26 and set(_angle_groups) ==
      {"机位高度", "俯仰角度", "水平朝向", "视角叙事", "创意特殊"})
_lens_groups = [v["group"] for v in _data["Camera Lens"].values()]
check("Camera Lens 全部有分组", len(_lens_groups) == 43 and set(_lens_groups) ==
      {"广角镜头", "标准镜头", "人像镜头", "长焦镜头", "微距镜头", "特殊镜头", "变焦镜头",
       "电影定焦", "变形宽银幕", "电影变焦", "复古镜头"})
check("镜头分组抽查", _data["Camera Lens"]["14mm Fisheye"]["group"] == "广角镜头"
      and _data["Camera Lens"]["Anamorphic 50mm"]["group"] == "变形宽银幕"
      and _data["Camera Angle"]["Dutch Angle"]["group"] == "创意特殊"
      and _data["Camera Angle"]["Eye Level"]["group"] == "机位高度")

# Lighting 数据驱动分组
_light_groups = [v["group"] for v in _data["Lighting"].values()]
check("Lighting 全部有分组", len(_light_groups) == 62 and set(_light_groups) ==
      {"自然日光", "人工光源", "光效氛围", "柔光漫射", "人像布光", "夜晚星光", "黄昏日落"})
check("Lighting 分组抽查", _data["Lighting"]["Natural Daylight Window"]["group"] == "自然日光"
      and _data["Lighting"]["Studio Professional"]["group"] == "人像布光"
      and _data["Lighting"]["Neon Urban Night"]["group"] == "人工光源"
      and _data["Lighting"]["Volumetric Moonlight"]["group"] == "夜晚星光")

# Camera Distance 数据驱动分组
_dist_groups = [v["group"] for v in _data["Camera Distance"].values()]
check("Camera Distance 全部有分组", len(_dist_groups) == 11 and set(_dist_groups) ==
      {"特写景别", "近景景别", "全景景别", "远景景别"})
check("景别分组抽查", _data["Camera Distance"]["Extreme Close-Up"]["group"] == "特写景别"
      and _data["Camera Distance"]["Full Shot"]["group"] == "全景景别"
      and _data["Camera Distance"]["Establishing Shot"]["group"] == "远景景别")
check("全部分类均有分组", all("group" in v for cat in _data.values() for v in cat.values()))

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
_, pack = node.execute("", seed=1, camera_distance_preset="特写", camera_lens_preset="85mm经典人像")
_c1, *_ = node.execute("test subject.", seed=1, camera_distance_preset="特写", camera_lens_preset="85mm经典人像")
check("拼接无 . , 粘连", "., " not in _c1)
_c2, *_ = node.execute("test subject", seed=1, camera_distance_preset="特写", camera_lens_preset="85mm经典人像")
check("拼接无 . , 粘连(无输入句号)", "., " not in _c2)
check("片段首字母小写化", _c2.split(", ")[1].startswith("close-up shot"))
check("input_text 首字母保持原样", _c2.startswith("test subject,"))
check("数字开头片段不受影响", "85mm classic" in _c2)

# 7d. pose/couple mutual exclusion (backend fallback, pose wins)
_, pack = node.execute("x", seed=1, pose_preset="回眸", couple_preset="公主抱")
p_m = pack['Pose']
c_m = pack['Couple Pose']
check("互斥兜底 pose 生效", "looking back over the shoulder" in p_m)
check("互斥兜底 couple 忽略", c_m == "")
_, pack = node.execute("x", seed=1, pose_preset="禁用", couple_preset="公主抱")
p_m2 = pack['Pose']
c_m2 = pack['Couple Pose']
check("仅 couple 时生效", c_m2 != "" and "bridal carry" in c_m2 and p_m2 == "")
_, pack = node.execute("x", seed=1, pose_preset="随机", couple_preset="公主抱")
p_m3 = pack['Pose']
check("互斥 pose=随机 生效", p_m3 != "")

# 服装 NSFW 反查
_, pack = node.execute("x", seed=1, outfit_preset="全透明连衣裙")
outfit_nsfw = pack['Outfit']
check("服装 NSFW 命中", "transparent" in outfit_nsfw and "adult" in outfit_nsfw.lower() or "see-through" in outfit_nsfw)
_, pack = node.execute("x", seed=1, outfit_preset="全裸")
outfit_nude = pack['Outfit']
check("全裸选项命中", "fully nude" in outfit_nude and "no clothing" in outfit_nude)
_, pack = node.execute("x", seed=1, outfit_preset="死库水（深蓝）")
outfit_swim = pack['Outfit']
check("死库水命中", "sukumizu" in outfit_swim and "navy" in outfit_swim)
_, pack = node.execute("x", seed=1, outfit_preset="黑色比基尼")
outfit_bikini = pack['Outfit']
check("纯色比基尼命中", "black bikini" in outfit_bikini and "solid" in outfit_bikini)

# 7e. 分类正交化：姿势不含场景/灯光，Style 不含镜头参数
import re as _re_orth
_pose_allowed = {"Sitting on a Chair", "Sitting Crossed Ankle", "Sitting Masturbating",
                 "Leaning on Doorframe", "In Bathtub", "Sitting on Bed Edge", "Grinding on Pillow"}
_pose_bad = [n for n, v in _data["Pose"].items()
             if n not in _pose_allowed and _re_orth.search(r"\bbed\b|bedroom|lighting", v["prompt"], _re_orth.I)]
check(f"Pose 无场景/灯光残留 ({len(_pose_bad)})", len(_pose_bad) == 0)
_couple_bad = [n for n, v in _data["Couple Pose"].items()
               if _re_orth.search(r"\bbed\b|bedroom|lighting", v["prompt"], _re_orth.I)]
check(f"Couple 无场景/灯光残留 ({len(_couple_bad)})", len(_couple_bad) == 0)
_style_bad = [n for n, v in _data["Style"].items()
              if _re_orth.search(r"\d+mm|f/\d|bokeh|depth of field", v["prompt"], _re_orth.I)]
check(f"Style 无镜头参数残留 ({len(_style_bad)})", len(_style_bad) == 0)
# 动作要素保留
check("动作要素保留", all(_re_orth.search(r"chair|doorframe|bathtub|pillow|\bbed\b", _data["Pose"][n]["prompt"], _re_orth.I)
      for n in _pose_allowed))

# 7f. Krea2SystemPrompt 预设含官方扩展
import ast as _ast
_k2 = open(os.path.join(root, "nodes", "model", "krea2.py"), encoding="utf-8").read()
_k2_tree = _ast.parse(_k2)
_k2_keys = []
for _stmt in _k2_tree.body:
    if isinstance(_stmt, _ast.Assign) and any(getattr(t, "id", "") == "KREA2_PRESETS" for t in _stmt.targets):
        for _k in _stmt.value.keys:
            _k2_keys.append(_k.value if isinstance(_k, _ast.Constant) else None)
        break
check("Krea2 预设含官方扩展", "Krea2 提示词扩展（官方规则）" in _k2_keys)
check("Krea2 预设键无重复", len(_k2_keys) == len(set(_k2_keys)))

# 7g. Krea2 适配：非镜头分类无 SD 质量标签/营销词
import re as _re_k2
_K2_BAD = _re_k2.compile(r"\b(masterpiece|best quality|8K|DSLR|ultra sharp|hyperrealistic|highly detailed|IMAX|Netflix)\b", _re_k2.I)
_k2_left = {}
for _cat, _items in _data.items():
    for _n, _v in _items.items():
        _m = _K2_BAD.findall(_v["prompt"])
        if _m:
            _k2_left.setdefault(_cat, {})[_n] = sorted(set(_m))
check(f"非镜头分类无 SD 标签残留 ({_k2_left})", len(_k2_left) == 0)
_lens_mkt = [n for n, v in _data["Camera Lens"].items() if _re_k2.search(r"IMAX|Netflix", v["prompt"], _re_k2.I)]
check(f"镜头分类无营销词 ({len(_lens_mkt)})", len(_lens_mkt) == 0)

# 7h. SFUnpackPromptPreset 解包节点
_unpack_node = mod.SFUnpackPromptPreset()
_u = _unpack_node.execute({"Celebrity": "c1", "Expression": "e1", "Outfit": "o1", "Pose": "p1",
                           "Couple Pose": "cp1", "Environment": "env1", "Lighting": "l1",
                           "Style": "s1", "Camera Angle": "a1", "Camera Distance": "d1",
                           "Camera Lens": "lens1"})
check("解包正常", _u == ("c1", "e1", "o1", "p1", "cp1", "env1", "l1", "s1", "a1", "d1", "lens1"))
check("解包容错 None", _unpack_node.execute(None) == ("",) * 11)
check("解包容错非 dict", _unpack_node.execute([1, 2]) == ("",) * 11)
check("解包容错缺键", _unpack_node.execute({"Celebrity": "c1"})[0] == "c1" and _unpack_node.execute({"Celebrity": "c1"})[1] == "")
check("打包输出结构", isinstance(node.execute("x", seed=1)[1], dict)
      and len(node.execute("x", seed=1)[1]) == 11
      and node.execute("x", seed=1)[1]["Celebrity"] == "")
check("打包输出 combined 保留", node.execute("x", seed=1)[0] == "x")

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
