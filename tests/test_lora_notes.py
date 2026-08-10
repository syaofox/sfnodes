# lora_notes 统一存储网关测试（Python 直接运行：python tests/test_lora_notes.py）
# 覆盖（2026-08 统一用户数据存储：Power 系 lora_notes 与 SFLoraStack 共用
# user/sfnodes/lora_triggers.json 真源，旧 <base>.sf.json 侧车惰性迁移）：
#   - split_trigger_text：逗号/中文逗号/换行拆分、清洗、垃圾输入
#   - get_merged_metadata：embedded / .civitai.info 侧车 / 统一存储优先级、
#     _not_found / _has_custom / 返回形状兼容旧 /lora_notes
#   - 惰性迁移：.sf.json -> 统一存储 + 删除侧车 + 幂等 + store 已有数据不迁
#   - set_custom_notes：拆分写入、空数据清条目、返回 merged 形状
import importlib.util
import json
import os
import struct
import sys
import tempfile
import types

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

failures = []

def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")

# ── mock folder_paths / aiohttp / PIL（本机无 ComfyUI 运行时与依赖）─────────
folder_paths = types.ModuleType("folder_paths"); sys.modules["folder_paths"] = folder_paths

LORAS_DIR = tempfile.mkdtemp(prefix="sf_lora_notes_loras_")
USER_DIR = tempfile.mkdtemp(prefix="sf_lora_notes_user_")

def fake_get_full_path(folder, name):
    if folder != "loras":
        return None
    return os.path.join(LORAS_DIR, name.replace("/", os.sep))

def fake_get_folder_paths(folder):
    return [LORAS_DIR] if folder == "loras" else []

def fake_get_user_directory():
    return USER_DIR

folder_paths.get_full_path = fake_get_full_path
folder_paths.get_folder_paths = fake_get_folder_paths
folder_paths.get_user_directory = fake_get_user_directory

aiohttp = types.ModuleType("aiohttp"); sys.modules["aiohttp"] = aiohttp
web_mod = types.ModuleType("aiohttp.web")
web_mod.json_response = lambda *a, **k: None
web_mod.Response = lambda *a, **k: None
web_mod.FileResponse = lambda *a, **k: None
sys.modules["aiohttp.web"] = web_mod
aiohttp.web = web_mod

pil = types.ModuleType("PIL"); sys.modules["PIL"] = pil
pil_image = types.ModuleType("PIL.Image"); sys.modules["PIL.Image"] = pil_image
pil_ops = types.ModuleType("PIL.ImageOps"); sys.modules["PIL.ImageOps"] = pil_ops
pil.Image = pil_image
pil.ImageOps = pil_ops

# ── 注册 sfnodes 包结构（相对导入 from .xxx 需要）───────────────────────────
pkg = types.ModuleType("sfnodes"); pkg.__path__ = [root]; sys.modules["sfnodes"] = pkg
pkg2 = types.ModuleType("sfnodes.nodes"); pkg2.__path__ = [os.path.join(root, "nodes")]; sys.modules["sfnodes.nodes"] = pkg2
pkg3 = types.ModuleType("sfnodes.nodes.model"); pkg3.__path__ = [os.path.join(root, "nodes", "model")]; sys.modules["sfnodes.nodes.model"] = pkg3
pkg4 = types.ModuleType("sfnodes.sf_utils"); pkg4.__path__ = [os.path.join(root, "sf_utils")]; sys.modules["sfnodes.sf_utils"] = pkg4

spec_logger = importlib.util.spec_from_file_location(
    "sfnodes.sf_utils.logger",
    os.path.join(root, "sf_utils", "logger.py"),
)
logger_mod = importlib.util.module_from_spec(spec_logger)
sys.modules["sfnodes.sf_utils.logger"] = logger_mod
spec_logger.loader.exec_module(logger_mod)

def load_as(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# lora_reader 纯逻辑（无相对导入）
reader = load_as("sf_utils_lora_reader", os.path.join(root, "sf_utils", "lora_reader.py"))
# lora_routes（import lora_reader；路由注册因无 server 被 try/except 吞掉）
load_as("sfnodes.sf_utils.lora_routes", os.path.join(root, "sf_utils", "lora_routes.py"))
# lora_samples（import PIL mock）
load_as("sfnodes.sf_utils.lora_samples", os.path.join(root, "sf_utils", "lora_samples.py"))
# lora_notes（统一存储网关）
notes = load_as("sfnodes.sf_utils.lora_notes", os.path.join(root, "sf_utils", "lora_notes.py"))

# 统一存储路径（与 lora_routes 一致 -> USER_DIR/sfnodes/lora_triggers.json）
STORE = os.path.join(USER_DIR, "sfnodes", "lora_triggers.json")

# ── 工具 ──
def make_safetensors(meta):
    header = {"__metadata__": meta}
    h = json.dumps(header).encode()
    return struct.pack("<Q", len(h)) + h + b"\x00" * 8

def write_lora(name, meta):
    path = os.path.join(LORAS_DIR, name.replace("/", os.sep))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(make_safetensors(meta))
    return path

def read_store():
    if not os.path.isfile(STORE):
        return {}
    with open(STORE, "r", encoding="utf-8") as f:
        return json.load(f)

# ── split_trigger_text ──
check("split: 逗号+中文逗号+换行混合", reader.split_trigger_text("a, b\nc，d") == ["a", "b", "c", "d"])
check("split: 单个词", reader.split_trigger_text("alpha") == ["alpha"])
check("split: 去空与去重", reader.split_trigger_text(", a, , a, \n") == ["a"])
check("split: 空串/垃圾", reader.split_trigger_text("") == [] and reader.split_trigger_text(None) == [] and reader.split_trigger_text(42) == [])

# ── get_merged_metadata：文件缺失 ──
check("merged: 文件缺失 _not_found", notes.get_merged_metadata("missing.safetensors") == {"_not_found": True})

# ── get_merged_metadata：embedded 兜底 ──
write_lora("emb.safetensors", {
    "modelspec.trigger_phrase": "alpha, beta",
    "modelspec.description": "embedded desc",
    "ss_base_model_version": "sd_xl_base_1.0",
    "source_url": "https://example.com/m",
})
m = notes.get_merged_metadata("emb.safetensors")
check("merged: embedded 触发词", m["trigger_words"] == "alpha, beta")
check("merged: embedded 描述", m["description"] == "embedded desc")
check("merged: embedded base_model", m["base_model"] == "sd_xl_base_1.0")
check("merged: embedded source_url", m["source_url"] == "https://example.com/m")
check("merged: 无自定义 _has_custom=False", m["_has_custom"] is False and m["_has_embedded"] is True)

# ── get_merged_metadata：.civitai.info 侧车优先于 embedded ──
# 侧车命名约定（lora_reader.read_sidecar_info）：<base>（去扩展名）+ .civitai.info
civ_path = os.path.join(LORAS_DIR, "emb.civitai.info")
with open(civ_path, "w", encoding="utf-8") as f:
    json.dump({"trainedWords": ["civ1", "civ2"], "description": "civ desc"}, f)
m = notes.get_merged_metadata("emb.safetensors")
check("merged: sidecar 触发词优先于 embedded", m["trigger_words"] == "civ1, civ2")
check("merged: sidecar 描述优先于 embedded", m["description"] == "civ desc")
check("merged: 仅 sidecar 有信息 _has_custom=True", m["_has_custom"] is True)
os.remove(civ_path)

# ── get_merged_metadata：统一存储优先于 sidecar/embedded ──
notes.set_custom_notes("emb.safetensors", {"trigger_words": "mine1, mine2", "description": "my desc"})
m = notes.get_merged_metadata("emb.safetensors")
check("merged: 统一存储触发词优先", m["trigger_words"] == "mine1, mine2")
check("merged: 统一存储描述优先", m["description"] == "my desc")
check("merged: 有自定义 _has_custom=True", m["_has_custom"] is True)
store = read_store()
check("store: words 数组形状", store["emb.safetensors"]["words"] == ["mine1", "mine2"])
check("store: description 直写", store["emb.safetensors"]["description"] == "my desc")

# ── 惰性迁移：.sf.json -> 统一存储 + 删除侧车 + 幂等 ──
legacy = write_lora("legacy.safetensors", {"modelspec.trigger_phrase": "oldfile"})
sf_path = legacy + ".sf.json"
with open(sf_path, "w", encoding="utf-8") as f:
    json.dump({"trigger_words": "leg1, leg2", "description": "legacy desc"}, f)
m = notes.get_merged_metadata("legacy.safetensors")
check("migrate: 旧侧车词被读取", m["trigger_words"] == "leg1, leg2")
check("migrate: 旧侧车描述被读取", m["description"] == "legacy desc")
check("migrate: 迁移后侧车已删除", not os.path.isfile(sf_path))
check("migrate: 数据并入统一存储", read_store().get("legacy.safetensors", {}).get("words") == ["leg1", "leg2"])
check("migrate: 幂等（侧车已删不再迁移）", notes.get_merged_metadata("legacy.safetensors")["trigger_words"] == "leg1, leg2")

# store 已有数据时 .sf.json 不迁移、不删除
sf_path2 = os.path.join(LORAS_DIR, "emb.safetensors.sf.json")
with open(sf_path2, "w", encoding="utf-8") as f:
    json.dump({"trigger_words": "stale"}, f)
m = notes.get_merged_metadata("emb.safetensors")
check("migrate: store 有数据时旧侧车不覆盖", m["trigger_words"] == "mine1, mine2")
check("migrate: store 有数据时旧侧车保留", os.path.isfile(sf_path2))
os.remove(sf_path2)

# 空侧车（无可迁移内容）跳过
empty_path = write_lora("empty.sf.safetensors", {})
e_sf = empty_path + ".sf.json"
with open(e_sf, "w", encoding="utf-8") as f:
    json.dump({"trigger_words": "", "description": ""}, f)
m = notes.get_merged_metadata("empty.sf.safetensors")
check("migrate: 空侧车跳过", not os.path.isfile(e_sf) or read_store().get("empty.sf.safetensors") is None)

# ── set_custom_notes：清空删条目 ──
m = notes.set_custom_notes("emb.safetensors", {})
check("set: 空数据回到 embedded 词", m["trigger_words"] == "alpha, beta")
check("set: 空数据删除条目", "emb.safetensors" not in read_store())
check("set: 文件缺失返回 {}", notes.set_custom_notes("nope.safetensors", {"trigger_words": "x"}) == {})

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("ALL PASS")
