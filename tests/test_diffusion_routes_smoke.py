# SF Load Diffusion Model 后端冒烟测试（Python 直接运行：python tests/test_diffusion_routes_smoke.py）
# 覆盖：
#   - diffusion_routes 纯逻辑：_human_size / _arch_from_meta / _build_dmodel_info
#     （safetensors 头部元数据 -> info 形状，触发词恒空、size/mtime、侧车胜出）
#   - dmodel_info 孤儿兜底形状所需函数在 lora_routes 域分派下的可用性
#     （_is_dmodel_req/_dom_* 与 _resolve_model_path fail-closed）
#   - lora_samples kind 参数化：_norm_kind / _model_exts / _resolve_lora_dir
#     （diffusion_models 域与 folder_paths.supported_pt_extensions 对齐）
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

LORAS_DIR = tempfile.mkdtemp(prefix="sf_dmodel_loras_")
DMODEL_DIR = tempfile.mkdtemp(prefix="sf_dmodel_models_")
USER_DIR = tempfile.mkdtemp(prefix="sf_dmodel_user_")

# 与 ComfyUI folder_paths.py:10 对齐（无 .gguf，同原生 UNETLoader 行为）
folder_paths.supported_pt_extensions = {".ckpt", ".pt", ".pt2", ".bin", ".pth",
                                        ".safetensors", ".pkl", ".sft"}

def fake_get_full_path(folder, name):
    base = {"loras": LORAS_DIR, "diffusion_models": DMODEL_DIR}.get(folder)
    if base is None:
        return None
    p = os.path.join(base, name.replace("/", os.sep))
    return p if os.path.isfile(p) else None

def fake_get_folder_paths(folder):
    return {"loras": [LORAS_DIR], "diffusion_models": [DMODEL_DIR]}.get(folder, [])

def fake_get_user_directory():
    return USER_DIR

def fake_get_filename_list(folder):
    try:
        base = {"loras": LORAS_DIR, "diffusion_models": DMODEL_DIR}[folder]
    except KeyError:
        return []
    out = []
    for dirpath, _dirnames, filenames in os.walk(base):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in folder_paths.supported_pt_extensions:
                rel = os.path.relpath(os.path.join(dirpath, fn), base)
                out.append(rel.replace(os.sep, "/"))
    return sorted(out)

folder_paths.get_full_path = fake_get_full_path
folder_paths.get_folder_paths = fake_get_folder_paths
folder_paths.get_user_directory = fake_get_user_directory
folder_paths.get_filename_list = fake_get_filename_list

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

def load_pkg(name, path):
    m = types.ModuleType(name); m.__path__ = [path]; sys.modules[name] = m; return m

load_pkg("sfnodes.nodes", os.path.join(root, "nodes"))
load_pkg("sfnodes.nodes.model", os.path.join(root, "nodes", "model"))
load_pkg("sfnodes.sf_utils", os.path.join(root, "sf_utils"))

def load_as(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

load_as("sfnodes.sf_utils.logger", os.path.join(root, "sf_utils", "logger.py"))
reader = load_as("sf_utils_lora_reader", os.path.join(root, "sf_utils", "lora_reader.py"))
lr = load_as("sfnodes.sf_utils.lora_routes", os.path.join(root, "sf_utils", "lora_routes.py"))
ls = load_as("sfnodes.sf_utils.lora_samples", os.path.join(root, "sf_utils", "lora_samples.py"))
dr = load_as("sfnodes.sf_utils.diffusion_routes", os.path.join(root, "sf_utils", "diffusion_routes.py"))
node_mod = load_as("sfnodes.nodes.model.load_diffusion_model",
                   os.path.join(root, "nodes", "model", "load_diffusion_model.py"))

DMODELS_FILE = os.path.join(USER_DIR, "sfnodes", "dmodels.json")

# ── 工具 ──
def make_safetensors(meta):
    header = {"__metadata__": meta}
    h = json.dumps(header).encode()
    return struct.pack("<Q", len(h)) + h + b"\x00" * 8

def write_model(name, meta):
    path = os.path.join(DMODEL_DIR, name.replace("/", os.sep))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(make_safetensors(meta))
    return path

def read_store():
    if not os.path.isfile(DMODELS_FILE):
        return {}
    with open(DMODELS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

# ── _human_size ──
check("size: B/KB/GB", dr._human_size(512) == "512 B" and dr._human_size(2048) == "2.0 KB"
      and dr._human_size(7 * 1024 ** 3) == "7.0 GB")
check("size: 垃圾输入", dr._human_size(None) == "" and dr._human_size(-1) == "")

# ── _arch_from_meta ──
check("arch: modelspec.architecture 优先",
      dr._arch_from_meta({"modelspec.architecture": "flux1-dev",
                          "ss_base_model_version": "SDXL"}) == "flux1-dev")
check("arch: config JSON _class_name",
      dr._arch_from_meta({"config": json.dumps({"_class_name": "Flux2Pipeline", "model_type": "x"})})
      == "Flux2Pipeline")
check("arch: 空/垃圾 meta", dr._arch_from_meta({}) == "" and dr._arch_from_meta({"config": "{bad"}) == "")

# ── 域分派（lora_routes）──
class FakeReq:
    def __init__(self, path):
        self.path = path

check("dom: 路径分派", lr._is_dmodel_req(FakeReq("/api/sfnodes/dmodel/civitai")) is True
      and lr._is_dmodel_req(FakeReq("/api/sfnodes/lora/civitai")) is False)
check("dom: notes/previews 目录分域",
      lr._dmodels_file().endswith("dmodels.json")
      and lr._previews_model_dir().endswith("previews_model"))

# ── _resolve_model_path fail-closed ──
p1 = write_model("test_model.safetensors", {"config": json.dumps({"architecture": "wan2.2"})})
check("resolve: 域内文件命中", lr._resolve_model_path("test_model.safetensors") == p1)
check("resolve: 域外名字拒绝", lr._resolve_model_path("../escape.safetensors") is None
      and lr._resolve_model_path("") is None and lr._resolve_model_path(None) is None)

# ── lora_samples kind 参数化 ──
check("samples: _norm_kind 白名单", ls._norm_kind("diffusion_models") == "diffusion_models"
      and ls._norm_kind("loras") == "loras" and ls._norm_kind("hacker") == "loras")
check("samples: 扩展名随域切换", ".gguf" in ls._model_exts("loras")
      and ".gguf" not in ls._model_exts("diffusion_models")
      and ".safetensors" in ls._model_exts("diffusion_models"))
check("samples: diffusion 域解析命中目录", ls._resolve_lora_dir("test_model.safetensors", "diffusion_models") == DMODEL_DIR)
check("samples: loras 域不见 diffusion 文件", ls._resolve_lora_dir("test_model.safetensors") is None)

# ── _build_dmodel_info ──
meta = {
    "modelspec.title": "My Model",
    "modelspec.description": "<b>hello</b> world",
    "modelspec.date": "2026-08-23T00:00:00",
    "config": json.dumps({"architecture": "qwen-image"}),
}
with open(p1, "wb") as f:
    f.write(make_safetensors(meta))   # 覆写为完整元数据（此前仅为 resolve 检查写过最小头）
info = dr._build_dmodel_info(p1, "sub/test_model.safetensors")
check("info: 标题取 modelspec.title", info["title"] == "My Model")
check("info: 架构串进 base_model", info["base_model"] == "qwen-image")
check("info: 触发词恒空三组", info["triggers"] == [] and info["file_triggers"] == []
      and info["sidecar_triggers"] == [])
check("info: size/mtime 附带", info["size"].endswith("B") and isinstance(info["mtime"], int))
check("info: 描述经 markdown 清洗", "hello" in info["description"])
check("info: source=file 且 has_preview 键存在", info["source"] == "file" and "has_preview" in info)

# 侧车胜出语义（与 build_lora_info 一致）。侧车是原始 Civitai API 形状
# （read_sidecar_info 消费 trainedWords/model.name/baseModel/modelId/id）。
sidecar = {
    "model": {"name": "Civitai Name"},
    "description": "civ desc",
    "baseModel": "Flux.1 Dev",
    "modelId": 123,
    "id": 456,
    "trainedWords": ["ignored"],
}
with open(os.path.splitext(p1)[0] + ".civitai.info", "w", encoding="utf-8") as f:
    json.dump(sidecar, f)
info2 = dr._build_dmodel_info(p1, "test_model.safetensors")
check("info: 侧车标题/架构/描述/id 胜出",
      info2["title"] == "Civitai Name" and info2["base_model"] == "Flux.1 Dev"
      and info2["description"] == "civ desc" and info2["civitai_description"] == "civ desc"
      and info2["model_id"] == 123 and info2["version_id"] == 456
      and info2["source"] == "sidecar")
check("info: 侧车触发词仍不采纳", info2["triggers"] == [])

# ── 用户数据域隔离（dmodels.json，不碰 lora_triggers.json）──
R = reader
R.set_custom_description(DMODELS_FILE, "test_model.safetensors", "my custom note", fp=None)
store = read_store()
check("store: 写入 dmodels.json", any("my custom note" == v.get("description")
                                      for v in store.values()))
check("store: get_custom_description 读回",
      R.get_custom_description(DMODELS_FILE, "test_model.safetensors") == "my custom note")

# 孤儿检测（基名兜底）：旧键数据在新名下应被找到
orphan = R.find_orphan_key(store, "moved/test_model.safetensors")
check("orphan: 基名唯一匹配命中旧键", orphan is not None and orphan.endswith("test_model.safetensors"))

# ── 节点模块 ──
check("node: 类与注册常量", hasattr(node_mod, "SFLoadDiffusionModel")
      and node_mod.SFLoadDiffusionModel.CATEGORY == "sfnodes/model"
      and node_mod.SFLoadDiffusionModel.FUNCTION == "execute")
it = node_mod.SFLoadDiffusionModel.INPUT_TYPES()["required"]
check("node: INPUT_TYPES 形状", list(it.keys()) == ["unet_name", "weight_dtype"]
      and "test_model.safetensors" in it["unet_name"][0]
      and it["weight_dtype"][1].get("advanced") is True)
check("node: RETURN_TYPES", node_mod.SFLoadDiffusionModel.RETURN_TYPES == ("MODEL",))

print()
if failures:
    print("FAILED: {} check(s)".format(len(failures)))
    sys.exit(1)
print("ALL PASS")
