# VOSR2 loader / inferencer 纯逻辑测试（stub torch / comfy / folder_paths，不联网不加载权重）：
#  - bundle 目录发现（主目录 + 旧包目录兼容）与安全路径校验
#  - args.json 严格校验、权重文件查找优先级、DINO .pth → .safetensors 识别
#  - dtype 解析、checkpoint key 清洗规则、缺失文件且关闭下载时报错
#  - 瓦片/显存降级辅助函数、DinoTemporalCache 基本行为
# 运行：python tests/test_vosr2_loader.py
import importlib.util
import os
import sys
import tempfile
import types
from pathlib import Path

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

failures = []

def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")

# ── stub torch（只满足 import 期求值：nn.Module / torch.compiler.disable / F / cuda）──
class _Module:
    pass

class _NNAttr(types.ModuleType):
    """torch.nn stub：未显式定义的属性按需返回占位类（import 期只求值类基/默认参数）。"""
    def __getattr__(self, name):
        return type(name, (), {})


torch = types.ModuleType("torch")
torch.nn = _NNAttr("torch.nn")
torch.nn.Module = _Module
torch.nn.functional = types.ModuleType("torch.nn.functional")
torch.compiler = types.ModuleType("torch.compiler")
torch.compiler.disable = lambda fn: fn
torch.compiler.is_compiling = lambda: False
torch.cuda = types.ModuleType("torch.cuda")
torch.cuda.OutOfMemoryError = type("OutOfMemoryError", (RuntimeError,), {})
torch.float16 = "float16"
torch.bfloat16 = "bfloat16"
torch.float32 = "float32"
torch.Tensor = type("Tensor", (), {})
sys.modules["torch"] = torch
sys.modules["torch.nn"] = torch.nn
sys.modules["torch.nn.functional"] = torch.nn.functional
sys.modules["torch.compiler"] = torch.compiler
sys.modules["torch.cuda"] = torch.cuda

# ── stub einops / safetensors ──
einops = types.ModuleType("einops")
einops.rearrange = lambda *a, **k: None
einops.repeat = lambda *a, **k: None
sys.modules["einops"] = einops

safetensors = types.ModuleType("safetensors")
safetensors.safe_open = lambda *a, **k: None
safetensors_torch = types.ModuleType("safetensors.torch")
safetensors_torch.save_file = lambda *a, **k: None
safetensors_torch.load_file = lambda *a, **k: {}
sys.modules["safetensors"] = safetensors
sys.modules["safetensors.torch"] = safetensors_torch

# ── stub comfy / folder_paths ──
models_dir = tempfile.mkdtemp(prefix="vosr2_models_")
folder_paths = types.ModuleType("folder_paths")
folder_paths.models_dir = models_dir
folder_paths.get_temp_directory = lambda: tempfile.gettempdir()
folder_paths.get_folder_paths = lambda name: [str(Path(models_dir) / name)]
# 已注册类别 → 用于推导额外模型根目录（模拟 extra_model_paths 合并后的状态）
folder_paths.folder_names_and_paths = {"checkpoints": ([str(Path(models_dir) / "checkpoints")], set())}
sys.modules["folder_paths"] = folder_paths

comfy = types.ModuleType("comfy")
comfy.ops = types.SimpleNamespace(disable_weight_init=types.SimpleNamespace())
comfy.utils = types.ModuleType("comfy.utils")
comfy.utils.load_torch_file = lambda *a, **k: {}
comfy.utils.ProgressBar = type(
    "ProgressBar", (), {"__init__": lambda self, total: setattr(self, "total", total),
                        "current": 0, "update_absolute": lambda self, v, total=None: None}
)
comfy.model_management = types.ModuleType("comfy.model_management")
comfy.model_management.unet_dtype = lambda device=None: "default_dtype"
comfy.model_management.get_torch_device = lambda: "cpu"
comfy.model_management.unet_offload_device = lambda: "cpu"
comfy.model_management.module_size = lambda m: 0
comfy.model_management.get_free_memory = lambda dev=None, torch_free_too=False: 0
comfy.model_management.free_memory = lambda *a, **k: []
comfy.model_management.soft_empty_cache = lambda force=False: None
comfy.model_management.load_models_gpu = lambda *a, **k: None
comfy.model_patcher = types.ModuleType("comfy.model_patcher")
comfy.model_patcher.ModelPatcher = type("ModelPatcher", (), {})
sys.modules["comfy"] = comfy
sys.modules["comfy.ops"] = comfy.ops
sys.modules["comfy.utils"] = comfy.utils
sys.modules["comfy.model_management"] = comfy.model_management
sys.modules["comfy.model_patcher"] = comfy.model_patcher

# ── 包骨架 ──
for name, path in [("sfnodes", "."), ("sfnodes.nodes", "nodes"), ("sfnodes.nodes.model", "nodes/model"),
                   ("sfnodes.nodes.model.vosr2", "nodes/model/vosr2")]:
    m = types.ModuleType(name)
    m.__path__ = [os.path.join(root, path)]
    sys.modules[name] = m


def _load(mod_name, rel_path):
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(root, rel_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


loader = _load("sfnodes.nodes.model.vosr2.loader", "nodes/model/vosr2/loader.py")
inferencer = _load("sfnodes.nodes.model.vosr2.inferencer", "nodes/model/vosr2/inferencer.py")
settings_mod = _load("sfnodes.nodes.model.vosr2.settings", "nodes/model/vosr2/settings.py")

# ── _strip_key ──
check("strip module. 前缀", loader._strip_key("module.blocks.0.weight") == "blocks.0.weight")
check("strip ema_model. 前缀", loader._strip_key("ema_model.x") == "x")
check("drop n_averaged", loader._strip_key("n_averaged") is None)
check("drop step_count", loader._strip_key("step_count") is None)
check("普通键保留", loader._strip_key("final_layer.linear.weight") == "final_layer.linear.weight")

# ── _safe_child_dir ──
try:
    loader._safe_child_dir(Path("/tmp"), "../etc")
    check("拒绝 .. 路径", False)
except loader.VOSR2LoadError:
    check("拒绝 .. 路径", True)
try:
    loader._safe_child_dir(Path("/tmp"), "a/b")
    check("拒绝含斜杠名", False)
except loader.VOSR2LoadError:
    check("拒绝含斜杠名", True)
check("合法名通过", str(loader._safe_child_dir(Path("/tmp"), "VOSR2")) == "/tmp/VOSR2")

# ── _load_args_json ──
valid_args = dict(loader.REQUIRED_ARGS)
for key in loader.DIT_ARG_KEYS:
    valid_args[key] = 1
with tempfile.TemporaryDirectory() as tmp:
    bundle = Path(tmp) / "VOSR2"
    bundle.mkdir()
    import json as _json
    (bundle / "args.json").write_text(_json.dumps(valid_args))
    args = loader._load_args_json(bundle)
    check("args.json 合法通过", args["dim"] == 1536)

    bad = dict(valid_args, dim=768)
    (bundle / "args.json").write_text(_json.dumps(bad))
    try:
        loader._load_args_json(bundle)
        check("架构不符报错", False)
    except loader.VOSR2LoadError:
        check("架构不符报错", True)

    missing = {k: v for k, v in valid_args.items() if k != "use_rope"}
    (bundle / "args.json").write_text(_json.dumps(missing))
    try:
        loader._load_args_json(bundle)
        check("缺字段报错", False)
    except loader.VOSR2LoadError:
        check("缺字段报错", True)

    (bundle / "args.json").unlink()
    try:
        loader._load_args_json(bundle)
        check("缺 args.json 报错", False)
    except loader.VOSR2LoadError:
        check("缺 args.json 报错", True)

# ── 权重查找优先级 ──
with tempfile.TemporaryDirectory() as tmp:
    bundle = Path(tmp)
    (bundle / "clean_weights").mkdir()
    (bundle / "checkpoints").mkdir()
    (bundle / "clean_weights" / "ema_model.safetensors").write_text("x")
    (bundle / "checkpoints" / "ema_model.safetensors").write_text("x")
    (bundle / "ema_model.safetensors").write_text("x")
    found = loader._find_dit_weight(bundle)
    check("DiT 权重优先 clean_weights", found == bundle / "clean_weights" / "ema_model.safetensors")
    (bundle / "clean_weights" / "ema_model.safetensors").unlink()
    check("DiT 回退 checkpoints", loader._find_dit_weight(bundle) == bundle / "checkpoints" / "ema_model.safetensors")
    (bundle / "checkpoints" / "ema_model.safetensors").unlink()
    check("DiT 回退 bundle 根", loader._find_dit_weight(bundle) == bundle / "ema_model.safetensors")

    check("无 DiT 权重返回 None", loader._find_dit_weight(bundle / "nope") is None)

    (bundle / "dinov2_vitl14_pretrain.pth").write_text("x")
    path, convert = loader._find_dino_weight(bundle)
    check("DINO .pth 需转换", convert and path.name == "dinov2_vitl14_pretrain.pth")
    (bundle / "dinov2_vitl14.safetensors").write_text("x")
    path, convert = loader._find_dino_weight(bundle)
    check("DINO safetensors 优先", not convert and path.name == "dinov2_vitl14.safetensors")

# ── dtype 解析 ──
check("dtype fp16", loader._resolve_dtype("fp16", "cpu") == "float16")
check("dtype bf16", loader._resolve_dtype("bf16", "cpu") == "bfloat16")
check("dtype default 走 ComfyUI", loader._resolve_dtype("default", "cpu") == "default_dtype")
try:
    loader._resolve_dtype("int8", "cpu")
    check("未知 dtype 报错", False)
except loader.VOSR2LoadError:
    check("未知 dtype 报错", True)

# ── 目录发现：主目录 + 旧包目录 ──
primary = Path(models_dir) / "sfnodes" / "vosr2"
legacy = Path(models_dir) / "vosr2"
(primary / "VOSR2").mkdir(parents=True)
(primary / "VOSR2" / "args.json").write_text("{}")
(legacy / "VOSR2").mkdir(parents=True)
(legacy / "VOSR2" / "args.json").write_text("{}")
(legacy / "Custom").mkdir(parents=True)
(legacy / "Custom" / "args.json").write_text("{}")
check("bundle 列表去重合并", loader.list_model_bundles() == ["Custom", "VOSR2"])
check("combo 含已知模型", "VOSR2" in loader.model_options())
check("主目录优先", loader.resolve_bundle_dir("VOSR2")[0] == primary / "VOSR2")
check("旧目录回退", loader.resolve_bundle_dir("Custom")[0] == legacy / "Custom")
check("未知 bundle 落到主目录", loader.resolve_bundle_dir("Nope")[0] == primary / "Nope")

# 空目录场景：combo 仍提供 KNOWN_MODEL
empty_models = tempfile.mkdtemp(prefix="vosr2_empty_")
folder_paths.models_dir = empty_models
folder_paths.folder_names_and_paths = {"checkpoints": ([str(Path(empty_models) / "checkpoints")], set())}
loader.BUNDLE_ROOT = Path(empty_models) / "sfnodes" / "vosr2"
loader.LEGACY_BUNDLE_ROOT = Path(empty_models) / "vosr2"
check("空目录 combo 含已知模型", loader.model_options() == [loader.KNOWN_MODEL])
try:
    loader.ensure_bundle_files("VOSR2", auto_download=False)
    check("缺文件且关下载报错", False)
except loader.VOSR2LoadError:
    check("缺文件且关下载报错", True)
try:
    loader.ensure_bundle_files("Custom", auto_download=True)
    check("非已知 bundle 不下载", False)
except loader.VOSR2LoadError:
    check("非已知 bundle 不下载", True)

# 完整 bundle：ensure 不联网直接返回
complete = loader.BUNDLE_ROOT / "VOSR2"
(complete / "checkpoints").mkdir(parents=True)
(complete / "args.json").write_text("{}")
(complete / "checkpoints" / "ema_model.safetensors").write_text("x")
(complete / "dinov2_vitl14.safetensors").write_text("x")
(complete / "Qwen-Image-vae-2d").mkdir()
(complete / "Qwen-Image-vae-2d" / "config.json").write_text("{}")
(complete / "Qwen-Image-vae-2d" / "diffusion_pytorch_model.safetensors").write_text("x")
try:
    loader.ensure_bundle_files("VOSR2", auto_download=True)
    check("完整 bundle 不联网通过", True)
except Exception as exc:
    check(f"完整 bundle 不联网通过 ({exc})", False)

# 额外模型根（extra_model_paths / models-ext）发现：<ext>/vosr2/<bundle>
ext_root = Path(tempfile.mkdtemp(prefix="vosr2_ext_"))
(ext_root / "vosr2" / "ExtBundle").mkdir(parents=True)
(ext_root / "vosr2" / "ExtBundle" / "args.json").write_text("{}")
(ext_root / "sfnodes" / "vosr2" / "ExtSfnodes").mkdir(parents=True)
(ext_root / "sfnodes" / "vosr2" / "ExtSfnodes" / "args.json").write_text("{}")
folder_paths.folder_names_and_paths = {"checkpoints": ([str(ext_root / "checkpoints")], set())}
found = loader.list_model_bundles()
check("额外根 vosr2 被发现", "ExtBundle" in found)
check("额外根 sfnodes/vosr2 被发现", "ExtSfnodes" in found)
check("额外根优先于下载兜底", loader.resolve_bundle_dir("ExtBundle")[0] == ext_root / "vosr2" / "ExtBundle")
check("未找到时仍回内置主目录", loader.resolve_bundle_dir("Missing")[0] == loader.BUNDLE_ROOT / "Missing")
# 恢复内置根，避免影响后续用例
folder_paths.folder_names_and_paths = {"checkpoints": ([str(Path(empty_models) / "checkpoints")], set())}

# ── 运行时封装：策略与编译幂等 ──
class _FakePatcher:
    model = types.SimpleNamespace()
    load_device = "cpu"
    def cleanup(self):
        pass

bundle = loader.VOSR2ModelBundle(_FakePatcher(), _FakePatcher(), _FakePatcher(),
                                 {"layer_dinov2b_list": [17], "dinov2_size": 448})
bundle.set_memory_policy("staged")
check("显存策略设置", bundle.memory_policy == "staged")
bundle.set_memory_policy("bogus")
check("非法策略回退 auto", bundle.memory_policy == "auto")
check("关闭编译返回 False", bundle.set_torch_compile(False) is False)
check("compile_active 默认 False", bundle.compile_active is False)

# ── inferencer 辅助函数 ──
d = settings_mod.default_settings()
check("tile_size>0 原样", inferencer.VOSR2Inferencer._effective_tile_size(512, d) == 512)
check("tile_size=0 auto 关闭", inferencer.VOSR2Inferencer._effective_tile_size(0, d) == 0)
tiled = settings_mod.VOSR2Settings(tile_strategy="tiled")
check("tiled 策略兜底 512", inferencer.VOSR2Inferencer._effective_tile_size(0, tiled) == 512)

check("VAE 降级 1024→512", inferencer.VOSR2Inferencer._fallback_vae_tile(1024, 4096, 4096) == 512)
check("VAE 降级 256→None", inferencer.VOSR2Inferencer._fallback_vae_tile(256, 4096, 4096) is None)
check("VAE 整图 OOM → 1024 分块", inferencer.VOSR2Inferencer._fallback_vae_tile(0, 4096, 4096) == 1024)
check("VAE 整图小图 OOM → None", inferencer.VOSR2Inferencer._fallback_vae_tile(0, 512, 512) is None)

# ── DinoTemporalCache：关闭态与几何失效 ──
cache = inferencer.DinoTemporalCache(enabled=False)
check("缓存关闭不复用", cache.reuse(("a",), None, 0, "cpu", None) is None)
cache = inferencer.DinoTemporalCache(enabled=True, threshold=0.05, refresh=0)
cache._feats[("a",)] = "feat"
cache._signatures[("a",)] = "sig"
cache.begin_frame((512, 512, 64, [(0, 0)]))
check("几何变化清空缓存", cache._feats == {} and cache._signatures == {})


# ── 批次分组 / 进度总量 ──
class _FakeBatch:
    """最小 IMAGE 批次桩：支持 shape / 整数索引 / 切片（用于 _shape_groups / _make_progress）。"""
    def __init__(self, shapes):
        self._shapes = list(shapes)
        self.shape = (len(self._shapes),) + tuple(self._shapes[0]) if self._shapes else (0,)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return types.SimpleNamespace(shape=self._shapes[idx])
        sub = _FakeBatch.__new__(_FakeBatch)
        sub._shapes = self._shapes[idx]
        sub.shape = (len(sub._shapes),) + tuple(sub._shapes[0]) if sub._shapes else (0,)
        return sub


check("同形状合并分组", inferencer.VOSR2Inferencer._shape_groups(
    _FakeBatch([(8, 8, 3), (8, 8, 3), (16, 16, 3)])) == [(0, 2), (2, 3)])
check("单张分组", inferencer.VOSR2Inferencer._shape_groups(_FakeBatch([(4, 4, 3)])) == [(0, 1)])

inf = inferencer.VOSR2Inferencer(None)
pbar = inf._make_progress(_FakeBatch([(512, 512, 3)]), upscale=4, tile_size=512,
                          tile_overlap=32, settings=d, progress=True)
expected_tiles = inferencer.dit_tile_count(2048, 2048, 512, 32)
check("进度总量 = 项数 × 瓦片数", pbar.total == expected_tiles and expected_tiles > 1)
pbar_small = inf._make_progress(_FakeBatch([(64, 64, 3)]), upscale=1, tile_size=512,
                                tile_overlap=32, settings=d, progress=True)
check("小图进度总量 = 项数", pbar_small.total == 1)
check("关闭进度返回 None", inf._make_progress(_FakeBatch([(64, 64, 3)]), 1, 512, 32, d, False) is None)

if failures:
    print(f"\n{len(failures)} 项失败: {failures}")
    sys.exit(1)
print("\n全部通过")
