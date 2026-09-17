# SF SCAIL-2 后端测试（Node/Python 直接运行：python3 tests/test_scail2.py）
# 覆盖：
#   - sf_utils/scail2_easy.py 纯逻辑：帧数 4n+1 取整、32 对齐、Fit Video 尺寸策略、
#     色板、多主体拼图布局候选、外部分段锚帧归一/首段丢弃（无 torch 依赖）
#   - nodes/video/scail2.py 四节点结构：CATEGORY/RETURN_TYPES/RETURN_NAMES/FUNCTION/
#     DESCRIPTION、INPUT_TYPES 关键项、模块内双注册字典
#   - 根 __init__.py 注册键一致（4 键各出现两次：类映射 + 显示名映射）
# mock：torch（节点模块顶层 import torch / torch.nn.functional；方法内才用张量）
import importlib.util
import inspect
import os
import sys
import types

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

failures = []


def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")


def assert_eq(a, b, msg=""):
    check(f"{msg}: {a!r} == {b!r}", a == b)


# ── mock torch（仅需顶层符号，方法内不实际执行）──
torch = types.ModuleType("torch")
torch.nn = types.ModuleType("torch.nn")
torch.nn.functional = types.ModuleType("torch.nn.functional")
torch.Tensor = object
sys.modules["torch"] = torch
sys.modules["torch.nn"] = torch.nn
sys.modules["torch.nn.functional"] = torch.nn.functional

# ── 纯逻辑模块（无依赖，直接加载）──
spec = importlib.util.spec_from_file_location("scail2_easy", os.path.join(root, "sf_utils", "scail2_easy.py"))
easy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(easy)

# 帧数 4n+1
assert_eq(easy.wan_frame_count_cover(1), 1, "cover(1)")
assert_eq(easy.wan_frame_count_cover(2), 5, "cover(2)")
assert_eq(easy.wan_frame_count_cover(5), 5, "cover(5)")
assert_eq(easy.wan_frame_count_cover(81), 81, "cover(81)")
assert_eq(easy.wan_frame_count_floor(1), 1, "floor(1)")
assert_eq(easy.wan_frame_count_floor(81), 81, "floor(81)")
assert_eq(easy.wan_frame_count_floor(82), 81, "floor(82)")
assert_eq(easy.wan_frame_count_floor(85), 85, "floor(85)")

# 32 对齐
assert_eq(easy.round_32(31), 32, "round_32(31)")
assert_eq(easy.round_32(480), 480, "round_32(480)")
assert_eq(easy.round_nearest_32(47), 32, "round_nearest_32(47)")
assert_eq(easy.round_nearest_32(48), 64, "round_nearest_32(48)")

# 上下文窗口调度枚举（对齐 comfy.context_windows.ContextSchedules）
assert_eq(easy.CONTEXT_SCHEDULES, ("standard_static", "standard_uniform", "looped_uniform", "batched"), "context schedules")

# Fit Video 尺寸策略
assert_eq(easy.target_size_for_video(480, 832, "512p"), (896, 512), "target 512p")
assert_eq(easy.target_size_for_video(480, 832, "704p"), (1216, 704), "target 704p")
assert_eq(easy.target_size_for_video(100, 100, "custom", 100, 200), (96, 192), "target custom")

# 生成尺寸推断
assert_eq(easy.infer_generation_size(480, 832), (832, 480), "infer size")

# 外部分段锚帧归一 / 首段丢弃
assert_eq(easy.normalize_external_anchor(0, 5), 0, "anchor 空输入")
assert_eq(easy.normalize_external_anchor(5, 0), 0, "anchor overlap 关")
assert_eq(easy.normalize_external_anchor(3, 5), 3, "anchor 少于上限")
assert_eq(easy.normalize_external_anchor(9, 5), 5, "anchor 截到上限")
assert_eq(easy.normalize_external_anchor(None, 5), 0, "anchor None")
assert_eq(easy.chunk_discard_head(0, 5, 0), 0, "无外锚首段不丢")
assert_eq(easy.chunk_discard_head(0, 5, 5), 5, "外锚首段丢锚帧")
assert_eq(easy.chunk_discard_head(0, 5, 3), 3, "外锚首段丢实际锚帧数")
assert_eq(easy.chunk_discard_head(1, 5, 0), 5, "内部衔接段丢重叠")
assert_eq(easy.chunk_discard_head(1, 5, 5), 5, "内部衔接段丢重叠（有外锚）")
assert_eq(easy.chunk_discard_head(2, 0, 5), 0, "overlap 关不丢")

# 类型/色板
check("is_reference_pack true", easy.is_reference_pack({"type": "SCAIL2_REFERENCE_PACK"}))
check("is_reference_pack false", not easy.is_reference_pack({"type": "other"}))
check("is_reference_pack non-dict", not easy.is_reference_pack(None))
assert_eq(easy.subject_color(0), (0.0, 0.0, 1.0), "color 0")
assert_eq(easy.subject_color(6), (0.0, 0.0, 1.0), "color wrap")

# 布局
assert_eq(easy.layout_stage_entries([], 64, 64), ([], []), "layout empty")
specs, scales = easy.layout_stage_entries([{"metrics": {"crop_w": 10.0, "crop_h": 10.0, "image_w": 10.0, "image_h": 10.0}}], 64, 64)
assert_eq(specs[0]["max_w"], 0.86, "single subject spec")
assert_eq(scales, [None], "single subject scale")
assert_eq(easy.stage_row_candidates(1, "square"), [[[0]]], "candidates 1")
assert_eq(easy.stage_row_candidates(3, "portrait"), [[[0, 2], [1]]], "candidates 3 portrait")
rows_ok = easy.stage_row_candidates(4, "landscape")
check("landscape 4 candidate rows", any(sorted(r) for r in rows_ok))

# ── 节点模块（stub sfnodes 包结构，使相对导入可解析）──
def stub_pkg(name, path):
    mod = types.ModuleType(name)
    mod.__path__ = [path]
    sys.modules[name] = mod
    return mod


stub_pkg("sfnodes", root)
stub_pkg("sfnodes.sf_utils", os.path.join(root, "sf_utils"))
stub_pkg("sfnodes.nodes", os.path.join(root, "nodes"))
stub_pkg("sfnodes.nodes.video", os.path.join(root, "nodes", "video"))

nodes_mod = importlib.import_module("sfnodes.nodes.video.scail2")

CLASS_KEYS = ("SFSCAIL2FitVideo", "SFSCAIL2ReferencePack", "SFSCAIL2ReferenceSAMBuilder", "SFSCAIL2SimpleVideo")
assert_eq(set(nodes_mod.NODE_CLASS_MAPPINGS.keys()), set(CLASS_KEYS), "node class mapping keys")
assert_eq(set(nodes_mod.NODE_DISPLAY_NAME_MAPPINGS.keys()), set(CLASS_KEYS), "display mapping keys")
for key in CLASS_KEYS:
    cls = nodes_mod.NODE_CLASS_MAPPINGS[key]
    check(f"{key} CATEGORY", cls.CATEGORY == "sfnodes/video")
    check(f"{key} DESCRIPTION", bool(cls.DESCRIPTION))
    check(f"{key} INPUT_TYPES", isinstance(cls.INPUT_TYPES(), dict))
    check(f"{key} display prefix", nodes_mod.NODE_DISPLAY_NAME_MAPPINGS[key].startswith("SF SCAIL-2"))

fit = nodes_mod.SFSCAIL2FitVideo
assert_eq(fit.RETURN_TYPES, ("IMAGE", "INT", "INT", "STRING"), "fit returns")
assert_eq(fit.RETURN_NAMES, ("video", "width", "height", "summary"), "fit return names")
assert_eq(fit.FUNCTION, "fit", "fit function")

pack = nodes_mod.SFSCAIL2ReferencePack
assert_eq(pack.RETURN_TYPES, ("SCAIL2_REFERENCE_PACK",), "pack returns")
assert_eq(pack.FUNCTION, "pack", "pack function")
pack_required = pack.INPUT_TYPES()["required"]
assert_eq(set(pack_required.keys()), {"subject_count", "reference_count"}, "pack required")
check("pack optional has scene_image", "scene_image" in pack.INPUT_TYPES()["optional"])
check("pack optional has legacy", "subject_1_image_3" in pack.INPUT_TYPES()["optional"])

simple = nodes_mod.SFSCAIL2SimpleVideo
assert_eq(simple.RETURN_TYPES, ("IMAGE", "STRING"), "simple returns")
required = simple.INPUT_TYPES()["required"]
check("simple has mode combo", required["mode"][0] == ["replacement", "animation"])
check("simple has long_video_mode", required["long_video_mode"][0] == ["chunk", "context_sampling"])
check("simple optional track data", "driving_track_data" in simple.INPUT_TYPES()["optional"])
check("simple optional previous_frames", "previous_frames" in simple.INPUT_TYPES()["optional"])
check("simple previous_frames IMAGE 类型", simple.INPUT_TYPES()["optional"]["previous_frames"][0] == "IMAGE")
check("simple generate 签名含 previous_frames",
      "previous_frames" in inspect.signature(simple.generate).parameters)

# 每个节点的每个输入/选项都必须带非空中文 tooltip
for key, cls in nodes_mod.NODE_CLASS_MAPPINGS.items():
    spec = cls.INPUT_TYPES()
    for group in ("required", "optional"):
        for name, config in spec.get(group, {}).items():
            has_tooltip = isinstance(config, tuple) and len(config) > 1 and isinstance(config[1], dict) and bool(config[1].get("tooltip"))
            check(f"{key}.{name} tooltip", has_tooltip)

# tiled_decode 开关存在且默认关
check("simple has tiled_decode", "tiled_decode" in required)
check("tiled_decode default off", required["tiled_decode"][1].get("default") is False)

# context_schedule / freenoise 选项（对齐原生 WanContextWindowsManual，默认保持现状）
check("simple has context_schedule", required["context_schedule"][0] == list(easy.CONTEXT_SCHEDULES))
check("context_schedule default static", required["context_schedule"][1].get("default") == "standard_static")
check("simple has freenoise", required["freenoise"][0] == "BOOLEAN")
check("freenoise default off", required["freenoise"][1].get("default") is False)
check("simple has context_stride", required["context_stride"][0] == "INT")
check("context_stride default 1", required["context_stride"][1].get("default") == 1)
check("simple has closed_loop", required["closed_loop"][0] == "BOOLEAN")
check("closed_loop default off", required["closed_loop"][1].get("default") is False)

# ── _decode_latent_to_frames：tiled 分块解码分支选择 ──
class FakeTensor:
    def detach(self):
        return self

    def cpu(self):
        return self

    def contiguous(self):
        return self

    def clamp(self, low, high):
        return self


decode_calls = []


class FakeVAEDecode:
    def decode(self, vae, latent):
        decode_calls.append("plain")
        return (FakeTensor(),)


class FakeVAEDecodeTiled:
    def decode(self, vae, latent, tile_size, overlap, temporal_size, temporal_overlap):
        decode_calls.append(("tiled", tile_size, overlap, temporal_size, temporal_overlap))
        return (FakeTensor(),)


nodes_stub = types.ModuleType("nodes")
nodes_stub.VAEDecode = FakeVAEDecode
nodes_stub.VAEDecodeTiled = FakeVAEDecodeTiled
sys.modules["nodes"] = nodes_stub
nodes_mod._empty_cache = lambda *a, **k: None

nodes_mod._decode_latent_to_frames(None, {"samples": None})
assert_eq(decode_calls[-1], "plain", "decode plain")
nodes_mod._decode_latent_to_frames(None, {"samples": None}, True)
assert_eq(decode_calls[-1], ("tiled", 512, 64, 64, 8), "decode tiled args")

# ── apply_scail2_easy_context：schedule / freenoise 透传（stub comfy.context_windows）──
cw = types.ModuleType("comfy.context_windows")


class FakeSchedules:
    STATIC_STANDARD = "standard_static"
    UNIFORM_STANDARD = "standard_uniform"
    UNIFORM_LOOPED = "looped_uniform"
    BATCHED = "batched"


class FakeFuseMethods:
    PYRAMID = "pyramid"


class FakeNamed:
    def __init__(self, name):
        self.name = name


handler_kwargs_log = []


class FakeContextHandler:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        handler_kwargs_log.append(kwargs)

    def get_resized_cond(self, cond_in, x_in, window, device=None):
        return cond_in


prepare_wrapper_calls = []
sampler_wrapper_calls = []
cw.ContextSchedules = FakeSchedules
cw.ContextFuseMethods = FakeFuseMethods
cw.IndexListContextHandler = FakeContextHandler
cw.get_matching_context_schedule = lambda name: FakeNamed(name)
cw.get_matching_fuse_method = lambda name: FakeNamed(name)
cw.create_prepare_sampling_wrapper = lambda model: prepare_wrapper_calls.append(model)
cw.create_sampler_sample_wrapper = lambda model: sampler_wrapper_calls.append(model)

comfy_stub = types.ModuleType("comfy")
comfy_stub.context_windows = cw
sys.modules["comfy"] = comfy_stub
sys.modules["comfy.context_windows"] = cw

context_mod = importlib.import_module("sfnodes.sf_utils.scail2_context")


class FakeModel:
    def __init__(self):
        self.model_options = {}

    def clone(self):
        return self


default_model = FakeModel()
_, default_summary = context_mod.apply_scail2_easy_context(default_model, 81, 20)
assert_eq(default_summary["context_schedule"], "standard_static", "context default schedule")
assert_eq(default_summary["freenoise"], False, "context default freenoise")
assert_eq(default_summary["context_stride"], 1, "context default stride")
assert_eq(default_summary["closed_loop"], False, "context default closed_loop")
assert_eq(default_summary["context_latent_frames"], 21, "context latent length")
assert_eq(default_summary["context_overlap_latent_frames"], 5, "context latent overlap")
assert_eq(handler_kwargs_log[-1]["context_schedule"].name, "standard_static", "handler schedule default")
assert_eq(handler_kwargs_log[-1]["fuse_method"].name, "pyramid", "handler fuse default")
assert_eq(handler_kwargs_log[-1]["freenoise"], False, "handler freenoise default")
assert_eq(handler_kwargs_log[-1]["context_stride"], 1, "handler stride")
assert_eq(handler_kwargs_log[-1]["closed_loop"], False, "handler closed_loop")
assert_eq(handler_kwargs_log[-1]["dim"], 2, "handler dim")
assert_eq(sampler_wrapper_calls, [], "no sampler wrapper without freenoise")

uniform_model = FakeModel()
_, uniform_summary = context_mod.apply_scail2_easy_context(
    uniform_model, 81, 20, context_schedule="standard_uniform", freenoise=True
)
assert_eq(uniform_summary["context_schedule"], "standard_uniform", "context uniform schedule")
assert_eq(uniform_summary["freenoise"], True, "context freenoise on")
assert_eq(handler_kwargs_log[-1]["context_schedule"].name, "standard_uniform", "handler schedule uniform")
assert_eq(handler_kwargs_log[-1]["freenoise"], True, "handler freenoise on")
assert_eq(len(prepare_wrapper_calls), 2, "prepare wrapper per call")
assert_eq(len(sampler_wrapper_calls), 1, "sampler wrapper with freenoise")
assert_eq(uniform_model.model_options["context_handler"].kwargs["freenoise"], True, "handler attached")

looped_model = FakeModel()
_, looped_summary = context_mod.apply_scail2_easy_context(
    looped_model, 81, 20, context_schedule="looped_uniform", context_stride=2, closed_loop=True
)
assert_eq(looped_summary["context_schedule"], "looped_uniform", "context looped schedule")
assert_eq(looped_summary["context_stride"], 2, "context stride passthrough")
assert_eq(looped_summary["closed_loop"], True, "context closed_loop passthrough")
assert_eq(handler_kwargs_log[-1]["context_stride"], 2, "handler stride looped")
assert_eq(handler_kwargs_log[-1]["closed_loop"], True, "handler closed_loop looped")
assert_eq(len(prepare_wrapper_calls), 3, "prepare wrapper per call (looped)")

context_mod.apply_scail2_easy_context(FakeModel(), 81, 20, context_stride=0)
assert_eq(handler_kwargs_log[-1]["context_stride"], 1, "handler stride clamped to 1")

try:
    context_mod.apply_scail2_easy_context(FakeModel(), 81, 81)
    check("overlap guard raises", False)
except ValueError:
    check("overlap guard raises", True)

# ── _set_scail_single_reference_conditioning：正/负向共享同一 reference_latents ──
comfy_utils = types.ModuleType("comfy.utils")
comfy_utils.common_upscale = lambda tensor, width, height, mode, crop: tensor
comfy_stub.utils = comfy_utils
sys.modules["comfy.utils"] = comfy_utils

cond_set_calls = []

node_helpers_stub = types.ModuleType("node_helpers")
node_helpers_stub.conditioning_set_values = lambda cond, values, append=False: (
    cond_set_calls.append((cond, values, append)) or cond
)
sys.modules["node_helpers"] = node_helpers_stub


class FakeImageTensor:
    def __getitem__(self, item):
        return self

    def movedim(self, a, b):
        return self


ref_latent_sentinel = object()


class FakeVAE:
    def encode(self, pixels):
        return ref_latent_sentinel


positive_cond = {"positive": True}
negative_cond = {"negative": True}
nodes_mod._set_scail_single_reference_conditioning(
    positive=positive_cond,
    negative=negative_cond,
    vae=FakeVAE(),
    latent=None,
    reference_image=FakeImageTensor(),
    reference_image_mask=None,
    width=64,
    height=64,
    replacement_mode=False,
)
positive_refs = [values for cond, values, _ in cond_set_calls if cond is positive_cond]
negative_refs = [values for cond, values, _ in cond_set_calls if cond is negative_cond]
assert_eq(positive_refs, [{"reference_latents": [ref_latent_sentinel]}], "positive ref latents")
assert_eq(negative_refs, positive_refs, "negative shares positive reference_latents")

# ── 根 __init__.py 注册键（4 键各出现两次：类映射 + 显示名映射）──
with open(os.path.join(root, "__init__.py"), encoding="utf-8") as fh:
    root_src = fh.read()
for key in CLASS_KEYS:
    assert_eq(root_src.count(f'"{key}"'), 2, f"root registration {key}")

if failures:
    print(f"\n{len(failures)} FAILURES")
    sys.exit(1)
print("\ntest_scail2: all assertions passed")
