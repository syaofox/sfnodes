# SF SCAIL-2 后端测试（Node/Python 直接运行：python3 tests/test_scail2.py）
# 覆盖：
#   - sf_utils/scail2_easy.py 纯逻辑：帧数 4n+1 取整、32 对齐、Fit Video 尺寸策略、
#     色板、多主体拼图布局候选（无 torch 依赖）
#   - nodes/video/scail2.py 四节点结构：CATEGORY/RETURN_TYPES/RETURN_NAMES/FUNCTION/
#     DESCRIPTION、INPUT_TYPES 关键项、模块内双注册字典
#   - 根 __init__.py 注册键一致（4 键各出现两次：类映射 + 显示名映射）
# mock：torch（节点模块顶层 import torch / torch.nn.functional；方法内才用张量）
import importlib.util
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

# Fit Video 尺寸策略
assert_eq(easy.target_size_for_video(480, 832, "512p"), (896, 512), "target 512p")
assert_eq(easy.target_size_for_video(480, 832, "704p"), (1216, 704), "target 704p")
assert_eq(easy.target_size_for_video(100, 100, "custom", 100, 200), (96, 192), "target custom")

# 生成尺寸推断
assert_eq(easy.infer_generation_size(480, 832), (832, 480), "infer size")

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

# 每个节点的每个输入/选项都必须带非空中文 tooltip
for key, cls in nodes_mod.NODE_CLASS_MAPPINGS.items():
    spec = cls.INPUT_TYPES()
    for group in ("required", "optional"):
        for name, config in spec.get(group, {}).items():
            has_tooltip = isinstance(config, tuple) and len(config) > 1 and isinstance(config[1], dict) and bool(config[1].get("tooltip"))
            check(f"{key}.{name} tooltip", has_tooltip)

# ── 根 __init__.py 注册键（4 键各出现两次：类映射 + 显示名映射）──
with open(os.path.join(root, "__init__.py"), encoding="utf-8") as fh:
    root_src = fh.read()
for key in CLASS_KEYS:
    assert_eq(root_src.count(f'"{key}"'), 2, f"root registration {key}")

if failures:
    print(f"\n{len(failures)} FAILURES")
    sys.exit(1)
print("\ntest_scail2: all assertions passed")
