# SFWanMotionBoost 后端测试（Node/Python 直接运行：python3 tests/test_wan_motion_boost.py）
# 覆盖：
#   - sf_utils/wan_motion_boost.py 纯逻辑（numpy 代理 torch）：
#     placeholder_start 结构识别（单帧/多帧/全条件/全占位/非后段连续/无 mask/短序列）、
#     boost_concat_latent 逐通道保均值缩放/多帧基准=最后条件帧/amp<=1 直通/
#     clamp 与 color_protect 均值恢复/Painter 原版跨通道模式/latent_clamp=0/
#     非 5D 拒绝/不改原张量、
#     boost_conditioning 同张量去重/无 concat 条目直通/不改原 conditioning
#   - nodes/video/wan_motion_boost.py 节点结构（CATEGORY/RETURN_TYPES/RETURN_NAMES/
#     FUNCTION/DESCRIPTION、INPUT_TYPES 默认值与 tooltip）与 execute
#     （委托原生 WanImageToVideo 的参数透传、amp=1/无 start_image/非 Wan model 直通）
#   - 根 __init__.py 注册键一致
# mock：torch（numpy 代理）/ nodes / comfy.model_base / comfy_extras.nodes_wan
import importlib.util
import os
import sys
import types

import numpy as np

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

failures = []


def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")


# ── mock torch（numpy 代理：仅 helper 用到的三个模块级函数）──
torch = types.ModuleType("torch")
torch.mean = lambda x, dim=None, keepdim=False: np.mean(np.asarray(x), axis=dim, keepdims=keepdim)
torch.clamp = lambda x, lo, hi: np.clip(np.asarray(x), lo, hi)
torch.cat = lambda xs, dim=0: np.concatenate([np.asarray(x) for x in xs], axis=dim)
sys.modules["torch"] = torch

# ── mock nodes（节点顶层 from nodes import MAX_RESOLUTION）──
nodes_mod = types.ModuleType("nodes")
nodes_mod.MAX_RESOLUTION = 16384
sys.modules["nodes"] = nodes_mod

# ── mock comfy.model_base（_is_wan_model 类型校验）──
class _WAN21:
    pass


comfy_mod = types.ModuleType("comfy")
comfy_mod.__path__ = []
model_base = types.ModuleType("comfy.model_base")
model_base.WAN21 = _WAN21
comfy_mod.model_base = model_base  # 模拟真实 import 后父包属性
sys.modules["comfy"] = comfy_mod
sys.modules["comfy.model_base"] = model_base


# ── 注册 sfnodes 包结构（相对导入可解析）──
def stub_pkg(name, path):
    mod = types.ModuleType(name)
    mod.__path__ = [path]
    sys.modules[name] = mod
    return mod


stub_pkg("sfnodes", root)
stub_pkg("sfnodes.sf_utils", os.path.join(root, "sf_utils"))
stub_pkg("sfnodes.nodes", os.path.join(root, "nodes"))
stub_pkg("sfnodes.nodes.video", os.path.join(root, "nodes", "video"))

helper = importlib.import_module("sfnodes.sf_utils.wan_motion_boost")
placeholder_start = helper.placeholder_start
boost_concat_latent = helper.boost_concat_latent
boost_conditioning = helper.boost_conditioning


def mask_for(total, conditioned):
    m = np.ones((1, 1, total, 2, 2), dtype=np.float32)
    m[:, :, :conditioned] = 0.0
    return m


# ── 1. placeholder_start ──
check("无 mask 短序列返回 None", placeholder_start(None, 1) is None)
check("无 mask 单帧假设 start=1", placeholder_start(None, 5) == 1)
check("单帧条件 mask -> start=1", placeholder_start(mask_for(5, 1), 5) == 1)
check("多帧条件 mask -> start=2", placeholder_start(mask_for(5, 2), 5) == 2)
check("全条件帧返回 None", placeholder_start(mask_for(5, 5), 5) is None)
check("全占位帧（start=0）返回 None", placeholder_start(mask_for(5, 0), 5) is None)
check("单帧序列返回 None", placeholder_start(mask_for(1, 0), 1) is None)
bad = mask_for(5, 2)
bad[:, :, 0] = 1.0  # 条件段被打断：首帧变占位 → 非后段连续
check("非后段连续模式返回 None", placeholder_start(bad, 5) is None)
check("4D mask 回退单帧假设", placeholder_start(np.zeros((1, 5, 2, 2), dtype=np.float32), 5) == 1)


# ── 2. boost_concat_latent ──
def make_latent():
    """[1,2,3,2,2]：帧0=条件帧，帧1/2 空间上有差异（有结构可放大）。"""
    x = np.zeros((1, 2, 3, 2, 2), dtype=np.float32)
    x[0, 0, 0] = [[1.0, 1.0], [1.0, 1.0]]
    x[0, 1, 0] = [[2.0, 2.0], [2.0, 2.0]]
    x[0, 0, 1] = [[0.5, 0.6], [0.7, 0.8]]
    x[0, 1, 1] = [[1.0, 1.2], [1.4, 1.6]]
    x[0, 0, 2] = [[1.5, 1.6], [1.7, 1.8]]
    x[0, 1, 2] = [[3.0, 3.2], [3.4, 3.6]]
    return x


base = make_latent()
orig = base.copy()
mask = mask_for(3, 1)

check("amp=1.0 原样返回同一对象", boost_concat_latent(base, mask, 1.0, 6.0, True) is base)
check("非 5D 原样返回", boost_concat_latent(np.zeros((2, 3)), mask, 1.5, 6.0, True).shape == (2, 3))

out = boost_concat_latent(base, mask, 2.0, 6.0, True)
check("条件帧逐元素不变", np.allclose(out[:, :, 0], orig[:, :, 0]))
check("不改原张量", np.array_equal(base, orig))
check("形状不变", out.shape == orig.shape)
check("返回值是新对象", out is not base)

# 逐帧逐通道均值严格保持（color_protect 默认）
for t in (1, 2):
    for c in (0, 1):
        check(f"帧{t}通道{c}均值保持",
              np.allclose(out[:, c, t].mean(), orig[:, c, t].mean(), atol=1e-6))
# 空间结构被放大（帧1通道0 的 0.8 相对均值 0.65 偏差 +0.15 → ×2）
check("空间结构偏差被放大",
      abs(out[0, 0, 1, 1, 1] - orig[:, 0, 1].mean()) > abs(orig[0, 0, 1, 1, 1] - orig[:, 0, 1].mean()))

# 独立按公式复算（逐通道均值版本）
rest = orig[:, :, 1:]
b = orig[:, :, 0:1]
diff = rest - b
dm = diff.mean(axis=(3, 4), keepdims=True)
expected = b + (diff - dm) * 2.0 + dm
check("缩放数学与公式一致", np.allclose(out[:, :, 1:], expected, atol=1e-6))

# PainterI2V 原版（跨通道均值）可复现
out_legacy = boost_concat_latent(base, mask, 2.0, 6.0, False)
legacy_dm = diff.mean(axis=(1, 3, 4), keepdims=True)
check("color_protect=False 为跨通道均值公式",
      np.allclose(out_legacy[:, :, 1:], b + (diff - legacy_dm) * 2.0 + legacy_dm, atol=1e-6))

# clamp 截断 -> 逐通道均值仍被恢复；关闭保护则漂移
big = make_latent() * 10.0
big_orig = big.copy()
out_clamped = boost_concat_latent(big, mask, 2.0, 2.0, True)
for t in (1, 2):
    for c in (0, 1):
        check(f"clamp 后帧{t}通道{c}均值恢复",
              np.allclose(out_clamped[:, c, t].mean(), big_orig[:, c, t].mean(), atol=1e-5))
out_unprotected = boost_concat_latent(big, mask, 2.0, 2.0, False)
check("color_protect=False 保留截断漂移",
      not np.allclose(out_unprotected[:, 0, 2].mean(), big_orig[:, 0, 2].mean(), atol=1e-4))

# latent_clamp=0 不截断（值与公式一致）
no_clamp = boost_concat_latent(big, mask, 2.0, 0.0, True)
formula = big_orig[:, :, 0:1] + (big_orig[:, :, 1:] - big_orig[:, :, 0:1]
                                 - (big_orig[:, :, 1:] - big_orig[:, :, 0:1]).mean(axis=(3, 4), keepdims=True)) * 2.0 \
    + (big_orig[:, :, 1:] - big_orig[:, :, 0:1]).mean(axis=(3, 4), keepdims=True)
check("latent_clamp=0 走未截断公式", np.allclose(no_clamp[:, :, 1:], formula, atol=1e-4))

# 多帧：基准=最后条件帧（帧1），帧0/1 不变、帧2 相对帧1 放大
multi = make_latent()
multi[0, 0, 2] = [[1.5, 2.0], [2.5, 3.0]]  # 与帧1 有非均匀差异（可放大）
multi_orig = multi.copy()
out_multi = boost_concat_latent(multi, mask_for(3, 2), 2.0, 6.0, True)
check("多帧：前两条件帧不变", np.array_equal(out_multi[:, :, :2], multi_orig[:, :, :2]))
dev_orig = np.abs(multi_orig[0, 0, 2] - multi_orig[0, 0, 1].mean()).max()
dev_boost = np.abs(out_multi[0, 0, 2] - multi_orig[0, 0, 1].mean()).max()
check("多帧：占位帧相对最后条件帧放大", dev_boost > dev_orig)
check("多帧：占位帧通道均值保持",
      np.allclose(out_multi[:, 0, 2].mean(), multi_orig[:, 0, 2].mean(), atol=1e-6))

# ── 3. boost_conditioning ──
shared = make_latent()
ctx = {"concat_latent_image": shared, "concat_mask": mask_for(3, 1), "other": 1}
cond = [[np.zeros((1, 4)), dict(ctx)], [np.ones((1, 4)), dict(ctx)], [np.zeros((1, 4)), {"no_concat": True}]]
out_cond = boost_conditioning(cond, 2.0, 6.0, True)
check("共享张量去重：两条同对象", out_cond[0][1]["concat_latent_image"] is out_cond[1][1]["concat_latent_image"])
check("增强张量为新对象", out_cond[0][1]["concat_latent_image"] is not shared)
check("原 conditioning 未改", cond[0][1]["concat_latent_image"] is shared and cond[1][1]["concat_latent_image"] is shared)
check("原 dict 未改", cond[0][1] == ctx)
check("无 concat 条目直通且 dict 为新拷贝", out_cond[2][1] is not cond[2][1] and out_cond[2][1] == {"no_concat": True})
check("文本张量保留", out_cond[0][0] is cond[0][0])
check("空 conditioning 直通", boost_conditioning([], 2.0, 6.0, True) == [])

# ── 4. 节点结构与 execute ──
calls = []
fake_outputs = []


class _FakeNodeOutput:
    def __init__(self, *args):
        self.result = args


class _FakeWanImageToVideo:
    @classmethod
    def execute(cls, **kwargs):
        calls.append(kwargs)
        total = ((int(kwargs["length"]) - 1) // 4) + 1
        if kwargs.get("start_image") is None:
            concat, concat_mask = None, None
        else:
            concat = make_latent()[:, :, :total]
            concat_mask = mask_for(total, 1)
        pos, neg = [], []
        for text in (np.zeros((1, 4)), np.ones((1, 4))):
            ctx2 = {}
            if concat is not None:
                ctx2 = {"concat_latent_image": concat, "concat_mask": concat_mask}
            pos.append([text, dict(ctx2)])
            neg.append([text, dict(ctx2)])
        out = _FakeNodeOutput(pos, neg, {"samples": np.zeros((1, 16, total, 2, 2))})
        fake_outputs.append(out)
        return out


comfy_extras = types.ModuleType("comfy_extras")
comfy_extras.__path__ = []
wan_mod = types.ModuleType("comfy_extras.nodes_wan")
wan_mod.WanImageToVideo = _FakeWanImageToVideo
sys.modules["comfy_extras"] = comfy_extras
sys.modules["comfy_extras.nodes_wan"] = wan_mod

node_mod = importlib.import_module("sfnodes.nodes.video.wan_motion_boost")
SFWanMotionBoost = node_mod.SFWanMotionBoost
node = SFWanMotionBoost()

check("CATEGORY", SFWanMotionBoost.CATEGORY == "sfnodes/video")
check("RETURN_TYPES", SFWanMotionBoost.RETURN_TYPES == ("CONDITIONING", "CONDITIONING", "LATENT"))
check("RETURN_NAMES", SFWanMotionBoost.RETURN_NAMES == ("positive", "negative", "latent"))
check("FUNCTION", SFWanMotionBoost.FUNCTION == "execute")
check("DESCRIPTION", bool(SFWanMotionBoost.DESCRIPTION))

spec = SFWanMotionBoost.INPUT_TYPES()
check("required 键", set(spec["required"]) == {
    "positive", "negative", "vae", "width", "height", "length", "batch_size",
    "motion_amplitude", "color_protect", "latent_clamp"})
check("optional 键", set(spec["optional"]) == {"model", "start_image", "clip_vision_output"})
check("motion_amplitude 默认 1.15", spec["required"]["motion_amplitude"][1]["default"] == 1.15)
check("color_protect 默认开", spec["required"]["color_protect"][1]["default"] is True)
check("latent_clamp 默认 6.0", spec["required"]["latent_clamp"][1]["default"] == 6.0)
check("latent_clamp 0 合法（关闭）", spec["required"]["latent_clamp"][1]["min"] == 0.0)
check("width 上限 MAX_RESOLUTION", spec["required"]["width"][1]["max"] == 16384)
check("全部输入带 tooltip", all(
    cfg[1].get("tooltip") for group in spec.values() for cfg in group.values()))

# execute：委托参数透传 + 增强
start = np.zeros((1, 4, 4, 3), dtype=np.float32)
pos = [[np.zeros((1, 4)), {}]]
neg = [[np.ones((1, 4)), {}]]
p_out, n_out, lat = node.execute(pos, neg, vae=object(), width=832, height=480,
                                 length=9, batch_size=1, motion_amplitude=1.5,
                                 color_protect=True, latent_clamp=6.0, start_image=start)
check("原生调用一次", len(calls) == 1)
check("委托参数透传", calls[0]["width"] == 832 and calls[0]["height"] == 480
      and calls[0]["length"] == 9 and calls[0]["start_image"] is start)
check("positive 被增强", p_out[0][1]["concat_latent_image"] is not fake_outputs[-1].result[0][0][1]["concat_latent_image"])
check("增强后逐通道均值保持", np.allclose(
    p_out[0][1]["concat_latent_image"][:, 0, 1].mean(),
    fake_outputs[-1].result[0][0][1]["concat_latent_image"][:, 0, 1].mean(), atol=1e-6))
check("negative 同样增强", "concat_latent_image" in n_out[0][1])
check("latent 透传", lat is fake_outputs[-1].result[2])

# execute：amp=1.0 -> 原样返回原生结果（同一对象）
p_amp1, _, _ = node.execute(pos, neg, vae=object(), width=64, height=64, length=9,
                            batch_size=1, motion_amplitude=1.0, start_image=start)
check("amp=1.0 直通原生对象", p_amp1 is fake_outputs[-1].result[0])

# execute：无 start_image -> 直通（原生无 concat）
p_noimg, _, _ = node.execute(pos, neg, vae=object(), width=64, height=64, length=9,
                             batch_size=1, motion_amplitude=1.5, start_image=None)
check("无 start_image 直通原生", p_noimg is fake_outputs[-1].result[0])
check("无 start_image 无 concat", p_noimg[0][1].get("concat_latent_image") is None)

# execute：model 类型校验
p_wan, _, _ = node.execute(pos, neg, vae=object(), width=64, height=64, length=9,
                           batch_size=1, motion_amplitude=1.5,
                           model=types.SimpleNamespace(model=_WAN21()), start_image=start)
check("Wan model 正常增强", p_wan[0][1]["concat_latent_image"] is not fake_outputs[-1].result[0][0][1]["concat_latent_image"])
p_nonwan, _, _ = node.execute(pos, neg, vae=object(), width=64, height=64, length=9,
                              batch_size=1, motion_amplitude=1.5,
                              model=types.SimpleNamespace(model=object()), start_image=start)
check("非 Wan model 跳过增强", p_nonwan is fake_outputs[-1].result[0])

# GGUF：ComfyUI-GGUF 的 GGUFModelPatcher.clone 只换 patcher 类，.model 仍是 WAN21


class _FakeGGUFModelPatcher:
    patch_on_device = False

    def __init__(self):
        self.model = _WAN21()

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False):
        pass


p_gguf, _, _ = node.execute(pos, neg, vae=object(), width=64, height=64, length=9,
                            batch_size=1, motion_amplitude=1.5,
                            model=_FakeGGUFModelPatcher(), start_image=start)
check("GGUFModelPatcher（.model=WAN21）正常增强",
      p_gguf[0][1]["concat_latent_image"] is not fake_outputs[-1].result[0][0][1]["concat_latent_image"])

# ── 5. 根 __init__.py 注册键一致 ──
init_src = open(os.path.join(root, "__init__.py"), encoding="utf-8").read()
check("根注册：类映射 + 显示名映射", init_src.count('"SFWanMotionBoost"') == 2)
check("根注册：import 语句", "from .nodes.video.wan_motion_boost import SFWanMotionBoost" in init_src)

# ── 6. 模块导出契约 ──
check("__all__", set(helper.__all__) == {"placeholder_start", "boost_concat_latent", "boost_conditioning"})
check("节点模块 __all__", node_mod.__all__ == ["SFWanMotionBoost"])

if failures:
    print(f"\n{len(failures)} FAILED:")
    for name in failures:
        print(f"  - {name}")
    sys.exit(1)
print("test_wan_motion_boost: all assertions passed")
