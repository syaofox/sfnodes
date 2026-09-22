# SFTESpeedQwenImage21 模拟测试（Python 直接运行：python3 tests/test_te_speed_qwen.py）
# 无 torch 环境：numpy 版 FakeTensor + mock comfy.patcher_extension.WrappersMP，
# 验证纯逻辑（sf_utils/te_speed_qwen.py）与节点链路（nodes/model/te_speed_qwen.py）：
#   - 复刻常量/窗口 auto/签名 stride/相对变化公式（clamp_min 分母）
#   - _step_index 幂等 + 首次即 1、_active 窗口边界
#   - decide 8 闸门（顺序/标签/计数）、predict_output 外推公式/因子钳制/质量衰减/
#     不可预测回退、record_full 校准（误差→冷却 2 步）/历史滚动、record_hit
#   - 节点：非 QwenImage21 透传、INPUT_TYPES/RETURN_*、wrapper 注册、
#     运行全程（full→full→hit→full…）的模型调用次数与命中统计、bypass、
#     attention=default 不覆盖（可用性缺失时 status 提示回退）
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


# ── numpy 版 FakeTensor（纯逻辑里的张量操作为鸭子类型）──────────────────────
class FakeTensor:
    def __init__(self, data, device="cuda", dtype="float32"):
        self._data = np.asarray(data, dtype=np.float32)
        self.device = device
        self.dtype = dtype

    @property
    def shape(self):
        return tuple(self._data.shape)

    def dim(self):
        return self._data.ndim

    def detach(self):
        return self

    def clone(self):
        return FakeTensor(self._data.copy(), self.device, self.dtype)

    def cpu(self):
        return self

    def reshape(self, *shape):
        return FakeTensor(self._data.reshape(*shape), self.device, self.dtype)

    def __getitem__(self, item):
        return FakeTensor(self._data[item], self.device, self.dtype)

    def abs(self):
        return FakeTensor(np.abs(self._data), self.device, self.dtype)

    __abs__ = abs

    def mean(self):
        return FakeTensor(float(np.mean(self._data)), self.device, self.dtype)

    def clamp_min(self, value):
        return FakeTensor(np.maximum(self._data, value), self.device, self.dtype)

    def item(self):
        return float(self._data.reshape(-1)[0])

    def float(self):
        return self

    def __float__(self):
        return float(self._data.reshape(-1)[0])

    def to(self, device=None, dtype=None):
        return FakeTensor(self._data, device or self.device, dtype or self.dtype)

    def _other(self, other):
        return other._data if isinstance(other, FakeTensor) else other

    def __sub__(self, other):
        return FakeTensor(self._data - self._other(other), self.device, self.dtype)

    def __add__(self, other):
        return FakeTensor(self._data + self._other(other), self.device, self.dtype)

    def __mul__(self, other):
        return FakeTensor(self._data * self._other(other), self.device, self.dtype)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return FakeTensor(self._data / self._other(other), self.device, self.dtype)


# ── import 被测模块（纯逻辑 + 节点，相对导入用假包）────────────────────────
spec = importlib.util.spec_from_file_location(
    "sfnodes.sf_utils.te_speed_qwen", os.path.join(root, "sf_utils", "te_speed_qwen.py"))
pure = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pure
spec.loader.exec_module(pure)

# mock torch / comfy.patcher_extension（节点模块 import 时需要）
torch_mod = types.ModuleType("torch")
torch_mod.cat = lambda tensors, dim=0: FakeTensor(
    np.concatenate([t._data for t in tensors], axis=dim))
sys.modules["torch"] = torch_mod


class FakeWrappersMP:
    OUTER_SAMPLE = "outer_sample"
    DIFFUSION_MODEL = "diffusion_model"


pe_mod = types.ModuleType("comfy.patcher_extension")
pe_mod.WrappersMP = FakeWrappersMP
comfy_mod = types.ModuleType("comfy")
comfy_mod.patcher_extension = pe_mod
sys.modules["comfy"] = comfy_mod
sys.modules["comfy.patcher_extension"] = pe_mod

attn_mod = types.ModuleType("comfy.ldm.modules.attention")
attn_mod.attention_pytorch = lambda *a, **k: ("sdpa", a, k)
attn_mod.attention_sage = lambda *a, **k: ("sage",)
attn_mod.attention_flash = lambda *a, **k: ("flash",)
attn_mod.attention_comfy_kitchen_int8 = lambda *a, **k: ("kitchen",)
attn_mod.SAGE_ATTENTION_IS_AVAILABLE = False
attn_mod.FLASH_ATTENTION_IS_AVAILABLE = False
attn_mod.COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE = False
for name, mod in (("comfy.ldm", types.ModuleType("comfy.ldm")),
                  ("comfy.ldm.modules", types.ModuleType("comfy.ldm.modules")),
                  ("comfy.ldm.modules.attention", attn_mod)):
    mod.__path__ = []
    sys.modules[name] = mod

pkg = types.ModuleType("sfnodes"); pkg.__path__ = [root]
sys.modules["sfnodes"] = pkg
pkg2 = types.ModuleType("sfnodes.nodes"); pkg2.__path__ = [os.path.join(root, "nodes")]
sys.modules["sfnodes.nodes"] = pkg2
pkg3 = types.ModuleType("sfnodes.nodes.model"); pkg3.__path__ = [os.path.join(root, "nodes", "model")]
sys.modules["sfnodes.nodes.model"] = pkg3
pkg4 = types.ModuleType("sfnodes.sf_utils"); pkg4.__path__ = [os.path.join(root, "sf_utils")]
sys.modules["sfnodes.sf_utils"] = pkg4

node_spec = importlib.util.spec_from_file_location(
    "sfnodes.nodes.model.te_speed_qwen", os.path.join(root, "nodes", "model", "te_speed_qwen.py"))
node = importlib.util.module_from_spec(node_spec)
sys.modules[node_spec.name] = node
node_spec.loader.exec_module(node)


# ── 纯逻辑：常量 / 窗口 / 签名 / 相对变化 ───────────────────────────────────
check("threshold 默认 0.06", pure.DEFAULT_THRESHOLD == 0.06)
check("误差上限默认 0.08", pure.DEFAULT_ERROR_LIMIT == 0.08)
check("profile 窗口 0.18-0.86",
      (pure.PROFILE_START_PERCENT, pure.PROFILE_END_PERCENT) == (0.18, 0.86))
check("max_consecutive=1 / refresh=4 / stride=64 / 冷却=2",
      (pure.DEFAULT_MAX_CONSECUTIVE, pure.DEFAULT_REFRESH_INTERVAL,
       pure.DEFAULT_SIGNATURE_STRIDE, pure.DEFAULT_COOLDOWN) == (1, 4, 64, 2))
check("因子钳 -1..2 / 质量下限 0.25",
      (pure.FACTOR_MIN, pure.FACTOR_MAX, pure.QUALITY_FLOOR) == (-1.0, 2.0, 0.25))
check("阈值兜底：0 → 0.018（profile[0]）", pure.resolve_threshold(0.0) == 0.018
      and pure.resolve_threshold(None) == 0.018 and pure.resolve_threshold(0.06) == 0.06)
check("阈值兜底接入 TEPredictor", pure.TEPredictor(threshold=0.0).threshold == 0.018)
check("窗口 auto：0 → profile", pure.resolve_window(0.0, 0.0) == (0.18, 0.86))
check("窗口显式：>0 生效", pure.resolve_window(0.3, 0.7) == (0.3, 0.7))
check("窗口负值回退 auto", pure.resolve_window(-1.0, -1.0) == (0.18, 0.86))
check("timestep_value 张量", pure.timestep_value(FakeTensor([[[0.75]]])) == 0.75)
check("timestep_value dict", pure.timestep_value({"timestep": 1.5}) == 1.5)

sig = pure.make_signature(FakeTensor(list(range(8))), stride=4)
check("signature stride=4 取 0/4", list(sig._data) == [0.0, 4.0])
check("signature None 历史 → inf", pure.relative_change(FakeTensor([1.0]), None) == float("inf"))
check("signature 形状不符 → inf",
      pure.relative_change(FakeTensor([1.0, 2.0]), FakeTensor([1.0])) == float("inf"))
# mean|3-1|=2；denominator=max(mean|1|,1e-6)=1 → 2.0
check("相对变化公式", pure.relative_change(FakeTensor([3.0]), FakeTensor([1.0])) == 2.0)
# 分母 clamp_min：prev 全 0 → 用 1e-6
check("相对变化分母 clamp_min",
      pure.relative_change(FakeTensor([1e-6]), FakeTensor([0.0])) == 1.0)


# ── 纯逻辑：step / 窗口 ────────────────────────────────────────────────────
predictor = pure.TEPredictor()
predictor.begin([0.9, 0.6, 0.3, 0.0])
check("begin: total_steps = len(sigmas) = 4", predictor.total_steps == 4)
check("begin: sample_timesteps 保留", predictor.sample_timesteps == [0.9, 0.6, 0.3, 0.0])
check("_step_index 首次即 1", predictor._step_index(0.9) == 1.0)
check("_step_index 同步幂等", predictor._step_index(0.9) == 1.0)
check("_step_index 换 timestep 递增", predictor._step_index(0.6) == 2.0)
predictor.reset()
predictor.begin([0.9, 0.6, 0.3, 0.0])  # total 4，窗口 0.18-0.86
check("_active: step1 percent=0.25 命中",
      predictor._active(1) is True)
check("_active: step0 窗口外（0 < 0.18）", predictor._active(0) is False)
check("_active: step4 越界（>= total）", predictor._active(4) is False)
windowed = pure.TEPredictor(start_percent=0.5, end_percent=1.0)
windowed.begin([0.9, 0.6, 0.3])  # total 3
check("_active: step1 percent=0.33 < 0.5 窗口外", windowed._active(1) is False)
check("_active: step2 percent=0.67 命中", windowed._active(2) is True)
check("_active: step3 越界", windowed._active(3) is False)


# ── 纯逻辑：闸门顺序 / 预测公式 / 校准 ─────────────────────────────────────
p = pure.TEPredictor(threshold=0.06)
p.begin([0.9, 0.6, 0.3, 0.0])
decision, state, signature = p.decide("uuid-a", FakeTensor([1.0]), 0.9)
check("首步闸门：no_previous_input", decision.reason == "no_previous_input")
check("首步不可预测", decision.predict is False)
check("决策计数累加", p.decision_counts["no_previous_input"] == 1)
check("predict_output 无历史 → None", p.predict_output(state, 0.6) is None)
check("闸门明细 8 项含标签", [label for label, _ in decision.gates] == list(pure.GATE_LABELS))

# 构造历史：两个真实步（同一分支，输入潜变量恒定 → 变化 0）
p2 = pure.TEPredictor(threshold=0.06, verbose=False)
p2.begin([0.9, 0.6, 0.3, 0.0])
for t, out in ((0.9, 11.0), (0.6, 9.0)):
    _, st, sg = p2.decide("u", FakeTensor([6.0]), t)
    p2.record_full(st, sg, FakeTensor([6.0]), FakeTensor([out]), t)
# 第 3 步 t=0.3：历史齐备 + 变化 0 → 全闸门通过
decision3, state3, sig3 = p2.decide("u", FakeTensor([6.0]), 0.3)
check("第 3 步可预测（窗口内/变化 0）", decision3.predict is True)
check("第 3 步 reason=predict", decision3.reason == "predict")
# prev_t=0.6 older_t=0.9 signed span=-0.3；factor=(0.3-0.6)/(-0.3)=1
# delta = 9-11 = -2 → pred = 9 + (-2)*1 = 7（首次预测无误差 → quality=1）
pred3 = p2.predict_output(state3, 0.3)
check("predict_output 线性外推", abs(float(pred3) - 7.0) < 1e-5)
check("predict_output 写入 pending", state3.pending_prediction is pred3)

# 因子钳制：t 远离 prev（比值 35.3 → 钳到 2.0）
_, st3, _ = p2.decide("u", FakeTensor([6.0]), -10.0)
pred = p2.predict_output(st3, -10.0)
check("因子钳制上限 2.0", abs(float(pred) - (9.0 + (-2.0) * 2.0)) < 1e-5)

# 质量衰减 / 校准：误差超限 → cooldown 置 2 并当步 -1（原版顺序）
p4 = pure.TEPredictor(predictor_error_limit=0.1, verbose=False)
p4.begin([0.9, 0.6, 0.3, 0.0, -0.3])
_, st4, sg4 = p4.decide("u", FakeTensor([6.0]), 0.9)
p4.record_full(st4, sg4, FakeTensor([6.0]), FakeTensor([1.0]), 0.9)
_, st4, sg4 = p4.decide("u", FakeTensor([6.0]), 0.6)
p4.record_full(st4, sg4, FakeTensor([6.0]), FakeTensor([1.0]), 0.6)
# 手工注入偏差极大的 pending（模拟上一次预测），用同一 shape 的真实输出校准
st4.pending_prediction = FakeTensor([100.0])
_, st4, sg4 = p4.decide("u", FakeTensor([6.0]), 0.3)
p4.record_full(st4, sg4, FakeTensor([6.0]), FakeTensor([1.0]), 0.3)
# err = mean|100-1| / clamp_min(mean|1|) = 99 → 超限 → cooldown 2 → 当步 -1
check("校准：误差记录", abs(p4.prediction_errors[-1] - 99.0) < 1e-4)
check("校准：超限冷却（置 2 当步 -1）", st4.prediction_cooldown == 1.0)
check("校准：超限后 pending 清空", st4.pending_prediction is None)
check("校准：error 保留最近值", st4.prediction_error == p4.prediction_errors[-1] and st4.prediction_error > 0.1)
next_decision, st4, sg4 = p4.decide("u", FakeTensor([6.0]), 0.0)
check("冷却中闸门：prediction_cooldown", next_decision.reason == "prediction_cooldown")
p4.record_full(st4, sg4, FakeTensor([6.0]), FakeTensor([1.0]), 0.0)
check("校准：下次真实步冷却归零", st4.prediction_cooldown == 0.0)

# 质量衰减参与外推：prediction_error 未超限 → factor 乘 clamp(1-err/limit, 0.25, 1)
p4b = pure.TEPredictor(predictor_error_limit=0.2, verbose=False)
p4b.begin([0.9, 0.6, 0.3, 0.0])
for t, out in ((0.9, 11.0), (0.6, 9.0)):
    _, stb, sgb = p4b.decide("u", FakeTensor([6.0]), t)
    p4b.record_full(stb, sgb, FakeTensor([6.0]), FakeTensor([out]), t)
stb.prediction_error = 0.1  # quality = 1 - 0.1/0.2 = 0.5
_, stb, _ = p4b.decide("u", FakeTensor([6.0]), 0.3)
pred_damped = p4b.predict_output(stb, 0.3)
check("质量衰减参与外推", abs(float(pred_damped) - (9.0 + (-2.0) * 1.0 * 0.5)) < 1e-5)
stb.prediction_error = 10.0  # quality 钳到下限 0.25
_, stb, _ = p4b.decide("u", FakeTensor([6.0]), 0.3)
check("质量下限 0.25", abs(float(p4b.predict_output(stb, 0.3)) - (9.0 + (-2.0) * 1.0 * 0.25)) < 1e-5)

# record_hit：只动指纹/计数，真实输出历史保持
p5 = pure.TEPredictor(verbose=False)
p5.begin([0.9, 0.6, 0.3, 0.0])
for t, out in ((0.9, 11.0), (0.6, 9.0)):
    _, st5, sg5 = p5.decide("u", FakeTensor([6.0]), t)
    p5.record_full(st5, sg5, FakeTensor([6.0]), FakeTensor([out]), t)
prev_output, prev_older = st5.previous_output, st5.older_output
p5.record_hit(st5, sg5)
check("record_hit: consecutive=1", st5.consecutive == 1.0)
check("record_hit: age=1", st5.age == 1.0)
check("record_hit: hits 累加", p5.hits == 1.0 and p5.predicted_steps == 1.0)
check("record_hit: 真实输出历史不动",
      st5.previous_output is prev_output and st5.older_output is prev_older)
check("连续命中闸门：consecutive_limit",
      p5.decide("u", FakeTensor([6.0]), 0.3)[0].reason == "consecutive_limit")

# refresh_interval 闸门（age >= 4）
p6 = pure.TEPredictor(verbose=False)
p6.begin([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
for t, out in ((1.0, 11.0), (0.8, 9.0)):
    _, st6, sg6 = p6.decide("u", FakeTensor([6.0]), t)
    p6.record_full(st6, sg6, FakeTensor([6.0]), FakeTensor([out]), t)
st6.age = 4.0
check("刷新闸门：refresh_due", p6.decide("u", FakeTensor([6.0]), 0.6)[0].reason == "refresh_due")

# shape_changed 闸门（第三步仍在窗口内 → 需 4 个 sigma）
p7 = pure.TEPredictor(verbose=False)
p7.begin([0.9, 0.6, 0.3, 0.0])
for t, out in ((0.9, 11.0), (0.6, 9.0)):
    _, st7, sg7 = p7.decide("u", FakeTensor([[6.0, 7.0]]), t)
    p7.record_full(st7, sg7, FakeTensor([[6.0, 7.0]]), FakeTensor([[out, out]]), t)
check("shape 闸门：形状变化", p7.decide("u", FakeTensor([6.0]), 0.3)[0].reason == "shape_changed")

# latent_change 闸门：变化 > 阈值
p8 = pure.TEPredictor(threshold=0.06, verbose=False)
p8.begin([0.9, 0.6, 0.3, 0.0])
for t, cur, out in ((0.9, 2.0, 11.0), (0.6, 2.0, 9.0)):
    _, st8, sg8 = p8.decide("u", FakeTensor([cur]), t)
    p8.record_full(st8, sg8, FakeTensor([cur]), FakeTensor([out]), t)
# change = mean|3-2| / mean|2| = 0.5 > 0.06
check("变化闸门：latent_change", p8.decide("u", FakeTensor([3.0]), 0.3)[0].reason == "latent_change")

# 统计
avg, maximum, samples = p4.error_stats()
check("error_stats samples", samples >= 1 and maximum >= avg > 0)
check("decision_summary 含标签", "no_previous_input=" in p.decision_summary())


# ── 节点：结构 / 检测 / wrapper 注册 / 全流程 ──────────────────────────────
class FakeTransformer:
    def __init__(self):
        self.value = None


class FakeDiffusionModel:
    def __init__(self):
        self.t = FakeTransformer()


class FakeQwenImage21Transformer(FakeDiffusionModel):
    pass


FakeQwenImage21Transformer.__name__ = "QwenImage21Transformer2DModel"


class FakeBaseModel:
    def __init__(self, transformer):
        self.diffusion_model = transformer  # 对齐 BaseModel.diffusion_model 层级


class FakePatcher:
    def __init__(self, transformer_cls):
        self.model = FakeBaseModel(transformer_cls())
        self.model_options = {}
        self.wrappers = {}

    def clone(self):
        new = FakePatcher.__new__(FakePatcher)
        new.model = self.model
        new.model_options = {k: dict(v) if isinstance(v, dict) else v
                             for k, v in self.model_options.items()}
        new.wrappers = {k: {k1: list(v1) for k1, v1 in v.items()} for k, v in self.wrappers.items()}
        return new

    def add_wrapper_with_key(self, wrapper_type, key, wrapper):
        self.wrappers.setdefault(wrapper_type, {}).setdefault(key, []).append(wrapper)


NODE = node.SFTESpeedQwenImage21
required = NODE.INPUT_TYPES()["required"]
check("INPUT_TYPES: model", required["model"][0] == "MODEL")
check("INPUT_TYPES: threshold 默认 0.06", required["reuse_threshold"][1]["default"] == 0.06)
check("INPUT_TYPES: 窗口默认 auto(0)", required["start_percent"][1]["default"] == 0.0
      and required["end_percent"][1]["default"] == 0.0)
check("INPUT_TYPES: error_limit 默认 0.08", required["predictor_error_limit"][1]["default"] == 0.08)
check("INPUT_TYPES: attention 默认 default", required["attention"][1]["default"] == "default")
check("INPUT_TYPES: attention 档位", list(required["attention"][0]) == ["default", "sdpa", "sageattn",
                                                                        "flashattn", "kitchen_int8"])
check("INPUT_TYPES: 无 step_cache 占位", "step_cache" not in required)
check("RETURN_TYPES/RETURN_NAMES", NODE.RETURN_TYPES == ("MODEL", "STRING")
      and NODE.RETURN_NAMES == ("model", "status"))
check("FUNCTION/CATEGORY", NODE.FUNCTION == "patch" and NODE.CATEGORY == "sfnodes/model")

# 非 QwenImage21 → 透传
class FakeOtherTransformer(FakeDiffusionModel):
    pass


FakeOtherTransformer.__name__ = "FluxTransformer2DModel"
patcher = FakePatcher(FakeOtherTransformer)
out, status = NODE().patch(patcher, attention="default", verbose=True)
check("非目标模型：原样透传", out is patcher and "not detected" in status)

# QwenImage21 → 挂 cache + 两个 wrapper
patcher = FakePatcher(FakeQwenImage21Transformer)
patched, status = NODE().patch(patcher, reuse_threshold=0.05, start_percent=0.2,
                               end_percent=0.8, predictor_error_limit=0.1,
                               attention="default", verbose=True)
cache = patched.model_options["transformer_options"][node._CACHE_KEY]
check("cache 类型/参数", isinstance(cache, pure.TEPredictor) and cache.threshold == 0.05
      and (cache.start_percent, cache.end_percent) == (0.2, 0.8))
check("wrapper 注册：OUTER_SAMPLE + DIFFUSION_MODEL",
      list(patched.wrappers.get("outer_sample", {}).get(node._CACHE_KEY, []))
      and list(patched.wrappers.get("diffusion_model", {}).get(node._CACHE_KEY, [])))
check("status 摘要", "step_cache=te_predictor" in status and "attention=default" in status
      and "window=0.20-0.80" in status)
check("attention=default 不覆盖",
      "optimized_attention_override" not in patched.model_options["transformer_options"])
check("原 patcher 不被污染（clone）",
      patcher.model_options == {} and patcher.wrappers == {})

# attention=kitchen_int8：核心不可用 → status 提示回退（不注入 override）
patcher = FakePatcher(FakeQwenImage21Transformer)
patched, status = NODE().patch(patcher, attention="kitchen_int8", verbose=False)
check("attention 不可用 → status 回退提示", "unavailable (default used)" in status)
check("attention 不可用 → 不注入 override",
      "optimized_attention_override" not in patched.model_options["transformer_options"])

patcher = FakePatcher(FakeQwenImage21Transformer)
patched, status = NODE().patch(patcher, attention="sageattn", verbose=False)
check("attention 库缺失（sage）→ 回退提示", "unavailable (default used)" in status)

patcher = FakePatcher(FakeQwenImage21Transformer)
patched, status = NODE().patch(patcher, attention="sdpa", verbose=False)
override = patched.model_options["transformer_options"]["optimized_attention_override"]
check("attention=sdpa 注入 override 且 status 记录", "attention=sdpa" in status and callable(override))
check("override 调用选中后端", override(lambda *a, **k: "orig", 1, 2) == ("sdpa", (1, 2), {}))

def _boom(*a, **k):
    raise RuntimeError("boom")


override_boom = node._make_attention_override(_boom, "boom")
check("override 后端异常 → 回退 Comfy 原 attention",
      override_boom(lambda *a, **k: "orig", 1) == "orig")

# ── 全流程：DIFFUSION_MODEL wrapper 逐步调用 ───────────────────────────────
calls = []


def _fake_executor(*args, **kwargs):
    """真实模型：输出 = -2 * 当前 timestep + 1（对 t 线性，外推应精确命中）。"""
    calls.append("model")
    x, timestep = args[0], args[1]
    return FakeTensor(np.full(x.shape, float(timestep) * -2.0 + 1.0))


def run_step(cache, options, x, timestep, uuid="uuid-1"):
    # 生产路径：transformer_options 作为最后一个位置参数（Qwen 模型 forward 传参）
    return node._predictor_diffusion_wrapper(_fake_executor, x, timestep, options)


class FakeExecutor:
    def __init__(self, model_options):
        self.class_obj = type("FakeGuider", (), {"model_options": model_options})()

    def __call__(self, *args, **kwargs):
        return None


patcher = FakePatcher(FakeQwenImage21Transformer)
patched, _status = NODE().patch(patcher, verbose=True)
model_options = patched.model_options
cache = model_options["transformer_options"][node._CACHE_KEY]
options = {"uuids": ("uuid-1",), node._CACHE_KEY: cache}
sigmas = [0.9, 0.6, 0.3, 0.0]

before = len(calls)
node._predictor_sample_wrapper(FakeExecutor(model_options), FakeTensor([0.0]), FakeTensor([0.0]),
                               None, sigmas)
check("OUTER_SAMPLE: begin total_steps=len(sigmas)", cache.total_steps == 4)

STEP = FakeTensor([[[[1.0]]]])  # (1,1,1,1) 四维潜变量
# step1（t=0.9）：无历史 → full
run_step(cache, dict(options), STEP, 0.9)
check("运行 step1 调用模型", len(calls) - before == 1)
# step2（t=0.6）：older 缺失 → full
run_step(cache, dict(options), STEP, 0.6)
check("运行 step2 调用模型", len(calls) - before == 2)
# step3（t=0.3）：历史齐备 + 变化 0 + 窗口内 → 预测，不调模型
out3 = run_step(cache, dict(options), STEP, 0.3)
check("运行 step3 跳过模型", len(calls) - before == 2)
# prev=o(0.6)=-0.2 older=o(0.9)=-0.8 delta=0.6 factor=1 → pred=0.4 = 真实 o(0.3)
check("运行 step3 输出=外推值", abs(float(out3) - 0.4) < 1e-5)
check("运行 step3 命中计数", cache.hits == 1.0 and cache.predicted_steps == 1.0)
# step4（t=0.0）：窗口外（step==total）→ full；用本步预测校准（线性外推 → 误差 0）
run_step(cache, dict(options), STEP, 0.0)
check("运行 step4 调用模型（窗口外）", len(calls) - before == 3)
check("运行 step4 记录校准误差", len(cache.prediction_errors) == 1
      and abs(cache.prediction_errors[0]) < 1e-5)
check("运行 step4 计数", cache.full_steps == 3.0
      and cache.states["uuid-1"].prediction_cooldown == 0.0)
# step5（t=-0.3）：步序越界 → full
run_step(cache, dict(options), STEP, -0.3)
check("运行 step5 越界 → full", len(calls) - before == 4)
check("diffusion_calls 计数", cache.diffusion_calls == 5.0)

# bypass：ndim != 4 / uuids 为空
before = len(calls)
run_step(cache, dict(options), FakeTensor([1.0, 2.0]), 0.3)
check("bypass: ndim!=4 调模型", len(calls) - before == 1)
run_step(cache, {"uuids": (), node._CACHE_KEY: cache}, FakeTensor([[1.0]]), 0.3)
check("bypass: uuids 空调模型", len(calls) - before == 2)
# 无 cache（未挂本节点）→ 直通
run_step(None, {"uuids": ("u",)}, FakeTensor([[1.0]]), 0.3)
check("无 cache 直通", len(calls) - before == 3)

# 多分支（CFG：cond+uncond 两个 uuid）
cache2 = pure.TEPredictor(verbose=False)
cache2.begin([0.9, 0.6, 0.3, 0.0])
opts2 = {"uuids": ("cond", "uncond"), node._CACHE_KEY: cache2}
X2 = FakeTensor(np.ones((2, 1, 1, 1)))
for t in (0.9, 0.6):
    node._predictor_diffusion_wrapper(_fake_executor, X2, t, opts2)
before2 = len(calls)
out = node._predictor_diffusion_wrapper(_fake_executor, X2, 0.3, opts2)
check("多分支：两分支同时预测 → 不调模型", len(calls) - before2 == 0)
check("多分支：预测 shape 原样", out.shape == (2, 1, 1, 1))
check("多分支：两分支各自 state", set(cache2.states.keys()) == {"cond", "uncond"})
check("多分支：命中计数按分支", cache2.hits == 2.0)

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("all checks passed")
