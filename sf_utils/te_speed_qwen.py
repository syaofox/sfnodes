"""TE-Speed Qwen Image 2.1 步骤预测器纯逻辑（复刻 tl2012tl/TE-Speed-QwenImage21 v1.0）。

干净室复刻：原插件只发 Windows Cython `.pyd`（无源码）。本模块的机制、参数与
判据全部来自对 `nodes.pyd` 的逆向（PE 字符串表/常量池/方法表 + 逐函数反汇编）：
阈值 0.06、窗口 0.18~0.86、max_consecutive=1、refresh_interval=4、
signature_stride=64、误差上限 0.08、冷却 2 步、外推因子钳 [-1, 2]、
质量衰减下限 0.25、相对量 eps 1e-6 均为原值。

机制（每一步模型调用，按 `transformer_options["uuids"]` 逐分支独立状态）：

1. `signature` = 潜变量展平后每 `signature_stride` 取 1（廉价指纹）；
2. `relative_change` = ``mean|Δsig| / clamp_min(mean|prev_sig|, 1e-6)``；
3. 8 道闸门全过才跳过模型：窗口（`step/total_steps ∈ [start, end]`）、signature
   历史、previous/older 输出齐备、shape 一致、连续命中 < max_consecutive、
   距上次真实步 < refresh_interval、冷却 = 0、latent 变化 ≤ 阈值；
4. 预测：``prev + (prev - older) * clamp(factor, -1, 2) * quality``，其中
   ``factor = (t - prev_t) / |prev_t - older_t|``，
   ``quality = clamp(1 - prediction_error / error_limit, 0.25, 1)``；
5. 真实步顺带校准：``mean|pred - real| / clamp_min(mean|real|, 1e-6)`` 超上限
   → 冷却 `prediction_cooldown` 步。

纯逻辑（不 import torch / comfy）：张量为鸭子类型，只调用 detach/reshape/
abs/mean/clamp_min/clone/item/shape 等；节点侧（nodes/model/te_speed_qwen.py）
负责 model_patcher 包装、注意力 override、日志与统计展示。
"""

import math

# ── 复刻常量（原 pyd 常量池 / 反汇编还原，勿随意改动）───────────────────────
# attention 下拉（原版顺序 kitchen_int8/default/sdpa/sageattn/flashattn；本包默认 default）
ATTENTION_MODES = ("default", "sdpa", "sageattn", "flashattn", "kitchen_int8")

DEFAULT_THRESHOLD = 0.06
PROFILE_THRESHOLD_FALLBACK = 0.018  # 原版 profile[0]：widget 阈值 <= 0 时回退（逆向推断）
THRESHOLD_MIN = 0.0
THRESHOLD_MAX = 3.0
THRESHOLD_STEP = 0.01

DEFAULT_ERROR_LIMIT = 0.08
ERROR_LIMIT_MIN = 0.01
ERROR_LIMIT_MAX = 1.0
ERROR_LIMIT_STEP = 0.01

# widget 0.0 = auto（用 profile 窗口）
WIDGET_WINDOW_MIN = 0.0
WIDGET_WINDOW_MAX = 1.0
PROFILE_START_PERCENT = 0.18
PROFILE_END_PERCENT = 0.86

DEFAULT_MAX_CONSECUTIVE = 1
DEFAULT_REFRESH_INTERVAL = 4
DEFAULT_SIGNATURE_STRIDE = 64
DEFAULT_COOLDOWN = 2
FACTOR_MIN = -1.0
FACTOR_MAX = 2.0
QUALITY_FLOOR = 0.25
EPS = 1e-06
ERROR_HISTORY_MAX = 64

# 闸门标签（原版 decide 的 checks 名称，日志/summary 逐字对齐）
GATE_OUTSIDE_WINDOW = "outside_window"
GATE_NO_PREVIOUS_INPUT = "no_previous_input"
GATE_HISTORY_INCOMPLETE = "history_incomplete"
GATE_SHAPE_CHANGED = "shape_changed"
GATE_CONSECUTIVE_LIMIT = "consecutive_limit"
GATE_REFRESH_DUE = "refresh_due"
GATE_PREDICTION_COOLDOWN = "prediction_cooldown"
GATE_LATENT_CHANGE = "latent_change"
GATE_PREDICT = "predict"
GATE_LABELS = (
    GATE_OUTSIDE_WINDOW,
    GATE_NO_PREVIOUS_INPUT,
    GATE_HISTORY_INCOMPLETE,
    GATE_SHAPE_CHANGED,
    GATE_CONSECUTIVE_LIMIT,
    GATE_REFRESH_DUE,
    GATE_PREDICTION_COOLDOWN,
    GATE_LATENT_CHANGE,
)


def resolve_threshold(threshold):
    """widget 阈值 <= 0（或 None）→ profile 兜底 0.018（原版 `> 0` 判定）。"""
    return float(threshold) if threshold and threshold > 0 else PROFILE_THRESHOLD_FALLBACK


def resolve_window(start_percent, end_percent):
    """widget 0.0 = auto → profile 窗口（原版 `> 0` 判定，负值同样回退）。"""
    start = start_percent if start_percent and start_percent > 0 else PROFILE_START_PERCENT
    end = end_percent if end_percent and end_percent > 0 else PROFILE_END_PERCENT
    return float(start), float(end)


def timestep_value(timestep):
    """timestep 张量/标量 → float（原版 `timestep.reshape(-1)[0].detach().cpu()`）。"""
    if isinstance(timestep, dict):
        timestep = timestep.get("timestep", timestep)
    if hasattr(timestep, "reshape"):
        return float(timestep.reshape(-1)[0].detach().cpu())
    return float(timestep)


def make_signature(value, stride=DEFAULT_SIGNATURE_STRIDE):
    """廉价指纹：展平后每 `stride` 个元素取 1 并转 fp32。

    原版为 `value.detach().reshape(-1)[::stride].float()`——潜变量常为 bf16，
    指纹比较必须转 fp32，否则相对变化会被 bf16 量化误差干扰。
    """
    stride = int(stride) if stride else 1
    return value.detach().reshape(-1)[::stride].float()


def relative_change(current, previous, eps=EPS):
    """相对变化：mean|Δ| / clamp_min(mean|prev|, eps)；无历史返回 inf。"""
    if previous is None:
        return math.inf
    if hasattr(current, "shape") and hasattr(previous, "shape"):
        if tuple(current.shape) != tuple(previous.shape):
            return math.inf
    denominator = abs(previous).mean().clamp_min(eps)
    return float((abs(current - previous).mean() / denominator).item())


class BranchState:
    """单分支（transformer_options['uuids'] 一项）的预测器状态。"""

    __slots__ = (
        "shape",
        "previous_output",
        "previous_timestep",
        "older_output",
        "older_timestep",
        "pending_prediction",
        "prediction_error",
        "previous",
        "consecutive",
        "age",
        "prediction_cooldown",
    )

    def __init__(self):
        self.shape = None
        self.previous_output = None
        self.previous_timestep = None
        self.older_output = None
        self.older_timestep = None
        self.pending_prediction = None
        self.prediction_error = None
        self.previous = None
        self.consecutive = 0.0
        self.age = 0.0
        self.prediction_cooldown = 0.0


class Decision:
    """decide() 返回：是否可预测 + 失败原因 + 8 闸门明细。"""

    __slots__ = ("predict", "reason", "change", "gates")

    def __init__(self, predict, reason, change, gates):
        self.predict = predict
        self.reason = reason
        self.change = change
        self.gates = gates


class TEPredictor:
    """Qwen Image 2.1 自适应单步输出预测器（逐分支状态 + 误差校准）。"""

    def __init__(
        self,
        threshold=DEFAULT_THRESHOLD,
        max_consecutive=DEFAULT_MAX_CONSECUTIVE,
        start_percent=0.0,
        end_percent=0.0,
        refresh_interval=DEFAULT_REFRESH_INTERVAL,
        signature_stride=DEFAULT_SIGNATURE_STRIDE,
        verbose=False,
        predictor_error_limit=DEFAULT_ERROR_LIMIT,
    ):
        self.threshold = resolve_threshold(threshold)
        self.max_consecutive = float(max_consecutive)
        self.refresh_interval = float(refresh_interval)
        self.signature_stride = int(signature_stride) if signature_stride else 1
        self.verbose = bool(verbose)
        self.predictor_error_limit = float(predictor_error_limit)
        self.start_percent, self.end_percent = resolve_window(start_percent, end_percent)
        self.reset()

    # ── 采样生命周期 ────────────────────────────────────────────────────────
    def reset(self):
        self.states = {}
        self.total_steps = 0
        self.step = 0.0
        self.last_timestep = None
        self.sample_timesteps = []
        self.hits = 0.0
        self.predicted_steps = 0.0
        self.full_steps = 0.0
        self.prediction_errors = []
        self.decision_counts = {}
        self.diffusion_calls = 0.0

    def begin(self, sigmas):
        """一次采样开始：复位并用 sigma 序列确定步数（原版 total_steps = len(sigmas)）。"""
        self.reset()
        try:
            count = len(sigmas)
        except TypeError:
            count = int(sigmas)
        self.total_steps = max(int(count), 1)
        try:
            self.sample_timesteps = [float(s) for s in sigmas]
        except (TypeError, ValueError, RuntimeError):
            self.sample_timesteps = []

    # ── step 序号 / 窗口 ────────────────────────────────────────────────────
    def _step_index(self, timestep):
        """同一步内（decide 与 predict_output 各调一次）幂等；首次调用即 1.0。"""
        value = timestep_value(timestep)
        if self.last_timestep is not None and abs(value - self.last_timestep) < EPS:
            return self.step
        self.step = self.step + 1.0
        self.last_timestep = value
        return self.step

    def _active(self, step):
        if step < 0 or step >= self.total_steps:
            return False
        percent = step / self.total_steps
        return self.start_percent <= percent <= self.end_percent

    # ── 闸门 / 预测 / 记录 ─────────────────────────────────────────────────
    def decide(self, branch_id, current, timestep):
        """8 道闸门全过才可跳过模型；返回 Decision（含明细，便于日志/统计）。"""
        state = self.states.setdefault(branch_id, BranchState())
        signature = make_signature(current, self.signature_stride)
        step = self._step_index(timestep)
        change = relative_change(signature, state.previous)
        has_history = state.previous_output is not None and state.older_output is not None
        shape_ok = (
            state.shape is not None
            and hasattr(current, "shape")
            and tuple(state.shape) == tuple(current.shape)
        )
        gates = (
            (GATE_OUTSIDE_WINDOW, self._active(step)),
            (GATE_NO_PREVIOUS_INPUT, state.previous is not None),
            (GATE_HISTORY_INCOMPLETE, has_history),
            (GATE_SHAPE_CHANGED, shape_ok),
            (GATE_CONSECUTIVE_LIMIT, state.consecutive < self.max_consecutive),
            (GATE_REFRESH_DUE, state.age < self.refresh_interval),
            (GATE_PREDICTION_COOLDOWN, state.prediction_cooldown <= 0.0),
            (GATE_LATENT_CHANGE, change <= self.threshold),
        )
        reason = GATE_PREDICT
        for label, ok in gates:
            if not ok:
                reason = label
                break
        self.decision_counts[reason] = self.decision_counts.get(reason, 0) + 1
        return Decision(reason == GATE_PREDICT, reason, change, gates), state, signature

    def predict_output(self, state, timestep, device=None, dtype=None):
        """按 timestep 线性外推当前步输出；不可预测（历史/跨度不足）返回 None。

        预测同时写入 ``state.pending_prediction``（原版语义），供跳过分支直接取用，
        并在下一次真实步与真实输出比对（校准）。
        """
        if state.previous_output is None or state.older_output is None:
            self._clear_pending(state)
            return None
        if state.previous_timestep is None or state.older_timestep is None:
            self._clear_pending(state)
            return None
        span = float(state.previous_timestep) - float(state.older_timestep)
        if abs(span) < 1e-06:
            self._clear_pending(state)
            return None
        current_t = timestep_value(timestep)
        factor = (current_t - float(state.previous_timestep)) / span
        factor = min(max(factor, FACTOR_MIN), FACTOR_MAX)
        delta = state.previous_output - state.older_output
        if state.prediction_error is not None:
            denominator = self.predictor_error_limit if self.predictor_error_limit else DEFAULT_ERROR_LIMIT
            quality = 1.0 - float(state.prediction_error) / denominator
            quality = min(max(quality, QUALITY_FLOOR), 1.0)
        else:
            quality = 1.0
        prediction = state.previous_output + delta * factor * quality
        prediction = prediction.detach()
        if device is not None or dtype is not None:
            try:
                prediction = prediction.to(device=device, dtype=dtype)
            except TypeError:
                pass
        state.pending_prediction = prediction
        return prediction

    @staticmethod
    def _clear_pending(state):
        state.pending_prediction = None

    def record_hit(self, state, signature):
        """跳过步：只更新指纹/计数（真实输出历史不动，避免误差累积）。"""
        state.previous = signature.detach().clone()
        state.consecutive += 1.0
        state.age += 1.0
        self.hits += 1.0
        self.predicted_steps += 1.0

    def record_full(self, state, signature, current, output, timestep):
        """真实步：与 pending 预测校准误差（超上限 → 冷却），并滚动历史。"""
        if state.pending_prediction is not None:
            if hasattr(output, "shape") and tuple(output.shape) == tuple(state.pending_prediction.shape):
                denom = output.detach().abs().mean().clamp_min(EPS)
                error = float(
                    ((output.detach() - state.pending_prediction).abs().mean() / denom).item()
                )
                state.prediction_error = error
                self.prediction_errors.append(error)
                if len(self.prediction_errors) > ERROR_HISTORY_MAX:
                    del self.prediction_errors[: -ERROR_HISTORY_MAX]
                if error > self.predictor_error_limit:
                    state.prediction_cooldown = float(DEFAULT_COOLDOWN)
                    if self.verbose:
                        print(
                            "[TE-Speed Qwen] predictor calibration: error=%.6f limit=%.6f cooldown=%d"
                            % (error, self.predictor_error_limit, DEFAULT_COOLDOWN)
                        )
            state.pending_prediction = None
        if state.prediction_cooldown > 0.0:
            state.prediction_cooldown -= 1.0
        state.older_output = state.previous_output
        state.older_timestep = state.previous_timestep
        state.previous_output = output.detach().clone()
        state.previous_timestep = timestep_value(timestep)
        state.previous = signature.detach().clone()
        state.shape = tuple(current.shape) if hasattr(current, "shape") else None
        state.consecutive = 0.0
        state.age = 0.0
        # prediction_error 保留最近一次校准值（原版不在此清除，下一次真实步再覆盖）
        self.full_steps += 1.0

    # ── 统计 ───────────────────────────────────────────────────────────────
    def error_stats(self):
        errors = self.prediction_errors
        if not errors:
            return 0.0, 0.0, 0
        return sum(errors) / len(errors), max(errors), len(errors)

    def decision_summary(self):
        return "; ".join(
            "%s=%d" % (label, count) for label, count in sorted(self.decision_counts.items())
        )
