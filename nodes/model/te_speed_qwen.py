"""SF TE-Speed Qwen Image 2.1 —— Qwen Image 2.1 采样步预测加速节点。

干净室复刻 `tl2012tl/TE-Speed-QwenImage21 v1.0`（原插件只发 Windows Cython
`.pyd`，机制/参数/判据由逆向还原，详见 `sf_utils/te_speed_qwen.py` docstring）。
用法：`UNETLoader → [QwenImage21Cache（官方 Prefix KV Cache，可选）] → 本节点 → KSampler`。

与原版的差异（刻意，已与需求确认）：
- `attention` 默认 `default`（不覆盖注意力；原版默认 kitchen_int8），
  其余选项 sdpa / sageattn / flashattn / kitchen_int8 复用核心
  `comfy.ldm.modules.attention` 的函数，库缺失自动回退（不加依赖）；
- 去掉原版只有一个取值的 `step_cache` 占位下拉；
- 保留第二输出 STRING `status`（配置摘要；运行时命中/误差统计打印到控制台）。

机制（详见纯逻辑模块）：
- 把 predictor 挂到 model 的 wrappers：`WrappersMP.OUTER_SAMPLE`（采样开始
  `begin(sigmas)` / 结束打印统计）+ `WrappersMP.DIFFUSION_MODEL`（逐步决策）；
- 逐 `transformer_options["uuids"]` 分支独立状态；8 道闸门全过才跳过本次模型
  调用，用「按 timestep 线性外推」的输出代替；真实步顺带校准预测误差，
  超上限进入冷却。仅对类名含 `QwenImage21` 的模型生效，其余原样透传。

不替代官方 Prefix KV Cache，也不建议把标准模型改成极低步数：本节点保持原采样
调度、只减少模型调用次数。实现细节与逆向依据见 §133（doc/experience）。
"""

import torch

from ...sf_utils.logger import get_logger
from ...sf_utils.te_speed_qwen import (
    ATTENTION_MODES,
    DEFAULT_ERROR_LIMIT,
    DEFAULT_REFRESH_INTERVAL,
    DEFAULT_SIGNATURE_STRIDE,
    DEFAULT_THRESHOLD,
    ERROR_LIMIT_MAX,
    ERROR_LIMIT_MIN,
    ERROR_LIMIT_STEP,
    PROFILE_END_PERCENT,
    PROFILE_START_PERCENT,
    THRESHOLD_MAX,
    THRESHOLD_MIN,
    THRESHOLD_STEP,
    TEPredictor,
    WIDGET_WINDOW_MAX,
    WIDGET_WINDOW_MIN,
    timestep_value,
)

logger = get_logger(__name__)

_CATEGORY = "sfnodes/model"
_CACHE_KEY = "te_speed_qwen_predictor_cache"  # transformer_options / wrapper key（原版同名）


def _wrappers_mp():
    """运行时取 WrappersMP（核心位置 comfy.patcher_extension，graph.py 只是 re-export）。"""
    import comfy.patcher_extension as pe

    return pe.WrappersMP


def _is_qwen_image21(model):
    """模型类名含 QwenImage21 才生效（原版同名判定；旧 Qwen-Image 1.0 不匹配）。"""
    base = getattr(model, "model", model)
    net = getattr(base, "diffusion_model", base)
    cls = net if isinstance(net, type) else type(net)
    return "QwenImage21" in getattr(cls, "__name__", "")


# ── 注意力后端（复用核心函数；缺失/不兼容返回 None → 调用方回退）────────────
def _attention_function(name):
    try:
        from comfy.ldm.modules import attention as attn
    except Exception:  # noqa: BLE001 - 核心布局变化时静默回退默认注意力
        return None

    if name == "sdpa":
        return attn.attention_pytorch
    if name == "sageattn":
        return attn.attention_sage if getattr(attn, "SAGE_ATTENTION_IS_AVAILABLE", False) else None
    if name == "flashattn":
        return attn.attention_flash if getattr(attn, "FLASH_ATTENTION_IS_AVAILABLE", False) else None
    if name == "kitchen_int8":
        if not getattr(attn, "COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE", False):
            return None
        return attn.attention_comfy_kitchen_int8
    return None


def _make_attention_override(func, name):
    """包装核心 attention 函数：异常回退 Comfy 原 attention（原版守卫语义）。"""

    def override(original, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - 任何后端错误都不应炸整个采样
            logger.warning("%s failed; falling back to Comfy attention: %s", name, exc)
            return original(*args, **kwargs)

    if hasattr(func, "container_function") and func.container_function is not None:
        override.container_function = func.container_function
    return override


def _find_cache(options):
    if not isinstance(options, dict):
        return None
    return options.get(_CACHE_KEY)


# ── DIFFUSION_MODEL wrapper：逐步决策 / 预测 / 校准 ──────────────────────────
def _predictor_diffusion_wrapper(executor, *args, **kwargs):
    options = kwargs.get("transformer_options")
    if options is None and args and isinstance(args[-1], dict):
        options = args[-1]
    cache = _find_cache(options)
    if cache is None or len(args) < 2:
        return executor(*args, **kwargs)

    x, timestep = args[0], args[1]
    cache.diffusion_calls += 1.0
    uuids = options.get("uuids") or ()
    shape = getattr(x, "shape", ())
    if len(shape) != 4 or len(uuids) == 0 or shape[0] % len(uuids) != 0:
        if cache.verbose:
            logger.info(
                "Qwen TE predictor bypass: uuids=%d input_shape=%s ndim=%s",
                len(uuids),
                shape or None,
                getattr(x, "ndim", None),
            )
        return executor(*args, **kwargs)

    group = shape[0] // len(uuids)  # 单个 uuid 的分块跨度（含 cond/uncond 全部行）
    plan = []
    need_full = False
    for index, branch_id in enumerate(uuids):
        chunk = x[index * group : (index + 1) * group]
        decision, state, signature = cache.decide(branch_id, chunk, timestep)
        prediction = cache.predict_output(state, timestep, device=x.device, dtype=x.dtype)
        use_prediction = decision.predict and prediction is not None
        need_full = need_full or not use_prediction
        plan.append((index, chunk, state, signature, prediction if use_prediction else None))
        if cache.verbose:
            try:
                step_t = timestep_value(timestep)
            except Exception:  # noqa: BLE001 - 日志不应影响采样
                step_t = -1.0
            logger.info(
                "Qwen TE predictor decision: branch=%s t=%.6g change=%.6f threshold=%.6f result=%s",
                branch_id,
                step_t,
                decision.change if decision.change != float("inf") else -1.0,
                cache.threshold,
                decision.reason,
            )

    if not need_full:
        for _, _, state, signature, prediction in plan:
            cache.record_hit(state, signature)
        outputs = [entry[4] for entry in plan]
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)

    output = executor(*args, **kwargs)
    for index, chunk, state, signature, _ in plan:
        cache.record_full(state, signature, chunk, output[index * group : (index + 1) * group], timestep)
    return output


# ── OUTER_SAMPLE wrapper：采样生命周期与统计 ────────────────────────────────
def _find_sigmas(args):
    if len(args) > 3 and hasattr(args[3], "__iter__"):
        return args[3]
    return ()


def _predictor_sample_wrapper(executor, *args, **kwargs):
    guider = getattr(executor, "class_obj", None)
    model_options = getattr(guider, "model_options", None) or {}
    transformer_options = model_options.get("transformer_options") or {}
    cache = _find_cache(transformer_options)
    if cache is None:
        return executor(*args, **kwargs)

    sigmas = kwargs.get("sigmas")
    if sigmas is None:
        sigmas = _find_sigmas(args)
    cache.begin(sigmas)
    if cache.verbose:
        logger.info(
            "Qwen TE predictor begin: sigma_count=%d total_steps=%d first_sigma=%s last_sigma=%s",
            len(cache.sample_timesteps),
            cache.total_steps,
            cache.sample_timesteps[0] if cache.sample_timesteps else None,
            cache.sample_timesteps[-1] if cache.sample_timesteps else None,
        )
        logger.info(
            "Qwen TE predictor config: threshold=%.6f window=%.2f-%.2f max_consecutive=%d"
            " refresh_interval=%d error_limit=%.4f verbose=%s",
            cache.threshold,
            cache.start_percent,
            cache.end_percent,
            int(cache.max_consecutive),
            int(cache.refresh_interval),
            cache.predictor_error_limit,
            cache.verbose,
        )
    try:
        return executor(*args, **kwargs)
    finally:
        if cache.verbose:
            total = cache.predicted_steps + cache.full_steps
            speedup = (total / cache.full_steps) if cache.full_steps else 0.0
            logger.info(
                "Qwen TE predictor: predicted %d/%d steps (%.2fx model-call speedup)",
                int(cache.predicted_steps),
                int(total),
                speedup,
            )
            logger.info(
                "Qwen TE predictor calls: diffusion_wrapper=%d full_steps=%d",
                int(cache.diffusion_calls),
                int(cache.full_steps),
            )
            logger.info("Qwen TE predictor decision summary: %s", cache.decision_summary())
            avg, maximum, samples = cache.error_stats()
            logger.info("Qwen TE predictor error: avg=%.4f max=%.4f samples=%d", avg, maximum, samples)


class SFTESpeedQwenImage21:
    """Qwen Image 2.1 自适应单步输出预测（TE-Speed 复刻）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "reuse_threshold": (
                    "FLOAT",
                    {
                        "default": DEFAULT_THRESHOLD,
                        "min": THRESHOLD_MIN,
                        "max": THRESHOLD_MAX,
                        "step": THRESHOLD_STEP,
                        "tooltip": (
                            "默认 0.06。越小画质越保守、预测越少；越大越快。"
                            "潜变量指纹的相对变化超过该阈值时不预测"
                        ),
                    },
                ),
                "start_percent": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": WIDGET_WINDOW_MIN,
                        "max": WIDGET_WINDOW_MAX,
                        "step": 0.01,
                        "tooltip": "预测窗口起点（按步数百分比）。0 = 自动（%.2f）" % PROFILE_START_PERCENT,
                    },
                ),
                "end_percent": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": WIDGET_WINDOW_MIN,
                        "max": WIDGET_WINDOW_MAX,
                        "step": 0.01,
                        "tooltip": "预测窗口终点（按步数百分比）。0 = 自动（%.2f）" % PROFILE_END_PERCENT,
                    },
                ),
                "predictor_error_limit": (
                    "FLOAT",
                    {
                        "default": DEFAULT_ERROR_LIMIT,
                        "min": ERROR_LIMIT_MIN,
                        "max": ERROR_LIMIT_MAX,
                        "step": ERROR_LIMIT_STEP,
                        "tooltip": "预测相对误差超过该值时进入短冷却（触发预测回退真实步）",
                    },
                ),
                "attention": (
                    list(ATTENTION_MODES),
                    {
                        "default": "default",
                        "tooltip": (
                            "可选注意力后端：default = 不改动；sdpa / sageattn / flashattn / "
                            "kitchen_int8 复用 ComfyUI 核心实现，库缺失或不兼容时自动回退"
                        ),
                    },
                ),
                "verbose": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "在 ComfyUI 控制台打印预测器决策、命中率与误差统计",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "status")
    FUNCTION = "patch"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Qwen Image 2.1 采样加速（复刻 TE-Speed-QwenImage21）：按 uuids 分支跟踪潜变量"
        "指纹，窗口内小变化时用上一/上两步输出线性外推跳过本次模型调用，并用真实步做"
        "误差校准（超限冷却）。仅对 Qwen Image 2.1 生效，其余模型原样透传"
    )

    def patch(
        self,
        model,
        reuse_threshold=DEFAULT_THRESHOLD,
        start_percent=0.0,
        end_percent=0.0,
        predictor_error_limit=DEFAULT_ERROR_LIMIT,
        attention="default",
        verbose=True,
    ):
        if not _is_qwen_image21(model):
            logger.warning(
                "SF TE-Speed Qwen Image 2.1: 未检测到 Qwen Image 2.1 模型（%s），原样透传"
                % type(getattr(model, "model", model)).__name__
            )
            return (model, "Qwen Image 2.1 model not detected; model passed through")

        cache = TEPredictor(
            threshold=DEFAULT_THRESHOLD if reuse_threshold is None else reuse_threshold,
            max_consecutive=1,
            start_percent=0.0 if start_percent is None else start_percent,
            end_percent=0.0 if end_percent is None else end_percent,
            refresh_interval=DEFAULT_REFRESH_INTERVAL,
            signature_stride=DEFAULT_SIGNATURE_STRIDE,
            verbose=bool(verbose),
            predictor_error_limit=(
                DEFAULT_ERROR_LIMIT if predictor_error_limit is None else predictor_error_limit
            ),
        )

        wrappers = _wrappers_mp()
        patched = model.clone()
        transformer_options = patched.model_options.setdefault("transformer_options", {})
        transformer_options[_CACHE_KEY] = cache
        patched.add_wrapper_with_key(wrappers.OUTER_SAMPLE, _CACHE_KEY, _predictor_sample_wrapper)
        patched.add_wrapper_with_key(wrappers.DIFFUSION_MODEL, _CACHE_KEY, _predictor_diffusion_wrapper)

        status = [
            "te_predictor: threshold=%.2f window=%.2f-%.2f error_limit=%.2f"
            % (cache.threshold, cache.start_percent, cache.end_percent, cache.predictor_error_limit),
            "step_cache=te_predictor",
        ]
        if attention and attention != "default":
            func = _attention_function(attention)
            if func is None:
                status.append("attention=%s unavailable (default used)" % attention)
            else:
                transformer_options["optimized_attention_override"] = _make_attention_override(func, attention)
                status.append("attention=" + attention)
        else:
            status.append("attention=default")
        logger.info("SF TE-Speed Qwen Image 2.1 enabled: %s", " | ".join(status))
        return (patched, "\n".join(status))
