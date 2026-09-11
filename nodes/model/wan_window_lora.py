"""SF Wan Window LoRA — 长视频逐窗口位置式注入不同 LoRA（Wan Context Windows 伴侣节点）。

链路（顺序重要）：LoadCheckpoint → Wan Context Windows（原生，不改）
→ 本节点 → KSampler。每槽连一个 SFLoraPreset（含多行 LoRA：开关/强度）；
槽位即窗口位置：`slot = window_idx % N`，N 为当前动态槽总数（含未连接
的空槽）。空槽（未连接 preset）= 该窗口位置只跑上游补丁（通常即基模）。

原理（patch-swap + 后装流式，见 experience/nodes-lora.md §39.5/§39.7）：
按官方 `LoraLoader` 同款路径（`convert_lora` → `load_lora`）为每槽预建
补丁列表，存**独立槽表**（绝不写入 `ModelPatcher.patches`，上游 base 的
烘焙/流式命运原封不动）；采样时经原生 context handler 的
`EVALUATE_CONTEXT_WINDOWS` 回调（逐窗、逐 step 触发，自带 `window_idx`）
把当窗槽位的列表换进槽表。普通层在首次换槽前为并集键装上
`LowVramPatch(key, 槽表, …)` 并追加进各模块的 `weight/bias_function`
（forward 经 `ops.cast_bias_weight` 现场读取，与静态 Stack 同代码路径、
同开销）；GGUF 量化层走第二通道（见 §39.11：ComfyUI-GGUF
`get_weight` 只读张量 `patches`、绕过 `weight_function`），逐窗改写
`weight.patches = base + [(本槽行, key)]`；`EXECUTE_CLEANUP` 只打换槽
计数，不复位（无可复位之物）。

为何是这套（§39.7 血泪）：官方 `load()` 对全载模块把补丁**烘焙**进权重
且不再装函数——往 `patcher.patches` 里换表对烘焙层完全无效（§39.5 旧
架构的死因，`lowvram patches: 0` 即全载计数的旁证）。独立表 + 后装函数
绕开烘焙，上游 base 烘焙态也不受影响（分表隔离，无双重计数）。

动态槽（SFConditioningCombine §49 同款）：后端灵活 optional schema
（`_WindowPresetInputs`，任意 `window_N` 槽名按 SF_LORA_PRESET 放行，
`VALIDATE_INPUTS → True` 接管校验）+ 前端 web/sf_wan_window_lora.js
（复用 sf_dynamic_slots.installDynamicSlots：初始 2、上限 10，
全连追加、断尾回收）。

约束：
  - 本节点必须接在 Wan Context Windows **之后**（需要它建好的
    `model.model_options["context_handler"]` 来注册回调）；链错顺序则
    warning + 回退官方 `add_patches(slot0)` 静态（短视频/无回调同理）。
  - 上游静态补丁（SFLoraStack 等）原样保留并与每槽叠加（base 烘焙/
    张量附着态 + 本槽注入态，生效顺序 = base 先、本槽后，与串联一致）。
  - preset 行的 `strengthTwo`（CLIP 强度）被忽略，只用 `strength`（模型侧）；
    CLIP 本体不在逐窗路径内。
  - 要严格"第 k 段 = 槽 k"请用 STATIC_STANDARD 或 BATCHED schedule；
    UNIFORM 系每步按 ordered_halving 漂移窗口划分（连贯性仍由原生
    overlap 融合保证，建议 fuse=overlap-linear）。
"""

import json

import safetensors.torch

import folder_paths

from ...sf_utils.logger import get_logger
from .conditioning_combine import conditioning_slot_key
from .lora_preset import PRESET_TYPE

try:
    import comfy.lora as _lora
    import comfy.lora_convert as _lora_convert
    from comfy.context_windows import IndexListCallbacks as _CB
    from comfy.model_patcher import LowVramPatch as _LowVramPatch
    from comfy.model_patcher import get_key_weight as _get_key_weight
    _EVALUATE = _CB.EVALUATE_CONTEXT_WINDOWS
    _CLEANUP = _CB.EXECUTE_CLEANUP
    _HAS_STREAM = True
except Exception:
    _lora = None
    _lora_convert = None
    _LowVramPatch = None
    _get_key_weight = None
    # 回退字面量（comfy/context_windows.py 的稳定 API 名）
    _EVALUATE = "evaluate_context_windows"
    _CLEANUP = "execute_cleanup"
    _HAS_STREAM = False

_CATEGORY = "sfnodes/model"

WINDOW_PREFIX = "window_"
INITIAL_WINDOW_INPUTS = 2
MAX_WINDOW_INPUTS = 10
CALLBACK_KEY = "sf_wan_window_lora"
FN_TAG = "_sf_window_lora_stream"
# GGUF 张量上挂载的上游 base 快照属性（跨会话复用，换槽重建时剔除我方旧条目）。
_GGUF_BASE_ATTR = "_sf_window_lora_base"


def _is_ggml_quantized(mod):
    """GGUF 量化层判定（ComfyUI-GGUF 的 GGMLLayer.is_ggml_quantized）。

    duck-typing 探测，不 import 外部自定义节点；核心 ComfyUI 模块无此方法
    恒为 False。量化层走张量 patches 通道，非量化（F16/F32）走 weight_function。
    """
    try:
        probe = getattr(mod, "is_ggml_quantized", None)
        if callable(probe):
            return bool(probe())
    except Exception:
        pass
    return False

logger = get_logger(__name__)


class _WindowPresetInputs(dict):
    """灵活 optional 输入：前端动态添加的 window_N 槽名都能通过校验。

    复刻 conditioning_combine._ConditioningCombineInputs 模式（该类硬编码
    CONDITIONING 不可直复用），类型换成 SFLoraPreset 的 SF_LORA_PRESET。
    """

    def __contains__(self, key):
        return True

    def __getitem__(self, key):
        return (PRESET_TYPE,)


def _resolve_lora_path(name):
    try:
        p = folder_paths.get_full_path("loras", name)
        if p:
            return p
    except Exception:
        pass
    return name


def _parse_preset_rows(preset):
    """SFLoraPreset 形状 {loras: [{lora, on, strength, strengthTwo}]} → [(文件名, sm)]。

    只收 on 开的行；强度取 strength（模型侧，strengthTwo 是 CLIP 侧、
    逐窗路径用不到）；坏行丢弃永不抛错（parse_state 同款宽容）。
    """
    rows = []
    if not isinstance(preset, dict):
        return rows
    loras = preset.get("loras")
    if not isinstance(loras, list):
        return rows
    for item in loras:
        if not isinstance(item, dict):
            continue
        name = item.get("lora")
        if not isinstance(name, str) or not name.strip() or name == "None":
            continue
        if not item.get("on", True):
            continue
        try:
            sm = float(item.get("strength", 1.0))
        except Exception:
            continue
        if sm == 0.0:
            continue
        rows.append((name, sm))
    return rows


def _file_lora_keys(sd):
    """LoRA 文件内的 lora-down 键数（诊断分母：0 匹配警告用）。"""
    try:
        return sum(1 for k in sd
                   if str(k).endswith((".lora_down.weight", ".lora_A.weight")))
    except Exception:
        return 0


class _PatchSwapSession:
    """逐窗补丁交换会话（纯 CPU 数据操作，无 torch 依赖）。

    tables: 独立槽表 {model_key: [(sm, data, 1.0, None, None)...]}，键集恒为
      并集（含空槽位，空表 [] 占位——LowVramPatch 读表不存在的键会 KeyError）。
    set_slot() 只替换本表列表对象，不碰 patcher.patches。
    _ensure_stream_functions() 在首次换槽前为并集键装注入点，双通道：
      - 普通层：模块 weight/bias_function 追加 LowVramPatch（绑定独立槽表）；
        官方表函数原样保留（base 流式态 + 本槽流式态叠加）、旧会话僵尸清除、
        本会话幂等（二次 ensure 不双重应用）。
      - GGUF 量化层（ComfyUI-GGUF GGMLOps）：其 forward 经 get_weight 只读
        张量 patches、完全绕过 weight_function，故登记到 self._gguf、改由
        set_slot() 逐窗改写 `weight.patches = base + [(本槽行, key)]`
        （GGUF 自产条目同形；上游 base 首次写入前快照到 _GGUF_BASE_ATTR，
        换槽恒从 base 重建，不累积、不丢 base）。
    """

    def __init__(self, patcher, model_root, slot_tables, n_slots):
        self.patcher = patcher
        self._root = model_root
        # tables 只定并集键集（全 [] 占位，LowVramPatch 读表不存在的键会 KeyError）；
        # 各槽内容独立存 _slot_tables，set_slot() 逐键填写。
        self.tables = {}
        for table in slot_tables:
            if table:
                for k in table:
                    self.tables.setdefault(k, [])
        self._slot_tables = slot_tables  # per slot: {key: [tuples]} or None
        self._gguf = {}  # {key: (module, param_name)} GGUF 量化键的张量通道
        self.n = n_slots
        self.swap_count = 0  # 实际切换次数（同槽重复命中不计）
        self._last_si = None
        self._stream_ready = False

    def _iter_target_modules(self):
        try:
            mods = self._root.named_modules()
        except Exception:
            return
        for name, mod in mods:
            yield name, mod

    def _fn_list(self, mod, key):
        if key.endswith(".bias"):
            return getattr(mod, "bias_function", None)
        return getattr(mod, "weight_function", None)

    def _ensure_stream_functions(self):
        if self._stream_ready:
            return
        for name, mod in self._iter_target_modules():
            for suffix, attr, pname in ((".weight", "weight_function", "weight"),
                                        (".bias", "bias_function", "bias")):
                key = f"{name}{suffix}"
                if key not in self.tables:
                    continue
                if _is_ggml_quantized(mod):
                    # GGUF 量化层：get_weight 只读张量 patches，weight_function
                    # 永不被调用——改登记张量通道（内容由 set_slot 逐窗填写）。
                    if getattr(mod, pname, None) is None:
                        continue  # 该层无此参数（如 Linear bias=None）
                    if key not in self._gguf:
                        self._gguf[key] = (mod, pname)
                    continue
                fns = getattr(mod, attr, None)
                if not isinstance(fns, list):
                    # 非 comfy ops 模块可能没有该属性；建一个
                    #（forward 经 cast_bias_weight 现场读此表）。
                    fns = []
                    try:
                        setattr(mod, attr, fns)
                    except Exception:
                        continue
                # 清僵尸（旧会话表）+ 官方表函数保留共存 + 本会话幂等
                kept = []
                mine = False
                for fn in fns:
                    tag = getattr(fn, FN_TAG, None)
                    if tag is not None:
                        if getattr(fn, "patches", None) is self.tables:
                            if not mine:
                                kept.append(fn)  # 本会话已装：只留一个
                                mine = True
                            # 重复的本会话函数：丢弃（二次 ensure 不双重应用）
                        continue  # 旧会话僵尸：丢弃
                    kept.append(fn)
                if not mine:
                    try:
                        _, set_func, convert_func = _get_key_weight(self._root, key)
                    except Exception:
                        set_func, convert_func = None, None
                    try:
                        fn = _LowVramPatch(key, self.tables, convert_func, set_func)
                        setattr(fn, FN_TAG, True)
                        kept.append(fn)
                    except Exception as e:
                        logger.warning("[SFWanWindowLoRA] stream fn install failed for %s (%s).",
                                       key, e)
                        continue
                if len(kept) != len(fns):
                    try:
                        setattr(mod, attr, kept)
                    except Exception:
                        pass
        self._stream_ready = True
        if self._gguf:
            logger.info("[SFWanWindowLoRA] %d GGUF quantized key(s) injected via "
                        "tensor.patches (ComfyUI-GGUF get_weight path).",
                        len(self._gguf))

    def _swap_gguf_slot(self):
        """把当前 tables 内容写进 GGUF 张量的 patches（逐窗唯一写点）。

        entries 恒从首次快照的 base 重建：`[(base…), ([本槽行], key)]`，
        空槽即纯 base——base 永不被污染、旧窗条目永不累积。首次写入前才
        快照 base（param 换过对象则自动重拍，跨会话 marker 常驻同一 param）。
        """
        for key, ref in self._gguf.items():
            try:
                mod, pname = ref
                param = getattr(mod, pname, None)
                if param is None:
                    continue
                base = getattr(param, _GGUF_BASE_ATTR, None)
                if base is None:
                    base = list(getattr(param, "patches", None) or [])
                    try:
                        setattr(param, _GGUF_BASE_ATTR, base)
                    except Exception:
                        pass
                entries = list(base)
                ours = self.tables.get(key) or []
                if ours:
                    entries.append((list(ours), key))
                try:
                    param.patches = entries
                except Exception as e:
                    logger.warning("[SFWanWindowLoRA] gguf patch swap failed for %s (%s).",
                                   key, e)
            except Exception:
                continue

    def _clear_stale_prepared(self):
        """防御性清理：异常中断的 forward 可能留下 prepared_patches 快照。"""
        for _, mod in self._iter_target_modules():
            try:
                for attr in ("weight_function", "bias_function"):
                    for fn in list(getattr(mod, attr, None) or []):
                        clear = getattr(fn, "clear_prepared", None)
                        if clear is not None:
                            clear()
            except Exception:
                continue

    def set_slot(self, si):
        self._ensure_stream_functions()
        content = self._slot_tables[si] if self._slot_tables[si] else {}
        for k in self.tables:
            lst = content.get(k)
            self.tables[k] = list(lst) if lst else []
        self._swap_gguf_slot()
        self._clear_stale_prepared()
        if si != self._last_si:
            self.swap_count += 1
            self._last_si = si


class SFWanWindowLoRA:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Wan Context Windows 之后的扩散模型（需带 context_handler）。"}),
            },
            "optional": _WindowPresetInputs(),
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # window_N 槽由前端动态增删，超出静态 schema 时接管校验
        return True

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "info")
    FUNCTION = "apply"
    CATEGORY = _CATEGORY
    DESCRIPTION = ("SF Wan Window LoRA：每窗口位置连一个 SFLoraPreset（槽内多 LoRA），"
                   "按 window_idx 逐窗轮换注入官方格式补丁（普通层流式函数、"
                   "GGUF 量化层张量 patches），接在原生 Wan Context Windows 之后使用。"
                   "空槽（未连接）= 该窗口位置只跑上游补丁。严格分段请用 STATIC/BATCHED schedule。"
                   "仅 DiT 侧生效（preset 的 CLIP 强度被忽略）。")

    def apply(self, model, **kwargs):
        if _lora is None or _lora_convert is None:
            raise RuntimeError(
                "[SFWanWindowLoRA] This ComfyUI build lacks comfy.lora. Update ComfyUI.")
        keys = sorted((k for k in kwargs if k.startswith(WINDOW_PREFIX)),
                      key=conditioning_slot_key)
        # 位置式：槽位列表与动态槽一一对应（含空槽），不可压紧
        slot_rows = []
        for key in keys[:MAX_WINDOW_INPUTS]:
            rows = _parse_preset_rows(kwargs.get(key))
            slot_rows.append((key, rows if rows else None))

        patched = model.clone()
        try:
            key_map = _lora.model_lora_keys_unet(patched.model, {})
        except Exception:
            key_map = {}
        try:
            model_keys = set(patched.model.state_dict().keys())
        except Exception:
            model_keys = set()

        file_cache = {}  # path -> (sd, loaded {model_key: adapter})
        slot_tables = []  # per slot: {model_key: [(sm, data, 1.0, None, None)]} or None
        slot_files = []  # per slot: [(loaded, sm)] 回退静态用
        info_slots = []
        for i, (key, rows) in enumerate(slot_rows):
            if rows is None:
                slot_tables.append(None)
                slot_files.append([])
                info_slots.append({"slot": i, "key": key, "status": "empty (upstream patches only)"})
                continue
            content = {}
            raws = []
            entries = []
            failed = []
            for name, sm in rows:
                path = _resolve_lora_path(name)
                if path not in file_cache:
                    try:
                        sd = safetensors.torch.load_file(path)
                    except Exception as e:
                        logger.warning("[SFWanWindowLoRA] could not load '%s' (%s) -- skipped.",
                                       name, e)
                        failed.append(f"{name}: unreadable")
                        continue
                    try:
                        converted = _lora_convert.convert_lora(sd)
                        loaded = _lora.load_lora(converted, key_map, log_missing=False)
                    except Exception as e:
                        logger.warning("[SFWanWindowLoRA] could not parse '%s' (%s) -- skipped.",
                                       name, e)
                        failed.append(f"{name}: unparsable")
                        continue
                    file_cache[path] = (sd, loaded)
                sd, loaded = file_cache[path]
                if not loaded:
                    logger.warning("[SFWanWindowLoRA] '%s' matched 0 model keys -- skipped.", name)
                    failed.append(f"{name}: 0 model keys")
                    continue
                matched = 0
                for mk, adapter in loaded.items():
                    if model_keys and mk not in model_keys:
                        continue
                    # 对齐 add_patches(loaded, sm) 元组语义：(strength_patch, data, strength_model, offset, function)
                    content.setdefault(mk, []).append((sm, adapter, 1.0, None, None))
                    matched += 1
                total = _file_lora_keys(sd)
                logger.info("[SFWanWindowLoRA] slot %d '%s' (strength %.2f): matched %d model keys (%d lora keys)",
                            i, name, sm, matched, total)
                if matched == 0:
                    logger.warning("[SFWanWindowLoRA] slot %d '%s': 0 model keys matched -- "
                                   "this LoRA will NOT take effect (wrong architecture).", i, name)
                    failed.append(f"{name}: 0 model keys")
                    continue
                raws.append((loaded, sm))
                entries.append({"lora": name, "strength": sm,
                                "model_keys_matched": matched, "lora_keys_total": total})
            if not content:
                slot_tables.append(None)
                slot_files.append([])
                reason = "; ".join(failed) if failed else "all LoRAs failed"
                info_slots.append({"slot": i, "key": key, "status": f"empty ({reason})"})
            else:
                slot_tables.append(content)
                slot_files.append(raws)
                info_slots.append({"slot": i, "key": key, "loras": entries})

        session = _PatchSwapSession(patched, patched.model, slot_tables, len(slot_rows))
        try:
            patched._sf_window_session = session  # 诊断可达（换槽计数等）
        except Exception:
            pass

        handler = None
        try:
            handler = (patched.model_options or {}).get("context_handler")
        except Exception:
            handler = None
        use_stream = bool(handler is not None and hasattr(handler, "callbacks")) and _HAS_STREAM
        if not use_stream:
            # 回退：官方静态 slot0（短视频/链错顺序/旧版无流式类时行为 = 普通 LoRA 加载）
            if slot_tables and slot_tables[0]:
                for loaded, sm in slot_files[0]:
                    try:
                        patched.add_patches(loaded, sm)
                    except Exception as e:
                        logger.warning("[SFWanWindowLoRA] static fallback add_patches failed (%s).", e)
                        break
                logger.warning("[SFWanWindowLoRA] no context_handler on MODEL "
                               "(put this node AFTER Wan Context Windows); running static slot 0.")
            else:
                logger.warning("[SFWanWindowLoRA] no context_handler on MODEL and slot 0 empty; "
                               "passing model through unchanged.")
        else:
            def _on_evaluate(handler_arg, model_arg, x_in, conds, timestep, model_options,
                             window_idx, window, *rest):
                if not session.n:
                    return
                try:
                    session.set_slot(int(window_idx) % session.n)
                except Exception as e:
                    logger.warning("[SFWanWindowLoRA] window %s swap failed (%s).",
                                   window_idx, e)

            def _on_cleanup(handler_arg, model_arg, x_in, conds, timestep, model_options):
                if session.swap_count:
                    logger.info("[SFWanWindowLoRA] done | %d window evaluations across %d slot(s).",
                                session.swap_count, session.n)

            # 原生 get_all_callbacks(call_type, handler.callbacks) 内部按
            # transformer_options 形状再取 ["callbacks"]（§39.8：直接形状永远
            # 读不到——上游 reader/shape 错配，所有 handler 回调天然全哑）。
            # 双形状注册：当前 reader 走 ["callbacks"] 分支；若上游日后修正，
            # 直接分支生效——set_slot 幂等，双重触发无害。
            try:
                cbs = handler.callbacks
                if not isinstance(cbs, dict):
                    raise TypeError("handler.callbacks is not a dict")
                for root in (cbs, cbs.setdefault("callbacks", {})):
                    root.setdefault(_EVALUATE, {}).setdefault(
                        CALLBACK_KEY, []).append(_on_evaluate)
                    root.setdefault(_CLEANUP, {}).setdefault(
                        CALLBACK_KEY, []).append(_on_cleanup)
            except Exception as e:
                logger.warning("[SFWanWindowLoRA] callback register failed (%s); running static slot 0.", e)
                if slot_tables and slot_tables[0]:
                    for loaded, sm in slot_files[0]:
                        try:
                            patched.add_patches(loaded, sm)
                        except Exception:
                            break

        n_loras = sum(len(r) for _, r in slot_rows if r)
        info = json.dumps({
            "n_slots": len(slot_rows),
            "mode": "patch-swap per window_idx % n_slots (positional, empty=upstream only)",
            "note": ("strict segment mapping needs STATIC_STANDARD/BATCHED schedule; "
                     "UNIFORM schedules shift windows per step; "
                     "overlap zones are blended by the native fuse method; "
                     "GPU extra ~= 0 (official lowvram streaming); "
                     "this node must come AFTER Wan Context Windows"),
            "slots": info_slots,
        }, indent=2, ensure_ascii=False)
        logger.info("[SFWanWindowLoRA] armed %d slot(s), %d LoRA row(s).",
                    len(slot_rows), n_loras)
        return (patched, info)
