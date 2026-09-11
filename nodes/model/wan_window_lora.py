"""SF Wan Window LoRA — 长视频逐窗口位置式注入不同 LoRA（Wan Context Windows 伴侣节点）。

链路（顺序重要）：LoadCheckpoint → Wan Context Windows（原生，不改）
→ 本节点 → KSampler。每槽连一个 SFLoraPreset（含多行 LoRA：开关/强度）；
槽位即窗口位置：`slot = window_idx % N`，N 为当前动态槽总数（含未连接
的空槽）。空槽（未连接 preset）= 该窗口位置只跑上游补丁（通常即基模）。

原理（patch-swap，见 experience/nodes-lora.md §39.5）：不合并权重、不做
激活注入——按官方 `LoraLoader` 同款路径（`convert_lora` → `load_lora` →
`add_patches` 元组语义）为每槽预建补丁列表，采样时经原生 context handler
的 `EVALUATE_CONTEXT_WINDOWS` 回调（逐窗、逐 step 触发，自带 `window_idx`）
把当窗槽位的补丁列表换进 `ModelPatcher.patches`，`EXECUTE_CLEANUP` 时复位。

为何是 patch-swap 而非 forward-hook 注入（§39.1-39.4 的旧架构，已删除）：
同形 LoRA 在官方加载期走 `LowVramPatch` 延迟合并（`model_patcher.py`），
`weight_function` 持有 `patcher.patches` 的 live 引用、forward 时现算——
补丁数据走官方低显存流式（pin + cast_buffer + 按 `lowvram_model_memory`
逐层搬运），GPU 额外常驻 ≈ 0；而 hook 注入需自建 bf16 常驻（N 槽则 N 份）
+ 270MB 级瞬时量，在半卸载模型上正是 OOM 主凶。uuid 在 forward 热路径
零读取（仅 load 路径用），中途换列表不 bump 也安全。

动态槽（SFConditioningCombine §49 同款）：后端灵活 optional schema
（`_WindowPresetInputs`，任意 `window_N` 槽名按 SF_LORA_PRESET 放行，
`VALIDATE_INPUTS → True` 接管校验）+ 前端 web/sf_wan_window_lora.js
（复用 sf_dynamic_slots.installDynamicSlots：初始 2、上限 10，
全连追加、断尾回收）。

约束：
  - 本节点必须接在 Wan Context Windows **之后**（需要它建好的
    `model.model_options["context_handler"]` 来注册回调）；链错顺序则
    warning + 全程静态 slot0。
  - 上游静态补丁（SFLoraStack 等）会被保留并与每槽叠加（官方列表语义），
    与旧 hook 架构"二选一"不同，现在可共存。
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
    _EVALUATE = _CB.EVALUATE_CONTEXT_WINDOWS
    _CLEANUP = _CB.EXECUTE_CLEANUP
except Exception:
    _lora = None
    _lora_convert = None
    # 回退字面量（comfy/context_windows.py 的稳定 API 名）
    _EVALUATE = "evaluate_context_windows"
    _CLEANUP = "execute_cleanup"

_CATEGORY = "sfnodes/model"

WINDOW_PREFIX = "window_"
INITIAL_WINDOW_INPUTS = 2
MAX_WINDOW_INPUTS = 10
CALLBACK_KEY = "sf_wan_window_lora"

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

    full[si]: 槽 si 的完整补丁表 {model_key: [(sm, data, 1.0, None, None)...]}
     （含上游 base + 本槽行；空槽 = 纯 base）。set_slot() 整键替换
      `patcher.patches` 内容（dict 对象本身不动，LowVramPatch 的 live
      引用持续有效；uuid 不 bump——forward 热路径不读它）。
    """

    def __init__(self, patcher, dm, full, initial, n_slots):
        self.patcher = patcher
        self._dm = dm
        self.full = full
        self.initial = initial
        self.n = n_slots

    def _clear_stale_prepared(self):
        """防御性清理：异常中断的 forward 可能留下 prepared_patches 快照
       （正常路径官方逐层配对清理，此处只处理异常残留）。"""
        try:
            mods = self._dm.named_modules()
        except Exception:
            return
        for _, mod in mods:
            try:
                for attr in ("weight_function", "bias_function"):
                    for fn in list(getattr(mod, attr, None) or []):
                        clear = getattr(fn, "clear_prepared", None)
                        if clear is not None:
                            clear()
                for attr in dir(mod):
                    if attr.endswith("_lowvram_function"):
                        clear = getattr(getattr(mod, attr, None), "clear_prepared", None)
                        if clear is not None:
                            clear()
            except Exception:
                continue

    def set_slot(self, si):
        lists = self.full[si]
        P = self.patcher.patches
        for k, lst in lists.items():
            P[k] = lst
        self._clear_stale_prepared()

    def reset_initial(self):
        P = self.patcher.patches
        for k, lst in self.initial.items():
            P[k] = lst


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
                   "按 window_idx 轮换官方补丁注入，接在原生 Wan Context Windows 之后使用。"
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
        dm = getattr(patched.model, "diffusion_model", patched.model)
        try:
            key_map = _lora.model_lora_keys_unet(patched.model, {})
        except Exception:
            key_map = {}
        try:
            model_keys = set(patched.model.state_dict().keys())
        except Exception:
            try:
                model_keys = set(dm.state_dict().keys())
            except Exception:
                model_keys = set()

        # 上游静态补丁快照（SFLoraStack 等）：每窗列表 = base + 本槽行
        try:
            base = {k: list(v) for k, v in patched.patches.items()}
        except Exception:
            base = {}

        file_cache = {}  # path -> (converted_sd, loaded {model_key: adapter})
        slot_content = []  # per slot: {model_key: [(sm, data, 1.0, None, None)]} or None
        info_slots = []
        for i, (key, rows) in enumerate(slot_rows):
            if rows is None:
                slot_content.append(None)
                info_slots.append({"slot": i, "key": key, "status": "empty (upstream patches only)"})
                continue
            content = {}
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
                logger.info("[SFWanWindowLoRA] slot %d '%s': matched %d model keys (%d lora keys)",
                            i, name, matched, total)
                if matched == 0:
                    logger.warning("[SFWanWindowLoRA] slot %d '%s': 0 model keys matched -- "
                                   "this LoRA will NOT take effect (wrong architecture).", i, name)
                    failed.append(f"{name}: 0 model keys")
                    continue
                entries.append({"lora": name, "strength": sm,
                                "model_keys_matched": matched, "lora_keys_total": total})
            if not content:
                slot_content.append(None)
                reason = "; ".join(failed) if failed else "all LoRAs failed"
                info_slots.append({"slot": i, "key": key, "status": f"empty ({reason})"})
            else:
                slot_content.append(content)
                info_slots.append({"slot": i, "key": key, "loras": entries})

        # 全并集键（加载期 hook 落点）：base ∪ 各槽。slot0 之外独占键用零强度
        # 占位（同形数据借位，仅为让官方装上 LowVramPatch；短视频/无回调时
        # 这些占位零生效，行为 = base + slot0）。
        union_keys = set(base)
        for content in slot_content:
            if content:
                union_keys.update(content)
        placeholder_data = {}
        for content in slot_content:
            if not content:
                continue
            for mk, tuples in content.items():
                if mk not in placeholder_data and tuples:
                    placeholder_data[mk] = tuples[0][1]
        initial = {}
        slot0 = slot_content[0] if slot_content else None
        for k in union_keys:
            lst = list(base.get(k, []))
            if slot0 and k in slot0:
                lst = lst + list(slot0[k])
            elif k not in base and k in placeholder_data:
                lst = lst + [(0.0, placeholder_data[k], 1.0, None, None)]
            initial[k] = lst
        for k, lst in initial.items():
            try:
                patched.patches[k] = lst
            except Exception:
                pass

        # 每槽完整表 = base + 本槽行（空槽 = 纯 base）
        full = []
        for content in slot_content:
            table = {}
            for k in union_keys:
                lst = list(base.get(k, []))
                if content and k in content:
                    lst = lst + list(content[k])
                table[k] = lst
            full.append(table)

        session = _PatchSwapSession(patched, dm, full, initial, len(slot_rows))

        def _on_evaluate(handler, model_arg, x_in, conds, timestep, model_options,
                         window_idx, window, *rest):
            if not session.n:
                return
            try:
                session.set_slot(int(window_idx) % session.n)
            except Exception as e:
                logger.warning("[SFWanWindowLoRA] window %s swap failed (%s).",
                               window_idx, e)

        def _on_cleanup(handler, model_arg, x_in, conds, timestep, model_options):
            session.reset_initial()

        handler = None
        try:
            handler = (patched.model_options or {}).get("context_handler")
        except Exception:
            handler = None
        if handler is None or not hasattr(handler, "callbacks"):
            logger.warning("[SFWanWindowLoRA] no context_handler on MODEL "
                           "(put this node AFTER Wan Context Windows); running static slot 0.")
        else:
            try:
                handler.callbacks.setdefault(_EVALUATE, {}).setdefault(
                    CALLBACK_KEY, []).append(_on_evaluate)
                handler.callbacks.setdefault(_CLEANUP, {}).setdefault(
                    CALLBACK_KEY, []).append(_on_cleanup)
            except Exception as e:
                logger.warning("[SFWanWindowLoRA] callback register failed (%s); running static slot 0.", e)

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
