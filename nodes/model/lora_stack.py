"""SFLoraStack —— 多行 LoRA 栈节点（复刻 PixaromaLoraLoader）。

前端驱动（Vue Compat #9）：每个 LoRA 行（文件名、开关、model/clip 强度、用户
勾选的触发词）存在浏览器 node.properties.loraStackState，由 graphToPrompt 钩子
注入隐藏 LoraLoaderState 输入（web/sf_lora_stack.js）。因为 LoraLoaderState 是
节点输入的一部分，编辑一行就改变节点缓存签名，运行总是拿到新值，无需
IS_CHANGED。

Python 职责很小：对每个开着的行，用各自强度把 LoRA 应用到 MODEL（和 CLIP），
链式叠加，并把用户勾选的触发词连接成 `triggers` STRING 输出。元数据/触发词
读取的离线逻辑在 sf_utils/lora_reader.py，Civitai 等路由在 sf_utils/lora_routes.py。
ortho_gs 模式（齿轮 Stacking method 切换）：UNet 侧同 key 多 LoRA 的 down 矩阵
先做 Gram-Schmidt 输入空间正交化再应用（sf_utils/lora_ortho.py），CLIP 仍顺序
叠加；正交化失败一律 fallback 顺序路径，绝不报错。
"""
import os

import folder_paths
import comfy.sd
import comfy.utils

from ...sf_utils import lora_reader as R
from ...sf_utils import lora_ortho as LO
from ...sf_utils.logger import get_logger

logger = get_logger(__name__)

_CATEGORY = "sfnodes/model"

_NO_LORAS = "(put LoRAs in models/loras)"


class SFLoraStack:
    DESCRIPTION = (
        "在一个节点里叠加任意多个 LoRA。每个 LoRA 有独立的开/关开关和强度，"
        "模型与 CLIP 可分开设置，且可串联多个本节点。点击行上的 i 可查看该 "
        "LoRA 的信息并勾选它的触发词；开着的行勾选的触发词会从 triggers 输出"
        "以纯文本给出，可直接接入提示词。触发词直接读自文件，离线可用；"
        "可选按 LoRA 的 Civitai 查询（仅在你点击时才联网）。Add LoRA 添加行，"
        "全部开/关与齿轮设置位于节点中部；右键行可上移/下移/复制/删除。"
        "可选 preset 输入（SF Power Lora Preset 输出）：连接后自动把预设的"
        "顺序与强度加载到行上，执行时预设优先（行上勾选的触发词仍保留）。"
        "齿轮里的 Stacking method 可切换叠加方式：Sequential（标准，逐行相加）"
        "或 Orthogonal（Gram-Schmidt 输入空间正交化，减少相似 LoRA 之间的干扰；"
        "行顺序即优先级——第一个 LoRA 保持原样、后续让位、可能损失幅度；"
        "仅 UNet 层正交化，CLIP 仍按顺序叠加）。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "每个开着的 LoRA 都将应用到的扩散模型。"}),
            },
            "optional": {
                "clip": ("CLIP", {"tooltip": "LoRA 应用到的 CLIP（文本编码器）。可选但建议连接（checkpoint CLIP 进这里，CLIP 输出再去你的文本编码），这样 LoRA 也能调整触发词如何被读取。仅模型设置时可留空。"}),
                "preset": ("SF_LORA_PRESET", {"tooltip": "SF Power Lora Preset 的选择输出。连接后自动把预设的顺序与强度加载到行上；执行时预设优先（行状态仅保留同名行的触发词勾选）。"}),
            },
            "hidden": {"LoraLoaderState": ("STRING", {"default": "{}"})},
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "triggers")
    OUTPUT_TOOLTIPS = (
        "按行顺序应用了每个开着 LoRA 的模型。",
        "应用了每个开着 LoRA 的 CLIP（未接 CLIP 时原样直通）。",
        "你勾选、且所在行处于开启状态的触发词，按分隔符连接的纯文本。",
    )
    FUNCTION = "apply"
    CATEGORY = _CATEGORY

    def __init__(self):
        # path -> (lora_state_dict, lora_metadata)。跨 run 存活的是用户的
        # 选择（齿轮里的 "LoRA memory use"，状态里以 cacheMode 携带）：
        #   "last"（默认）= ComfyUI 对齐，只留最近使用的文件；
        #   "all"  = 保留整个当前栈（重跑最快，大栈可达 GB 级）；
        #   "none" = 什么都不留，每次 run 重读（内存最低）。
        # "last"/"none" 下本次 run 每个条目应用完下一个时立即释放，峰值内存
        # 保持在几个文件而不是整个栈（跨 run 保留条目一直随行到结束）。
        self._cache = {}
        self._last_path = None

    def _get_lora(self, path):
        cached = self._cache.get(path)
        if cached is not None:
            return cached
        try:
            lora, meta = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
        except TypeError:
            # 旧 ComfyUI 没有 return_metadata 参数。没有这个回退每一行都会
            # TypeError，调用方吞掉后节点静默交回未触碰的模型——LoRA 从未应用。
            lora, meta = comfy.utils.load_torch_file(path, safe_load=True), None
        self._cache[path] = (lora, meta)
        return (lora, meta)

    def apply(self, model, clip=None, preset=None, LoraLoaderState="{}"):
        state = R.parse_state(LoraLoaderState)
        if isinstance(preset, dict):
            # 预设优先（Power 同语义）：preset 覆盖行，触发词继承自行状态。
            state = R.preset_override(state, preset)
        cache_mode = state.get("cacheMode", "last")
        # 行解析（文件存在性/强度/override/零强度语义）两路径共用同一过滤。
        plan = self._build_plan(state, clip)

        if state.get("mergeMethod") == "ortho_gs":
            result = self._apply_ortho(model, clip, plan, cache_mode)
            if result is None:
                # key map 构建失败等整体兜底：顺序路径（绝不报错，DuoNodes 同款）。
                result = self._apply_sequential(model, clip, plan, cache_mode)
            model, clip, resolved, used_paths, last_this_run, applied = result
        else:
            model, clip, resolved, used_paths, last_this_run, applied = \
                self._apply_sequential(model, clip, plan, cache_mode)

        # 触发词只来自 resolved 行（collect_triggers 以 `on` 为门槛；每个
        # resolved 行都是开的，所以去重连接它们勾选的词）。
        triggers = R.collect_triggers({"loras": resolved, "sep": state.get("sep", ", ")})

        # 按用户的内存模式修剪（见 __init__）。
        self._trim_cache(cache_mode, used_paths, last_this_run)

        logger.info("[SFLoraStack] applied {} LoRA(s).".format(applied))
        return (model, clip, triggers)

    def _build_plan(self, state, clip):
        """把状态解析为执行计划 [(entry, path, sm, sc, zero)]。

        entry 是归一化行（含 triggers，resolved 收集用）；zero = sm 与 sc 均为 0
        （刻意无操作行）。关闭/占位/文件缺失行在此过滤并打 warning——顺序与
        ortho 两条路径共用同一过滤语义。
        """
        plan = []
        for entry in state["loras"]:
            if not entry.get("on"):
                continue
            name = entry["name"]
            if name == _NO_LORAS:
                continue
            try:
                path = folder_paths.get_full_path("loras", name)
            except Exception:
                path = None
            if not path or not os.path.isfile(path):
                logger.warning("[SFLoraStack] skipped (not found): {}".format(name))
                continue
            sm = float(entry.get("sm", 0.0))
            sc = float(entry.get("sc", 0.0)) if clip is not None else 0.0
            plan.append((entry, path, sm, sc, sm == 0 and sc == 0))
        return plan

    def _apply_sequential(self, model, clip, plan, cache_mode):
        """顺序叠加（默认行为，与 ComfyUI 官方 LoraLoader 逐行等价）。

        返回 (model, clip, resolved, used_paths, last_this_run, applied)；
        实际应用了 LoRA（或故意停在强度 0）的行进 resolved——只有这些行贡献
        触发词，所以 triggers 输出永远不会声称一个不在模型里的 LoRA 的词。
        """
        last_this_run = None
        resolved = []
        used_paths = set()
        applied = 0
        for entry, path, sm, sc, zero in plan:
            if zero:
                # 刻意无操作（文件在、强度零）：留在缓存里，让勾选的触发词
                # 计数（用户故意开着它）。
                used_paths.add(path)
                resolved.append(entry)
                continue
            try:
                lora, meta = self._get_lora(path)
                try:
                    model, clip = comfy.sd.load_lora_for_models(
                        model, clip, lora, sm, sc, lora_metadata=meta
                    )
                except TypeError:
                    # 旧 ComfyUI：没有 lora_metadata 参数。去掉重试，让 LoRA
                    # 仍能应用而不是静默什么都不做。
                    model, clip = comfy.sd.load_lora_for_models(model, clip, lora, sm, sc)
                used_paths.add(path)
                applied += 1
                resolved.append(entry)  # 实际应用 -> 触发词计数
                if cache_mode != "all":
                    # 释放本次 run 加载的上一行，让 10 行栈的峰值停在几个
                    # 文件而不是十个。跨 run 保留条目（self._last_path）
                    # 刻意不在这里碰——它后面可能还被复用。
                    if last_this_run is not None and last_this_run != path:
                        self._cache.pop(last_this_run, None)
                    last_this_run = path
            except Exception as exc:
                # 加载失败 -> 不进 resolved，词到不了输出。
                logger.warning("[SFLoraStack] failed to apply {}: {}".format(entry["name"], exc))
        return model, clip, resolved, used_paths, last_this_run, applied

    def _apply_ortho(self, model, clip, plan, cache_mode):
        """ortho_gs 模式：UNet 侧同 key 多 LoRA 先做 Gram-Schmidt 正交化。

        与 comfy.sd.load_lora_for_models 对齐的加载路径：model_lora_keys_unet /
        model_lora_keys_clip 建 key map、convert_lora 转换、load_lora 解析、
        clone + add_patches + set_attachments("lora_metadata")。区别：
          - UNet 侧：同一模型 key 的多个 LoRA down 矩阵提取后正交化再替换
            （首行不动，后续让位），一次 clone 全栈应用；非 LoRA patch
            （conv/diff/set 等）或提取失败 -> 该 key 全部顺序叠加；
          - CLIP 侧：仍按行顺序叠加（模型侧正交化，文本编码器不参与）；
          - 全部 LoRA sd 驻留到本 run 结束（分组需要），随后按 cacheMode
            修剪——ortho 模式峰值内存 = 整个栈（与 "last" 的逐行释放不同）。
        返回 None 表示 key map 构建失败（调用方整体 fallback 顺序路径）；
        单行加载失败只跳过该行（不进 resolved，触发词不计入）。
        """
        import comfy.lora
        import comfy.lora_convert

        try:
            unet_key_map = comfy.lora.model_lora_keys_unet(model.model, {})
        except Exception as exc:
            logger.warning("[SFLoraStack] ortho: model key map failed ({}); "
                           "falling back to sequential".format(exc))
            return None
        if clip is not None:
            try:
                clip_key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, {})
            except Exception as exc:
                logger.warning("[SFLoraStack] ortho: clip key map failed ({}); "
                               "falling back to sequential".format(exc))
                return None
        else:
            clip_key_map = None

        # 逐行加载（复用 _cache 的 sd，免二次读盘）。
        unet_entries = []  # [(patch_dict, sm, meta)] —— sm != 0 的行
        clip_entries = []  # [(patch_dict, sc, meta)] —— clip 且 sc != 0 的行
        resolved = []
        used_paths = set()
        last_this_run = None
        applied = 0
        for entry, path, sm, sc, zero in plan:
            if zero:
                # 与顺序路径同语义：刻意无操作行算 resolved（触发词计数）。
                used_paths.add(path)
                resolved.append(entry)
                continue
            try:
                lora_sd, meta = self._get_lora(path)
                # 与官方 load_lora_for_models 对齐（DuoNodes 漏掉的步骤）。
                lora_sd = comfy.lora_convert.convert_lora(lora_sd)
            except Exception as exc:
                logger.warning("[SFLoraStack] ortho: failed to load {}: {}".format(entry["name"], exc))
                continue
            ok = False
            if sm != 0:
                try:
                    patches = comfy.lora.load_lora(lora_sd, unet_key_map)
                except Exception as exc:
                    logger.warning("[SFLoraStack] ortho: failed to parse {}: {}".format(entry["name"], exc))
                    patches = None
                if patches is not None:
                    unet_entries.append((patches, sm, meta))
                    ok = True
            if clip is not None and clip_key_map is not None and sc != 0:
                try:
                    patches = comfy.lora.load_lora(lora_sd, clip_key_map)
                except Exception as exc:
                    logger.warning("[SFLoraStack] ortho: failed to parse {} (clip): {}".format(entry["name"], exc))
                    patches = None
                if patches is not None:
                    clip_entries.append((patches, sc, meta))
                    ok = True
            if ok:
                used_paths.add(path)
                resolved.append(entry)
                applied += 1
                # 与顺序路径不同：ortho 需要全部 sd 驻留到分组完成，循环内
                # 不逐出；last_this_run 只作 run 后修剪的游标。
                last_this_run = path

        if unet_entries:
            # 按模型 key 分组（同一 key 的多个 LoRA 才需要正交化）。
            key_to_entries = {}  # key -> [(patch, sm)]（栈顺序）
            for patches, sm, _meta in unet_entries:
                for key, patch in patches.items():
                    key_to_entries.setdefault(key, []).append((patch, sm))

            # 每 key 预处理：ortho_by_key[key] 为 None 表示直通原 patch；
            # 否则是 {原 patch: 替换后 patch}（同一 key 的多 LoRA 正交化产物）。
            ortho_by_key = {}
            ortho_keys = 0
            for key, entries in key_to_entries.items():
                if len(entries) == 1:
                    ortho_by_key[key] = None
                    continue
                components = []  # [(patch, down)]
                fallback = False
                for patch, _sm in entries:
                    up, down = LO.extract_up_down(patch)
                    if up is None or down is None or down.dim() != 2:
                        # conv 等非 LoRA patch：该 key 全部顺序叠加。
                        fallback = True
                        break
                    components.append((patch, down))
                if fallback:
                    ortho_by_key[key] = None
                    continue
                ortho_downs = LO.gram_schmidt_ortho_downs([d for _p, d in components])
                ortho_by_key[key] = {
                    p: LO.replace_down(p, od.to(d.dtype))
                    for (p, d), od in zip(components, ortho_downs)
                }
                ortho_keys += 1
            logger.info("[SFLoraStack] ortho: {} key(s) orthogonalized, {} pass-through"
                        .format(ortho_keys, len(key_to_entries) - ortho_keys))

            new_model = model.clone()
            for patches, sm, meta in unet_entries:
                replaced = {}
                for key, patch in patches.items():
                    entry_map = ortho_by_key[key]
                    replaced[key] = patch if entry_map is None else entry_map[patch]
                new_model.add_patches(replaced, sm)
                if meta:
                    new_model.set_attachments("lora_metadata", meta)
        else:
            new_model = model

        if clip is not None and clip_entries:
            new_clip = clip.clone()
            for patches, sc, meta in clip_entries:
                new_clip.add_patches(patches, sc)
                if meta:
                    new_clip.patcher.set_attachments("lora_metadata", meta)
        else:
            new_clip = clip

        return new_model, new_clip, resolved, used_paths, last_this_run, applied

    def _trim_cache(self, cache_mode, used_paths, last_this_run):
        """按用户的内存模式修剪（last/all/none 语义见 __init__）。"""
        if cache_mode == "none":
            self._cache.clear()
            self._last_path = None
        elif cache_mode == "all":
            # 释放用户删掉的 LoRA 条目，让内存跟随节点。
            for path in list(self._cache):
                if path not in used_paths:
                    del self._cache[path]
        else:  # "last"：ComfyUI 对齐——最多一个条目活过本次 run：
            # 本次 run 最近一次加载。本次没加载任何东西时，先前保留的文件
            # 只在它仍是栈的一部分时存活（强度 0 的行也算）——清空的栈
            # 真的释放它。
            keep = last_this_run
            if keep is None and self._last_path in used_paths:
                keep = self._last_path
            for path in list(self._cache):
                if path != keep:
                    del self._cache[path]
            self._last_path = keep


# 导入以触发 LoRA 栈 API 路由注册（lora_notes 同款先例）
from ...sf_utils import lora_routes  # noqa: F401, E402
