"""SFLoraStack —— 多行 LoRA 栈节点（复刻 PixaromaLoraLoader）。

前端驱动（Vue Compat #9）：每个 LoRA 行（文件名、开关、model/clip 强度、用户
勾选的触发词）存在浏览器 node.properties.loraStackState，由 graphToPrompt 钩子
注入隐藏 LoraLoaderState 输入（web/sf_lora_stack.js）。因为 LoraLoaderState 是
节点输入的一部分，编辑一行就改变节点缓存签名，运行总是拿到新值，无需
IS_CHANGED。

Python 职责很小：对每个开着的行，用各自强度把 LoRA 应用到 MODEL（和 CLIP），
链式叠加，并把用户勾选的触发词连接成 `triggers` STRING 输出。元数据/触发词
读取的离线逻辑在 sf_utils/lora_reader.py，Civitai 等路由在 sf_utils/lora_routes.py。
"""
import os

import folder_paths
import comfy.sd
import comfy.utils

from ...sf_utils import lora_reader as R
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
        # 本次 run 实际加载的最近一行。与 self._last_path（上一 run 保留的条目）
        # 分开——本 run 第一行应用时就逐出保留条目，会让 "last" 对任何
        # 2+ 行的栈表现得像 "none"（暖文件在被复用前一刻被丢掉）。
        last_this_run = None

        # 实际应用了 LoRA（或故意停在强度 0）的行。只有这些行贡献触发词，
        # 所以 triggers 输出永远不会声称一个不在模型里的 LoRA 的词——缺失/
        # 改名文件或加载失败（损坏/不兼容）什么都不加。
        resolved = []
        used_paths = set()
        applied = 0
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
            if sm == 0 and sc == 0:
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
                logger.warning("[SFLoraStack] failed to apply {}: {}".format(name, exc))

        # 触发词只来自 resolved 行（collect_triggers 以 `on` 为门槛；每个
        # resolved 行都是开的，所以去重连接它们勾选的词）。
        triggers = R.collect_triggers({"loras": resolved, "sep": state.get("sep", ", ")})

        # 按用户的内存模式修剪（见 __init__）。
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

        logger.info("[SFLoraStack] applied {} LoRA(s).".format(applied))
        return (model, clip, triggers)


# 导入以触发 LoRA 栈 API 路由注册（lora_notes 同款先例）
from ...sf_utils import lora_routes  # noqa: F401, E402
