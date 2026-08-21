"""LoRA 正交堆叠加载路径（SFLoraStack SFLoraStack 专用，规则 14）。

与 comfy.sd.load_lora_for_models 对齐的加载路径：model_lora_keys_unet /
model_lora_keys_clip 建 key map、convert_lora 转换、load_lora 解析；UNet 侧
按模型 key 分组做 Gram-Schmidt 正交化（lora_ortho.build_ortho_replacements），
clone + add_patches + set_attachments("lora_metadata")；CLIP 侧仍顺序叠加。

依赖 comfy.lora / comfy.lora_convert（运行时由 ComfyUI 提供）；文件读盘由
调用方的 load_sd 回调负责（folder_paths 不在此 import）；数学与 patch 格式
探测在 lora_ortho.py（仅 torch，可在 ComfyUI 之外单测）。
"""
import comfy.lora
import comfy.lora_convert

from .lora_ortho import build_ortho_replacements
from .logger import get_logger

logger = get_logger(__name__)


def ortho_apply(model, clip, entries, load_sd):
    """Gram-Schmidt 正交化应用路径（key map 构建失败返回 None，调用方 fallback）。

    entries: [(name, path, sm, sc)] —— 栈顺序（第一个 LoRA 不动），sm/sc 为
      最终应用强度（调用方归一化后）；sm=0 的行跳过 model 侧、sc=0 跳过
      clip 侧。零强度行应在调用方预处理（本函数不处理）。
    load_sd: callable(path) -> (sd, meta) —— 调用方决定缓存策略
      （SFLoraStack 传 self._get_lora 复用缓存；Power 传直接读盘）。
    返回 (new_model, new_clip, ok_paths, (ortho_keys, pass_keys))：
      ok_paths = 成功应用的行 path 集合（加载/解析失败的行不计入）；
      stats 供日志展示。返回 None = key map 构建失败（整体 fallback 顺序）。
    失败语义：单行加载/解析失败只跳过该行，绝不报错。
    """
    try:
        unet_key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    except Exception as exc:
        logger.warning("[sfnodes ortho] model key map failed ({}); "
                       "falling back to sequential".format(exc))
        return None
    if clip is not None:
        try:
            clip_key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, {})
        except Exception as exc:
            logger.warning("[sfnodes ortho] clip key map failed ({}); "
                           "falling back to sequential".format(exc))
            return None
    else:
        clip_key_map = None

    # 逐行加载（load_sd 由调用方提供缓存/读盘策略）。
    unet_entries = []  # [(patch_dict, sm, meta, path)]
    clip_entries = []  # [(patch_dict, sc, meta, path)]
    ok_paths = set()
    for name, path, sm, sc in entries:
        try:
            lora_sd, meta = load_sd(path)
            # 与官方 load_lora_for_models 对齐（DuoNodes 漏掉的步骤）。
            lora_sd = comfy.lora_convert.convert_lora(lora_sd)
        except Exception as exc:
            logger.warning("[sfnodes ortho] failed to load {}: {}".format(name, exc))
            continue
        ok = False
        if sm != 0:
            try:
                patches = comfy.lora.load_lora(lora_sd, unet_key_map)
            except Exception as exc:
                logger.warning("[sfnodes ortho] failed to parse {}: {}".format(name, exc))
                patches = None
            if patches is not None:
                unet_entries.append((patches, sm, meta, path))
                ok = True
        if clip is not None and clip_key_map is not None and sc != 0:
            try:
                patches = comfy.lora.load_lora(lora_sd, clip_key_map)
            except Exception as exc:
                logger.warning("[sfnodes ortho] failed to parse {} (clip): {}".format(name, exc))
                patches = None
            if patches is not None:
                clip_entries.append((patches, sc, meta, path))
                ok = True
        if ok:
            ok_paths.add(path)

    if unet_entries:
        replaced, ortho_keys, pass_keys = build_ortho_replacements(
            [(p, s) for p, s, _m, _x in unet_entries]
        )
        new_model = model.clone()
        for (patches, sm, meta, _path), (new_dict, _s) in zip(unet_entries, replaced):
            new_model.add_patches(new_dict, sm)
            if meta:
                new_model.set_attachments("lora_metadata", meta)
    else:
        new_model = model
        ortho_keys = 0
        pass_keys = 0

    if clip is not None and clip_entries:
        new_clip = clip.clone()
        for patches, sc, meta, _path in clip_entries:
            new_clip.add_patches(patches, sc)
            if meta:
                new_clip.patcher.set_attachments("lora_metadata", meta)
    else:
        new_clip = clip

    return new_model, new_clip, ok_paths, (ortho_keys, pass_keys)
