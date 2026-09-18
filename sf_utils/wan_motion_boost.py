"""Wan I2V 运动增强纯逻辑（无 ComfyUI 依赖，仅 torch）。

`SFWanMotionBoost` 的后处理核心：从原生 `WanImageToVideo` 产出的
conditioning 中取出 `concat_latent_image` / `concat_mask`，对 mask 标记的
灰填充占位帧做**保均值运动缩放**（放大占位帧与最后一个条件帧 latent 的
差异，间接驱动更大动作幅度），缩放后可选 clamp，clamp 开启且
`color_protect` 时再**精确恢复占位帧每通道均值**——抵消截断造成的整段
偏色/偏亮（PainterI2V 系列偏灰偏绿的 latent 层来源）。

仅用 `torch.mean` / `torch.clamp` / `torch.cat` 三个模块级函数与张量
索引/算术，故测试可用 numpy 代理直接验证数学（tests/test_wan_motion_boost.py）。

设计取舍与背景见 doc/experience/nodes-video.md §105。
"""

import torch


def placeholder_start(concat_mask, total_frames):
    """返回首个占位帧下标；结构不适用（未知 mask / 无占位帧 / 非后段连续）时返回 None。

    `concat_mask` 为原生 `WanImageToVideo` 写入的 `[1,1,T,H,W]`：
    0=条件帧（首帧/多帧 start_image），1=占位帧（concat_cond 采样时才取反）。
    无 mask 时按单帧条件返回 1（与 PainterI2V 单帧行为一致）；start=0
    （全占位）不属于原生输出，视为不适用。
    """
    if total_frames <= 1:
        return None
    if concat_mask is None:
        return 1
    if getattr(concat_mask, "ndim", 0) != 5 or int(concat_mask.shape[2]) != total_frames:
        return 1
    flags = [bool(v) for v in (torch.mean(concat_mask, dim=(0, 1, 3, 4)) > 0.5)]
    if not any(flags):
        return None
    start = flags.index(True)
    if start == 0 or not all(flags[start:]) or any(flags[:start]):
        return None
    return start


def boost_concat_latent(concat_latent, concat_mask=None, motion_amplitude=1.15,
                        latent_clamp=6.0, color_protect=True):
    """对占位帧做保均值运动缩放，返回新张量；不适用/幅度<=1 时原样返回入参（不改原张量）。

    基准取**最后一个条件帧**的 latent（多帧 start_image 时为最后一张的 latent，
    单帧时即首帧）：`diff = 占位帧 - base`，去掉每帧均值分量后放大
    `motion_amplitude` 倍再补回 —— 只放大空间结构差异（动作信息），不改变
    颜色/亮度统计。

    - `color_protect=True`（默认）：均值按**每帧每通道**计算（latent 逐通道
      DC 对应颜色/亮度），缩放前后每帧每通道均值严格不变；`latent_clamp`
      截断后再精确恢复一次，消除截断漂移。
    - `color_protect=False`：对齐 PainterI2V 原版——均值跨通道+空间计算
      （per-frame 标量），通道间可能漂移导致偏灰/偏绿，仅供复现/对比。

    `latent_clamp > 0` 时截断绝对值（0=不限制）。
    """
    if concat_latent is None or motion_amplitude <= 1.0:
        return concat_latent
    if getattr(concat_latent, "ndim", 0) != 5:
        return concat_latent
    total = int(concat_latent.shape[2])
    start = placeholder_start(concat_mask, total)
    if start is None or start >= total:
        return concat_latent

    base = concat_latent[:, :, start - 1:start]
    rest = concat_latent[:, :, start:]
    diff = rest - base
    if color_protect:
        diff_mean = torch.mean(diff, dim=(3, 4), keepdim=True)
    else:
        diff_mean = torch.mean(diff, dim=(1, 3, 4), keepdim=True)
    scaled = base + (diff - diff_mean) * motion_amplitude + diff_mean

    limit = float(latent_clamp or 0.0)
    if limit > 0:
        scaled = torch.clamp(scaled, -limit, limit)
        if color_protect:
            scaled = scaled + (torch.mean(rest, dim=(3, 4), keepdim=True)
                               - torch.mean(scaled, dim=(3, 4), keepdim=True))

    return torch.cat([concat_latent[:, :, :start], scaled], dim=2)


def boost_conditioning(conditioning, motion_amplitude=1.15, latent_clamp=6.0,
                       color_protect=True):
    """对 conditioning 中每条带 `concat_latent_image` 的条目做增强。

    同一条 conditioning 内各条目常共享同一张 concat 张量（原生节点对所有
    条目写入同一对象），按 `id` 去重避免重复计算；返回新列表与新 dict，
    不改原输入（未命中的条目 dict 也复制，语义与原生 conditioning_set_values
    的拷贝行为一致）。无 concat 的条目（T2V / 其他模型条件）原样通过。
    """
    if not conditioning:
        return conditioning
    cache = {}
    out = []
    for entry in conditioning:
        tensor, ctx = entry[0], entry[1]
        new_ctx = dict(ctx)
        concat = ctx.get("concat_latent_image")
        if concat is not None:
            key = id(concat)
            boosted = cache.get(key)
            if boosted is None:
                boosted = boost_concat_latent(
                    concat, ctx.get("concat_mask"), motion_amplitude,
                    latent_clamp, color_protect)
                cache[key] = boosted
            if boosted is not concat:
                new_ctx["concat_latent_image"] = boosted
        out.append([tensor, new_ctx])
    return out


__all__ = ["placeholder_start", "boost_concat_latent", "boost_conditioning"]
