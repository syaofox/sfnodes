"""VOSR2 分块几何 — 纯 Python（无 torch 依赖，可在宿主机直接单测）。

统一收敛三处重复实现（ylchen333/ComfyUI-VOSR2 的 tiled_vae._make_tile_grid /
tiled_vae._tile_params / inference._tile_params，Apache-2.0）为单源：

- 瓦片起点网格（`make_tile_grid`）：stride = tile - overlap，末尾补一块贴边，
  去重排序 —— DiT 分块与 VAE 分块共用。
- DiT 潜空间瓦片参数（`dit_tile_params`）：像素 → 潜空间（/8）后再对齐 DiT
  patch（2），重叠至少 1 个 patch。
- VAE 潜空间瓦片参数（`vae_tile_params`）：像素 → 潜空间（/8），无 patch 对齐。

坐标约定：潜空间尺寸 lh/lw 均按「已 pad 到 16 倍数」的像素尺寸 / 8 计算，
与上游契约一致（DiT 输入必须是 patch 整数倍）。
"""

# Qwen-Image 2D VAE 空间压缩比
AE_FACTOR = 8
# LightningDiT patch 尺寸
DIT_PATCH_SIZE = 2
# 像素侧对齐倍数：VAE 压缩 × DiT patch（未分块路径 pad 用）
PAD_MULTIPLE = AE_FACTOR * DIT_PATCH_SIZE


def pad_to_multiple_amount(length, multiple=PAD_MULTIPLE):
    """返回把 length 补齐到 multiple 整数倍所需的像素数（0 表示已对齐）。"""
    return (multiple - length % multiple) % multiple


def make_tile_grid(length, tile, overlap):
    """覆盖 length 的瓦片起点列表（排序去重，末尾贴边）。

    tile >= length 时只有一块 [0]；stride 至少为 1 防止 overlap >= tile 死循环。
    """
    stride = max(tile - overlap, 1)
    if length <= tile:
        return [0]
    positions = list(range(0, length - tile + 1, stride))
    if positions[-1] + tile < length:
        positions.append(length - tile)
    return sorted(set(positions))


def _clamp_tile_params(lt_size, lt_overlap, lh, lw):
    lt_size = min(lt_size, min(lh, lw))
    lt_size = max(lt_size, 1)
    lt_overlap = min(lt_overlap, lt_size - 1)
    lt_overlap = max(lt_overlap, 0)
    return lt_size, lt_overlap


def dit_tile_params(tile_size, tile_overlap, lh, lw):
    """DiT 分块：像素瓦片 → 潜空间并向上对齐到 patch 整数倍。"""
    lt_size = max((tile_size // AE_FACTOR // DIT_PATCH_SIZE) * DIT_PATCH_SIZE, DIT_PATCH_SIZE)
    lt_overlap = max(tile_overlap // AE_FACTOR, lt_size // 8)
    return _clamp_tile_params(lt_size, lt_overlap, lh, lw)


def vae_tile_params(tile_size, tile_overlap, lh, lw):
    """VAE 分块：像素瓦片 → 潜空间（无 patch 对齐）。"""
    lt_size = max(tile_size // AE_FACTOR, 1)
    lt_overlap = max(tile_overlap // AE_FACTOR, lt_size // 8)
    return _clamp_tile_params(lt_size, lt_overlap, lh, lw)


def latent_size(h, w):
    """像素尺寸 → 潜空间尺寸（要求已 pad 到 8 的倍数）。"""
    return h // AE_FACTOR, w // AE_FACTOR


def dit_tile_positions(padded_h, padded_w, tile_size, tile_overlap):
    """DiT 瓦片位置列表 [(hi, wi), ...]；瓦片边长取潜空间参数。"""
    lh, lw = latent_size(padded_h, padded_w)
    lt_size, lt_overlap = dit_tile_params(tile_size, tile_overlap, lh, lw)
    return [(hi, wi) for hi in make_tile_grid(lh, lt_size, lt_overlap)
            for wi in make_tile_grid(lw, lt_size, lt_overlap)]


def dit_tile_count(padded_h, padded_w, tile_size, tile_overlap):
    """DiT 瓦片数量（进度条总量用）。"""
    lh, lw = latent_size(padded_h, padded_w)
    lt_size, lt_overlap = dit_tile_params(tile_size, tile_overlap, lh, lw)
    return len(make_tile_grid(lh, lt_size, lt_overlap)) * len(make_tile_grid(lw, lt_size, lt_overlap))


def needs_dit_tiling(padded_h, padded_w, tile_size):
    """是否必须走 DiT 分块：输出超过 tile_size 时（0 = 不分块由调用方判断）。"""
    if tile_size <= 0:
        return False
    lh, lw = latent_size(padded_h, padded_w)
    lt_size, _ = dit_tile_params(tile_size, 0, lh, lw)
    return lh > lt_size or lw > lt_size
