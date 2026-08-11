"""SF Image Tile / SF Image Untile — 移植自 cubiq/ComfyUI_essentials
image.py 的 ImageTile / ImageUntile。

Tile: 把图片按 rows×cols 切成均匀块，支持重叠（FLOAT 比例 + INT 像素相加，
自动钳制到块尺寸一半；单行/列对应方向重叠清零）。块按行优先顺序拼接成批次，
并输出 "SF_TILE_INFO" 自定义线型，携带网格几何与原始大小等信息。

Untile: 把批次按行优先顺序贴回网格，重叠区顶部/左侧羽化混合（linspace 渐变），
输出单帧。rows/cols/重叠量全部从 tile_info 读取（不再手填）；若块被放大/缩小
过（尺寸与原始块尺寸不符），先 bilinear 恢复回原始块尺寸再合并，输出尺寸恒为
原始画布 out_w × out_h。

几何计算收敛于 sf_utils/tiling.py（纯逻辑，两侧共用保证契约一致）。
纯 torch，无磁盘、无 JS。
"""

import torch
import torch.nn.functional as F

from nodes import MAX_RESOLUTION

from ...sf_utils.tiling import tile_plan, tile_rects

_CATEGORY = "sfnodes/image"

# 自定义线型：携带 Untile 需要的一切——网格几何（rows/cols）、净块尺寸、
# 实际重叠量、截断后画布尺寸与原始图片尺寸。纯 dict（int 标量，无张量），
# 与节点类保持解耦。Untile 必须接此线才能还原。
SF_TILE_INFO = "SF_TILE_INFO"


class SFImageTile:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "要切块的图片。任意 IMAGE 源都可接入（LoadImage、VAE Decode、生成结果等）。",
                }),
                "rows": ("INT", {
                    "default": 2, "min": 1, "max": 256, "step": 1,
                    "tooltip": "垂直方向切几行。行数越多每块越矮；图片高度不能整除时，底部不足一块的部分会被丢弃。单行时垂直重叠自动清零。",
                }),
                "cols": ("INT", {
                    "default": 2, "min": 1, "max": 256, "step": 1,
                    "tooltip": "水平方向切几列。列数越多每块越窄；图片宽度不能整除时，右侧不足一块的部分会被丢弃。单列时水平重叠自动清零。",
                }),
                "overlap": ("FLOAT", {
                    "default": 0, "min": 0, "max": 0.5, "step": 0.01,
                    "tooltip": "块之间重叠的比例（相对块尺寸，0~0.5）。与 overlap_x/y 的像素值相加得到实际重叠量。重叠让相邻块共享边缘像素，Untile 还原时接缝会被羽化混合，适合局部重绘场景。",
                }),
                "overlap_x": ("INT", {
                    "default": 0, "min": 0, "max": MAX_RESOLUTION // 2, "step": 1,
                    "tooltip": "水平重叠的固定像素数，叠加在 overlap 比例之上。实际重叠量会被钳制到块宽的一半以内。",
                }),
                "overlap_y": ("INT", {
                    "default": 0, "min": 0, "max": MAX_RESOLUTION // 2, "step": 1,
                    "tooltip": "垂直重叠的固定像素数，叠加在 overlap 比例之上。实际重叠量会被钳制到块高的一半以内。",
                }),
                "as_list": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "tiles 输出的形态：关闭 = 批次（batch，可直接接 VAE Encode、Save Image 等普通节点）；打开 = 图片列表（每帧一项，ComfyUI 会为下游逐项执行，适合逐块单独处理如循环的场景）。",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", SF_TILE_INFO)
    RETURN_NAMES = ("tiles", "tile_info")
    OUTPUT_IS_LIST = (True, False)
    OUTPUT_TOOLTIPS = (
        "全部切块，按行优先顺序排列（第 0 行从左到右，然后第 1 行……）。"
        "每块尺寸为原始块尺寸（净块 + 实际重叠），总帧数 = 输入帧数 × rows × cols。"
        "形态由 as_list 开关决定：关闭 = 批次，打开 = 逐帧列表（下游逐项执行）。",
        "供 SF Image Untile 使用的信息——携带网格几何（rows/cols）、净块尺寸、"
        "实际重叠量、截断后画布尺寸与原始图片尺寸。可选，不接也可。",
    )
    FUNCTION = "tile"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "把图片按 rows×cols 切成均匀块，输出为一个批次（行优先顺序）。\n\n"
        "overlap（0~0.5，块尺寸的比例）与 overlap_x/y（像素）相加得到实际重叠量，"
        "自动钳制到块尺寸的一半以内；单行/单列时对应方向的重叠清零。"
        "图片尺寸不能整除 rows×cols 时，右/下边缘像素被丢弃。\n\n"
        "tiles 输出的形态由 as_list 开关决定：关闭（默认）= 批次，可直接接普通"
        "节点；打开 = 图片列表，ComfyUI 会为下游逐项执行——适合逐块单独处理"
        "（如循环）的场景。两种形态都可直接接入 SF Image Untile 还原。\n\n"
        "tile_info 携带网格几何与原始大小等信息，直接接给 SF Image Untile 即可"
        "还原；重叠区域的接缝会被羽化混合。即使块被放大/缩小过（如经局部重绘、"
        "超分等中间处理），Untile 也会按 tile_info 自动恢复为原始尺寸。"
    )

    def tile(self, image, rows, cols, overlap, overlap_x, overlap_y, as_list=False):
        h, w = image.shape[1:3]
        plan = tile_plan(h, w, rows, cols, overlap, overlap_x, overlap_y)

        tiles = [
            image[:, y1:y2, x1:x2, :]
            for (y1, x1, y2, x2) in plan["rects"]
        ]
        tiles = torch.cat(tiles, dim=0)

        # OUTPUT_IS_LIST 声明下：批次形态必须包成单元素列表（元素是完整批次），
        # 下游逐项执行一次即等效 batch 直连；列表形态返回逐帧。
        tiles_out = (
            [tiles[i].unsqueeze(0) for i in range(tiles.shape[0])]
            if as_list else [tiles]
        )

        info = {
            "rows": rows,
            "cols": cols,
            "tile_w": plan["tile_w"],
            "tile_h": plan["tile_h"],
            "overlap_w": plan["overlap_w"],
            "overlap_h": plan["overlap_h"],
            "out_w": plan["tile_w"] * cols,
            "out_h": plan["tile_h"] * rows,
            "orig_w": int(w),
            "orig_h": int(h),
        }
        return (tiles_out, info)


class SFImageUntile:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tiles": ("IMAGE", {
                    "tooltip": "SF Image Tile 输出的块——批次（tiles）或图片列表（tiles_list）均可直接接入，列表会自动合并为批次。帧数必须 ≥ rows × cols，多余的帧会被忽略（多帧输入经 Tile 后帧序为逐块展开，原版行为）。",
                }),
                "tile_info": (SF_TILE_INFO, {
                    "tooltip": "接入 SF Image Tile 的 tile_info 输出。携带网格几何、块尺寸、重叠量与原始大小；块被放大/缩小过也会按其自动恢复。",
                }),
                "resize_to_original": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "块被中间处理放大/缩小过（如局部重绘、超分）时的处理方式：开启（默认）= 双线性恢复回原始块尺寸再合并，输出恒为原始画布尺寸；关闭 = 保持当前尺寸合并，输出按当前块尺寸推导的画布（如 2x 放大后拼出 2x 大图）。关闭时列表内块尺寸必须一致，且块尺寸需大于重叠量。",
                }),
            },
        }

    # 声明后节点只执行一次、所有输入以列表传入（普通输出包装成单元素列表，
    # 列表输出原样传入）——batch 与 list 两种形态在 untile() 中统一合并。
    INPUT_IS_LIST = True

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_TOOLTIPS = (
        "还原后的单帧图片：把块批次按行优先顺序贴回网格，重叠区顶部/左侧羽化混合。"
        "输出尺寸由 resize_to_original 决定：开启 = 原始画布（Tile 截断后的尺寸）；"
        "关闭 = 当前块尺寸推导的画布。原始图不可整除丢弃的边缘无法恢复。",
    )
    FUNCTION = "untile"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "把 SF Image Tile 输出的块按行优先顺序贴回网格，还原为一张图。\n\n"
        "网格几何（rows/cols）、块尺寸与重叠量全部从 tile_info 读取，不再需要"
        "手填。若块在中间被放大/缩小过（如局部重绘、超分等处理改变了尺寸）：\n"
        "  - resize_to_original 开启（默认）：自动以双线性插值恢复回原始块尺寸"
        "再合并，输出恒为原始画布尺寸；\n"
        "  - 关闭：保持当前尺寸合并，输出按当前块尺寸推导的画布（如 2x 超分后"
        "拼出 2x 大图）。\n\n"
        "tiles 接受批次或列表两种输入：SF Image Tile 的 tiles（批次）与 "
        "tiles_list（列表）输出都可直接接入同一个输入，列表自动合并为批次——"
        "适合块经列表型节点（如循环）逐块处理后再还原的场景。\n\n"
        "重叠区域的顶部/左侧做羽化混合（渐入），消除接缝。"
    )

    def _resize_bhwc(self, t, target_w, target_h):
        """把图像张量 [B,H,W,C] 缩放到 [B,target_h,target_w,C]。双线性，
        与 SFImageOutpaintStitch 的 _resize_bhwc 一致，两节点缩放行为相同。"""
        x = t.permute(0, 3, 1, 2)  # [B,C,H,W]
        x = F.interpolate(x, size=(int(target_h), int(target_w)),
                          mode="bilinear", align_corners=False)
        return x.permute(0, 2, 3, 1).contiguous()

    @staticmethod
    def _first(x):
        """INPUT_IS_LIST 解包：所有输入以列表传入（未连接为 (None,)，普通输出
        包装为 [value]，列表输出原样）。取首项；None 原样返回。"""
        if isinstance(x, (list, tuple)) and len(x) > 0:
            return x[0]
        return x

    def untile(self, tiles, tile_info, resize_to_original=True):
        tile_info = self._first(tile_info)
        resize_to_original = self._first(resize_to_original)
        if resize_to_original is None:
            resize_to_original = True
        resize_to_original = bool(resize_to_original)

        if not isinstance(tile_info, dict):
            raise ValueError(
                "tile_info 缺失或非法：请接入 SF Image Tile 的 tile_info 输出，"
                "或重新运行 Tile 节点"
            )
        missing = {"rows", "cols", "tile_w", "tile_h", "overlap_w", "overlap_h",
                   "out_w", "out_h"} - set(tile_info.keys())
        if missing:
            raise ValueError(
                f"tile_info 缺少字段 {sorted(missing)}：请接入 SF Image Tile 的 "
                f"tile_info 输出，或重新运行 Tile 节点"
            )

        rows = int(tile_info["rows"])
        cols = int(tile_info["cols"])
        tile_h = int(tile_info["tile_h"])
        tile_w = int(tile_info["tile_w"])
        overlap_h = int(tile_info["overlap_h"])
        overlap_w = int(tile_info["overlap_w"])
        out_h = int(tile_info["out_h"])
        out_w = int(tile_info["out_w"])

        if tile_h <= 0 or tile_w <= 0:
            raise ValueError(
                f"tile_info 块尺寸非法：净块 {tile_h}×{tile_w}，"
                f"overlap 不能超过块尺寸"
            )

        # tiles 归一：batch（包装成单元素列表）与列表输出统一合并为批次。
        # resize_to_original 开启时，列表内块尺寸不一致（逐块处理改变了尺寸）
        # 先按原始块尺寸统一再合并；关闭时尺寸必须一致。
        orig_block_h = tile_h + overlap_h
        orig_block_w = tile_w + overlap_w
        if isinstance(tiles, (list, tuple)):
            items = [t for t in tiles if t is not None]
            if not items:
                raise ValueError(
                    "tiles 未连接：请接入 SF Image Tile 的 tiles 或 tiles_list 输出"
                )
            sizes = {(t.shape[1], t.shape[2]) for t in items}
            if len(sizes) > 1:
                if resize_to_original:
                    items = [self._resize_bhwc(t, orig_block_w, orig_block_h) for t in items]
                else:
                    raise ValueError(
                        "关闭缩放回原始大小（resize_to_original = 关）时，列表内"
                        "块尺寸必须一致；尺寸不一致请开启缩放，或先统一块尺寸"
                    )
            tiles = torch.cat(items, dim=0)
        else:
            tiles = self._first(tiles)
        if tiles is None:
            raise ValueError(
                "tiles 未连接：请接入 SF Image Tile 的 tiles 或 tiles_list 输出"
            )

        need = rows * cols
        if tiles.shape[0] < need:
            raise ValueError(
                f"tiles 帧数不足：需要 {need} 帧（{rows} 行 × {cols} 列），"
                f"实际只有 {tiles.shape[0]} 帧"
            )

        # 几何决策：resize_to_original 开启 = 恢复回原始块尺寸、原始画布；
        # 关闭 = 按当前块尺寸推导净块与画布。关闭时重叠量必须随块按比例缩放
        # （块 2x 放大后其重叠区也是 2x）——cell 按比例取整、ov 由
        # cur - cell 补足，保证 cell + ov == 当前块尺寸恒等（几何自洽）。
        cur_h, cur_w = tiles.shape[1:3]
        if resize_to_original:
            if (cur_h, cur_w) != (orig_block_h, orig_block_w):
                if cur_h <= 0 or cur_w <= 0:
                    raise ValueError(
                        f"tiles 尺寸非法：块 {cur_h}×{cur_w}"
                    )
                tiles = self._resize_bhwc(tiles, orig_block_w, orig_block_h)
            cell_h, cell_w = tile_h, tile_w
            ov_h, ov_w = overlap_h, overlap_w
            canvas_h, canvas_w = out_h, out_w
        else:
            scale_h = cur_h / orig_block_h
            scale_w = cur_w / orig_block_w
            cell_h = round(tile_h * scale_h)
            cell_w = round(tile_w * scale_w)
            ov_h = cur_h - cell_h
            ov_w = cur_w - cell_w
            if cell_h <= 0 or cell_w <= 0 or ov_h < 0 or ov_w < 0:
                raise ValueError(
                    f"关闭缩放时块尺寸异常：块 {cur_h}×{cur_w}，"
                    f"原始块 {orig_block_h}×{orig_block_w}，"
                    f"推导净块 {cell_h}×{cell_w}"
                )
            canvas_h = rows * cell_h
            canvas_w = cols * cell_w

        rects = tile_rects(
            rows, cols, cell_h, cell_w, ov_h, ov_w, canvas_h, canvas_w
        )

        out = torch.zeros(
            (1, canvas_h, canvas_w, tiles.shape[3]),
            device=tiles.device,
            dtype=tiles.dtype,
        )

        for n, (y1, x1, y2, x2) in enumerate(rects):
            i, j = divmod(n, cols)
            mask = torch.ones(
                (1, cell_h + ov_h, cell_w + ov_w),
                device=tiles.device,
                dtype=tiles.dtype,
            )
            if i > 0 and ov_h > 0:
                mask[:, :ov_h, :] *= torch.linspace(
                    0, 1, ov_h, device=tiles.device, dtype=tiles.dtype
                ).unsqueeze(1)
            if j > 0 and ov_w > 0:
                mask[:, :, :ov_w] *= torch.linspace(
                    0, 1, ov_w, device=tiles.device, dtype=tiles.dtype
                ).unsqueeze(0)

            mask = mask.unsqueeze(-1).repeat(1, 1, 1, tiles.shape[3])
            tile = tiles[n] * mask
            out[:, y1:y2, x1:x2, :] = out[:, y1:y2, x1:x2, :] * (1 - mask) + tile

        return (out,)


class SFImageTileInfo:
    """解析 SF Image Tile 的 tile_info，输出各项数字（含分块数）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tile_info": (SF_TILE_INFO, {
                    "tooltip": "接入 SF Image Tile 的 tile_info 输出。",
                }),
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "INT",
                    "INT", "INT", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("rows", "cols", "tile_w", "tile_h", "overlap_x", "overlap_y",
                    "out_w", "out_h", "orig_w", "orig_h", "block_count",
                    "full_tile_w", "full_tile_h")
    OUTPUT_TOOLTIPS = (
        "网格行数。",
        "网格列数。",
        "净块宽度（不含重叠）。",
        "净块高度（不含重叠）。",
        "实际使用的水平重叠量（钳制后）。",
        "实际使用的垂直重叠量（钳制后）。",
        "截断后的画布宽度（净块宽 × cols），即 SF Image Untile 的输出宽度。",
        "截断后的画布高度（净块高 × rows），即 SF Image Untile 的输出高度。",
        "原始图片完整宽度（未截断，含被丢弃的边缘）。",
        "原始图片完整高度（未截断，含被丢弃的边缘）。",
        "分块数 = rows × cols，即 tiles 批次中 SF Image Untile 实际使用的帧数。",
        "含重叠的完整块宽度（净块宽 + 实际水平重叠），即 tiles 输出中每块的尺寸。",
        "含重叠的完整块高度（净块高 + 实际垂直重叠），即 tiles 输出中每块的尺寸。",
    )
    FUNCTION = "parse"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "解析 SF Image Tile 输出的 tile_info，把网格几何与原始大小等各项数字"
        "拆成独立输出：rows / cols、净块尺寸（不含重叠）、完整块尺寸（净块 + "
        "实际重叠，即 tiles 输出中每块的实际尺寸）、实际重叠量、截断后画布尺寸、"
        "原始图片尺寸，以及分块数（rows × cols，即 Untile 需要的 tiles 帧数）。\n\n"
        "输出均为 INT，可直接接入数字计算、尺寸计算或文本拼接等节点。"
    )

    def parse(self, tile_info):
        if not isinstance(tile_info, dict):
            raise ValueError(
                "tile_info 缺失或非法：请接入 SF Image Tile 的 tile_info 输出，"
                "或重新运行 Tile 节点"
            )
        missing = {"rows", "cols", "tile_w", "tile_h", "overlap_w", "overlap_h",
                   "out_w", "out_h", "orig_w", "orig_h"} - set(tile_info.keys())
        if missing:
            raise ValueError(
                f"tile_info 缺少字段 {sorted(missing)}：请接入 SF Image Tile 的 "
                f"tile_info 输出，或重新运行 Tile 节点"
            )

        rows = int(tile_info["rows"])
        cols = int(tile_info["cols"])
        tile_w = int(tile_info["tile_w"])
        tile_h = int(tile_info["tile_h"])
        overlap_w = int(tile_info["overlap_w"])
        overlap_h = int(tile_info["overlap_h"])
        return (
            rows,
            cols,
            tile_w,
            tile_h,
            overlap_w,
            overlap_h,
            int(tile_info["out_w"]),
            int(tile_info["out_h"]),
            int(tile_info["orig_w"]),
            int(tile_info["orig_h"]),
            rows * cols,
            tile_w + overlap_w,
            tile_h + overlap_h,
        )
