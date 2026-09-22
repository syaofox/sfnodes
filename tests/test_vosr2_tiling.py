# VOSR2 分块几何纯逻辑测试（无 torch 依赖）：
#  - 瓦片网格覆盖完整、去重、末尾贴边、overlap>=tile 不死循环
#  - DiT 潜空间参数按 patch 对齐、clamp 正确
#  - VAE 潜空间参数按 AE_FACTOR 对齐
#  - 瓦片数量 / pad 计算与上游契约一致
# 运行：python tests/test_vosr2_tiling.py
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

failures = []

def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")

spec = importlib.util.spec_from_file_location(
    "vosr2_tiling", os.path.join(root, "nodes/model/vosr2/tiling.py")
)
tiling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tiling)

# ── pad_to_multiple_amount ──
check("pad 0 当已对齐", tiling.pad_to_multiple_amount(512) == 0)
check("pad 16-倍数补足", tiling.pad_to_multiple_amount(513) == 15)
check("pad 小尺寸", tiling.pad_to_multiple_amount(1) == 15)
check("pad 8 倍数但非 16", tiling.pad_to_multiple_amount(8) == 8)

# ── make_tile_grid ──
check("短边单块", tiling.make_tile_grid(10, 16, 4) == [0])
grid = tiling.make_tile_grid(100, 32, 8)
check("网格首块 0", grid[0] == 0)
check("网格末块贴边", grid[-1] == 100 - 32)
check("网格覆盖完整", all(grid[i + 1] - grid[i] <= 32 - 8 for i in range(len(grid) - 1)))
check("网格去重有序", grid == sorted(set(grid)))
oversized_overlap = tiling.make_tile_grid(100, 32, 64)
check("overlap>=tile 退化为 stride 1 且不挂", oversized_overlap[0] == 0 and oversized_overlap[-1] == 68)
check("整除无多余块", tiling.make_tile_grid(64, 32, 0) == [0, 32])

# ── dit_tile_params：像素 512 / overlap 32 → 潜空间 64 / 4 ──
size, overlap = tiling.dit_tile_params(512, 32, 128, 128)
check("DiT 瓦片潜空间边长", size == 64)
check("DiT 瓦片重叠", overlap == 8)
# 对齐到 patch=2：像素 520 → 65 → 向下取偶 64
size, _ = tiling.dit_tile_params(520, 0, 128, 128)
check("DiT 瓦片按 patch 对齐", size == 64)
# 最小 patch
size, overlap = tiling.dit_tile_params(8, 0, 128, 128)
check("DiT 瓦片下限 1 patch", size == 2)
check("DiT 重叠 <= size-1", overlap <= size - 1)
# 小于整图时钳到整图
size, _ = tiling.dit_tile_params(4096, 0, 32, 16)
check("DiT 瓦片钳到整图", size == 16)

# ── vae_tile_params：像素 1024 / overlap 32 → 潜空间 128 / max(4, 128//8) ──
size, overlap = tiling.vae_tile_params(1024, 32, 256, 256)
check("VAE 瓦片潜空间边长", size == 128)
check("VAE 瓦片重叠下限 tile/8", overlap == 16)
size, _ = tiling.vae_tile_params(1020, 0, 256, 256)
check("VAE 瓦片向下取整", size == 127)

# ── 位置 / 数量 ──
positions = tiling.dit_tile_positions(512, 512, 512, 32)
check("整图单瓦片", positions == [(0, 0)])
positions = tiling.dit_tile_positions(1536, 1536, 1024, 128)
check("2x2 瓦片", len(positions) == 4)
check("瓦片位置去重", len(set(positions)) == len(positions))
check("瓦片数一致", tiling.dit_tile_count(1536, 1536, 1024, 128) == len(positions))

# 非方形：只在一个方向分块（高度整图一块、宽度多块）
positions = tiling.dit_tile_positions(512, 1536, 512, 64)
check("非方形单向分块", {p[0] for p in positions} == {0} and len({p[1] for p in positions}) > 1)

# ── needs_dit_tiling ──
check("512 输出不分块", not tiling.needs_dit_tiling(512, 512, 512))
check("1024 输出需分块", tiling.needs_dit_tiling(1024, 1024, 512))
check("tile_size=0 不分块", not tiling.needs_dit_tiling(4096, 4096, 0))
check("非方形长边超限需分块", tiling.needs_dit_tiling(512, 1024, 512))

# ── 常量契约 ──
check("PAD_MULTIPLE=16", tiling.PAD_MULTIPLE == 16)
check("latent_size", tiling.latent_size(1024, 512) == (128, 64))

if failures:
    print(f"\n{len(failures)} 项失败: {failures}")
    sys.exit(1)
print("\n全部通过")
