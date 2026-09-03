# SFImageSceneSplit 后端测试（Python 直接运行：python tests/test_scene_detect.py）
# 覆盖：
#   - sf_utils.scene_detect 纯逻辑：硬切/黑场/白闪/溶解/min_scene_len 去抖
#   - 节点模块（mock torch）：INPUT_TYPES / RETURN_TYPES / 选段负索引/max_frames/越界抛错/all_segments LIST

import os
import sys
import json
import types
import importlib.util

import numpy as np

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

failures = []


def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")


# ── 1. 纯逻辑 ──
from sf_utils.scene_detect import detect_scenes, split_scenes  # noqa: E402


def solid_frame(h, w, rgb):
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :] = np.array(rgb, dtype=np.uint8)
    return arr


def gray_frame(h, w, v):
    return solid_frame(h, w, (v, v, v))


# 硬切：红(5) -> 蓝(5) -> 绿(5)，min_len=1 时应 [0,5,10,15]
frames = [solid_frame(32, 32, (255, 0, 0))] * 5 + [solid_frame(32, 32, (0, 0, 255))] * 5 + [solid_frame(32, 32, (0, 255, 0))] * 5
cuts = detect_scenes(frames, threshold=0.25, min_scene_len=1, method="hist", longest=16)
check("硬切 hist 切点", cuts == [0, 5, 10, 15])

cuts_diff = detect_scenes(frames, threshold=0.20, min_scene_len=1, method="diff", longest=16)
check("硬切 diff 切点", cuts_diff == [0, 5, 10, 15])

# 相同帧无切
frames_same = [solid_frame(32, 32, (100, 100, 100))] * 6
cuts_same = detect_scenes(frames_same, threshold=0.30, min_scene_len=2, longest=16)
check("无切 6 帧 -> [0,6]", cuts_same == [0, 6])

# 黑场：红5 + 黑2 + 蓝5，min_len=2 应在黑场前后各切 -> [0,5,7,12]
frames_black = [solid_frame(32, 32, (200, 0, 0))] * 5 + [gray_frame(32, 32, 0)] * 2 + [solid_frame(32, 32, (0, 0, 200))] * 5
cuts_black = detect_scenes(frames_black, threshold=0.30, black_threshold=0.08, min_scene_len=2, longest=16)
check("黑场 切点", cuts_black == [0, 5, 7, 12])

# 白闪：红5 + 白2 + 蓝5 -> 同理
frames_white = [solid_frame(32, 32, (200, 0, 0))] * 5 + [gray_frame(32, 32, 255)] * 2 + [solid_frame(32, 32, (0, 0, 200))] * 5
cuts_white = detect_scenes(frames_white, threshold=0.30, white_threshold=0.92, min_scene_len=2, longest=16)
check("白闪 切点", cuts_white == [0, 5, 7, 12])

# 溶解：红5 -> 渐变8 -> 蓝5，窗口8 时累积距离应触发溶解切
# 构造渐变：线性插值 R->B
grad = []
for t in range(8):
    r = int(200 * (1 - t / 7))
    b = int(200 * (t / 7))
    grad.append(solid_frame(32, 32, (r, 0, b)))
frames_dissolve = [solid_frame(32, 32, (200, 0, 0))] * 5 + grad + [solid_frame(32, 32, (0, 0, 200))] * 5
cuts_dissolve = detect_scenes(frames_dissolve, threshold=0.30, min_scene_len=2, dissolve_window=8, dissolve_threshold=0.05, longest=16)
# 溶解应至少产生一次切（不要求精确位置，只要段数>2）
check("溶解 检测出 >2 段", len(cuts_dissolve) >= 3 and cuts_dissolve[0] == 0 and cuts_dissolve[-1] == len(frames_dissolve))

# min_scene_len 去抖：红2 + 蓝2 + 红2，min_len=3 时 2 帧段被合并 -> 仅 [0,6]
frames_short = [solid_frame(32, 32, (255, 0, 0))] * 2 + [solid_frame(32, 32, (0, 0, 255))] * 2 + [solid_frame(32, 32, (255, 0, 0))] * 2
cuts_short = detect_scenes(frames_short, threshold=0.25, min_scene_len=3, longest=16)
check("min_scene_len 去抖", cuts_short == [0, 6])

# 单帧
cuts_one = detect_scenes([solid_frame(16, 16, (10, 20, 30))], longest=16)
check("单帧 -> [0,1]", cuts_one == [0, 1])

# 空
cuts_empty = detect_scenes([], longest=16)
check("空 -> [0,0]", cuts_empty == [0, 0])

# split_scenes
segs = split_scenes([0, 5, 10, 15])
check("split_scenes", segs == [(0, 5), (5, 10), (10, 15)])

# 形状 float 输入也支持
frames_float = [np.full((16, 16, 3), 0.2, dtype=np.float32)] * 3 + [np.full((16, 16, 3), 0.8, dtype=np.float32)] * 3
cuts_float = detect_scenes(frames_float, threshold=0.25, min_scene_len=1, longest=16)
check("float 输入", cuts_float == [0, 3, 6])

# ── 2. 节点模块（mock torch）──
try:
    import torch as real_torch  # noqa: F401
except ModuleNotFoundError:
    real_torch = None

# 构造最小 FakeTensor 替身，适配节点逻辑（detach/cpu/numpy/slice）
class FakeTensor:
    def __init__(self, arr):
        self._arr = np.asarray(arr)
        # keep shape attr as tuple
        self.shape = self._arr.shape
        self.ndim = self._arr.ndim

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._arr

    def __getitem__(self, idx):
        res = self._arr[idx]
        # 切片返回 FakeTensor（保持 ndim 语义）
        if isinstance(res, np.ndarray):
            return FakeTensor(res)
        # 单元素标量直接返回标量
        return res

    def __len__(self):
        return self._arr.shape[0]


# mock torch 模块
fake_torch = types.ModuleType("torch")
fake_torch.Tensor = FakeTensor
sys.modules["torch"] = fake_torch

# 注册包以便相对导入解析
_sf_pkg = types.ModuleType("sfnodes")
_sf_pkg.__path__ = [root]
sys.modules["sfnodes"] = _sf_pkg
_sf_nodes_pkg = types.ModuleType("sfnodes.nodes")
_sf_nodes_pkg.__path__ = [os.path.join(root, "nodes")]
sys.modules["sfnodes.nodes"] = _sf_nodes_pkg
_sf_image_pkg = types.ModuleType("sfnodes.nodes.image")
_sf_image_pkg.__path__ = [os.path.join(root, "nodes", "image")]
sys.modules["sfnodes.nodes.image"] = _sf_image_pkg
_sf_utils_pkg = types.ModuleType("sfnodes.sf_utils")
_sf_utils_pkg.__path__ = [os.path.join(root, "sf_utils")]
sys.modules["sfnodes.sf_utils"] = _sf_utils_pkg

# 需让 sf_utils.scene_detect 真实加载（已在 sys.path）
spec = importlib.util.spec_from_file_location(
    "sfnodes.nodes.image.scene_split",
    os.path.join(root, "nodes", "image", "scene_split.py"),
)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

check("模块已加载", hasattr(mod, "SFImageSceneSplit"))
Node = mod.SFImageSceneSplit
check("CATEGORY", Node.CATEGORY == "sfnodes/image")
check("DESCRIPTION 非空", isinstance(Node.DESCRIPTION, str) and len(Node.DESCRIPTION) > 0)
check("RETURN_TYPES", Node.RETURN_TYPES == ("IMAGE", "INT", "STRING", "INT", "IMAGE"))
check("RETURN_NAMES", Node.RETURN_NAMES == ("images", "count", "cuts", "scene_count", "all_segments"))
check("OUTPUT_IS_LIST", Node.OUTPUT_IS_LIST == (False, False, False, False, True))
check("FUNCTION", Node.FUNCTION == "execute")

it = Node.INPUT_TYPES()
check("required 含 images", "images" in it["required"])
for k in ("threshold", "black_threshold", "white_threshold", "min_scene_len", "segment_index", "max_frames", "method", "dissolve_window", "dissolve_threshold"):
    check(f"required 含 {k}", k in it["required"])

# 节点执行：构造 3 段红/蓝/绿 各 4 帧 32x32
def make_batch(colors):
    # colors: list[(r,g,b)]
    arrs = []
    for c in colors:
        a = np.zeros((32, 32, 3), dtype=np.float32)
        a[:, :] = np.array(c, dtype=np.float32) / 255.0
        arrs.append(a)
    batch = np.stack(arrs, axis=0)  # [B,H,W,C]
    return FakeTensor(batch)


colors = [(255, 0, 0)] * 4 + [(0, 0, 255)] * 4 + [(0, 255, 0)] * 4
batch = make_batch(colors)
node = Node()
res = node.execute(
    images=batch,
    threshold=0.25,
    black_threshold=0.08,
    white_threshold=0.92,
    min_scene_len=2,
    segment_index=0,
    max_frames=0,
    method="hist",
    dissolve_window=8,
    dissolve_threshold=0.18,
)
selected, count, cuts_json, scene_count, all_segments = res
cuts = json.loads(cuts_json)
check("节点 cuts", cuts == [0, 4, 8, 12])
check("scene_count", scene_count == 3)
check("segment_index 0 帧数", count == 4 and selected.shape[0] == 4)
check("all_segments 长度", len(all_segments) == 3 and all_segments[1].shape[0] == 4)

# 负索引
res2 = node.execute(
    images=batch,
    threshold=0.25,
    black_threshold=0.08,
    white_threshold=0.92,
    min_scene_len=2,
    segment_index=-1,
    max_frames=0,
    method="hist",
    dissolve_window=8,
    dissolve_threshold=0.18,
)
selected2 = res2[0]
# -1 应为绿段，首像素 G≈1
check("负索引 -1", selected2._arr[0, 0, 0, 1] > 0.9)

# max_frames 首 N 帧
res3 = node.execute(
    images=batch,
    threshold=0.25,
    black_threshold=0.08,
    white_threshold=0.92,
    min_scene_len=2,
    segment_index=1,
    max_frames=2,
    method="hist",
    dissolve_window=8,
    dissolve_threshold=0.18,
)
check("max_frames 截断", res3[0].shape[0] == 2 and res3[1] == 2)

# 越界抛错
try:
    node.execute(
        images=batch,
        threshold=0.25,
        black_threshold=0.08,
        white_threshold=0.92,
        min_scene_len=2,
        segment_index=10,
        max_frames=0,
        method="hist",
        dissolve_window=8,
        dissolve_threshold=0.18,
    )
    check("越界应抛错", False)
except ValueError as e:
    check("越界抛 ValueError", "越界" in str(e))

# 越界负数
try:
    node.execute(
        images=batch,
        threshold=0.25,
        black_threshold=0.08,
        white_threshold=0.92,
        min_scene_len=2,
        segment_index=-10,
        max_frames=0,
        method="hist",
        dissolve_window=8,
        dissolve_threshold=0.18,
    )
    check("负越界应抛错", False)
except ValueError as e:
    check("负越界抛 ValueError", "越界" in str(e))

# 单帧
single = make_batch([(10, 20, 30)])
res_single = node.execute(
    images=single,
    threshold=0.30,
    black_threshold=0.08,
    white_threshold=0.92,
    min_scene_len=1,
    segment_index=0,
    max_frames=0,
    method="hist",
    dissolve_window=8,
    dissolve_threshold=0.18,
)
check("单帧 scene_count 1", res_single[3] == 1)

print()
if failures:
    print(f"{len(failures)} FAILURES: {failures}")
    sys.exit(1)
print("ALL PASS")
