# SFPauseImage 后端逻辑测试（Node/Python 直接运行：python tests/test_pause_image.py）
# 覆盖：INPUT_TYPES 结构、快照写读 round-trip、continue 无快照/损坏快照报错、
#       无线报错、pause 透传+emit+meta 嵌入、无 IS_CHANGED
# mock：torch/folder_paths（numpy/PIL 本机真实可用）
import importlib.util
import os
import sys
import tempfile
import types

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── mock torch / folder_paths ──
torch = types.ModuleType("torch")

class MockTensor:
    """模拟 torch.Tensor：支持下标（image[0]）、[None, ...] 加 batch 维、.shape、.numpy()。"""
    def __init__(self, arr):
        self._arr = arr  # [B,H,W,C] 或 [H,W,C]
    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            if len(idx) == 2 and idx[0] is None:  # [None, ...] -> 加 batch 维
                return MockTensor(self._arr[None, ...])
            return self
        if idx is None:
            return self
        frame = self._arr[idx]
        return types.SimpleNamespace(cpu=lambda: types.SimpleNamespace(numpy=lambda: frame))
    @property
    def shape(self):
        return self._arr.shape
    def numpy(self):
        return self._arr

torch.from_numpy = MockTensor
sys.modules["torch"] = torch

tmp_root = tempfile.mkdtemp(prefix="sf_pause_img_test_")
folder_paths = types.ModuleType("folder_paths")
folder_paths.get_temp_directory = lambda: os.path.join(tmp_root, "temp")
sys.modules["folder_paths"] = folder_paths

spec = importlib.util.spec_from_file_location(
    "sfnodes.nodes.image.pause_image",
    os.path.join(root, "nodes", "image", "pause_image.py"),
)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

failures = []

def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")

node = mod.SFPauseImage()
check("CATEGORY", node.CATEGORY == "sfnodes/image")
check("DESCRIPTION 存在", isinstance(node.DESCRIPTION, str) and len(node.DESCRIPTION) > 0)
check("OUTPUT_NODE", node.OUTPUT_NODE is True)
check("无 IS_CHANGED", not hasattr(node, "IS_CHANGED"))

it = node.INPUT_TYPES()
check("required 为空", it["required"] == {})
check("INPUT_TYPES 含 image", "image" in it["optional"])
check("image 类型 IMAGE", it["optional"]["image"][0] == "IMAGE")
for key in ("PauseState", "unique_id", "prompt", "extra_pnginfo"):
    check(f"hidden 含 {key}", key in it["hidden"])
check("返回类型 image", node.RETURN_TYPES == ("IMAGE",) and node.RETURN_NAMES == ("image",))
check("FUNCTION = run", node.FUNCTION == "run")

# ── 输入张量构造（真实 numpy，mock tensor 壳）──
import numpy as np
from PIL import Image  # 路由测试（_decode_image/_build_pnginfo round-trip）使用
arr = np.zeros((1, 2, 4, 3), dtype=np.float32)  # [B,H,W,C]，1 帧 2x4x3
arr[0, 0, 0, 0] = 1.0  # 一个非零像素防全黑
image = torch.from_numpy(arr)

def snap_path(node_id):
    return os.path.join(tmp_root, "temp", f"sf_pause_{node_id}.png")

# ── pause 有线：透传 + emit + 快照写入 + meta 嵌入 ──
r = node.run(image=image, PauseState='{"mode": "pause"}', unique_id="42",
             prompt={"x": 1}, extra_pnginfo={"workflow": {"w": 1}})
check("pause 有线透传", r["result"][0] is image)
check("pause emit frame", "sf_pause_frame" in r["ui"] and r["ui"]["sf_pause_frame"][0]["filename"] == "sf_pause_42.png")
check("快照文件已写入", os.path.isfile(snap_path("42")))
check("meta 嵌入（NaN 清洗）", r["ui"]["sf_pause_frame"][0]["_sf_pause_meta"]["prompt"] == {"x": 1})
r2 = node.run(image=image, PauseState='{"mode": "pause"}', unique_id="42",
              prompt={"n": float("nan")}, extra_pnginfo={})
check("meta NaN 清洗为字符串", r2["ui"]["sf_pause_frame"][0]["_sf_pause_meta"]["prompt"]["n"] == "nan")

# ── continue：读回快照 round-trip ──
r = node.run(image=None, PauseState='{"mode": "continue"}', unique_id="42")
out = r["result"][0]
check("continue 读回快照（1xHxWxC）", out.shape[0] == 1 and out.shape[1:] == (2, 4, 3))
check("continue emit frame", r["ui"]["sf_pause_frame"][0]["filename"] == "sf_pause_42.png")
check("continue 输出与快照一致", bool((out.numpy() == arr).all()))

# ── continue 无快照：清晰报错 ──
try:
    node.run(image=None, PauseState='{"mode": "continue"}', unique_id="nope")
    check("continue 无快照报错", False)
except RuntimeError as e:
    check("continue 无快照报错", "快照已过期" in str(e))

# ── continue 损坏快照：清晰报错 ──
with open(snap_path("bad"), "wb") as f:
    f.write(b"not a png")
try:
    node.run(image=None, PauseState='{"mode": "continue"}', unique_id="bad")
    check("continue 损坏快照报错", False)
except RuntimeError as e:
    check("continue 损坏快照报错", "无法读取" in str(e))

# ── pause/pass 无线：报错 ──
try:
    node.run(image=None, PauseState='{"mode": "pause"}', unique_id="x")
    check("pause 无线报错", False)
except RuntimeError as e:
    check("pause 无线报错", "未连接图片" in str(e))
try:
    node.run(image=None, PauseState='{"mode": "pass"}', unique_id="x")
    check("pass 无线报错", False)
except RuntimeError as e:
    check("pass 无线报错", "未连接图片" in str(e))

# ── pass 有线：同 pause 行为（透传 + emit + 快照） ──
r = node.run(image=image, PauseState='{"mode": "pass"}', unique_id="pass1")
check("pass 有线透传", r["result"][0] is image and "sf_pause_frame" in r["ui"])
check("pass 也写快照", os.path.isfile(snap_path("pass1")))

# ── 模式/状态容错 ──
r = node.run(image=image, PauseState="not json", unique_id="f1")
check("非法 JSON 回退 pause（透传）", r["result"][0] is image)
r = node.run(image=image, PauseState='{"mode": "bogus"}', unique_id="f2")
check("未知模式回退 pause", r["result"][0] is image)
# 快照保存失败不炸 run（temp 目录只读）
ro_dir = os.path.join(tmp_root, "ro")
os.makedirs(ro_dir)
os.chmod(ro_dir, 0o500)  # r-x：目录存在但不可写 -> pil.save 抛 OSError -> 降级
folder_paths.get_temp_directory = lambda: ro_dir
r = node.run(image=image, PauseState='{"mode": "pause"}', unique_id="f3")
check("快照保存失败降级（仍透传，ui 无 frame）", r["result"][0] is image and "sf_pause_frame" not in r["ui"])
os.chmod(ro_dir, 0o700)

# ── meta 仅新鲜捕获（continue 的 frame 无 meta） ──
folder_paths.get_temp_directory = lambda: os.path.join(tmp_root, "temp")
r = node.run(image=None, PauseState='{"mode": "continue"}', unique_id="42")
check("continue frame 无 meta", "_sf_pause_meta" not in r["ui"]["sf_pause_frame"][0])

# ── preview_routes 模块测试（mock folder_paths，server/comfy 缺失时降级）──
import base64 as _b64
import io as _io

spec_r = importlib.util.spec_from_file_location(
    "sfnodes.nodes.image.preview_routes",
    os.path.join(root, "nodes", "image", "preview_routes.py"),
)
mod_r = importlib.util.module_from_spec(spec_r)
sys.modules[spec_r.name] = mod_r
spec_r.loader.exec_module(mod_r)

check("_safe_prefix 正常前缀", mod_r._safe_prefix("PauseImage") == "PauseImage")
check("_safe_prefix 子目录", mod_r._safe_prefix("a/b") == "a/b")
check("_safe_prefix 非法字符替换为下划线（尾 _ 被边沿剥离）", mod_r._safe_prefix("Bad:Name*") == "Bad_Name")
check("_safe_prefix 保留空格/中文", mod_r._safe_prefix("My 分类") == "My 分类")
check("_safe_prefix 路径穿越拒绝", mod_r._safe_prefix("..") == "" and mod_r._safe_prefix("a/../b") == "")
check("_safe_prefix leading 斜杠拒绝", mod_r._safe_prefix("/abs") == "")
check("_safe_prefix Windows 保留名加后缀", mod_r._safe_prefix("CON") == "CON_")
check("_safe_prefix 超长拒绝", mod_r._safe_prefix("x" * 300) == "")

buf = _io.BytesIO()
Image.new("RGB", (4, 4)).save(buf, "PNG")
png_b64 = "data:image/png;base64," + _b64.b64encode(buf.getvalue()).decode("ascii")
pil = mod_r._decode_image(png_b64)
check("_decode_image data URI", pil is not None and pil.size == (4, 4))
check("_decode_image 非法输入", mod_r._decode_image("not base64!") is None and mod_r._decode_image(None) is None)

pi = mod_r._build_pnginfo(prompt={"n": float("nan")}, workflow={"w": 1})
# 保存并读回 tEXt 块，验证 prompt/workflow 嵌入且 NaN 已清洗
out_buf = _io.BytesIO()
Image.new("RGB", (4, 4)).save(out_buf, "PNG", pnginfo=pi)
out_buf.seek(0)
with Image.open(out_buf) as reopened:
    t = reopened.text
check("_build_pnginfo 嵌入 prompt（NaN 清洗）", '"n": "nan"' in t.get("prompt", ""))
check("_build_pnginfo 嵌入 workflow", '"w": 1' in t.get("workflow", ""))
check("_build_pnginfo 参数块", t.get("parameters") is None)  # 无 parameters 时跳过

print("\nFAILURES:", len(failures))
sys.exit(1 if failures else 0)
