# SFPauseMask 后端逻辑测试（Node/Python 直接运行：python tests/test_pause_mask.py）
# 覆盖：INPUT_TYPES 结构、MASK 灰度快照写读 round-trip（量化检查）、continue 无/
# 损坏快照报错、无线报错、pause 透传+emit+meta、无 IS_CHANGED
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
    """模拟 torch.Tensor：支持下标（mask[0]）、[None, ...] 加 batch 维、.shape、.numpy()。"""
    def __init__(self, arr):
        self._arr = arr  # [B,H,W] 或 [H,W]
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

tmp_root = tempfile.mkdtemp(prefix="sf_pause_mask_test_")
folder_paths = types.ModuleType("folder_paths")
folder_paths.get_temp_directory = lambda: os.path.join(tmp_root, "temp")
sys.modules["folder_paths"] = folder_paths

spec = importlib.util.spec_from_file_location(
    "sfnodes.nodes.mask.pause_mask",
    os.path.join(root, "nodes", "mask", "pause_mask.py"),
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

node = mod.SFPauseMask()
check("CATEGORY", node.CATEGORY == "sfnodes/mask")
check("DESCRIPTION 存在", isinstance(node.DESCRIPTION, str) and len(node.DESCRIPTION) > 0)
check("OUTPUT_NODE", node.OUTPUT_NODE is True)
check("无 IS_CHANGED", not hasattr(node, "IS_CHANGED"))

it = node.INPUT_TYPES()
check("required 为空", it["required"] == {})
check("INPUT_TYPES 含 mask", "mask" in it["optional"])
check("mask 类型 MASK", it["optional"]["mask"][0] == "MASK")
for key in ("PauseState", "unique_id", "prompt", "extra_pnginfo"):
    check(f"hidden 含 {key}", key in it["hidden"])
check("返回类型 mask", node.RETURN_TYPES == ("MASK",) and node.RETURN_NAMES == ("mask",))
check("FUNCTION = run", node.FUNCTION == "run")

# ── MASK 输入构造（[B,H,W] 无 C 通道）──
import numpy as np
arr = np.zeros((1, 2, 4), dtype=np.float32)  # 1 帧 2x4 遮罩
arr[0, 0, 0] = 1.0
arr[0, 1, 3] = 0.5
mask = torch.from_numpy(arr)

def snap_path(node_id):
    return os.path.join(tmp_root, "temp", f"sf_pause_mask_{node_id}.png")

# ── pause 有线：透传 + emit + 快照写入 + meta ──
r = node.run(mask=mask, PauseState='{"mode": "pause"}', unique_id="42",
             prompt={"x": 1}, extra_pnginfo={"workflow": {"w": 1}})
check("pause 有线透传", r["result"][0] is mask)
check("pause emit frame", "sf_pause_mask_frame" in r["ui"] and r["ui"]["sf_pause_mask_frame"][0]["filename"] == "sf_pause_mask_42.png")
check("快照文件已写入", os.path.isfile(snap_path("42")))
check("meta 嵌入（NaN 清洗）", r["ui"]["sf_pause_mask_frame"][0]["_sf_pause_meta"]["prompt"] == {"x": 1})

# ── continue：读回快照 round-trip（灰度 PNG，量化容差）──
r = node.run(mask=None, PauseState='{"mode": "continue"}', unique_id="42")
out = r["result"][0]
check("continue 读回快照（1xHxW）", out.shape[0] == 1 and out.shape[1:] == (2, 4))
check("continue emit frame", r["ui"]["sf_pause_mask_frame"][0]["filename"] == "sf_pause_mask_42.png")
# 量化检查：0.5 -> 128/255 ≈ 0.50196，容差 1/255
check("continue 输出与快照一致（量化容差）",
      bool(np.max(np.abs(out.numpy() - arr)) <= 1.0 / 255.0))

# ── continue 无快照：清晰报错 ──
try:
    node.run(mask=None, PauseState='{"mode": "continue"}', unique_id="nope")
    check("continue 无快照报错", False)
except RuntimeError as e:
    check("continue 无快照报错", "快照已过期" in str(e))

# ── continue 损坏快照：清晰报错 ──
with open(snap_path("bad"), "wb") as f:
    f.write(b"not a png")
try:
    node.run(mask=None, PauseState='{"mode": "continue"}', unique_id="bad")
    check("continue 损坏快照报错", False)
except RuntimeError as e:
    check("continue 损坏快照报错", "无法读取" in str(e))

# ── pause/pass 无线：报错 ──
for mode in ("pause", "pass"):
    try:
        node.run(mask=None, PauseState=f'{{"mode": "{mode}"}}', unique_id="x")
        check(f"{mode} 无线报错", False)
    except RuntimeError as e:
        check(f"{mode} 无线报错", "未连接遮罩" in str(e))

# ── pass 有线：同 pause 行为 ──
r = node.run(mask=mask, PauseState='{"mode": "pass"}', unique_id="pass1")
check("pass 有线透传", r["result"][0] is mask and "sf_pause_mask_frame" in r["ui"])
check("pass 也写快照", os.path.isfile(snap_path("pass1")))

# ── 模式/状态容错 ──
r = node.run(mask=mask, PauseState="not json", unique_id="f1")
check("非法 JSON 回退 pause（透传）", r["result"][0] is mask)
r = node.run(mask=mask, PauseState='{"mode": "bogus"}', unique_id="f2")
check("未知模式回退 pause", r["result"][0] is mask)

# ── 快照保存失败降级（temp 只读）──
ro_dir = os.path.join(tmp_root, "ro")
os.makedirs(ro_dir)
os.chmod(ro_dir, 0o500)
folder_paths.get_temp_directory = lambda: ro_dir
r = node.run(mask=mask, PauseState='{"mode": "pause"}', unique_id="f3")
check("快照保存失败降级（仍透传，ui 无 frame）", r["result"][0] is mask and "sf_pause_mask_frame" not in r["ui"])
os.chmod(ro_dir, 0o700)

# ── meta 仅新鲜捕获（continue 的 frame 无 meta）──
folder_paths.get_temp_directory = lambda: os.path.join(tmp_root, "temp")
r = node.run(mask=None, PauseState='{"mode": "continue"}', unique_id="42")
check("continue frame 无 meta", "_sf_pause_meta" not in r["ui"]["sf_pause_mask_frame"][0])

# ── 转换函数直接验证 ──
pil_img = mod._mask_to_pil(mask[0])   # mask[0] = mock 帧壳（cpu().numpy()）
check("_mask_to_pil L 模式", pil_img.mode == "L" and pil_img.size == (4, 2))
back = mod._pil_to_mask(pil_img)
check("_pil_to_mask 形状与值", back.shape == (1, 2, 4) and bool(np.max(np.abs(back.numpy() - arr)) <= 1.0 / 255.0))

# 非标准 [1,H,W] 帧防御
odd = np.zeros((1, 2, 4), dtype=np.float32)
odd[0, 0, 0] = 0.75
odd_frame = types.SimpleNamespace(cpu=lambda: types.SimpleNamespace(numpy=lambda: odd))
pil_odd = mod._mask_to_pil(odd_frame)
check("_mask_to_pil 非标准 [1,H,W] 压平防御", pil_odd.mode == "L" and pil_odd.size == (4, 2))

print("\nFAILURES:", len(failures))
sys.exit(1 if failures else 0)
