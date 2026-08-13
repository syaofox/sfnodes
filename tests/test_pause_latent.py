# SFPauseLatent 后端逻辑测试（Node/Python 直接运行：python tests/test_pause_latent.py）
# 覆盖：INPUT_TYPES 结构、latent 快照写读 round-trip（samples + noise_mask 全
# batch）、continue 无/损坏快照报错、无 latent 报错、pause 透传+emit+meta、
# 无 image 时无 frame、无 IS_CHANGED
# mock：torch/folder_paths/safetensors（numpy/PIL 本机真实可用）
import importlib.util
import os
import sys
import tempfile
import types

import numpy as np

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── mock torch / folder_paths / safetensors ──
torch = types.ModuleType("torch")

class MockTensor:
    """模拟 torch.Tensor：支持下标（image[0] 返回帧壳）、.shape、.numpy()、
    .contiguous()。构造自 numpy 数组（safetensors mock 存取 _arr）。"""
    def __init__(self, arr):
        self._arr = arr
    def __getitem__(self, idx):
        if idx is None or (isinstance(idx, tuple) and idx[0] is None):
            return self
        frame = self._arr[idx]
        return types.SimpleNamespace(cpu=lambda: types.SimpleNamespace(numpy=lambda: frame))
    @property
    def shape(self):
        return self._arr.shape
    def numpy(self):
        return self._arr
    def contiguous(self):
        return self

torch.from_numpy = lambda arr: MockTensor(arr) if isinstance(arr, np.ndarray) else (_ for _ in ()).throw(ValueError("must be an ndarray"))
torch.tensor = lambda arr: MockTensor(np.asarray(arr))
torch.Tensor = MockTensor
sys.modules["torch"] = torch

tmp_root = tempfile.mkdtemp(prefix="sf_pause_latent_test_")
folder_paths = types.ModuleType("folder_paths")
folder_paths.get_temp_directory = lambda: os.path.join(tmp_root, "temp")
sys.modules["folder_paths"] = folder_paths

# safetensors mock：内存 dict 存取 + 磁盘占位文件（真实 safetensors 未安装；
# 占位文件让 run() 的 os.path.isfile 存在性检查语义与真实环境一致）
_store = {}
st = types.ModuleType("safetensors")
stt = types.ModuleType("safetensors.torch")

def save_file(d, path):
    _store[path] = {k: (v._arr if isinstance(v, MockTensor) else v) for k, v in d.items()}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"mock")

def load_file(path, device="cpu"):
    if path not in _store:
        raise OSError(f"no such file: {path}")
    return {k: torch.from_numpy(v) for k, v in _store[path].items()}

stt.save_file = save_file
stt.load_file = load_file
st.torch = stt
sys.modules["safetensors"] = st
sys.modules["safetensors.torch"] = stt

# ── 加载节点模块（pause_latent 相对导入 .pause_image，先加载它）──
def load_mod(name, rel_path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(root, rel_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

load_mod("sfnodes.nodes.image.pause_image", "nodes/image/pause_image.py")
mod = load_mod("sfnodes.nodes.image.pause_latent", "nodes/image/pause_latent.py")

failures = []

def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")

node = mod.SFPauseLatent()
check("CATEGORY", node.CATEGORY == "sfnodes/image")
check("DESCRIPTION 存在", isinstance(node.DESCRIPTION, str) and len(node.DESCRIPTION) > 0)
check("OUTPUT_NODE", node.OUTPUT_NODE is True)
check("无 IS_CHANGED", not hasattr(node, "IS_CHANGED"))

it = node.INPUT_TYPES()
check("required 为空", it["required"] == {})
check("INPUT_TYPES 含 latent", "latent" in it["optional"])
check("latent 类型 LATENT", it["optional"]["latent"][0] == "LATENT")
check("INPUT_TYPES 含 image 预览", "image" in it["optional"])
check("image 类型 IMAGE", it["optional"]["image"][0] == "IMAGE")
for key in ("PauseState", "unique_id", "prompt", "extra_pnginfo"):
    check(f"hidden 含 {key}", key in it["hidden"])
check("返回类型 latent", node.RETURN_TYPES == ("LATENT",) and node.RETURN_NAMES == ("latent",))
check("FUNCTION = run", node.FUNCTION == "run")

# ── latent 输入构造（[B,C,H,W]）──
samples = np.random.rand(1, 4, 2, 3).astype(np.float32)  # 1 帧 4 通道 2x3
noise_mask = np.random.rand(1, 2, 3).astype(np.float32)
latent = {"samples": torch.from_numpy(samples), "noise_mask": torch.from_numpy(noise_mask)}

img_arr = np.random.rand(1, 3, 4, 3).astype(np.float32)  # 1 帧 3x4 RGB（[B,H,W,C]）
image = torch.from_numpy(img_arr)


def lpath(node_id):
    return os.path.join(tmp_root, "temp", f"sf_pause_latent_{node_id}.latent")


def ppath(node_id):
    return os.path.join(tmp_root, "temp", f"sf_pause_latent_{node_id}.png")


# ── pause 有线（latent + image）：透传 + emit + 双快照 + meta ──
r = node.run(latent=latent, image=image, PauseState='{"mode": "pause"}', unique_id="42",
             prompt={"x": 1}, extra_pnginfo={"workflow": {"w": 1}})
check("pause 有线透传", r["result"][0] is latent)
check("pause emit frame", "sf_pause_latent_frame" in r["ui"] and r["ui"]["sf_pause_latent_frame"][0]["filename"] == "sf_pause_latent_42.png")
check("latent 快照已写入", os.path.isfile(lpath("42")))
check("预览快照已写入", os.path.isfile(ppath("42")))
check("meta 嵌入（NaN 清洗）", r["ui"]["sf_pause_latent_frame"][0]["_sf_pause_meta"]["prompt"] == {"x": 1})

# ── continue：读回 latent 快照 round-trip（全 batch + noise_mask）──
r = node.run(latent=None, image=None, PauseState='{"mode": "continue"}', unique_id="42")
out = r["result"][0]
check("continue 读回 samples 形状", out["samples"].shape == samples.shape)
check("continue samples 值一致", bool(np.max(np.abs(out["samples"].numpy() - samples)) < 1e-6))
check("continue noise_mask 保留", out["noise_mask"].shape == noise_mask.shape
      and bool(np.max(np.abs(out["noise_mask"].numpy() - noise_mask)) < 1e-6))
check("continue emit 预览 frame", r["ui"]["sf_pause_latent_frame"][0]["filename"] == "sf_pause_latent_42.png")
check("continue frame 无 meta", "_sf_pause_meta" not in r["ui"]["sf_pause_latent_frame"][0])

# ── continue 无快照：清晰报错 ──
try:
    node.run(latent=None, PauseState='{"mode": "continue"}', unique_id="nope")
    check("continue 无快照报错", False)
except RuntimeError as e:
    check("continue 无快照报错", "快照已过期" in str(e))

# ── continue 损坏快照：清晰报错 ──
_store[lpath("bad")] = {"latent_tensor": "junk"}  # 非张量 → load 抛错
with open(lpath("bad"), "wb") as f:
    f.write(b"junk")  # 占位文件（isfile 通过，load 阶段才失败）
try:
    node.run(latent=None, PauseState='{"mode": "continue"}', unique_id="bad")
    check("continue 损坏快照报错", False)
except RuntimeError as e:
    check("continue 损坏快照报错", "无法读取" in str(e))

# ── pause/pass 无 latent：报错 ──
for mode in ("pause", "pass"):
    try:
        node.run(latent=None, PauseState=f'{{"mode": "{mode}"}}', unique_id="x")
        check(f"{mode} 无 latent 报错", False)
    except RuntimeError as e:
        check(f"{mode} 无 latent 报错", "未连接 latent" in str(e))

# ── pass 有线：同 pause 行为 ──
r = node.run(latent=latent, image=image, PauseState='{"mode": "pass"}', unique_id="pass1")
check("pass 有线透传", r["result"][0] is latent and "sf_pause_latent_frame" in r["ui"])
check("pass 也写快照", os.path.isfile(lpath("pass1")) and os.path.isfile(ppath("pass1")))

# ── pause 无 image：latent 快照照存，ui 无 frame ──
r = node.run(latent=latent, image=None, PauseState='{"mode": "pause"}', unique_id="noprev")
check("无 image 时 latent 快照照存", os.path.isfile(lpath("noprev")))
check("无 image 时 ui 无 frame", "sf_pause_latent_frame" not in r["ui"])
check("无 image 时透传", r["result"][0] is latent)

# ── continue 无预览 png：无 frame 只出 latent ──
r = node.run(latent=None, PauseState='{"mode": "continue"}', unique_id="noprev")
check("continue 无预览 png 无 frame", "sf_pause_latent_frame" not in r["ui"])
check("continue 无预览仍出 latent", r["result"][0]["samples"].shape == samples.shape)

# ── 模式/状态容错 ──
r = node.run(latent=latent, image=image, PauseState="not json", unique_id="f1")
check("非法 JSON 回退 pause（透传）", r["result"][0] is latent)
r = node.run(latent=latent, image=image, PauseState='{"mode": "bogus"}', unique_id="f2")
check("未知模式回退 pause", r["result"][0] is latent)

# ── latent 快照保存失败降级（save_file 抛 OSError）──
_save_orig = save_file
def save_fail(d, path):
    raise OSError("readonly")
stt.save_file = save_fail
r = node.run(latent=latent, image=image, PauseState='{"mode": "pause"}', unique_id="f3")
check("快照保存失败降级（仍透传，ui 无 frame）", r["result"][0] is latent and "sf_pause_latent_frame" not in r["ui"])
stt.save_file = _save_orig

# ── 转换/存取函数直接验证（batch>1 全保存）──
batch = {"samples": torch.from_numpy(np.random.rand(3, 4, 2, 3).astype(np.float32))}
mod._save_latent(batch, lpath("batch3"))
back = mod._load_latent(lpath("batch3"))
check("batch=3 全保存读回", back["samples"].shape == (3, 4, 2, 3))
check("version 键不泄漏进 latent", "latent_format_version_0" not in back)

# 复用 pause_image 的转换函数（导入而非复制）
from PIL import Image
pil_img = mod._tensor_to_pil(image[0])
check("_tensor_to_pil RGB 预览", pil_img.mode == "RGB" and pil_img.size == (4, 3))
check("_json_safe 复用 pause_image", mod._json_safe is not None)

print("\nFAILURES:", len(failures))
sys.exit(1 if failures else 0)
