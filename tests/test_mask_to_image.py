#!/usr/bin/env python3
"""SFMasksToImage 模拟测试（无 torch 依赖时用 FakeTensor，复用 test_mask_fill mock 先例）。"""
import sys, os, importlib.util
import types
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

# --- FakeTensor (subset needed by MasksToImage: permute/reshape/unsqueeze/expand/getitem) ---
class _FakeTensor:
    def __init__(self, data):
        import numpy as _np
        self._np = _np
        if isinstance(data, _np.ndarray):
            self._a = data.astype(_np.float32)
        else:
            self._a = _np.array(data, dtype=_np.float32)
        self.shape = self._a.shape
        self.dtype = self._a.dtype
        self.device = "cpu"

    def reshape(self, *args):
        if len(args) == 1 and isinstance(args[0], tuple):
            args = args[0]
        return _FakeTensor(self._a.reshape(*args))

    def unsqueeze(self, dim):
        import numpy as _np
        return _FakeTensor(_np.expand_dims(self._a, axis=dim))

    def permute(self, *dims):
        import numpy as _np
        return _FakeTensor(_np.transpose(self._a, dims))

    def expand(self, *shape):
        import numpy as _np
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        tgt = []
        for idx, s in enumerate(shape):
            if s == -1:
                tgt.append(self._a.shape[idx])
            else:
                tgt.append(s)
        return _FakeTensor(_np.broadcast_to(self._a, tuple(tgt)).copy())

    def __getitem__(self, idx):
        import numpy as _np
        res = self._a[idx]
        if isinstance(res, (_np.floating, _np.integer)):
            return float(res)
        return _FakeTensor(res)

def _fake_torch_module():
    m = types.ModuleType("torch")
    m.Tensor = _FakeTensor
    return m

# --- install mocks before import ---
sys.modules["torch"] = _fake_torch_module()

fp = types.ModuleType("folder_paths")
fp.get_temp_directory = lambda: "/tmp"
fp.get_save_image_path = lambda *a, **k: ("/tmp", "test", 1, "test", "test")
fp.get_output_directory = lambda: "/tmp"
fp.get_input_directory = lambda: "/tmp"
sys.modules["folder_paths"] = fp

cu = types.ModuleType("comfy.utils")
cu.common_upscale = lambda *a, **k: a[0]
sys.modules["comfy.utils"] = cu

nodes_mod = types.ModuleType("nodes")
class _SaveImage: pass
nodes_mod.SaveImage = _SaveImage
nodes_mod.MAX_RESOLUTION = 8192
sys.modules["nodes"] = nodes_mod

# sfnodes package shells
pkg = types.ModuleType("sfnodes")
pkg.__path__ = [root]
sys.modules["sfnodes"] = pkg
pkg2 = types.ModuleType("sfnodes.sf_utils")
pkg2.__path__ = [os.path.join(root, "sf_utils")]
sys.modules["sfnodes.sf_utils"] = pkg2
pkg3 = types.ModuleType("sfnodes.nodes")
pkg3.__path__ = [os.path.join(root, "nodes")]
sys.modules["sfnodes.nodes"] = pkg3
pkg4 = types.ModuleType("sfnodes.nodes.mask")
pkg4.__path__ = [os.path.join(root, "nodes", "mask")]
sys.modules["sfnodes.nodes.mask"] = pkg4

# trivial mocks for sibling utils (MasksToImage 不使用它们，仅供 masks.py import 成功)
mod_ic = types.ModuleType("sfnodes.sf_utils.image_convert")
mod_ic.mask2tensor = lambda m: m
mod_ic.tensor2mask = lambda i, channel="red": i
mod_ic.rescale_image = lambda i, w, h: i
mod_ic.np2tensor = lambda i: i
sys.modules["sfnodes.sf_utils.image_convert"] = mod_ic

mod_re = types.ModuleType("sfnodes.sf_utils.resize_engine")
mod_re.floor_divisible = lambda x, d=8: x
sys.modules["sfnodes.sf_utils.resize_engine"] = mod_re

mod_mu = types.ModuleType("sfnodes.sf_utils.mask_utils")
mod_mu.combine_mask = lambda *a, **k: a[0]
mod_mu.expand_mask = lambda a, *k: a
mod_mu.invert_mask = lambda a: a
mod_mu.apply_mask_area = lambda a, b, c: a
mod_mu.mask_unsqueeze = lambda a: a
mod_mu.mask_floor = lambda a, *k, **kw: a
mod_mu.make_odd = lambda x: x
mod_mu.binary_erosion = lambda a, r: a
mod_mu.gaussian_blur = lambda a, r, sigma=0: a
mod_mu.mask_process = lambda a, b, **k: a
sys.modules["sfnodes.sf_utils.mask_utils"] = mod_mu

spec = importlib.util.spec_from_file_location(
    "sfnodes.nodes.mask.masks", os.path.join(root, "nodes", "mask", "masks.py"))
masks = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = masks
spec.loader.exec_module(masks)

import numpy as np

def assert_eq(a, b, msg=""):
    if isinstance(a, _FakeTensor): a = a._a
    if not np.allclose(np.array(a), np.array(b)):
        print(f"FAIL {msg}: {a} != {b}")
        sys.exit(1)

node = masks.MasksToImage()

# 结构检查
inp = masks.MasksToImage.INPUT_TYPES()
assert "masks" in inp["required"], "masks input missing"
assert node.RETURN_TYPES == ("IMAGE",), "RETURN_TYPES wrong"
assert node.FUNCTION == "execute", "FUNCTION wrong"
assert node.CATEGORY == "sfnodes/mask", "CATEGORY wrong"
assert isinstance(node.DESCRIPTION, str) and node.DESCRIPTION, "DESCRIPTION missing"
print("test node structure OK")

# 1. [B,H,W] 标准形态
m = _FakeTensor(np.arange(8, dtype=np.float32).reshape(1, 2, 4) / 8.0)
out = node.execute(m)[0]
assert out._a.shape == (1, 2, 4, 3), f"BHW shape {out._a.shape}"
assert_eq(out._a[..., 0], out._a[..., 1], "channels equal r-g")
assert_eq(out._a[..., 1], out._a[..., 2], "channels equal g-b")
assert_eq(out._a[..., 0], m._a, "grayscale preserved")
print("test BHW OK")

# 2. [H,W] 二维形态 → 升为 B=1
m2 = _FakeTensor(np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32))
out2 = node.execute(m2)[0]
assert out2._a.shape == (1, 2, 2, 3), f"HW shape {out2._a.shape}"
assert_eq(out2._a[..., 0], m2._a, "HW grayscale preserved")
print("test HW OK")

# 3. [N,C,H,W] 四维形态（取首通道，恒输出 3 通道）
m4 = _FakeTensor(np.stack([
    np.full((1, 2, 2), 0.3, dtype=np.float32),
    np.full((1, 2, 2), 0.9, dtype=np.float32),
], axis=1))  # N=1, C=2, H=2, W=2
out4 = node.execute(m4)[0]
assert out4._a.shape == (1, 2, 2, 3), f"NCHW shape {out4._a.shape}"
assert_eq(out4._a, 0.3, "NCHW takes first channel (not C*3 channels)")
print("test NCHW OK")

# 4. 批量 B>1
mb = _FakeTensor(np.zeros((3, 2, 2), dtype=np.float32))
outb = node.execute(mb)[0]
assert outb._a.shape == (3, 2, 2, 3), f"batch shape {outb._a.shape}"
print("test batch OK")

# 5. 非法维度 fail-fast
for bad in (_FakeTensor(np.zeros(4, dtype=np.float32)),):
    try:
        node.execute(bad)
        print("FAIL: 1-dim input should raise")
        sys.exit(1)
    except ValueError:
        pass
print("test invalid dims raise OK")

print("All SFMasksToImage tests passed")
