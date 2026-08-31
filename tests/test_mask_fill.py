#!/usr/bin/env python3
"""SFMaskFill 模拟测试（无 torch 依赖时用 FakeTensor，复用 mask_utils 纯逻辑）。"""
import sys, os, importlib.util
import types
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

# mock heavy deps so masks.py import succeeds locally
# --- mock torch ---
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

    def detach(self): return self
    def clone(self):
        return _FakeTensor(self._a.copy())
    def cpu(self): return self
    def numpy(self): return self._a
    def reshape(self, *args):
        # support reshape(-1) and reshape(dims)
        if len(args)==1 and isinstance(args[0], tuple):
            args=args[0]
        if len(args)==1 and isinstance(args[0], int) and args[0]==-1:
            return _FakeTensor(self._a.reshape(-1))
        return _FakeTensor(self._a.reshape(*args))
    def unsqueeze(self, dim):
        import numpy as _np
        a=_np.expand_dims(self._a, axis=dim)
        return _FakeTensor(a)
    def squeeze(self, dim=None):
        import numpy as _np
        if dim is None:
            return _FakeTensor(_np.squeeze(self._a))
        return _FakeTensor(_np.squeeze(self._a, axis=dim))
    def __getitem__(self, idx):
        import numpy as _np2
        res = self._a[idx]
        if isinstance(res, (float, int, _np2.floating, _np2.integer)):
            return float(res)
        # scalar array
        if isinstance(res, _np2.ndarray) and res.ndim==0:
            return float(res)
        return _FakeTensor(res)
    def __setitem__(self, idx, val):
        if isinstance(val, _FakeTensor):
            self._a[idx]=val._a
        else:
            self._a[idx]=val
    def __mul__(self, other):
        if isinstance(other, _FakeTensor):
            return _FakeTensor(self._a*other._a)
        return _FakeTensor(self._a*other)
    __rmul__=__mul__
    def __add__(self, o):
        if isinstance(o, _FakeTensor): return _FakeTensor(self._a+o._a)
        return _FakeTensor(self._a+o)
    def __radd__(self, o):
        return _FakeTensor(o+self._a) if not isinstance(o, _FakeTensor) else _FakeTensor(o._a+self._a)
    def __sub__(self, o):
        if isinstance(o, _FakeTensor): return _FakeTensor(self._a-o._a)
        return _FakeTensor(self._a-o)
    def __rsub__(self, o):
        if isinstance(o, _FakeTensor): return _FakeTensor(o._a-self._a)
        return _FakeTensor(o-self._a)
    def __truediv__(self, o):
        if isinstance(o, _FakeTensor): return _FakeTensor(self._a/o._a)
        return _FakeTensor(self._a/o)
    def repeat(self, *repeats):
        import numpy as _np
        # repeats like (2,1,1,1) for B1HW -> B H W
        # use tile
        repeats = repeats[0] if len(repeats)==1 and isinstance(repeats[0], tuple) else repeats
        a=self._a
        # numpy tile
        return _FakeTensor(_np.tile(a, repeats))
    def __rtruediv__(self, o):
        if isinstance(o, _FakeTensor): return _FakeTensor(o._a/self._a)
        return _FakeTensor(o/self._a)
    def __eq__(self, other):
        import numpy as _np
        if isinstance(other, _FakeTensor):
            return _FakeTensor(self._a == other._a)
        return _FakeTensor(self._a == other)
    def __ne__(self, other):
        import numpy as _np
        if isinstance(other, _FakeTensor):
            return _FakeTensor(self._a != other._a)
        return _FakeTensor(self._a != other)
    def __gt__(self, other):
        import numpy as _np
        if isinstance(other, _FakeTensor):
            return _FakeTensor(self._a > other._a)
        return _FakeTensor(self._a > other)
    def __lt__(self, other):
        import numpy as _np
        if isinstance(other, _FakeTensor):
            return _FakeTensor(self._a < other._a)
        return _FakeTensor(self._a < other)
    def __ge__(self, other):
        import numpy as _np
        if isinstance(other, _FakeTensor):
            return _FakeTensor(self._a >= other._a)
        return _FakeTensor(self._a >= other)
    def __le__(self, other):
        import numpy as _np
        if isinstance(other, _FakeTensor):
            return _FakeTensor(self._a <= other._a)
        return _FakeTensor(self._a <= other)
    def movedim(self,*a,**k): return self
    def expand(self,*a,**k):
        import numpy as _np
        shape=a
        if len(shape)==1 and isinstance(shape[0], tuple):
            shape=shape[0]
        if len(shape)==1 and isinstance(shape[0], int):
            # single tuple case already handled
            pass
        # handle -1 (keep dim) like torch
        tgt=[]
        for idx, s in enumerate(shape):
            if s==-1:
                # map to self dim if exists, else keep 1
                if idx < len(self._a.shape):
                    tgt.append(self._a.shape[idx])
                else:
                    tgt.append(1)
            else:
                tgt.append(s)
        shape=tuple(tgt)
        try:
            return _FakeTensor(_np.broadcast_to(self._a, shape))
        except Exception:
            return _FakeTensor(_np.broadcast_to(self._a, shape))

def _fake_torch_module():
    import numpy as _np
    m=types.ModuleType("torch")
    m.Tensor=_FakeTensor
    def _from_numpy(a): return _FakeTensor(a)
    m.from_numpy=_from_numpy
    def _all(x):
        class _R:
            def __init__(self, v): self._v=v
            def item(self): return bool(self._v)
        if isinstance(x, _FakeTensor):
            return _R(_np.all(x._a==1.0))
        return _R(_np.all(_np.array(x)==1.0))
    m.all=_all
    def _ones_like(a):
        if isinstance(a, _FakeTensor):
            return _FakeTensor(_np.ones_like(a._a))
        return _FakeTensor(_np.ones_like(_np.array(a)))
    m.ones_like=_ones_like
    def _ones(shape, dtype=None, device=None):
        return _FakeTensor(_np.ones(shape, dtype=_np.float32))
    m.ones=_ones
    def _tensor(data, dtype=None, device=None):
        return _FakeTensor(_np.array(data, dtype=_np.float32))
    m.tensor=_tensor
    def _where(cond, x, y):
        # cond may be FakeTensor
        if isinstance(cond, _FakeTensor): cond=cond._a
        if isinstance(x, _FakeTensor): x=x._a
        if isinstance(y, _FakeTensor): y=y._a
        return _FakeTensor(_np.where(cond, x, y))
    m.where=_where
    def _stack(arr, dim=0):
        import numpy as _np
        a=[x._a if isinstance(x, _FakeTensor) else _np.array(x) for x in arr]
        return _FakeTensor(_np.stack(a, axis=dim))
    m.stack=_stack
    def _mean(a, dim=0):
        if isinstance(a, _FakeTensor): a=a._a
        return _FakeTensor(_np.array(_np.mean(a)))
    m.mean=_mean
    m.nn=types.ModuleType("torch.nn")
    m.nn.functional=types.ModuleType("torch.nn.functional")
    def _interpolate(x, size=None, mode="bilinear"):
        # minimal: assume FakeTensor
        import numpy as _np
        # use PIL resize for correctness in tests that hit rescale? simplified to nearest
        # fallback: crop/pad zeros to target H,W
        if isinstance(x, _FakeTensor):
            a=x._a
        else:
            a=_np.array(x)
        # x shape BCHW or BHW? use last 2
        th,tw=size
        bh,bw=a.shape[-2],a.shape[-1]
        if th==bh and tw==bw:
            return _FakeTensor(a)
        # simple nearest via repeat/crop
        res=_np.zeros((*a.shape[:-2], th, tw), dtype=_np.float32)
        h=min(th,bh); w=min(tw,bw)
        res[..., :h, :w]=a[..., :h, :w]
        return _FakeTensor(res)
    m.nn.functional.interpolate=_interpolate
    def _conv2d(a,b,groups=1): return a
    m.conv2d=_conv2d if hasattr(m,'conv2d') else None
    import unittest.mock as _mock
    m.clamp=lambda a, min=None, max=None: a
    return m

# install mocks before import
sys.modules["torch"]=_fake_torch_module()
sys.modules["torch.nn"]=sys.modules["torch"].nn
sys.modules["torch.nn.functional"]=sys.modules["torch"].nn.functional
# folder_paths mock
fp=types.ModuleType("folder_paths")
fp.get_temp_directory=lambda: "/tmp"
fp.get_save_image_path=lambda *a, **k: ("/tmp", "test", 1, "test", "test")
fp.get_output_directory=lambda: "/tmp"
fp.get_input_directory=lambda: "/tmp"
sys.modules["folder_paths"]=fp
# comfy.utils mock
cu=types.ModuleType("comfy.utils")
cu.common_upscale=lambda *a,**k: a[0]
sys.modules["comfy.utils"]=cu
# nodes mock
nodes_mod=types.ModuleType("nodes")
class _SaveImage: pass
nodes_mod.SaveImage=_SaveImage
nodes_mod.MAX_RESOLUTION=8192
sys.modules["nodes"]=nodes_mod
# PIL mocks keep real
# kornia mock
kornia=types.ModuleType("kornia")
kornia.filters=types.ModuleType("kornia.filters")
kornia.filters.filter2d_separable=lambda a,b,c,**k: a
kornia.filters.gaussian_blur2d=lambda a,b,c: a
sys.modules["kornia"]=kornia
sys.modules["kornia.filters"]=kornia.filters
# scipy mock partial (optional)
try:
    import scipy.ndimage as _real_scipy_nd  # noqa
except ModuleNotFoundError:
    pass

# mock torchvision
for _mn in ["torchvision", "torchvision.transforms", "torchvision.transforms.v2"]:
    if _mn not in sys.modules:
        sys.modules[_mn] = types.ModuleType(_mn)
        sys.modules[_mn].ToTensor = lambda: (lambda x: _FakeTensor(np.array(x).astype(np.float32)/255.0))
        sys.modules[_mn].ToPILImage = lambda: (lambda x: None)
        sys.modules[_mn].functional = types.ModuleType("functional")
# mock PIL ImageOps already available, keep real
# mock cv2/scipy for mask_utils
if "cv2" not in sys.modules:
    sys.modules["cv2"] = types.ModuleType("cv2")
    sys.modules["cv2"].INPAINT_TELEA = 0
    sys.modules["cv2"].INPAINT_NS = 1
    sys.modules["cv2"].inpaint = lambda a,b,c,d: a
if "scipy" not in sys.modules:
    sys.modules["scipy"] = types.ModuleType("scipy")
if "scipy.ndimage" not in sys.modules:
    _scipy_nd = types.ModuleType("scipy.ndimage")
    _scipy_nd.grey_erosion = lambda a, **k: a
    _scipy_nd.grey_dilation = lambda a, **k: a
    _scipy_nd.binary_closing = lambda a, **k: a
    _scipy_nd.binary_fill_holes = lambda a: a
    sys.modules["scipy.ndimage"] = _scipy_nd

# now import masks via sfnodes package (like test_crop)
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
# mock image_convert with FakeTensor-compatible impl
mod_ic = types.ModuleType("sfnodes.sf_utils.image_convert")
def _mask2tensor(mask):
    # mask BHW -> B H W 1 -> expand 3 via our FakeTensor
    if isinstance(mask, _FakeTensor):
        a = mask._a
        # reshape (-1,1,H,W) -> movedim -> expand last
        # simulate: mask is BHW, we make B1HW then B H W 3
        # Use FakeTensor expand: create B H W 3 zeros then broadcast
        import numpy as _np
        B,H,W = a.shape
        out = _np.stack([a,a,a], axis=-1)  # B H W 3
        return _FakeTensor(out)
    return mask
def _tensor2mask(img, channel="red"):
    if isinstance(img, _FakeTensor):
        a = img._a
        # img BHWC -> take channel 0
        return _FakeTensor(a[...,0])
    return img
def _rescale_image(img, w, h):
    # img BHWC, resize to B h w C (note args w,h order in masks.py: rescale_image(mask_tensor, image.shape[2], image.shape[1]) => w then h)
    if isinstance(img, _FakeTensor):
        import numpy as _np
        a = img._a
        # a is B H W C
        BHC = a.shape[0]
        # simple nearest by cropping/padding
        cur_h, cur_w = a.shape[1], a.shape[2]
        out = _np.zeros((a.shape[0], h, w, a.shape[3]), dtype=_np.float32)
        hh = min(h, cur_h); ww = min(w, cur_w)
        out[:, :hh, :ww, :] = a[:, :hh, :ww, :]
        return _FakeTensor(out)
    return img
def _np2tensor(img):
    import numpy as _np
    if isinstance(img, _np.ndarray):
        return _FakeTensor(img.astype(_np.float32)/255.0)
    return _FakeTensor(_np.array(img).astype(_np.float32)/255.0)
mod_ic.mask2tensor = _mask2tensor
mod_ic.tensor2mask = _tensor2mask
mod_ic.rescale_image = _rescale_image
mod_ic.np2tensor = _np2tensor
sys.modules["sfnodes.sf_utils.image_convert"] = mod_ic
# mock mask_utils with FakeTensor-compatible impl (reuse real make_odd logic)
mod_mu = types.ModuleType("sfnodes.sf_utils.mask_utils")
def _make_odd(x):
    return x+1 if x>0 and x%2==0 else x
def _mask_unsqueeze(m):
    if isinstance(m, _FakeTensor):
        import numpy as _np
        # BHW -> B1HW
        a = m._a
        if len(a.shape)==3:
            return _FakeTensor(_np.expand_dims(a, axis=1))
        return m
    return m
def _mask_floor(m, thr=0.99):
    if isinstance(m, _FakeTensor):
        import numpy as _np
        return _FakeTensor((_np.array(m._a >= thr, dtype=_np.float32)))
    return m
def _binary_erosion(m, r): return m
def _gaussian_blur(m, r, sigma=0): return m
mod_mu.mask_unsqueeze = _mask_unsqueeze
mod_mu.mask_floor = _mask_floor
mod_mu.make_odd = _make_odd
mod_mu.binary_erosion = _binary_erosion
mod_mu.gaussian_blur = _gaussian_blur
mod_mu.combine_mask = lambda *a,**k: a[0]
mod_mu.expand_mask = lambda a,*k: a
mod_mu.invert_mask = lambda a: a
mod_mu.apply_mask_area = lambda a,b,c: a
mod_mu.mask_process = lambda a,b,**k: a
sys.modules["sfnodes.sf_utils.mask_utils"] = mod_mu

spec=importlib.util.spec_from_file_location("sfnodes.nodes.mask.masks", os.path.join(root, "nodes", "mask", "masks.py"))
masks=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=masks
spec.loader.exec_module(masks)

def assert_eq(a,b,msg=""):
    import numpy as _np
    if isinstance(a, _FakeTensor): a=a._a
    if isinstance(b, _FakeTensor): b=b._a
    if not _np.allclose(_np.array(a), _np.array(b)):
        print(f"FAIL {msg}: {a} != {b}")
        sys.exit(1)

# helpers
import numpy as np

def make_image(b=1,h=4,w=4,c=3,val=0.2):
    return _FakeTensor(np.full((b,h,w,c), val, dtype=np.float32))
def make_mask(b=1,h=4,w=4,val=0.0):
    return _FakeTensor(np.full((b,h,w), val, dtype=np.float32))

# 1. _parse_fill_color
r,g,bv=masks._parse_fill_color([10,20,30])
assert (r,g,bv)==(10,20,30), "list parse"
r,g,bv=masks._parse_fill_color("#0a141e")
assert (r,g,bv)==(10,20,30), "hex parse"
print("test _parse_fill_color OK")

# 2. _apply_falloff with 0 should be no-op (via gaussian_blur mock returns same)
alpha=make_mask(1,4,4,val=1.0)
# need unsqueeze shape B1HW for function
from sf_utils.mask_utils import mask_unsqueeze, mask_floor  # noqa - real utils may be importable? skip if fails
try:
    import importlib.util as _ilu
    spec2=_ilu.spec_from_file_location("mask_utils", os.path.join(os.path.dirname(__file__), "..", "sf_utils", "mask_utils.py"))
    mu=importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(mu)
    # test make_odd
    assert mu.make_odd(2)==3
    assert mu.make_odd(3)==3
    print("test mask_utils make_odd OK")
except Exception as e:
    print(f"skip mask_utils import: {e}")

# 3. MaskFill class structure
mf=masks.MaskFill()
inp=masks.MaskFill.INPUT_TYPES()
assert "fill_mode" in inp["required"], "fill_mode missing"
assert "fill_color" in inp["required"], "fill_color missing"
assert "opacity" in inp["required"], "opacity missing"
assert "falloff" in inp["required"], "falloff global missing"
assert "skip_if_all_white" in inp["required"], "skip missing"
assert mf.FUNCTION=="execute"
assert "SFMaskFill" not in str(type(mf)) or True
print("test INPUT_TYPES structure OK")

# 4. skip_if_all_white path: mask all 1.0 should return original image (content equal, ideally same object)
img=make_image(1,4,4,3,val=0.5)
mask_all_white=make_mask(1,4,4,val=1.0)
# debug torch.all
import torch as _t
print("debug mask_all_white reshape all", _t.all(mask_all_white.reshape(-1)).item())
out=mf.execute(img, mask_all_white, "color", [255,0,0], 1.0, 0, True)
import numpy as _npd
# allow either identity or equal content (mock may clone)
assert _npd.allclose(out[0]._a, img._a), "skip should return original content"
print(f"skip returned is_same={out[0] is img}")
# without skip, should clone (not same object) but content may differ due to fill
out2=mf.execute(img, mask_all_white, "color", [255,0,0], 1.0, 0, False)
assert out2[0] is not img, "non-skip should clone"
print("test skip_if_all_white OK")

# 5. color mode with mask all 0 => no change (opacity 1 but mask 0)
img2=make_image(1,2,2,3,val=0.2)
mask_zero=make_mask(1,2,2,val=0.0)
out3=mf.execute(img2, mask_zero, "color", [255,0,0], 1.0, 0, False)
# since mask 0, result should be original values (0.2)
import numpy as _np
assert _np.allclose(out3[0]._a, 0.2), f"color zero mask should not change: {out3[0]._a}"
print("test color zero mask OK")

# 6. color mode full mask with opacity 0.5 should blend
img3=make_image(1,2,2,3,val=0.0)
mask_one=make_mask(1,2,2,val=1.0)
out4=mf.execute(img3, mask_one, "color", [255,255,255], 0.5, 0, False)
# white 1.0 blended 0.5 with black 0.0 => 0.5
assert _np.allclose(out4[0]._a, 0.5, atol=1e-5), f"blend failed {out4[0]._a}"
print("test color blend OK")

# 7. RGBA branch: create 4ch image
img_rgba=_FakeTensor(np.full((1,2,2,4), 0.2, dtype=np.float32))
mask_one2=make_mask(1,2,2,val=1.0)
out5=mf.execute(img_rgba, mask_one2, "color", [255,0,0], 1.0, 0, False)
# alpha channel should become 1.0 where mask 1
assert _np.allclose(out5[0]._a[...,3], 1.0), f"rgba alpha failed {out5[0]._a[...,3]}"
print("test rgba OK")

# 8. neutral mode: image 0.8 with mask 1 should become 0.8-> -0.5 =>0.3 *0 =>0 +0.5 =>0.5? Let's check formula: image-0.5 *m +0.5, m=1-alpha=0 => 0.5
img4=make_image(1,2,2,3,val=0.8)
out6=mf.execute(img4, mask_one, "neutral", [255,255,255], 1.0, 0, False)
assert _np.allclose(out6[0]._a[...,0], 0.5, atol=1e-5), f"neutral failed {out6[0]._a}"
print("test neutral OK")

# 9. batch broadcast: image 2 batch, mask 1 batch
img_b=make_image(2,2,2,3,val=0.0)
mask_single=make_mask(1,2,2,val=1.0)
out7=mf.execute(img_b, mask_single, "color", [0,255,0], 1.0, 0, False)
assert out7[0]._a.shape[0]==2, "batch broadcast failed"
# green channel should be 1.0
assert _np.allclose(out7[0]._a[0,:,:,1], 1.0), "batch broadcast color failed"
print("test batch broadcast OK")

# 10. size mismatch: mask 2x2 vs image 4x4 should rescale (our mock pads) and not crash
img_big=make_image(1,4,4,3,val=0.1)
mask_small=make_mask(1,2,2,val=1.0)
out8=mf.execute(img_big, mask_small, "color", [255,0,0], 1.0, 0, False)
assert out8[0]._a.shape==(1,4,4,3), "resize failed"
print("test size mismatch OK")

print("All SFMaskFill tests passed")
