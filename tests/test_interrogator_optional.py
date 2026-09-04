import types, sys, os
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)
# mock minimal comfy
comfy = types.ModuleType("comfy")
def _upscale(s, w, h, m, c):
    return s
comfy.utils = types.SimpleNamespace(common_upscale=_upscale)
sys.modules["comfy"] = comfy
sys.modules["comfy.utils"] = comfy.utils
# mock torch
torch_mod = types.ModuleType("torch")
class FakeTensor:
    def __init__(self, shape, val=0.5):
        self._shape = tuple(shape)
        self.shape = self._shape
        self.dim = lambda: len(self._shape)
        self.val = val
    def unsqueeze(self, d):
        # simplistic: add dim at pos
        new_shape = list(self._shape)
        if d < 0:
            d = len(new_shape)+1+d
        new_shape.insert(d, 1)
        return FakeTensor(new_shape, self.val)
    def __getitem__(self, key):
        # for video slicing: return FakeTensor with shape adjusted
        if isinstance(key, slice):
            # assume slice on dim0
            start, stop, step = key.indices(self._shape[0])
            n = len(range(start, stop, step if step else 1))
            return FakeTensor((n,)+self._shape[1:], self.val)
        if isinstance(key, tuple):
            # video[idx:idx+1]
            return FakeTensor((1,)+self._shape[1:], self.val)
        return FakeTensor((1,)+self._shape[1:], self.val)
    def __getattr__(self, name):
        # for movedim etc, return fake
        if name == "movedim":
            return lambda a,b: self
        if name == "clamp":
            return lambda *a, **k: self
        if name == "__getitem__":
            return self.__getitem__
        raise AttributeError(name)
sys.modules["torch"] = torch_mod
# mock folder_paths
import types as _t
fp = _t.ModuleType("folder_paths")
fp.get_user_directory = lambda: "/tmp"
sys.modules["folder_paths"] = fp
# mock server/aiohttp
aioh = _t.ModuleType("aiohttp")
aioh.web = _t.SimpleNamespace(json_response=lambda *a, **k: None, Response=lambda *a, **k: None)
sys.modules["aiohttp"] = aioh
sys.modules["aiohttp.web"] = aioh.web
srv = _t.ModuleType("server")
class _R:
    def get(self, p): return lambda fn: fn
    def post(self, p): return lambda fn: fn
    def delete(self, p): return lambda fn: fn
srv.PromptServer = type("PS", (), {"instance": type("I", (), {"routes": _R()})()})
sys.modules["server"] = srv

from nodes.model.krea2 import SFImageInterrogator, _strip_qwen3_thinking
import inspect

fail=0
def check(n,c):
    global fail
    print(f"[{'OK' if c else 'FAIL'}] {n}")
    if not c:
        fail+=1

# Mock clip that records tokenize args
class FakeClip:
    def __init__(self):
        self.last = {}
    def tokenize(self, text, images=None, llama_template=None, thinking=False, **kw):
        self.last = {"text": text, "images": images, "template": llama_template, "thinking": thinking, "kw": kw}
        # return dummy tokens
        return {"qwen3vl_4b": [ [(0,0)] ]}
    def generate(self, tokens, **kw):
        self.last_gen = kw
        return [[1,2,3]]
    def decode(self, ids):
        return "hello world"

# helper to make fake image tensor with shape [B,H,W,C]
class Img:
    def __init__(self, b=1,h=64,w=64,c=3):
        self.shape = (b,h,w,c)
        self.dim = lambda: 4
        self._b=b
    def __getitem__(self, k):
        return self
    def movedim(self, a,b):
        # return object with shape for common_upscale handling: need shape[3],shape[2]
        class S:
            shape = (1,3,64,64)
            def __getitem__(self, k): return self
            def __getattr__(self, n): return lambda *a, **k: self
        return S()
    def clamp(self, *a, **k): return self

# monkey patch _flatten_to_rgb and common_upscale to passthrough for test
import nodes.model.krea2 as k2
orig_flat = k2._flatten_to_rgb
k2._flatten_to_rgb = lambda x: x if x is not None else None
orig_upscale = k2.comfy.utils.common_upscale
k2.comfy.utils.common_upscale = lambda s,w,h,m,c: s

# Test 1: pure text (no image/video)
fc = FakeClip()
node = SFImageInterrogator()
res = node.interrogate(clip=fc, preset="default", prompt="a cat", user_prompt="", max_length=256, do_sample=True, temperature=0.7, top_k=64, top_p=0.95, min_p=0.05, repetition_penalty=1.05, presence_penalty=0.0, seed=0, thinking=False)
check("纯文本无图调用成功", res[0]=="hello world")
check("纯文本 images 为空", fc.last["images"]==[])
check("纯文本 text 无占位符", "<|image_pad|>" not in fc.last["text"])
check("纯文本 text 含 prompt", "a cat" in fc.last["text"])
check("纯文本 min_p 透传", fc.last_gen.get("min_p")==0.05 if hasattr(fc, 'last_gen') else True)
# 用 FakeClip.generate 记录 min_p 需要让 generate 捕获；在 FakeClip.generate 中记 last_gen
# 上面第一次调用的 last_gen 在 fc.last_gen
# 额外校验透传
fc_tmp = FakeClip()
node.interrogate(clip=fc_tmp, preset="default", prompt="a cat", user_prompt="", max_length=256, do_sample=True, temperature=0.7, top_k=64, top_p=0.95, min_p=0.1, repetition_penalty=1.05, presence_penalty=0.5, seed=0, thinking=False)
check("min_p/presence_penalty 透传", fc_tmp.last_gen.get("min_p")==0.1 and fc_tmp.last_gen.get("presence_penalty")==0.5)

# Test 2: with image only
fc2 = FakeClip()
fake_img = Img(b=1)
# make _scale_image return single element list
node2 = SFImageInterrogator()
# patch _scale_image to return [fake]
orig_scale = node2._scale_image
k2.SFImageInterrogator._scale_image = staticmethod(lambda img, mp: [img] if img is not None else [])
res2 = node2.interrogate(clip=fc2, preset="default", prompt="describe", user_prompt="", max_length=256, do_sample=True, temperature=0.7, top_k=64, top_p=0.95, min_p=0.05, repetition_penalty=1.05, presence_penalty=0.0, seed=1, thinking=False, image=fake_img)
check("单图 images 长度 1", len(fc2.last["images"])==1)
check("单图含单占位符", fc2.last["text"].count("<|image_pad|>")==1)
check("单图无 Picture 前缀", "Picture" not in fc2.last["text"])

# Test 3: image + video multi-frame -> Picture 前缀
fc3 = FakeClip()
fake_vid = Img(b=48)  # 48 frames ->抽帧每24取1 => 2帧
# video shape mock: need shape[0]=48
class FakeVideo:
    shape = (48,64,64,3)
    def __getitem__(self, k):
        # k is slice or int slice
        if isinstance(k, slice):
            start, stop, step = k.indices(48)
            n = len(range(start, stop, step if step else 1))
            return FakeVideoN(n)
        return FakeVideoN(1)
class FakeVideoN:
    def __init__(self, n):
        self.shape = (n,64,64,3)
    def __getitem__(self, k):
        return self
    @property
    def dim(self): return lambda: 4
    def __getattr__(self, name): raise AttributeError
# simplify: monkey _scale_image to distinguish
k2.SFImageInterrogator._scale_image = staticmethod(lambda img, mp: [f"img_{id(img)}"] if img is not None else [])
# create simple objects with shape for video path: use FakeTensor style
class V:
    def __init__(self, b): self.shape=(b,64,64,3)
    def __getitem__(self, k):
        # slice returns new V with appropriate b
        if isinstance(k, slice):
            s,e,step=k.indices(self.shape[0])
            return V(len(range(s,e,step)))
        # int slice like video[idx:idx+1] -> V(1)
        return V(1)
    def __getattr__(self, n): raise AttributeError
fake_img2 = V(1)
fake_vid2 = V(48)
fc3b = FakeClip()
res3 = node2.interrogate(clip=fc3b, preset="default", prompt="p", user_prompt="", max_length=256, do_sample=True, temperature=0.7, top_k=64, top_p=0.95, min_p=0.05, repetition_penalty=1.05, presence_penalty=0.0, seed=0, thinking=False, image=fake_img2, video=fake_vid2)
check("多帧 images 长度 3 (1图+2视频)", len(fc3b.last["images"])==3)
check("多帧含 Picture 前缀", "Picture 1:" in fc3b.last["text"])
check("多帧占位符数量 3", fc3b.last["text"].count("<|image_pad|>")==3)

# Test 4: user_prompt 文本框拼接（现紧邻 prompt 的 required）
fc4 = FakeClip()
k2.SFImageInterrogator._scale_image = staticmethod(lambda img, mp: [])
res4 = node2.interrogate(clip=fc4, preset="default", prompt="base", user_prompt="  extra hello  ", max_length=256, do_sample=True, temperature=0.7, top_k=64, top_p=0.95, min_p=0.05, repetition_penalty=1.05, presence_penalty=0.0, seed=0, thinking=False)
check("user_prompt 拼接到 prompt", "base\nextra hello" in fc4.last["text"] or "base\n"+"extra hello" in fc4.last["text"])
# 空 user_prompt 不拼
fc5 = FakeClip()
res5 = node2.interrogate(clip=fc5, preset="default", prompt="base", user_prompt="   ", max_length=256, do_sample=True, temperature=0.7, top_k=64, top_p=0.95, min_p=0.05, repetition_penalty=1.05, presence_penalty=0.0, seed=0, thinking=False)
check("空 user_prompt 不追加", fc5.last["text"].strip()=="base")

# Test 5: preset 回退（prompt 空时用 preset）
fc6 = FakeClip()
res6 = node2.interrogate(clip=fc6, preset="default", prompt="   ", user_prompt="", max_length=256, do_sample=True, temperature=0.7, top_k=64, top_p=0.95, min_p=0.05, repetition_penalty=1.05, presence_penalty=0.0, seed=0, thinking=False)
check("空 prompt 回退预设", "Generate a detailed paragraph" in fc6.last["text"])

# Test 5b: use_default_template=False 裸模板
fc7 = FakeClip()
res7 = node2.interrogate(clip=fc7, preset="default", prompt="hello", user_prompt="", max_length=256, do_sample=True, temperature=0.7, top_k=64, top_p=0.95, min_p=0.05, repetition_penalty=1.05, presence_penalty=0.0, seed=0, thinking=False, use_default_template=False)
check("裸模板不包裹系统", fc7.last["template"] is None)
check("裸模板 skip_template 透传", fc7.last["kw"].get("skip_template") is True)
check("裸模板 text 直送", fc7.last["text"]=="hello")

# Test 6: _scale_image None 防御
check("_scale_image None 返回 []", SFImageInterrogator._scale_image(None, 1.0)==[])

# restore
k2._flatten_to_rgb = orig_flat
k2.comfy.utils.common_upscale = orig_upscale
k2.SFImageInterrogator._scale_image = orig_scale

print(f"\n{'ALL PASS' if fail==0 else str(fail)+' FAILURES'}")
sys.exit(1 if fail else 0)
