# TextEncodeKrea2 编码纯逻辑测试（Node/Python 直接运行：python tests/test_krea2_encode.py）
# 覆盖：
#   - _build_text：system 回退默认、vision_position before/after 拼接顺序
#   - _collect_indexed：imageN/maskN 编号收集（乱序、None 忽略、非匹配名忽略）
#   - _flatten_to_rgb：RGB clamp、RGBA 黑底预乘、(H,W,C) 升维、None 透传
#   - _crop_to_mask：无/空遮罩原样返回、包围盒裁剪、padding 扩展与边界 clamp、
#     遮罩尺寸不匹配时 bilinear 对齐、dim==2/dim==4 归一、多帧并集语义
#   - _prepare_vision：megapixels 上限只缩不放大、小图保持原尺寸、多图按编号
#     排序生成 "Picture N:"、单图无前缀、batch>1 只取首帧
#   - _fp8_hint：FP8 视觉崩溃映射可操作提示，其余异常不误报
# mock：torch stub（where/any）/ comfy.utils.common_upscale（最近邻近似缩放+记录调用）
#       / FakeTensor numpy 代理（test_inpaint_helpers.py 同款思路，补 movedim 等）。
import os
import sys
import types

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

import numpy as np  # noqa: E402

failures = []


def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")


# ── torch / comfy stub（test-only, no real ComfyUI runtime）──
class FakeTensor:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float32)

    @property
    def shape(self):
        return self.data.shape

    def dim(self):
        return self.data.ndim

    def unsqueeze(self, i):
        return FakeTensor(np.expand_dims(self.data, i))

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return FakeTensor(self.data.reshape(shape))

    def movedim(self, src, dst):
        return FakeTensor(np.moveaxis(self.data, src, dst))

    def any(self, dim=None):
        r = np.any(self.data, axis=dim)
        return FakeTensor(r) if isinstance(r, np.ndarray) else bool(r)

    def clamp(self, lo=None, hi=None):
        return FakeTensor(np.clip(self.data, lo, hi))

    def __mul__(self, o):
        return FakeTensor(self.data * (o.data if isinstance(o, FakeTensor) else o))

    def __gt__(self, o):
        return FakeTensor(self.data > o)

    def __getitem__(self, k):
        r = self.data[k]
        return FakeTensor(r) if isinstance(r, np.ndarray) else r


comfy = types.ModuleType("comfy")
comfy.utils = types.ModuleType("comfy.utils")
comfy.cli_args = types.ModuleType("comfy.cli_args")
comfy.cli_args.args = types.SimpleNamespace()
sys.modules["comfy"] = comfy
sys.modules["comfy.utils"] = comfy.utils
sys.modules["comfy.cli_args"] = comfy.cli_args

fake_torch = types.ModuleType("torch")


def _unwrap(t):
    return t.data if isinstance(t, FakeTensor) else np.asarray(t)


fake_torch.where = lambda cond: np.where(_unwrap(cond))
fake_torch.any = lambda t, dim=None: np.any(_unwrap(t), axis=dim)
sys.modules["torch"] = fake_torch

upscale_calls = []


def fake_common_upscale(samples, width, height, method="area", crop="disabled"):
    """最近邻近似缩放 + 记录目标尺寸（测试只关心参数与形状，不追求插值保真）。"""
    arr = _unwrap(samples)
    idx_h = np.arange(height) * arr.shape[2] // max(height, 1)
    idx_w = np.arange(width) * arr.shape[3] // max(width, 1)
    out = arr[:, :, idx_h][:, :, :, idx_w]
    upscale_calls.append({"in_shape": tuple(arr.shape), "w": int(width), "h": int(height),
                          "method": method, "in_data": arr})
    return FakeTensor(out)


comfy.utils.common_upscale = fake_common_upscale
# -------------------------------------------------------------

from nodes.model.krea2 import KREA2_SYSTEM_DEFAULT, TextEncodeKrea2, _flatten_to_rgb  # noqa: E402


def run():
    # ── _build_text ──
    text, template = TextEncodeKrea2._build_text("", "a cat", "<VIS>", "before prompt")
    check("build: 空 system 回退默认指令", template.startswith("<|im_start|>system\n" + KREA2_SYSTEM_DEFAULT))
    check("build: before 视觉在前", text == "<VIS>a cat")

    text, template = TextEncodeKrea2._build_text("SYS", "p", "<VIS>", "after prompt")
    check("build: 自定义 system 注入模板", "system\nSYS<|im_end|>" in template)
    check("build: after 文本在前", text == "p<VIS>")

    # ── _collect_indexed ──
    A, B, M = object(), object(), object()
    got = TextEncodeKrea2._collect_indexed(
        {"image2": B, "image10": A, "mask2": M, "image1": None, "prompt": "x"}, "image")
    check("collect: 编号收集且 None 忽略", got == {2: B, 10: A})
    check("collect: mask 前缀不串", TextEncodeKrea2._collect_indexed({"mask3": M}, "mask") == {3: M})

    # ── _flatten_to_rgb ──
    check("rgb: None 透传", _flatten_to_rgb(None) is None)
    rgb_in = FakeTensor([[[[1.5, -0.5, 0.7]]]])
    out = _flatten_to_rgb(rgb_in)
    check("rgb: clamp 到 [0,1]", np.allclose(out.data, [[[[1.0, 0.0, 0.7]]]]))
    rgba = FakeTensor([[[[1.0, 0.5, 0.25, 0.5], [9.0, 9.0, 9.0, 0.0]]]])
    out = _flatten_to_rgb(rgba)
    check("rgba: alpha 预乘黑底", np.allclose(out.data, [[[[0.5, 0.25, 0.125], [0.0, 0.0, 0.0]]]]))
    check("rgba: 输出 3 通道", out.shape[-1] == 3)
    hw3 = FakeTensor(np.zeros((2, 2, 4), dtype=np.float32))
    check("rgb: (H,W,C) 升维为 (1,H,W,C)", _flatten_to_rgb(hw3).shape == (1, 2, 2, 3))

    # ── _crop_to_mask ──
    img6 = FakeTensor(np.arange(6 * 6 * 3, dtype=np.float32).reshape(1, 6, 6, 3))
    check("crop: 无遮罩原样返回", TextEncodeKrea2._crop_to_mask(img6, None) is img6)

    zero_mask = FakeTensor(np.zeros((1, 6, 6), dtype=np.float32))
    check("crop: 全零遮罩保留整图", TextEncodeKrea2._crop_to_mask(img6, zero_mask) is img6)

    mask_pt = np.zeros((1, 6, 6), dtype=np.float32)
    mask_pt[0, 2, 3] = 1.0
    cropped = TextEncodeKrea2._crop_to_mask(img6, FakeTensor(mask_pt))
    check("crop: 单点包围盒裁剪", cropped.shape == (1, 1, 1, 3)
          and np.allclose(cropped.data, img6.data[:, 2:3, 3:4, :]))

    pad_mask = np.zeros((1, 6, 6), dtype=np.float32)
    pad_mask[0, 0, 0] = 1.0
    padded = TextEncodeKrea2._crop_to_mask(img6, FakeTensor(pad_mask), padding=0.5)
    check("crop: padding 扩展并 clamp 边界", padded.shape == (1, 4, 4, 3)
          and np.allclose(padded.data, img6.data[:, 0:4, 0:4, :]))

    n_before = len(upscale_calls)
    small_mask = np.zeros((1, 3, 3), dtype=np.float32)
    small_mask[0, 1, 1] = 1.0
    aligned = TextEncodeKrea2._crop_to_mask(img6, FakeTensor(small_mask))
    call = upscale_calls[n_before]
    check("crop: 尺寸不匹配先 bilinear 对齐", call["method"] == "bilinear"
          and (call["w"], call["h"]) == (6, 6))
    # 最近邻放大把单点扩散为 2x2 块（idx=[0,0,1,1,2,2]），包围盒随之扩为 [:, 2:4, 2:4]
    check("crop: 对齐后裁剪正确", aligned.shape == (1, 2, 2, 3)
          and np.allclose(aligned.data, img6.data[:, 2:4, 2:4, :]))

    mask_2d = np.zeros((6, 6), dtype=np.float32)
    mask_2d[5, 5] = 1.0
    cropped = TextEncodeKrea2._crop_to_mask(img6, FakeTensor(mask_2d))
    check("crop: dim==2 归一", cropped.shape == (1, 1, 1, 3)
          and np.allclose(cropped.data, img6.data[:, 5:6, 5:6, :]))

    mask_4d = np.zeros((2, 1, 6, 6), dtype=np.float32)
    mask_4d[0, 0, 1, 1] = 1.0
    mask_4d[1, 0, 4, 4] = 1.0
    unioned = TextEncodeKrea2._crop_to_mask(img6, FakeTensor(mask_4d))
    check("crop: dim==4 多帧并集包围盒", unioned.shape == (1, 4, 4, 3)
          and np.allclose(unioned.data, img6.data[:, 1:5, 1:5, :]))

    # ── _prepare_vision ──
    upscale_calls.clear()
    small_img = {"image1": FakeTensor(np.zeros((1, 80, 100, 3), dtype=np.float32))}
    imgs, prompt = TextEncodeKrea2._prepare_vision(small_img, 1.0, 0.0)
    check("vision: 小图保持原尺寸不放大", (upscale_calls[0]["w"], upscale_calls[0]["h"]) == (100, 80))
    check("vision: 单图无 Picture 前缀", "Picture" not in prompt
          and prompt == "<|vision_start|><|image_pad|><|vision_end|>")
    check("vision: 单图输出一个视觉张量", len(imgs) == 1 and imgs[0].shape == (1, 80, 100, 3))

    upscale_calls.clear()
    big_img = {"image1": FakeTensor(np.zeros((1, 1024, 2048, 3), dtype=np.float32))}
    TextEncodeKrea2._prepare_vision(big_img, 1.0, 0.0)
    check("vision: 大图缩到 megapixels 上限", (upscale_calls[0]["w"], upscale_calls[0]["h"]) == (1448, 724))

    upscale_calls.clear()
    batch_img = {"image1": FakeTensor(np.zeros((2, 64, 64, 3), dtype=np.float32))}
    imgs, _ = TextEncodeKrea2._prepare_vision(batch_img, 1.0, 0.0)
    check("vision: batch>1 只取首帧", upscale_calls[0]["in_shape"][0] == 1 and imgs[0].shape == (1, 64, 64, 3))

    upscale_calls.clear()
    rgba_img = {"image1": FakeTensor([[[[1.0, 0.5, 0.25, 0.5]]]])}
    imgs, _ = TextEncodeKrea2._prepare_vision(rgba_img, 1.0, 0.0)
    check("vision: RGBA 预乘后进缩放（3 通道）", upscale_calls[0]["in_shape"][1] == 3
          and np.allclose(upscale_calls[0]["in_data"].flatten(), [0.5, 0.25, 0.125]))

    upscale_calls.clear()
    two_imgs = {"image2": FakeTensor(np.full((1, 8, 8, 3), 0.8, dtype=np.float32)),
                "image1": FakeTensor(np.full((1, 8, 8, 3), 0.2, dtype=np.float32))}
    imgs, prompt = TextEncodeKrea2._prepare_vision(two_imgs, 1.0, 0.0)
    check("vision: 多图按编号排序", len(imgs) == 2 and np.allclose(imgs[0].data, 0.2)
          and np.allclose(imgs[1].data, 0.8))
    check("vision: 多图 Picture N 前缀", prompt.index("Picture 1:") < prompt.index("Picture 2:")
          and "Picture 2:" in prompt)

    # ── _fp8_hint ──
    hint = TextEncodeKrea2._fp8_hint(NotImplementedError("add_stub not implemented for Float8_e4m3fn"), [object()])
    check("fp8: Float8 崩溃映射提示", isinstance(hint, RuntimeError) and "bf16" in str(hint))
    check("fp8: 无图片不误报", TextEncodeKrea2._fp8_hint(NotImplementedError("Float8"), []) is None)
    check("fp8: 其他异常不误报", TextEncodeKrea2._fp8_hint(ValueError("x"), [object()]) is None)

    print()
    if failures:
        print(f"{len(failures)} FAILURES: {failures}")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    run()
