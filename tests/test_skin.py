import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from sf_utils.skin import estimate_skin_color, rgb_to_lab

SKIN_RGB = np.array([235, 200, 180], dtype=np.uint8)
BLUE_RGB = np.array([0, 0, 255], dtype=np.uint8)

failures = []


def check(name, cond):
    if cond:
        print(f"PASS {name}")
    else:
        failures.append(name)
        print(f"FAIL {name}")


def make_image(left_rgb, right_rgb, size=100):
    img = np.empty((size, size, 3), dtype=np.uint8)
    img[:, : size // 2] = left_rgb
    img[:, size // 2 :] = right_rgb
    return img


# --- rgb_to_lab sanity ---
white = rgb_to_lab(np.array([[255, 255, 255]], dtype=np.uint8))[0]
check("lab white L~100", abs(white[0] - 100.0) < 1.0)
check("lab white a/b~0", abs(white[1]) < 1.0 and abs(white[2]) < 1.0)

red = rgb_to_lab(np.array([[255, 0, 0]], dtype=np.uint8))[0]
check("lab red a>0", red[1] > 40.0)
check("lab red b>0 (yellowish)", red[2] > 40.0)

blue = rgb_to_lab(np.array([[0, 0, 255]], dtype=np.uint8))[0]
check("lab blue b<0", blue[2] < -40.0)

skin_lab = rgb_to_lab(np.array([SKIN_RGB], dtype=np.uint8))[0]
check("lab skin in thresholds", 7.8 < skin_lab[0] < 98.0 and -8.0 < skin_lab[1] < 52.0 and -8.0 < skin_lab[2] < 62.0)

# --- estimate_skin_color ---
mask = np.ones((100, 100), dtype=np.float32)

# 全肤色 → 输出 ≈ 肤色
img = np.empty((100, 100, 3), dtype=np.uint8)
img[:] = SKIN_RGB
out = estimate_skin_color(img, mask)
check("all skin -> skin color", out is not None and np.abs(out - SKIN_RGB / 255.0).max() < 2.0 / 255.0)

# 半肤色半蓝 → 蓝色被过滤，输出 ≈ 肤色
img = make_image(SKIN_RGB, BLUE_RGB)
out = estimate_skin_color(img, mask)
check("mixed -> skin filtered", out is not None and np.abs(out - SKIN_RGB / 255.0).max() < 2.0 / 255.0)

# 全蓝 → 无肤色像素，回退区域全像素均值
img = np.empty((100, 100, 3), dtype=np.uint8)
img[:] = BLUE_RGB
out = estimate_skin_color(img, mask)
check("all blue -> fallback mean", out is not None and np.abs(out - BLUE_RGB / 255.0).max() < 2.0 / 255.0)

# 少量肤色占比不足 → 回退均值（10% 肤色 + 90% 蓝，min_ratio=0.2）
img = np.empty((100, 100, 3), dtype=np.uint8)
img[:, :10] = SKIN_RGB
img[:, 10:] = BLUE_RGB
out = estimate_skin_color(img, mask, min_ratio=0.2)
mean = img.reshape(-1, 3).mean(axis=0) / 255.0
check("low ratio -> fallback mean", out is not None and np.abs(out - mean).max() < 2.0 / 255.0)

# 空 mask → None
out = estimate_skin_color(img, np.zeros((100, 100), dtype=np.float32))
check("empty mask -> None", out is None)

# 4D mask [1,H,W] 输入
img = np.empty((100, 100, 3), dtype=np.uint8)
img[:] = SKIN_RGB
out = estimate_skin_color(img, mask[np.newaxis, ...])
check("4d mask ok", out is not None and np.abs(out - SKIN_RGB / 255.0).max() < 2.0 / 255.0)

# mask 区域为局部小块（仅右半蓝区）→ 输出 = 蓝
partial = np.zeros((100, 100), dtype=np.float32)
partial[:, 50:] = 1.0
img = make_image(SKIN_RGB, BLUE_RGB)
out = estimate_skin_color(img, partial)
check("partial mask region", out is not None and np.abs(out - BLUE_RGB / 255.0).max() < 2.0 / 255.0)

if failures:
    print(f"\n{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("\nALL PASS")
