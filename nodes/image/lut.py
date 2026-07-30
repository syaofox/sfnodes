import os
import numpy as np
import torch
import scipy.ndimage
import colour
from colour import LUT3D
from colour.io import write_LUT

_CATEGORY = "sfnodes/image"


def _get_luts_dir():
    try:
        import folder_paths
        luts_dir = os.path.join(folder_paths.get_output_directory(), "luts")
    except Exception:
        luts_dir = os.path.join(os.path.dirname(__file__), "..", "..", "output", "luts")
        luts_dir = os.path.abspath(luts_dir)
    os.makedirs(luts_dir, exist_ok=True)
    return luts_dir


class SFLoadLUT:
    DESCRIPTION = "加载 .cube 或其他格式的 3D LUT 文件"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "file_path": (
                    "STRING",
                    {"default": "", "tooltip": "LUT 文件路径，支持 .cube / .spi3d / .csp 等格式"},
                ),
            },
        }

    RETURN_TYPES = ("LUT",)
    RETURN_NAMES = ("lut",)
    FUNCTION = "load"
    CATEGORY = _CATEGORY

    def load(self, file_path):
        file_path = file_path.strip()
        if not file_path:
            raise ValueError("LUT file path cannot be empty")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"LUT file not found: {file_path}")
        lut = colour.read_LUT(file_path)
        return (lut,)


class SFApplyLUT:
    DESCRIPTION = "将 3D LUT 应用到图像，支持强度混合"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "输入图像"}),
                "lut": ("LUT", {"tooltip": "LUT 对象（来自 Load LUT 或 Extract LUT）"}),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "应用强度，0=完全保留原图，1=完全应用，>1=过度拉伸",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = _CATEGORY

    def apply(self, image, lut, strength=1.0):
        batch_size, height, width, channels = image.shape
        result = torch.zeros_like(image)

        for b in range(batch_size):
            img_np = image[b].cpu().numpy().astype(np.float32)
            orig_shape = img_np.shape
            pixels = img_np.reshape(-1, 3)

            mapped = lut.apply(pixels).reshape(orig_shape)
            mapped = np.clip(mapped, 0, 1)

            if strength != 1.0:
                mapped = np.clip(strength * mapped + (1.0 - strength) * img_np, 0, 1)

            result[b] = torch.from_numpy(mapped)

        return (result,)


class SFExtractLUT:
    DESCRIPTION = "从源图和参考图生成 3D LUT 并另存为 .cube 文件，可复用到其他图像"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source_image": ("IMAGE", {"tooltip": "源图像（调色前）"}),
                "reference_image": ("IMAGE", {"tooltip": "参考图像（调色后，目标风格）"}),
                "filename": (
                    "STRING",
                    {"default": "extracted_lut.cube", "tooltip": "输出的 .cube 文件名，保存到 ComfyUI/output/luts/"},
                ),
                "lut_size": (
                    "INT",
                    {
                        "default": 65,
                        "min": 17,
                        "max": 129,
                        "step": 16,
                        "tooltip": "LUT 网格精度，65 为标准精度，33 更小更快，129 更精细",
                    },
                ),
                "smooth_sigma": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.1,
                        "tooltip": "3D 高斯平滑强度，0=不平滑，越大 LUT 越平滑（推荐 0.5~1.5）",
                    },
                ),
            },
        }

    RETURN_TYPES = ("LUT",)
    RETURN_NAMES = ("lut",)
    FUNCTION = "extract"
    CATEGORY = _CATEGORY

    def extract(self, source_image, reference_image, filename, lut_size=65, smooth_sigma=1.0):
        src = source_image.cpu().numpy().reshape(-1, 3).astype(np.float64)
        ref = reference_image.cpu().numpy().reshape(-1, 3).astype(np.float64)

        src = np.clip(src, 0, 1)
        ref = np.clip(ref, 0, 1)

        step = 1.0 / (lut_size - 1)
        grid_coords = src / step
        r0 = np.floor(grid_coords[:, 0]).astype(np.int64)
        g0 = np.floor(grid_coords[:, 1]).astype(np.int64)
        b0 = np.floor(grid_coords[:, 2]).astype(np.int64)
        r1 = np.clip(r0 + 1, 0, lut_size - 1)
        g1 = np.clip(g0 + 1, 0, lut_size - 1)
        b1 = np.clip(b0 + 1, 0, lut_size - 1)
        r0 = np.clip(r0, 0, lut_size - 1)
        g0 = np.clip(g0, 0, lut_size - 1)
        b0 = np.clip(b0, 0, lut_size - 1)

        rf = grid_coords[:, 0] - r0
        gf = grid_coords[:, 1] - g0
        bf = grid_coords[:, 2] - b0

        w000 = (1 - rf) * (1 - gf) * (1 - bf)
        w100 = rf * (1 - gf) * (1 - bf)
        w010 = (1 - rf) * gf * (1 - bf)
        w001 = (1 - rf) * (1 - gf) * bf
        w101 = rf * (1 - gf) * bf
        w011 = (1 - rf) * gf * bf
        w110 = rf * gf * (1 - bf)
        w111 = rf * gf * bf

        lut_acc = np.zeros((lut_size, lut_size, lut_size, 3), dtype=np.float64)
        lut_wgt = np.zeros((lut_size, lut_size, lut_size), dtype=np.float64)

        out_r = ref[:, 0]
        out_g = ref[:, 1]
        out_b = ref[:, 2]

        np.add.at(lut_acc[:, :, :, 0], (r0, g0, b0), out_r * w000)
        np.add.at(lut_acc[:, :, :, 0], (r1, g0, b0), out_r * w100)
        np.add.at(lut_acc[:, :, :, 0], (r0, g1, b0), out_r * w010)
        np.add.at(lut_acc[:, :, :, 0], (r0, g0, b1), out_r * w001)
        np.add.at(lut_acc[:, :, :, 0], (r1, g0, b1), out_r * w101)
        np.add.at(lut_acc[:, :, :, 0], (r0, g1, b1), out_r * w011)
        np.add.at(lut_acc[:, :, :, 0], (r1, g1, b0), out_r * w110)
        np.add.at(lut_acc[:, :, :, 0], (r1, g1, b1), out_r * w111)

        np.add.at(lut_acc[:, :, :, 1], (r0, g0, b0), out_g * w000)
        np.add.at(lut_acc[:, :, :, 1], (r1, g0, b0), out_g * w100)
        np.add.at(lut_acc[:, :, :, 1], (r0, g1, b0), out_g * w010)
        np.add.at(lut_acc[:, :, :, 1], (r0, g0, b1), out_g * w001)
        np.add.at(lut_acc[:, :, :, 1], (r1, g0, b1), out_g * w101)
        np.add.at(lut_acc[:, :, :, 1], (r0, g1, b1), out_g * w011)
        np.add.at(lut_acc[:, :, :, 1], (r1, g1, b0), out_g * w110)
        np.add.at(lut_acc[:, :, :, 1], (r1, g1, b1), out_g * w111)

        np.add.at(lut_acc[:, :, :, 2], (r0, g0, b0), out_b * w000)
        np.add.at(lut_acc[:, :, :, 2], (r1, g0, b0), out_b * w100)
        np.add.at(lut_acc[:, :, :, 2], (r0, g1, b0), out_b * w010)
        np.add.at(lut_acc[:, :, :, 2], (r0, g0, b1), out_b * w001)
        np.add.at(lut_acc[:, :, :, 2], (r1, g0, b1), out_b * w101)
        np.add.at(lut_acc[:, :, :, 2], (r0, g1, b1), out_b * w011)
        np.add.at(lut_acc[:, :, :, 2], (r1, g1, b0), out_b * w110)
        np.add.at(lut_acc[:, :, :, 2], (r1, g1, b1), out_b * w111)

        np.add.at(lut_wgt, (r0, g0, b0), w000)
        np.add.at(lut_wgt, (r1, g0, b0), w100)
        np.add.at(lut_wgt, (r0, g1, b0), w010)
        np.add.at(lut_wgt, (r0, g0, b1), w001)
        np.add.at(lut_wgt, (r1, g0, b1), w101)
        np.add.at(lut_wgt, (r0, g1, b1), w011)
        np.add.at(lut_wgt, (r1, g1, b0), w110)
        np.add.at(lut_wgt, (r1, g1, b1), w111)

        mask = lut_wgt > 0
        lut_table = np.zeros_like(lut_acc)
        safe_wgt = np.maximum(lut_wgt, 1e-10)
        for c in range(3):
            lut_table[:, :, :, c] = np.where(mask, lut_acc[:, :, :, c] / safe_wgt, 0)

        if smooth_sigma > 0:
            lut_table = scipy.ndimage.gaussian_filter(
                lut_table, sigma=smooth_sigma, mode="nearest"
            )

        lut_table = np.clip(lut_table, 0, 1).astype(np.float32)

        name = os.path.splitext(filename)[0]
        lut = LUT3D(table=lut_table, name=name)

        luts_dir = _get_luts_dir()
        filepath = os.path.join(luts_dir, filename)
        write_LUT(lut, filepath)

        return (lut,)
