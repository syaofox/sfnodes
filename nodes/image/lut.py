import copy
import os
import hashlib
import cv2
import numpy as np
import torch
import scipy.ndimage
from scipy.cluster.vq import kmeans2
import colour
from colour import LUT3D
from colour.algebra import table_interpolation_tetrahedral
from colour.io import write_LUT

from ...sf_utils.disk_state import sanitize_filename

_CATEGORY = "sfnodes/image"


def _get_luts_dir():
    try:
        import folder_paths
        luts_dir = os.path.join(folder_paths.base_path, "user", "sfnodes", "lut")
    except Exception:
        luts_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "user", "sfnodes", "lut")
        luts_dir = os.path.abspath(luts_dir)
    os.makedirs(luts_dir, exist_ok=True)
    return luts_dir


def _list_lut_files():
    luts_dir = _get_luts_dir()
    if not os.path.exists(luts_dir):
        return []
    return sorted([f for f in os.listdir(luts_dir) if f.lower().endswith(('.cube', '.spi3d', '.csp', '.spi1d', '.spimtx'))])


def _identity_grid(lut_size, dtype=np.float64):
    grid = np.linspace(0, 1, lut_size, dtype=dtype)
    r, g, b = np.meshgrid(grid, grid, grid, indexing="ij")
    return np.stack([r, g, b], axis=-1)


def _build_lut_table(r_acc, g_acc, b_acc, w_acc, lut_size):
    safe_wgt = np.maximum(w_acc, 1e-10)
    learned = np.stack([r_acc, g_acc, b_acc], axis=-1) / safe_wgt[..., None]
    identity = _identity_grid(lut_size, dtype=learned.dtype)
    confidence = w_acc / (w_acc + 5.0)
    lut_table = confidence[..., None] * learned + (1.0 - confidence[..., None]) * identity
    return lut_table


def _distribute_to_grid(src_colors, ref_colors, weights, lut_size, smooth_sigma=0.0):
    step = 1.0 / (lut_size - 1)
    grid_coords = src_colors / step

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

    weighted_w000 = weights * w000
    weighted_w100 = weights * w100
    weighted_w010 = weights * w010
    weighted_w001 = weights * w001
    weighted_w101 = weights * w101
    weighted_w011 = weights * w011
    weighted_w110 = weights * w110
    weighted_w111 = weights * w111

    r_acc = np.zeros((lut_size, lut_size, lut_size), dtype=np.float64)
    g_acc = np.zeros((lut_size, lut_size, lut_size), dtype=np.float64)
    b_acc = np.zeros((lut_size, lut_size, lut_size), dtype=np.float64)
    w_acc = np.zeros((lut_size, lut_size, lut_size), dtype=np.float64)

    ref_r, ref_g, ref_b = ref_colors[:, 0], ref_colors[:, 1], ref_colors[:, 2]

    np.add.at(r_acc, (r0, g0, b0), ref_r * weighted_w000)
    np.add.at(r_acc, (r1, g0, b0), ref_r * weighted_w100)
    np.add.at(r_acc, (r0, g1, b0), ref_r * weighted_w010)
    np.add.at(r_acc, (r0, g0, b1), ref_r * weighted_w001)
    np.add.at(r_acc, (r1, g0, b1), ref_r * weighted_w101)
    np.add.at(r_acc, (r0, g1, b1), ref_r * weighted_w011)
    np.add.at(r_acc, (r1, g1, b0), ref_r * weighted_w110)
    np.add.at(r_acc, (r1, g1, b1), ref_r * weighted_w111)

    np.add.at(g_acc, (r0, g0, b0), ref_g * weighted_w000)
    np.add.at(g_acc, (r1, g0, b0), ref_g * weighted_w100)
    np.add.at(g_acc, (r0, g1, b0), ref_g * weighted_w010)
    np.add.at(g_acc, (r0, g0, b1), ref_g * weighted_w001)
    np.add.at(g_acc, (r1, g0, b1), ref_g * weighted_w101)
    np.add.at(g_acc, (r0, g1, b1), ref_g * weighted_w011)
    np.add.at(g_acc, (r1, g1, b0), ref_g * weighted_w110)
    np.add.at(g_acc, (r1, g1, b1), ref_g * weighted_w111)

    np.add.at(b_acc, (r0, g0, b0), ref_b * weighted_w000)
    np.add.at(b_acc, (r1, g0, b0), ref_b * weighted_w100)
    np.add.at(b_acc, (r0, g1, b0), ref_b * weighted_w010)
    np.add.at(b_acc, (r0, g0, b1), ref_b * weighted_w001)
    np.add.at(b_acc, (r1, g0, b1), ref_b * weighted_w101)
    np.add.at(b_acc, (r0, g1, b1), ref_b * weighted_w011)
    np.add.at(b_acc, (r1, g1, b0), ref_b * weighted_w110)
    np.add.at(b_acc, (r1, g1, b1), ref_b * weighted_w111)

    np.add.at(w_acc, (r0, g0, b0), weighted_w000)
    np.add.at(w_acc, (r1, g0, b0), weighted_w100)
    np.add.at(w_acc, (r0, g1, b0), weighted_w010)
    np.add.at(w_acc, (r0, g0, b1), weighted_w001)
    np.add.at(w_acc, (r1, g0, b1), weighted_w101)
    np.add.at(w_acc, (r0, g1, b1), weighted_w011)
    np.add.at(w_acc, (r1, g1, b0), weighted_w110)
    np.add.at(w_acc, (r1, g1, b1), weighted_w111)

    if smooth_sigma > 0:
        sigma_grid = smooth_sigma * (lut_size - 1)
        r_acc = scipy.ndimage.gaussian_filter(r_acc, sigma=sigma_grid, mode="nearest")
        g_acc = scipy.ndimage.gaussian_filter(g_acc, sigma=sigma_grid, mode="nearest")
        b_acc = scipy.ndimage.gaussian_filter(b_acc, sigma=sigma_grid, mode="nearest")
        w_acc = scipy.ndimage.gaussian_filter(w_acc, sigma=sigma_grid, mode="nearest")

    return _build_lut_table(r_acc, g_acc, b_acc, w_acc, lut_size)


class SFLoadLUT:
    DESCRIPTION = "从 user/sfnodes/lut/ 加载 3D LUT 文件"

    @classmethod
    def INPUT_TYPES(s):
        files = _list_lut_files()
        if not files:
            files = ["(no LUT files)"]
        return {
            "required": {
                "file_name": (files, {"tooltip": "选择要加载的 LUT 文件"}),
            },
        }

    RETURN_TYPES = ("LUT",)
    RETURN_NAMES = ("lut",)
    FUNCTION = "load"
    CATEGORY = _CATEGORY

    @classmethod
    def IS_CHANGED(s, file_name):
        file_path = os.path.join(_get_luts_dir(), file_name)
        if os.path.exists(file_path):
            m = hashlib.sha256()
            with open(file_path, "rb") as f:
                m.update(f.read())
            return m.hexdigest()
        # 文件缺失/未选择：返回稳定字符串而非 NaN（NaN 恒不等于自身，
        # 会让节点缓存键折叠所有祖先、下游每次 Run 全量重跑）。
        return f"missing:{file_name}"

    def load(self, file_name):
        if file_name == "(no LUT files)" or not file_name:
            raise FileNotFoundError("No LUT files in user/sfnodes/lut/. Please add .cube files or use SF Extract LUT to create one.")
        file_path = os.path.join(_get_luts_dir(), file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"LUT file not found: {file_path}")
        lut = colour.read_LUT(file_path)
        return (lut,)


class SFApplyLUT:
    DESCRIPTION = "将 3D LUT 应用到图像，支持四面体插值和 LUT 域强度混合"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "输入图像"}),
                "lut": ("LUT", {"tooltip": "LUT 对象（来自 Load LUT 或 Extract LUT）"}),
                "interpolation": (
                    ["trilinear", "tetrahedral"],
                    {"tooltip": "trilinear=三线性插值（平滑），tetrahedral=四面体插值（少色带、更快）"},
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "LUT 域强度混合：在 LUT 表与恒等映射之间插值，再应用到图像",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = _CATEGORY

    def apply(self, image, lut, interpolation="tetrahedral", strength=1.0):
        if strength != 1.0:
            lut = copy.deepcopy(lut)
            size = lut.table.shape[0]
            identity = _identity_grid(size, dtype=np.float32)
            lut.table = np.clip(strength * lut.table + (1.0 - strength) * identity, 0, 1)

        kwargs = {}
        if interpolation == "tetrahedral":
            kwargs["interpolator"] = table_interpolation_tetrahedral

        batch_size = image.shape[0]
        result = torch.zeros_like(image)

        for b in range(batch_size):
            img_np = image[b].cpu().numpy().astype(np.float32)
            orig_shape = img_np.shape
            pixels = img_np.reshape(-1, 3)
            mapped = lut.apply(pixels, **kwargs).reshape(orig_shape)
            result[b] = torch.from_numpy(np.clip(mapped, 0, 1))

        return (result,)


class SFExtractLUT:
    DESCRIPTION = "从源图和参考图生成 3D LUT 并另存为 .cube 文件，可复用到其他图像"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mode": (
                    ["pixel", "color"],
                    {"tooltip": "pixel=逐像素映射（适合内容对齐的图），color=按颜色聚类映射（适合不同内容的图）"},
                ),
                "source_image": ("IMAGE", {"tooltip": "源图像（调色前）"}),
                "reference_image": ("IMAGE", {"tooltip": "参考图像（调色后，目标风格）"}),
                "filename": (
                    "STRING",
                    {"default": "extracted_lut.cube", "tooltip": "输出的 .cube 文件名，保存到 user/sfnodes/lut/（与 Load 节点同级）"},
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
                        "default": 0.0,
                        "min": 0.0,
                        "max": 0.15,
                        "step": 0.01,
                        "tooltip": "RGB 色彩空间平滑半径（0=不平滑，0.01=轻度，0.03=适中），不再改变颜色，仅在网格稀疏时启用",
                    },
                ),
                "num_clusters": (
                    "INT",
                    {
                        "default": 512,
                        "min": 64,
                        "max": 8192,
                        "step": 64,
                        "tooltip": "仅 color 模式有效：颜色聚类数，越大映射越精细但噪声越多（推荐 256~1024）",
                    },
                ),
            },
        }

    RETURN_TYPES = ("LUT",)
    RETURN_NAMES = ("lut",)
    FUNCTION = "extract"
    CATEGORY = _CATEGORY

    def extract(self, source_image, reference_image, filename, mode="pixel", lut_size=65, smooth_sigma=0.0, num_clusters=512):
        src_np = source_image.cpu().numpy().astype(np.float64)
        ref_np = reference_image.cpu().numpy().astype(np.float64)

        if src_np.shape[1:3] != ref_np.shape[1:3]:
            ref_np = np.stack([cv2.resize(ref_np[b], (src_np.shape[2], src_np.shape[1]), interpolation=cv2.INTER_LINEAR) for b in range(ref_np.shape[0])], axis=0)

        if src_np.shape[0] != ref_np.shape[0]:
            if ref_np.shape[0] < src_np.shape[0]:
                repeats = src_np.shape[0] - ref_np.shape[0]
                ref_np = np.concatenate([ref_np, ref_np[-1:].repeat(repeats, axis=0)], axis=0)
            else:
                ref_np = ref_np[:src_np.shape[0]]

        if mode == "color":
            h, w = src_np.shape[1], src_np.shape[2]
            max_pixels = 131072
            if h * w > max_pixels:
                scale = np.sqrt(max_pixels / (h * w))
                new_h, new_w = int(h * scale), int(w * scale)
                src_resized = np.stack([cv2.resize(src_np[b], (new_w, new_h), interpolation=cv2.INTER_LINEAR) for b in range(src_np.shape[0])], axis=0)
                ref_resized = np.stack([cv2.resize(ref_np[b], (new_w, new_h), interpolation=cv2.INTER_LINEAR) for b in range(ref_np.shape[0])], axis=0)
            else:
                src_resized, ref_resized = src_np, ref_np

            src_pixels = src_resized.reshape(-1, 3)
            ref_pixels = ref_resized.reshape(-1, 3)

            src_pixels = np.clip(src_pixels, 0, 1)
            ref_pixels = np.clip(ref_pixels, 0, 1)

            k = min(num_clusters, len(src_pixels))
            centroids, labels = kmeans2(src_pixels, k, minit="++", iter=100)

            ref_means = np.zeros_like(centroids)
            cluster_sizes = np.zeros(k, dtype=np.float64)
            for i in range(k):
                mask_i = labels == i
                count = mask_i.sum()
                cluster_sizes[i] = count
                if count > 0:
                    ref_means[i] = ref_pixels[mask_i].mean(axis=0)
                else:
                    ref_means[i] = centroids[i]

            lut_table = _distribute_to_grid(centroids, ref_means, cluster_sizes, lut_size, smooth_sigma)
        else:
            src_pixels = src_np.reshape(-1, 3)
            ref_pixels = ref_np.reshape(-1, 3)
            src_pixels = np.clip(src_pixels, 0, 1)
            ref_pixels = np.clip(ref_pixels, 0, 1)
            lut_table = _distribute_to_grid(src_pixels, ref_pixels, np.ones(len(src_pixels), dtype=np.float64), lut_size, smooth_sigma)

        lut_table = np.clip(lut_table, 0, 1).astype(np.float32)

        # filename 是自由 STRING：净化成安全的单段文件名（拒绝 ../、绝对路径），
        # 并强制 .cube 后缀，否则可越过 user/sfnodes/lut/ 目录任意写文件。
        filename = sanitize_filename(filename, "extracted_lut")
        if not filename.lower().endswith(".cube"):
            filename += ".cube"
        name = os.path.splitext(filename)[0]
        lut = LUT3D(table=lut_table, name=name)

        luts_dir = _get_luts_dir()
        filepath = os.path.join(luts_dir, filename)
        write_LUT(lut, filepath)

        return (lut,)
