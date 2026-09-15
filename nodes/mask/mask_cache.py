"""SF Mask Cache —— 把追踪/分割得到的逐帧遮罩按名字持久化到
``user/sfnodes/mask_cache/``，下次同样的视频/图片可直接复用，跳过昂贵的
上游（如 SeC 视频分割的 4B 模型加载 + 推理）。

机制：
- 节点带一个 **lazy** ``masks`` 输入。命中缓存时 ``check_lazy_status`` 返回
  空列表，核心 ``graph.add_node`` 对 lazy 输入不建依赖 → 上游分割节点整条
  分支根本不执行（真跳过，不只是缓存结果）；未命中才请求 ``masks``，拿到
  上游结果后落盘，下次即命中。
- 缓存键 = ``name`` + ``signature``（可选，接点选坐标等）+ ``source``（可选，
  接源视频帧/参考图，首末帧哈希）。任一不匹配或 ``force=True`` 即视为失效
  并重算覆盖。``signature`` 语义为"完全相同才命中"，未接线时为空串参与比较。

落盘文件（同名前缀三件）：
- ``<name>.safetensors``：uint8 ``[T,H,W]`` 遮罩（0-255 量化，保留软遮罩）
- ``<name>.json``：元数据（n_frames/height/width/signature/source/name）
- ``<name>.png``：预览拼图（最多 4 帧横排，便于文件管理器查看）

无前端依赖，纯后端；列表路由供前端下拉刷新。
"""

import os

import numpy as np

from ...sf_utils import cache_store

_CATEGORY = "sfnodes/mask"
_CACHE_DIRNAME = "mask_cache"
_LAZY = {"lazy": True}
_PREVIEW_FRAMES = 4

# 通用缓存纯逻辑（名字清洗/路径/源指纹/元数据/命中/列表）单源收敛于
# sf_utils.cache_store（与 SFTrackDataCache 共用）；此处保留同名薄包装，
# 调用点与测试（monkeypatch cache_dir）零改动。


def cache_dir() -> str:
    return cache_store.sf_cache_dir(_CACHE_DIRNAME)


def clean_name(raw) -> str:
    return cache_store.clean_name(raw)


def _resolve_name(name, name_text=""):
    """缓存名解析单源：name_text 非空（列表取首个非空项）优先，否则回退下拉 name。"""
    candidates = name_text if isinstance(name_text, (list, tuple)) else (name_text,)
    for cand in candidates:
        if isinstance(cand, str) and cand.strip():
            return cand
    return name if isinstance(name, str) else ""


def cache_paths(name):
    return cache_store.cache_paths(cache_dir(), name)


def _to_numpy(x):
    return cache_store.to_numpy(x)


def quantize_masks(masks):
    """[H,W] / [T,H,W] 浮点遮罩 -> uint8 [T,H,W]（0-255）。"""
    a = _to_numpy(masks)
    if a.ndim == 2:
        a = a[None, ...]
    if a.ndim != 3:
        raise ValueError(f"SF Mask Cache: 期望 [T,H,W] 或 [H,W] 遮罩，实际 shape={a.shape}")
    a = np.clip(a.astype(np.float32), 0.0, 1.0) * 255.0
    return np.rint(a).astype(np.uint8)


def source_signature(source) -> str:
    return cache_store.source_signature(source)


def read_meta(name):
    return cache_store.read_meta(cache_dir(), name)


def cache_hit(name, signature="", source_sig="") -> bool:
    return cache_store.cache_hit(cache_dir(), name, signature, source_sig)


def list_cache_names() -> list:
    return cache_store.list_names(cache_dir())


def _write_preview(png_path, arr):
    from PIL import Image

    t = arr.shape[0]
    if t <= 0:
        return
    if t > _PREVIEW_FRAMES:
        idxs = [round(i * (t - 1) / (_PREVIEW_FRAMES - 1)) for i in range(_PREVIEW_FRAMES)]
    else:
        idxs = list(range(t))
    strip = np.concatenate([arr[i] for i in idxs], axis=1)
    Image.fromarray(strip, mode="L").save(png_path)


def save_mask_cache(name, masks, signature="", source_sig="", torch=None, sf=None,
                    write_preview=True):
    """遮罩落盘。返回 arr.shape。torch/sf 可注入（测试）。"""
    paths = cache_paths(name)
    if not paths:
        raise ValueError("SF Mask Cache: 缓存名称为空或非法")
    st_path, _meta_path, png_path = paths
    arr = quantize_masks(masks)
    if torch is None:
        import torch
    if sf is None:
        import safetensors.torch as sf
    sf.save_file({"masks": torch.from_numpy(arr)}, st_path)
    cache_store.write_meta(cache_dir(), name, {
        "name": clean_name(name),
        "n_frames": int(arr.shape[0]),
        "height": int(arr.shape[1]),
        "width": int(arr.shape[2]),
        "signature": str(signature or ""),
        "source": str(source_sig or ""),
    })
    if write_preview:
        try:
            _write_preview(png_path, arr)
        except Exception as e:
            print(f"[SFMaskCache] 预览图写入失败（忽略）: {e}")
    return arr.shape


def load_mask_cache(name, torch=None, sf=None):
    """从缓存读回 float32 [T,H,W] 遮罩（0-1）。"""
    paths = cache_paths(name)
    if not paths:
        raise ValueError("SF Mask Cache: 缓存名称为空或非法")
    if not os.path.isfile(paths[0]):
        raise FileNotFoundError(f"SF Mask Cache: 缓存不存在 {paths[0]}")
    if sf is None:
        import safetensors.torch as sf
    if torch is None:
        import torch
    arr = _to_numpy(sf.load_file(paths[0])["masks"])
    return torch.from_numpy(arr.astype(np.float32) / 255.0)


class SFMaskCache:
    @classmethod
    def INPUT_TYPES(cls):
        names = list_cache_names()
        return {
            "required": {
                "name": (
                    names if names else [""],
                    {"tooltip": "缓存名。下拉列出已有缓存；选「＋ 新建缓存…」输入新名字（跨工作流持久于 user/sfnodes/mask_cache/）"},
                ),
                "force": (
                    "BOOLEAN",
                    {"default": False, "label_on": "recompute", "label_off": "use cache",
                     "tooltip": "True=忽略缓存强制重算并覆盖（改了上游想更新时用）"},
                ),
            },
            "optional": {
                "masks": (
                    "MASK",
                    {**_LAZY, "tooltip": "上游遮罩（如 SeC 视频分割）。命中缓存时本输入不求值，上游整条不执行"},
                ),
                "signature": (
                    "STRING",
                    {"forceInput": True, "default": "", "tooltip": "可选签名（如 PointsEditor 点选坐标）；变化即缓存失效重算"},
                ),
                "source": (
                    "IMAGE",
                    {"tooltip": "可选源图/视频帧；首末帧哈希入缓存键，换源即失效"},
                ),
                "name_text": (
                    "STRING",
                    {"forceInput": True, "default": "",
                     "tooltip": "可选文本来源（如文件名/镜头名）：非空时覆盖上方下拉作为缓存名，空则用下拉。连接后下拉置灰"},
                ),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = ("按名字持久化逐帧遮罩到 user/sfnodes/mask_cache/：命中缓存时通过 lazy 输入跳过上游分割"
                   "（如 SeC）直接读盘，未命中/签名或源变化/force 才重算并覆盖。可选用 source 源图哈希与 "
                   "signature（点选坐标等）做缓存失效键，另存 PNG 预览拼图便于查看。"
                   "缓存名可由可选 name_text 文本输入覆盖（非空优先，接文件名/SFParsePath 等）。")

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # name 选项由前端从磁盘动态重建，超出 INPUT_TYPES 静态初始列表，跳过 "not in list" 校验
        return True

    def check_lazy_status(self, name, force=False, signature="", source=None, name_text="", **kwargs):
        # masks 未接线（纯读取场景）：无需拉取
        if "masks" not in kwargs:
            return []
        if force:
            return ["masks"]
        if cache_hit(_resolve_name(name, name_text), str(signature or ""), source_signature(source)):
            return []
        return ["masks"]

    def execute(self, name, force=False, signature="", source=None, masks=None, name_text=""):
        nm = clean_name(_resolve_name(name, name_text))
        if not nm:
            raise ValueError("SF Mask Cache: 请填写合法的缓存名（name_text 或下拉不能为空/非法）")
        sig = str(signature or "")
        src = source_signature(source)
        if masks is None:
            # 命中路径（上游被跳过）或纯读取
            if not os.path.isfile(cache_paths(nm)[0]):
                raise RuntimeError(f"SF Mask Cache: 缓存 '{nm}' 不存在，且未连接 masks（无法计算）")
            return (load_mask_cache(nm),)
        save_mask_cache(nm, masks, signature=sig, source_sig=src)
        return (masks,)


cache_store.register_list_route("/api/sfnodes/mask_cache/list", _CACHE_DIRNAME)
