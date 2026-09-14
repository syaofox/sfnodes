"""SF Track Data Cache —— 在 ``SAM3_TRACK_DATA`` 层做 lazy 磁盘缓存。

与 ``nodes/mask/mask_cache.py``（SFMaskCache，MASK 层）同构，区别是缓存的
是 SAM3 追踪的原始 track_data（``{packed_masks, n_frames, scores, orig_size}``），
因此**完整保留多对象身份与 scores**，不走 `SAM3_TrackToMask` 的并集塌缩。
命中时 ``check_lazy_status`` 返回空列表 → ``SAM3_VideoTrack`` 整条不执行。

通用缓存纯逻辑（名字清洗/路径/源指纹/元数据/命中/列表/路由）单源收敛于
``sf_utils.cache_store``；本文件只负责 track_data 的 safetensors 存取与
多对象预览。存储目录 ``user/sfnodes/track_cache/``：

- ``<name>.safetensors``：``packed_masks``（uint8 ``[T,N,H,W//8]``，原样位打包）、
  ``scores``（float32 ``[N]``）；无对象时写 ``_empty`` 占位以满足 save_file
- ``<name>.json``：``{name,n_frames,height,width,num_objects,signature,source}``
- ``<name>.png``：最多 4 帧预览，按对象上色（失败忽略）
"""

import os

import numpy as np

from ...sf_utils import cache_store

_CATEGORY = "sfnodes/image"
_CACHE_DIRNAME = "track_cache"
_LAZY = {"lazy": True}
_PREVIEW_FRAMES = 4
# 预览用调色板（数据常量；与核心 SAM3_TrackPreview.COLORS 同族配色）
_PREVIEW_COLORS = [
    (0.12, 0.47, 0.71), (1.0, 0.5, 0.05), (0.17, 0.63, 0.17), (0.84, 0.15, 0.16),
    (0.58, 0.4, 0.74), (0.55, 0.34, 0.29), (0.89, 0.47, 0.76), (0.5, 0.5, 0.5),
    (0.74, 0.74, 0.13), (0.09, 0.75, 0.81), (0.94, 0.76, 0.06), (0.42, 0.68, 0.84),
]


def cache_dir() -> str:
    return cache_store.sf_cache_dir(_CACHE_DIRNAME)


def clean_name(raw) -> str:
    return cache_store.clean_name(raw)


def cache_paths(name):
    return cache_store.cache_paths(cache_dir(), name)


def source_signature(source) -> str:
    return cache_store.source_signature(source)


def read_meta(name):
    return cache_store.read_meta(cache_dir(), name)


def cache_hit(name, signature="", source_sig="") -> bool:
    return cache_store.cache_hit(cache_dir(), name, signature, source_sig)


def list_cache_names() -> list:
    return cache_store.list_names(cache_dir())


def _write_preview(png_path, packed_np):
    """packed uint8 [T,N,H,W//8] -> 最多 4 帧、按对象上色的横排拼图。"""
    if packed_np is None or packed_np.shape[1] == 0 or packed_np.shape[0] == 0:
        return
    import torch
    from PIL import Image
    from comfy.ldm.sam3.tracker import unpack_masks

    t = packed_np.shape[0]
    if t > _PREVIEW_FRAMES:
        idxs = [round(i * (t - 1) / (_PREVIEW_FRAMES - 1)) for i in range(_PREVIEW_FRAMES)]
    else:
        idxs = list(range(t))

    strips = []
    for idx in idxs:
        masks = unpack_masks(torch.from_numpy(packed_np[idx]))  # [N,H,W] bool
        n, h, w = masks.shape
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for j in range(n):
            color = (np.array(_PREVIEW_COLORS[j % len(_PREVIEW_COLORS)]) * 255).astype(np.uint8)
            img[cache_store.to_numpy(masks[j])] = color
        strips.append(img)
    strip = np.concatenate(strips, axis=1)
    Image.fromarray(strip, mode="RGB").save(png_path)


def save_track_cache(name, track_data, signature="", source_sig="", torch=None, sf=None,
                     write_preview=True):
    """track_data 落盘。返回输入 dict。torch/sf 可注入（测试）。"""
    paths = cache_paths(name)
    if not paths:
        raise ValueError("SF Track Data Cache: 缓存名称为空或非法")
    if not isinstance(track_data, dict) or "packed_masks" not in track_data:
        raise ValueError("SF Track Data Cache: 输入不是有效的 SAM3_TRACK_DATA")
    st_path, _meta_path, png_path = paths
    if torch is None:
        import torch
    if sf is None:
        import safetensors.torch as sf

    packed = track_data.get("packed_masks")
    packed_np = None
    tensors = {}
    if packed is not None:
        packed_np = cache_store.to_numpy(packed).astype(np.uint8)
        tensors["packed_masks"] = torch.from_numpy(packed_np)
    scores = track_data.get("scores") or []
    tensors["scores"] = torch.from_numpy(
        np.asarray([float(s) for s in scores], dtype=np.float64))
    if packed is None:
        tensors["_empty"] = torch.from_numpy(np.zeros((0,), dtype=np.uint8))
    sf.save_file(tensors, st_path)

    n_frames = int(track_data.get("n_frames", packed_np.shape[0] if packed_np is not None else 0))
    orig = track_data.get("orig_size") or (0, 0)
    cache_store.write_meta(cache_dir(), name, {
        "name": clean_name(name),
        "n_frames": n_frames,
        "height": int(orig[0]),
        "width": int(orig[1]),
        "num_objects": int(packed_np.shape[1]) if packed_np is not None else 0,
        "signature": str(signature or ""),
        "source": str(source_sig or ""),
    })
    if write_preview:
        try:
            _write_preview(png_path, packed_np)
        except Exception as e:
            print(f"[SFTrackDataCache] 预览图写入失败（忽略）: {e}")
    return track_data


def load_track_cache(name, torch=None, sf=None):
    """从缓存重建 track_data dict（packed_masks 为张量或 None）。"""
    paths = cache_paths(name)
    if not paths:
        raise ValueError("SF Track Data Cache: 缓存名称为空或非法")
    if not os.path.isfile(paths[0]):
        raise FileNotFoundError(f"SF Track Data Cache: 缓存不存在 {paths[0]}")
    if sf is None:
        import safetensors.torch as sf
    if torch is None:
        import torch
    tensors = sf.load_file(paths[0])
    meta = read_meta(name) or {}
    packed = tensors.get("packed_masks")
    if packed is not None:
        packed = torch.from_numpy(cache_store.to_numpy(packed).astype(np.uint8))
    scores_t = tensors.get("scores")
    scores = ([float(x) for x in cache_store.to_numpy(scores_t).tolist()]
              if scores_t is not None else [])
    return {
        "packed_masks": packed,
        "n_frames": int(meta.get("n_frames", 0)),
        "scores": scores,
        "orig_size": (int(meta.get("height", 0)), int(meta.get("width", 0))),
    }


class SFTrackDataCache:
    @classmethod
    def INPUT_TYPES(cls):
        names = list_cache_names()
        return {
            "required": {
                "name": (
                    names if names else [""],
                    {"tooltip": "缓存名。下拉列出已有缓存；选「＋ 新建缓存…」输入新名字（跨工作流持久于 user/sfnodes/track_cache/）"},
                ),
                "force": (
                    "BOOLEAN",
                    {"default": False, "label_on": "recompute", "label_off": "use cache",
                     "tooltip": "True=忽略缓存强制重算并覆盖（改了上游想更新时用）"},
                ),
            },
            "optional": {
                "track_data": (
                    "SAM3_TRACK_DATA",
                    {**_LAZY, "tooltip": "上游 SAM3 追踪数据（SAM3 Video Track）。命中缓存时本输入不求值，追踪整条不执行"},
                ),
                "signature": (
                    "STRING",
                    {"forceInput": True, "default": "", "tooltip": "可选签名（如文本 prompt / 点选坐标）；变化即缓存失效重算"},
                ),
                "source": (
                    "IMAGE",
                    {"tooltip": "可选源图/视频帧；首末帧哈希入缓存键，换源即失效"},
                ),
            },
        }

    RETURN_TYPES = ("SAM3_TRACK_DATA",)
    RETURN_NAMES = ("track_data",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = ("在 SAM3_TRACK_DATA 层持久化追踪结果到 user/sfnodes/track_cache/：命中缓存时通过 lazy 输入"
                   "跳过 SAM3_VideoTrack 直接读盘，完整保留多对象身份与 scores（区别于 MASK 层缓存的并集塌缩）。"
                   "可选用 source 源图哈希与 signature 做缓存失效键，另存按对象上色的 PNG 预览拼图。")

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # name 选项由前端从磁盘动态重建，超出 INPUT_TYPES 静态初始列表，跳过 "not in list" 校验
        return True

    def check_lazy_status(self, name, force=False, signature="", source=None, **kwargs):
        # track_data 未接线（纯读取场景）：无需拉取
        if "track_data" not in kwargs:
            return []
        if force:
            return ["track_data"]
        if cache_hit(name, str(signature or ""), source_signature(source)):
            return []
        return ["track_data"]

    def execute(self, name, force=False, signature="", source=None, track_data=None):
        nm = clean_name(name)
        if not nm:
            raise ValueError("SF Track Data Cache: 请填写合法的缓存名（不能为空或含路径分隔符）")
        sig = str(signature or "")
        src = source_signature(source)
        if track_data is None:
            # 命中路径（上游被跳过）或纯读取
            if not os.path.isfile(cache_paths(nm)[0]):
                raise RuntimeError(f"SF Track Data Cache: 缓存 '{nm}' 不存在，且未连接 track_data（无法计算）")
            return (load_track_cache(nm),)
        save_track_cache(nm, track_data, signature=sig, source_sig=src)
        return (track_data,)


cache_store.register_list_route("/api/sfnodes/track_cache/list", _CACHE_DIRNAME)
