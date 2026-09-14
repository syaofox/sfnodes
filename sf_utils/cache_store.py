"""共享的磁盘缓存纯逻辑：SFMaskCache / SFTrackDataCache 共用。

两个缓存节点各持独立目录（``user/sfnodes/<subdir>/``），但"名字清洗 / 三件
路径 / 源图指纹 / 元数据读写 / 命中判定 / 列表 / 列表路由"完全同构，此前
在 mask_cache 内实现；抽到此处收敛为单源，节点只在 save/load 层处理各自的
张量布局（MASK / SAM3_TRACK_DATA），避免语义分叉。

纯函数（无 ComfyUI 硬依赖），可独立测试。``register_list_route`` 惰性 import
server/aiohttp（无 ComfyUI 时静默跳过）。
"""

import hashlib
import json
import os

import numpy as np

from .disk_state import atomic_write_json, sanitize_filename, sf_user_dir


def sf_cache_dir(subdir) -> str:
    d = os.path.join(sf_user_dir(), subdir)
    os.makedirs(d, exist_ok=True)
    return d


def clean_name(raw) -> str:
    """把用户输入净化为安全的单段缓存名（空/非法返回 ""）。"""
    if not isinstance(raw, str):
        return ""
    return sanitize_filename(raw.strip(), fallback="")


def cache_paths(base_dir, name):
    """返回 (safetensors, json, png) 三个绝对路径；名称非法返回 None。"""
    n = clean_name(name)
    if not n:
        return None
    base = os.path.join(base_dir, n)
    return base + ".safetensors", base + ".json", base + ".png"


def to_numpy(x):
    try:
        return x.detach().cpu().numpy()
    except AttributeError:
        return np.asarray(x)


def source_signature(source) -> str:
    """源图像批 [B,H,W,C] 的轻量指纹：形状 + 首/末帧 16×16 下采样哈希。

    源未接线/形状不符返回 ""。只取首末帧避免整批哈希的开销（驱动视频可能
    几十帧）。"""
    if source is None:
        return ""
    a = to_numpy(source)
    if a.ndim != 4:
        return ""
    b = int(a.shape[0])
    idxs = [0] if b <= 1 else [0, b - 1]
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    for i in idxs:
        f = a[i]
        step_y = max(1, f.shape[0] // 16)
        step_x = max(1, f.shape[1] // 16)
        f = f[::step_y, ::step_x]
        h.update(np.ascontiguousarray((np.clip(f, 0, 1) * 255).astype(np.uint8)).tobytes())
    return h.hexdigest()[:16]


def write_meta(base_dir, name, meta):
    paths = cache_paths(base_dir, name)
    if not paths:
        raise ValueError("缓存名称为空或非法")
    atomic_write_json(paths[1], meta)


def read_meta(base_dir, name):
    """读缓存元数据 dict；缺失/损坏返回 None。"""
    paths = cache_paths(base_dir, name)
    if not paths or not os.path.isfile(paths[0]):
        return None
    try:
        with open(paths[1], encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def cache_hit(base_dir, name, signature="", source_sig="") -> bool:
    meta = read_meta(base_dir, name)
    if meta is None:
        return False
    return (str(meta.get("signature", "")) == str(signature or "")
            and str(meta.get("source", "")) == str(source_sig or ""))


def list_names(base_dir) -> list:
    try:
        return sorted(fn[: -len(".safetensors")] for fn in os.listdir(base_dir)
                      if fn.endswith(".safetensors"))
    except OSError:
        return []


def register_list_route(route_path, subdir):
    """注册 GET route_path -> {"names": [...]}；无 PromptServer 时静默跳过。"""
    try:
        from server import PromptServer
        from aiohttp import web

        ins = getattr(PromptServer, "instance", None)
        if ins is None or not hasattr(ins, "routes"):
            return

        @ins.routes.get(route_path)
        async def _list_route(request):
            return web.json_response({"names": list_names(sf_cache_dir(subdir))})
    except Exception:
        pass
