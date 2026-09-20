"""共享的磁盘缓存纯逻辑：SFMaskCache / SFTrackDataCache 共用。

两个缓存节点各持独立目录（``user/sfnodes/<subdir>/``），但"名字清洗 / 三件
路径 / 失效键解析 / 元数据读写 / 命中判定 / 列表 / 列表路由"完全同构，此前
在 mask_cache 内实现；抽到此处收敛为单源，节点只在 save/load 层处理各自的
张量布局（MASK / SAM3_TRACK_DATA），避免语义分叉。

缓存键 = 缓存名 + ``source_key``（唯一失效键，required 非空；见 §111）：节点
不再做源图/帧哈希指纹，视频/图片的失效信号由用户以文本键显式提供。

纯函数（无 ComfyUI 硬依赖），可独立测试。``register_list_route`` 惰性 import
server/aiohttp（无 ComfyUI 时静默跳过）。
"""

import json
import os
import threading

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


def first_nonempty_text(raw) -> str:
    """取首个非空字符串（list/tuple 依序尝试）并 strip；无则返回 ""。

    link 输入在调度层可能表现为列表（多调用），"空则回退"的输入解析
    （缓存名 name_text / 失效键 source_key）共用此语义。
    """
    candidates = raw if isinstance(raw, (list, tuple)) else (raw,)
    for cand in candidates:
        if isinstance(cand, str) and cand.strip():
            return cand.strip()
    return ""


def required_source_key(raw) -> str:
    """解析并校验唯一失效键 ``source_key``：非空即返回（strip），空则报错。

    两个缓存节点共用同一错误文案与语义——键是 required 且必须非空，
    避免"接了线但为空串"退化成仅按名字命中的静默旧缓存。
    """
    key = first_nonempty_text(raw)
    if not key:
        raise ValueError("source_key 不能为空（缓存唯一失效键：如 视频路径/文件名 + 帧窗口/分辨率文本）")
    return key


def atomic_save_tensors(sf, tensors, path) -> None:
    """safetensors 原子写：pid+tid 临时名 + os.replace；失败清理临时文件后抛错。

    与 ``disk_state.atomic_write_bytes`` 同约定（临时名带 pid + 线程 id 防并发
    撞名）；不整块序列化进内存——``save_file`` 直接写临时路径再原子替换，
    避免中断留下半截缓存被后续命中误读。
    """
    tmp = "%s.%d.%d.tmp" % (path, os.getpid(), threading.get_ident())
    try:
        sf.save_file(tensors, tmp)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def to_numpy(x):
    try:
        return x.detach().cpu().numpy()
    except AttributeError:
        return np.asarray(x)


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


def cache_hit(base_dir, name, key="") -> bool:
    """命中判定：meta 的 ``source_key`` 与当前键完全相同。

    旧版缓存文件（§111 前）存的是 ``source``（帧哈希）与 ``signature``，
    这里回退读旧字段——旧值必然与文本键不同 → 首次 miss 重写一次后稳定。
    """
    meta = read_meta(base_dir, name)
    if meta is None:
        return False
    stored = meta.get("source_key", meta.get("source", ""))
    return str(stored) == str(key or "")


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
