"""SF Load Images Cursor — 目录游标式单图加载器（每次运行消费下一张）。

每次运行（每个 prompt）取目录里的下一张图：IS_CHANGED 只读返回"下一张"的
游标 token（prompt 开始、节点执行前求值），使本轮缓存键与上轮不同 → 节点与
全部下游重跑；执行时才真正推进并解码。配合"排队 N 次"即可批量跑单图流水线
（每张独立 history / 独立命名），队列耗尽按 cycle 回卷。

与 SFLoadImagesPath（一次 Run 把整目录加载成批，配合循环节点）语义互补。
机制参考 ostris_nodes_comfyui 的 OstrisBatchImageLoader，本实现修掉了它的
全局单例游标、glob 无序、无 RGB/MASK 处理与失败递归四个坑，见
experience/nodes-image.md §147。
"""

import json
import os
import random
import time

import numpy as np
import torch

from ...sf_utils.disk_state import atomic_write_json, sanitize_id, sf_user_dir
from ...sf_utils.image_sources import (
    image_alpha,
    open_image,
    resolve_folder,
    sorted_image_files,
)

_CATEGORY = "sfnodes/image"

_CAPTION_EXTS = ("txt", "caption")
_ORDERS = ("sequential", "shuffle", "random")
_CURSOR_SUBDIR = "image_cursor"

# 内存游标：key = unique_id（无则目录+参数），跨 Run 存活（节点实例每轮重建）。
_CURSORS = {}


def _as_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _params(file_prefix, image_load_cap, skip_first_images, select_every_nth, order):
    """切片参数归一：连线输入在 IS_CHANGED 里是 None（experience/platform.md §134），
    IS_CHANGED 与执行路径统一走本函数，保证两边得到同形参数。"""
    return {
        "prefix": str(file_prefix or "").lower(),
        "cap": max(0, _as_int(image_load_cap, 0)),
        "skip": max(0, _as_int(skip_first_images, 0)),
        "nth": max(1, _as_int(select_every_nth, 1)),
        "order": order if order in _ORDERS else _ORDERS[0],
    }


def _dir_fingerprint(directory):
    try:
        return os.stat(directory).st_mtime_ns
    except OSError:
        return None


def _state_key(unique_id, directory, params, reset_token):
    if unique_id:
        return str(unique_id)
    return f"anon|{directory}|{params}|{reset_token}"


def _cursor_path(state_name):
    return os.path.join(sf_user_dir(), _CURSOR_SUBDIR, sanitize_id(state_name, "cursor") + ".json")


def _caption_for(image_path):
    """同名 .txt / .caption 侧车（多行合并为一行标签串）；无侧车返回空串。"""
    stem = os.path.splitext(image_path)[0]
    for ext in _CAPTION_EXTS:
        path = f"{stem}.{ext}"
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = [ln.strip() for ln in f.read().split("\n")]
        except OSError:
            return ""
        return ", ".join(x for x in lines if x)
    return ""


def _new_state(directory, params, reset_token, state_name):
    return {
        "directory": directory,
        "params": params,
        "reset_token": reset_token,
        "state_name": state_name,
        "files": None,
        "fingerprint": None,
        "pos": 0,
        "cycle_index": 0,
        "wrap": True,
        "last_path": "",
        "last_index": 0,
    }


def _seed_from_disk(st):
    """state_name 非空时按磁盘记录续跑；目录/参数/令牌任一不一致则忽略（从头跑）。"""
    if not st["state_name"]:
        return
    try:
        with open(_cursor_path(st["state_name"]), "r", encoding="utf-8") as f:
            rec = json.load(f)
    except (OSError, ValueError):
        return
    if not isinstance(rec, dict):
        return
    if (rec.get("directory"), rec.get("params"), rec.get("reset_token")) != (
        st["directory"], st["params"], st["reset_token"],
    ):
        return
    st["last_path"] = str(rec.get("last_path") or "")
    st["last_index"] = max(0, _as_int(rec.get("last_index"), 0))
    st["cycle_index"] = max(0, _as_int(rec.get("cycle_index"), 0))
    print(f"[SFLoadImagesCursor] 游标续跑 {st['state_name']!r}：{st['last_path'] or st['last_index']}")


def _resume_pos(files, last_path, last_index):
    """按上一张路径在列表中的位置对齐（目录增删/重排/换档不丢进度），找不到退回索引夹取。"""
    if last_path:
        try:
            return files.index(last_path) + 1
        except ValueError:
            pass
    return min(max(0, last_index), len(files))


def _rebuild(st):
    try:
        files = sorted_image_files(st["directory"], st["params"]["cap"], st["params"]["skip"], st["params"]["nth"])
    except OSError:
        # 目录在排队期间被删除/移动：留空列表，执行侧报明确错误（IS_CHANGED 不抛）。
        files = []
    prefix = st["params"]["prefix"]
    if prefix:
        files = [p for p in files if os.path.basename(p).lower().startswith(prefix)]
    if st["params"]["order"] == "shuffle":
        random.shuffle(files)
    st["files"] = files
    st["fingerprint"] = _dir_fingerprint(st["directory"])
    st["pos"] = _resume_pos(files, st["last_path"], st["last_index"])


def _ensure_state(key, directory, params, reset_token, state_name):
    st = _CURSORS.get(key)
    if (st is None or st["params"] != params or st["reset_token"] != reset_token
            or st["directory"] != directory or st["state_name"] != state_name):
        st = _new_state(directory, params, reset_token, state_name)
        _CURSORS[key] = st
        _seed_from_disk(st)
        _rebuild(st)
        return st
    if st["files"] is None or st["fingerprint"] != _dir_fingerprint(directory):
        _rebuild(st)
    return st


def _token(st):
    """IS_CHANGED 返回值：只读 peek，绝不推进游标。"""
    files = st["files"] or []
    if not files:
        return f"empty|{st['directory']}|{st['params']}"
    if st["pos"] >= len(files):
        return f"exhausted|{st['cycle_index']}|{st['directory']}|{st['params']}"
    if st["params"]["order"] == "random":
        return f"random|{st['cycle_index']}|{st['pos']}"
    return f"next|{st['cycle_index']}|{st['pos']}|{files[st['pos']]}"


def _take(st):
    """推进游标并返回 (path, index, parked)；parked=停在最后一张（wrap 关）。"""
    files = st["files"]
    if st["pos"] >= len(files):
        if not st["wrap"]:
            return files[len(files) - 1], len(files) - 1, True
        st["cycle_index"] += 1
        st["pos"] = 0
        st["last_path"] = ""
        if st["params"]["order"] == "shuffle":
            random.shuffle(files)
    idx = st["pos"]
    path = files[random.randrange(len(files))] if st["params"]["order"] == "random" else files[idx]
    st["last_path"] = path
    st["last_index"] = idx
    st["pos"] += 1
    return path, idx, False


class SFLoadImagesCursor:
    DESCRIPTION = (
        "目录游标式单图加载器：每次运行消费目录里的下一张图片，配合排队 N 次做批量单图流水线"
        "（每张独立 history / 独立命名）。支持文件名前缀过滤、cap/skip/nth 切片、"
        "顺序/洗牌/随机、耗尽回卷开关、reset_token 重置，可选 state_name 磁盘续跑。"
        "输出图片、遮罩（alpha 反相）、文件名、文件路径与同名 txt 侧车提示词。\n\n"
        "需要一次 Run 内批量处理整目录时改用 SF Load Images Path + 循环节点。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": ("STRING", {"default": "input", "multiline": False, "tooltip": "图片目录：input / output / input/子目录 / output/子目录 或绝对路径"}),
            },
            "optional": {
                "file_prefix": ("STRING", {"default": "", "tooltip": "只加载文件名以此前缀开头的图片（大小写不敏感）"}),
                "image_load_cap": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1, "tooltip": "本轮最多加载的图片数（0 = 不限，与 SF Load Images Path 同义）"}),
                "skip_first_images": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1, "tooltip": "跳过排序后的前 N 张"}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": 100000, "step": 1, "tooltip": "每隔 N 张选取 1 张"}),
                "order": (list(_ORDERS), {"default": "sequential", "tooltip": "sequential 顺序 / shuffle 每轮回卷重洗（一轮内不重复）/ random 每次随机（可重复）"}),
                "cycle": ("BOOLEAN", {"default": True, "tooltip": "开 = 跑完目录后回卷到第一张；关 = 停在最后一张（后续运行输出同一张）"}),
                "reset_token": ("INT", {"default": 0, "min": 0, "max": 2147483647, "step": 1, "tooltip": "改为任意新值即重置游标：重新扫描目录、从第一张开始，磁盘续跑记录一并失效"}),
                "state_name": ("STRING", {"default": "", "tooltip": "非空时把游标持久化到 user/sfnodes/image_cursor/<名字>.json，重启或重建节点后续跑；留空仅内存"}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING", "STRING", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "filename", "filename_no_ext", "folder_path", "full_path", "caption", "index", "total")
    OUTPUT_TOOLTIPS = (
        "本轮图片（单张批次）",
        "遮罩：alpha 反相；无 alpha 时为与图等大的全 0",
        "文件名（含扩展名）",
        "文件名（不含扩展名），可直接接保存节点的文件名做逐张命名",
        "图片所在目录",
        "图片完整路径",
        "同名 .txt / .caption 侧车内容（多行合并），无侧车时为空串",
        "本轮第几张（0 起；random 档为已消费张数）",
        "本轮总张数",
    )
    FUNCTION = "load_next"
    CATEGORY = _CATEGORY

    @classmethod
    def IS_CHANGED(cls, folder, file_prefix="", image_load_cap=0, skip_first_images=0,
                   select_every_nth=1, order="sequential", cycle=True, reset_token=0,
                   state_name="", unique_id=None, **kwargs):
        params = _params(file_prefix, image_load_cap, skip_first_images, select_every_nth, order)
        directory = resolve_folder(folder)
        token = _as_int(reset_token, 0)
        st = _ensure_state(_state_key(unique_id, directory, params, token), directory, params, token, str(state_name or ""))
        return _token(st)

    @classmethod
    def VALIDATE_INPUTS(cls, folder, **kwargs):
        directory = resolve_folder(folder)
        if not os.path.isdir(directory):
            return f"Directory '{directory}' cannot be found."
        return True

    @staticmethod
    def _persist(st):
        if not st["state_name"]:
            return
        path = _cursor_path(st["state_name"])
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            atomic_write_json(path, {
                "version": 1,
                "directory": st["directory"],
                "params": st["params"],
                "reset_token": st["reset_token"],
                "last_path": st["last_path"],
                "last_index": st["last_index"],
                "cycle_index": st["cycle_index"],
                "updated": int(time.time()),
            })
        except OSError as e:
            print(f"[SFLoadImagesCursor] 游标写盘失败（忽略）：{e}")

    def load_next(self, folder, file_prefix="", image_load_cap=0, skip_first_images=0,
                  select_every_nth=1, order="sequential", cycle=True, reset_token=0,
                  state_name="", unique_id=None):
        params = _params(file_prefix, image_load_cap, skip_first_images, select_every_nth, order)
        directory = resolve_folder(folder)
        token = _as_int(reset_token, 0)
        st = _ensure_state(_state_key(unique_id, directory, params, token), directory, params, token, str(state_name or ""))
        st["wrap"] = True if cycle is None else bool(cycle)
        files = st["files"]
        if not files:
            raise ValueError(f"SFLoadImagesCursor: 目录中没有可加载的图片：{directory}（file_prefix={params['prefix']!r}）")

        failures = []
        for _ in range(len(files)):
            path, idx, parked = _take(st)
            try:
                img = open_image(path)
                rgb = img.convert("RGB")
                alpha = image_alpha(img)
                image = torch.from_numpy(np.array(rgb, dtype=np.float32)).div_(255.0).unsqueeze(0)
                if alpha is None:
                    mask_arr = np.zeros((rgb.size[1], rgb.size[0]), dtype=np.float32)
                else:
                    mask_arr = 1.0 - np.array(alpha, dtype=np.float32) / 255.0
                mask = torch.from_numpy(mask_arr).unsqueeze(0)
            except Exception as e:
                failures.append(f"{os.path.basename(path)}: {e}")
                print(f"[SFLoadImagesCursor] 跳过无法加载的图片 {path}: {e}")
                continue
            self._persist(st)
            if parked:
                print(f"[SFLoadImagesCursor] 已到目录末尾（cycle 关）：停在 {os.path.basename(path)}")
            filename = os.path.basename(path)
            return (
                image, mask, filename, os.path.splitext(filename)[0],
                os.path.dirname(path), path, _caption_for(path), idx, len(files),
            )

        raise ValueError(
            f"SFLoadImagesCursor: {directory} 本轮的图片全部加载失败（{len(files)} 张）："
            + "; ".join(failures[:5])
        )
