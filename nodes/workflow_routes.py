"""SF Workflows 后端路由（/api/sfnodes/workflows/*）。

复刻 Pixaroma Workflows 的 server 路由（改前缀与 sidecar/缓存文件名）：
- /index    一次返回浏览器绘制自身所需的全部（entries/folders/collections/issues）
- /meta     sidecar 元数据（notes/covers/folderColors + folderOrder/folderExpanded），
            GET 会自愈（迁移旧内嵌封面/遗忘图片已消失的封面），POST 按键合并
- /folder   文件夹创建/改名/删除（工作流文件本身永不在此触碰——走 ComfyUI 自己的 store）
- /reveal   在 OS 文件管理器打开工作流所在文件夹
- /cover    手选封面以真实 jpg 文件保存（sidecar 只存文件名）

注册方式沿用 lora_notes._register_routes 先例：模块导入时（__init__.py
import）副作用注册，try/except 包裹，环境异常降级不注册。
"""

import asyncio
import base64
import hashlib
import json
import os
import re
import shutil
import threading
import time

import folder_paths
from aiohttp import web

from ..sf_utils.workflow_index_helpers import (
    build_index,
    collections as _wf_collections,
    detect_issues as _wf_detect_issues,
    is_cover_name as _wf_is_cover_name,
    looks_like_image as _wf_looks_like_image,
    reserved_part as _wf_reserved_part,
)

# ── 路径与基础 ────────────────────────────────────────────────────────────

_WF_COVER_DIRNAME = "sf_covers"
_WF_COVER_MAX_BYTES = 8 * 1024 * 1024
# 迁移路径刻意远宽于新上传：图片已是用户选定的封面，唯一问题是这里能否
# 解码它，而不是是否保留。超限则保持内嵌。
_WF_MIGRATE_MAX_CHARS = 64 * 1024 * 1024

_WF_META_DICTS = ("notes", "covers", "folderColors")
_WF_META_LISTS = ("folderOrder", "folderExpanded")

# 每次写入都是对一个小文件的读-改-写。没有锁的话，一次文件夹重排与一次
# 笔记自动保存同时落地，可能各自读到对方写入前的版本，第二次写入把第一个
# 的分区还原回去——正是合并本身跨请求解决不了的"两个面板互擦"案例。
_WF_META_LOCK = asyncio.Lock()


def _sanitize_id(v, default="default"):
    s = str(v or "")
    s = re.sub(r"[^a-zA-Z0-9_.-]", "", s)
    return s or default


def _is_path_under(child, parent):
    from ..sf_utils.workflow_index_helpers import _is_under
    return _is_under(child, parent)


def _wf_user_dir(request):
    """ComfyUI 正在使用的用户文件夹。单用户安装是 'default'；多用户安装
    在 comfy-user 头里发送 id，core 也是这么解析的。"""
    base = folder_paths.get_user_directory()
    uid = "default"
    try:
        header = request.headers.get("comfy-user")
        if header and _sanitize_id(header, "") == header:
            uid = header
    except Exception:
        pass
    return os.path.join(base, uid)


def _wf_root(request):
    return os.path.join(_wf_user_dir(request), "workflows")


def _wf_cache_path(request):
    # 放在 workflows 文件夹外，否则缓存会索引它自己
    return os.path.join(_wf_user_dir(request), "sf_workflows_cache.json")


def _wf_meta_path(request):
    return os.path.join(_wf_user_dir(request), "sf_workflows_meta.json")


def _wf_resolve(root, rel):
    """浏览器传来的相对路径转为 workflows 文件夹内的真实路径，或 None。
    空返回 None，调用方绝不可能误操作到根本身。"""
    rel = (rel or "").replace("\\", "/").strip("/")
    if not rel or rel == ".":
        return None
    parts = [p for p in rel.split("/") if p not in ("", ".")]
    if not parts:
        return None
    p = os.path.normpath(os.path.join(root, *parts))
    if not _is_path_under(p, root):
        return None
    return p


def _wf_registered_types():
    """ComfyUI 已加载的类名，供缺失节点检查。"""
    try:
        import nodes as _comfy_nodes
        return set(_comfy_nodes.NODE_CLASS_MAPPINGS.keys())
    except Exception:
        return set()


def _wf_list_folders(root):
    """workflows 根下的每个文件夹，含空文件夹——空文件夹没有条目，否则在
    浏览器里不可见。"""
    out = []
    for dirpath, dirnames, _files in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        if os.path.abspath(dirpath) == os.path.abspath(root):
            continue
        out.append(os.path.relpath(dirpath, root).replace(os.sep, "/"))
    out.sort(key=lambda s: s.lower())
    return out


def _wf_read_meta(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except FileNotFoundError:
        return {}                     # 首次运行，无内容可保
    except (OSError, ValueError, UnicodeDecodeError, RecursionError):
        # 文件存在但解析不了。返回 {} 并让下次保存写回，会在一人打了
        # 一个字符后抹掉每张笔记/封面/文件夹颜色。先留一份副本。
        try:
            broken = path + ".broken"
            if not os.path.exists(broken):
                shutil.copy2(path, broken)
                print(f"[sfnodes] workflows sidecar unreadable; kept a copy at {broken}")
        except OSError:
            pass
    return {}


def _wf_write_meta(path, data):
    # 临时名带线程 id：并发写同一 sidecar 时抢同一个 .tmp 会写出混杂内容
    # （lora_reader.write_custom_store 同款做法）。
    tmp = f"{path}.{threading.get_ident()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, path)
        return True
    except OSError:
        try:
            os.remove(tmp)
        except OSError:
            pass
        return False


def _wf_build_payload(root, cache_path, registered):
    """在事件循环外运行——它 stat 并解析每个工作流文件，阻塞会拖慢所有人
    的出图进度。"""
    entries = build_index(root, cache_path)
    return {
        "ok": True,
        "entries": entries,
        "folders": _wf_list_folders(root),
        "collections": _wf_collections(entries),
        "issues": _wf_detect_issues(entries, registered),
    }


def _wf_read_and_heal_meta(request):
    path = _wf_meta_path(request)
    data = _wf_read_meta(path)
    # 自愈是礼貌，绝不是失败的理由：手改过（或旧版本写的）sidecar 可能
    # 带着这些遍不期望的形状，这里异常会在每次打开时 500 整个面板
    try:
        dirty = _wf_migrate_embedded_covers(request, data)
        dirty = _wf_drop_missing_covers(request, data) or dirty
    except Exception:
        return data
    if dirty:
        _wf_write_meta(path, data)
    return data


def _wf_apply_meta_patch(request, patch):
    path = _wf_meta_path(request)
    data = _wf_read_meta(path)

    # 本补丁停止引用的每张图（键被清或改指他处）。删除只发生一次，在最后、
     # 针对最终状态——这让整件事与顺序无关。在合并中途删除意味着对
     # 半应用补丁问"还有谁在用这个？"，答案可能仅因改指它的键尚未合并
     # 而为否。
    orphan_candidates = []

    # 字典分区按键合并，两个面板互不擦除
    for section in _WF_META_DICTS:
        incoming = patch.get(section)
        if not isinstance(incoming, dict):
            continue
        current = data.get(section)
        if not isinstance(current, dict):
            current = {}

        for k, v in incoming.items():
            old = current.get(k)
            if v is None:
                current.pop(k, None)            # 显式 null 清除单条
            else:
                # 封面记录命名一个我们稍后要删除的文件，拒绝我们不可能
                # 写出的文件名，而不是存下来等 os.remove 时才暴露
                if section == "covers" and isinstance(v, dict) and "file" in v \
                        and not _wf_is_cover_name(v.get("file")):
                    continue
                # 一次 run 自己的输出绝不能替换用户手选的图。自动捕获每次
                # 执行都触发、看不到这个文件（面板开没开都跑），保护只能
                # 放在这里——每个客户端都经过的唯一写入点。在孤儿检查前
                # 跳过是刻意的：什么都没变，旧图仍被引用，不能排队删除。
                if section == "covers" and isinstance(v, dict) \
                        and v.get("kind") == "output" \
                        and isinstance(old, dict) and old.get("kind") == "file":
                    continue
                current[k] = v
            # 覆盖封面同样搁浅它的旧图——过去只考虑了清除的情况
            if section == "covers" and isinstance(old, dict) and old.get("file"):
                if not (isinstance(v, dict) and v.get("file") == old["file"]):
                    orphan_candidates.append(old["file"])
        data[section] = current

    # 列表分区整体替换。顺序逐键合并没有意义——发送它的全部意义就是
    # 序列变了
    for section in _WF_META_LISTS:
        incoming = patch.get(section)
        if isinstance(incoming, list):
            data[section] = [x for x in incoming if isinstance(x, str)]

    ok = _wf_write_meta(path, data)

    # 只在写入落地后。先删图再保存停止引用它的记录，会留下引用已消失
    # 文件的封面
    if ok:
        for name in orphan_candidates:
            if _wf_cover_referenced(data, name):
                continue                        # 还有别处引用
            target = _wf_cover_path(request, name)
            if not target:
                continue                        # 不是我们可能写出的名字
            try:
                os.remove(target)
            except OSError:
                pass                            # 已消失或在用
    return web.json_response({"ok": ok, "meta": data})


# ── 封面图片 ──────────────────────────────────────────────────────────────
# 手选封面过去以内嵌 base64 存在 sidecar 里，每次打开面板都整体重新下载
# 该文件——三个封面就已 96KB，五十个就约 1.5MB。封面改为真实 jpg 文件，
# sidecar 只存文件名。存储名由工作流路径派生（稳定），每次保存递增版本号
# 放进 URL——图片可硬缓存，替换瞬间即更新。

def _wf_covers_dir(request, create=False):
    d = os.path.join(_wf_user_dir(request), _WF_COVER_DIRNAME)
    if create:
        try:
            os.makedirs(d, exist_ok=True)
        except OSError:
            pass
    return d


def _wf_cover_path(request, name):
    """我们某个封面文件的磁盘路径，或 None（名字不是我们的）。每次封面的
    os.remove 都经过这里。"""
    if not _wf_is_cover_name(name):
        return None
    folder = _wf_covers_dir(request)
    path = os.path.join(folder, name)
    # 正则已排除分隔符，但包含检查才是读者会找的防御
    return path if _is_path_under(path, folder) else None


def _wf_cover_name(rel):
    """按工作流稳定，且无论路径含什么都安全作文件名。"""
    return hashlib.sha1(rel.encode("utf-8")).hexdigest()[:16] + ".jpg"


def _wf_cover_referenced(meta, filename, skip_key=None):
    """其它某个工作流是否仍指向此文件。改名把同一张图改指新键，仅因旧键
    消失就删文件会丢掉仍在使用的封面。"""
    for k, v in (meta.get("covers") or {}).items():
        if k == skip_key:
            continue
        if isinstance(v, dict) and v.get("file") == filename:
            return True
    return False


def _wf_drop_missing_covers(request, data):
    """遗忘图片已不在磁盘的封面。文件夹是普通文件夹，人会删东西。返回
    True 表示有内容被丢。os.path.isfile 对"已删除"和"无法查看"（断网的
    网络盘、杀软锁、权限抖动）都答 False——同等对待会在一个坏时刻清掉
    全部封面引用。文件夹本身列不出来时假设不变。"""
    covers = data.get("covers")
    if not isinstance(covers, dict):
        return False
    folder = _wf_covers_dir(request)

    if not os.path.isdir(folder):
        return False
    try:
        present = set(os.listdir(folder))
    except OSError:
        return False

    # 也遗忘工作流已消失的封面。改名先把封面带到新路径（浏览器做），
    # 所以仍指向无文件路径的才是真孤儿，其图片否则永远留在那里
    wf_root = _wf_root(request)
    wf_root_ok = os.path.isdir(wf_root)

    changed = False
    for rel, rec in list(covers.items()):
        if not isinstance(rec, dict) or rec.get("kind") != "file":
            continue
        name = rec.get("file")
        # isinstance 而非真值：`name not in present` 对列表/字典抛 TypeError，
        # 而损坏/手改的 sidecar 正是这次自愈遍存在的原因——它绝不能 500 路由
        if not isinstance(name, str):
            name = None
        if name and name not in present:
            covers.pop(rel, None)
            changed = True
            continue
        if wf_root_ok:
            wf_path = _wf_resolve(wf_root, rel)
            if wf_path and not os.path.isfile(wf_path):
                covers.pop(rel, None)
                changed = True
                if name and not _wf_cover_referenced(data, name, skip_key=rel):
                    path = _wf_cover_path(request, name)
                    if path:
                        try:
                            os.remove(path)
                        except OSError:
                            pass
    return changed


def _wf_migrate_embedded_covers(request, data):
    """把第一版遗留的 base64 封面移到文件。读取时运行，一次性、自动、无需
    用户操作。返回 True 表示有变化、sidecar 需要写回。"""
    covers = data.get("covers")
    if not isinstance(covers, dict):
        return False
    changed = False
    for rel, rec in list(covers.items()):
        if not isinstance(rec, dict):
            continue
        # str() 而非 `or ""`：`or` 返回真值操作数，url 是 dict/数字时会
        # 到达 .startswith 并抛 AttributeError——出本函数、出自愈读、出
        # meta 路由，此后每次打开面板都 500
        url = rec.get("url")
        url = url if isinstance(url, str) else ""
        if not (rec.get("kind") == "file" and url.startswith("data:")):
            continue
        # 有界，让一个荒谬的遗留物不会被每次读取整体解码。宽松——这是
        # 用户已选定的现有封面：超限保持内嵌（慢但完好）而非丢弃
        if len(url) > _WF_MIGRATE_MAX_CHARS:
            continue
        try:
            payload = url.split(",", 1)[1]
            raw = base64.b64decode(payload)
        except Exception:
            covers.pop(rel, None)          # 不可读的遗留物，丢弃
            changed = True
            continue
        if not raw:
            covers.pop(rel, None)
            changed = True
            continue
        name = _wf_cover_name(rel)
        try:
            os.makedirs(_wf_covers_dir(request, create=True), exist_ok=True)
            with open(os.path.join(_wf_covers_dir(request), name), "wb") as f:
                f.write(raw)
        except OSError:
            continue                        # 保持内嵌而非丢失
        covers[rel] = {"kind": "file", "file": name, "v": 1}
        changed = True
    return changed


def _wf_record_cover(request, rel, name):
    # 在此处解析，而非继承调用方：曾读取调用方的一个 `folder` 局部变量，
    # 唯一需要它的路径（写失败清理）抛 NameError——而 NameError 不是
    # OSError，except 接不住，路由 500
    folder = _wf_covers_dir(request)
    meta_path = _wf_meta_path(request)
    meta = _wf_read_meta(meta_path)
    covers = meta.get("covers")
    if not isinstance(covers, dict):
        covers = {}
    # 时间戳而非计数器：文件名由工作流路径派生所以每次相同；计数器在条目
    # 被丢后从 1 重来（手删图片正是如此），产生与浏览器仍持有的 url 相同
    # 的地址——新封面会显示成旧的。毫秒时间戳不会重复
    version = int(time.time() * 1000)
    # 被替换的记录可能指向与刚写入不同的文件：文件名由路径哈希而来，改名后
    # 的旧记录仍持有旧路径的哈希。meta 补丁学会了清理被取代的图；此路由是
    # 自己的写入路径，曾让每次改名-再手选都在封面目录里搁浅一个 jpg
    prev = covers.get(rel)
    old_name = prev.get("file") if isinstance(prev, dict) else None
    # 在覆盖前、meta 仍描述磁盘状态时问：磁盘上有什么指向刚（覆盖）写入的
    # 文件？下面失败分支需要——sidecar 写入未落地时，磁盘记录才是要紧的
    name_was_referenced = _wf_cover_referenced(meta, name)
    covers[rel] = {"kind": "file", "file": name, "v": version}
    meta["covers"] = covers
    if not _wf_write_meta(meta_path, meta):
        # 记录写入失败，sidecar 仍持原状。只有磁盘上没人指向该名字时删除
        # 上传的文件才算清理——通常有人指：文件名由工作流路径哈希而来，
        # 给已有封面的工作流重选封面写的是同一个名字，无条件删除会毁掉
        # 已存在的封面
        if not name_was_referenced:
            try:
                os.remove(os.path.join(folder, name))
            except OSError:
                pass
        return web.json_response({"ok": False, "message": "Could not save the cover setting."})
    # 只在写入落地后，且与别处同样守卫：形状检查（损坏记录不能瞄准
    # os.remove）、对最终状态的引用检查、_wf_cover_path 作为名字变路径的
    # 唯一途径
    if old_name and old_name != name and not _wf_cover_referenced(meta, old_name):
        old_path = _wf_cover_path(request, old_name)
        if old_path:
            try:
                os.remove(old_path)
            except OSError:
                pass
    return web.json_response({"ok": True, "file": name, "v": version})


# ── 路由注册 ──────────────────────────────────────────────────────────────

def _register_routes():
    try:
        from server import PromptServer

        ins = getattr(PromptServer, "instance", None)
        if ins is None or not hasattr(ins, "routes"):
            print("[sfnodes] PromptServer instance not available, workflows routes not registered")
            return
        routes = ins.routes

        @routes.get("/api/sfnodes/workflows/index")
        async def api_workflows_index(request):
            """浏览器绘制自身所需的一切，一次请求。"""
            root = _wf_root(request)
            if not os.path.isdir(root):
                return web.json_response(
                    {"ok": True, "entries": [], "folders": [], "collections": [],
                     "issues": {"unsaved_names": [], "duplicates": [], "missing_nodes": []}},
                    headers={"Cache-Control": "no-store"},
                )
            loop = asyncio.get_event_loop()
            try:
                payload = await loop.run_in_executor(
                    None, _wf_build_payload, root, _wf_cache_path(request), _wf_registered_types()
                )
            except Exception as e:
                return web.json_response(
                    {"ok": False, "message": str(e), "entries": [], "folders": [],
                     "collections": [], "issues": {}},
                    headers={"Cache-Control": "no-store"},
                )
            # no-store，否则浏览器会启发式缓存并显示不再匹配磁盘的列表
            return web.json_response(payload, headers={"Cache-Control": "no-store"})

        @routes.get("/api/sfnodes/workflows/meta")
        async def api_workflows_meta_get(request):
            # 这个 GET 会写（迁移旧内嵌封面、遗忘图片已消失的封面），所以
            # 与 POST 同锁。磁盘工作放事件循环外
            async with _WF_META_LOCK:
                loop = asyncio.get_event_loop()
                data = await loop.run_in_executor(None, _wf_read_and_heal_meta, request)
            for k in _WF_META_DICTS:
                data.setdefault(k, {})
            for k in _WF_META_LISTS:
                data.setdefault(k, [])
            return web.json_response({"ok": True, "meta": data},
                                     headers={"Cache-Control": "no-store"})

        @routes.post("/api/sfnodes/workflows/meta")
        async def api_workflows_meta_post(request):
            """按键合并而非整体替换。两个面板同时开着不能擦掉彼此的笔记。"""
            try:
                patch = await request.json()
            except Exception:
                patch = {}
            if not isinstance(patch, dict):
                return web.json_response({"ok": False, "message": "bad payload"})

            async with _WF_META_LOCK:
                return _wf_apply_meta_patch(request, patch)

        @routes.post("/api/sfnodes/workflows/folder")
        async def api_workflows_folder(request):
            """创建/改名/删除文件夹。core 没有 API；工作流文件在此永不触碰
            ——那些走 ComfyUI 自己的 store。"""
            try:
                data = await request.json()
            except Exception:
                data = {}
            if not isinstance(data, dict):
                data = {}
            action = str(data.get("action", ""))
            root = _wf_root(request)
            path = _wf_resolve(root, data.get("path", ""))

            if action == "create":
                if not path:
                    return web.json_response({"ok": False, "message": "Give the folder a name."})
                bad = _wf_reserved_part(root, path)
                if bad:
                    return web.json_response(
                        {"ok": False, "message": f'"{bad}" is a name Windows keeps for itself. '
                                                 f"Pick another one."})
                if os.path.exists(path):
                    return web.json_response({"ok": False, "message": "That folder already exists."})
                try:
                    os.makedirs(path)
                except OSError as e:
                    return web.json_response({"ok": False, "message": str(e)})
                return web.json_response({"ok": True})

            if action == "rename":
                new_path = _wf_resolve(root, data.get("newPath", ""))
                if not path or not new_path:
                    return web.json_response({"ok": False, "message": "Bad folder name."})
                if not os.path.isdir(path):
                    return web.json_response({"ok": False, "message": "That folder is gone."})
                bad = _wf_reserved_part(root, new_path)
                if bad:
                    return web.json_response(
                        {"ok": False, "message": f'"{bad}" is a name Windows keeps for itself. '
                                                 f"Pick another one."})
                if os.path.exists(new_path):
                    return web.json_response({"ok": False, "message": "A folder with that name already exists."})
                try:
                    os.rename(path, new_path)
                except OSError as e:
                    return web.json_response({"ok": False, "message": str(e)})
                return web.json_response({"ok": True})

            if action == "delete":
                if not path or not os.path.isdir(path):
                    return web.json_response({"ok": False, "message": "That folder is gone."})
                try:
                    # 刻意 os.rmdir 而非递归删除：拒绝仍装着工作的文件夹正是
                    # 这里的全部安全故事，本版本没有撤销
                    os.rmdir(path)
                except OSError:
                    return web.json_response(
                        {"ok": False, "message": "That folder still has things in it. Empty it first."})
                return web.json_response({"ok": True})

            return web.json_response({"ok": False, "message": "Unknown action."})

        @routes.post("/api/sfnodes/workflows/reveal")
        async def api_workflows_reveal(request):
            """在 OS 文件管理器打开工作流所在文件夹。与 Save Image reveal
            同等信任级别：服务器就是用户自己的机器。"""
            try:
                data = await request.json()
            except Exception:
                data = {}
            if not isinstance(data, dict):
                data = {}
            root = _wf_root(request)
            target = _wf_resolve(root, data.get("path", "")) or root
            folder = target if os.path.isdir(target) else os.path.dirname(target)
            if not os.path.isdir(folder) or not _is_path_under(folder, root):
                return web.json_response({"ok": False, "message": "Folder not found."})
            try:
                import subprocess
                import sys
                if sys.platform == "win32":
                    os.startfile(folder)
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", folder])
                else:
                    subprocess.Popen(["xdg-open", folder])
                return web.json_response({"ok": True})
            except Exception as e:
                return web.json_response({"ok": False, "message": str(e)})

        @routes.post("/api/sfnodes/workflows/cover")
        async def api_workflows_cover_set(request):
            """把手选封面作为真实文件保存并让 sidecar 指向它。"""
            try:
                body = await request.json()
            except Exception:
                body = {}
            if not isinstance(body, dict):
                body = {}
            rel = str(body.get("rel", "") or "")
            data_url = str(body.get("dataUrl", "") or "")
            if not rel or "," not in data_url:
                return web.json_response({"ok": False, "message": "Nothing to save."})

            payload = data_url.split(",", 1)[1]
            # 解码前检查：base64 膨胀约三分之一，先解码会让上限无法约束
            # 用于拒绝超大 payload 的内存
            if len(payload) > _WF_COVER_MAX_BYTES * 4 // 3 + 8:
                return web.json_response({"ok": False, "message": "That picture is too large."})
            try:
                raw = base64.b64decode(payload)
            except Exception:
                return web.json_response({"ok": False, "message": "That picture could not be read."})
            if not raw or len(raw) > _WF_COVER_MAX_BYTES:
                return web.json_response({"ok": False, "message": "That picture is too large."})
            if not _wf_looks_like_image(raw):
                return web.json_response(
                    {"ok": False, "message": "That file is not a picture the browser can show."})

            name = _wf_cover_name(rel)
            folder = _wf_covers_dir(request, create=True)
            path = os.path.join(folder, name)
            if not _is_path_under(path, folder):
                return web.json_response({"ok": False, "message": "Bad cover path."})
            # 先写临时文件再移动到位。文件名对给定工作流每次相同，直接覆盖
            # 会让飞行中的请求读到写了一半的 jpg
            tmp = "%s.%d.tmp" % (path, threading.get_ident())
            try:
                with open(tmp, "wb") as f:
                    f.write(raw)
                os.replace(tmp, path)
            except OSError as e:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
                return web.json_response({"ok": False, "message": str(e)})

            async with _WF_META_LOCK:
                return _wf_record_cover(request, rel, name)

        @routes.get("/api/sfnodes/workflows/cover/{name}")
        async def api_workflows_cover_get(request):
            name = request.match_info.get("name", "")
            # 名字是我们的（hex 摘要 + .jpg），其它任何东西都不是我们的，
            # 没有理由去找它。删除路径用同一验证器
            path = _wf_cover_path(request, name)
            if not path or not os.path.isfile(path):
                return web.Response(status=404, text="Not found")
            # 非 immutable：这是用户可打开的普通文件夹里的文件，手删后旧图
            # 曾因浏览器不再询问而在屏上挂一年。no-cache 仍避免重新下载
            # （FileResponse 发送验证器，通常回答是便宜的 304），但文件消失
            # 现在立即 404，卡片回退到绘制图
            return web.FileResponse(path, headers={"Cache-Control": "no-cache"})

        print("[sfnodes] workflows routes registered (/api/sfnodes/workflows/*)")
    except Exception as e:
        print(f"[sfnodes] workflows routes registration failed: {e}")


_register_routes()
