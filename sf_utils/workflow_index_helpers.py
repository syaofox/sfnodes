"""SF Workflows - 读取工作流文件夹的纯逻辑。

复刻 Pixaroma Workflows 的 _workflow_index_helpers.py（无 ComfyUI 依赖，可独立
测试）。浏览器展示文件名之外的一切都来自这里：工作流内容、供封面用的小图
映射、哪些文件像垃圾或重复、自动填充的集合。

读取全部文件只做一次，并按每个文件的修改时间与大小缓存，二次打开只重解析
真正变化的部分。浏览器从不自行 fetch 文件。
"""

import hashlib
import json
import os
import re
import threading
from collections import deque

# 大于此大小的文件几乎不可能是手工工作流，解析会阻塞请求。24MB 远高于
# 现实中最大的工作流（作者自己文件夹里最大 75KB）。
_MAX_BYTES = 24 * 1024 * 1024

# 封面映射携带的节点矩形数。封面约 120x64 CSS 像素，几十个框之后不再可读，
# payload 只会变大。
_MAP_CAP = 60

# 每个工作流保留的提示词文本总量（供搜索）。足够找到记得的短语，也足够轻。
_TEXT_CAP = 2000

# 条目形状变化时递增，旧版本写的缓存被丢弃（v2：封面映射携带颜色字符串
# 而非调色板下标）。
_CACHE_VERSION = 2

_MODEL_EXT = (".safetensors", ".ckpt", ".gguf", ".pt", ".pth", ".sft", ".bin")

# 这些类的 widget 字符串视为值得搜索的提示词文本。
_TEXTY = ("cliptextencode", "text", "prompt", "string")

# ── 集合规则 ───────────────────────────────────────────────────────────────
# 一张表，新增分组是数据改动而非代码改动。每种是 (id, label, 谓词)。
# 顺序重要：第一个命中的"输出种类"获胜，视频工作流不会因为含采样器再被
# 归档为文生图。

def _has(entry, *needles):
    """工作流中任一节点类包含某个 needle 时返回 True。"""
    low = entry.get("_lower_types") or []
    return any(any(n in t for t in low) for n in needles)


def _kind_video(e):
    return _has(e, "savemp4", "vhs_", "savewebm", "videocombine", "imagetovideo",
                "svd_", "animatediff", "wanimage", "saveanimated")


def _kind_upscale(e):
    return _has(e, "upscalemodel", "imagescale", "upscale")


def _kind_inpaint(e):
    return _has(e, "inpaint", "setlatentnoisemask", "outpaint")


def _kind_img2img(e):
    return _has(e, "vaeencode", "loadimage") and _has(e, "sampler")


def _kind_txt2img(e):
    return _has(e, "sampler") and _has(e, "cliptextencode", "textencode")


# 按此顺序检查；工作流落在第一个命中的种类，所以对"它产出什么"的最精确
# 描述获胜。
_KINDS = [
    ("video", "Video", _kind_video),
    ("inpaint", "Inpaint / Outpaint", _kind_inpaint),
    ("upscale", "Upscale", _kind_upscale),
    ("img2img", "Image to Image", _kind_img2img),
    ("txt2img", "Text to Image", _kind_txt2img),
]

# 模型家族，按工作流里找到的模型文件名匹配。
_FAMILIES = [
    ("flux", "Flux", ("flux",)),
    ("qwen", "Qwen", ("qwen",)),
    ("wan", "Wan", ("wan",)),
    ("sdxl", "SDXL", ("sdxl", "sd_xl")),
    ("sd15", "SD 1.5", ("sd15", "v1-5", "sd_v1")),
    ("sd3", "SD 3", ("sd3", "sd_3")),
    ("hunyuan", "Hunyuan", ("hunyuan",)),
    ("krea", "Krea", ("krea",)),
    ("chroma", "Chroma", ("chroma",)),
]


# ── 小工具 ────────────────────────────────────────────────────────────────

def _is_under(child, parent):
    """child 是否在 parent 内。同时比较折叠路径与解析路径：经 junction
    （跨盘拆分常见）到达的 workflows 文件夹也被接受，'..' 无法逃逸。"""
    try:
        c_abs, p_abs = os.path.abspath(child), os.path.abspath(parent)
        if os.path.commonpath([c_abs, p_abs]) == p_abs:
            return True
    except ValueError:
        pass
    try:
        c_real, p_real = os.path.realpath(child), os.path.realpath(parent)
        return os.path.commonpath([c_real, p_real]) == p_real
    except ValueError:
        return False


def _rel(path, root):
    return os.path.relpath(path, root).replace(os.sep, "/")


def _num(v, default=0.0):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    # 坏导出器写出的工作流可能带 inf/nan，序列化为非法 JSON 会毁掉整个响应
    if f != f or f in (float("inf"), float("-inf")):
        return default
    return f


def _xy(v):
    """节点 pos/size 在现代文件里是 2 元素列表，某些旧文件是 {"0":x,"1":y}
    字典。两者都接受，而不是把节点从映射里丢掉。"""
    if isinstance(v, dict):
        return _num(v.get("0") if "0" in v else v.get(0)), _num(v.get("1") if "1" in v else v.get(1))
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        return _num(v[0]), _num(v[1])
    return 0.0, 0.0


_HEX_RE = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


def _node_color(color):
    """节点自己的颜色（纯 hex 字符串），无则 ""。浏览器会把颜色抬升到可读
    亮度（ComfyUI 节点颜色近乎黑，是深色画布上的标题色）。非纯 hex 值
    （rgba/css 名/手改文件垃圾）一律变 ""。"""
    if not isinstance(color, str):
        return ""
    c = color.strip()
    return c.lower() if _HEX_RE.match(c) else ""


def _clamp01(v):
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def _walk_strings(widgets):
    """widget 值可能嵌套（列表/字典）。逐个产出找到的字符串，模型文件名
    不会因为形状而漏掉。两个独立上限：visits 约束工作量，queue 长度约束
    内存；队列 FIFO，截断保留最早的值（widget 槽 0 正是模型文件名所在）。"""
    MAX_VISIT = 2000
    MAX_QUEUE = 2000
    queue = deque([widgets])
    visited = 0
    while queue and visited < MAX_VISIT:
        cur = queue.popleft()
        visited += 1
        if isinstance(cur, str):
            yield cur
            continue
        if isinstance(cur, dict):
            items = cur.values()
        elif isinstance(cur, (list, tuple)):
            items = cur
        else:
            continue
        for v in items:
            if len(queue) >= MAX_QUEUE:
                break
            queue.append(v)


# ── 单个工作流 ────────────────────────────────────────────────────────────

def summarize_workflow(path, root):
    """浏览器需要的关于一个工作流文件的一切。绝不抛错：缺失/过大/在根外/
    非法 JSON 的文件返回带 "error" 且其余为空的条目，一个坏文件弄不塌整个
    列表。"""
    name = os.path.splitext(os.path.basename(path))[0]
    blank = {
        "name": name, "rel": _rel(path, root) if root else os.path.basename(path),
        "folder": "", "size": 0, "modified": 0.0, "node_count": 0,
        "class_types": [], "models": [], "loras": [], "text": "",
        "map": [], "fingerprint": "", "error": None,
    }

    if root and not _is_under(path, root):
        blank["error"] = "outside the workflows folder"
        return blank

    try:
        st = os.stat(path)
    except OSError as e:
        blank["error"] = "cannot read: %s" % e.__class__.__name__
        return blank

    blank["size"] = st.st_size
    blank["modified"] = st.st_mtime
    rel = _rel(path, root)
    blank["rel"] = rel
    blank["folder"] = os.path.dirname(rel)

    if st.st_size > _MAX_BYTES:
        blank["error"] = "file is too large to read"
        return blank

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except RecursionError:
        blank["error"] = "this file is nested too deeply to read"
        return blank
    except (OSError, ValueError, UnicodeDecodeError) as e:
        blank["error"] = "not a readable workflow: %s" % e.__class__.__name__
        return blank

    if not isinstance(data, dict):
        blank["error"] = "not a workflow file"
        return blank

    nodes = data.get("nodes")
    if not isinstance(nodes, list):
        blank["error"] = "no nodes in this file"
        return blank

    types, lower, models, loras, texts, boxes = [], [], [], [], [], []
    for n in nodes:
        if not isinstance(n, dict):
            continue
        t = n.get("type")
        # `(t or "").lower()` 看着安全其实不是：Python 的 `or` 返回真值操作数，
        # "type": true 或 7 会到达 .lower() 并抛错，一个坏文件带崩整个列表
        t = t if isinstance(t, str) else ""
        if t:
            types.append(t)
            lower.append(t.lower())
        tl = t.lower()

        widgets = n.get("widgets_values")
        if widgets is not None:
            is_lora = "lora" in tl
            texty = any(k in tl for k in _TEXTY)
            for s in _walk_strings(widgets):
                low = s.lower()
                if low.endswith(_MODEL_EXT):
                    (loras if is_lora else models).append(s)
                elif texty and len(s) > 8:
                    texts.append(s)

        x, y = _xy(n.get("pos"))
        w, h = _xy(n.get("size"))
        boxes.append((x, y, w if w > 0 else 200.0, h if h > 0 else 80.0,
                      _node_color(n.get("color"))))

    # ── 封面映射：节点矩形归一化到 0..1 盒 ──
    cover = []
    if boxes:
        keep = boxes[:_MAP_CAP]
        min_x = min(b[0] for b in keep)
        min_y = min(b[1] for b in keep)
        max_x = max(b[0] + b[2] for b in keep)
        max_y = max(b[1] + b[3] for b in keep)
        span_x = max_x - min_x
        span_y = max_y - min_y
        # 单节点或全部堆在同一点：零跨度
        if span_x <= 0:
            span_x = 1.0
        if span_y <= 0:
            span_y = 1.0
        for (x, y, w, h, col) in keep:
            cover.append([
                round(_clamp01((x - min_x) / span_x), 4),
                round(_clamp01((y - min_y) / span_y), 4),
                round(_clamp01(w / span_x), 4),
                round(_clamp01(h / span_y), 4),
                col,
            ])

    uniq_types = sorted(set(types))
    uniq_models = sorted(set(models))
    uniq_loras = sorted(set(loras))

    # 相同图结构 + 相同模型 = 同一工作流穿了两个名字。刻意忽略提示词文本
    # 与节点位置——这正是发现人们堆积的副本所需的。序列化而非分隔符拼接：
    # 含分隔符的文件名可能与不同拆分产生相同字符串。
    fp_src = json.dumps([sorted(types), uniq_models, uniq_loras], separators=(",", ":"))
    fingerprint = hashlib.md5(fp_src.encode("utf-8")).hexdigest() if types else ""

    text = " ".join(texts)
    if len(text) > _TEXT_CAP:
        text = text[:_TEXT_CAP]

    blank.update({
        "node_count": len(nodes),
        "class_types": uniq_types,
        "models": uniq_models,
        "loras": uniq_loras,
        "text": text,
        "map": cover,
        "fingerprint": fingerprint,
    })
    return blank


# ── 整个文件夹 ────────────────────────────────────────────────────────────

def _cache_key(st):
    return [st.st_mtime_ns, st.st_size]


def _load_cache(cache_path):
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            c = json.load(f)
        if isinstance(c, dict) and c.get("version") == _CACHE_VERSION and isinstance(c.get("entries"), dict):
            return c["entries"]
    except (OSError, ValueError, UnicodeDecodeError):
        pass
    return {}


def _save_cache(cache_path, entries):
    """先写临时文件再移动到位：崩溃或磁盘满只留下旧缓存，而不是一个必须
    在每次打开时检测并丢弃的坏缓存。临时名带线程 id（跑在线程执行器里，
    共享的 .tmp 会被两次重叠构建写烂）。"""
    tmp = "%s.%d.tmp" % (cache_path, threading.get_ident())
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"version": _CACHE_VERSION, "entries": entries}, f)
        os.replace(tmp, cache_path)
    except OSError:
        try:
            os.remove(tmp)
        except OSError:
            pass


def build_index(root, cache_path):
    """汇总 root 下每个 .json，只重读修改时间或大小变化过的文件。返回条目
    列表。"""
    old = _load_cache(cache_path)
    new_entries = {}
    out = []

    for dirpath, dirnames, filenames in os.walk(root):
        # 跳过任何隐藏项与 ComfyUI 自己的簿记
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fn in filenames:
            if not fn.lower().endswith(".json") or fn.startswith("."):
                continue
            full = os.path.join(dirpath, fn)
            rel = _rel(full, root)
            try:
                key = _cache_key(os.stat(full))
            except OSError:
                continue
            hit = old.get(rel)
            if hit and hit.get("key") == key and isinstance(hit.get("data"), dict):
                data = hit["data"]
            else:
                # summarize_workflow 被写成不抛错，但整个列表是功能主体：
                # 单个不可读文件必须不能清空它——包括这里以后引入的 bug
                try:
                    data = summarize_workflow(full, root)
                except Exception as e:            # noqa: BLE001 - deliberate
                    data = {
                        "name": os.path.splitext(fn)[0], "rel": rel,
                        "folder": os.path.dirname(rel), "size": 0, "modified": 0.0,
                        "node_count": 0, "class_types": [], "models": [], "loras": [],
                        "text": "", "map": [], "fingerprint": "",
                        "error": "could not be read: %s" % e.__class__.__name__,
                    }
            new_entries[rel] = {"key": key, "data": data}
            out.append(data)

    _save_cache(cache_path, new_entries)
    out.sort(key=lambda e: (e.get("folder", ""), e.get("name", "").lower()))
    return out


# ── 文件夹有什么问题 ───────────────────────────────────────────────────────

# 前端注册、因此不会出现在 Python 的 NODE_CLASS_MAPPINGS 里的节点类型。
# 没有它，每个含便签的工作流都像坏的——首次运行给用户 143 个工作流里的
# 108 个打了标记。此列表只覆盖 ComfyUI 自带的；自定义包也能注册纯前端节点
# （rgthree 就是），所以浏览器会按 LiteGraph.registered_node_types 重算并
# 覆盖这里的结果。此值只是兜底，不是答案。
_FRONTEND_ONLY = frozenset({
    "Note", "MarkdownNote", "PrimitiveNode", "Reroute", "GroupNode",
})


# ── 名称与文件检查 ─────────────────────────────────────────────────────────
# 纯函数，放在这里（而非路由文件）供测试。两者都在守护一次写入。

# CON、NUL、COM1 等在 Windows 上指设备而非文件，任何扩展名都如此——
# "NUL" 与 "NUL.json" 都失败，失败以写入深处的无帮助 OSError 到达。
WIN_RESERVED_NAMES = frozenset({
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
})

# 浏览器会实际绘制的图片格式的前导字节
_IMAGE_MAGIC = (
    b"\xff\xd8\xff",                    # jpeg
    b"\x89PNG\r\n\x1a\n",               # png
    b"GIF87a", b"GIF89a",               # gif
    b"BM",                              # bmp
)


def looks_like_image(raw):
    """这些字节是否像浏览器能显示的图片。封面直接以图片回给浏览器，所以
    它应该是图片。检查几乎零成本，把打错的上传变成一句可行动的话，而不是
    一张永远渲染不出的卡片。"""
    if not isinstance(raw, (bytes, bytearray)):
        return False
    raw = bytes(raw)
    if raw.startswith(_IMAGE_MAGIC):
        return True
    if len(raw) < 12:
        return False
    # webp 是 RIFF....WEBP——字节长度在两个标记之间，仅前缀不够
    # （wav 文件也以 RIFF 开头）
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return True
    # avif/heic 是 ISO-BMFF：长度 + "ftyp" + 品牌
    return raw[4:8] == b"ftyp" and raw[8:12] in (
        b"avif", b"avis", b"heic", b"heix", b"hevc", b"mif1", b"msf1",
    )


# 封面文件名永远是 16 个 hex 字符 + .jpg（由工作流路径生成）。其它任何
# 东西都不是我们写的。
_COVER_NAME_RE = re.compile(r"[0-9a-f]{16}\.jpg")


def is_cover_name(name):
    """这是我们可能写出的文件名吗？对安全是承重的，不只是整洁：sidecar 由
    普通 HTTP body 写入，封面记录的 "file" 是客户端发来的任何东西——clear
    路径把它喂给 os.remove。没有它，把封面的 file 设为 "../../something"
    再清除该键，就能删除 ComfyUI 进程可达的任意文件。os.path.join 不是
    防御：第二部分是绝对路径时它会丢弃基目录。"""
    return isinstance(name, str) and bool(_COVER_NAME_RE.fullmatch(name))


def reserved_part(root, path):
    """`path` 在 `root` 之下的第一个 Windows 保留段，或 None。所有平台都
    检查：Linux 上建的文件夹仍可能在日后拷到的工作流所在的 Windows 机器上
    打开。刻意纯字符串处理、不用 os.path：显而易见的实现以 os.path.relpath
    开头，它会在最可能的输入上失败开放（ntpath 把裸 "NUL" 解析为设备挂载
    点，判定两路径在不同盘并抛 ValueError——于是守卫对它的目标名字返回
    None）。这里不允许调用任何路径函数，因为这些名字正是让路径函数行为
    怪异的原因。"""
    r = str(root or "").replace("\\", "/").rstrip("/")
    p = str(path or "").replace("\\", "/")
    # 剥掉根，让含保留词的根（Linux 上可能 /home/con/workflows）不拒绝其下
    # 的每个名字。两者无关时扫描整个路径：拒绝过多是一条消息，放过设备名
    # 是一次失败的写入。
    if r and p.lower().startswith(r.lower() + "/"):
        p = p[len(r) + 1:]
    elif r and p.lower() == r.lower():
        return None
    for part in p.split("/"):
        stem = part.split(".", 1)[0].strip().upper()
        if stem in WIN_RESERVED_NAMES:
            return part
    return None


def detect_issues(index, registered_types):
    """值得告诉用户的工作流文件夹三件事。"""
    unsaved, missing = [], []
    by_fp = {}

    for e in index:
        if e.get("error"):
            continue
        if e.get("name", "").lower().startswith("unsaved workflow"):
            unsaved.append({"rel": e["rel"], "name": e["name"]})

        gone = sorted(t for t in e.get("class_types", [])
                      if t not in registered_types and t not in _FRONTEND_ONLY)
        if gone:
            missing.append({"rel": e["rel"], "name": e["name"], "missing": gone})

        fp = e.get("fingerprint")
        if fp:
            by_fp.setdefault(fp, []).append(e)

    duplicates = [g for g in by_fp.values() if len(g) > 1]
    duplicates.sort(key=lambda g: -len(g))
    return {"unsaved_names": unsaved, "duplicates": duplicates, "missing_nodes": missing}


# ── 自动填充的集合 ─────────────────────────────────────────────────────────

def collections(index):
    """按工作流产出什么、用什么模型分组。真实文件夹不受影响；这些并排
    展示。"""
    kinds = {}
    families = {}
    lora_items = []

    for e in index:
        if e.get("error"):
            continue
        e["_lower_types"] = [t.lower() for t in e.get("class_types", [])]
        try:
            for kid, label, pred in _KINDS:
                if pred(e):
                    kinds.setdefault(kid, {"label": label, "items": []})["items"].append(e["rel"])
                    break

            if e.get("loras"):
                lora_items.append(e["rel"])

            hit = set()
            for m in e.get("models", []) + e.get("loras", []):
                low = m.lower()
                for fid, label, needles in _FAMILIES:
                    if fid not in hit and any(n in low for n in needles):
                        hit.add(fid)
                        families.setdefault(fid, {"label": label, "items": []})["items"].append(e["rel"])
        finally:
            del e["_lower_types"]

    out = []
    for kid, label, _ in _KINDS:
        if kid in kinds:
            out.append({"id": kid, "group": "kind", "label": label,
                        "items": kinds[kid]["items"], "count": len(kinds[kid]["items"])})
    if lora_items:
        out.append({"id": "lora", "group": "kind", "label": "Uses a LoRA",
                    "items": lora_items, "count": len(lora_items)})
    for fid, label, _ in _FAMILIES:
        if fid in families:
            out.append({"id": fid, "group": "model", "label": label,
                        "items": families[fid]["items"], "count": len(families[fid]["items"])})
    return out
