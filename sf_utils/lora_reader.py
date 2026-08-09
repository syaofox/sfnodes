"""LoRA 元数据 / 触发词 / 状态解析纯逻辑（SFLoraStack 节点共用）。

仅标准库——无 comfy、无 folder_paths、无 torch。这里每个函数都是"读文件或
读 dict 返回数据"，可在 ComfyUI 之外单测（tests/test_lora_reader.py 直跑）。

设计离线优先：
  - read_safetensors_metadata / derive_trigger_words / base_model_family /
    build_lora_info 只读文件的小 JSON 头部（永不触碰张量块）——即时、无网络。
  - read_sidecar_info / find_preview_path 读 Civitai 助手可能留在 LoRA 旁的
    文件——仍无网络。
  - file_sha256 / parse_civitai_modelversion / save_sidecar_cache 支撑可选的
    Civitai 在线查询（由路由执行，本模块绝不打开 socket）。
  - sanitize_civitai_key / mask_civitai_key / civitai_hosts /
    read_civitai_account / write_civitai_account 支撑可选的 Civitai API key。
    它们接收显式路径，保持无 folder_paths 依赖、可单测；路由决定存在哪里。
  - parse_state / collect_triggers 是 SFLoraStack 隐藏输入的容错契约
    （双端数字语义在 JS core 里 1:1 镜像，改一处必须同步另一处）。
"""
import hashlib
import json
import os
import re
import struct
import threading

# 真实 LoRA 头部几十 KB；上限远高于此，坏的长度字段永远无法让我们分配 GB 级内存。
_MAX_HEADER_BYTES = 200 * 1024 * 1024
# 频率推导出的候选触发词数量上限。
_MAX_TRIGGERS = 20
# 描述是说明文字，不是小说（Civitai 模型描述清洗后截断 + 自定义描述限长）。
_MAX_DESCRIPTION_LEN = 2000

_PREVIEW_EXTS = (
    ".preview.png", ".preview.jpeg", ".preview.jpg", ".preview.webp",
    ".png", ".jpg", ".jpeg", ".webp",
)


def read_safetensors_metadata(path):
    """返回文件 __metadata__ dict（str->str），任何问题返回 {}。

    只读头部（8 字节小端长度 + 那么多字节的 JSON），绝不触碰张量块。
    永不抛错：坏/缺失/超大文件 -> {}。
    """
    try:
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                return {}
            n = struct.unpack("<Q", raw)[0]
            if n <= 0 or n > _MAX_HEADER_BYTES:
                return {}
            head = f.read(n)
            if len(head) != n:
                return {}
        obj = json.loads(head)
    except Exception:
        return {}
    if not isinstance(obj, dict):
        return {}
    meta = obj.get("__metadata__")
    return meta if isinstance(meta, dict) else {}


def _clean_id(v):
    """Civitai model/version id -> 干净 int，或 None。拒绝手改侧车里的
    dict/list/垃圾，避免前端拼出无效 civitai.com URL。"""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, str) and v.isdigit():
        return int(v)
    return None


def _as_json(val):
    """safetensors 元数据值总是字符串；结构化的是需要二次解析的 JSON 字符串。
    返回解析后的对象，或 None。"""
    if isinstance(val, (dict, list)):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return None
    return None


def derive_trigger_words(meta, limit=_MAX_TRIGGERS):
    """从训练元数据尽力提取触发词。

    顺序：显式触发短语优先（modelspec.trigger_phrase / ss_trigger_words），
    然后 ss_tag_frequency 中出现频次最高的训练标签（跨所有数据集目录计数
    求和），大小写不敏感去重，上限 `limit`。无可用内容返回 []。永不抛错。
    """
    if not isinstance(meta, dict):
        return []
    out = []
    seen = set()

    def add(word):
        w = (word or "").strip()
        if not w:
            return
        key = w.lower()
        if key in seen:
            return
        seen.add(key)
        out.append(w)

    phrase = meta.get("modelspec.trigger_phrase") or meta.get("ss_trigger_words") or ""
    if isinstance(phrase, str):
        for part in phrase.split(","):
            add(part)

    freq = _as_json(meta.get("ss_tag_frequency"))
    counts = {}
    if isinstance(freq, dict):
        for dataset in freq.values():
            if not isinstance(dataset, dict):
                continue
            for tag, c in dataset.items():
                try:
                    counts[tag] = counts.get(tag, 0) + int(c)
                except (TypeError, ValueError):
                    continue
    # sorted() 稳定，同频保持先出现（插入）顺序。
    for tag, _c in sorted(counts.items(), key=lambda kv: -kv[1]):
        add(tag)
        if len(out) >= limit:
            break
    return out[:limit]


def base_model_family(meta):
    """粗粒度基础模型族（用于不匹配警告）：'SDXL' | 'SD1.5' | 'SD2' | 'SD3' |
    'Flux' | ''（未知）。永不抛错。"""
    if not isinstance(meta, dict):
        return ""
    hay = " ".join(
        str(meta.get(k, "")) for k in (
            "ss_base_model_version", "ss_sd_model_name", "modelspec.architecture",
            "modelspec.implementation", "ss_network_module",
        )
    ).lower()
    if not hay.strip():
        return ""
    if "flux" in hay:
        return "Flux"
    if "sd3" in hay or "sd_3" in hay or "stable-diffusion-3" in hay:
        return "SD3"
    if "sdxl" in hay or "xl_base" in hay or "xl-base" in hay or "illustrious" in hay or "pony" in hay:
        return "SDXL"
    if "sd_v2" in hay or "sd2" in hay or "v2-1" in hay or "768-v" in hay:
        return "SD2"
    if ("sd_v1" in hay or "sd1" in hay or "v1-5" in hay or "v1.5" in hay
            or "sd-v1" in hay or "1-5-pruned" in hay):
        return "SD1.5"
    return ""


def read_sidecar_info(lora_path):
    """读 LoRA 旁的 Civitai 助手侧车（<base>.civitai.info，再试 <base>.json）。
    返回 {name?, base_model?, triggers?, description?, model_id?, version_id?}
    或 {}。无网络。永不抛错。"""
    base = os.path.splitext(lora_path)[0]
    for ext in (".civitai.info", ".json"):
        sp = base + ext
        if not os.path.isfile(sp):
            continue
        try:
            with open(sp, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        info = {}
        tw = obj.get("trainedWords")
        if isinstance(tw, list):
            info["triggers"] = [str(w).strip() for w in tw if str(w).strip()]
        elif isinstance(obj.get("activation text"), str):
            info["triggers"] = [w.strip() for w in obj["activation text"].split(",") if w.strip()]
        model = obj.get("model")
        if isinstance(model, dict) and model.get("name"):
            info["name"] = str(model["name"])
        # description 在 version 顶层（API 实测）；兼容旧侧车的 model.description。
        desc = obj.get("description")
        if not desc and isinstance(model, dict):
            desc = model.get("description")
        if desc:
            info["description"] = _clean_description(desc)
        if obj.get("baseModel"):
            info["base_model"] = str(obj["baseModel"])
        # modelId / version id 让前端可链接到 Civitai 模型页。
        mid = _clean_id(obj.get("modelId"))
        if mid is not None:
            info["model_id"] = mid
        vid = _clean_id(obj.get("id"))
        if vid is not None:
            info["version_id"] = vid
        if info:
            return info
    return {}


def find_preview_path(lora_path):
    """返回 LoRA 旁的预览图路径（.preview.png 等），或 None。"""
    base = os.path.splitext(lora_path)[0]
    for ext in _PREVIEW_EXTS:
        p = base + ext
        if os.path.isfile(p):
            return p
    return None


def _title_from_meta(meta, lora_path):
    for k in ("modelspec.title", "ss_output_name"):
        v = meta.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return os.path.splitext(os.path.basename(lora_path))[0]


def build_lora_info(lora_path):
    """一个 LoRA 的统一离线 info：title/base_model/rank/alpha/num_images/date/
    triggers/description/source/has_preview（+ 可选 model_id/version_id）。
    侧车数据（来自先前 Civitai 抓取）优先于文件推导数据。永不抛错。

    触发词分三组返回：`triggers` 是合并默认（侧车胜出），`file_triggers` 恒为
    文件自己的词，`sidecar_triggers` 存已保存的 Civitai 词（信息面板据此做
    File / Civitai 源切换）。description 同理：侧车（Civitai）胜出，文件
    modelspec.description 兜底；用户自定义覆盖在路由层追加 custom_description。"""
    meta = read_safetensors_metadata(lora_path)
    file_triggers = derive_trigger_words(meta)
    info = {
        "title": _title_from_meta(meta, lora_path),
        "base_model": base_model_family(meta),
        "rank": meta.get("ss_network_dim", "") or "",
        "alpha": meta.get("ss_network_alpha", "") or "",
        "num_images": meta.get("ss_num_train_images", "") or "",
        "date": meta.get("modelspec.date", "") or "",
        "description": _clean_description(meta.get("modelspec.description")),
        "triggers": file_triggers,
        "file_triggers": file_triggers,
        "sidecar_triggers": [],
        "source": "file",
        "has_preview": find_preview_path(lora_path) is not None,
    }
    side = read_sidecar_info(lora_path)
    if side.get("triggers"):
        info["sidecar_triggers"] = side["triggers"]
        info["triggers"] = side["triggers"]
        info["source"] = "sidecar"
    if side.get("name"):
        info["title"] = side["name"]
    if side.get("base_model") and not info["base_model"]:
        info["base_model"] = side["base_model"]
    if side.get("description"):
        info["description"] = side["description"]  # Civitai 说明胜出（更全）
    if side.get("model_id") is not None:
        info["model_id"] = side["model_id"]
    if side.get("version_id") is not None:
        info["version_id"] = side["version_id"]
    return info


# ── SFLoraStack 隐藏输入状态契约（与 web/sf_lora_stack_core.js 双端镜像）──

_STATE_MAX_STRENGTH = 100.0


def _clamp_strength(v):
    """状态 JSON 里（可能被手改的）强度值 -> [-100, 100] 内的有限 float。
    垃圾 / nan / inf -> 0.0。"""
    try:
        f = float(v)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if f != f or f in (float("inf"), float("-inf")):
        return 0.0
    return max(-_STATE_MAX_STRENGTH, min(_STATE_MAX_STRENGTH, f))


def parse_state(state_str):
    """把隐藏 LoraLoaderState JSON 归一化为
    {'loras': [...], 'sep': str, 'cacheMode': 'last'|'all'|'none'}。

    刻意宽容（手写 API 工作流也必须能跑）：坏/空输入
    -> {'loras': [], 'sep': ', ', 'cacheMode': 'last'}；无名或非 dict 条目丢弃；
    每个保留条目为 {name, on, sm, sc, triggers}。sc 缺省取 sm（单强度驱动双端）。
    cacheMode 未知值钳到 'last'（ComfyUI 对齐，只留最近使用的文件）。
    永不抛错。
    """
    try:
        obj = json.loads(state_str) if isinstance(state_str, str) else (state_str or {})
    except Exception:
        obj = {}
    if not isinstance(obj, dict):
        obj = {}
    sep = obj.get("sep")
    if not isinstance(sep, str):
        sep = ", "
    cache_mode = obj.get("cacheMode")
    if cache_mode not in ("last", "all", "none"):
        cache_mode = "last"
    loras = []
    raw = obj.get("loras")
    if isinstance(raw, list):
        for e in raw:
            if not isinstance(e, dict):
                continue
            name = e.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            base_str = e.get("sm", e.get("strength", 1.0))
            trg = e.get("triggers")
            loras.append({
                "name": name,
                "on": bool(e.get("on", True)),
                "sm": _clamp_strength(base_str),
                "sc": _clamp_strength(e.get("sc", base_str)),
                "triggers": [str(w).strip() for w in trg if str(w).strip()]
                            if isinstance(trg, list) else [],
            })
    return {"loras": loras, "sep": sep, "cacheMode": cache_mode}


def collect_triggers(state):
    """仅从 ENABLED loras 连接并去重（大小写不敏感）的触发词，
    用 state['sep'] 作分隔符。顺序按首次出现。"""
    out, seen = [], set()
    for e in state.get("loras", []):
        if not e.get("on"):
            continue
        for w in e.get("triggers", []):
            k = w.lower()
            if w and k not in seen:
                seen.add(k)
                out.append(w)
    sep = state.get("sep")
    if not isinstance(sep, str):
        sep = ", "
    return sep.join(out)


# ── Civitai 在线查询支撑（本模块不开 socket，路由执行网络请求）─────────────

def file_sha256(path):
    """文件的完整 SHA256 hex（流式）。用于按精确文件匹配在 Civitai 上查找。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


_ORIGINAL_SEG_RE = re.compile(r"/original=true(?:,[^/]*)?/")


def _is_adult_image(nsfw, level):
    """Civitai 画廊图是否被标记为成人。`nsfwLevel` 是位掩码（1 PG, 2 PG13,
    4 R, 8 X, 16 XXX）；旧字段 `nsfw` 是 bool 或单词。仅用于阻止显式图成为
    节点缩略图，宁可过度拒绝。永不抛错。"""
    if nsfw in (True, "X", "XXX", "Mature"):
        return True
    try:
        if level is not None and int(level) >= 4:
            return True
    except (TypeError, ValueError):
        pass
    return False


def _thumb_url(url):
    """Civitai 图片 URL 带 transform 段；API 给的是 `/original=true/`，即全分辨率
    原图。缩略图只需要 256px——换上 width transform（可带逗号参数，按整段匹配）。"""
    if not isinstance(url, str):
        return url
    return _ORIGINAL_SEG_RE.sub("/width=256/", url, count=1)


_HTML_ENT_RE = re.compile(r"&(#x?[0-9a-fA-F]+|[a-zA-Z]+);")


def _clean_description(raw, limit=_MAX_DESCRIPTION_LEN):
    """Civitai 模型描述是 HTML——<br> 等换行标签转行、其余标签剥掉、解码
    实体、折叠空白、截断。坏输入返回 "". 永不抛错。"""
    if not isinstance(raw, str) or not raw.strip():
        return ""
    s = re.sub(r"<(br|/p|/div|/li)\s*/?>", "\n", raw, flags=re.IGNORECASE)
    s = re.sub(r"<[^>]*>", "", s)

    def _unesc(m):
        e = m.group(1)
        if e.startswith("#x"):
            try:
                return chr(int(e[2:], 16))
            except ValueError:
                return ""
        if e.startswith("#"):
            try:
                return chr(int(e[1:]))
            except ValueError:
                return ""
        return {"amp": "&", "lt": "<", "gt": ">", "quot": '"',
                "apos": "'", "nbsp": " "}.get(e, "")

    s = _HTML_ENT_RE.sub(_unesc, s)
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in s.split("\n")]
    return "\n".join(lines).strip()[:limit]


def parse_civitai_modelversion(obj, allow_adult=False):
    """从 Civitai model-version 响应提取所需字段：
    {name?, type?, base_model?, triggers?, description?, thumbnail?,
    model_id?, version_id?}。
    缩略图优先取第一张非显式图，回退第一张非成人图。永不抛错。

    `allow_adult` 允许用显式画廊图做缩略图，仅当用户在设置里主动开启：
    画廊全显式的模型否则完全没图，看起来像查询失败。"""
    if not isinstance(obj, dict):
        return {}
    out = {}
    tw = obj.get("trainedWords")
    if isinstance(tw, list):
        out["triggers"] = [str(w).strip() for w in tw if str(w).strip()]
    if obj.get("baseModel"):
        out["base_model"] = str(obj["baseModel"])
    # description 在 version 顶层（实测 model-versions API：model 对象只有
    # name/nsfw/poi/type，说明文字在顶层 description）。兼容旧侧车里偶见的
    # model.description 形状。
    model = obj.get("model")
    desc = obj.get("description")
    if not desc and isinstance(model, dict):
        desc = model.get("description")
    if desc:
        out["description"] = _clean_description(desc)
    if isinstance(model, dict):
        if model.get("name"):
            out["name"] = str(model["name"])
        if model.get("type"):
            out["type"] = str(model["type"])
    mid = _clean_id(obj.get("modelId"))
    if mid is not None:
        out["model_id"] = mid
    vid = _clean_id(obj.get("id"))
    if vid is not None:
        out["version_id"] = vid
    imgs = obj.get("images")
    if isinstance(imgs, list):
        fallback = None
        any_img = None
        for im in imgs:
            if not isinstance(im, dict) or not im.get("url"):
                continue
            nsfw = im.get("nsfw")
            level = im.get("nsfwLevel")
            if any_img is None:
                any_img = im["url"]
            if nsfw in (None, False, "None", "Soft") and level in (None, 0, 1, 2):
                out["thumbnail"] = _thumb_url(im["url"])
                break
            # 回退候选：第一张未被标成成人的图。全显式画廊 -> 干脆无缩略图。
            if fallback is None and not _is_adult_image(nsfw, level):
                fallback = im["url"]
        if "thumbnail" not in out and fallback:
            out["thumbnail"] = _thumb_url(fallback)
        # 最后手段，仅当用户主动要求：全显式画廊。
        if "thumbnail" not in out and allow_adult and any_img:
            out["thumbnail"] = _thumb_url(any_img)
    return out


# ── Civitai 账户（可选 API key + 首选主机）─────────────────────────────────
#
# 为什么有 key：Civitai 对匿名 API 请求隐藏成人评级模型，`model-versions/by-hash`
# 对它们返回普通 404——从节点看与"文件不在 Civitai"无法区分。key 让同一请求
# 返回记录。
#
# 存在哪里、为什么不放明显位置：
#   - 不放 node.properties：会被序列化进工作流 .json，分享工作流/图片时泄漏。
#   - 不放注册的 ComfyUI 设置：comfy.settings.json 会整份交给浏览器。
#   - 放服务器独读的文件（路由决定路径），浏览器只被告知"是否已配置 + 后 4 位"。

_CIVITAI_HOST_PREFS = ("com", "red")


def sanitize_civitai_key(raw):
    """清洗粘贴的 API key，无法成 key 返回 ""。

    发现任何意外内容即拒绝而非清洗。key 会进 HTTP 请求头，一个游离的 CR/LF
    就是头注入——对"看起来不像 key"的安全答复是拒绝。前后空白是唯一例外：
    复制粘贴几乎总带尾部换行。"""
    if not isinstance(raw, str):
        return ""
    k = raw.strip()
    if not k or len(k) > 200:
        return ""
    for ch in k:
        # 仅可打印 ASCII——无控制字符、无空格、无怪字符。
        if ord(ch) < 33 or ord(ch) > 126:
            return ""
    return k


def mask_civitai_key(key):
    """可安全显示的提示：仅后 4 位。足以区分两个 key，对窥屏/截图无用。"""
    if not isinstance(key, str) or not key:
        return ""
    tail = key[-4:] if len(key) > 4 else key
    return "•" * 6 + tail


def civitai_hosts(pref):
    """按主机偏好给出尝试顺序的 API 主机。两个主机总在列表里：偏好只决定
    先问谁，另一个作为已存在的备胎（某些网络按域名屏蔽其中一个）。"""
    if pref == "red":
        return ("civitai.red", "civitai.com")
    return ("civitai.com", "civitai.red")


def read_civitai_account(path):
    """读账户文件。恒返回完整形状、永不抛错：缺失/损坏文件必须让查询像
    完全没有 key 一样工作，而不是弄坏节点。"""
    out = {"key": "", "host": "com", "adult_thumbs": False}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return out
    if not isinstance(obj, dict):
        return out
    out["key"] = sanitize_civitai_key(obj.get("key"))
    if obj.get("host") in _CIVITAI_HOST_PREFS:
        out["host"] = obj["host"]
    out["adult_thumbs"] = bool(obj.get("adult_thumbs"))
    return out


def write_civitai_account(path, account):
    """写账户文件。成功返回 True，永不抛错。

    创建即 0600（Linux/macOS 上有意义）；每个值在进入前重新消毒，直接 POST
    无法把换行塞进文件让下次读取信任。"""
    data = {
        "key": sanitize_civitai_key(account.get("key")),
        "host": account.get("host") if account.get("host") in _CIVITAI_HOST_PREFS else "com",
        "adult_thumbs": bool(account.get("adult_thumbs")),
    }
    try:
        # 创建时即受限，而非先建后修：open(path,"w") 在常规 umask 下是 0644，
        # 随后的 chmod 是另一个系统调用——共享机器上首次写入的窗口期 key
        # 世界可读。窗口很短且只在首次写（后续重写复用已有 mode）。
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        # 仍要 chmod：修复旧构建留下的 0644 文件（os.open 的 mode 只在创建时生效）。
        try:
            os.chmod(path, 0o600)
        except Exception:
            pass
        return True
    except Exception:
        return False


# ── 用户自定义触发词（按 LoRA 存储）───────────────────────────────────────
#
# 为什么按文件存一份：用户自己的触发词属于 LoRA 文件，不属于某一行——
# 切走行再切回来、或另一个节点/工作流用同一 LoRA 都应该看到它。存
# ComfyUI user 目录的单一文件（路由决定路径），不写进 models 目录
# （可能是只读/网络盘，且避免与用户自己的 <base>.json 撞车）。
_MAX_CUSTOM_WORDS = 64      # 单个 LoRA 上限——与 JS core 的 cap 匹配
_MAX_CUSTOM_LEN = 200       # 触发短语，不是小作文
_MAX_CUSTOM_LORAS = 5000    # 全库 sanity 上限


def custom_trigger_key(name):
    """把 LoRA 名归一化为存储键。分隔符折叠为 `/`，Windows/Linux 间复制的
    存储仍能匹配。垃圾返回 ""。"""
    if not isinstance(name, str):
        return ""
    return name.strip().replace("\\", "/").strip("/")


def sanitize_custom_words(words):
    """干净、去重、限长的触发词列表。永不抛错。去重大小写不敏感但保留
    用户先输入的拼写（大写存活）。"""
    out, seen = [], set()
    if not isinstance(words, (list, tuple)):
        return out
    for w in words:
        if not isinstance(w, str):
            continue
        s = w.strip()[:_MAX_CUSTOM_LEN].strip()
        if not s:
            continue
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
        if len(out) >= _MAX_CUSTOM_WORDS:
            break
    return out


def sanitize_custom_description(desc):
    """自定义描述：str 校验 + 限长 + strip。垃圾返回 "". 永不抛错。"""
    if not isinstance(desc, str):
        return ""
    return desc.strip()[:_MAX_DESCRIPTION_LEN]


def _norm_store_entry(v):
    """旧 {key: [words]} 与新 {key: {"words": [...], "description": str}} 兼容
    归一。非 dict/list 垃圾 -> 空条目。"""
    if isinstance(v, dict):
        return {
            "words": sanitize_custom_words(v.get("words")),
            "description": sanitize_custom_description(v.get("description")),
        }
    return {"words": sanitize_custom_words(v), "description": ""}


def read_custom_store(path):
    """整个存储为 {key: {"words": [...], "description": str}}。旧形状
    （{key: [words]}）自动升级读取。永不抛错——缺失/损坏文件必须读作空，
    绝不能弄坏面板。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return {}
    if not isinstance(obj, dict):
        return {}
    out = {}
    for name, v in obj.items():
        key = custom_trigger_key(name)
        if not key:
            continue
        entry = _norm_store_entry(v)
        if entry["words"] or entry["description"]:
            out[key] = entry
        if len(out) >= _MAX_CUSTOM_LORAS:
            break
    return out


def write_custom_store(path, store):
    """写整个存储（新形状 {key: {"words", "description"}}）。成功返回 True，
    永不抛错。

    临时文件 + os.replace：这单个文件装着每个 LoRA 的词与描述，半路崩溃/
    磁盘满会毁掉全部而非刚编辑的那个。临时名带 pid 和线程 id——路由把它
    交给 run_in_executor，两次保存落在共享同一 pid 的两个池线程上。"""
    data = {}
    if isinstance(store, dict):
        for name, entry in store.items():
            key = custom_trigger_key(name)
            if not key:
                continue
            e = _norm_store_entry(entry)
            if e["words"] or e["description"]:
                data[key] = e
            if len(data) >= _MAX_CUSTOM_LORAS:
                break
    tmp = "%s.%d.%d.tmp" % (path, os.getpid(), threading.get_ident())
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
        return True
    except Exception:
        try:
            os.remove(tmp)
        except Exception:
            pass
        return False


def read_custom_triggers(path):
    """整个存储为 {key: [words]}（从 store 抽 words 的视图）。旧签名兼容。
    永不抛错。"""
    return {k: v["words"] for k, v in read_custom_store(path).items()}


def write_custom_triggers(path, store):
    """旧签名：写 {key: [words]}。合并保留已有 description。成功返回 True，
    永不抛错。"""
    old = read_custom_store(path)
    merged = {}
    if isinstance(store, dict):
        for name, words in store.items():
            key = custom_trigger_key(name)
            if not key:
                continue
            merged[key] = {
                "words": sanitize_custom_words(words),
                "description": old.get(key, {}).get("description", ""),
            }
    return write_custom_store(path, merged)


def get_custom_triggers(path, name):
    """一个 LoRA 的已存词（可能为空）。永不抛错。"""
    key = custom_trigger_key(name)
    if not key:
        return []
    return read_custom_store(path).get(key, {}).get("words", [])


def set_custom_triggers(path, name, words):
    """替换一个 LoRA 的词。空列表且无描述时移除条目（存储不积累死键）。
    返回实际存储的列表。永不抛错。"""
    key = custom_trigger_key(name)
    if not key:
        return []
    store = read_custom_store(path)
    clean = sanitize_custom_words(words)
    entry = store.get(key)
    desc = (entry or {}).get("description", "")
    if clean or desc:
        store[key] = {"words": clean, "description": desc}
    else:
        store.pop(key, None)
    write_custom_store(path, store)
    return clean


def get_custom_description(path, name):
    """一个 LoRA 的已存自定义描述（可能为空）。永不抛错。"""
    key = custom_trigger_key(name)
    if not key:
        return ""
    return read_custom_store(path).get(key, {}).get("description", "")


def set_custom_description(path, name, desc):
    """替换一个 LoRA 的自定义描述。空描述且无词时移除条目。
    返回实际存储的描述。永不抛错。"""
    key = custom_trigger_key(name)
    if not key:
        return ""
    store = read_custom_store(path)
    clean = sanitize_custom_description(desc)
    entry = store.get(key)
    words = (entry or {}).get("words", [])
    if clean or words:
        store[key] = {"words": words, "description": clean}
    else:
        store.pop(key, None)
    write_custom_store(path, store)
    return clean


# ── 文件移动/改名后的孤儿数据迁移 ─────────────────────────────────────────
#
# 自定义词/描述/预览图都以 LoRA 相对路径名为键。用户整理目录（移动/改名
# 文件）后键失配：数据还在存储里，但新路径名下读不到。侧车（<base>.
# .civitai.info）随文件走，天然跟上；这三样需要显式迁移。
#
# 匹配策略：基名唯一匹配——存储里恰有一个键的基名（去目录、去扩展名）与
# 当前文件相同。唯一才匹配：同名多目录会歧义，宁可放弃也不冒误配。迁移
# 由前端提示 + 用户确认触发，不自动执行。

_LORA_KEY_EXTS = (".safetensors", ".safetensor", ".ckpt", ".pt", ".pth", ".bin", ".sft")


def base_key(key):
    """存储键 -> 基名（去目录与扩展名）。垃圾返回 "". 永不抛错。"""
    if not isinstance(key, str):
        return ""
    b = key.replace("\\", "/").rsplit("/", 1)[-1]
    low = b.lower()
    for ext in _LORA_KEY_EXTS:
        if low.endswith(ext):
            return b[:-len(ext)]
    return b


def find_orphan_key(store, name):
    """store 中与 name 基名相同、键不同的条目。唯一才返回（同名多目录 ->
    歧义，放弃自动匹配），否则 None。永不抛错。"""
    key = custom_trigger_key(name)
    if not key:
        return None
    target = base_key(key)
    if not target:
        return None
    matches = [k for k in store if k != key and base_key(k) == target]
    return matches[0] if len(matches) == 1 else None


def migrate_custom_data(path, name):
    """把旧路径键下的自定义词/描述迁移到当前 name 键（基名唯一匹配时）。

    新键已有数据不迁移（不覆盖）；旧键空不迁移。返回
    {"ok": True, "old_key": ...} 或 {"ok": False, "reason": ...}。永不抛错。"""
    key = custom_trigger_key(name)
    if not key:
        return {"ok": False, "reason": "bad name"}
    store = read_custom_store(path)
    cur = store.get(key)
    if cur and (cur["words"] or cur["description"]):
        return {"ok": False, "reason": "already has data"}
    old = find_orphan_key(store, name)
    if old is None:
        return {"ok": False, "reason": "no unique match"}
    entry = store.get(old)
    if not entry or not (entry["words"] or entry["description"]):
        return {"ok": False, "reason": "old entry empty"}
    store[key] = entry
    del store[old]
    write_custom_store(path, store)
    return {"ok": True, "old_key": old}


def migrate_custom_preview(folder, name, old_key):
    """把旧键的预览图文件迁移到当前 name 的 hash 名下（同目录 rename）。

    旧文件存在且新目标不存在才迁移；已存在则不覆盖（保留现状）。
    成功返回 True。永不抛错。"""
    old = custom_preview_path(folder, old_key)
    new = custom_preview_path(folder, name)
    if not old or not new or old == new:
        return False
    if not os.path.isfile(old) or os.path.isfile(new):
        return False
    try:
        os.rename(old, new)
        return True
    except Exception:
        return False


# ── 用户自己的预览图 ──────────────────────────────────────────────────────
#
# 存在 user 目录（路由决定文件夹），原因同自定义词存储：models 目录常只读/
# 网络盘，且写 <base>.preview.png 会覆盖 Civitai 助手已放在 LoRA 旁的图。
# 这张是 OVERRIDE，胜过侧车预览和实时 Civitai 缩略图；删除即恢复自动图，
# 用户已有的东西永不丢失。
#
# 文件名从 LoRA 名推导，因此重复写同一路径。浏览器需要缓存破击版本，
# 这就是 custom_preview_version 返回文件 mtime（毫秒）的原因——计数器在
# 文件被手删后从 1 重启，会把浏览器仍持有的 URL 交回去。

# 名字是我们自己生成的，非此形状的都不是我们写的。对安全至关重要：
# delete_custom_preview 把结果喂给 os.remove，而 os.path.join 在第二段为
# 绝对路径时会丢弃文件夹。
_CUSTOM_PREVIEW_RE = re.compile(r"[0-9a-f]{16}\.jpg")


def custom_preview_name(name):
    """我们将为这个 LoRA 存储的自定义预览文件名，垃圾返回 ""。从与触发词
    存储相同的归一化 key 哈希而来，两者对子文件夹/跨 OS 的 LoRA 保持同步。"""
    key = custom_trigger_key(name)
    if not key:
        return ""
    return hashlib.sha1(key.encode("utf-8", "replace")).hexdigest()[:16] + ".jpg"


def is_custom_preview_name(name):
    """这是我们能写出的文件名吗？每个 os.remove 都经过它。"""
    return isinstance(name, str) and bool(_CUSTOM_PREVIEW_RE.fullmatch(name))


def custom_preview_path(folder, name):
    """一个 LoRA 自定义预览的完整路径，名字垃圾或拼接后落在 `folder` 外返回
    None。不检查文件是否存在。"""
    fn = custom_preview_name(name)
    if not fn or not folder or not is_custom_preview_name(fn):
        return None
    path = os.path.join(str(folder), fn)
    # 保险带：fn 构造上就是 16 hex + .jpg，这条检查当前不可能失败——
    # 但正是它让这从假设变成保证。
    try:
        base = os.path.abspath(str(folder))
        full = os.path.abspath(path)
    except Exception:
        return None
    if os.path.dirname(full) != base:
        return None
    return full


def find_custom_preview(folder, name):
    """这个 LoRA 的用户自定义预览路径，或 None。永不抛错。"""
    path = custom_preview_path(folder, name)
    try:
        if path and os.path.isfile(path):
            return path
    except Exception:
        pass
    return None


def custom_preview_version(folder, name):
    """用户自定义预览的 mtime（毫秒），不存在返回 0。预览 URL 永不变化
    （文件名从 LoRA 名推导）且缩略图路由缓存一小时——mtime 是让替换后的图
    在另一个节点/刷新后出现的东西。"""
    path = find_custom_preview(folder, name)
    if not path:
        return 0
    try:
        return int(os.path.getmtime(path) * 1000)
    except Exception:
        return 0


def write_custom_preview(folder, name, raw):
    """把 `raw` 存为该 LoRA 的自定义预览。返回路径或 None。

    临时文件 + os.replace，与所有重复文件名写入相同：同一 LoRA 的路径每次
    都相同，直接写会让在途请求读到写了一半的 jpg。临时名带 pid + 线程 id。"""
    path = custom_preview_path(folder, name)
    if not path or not isinstance(raw, (bytes, bytearray)) or not raw:
        return None
    try:
        os.makedirs(str(folder), exist_ok=True)
    except Exception:
        return None
    tmp = "%s.%d.%d.tmp" % (path, os.getpid(), threading.get_ident())
    try:
        with open(tmp, "wb") as f:
            f.write(bytes(raw))
        os.replace(tmp, path)
        return path
    except Exception:
        try:
            os.remove(tmp)
        except Exception:
            pass
        return None


def delete_custom_preview(folder, name):
    """删除用户自定义预览，让自动图回来。删掉了返回 True。永不抛错。"""
    path = find_custom_preview(folder, name)
    if not path:
        return False
    try:
        os.remove(path)
        return True
    except Exception:
        return False


def save_sidecar_cache(lora_path, civitai_obj):
    """把原始 Civitai 响应缓存到 LoRA 旁的 <base>.civitai.info，未来读取即时
    且离线。成功返回 True。永不抛错。"""
    try:
        base = os.path.splitext(lora_path)[0]
        with open(base + ".civitai.info", "w", encoding="utf-8") as f:
            json.dump(civitai_obj, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def sidecar_thumbnail(lora_path, allow_adult=False):
    """从侧车（原始 Civitai 响应）提取缩略图 URL，无侧车/无可用图返回 None。

    供"用户确认后保存封面"端点使用：查询时已跳过保存（用户有自定义预览），
    确认后从这里拿到同一张图重新下载。永不抛错。"""
    base = os.path.splitext(lora_path)[0]
    for ext in (".civitai.info", ".json"):
        sp = base + ext
        if not os.path.isfile(sp):
            continue
        try:
            with open(sp, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        parsed = parse_civitai_modelversion(obj, allow_adult=allow_adult)
        if parsed.get("thumbnail"):
            return parsed["thumbnail"]
    return None


def delete_sidecar_cache(lora_path):
    """删除 LoRA 旁的 Civitai 缓存侧车（<base>.civitai.info），信息回到文件
    自己的词（或重新查询）。消失（已删或本就不存在）返回 True。永不抛错。
    不动用户自己的 <base>.json。"""
    try:
        p = os.path.splitext(lora_path)[0] + ".civitai.info"
        if os.path.isfile(p):
            os.remove(p)
        return True
    except Exception:
        return False
