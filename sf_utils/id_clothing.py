"""SFIDClothingSelector 纯逻辑（证件照服装发型单选版）。

复用 styles JSON 生态（user/sfnodes/styles/*.json + samples/），本模块只放
styles_selector.py 没有的单选收敛语义，全部无 ComfyUI/torch 依赖、可独立测试：

- parse_state/first_selected/serialize_single：SFStylesState 同形 JSON 数组，
  但本节点只取首个有效名（单选；多值旧数据向前兼容取首个）。
- resolve_prompt：手改草稿优先（SFTextPreset text_override 同款），否则模板原词。
- find_style：按名查条目。
- parse_goohai_filename/goohai_entry：一次性转换脚本用的孤海
  `标题-提示词.扩展名` 映射（`标题-提示词` 按首个 `-` 切分，无 `-` 则提示词为空）。
- resolve_thumbnail_path：条目 thumbnail 落盘解析（远程 http(s) 返回 None；
  本地 `samples/x.jpg` 或 styles 图片路由 `?path=` 形式在 styles 目录内
  commonpath 钳位，防 `../../etc/passwd` 穿越）。
"""

import json
import os
import urllib.parse


LIB_PREFIX = "id_"

_PLACEHOLDER_TENSOR_SHAPE = (1, 1, 1, 3)


def is_remote_thumb(url):
    return isinstance(url, str) and url.startswith(("http://", "https://"))


def thumbnail_of(entry):
    t = (entry or {}).get("thumbnail") if isinstance(entry, dict) else None
    if isinstance(t, list):
        return t[0] if t else ""
    return t or ""


def parse_state(state):
    """解析隐藏 SFIDClothingState（JSON 数组），畸形输入容错为空列表。"""
    if isinstance(state, list):
        return [str(v) for v in state if str(v)]
    if not isinstance(state, str):
        return []
    try:
        v = json.loads(state) if state else []
    except Exception:
        return []
    return [str(x) for x in v if str(x)] if isinstance(v, list) else []


def serialize_single(name):
    """单选序列化：空名存 []，否则 [name]（与 styles 状态形状一致）。"""
    name = str(name or "")
    return json.dumps([name] if name else [])


def first_selected(state):
    """单选收敛：取解析后首个有效名，无选择返回 ""。"""
    names = parse_state(state)
    return names[0] if names else ""


def find_style(styles_data, name):
    if not name or not isinstance(styles_data, list):
        return None
    for d in styles_data:
        if isinstance(d, dict) and d.get("name") == name:
            return d
    return None


def resolve_prompt(styles_data, selected_name, draft=""):
    """提示词决策：手改草稿非空优先，否则模板原 prompt，缺省 ""。"""
    if draft is not None and str(draft) != "":
        return str(draft)
    entry = find_style(styles_data, selected_name)
    if entry is not None:
        prompt = entry.get("prompt", "")
        return str(prompt) if prompt is not None else ""
    return ""


def filter_libraries(names):
    """新节点 library 下拉只列 id_ 前缀库（服装库与风格库隔离，防串库）。"""
    return [n for n in (names or []) if isinstance(n, str) and n.startswith(LIB_PREFIX)]


def parse_goohai_filename(stem):
    """孤海文件名杆 `标题-提示词` 解析（一次性转换用；无 `-` 则提示词为空）。"""
    if not isinstance(stem, str):
        return ("", "")
    if "-" in stem:
        title, prompt = stem.split("-", 1)
    else:
        title, prompt = stem, ""
    return (title.strip(), prompt.strip())


def goohai_entry(title, prompt, sample_filename):
    """孤海模板 → styles JSON 条目（thumbnail 指向用户库 samples/ 下文件名）。"""
    entry = {"name": str(title or "").strip()}
    if prompt is not None and str(prompt) != "":
        entry["prompt"] = str(prompt)
    entry["thumbnail"] = "samples/" + str(sample_filename)
    return entry


def _candidate_rel_paths(thumbnail):
    """thumbnail 落盘候选相对路径（路由 ?path= 形式先解 query 取 path 参数）。"""
    if not isinstance(thumbnail, str) or not thumbnail:
        return []
    if thumbnail.startswith("/api/sfnodes/styles/image"):
        try:
            query = urllib.parse.urlparse(thumbnail).query
            rel = urllib.parse.parse_qs(query).get("path", [""])[0]
        except Exception:
            rel = ""
        return [rel] if rel else []
    return [thumbnail]


def resolve_thumbnail_path(thumbnail, styles_dirs):
    """thumbnail → styles 目录内的绝对路径；远程/缺失/穿越返回 None。

    styles_dirs: 用户优先的搜索目录列表（调用方传入 styles_selector._styles_dirs()，
    本模块不 import 节点层，避免循环依赖）。
    搜索顺序与图片路由一致：每个目录的根与 samples/ 子目录。
    """
    if is_remote_thumb(thumbnail):
        return None
    for rel in _candidate_rel_paths(thumbnail):
        if not isinstance(rel, str) or not rel:
            continue
        rel_norm = rel.replace("\\", "/").strip()
        if not rel_norm or rel_norm.startswith("/") or ".." in rel_norm.split("/"):
            continue
        for d in styles_dirs or []:
            try:
                base = os.path.abspath(d)
            except Exception:
                continue
            for sub in ("", "samples"):
                root = os.path.abspath(os.path.join(base, sub))
                try:
                    p = os.path.normpath(os.path.join(root, rel_norm))
                except Exception:
                    continue
                try:
                    if os.path.commonpath((root, p)) != root:
                        continue
                except Exception:
                    continue
                if os.path.isfile(p):
                    return p
                # thumbnail 已含 samples/ 前缀时，root=samples 会拼出 samples/samples/；
                # 回退按“相对库根”再试一次（幂等去重由调用方无感）。
                if sub == "samples" and rel_norm.startswith("samples/"):
                    slim = rel_norm[len("samples/"):]
                    q = os.path.normpath(os.path.join(root, slim))
                    try:
                        if os.path.commonpath((root, q)) != root:
                            continue
                    except Exception:
                        continue
                    if os.path.isfile(q):
                        return q
    return None
