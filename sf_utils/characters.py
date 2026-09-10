"""SFCharacterSelect 纯逻辑（角色三分镜：脸部特写/半身像/全身像）。

角色条目形状：{name, prompt?, face?, half?, full?}，三图为相对路径
（指向角色库 samples 独立目录，如 samples_id_chara/<库>/face_01.jpg）。
落盘解析直接复用 id_clothing.resolve_thumbnail_path（同款 commonpath 钳位），
本模块只放角色域独有的单选/草稿/分镜语义，无 ComfyUI/torch 依赖、可独立测试。
"""

import json

LIB_PREFIX = "character_"

SHOTS = ("face", "half", "full")


def parse_state(state):
    """解析隐藏 SFCharacterState（JSON 数组），畸形输入容错为空列表。"""
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


def find_role(roles_data, name):
    if not name or not isinstance(roles_data, list):
        return None
    for d in roles_data:
        if isinstance(d, dict) and d.get("name") == name:
            return d
    return None


def resolve_prompt(roles_data, selected_name, draft=""):
    """提示词决策：手改草稿非空优先，否则角色原 prompt，缺省 ""。"""
    if draft is not None and str(draft) != "":
        return str(draft)
    entry = find_role(roles_data, selected_name)
    if entry is not None:
        prompt = entry.get("prompt", "")
        return str(prompt) if prompt is not None else ""
    return ""


def shot_thumbnail(entry, shot):
    """角色某分镜的 thumbnail 取单值（数组取首项，缺省空串）。"""
    t = entry.get(shot) if isinstance(entry, dict) else None
    if isinstance(t, list):
        return t[0] if t else ""
    return t or ""


def filter_libraries(names):
    """角色库下拉只列 character_ 前缀库（与风格/服装库隔离，防串库）。"""
    return [n for n in (names or []) if isinstance(n, str) and n.startswith(LIB_PREFIX)]
