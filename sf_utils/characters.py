"""SFCharacterSelect 纯逻辑（角色内多选图 → batch 输出）。

角色条目形状（新）：
  {name, prompt?, images?: [{label, file, prompt?}]}
旧形状双读兼容：
  {name, prompt?, face?, half?, full?} → 映射为 3 项 images
  （label 取 脸部特写/半身像/全身像，prompt 回落角色级 prompt）。

选择状态形状：{"role": 名, "shots": [label]}；旧 `["名"]` 数组迁移为首图。
无 ComfyUI/torch 依赖、可独立测试。
"""

import json

LIB_PREFIX = "character_"

# 旧三分镜字段 → 新 images label（双读映射，保持中文分镜名稳定）
LEGACY_SHOTS = (("face", "脸部特写"), ("half", "半身像"), ("full", "全身像"))


def parse_state(state):
    """解析隐藏 SFCharacterState，新旧两态归一为 {"role", "shots"}。

新态 {"role", "shots"} 原样归一（shots 非数组时置空数组）；
旧数组 ["名"] 迁移为 {"role": 名, "shots": None}（None = 首图，保持默认语义统一）；
畸形输入容错为 {"role": "", "shots": []}。
"""
    if isinstance(state, dict):
        role = state.get("role")
        shots = state.get("shots")
        return {
            "role": str(role) if role else "",
            "shots": [str(s) for s in shots if str(s)] if isinstance(shots, list) else [],
        }
    if isinstance(state, list):
        names = [str(v) for v in state if str(v)]
        if names:
            return {"role": names[0], "shots": None}
        return {"role": "", "shots": []}
    if not isinstance(state, str):
        return {"role": "", "shots": []}
    try:
        v = json.loads(state) if state else None
    except Exception:
        return {"role": "", "shots": []}
    if isinstance(v, dict):
        return parse_state(v)
    if isinstance(v, list):
        return parse_state(v)
    return {"role": "", "shots": []}


def serialize_selection(role, shots):
    """选择序列化（数组保序去重，调用方按库内顺序传入）。"""
    seen = []
    for s in shots or []:
        s = str(s)
        if s and s not in seen:
            seen.append(s)
    return json.dumps({"role": str(role or ""), "shots": seen})


def find_role(roles_data, name):
    if not name or not isinstance(roles_data, list):
        return None
    for d in roles_data:
        if isinstance(d, dict) and d.get("name") == name:
            return d
    return None


def image_entries(entry):
    """角色条目 → 标准 images 列表 [{label, file, prompt}]（新旧双读，保库内顺序）。

    新形状取 images 数组（label/file 非空才收）；旧形状按 LEGACY_SHOTS 映射，
    单项 prompt 回落角色级 prompt。
    """
    if not isinstance(entry, dict):
        return []
    role_prompt = entry.get("prompt", "")
    role_prompt = str(role_prompt) if role_prompt is not None else ""
    imgs = entry.get("images")
    if isinstance(imgs, list):
        out = []
        for item in imgs:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", "") or "")
            file = item.get("file", "")
            file = str(file) if file is not None else ""
            if not label or not file:
                continue
            prompt = item.get("prompt", "")
            out.append({"label": label, "file": file,
                        "prompt": str(prompt) if prompt is not None else ""})
        return out
    out = []
    for field, label in LEGACY_SHOTS:
        file = entry.get(field, "")
        file = str(file) if file is not None else ""
        if file:
            out.append({"label": label, "file": file, "prompt": role_prompt})
    return out


def role_prompt(entry):
    """角色级 prompt 原值（缺省 ""；是否叠草稿由 resolve_role_prompt 决定）。"""
    if not isinstance(entry, dict):
        return ""
    prompt = entry.get("prompt", "")
    return str(prompt) if prompt is not None else ""


def resolve_role_prompt(entry, draft=""):
    """role_prompt 路输出决策：手改草稿非空优先，否则角色原词，缺省 ""。"""
    if draft is not None and str(draft) != "":
        return str(draft)
    return role_prompt(entry)


def coerce_selection(roles, state):
    """选择收敛：角色有效则保留并取交集分镜，否则回落首个角色首图。

    空 shots（含旧数组态的 None）一律收敛为该角色首图——显式全不选
    不跨重载保留（§47 同款语义）；空库回落 {"role": "", "shots": []}。
    返回 {"role", "shots"}（shots 恒为数组，按库内顺序）。
    """
    parsed = parse_state(state)
    data = roles if isinstance(roles, list) else []
    names = [d.get("name") for d in data if isinstance(d, dict) and d.get("name")]
    role_valid = parsed["role"] in names
    role = parsed["role"] if role_valid else (names[0] if names else "")
    entry = find_role(data, role)
    valid = [i["label"] for i in image_entries(entry)] if entry is not None else []
    if not role_valid or not parsed["shots"]:
        shots = valid[:1]
    else:
        wanted = set(parsed["shots"])
        shots = [label for label in valid if label in wanted]
        if not shots:
            shots = valid[:1]
    return {"role": role, "shots": shots}


def join_prompts(entry, labels):
    """选中分镜 prompt 逗号拼接（单项 prompt 为空回落角色级 prompt，保库内顺序）。"""
    if not isinstance(entry, dict):
        return ""
    fallback = role_prompt(entry)
    parts = []
    by_label = {i["label"]: i for i in image_entries(entry)}
    for label in labels or []:
        item = by_label.get(str(label))
        if item is None:
            continue
        text = item.get("prompt") or fallback
        if text:
            parts.append(text)
    return ", ".join(parts)


def resolve_prompt(entry, labels, draft=""):
    """拼接路提示词决策：手改草稿非空整体覆盖，否则分镜拼接，缺省 ""。"""
    if draft is not None and str(draft) != "":
        return str(draft)
    return join_prompts(entry, labels)


def filter_libraries(names):
    """角色库下拉只列 character_ 前缀库（与风格/服装库隔离，防串库）。"""
    return [n for n in (names or []) if isinstance(n, str) and n.startswith(LIB_PREFIX)]
