"""LoRA 正交堆叠纯逻辑（SFLoraStack 的 ortho_gs 模式使用）。

数学：顺序叠加 LoRA 时 ΔW = Σ s·(α/r)·(A_i·B_i)，其中 A_i = up（out×rank）、
B_i = down（rank×in）。若多个 LoRA 的 down 矩阵行（输入方向）重叠，它们读取
相同的输入方向却叠加不同的修改——干扰（相消/相涨），这就是多个相似 LoRA
叠在一起效果变糊的原因。

gram_schmidt_ortho_downs 在输入空间做 Gram-Schmidt：
  1. 第一个 down 原样保留，其行空间提取为正交基（SVD 右奇异向量）；
  2. 后续每个 down 的行投影到已提交基的正交补（d' = d - (d @ Qᵀ) @ Q）；
  3. 基扩展（SVD + QR 去线性相关），依序处理整个栈。
结果每个 LoRA 只读前面 LoRA 忽略的输入方向，干扰项消零；up 矩阵、alpha、
强度缩放全不动。投影后 down 被前面覆盖的行会损失幅度——tradeoff，设计使然。

仅依赖 torch（运行时由 ComfyUI 提供）——无 comfy/folder_paths 依赖，可在
ComfyUI 之外单测（tests/test_lora_ortho.py 的 numpy 参考对照）。
extract_up_down / replace_down 兼容当前与历史 ComfyUI patch 结构。
"""
import copy

import torch


def gram_schmidt_ortho_downs(downs_list):
    """对 down 投影矩阵列表做输入空间 Gram-Schmidt 正交化。

    downs_list: [(rank, in_dim) 二维 float 张量, ...]（栈顺序，第一个不动）。
    返回: 同顺序的投影后张量（保持各自原 dtype/device），非 2D 条目原样直通。
    数值细节：float32 计算防 fp16 精度损失；SVD 阈值 1e-6 相对 + 1e-10 绝对；
    QR 去线性相关；行空间被完全覆盖（fro < 1e-10）的 down 并入基时跳过。
    """
    result = []
    basis = None  # (k, in_dim) —— 累积的正交行基

    for down in downs_list:
        orig_dtype = down.dtype
        d = down.float()

        if d.dim() != 2:
            result.append(down)
            continue

        if basis is not None and basis.shape[0] > 0:
            # 每行投影到已提交子空间，再取正交补分量。
            d_ortho = d - (d @ basis.T) @ basis
        else:
            d_ortho = d.clone()

        result.append(d_ortho.to(orig_dtype))

        # 把 d_ortho 的行空间并入基（SVD 右奇异向量张成行空间）。
        if d_ortho.norm(p="fro").item() < 1e-10:
            continue  # 行空间已空（被前面完全覆盖），无可并入的方向
        try:
            _, S, Vh = torch.linalg.svd(d_ortho, full_matrices=False)
            tol = max(S[0].item() * 1e-6, 1e-10) if S.numel() > 0 else 1e-10
            keep = S > tol
            if not keep.any():
                continue
            basis_new = Vh[keep]
            if basis is None:
                basis = basis_new
            else:
                # 拼接后 QR 再正交化（直接拼接可能线性相关）。
                combined = torch.cat([basis, basis_new], dim=0).T
                basis_upd, R = torch.linalg.qr(combined, mode="reduced")
                r_diag = R.diagonal().abs()
                if r_diag.numel() > 0 and r_diag.max().item() > 0:
                    basis = basis_upd[:, r_diag > r_diag.max().item() * 1e-6].T
                else:
                    basis = basis_upd.T
        except Exception:
            # SVD/QR 数值失败：本次不扩展基（后续投影沿用旧基），永不抛错。
            pass

    return result


def build_ortho_replacements(patch_dicts):
    """对多个 LoRA 的 patch dict 做分组 Gram-Schmidt 正交化替换。

    patch_dicts: [(patch_dict, strength)] —— 每个 patch_dict 是一个 LoRA 的
    {model_key: patch} 映射，顺序即栈顺序（第一个 LoRA 不动）。
    返回 (replaced, ortho_keys, pass_keys)：
      replaced = [(new_patch_dict, strength)] —— 新 dict，仅重叠 key 的 down
                被替换（其余原样）；顺序与输入一致。
      ortho_keys = 正交化 key 数；pass_keys = 直通 key 数（单条目 / 非 LoRA
                  patch / conv 等提取失败，该 key 全部保留原 patch）。
    纯逻辑（无 comfy 依赖）——patch 已由 comfy.lora.load_lora 解析完毕。
    """
    # 按模型 key 分组（同一 key 的多个 LoRA 才需要正交化；栈顺序保留）。
    key_to_entries = {}  # key -> [(patch, strength)]
    for patches, _strength in patch_dicts:
        for key, patch in patches.items():
            key_to_entries.setdefault(key, []).append((patch, _strength))

    # 每 key 预处理：ortho_by_key[key] 为 None 表示直通原 patch；
    # 否则是 {原 patch: 替换后 patch}（同一 key 的多 LoRA 正交化产物）。
    ortho_by_key = {}
    ortho_keys = 0
    for key, entries in key_to_entries.items():
        if len(entries) == 1:
            ortho_by_key[key] = None
            continue
        components = []  # [(patch, down)]
        fallback = False
        for patch, _sm in entries:
            up, down = extract_up_down(patch)
            if up is None or down is None or down.dim() != 2:
                # conv 等非 LoRA patch：该 key 全部顺序叠加。
                fallback = True
                break
            components.append((patch, down))
        if fallback:
            ortho_by_key[key] = None
            continue
        ortho_downs = gram_schmidt_ortho_downs([d for _p, d in components])
        ortho_by_key[key] = {
            p: replace_down(p, od.to(d.dtype))
            for (p, d), od in zip(components, ortho_downs)
        }
        ortho_keys += 1
    pass_keys = len(key_to_entries) - ortho_keys

    replaced = []
    for patches, strength in patch_dicts:
        new_dict = {}
        for key, patch in patches.items():
            entry_map = ortho_by_key[key]
            new_dict[key] = patch if entry_map is None else entry_map[patch]
        replaced.append((new_dict, strength))
    return replaced, ortho_keys, pass_keys


def extract_up_down(patch):
    """从 ComfyUI patch 中提取 (lora_up, lora_down)，无法识别返回 (None, None)。

    兼容的 patch 结构：
      LoRAAdapter 对象 —— comfy.weight_adapter.lora，weights = (up, down, alpha,
                        mid, dora_scale, reshape)，up 是 weights[0]、down 是
                        weights[1]（当前 ComfyUI，见 weight_adapter/lora.py）
      ("lora", (up, down, alpha, dora))  字符串标签格式（现代）
      (up, down, ...)                    tensor 优先格式（旧）
      (alpha, (up, down, ...))           float 前缀格式（个别版本）
    conv/diff/set 等非 LoRA patch 返回 (None, None)，调用方 fallback 顺序叠加。
    """
    # comfy.weight_adapter.lora.LoRAAdapter —— 当前 ComfyUI
    if hasattr(patch, "weights"):
        w = patch.weights
        if isinstance(w, (tuple, list)) and len(w) >= 2:
            return w[0], w[1]
        return None, None

    if isinstance(patch, (list, tuple)) and len(patch) >= 2:
        first = patch[0]

        if isinstance(first, str):
            # ("lora", (up, down, alpha, dora))
            v = patch[1]
            if isinstance(v, (tuple, list)) and len(v) >= 2:
                return v[0], v[1]

        elif isinstance(first, torch.Tensor):
            # (up, down, ...)
            return patch[0], patch[1]

        elif isinstance(first, (int, float)):
            rest = patch[1]
            if isinstance(rest, str) and len(patch) >= 3:
                inner = patch[2]
                if isinstance(inner, (tuple, list)) and len(inner) >= 2:
                    return inner[0], inner[1]
            elif isinstance(rest, (tuple, list)) and len(rest) >= 2:
                if isinstance(rest[0], torch.Tensor):
                    return rest[0], rest[1]
                elif isinstance(rest[0], str) and len(rest) >= 2:
                    inner = rest[1]
                    if isinstance(inner, (tuple, list)) and len(inner) >= 2:
                        return inner[0], inner[1]

    return None, None


def replace_down(patch, new_down):
    """返回 patch 的副本，其 lora_down 被替换为 new_down（原 patch 不被修改）。

    与 extract_up_down 支持的格式一一对应；无法识别的格式原样返回。
    """
    # comfy.weight_adapter.lora.LoRAAdapter
    if hasattr(patch, "weights"):
        new_patch = copy.copy(patch)
        w = patch.weights
        new_patch.weights = (w[0], new_down) + tuple(w[2:])
        return new_patch

    if not isinstance(patch, (list, tuple)) or len(patch) < 2:
        return patch

    first = patch[0]

    if isinstance(first, str):
        v = patch[1]
        if not isinstance(v, (tuple, list)) or len(v) < 2:
            return patch  # ("diff", (w,)) 等非 LoRA 标签：原样返回
        v = list(v)
        v[1] = new_down
        return (patch[0], tuple(v))

    elif isinstance(first, torch.Tensor):
        v = list(patch)
        v[1] = new_down
        return tuple(v)

    elif isinstance(first, (int, float)):
        rest = patch[1]
        if isinstance(rest, str) and len(patch) >= 3:
            inner = list(patch[2])
            inner[1] = new_down
            return (patch[0], patch[1], tuple(inner))
        elif isinstance(rest, (tuple, list)) and len(rest) >= 2:
            if isinstance(rest[0], torch.Tensor):
                inner = list(rest)
                inner[1] = new_down
                return (patch[0], tuple(inner))
            elif isinstance(rest[0], str) and len(rest) >= 2:
                inner = list(rest[1])
                inner[1] = new_down
                return (patch[0], (rest[0], tuple(inner)))

    return patch
