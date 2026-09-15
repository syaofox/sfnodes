"""Krea2（SingleStreamDiT）参考图编辑补丁纯逻辑。

复刻自 ComfyUI-EditUtils 的 ``Krea2EditApply_EditUtils``（含内置参考图 KV 缓存），
逐行移植其 forward 改写与缓存实现，仅调整模块位置与少量命名。补丁语义：

* 序列顺序 ``[text | target | ref₁ | ref₂ | …]``；
  target 位置 id=(0,h,w)，refₙ=(+n,h,w)；
* ``mode="editutils"``（默认）ref token 用 t=0 调制（UnifiedTrainer 配方），
  ``mode="krea2edit"`` ref token 用真实 timestep（ai-toolkit predict_velocity_edit）；
* ``ref_pos_match_target``：把 ref 的 (h,w) 位置 id 中心对齐拉伸到 target token 网格；
* ``reference_rope_offsets``：按 patch_size 把像素偏移转成 token 偏移（区域编辑）；
* ``ref_kv_cache`` + ``ref_strength``：首步捕获每层 ref K/V，后续步复用；采样进度
  超过 ``ref_strength`` 后丢弃 ref（退化为纯文生图）。

重要：EditUtils 直接在补丁里重写了 Krea2 核心前向，依赖 ``dit`` 的
``_unpack_context/first/tmlp/tproj/txtfusion/txtmlp/pe_embedder/blocks/last`` 等属性
（ComfyUI 0.35.0 实测一致）。核心若改动 Krea2 前向，本模块需同步。
内部属性前缀保留原 ``_editutils_*`` 以免移植引入语义偏差。

纯逻辑，无节点注册副作用；torch/comfy 等重依赖在函数内惰性 import，便于本机 mock 测试。
"""

import math

import torch


# --- 无参考图时链回的原 forward 保存属性 ---
_EDITUTILS_ORIGINAL_FORWARD_ATTR = "_editutils_krea2_original_forward"

# KV 缓存状态属性
_KV_STATE_ATTR = "_editutils_ref_kv_state"       # 每次调用捕获/复用状态
_KV_CACHE_ATTR = "_editutils_ref_kv_cache"       # 持久缓存 dict
_KV_CFG_ATTR = "_editutils_ref_kv_cfg"           # 节点提供的调参配置

DEFAULT_INSTRUCTION = (
    "Describe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:"
)

_SYSTEM_PREFIX = "<|im_start|>system\n"
_SYSTEM_SUFFIX = "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"


def get_system_prompt(instruction):
    """构造 llama chat 模板（复刻 EditUtils get_system_prompt）。"""
    instruction_content = ""
    if instruction == "":
        instruction_content = (
            "Describe the key features of the input image (color, shape, size, texture, "
            "objects, background), then explain how the user's text instruction should "
            "alter or modify the image. Generate a new image that meets the user's "
            "requirements while maintaining consistency with the original input where "
            "appropriate."
        )
    else:
        if _SYSTEM_PREFIX in instruction:
            instruction = instruction.split(_SYSTEM_PREFIX)[1]
        if _SYSTEM_SUFFIX in instruction:
            instruction = instruction.split(_SYSTEM_SUFFIX)[0]
        if "{}" in instruction:
            instruction = instruction.replace("{}", "")
        instruction_content = instruction
    return _SYSTEM_PREFIX + instruction_content + _SYSTEM_SUFFIX


def crop_with_pad_info(image, pad_info):
    """按 pad_info 反向裁掉 padding，还原原始内容区。

    image: [B,H,W,C]；pad_info: {"x","y","width","height","scale_by"}。
    width/height 为右/下 padding 像素（resized 内容置于画布左上角 (0,0)）。
    返回 (cropped_image, scale_by)。
    """
    x = pad_info.get("x", 0)
    y = pad_info.get("y", 0)
    width_padding = pad_info.get("width", 0)
    height_padding = pad_info.get("height", 0)
    scale_by = pad_info.get("scale_by", 1.0)

    img = image.movedim(-1, 1)  # (B,H,W,C) -> (B,C,H,W)
    original_content_width = img.shape[3] - width_padding
    original_content_height = img.shape[2] - height_padding
    cropped_img = img[:, :, x:original_content_height, y:original_content_width]
    return cropped_img.movedim(1, -1), scale_by


def make_ref_positions(ref_latents_list, bs, patch_size, device, dtype,
                       scale_to_grid=None, rope_offsets=None):
    """为参考 latent 列表构造位置 id 块（复刻 EditUtils _editutils_make_ref_positions）。

    每张 ref 获得递增 frame index（ref₁=+1, ref₂=+2 …）。
    scale_to_grid=(th,tw) 时 ref (h,w) 中心对齐到目标网格：
    ``(i + 0.5) * (target / ref) - 0.5``。
    rope_offsets 为每张 ref 的 (x_offset, y_offset) 像素，按 patch_size 转 token 偏移。
    返回 (ref_tokens_list, ref_pos_list, total_reflen)。
    """
    from einops import rearrange
    import comfy.ldm.common_dit

    ref_tokens = []
    ref_positions = []
    total = 0

    for i, ref in enumerate(ref_latents_list, start=1):
        if ref.ndim == 5 and ref.shape[2] == 1:
            ref = ref.squeeze(2)
        elif ref.ndim == 5:
            raise ValueError(
                f"Krea2EditApply: reference latent has {ref.ndim}D shape "
                f"{tuple(ref.shape)} — expected 4D (B,C,H,W) or 5D with temporal=1"
            )

        if ref.shape[0] != bs:
            ref = ref.expand(bs, *ref.shape[1:])

        ref_pad = comfy.ldm.common_dit.pad_to_patch_size(ref, (patch_size, patch_size))
        rH, rW = ref_pad.shape[-2], ref_pad.shape[-1]
        rh, rw = rH // patch_size, rW // patch_size

        rt = rearrange(ref_pad, "b c (h ph) (w pw) -> b (h w) (c ph pw)",
                       ph=patch_size, pw=patch_size)
        ref_tokens.append(rt)

        frame_idx = float(i)
        rids = torch.zeros(rh, rw, 3, device=device, dtype=dtype)
        rids[..., 0] = frame_idx
        h_ids = torch.arange(rh, device=device, dtype=dtype)
        w_ids = torch.arange(rw, device=device, dtype=dtype)
        if scale_to_grid is not None:
            th, tw = scale_to_grid
            h_ids = (h_ids + 0.5) * (float(th) / float(rh)) - 0.5
            w_ids = (w_ids + 0.5) * (float(tw) / float(rw)) - 0.5
        if rope_offsets is not None and (i - 1) < len(rope_offsets):
            x_off, y_off = rope_offsets[i - 1]
            if x_off != 0 or y_off != 0:
                h_ids = h_ids + (float(y_off) / float(patch_size))
                w_ids = w_ids + (float(x_off) / float(patch_size))
        rids[..., 1] = h_ids[:, None]
        rids[..., 2] = w_ids[None, :]
        ref_positions.append(rids.reshape(1, rh * rw, 3).repeat(bs, 1, 1))

        total += rh * rw

    return ref_tokens, ref_positions, total


def krea2_edit_forward(self, x, timesteps, context, attention_mask=None,
                       transformer_options=None, ref_latents=None, **kwargs):
    """参考图感知的 SingleStreamDiT 前向（复刻 EditUtils _editutils_krea2_edit_forward）。"""
    from einops import rearrange
    from comfy.conds import CONDList

    if ref_latents is None:
        ref_list = []
    elif isinstance(ref_latents, torch.Tensor):
        ref_list = [ref_latents]
    elif isinstance(ref_latents, CONDList):
        ref_list = [r for r in ref_latents.cond if r is not None]
    elif isinstance(ref_latents, (list, tuple)):
        ref_list = [r for r in ref_latents if r is not None and isinstance(r, torch.Tensor)]
    else:
        ref_list = []

    mode = getattr(self, "_editutils_ref_timestep_mode", "editutils")

    if len(ref_list) == 0:
        original = getattr(self, _EDITUTILS_ORIGINAL_FORWARD_ATTR)
        return original(x, timesteps, context,
                        attention_mask=attention_mask,
                        transformer_options=transformer_options,
                        **kwargs)

    temporal = x.ndim == 5
    if temporal:
        b5, c5, t5, h5, w5 = x.shape
        x = x.reshape(b5 * t5, c5, h5, w5)

    bs, c, H_orig, W_orig = x.shape
    patch_size = self.patch
    import comfy.ldm.common_dit

    x_pad = comfy.ldm.common_dit.pad_to_patch_size(x, (patch_size, patch_size))
    H, W = x_pad.shape[-2], x_pad.shape[-1]
    h_grid, w_grid = H // patch_size, W // patch_size

    device = x_pad.device
    dtype = torch.float32
    pos_scale = (h_grid, w_grid) if getattr(
        self, "_editutils_ref_pos_match_target", False) else None
    ref_rope_offsets = kwargs.get("ref_rope_offsets", None)
    ref_tokens_list, ref_pos_list, total_reflen = make_ref_positions(
        ref_list, bs, patch_size, device, dtype, scale_to_grid=pos_scale,
        rope_offsets=ref_rope_offsets,
    )

    context = self._unpack_context(context)

    img = rearrange(x_pad, "b c (h ph) (w pw) -> b (h w) (c ph pw)",
                    ph=patch_size, pw=patch_size)
    img = self.first(img)

    ref_imgs = [self.first(rt) for rt in ref_tokens_list]

    from comfy.ldm.flux.layers import timestep_embedding

    t_emb_real = self.tmlp(
        timestep_embedding(timesteps, self.tdim).unsqueeze(1).to(img.dtype))
    tvec_real = self.tproj(t_emb_real)

    context = self.txtfusion(context, mask=None,
                             transformer_options=transformer_options or {})
    context = self.txtmlp(context)

    txtlen = context.shape[1]
    imglen = img.shape[1]

    combined = torch.cat([context, img] + ref_imgs, dim=1)

    txtpos = torch.zeros(bs, txtlen, 3, device=device, dtype=dtype)
    tgtids = torch.zeros(h_grid, w_grid, 3, device=device, dtype=dtype)
    tgtids[..., 1] = torch.arange(h_grid, device=device, dtype=dtype)[:, None]
    tgtids[..., 2] = torch.arange(w_grid, device=device, dtype=dtype)[None, :]
    tgtpos = tgtids.reshape(1, h_grid * w_grid, 3).repeat(bs, 1, 1)
    pos = torch.cat([txtpos, tgtpos] + ref_pos_list, dim=1)
    freqs = self.pe_embedder(pos)

    if mode == "krea2edit":
        tvec_ref = tvec_real.expand(-1, total_reflen, -1)
    else:
        t_emb_zero = self.tmlp(
            timestep_embedding(torch.zeros_like(timesteps), self.tdim).unsqueeze(1).to(img.dtype))
        tvec_ref = self.tproj(t_emb_zero).expand(-1, total_reflen, -1)
    tvec = torch.cat([tvec_real.expand(-1, txtlen, -1),
                      tvec_real.expand(-1, imglen, -1),
                      tvec_ref], dim=1)

    for block in self.blocks:
        combined = block(combined, tvec, freqs, attention_mask,
                         transformer_options=transformer_options or {})

    final = self.last(combined, t_emb_real)
    out = final[:, txtlen: txtlen + imglen, :]

    out = rearrange(out,
                    "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                    h=h_grid, w=w_grid, ph=patch_size, pw=patch_size, c=self.channels)
    out = out[:, :, :H_orig, :W_orig]

    if not torch.isfinite(out).all():
        raise ValueError("Krea2EditApply: edit forward produced NaN/Inf output")

    if temporal:
        out = out.reshape(b5, t5, self.channels, H_orig, W_orig).movedim(1, 2)

    return out


def ref_fingerprint(tensor):
    """参考 latent 的廉价内容指纹（memoize 在张量自身属性上，随张量销毁）。"""
    fp = getattr(tensor, "_editutils_fp", None)
    if fp is not None:
        return fp
    t = tensor.detach()
    if t.device.type != "cpu":
        t = t.to("cpu", dtype=torch.float32)
    else:
        t = t.float()
    flat = t.flatten()
    stride = max(1, flat.numel() // 64)
    sample = tuple(round(float(v), 4) for v in flat[::stride][:64])
    fp = (
        tuple(t.shape),
        round(float(t.mean()), 6),
        round(float(t.std()), 6),
        round(float(t.abs().sum()), 3),
        sample,
    )
    try:
        tensor._editutils_fp = fp
    except Exception:
        pass
    return fp


def build_cache_key(model_uuid, dtype, ref_list):
    return (
        model_uuid,
        str(dtype),
        tuple(ref_fingerprint(r) for r in ref_list),
    )


def make_ref_kv_attn_wrapper(attn_module, dit):
    """重写 Krea2 Attention.forward，加入 ref K/V 捕获/复用钩子。

    捕获：存 post qk-norm、post-RoPE、pre GQA repeat 的 ref 切片 k/v。
    复用：把缓存的 ref k/v 拼到新算出的 k/v 尾部。
    """
    from einops import rearrange
    from comfy.ldm.flux.math import apply_rope
    from comfy.ldm.modules.attention import optimized_attention_masked

    def forward(x, freqs=None, mask=None, transformer_options={}):
        q, k, v, gate = attn_module.wq(x), attn_module.wk(x), attn_module.wv(x), attn_module.gate(x)
        q = rearrange(q, "B L (H D) -> B H L D", H=attn_module.heads)
        k = rearrange(k, "B L (H D) -> B H L D", H=attn_module.kvheads)
        v = rearrange(v, "B L (H D) -> B H L D", H=attn_module.kvheads)
        q, k = attn_module.qknorm(q, k)
        if freqs is not None:
            q, k = apply_rope(q, k, freqs)

        state = getattr(dit, _KV_STATE_ATTR, None)
        if state is not None:
            layer = state["layer_counter"]
            state["layer_counter"] = layer + 1
            ref_toks = state["ref_toks"]
            if state["mode"] == "capture":
                state["store"][layer] = (
                    k[:, :, -ref_toks:].detach().to("cpu", copy=True),
                    v[:, :, -ref_toks:].detach().to("cpu", copy=True),
                )
            elif state["mode"] == "reuse" and not state.get("drop_refs"):
                kk, vv = state["store"][layer]
                if kk.device != k.device:
                    kk = kk.to(k.device, non_blocking=True)
                    vv = vv.to(v.device, non_blocking=True)
                kk = comfy.utils.repeat_to_batch_size(kk, k.shape[0])
                vv = comfy.utils.repeat_to_batch_size(vv, v.shape[0])
                k = torch.cat((k, kk), dim=2)
                v = torch.cat((v, vv), dim=2)

        if attn_module.kvheads != attn_module.heads:
            rep = attn_module.heads // attn_module.kvheads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        out = optimized_attention_masked(
            q, k, v, attn_module.heads, mask=mask, skip_reshape=True,
            transformer_options=transformer_options,
        )
        return attn_module.wo(out * torch.nn.functional.sigmoid(gate))

    return forward


class ref_kv_attn_patch_guard:
    """临时包裹所有 block 的 attention forward，退出时必定还原。"""

    def __init__(self, dit):
        self.dit = dit
        self.saved = []

    def __enter__(self):
        for block in self.dit.blocks:
            orig = block.attn.forward
            self.saved.append((block, orig))
            block.attn.forward = make_ref_kv_attn_wrapper(block.attn, self.dit)
        return self

    def __exit__(self, *exc):
        for block, orig in self.saved:
            block.attn.forward = orig
        self.saved = []
        return False


def krea2_edit_forward_cached(self, x, timesteps, context, attention_mask=None,
                              transformer_options=None, ref_latents=None, **kwargs):
    """KV 缓存版 edit forward（复刻 EditUtils _editutils_krea2_edit_forward_cached）。

    缓存未命中：完整前向 + 逐 block 捕获 ref K/V。
    缓存命中：仅对 [text | target] 前向，各 attention 内拼接缓存 ref K/V。
    异常回退：任意异常 → 非缓存 edit forward。
    """
    from einops import rearrange
    import comfy.ldm.common_dit
    from comfy.conds import CONDList
    from comfy.ldm.flux.layers import timestep_embedding

    if transformer_options is None:
        transformer_options = {}

    if ref_latents is None:
        ref_list = []
    elif isinstance(ref_latents, torch.Tensor):
        ref_list = [ref_latents]
    elif isinstance(ref_latents, CONDList):
        ref_list = [r for r in ref_latents.cond if r is not None]
    elif isinstance(ref_latents, (list, tuple)):
        ref_list = [r for r in ref_latents if r is not None and isinstance(r, torch.Tensor)]
    else:
        ref_list = []

    if len(ref_list) == 0:
        if not getattr(self, "_editutils_ref_kv_logged_norefs", False):
            setattr(self, "_editutils_ref_kv_logged_norefs", True)
            print(f"[SFKrea2EditApply] forward called but NO refs received "
                  f"(type={type(ref_latents).__name__}); chaining through.")
        chained = getattr(self, "_editutils_ref_kv_chained_forward")
        return chained(x, timesteps, context, attention_mask=attention_mask,
                       transformer_options=transformer_options, **kwargs)

    temporal = x.ndim == 5
    if temporal:
        b5, c5, t5, h5, w5 = x.shape
        x = x.reshape(b5 * t5, c5, h5, w5)
        if not getattr(self, "_editutils_ref_kv_logged_5d", False):
            setattr(self, "_editutils_ref_kv_logged_5d", True)
            print(f"[SFKrea2EditApply] 5D input detected "
                  f"(B={b5}, C={c5}, T={t5}, H={h5}, W={w5}); reshaping to 4D.")

    _cfg = getattr(self, _KV_CFG_ATTR, None) or {}
    debug = bool(_cfg.get("debug_log", False))
    ref_strength = float(_cfg.get("ref_strength", 1.0))

    if ref_strength <= 0.0:
        chained = getattr(self, "_editutils_ref_kv_chained_forward")
        return chained(x, timesteps, context, attention_mask=attention_mask,
                       transformer_options=transformer_options, **kwargs)

    cache = getattr(self, _KV_CACHE_ATTR, None)
    if cache is None:
        cache = {}
        setattr(self, _KV_CACHE_ATTR, cache)
    _rope_offsets_for_key = kwargs.get("ref_rope_offsets", None)
    key = build_cache_key(id(self), x.dtype, ref_list) + (
        tuple(x.shape),
        bool(getattr(self, "_editutils_ref_pos_match_target", False)),
        tuple(tuple(o) for o in _rope_offsets_for_key) if _rope_offsets_for_key else None,
    )
    entry = cache.get(key)
    reuse = entry is not None
    if debug and not getattr(self, "_editutils_ref_kv_logged_call", False):
        setattr(self, "_editutils_ref_kv_logged_call", True)
        print(f"[SFKrea2EditApply] KV cache active: {len(ref_list)} ref(s), "
              f"mode={'reuse' if reuse else 'capture'}")

    # 采样进度 [0,1]，来自完整 sigma 序列（"sample_sigmas"；"sigmas" 仅当前步）。
    progress = 0.0
    sig = transformer_options.get("sample_sigmas", None)
    if sig is None:
        sig = transformer_options.get("sigmas", None)
    if sig is not None:
        try:
            flat = sig.flatten().float()
            cur = transformer_options.get("sigmas", None)
            t_now = float(cur.flatten()[0]) if cur is not None \
                else float(timesteps.flatten()[0])
            idx = int((flat - t_now).abs().argmin())
            n = int(flat.numel())
            if n > 1 and float(flat[-1]) == 0.0:
                n -= 1
            progress = min(idx / max(n - 1, 1), 1.0)
        except Exception:
            progress = 0.0
    drop_refs = reuse and progress >= ref_strength
    if debug:
        alloc = torch.cuda.memory_allocated() // 2**20 if torch.cuda.is_available() else 0
        resv = torch.cuda.memory_reserved() // 2**20 if torch.cuda.is_available() else 0
        n_cache = len(getattr(self, _KV_CACHE_ATTR, {}) or {})
        print(f"[SFKrea2EditApply] step progress={progress:.2f} "
              f"ref_strength={ref_strength} -> {'DROP ref' if drop_refs else 'ref active'} "
              f"| VRAM alloc={alloc}MB reserved={resv}MB | cache_entries={n_cache}")
    if drop_refs and debug and not getattr(self, "_editutils_ref_kv_logged_drop", False):
        setattr(self, "_editutils_ref_kv_logged_drop", True)
        print(f"[SFKrea2EditApply] ref dropped at progress={progress:.2f} "
              f"(ref_strength={ref_strength}); continuing as text-to-image")

    bs, c, H_orig, W_orig = x.shape
    patch_size = self.patch
    x_pad = comfy.ldm.common_dit.pad_to_patch_size(x, (patch_size, patch_size))
    H, W = x_pad.shape[-2], x_pad.shape[-1]
    h_grid, w_grid = H // patch_size, W // patch_size
    device, dtype = x_pad.device, torch.float32

    pos_scale = (h_grid, w_grid) if getattr(
        self, "_editutils_ref_pos_match_target", False) else None
    ref_rope_offsets = kwargs.get("ref_rope_offsets", None)
    ref_tokens_list, ref_pos_list, total_reflen = make_ref_positions(
        ref_list, bs, patch_size, device, dtype, scale_to_grid=pos_scale,
        rope_offsets=ref_rope_offsets,
    )

    context = self._unpack_context(context)
    img = rearrange(x_pad, "b c (h ph) (w pw) -> b (h w) (c ph pw)",
                    ph=patch_size, pw=patch_size)
    img = self.first(img)

    t_emb_real = self.tmlp(
        timestep_embedding(timesteps, self.tdim).unsqueeze(1).to(img.dtype))
    tvec_real = self.tproj(t_emb_real)

    context = self.txtfusion(context, mask=None,
                             transformer_options=transformer_options)
    context = self.txtmlp(context)

    txtlen, imglen = context.shape[1], img.shape[1]

    txtpos = torch.zeros(bs, txtlen, 3, device=device, dtype=dtype)
    tgtids = torch.zeros(h_grid, w_grid, 3, device=device, dtype=dtype)
    tgtids[..., 1] = torch.arange(h_grid, device=device, dtype=dtype)[:, None]
    tgtids[..., 2] = torch.arange(w_grid, device=device, dtype=dtype)[None, :]
    tgtpos = tgtids.reshape(1, h_grid * w_grid, 3).repeat(bs, 1, 1)

    if reuse:
        combined = torch.cat([context, img], dim=1)
        pos = torch.cat([txtpos, tgtpos], dim=1)
        tvec = tvec_real.expand(-1, txtlen + imglen, -1)
    else:
        ref_imgs = [self.first(rt) for rt in ref_tokens_list]
        combined = torch.cat([context, img] + ref_imgs, dim=1)
        pos = torch.cat([txtpos, tgtpos] + ref_pos_list, dim=1)
        if getattr(self, "_editutils_ref_timestep_mode", "editutils") == "krea2edit":
            tvec_ref = tvec_real.expand(-1, total_reflen, -1)
        else:
            t_emb_zero = self.tmlp(
                timestep_embedding(torch.zeros_like(timesteps), self.tdim)
                .unsqueeze(1).to(img.dtype))
            tvec_ref = self.tproj(t_emb_zero).expand(-1, total_reflen, -1)
        tvec = torch.cat([
            tvec_real.expand(-1, txtlen, -1),
            tvec_real.expand(-1, imglen, -1),
            tvec_ref,
        ], dim=1)

    freqs = self.pe_embedder(pos)

    state = {
        "mode": "reuse" if reuse else "capture",
        "ref_toks": total_reflen,
        "layer_counter": 0,
        "store": entry["kv"] if reuse else {},
        "drop_refs": drop_refs,
    }
    setattr(self, _KV_STATE_ATTR, state)
    try:
        with ref_kv_attn_patch_guard(self):
            for block in self.blocks:
                combined = block(combined, tvec, freqs, attention_mask,
                                 transformer_options=transformer_options)
    finally:
        setattr(self, _KV_STATE_ATTR, None)

    if not reuse:
        cache[key] = {"kv": state["store"], "ref_toks": total_reflen, "hits": 0}
        if debug:
            cpu_mb = sum(t.nelement() * t.element_size() for kv in state["store"].values() for t in kv) // 2**20
            dev = next(iter(state["store"].values()))[0].device if state["store"] else "?"
            print(f"[SFKrea2EditApply] KV cache captured: {total_reflen} ref tokens, "
                  f"{len(state['store'])} layers, {cpu_mb}MB on {dev}")
        while len(cache) > 4:
            cache.pop(next(iter(cache)))
    elif debug and entry.get("hits", 0) == 0:
        print("[SFKrea2EditApply] reusing cached ref K/V")
    if reuse:
        entry["hits"] = entry.get("hits", 0) + 1

    final = self.last(combined, t_emb_real)
    out = final[:, txtlen: txtlen + imglen, :]
    out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                    h=h_grid, w=w_grid, ph=patch_size, pw=patch_size,
                    c=self.channels)
    out = out[:, :, :H_orig, :W_orig]

    if temporal:
        out = out.reshape(b5, t5, self.channels, H_orig, W_orig).movedim(1, 2)

    return out


def install_ref_kv_cache_patch(m, dit, reset_cache=False, ref_strength=1.0,
                               debug_log=False):
    """在已由 install_krea2_edit_patch 打过补丁的 ModelPatcher 上安装 KV 缓存 forward。"""
    if reset_cache:
        setattr(dit, _KV_CACHE_ATTR, {})

    setattr(dit, _KV_CFG_ATTR, {
        "debug_log": bool(debug_log),
        "ref_strength": float(ref_strength),
    })

    # 关键：链到已存在的对象补丁（Krea2EditApply 的分发 forward），而非 dit.forward。
    chained = m.object_patches.get("diffusion_model.forward", None)
    if chained is None:
        chained = dit.forward
    setattr(dit, "_editutils_ref_kv_chained_forward", chained)
    if debug_log:
        print("[SFKrea2EditApply] ref KV cache patch installed")

    def forward(x, timesteps, context, attention_mask=None,
                transformer_options=None, ref_latents=None, **kwargs):
        if transformer_options is None:
            transformer_options = {}
        try:
            import comfy.patcher_extension
            return comfy.patcher_extension.WrapperExecutor.new_class_executor(
                krea2_edit_forward_cached, dit,
                comfy.patcher_extension.get_all_wrappers(
                    comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                    transformer_options),
            ).execute(dit, x, timesteps, context,
                      attention_mask=attention_mask,
                      transformer_options=transformer_options,
                      ref_latents=ref_latents, **kwargs)
        except Exception as e:
            seen = getattr(dit, "_editutils_ref_kv_err_seen", None)
            if seen is None:
                seen = set()
                setattr(dit, "_editutils_ref_kv_err_seen", seen)
            msg = repr(e)
            if msg not in seen:
                seen.add(msg)
                import traceback
                print(f"[SFKrea2EditApply] KV cache forward failed, "
                      f"falling back to uncached: {msg}")
                traceback.print_exc()
            saved = getattr(dit, "_editutils_ref_kv_chained_forward")
            return saved(x, timesteps, context,
                         attention_mask=attention_mask,
                         transformer_options=transformer_options,
                         ref_latents=ref_latents, **kwargs)

    m.add_object_patch("diffusion_model.forward", forward)


def apply_krea2_edit_patch(model_patcher):
    """安装参考图 edit forward 补丁（复刻 Krea2EditApply._apply_krea2_edit_patch）。

    注册三个对象补丁：
      - extra_conds：从 conditioning 提取 reference_latents（EditUtils 约定）
      - extra_conds_shapes：声明显存分配形状
      - diffusion_model.forward：参考图感知分发（无 ref 时回退原 forward）
    """
    import comfy.conds

    base_model = model_patcher.model
    dit = model_patcher.get_model_object("diffusion_model")

    orig_extra_conds = base_model.extra_conds
    orig_extra_conds_shapes = base_model.extra_conds_shapes
    orig_forward = dit.forward

    if not hasattr(dit, _EDITUTILS_ORIGINAL_FORWARD_ATTR):
        setattr(dit, _EDITUTILS_ORIGINAL_FORWARD_ATTR, orig_forward)

    def extra_conds(**kwargs):
        out = orig_extra_conds(**kwargs)
        cond_refs = kwargs.get("reference_latents", None)
        rope_offsets = kwargs.get("reference_rope_offsets", None)
        if cond_refs is not None and len(cond_refs) > 0:
            ref_tensors = [
                r["samples"] if isinstance(r, dict) else r
                for r in cond_refs
            ]
            out["ref_latents"] = comfy.conds.CONDList([
                base_model.process_latent_in(t) for t in ref_tensors
            ])
            if rope_offsets is not None:
                out["ref_rope_offsets"] = comfy.conds.CONDConstant(rope_offsets)
        return out

    def extra_conds_shapes(**kwargs):
        out = orig_extra_conds_shapes(**kwargs)
        cond_refs = kwargs.get("reference_latents", None)
        if cond_refs is not None and len(cond_refs) > 0:
            ref_tensors = [
                r["samples"] if isinstance(r, dict) else r
                for r in cond_refs
            ]
            out["ref_latents"] = list(
                [1, dit.channels,
                 sum(math.prod(t.size()[2:]) for t in ref_tensors) // dit.channels]
            )
        return out

    def forward(x, timesteps, context, attention_mask=None,
                transformer_options=None, ref_latents=None, **kwargs):
        if transformer_options is None:
            transformer_options = {}

        if ref_latents is None or len(ref_latents) == 0:
            saved = getattr(dit, _EDITUTILS_ORIGINAL_FORWARD_ATTR)
            return saved(
                x, timesteps, context,
                attention_mask=attention_mask,
                transformer_options=transformer_options,
                **kwargs,
            )
        import comfy.patcher_extension
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            krea2_edit_forward, dit,
            comfy.patcher_extension.get_all_wrappers(
                comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options),
        ).execute(dit, x, timesteps, context,
                  attention_mask=attention_mask,
                  transformer_options=transformer_options,
                  ref_latents=ref_latents,
                  **kwargs)

    model_patcher.add_object_patch("extra_conds", extra_conds)
    model_patcher.add_object_patch("extra_conds_shapes", extra_conds_shapes)
    model_patcher.add_object_patch("memory_usage_factor_conds", ("ref_latents",))
    model_patcher.add_object_patch("diffusion_model.forward", forward)


def is_krea2_model(model):
    """按 _unpack_context 属性探测 Krea2 模型（复刻 EditUtils 的探测方式）。"""
    return (
        hasattr(model, "model")
        and hasattr(model.model, "diffusion_model")
        and hasattr(model.model.diffusion_model, "_unpack_context")
    )
