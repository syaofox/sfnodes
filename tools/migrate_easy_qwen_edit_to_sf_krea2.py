#!/usr/bin/env python3
"""把工作流里的 Easy_QwenEdit2509 编码节点替换为 sfnodes 的 Krea2 编码链。

适用场景：Krea2 工作流把 Easy_QwenEdit2509 当作"编码器"用（VLM 文本条件 +
reference_latents + 初始 latent + 零负条件），可用纯 sfnodes 节点替代：

    Easy_QwenEdit2509 ──> SFKrea2ModelConfig + SFKrea2ConfigPreparer
                          + SFKrea2EditTextEncode + ConditioningZeroOut

等价性依据（同源算法 + 同参数）见 doc/experience/nodes-lora.md §151。
脚本把 `ref_longest_edge` 自动换算为 `round(1024 * sqrt(megapixels))`，
从 image1 上游的 ImageScaleToTotalPixels 读取 megapixels（或用 --mp/--edge 覆盖），
以此抵消 1.5MP 缩放，保持参考/初始 latent 与 Easy 同尺寸——改图源分辨率后重跑即可。

用法：
    python3 migrate_easy_qwen_edit_to_sf_krea2.py <workflow.json> [--out OUT]
                                                 [--mp 1.5] [--edge 1254] [--dry-run]

输出默认写到同目录 `<原名>(SF-Krea2版).json`，不改动原文件。
仅支持 Easy 的 image1 与 latent_image 同一来源（该场景 auto_resize 为空操作，
strength=1）；其它拓扑会报错，不猜测映射。
"""

import argparse
import json
import math
import sys
import uuid
from pathlib import Path

EASY_TYPE = "Easy_QwenEdit2509"

# Easy 的默认 system_prompt（widget 默认值）。widget 清空时 Easy 内部回退到同一文本；
# 而 SFKrea2ModelConfig 的空 instruction 会回退到 Krea2 描述指令，因此需显式补文本。
EASY_DEFAULT_SYSTEM_PROMPT = (
    "Describe the key features of the input image (color, shape, size, texture, "
    "objects, background), then explain how the user's text instruction should "
    "alter or modify the image. Generate a new image that meets the user's "
    "requirements while maintaining consistency with the original input where "
    "appropriate."
)

# SFKrea2ConfigPreparer 的 widget 顺序（与 INPUT_TYPES / 前端序列化一致）
PREPARER_WIDGET_ORDER = [
    "to_ref", "ref_main_image", "ref_longest_edge", "ref_crop", "ref_upscale",
    "to_vl", "vl_resize", "vl_target_size", "vl_crop", "vl_upscale",
    "ref_resize_mode", "rope_x_offset", "rope_y_offset",
]


def _conn(name, type_name, optional=False, link=None):
    d = {"localized_name": name, "name": name, "type": type_name, "link": link}
    if optional:
        d["shape"] = 7
    return d


def _widget(name, type_name, optional=True):
    d = {"localized_name": name, "name": name, "type": type_name,
         "widget": {"name": name}, "link": None}
    if optional:
        d["shape"] = 7
    return d


def _out(name, type_name):
    return {"localized_name": name, "name": name, "type": type_name, "links": []}


def make_model_config(nid, order, pos, instruction):
    return {
        "id": nid, "type": "SFKrea2ModelConfig", "pos": pos, "size": [380, 120],
        "flags": {}, "order": order, "mode": 0,
        "inputs": [_widget("instruction", "STRING")],
        "outputs": [_out("model_config", "DICT")],
        "properties": {"cnr_id": "sfnodes", "Node name for S&R": "SFKrea2ModelConfig"},
        "widgets_values": [instruction],
        "widgets_values_named": {"instruction": instruction},
    }


def make_preparer(nid, order, pos, edge, vl_target, main=True):
    values = {
        "to_ref": True, "ref_main_image": main, "ref_longest_edge": edge,
        "ref_crop": "disabled", "ref_upscale": "bicubic",
        "to_vl": True, "vl_resize": True, "vl_target_size": vl_target,
        "vl_crop": "disabled", "vl_upscale": "area",
        "ref_resize_mode": "area", "rope_x_offset": 0, "rope_y_offset": 0,
    }
    inputs = [
        _conn("image", "IMAGE"),
        _conn("configs", "LIST", optional=True),
        _conn("mask", "MASK", optional=True),
    ] + [_widget(k, "BOOLEAN" if isinstance(values[k], bool) else
                 ("INT" if isinstance(values[k], int) else "COMBO"))
         for k in PREPARER_WIDGET_ORDER]
    return {
        "id": nid, "type": "SFKrea2ConfigPreparer", "pos": pos, "size": [365, 400],
        "flags": {}, "order": order, "mode": 0,
        "inputs": inputs,
        "outputs": [_out("configs", "LIST"), _out("config", "ANY")],
        "properties": {"cnr_id": "sfnodes", "Node name for S&R": "SFKrea2ConfigPreparer"},
        "widgets_values": [values[k] for k in PREPARER_WIDGET_ORDER],
        "widgets_values_named": values,
    }


def make_encoder(nid, order, pos, prompt):
    return {
        "id": nid, "type": "SFKrea2EditTextEncode", "pos": pos, "size": [430, 300],
        "flags": {}, "order": order, "mode": 0,
        "inputs": [
            _conn("clip", "CLIP"),
            _conn("vae", "VAE"),
            _conn("model_config", "DICT"),
            _conn("configs", "LIST", optional=True),
            _widget("prompt", "STRING", optional=False),
        ],
        "outputs": [
            _out("conditioning", "CONDITIONING"), _out("latent", "LATENT"),
            _out("custom_output", "ANY"), _out("main_image", "IMAGE"),
            _out("mask", "MASK"), _out("pad_info", "ANY"),
        ],
        "properties": {"cnr_id": "sfnodes", "Node name for S&R": "SFKrea2EditTextEncode"},
        "widgets_values": [prompt],
        "widgets_values_named": {"prompt": prompt},
    }


def make_zero_out(nid, order, pos):
    return {
        "id": nid, "type": "ConditioningZeroOut", "pos": pos, "size": [210, 30],
        "flags": {}, "order": order, "mode": 0,
        "inputs": [_conn("conditioning", "CONDITIONING")],
        "outputs": [_out("CONDITIONING", "CONDITIONING")],
        "properties": {"cnr_id": "comfy-core", "Node name for S&R": "ConditioningZeroOut"},
        "widgets_values": [],
    }


def next_id(data):
    data["last_node_id"] = max(
        data.get("last_node_id", 0), max((n["id"] for n in data["nodes"]), default=0)
    ) + 1
    return data["last_node_id"]


def _next_link_id(data):
    data["last_link_id"] = max(
        data.get("last_link_id", 0), max((l[0] for l in data["links"]), default=0)
    ) + 1
    return data["last_link_id"]


def add_link(data, src, src_slot, dst, dst_slot, link_type):
    by_id = {n["id"]: n for n in data["nodes"]}
    lid = _next_link_id(data)
    data["links"].append([lid, src, src_slot, dst, dst_slot, link_type])
    out = by_id[src]["outputs"][src_slot]
    if not out.get("links"):
        out["links"] = []
    out["links"].append(lid)
    by_id[dst]["inputs"][dst_slot]["link"] = lid
    return lid


def disconnect_node(data, node):
    """摘除节点并清理其全部连线（双向）。"""
    nid = node["id"]
    by_id = {n["id"]: n for n in data["nodes"]}
    kept = []
    for link in data["links"]:
        lid, src, sslot, dst, dslot = link[:5]
        if src != nid and dst != nid:
            kept.append(link)
            continue
        outs = by_id[src].get("outputs") or []
        if sslot < len(outs):
            outs[sslot]["links"] = [x for x in (outs[sslot].get("links") or []) if x != lid]
        ins = by_id[dst].get("inputs") or []
        if dslot < len(ins) and ins[dslot].get("link") == lid:
            ins[dslot]["link"] = None
    data["links"] = kept
    data["nodes"] = [n for n in data["nodes"] if n["id"] != nid]


def check_graph(data):
    by_id = {n["id"]: n for n in data["nodes"]}
    link_ids = set()
    for link in data["links"]:
        lid, src, sslot, dst, dslot = link[:5]
        if lid in link_ids:
            raise ValueError(f"重复 link id {lid}")
        link_ids.add(lid)
        if src not in by_id or dst not in by_id:
            raise ValueError(f"link {lid} 指向不存在的节点")
        outs = by_id[src].get("outputs") or []
        if sslot >= len(outs) or lid not in (outs[sslot].get("links") or []):
            raise ValueError(f"link {lid} 未挂到源节点输出")
        ins = by_id[dst].get("inputs") or []
        if dslot >= len(ins) or ins[dslot].get("link") != lid:
            raise ValueError(f"link {lid} 未挂到目标节点输入")
    for n in data["nodes"]:
        for i, inp in enumerate(n.get("inputs") or []):
            if inp.get("link") is not None and inp["link"] not in link_ids:
                raise ValueError(f"节点 {n['id']} 输入 {inp.get('name')} 指向不存在的 link")
        for out in n.get("outputs") or []:
            for lid in out.get("links") or []:
                if lid not in link_ids:
                    raise ValueError(f"节点 {n['id']} 输出 {out.get('name')} 指向不存在的 link")


def fail(msg):
    print("错误：" + msg, file=sys.stderr)
    sys.exit(2)


def main():
    ap = argparse.ArgumentParser(description="Easy_QwenEdit2509 → sfnodes Krea2 编码链迁移")
    ap.add_argument("workflow", help="输入工作流 JSON 路径（需包含 Easy_QwenEdit2509）")
    ap.add_argument("--out", default=None, help="输出路径（默认同目录 <原名>(SF-Krea2版).json）")
    ap.add_argument("--mp", type=float, default=None,
                    help="覆盖 megapixels（换算 ref_longest_edge=round(1024*sqrt(mp)))")
    ap.add_argument("--edge", type=int, default=None, help="直接指定 ref_longest_edge")
    ap.add_argument("--dry-run", action="store_true", help="只校验与打印，不写文件")
    args = ap.parse_args()

    src_path = Path(args.workflow)
    if not src_path.is_file():
        fail(f"工作流不存在：{src_path}")
    data = json.loads(src_path.read_text(encoding="utf-8"))

    easies = [n for n in data.get("nodes", []) if n.get("type") == EASY_TYPE]
    if not easies:
        fail(f"未找到 {EASY_TYPE}")
    if len(easies) > 1:
        fail(f"找到 {len(easies)} 个 {EASY_TYPE}（id={[n['id'] for n in easies]}），请先拆分为多个工作流")

    easy = easies[0]
    node_by_id = {n["id"]: n for n in data["nodes"]}
    link_by_id = {l[0]: l for l in data["links"]}

    named = easy.get("widgets_values_named") or {}
    wv = easy.get("widgets_values") or []

    def widget(name, idx, default=None):
        if name in named:
            return named[name]
        return wv[idx] if idx < len(wv) else default

    # auto_resize 仅在 image1 与 latent_image 尺寸不一致时生效；本脚本只支持同源，忽略
    prompt = widget("prompt", 2, "") or ""
    system_prompt = widget("system_prompt", 3, "") or ""
    vl_size = int(widget("vl_size", 1, 384))
    strengths = [float(widget(f"image{i}_strength", 3 + i, 1.0)) for i in (1, 2, 3)]

    if any(abs(s - 1.0) > 1e-6 for s in strengths):
        print("警告：Easy 的 image strength 非 1，sfnodes 链无此能力，已忽略（输出会与 Easy 不同）")

    links_in = {i["name"]: (idx, i["link"]) for idx, i in enumerate(easy.get("inputs") or [])
                if i.get("link") is not None}

    def source_of(link_id):
        link = link_by_id[link_id]
        return link[1], link[2]

    if "image1" not in links_in:
        fail("image1 未连接")
    if "latent_image" not in links_in:
        fail("latent_image 未连接")
    if source_of(links_in["image1"][1]) != source_of(links_in["latent_image"][1]):
        fail("image1 与 latent_image 不是同一来源；本脚本仅支持同源（此时 auto_resize 为空操作）。"
             "独立 latent_image 场景需手工改用 core VAEEncode + SetLatentNoiseMask")
    if "clip" not in links_in or "vae" not in links_in:
        fail("clip/vae 未连接")
    for name in ("auto_resize", "vl_size"):
        if name in links_in:
            print(f"警告：{name} 为前端接线输入，sfnodes 链只能取 widget 值，已忽略接线")

    edge = args.edge
    if edge is None:
        mp = args.mp
        if mp is None:
            src_id, _ = source_of(links_in["image1"][1])
            src_node = node_by_id[src_id]
            if src_node.get("type") == "ImageScaleToTotalPixels":
                sn = src_node.get("widgets_values_named") or {}
                if "megapixels" in sn:
                    mp = float(sn["megapixels"])
                else:
                    sv = src_node.get("widgets_values") or []
                    if len(sv) >= 2:
                        mp = float(sv[1])
            if mp is None:
                fail("无法从 image1 上游读取 megapixels，请用 --mp 或 --edge 指定")
        edge = round(1024 * math.sqrt(mp))
    if not 8 <= edge <= 4096:
        fail(f"ref_longest_edge={edge} 超出 [8, 4096]")

    vl_target = max(384, min(2048, vl_size))
    if vl_target != vl_size:
        print(f"警告：vl_size={vl_size} 超范围，已夹到 {vl_target}（sfnodes 链最小 384）")

    instruction = system_prompt if system_prompt else EASY_DEFAULT_SYSTEM_PROMPT

    def targets(slot):
        outs = easy.get("outputs") or []
        if slot >= len(outs):
            return []
        return [link_by_id[lid] for lid in (outs[slot].get("links") or []) if lid in link_by_id]

    positive_targets = targets(0)
    negative_targets = targets(1)
    latent_targets = targets(2)
    if not latent_targets:
        print("警告：Easy 的 latent 输出未连接；迁移后 SFKrea2EditTextEncode.latent 将悬空")

    extra_names = [f"image{i}" for i in (2, 3) if f"image{i}" in links_in]
    extra_images = [links_in[name][1] for name in extra_names]
    if extra_images:
        print(f"警告：检测到 {'/'.join(extra_names)} 参考图，"
              "sfnodes 链按 area 归一化到同像素面积（Easy 是 crop/pad/stretch 到 latent_image 尺寸），"
              "尺寸/取景不完全一致")

    ex, ey = easy["pos"]
    base_order = max([n.get("order") or 0 for n in data["nodes"]] + [0])
    cfg_id, prep_id, enc_id, zero_id = (next_id(data), next_id(data), next_id(data), next_id(data))
    prep_ids = [prep_id]
    for _ in extra_images:
        prep_ids.append(next_id(data))

    new_nodes = [
        make_model_config(cfg_id, base_order + 1, [ex - 560, ey], instruction),
        make_preparer(prep_id, base_order + 2, [ex, ey], edge, vl_target, main=True),
        make_encoder(enc_id, base_order + 3 + len(extra_images), [ex + 380, ey + 100], prompt),
        make_zero_out(zero_id, base_order + 4 + len(extra_images), [ex + 780, ey + 100]),
    ]
    for k, pid in enumerate(prep_ids[1:], start=1):
        new_nodes.insert(1 + k, make_preparer(pid, base_order + 2 + k,
                                              [ex, ey + 440 * k], edge, vl_target, main=False))

    disconnect_node(data, easy)
    data["nodes"].extend(new_nodes)

    # 输入源接线
    img_src, img_slot = source_of(links_in["image1"][1])
    add_link(data, img_src, img_slot, prep_id, 0, "IMAGE")
    for k, link_id in enumerate(extra_images, start=1):
        s, sslot = source_of(link_id)
        add_link(data, s, sslot, prep_ids[k], 0, "IMAGE")
    clip_src, clip_slot = source_of(links_in["clip"][1])
    add_link(data, clip_src, clip_slot, enc_id, 0, "CLIP")
    vae_src, vae_slot = source_of(links_in["vae"][1])
    add_link(data, vae_src, vae_slot, enc_id, 1, "VAE")
    if "system_prompt" in links_in:
        s, sslot = source_of(links_in["system_prompt"][1])
        add_link(data, s, sslot, cfg_id, 0, "STRING")
    if "mask" in links_in or "latent_mask" in links_in:
        mask_link = links_in.get("latent_mask") or links_in.get("mask")
        s, sslot = source_of(mask_link[1])
        add_link(data, s, sslot, prep_id, 2, "MASK")
    if "prompt" in links_in:
        s, sslot = source_of(links_in["prompt"][1])
        add_link(data, s, sslot, enc_id, 4, "STRING")

    # 链内接线
    add_link(data, cfg_id, 0, enc_id, 2, "DICT")
    add_link(data, prep_id, 0, enc_id, 3, "LIST")
    for k in range(1, len(prep_ids)):
        add_link(data, prep_ids[k - 1], 0, prep_ids[k], 1, "LIST")

    # 输出接线（沿用 Easy 的消费端）
    add_link(data, enc_id, 0, zero_id, 0, "CONDITIONING")
    for link in positive_targets:
        add_link(data, enc_id, 0, link[3], link[4], link[5])
    for link in negative_targets:
        add_link(data, zero_id, 0, link[3], link[4], link[5])
    for link in latent_targets:
        add_link(data, enc_id, 1, link[3], link[4], link[5])

    check_graph(data)

    out_path = Path(args.out) if args.out else src_path.with_name(
        src_path.stem + "(SF-Krea2版)" + src_path.suffix)
    if out_path.resolve() == src_path.resolve():
        fail("输出路径与输入相同，拒绝覆盖原工作流")
    if args.dry_run:
        print(f"[dry-run] 校验通过；将写入 {out_path}")
    else:
        payload = dict(data)
        payload["id"] = str(uuid.uuid4())
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                            encoding="utf-8")
        print(f"已生成：{out_path}")
    print(f"  ref_longest_edge={edge}（area 模式，抵消 {edge * edge / (1024 * 1024):.3f}MP 缩放），"
          f"vl_target_size={vl_target}，参考图 {1 + len(extra_images)} 张")
    print(f"  替换：{EASY_TYPE} → SFKrea2ModelConfig + SFKrea2ConfigPreparer×{len(prep_ids)} "
          f"+ SFKrea2EditTextEncode + ConditioningZeroOut")
    print("  请在前端硬刷新后打开对比（seed 固定时输出应一致）")


if __name__ == "__main__":
    main()
