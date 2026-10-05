# tools/migrate_easy_qwen_edit_to_sf_krea2.py 回归测试（python3 tests/test_migrate_easy_krea2.py）
# 覆盖：
#   - 最小工作流替换：Easy 移除、4 节点接入、9 条新链接、output 全保留
#   - ref_longest_edge 换算：自动读上游 megapixels / --mp / --edge 覆盖
#   - 原文件不动、--dry-run 不落盘
#   - 无 Easy、image1 与 latent_image 不同源时非零退出（不猜测映射）
import json
import os
import subprocess
import sys
import tempfile

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(BASE, "tools", "migrate_easy_qwen_edit_to_sf_krea2.py")

failures = []


def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")


def _inp(name, type_, link=None, widget=False, optional=False, shape=False):
    d = {"localized_name": name, "name": name, "type": type_, "link": link}
    if shape or optional:
        d["shape"] = 7
    if widget:
        d["widget"] = {"name": name}
    return d


def _out(name, type_, links=None):
    return {"localized_name": name, "name": name, "type": type_, "links": links or []}


def _node(nid, ntype, inputs=None, outputs=None, widgets=None, named=None):
    n = {"id": nid, "type": ntype, "pos": [nid * 100, nid * 50], "size": [200, 100],
         "flags": {}, "order": nid, "mode": 0, "inputs": inputs or [], "outputs": outputs or [],
         "properties": {}, "widgets_values": widgets if widgets is not None else []}
    if named is not None:
        n["widgets_values_named"] = named
    return n


def easy_node():
    inputs = [
        _inp("clip", "CLIP", 11), _inp("vae", "VAE", 12),
        _inp("image1", "IMAGE", 10, optional=True), _inp("image2", "IMAGE", optional=True),
        _inp("image3", "IMAGE", optional=True), _inp("latent_image", "IMAGE", 13, optional=True),
        _inp("latent_mask", "MASK", optional=True),
        _inp("auto_resize", "COMBO", widget=True, optional=True),
        _inp("vl_size", "INT", widget=True, optional=True),
        _inp("prompt", "STRING", widget=True, optional=True),
        _inp("system_prompt", "STRING", widget=True, optional=True),
        _inp("image1_strength", "FLOAT", widget=True, optional=True),
        _inp("image2_strength", "FLOAT", widget=True, optional=True),
        _inp("image3_strength", "FLOAT", widget=True, optional=True),
    ]
    named = {"auto_resize": "crop", "vl_size": 384, "prompt": "make it real",
             "system_prompt": "", "image1_strength": 1, "image2_strength": 1, "image3_strength": 1}
    return _node(4, "Easy_QwenEdit2509", inputs,
                 [_out("positive", "CONDITIONING", [14]), _out("zero_negative", "CONDITIONING", [15]),
                  _out("latent", "LATENT", [16])],
                 ["crop", 384, "make it real", "", 1, 1, 1], named)


def workflow(image_src=1, latent_src=1):
    nodes = [
        _node(1, "ImageScaleToTotalPixels",
              [_inp("image", "IMAGE"), _inp("upscale_method", "COMBO", widget=True),
               _inp("megapixels", "FLOAT", widget=True), _inp("resolution_steps", "INT", widget=True)],
              [_out("IMAGE", "IMAGE", [10, 13])],
              ["lanczos", 1.5, 8], {"upscale_method": "lanczos", "megapixels": 1.5, "resolution_steps": 8}),
        _node(2, "CLIPLoader", [], [_out("CLIP", "CLIP", [11])]),
        _node(3, "VAELoader", [], [_out("VAE", "VAE", [12])]),
        easy_node(),
        _node(5, "FakeCond", [_inp("conditioning", "CONDITIONING", 14)], []),
        _node(6, "FakeCond", [_inp("conditioning", "CONDITIONING", 15)], []),
        _node(7, "FakeSampler", [_inp("latent", "LATENT", 16)], []),
    ]
    links = [[10, image_src, 0, 4, 2, "IMAGE"], [13, latent_src, 0, 4, 5, "IMAGE"],
             [11, 2, 0, 4, 0, "CLIP"], [12, 3, 0, 4, 1, "VAE"],
             [14, 4, 0, 5, 0, "CONDITIONING"], [15, 4, 1, 6, 0, "CONDITIONING"],
             [16, 4, 2, 7, 0, "LATENT"]]
    return {"id": "test-wf", "revision": 0, "last_node_id": 7, "last_link_id": 16,
            "nodes": nodes, "links": links, "groups": [], "config": {}, "extra": {}, "version": 0.4}


def run(fixture, out, *args):
    return subprocess.run([sys.executable, SCRIPT, fixture, "--out", out, *args],
                          capture_output=True, text=True)


def load(path):
    d = json.load(open(path, encoding="utf-8"))
    return d, {n["id"]: n for n in d["nodes"]}, {l[0]: l for l in d["links"]}


with tempfile.TemporaryDirectory() as td:
    src = os.path.join(td, "wf.json")
    out = os.path.join(td, "wf_out.json")
    with open(src, "w", encoding="utf-8") as f:
        json.dump(workflow(), f, ensure_ascii=False)
    before = open(src, "rb").read()

    r = run(src, out)
    check("退出码 0", r.returncode == 0)
    check("原文件未改动", open(src, "rb").read() == before)
    d, nby, lby = load(out)
    types = {n["type"] for n in d["nodes"]}
    check("Easy 已移除", "Easy_QwenEdit2509" not in types)
    check("新节点齐全", {"SFKrea2ModelConfig", "SFKrea2ConfigPreparer",
                        "SFKrea2EditTextEncode", "ConditioningZeroOut"} <= types)
    check("workflow id 重新生成", d["id"] != "test-wf")

    prep = [n for n in d["nodes"] if n["type"] == "SFKrea2ConfigPreparer"][0]
    enc = [n for n in d["nodes"] if n["type"] == "SFKrea2EditTextEncode"][0]
    zero = [n for n in d["nodes"] if n["type"] == "ConditioningZeroOut"][0]
    cfg = [n for n in d["nodes"] if n["type"] == "SFKrea2ModelConfig"][0]
    named = prep["widgets_values_named"]
    check("edge=round(1024*sqrt(1.5))", named["ref_longest_edge"] == 1254)
    check("area 模式 + 与 Easy 同款缩放参数",
          (named["ref_resize_mode"], named["ref_crop"], named["ref_upscale"],
           named["vl_crop"], named["vl_upscale"]) == ("area", "disabled", "bicubic", "disabled", "area"))
    check("system_prompt 空值补 Easy 默认文本",
          cfg["widgets_values_named"]["instruction"].startswith("Describe the key features"))
    check("prompt 透传", enc["widgets_values"] == ["make it real"])

    def link_of(src_id, sslot, dst_id, dslot):
        return [l for l in d["links"] if l[1] == src_id and l[2] == sslot and l[3] == dst_id and l[4] == dslot]

    check("image → Preparer.image", link_of(1, 0, prep["id"], 0))
    check("CLIP/VAE → Encode", link_of(2, 0, enc["id"], 0) and link_of(3, 0, enc["id"], 1))
    check("ModelConfig → Encode.model_config", link_of(cfg["id"], 0, enc["id"], 2))
    check("Preparer → Encode.configs", link_of(prep["id"], 0, enc["id"], 3))
    check("positive 消费端保留（conditioning）", link_of(enc["id"], 0, 5, 0))
    check("negative 经 ZeroOut 保留", link_of(enc["id"], 0, zero["id"], 0) and link_of(zero["id"], 0, 6, 0))
    check("latent 消费端保留", link_of(enc["id"], 1, 7, 0))
    check("消费端 link 指向新链接",
          nby[5]["inputs"][0]["link"] is not None and nby[5]["inputs"][0]["link"] in lby)
    check("下游 FakeSampler 保留", nby[7]["type"] == "FakeSampler")

    # --mp / --edge 覆盖
    out2 = os.path.join(td, "out2.json")
    r2 = run(src, out2, "--mp", "1.0")
    _, _, _ = load(out2)
    p2 = [n for n in json.load(open(out2, encoding="utf-8"))["nodes"]
          if n["type"] == "SFKrea2ConfigPreparer"][0]
    check("--mp 1.0 → edge 1024", r2.returncode == 0 and p2["widgets_values_named"]["ref_longest_edge"] == 1024)
    out3 = os.path.join(td, "out3.json")
    r3 = run(src, out3, "--edge", "640")
    p3 = [n for n in json.load(open(out3, encoding="utf-8"))["nodes"]
          if n["type"] == "SFKrea2ConfigPreparer"][0]
    check("--edge 640 生效", r3.returncode == 0 and p3["widgets_values_named"]["ref_longest_edge"] == 640)

    # --dry-run 不落盘
    out4 = os.path.join(td, "out4.json")
    r4 = run(src, out4, "--dry-run")
    check("--dry-run 退出 0 且不写文件", r4.returncode == 0 and not os.path.exists(out4))

    # 无 Easy
    no_easy = os.path.join(td, "no_easy.json")
    wf = workflow()
    wf["nodes"] = [n for n in wf["nodes"] if n["type"] != "Easy_QwenEdit2509"]
    json.dump(wf, open(no_easy, "w", encoding="utf-8"))
    r5 = run(no_easy, os.path.join(td, "out5.json"))
    check("无 Easy 时非零退出", r5.returncode == 2)

    # image1 与 latent_image 不同源
    diff = os.path.join(td, "diff.json")
    wf2 = workflow(image_src=1, latent_src=2)
    # 让 node2 也输出 IMAGE，供 latent_image 链接
    wf2["nodes"][1]["outputs"].append(_out("IMAGE", "IMAGE", [13]))
    wf2["links"] = [l for l in wf2["links"] if l[0] != 13] + [[13, 2, 1, 4, 5, "IMAGE"]]
    json.dump(wf2, open(diff, "w", encoding="utf-8"))
    r6 = run(diff, os.path.join(td, "out6.json"))
    check("image1≠latent_image 非零退出", r6.returncode == 2)

    # 默认输出名
    r7 = subprocess.run([sys.executable, SCRIPT, src], capture_output=True, text=True)
    check("默认输出名带 (SF-Krea2版)", r7.returncode == 0 and
          os.path.exists(os.path.join(td, "wf(SF-Krea2版).json")))

print()
if failures:
    print(f"FAILED: {len(failures)} 项 -> {failures}")
    sys.exit(1)
print("ALL PASSED")
