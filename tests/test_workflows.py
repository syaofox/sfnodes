# SF Workflows 后端测试（Node/Python 直接运行：python tests/test_workflows.py）
# 覆盖：summarize_workflow（各字段/容错/封面映射/指纹）、build_index（缓存命中/
# 重解析）、detect_issues、collections、looks_like_image/is_cover_name/reserved_part、
# 路由的 _wf_resolve/_wf_cover_name（mock folder_paths）
import importlib.util
import json
import os
import sys
import tempfile
import time
import types

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

tmp_root = tempfile.mkdtemp(prefix="sf_wf_test_")

folder_paths = types.ModuleType("folder_paths")
folder_paths.get_user_directory = lambda: os.path.join(tmp_root, "user")
sys.modules["folder_paths"] = folder_paths

# mock aiohttp（路由模块 import 用；测试不真正调用 web 响应构造）
aiohttp = types.ModuleType("aiohttp")
web_mock = types.ModuleType("aiohttp.web")
web_mock.json_response = lambda *a, **k: None
web_mock.Response = lambda *a, **k: None
web_mock.FileResponse = lambda *a, **k: None
aiohttp.web = web_mock
sys.modules["aiohttp"] = aiohttp
sys.modules["aiohttp.web"] = web_mock

spec = importlib.util.spec_from_file_location(
    "sfnodes.sf_utils.workflow_index_helpers",
    os.path.join(root, "sf_utils", "workflow_index_helpers.py"),
)
H = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = H
spec.loader.exec_module(H)

failures = []

def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")

# ── 工作流目录准备 ──
wf_root = os.path.join(tmp_root, "user", "workflows")
os.makedirs(os.path.join(wf_root, "sub"), exist_ok=True)
os.makedirs(os.path.join(tmp_root, "user", "default"), exist_ok=True)  # meta sidecar 目录

def write_wf(rel, nodes, name=None):
    path = os.path.join(wf_root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {"nodes": nodes}
    if name:
        data["name"] = name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return path

write_wf("simple.json", [
    {"type": "KSampler", "widgets_values": ["flux1-dev.safetensors", "model"], "pos": [0, 0], "size": [200, 80], "color": "#1d1d1d"},
    {"type": "CLIPTextEncode", "widgets_values": ["a red fox in the snow"], "pos": [100, 50], "size": [150, 70]},
    {"type": "Note", "widgets_values": ["just a note"], "pos": [0, 200]},
])
write_wf("sub/nested.json", [
    {"type": "LoadImage", "widgets_values": ["photo.png"], "pos": [10, 10], "size": [100, 100]},
    {"type": "VAEEncode", "widgets_values": []},
    {"type": "KSampler", "widgets_values": ["sdxl_base.safetensors"]},
])
write_wf("broken.json", b"not json" if False else "not json{{{" )
write_wf("nofolder.json", [{"type": "Something"}])

# ── summarize_workflow ──
e = H.summarize_workflow(os.path.join(wf_root, "simple.json"), wf_root)
check("基础字段", e["name"] == "simple" and e["rel"] == "simple.json" and e["folder"] == "")
check("node_count", e["node_count"] == 3)
check("class_types 去重排序", "KSampler" in e["class_types"] and "Note" in e["class_types"])
check("模型识别", any("flux1-dev.safetensors" in m for m in e["models"]))
check("prompt 文本收集", "red fox" in e["text"])
check("封面映射带颜色", len(e["map"]) == 3 and e["map"][0][4] == "#1d1d1d")
check("指纹非空", len(e["fingerprint"]) == 32)
check("子文件夹 rel", H.summarize_workflow(os.path.join(wf_root, "sub", "nested.json"), wf_root)["folder"] == "sub")
check("模型家族 sdxl", any("sdxl" in m for m in H.summarize_workflow(os.path.join(wf_root, "sub", "nested.json"), wf_root)["models"]))
e = H.summarize_workflow(os.path.join(wf_root, "broken.json"), wf_root)
check("坏文件 error", e["error"] is not None and e["node_count"] == 0)
e = H.summarize_workflow(os.path.join(wf_root, "missing.json"), wf_root)
check("缺失文件 error", e["error"] is not None)
e = H.summarize_workflow(os.path.join(tmp_root, "outside.json"), wf_root)
check("根外文件 error", e["error"] == "outside the workflows folder")
# 手改文件防御：type 非字符串
write_wf("junk_type.json", [{"type": True}, {"type": 7}, {"type": "OK"}])
e = H.summarize_workflow(os.path.join(wf_root, "junk_type.json"), wf_root)
check("非字符串 type 防御", e["node_count"] == 3 and "OK" in e["class_types"] and not e["error"])
# 大文件
big = os.path.join(wf_root, "big.json")
with open(big, "w") as f:
    f.write("x" * (H._MAX_BYTES + 10))
check("超大文件 error", H.summarize_workflow(big, wf_root)["error"] is not None)
os.remove(big)

# ── build_index + 缓存 ──
cache_path = os.path.join(tmp_root, "cache.json")
entries = H.build_index(wf_root, cache_path)
check("索引条目数", len(entries) >= 5)
check("排序 folder 优先", entries[0]["folder"] <= entries[-1]["folder"])
# 二次构建命中缓存（文件未变）
entries2 = H.build_index(wf_root, cache_path)
check("二次索引一致", len(entries2) == len(entries))
# 修改文件触发重解析
mtime = os.path.getmtime(os.path.join(wf_root, "simple.json"))
os.utime(os.path.join(wf_root, "simple.json"), (mtime + 10, mtime + 10))
entries3 = H.build_index(wf_root, cache_path)
check("mtime 变化后重解析", len(entries3) == len(entries))

# ── detect_issues ──
issues = H.detect_issues(entries, {"KSampler", "CLIPTextEncode", "LoadImage", "VAEEncode", "Something"})
check("missing_nodes 检测（Note 为前端节点不报）",
      all("Note" not in m["missing"] for m in issues["missing_nodes"]))
issues2 = H.detect_issues(entries, set())
check("注册表为空时报缺失", any("KSampler" in m["missing"] for m in issues2["missing_nodes"]))
# 未保存名
write_wf("Unsaved Workflow 3.json", [{"type": "X"}])
issues3 = H.detect_issues(H.build_index(wf_root, cache_path), {"X"})
check("unsaved_names 检测", any(u["name"].startswith("Unsaved Workflow") for u in issues3["unsaved_names"]))
# 重复指纹
write_wf("dup_a.json", [{"type": "A", "widgets_values": ["m1.safetensors"]}])
write_wf("dup_b.json", [{"type": "A", "widgets_values": ["m1.safetensors"]}])
issues4 = H.detect_issues(H.build_index(wf_root, cache_path), {"A"})
check("duplicates 检测", any(len(g) == 2 for g in issues4["duplicates"]))

# ── collections ──
idx = H.build_index(wf_root, cache_path)
col = H.collections(idx)
check("collections 有 kind 组", any(c["group"] == "kind" for c in col))
check("collections txt2img 判定（含 sampler+textencode）", any(c["id"] == "txt2img" for c in col))
check("collections img2img 判定", any(c["id"] == "img2img" for c in col))
check("collections 模型族 sdxl", any(c["id"] == "sdxl" for c in col))

# ── 图片/封面名/保留名 ──
check("looks_like_image png", H.looks_like_image(b"\x89PNG\r\n\x1a\nxxxx"))
check("looks_like_image jpeg", H.looks_like_image(b"\xff\xd8\xff\xe0"))
check("looks_like_image webp", H.looks_like_image(b"RIFF\x00\x00\x00\x00WEBPVP8"))
check("looks_like_image 拒绝文本", not H.looks_like_image(b"plain text"))
check("is_cover_name", H.is_cover_name("0123456789abcdef.jpg") is True)
check("is_cover_name 拒绝穿越", H.is_cover_name("../../etc.jpg") is False and H.is_cover_name("x.jpg") is False)
check("reserved_part", H.reserved_part(wf_root, os.path.join(wf_root, "NUL")) is not None)
check("reserved_part 正常", H.reserved_part(wf_root, os.path.join(wf_root, "ok")) is None)
check("reserved_part 根含保留词不误伤", H.reserved_part("/home/con/workflows", "/home/con/workflows/ok") is None)

# ── 路由纯逻辑（mock folder_paths 已就位）──
# 相对导入 ..sf_utils 需要包上下文
_sf_pkg = types.ModuleType("sfnodes"); _sf_pkg.__path__ = [root]
_sf_utils_pkg = types.ModuleType("sfnodes.sf_utils"); _sf_utils_pkg.__path__ = [os.path.join(root, "sf_utils")]
sys.modules.setdefault("sfnodes", _sf_pkg)
sys.modules.setdefault("sfnodes.sf_utils", _sf_utils_pkg)

spec_r = importlib.util.spec_from_file_location(
    "sfnodes.nodes.workflow_routes",
    os.path.join(root, "nodes", "workflow_routes.py"),
)
R = importlib.util.module_from_spec(spec_r)
sys.modules[spec_r.name] = R
spec_r.loader.exec_module(R)

class FakeReq:
    headers = {}

req = FakeReq()
check("_wf_root", R._wf_root(req).endswith(os.path.join("user", "default", "workflows")))
check("_wf_resolve 正常", R._wf_resolve(wf_root, "sub/nested.json") == os.path.join(wf_root, "sub", "nested.json"))
check("_wf_resolve 拒绝穿越", R._wf_resolve(wf_root, "../outside.json") is None)
check("_wf_resolve 空拒绝", R._wf_resolve(wf_root, "") is None and R._wf_resolve(wf_root, ".") is None)
check("_wf_resolve 反斜杠", R._wf_resolve(wf_root, "sub\\nested.json").endswith("nested.json"))
check("_wf_cover_name 稳定", R._wf_cover_name("a/b.json") == R._wf_cover_name("a/b.json"))
check("_wf_cover_name 16hex.jpg", len(R._wf_cover_name("x")) == 16 + 4 and R._wf_cover_name("x").endswith(".jpg"))
# meta 读写
meta_path = R._wf_meta_path(req)
ok = R._wf_write_meta(meta_path, {"notes": {"a": "note"}})
check("_wf_write_meta", ok and R._wf_read_meta(meta_path)["notes"]["a"] == "note")
# 坏 sidecar 备份
with open(meta_path, "w") as f:
    f.write("{broken")
check("坏 sidecar 返回 {} 且留副本", R._wf_read_meta(meta_path) == {} and os.path.exists(meta_path + ".broken"))
os.remove(meta_path + ".broken")

print("\nFAILURES:", len(failures))
sys.exit(1 if failures else 0)
