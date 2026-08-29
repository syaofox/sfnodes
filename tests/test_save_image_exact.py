#!/usr/bin/env python3
# SFSaveImageExact 后端逻辑测试（Node/Python 直接运行：python tests/test_save_image_exact.py）
# 覆盖：INPUT_TYPES、_safe_filename 清洗、精确名保存、overwrite/increment、batch>1、format/quality、越界拒绝、png metadata
import importlib.util
import json
import os
import sys
import tempfile
import types

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── mock torch / folder_paths / comfy.cli_args ──
import numpy as np
from PIL import Image

# Minimal torch mock with tensor semantics used by SFSaveImageExact
torch_mock = types.ModuleType("torch")

class MockTensorFrame:
    def __init__(self, arr):  # arr is single frame HWC
        self._arr = arr
    def cpu(self):
        return self
    def numpy(self):
        return self._arr

class MockBatchTensor:
    def __init__(self, arr):  # arr [B,H,W,C]
        self._arr = arr
    def __iter__(self):
        for i in range(self._arr.shape[0]):
            yield MockTensorFrame(self._arr[i])
    def __len__(self):
        return self._arr.shape[0]
    @property
    def shape(self):
        return self._arr.shape

torch_mock.from_numpy = lambda arr: None  # not used
sys.modules["torch"] = torch_mock

# comfy.cli_args mock
comfy = types.ModuleType("comfy")
cli_args = types.ModuleType("comfy.cli_args")
cli_args.args = types.SimpleNamespace(disable_metadata=False)
sys.modules["comfy"] = comfy
sys.modules["comfy.cli_args"] = cli_args

tmp_output = tempfile.mkdtemp(prefix="sf_save_exact_out_")
tmp_input = tempfile.mkdtemp(prefix="sf_save_exact_in_")

folder_paths = types.ModuleType("folder_paths")
folder_paths.get_output_directory = lambda: tmp_output
folder_paths.get_input_directory = lambda: tmp_input
folder_paths.get_temp_directory = lambda: tmp_output
# real is_within_directory logic (copy from folder_paths.py)
def is_within_directory(directory, target):
    try:
        directory = os.path.realpath(directory)
        target = os.path.realpath(target)
        return os.path.commonpath((directory, target)) == directory
    except ValueError:
        return False
folder_paths.is_within_directory = is_within_directory
sys.modules["folder_paths"] = folder_paths

# ── load module ──
spec = importlib.util.spec_from_file_location(
    "sfnodes.nodes.image.save_image_exact",
    os.path.join(root, "nodes", "image", "save_image_exact.py"),
)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

failures = []
def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")

node = mod.SFSaveImageExact()
check("CATEGORY", node.CATEGORY == "sfnodes/image")
check("DESCRIPTION", isinstance(node.DESCRIPTION, str) and len(node.DESCRIPTION) > 0)
check("OUTPUT_NODE", node.OUTPUT_NODE is True)
check("FUNCTION save", node.FUNCTION == "save")
check("RETURN_TYPES", node.RETURN_TYPES == ("IMAGE",))

it = node.INPUT_TYPES()
check("required images", it["required"]["images"][0] == "IMAGE")
check("required filename", "filename" in it["required"] and it["required"]["filename"][0] == "STRING")
check("optional overwrite", "overwrite" in it["optional"])
check("optional format", "format" in it["optional"])
check("optional quality", "quality" in it["optional"])
check("hidden prompt", "prompt" in it["hidden"])

# ── _safe_filename 清洗 ──
sf = mod._safe_filename
check("_safe_filename 正常", sf("ComfyUI") == "ComfyUI")
check("_safe_filename 子目录", sf("seedvr/xiaoguo-v3gai/a1") == "seedvr/xiaoguo-v3gai/a1")
check("_safe_filename 带扩展名剥离", sf("seedvr/xiaoguo-v3gai/a1.png") == "seedvr/xiaoguo-v3gai/a1")
check("_safe_filename .JPG 大小写剥离", sf("a/B.JPG") == "a/B")
check("_safe_filename 非法字符", sf("Bad:Name*") == "Bad_Name")
check("_safe_filename 中文保留", sf("seedvr/测试/a1") == "seedvr/测试/a1")
check("_safe_filename 空格保留", sf("My Folder/a1") == "My Folder/a1")
check("_safe_filename 路径穿越拒绝", sf("..") == "" and sf("a/../b") == "")
check("_safe_filename leading 斜杠拒绝", sf("/abs") == "")
check("_safe_filename Windows 保留名", sf("CON") == "CON_")
check("_safe_filename aa.png_ 尾下划线剥离", sf("a1.png") == "a1")
check("_safe_filename 超长拒绝", sf("x"*300) == "")

# ── helper to build mock batch images ──
def make_images(b=1, h=4, w=4):
    arr = np.random.rand(b, h, w, 3).astype(np.float32)
    return MockBatchTensor(arr)

def exists(fname, sub=""):
    base = os.path.join(tmp_output, sub) if sub else tmp_output
    return os.path.isfile(os.path.join(base, fname))

# 清理输出目录
import shutil

def clean_output():
    for rootdir, dirs, files in os.walk(tmp_output):
        for f in files:
            os.remove(os.path.join(rootdir, f))
        # 不删子目录，保留结构

clean_output()

# ── 精确名保存：seedvr/xiaoguo-v3gai/a1.png ──
imgs = make_images(1)
r = node.save(imgs, filename="seedvr/xiaoguo-v3gai/a1", overwrite=True, format="png")
check("精确名保存 filename seedvr/xiaoguo-v3gai/a1 -> a1.png", r["ui"]["images"][0]["filename"] == "a1.png" and r["ui"]["images"][0]["subfolder"] == "seedvr/xiaoguo-v3gai")
check("精确名文件存在", exists("a1.png", "seedvr/xiaoguo-v3gai"))
# 验证不是 _00001_ 格式
check("非计数后缀", not exists("a1_00001_.png", "seedvr/xiaoguo-v3gai"))

# ── 双扩展名剥离：a1.png 不应得到 a1.png_00001_.png ──
clean_output()
r = node.save(make_images(1), filename="seedvr/xiaoguo-v3gai/a1.png", overwrite=True, format="png")
check("带 .png 输入仍为 a1.png", r["ui"]["images"][0]["filename"] == "a1.png")
check("双扩展名未出现", not exists("a1.png_00001_.png", "seedvr/xiaoguo-v3gai") and not exists("a1.png.png", "seedvr/xiaoguo-v3gai"))

# ── overwrite=True 重复执行覆盖（同名文件 mtime 更新但不新增 _1）──
clean_output()
node.save(make_images(1), filename="overwrite_test/a", overwrite=True, format="png")
mtime1 = os.path.getmtime(os.path.join(tmp_output, "overwrite_test", "a.png"))
import time
time.sleep(0.02)
node.save(make_images(1), filename="overwrite_test/a", overwrite=True, format="png")
mtime2 = os.path.getmtime(os.path.join(tmp_output, "overwrite_test", "a.png"))
check("overwrite 覆盖 mtime 更新", mtime2 >= mtime1)
check("overwrite 不产生 _1", not exists("a_1.png", "overwrite_test"))

# ── overwrite=False 递增 _1 ──
clean_output()
node.save(make_images(1), filename="inc_test/a", overwrite=False, format="png")
check("increment 首帧 a.png", exists("a.png", "inc_test"))
node.save(make_images(1), filename="inc_test/a", overwrite=False, format="png")
check("increment 次帧 a_1.png", exists("a_1.png", "inc_test"))
node.save(make_images(1), filename="inc_test/a", overwrite=False, format="png")
check("increment 第三帧 a_2.png", exists("a_2.png", "inc_test"))
# 确保不使用 _00001_
check("increment 非 _00001_", not exists("a_00001_.png", "inc_test"))

# ── batch>1 overwrite=True：a.png, a_1.png, a_2.png ──
clean_output()
r = node.save(make_images(3), filename="batch/a", overwrite=True, format="png")
fns = [x["filename"] for x in r["ui"]["images"]]
check("batch overwrite 3帧命名", fns == ["a.png", "a_1.png", "a_2.png"])
check("batch 文件均存在 overwrite", all(exists(f, "batch") for f in fns))

# ── batch>1 overwrite=False：首次 batch 2 帧，空目录 -> a.png, a_1.png ──
clean_output()
r = node.save(make_images(2), filename="batch2/b", overwrite=False, format="png")
fns = [x["filename"] for x in r["ui"]["images"]]
check("batch increment 空目录 2帧", fns == ["b.png", "b_1.png"])
# 已存在 a.png, a_1.png 时再跑 batch 2 帧 -> 分配 b_2.png, b_3.png
r2 = node.save(make_images(2), filename="batch2/b", overwrite=False, format="png")
fns2 = [x["filename"] for x in r2["ui"]["images"]]
check("batch increment 已占用后继续递增", fns2 == ["b_2.png", "b_3.png"])

# ── format jpeg/webp ──
clean_output()
r = node.save(make_images(1), filename="fmt_test/img", overwrite=True, format="jpeg", quality=80)
check("jpeg 扩展名 .jpg", r["ui"]["images"][0]["filename"] == "img.jpg" and exists("img.jpg", "fmt_test"))
clean_output()
r = node.save(make_images(1), filename="fmt_test/img", overwrite=True, format="webp", quality=80)
check("webp 扩展名 .webp", r["ui"]["images"][0]["filename"] == "img.webp" and exists("img.webp", "fmt_test"))
# format 决定扩展名，输入中的 .png 被剥离
clean_output()
r = node.save(make_images(1), filename="fmt_test/img.png", overwrite=True, format="jpeg")
check("输入 .png + format jpeg -> .jpg", r["ui"]["images"][0]["filename"] == "img.jpg")

# ── png metadata 嵌入（prompt/workflow）──
clean_output()
imgs = make_images(1)
node.save(imgs, filename="meta_test/m", overwrite=True, format="png", prompt={"test": 123}, extra_pnginfo={"workflow": {"w": 1}})
with Image.open(os.path.join(tmp_output, "meta_test", "m.png")) as im:
    check("png metadata prompt", "prompt" in im.text and '"test": 123' in im.text["prompt"])
    check("png metadata workflow", "workflow" in im.text and '"w": 1' in im.text["workflow"])

# ── 越界拒绝 ──
try:
    r = node.save(make_images(1), filename="../../etc/passwd", overwrite=True, format="png")
    check("越界 filename 回退而非抛异常（清洗为空回退 ComfyUI）", r["ui"]["images"][0]["filename"] == "ComfyUI.png" and r["ui"]["images"][0]["subfolder"] == "")
    check("越界回退文件在 output 内", exists("ComfyUI.png"))
except Exception as e:
    check("越界不应抛未捕获异常", False)

# 直接测试 is_within_directory 拒绝绝对逃逸（构造 subfolder 含 .. 的清洗已拒绝，回退）
check("清洗后无 ..", sf("../../etc/passwd") == "")

print("\nFAILURES:", len(failures))
if failures:
    print(failures)
sys.exit(1 if failures else 0)
