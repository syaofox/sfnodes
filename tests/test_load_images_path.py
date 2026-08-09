# SFLoadImagesPath 后端测试（Node/Python 直接运行：python tests/test_load_images_path.py）
# 覆盖：
#   - _resolve_folder：default / images 根 / input / output / 前缀子目录 / 绝对路径
#   - _list_folders：三源根 + 一级子目录（含新增的 "images" 根）
#   - VALIDATE_INPUTS：目录存在校验
# mock：torch / aiohttp / folder_paths / comfy.utils（numpy/PIL 本机真实可用）
import importlib.util
import os
import sys
import tempfile
import types

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

failures = []

def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")

# ── mock torch ──
torch = types.ModuleType("torch")
torch.float32 = "float32"
torch.Tensor = type("Tensor", (), {})
torch.zeros = lambda *a, **k: "zeros"
torch.ones = lambda *a, **k: "ones"
torch.stack = lambda *a, **k: "stack"
sys.modules["torch"] = torch

# ── mock aiohttp ──
aiohttp = types.ModuleType("aiohttp")
aiohttp.web = types.ModuleType("aiohttp.web")
aiohttp.web.json_response = lambda *a, **k: types.SimpleNamespace(status=200, body=a)
aiohttp.web.Response = types.SimpleNamespace
sys.modules["aiohttp"] = aiohttp
sys.modules["aiohttp.web"] = aiohttp.web

# ── mock folder_paths / comfy.utils（目录用真实 tmp）──
tmp_user = tempfile.mkdtemp(prefix="sf_lip_user_")
tmp_in = tempfile.mkdtemp(prefix="sf_lip_input_")
tmp_out = tempfile.mkdtemp(prefix="sf_lip_output_")

folder_paths = types.ModuleType("folder_paths")
folder_paths.get_user_directory = lambda: tmp_user
folder_paths.get_input_directory = lambda: tmp_in
folder_paths.get_output_directory = lambda: tmp_out
folder_paths.filter_files_content_types = lambda files, types_: files
sys.modules["folder_paths"] = folder_paths

comfy = types.ModuleType("comfy")
comfy.utils = types.ModuleType("comfy.utils")
comfy.utils.ProgressBar = lambda *a, **k: types.SimpleNamespace(update_absolute=lambda *a, **k: None)
comfy.utils.common_upscale = lambda *a, **k: a[0]
sys.modules["comfy"] = comfy
sys.modules["comfy.utils"] = comfy.utils

# ── 注册 sfnodes 包结构 ──
pkg = types.ModuleType("sfnodes"); pkg.__path__ = [root]; sys.modules["sfnodes"] = pkg
pkg2 = types.ModuleType("sfnodes.nodes"); pkg2.__path__ = [os.path.join(root, "nodes")]; sys.modules["sfnodes.nodes"] = pkg2
pkg3 = types.ModuleType("sfnodes.nodes.image"); pkg3.__path__ = [os.path.join(root, "nodes", "image")]; sys.modules["sfnodes.nodes.image"] = pkg3

spec = importlib.util.spec_from_file_location(
    "sfnodes.nodes.image.load_images_path",
    os.path.join(root, "nodes", "image", "load_images_path.py"),
)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

# ── 目录准备 ──
images_base = os.path.join(tmp_user, "sfnodes", "images")
os.makedirs(os.path.join(images_base, "default"), exist_ok=True)
os.makedirs(os.path.join(images_base, "anime"), exist_ok=True)
os.makedirs(os.path.join(tmp_in, "faces"), exist_ok=True)
os.makedirs(os.path.join(tmp_out, "render"), exist_ok=True)

# ── _list_folders ──
folders = mod._list_folders()
check("列表含 default", "default" in folders)
check("列表含 images 根", "images" in folders)
check("列表含 images/anime", "images/anime" in folders)
check("列表含 input/output 根", "input" in folders and "output" in folders)
check("列表含 input/faces", "input/faces" in folders)
check("列表含 output/render", "output/render" in folders)

# ── _resolve_folder ──
rf = mod._resolve_folder
check("default 解析到 images/default", rf("default") == os.path.join(images_base, "default"))
check("images 根解析", rf("images") == os.path.normpath(images_base))
check("input 根解析", rf("input") == os.path.normpath(tmp_in))
check("output 根解析", rf("output") == os.path.normpath(tmp_out))
check("input/faces 子目录", rf("input/faces") == os.path.join(tmp_in, "faces"))
check("images/anime 子目录", rf("images/anime") == os.path.join(images_base, "anime"))
abs_dir = os.path.join(tmp_in, "faces")
check("绝对路径直通", rf(abs_dir) == os.path.normpath(abs_dir))
check("绝对路径不存在的目录也直通（校验层提示）", rf(os.path.join(tmp_in, "nope")) == os.path.normpath(os.path.join(tmp_in, "nope")))

# ── VALIDATE_INPUTS ──
check("VALIDATE 目录存在 True", mod.SFLoadImagesPath.VALIDATE_INPUTS("input/faces") is True)
check("VALIDATE 目录不存在提示", isinstance(mod.SFLoadImagesPath.VALIDATE_INPUTS("/no/such/dir"), str))

# ── INPUT_TYPES 结构 ──
it = mod.SFLoadImagesPath.INPUT_TYPES()
check("folder combo 列表非空", len(it["required"]["folder"][0]) > 0)
check("folder combo 首项 default（ComfyUI 默认值）", it["required"]["folder"][0][0] == "default")

# ── 空目录/目录不存在：不抛错，返回空占位 ──
os.makedirs(os.path.join(tmp_in, "empty"), exist_ok=True)
node = mod.SFLoadImagesPath()
res_empty = node.load_images("input/empty")
check("空目录不抛错且五元组", isinstance(res_empty, tuple) and len(res_empty) == 5)
check("空目录 count=0", res_empty[2] == 0)
check("空目录文件名列表空", res_empty[3] == [] and res_empty[4] == [])
check("空目录返回占位图与遮罩", res_empty[0] == "ones" and res_empty[1] == "zeros")
res_missing = node.load_images("input/no_such_dir")
check("目录不存在不抛错且 count=0", res_missing[2] == 0 and res_missing[3] == [])

# ── _list_subdirs：渐进式按需加载（多级 + 隐藏目录跳过 + 越界空）──
os.makedirs(os.path.join(tmp_in, "faces", "sub1"), exist_ok=True)
os.makedirs(os.path.join(tmp_in, "faces", "sub2"), exist_ok=True)
os.makedirs(os.path.join(tmp_in, "faces", ".hidden"), exist_ok=True)
os.makedirs(os.path.join(tmp_out, "render", "deep"), exist_ok=True)
check("根层列一级子目录", mod._list_subdirs("input") == ["empty", "faces"])
check("子层列一级（多级路径）", mod._list_subdirs("input/faces") == ["sub1", "sub2"])
check("隐藏目录跳过", ".hidden" not in mod._list_subdirs("input/faces"))
check("三层路径", mod._list_subdirs("output/render") == ["deep"])
check("不存在的目录返回空", mod._list_subdirs("input/nope") == [])
check("越界路径返回空", mod._list_subdirs("../../etc") == [])
check("images 根列子目录", mod._list_subdirs("images") == ["anime", "default"])

print()
if failures:
    print(f"{len(failures)} FAILURES: {failures}")
    sys.exit(1)
print("ALL PASS")
