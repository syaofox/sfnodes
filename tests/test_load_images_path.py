# SFLoadImagesPath 后端测试（Node/Python 直接运行：python tests/test_load_images_path.py）
# 覆盖：
#   - resolve_folder：default（→input 根）/ input / output / 前缀子目录 / 绝对路径（sf_utils/image_sources）
#   - list_folders：两源根（input/output）+ 一级子目录
#   - sort_key：数字序 + 同数字按文件名兜底（不依赖 os.listdir 顺序）
#   - VALIDATE_INPUTS：目录存在校验
#   - IS_CHANGED：文件 mtime 哈希 / 切片参与 / 连线输入（None）退化为全量目录哈希
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
_IMG_EXT = {"png", "jpg", "jpeg", "webp", "bmp", "gif", "tif", "tiff"}
def _filter_files_content_types(files, types_):
    if "image" not in types_:
        return []
    return [f for f in files if f.rsplit(".", 1)[-1].lower() in _IMG_EXT]
folder_paths.filter_files_content_types = _filter_files_content_types
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
os.makedirs(os.path.join(tmp_in, "faces"), exist_ok=True)
os.makedirs(os.path.join(tmp_out, "render"), exist_ok=True)
with open(os.path.join(tmp_in, "root.png"), "w") as f:
    f.write("x")
for name in ("a.png", "b.jpg", "note.txt"):
    with open(os.path.join(tmp_in, "faces", name), "w") as f:
        f.write("x")

# ── list_folders（共享模块 sf_utils/image_sources）──
sources = sys.modules["sfnodes.sf_utils.image_sources"]
folders = sources.list_folders()
check("列表含 default", "default" in folders)
check("列表无 images 源", "images" not in folders and not any(f.startswith("images/") for f in folders))
check("列表含 input/output 根", "input" in folders and "output" in folders)
check("列表含 input/faces", "input/faces" in folders)
check("列表含 output/render", "output/render" in folders)

# ── resolve_folder + sort_key ──
rf = sources.resolve_folder
check("default 解析到 input 根", rf("default") == os.path.normpath(tmp_in))
check("空值解析到 input 根", rf("") == os.path.normpath(tmp_in))
check("input 根解析", rf("input") == os.path.normpath(tmp_in))
check("output 根解析", rf("output") == os.path.normpath(tmp_out))
check("input/faces 子目录", rf("input/faces") == os.path.join(tmp_in, "faces"))
check("裸名走 input 相对解析", rf("faces") == os.path.join(tmp_in, "faces"))
abs_dir = os.path.join(tmp_in, "faces")
check("绝对路径直通", rf(abs_dir) == os.path.normpath(abs_dir))
check("绝对路径不存在的目录也直通（校验层提示）", rf(os.path.join(tmp_in, "nope")) == os.path.normpath(os.path.join(tmp_in, "nope")))
check("sort_key 数字序", sources.sort_key("10.png") > sources.sort_key("2.png"))
check("sort_key 同数字按文件名兜底", sources.sort_key("a1.png") < sources.sort_key("b1.png"))

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
check("越界路径钳制到 input 根（无逃逸）", mod._list_subdirs("../../etc") == mod._list_subdirs("input"))

# ── _count_image_files：当前目录一级图片数（内容类型过滤，不含子目录）──
check("根层图片数（1 张）", mod._count_image_files("input") == 1)
check("子层图片数（a.png + b.jpg，note.txt 不算）", mod._count_image_files("input/faces") == 2)
check("绝对路径图片数", mod._count_image_files(os.path.join(tmp_in, "faces")) == 2)
check("不存在的目录图片数为 0", mod._count_image_files("input/nope") == 0)
check("越界路径图片数钳到 input 根（无逃逸）", mod._count_image_files("../../etc") == 1)

# ── IS_CHANGED：文件哈希；连线输入（None）退化为全量目录哈希而非报错 ──
ip = mod.SFLoadImagesPath
h_full = ip.IS_CHANGED("input/faces", 0, 0, 1)
check("IS_CHANGED 返回 sha256 十六进制", isinstance(h_full, str) and len(h_full) == 64)
check("IS_CHANGED 稳定（同输入同值）", ip.IS_CHANGED("input/faces", 0, 0, 1) == h_full)
check("IS_CHANGED 切片参与（cap=1 不同于全量）", ip.IS_CHANGED("input/faces", 1, 0, 1) != h_full)
check("IS_CHANGED 切片参与（skip=1 不同于全量）", ip.IS_CHANGED("input/faces", 0, 1, 1) != h_full)
check("IS_CHANGED 连线 cap（None）退全量且不抛错", ip.IS_CHANGED("input/faces", None, 0, 1) == h_full)
check("IS_CHANGED 连线 skip（None）退全量", ip.IS_CHANGED("input/faces", 0, None, 1) == h_full)
check("IS_CHANGED 连线 nth（None）退全量", ip.IS_CHANGED("input/faces", 0, 0, None) == h_full)
check("IS_CHANGED 三输入全连线（None）退全量", ip.IS_CHANGED("input/faces", None, None, None) == h_full)
check("IS_CHANGED 目录不存在 False", ip.IS_CHANGED("input/no_such_dir") is False)
_a = os.path.join(tmp_in, "faces", "a.png")
_old_ns = os.stat(_a).st_mtime_ns
os.utime(_a, ns=(_old_ns + 10_000_000_000, _old_ns + 10_000_000_000))
check("IS_CHANGED 感知文件 mtime 变化", ip.IS_CHANGED("input/faces", 0, 0, 1) != h_full)
os.utime(_a, ns=(_old_ns, _old_ns))
check("IS_CHANGED mtime 复原后哈希复原", ip.IS_CHANGED("input/faces", 0, 0, 1) == h_full)

# ── 路由响应：subdirs + file_count（前端计数器数据源）──
class _FakeRoutes:
    def __init__(self): self.handlers = {}
    def get(self, path):
        def deco(fn):
            self.handlers[path] = fn
            return fn
        return deco
fake_routes = _FakeRoutes()
server_mod = types.ModuleType("server")
class _PromptServer: pass
_PromptServer.instance = types.SimpleNamespace(routes=fake_routes)
server_mod.PromptServer = _PromptServer
sys.modules["server"] = server_mod
mod._register_routes()
route = fake_routes.handlers.get("/api/sfnodes/images_path/subdirs")
check("subdirs 路由已注册", callable(route))
import asyncio
_resp = asyncio.run(route(types.SimpleNamespace(query={"folder": "input/faces"})))
_payload = _resp.body[0]
check("路由返回 subdirs", _payload["subdirs"] == ["sub1", "sub2"])
check("路由返回 file_count", _payload["file_count"] == 2)

print()
if failures:
    print(f"{len(failures)} FAILURES: {failures}")
    sys.exit(1)
print("ALL PASS")
