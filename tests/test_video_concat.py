# SFVideoConcat 后端测试（Node/Python 直接运行：python3 tests/test_video_concat.py）
# 覆盖：
#   - sf_utils/video_concat.py 纯逻辑：VHS_FILENAMES (bool, [paths]) 解析、
#     SFBatchAnything 累加后的 list/tuple 混合结构展平、保序去重、
#     PNG 元数据/被 -audio 终版覆盖的中间视频剔除、非路径值忽略
#   - nodes/video/video_concat.py 节点结构：CATEGORY/RETURN_TYPES/RETURN_NAMES/
#     FUNCTION/DESCRIPTION/OUTPUT_NODE、INPUT_TYPES 关键项与全部 tooltip
#   - 根 __init__.py 注册键一致（SFVideoConcat 出现两次）
#   - execute 返回形态（回归）：必须走 io.NodeOutput + ui.PreviewVideo，
#     不能返回 {"ui": PreviewVideo, ...}（legacy 分支只收 dict，会 .keys() 报错）
import importlib.util
import os
import sys
import tempfile
import types

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

failures = []


def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")


def load(modpath, modname):
    spec = importlib.util.spec_from_file_location(modname, modpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


collect_mod = load(os.path.join(root, "sf_utils", "video_concat.py"),
                   "sfnodes.sf_utils.video_concat")
collect = collect_mod.collect_segment_paths
node_mod = load(os.path.join(root, "nodes", "video", "video_concat.py"),
                "sfnodes.nodes.video.video_concat")
SFVideoConcat = node_mod.SFVideoConcat

# ── 1. collect_segment_paths ──
vhs = (True, ["/out/seg_00001.png", "/out/seg_00001.mp4"])
check("VHS_FILENAMES 剥离 PNG", collect(vhs) == ["/out/seg_00001.mp4"])
check("SFBatchAnything 累加 tuple 拼接",
      collect((True, ["/o/a.mp4"], True, ["/o/b.mp4"])) == ["/o/a.mp4", "/o/b.mp4"])
check("list 混合标量", collect([["/o/a.mp4"], True, "/o/b.mp4"]) == ["/o/a.mp4", "/o/b.mp4"])
check("保序去重", collect(["/o/a.mp4", "/o/a.mp4", "/o/b.mkv"]) == ["/o/a.mp4", "/o/b.mkv"])
check("-audio 终版覆盖中间视频",
      collect((True, ["/o/a_00001.png", "/o/a_00001.mp4", "/o/a_00001-audio.mp4"]))
      == ["/o/a_00001-audio.mp4"])
check("无 -audio 时保留视频", collect(["/o/a.mp4"]) == ["/o/a.mp4"])
check("扩展名大小写不敏感", collect(["/o/A.MP4"]) == ["/o/A.MP4"])
check("忽略非路径值", collect(None) == [] and collect(0) == [] and collect(False) == [])
check("忽略非视频扩展名", collect(["/o/a.png", "/o/a.txt", "/o/a.webp"]) == [])
check("空结构返回空", collect([]) == [] and collect(()) == [])

# ── 2. 节点结构 ──
check("CATEGORY", SFVideoConcat.CATEGORY == "sfnodes/video")
check("FUNCTION", SFVideoConcat.FUNCTION == "execute")
check("RETURN_TYPES", SFVideoConcat.RETURN_TYPES == ("STRING",))
check("RETURN_NAMES", SFVideoConcat.RETURN_NAMES == ("video_path",))
check("DESCRIPTION 存在", isinstance(getattr(SFVideoConcat, "DESCRIPTION", None), str)
      and SFVideoConcat.DESCRIPTION.strip() != "")
check("OUTPUT_NODE", getattr(SFVideoConcat, "OUTPUT_NODE", False) is True)

schema = SFVideoConcat.INPUT_TYPES()
check("segments any 输入", schema["required"]["segments"][0] == "*")
check("format 选项", schema["required"]["format"][0] == ["mp4", "mkv", "webm"])
check("audio 可选 AUDIO", schema["optional"]["audio"][0] == "AUDIO")
check("cleanup 默认关", schema["optional"]["cleanup"][0] == "BOOLEAN"
      and schema["optional"]["cleanup"][1].get("default") is False)

for group in ("required", "optional"):
    for name, config in schema.get(group, {}).items():
        has_tooltip = isinstance(config, tuple) and len(config) > 1 and isinstance(config[1], dict) and bool(config[1].get("tooltip"))
        check(f"tooltip {name}", has_tooltip)

# ── 3. 根注册 ──
with open(os.path.join(root, "__init__.py"), encoding="utf-8") as fh:
    root_src = fh.read()
check("根注册 SFVideoConcat 双字典", root_src.count('"SFVideoConcat"') == 2)

# ── 4. execute 返回形态（回归：PreviewVideo 不能塞 legacy ui）──
# 注册 sfnodes 包结构使节点内 `from ...sf_utils.video_concat import ...` 可解析
for _pkg, _rel in [("sfnodes", "."), ("sfnodes.nodes", "nodes"),
                   ("sfnodes.nodes.video", "nodes/video"), ("sfnodes.sf_utils", "sf_utils")]:
    _m = types.ModuleType(_pkg)
    _m.__path__ = [os.path.join(root, _rel)]
    sys.modules.setdefault(_pkg, _m)

calls = {}


class FakeVideoFromFile:
    def __init__(self, path):
        self.path = path


class FakeVideoFromList:
    def __init__(self, videos, complete_audio=None, codec=None):
        calls["videos"] = [v.path for v in videos]
        calls["complete_audio"] = complete_audio

    def get_dimensions(self):
        return (64, 64)

    def save_to(self, path, format=None, codec=None):
        calls["save_path"] = path
        with open(path, "wb") as fh:
            fh.write(b"fake")


class FakeVideoContainer:
    def __init__(self, fmt):
        self.fmt = fmt

    @staticmethod
    def get_extension(fmt):
        return {"mp4": "mp4", "mkv": "mkv", "webm": "webm"}[fmt]


class FakeVideoCodec:
    def __init__(self, name):
        self.name = name


class FakeNodeOutput:
    def __init__(self, *args, ui=None):
        self.args = args
        self.ui = ui

    @property
    def result(self):
        return self.args if self.args else None


class FakeSavedResult:
    def __init__(self, file, subfolder, folder_type):
        self.file, self.subfolder, self.folder_type = file, subfolder, folder_type


class FakePreviewVideo:
    def __init__(self, saved):
        self.saved = saved

    def as_dict(self):
        return {"images": [{"filename": s.file, "subfolder": s.subfolder} for s in self.saved],
                "animated": (True,)}


class FakeFolderType:
    output = "output"
    temp = "temp"


_mock_api = types.ModuleType("comfy_api")
_mock_api.__path__ = []
_mock_latest = types.ModuleType("comfy_api.latest")
_mock_latest.InputImpl = types.SimpleNamespace(VideoFromFile=FakeVideoFromFile,
                                               VideoFromList=FakeVideoFromList)
_mock_latest.Types = types.SimpleNamespace(VideoContainer=FakeVideoContainer, VideoCodec=FakeVideoCodec)
_mock_latest.io = types.SimpleNamespace(NodeOutput=FakeNodeOutput, FolderType=FakeFolderType)
_mock_latest.ui = types.SimpleNamespace(PreviewVideo=FakePreviewVideo, SavedResult=FakeSavedResult)
sys.modules["comfy_api"] = _mock_api
sys.modules["comfy_api.latest"] = _mock_latest

with tempfile.TemporaryDirectory() as tmpdir:
    fake_fp = types.ModuleType("folder_paths")
    fake_fp.get_output_directory = lambda: tmpdir
    fake_fp.get_save_image_path = lambda prefix, out_dir, width, height: (tmpdir, "verify", 1, "Wan21_SCAIL2", None)
    sys.modules["folder_paths"] = fake_fp

    p1 = os.path.join(tmpdir, "seg_a.mp4")
    p2 = os.path.join(tmpdir, "seg_b.mp4")
    for path in (p1, p2):
        with open(path, "wb") as fh:
            fh.write(b"x")

    # 真实工作流形态：SFBatchAnything 累加后的 (bool, [paths], bool, [paths])
    res = SFVideoConcat().execute(segments=(True, [p1], True, [p2]),
                                  filename_prefix="Wan21_SCAIL2/verify", format="mp4")
    out_path = os.path.join(tmpdir, "verify_00001_.mp4")

check("execute 返回 NodeOutput（非 legacy dict）", isinstance(res, FakeNodeOutput))
check("execute result 为输出路径 tuple", getattr(res, "result", None) == (out_path,))
check("execute ui 可 as_dict（PreviewVideo）",
      isinstance(getattr(res, "ui", None), FakePreviewVideo) and bool(res.ui.as_dict().get("images")))
check("VideoFromList 收到两段且 complete_audio=None",
      calls.get("videos") == [p1, p2] and calls.get("complete_audio") is None)
check("save_to 输出路径与 result 一致", calls.get("save_path") == out_path)

print()
if failures:
    print(f"FAILED: {len(failures)} -> {failures}")
    sys.exit(1)
print("ALL PASS")
