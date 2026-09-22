# SFVideoCompare 后端测试（Node/Python 直接运行：python3 tests/test_video_compare.py）
# 覆盖：
#   - sf_utils/video_compare.py 纯逻辑：preview_filename 唯一/前缀/扩展名、
#     fallback_frame_count 分支、build_entry 字段归一、
#     probe_video_meta（桩 av：帧数/帧率/时长、container.duration 回退、
#     frames=0 时 时长×帧率 兜底、无视频流全 0）
#   - nodes/video/video_compare.py 节点结构：CATEGORY/FUNCTION/OUTPUT_NODE/
#     空 RETURN_TYPES、两 optional VIDEO 输入与 tooltip、DESCRIPTION
#   - 根 __init__.py 注册键一致（SFVideoCompare 两字典各一次）
#   - execute：FakeVideo 记录 save_to 参数（MP4 + H264 + temp 路径）与 ui 形状、
#     单路输入、全空返回空列表、save_to 失败抛 RuntimeError
import importlib.util
import os
import sys
import tempfile
import types
from fractions import Fraction

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


# 注册 sfnodes 包结构使节点内 `from ...sf_utils.video_compare import ...` 可解析
for _pkg, _rel in [("sfnodes", "."), ("sfnodes.nodes", "nodes"),
                   ("sfnodes.nodes.video", "nodes/video"), ("sfnodes.sf_utils", "sf_utils")]:
    _m = types.ModuleType(_pkg)
    _m.__path__ = [os.path.join(root, _rel)]
    sys.modules.setdefault(_pkg, _m)

helper = load(os.path.join(root, "sf_utils", "video_compare.py"),
              "sfnodes.sf_utils.video_compare")
node_mod = load(os.path.join(root, "nodes", "video", "video_compare.py"),
                "sfnodes.nodes.video.video_compare")
SFVideoCompare = node_mod.SFVideoCompare

# ── 1. preview_filename / fallback_frame_count / build_entry ──
name_a = helper.preview_filename()
name_b = helper.preview_filename()
check("preview_filename 前缀与扩展名",
      name_a.startswith("sf_video_compare_") and name_a.endswith(".mp4"))
check("preview_filename 每次唯一", name_a != name_b)

check("fallback：帧数已有不覆盖", helper.fallback_frame_count(90, 30.0, 3.0) == 90)
check("fallback：帧数缺失按时长×帧率", helper.fallback_frame_count(0, 30.0, 3.0) == 90)
check("fallback：帧率缺失不猜", helper.fallback_frame_count(0, 0.0, 3.0) == 0)
check("fallback：时长缺失不猜", helper.fallback_frame_count(0, 30.0, 0.0) == 0)

entry = helper.build_entry("x.mp4", None, "temp", {"frame_count": "12", "frame_rate": "24.5", "duration": None})
check("build_entry 归一", entry == {"filename": "x.mp4", "subfolder": "", "type": "temp",
                                    "frame_count": 12, "frame_rate": 24.5, "duration": 0.0})
check("build_entry 默认类型 temp", helper.build_entry("y.mp4", "", None, {})["type"] == "temp")

# ── 2. probe_video_meta（桩 av）──
class FakeStream:
    def __init__(self, frames=0, average_rate=30, duration=None, time_base=Fraction(1, 30)):
        self.frames = frames
        self.average_rate = average_rate
        self.duration = duration
        self.time_base = time_base


class FakeContainer:
    def __init__(self, streams, duration=0):
        self.streams = types.SimpleNamespace(video=streams)
        self.duration = duration

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def install_fake_av(container):
    fake_av = types.ModuleType("av")
    fake_av.time_base = 1000000
    fake_av.open = lambda path, mode="r": container
    sys.modules["av"] = fake_av


install_fake_av(FakeContainer([FakeStream(frames=90, average_rate=30, duration=90)]))
meta = helper.probe_video_meta("/tmp/whatever.mp4")
check("probe：帧数/帧率/时长", meta == {"frame_count": 90, "frame_rate": 30.0, "duration": 3.0})

install_fake_av(FakeContainer([FakeStream(frames=0, average_rate=30, duration=90)]))
meta = helper.probe_video_meta("/tmp/whatever.mp4")
check("probe：frames=0 兜底为 时长×帧率", meta["frame_count"] == 90)

install_fake_av(FakeContainer([FakeStream(frames=0, average_rate=30, duration=None)], duration=2_500_000))
meta = helper.probe_video_meta("/tmp/whatever.mp4")
check("probe：stream.duration 缺失回退 container.duration", meta["duration"] == 2.5)

install_fake_av(FakeContainer([]))
check("probe：无视频流全 0",
      helper.probe_video_meta("/tmp/whatever.mp4") == {"frame_count": 0, "frame_rate": 0.0, "duration": 0.0})

# ── 3. 节点结构 ──
check("CATEGORY", SFVideoCompare.CATEGORY == "sfnodes/video")
check("FUNCTION", SFVideoCompare.FUNCTION == "execute")
check("OUTPUT_NODE", getattr(SFVideoCompare, "OUTPUT_NODE", False) is True)
check("RETURN_TYPES 空（纯查看器）", SFVideoCompare.RETURN_TYPES == ())
check("DESCRIPTION 存在", isinstance(getattr(SFVideoCompare, "DESCRIPTION", None), str)
      and SFVideoCompare.DESCRIPTION.strip() != "")

schema = SFVideoCompare.INPUT_TYPES()
check("required 为空", schema["required"] == {})
check("video_a/video_b 均 optional VIDEO",
      schema["optional"]["video_a"][0] == "VIDEO" and schema["optional"]["video_b"][0] == "VIDEO")
for name, config in schema["optional"].items():
    has_tooltip = isinstance(config, tuple) and len(config) > 1 and isinstance(config[1], dict) and bool(config[1].get("tooltip"))
    check(f"tooltip {name}", has_tooltip)

# ── 4. 根注册 ──
with open(os.path.join(root, "__init__.py"), encoding="utf-8") as fh:
    root_src = fh.read()
check("根注册 SFVideoCompare 双字典", root_src.count('"SFVideoCompare"') == 2)

# ── 5. execute ──
calls = {}


class FakeVideoContainer:
    MP4 = "mp4"


class FakeVideoCodec:
    H264 = "h264"


class FakeVideo:
    def __init__(self, label, fail=False):
        self.label = label
        self.fail = fail

    def save_to(self, path, format=None, codec=None):
        if self.fail:
            raise ValueError("boom")
        calls.setdefault("save", []).append({"path": path, "format": format, "codec": codec, "label": self.label})
        with open(path, "wb") as fh:
            fh.write(b"fake")


_mock_api = types.ModuleType("comfy_api")
_mock_api.__path__ = []
_mock_latest = types.ModuleType("comfy_api.latest")
_mock_latest.Types = types.SimpleNamespace(VideoContainer=FakeVideoContainer, VideoCodec=FakeVideoCodec)
sys.modules["comfy_api"] = _mock_api
sys.modules["comfy_api.latest"] = _mock_latest

with tempfile.TemporaryDirectory() as tmpdir:
    fake_fp = types.ModuleType("folder_paths")
    fake_fp.get_temp_directory = lambda: tmpdir
    sys.modules["folder_paths"] = fake_fp

    install_fake_av(FakeContainer([FakeStream(frames=48, average_rate=24, duration=48,
                                              time_base=Fraction(1, 24))]))
    result = SFVideoCompare().execute(video_a=FakeVideo("a"), video_b=FakeVideo("b"))
    check("execute ui 两路列表", list(result.keys()) == ["ui"]
          and len(result["ui"]["a_videos"]) == 1 and len(result["ui"]["b_videos"]) == 1)
    entry_a = result["ui"]["a_videos"][0]
    check("execute 条目含帧元数据",
          entry_a["frame_count"] == 48 and entry_a["frame_rate"] == 24.0
          and entry_a["duration"] == 2.0 and entry_a["type"] == "temp")
    check("execute 落盘 temp 且文件名唯一",
          entry_a["filename"].startswith("sf_video_compare_")
          and os.path.isfile(os.path.join(tmpdir, entry_a["filename"]))
          and entry_a["filename"] != result["ui"]["b_videos"][0]["filename"])
    check("execute save_to 参数 MP4 + H264",
          all(c["format"] == "mp4" and c["codec"] == "h264" and os.path.dirname(c["path"]) == tmpdir
              for c in calls["save"]) and len(calls["save"]) == 2)

    calls.clear()
    result = SFVideoCompare().execute(video_a=FakeVideo("a"), video_b=None)
    check("execute 单路 A：B 为空列表", len(result["ui"]["a_videos"]) == 1 and result["ui"]["b_videos"] == [])

    calls.clear()
    result = SFVideoCompare().execute(video_a=None, video_b=None)
    check("execute 全空：空 ui 且不落盘",
          result == {"ui": {"a_videos": [], "b_videos": []}} and not calls)

    try:
        SFVideoCompare().execute(video_a=FakeVideo("a", fail=True), video_b=None)
        check("execute 失败抛 RuntimeError", False)
    except RuntimeError as e:
        check("execute 失败抛 RuntimeError", "[SFVideoCompare]" in str(e) and "boom" in str(e))

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("test_video_compare: all assertions passed")
