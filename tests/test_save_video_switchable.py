#!/usr/bin/env python3
# SFSaveVideoSwitchable 后端逻辑测试（Node/Python 直接运行：python tests/test_save_video_switchable.py）
import importlib.util
import os
import sys
import tempfile
import types

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
comfy_root = os.path.abspath(os.path.join(root, "..", ".."))
if comfy_root not in sys.path:
    sys.path.insert(0, comfy_root)

# ── mock torch ──
if "torch" not in sys.modules:
    tm = types.ModuleType("torch")
    tm.Tensor = object
    tm.device = object
    sys.modules["torch"] = tm
else:
    if not hasattr(sys.modules["torch"], "Tensor"):
        sys.modules["torch"].Tensor = object

# ── mock folder_paths ──
tmp_output = tempfile.mkdtemp(prefix="sf_save_video_out_")
tmp_temp = tempfile.mkdtemp(prefix="sf_save_video_tmp_")

# 先尝试导入真实 folder_paths，若失败则 mock
try:
    import folder_paths as _real_fp
    # 覆盖其目录为 tmp，避免污染真实 output
    _real_fp.get_output_directory = lambda: tmp_output
    _real_fp.get_temp_directory = lambda: tmp_temp
    folder_paths = _real_fp
    sys.modules["folder_paths"] = folder_paths
except Exception:
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_output_directory = lambda: tmp_output
    folder_paths.get_temp_directory = lambda: tmp_temp
    def _get_save_image_path(prefix, output_dir, width, height):
        raw = prefix.replace("\\", "/").strip("/")
        if "/" in raw:
            subfolder, filename = raw.rsplit("/", 1)
        else:
            subfolder, filename = "", raw
        if not filename:
            filename = "ComfyUI"
        full_output_folder = os.path.join(output_dir, subfolder) if subfolder else output_dir
        os.makedirs(full_output_folder, exist_ok=True)
        counter = 1
        try:
            files = os.listdir(full_output_folder)
        except FileNotFoundError:
            files = []
        while any(f.startswith(f"{filename}_{counter:05}_") for f in files):
            counter += 1
        return full_output_folder, filename, counter, subfolder, prefix
    folder_paths.get_save_image_path = _get_save_image_path
    def is_within_directory(directory, target):
        try:
            directory = os.path.realpath(directory)
            target = os.path.realpath(target)
            return os.path.commonpath((directory, target)) == directory
        except ValueError:
            return False
    folder_paths.is_within_directory = is_within_directory
    sys.modules["folder_paths"] = folder_paths
else:
    # 已有真实 folder_paths，仍需 mock get_save_image_path 以隔离
    orig_get = folder_paths.get_save_image_path
    def _wrapped_get(prefix, output_dir, width, height):
        # 强制 output_dir 为 tmp_output，忽略传入
        return orig_get(prefix, tmp_output, width, height) if "tmp_output" in str(output_dir) else orig_get(prefix, output_dir, width, height)
    # 简单包装：若传入的是真实 output，保持tmp隔离
    # 为简化，直接重定义为基于 tmp_output 的计数逻辑
    def _mock_get(prefix, output_dir, width, height):
        raw = prefix.replace("\\", "/").strip("/")
        if "/" in raw:
            subfolder, filename = raw.rsplit("/", 1)
        else:
            subfolder, filename = "", raw
        if not filename:
            filename = "ComfyUI"
        full_output_folder = os.path.join(tmp_output, subfolder) if subfolder else tmp_output
        os.makedirs(full_output_folder, exist_ok=True)
        counter = 1
        try:
            files = os.listdir(full_output_folder)
        except FileNotFoundError:
            files = []
        while any(f.startswith(f"{filename}_{counter:05}_") for f in files):
            counter += 1
        return full_output_folder, filename, counter, subfolder, prefix
    folder_paths.get_save_image_path = _mock_get

# ── mock comfy.cli_args ──
try:
    import comfy.cli_args as _real_cli
    cli_args = _real_cli
    if not hasattr(cli_args.args, "disable_metadata"):
        cli_args.args.disable_metadata = False
except Exception:
    comfy = types.ModuleType("comfy")
    cli_args = types.ModuleType("comfy.cli_args")
    cli_args.args = types.SimpleNamespace(disable_metadata=False)
    sys.modules["comfy"] = comfy
    sys.modules["comfy.cli_args"] = cli_args

# ── mock av (避免真实 av 导入) ──
for _m in ["av", "av.bitstream", "av.container", "av.subtitles.stream", "av.video.reformatter"]:
    if _m not in sys.modules:
        sys.modules[_m] = types.ModuleType(_m)
# 补齐必要符号
sys.modules["av.bitstream"].BitStreamFilterContext = object
sys.modules["av.container"].InputContainer = object
sys.modules["av.subtitles.stream"].SubtitleStream = object
sys.modules["av.video.reformatter"].ColorPrimaries = object
sys.modules["av.video.reformatter"].ColorRange = object
sys.modules["av.video.reformatter"].ColorTrc = object

# ── mock comfy_api.latest with minimal io/ui/Types/Input ──
# 若真实可导入则优先用真实（需 torch/av 已 mock），否则用桩
use_real = False
try:
    # 尝试导入真实 latest 的 io/ui/Types（此时 torch/av 已 mock，应该成功）
    from comfy_api.latest import io as real_io, ui as real_ui, Types as real_Types, Input as real_Input
    # 验证关键属性存在
    assert hasattr(real_io, "Schema") and hasattr(real_io, "ComfyNode")
    io = real_io
    ui = real_ui
    Types = real_Types
    Input = real_Input
    use_real = True
except Exception as e:
    # 回退到桩
    # print(f"fallback mock comfy_api.latest: {e}")
    mock_latest = types.ModuleType("comfy_api.latest")
    # --- io ---
    mock_io = types.ModuleType("comfy_api.latest.io")
    class _FakeComfyNode:
        hidden = types.SimpleNamespace(prompt=None, extra_pnginfo=None)
    # 简易 Schema
    class _Schema:
        def __init__(self, node_id, display_name=None, category=None, essentials_category=None, description=None, inputs=None, hidden=None, is_output_node=False, outputs=None, **kw):
            self.node_id = node_id
            self.display_name = display_name
            self.category = category
            self.essentials_category = essentials_category
            self.description = description
            self.inputs = inputs or []
            self.hidden = hidden or []
            self.is_output_node = is_output_node
            self.outputs = outputs or []
            for k, v in kw.items():
                setattr(self, k, v)
    def _make_input(id, display_name=None, optional=False, tooltip=None, default=None, **kw):
        ns = types.SimpleNamespace(id=id, display_name=display_name, optional=optional, tooltip=tooltip, default=default)
        for k, v in kw.items():
            setattr(ns, k, v)
        return ns
    class _Boolean:
        Type = bool
        class Input:
            def __init__(self, id, display_name=None, optional=False, tooltip=None, default=None, label_on=None, label_off=None, **kw):
                self.id = id
                self.display_name = display_name
                self.optional = optional
                self.tooltip = tooltip
                self.default = default
                self.label_on = label_on
                self.label_off = label_off
                for k, v in kw.items():
                    setattr(self, k, v)
    class _String:
        Type = str
        class Input:
            def __init__(self, id, display_name=None, optional=False, tooltip=None, default=None, **kw):
                self.id = id
                self.display_name = display_name
                self.optional = optional
                self.tooltip = tooltip
                self.default = default
                for k, v in kw.items():
                    setattr(self, k, v)
    class _Video:
        Type = object
        class Input:
            def __init__(self, id, display_name=None, optional=False, tooltip=None, **kw):
                self.id = id
                self.display_name = display_name
                self.optional = optional
                self.tooltip = tooltip
                for k, v in kw.items():
                    setattr(self, k, v)
        class Output:
            def __init__(self, id=None, display_name=None, tooltip=None, **kw):
                self.id = id
                self.display_name = display_name
                self.tooltip = tooltip
    class _DynamicCombo:
        Type = dict
        class Option:
            def __init__(self, key, inputs):
                self.key = key
                self.inputs = inputs
        class Input:
            def __init__(self, id, options, display_name=None, optional=False, tooltip=None, extra_dict=None, **kw):
                self.id = id
                self.options = options
                self.display_name = display_name
                self.optional = optional
                self.tooltip = tooltip
                self.extra_dict = extra_dict
                for k, v in kw.items():
                    setattr(self, k, v)
    class _Hidden:
        prompt = types.SimpleNamespace(id="prompt")
        extra_pnginfo = types.SimpleNamespace(id="extra_pnginfo")
        def __getattr__(self, name):
            return types.SimpleNamespace(id=name)
    class _Float:
        Type = float
        class Input:
            def __init__(self, id, display_name=None, optional=False, tooltip=None, default=None, **kw):
                self.id = id
                self.display_name = display_name
                self.optional = optional
                self.tooltip = tooltip
                self.default = default
                for k, v in kw.items():
                    setattr(self, k, v)
    # 组装 mock_io
    mock_io.Schema = _Schema
    mock_io.ComfyNode = _FakeComfyNode
    mock_io.Boolean = _Boolean
    mock_io.String = _String
    mock_io.Video = _Video
    mock_io.DynamicCombo = _DynamicCombo
    mock_io.Hidden = _Hidden()
    mock_io.Float = _Float
    class _FolderType:
        output = "output"
        temp = "temp"
        input = "input"
    mock_io.FolderType = _FolderType
    # NodeOutput
    class _NodeOutput:
        def __init__(self, *args, ui=None):
            self.args = args
            self.result = args
            self.ui = ui
        def __iter__(self):
            return iter(self.args)
    mock_io.NodeOutput = _NodeOutput
    # --- ui ---
    mock_ui = types.ModuleType("comfy_api.latest.ui")
    class _FolderType:
        output = "output"
        temp = "temp"
        input = "input"
    class _SavedResult:
        def __init__(self, filename, subfolder, type):
            self.filename = filename
            self.subfolder = subfolder
            self.type = type
    class _PreviewVideo:
        def __init__(self, results):
            self.results = results
        def __getitem__(self, key):
            return self.results
    mock_ui.PreviewVideo = _PreviewVideo
    mock_ui.SavedResult = _SavedResult
    mock_ui.FolderType = _FolderType
    # --- Types ---
    mock_Types = types.SimpleNamespace()
    class _VideoContainer:
        MP4 = "mp4"
        MKV = "mkv"
        WEBM = "webm"
        AUTO = "auto"
        def __init__(self, v):
            self.v = v
        @classmethod
        def get_extension(cls, v):
            if isinstance(v, cls):
                v = v.v
            if v in ("mp4", "auto"):
                return "mp4"
            if v == "mkv":
                return "mkv"
            if v == "webm":
                return "webm"
            return "mp4"
        def __str__(self):
            return self.v
    class _VideoCodec:
        AUTO = "auto"
        H264 = "h264"
        AV1 = "av1"
        def __init__(self, v):
            self.v = v
    mock_Types.VideoContainer = _VideoContainer
    mock_Types.VideoCodec = _VideoCodec
    # --- Input ---
    mock_Input = types.SimpleNamespace(Video=object)
    mock_Input.Video = object
    # 注入 sys.modules
    sys.modules["comfy_api"] = types.ModuleType("comfy_api")
    sys.modules["comfy_api.latest"] = mock_latest
    mock_latest.io = mock_io
    mock_latest.ui = mock_ui
    mock_latest.Types = mock_Types
    mock_latest.Input = mock_Input
    io = mock_io
    ui = mock_ui
    Types = mock_Types
    Input = mock_Input

# ── mock comfy_extras.nodes_video.save_video_preview ──
_preview_calls = []
# 若真实模块已存在则补丁，否则 mock
try:
    import comfy_extras.nodes_video as _real_nodes_video
    nodes_video = _real_nodes_video
except Exception:
    comfy_extras = types.ModuleType("comfy_extras")
    nodes_video = types.ModuleType("comfy_extras.nodes_video")
    sys.modules["comfy_extras"] = comfy_extras
    sys.modules["comfy_extras.nodes_video"] = nodes_video

_orig_preview = getattr(nodes_video, "save_video_preview", None)
def fake_save_video_preview(video):
    _preview_calls.append(video)
    try:
        return ui.PreviewVideo([ui.SavedResult("preview_00001_.mp4", "", getattr(ui.FolderType, "temp", "temp"))])
    except Exception:
        return ui.PreviewVideo([ui.SavedResult("preview_00001_.mp4", "", "temp")]) if hasattr(ui, "PreviewVideo") else {"gifs": []}
nodes_video.save_video_preview = fake_save_video_preview
if "comfy_extras.nodes_video" not in sys.modules:
    sys.modules["comfy_extras.nodes_video"] = nodes_video
if "comfy_extras" not in sys.modules:
    sys.modules["comfy_extras"] = types.ModuleType("comfy_extras")

# ── load module under test ──
spec = importlib.util.spec_from_file_location(
    "sfnodes.nodes.video.save_video",
    os.path.join(root, "nodes", "video", "save_video.py"),
)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

# ── mock Video ──
class FakeVideo:
    def __init__(self, w=64, h=64):
        self._w = w
        self._h = h
        self.save_calls = []
    def get_dimensions(self):
        return self._w, self._h
    def save_to(self, path, format=None, codec=None, metadata=None, crf=None, **kw):
        self.save_calls.append((path, format, codec, metadata, crf))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"fake video")

failures = []
def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")

cls = mod.SFSaveVideoSwitchable
schema = cls.define_schema()
check("schema node_id", schema.node_id == "SFSaveVideoSwitchable")
check("schema display_name", schema.display_name == "SF Save Video (Switchable)")
check("schema is_output_node", getattr(schema, "is_output_node", False) is True)
input_ids = [i.id for i in schema.inputs]
check("inputs contain video", "video" in input_ids)
check("inputs contain filename_prefix", "filename_prefix" in input_ids)
check("inputs contain format", "format" in input_ids)
check("inputs contain save_enabled", "save_enabled" in input_ids)
check("inputs contain overwrite", "overwrite" in input_ids)
for inp in schema.inputs:
    if inp.id == "save_enabled":
        check("save_enabled default True", getattr(inp, "default", None) is True)
    if inp.id == "overwrite":
        check("overwrite default False", getattr(inp, "default", None) is False)
check("hidden prompt", any("prompt" in str(h) for h in schema.hidden))
check("hidden extra_pnginfo", any("extra_pnginfo" in str(h) for h in schema.hidden))
check("CATEGORY sfnodes/video", getattr(mod, "_CATEGORY", None) == "sfnodes/video" or schema.category == "sfnodes/video")
check("DESCRIPTION non-empty", isinstance(cls.DESCRIPTION, str) and len(cls.DESCRIPTION) > 10)

# hidden 赋值
cls.hidden = types.SimpleNamespace(prompt={"prompt": 1}, extra_pnginfo={"workflow": {"w": 1}})

# 清理
import shutil
for d in [tmp_output, tmp_temp]:
    for r, dirs, files in os.walk(d):
        for f in files:
            os.remove(os.path.join(r, f))

vid = FakeVideo(128, 64)
out = cls.execute(vid, "video/ComfyUI", "mp4", {"codec": "h264"}, save_enabled=True, overwrite=False)
check("save_enabled True returns NodeOutput", hasattr(out, "args") or hasattr(out, "result") or isinstance(out, tuple) or hasattr(out, "ui"))
try:
    result_video = out.args[0] if hasattr(out, "args") else (out.result[0] if hasattr(out, "result") else out[0])
except Exception:
    result_video = None
check("video passthrough save True", result_video is vid)
check("save_to called once", len(vid.save_calls) == 1)
saved_path = vid.save_calls[0][0]
check("saved to output dir", tmp_output in saved_path)
check("saved ext mp4", saved_path.endswith(".mp4"))
check("file exists output", os.path.isfile(saved_path))

vid2 = FakeVideo(128, 64)
out2 = cls.execute(vid2, "video/ComfyUI", "mp4", {"codec": "h264"}, save_enabled=True, overwrite=False)
check("second save new file", vid2.save_calls[0][0] != saved_path)
check("second file exists", os.path.isfile(vid2.save_calls[0][0]))
check("counter incremented (_00002_)", "_00002_" in vid2.save_calls[0][0])

# overwrite prep
for r, dirs, files in os.walk(tmp_output):
    for f in files:
        if f.startswith("ComfyUI_"):
            os.remove(os.path.join(r, f))
vid_a = FakeVideo(64, 64)
out_a = cls.execute(vid_a, "video/ComfyUI", "mp4", {"codec": "h264"}, save_enabled=True, overwrite=False)
path_a = vid_a.save_calls[0][0]
check("overwrite prep file exists", os.path.isfile(path_a))
vid_b = FakeVideo(64, 64)
out_b = cls.execute(vid_b, "video/ComfyUI", "mp4", {"codec": "h264"}, save_enabled=True, overwrite=True)
check("overwrite True still saves", len(vid_b.save_calls) == 1)
check("overwrite file exists", os.path.isfile(vid_b.save_calls[0][0]))

# save_enabled=False
before_count = sum(len(files) for _, _, files in os.walk(tmp_output))
_preview_calls.clear()
vid3 = FakeVideo(32, 32)
out3 = cls.execute(vid3, "video/ComfyUI", "mp4", {"codec": "h264"}, save_enabled=False, overwrite=False)
after_count = sum(len(files) for _, _, files in os.walk(tmp_output))
check("save_enabled False not write output", before_count == after_count)
check("save_enabled False calls save_video_preview", len(_preview_calls) == 1 and _preview_calls[0] is vid3)
check("save_enabled False no save_to", len(vid3.save_calls) == 0)
try:
    rv3 = out3.args[0] if hasattr(out3, "args") else (out3.result[0] if hasattr(out3, "result") else out3[0])
except Exception:
    rv3 = None
check("video passthrough skip", rv3 is vid3)

# auto
vid4 = FakeVideo(16, 16)
cls.execute(vid4, "video/ComfyUI", {"format": "auto", "codec": {"codec": "h264"}}, None, save_enabled=True, overwrite=True)
check("auto h264 -> mp4", vid4.save_calls[0][0].endswith(".mp4"))
vid5 = FakeVideo(16, 16)
cls.execute(vid5, "video/ComfyUI", {"format": "auto", "codec": {"codec": "av1"}}, None, save_enabled=True, overwrite=True)
check("auto av1 -> webm", vid5.save_calls[0][0].endswith(".webm"))

# metadata
vid6 = FakeVideo(8, 8)
cls.execute(vid6, "video/ComfyUI", "mp4", {"codec": "h264"}, save_enabled=True, overwrite=True)
meta = vid6.save_calls[0][3]
check("metadata contains prompt", meta is not None and "prompt" in meta)

old_disable = cli_args.args.disable_metadata
cli_args.args.disable_metadata = True
vid7 = FakeVideo(8, 8)
cls.execute(vid7, "video/ComfyUI", "mp4", {"codec": "h264"}, save_enabled=True, overwrite=True)
check("disable_metadata -> None", vid7.save_calls[0][3] is None)
cli_args.args.disable_metadata = old_disable

print("\nFAILURES:", len(failures))
if failures:
    print(failures)
sys.exit(1 if failures else 0)
