# SFPromptReader 后端测试（Python 直接运行：python tests/test_prompt_reader.py）
# 覆盖：
#   - 结构：SFPromptReader 类、CATEGORY、DESCRIPTION、INPUT_TYPES（image combo +
#     optional filename）、RETURN_TYPES、FUNCTION、注册键、IS_CHANGED、
#     VALIDATE_INPUTS
#   - extract_positive_from_comfy_prompt：基础 encode、SDXL 双文本、Combine、
#     StringConcatenate、sf 自家节点（SFPromptTags / SFValueDropdown /
#     SFTextPreset / SFAnythingIndexSwitch / SFPauseText / SFPromptList /
#     SFPromptPreset）、Pixaroma 生态兼容（Switch / Stack / Multi / Pack /
#     Dropdown / FromList / Prompt / SwitchSource / rgthree Any Switch）、
#     自追链、循环防护、深度上限、去重
#   - extract_positive_from_a1111：正负分隔、参数行分隔
#   - read_png_text_chunks + read_prompt_from_image：真实 PNG（PIL 写 tEXt）
#   - resolve_input_image_name：裸名 / 注解 / 子目录 / 无匹配
import importlib.util
import json
import os
import re
import sys
import tempfile
import types

from PIL import Image
from PIL.PngImagePlugin import PngInfo

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

failures = []

def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")

# ── 临时环境 ──
tmp = tempfile.mkdtemp(prefix="sf_prompt_reader_test_")
input_dir = os.path.join(tmp, "input")
os.makedirs(input_dir, exist_ok=True)
output_dir = os.path.join(tmp, "output")
os.makedirs(output_dir, exist_ok=True)
temp_dir = os.path.join(tmp, "temp")
os.makedirs(temp_dir, exist_ok=True)


def _strip_annotation(name):
    return re.sub(r"\s*\[(?:input|output|temp)\]\s*$", "", str(name)).replace("\\", "/").lstrip("/")


def get_annotated_filepath(name):
    return os.path.join(input_dir, _strip_annotation(name))


# folder_paths mock（在加载 helpers 之前注入，让模块拿到 mock 而非 None）
_IMG_VIDEO_EXTS = {
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff", ".tif",
    ".mp4", ".m4v", ".mov", ".webm", ".mkv",
}
fp_mock = types.ModuleType("folder_paths")
fp_mock.get_annotated_filepath = get_annotated_filepath
fp_mock.get_input_directory = lambda: input_dir
fp_mock.get_output_directory = lambda: output_dir
fp_mock.get_temp_directory = lambda: temp_dir
fp_mock.filter_files_content_types = lambda files, content_types: [
    f for f in files if os.path.splitext(f)[1].lower() in _IMG_VIDEO_EXTS
]
fp_mock.get_save_image_path = lambda prefix, base, w, h: (base, "x", 1, "", "")
sys.modules["folder_paths"] = fp_mock

# 注册 sfnodes 包结构（相对导入 from ...sf_utils.common import AnyType）
pkg = types.ModuleType("sfnodes"); pkg.__path__ = [root]; sys.modules["sfnodes"] = pkg
pkg2 = types.ModuleType("sfnodes.nodes"); pkg2.__path__ = [os.path.join(root, "nodes")]; sys.modules["sfnodes.nodes"] = pkg2
pkg3 = types.ModuleType("sfnodes.nodes.text"); pkg3.__path__ = [os.path.join(root, "nodes", "text")]; sys.modules["sfnodes.nodes.text"] = pkg3

# helpers 纯逻辑模块
spec_utils = importlib.util.spec_from_file_location(
    "sf_utils_prompt_reader",
    os.path.join(root, "sf_utils", "prompt_reader.py"),
)
helpers = importlib.util.module_from_spec(spec_utils)
sys.modules[spec_utils.name] = helpers
spec_utils.loader.exec_module(helpers)

# 节点类（含相对导入）
spec = importlib.util.spec_from_file_location(
    "sfnodes.nodes.text.prompt_reader",
    os.path.join(root, "nodes", "text", "prompt_reader.py"),
)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

# 路由模块（副作用注册依赖真实 server，导入不应抛异常）
spec_r = importlib.util.spec_from_file_location(
    "sfnodes.nodes.text.prompt_reader_routes",
    os.path.join(root, "nodes", "text", "prompt_reader_routes.py"),
)
mod_r = importlib.util.module_from_spec(spec_r)
sys.modules[spec_r.name] = mod_r
spec_r.loader.exec_module(mod_r)

# ── helpers ──
def write_png(path, prompt_json=None, parameters=None):
    """写一个带 tEXt chunks 的 1x1 PNG。"""
    info = PngInfo()
    if prompt_json is not None:
        info.add_text("prompt", prompt_json if isinstance(prompt_json, str) else json.dumps(prompt_json))
    if parameters is not None:
        info.add_text("parameters", parameters)
    Image.new("RGB", (1, 1), (255, 255, 255)).save(path, "PNG", pnginfo=info)
    return path


def enc(nid, text, cls="CLIPTextEncode"):
    return {nid: {"class_type": cls, "inputs": {"text": text}}}


def node(nid, cls, inputs):
    return {nid: {"class_type": cls, "inputs": inputs}}


def sampler(nid, positive):
    return {nid: {"class_type": "KSampler", "inputs": {"positive": positive}}}


def extract(prompt):
    return helpers.extract_positive_from_comfy_prompt(json.dumps(prompt))

# ── 基础提取 ──
w = {**sampler("1", ["2", 0]), **enc("2", "masterpiece, best quality")}
r = extract(w)
check("基础: KSampler -> CLIPTextEncode", r == "masterpiece, best quality")

w = {**sampler("1", ["2", 0]), **enc("2", "hello"), "3": {"class_type": "KSampler", "inputs": {"positive": ["2", 0]}}}
r = extract(w)
check("多个 sampler 共享 encode 去重", r == "hello")

w = {**sampler("1", ["2", 0]), "2": {"class_type": "CLIPTextEncodeSDXL", "inputs": {"text_g": ["3", 0], "text_l": ["4", 0]}},
     **enc("3", "g-prompt"), **enc("4", "l-prompt")}
r = extract(w)
check("SDXL 双文本: g + l 用段落分隔", r == "g-prompt\n\nl-prompt")

w = {**sampler("1", ["2", 0]),
     "2": {"class_type": "ConditioningCombine", "inputs": {"conditioning_1": ["3", 0], "conditioning_2": ["4", 0]}},
     **enc("3", "a"), **enc("4", "b")}
r = extract(w)
check("ConditioningCombine 两个上游", r == "a\n\nb")

w = {**sampler("1", ["2", 0]),
     "2": {"class_type": "SFTextConcatenate", "inputs": {"delimiter": ", ", "clean_whitespace": True, "text_1": ["3", 0], "text_2": ["4", 0]}},
     **enc("3", "x"), **enc("4", "y")}
r = extract(w)
check("SFTextConcatenate 链", r == "x\n\ny")

w = {**sampler("1", ["2", 0]),
     "2": {"class_type": "CLIPTextEncode", "inputs": {"text": ["3", 0]}},
     "3": {"class_type": "SFPromptTags", "inputs": {
         "PromptState": json.dumps({"text": "@tag expanded prompt", "order": "mine", "sep": ", "})}}}
r = extract(w)
check("SFPromptTags: PromptState 直读", r == "@tag expanded prompt")

w = {**sampler("1", ["2", 0]),
     "2": {"class_type": "CLIPTextEncode", "inputs": {"text": ["3", 0]}},
     "3": {"class_type": "SFPromptTags", "inputs": {
         "PromptState": json.dumps({"text": "mine", "order": "wired", "sep": " | "}),
         "text_in": ["4", 0]}},
     **enc("4", "wired-part")}
r = extract(w)
check("SFPromptTags: wired 顺序拼接", r == "wired-part | mine")

w = {**sampler("1", ["2", 0]),
     "2": {"class_type": "CLIPTextEncode", "inputs": {"text": ["3", 0]}},
     "3": {"class_type": "SFValueDropdown", "inputs": {
         "DropdownState": json.dumps({"type": "text", "value": "trigger words"})}}}
r = extract(w)
check("SFValueDropdown: text 类型", r == "trigger words")

w = {**sampler("1", ["2", 0]),
     "2": {"class_type": "CLIPTextEncode", "inputs": {"text": ["3", 0]}},
     "3": {"class_type": "SFValueDropdown", "inputs": {
         "DropdownState": json.dumps({"type": "int", "value": 1024})}}}
r = extract(w)
check("SFValueDropdown: 非 text 类型不贡献", r is None)

w = {**sampler("1", ["2", 0]),
     "2": {"class_type": "CLIPTextEncode", "inputs": {"text": ["3", 0]}},
     "3": {"class_type": "SFValueDropdown", "inputs": {
         "DropdownState": json.dumps({"type": "text", "index": 1,
                                      "options": [{"name": "a", "value": "v0"}, {"name": "b", "value": "v1"}]})}}}
r = extract(w)
check("SFValueDropdown: full 形状", r == "v1")

w = {**sampler("1", ["2", 0]),
     "2": {"class_type": "CLIPTextEncode", "inputs": {"text": ["3", 0]}},
     "3": {"class_type": "SFTextPreset", "inputs": {
         "preset": "portrait", "presets_json": json.dumps(
             [{"name": "portrait", "text": "portrait style prompt"}, {"name": "b", "text": "other"}])}}}
r = extract(w)
check("SFTextPreset: 选中预设文本", r == "portrait style prompt")

w = {**sampler("1", ["2", 0]),
     "2": {"class_type": "CLIPTextEncode", "inputs": {"text": ["3", 0]}},
     "3": {"class_type": "SFTextPreset", "inputs": {"preset": "missing", "presets_json": "[]"}}}
r = extract(w)
check("SFTextPreset: 未命中返回 None", r is None)

w = {**sampler("1", ["2", 0]),
     "2": {"class_type": "CLIPTextEncode", "inputs": {"text": ["3", 0]}},
     "3": {"class_type": "SFAnythingIndexSwitch", "inputs": {"index": 2, "value0": ["4", 0], "value2": ["5", 0]}},
     **enc("4", "lane0"), **enc("5", "lane2")}
r = extract(w)
check("SFAnythingIndexSwitch: index=2 -> value2", r == "lane2")

w = {**sampler("1", ["2", 0]),
     "2": {"class_type": "CLIPTextEncode", "inputs": {"text": ["3", 0]}},
     "3": {"class_type": "SFPauseText", "inputs": {
         "PauseState": json.dumps({"mode": "continue", "text": "edited by user"})}}}
r = extract(w)
check("SFPauseText: continue 读盒子文本", r == "edited by user")

w = {**sampler("1", ["2", 0]),
     "2": {"class_type": "CLIPTextEncode", "inputs": {"text": ["3", 0]}},
     "3": {"class_type": "SFPauseText", "inputs": {"PauseState": json.dumps({"mode": "pass", "text": "box"})}},
     "4": {"class_type": "CLIPTextEncode", "inputs": {"text": ["5", 0]}},
     **enc("5", "model-text")}
w["3"]["inputs"]["text"] = ["4", 0]
r = extract(w)
check("SFPauseText: pass 模式跟随 text 输入", r == "model-text")

w = {**sampler("1", ["2", 0]),
     "2": {"class_type": "CLIPTextEncode", "inputs": {"text": ["3", 0]}},
     "3": {"class_type": "SFPromptList", "inputs": {
         "multiline_text": "row one\nrow two", "prepend_text": "PRE ", "append_text": " POST"}}}
r = extract(w)
check("SFPromptList: 行拆分 + 前后缀", r == "PRE row one POST\n\nPRE row two POST")

w = {**sampler("1", ["2", 0]),
     "2": {"class_type": "CLIPTextEncode", "inputs": {"text": ["3", 0]}},
     "3": {"class_type": "SFPromptList", "inputs": {
         "multiline_text": "row one\n   \nrow two\n\n", "prepend_text": "", "append_text": ""}}}
r = extract(w)
check("SFPromptList: skip_empty 默认 True 过滤空白行", r == "row one\n\nrow two")

w = {**sampler("1", ["2", 0]),
     "2": {"class_type": "CLIPTextEncode", "inputs": {"text": ["3", 0]}},
     "3": {"class_type": "SFPromptList", "inputs": {
         "multiline_text": "row one\n\nrow two", "skip_empty": False}}}
r = extract(w)
check("SFPromptList: skip_empty=False 保留空行（空段落）", r == "row one\n\n\n\nrow two")

w = {**sampler("1", ["2", 0]),
     "2": {"class_type": "CLIPTextEncode", "inputs": {"text": ["3", 0]}},
     "3": {"class_type": "SFPromptPreset", "inputs": {"input_text": "base prompt [a, b]", "seed": 5}}}
r = extract(w)
check("SFPromptPreset: 基础文本（预设部分不可恢复）", r == "base prompt [a, b]")

# ── Pixaroma 生态兼容 ──
w = {**sampler("1", ["2", 0]),
     "2": {"class_type": "CLIPTextEncode", "inputs": {"text": ["3", 0]}},
     "3": {"class_type": "PixaromaPromptStack", "inputs": {
         "PromptStackState": json.dumps({
             "version": 1,
             "rows": [{"enabled": True, "label": "a", "text": "stack1"},
                      {"enabled": False, "label": "b", "text": "skip"},
                      {"enabled": True, "label": "c", "text": "stack2,"}],
             "separator": ", "})}}}
r = extract(w)
check("PixaromaPromptStack: 启用的行按分隔符拼接", r == "stack1, stack2")

w = {**sampler("1", ["2", 0]),
     "2": {"class_type": "CLIPTextEncode", "inputs": {"text": ["3", 0]}},
     "3": {"class_type": "PixaromaDropdown", "inputs": {
         "DropdownState": json.dumps({"type": "text", "value": "pix-value"})}}}
r = extract(w)
check("PixaromaDropdown: text 类型", r == "pix-value")

w = {**sampler("1", ["2", 0]),
     "2": {"class_type": "CLIPTextEncode", "inputs": {"text": ["3", 0]}},
     "3": {"class_type": "PixaromaPrompt", "inputs": {
         "PromptState": json.dumps({"text": "pix prompt", "order": "mine", "sep": ", "})}}}
r = extract(w)
check("PixaromaPrompt: PromptState 直读", r == "pix prompt")

w = {**sampler("1", ["2", 0]),
     "2": {"class_type": "CLIPTextEncode", "inputs": {"text": ["3", 0]}},
     "3": {"class_type": "PixaromaSwitch", "inputs": {"SwitchState": "3", "input_1": ["4", 0], "input_3": ["5", 0]}},
     **enc("4", "in1"), **enc("5", "in3")}
r = extract(w)
check("PixaromaSwitch: SwitchState 选中行", r == "in3")

w = {**sampler("1", ["2", 0]),
     "2": {"class_type": "CLIPTextEncode", "inputs": {"text": ["3", 0]}},
     "3": {"class_type": "Any Switch (rgthree)", "inputs": {"any_2": ["4", 0], "any_1": ["5", 0]}},
     **enc("4", "any2"), **enc("5", "any1")}
r = extract(w)
check("rgthree Any Switch: 数字序最小连线的 any_NN", r == "any1")

w = {**sampler("1", ["2", 0]),
     "2": {"class_type": "CLIPTextEncode", "inputs": {"text": ["3", 0]}},
     "3": {"class_type": "PixaromaPromptMulti", "inputs": {
         "PromptMultiState": json.dumps({"version": 2, "mode": "queue", "activePrompt": "multi active", "rowTexts": ["x"]})}}}
r = extract(w)
check("PixaromaPromptMulti: activePrompt 直读", r == "multi active")

w = {**sampler("1", ["2", 0]),
     "2": {"class_type": "CLIPTextEncode", "inputs": {"text": ["3", 0]}},
     "3": {"class_type": "PixaromaPromptFromList", "inputs": {"index": 2, "prompts": ["5", 0]}},
     "5": {"class_type": "PixaromaPromptMulti", "inputs": {
         "PromptMultiState": json.dumps({"version": 2, "mode": "list", "activePrompt": "",
                                         "rowTexts": ["row1", "row2", "row3"]})}}}
r = extract(w)
check("PixaromaPromptFromList: rowTexts 索引", r == "row2")

w = {**sampler("1", ["2", 0]),
     "2": {"class_type": "CLIPTextEncode", "inputs": {"text": ["3", 0]}},
     "3": {"class_type": "PixaromaPromptPack", "inputs": {
         "PromptPackState": json.dumps({"version": 1, "activePrompt": "pack prompt"})}}}
r = extract(w)
check("PixaromaPromptPack: activePrompt 直读", r == "pack prompt")

w = {**sampler("1", ["2", 0]),
     "2": {"class_type": "CLIPTextEncode", "inputs": {"text": ["3", 0]}},
     "3": {"class_type": "PixaromaSwitchSource", "inputs": {
         "SwitchSourceState": json.dumps({"version": 1, "active": "A", "rows": 2}),
         "a_1": ["4", 0], "b_1": ["5", 0]}},
     **enc("4", "sideA"), **enc("5", "sideB")}
r = extract(w)
check("PixaromaSwitchSource: 从 output 槽 origin_slot 选行", r == "sideA")

# ── 鲁棒性 ──
w = {**sampler("1", ["2", 0]), "2": {"class_type": "CLIPTextEncode", "inputs": {"text": ["2", 0]}}}
r = extract(w)
check("自环不炸", r is None)

w = {**sampler("1", ["2", 0]), "2": {"class_type": "CLIPTextEncode", "inputs": {"text": ["3", 0]}},
     "3": {"class_type": "CLIPTextEncode", "inputs": {"text": ["2", 0]}}}
r = extract(w)
check("互环不炸", r is None)

# 深度上限：22 层文本链接链（键从 100 起，避免覆盖 sampler 的 "1"/"2"；
# 最深层节点 depth 23 < 24 上限，若截断逻辑破坏则取不到 "deep"）
chain = {}
for i in range(22):
    chain[str(100 + i)] = {"class_type": "CLIPTextEncode", "inputs": {"text": [str(101 + i), 0]}}
chain["122"] = {"class_type": "CLIPTextEncode", "inputs": {"text": "deep"}}
w = {**sampler("1", ["2", 0]), "2": {"class_type": "CLIPTextEncode", "inputs": {"text": ["100", 0]}}, **chain}
r = extract(w)
check("深度上限: 22 层链不炸", r == "deep")

w = {**sampler("1", ["2", 0]), **enc("2", "same"), **enc("3", "same")}
w["1"]["inputs"]["positive"] = ["2", 0]
w["3"] = {"class_type": "KSampler", "inputs": {"positive": ["2", 0]}}
r = extract(w)
check("去重保持顺序", r == "same")

r = helpers.extract_positive_from_comfy_prompt("not json")
check("非法 JSON 返回 None", r is None)

w = {"1": {"class_type": "VAELoader", "inputs": {}}}
r = extract(w)
check("无 sampler 返回 None", r is None)

# ── A1111 ──
r = helpers.extract_positive_from_a1111("masterpiece, cat\nNegative prompt: ugly\nSteps: 20, Sampler: Euler")
check("A1111: 负向分隔", r == "masterpiece, cat")
r = helpers.extract_positive_from_a1111("a pretty girl\nSteps: 20, Sampler: Euler, CFG scale: 7")
check("A1111: 参数行分隔", r == "a pretty girl")
r = helpers.extract_positive_from_a1111("no markers here at all")
check("A1111: 无标记整段", r == "no markers here at all")
r = helpers.extract_positive_from_a1111("")
check("A1111: 空串 None", r is None)

# ── 真实 PNG 读取 ──
png_path = os.path.join(input_dir, "gen1.png")
write_png(png_path, prompt_json={**sampler("1", ["2", 0]), **enc("2", "from real png")})
res = helpers.read_prompt_from_image(png_path)
check("read_prompt_from_image: comfyui 来源", res == {"found": True, "text": "from real png", "source": "comfyui"})

a1111_path = os.path.join(input_dir, "a1111.png")
write_png(a1111_path, parameters="dog portrait\nNegative prompt: blur\nSteps: 20")
res = helpers.read_prompt_from_image(a1111_path)
check("read_prompt_from_image: a1111 来源", res == {"found": True, "text": "dog portrait", "source": "a1111"})

plain_path = os.path.join(input_dir, "plain.png")
Image.new("RGB", (1, 1), (0, 0, 0)).save(plain_path, "PNG")
res = helpers.read_prompt_from_image(plain_path)
check("read_prompt_from_image: 无元数据", res.get("found") is False and "metadata" in res.get("message", ""))

res = helpers.read_prompt_from_image(os.path.join(input_dir, "nope.png"))
check("read_prompt_from_image: 不存在", res.get("found") is False)

# ── 自追链（SFPromptReader 嵌套）──
inner_path = os.path.join(input_dir, "inner.png")
write_png(inner_path, prompt_json={**sampler("1", ["2", 0]), **enc("2", "from inner png")})
outer_path = os.path.join(input_dir, "outer.png")
write_png(outer_path, prompt_json={**sampler("1", ["2", 0]),
                                   "2": {"class_type": "SFPromptReader", "inputs": {"image": "inner.png"}}})
res = helpers.read_prompt_from_image(outer_path)
check("自追链: SFPromptReader -> 源图元数据", res == {"found": True, "text": "from inner png", "source": "comfyui"})

res = helpers.read_prompt_from_image(outer_path.replace("outer", "outer2") if False else outer_path)
outer_missing = os.path.join(input_dir, "outer_missing.png")
write_png(outer_missing, prompt_json={**sampler("1", ["2", 0]),
                                      "2": {"class_type": "SFPromptReader", "inputs": {"image": "deleted.png"}}})
res = helpers.read_prompt_from_image(outer_missing)
check("自追链: 源图缺失给出专属提示", res.get("found") is False and "source image" in res.get("message", ""))

# ── 视频元数据（真实 ffmpeg 生成；不可用时跳过）──
import shutil
import subprocess

_ffmpeg = shutil.which("ffmpeg")


def _make_video(video_path, meta_entries, extra_args):
    """用 ffmpeg 生成带元数据的测试视频。失败返回 False。"""
    meta_file = os.path.join(tmp, os.path.basename(video_path) + ".meta.txt")
    with open(meta_file, "w") as f:
        for k, v in meta_entries:
            f.write(f"{k}={v}\n")
    cmd = [_ffmpeg, "-y", "-loglevel", "error",
           "-f", "lavfi", "-i", "color=c=red:s=64x64:d=0.3",
           "-f", "ffmetadata", "-i", meta_file,
           "-map", "0:v", "-map_metadata", "1", *extra_args, video_path]
    return subprocess.run(cmd, capture_output=True).returncode == 0


if _ffmpeg:
    mp4_path = os.path.join(input_dir, "gen1.mp4")
    mp4_ok = _make_video(mp4_path, [
        ("prompt", '{"3":{"class_type":"KSampler","inputs":{"positive":["4",0]}},"4":{"class_type":"CLIPTextEncode","inputs":{"text":"hello from mp4"}}}'),
        ("workflow", '{"nodes":[]}'),
    ], ["-movflags", "use_metadata_tags", "-c:v", "libx264", "-pix_fmt", "yuv420p"])
    if mp4_ok:
        chunks = helpers.read_video_text_chunks(mp4_path)
        check("mp4: 元数据键存在", "prompt" in chunks and "workflow" in chunks)
        check("mp4: 值是字符串", isinstance(chunks.get("prompt"), str) and "CLIPTextEncode" in chunks["prompt"])
        res = helpers.read_prompt_from_image(mp4_path)
        check("mp4: 完整链恢复 prompt", res == {"found": True, "text": "hello from mp4", "source": "comfyui"})
    else:
        print("SKIP: ffmpeg mp4 生成失败")

    webm_path = os.path.join(input_dir, "gen1.webm")
    webm_ok = _make_video(webm_path, [
        ("prompt", '{"3":{"class_type":"KSampler","inputs":{"positive":["4",0]}},"4":{"class_type":"CLIPTextEncode","inputs":{"text":"hello from webm"}}}'),
    ], ["-c:v", "libvpx-vp9", "-b:v", "100k"])
    if webm_ok:
        chunks = helpers.read_video_text_chunks(webm_path)
        check("webm: 大写键归一小写", "prompt" in chunks)
        res = helpers.read_prompt_from_image(webm_path)
        check("webm: 完整链恢复 prompt", res == {"found": True, "text": "hello from webm", "source": "comfyui"})
    else:
        print("SKIP: ffmpeg webm 生成失败")

    plain_mp4 = os.path.join(input_dir, "plain.mp4")
    if _make_video(plain_mp4, [], ["-c:v", "libx264", "-pix_fmt", "yuv420p"]):
        res = helpers.read_prompt_from_image(plain_mp4)
        check("mp4: 无元数据", res.get("found") is False and "metadata" in res.get("message", ""))
    else:
        print("SKIP: ffmpeg 无元数据 mp4 生成失败")

    mkv_path = os.path.join(input_dir, "gen1.mkv")
    if _make_video(mkv_path, [
        ("prompt", '{"3":{"class_type":"KSampler","inputs":{}}}'),
    ], ["-c:v", "libx264", "-pix_fmt", "yuv420p"]):
        chunks = helpers.read_video_text_chunks(mkv_path)
        check("mkv: EBML 解析同样生效", "prompt" in chunks)
    else:
        print("SKIP: ffmpeg mkv 生成失败")

    # 裸名视频 resolve：无同名 PNG 时落到 .mp4（优先级表末位）
    if _make_video(os.path.join(input_dir, "video_only.mp4"), [], ["-c:v", "libx264", "-pix_fmt", "yuv420p"]):
        check("resolve: 裸名视频", helpers.resolve_input_image_name("video_only") == "video_only.mp4")
    else:
        print("SKIP: ffmpeg video_only 生成失败")
else:
    print("SKIP: ffmpeg 不可用，视频测试跳过")

# ── resolve_input_image_name ──
os.makedirs(os.path.join(input_dir, "sub"), exist_ok=True)
write_png(os.path.join(input_dir, "BunnyExplorer.png"), prompt_json={})
write_png(os.path.join(input_dir, "sub", "cat.png"), prompt_json={})
check("resolve: 完整名直通", helpers.resolve_input_image_name("BunnyExplorer.png") == "BunnyExplorer.png")
check("resolve: 裸名找到 PNG", helpers.resolve_input_image_name("BunnyExplorer") == "BunnyExplorer.png")
check("resolve: 带注解直接命中原值直通", helpers.resolve_input_image_name("BunnyExplorer.png [input]") == "BunnyExplorer.png [input]")
check("resolve: 注解 + 裸名走剥离 fallback", helpers.resolve_input_image_name("BunnyExplorer [input]") == "BunnyExplorer.png")
check("resolve: 子目录裸名", helpers.resolve_input_image_name("cat") == "sub/cat.png")
check("resolve: 空名 None", helpers.resolve_input_image_name("") is None)
check("resolve: 不存在 None", helpers.resolve_input_image_name("ghost") is None)

# ── 目录列表（_list_media_recursive：input/output 切换用）──
# 放在 resolve 段之后：sub/ 子目录在此前已创建
check("list: input 根文件", "gen1.png" in mod_r._list_media_recursive("input")
      and "gen1.mp4" in mod_r._list_media_recursive("input"))
check("list: input 子目录递归", "sub/cat.png" in mod_r._list_media_recursive("input"))
check("list: input 含视频", "gen1.webm" in mod_r._list_media_recursive("input"))
check("list: input 排序", mod_r._list_media_recursive("input") == sorted(mod_r._list_media_recursive("input")))
# 非媒体文件被过滤
with open(os.path.join(input_dir, "notes.txt"), "w") as f:
    f.write("x")
check("list: 非媒体被过滤", "notes.txt" not in mod_r._list_media_recursive("input"))
os.remove(os.path.join(input_dir, "notes.txt"))
# output 目录
write_png(os.path.join(output_dir, "out1.png"), prompt_json={})
os.makedirs(os.path.join(output_dir, "subout"), exist_ok=True)
write_png(os.path.join(output_dir, "subout", "out2.png"), prompt_json={})
files_out = mod_r._list_media_recursive("output")
check("list: output 列出文件", "out1.png" in files_out and "subout/out2.png" in files_out)
check("list: output 不含 input 文件", "gen1.png" not in files_out)
check("list: 非法类型回落 input", "out1.png" not in mod_r._list_media_recursive("bogus"))

# ── 节点结构 ──
node = mod.SFPromptReader()
check("CATEGORY", node.CATEGORY == "sfnodes/text")
check("DESCRIPTION 存在", isinstance(node.DESCRIPTION, str) and len(node.DESCRIPTION) > 0)
it = node.INPUT_TYPES()
check("required.image 为 combo", it["required"]["image"][0] == sorted([]) or isinstance(it["required"]["image"], tuple))
check("optional.filename forceInput", it["optional"]["filename"][1].get("forceInput") is True)
check("RETURN_TYPES", node.RETURN_TYPES == ("STRING",) and node.RETURN_NAMES == ("text",))
check("FUNCTION = read", node.FUNCTION == "read")
check("OUTPUT_NODE", node.OUTPUT_NODE is True)
check("注册键", mod.NODE_CLASS_MAPPINGS == {"SFPromptReader": mod.SFPromptReader})
check("显示名", mod.NODE_DISPLAY_NAME_MAPPINGS == {"SFPromptReader": "SF Prompt Reader"})

# IS_CHANGED：存在文件 -> mtime:size；不存在 -> name:...；未选 -> nan
out = node.IS_CHANGED("gen1.png")
check("IS_CHANGED 有效文件", isinstance(out, str) and ":" in out)
out = node.IS_CHANGED("ghost.png")
check("IS_CHANGED 缺失文件", isinstance(out, str) and out.startswith("name:"))
out = node.IS_CHANGED("")
import math
check("IS_CHANGED 未选 nan", isinstance(out, float) and math.isnan(out))
out = node.IS_CHANGED("", filename="ghost")
check("IS_CHANGED wired 未解析", out == "unresolved:ghost")
check("VALIDATE_INPUTS 恒 True", node.VALIDATE_INPUTS("whatever", "x") is True)

# read()：真实 PNG
res = node.read("gen1.png")
check("read() 返回文本 + ui", res["result"] == ("from real png",) and res["ui"]["text"] == ["from real png"])
res = node.read("", filename="gen1")
check("read() wired 裸名", res["result"] == ("from real png",))
res = node.read("", filename="ghost")
check("read() wired 缺失提示", res["result"][0].startswith("Could not find an image"))
res = node.read("plain.png")
check("read() 无元数据提示", res["result"][0].startswith("No prompt metadata"))

# ── 收尾 ──
print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("ALL PASS")
