# SFTextPreset 后端逻辑测试（Node/Python 直接运行：python tests/test_text_preset.py）
# 覆盖：INPUT_TYPES 结构、execute 的全局库优先/工作流回退/容错
import importlib.util
import os
import sys
import tempfile
import types

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

comfy = types.ModuleType("comfy")
node_typing = types.ModuleType("comfy.comfy_types")
node_typing_module = types.ModuleType("comfy.comfy_types.node_typing")
class IO:
    STRING = "STRING"
node_typing_module.IO = IO
comfy.comfy_types = node_typing
comfy.comfy_types.node_typing = node_typing_module
sys.modules["comfy"] = comfy
sys.modules["comfy.comfy_types"] = node_typing
sys.modules["comfy.comfy_types.node_typing"] = node_typing_module

# mock aiohttp.web + server.PromptServer（sf_utils.text_presets import 时注册路由用，
# test_prompt_preset.py 范式）
aiohttp = types.ModuleType("aiohttp")
web_mod = types.ModuleType("aiohttp.web")
class _FakeJsonResponse:
    def __init__(self, data):
        self.data = data
web_mod.json_response = lambda data, status=200: _FakeJsonResponse(data)
web_mod.Response = lambda status=200, text="": type("R", (), {"status": status})()
aiohttp.web = web_mod
sys.modules["aiohttp"] = aiohttp
sys.modules["aiohttp.web"] = web_mod


class _FakeRoutes:
    def __init__(self):
        self.handlers = {}
    def get(self, path):
        def deco(fn):
            self.handlers[path] = fn
            return fn
        return deco
    def post(self, path):
        return self.get(path)
    def delete(self, path):
        return self.get(path)


server_mod = types.ModuleType("server")
server_mod.PromptServer = type("PS", (), {"instance": type("I", (), {"routes": _FakeRoutes()})()})
sys.modules["server"] = server_mod

# mock folder_paths：隔离 user 目录（避免回退路径在仓库内创建 user/sfnodes）
_tmp_user = tempfile.mkdtemp(prefix="sf_tpn_test_")
fp = types.ModuleType("folder_paths")
fp.get_user_directory = lambda: _tmp_user
sys.modules["folder_paths"] = fp

# 包上下文（相对 import `from ...sf_utils import text_presets` 需要）
pkg = types.ModuleType("sfnodes")
pkg.__path__ = [root]
nodes_pkg = types.ModuleType("sfnodes.nodes")
nodes_pkg.__path__ = [os.path.join(root, "nodes")]
text_pkg = types.ModuleType("sfnodes.nodes.text")
text_pkg.__path__ = [os.path.join(root, "nodes", "text")]
sys.modules["sfnodes"] = pkg
sys.modules["sfnodes.nodes"] = nodes_pkg
sys.modules["sfnodes.nodes.text"] = text_pkg

spec = importlib.util.spec_from_file_location(
    "sfnodes.nodes.text.text_preset",
    os.path.join(root, "nodes", "text", "text_preset.py"),
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

node = mod.SFTextPreset()
check("CATEGORY", node.CATEGORY == "sfnodes/text")
check("DESCRIPTION 存在", isinstance(node.DESCRIPTION, str) and len(node.DESCRIPTION) > 0)

it = node.INPUT_TYPES()
required = it["required"]
check("INPUT_TYPES 含 preset", "preset" in required)
check("preset 初始选项为空占位", required["preset"][0] == [""])
check("INPUT_TYPES 含 presets_json", "presets_json" in required)
check("presets_json 隐藏", required["presets_json"][1].get("display") == "hidden")
check("presets_json 默认空数组", required["presets_json"][1].get("default") == "[]")
check("INPUT_TYPES 含 text_override", "text_override" in required)
check("text_override 隐藏", required["text_override"][1].get("display") == "hidden")
check("text_override 默认空串", required["text_override"][1].get("default") == "")
check("返回类型 text+preset_name", node.RETURN_TYPES == ("STRING", "STRING") and node.RETURN_NAMES == ("text", "preset_name"))
check("VALIDATE_INPUTS 跳过 combo 校验", node.VALIDATE_INPUTS(preset="a", presets_json="[]") is True)

# 全局库优先：写入全局库后命中全局文本而非工作流载体
text_presets = mod.text_presets
text_presets.save_presets([{"name": "G", "text": "global"}])
t, n = node.execute("G", '[{"name": "G", "text": "workflow"}]')
check("全局库优先命中", t == "global" and n == "G")

# 草稿优先：text_override 非空时覆盖全局库与工作流载体（仅本节点输出）
t, n = node.execute("G", '[{"name": "G", "text": "workflow"}]', "draft text")
check("草稿优先于全局库", t == "draft text" and n == "G")
t, n = node.execute("WF", '[{"name": "WF", "text": "workflow"}]', "draft wf")
check("草稿优先于工作流回退", t == "draft wf" and n == "WF")
t, n = node.execute("G", "[]", "")
check("空草稿不生效", t == "global" and n == "G")
t, n = node.execute("G", "[]")
check("缺省 text_override 容错", t == "global" and n == "G")

# 工作流回退：全局库未命中时解析 presets_json
json_data = '[{"name": "A", "text": "hello"}, {"name": "B", "text": "world"}]'
t, n = node.execute("B", json_data)
check("按名命中返回文本", t == "world" and n == "B")
t, n = node.execute("A", json_data)
check("命中第一个预设", t == "hello" and n == "A")
t, n = node.execute("Nope", json_data)
check("未命中返回空文本", t == "" and n == "Nope")
t, n = node.execute("", json_data)
check("空选择返回空文本", t == "" and n == "")
t, n = node.execute("B", "")
check("空 presets_json 容错", t == "" and n == "B")
t, n = node.execute("B", None)
check("None presets_json 容错", t == "" and n == "B")
t, n = node.execute("B", "not json{{{")
check("非法 JSON 容错", t == "" and n == "B")
t, n = node.execute("B", '{"not": "list"}')
check("非数组 JSON 容错", t == "" and n == "B")
t, n = node.execute("B", '[{"name": 123, "text": 456}]')
check("name 非字符串时按 str 比较", t == "" and n == "B")
t, n = node.execute(123, '[{"name": "123", "text": "num"}]')
check("preset 非字符串输入", t == "num" and n == "123")
t, n = node.execute("B", '[{"name": "B", "text": 456}]')
check("text 非字符串转字符串", t == "456" and n == "B")
t, n = node.execute("B", '[{"name": "B"}]')
check("缺 text 字段输出空", t == "" and n == "B")

# 缺省 presets_json（隐藏输入首轮可能缺键，platform §1）
t, n = node.execute("B")
check("缺省 presets_json 容错", t == "" and n == "B")

# 清理全局库缓存，避免影响同进程后续断言
text_presets.save_presets([])

print("\nFAILURES:", len(failures))
sys.exit(1 if failures else 0)
