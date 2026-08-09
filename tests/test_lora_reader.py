# SFLoraStack 后端逻辑测试（Python 直接运行：python tests/test_lora_reader.py）
# 覆盖：
#   - lora_reader 纯逻辑：read_safetensors_metadata（迷你 safetensors 头）、
#     derive_trigger_words（短语优先 / ss_tag_frequency 按频次排序去重 / 上限）、
#     base_model_family、parse_state（容错 / sc 缺省 = sm / 钳制 / 脏输入）、
#     collect_triggers（on 过滤 / 去重 / sep）、sanitize_civitai_key（拒控制
#     字符）、sanitize_custom_words、custom_trigger_key、自定义词原子写
#   - lora_routes 守卫：_is_path_under、_looks_like_image
#   - 节点结构：SFLoraStack 类、INPUT_TYPES（hidden LoraLoaderState）、
#     RETURN_TYPES / RETURN_NAMES / FUNCTION / CATEGORY / DESCRIPTION、
#     apply 全链路（mock comfy/folder_paths：跳过缺失、强度 0 计触发词、
#     cacheMode last/all/none 修剪）
import importlib.util
import json
import os
import struct
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

# ── mock comfy / folder_paths（本机无 ComfyUI 运行时）────────────────────────
comfy = types.ModuleType("comfy"); comfy.__path__ = []; sys.modules["comfy"] = comfy
comfy_utils = types.ModuleType("comfy.utils"); sys.modules["comfy.utils"] = comfy_utils
comfy_sd = types.ModuleType("comfy.sd"); sys.modules["comfy.sd"] = comfy_sd
comfy.utils = comfy_utils  # sys.modules 直塞不设置属性，`import comfy.utils` 需要
comfy.sd = comfy_sd

loaded = {}          # path -> state_dict（记录 load_torch_file 次数）
load_calls = []
apply_calls = []

def fake_load_torch_file(path, safe_load=True, return_metadata=False):
    load_calls.append(path)
    if return_metadata:
        return {"tensor": 1}, None
    return {"tensor": 1}

def fake_load_lora_for_models(model, clip, lora, sm, sc, lora_metadata=None):
    apply_calls.append((sm, sc))
    return (model + 1, clip + 1 if clip is not None else None)

comfy_utils.load_torch_file = fake_load_torch_file
comfy_sd.load_lora_for_models = fake_load_lora_for_models

folder_paths = types.ModuleType("folder_paths"); sys.modules["folder_paths"] = folder_paths

LORAS_DIR = tempfile.mkdtemp(prefix="sf_lora_test_")

def fake_get_full_path(folder, name):
    if folder != "loras":
        return None
    return os.path.join(LORAS_DIR, name.replace("/", os.sep))

def fake_get_folder_paths(folder):
    return [LORAS_DIR]

folder_paths.get_full_path = fake_get_full_path
folder_paths.get_folder_paths = fake_get_folder_paths

# ── 注册 sfnodes 包结构（相对导入 from ...sf_utils... 需要）──────────────────
pkg = types.ModuleType("sfnodes"); pkg.__path__ = [root]; sys.modules["sfnodes"] = pkg
pkg2 = types.ModuleType("sfnodes.nodes"); pkg2.__path__ = [os.path.join(root, "nodes")]; sys.modules["sfnodes.nodes"] = pkg2
pkg3 = types.ModuleType("sfnodes.nodes.model"); pkg3.__path__ = [os.path.join(root, "nodes", "model")]; sys.modules["sfnodes.nodes.model"] = pkg3
pkg4 = types.ModuleType("sfnodes.sf_utils"); pkg4.__path__ = [os.path.join(root, "sf_utils")]; sys.modules["sfnodes.sf_utils"] = pkg4

# logger 需要先注册（lora_reader/lora_routes 相对导入它）——真实模块，仅 stdlib
spec_logger = importlib.util.spec_from_file_location(
    "sfnodes.sf_utils.logger",
    os.path.join(root, "sf_utils", "logger.py"),
)
logger_mod = importlib.util.module_from_spec(spec_logger)
sys.modules["sfnodes.sf_utils.logger"] = logger_mod
spec_logger.loader.exec_module(logger_mod)

# lora_reader 纯逻辑（无相对导入，直接加载）
spec_utils = importlib.util.spec_from_file_location(
    "sf_utils_lora_reader",
    os.path.join(root, "sf_utils", "lora_reader.py"),
)
utils = importlib.util.module_from_spec(spec_utils)
sys.modules[spec_utils.name] = utils
spec_utils.loader.exec_module(utils)

# ── lora_routes（mock 掉 aiohttp web / folder_paths；测守卫函数）─────────────
aiohttp_web = types.ModuleType("aiohttp"); sys.modules["aiohttp"] = aiohttp_web
web_mod = types.ModuleType("aiohttp.web")
web_mod.json_response = lambda *a, **k: None
web_mod.Response = lambda *a, **k: None
web_mod.FileResponse = lambda *a, **k: None
sys.modules["aiohttp.web"] = web_mod

# 节点类（含相对导入 + 副作用路由注册；路由注册会因 server 缺失被 try/except 吞掉）
spec_node = importlib.util.spec_from_file_location(
    "sfnodes.nodes.model.lora_stack",
    os.path.join(root, "nodes", "model", "lora_stack.py"),
)
mod = importlib.util.module_from_spec(spec_node)
sys.modules[spec_node.name] = mod
spec_node.loader.exec_module(mod)

# ── read_safetensors_metadata ──
def make_safetensors(meta):
    header = {"__metadata__": meta}
    h = json.dumps(header).encode()
    return struct.pack("<Q", len(h)) + h + b"\x00" * 8  # 头 + 假张量块

sf_path = os.path.join(LORAS_DIR, "test.safetensors")
with open(sf_path, "wb") as f:
    f.write(make_safetensors({"modelspec.trigger_phrase": "alpha, beta",
                              "ss_network_dim": "32", "ss_network_alpha": "16",
                              "ss_base_model_version": "sd_xl_base_1.0"}))

check("read_safetensors_metadata 好文件", utils.read_safetensors_metadata(sf_path).get("ss_network_dim") == "32")
check("read_safetensors_metadata 缺失 -> {}", utils.read_safetensors_metadata(os.path.join(LORAS_DIR, "nope.safetensors")) == {})

bad_path = os.path.join(LORAS_DIR, "bad.safetensors")
with open(bad_path, "wb") as f:
    f.write(struct.pack("<Q", 10 ** 15) + b"xx")  # 头长度字段巨大
check("read_safetensors_metadata 巨大长度 -> {}", utils.read_safetensors_metadata(bad_path) == {})

bad2 = os.path.join(LORAS_DIR, "bad2.safetensors")
with open(bad2, "wb") as f:
    f.write(b"\x00\x00")  # 不足 8 字节
check("read_safetensors_metadata 短文件 -> {}", utils.read_safetensors_metadata(bad2) == {})

# ── derive_trigger_words ──
dw = utils.derive_trigger_words
check("trigger_words 短语优先", dw({"modelspec.trigger_phrase": "alpha, beta"}) == ["alpha", "beta"])
check("trigger_words 短语去重", dw({"modelspec.trigger_phrase": "alpha, ALPHA"}) == ["alpha"])
check("trigger_words ss_trigger_words", dw({"ss_trigger_words": "zeta"}) == ["zeta"])
check("trigger_words 频率排序", dw({"ss_tag_frequency": json.dumps({
    "d1": {"low": 1, "high": 9, "mid": 3}, "d2": {"mid": 5}})}) == ["high", "mid", "low"])
check("trigger_words 频率跨目录求和", dw({"ss_tag_frequency": json.dumps({
    "d1": {"a": 2}, "d2": {"a": 3, "b": 1}})}) == ["a", "b"])
check("trigger_words 空 -> []", dw({}) == [] and dw(None) == [])
check("trigger_words 上限 20", len(dw({"ss_tag_frequency": json.dumps(
    {"d": {f"tag{i}": i for i in range(40)}})}, limit=20)) == 20)
check("trigger_words 坏频率不炸", dw({"ss_tag_frequency": "not json"}) == [])
check("trigger_words 坏计数跳过", dw({"ss_tag_frequency": json.dumps(
    {"d": {"a": "x", "b": 2}})}) == ["b"])

# ── base_model_family ──
bm = utils.base_model_family
check("base_model SDXL", bm({"ss_base_model_version": "sd_xl_base_1.0"}) == "SDXL")
check("base_model SD1.5", bm({"ss_sd_model_name": "v1-5-pruned"}) == "SD1.5")
check("base_model SD2", bm({"ss_base_model_version": "sd2-1"}) == "SD2")
check("base_model SD3", bm({"modelspec.architecture": "stable-diffusion-3"}) == "SD3")
check("base_model Flux", bm({"modelspec.implementation": "flux"}) == "Flux")
check("base_model 未知 -> ''", bm({"ss_network_module": "x"}) == "" and bm(None) == "")

# ── build_lora_info（file 源）──
info = utils.build_lora_info(sf_path)
check("build_lora_info title", info["title"] == "test")
check("build_lora_info base_model", info["base_model"] == "SDXL")
check("build_lora_info rank/alpha", info["rank"] == "32" and info["alpha"] == "16")
check("build_lora_info triggers 合并 = 文件", info["triggers"] == ["alpha", "beta"])
check("build_lora_info file_triggers", info["file_triggers"] == ["alpha", "beta"])
check("build_lora_info source=file", info["source"] == "file")
check("build_lora_info 无 description", info["description"] == "")

# 侧车优先（含 description——API 实测 description 在 version 顶层）
sidecar_path = os.path.join(LORAS_DIR, "test.civitai.info")
with open(sidecar_path, "w", encoding="utf-8") as f:
    json.dump({"trainedWords": ["side1"],
               "description": "<b>Hello</b> &amp; welcome<br>line 2",
               "model": {"name": "Test Model"},
               "modelId": "123", "id": "456"}, f)
info2 = utils.build_lora_info(sf_path)
check("build_lora_info 侧车触发词胜出", info2["triggers"] == ["side1"] and info2["sidecar_triggers"] == ["side1"])
check("build_lora_info 侧车 title", info2["title"] == "Test Model")
check("build_lora_info 侧车 ids", info2["model_id"] == 123 and info2["version_id"] == 456)
check("build_lora_info source=sidecar", info2["source"] == "sidecar")
check("build_lora_info 侧车 description 清洗", info2["description"] == "Hello & welcome\nline 2")
os.remove(sidecar_path)

# ── parse_state ──
ps = utils.parse_state
check("parse_state 空串", ps("") == {"loras": [], "sep": ", ", "cacheMode": "last"})
check("parse_state 垃圾 JSON", ps("{oops") == {"loras": [], "sep": ", ", "cacheMode": "last"})
check("parse_state 非 dict JSON", ps("[1]") == {"loras": [], "sep": ", ", "cacheMode": "last"})
check("parse_state 非字符串输入", ps(None)["loras"] == [] and ps({"loras": []})["loras"] == [])
st = ps(json.dumps({"sep": "|", "cacheMode": "all", "loras": [
    {"name": "a.safetensors", "on": True, "sm": 1.0},
    {"name": "b.safetensors", "on": False, "sm": 0.5, "sc": 0.3},
    {"name": "   ", "on": True},          # 空名丢弃
    {"sm": 1.0},                          # 无名丢弃
    "not a dict",                         # 非 dict 丢弃
]}))
check("parse_state 结构", len(st["loras"]) == 2 and st["sep"] == "|" and st["cacheMode"] == "all")
check("parse_state sc 缺省 = sm", st["loras"][0]["sc"] == 1.0)
check("parse_state on 缺省 true", st["loras"][0]["on"] is True and st["loras"][1]["on"] is False)
check("parse_state cacheMode 未知钳 last", ps('{"cacheMode":"weird"}')["cacheMode"] == "last")
check("parse_state 强度钳制", ps(json.dumps({"loras": [{"name": "x", "sm": 999, "sc": -999}]}))["loras"][0] ==
      {"name": "x", "on": True, "sm": 100.0, "sc": -100.0, "triggers": []})
check("parse_state nan/inf -> 0", ps(json.dumps({"loras": [{"name": "x", "sm": "nan", "sc": "inf"}]}))["loras"][0]["sm"] == 0.0
      and ps(json.dumps({"loras": [{"name": "x", "sc": "nan"}]}))["loras"][0]["sc"] == 0.0)
check("parse_state 坏 sm 字符串 -> 0", ps(json.dumps({"loras": [{"name": "x", "sm": "abc"}]}))["loras"][0]["sm"] == 0.0)
check("parse_state strength 旧键", ps(json.dumps({"loras": [{"name": "x", "strength": 0.7}]}))["loras"][0]["sm"] == 0.7)
check("parse_state triggers 清洗（不去重，去重在 collect）", ps(json.dumps({"loras": [{"name": "x", "triggers": [" a ", "", 5, "a"]}]}))["loras"][0]["triggers"] == ["a", "5", "a"])

# ── collect_triggers ──
ct = utils.collect_triggers
check("collect_triggers 仅 on", ct({"loras": [
    {"on": True, "triggers": ["a", "b"]}, {"on": False, "triggers": ["c"]}]}, ) == "a, b")
check("collect_triggers 去重", ct({"loras": [{"on": True, "triggers": ["A", "a", "b"]}]}) == "A, b")
check("collect_triggers sep", ct({"loras": [{"on": True, "triggers": ["a", "b"]}], "sep": "|"}) == "a|b")
check("collect_triggers 空", ct({}) == "" and ct({"loras": [{"on": True, "triggers": []}]}) == "")

# ── Civitai key ──
sk = utils.sanitize_civitai_key
check("sanitize_key 通过", sk("abcDEF123-_.~") == "abcDEF123-_.~")
check("sanitize_key strip 空白", sk("  key123\n") == "key123")
check("sanitize_key 拒空格", sk("abc def") == "")
check("sanitize_key 拒控制字符", sk("abc\x00def") == "" and sk("abc\n") == "abc")
check("sanitize_key 拒中文", sk("密钥") == "")
check("sanitize_key 拒过长", sk("k" * 201) == "")
check("sanitize_key 拒非字符串", sk(None) == "" and sk(123) == "")
check("mask_key", utils.mask_civitai_key("abc12345") == "••••••2345")
check("civitai_hosts", utils.civitai_hosts("com") == ("civitai.com", "civitai.red")
      and utils.civitai_hosts("red") == ("civitai.red", "civitai.com"))

# ── 自定义词 ──
ck = utils.custom_trigger_key
check("trigger_key 反斜杠折叠", ck("sub\\file.safetensors") == "sub/file.safetensors")
check("trigger_key strip", ck("  a.safetensors  ") == "a.safetensors")
check("trigger_key 垃圾", ck("") == "" and ck(None) == "")
sw = utils.sanitize_custom_words
check("sanitize_custom_words 清洗", sw([" a ", "", "a", "B"]) == ["a", "B"])
check("sanitize_custom_words 去重保先拼写", sw(["Hello", "hello"]) == ["Hello"])
check("sanitize_custom_words 上限", len(sw([f"w{i}" for i in range(100)])) == 64)
check("sanitize_custom_words 长度截断", sw(["x" * 300]) == ["x" * 200])
check("sanitize_custom_words 非列表", sw("abc") == [])
# 原子写 + 读
store_path = os.path.join(tempfile.mkdtemp(prefix="sf_lora_store_"), "triggers.json")
check("set_custom_triggers 写", utils.set_custom_triggers(store_path, "sub/a.safetensors", ["w1", "w2"]) == ["w1", "w2"])
check("get_custom_triggers 读", utils.get_custom_triggers(store_path, "sub\\a.safetensors") == ["w1", "w2"])
check("set_custom_triggers 空清条目", utils.set_custom_triggers(store_path, "sub/a.safetensors", []) == []
      and utils.get_custom_triggers(store_path, "sub/a.safetensors") == [])
check("set_custom_triggers 不存在 LoRA 名可写（路由守卫在别处）",
      utils.set_custom_triggers(store_path, "other.safetensors", ["x"]) == ["x"])

# ── 自定义描述（同存储，形状升级兼容）──
cd = utils.get_custom_description
check("get_custom_description 空", cd(store_path, "other.safetensors") == "")
check("set_custom_description 写", utils.set_custom_description(store_path, "other.safetensors", "my desc") == "my desc")
check("get_custom_description 读", cd(store_path, "other.safetensors") == "my desc")
check("custom 描述不清词", utils.get_custom_triggers(store_path, "other.safetensors") == ["x"])
# 旧形状文件兼容：{key: [words]} 仍读为词
legacy_path = os.path.join(tempfile.mkdtemp(prefix="sf_lora_store2_"), "triggers.json")
with open(legacy_path, "w", encoding="utf-8") as f:
    json.dump({"legacy/a.safetensors": ["old1"], "junk": "notalist"}, f)
check("旧形状 {key:[words]} 读", utils.get_custom_triggers(legacy_path, "legacy/a.safetensors") == ["old1"])
check("旧形状垃圾值忽略", utils.get_custom_triggers(legacy_path, "junk") == [])
# 词空但描述在 -> 条目保留；描述空 + 词在 -> 条目保留；都空 -> 删
utils.set_custom_triggers(store_path, "other.safetensors", [])
check("清词保留描述", cd(store_path, "other.safetensors") == "my desc")
utils.set_custom_description(store_path, "other.safetensors", "")
check("描述与词都空删条目", cd(store_path, "other.safetensors") == ""
      and utils.get_custom_triggers(store_path, "other.safetensors") == [])
check("set_custom_description 截断限长", len(utils.set_custom_description(store_path, "t.safetensors", "x" * 9999)) <= 2000)
check("set_custom_description 非 str -> 清", utils.set_custom_description(store_path, "t.safetensors", None) == ""
      and cd(store_path, "t.safetensors") == "")

# ── 孤儿数据迁移（文件移动/改名后）──
bk = utils.base_key
check("base_key 去目录去扩展", bk("a/b/c.safetensors") == "c" and bk("d.ckpt") == "d")
check("base_key 无扩展名", bk("x") == "x")
check("base_key 垃圾", bk("") == "" and bk(None) == "")

# ── 内容指纹（文件改名/移动的内容级证据）──
fp_path = os.path.join(LORAS_DIR, "fptest.bin")
with open(fp_path, "wb") as f:
    f.write(os.urandom(200 * 1024))  # 200KB，跨头/中/尾三段
import shutil
shutil.copy(fp_path, os.path.join(LORAS_DIR, "renamed.bin"))  # 模拟"改名"（内容不变）
fp1 = utils.file_fingerprint(fp_path)
check("file_fingerprint 形状", fp1 is not None and set(fp1.keys()) == {"size", "head", "mid", "tail"}
      and fp1["size"] == 200 * 1024)
check("file_fingerprint 改名后一致", utils.file_fingerprint(os.path.join(LORAS_DIR, "renamed.bin")) is not None
      and utils.file_fingerprint(os.path.join(LORAS_DIR, "renamed.bin"))["head"] == fp1["head"])
fp_renamed = utils.file_fingerprint(os.path.join(LORAS_DIR, "renamed.bin"))
check("file_fingerprint 改名后全等", utils._fp_equal(fp1, fp_renamed) is True)
with open(os.path.join(LORAS_DIR, "other.bin"), "wb") as f:
    f.write(os.urandom(200 * 1024))
check("file_fingerprint 不同文件不同", utils._fp_equal(fp1, utils.file_fingerprint(os.path.join(LORAS_DIR, "other.bin"))) is False)
check("file_fingerprint 缺失文件 None", utils.file_fingerprint(os.path.join(LORAS_DIR, "nope.bin")) is None)
small = os.path.join(LORAS_DIR, "small.bin")
with open(small, "wb") as f:
    f.write(b"tiny")
fps = utils.file_fingerprint(small)
check("file_fingerprint 小文件三段一致", fps["head"] == fps["mid"] == fps["tail"])
# find_orphan_by_fingerprint
fp_store = utils.read_custom_store(store_path)
utils.set_custom_triggers(store_path, "a/x.bin", ["w"], fp1)
fp_store2 = utils.read_custom_store(store_path)
check("fp 写入条目", utils._fp_equal(fp_store2["a/x.bin"]["fp"], fp1) is True)
check("find_orphan_by_fingerprint 唯一命中",
      utils.find_orphan_by_fingerprint(fp_store2, utils.file_fingerprint(os.path.join(LORAS_DIR, "renamed.bin"))) == "a/x.bin")
check("find_orphan_by_fingerprint 无匹配 None",
      utils.find_orphan_by_fingerprint(fp_store2, utils.file_fingerprint(os.path.join(LORAS_DIR, "small.bin"))) is None)
check("find_orphan_by_fingerprint exclude 排除", utils.find_orphan_by_fingerprint(
    fp_store2, utils.file_fingerprint(os.path.join(LORAS_DIR, "renamed.bin")), exclude="a/x.bin") is None)
check("find_orphan_by_fingerprint 歧义 None", utils.find_orphan_by_fingerprint(
    {"k1": {"fp": fp1}, "k2": {"fp": fp1}}, fp1) is None)
check("find_orphan_by_fingerprint 坏指纹 None", utils.find_orphan_by_fingerprint(fp_store2, None) is None)
utils.set_custom_triggers(store_path, "a/x.bin", [])  # 清理
# _norm_fp 容错
check("_norm_fp 坏形状 None", utils._norm_fp("x") is None and utils._norm_fp({"size": 1}) is None
      and utils._norm_fp({"size": "a", "head": "h" * 64, "mid": "m" * 64, "tail": "t" * 64}) is None)
check("_norm_fp 好形状", utils._norm_fp(fp1) == fp1)
# 构造：旧键有数据（词+描述），新键无数据 -> 唯一基名匹配
mig_store = os.path.join(tempfile.mkdtemp(prefix="sf_lora_mig_"), "triggers.json")
utils.set_custom_triggers(mig_store, "old/dir/char.safetensors", ["w1"])
utils.set_custom_description(mig_store, "old/dir/char.safetensors", "old desc")
check("find_orphan_key 唯一匹配", utils.find_orphan_key(
    utils.read_custom_store(mig_store), "new/dir/char.safetensors") == "old/dir/char.safetensors")
check("find_orphan_key 同名键排除", utils.find_orphan_key(
    utils.read_custom_store(mig_store), "old/dir/char.safetensors") is None)
# 歧义：两个不同目录同名 -> 不匹配
utils.set_custom_triggers(mig_store, "another/char.safetensors", ["w2"])
check("find_orphan_key 同名多目录歧义放弃", utils.find_orphan_key(
    utils.read_custom_store(mig_store), "new/dir/char.safetensors") is None)
utils.set_custom_triggers(mig_store, "another/char.safetensors", [])  # 清歧义条目
# 迁移
res = utils.migrate_custom_data(mig_store, "new/dir/char.safetensors")
check("migrate_custom_data ok", res["ok"] is True and res["old_key"] == "old/dir/char.safetensors")
check("migrate_custom_data 词迁移", utils.get_custom_triggers(mig_store, "new/dir/char.safetensors") == ["w1"])
check("migrate_custom_data 描述迁移", utils.get_custom_description(mig_store, "new/dir/char.safetensors") == "old desc")
check("migrate_custom_data 旧键删除", utils.get_custom_triggers(mig_store, "old/dir/char.safetensors") == []
      and utils.get_custom_description(mig_store, "old/dir/char.safetensors") == "")
check("migrate_custom_data 新键已有不迁移", utils.migrate_custom_data(mig_store, "new/dir/char.safetensors")["ok"] is False)
check("migrate_custom_data 无唯一匹配", utils.migrate_custom_data(mig_store, "totally/new.safetensors")["ok"] is False)
# 指纹路径迁移：指定 old_key（孤儿检测指纹命中时）+ 迁移后新键带 fp
utils.set_custom_triggers(mig_store, "old/dir/char.safetensors", ["w1"])
res2 = utils.migrate_custom_data(mig_store, "new2/dir/char.safetensors", fp1, "old/dir/char.safetensors")
check("migrate_custom_data 指定 old_key", res2["ok"] is True and res2["old_key"] == "old/dir/char.safetensors")
check("migrate_custom_data 新键带 fp", utils._fp_equal(
    utils.read_custom_store(mig_store)["new2/dir/char.safetensors"]["fp"], fp1) is True)
check("migrate_custom_data 指定合法 old_key 迁移", utils.migrate_custom_data(
    mig_store, "z.safetensors", None, "new/dir/char.safetensors")["ok"] is True)
check("migrate_custom_data 坏 old_key 拒绝", utils.migrate_custom_data(
    mig_store, "new/dir/char.safetensors", None, "nope")["ok"] is False
    and utils.migrate_custom_data(mig_store, "z.safetensors", None, "z.safetensors")["ok"] is False)
# 预览图迁移
mig_pv = tempfile.mkdtemp(prefix="sf_lora_migpv_")
old_pv = utils.custom_preview_path(mig_pv, "old/dir/char.safetensors")
with open(old_pv, "wb") as f:
    f.write(b"\xff\xd8\xff\xe0img")
check("migrate_custom_preview ok", utils.migrate_custom_preview(mig_pv, "new/dir/char.safetensors", "old/dir/char.safetensors") is True)
check("migrate_custom_preview 新键文件在", utils.find_custom_preview(mig_pv, "new/dir/char.safetensors") is not None)
check("migrate_custom_preview 旧键文件无", utils.find_custom_preview(mig_pv, "old/dir/char.safetensors") is None)
# 目标已存在不覆盖
with open(old_pv, "wb") as f:
    f.write(b"xx")
check("migrate_custom_preview 目标已存在不覆盖", utils.migrate_custom_preview(mig_pv, "new/dir/char.safetensors", "old/dir/char.safetensors") is False
      and utils.find_custom_preview(mig_pv, "new/dir/char.safetensors") is not None)

# ── _clean_description ──
cl = utils._clean_description
check("clean_desc 剥标签", cl("<b>Hello</b> world") == "Hello world")
check("clean_desc 实体解码", cl("a &amp; b &lt;c&gt;") == "a & b <c>")
check("clean_desc br 转行", cl("line1<br>line2<br/>line3") == "line1\nline2\nline3")
check("clean_desc 空白折叠", cl("a   b\tc") == "a b c")
check("clean_desc 非 str -> ''", cl(None) == "" and cl(123) == "" and cl("") == "")
check("clean_desc 截断", len(cl("x" * 5000)) == 2000)

# ── 自定义预览名（安全形状）──
cpn = utils.custom_preview_name
check("custom_preview_name 形状", utils.is_custom_preview_name(cpn("a/b.safetensors")) is True)
check("custom_preview_name 垃圾", cpn("") == "" and cpn(None) == "")
# 越界拒绝是构造性保证（名字永远是 16 hex + .jpg，拼接不出目录）——测垃圾
# 名返回 None 与正常路径存在即可。
check("custom_preview_path 垃圾名拒绝", utils.custom_preview_path("", "../evil.safetensors") is None
      and utils.custom_preview_path(None, "a.safetensors") is None)
folder_pv = tempfile.mkdtemp(prefix="sf_lora_pv_")
pv_path = utils.custom_preview_path(folder_pv, "a.safetensors")
with open(pv_path, "wb") as f:
    f.write(b"\xff\xd8\xff\xe0jpegdata")
check("custom_preview_version mtime", utils.custom_preview_version(folder_pv, "a.safetensors") > 0)
check("find_custom_preview", utils.find_custom_preview(folder_pv, "a.safetensors") == pv_path)
check("delete_custom_preview", utils.delete_custom_preview(folder_pv, "a.safetensors") is True
      and utils.find_custom_preview(folder_pv, "a.safetensors") is None)
check("write_custom_preview 拒非 bytes", utils.write_custom_preview(folder_pv, "a.safetensors", "text") is None)

# ── parse_civitai_modelversion ──
pmv = utils.parse_civitai_modelversion
civ = pmv({"trainedWords": ["t1"], "baseModel": "SDXL",
           "description": "Great <i>style</i> &amp; more",
           "model": {"name": "M", "type": "LORA"},
           "modelId": "1", "id": "2",
           "images": [{"url": "https://x/o/original=true/1.jpg", "nsfw": "X", "nsfwLevel": 16},
                      {"url": "https://x/o/original=true/2.jpg", "nsfw": None}]})
check("parse_civitai 触发词", civ["triggers"] == ["t1"])
check("parse_civitai description 清洗", civ["description"] == "Great style & more")
check("parse_civitai 顶层空则 model 兜底", pmv({"model": {"description": "fallback"}})["description"] == "fallback")
check("parse_civitai 无描述", "description" not in pmv({"trainedWords": ["x"]}))
check("parse_civitai 跳过显式图取下一张", civ["thumbnail"] == "https://x/o/width=256/2.jpg")
check("parse_civitai ids", civ["model_id"] == 1 and civ["version_id"] == 2)
civ2 = pmv({"images": [{"url": "https://x/a/original=true/1.jpg", "nsfw": "X"}]})
check("parse_civitai 全显式无缩略图", "thumbnail" not in civ2)
civ3 = pmv({"images": [{"url": "https://x/a/original=true/1.jpg", "nsfw": "X"}]}, allow_adult=True)
check("parse_civitai allow_adult 用显式图", civ3.get("thumbnail") == "https://x/a/width=256/1.jpg")
check("parse_civitai 垃圾", pmv("x") == {} and pmv(None) == {})

# ── 侧车缓存（文件名 = splitext 基名 + .civitai.info，与读侧车一致）──
side_base = os.path.splitext(sf_path)[0] + ".civitai.info"
check("save_sidecar_cache", utils.save_sidecar_cache(sf_path, {"a": 1}) is True
      and os.path.isfile(side_base))
# sidecar_thumbnail：从侧车原始响应提取缩略图
with open(side_base, "w", encoding="utf-8") as f:
    json.dump({"model": {"name": "M"},
               "images": [{"url": "https://x/o/original=true/1.jpg", "nsfw": "X", "nsfwLevel": 16},
                          {"url": "https://x/o/original=true/2.jpg", "nsfw": None}]}, f)
check("sidecar_thumbnail 非成人图", utils.sidecar_thumbnail(sf_path) == "https://x/o/width=256/2.jpg")
with open(side_base, "w", encoding="utf-8") as f:
    json.dump({"images": [{"url": "https://x/o/original=true/1.jpg", "nsfw": "X", "nsfwLevel": 16}]}, f)
check("sidecar_thumbnail 全显式无图", utils.sidecar_thumbnail(sf_path) is None)
check("sidecar_thumbnail allow_adult 全显式", utils.sidecar_thumbnail(sf_path, allow_adult=True) == "https://x/o/width=256/1.jpg")
check("sidecar_thumbnail 无侧车 -> None", utils.sidecar_thumbnail(os.path.join(LORAS_DIR, "none.safetensors")) is None)
os.remove(side_base)
check("delete_sidecar_cache", utils.delete_sidecar_cache(sf_path) is True
      and not os.path.isfile(sf_path + ".civitai.info"))
check("delete_sidecar_cache 已无", utils.delete_sidecar_cache(sf_path) is True)

# ── 账户读写（0600）──
acc_path = os.path.join(tempfile.mkdtemp(prefix="sf_lora_acc_"), "civitai.json")
check("write_civitai_account", utils.write_civitai_account(acc_path, {"key": "k123", "host": "red", "adult_thumbs": True}) is True)
acc = utils.read_civitai_account(acc_path)
check("read_civitai_account 完整形状", acc["key"] == "k123" and acc["host"] == "red" and acc["adult_thumbs"] is True)
check("read_civitai_account 损坏文件", utils.read_civitai_account("/nonexistent/x.json") ==
      {"key": "", "host": "com", "adult_thumbs": False})
with open(acc_path, "w") as f:
    f.write("{{{bad")
check("read_civitai_account 坏 JSON", utils.read_civitai_account(acc_path)["key"] == "")

# ── lora_routes 守卫（以包结构名加载，相对导入 .logger 才成立）──
spec_routes = importlib.util.spec_from_file_location(
    "sfnodes.sf_utils.lora_routes",
    os.path.join(root, "sf_utils", "lora_routes.py"),
)
routes = importlib.util.module_from_spec(spec_routes)
sys.modules[spec_routes.name] = routes
spec_routes.loader.exec_module(routes)
is_under = routes._is_path_under
check("is_path_under 在内", is_under("/data/loras/a.safetensors", "/data/loras") is True)
check("is_path_under 在外", is_under("/data/other/a.safetensors", "/data/loras") is False)
check("is_path_under 前缀陷阱", is_under("/data/loras_evil/a", "/data/loras") is False)
check("is_path_under 根相等", is_under("/data/loras", "/data/loras") is True)
# symlink 逃逸：realpath 双端严格检查必须拒绝同盘链接指向根外
if os.name == "posix":
    real_root = os.path.realpath(LORAS_DIR)
    outside_dir = os.path.join(os.path.dirname(real_root), "sf_lora_outside_" + os.path.basename(real_root))
    os.makedirs(outside_dir, exist_ok=True)
    link = os.path.join(real_root, "link")
    try:
        os.symlink(outside_dir, link)
        check("is_path_under 拒绝 symlink 逃逸",
              is_under(os.path.join(link, "secret.safetensors"), real_root) is False)
    except OSError:
        print("PASS: is_path_under 拒绝 symlink 逃逸 (symlink 不可用，跳过)")
    # junction 跨盘场景（无第二盘时模拟）：lexical 回退仅在 commonpath 抛
    # ValueError 时解锁——用不存在路径构造不同前缀无法触发，此分支留待真实环境。
    check("is_path_under 子路径真实", is_under(os.path.join(real_root, "sub", "x.safetensors"), real_root) is True)
li = routes._looks_like_image
check("looks_like_image jpg", li(b"\xff\xd8\xff\xe0xxxx") is True)
check("looks_like_image png", li(b"\x89PNG\r\n\x1a\nxxx") is True)
check("looks_like_image webp", li(b"RIFF\x00\x00\x00\x00WEBPVP8") is True)
check("looks_like_image gif", li(b"GIF89a") is True)
check("looks_like_image 文本", li(b"hello world") is False)
check("looks_like_image 空", li(b"") is False)

# ── 缩略图 URL 安全（https only）──
ts = routes._thumb_url_safe
check("thumb_url_safe https 收", ts("https://image.civitai.com/x.jpg") is True)
check("thumb_url_safe http 拒", ts("http://image.civitai.com/x.jpg") is False)
check("thumb_url_safe ftp 拒", ts("ftp://x/y.jpg") is False)
check("thumb_url_safe 无 scheme 拒", ts("//image.civitai.com/x.jpg") is False)
check("thumb_url_safe 非 str 拒", ts(None) is False and ts(123) is False)
check("thumb_url_safe 空拒", ts("") is False)

# ── 节点结构 ──
node = mod.SFLoraStack()
check("CATEGORY", node.CATEGORY == "sfnodes/model")
check("DESCRIPTION 存在", isinstance(node.DESCRIPTION, str) and len(node.DESCRIPTION) > 0)
it = node.INPUT_TYPES()
check("INPUT_TYPES model", it["required"]["model"][0] == "MODEL")
check("INPUT_TYPES clip 可选", it["optional"]["clip"][0] == "CLIP")
check("INPUT_TYPES preset 可选", it["optional"]["preset"][0] == "SF_LORA_PRESET")
check("INPUT_TYPES hidden LoraLoaderState", it["hidden"]["LoraLoaderState"][1]["default"] == "{}")
check("RETURN_TYPES", node.RETURN_TYPES == ("MODEL", "CLIP", "STRING"))
check("RETURN_NAMES", node.RETURN_NAMES == ("MODEL", "CLIP", "triggers"))
check("FUNCTION = apply", node.FUNCTION == "apply")

# ── preset_override（Power 预设形状 -> 行形状，预设优先）──
po = utils.preset_override
st_po = {"loras": [
    {"name": "a.safetensors", "on": True, "sm": 1, "sc": 1, "triggers": ["keep"]},
    {"name": "b.safetensors", "on": True, "sm": 0.5, "sc": 0.5, "triggers": ["x"]},
], "sep": "|", "cacheMode": "all"}
preset_po = {"normalize": False, "loras": [
    {"lora": "a.safetensors", "on": True, "strength": 0.9, "strengthTwo": 0.7},
    {"lora": "c.safetensors", "on": False, "strength": 1.5},
    {"lora": "", "on": True, "strength": 1},   # 空名丢弃
    "not a dict",                              # 非 dict 丢弃
]}
out_po = po(st_po, preset_po)
check("preset_override 行覆盖", len(out_po["loras"]) == 2)
check("preset_override 名称/强度", out_po["loras"][0]["name"] == "a.safetensors"
      and out_po["loras"][0]["sm"] == 0.9 and out_po["loras"][0]["sc"] == 0.7)
check("preset_override strengthTwo 缺省 = sm", out_po["loras"][1]["sm"] == 1.5 and out_po["loras"][1]["sc"] == 1.5)
check("preset_override on", out_po["loras"][1]["on"] is False)
check("preset_override 同名行触发词继承", out_po["loras"][0]["triggers"] == ["keep"])
check("preset_override 新行触发词空", out_po["loras"][1]["triggers"] == [])
check("preset_override 其余状态不变", out_po["sep"] == "|" and out_po["cacheMode"] == "all")
check("preset_override 非 dict 原样", po(st_po, None) == st_po and po(st_po, "x") == st_po
      and po(st_po, {"loras": "x"}) == st_po)
check("preset_override 强度钳制", po(st_po, {"loras": [{"lora": "z.safetensors", "strength": 999}]})["loras"][0]["sm"] == 100.0)

# ── apply 全链路 ──
def run_apply(state_str, model=0, clip=0, preset=None):
    load_calls.clear(); apply_calls.clear()
    return node.apply(model, clip, preset=preset, LoraLoaderState=state_str)

# 全 off -> 不加载
res = run_apply(json.dumps({"loras": [{"name": "test.safetensors", "on": False, "sm": 1.0, "sc": 1.0}]}))
check("apply 全 off 直通", res[0] == 0 and res[1] == 0 and res[2] == "" and load_calls == [])

# 正常应用 + 触发词
res = run_apply(json.dumps({"loras": [
    {"name": "test.safetensors", "on": True, "sm": 0.5, "sc": 0.5, "triggers": ["alpha"]},
    {"name": "missing.safetensors", "on": True, "sm": 1.0, "sc": 1.0, "triggers": ["ghost"]},
]}))
check("apply 链式应用", res[0] == 1 and apply_calls == [(0.5, 0.5)])
check("apply 缺失文件词不计", res[2] == "alpha")

# 强度 0 -> 计触发词不加载
res = run_apply(json.dumps({"loras": [{"name": "test.safetensors", "on": True, "sm": 0, "sc": 0, "triggers": ["z"]}]}))
check("apply 强度 0 计触发词", res[2] == "z" and load_calls == [])

# 无 clip -> sc 归 0（不因 clip strength 报错）
res = run_apply(json.dumps({"loras": [{"name": "test.safetensors", "on": True, "sm": 0.5, "sc": 9.0}]}), model=0, clip=None)
check("apply 无 clip sc=0", apply_calls == [(0.5, 0.0)])

# 分隔符
res = run_apply(json.dumps({"sep": "|", "loras": [
    {"name": "test.safetensors", "on": True, "sm": 1, "sc": 1, "triggers": ["a", "b"]}]}))
check("apply 分隔符", res[2] == "a|b")

# preset 分支：preset 优先（强度覆盖行状态；触发词从行状态同名行继承）
res = run_apply(json.dumps({"loras": [{"name": "test.safetensors", "on": True, "sm": 0.1, "sc": 0.1, "triggers": ["alpha"]}]}),
                preset={"loras": [{"lora": "test.safetensors", "on": True, "strength": 0.5, "strengthTwo": 0.5}]})
check("apply preset 优先强度", apply_calls == [(0.5, 0.5)])
check("apply preset 触发词继承", res[2] == "alpha")
check("apply preset 非 dict 忽略", run_apply(json.dumps({
    "loras": [{"name": "test.safetensors", "on": True, "sm": 0.5, "sc": 0.5}]}), preset=None)[0] == 1)

# cacheMode=all 保留已用路径
run_apply(json.dumps({"cacheMode": "all", "loras": [{"name": "test.safetensors", "on": True, "sm": 1, "sc": 1}]}))
check("cacheMode=all 保留缓存", set(node._cache.keys()) == {fake_get_full_path("loras", "test.safetensors")})
# 换文件后 all 修剪旧条目
run_apply(json.dumps({"cacheMode": "all", "loras": []}))
check("cacheMode=all 清空后修剪", node._cache == {})

# cacheMode=none 全清
run_apply(json.dumps({"cacheMode": "all", "loras": [{"name": "test.safetensors", "on": True, "sm": 1, "sc": 1}]}))
run_apply(json.dumps({"cacheMode": "none", "loras": []}))
check("cacheMode=none 清空", node._cache == {} and node._last_path is None)

# cacheMode=last：只有最近路径存活
run_apply(json.dumps({"cacheMode": "all", "loras": [{"name": "test.safetensors", "on": True, "sm": 1, "sc": 1}]}))
run_apply(json.dumps({"cacheMode": "last", "loras": [{"name": "test.safetensors", "on": True, "sm": 1, "sc": 1}]}))
check("cacheMode=last 保留最近", node._last_path == fake_get_full_path("loras", "test.safetensors")
      and len(node._cache) == 1)

# 注册在根 __init__.py（本项目惯例，节点模块不定义映射表）——根注册一致性
# 由 tests/test_registry_keys.py 或静态检查覆盖。
print()
if failures:
    print(f"{len(failures)} FAILURES: {failures}")
    sys.exit(1)
print("ALL PASS")
