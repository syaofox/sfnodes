# sf_utils/common.py + disk_state.sf_user_dir 纯函数测试（Node/Python 直接运行）：
# 覆盖本次提炼收敛的共享实现 —— json_safe / parse_json_dict / lora_stem /
# valid_name / _parse_fill_color / sf_user_dir（pause_image / pause_mask /
# preview_routes / brush_mask / crop_expand / lora_loader* / lora_selector /
# krea2_presets / text_presets / scale.py 的调用方行为由各自既有测试锁定）。
# 运行：python tests/test_common_pure.py
import math
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_sf_loader as L

common = L.load_node("sf_utils/common.py")

failures = []


def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")


# ── json_safe ──
js = common.json_safe
check("NaN 转字符串", js(float("nan")) == "nan")
check("+Inf 转字符串", js(float("inf")) == "inf")
check("-Inf 转字符串", js(float("-inf")) == "-inf")
check("普通 float 原样", js(1.5) == 1.5)
check("嵌套 dict 清洗", js({"a": [float("nan"), {"b": float("inf")}]}) == {"a": ["nan", {"b": "inf"}]})
check("tuple 按 list 处理", js((float("nan"), 1)) == ["nan", 1])
check("字符串/整数/None 原样", js("x") == "x" and js(3) == 3 and js(None) is None)

# ── parse_json_dict ──
pj = common.parse_json_dict
check("dict 直通", pj({"a": 1}) == {"a": 1})
check("合法 JSON 字符串", pj('{"a": 1}') == {"a": 1})
check("坏 JSON -> {}", pj("{") == {})
check("非 dict JSON -> {}", pj("[1,2]") == {} and pj("123") == {})
check("空/None/非字符串 -> {}", pj("") == {} and pj(None) == {} and pj(123) == {})

# ── lora_stem ──
ls = common.lora_stem
check("子目录去路径去扩展", ls("subdir/name.safetensors") == "name")
check("裸文件名", ls("name.safetensors") == "name")
check("多点文件名", ls("a.b.safetensors") == "a.b")

# ── valid_name ──
vn = common.valid_name
check("合法名", vn("my_preset") and vn("中文预设 1"))
check("空/空白/非字符串拒绝", not vn("") and not vn("   ") and not vn(None) and not vn(123))
check("路径分隔符拒绝", not vn("a/b") and not vn("a\\b"))
check("控制字符拒绝", not vn("a\x01b"))
check("无 max_len 不限长", vn("x" * 500))
check("max_len=200 通过/拒绝", vn("x" * 200, max_len=200) and not vn("x" * 201, max_len=200))

# ── _parse_fill_color（scale.py pad 复用同一实现）──
pf = common._parse_fill_color
check("hex 带 #", pf("#808080") == (128, 128, 128))
check("hex 不带 #", pf("ff0000") == (255, 0, 0))
check("三元组直通", pf((1, 2, 3)) == (1, 2, 3))

# ── sf_user_dir（mock folder_paths）──
tmp = tempfile.mkdtemp(prefix="sf_common_pure_")
fp = types.ModuleType("folder_paths")
fp.get_user_directory = lambda: tmp
sys.modules["folder_paths"] = fp
ds = L.load_node("sf_utils/disk_state.py")
d = ds.sf_user_dir()
check("指向 <user>/sfnodes", d == os.path.join(tmp, "sfnodes"))
check("目录已创建", os.path.isdir(d))
del sys.modules["folder_paths"]

if failures:
    print(f"\n{failures}")
    sys.exit(1)
print("\nALL PASS")
