# SFInputPath 后端逻辑测试（python3 tests/test_input_path.py）
# 覆盖：
#   - 结构：CATEGORY、RETURN_TYPES/NAMES、FUNCTION、DESCRIPTION
#   - execute：空/None → 空串（不报错）；input 相对名/子路径/[input] 注解 →
#     folder_paths.get_annotated_filepath 的绝对路径；已是存在的绝对路径原样
#     （normpath）；解析失败（目录穿越）→ ValueError 带 SF Input Path 上下文
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_sf_loader as L

INPUT_DIR = tempfile.mkdtemp(prefix="sf_input_path_")

# 注入最小 folder_paths 桩（真实实现：input 相对名 → 输入目录绝对路径、防穿越）
fake = types.ModuleType("folder_paths")


def _strip_annotation(name):
    name = name.strip()
    return name[:-len(" [input]")].strip() if name.endswith("[input]") else name


def _get_annotated_filepath(name):
    rel = _strip_annotation(name)
    if ".." in rel.replace("\\", "/").split("/"):
        raise ValueError("Invalid file path: {!r}".format(rel))
    return os.path.abspath(os.path.join(INPUT_DIR, rel))


fake.get_annotated_filepath = _get_annotated_filepath
sys.modules["folder_paths"] = fake

mod = L.load_node("nodes/utils/input_path.py")
SFInputPath = mod.SFInputPath

failures = []


def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")


# ── 1. 结构 ──
check("CATEGORY", SFInputPath.CATEGORY == "sfnodes/utils")
check("FUNCTION", SFInputPath.FUNCTION == "execute")
check("RETURN_TYPES 单 STRING", SFInputPath.RETURN_TYPES == ("STRING",))
check("RETURN_NAMES", SFInputPath.RETURN_NAMES == ("path",))
check("DESCRIPTION 存在", isinstance(getattr(SFInputPath, "DESCRIPTION", None), str)
      and SFInputPath.DESCRIPTION.strip() != "")
check("INPUT_TYPES 含 name", "name" in SFInputPath.INPUT_TYPES()["required"])
check("模块级注册", mod.NODE_CLASS_MAPPINGS.get("SFInputPath") is SFInputPath)

node = SFInputPath()

# ── 2. 空输入 ──
check("空串 → 空串", node.execute("") == ("",))
check("None → 空串", node.execute(None) == ("",))
check("纯空白 → 空串", node.execute("   ") == ("",))

# ── 3. input 相对名解析 ──
check("纯文件名", node.execute("clip.mp4") == (os.path.join(INPUT_DIR, "clip.mp4"),))
check("子路径", node.execute("sub/clip.mp4") == (os.path.join(INPUT_DIR, "sub", "clip.mp4"),))
check("首尾空白剔除", node.execute("  clip.mp4  ") == (os.path.join(INPUT_DIR, "clip.mp4"),))
check("[input] 注解", node.execute("clip.mp4 [input]") == (os.path.join(INPUT_DIR, "clip.mp4"),))

# ── 4. 已是绝对路径 ──
abs_file = os.path.join(INPUT_DIR, "abs.mp4")
open(abs_file, "w").close()
check("绝对存在路径原样", node.execute(abs_file) == (abs_file,))
check("绝对路径 normpath", node.execute(INPUT_DIR + "//abs.mp4") == (abs_file,))

# ── 5. 解析失败 ──
try:
    node.execute("../escape.mp4")
    check("目录穿越抛 ValueError", False)
except ValueError as exc:
    check("目录穿越抛 ValueError", "SF Input Path" in str(exc) and "escape.mp4" in str(exc))
except Exception as exc:  # noqa: BLE001
    check("目录穿越抛 ValueError", False)

print()
if failures:
    print(f"FAILED: {len(failures)} 项 -> {failures}")
    sys.exit(1)
print("ALL PASSED")
