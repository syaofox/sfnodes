# SFParsePath 后端逻辑测试（Node/Python 直接运行：python tests/test_path_parse.py）
# 覆盖：
#   - 结构：CATEGORY、RETURN_TYPES/NAMES、OUTPUT_IS_LIST、OUTPUT_TOOLTIPS 长度与
#     RETURN_* 对齐（新增 parent_folder 时 tooltip 不脱节）、FUNCTION、DESCRIPTION
#   - _parse_entry 解析：全路径/相对子目录/纯文件名/Windows 反斜杠/末尾斜杠/
#     根目录文件/无扩展名，parent_folder = 目录最后一级名称
#   - execute：OUTPUT_IS_LIST 语义（每路输出单项列表）
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_sf_loader as L

mod = L.load_node("nodes/utils/path_parse.py")
SFParsePath = mod.SFParsePath
parse = mod._parse_entry

failures = []


def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")


# ── 1. 结构 ──
check("CATEGORY", SFParsePath.CATEGORY == "sfnodes/utils")
check("FUNCTION", SFParsePath.FUNCTION == "execute")
check("RETURN_TYPES 5 路全 STRING", SFParsePath.RETURN_TYPES == ("STRING",) * 5)
check("RETURN_NAMES",
      SFParsePath.RETURN_NAMES == ("path", "filename", "extension", "stem", "parent_folder"))
check("OUTPUT_IS_LIST 5 路全 True", SFParsePath.OUTPUT_IS_LIST == (True,) * 5)
check("OUTPUT_TOOLTIPS 与 RETURN_TYPES 等长",
      len(SFParsePath.OUTPUT_TOOLTIPS) == len(SFParsePath.RETURN_TYPES))
check("DESCRIPTION 存在", isinstance(getattr(SFParsePath, "DESCRIPTION", None), str)
      and SFParsePath.DESCRIPTION.strip() != "")

# ── 2. _parse_entry ──
check("全路径", parse("/a/b/c.png") == ("/a/b", "c.png", ".png", "c", "b"))
check("相对子目录", parse("sub/clip.mp4") == ("sub", "clip.mp4", ".mp4", "clip", "sub"))
check("多级子目录", parse("x/y/z/img.jpg") == ("x/y/z", "img.jpg", ".jpg", "img", "z"))
check("纯文件名", parse("c.png") == ("", "c.png", ".png", "c", ""))
check("Windows 反斜杠", parse("C:\\a\\b\\c.png") == ("C:/a/b", "c.png", ".png", "c", "b"))
check("Windows 盘根", parse("C:\\c.png") == ("C:", "c.png", ".png", "c", "C:"))
check("末尾斜杠", parse("/a/b/") == ("/a/b", "", "", "", "b"))
check("重复斜杠", parse("a//b/c.png") == ("a//b", "c.png", ".png", "c", "b"))
check("根目录文件", parse("/c.png") == ("", "c.png", ".png", "c", ""))
check("无扩展名", parse("/a/b/README") == ("/a/b", "README", "", "README", "b"))
check("空串", parse("") == ("", "", "", "", ""))

# ── 3. execute（OUTPUT_IS_LIST：每路单项列表）──
out = SFParsePath().execute("/a/b/c.png")
check("execute 返回 5 路", isinstance(out, tuple) and len(out) == 5)
check("execute 各路列表化", all(isinstance(x, list) and len(x) == 1 for x in out))
check("execute 值", tuple(x[0] for x in out) == ("/a/b", "c.png", ".png", "c", "b"))

print()
if failures:
    print(f"FAILED: {len(failures)} 项 -> {failures}")
    sys.exit(1)
print("ALL PASSED")
