#!/usr/bin/env python3
"""SF Prompt Batcher 基目录测试：读写根为 output/prompt（Node 直接运行）。

锁定 2026-09 换根行为：_get_prompt_base_dir == <output>/prompt，
_resolve_folder 钳制域随之迁移（越界/symlink 回退 default 子目录）。
运行：python3 tests/test_prompt_batcher.py
"""
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_sf_loader as L

failures = []


def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")


# ── mock folder_paths / aiohttp ──
tmp_out = tempfile.mkdtemp(prefix="sf_batcher_output_")
fp = types.ModuleType("folder_paths")
fp.get_output_directory = lambda: tmp_out
fp.get_user_directory = lambda: "/nonexistent/user"
sys.modules["folder_paths"] = fp
aiohttp = types.ModuleType("aiohttp")
aiohttp.web = types.ModuleType("aiohttp.web")
aiohttp.web.json_response = lambda *a, **k: None
aiohttp.web.Response = type("Response", (), {})
sys.modules["aiohttp"] = aiohttp
sys.modules["aiohttp.web"] = aiohttp.web

mod = L.load_node("nodes/text/prompt_batcher.py")

expect = os.path.join(tmp_out, "prompt")
check("基目录 == output/prompt", mod._get_prompt_base_dir() == expect)
check("基目录被创建", os.path.isdir(expect))
check("不再读 user/sfnodes", not mod._get_prompt_base_dir().startswith("/nonexistent"))

os.makedirs(os.path.join(expect, "lib1"), exist_ok=True)
check("子目录枚举", mod._list_subdirs() == ["lib1"])
check("子目录解析", mod._resolve_folder("lib1") == os.path.join(expect, "lib1"))
check("缺省回退 default", mod._resolve_folder("") == os.path.join(expect, "default"))
check("越界回退 default", mod._resolve_folder("../../etc") == os.path.join(expect, "default"))
check("DESCRIPTION 指向 output/prompt",
      "output/prompt/" in mod.SFLoadPromptsFromFolder.DESCRIPTION
      and "output/prompt/" in mod.SFSaveTextToFiles.DESCRIPTION)

if failures:
    print(f"{len(failures)} FAILURES")
    sys.exit(1)
print("ALL PASS")
