#!/usr/bin/env python3
"""SF LUT 目录收敛测试：_get_luts_dir 经 disk_state.sf_user_dir 单源（Node 直接运行）。

锁定 P2 收敛行为：Load/Extract 共用的 luts 目录 == <user>/sfnodes/lut，
与包内其余存储（text/krea2/crop_expand presets、lora_routes、character、
styles_selector）同一单源，mock folder_paths.get_user_directory 指向 tmp 隔离。
运行：python3 tests/test_lut_path.py
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


# ── 重依赖打桩（lut.py 顶层 import cv2/torch/scipy/colour，本机无运行时）──
for _name in ("cv2", "torch", "colour"):
    if _name not in sys.modules:
        sys.modules[_name] = types.ModuleType(_name)
scipy_mod = types.ModuleType("scipy")
ndimage_mod = types.ModuleType("scipy.ndimage")
ndimage_mod.gaussian_filter = lambda *a, **k: a[0]
cluster_mod = types.ModuleType("scipy.cluster")
vq_mod = types.ModuleType("scipy.cluster.vq")
vq_mod.kmeans2 = lambda *a, **k: (a[0], [0] * len(a[0]))
sys.modules["scipy"] = scipy_mod
sys.modules["scipy.ndimage"] = ndimage_mod
sys.modules["scipy.cluster"] = cluster_mod
sys.modules["scipy.cluster.vq"] = vq_mod
sys.modules["colour"].LUT3D = object
algebra_mod = types.ModuleType("colour.algebra")
algebra_mod.table_interpolation_tetrahedral = lambda *a, **k: a[0]
sys.modules["colour.algebra"] = algebra_mod
io_mod = types.ModuleType("colour.io")
io_mod.write_LUT = lambda *a, **k: None
sys.modules["colour.io"] = io_mod
sys.modules["colour"].read_LUT = lambda *a, **k: None

# ── mock folder_paths：隔离 user 目录 ──
tmp_user = tempfile.mkdtemp(prefix="sf_lut_path_")
fp = types.ModuleType("folder_paths")
fp.get_user_directory = lambda: tmp_user
fp.base_path = "/nonexistent/old_base_path"
sys.modules["folder_paths"] = fp

disk_state = L.load_node("sf_utils/disk_state.py")
lut = L.load_node("nodes/image/lut.py")

expect = os.path.join(tmp_user, "sfnodes", "lut")
check("luts 目录 == sf_user_dir()/lut", lut._get_luts_dir() == expect)
check("目录被创建", os.path.isdir(expect))
check("与 disk_state.sf_user_dir() 同源",
      lut._get_luts_dir() == os.path.join(disk_state.sf_user_dir(), "lut"))
check("不再读 folder_paths.base_path",
      not lut._get_luts_dir().startswith("/nonexistent"))
check("空目录列出为空", lut._list_lut_files() == [])
check("CATEGORY 规范", lut.SFLoadLUT.CATEGORY == "sfnodes/image"
      and lut.SFApplyLUT.CATEGORY == "sfnodes/image"
      and lut.SFExtractLUT.CATEGORY == "sfnodes/image")
check("DESCRIPTION 齐备", all(bool(c.DESCRIPTION) for c in
      (lut.SFLoadLUT, lut.SFApplyLUT, lut.SFExtractLUT)))

if failures:
    print(f"{len(failures)} FAILURES")
    sys.exit(1)
print("ALL PASS")
