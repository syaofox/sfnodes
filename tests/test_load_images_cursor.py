# SFLoadImagesCursor 后端测试（Node/Python 直接运行：python tests/test_load_images_cursor.py）
# 覆盖：
#   - 结构：CATEGORY/FUNCTION/RETURN_NAMES/DESCRIPTION/INPUT_TYPES + 根 __init__.py 双字典注册
#   - 游标推进与回卷：顺序三张 → 每 Run 一张 → 耗尽回卷；cycle=False 停在最后一张（token 稳定）
#   - IS_CHANGED 只读 peek：连续两次同值且不推进；token 与实际返回文件一致
#   - 实例隔离（unique_id）、file_prefix、cap/skip/nth、reset_token 重置
#   - shuffle 一轮不重复 / random 全覆盖；None 形参（§134 连线输入）容忍
#   - 解码：RGB / 灰度 / RGBA(alpha 反相) / 调色板 transparency
#   - 目录内容变化按 last_path 续跑；state_name 磁盘续跑与 reset_token 失效
#   - 空目录/目录不存在明确报错且 token 稳定；坏图跳过、全坏图报错（不递归）
#   - caption 侧车、filename/folder_path/full_path/index/total 输出
# mock：torch（numpy FakeTensor）/ folder_paths（真实 tmp 目录）/ sfnodes 包结构
import importlib.util
import json
import os
import sys
import tempfile
import types

import numpy as np
from PIL import Image

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

failures = []


def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")


# ── mock torch：from_numpy → FakeTensor（div_/unsqueeze/shape/numpy）──
class FakeTensor:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float32)

    @property
    def shape(self):
        return self.data.shape

    def div_(self, value):
        self.data = self.data / value
        return self

    def unsqueeze(self, dim):
        return FakeTensor(np.expand_dims(self.data, dim))

    def numpy(self):
        return self.data


torch = types.ModuleType("torch")
torch.from_numpy = lambda arr: FakeTensor(arr)
sys.modules["torch"] = torch

# ── mock folder_paths（真实 tmp 目录）──
tmp_user = tempfile.mkdtemp(prefix="sf_lic_user_")
tmp_in = tempfile.mkdtemp(prefix="sf_lic_input_")
tmp_out = tempfile.mkdtemp(prefix="sf_lic_output_")

folder_paths = types.ModuleType("folder_paths")
folder_paths.get_user_directory = lambda: tmp_user
folder_paths.get_input_directory = lambda: tmp_in
folder_paths.get_output_directory = lambda: tmp_out
_IMG_EXT = {"png", "jpg", "jpeg", "webp", "bmp", "gif", "tif", "tiff"}
folder_paths.filter_files_content_types = lambda files, types_: [
    f for f in files if "image" in types_ and f.rsplit(".", 1)[-1].lower() in _IMG_EXT
]
sys.modules["folder_paths"] = folder_paths

# ── 注册 sfnodes 包结构（相对导入可解析）──
for _pkg, _rel in [("sfnodes", "."), ("sfnodes.nodes", "nodes"),
                   ("sfnodes.nodes.image", "nodes/image"),
                   ("sfnodes.sf_utils", "sf_utils")]:
    _m = types.ModuleType(_pkg)
    _m.__path__ = [os.path.join(root, _rel)]
    sys.modules[_pkg] = _m
spec = importlib.util.spec_from_file_location(
    "sfnodes.nodes.image.load_images_cursor",
    os.path.join(root, "nodes", "image", "load_images_cursor.py"),
)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

SFLoadImagesCursor = mod.SFLoadImagesCursor
NODE = SFLoadImagesCursor()


def _save(name, mode="RGB", size=(4, 3), color=(200, 100, 50)):
    path = os.path.join(tmp_in, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new(mode, size, color).save(path)
    return path


# ── 测试素材 ──
_save("set/1.png"); _save("set/2.png", size=(5, 4)); _save("set/10.png")
with open(os.path.join(tmp_in, "set", "note.txt"), "w") as f:
    f.write("x")
_save("prefix/a1.png"); _save("prefix/a2.png"); _save("prefix/b1.png")
_save("decode/rgb.png")
_save("decode/gray.png", mode="L", color=128)
_save("decode/rgba.png", mode="RGBA", size=(2, 2), color=(255, 0, 0, 128))
_pal = Image.new("P", (2, 2), 0)
_pal.putpalette([255, 0, 0, 0, 255, 0] + [0, 0, 0] * 254)
_pal.putpixel((1, 0), 1)
_pal.info["transparency"] = 0
_pal.save(os.path.join(tmp_in, "decode", "pal.png"))
os.makedirs(os.path.join(tmp_in, "empty"), exist_ok=True)
# bad 目录需先有目录再写文件
os.makedirs(os.path.join(tmp_in, "bad"), exist_ok=True)
with open(os.path.join(tmp_in, "bad", "1.png"), "wb") as f:
    f.write(b"not an image")
_save("bad/2.png"); _save("bad/3.png")
os.makedirs(os.path.join(tmp_in, "allbad"), exist_ok=True)
for _n in ("1.png", "2.png"):
    with open(os.path.join(tmp_in, "allbad", _n), "wb") as f:
        f.write(b"broken")
_save("grow/a.png"); _save("grow/b.png"); _save("grow/c.png")
for _n in ("1", "2", "3", "4", "5"):
    _save(f"slice/{_n}.png")
_save("misc/2.png")
with open(os.path.join(tmp_in, "misc", "2.txt"), "w") as f:
    f.write("tag one\n\ntag two\n")


def step(**kw):
    """一轮 = IS_CHANGED（peek）+ load_next（推进）。返回 (token, 输出元组)。"""
    kw = {"folder": "input/set", "unique_id": "u", **kw}
    token = SFLoadImagesCursor.IS_CHANGED(**kw)
    return token, NODE.load_next(**kw)


def names(token, out):
    return out[2]


# ── 1. 结构 ──
check("CATEGORY", SFLoadImagesCursor.CATEGORY == "sfnodes/image")
check("FUNCTION", SFLoadImagesCursor.FUNCTION == "load_next")
check("RETURN_NAMES", SFLoadImagesCursor.RETURN_NAMES == (
    "image", "mask", "filename", "filename_no_ext", "folder_path", "full_path", "caption", "index", "total"))
check("OUTPUT_TOOLTIPS 数量与输出一致", len(SFLoadImagesCursor.OUTPUT_TOOLTIPS) == len(SFLoadImagesCursor.RETURN_TYPES))
check("DESCRIPTION 存在", isinstance(SFLoadImagesCursor.DESCRIPTION, str) and len(SFLoadImagesCursor.DESCRIPTION) > 0)
it = SFLoadImagesCursor.INPUT_TYPES()
check("folder 为 STRING widget", it["required"]["folder"][0] == "STRING")
check("folder 默认 input", it["required"]["folder"][1]["default"] == "input")
check("order 三档", it["optional"]["order"][0] == ["sequential", "shuffle", "random"])
check("cycle 默认开", it["optional"]["cycle"][1]["default"] is True)
check("hidden unique_id", it["hidden"]["unique_id"] == "UNIQUE_ID")
_init_src = open(os.path.join(root, "__init__.py"), encoding="utf-8").read()
check("根注册 CLASS", '"SFLoadImagesCursor": SFLoadImagesCursor,' in _init_src)
check("根注册 DISPLAY", '"SFLoadImagesCursor": "SF Load Images Cursor",' in _init_src)

# ── 2. 顺序推进 / token 变化 / 回卷 ──
_tok1, _o1 = step(unique_id="seq", folder="input/set")
check("第 1 张 1.png", names(_tok1, _o1) == "1.png")
check("index 0 / total 3", _o1[7] == 0 and _o1[8] == 3)
check("filename_no_ext", _o1[3] == "1")
check("folder_path 与 full_path", _o1[4] == os.path.join(tmp_in, "set") and _o1[5] == os.path.join(tmp_in, "set", "1.png"))
_tok2, _o2 = step(unique_id="seq", folder="input/set")
_tok3, _o3 = step(unique_id="seq", folder="input/set")
check("自然序 2.png / 10.png", names(_tok2, _o2) == "2.png" and names(_tok3, _o3) == "10.png")
check("token 每轮不同", len({_tok1, _tok2, _tok3}) == 3)
_tok4, _o4 = step(unique_id="seq", folder="input/set")
check("耗尽回卷到 1.png", names(_tok4, _o4) == "1.png")
check("回卷 token 与首轮不同（cycle 计数）", _tok4 != _tok1)

# ── 3. IS_CHANGED 只读 peek ──
_tokp1 = SFLoadImagesCursor.IS_CHANGED(folder="input/set", unique_id="peek")
_tokp2 = SFLoadImagesCursor.IS_CHANGED(folder="input/set", unique_id="peek")
check("peek 连续两次同值", _tokp1 == _tokp2)
check("peek token 携带下一张路径", _tokp1.endswith(os.path.join(tmp_in, "set", "1.png")))
check("peek 不推进（执行返回同张）", names(_tokp1, NODE.load_next(folder="input/set", unique_id="peek")) == "1.png")

# ── 4. cycle=False：停在最后一张且 token 稳定 ──
for _ in range(3):
    _, _o = step(unique_id="hold", folder="input/set", cycle=False)
check("cycle=False 全程走完", names("", _o) == "10.png")
_tokh1, _oh1 = step(unique_id="hold", folder="input/set", cycle=False)
_tokh2, _oh2 = step(unique_id="hold", folder="input/set", cycle=False)
check("cycle=False 停在最后一张", names(_tokh1, _oh1) == "10.png" and names(_tokh2, _oh2) == "10.png")
check("cycle=False 停住后 token 稳定", _tokh1 == _tokh2)

# ── 5. 实例隔离 ──
_, _a = step(unique_id="inst-a", folder="input/set")
_, _b = step(unique_id="inst-b", folder="input/prefix")
check("两实例各走自己的目录", names("", _a) == "1.png" and names("", _b) == "a1.png")
_, _a2 = step(unique_id="inst-a", folder="input/set")
check("实例 A 不受 B 影响", names("", _a2) == "2.png")

# ── 6. file_prefix / cap / skip / nth ──
_, _p = step(unique_id="pfx", folder="input/prefix", file_prefix="A")
check("前缀过滤（大小写不敏感）", names("", _p) == "a1.png" and _p[8] == 2)
_, _s = step(unique_id="slice", folder="input/slice", skip_first_images=1, select_every_nth=2)
check("skip+nth 切片", names("", _s) == "2.png" and _s[8] == 2)
_, _c = step(unique_id="cap", folder="input/slice", image_load_cap=3)
check("cap 截断", _c[8] == 3)

# ── 7. reset_token 重置 ──
_, _r1 = step(unique_id="reset", folder="input/set")
_, _r2 = step(unique_id="reset", folder="input/set")
check("reset 前推进到 2.png", names("", _r1) == "1.png" and names("", _r2) == "2.png")
_, _r3 = step(unique_id="reset", folder="input/set", reset_token=1)
check("reset_token 改值后回第一张", names("", _r3) == "1.png")

# ── 8. shuffle 一轮不重复 / random 全覆盖 ──
_sh = [names("", step(unique_id="shuf", folder="input/set")[1]) for _ in range(3)]
check("shuffle 一轮不重复且全覆盖", sorted(_sh) == ["1.png", "10.png", "2.png"])
_sh2 = [names("", step(unique_id="shuf", folder="input/set")[1]) for _ in range(3)]
check("shuffle 下一轮重新洗牌且不重复", sorted(_sh2) == ["1.png", "10.png", "2.png"])
_rd_tokens, _rd = [], []
for _ in range(4):
    _t, _o = step(unique_id="rand", folder="input/set", order="random")
    _rd_tokens.append(_t)
    _rd.append(names(_t, _o))
check("random 均来自目录", all(n in ("1.png", "2.png", "10.png") for n in _rd))
check("random token 每轮不同", len(set(_rd_tokens)) == 4)

# ── 9. None 形参容忍（§134 连线输入在 IS_CHANGED 里是 None）──
_tok_none = SFLoadImagesCursor.IS_CHANGED(
    folder="input/set", file_prefix=None, image_load_cap=None,
    skip_first_images=None, select_every_nth=None, order=None, cycle=None,
    reset_token=None, state_name=None, unique_id="none",
)
check("None 形参不抛错且稳定", _tok_none == SFLoadImagesCursor.IS_CHANGED(
    folder="input/set", image_load_cap=None, unique_id="none"))
_, _on = step(unique_id="none", folder="input/set", image_load_cap=None, cycle=None)
check("None 形参执行走默认全量", names("", _on) == "1.png" and _on[8] == 3)

# ── 10. 解码：RGB/灰度/RGBA/调色板 ──
_, _rgb = step(unique_id="dec-rgb", folder="input/decode", file_prefix="rgb")
check("RGB 形状 [1,H,W,3]", _rgb[0].shape == (1, 3, 4, 3))
check("RGB 无 alpha → 全 0 遮罩且与图等大", _rgb[1].shape == (1, 3, 4) and float(_rgb[1].data.sum()) == 0.0)
_, _gray = step(unique_id="dec-gray", folder="input/decode", file_prefix="gray")
check("灰度转 RGB 三通道", _gray[0].shape == (1, 3, 4, 3))
check("灰度值正确（128/255）", abs(float(_gray[0].data.mean()) - 128 / 255.0) < 1e-6)
_, _rgba = step(unique_id="dec-rgba", folder="input/decode", file_prefix="rgba")
check("RGBA → RGB 三通道", _rgba[0].shape == (1, 2, 2, 3))
check("alpha 反相为遮罩", _rgba[1].shape == (1, 2, 2) and abs(float(_rgba[1].data.mean()) - (1 - 128 / 255.0)) < 1e-6)
_, _pal = step(unique_id="dec-pal", folder="input/decode", file_prefix="pal")
check("调色板 transparency → 遮罩区分透明/不透明",
      float(_pal[1].data[0, 0, 0]) == 1.0 and float(_pal[1].data[0, 0, 1]) == 0.0)

# ── 11. 目录内容变化：按 last_path 续跑，不重复已消费 ──
_, _g1 = step(unique_id="grow", folder="input/grow")
check("grow 首张 a.png", names("", _g1) == "a.png")
_save("grow/0.png")  # 插入排序最前的新文件
_d = os.path.join(tmp_in, "grow")
_st = os.stat(_d)
os.utime(_d, ns=(_st.st_mtime_ns + 10 ** 9, _st.st_mtime_ns + 10 ** 9))  # 强制指纹变化
_, _g2 = step(unique_id="grow", folder="input/grow")
check("目录变化后仍接着 b.png（不重复 a）", names("", _g2) == "b.png")

# ── 12. state_name 磁盘续跑 / reset_token 失效 ──
_, _d1 = step(unique_id="disk", folder="input/set", state_name="t1")
_, _d2 = step(unique_id="disk", folder="input/set", state_name="t1")
check("续跑前推进两张", names("", _d1) == "1.png" and names("", _d2) == "2.png")
_rec_path = os.path.join(tmp_user, "sfnodes", "image_cursor", "t1.json")
check("游标文件已写盘", os.path.isfile(_rec_path))
_rec = json.load(open(_rec_path, encoding="utf-8"))
check("记录 last_path/last_index", _rec["last_path"].endswith("2.png") and _rec["last_index"] == 1)
mod._CURSORS.clear()  # 模拟重启/节点重建
_, _d3 = step(unique_id="disk", folder="input/set", state_name="t1")
check("重启后续跑（返回 10.png）", names("", _d3) == "10.png")
mod._CURSORS.clear()
_, _d4 = step(unique_id="disk", folder="input/set", state_name="t1", reset_token=1)
check("reset_token 失效续跑记录（回 1.png）", names("", _d4) == "1.png")
mod._CURSORS.clear()
_, _d5 = step(unique_id="disk", folder="input/set", state_name="t1", file_prefix="1")
check("参数变化不被旧记录污染（prefix 后从 1.png 起）", names("", _d5) == "1.png" and _d5[8] == 2)

# ── 13. 空目录 / 目录不存在：token 稳定 + 明确报错 ──
_tok_empty = SFLoadImagesCursor.IS_CHANGED(folder="input/empty", unique_id="empty")
_tok_empty2 = SFLoadImagesCursor.IS_CHANGED(folder="input/empty", unique_id="empty")
check("空目录 token 稳定且不抛错", _tok_empty == _tok_empty2 and _tok_empty.startswith("empty|"))
try:
    NODE.load_next(folder="input/empty", unique_id="empty")
    check("空目录执行报错", False)
except ValueError as e:
    check("空目录执行报错", "没有可加载的图片" in str(e))
check("目录不存在 VALIDATE 提示", isinstance(SFLoadImagesCursor.VALIDATE_INPUTS("/no/such/dir"), str))
try:
    NODE.load_next(folder="input/no_such_dir", unique_id="missing")
    check("目录不存在执行报错", False)
except ValueError as e:
    check("目录不存在执行报错", "没有可加载的图片" in str(e))

# ── 14. 坏图跳过 / 全坏图报错（不递归）──
_, _b1 = step(unique_id="bad", folder="input/bad")
check("坏图跳过取下一张", names("", _b1) == "2.png")
_, _b2 = step(unique_id="bad", folder="input/bad")
check("坏图后继续推进", names("", _b2) == "3.png")
try:
    NODE.load_next(folder="input/allbad", unique_id="allbad")
    check("全坏图报错（无递归）", False)
except ValueError as e:
    check("全坏图报错（无递归）", "全部加载失败" in str(e))

# ── 15. caption 侧车 / 输出细节 ──
_, _m = step(unique_id="cap2", folder="input/misc")
check("caption 多行合并且过滤空行", _m[6] == "tag one, tag two")
_, _m2 = step(unique_id="nocaption", folder="input/set")
check("无侧车 caption 为空串", _m2[6] == "")

print()
if failures:
    print(f"{len(failures)} FAILURES: {failures}")
    sys.exit(1)
print("ALL PASS")
