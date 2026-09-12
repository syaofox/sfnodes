# disk_state.sanitize_filename 测试（H3/H5 共用净化）：
#  - 路径穿越/绝对路径/.. / 空段拒绝
#  - Unicode 保留、非法字符替换、设备名、隐藏文件、限长
# 运行：python tests/test_disk_state.py
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_sf_loader as L

ds = L.load_node("sf_utils/disk_state.py")
sf = ds.sanitize_filename

failures = []

def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")

# 拒绝：路径穿越 / 绝对路径 / 危险段
check("拒绝 ../evil", sf("../evil", "fb") == "fb")
check("拒绝 /etc/passwd", sf("/etc/passwd", "fb") == "fb")
check("拒绝 ..", sf("..", "fb") == "fb")
check("拒绝 .", sf(".", "fb") == "fb")
check("拒绝 a/../b", sf("a/../b", "fb") == "fb")
check("拒绝空串", sf("", "fb") == "fb")
check("拒绝 None", sf(None, "fb") == "fb")
check("拒绝空白", sf("   ", "fb") == "fb")
check("拒绝隐藏文件", sf(".hidden", "fb") == "fb")
# 路径分隔符拍平为 _
check("a/b 拍平", sf("a/b", "fb") == "a_b")
check("a\\b 拍平", sf("a\\b", "fb") == "a_b")
# 保留 Unicode / 空格
check("中文保留", sf("我的角色", "fb") == "我的角色")
check("空格保留", sf("my char", "fb") == "my char")
# 非法字符替换
check("非法字符替换", sf("bad<char>:name?", "fb") == "bad_char__name")
# 设备名
check("CON 加后缀", sf("CON", "fb") == "CON_")
check("con.txt 加后缀", sf("con.txt", "fb") == "con.txt_")
# 常规
check("file.cube 原样", sf("file.cube", "fb") == "file.cube")
check("非字符串回退", sf(123, "fb") == "fb")
# 限长
check("超长截断", len(sf("a" * 500, "fb")) <= 128)

# ── safe_prefix（多段文件名前缀清洗；preview_routes/save_image_exact 共用）──
sp = ds.safe_prefix
check("正常前缀", sp("PauseImage") == "PauseImage")
check("子目录", sp("a/b") == "a/b")
check("非法字符替换", sp("Bad:Name*") == "Bad_Name")
check("空格/中文保留", sp("My 分类") == "My 分类")
check("穿越拒绝", sp("..") == "" and sp("a/../b") == "" and sp("/abs") == "")
check("保留设备名", sp("CON") == "CON_")
check("超长拒绝", sp("x" * 300) == "")
check("非字符串拒绝", sp(None) == "" and sp(123) == "")

# ── mtime_size_sig（缓存失效键）──
import tempfile as _tf
_tmpd = _tf.mkdtemp(prefix="sf_disk_state_")
_fp = os.path.join(_tmpd, "s.json")
with open(_fp, "w") as _f:
    _f.write("{}")
_sig = ds.mtime_size_sig(_fp)
check("签名为 (mtime, size)", isinstance(_sig, tuple) and len(_sig) == 2)
check("缺失返回 None", ds.mtime_size_sig(_fp + ".nope") is None)

# ── atomic_write_json / atomic_write_bytes（tmp+replace 原子写盘）──
_jp = os.path.join(_tmpd, "w.json")
ds.atomic_write_json(_jp, {"a": [1, 2]})
import json as _json
check("json 写读一致", _json.load(open(_jp, encoding="utf-8")) == {"a": [1, 2]})
check("无残留 tmp", not [n for n in os.listdir(_tmpd) if n.endswith(".tmp")])
_bp = os.path.join(_tmpd, "w.bin")
ds.atomic_write_bytes(_bp, b"\x00\x01")
check("bytes 写读一致", open(_bp, "rb").read() == b"\x00\x01")
try:
    ds.atomic_write_bytes(os.path.join(_tmpd, "no-such-dir", "x.bin"), b"y")
    check("失败抛错（调用方自管）", False)
except OSError:
    check("失败抛错（调用方自管）", True)

if failures:
    print(f"\n{failures}")
    sys.exit(1)
print("\nALL PASS")
