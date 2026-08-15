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

if failures:
    print(f"\n{failures}")
    sys.exit(1)
print("\nALL PASS")
