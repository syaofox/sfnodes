# SF 原生 LoadImage filename 输出补丁测试（python3 tests/test_native_load_image_filename.py）
# 覆盖 sf_utils/native_load_image_filename.py：
#   - read_enabled：默认开 / 显式关 / 非 dict 注入
#   - install：注册表条目替换、RETURN_TYPES/RETURN_NAMES 末尾追加、load_image
#     返回 3 元组（第 3 = image 原始值）、原类零改动（继承者 super 透传安全）、
#     类属性继承、开关关闭不替换、幂等
#   - 守卫：缺 key / 非子类 / FUNCTION 不符 / 已带 filename / 第三方覆盖
#     load_image / RETURN_TYPES 结构不符 / nodes 模块不可用
#   - 第三方子类未覆盖 load_image：保留其属性 + OUTPUT_IS_LIST 同步
# 全部 fake 类模拟核心结构，无 ComfyUI/torch 依赖。
import os
import sys
import types

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

failures = []


def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")


def assert_eq(a, b, msg=""):
    check(f"{msg}: {a!r} == {b!r}", a == b)


class FakeCoreLoadImage:
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"
    CATEGORY = "image"

    def load_image(self, image):
        self.seen = image
        return ("img", "mask")

    @classmethod
    def IS_CHANGED(cls, image):
        return "hash"

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        return True


class FakeWebcamCapture(FakeCoreLoadImage):
    """模拟 comfy_extras/nodes_webcam.py：super().load_image 直接透传（2 元组）。"""

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_capture"

    def load_capture(self, image, **kwargs):
        return super().load_image(image)


def make_fake_nodes():
    mod = types.ModuleType("nodes")
    mod.LoadImage = FakeCoreLoadImage
    mod.NODE_CLASS_MAPPINGS = {"LoadImage": FakeCoreLoadImage}
    return mod


fake_nodes = make_fake_nodes()
sys.modules["nodes"] = fake_nodes

from sf_utils import native_load_image_filename as mod  # noqa: E402

SETTING = mod.SETTING_ENABLED

# ── read_enabled 语义 ──
assert_eq(mod.read_enabled({}), True, "默认开")
assert_eq(mod.read_enabled({SETTING: False}), False, "显式关")
assert_eq(mod.read_enabled({SETTING: True}), True, "显式开")
assert_eq(mod.read_enabled("not-a-dict"), True, "非 dict 注入回退默认开")

# ── install 正常路径（默认开）──
registry = fake_nodes.NODE_CLASS_MAPPINGS
check("install 返回 True", mod.install(registry=registry, settings={}) is True)
Patched = registry["LoadImage"]
check("注册表条目已替换", Patched is not FakeCoreLoadImage)
assert_eq(list(Patched.RETURN_TYPES), ["IMAGE", "MASK", "STRING"], "RETURN_TYPES 末尾追加")
assert_eq(list(Patched.RETURN_NAMES), ["IMAGE", "MASK", "filename"], "RETURN_NAMES 末尾追加")
node = Patched()
out = node.load_image("sub/a.png")
assert_eq(out, ("img", "mask", "sub/a.png"), "包装返回 3 元组 + image 原始值")
assert_eq(node.seen, "sub/a.png", "原参数未被改写")
assert_eq(Patched().load_image(None)[-1], "", "非字符串给空串")
assert_eq(Patched().load_image("[output]b.png")[-1], "[output]b.png", "注解值原样")
check("patched 标记", getattr(Patched, mod._MARK, False) is True)

# ── 原类零改动（继承者与直接引用安全）──
assert_eq(list(FakeCoreLoadImage.RETURN_TYPES), ["IMAGE", "MASK"], "原类 RETURN_TYPES 不变")
assert_eq(FakeCoreLoadImage().load_image("x"), ("img", "mask"), "原类方法仍 2 元组")
assert_eq(FakeWebcamCapture().load_capture("x"), ("img", "mask"), "WebcamCapture 式 super 透传安全")
assert_eq(FakeCoreLoadImage.IS_CHANGED("x"), "hash", "原类 IS_CHANGED 不变")

# ── 类属性继承 ──
assert_eq(Patched.FUNCTION, "load_image", "FUNCTION 继承")
assert_eq(Patched.CATEGORY, "image", "CATEGORY 继承")
assert_eq(Patched.IS_CHANGED("x"), "hash", "IS_CHANGED 继承")
assert_eq(Patched.VALIDATE_INPUTS("x"), True, "VALIDATE_INPUTS 继承")

# ── 幂等 ──
check("二次 install 幂等返回 False", mod.install(registry=registry, settings={}) is False)
assert_eq(len(registry["LoadImage"].RETURN_TYPES), 3, "二次调用不重复追加")
assert_eq(registry["LoadImage"] is Patched, True, "二次调用类对象不变")

# ── 开关关闭不替换 ──
off_registry = {"LoadImage": FakeCoreLoadImage}
check("开关关闭 install 返回 False", mod.install(registry=off_registry, settings={SETTING: False}) is False)
check("开关关闭注册表不变", off_registry["LoadImage"] is FakeCoreLoadImage)

# ── 守卫 ──
check("缺 key 跳过", mod.install(registry={}, settings={}) is False)


class Foreign:
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"

    def load_image(self, image):
        return ("i", "m")


foreign_registry = {"LoadImage": Foreign}
check("非 LoadImage 子类跳过", mod.install(registry=foreign_registry, settings={}) is False)
check("非子类注册表不变", foreign_registry["LoadImage"] is Foreign)


class WrongFunction(FakeCoreLoadImage):
    FUNCTION = "load_capture"


check("FUNCTION 不符跳过", mod.install(registry={"LoadImage": WrongFunction}, settings={}) is False)
check("FUNCTION 不符未打标记", getattr(WrongFunction, mod._MARK, False) is False)


class AlreadyHasFilename(FakeCoreLoadImage):
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("IMAGE", "MASK", "filename")


check("已带 filename 输出跳过", mod.install(registry={"LoadImage": AlreadyHasFilename}, settings={}) is False)
check("已带 filename 打幂等标记", getattr(AlreadyHasFilename, mod._MARK, False) is True)


class OverridesLoadImage(FakeCoreLoadImage):
    def load_image(self, image):
        return ("i", "m", "extra")


check("第三方覆盖 load_image 跳过", mod.install(registry={"LoadImage": OverridesLoadImage}, settings={}) is False)


class BadReturnTypes(FakeCoreLoadImage):
    RETURN_TYPES = ("LATENT",)


check("RETURN_TYPES 结构不符跳过", mod.install(registry={"LoadImage": BadReturnTypes}, settings={}) is False)


class MismatchNames(FakeCoreLoadImage):
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("IMAGE",)


check("RETURN_NAMES 长度不符跳过", mod.install(registry={"LoadImage": MismatchNames}, settings={}) is False)


# ── 第三方子类未覆盖 load_image：保留属性 + OIL 同步 ──
class ExtendedNoOverride(FakeCoreLoadImage):
    CATEGORY = "custom"
    RETURN_NAMES = ("IMAGE", "MASK")
    OUTPUT_IS_LIST = (False, False)


ext_registry = {"LoadImage": ExtendedNoOverride}
check("子类未覆盖 load_image → 替换", mod.install(registry=ext_registry, settings={}) is True)
Ext = ext_registry["LoadImage"]
assert_eq(list(Ext.RETURN_NAMES), ["IMAGE", "MASK", "filename"], "子类 RETURN_NAMES 追加")
assert_eq(Ext.CATEGORY, "custom", "子类扩展属性保留")
assert_eq(list(Ext.OUTPUT_IS_LIST), [False, False, False], "OUTPUT_IS_LIST 同步追加")
assert_eq(Ext().load_image("x"), ("img", "mask", "x"), "子类路径返回值 3 元组")

# ── 默认注册表路径（sys.modules["nodes"]）──
default_nodes = make_fake_nodes()
sys.modules["nodes"] = default_nodes
check("默认注册表：install 返回 True", mod.install(settings={}) is True)
assert_eq(list(default_nodes.NODE_CLASS_MAPPINGS["LoadImage"].RETURN_TYPES), ["IMAGE", "MASK", "STRING"], "默认注册表已替换")

real_nodes = sys.modules.pop("nodes", None)
sys.modules["nodes"] = None  # import nodes 会抛 ImportError
check("nodes 模块不可用返回 False", mod.install(settings={}) is False)
if real_nodes is None:
    del sys.modules["nodes"]
else:
    sys.modules["nodes"] = real_nodes

# ── 根 __init__.py 接线（导入 + 调用各一次，防丢）──
with open(os.path.join(root, "__init__.py"), encoding="utf-8") as fh:
    root_src = fh.read()
assert_eq(root_src.count("_install_load_image_filename"), 2, "根 __init__ 接线 2 处")

if failures:
    print(f"\n{len(failures)} FAILURES")
    sys.exit(1)
print("\ntest_native_load_image_filename: all assertions passed")
