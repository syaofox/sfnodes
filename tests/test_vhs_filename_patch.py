# SF VHS filename 输出补丁测试（python3 tests/test_vhs_filename_patch.py）
# 覆盖 sf_utils/vhs_loadvideo_filename.py：
#   - patch_class：守卫（模块标记 / 已带 filename / FUNCTION 不符 / 空 RETURN_TYPES / 名称长度不匹配）、
#     属性追加、包装返回值
#   - install：注册表扫描命中数、幂等、默认注册表路径
#   - install_deferred：就绪直返、有限重试调度、超时告警不崩
# 全部 fake 类模拟 VHS 结构，无 ComfyUI/torch 依赖。
import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

from sf_utils import vhs_loadvideo_filename as mod  # noqa: E402

failures = []


def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")


def assert_eq(a, b, msg=""):
    check(f"{msg}: {a!r} == {b!r}", a == b)


VHS_MODULE = "custom_nodes.ComfyUI-VideoHelperSuite.videohelpersuite.load_video_nodes"


def make_upload_class():
    class FakeUpload:
        __module__ = VHS_MODULE
        RETURN_TYPES = ("IMAGE", "INT", "AUDIO", "VHS_VIDEOINFO")
        RETURN_NAMES = ("IMAGE", "frame_count", "audio", "video_info")
        FUNCTION = "load_video"

        def load_video(self, **kwargs):
            self.seen = dict(kwargs)
            return ("img", 2, "audio", {"loaded_frame_count": 2})

    return FakeUpload


class FakePathWithOIL:
    __module__ = VHS_MODULE
    RETURN_TYPES = ("IMAGE", "MASK", "AUDIO", "VHS_VIDEOINFO")
    RETURN_NAMES = ("IMAGE", "mask", "audio", "video_info")
    OUTPUT_IS_LIST = (False, False, False, False)
    FUNCTION = "load_video"

    def load_video(self, **kwargs):
        return ("img", "mask", "audio", {})


class FakeRaises:
    __module__ = VHS_MODULE
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "load_video"

    def load_video(self, **kwargs):
        raise ValueError("boom")


class FakeNonTuple:
    __module__ = VHS_MODULE
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "load_video"

    def load_video(self, **kwargs):
        return "only"


class FakeForeign:
    __module__ = "some_other_pack.nodes"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "load_video"

    def load_video(self, **kwargs):
        return ("img",)


class FakeAlreadyHasFilename:
    __module__ = VHS_MODULE
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("IMAGE", "filename")
    FUNCTION = "load_video"

    def load_video(self, **kwargs):
        return ("img", "keep.mp4")


class FakeWrongFunction:
    __module__ = VHS_MODULE
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "load_image"

    def load_video(self, **kwargs):
        return ("img",)


class FakeNoReturnTypes:
    __module__ = VHS_MODULE
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "load_video"

    def load_video(self, **kwargs):
        return ()


class FakeMismatchNames:
    __module__ = VHS_MODULE
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "load_video"

    def load_video(self, **kwargs):
        return ("img", 1)


# ── _extract_filename 语义：原始值 ──
assert_eq(mod._extract_filename(None), "", "None → 空串")
assert_eq(mod._extract_filename("sub/clip.mp4"), "sub/clip.mp4", "字符串原样")
assert_eq(mod._extract_filename(3), "3", "非字符串转 str")

# ── patch_class 正常路径 ──
Upload = make_upload_class()
check("patch_class 返回 True", mod.patch_class(Upload) is True)
assert_eq(list(Upload.RETURN_TYPES), ["IMAGE", "INT", "AUDIO", "VHS_VIDEOINFO", "STRING"], "RETURN_TYPES 追加")
assert_eq(list(Upload.RETURN_NAMES), ["IMAGE", "frame_count", "audio", "video_info", "filename"], "RETURN_NAMES 追加")
node = Upload()
out = node.load_video(video="sub/clip.mp4", frame_load_cap=0)
assert_eq(len(out), 5, "包装返回 5 元组")
assert_eq(out[-1], "sub/clip.mp4", "filename = 原始 widget 值")
assert_eq(node.seen["video"], "sub/clip.mp4", "原 kwargs 未被改写")
assert_eq(out[:4], ("img", 2, "audio", {"loaded_frame_count": 2}), "原输出保持")
assert_eq(Upload().load_video(video="https://x/y.mp4")[-1], "https://x/y.mp4", "URL 原样")
assert_eq(Upload().load_video()[-1], "", "缺 video 键给空串")

# ── OUTPUT_IS_LIST 同步扩展 ──
check("patch_class 带 OUTPUT_IS_LIST 返回 True", mod.patch_class(FakePathWithOIL) is True)
assert_eq(list(FakePathWithOIL.OUTPUT_IS_LIST), [False, False, False, False, False], "OUTPUT_IS_LIST 追加")
assert_eq(list(FakePathWithOIL.RETURN_NAMES), ["IMAGE", "mask", "audio", "video_info", "filename"], "带 OIL 的名称追加")

# ── 幂等 ──
check("二次 patch_class 幂等返回 False", mod.patch_class(Upload) is False)
assert_eq(len(Upload.RETURN_TYPES), 5, "二次调用不重复追加")
assert_eq(len(Upload().load_video(video="a.mp4")), 5, "二次调用返回值仍 5 元组")

# ── 守卫 ──
check("外来模块类跳过", mod.patch_class(FakeForeign) is False)
assert_eq(list(FakeForeign.RETURN_TYPES), ["IMAGE"], "外来类属性不变")
check("已带 filename 输出跳过", mod.patch_class(FakeAlreadyHasFilename) is False)
assert_eq(list(FakeAlreadyHasFilename.RETURN_NAMES), ["IMAGE", "filename"], "不重复追加 filename")
assert_eq(FakeAlreadyHasFilename().load_video()[-1], "keep.mp4", "跳过类不被包装")
check("FUNCTION 不符跳过", mod.patch_class(FakeWrongFunction) is False)
assert_eq(list(FakeWrongFunction.RETURN_TYPES), ["IMAGE"], "FUNCTION 不符属性不变")
check("空 RETURN_TYPES 跳过", mod.patch_class(FakeNoReturnTypes) is False)
assert_eq(list(FakeNoReturnTypes.RETURN_TYPES), [], "空 RETURN_TYPES 不变")
check("名称长度不匹配跳过", mod.patch_class(FakeMismatchNames) is False)
assert_eq(list(FakeMismatchNames.RETURN_NAMES), ["IMAGE"], "长度不匹配名称不变")

# ── 异常传播 / 非元组结果兜底 ──
check("异常类 patch 成功", mod.patch_class(FakeRaises) is True)
try:
    FakeRaises().load_video(video="a.mp4")
    check("原函数异常照常抛出（未抛）", False)
except ValueError as exc:
    check(f"原函数异常照常抛出（{exc}）", str(exc) == "boom")
check("非元组类 patch 成功", mod.patch_class(FakeNonTuple) is True)
assert_eq(FakeNonTuple().load_video(video="b.mp4"), ("only", "b.mp4"), "非元组结果包装")

# ── install 注册表扫描 ──
registry4 = {key: make_upload_class() for key in mod._TARGET_KEYS}
assert_eq(mod.install(registry={}), 0, "空注册表 0 命中")
assert_eq(mod.install(registry=registry4), 4, "4 键全部补丁")
assert_eq(mod.install(registry=registry4), 0, "重复 install 幂等")
assert_eq(mod.install(registry={"VHS_LoadVideo": FakeForeign}), 0, "外来类不计数")
assert_eq(len(registry4["VHS_LoadVideo"].RETURN_TYPES), 5, "注册表类已扩展")

# ── install_deferred ──


class FakeLoop:
    def __init__(self):
        self.calls = []

    def call_later(self, delay, cb):
        self.calls.append((delay, cb))


ready = {key: make_upload_class() for key in mod._TARGET_KEYS}
check("已就绪直接返回 True", mod.install_deferred(registry=ready, loop=None) is True)

Pending = make_upload_class()
pending = {"VHS_LoadVideo": Pending}
loop = FakeLoop()
check("缺 key 时返回 False 并调度", mod.install_deferred(registry=pending, loop=loop, retries=3, interval=0) is False)
assert_eq(len(loop.calls), 1, "首次调度一次")
loop.calls[0][1]()  # 仍缺 key → 再调度一次
assert_eq(len(loop.calls), 2, "未就绪时再调度一次")
pending.update({key: make_upload_class() for key in mod._TARGET_KEYS})
loop.calls[1][1]()  # 注册表已补齐 → 命中后不再调度
assert_eq(len(loop.calls), 2, "补齐后不再调度")
check("重试后完成补丁", getattr(Pending, "_sf_filename_output_patched", False) is True)

exhaust = {}
loop2 = FakeLoop()
check("重试耗尽返回 False", mod.install_deferred(registry=exhaust, loop=loop2, retries=1, interval=0) is False)
loop2.calls[0][1]()
assert_eq(len(loop2.calls), 1, "耗尽后不再调度")

# ── 默认注册表路径（模拟 ComfyUI 运行时 sys.modules["nodes"]）──
import types  # noqa: E402

real_nodes = sys.modules.pop("nodes", None)
sys.modules["nodes"] = None  # import nodes 会抛 ImportError
assert_eq(mod.install(), 0, "注册表不可用 install 返回 0")
check("注册表不可用 deferred 返回 False", mod.install_deferred(retries=0) is False)

FakeNodesModule = types.ModuleType("nodes")
DefaultUpload = make_upload_class()
FakeNodesModule.NODE_CLASS_MAPPINGS = {"VHS_LoadVideo": DefaultUpload}
sys.modules["nodes"] = FakeNodesModule
check("默认注册表：缺其余 key 时 deferred 返回 False", mod.install_deferred(loop=FakeLoop(), retries=0) is False)
check("默认注册表：已知 key 已补丁", getattr(DefaultUpload, "_sf_filename_output_patched", False) is True)
for key in mod._TARGET_KEYS[1:]:
    FakeNodesModule.NODE_CLASS_MAPPINGS[key] = make_upload_class()
assert_eq(mod.install_deferred(loop=None) is True, True, "默认注册表：补齐后 ready")
if real_nodes is None:
    del sys.modules["nodes"]
else:
    sys.modules["nodes"] = real_nodes

# ── 根 __init__.py 接线（导入 + 调用各一次，防丢）──
with open(os.path.join(root, "__init__.py"), encoding="utf-8") as fh:
    root_src = fh.read()
assert_eq(root_src.count("_install_vhs_filename_output"), 2, "根 __init__ 接线 2 处")

if failures:
    print(f"\n{len(failures)} FAILURES")
    sys.exit(1)
print("\ntest_vhs_filename_patch: all assertions passed")
