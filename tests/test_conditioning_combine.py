"""SFConditioningCombine mock tests（纯列表拼接，无 torch/comfy 依赖）."""
import importlib.util

spec = importlib.util.spec_from_file_location(
    "conditioning_combine", "nodes/model/conditioning_combine.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
SFConditioningCombine = mod.SFConditioningCombine
_ConditioningCombineInputs = mod._ConditioningCombineInputs


def assert_eq(a, b, msg=""):
    if a != b:
        raise AssertionError(f"{msg}: {a!r} != {b!r}")


def cond(*tags):
    return [[[t], {"pooled": t}] for t in tags]


node = SFConditioningCombine()

# 1. 双路按序拼接（原生 Combine 语义）
res, = node.execute(conditioning_1=cond("a"), conditioning_2=cond("b"))
assert_eq([c[0][0] for c in res], ["a", "b"], "two-way order")

# 2. 三路拼接
res, = node.execute(conditioning_1=cond("a"), conditioning_2=cond("b"), conditioning_3=cond("c"))
assert_eq([c[0][0] for c in res], ["a", "b", "c"], "three-way order")

# 3. 未连接槽（None）自动跳过
res, = node.execute(conditioning_1=cond("a"), conditioning_2=None, conditioning_3=cond("c"))
assert_eq([c[0][0] for c in res], ["a", "c"], "None skipped")

# 4. 全空返回空列表（不抛错）
res, = node.execute(conditioning_1=None, conditioning_2=None)
assert_eq(res, [], "all None -> empty")

# 5. 无输入返回空列表
res, = node.execute()
assert_eq(res, [], "no inputs -> empty")

# 6. 非前缀键忽略
res, = node.execute(conditioning_1=cond("a"), other="x")
assert_eq([c[0][0] for c in res], ["a"], "non-prefix ignored")

# 7. 数字排序（kwargs 乱序仍按槽号拼接，conditioning_10 在 conditioning_2 之后）
res, = node.execute(conditioning_10=cond("j"), conditioning_2=cond("b"), conditioning_1=cond("a"))
assert_eq([c[0][0] for c in res], ["a", "b", "j"], "numeric sort")

# 8. 多条目 conditioning（单路多 cond 条目保持内部顺序）
res, = node.execute(conditioning_1=cond("a1", "a2"), conditioning_2=cond("b1"))
assert_eq([c[0][0] for c in res], ["a1", "a2", "b1"], "multi-entry order")

# 9. 元数据契约
assert_eq(SFConditioningCombine.RETURN_TYPES, ("CONDITIONING",))
assert_eq(SFConditioningCombine.RETURN_NAMES, ("conditioning",))
assert SFConditioningCombine.CATEGORY == "sfnodes/model"
assert SFConditioningCombine.DESCRIPTION
assert SFConditioningCombine.VALIDATE_INPUTS(conditioning_99=("CONDITIONING",)) is True
assert "conditioning_99" in _ConditioningCombineInputs()
assert _ConditioningCombineInputs()["conditioning_99"] == ("CONDITIONING",)

print("test_conditioning_combine: 13 assertions passed")
