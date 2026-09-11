"""SFConditioningConcat mock tests（本机无 torch，注入 stub；无 comfy 依赖）."""
import importlib.util
import logging
import sys
import types


class _StubTorch:
    """最小 torch 桩：cat 仅支持嵌套 list 沿 dim=1 拼接（行内拼接 token）。"""

    @staticmethod
    def cat(tensors, dim=0):
        assert dim == 1, f"stub torch.cat only supports dim=1, got {dim}"
        rows = None
        for t in tensors:
            if rows is None:
                rows = [list(r) for r in t]
            else:
                assert len(rows) == len(t), "row count mismatch"
                for i, r in enumerate(t):
                    rows[i] = rows[i] + list(r)
        return rows


sys.modules["torch"] = _StubTorch

# 先加载被复用的 combine 模块（纯逻辑，零依赖）
spec_c = importlib.util.spec_from_file_location(
    "conditioning_combine", "nodes/model/conditioning_combine.py")
mod_c = importlib.util.module_from_spec(spec_c)
sys.modules["conditioning_combine"] = mod_c
spec_c.loader.exec_module(mod_c)

# concat 用相对导入，改写为绝对后 exec（tests/test_any_to_string.py 先例）
import pathlib

src = pathlib.Path("nodes/model/conditioning_concat.py").read_text()
src = src.replace("from .conditioning_combine import", "from conditioning_combine import")
g = {}
exec(src, g)
SFConditioningConcat = g["SFConditioningConcat"]


def assert_eq(a, b, msg=""):
    if a != b:
        raise AssertionError(f"{msg}: {a!r} != {b!r}")


def cond(*token_lists, **kw):
    """构造 conditioning：每条目 [tokens, dict]。"""
    ctx = kw.get("ctx", {})
    return [[[list(t) for t in toks], dict(ctx)] for toks in token_lists]


node = SFConditioningConcat()

# 1. 双路拼接（原生语义）：to 每条之后拼 from 首条
res, = node.execute(conditioning_1=cond([[1, 2]], [[3, 4]]), conditioning_2=cond([[5, 6]]))
assert_eq([c[0] for c in res], [[[1, 2, 5, 6]], [[3, 4, 5, 6]]], "two-way concat")

# 2. 三路按槽序拼接
res, = node.execute(
    conditioning_1=cond([[1]]),
    conditioning_2=cond([[2]]),
    conditioning_3=cond([[3]]),
)
assert_eq(res[0][0], [[1, 2, 3]], "three-way order")

# 3. kwargs 乱序仍按槽号拼接
res, = node.execute(
    conditioning_3=cond([[3]]),
    conditioning_1=cond([[1]]),
    conditioning_2=cond([[2]]),
)
assert_eq(res[0][0], [[1, 2, 3]], "numeric sort")

# 4. None / 空 from 跳过
res, = node.execute(
    conditioning_1=cond([[1]]),
    conditioning_2=None,
    conditioning_3=[],
    conditioning_4=cond([[4]]),
)
assert_eq(res[0][0], [[1, 4]], "None/empty from skipped")

# 5. 无 from 时 to 原样透传
to = cond([[7, 8]], ctx={"pooled": 1})
res, = node.execute(conditioning_1=to)
assert_eq([c[0] for c in res], [[[7, 8]]], "no from passthrough")

# 6. dict 上下文保留且为拷贝（改结果不影响输入）
res, = node.execute(conditioning_1=cond([[1]], ctx={"a": 1}), conditioning_2=cond([[2]]))
assert_eq(res[0][1], {"a": 1}, "ctx kept")
assert res[0][1] is not None
res[0][1]["a"] = 999  # 不应影响下次执行（每次 copy）

# 7. from 多条仅取首条
res, = node.execute(conditioning_1=cond([[1]]), conditioning_2=cond([[2]], [[9]]))
assert_eq(res[0][0], [[1, 2]], "from first only")

# 8. 非前缀键忽略
res, = node.execute(conditioning_1=cond([[1]]), other="x")
assert_eq(res[0][0], [[1]], "non-prefix ignored")

# 9. 缺 base 抛错（fail-fast，明示信息）
for bad in ({}, {"conditioning_1": None}, {"conditioning_2": cond([[2]])}):
    try:
        node.execute(**bad)
    except ValueError as e:
        assert "conditioning_1" in str(e), f"error names slot: {e}"
    else:
        raise AssertionError(f"missing base must raise: {bad!r}")

# 10. 元数据契约
assert_eq(SFConditioningConcat.RETURN_TYPES, ("CONDITIONING",))
assert_eq(SFConditioningConcat.RETURN_NAMES, ("conditioning",))
assert SFConditioningConcat.CATEGORY == "sfnodes/model"
assert SFConditioningConcat.DESCRIPTION
assert SFConditioningConcat.VALIDATE_INPUTS(conditioning_99=("CONDITIONING",)) is True

print("test_conditioning_concat: 16 assertions passed")
