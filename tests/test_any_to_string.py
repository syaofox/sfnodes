"""SFAnyToString pad_digits mock tests."""
import importlib.util
import pathlib

# load sf_utils/string
spec_u = importlib.util.spec_from_file_location("sf_utils_string", "sf_utils/string.py")
mod_u = importlib.util.module_from_spec(spec_u)
spec_u.loader.exec_module(mod_u)
pad_number_text = mod_u.pad_number_text

# load node via exec (avoid relative import complexity)
src = pathlib.Path("nodes/text/any_to_string.py").read_text()
src = src.replace("from ...sf_utils.common import AnyType", "class AnyType:\n    def __init__(self, t):\n        self.t = t")
src = src.replace("from ...sf_utils.string import pad_number_text", "")
g = {"pad_number_text": pad_number_text}
exec(src, g)
SFAnyToString = g["SFAnyToString"]
any_type = g["any_type"]


def assert_eq(a, b, msg=""):
    if a != b:
        raise AssertionError(f"{msg}: {a!r} != {b!r}")


# ---- pure pad_number_text tests ----
assert_eq(pad_number_text("7", 2), "07", "int pad")
assert_eq(pad_number_text("7", 0), "7", "digits=0 no pad")
assert_eq(pad_number_text("-5", 2), "-05", "negative keeps sign outside")
assert_eq(pad_number_text("+3", 2), "+03", "plus sign kept")
assert_eq(pad_number_text("123", 2), "123", "already long enough")
assert_eq(pad_number_text("007", 2), "007", "no truncate when longer")
assert_eq(pad_number_text("3.5", 2), "3.5", "float not padded")
assert_eq(pad_number_text("2.0", 2), "2.0", "whole float not padded")
assert_eq(pad_number_text("abc", 2), "abc", "non-numeric not padded")
assert_eq(pad_number_text("", 2), "", "empty string")
assert_eq(pad_number_text("7", 5), "00007", "digits larger than len")
assert_eq(pad_number_text("a7", 2), "a7", "mixed not padded")

# ---- node execute tests ----
node = SFAnyToString()

# default pad_digits=2: int value padded
res, = node.execute(prefix="", suffix="", pad_digits=2, value=7)
assert_eq(res, "07", "node default pad int")
# float untouched
res, = node.execute(prefix="", suffix="", pad_digits=2, value=3.5)
assert_eq(res, "3.5", "node float untouched")
# string numeric input padded too
res, = node.execute(prefix="", suffix="", pad_digits=3, value="42")
assert_eq(res, "042", "node string int padded")
# pad_digits=0 disables
res, = node.execute(prefix="", suffix="", pad_digits=0, value=7)
assert_eq(res, "7", "node digits=0")
# None -> empty
res, = node.execute(prefix="", suffix="", pad_digits=2, value=None)
assert_eq(res, "", "node None empty")
# prefix/suffix applied after padding
res, = node.execute(prefix="P", suffix="S", pad_digits=2, value=5)
assert_eq(res, "P05S", "node prefix/suffix after pad")
# negative int
res, = node.execute(prefix="", suffix="", pad_digits=2, value=-9)
assert_eq(res, "-09", "node negative int")

# check metadata
it = SFAnyToString.INPUT_TYPES()
assert "pad_digits" in it["required"]
assert it["required"]["pad_digits"][1]["default"] == 2
assert_eq(SFAnyToString.RETURN_TYPES, ("STRING",))
assert_eq(SFAnyToString.RETURN_NAMES, ("text",))
assert SFAnyToString.CATEGORY == "sfnodes/text"
assert SFAnyToString.DESCRIPTION
assert any_type.t == "*"

print("test_any_to_string: 24 assertions passed")
