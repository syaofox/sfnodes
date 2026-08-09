"""SFValueDropdown - a list you write, one value out.

复刻 Pixaroma Dropdown：每行条目 = 短名 + 它代表的实际值。选中 'warm light'
而不是粘贴一整句触发词。列表、类型与运行模式都归属节点自身，存于
node.properties（随工作流保存），与接线无关。

后端仅薄封装：真实逻辑全部在 sf_utils/dropdown.py（纯函数，可独立测试）。
前端 JS（web/sf_dropdown*.js，Pattern #9）在 graphToPrompt 时把选中的值以
LEAN 形状 {"type", "value"} 注入隐藏 DropdownState 输入；本节点对拿到的
字符串做解析与类型强制转换后输出。
"""

from ...sf_utils.common import AnyType
from ...sf_utils.dropdown import selected_value

_CATEGORY = "sfnodes/text"

any_type = AnyType("*")


class ValueDropdown:
    DESCRIPTION = (
        "SF Value Dropdown - 自己填写的下拉列表：每个条目有一个短名和它代表的实际值，"
        "于是你选 'warm light' 而不是每次粘贴一整句。适合 LoRA 触发词、常用尺寸、"
        "步数，或任何你经常重打的值。\n\n"
        "点节点上的齿轮（或右键）打开设置：添加条目、选择输出类型——文本、整数、"
        "小数或开关——列表随工作流保存，分享工作流即分享条目。输出点随类型改名，"
        "一眼可见会输出什么。\n\n"
        "节点上的小字母决定每次运行发送哪一条，点击即可切换：F 固定你选中的条目，"
        "I 每次运行顺延到下一条（到底回绕），R 每次随机一条。\n\n"
        "列表和类型都归节点自身，与接了什么线无关。Export/Import 可在工作流之间"
        "搬运列表。搜索 dropdown、list、options、preset、choose、pick 可找到它。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        # Hidden, not required: a required STRING would show as a widget AND as
        # a convertible input dot in the Vue frontend. The browser injects the
        # real value at graphToPrompt time.
        return {
            "required": {},
            "hidden": {"DropdownState": ("STRING", {"default": "{}"})},
        }

    # ANY, exactly as Control Panel declares its outputs. The TYPED appearance
    # is a frontend concern: web/sf_dropdown.js sets node.outputs[0].type so
    # LiteGraph refuses an incompatible drag on the canvas. There is no second,
    # server-side type check behind that.
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("value",)
    OUTPUT_TOOLTIPS = (
        "所选条目背后的实际值。值的类型跟随节点设置：文本、整数、小数或开关。",
    )
    FUNCTION = "run"
    CATEGORY = _CATEGORY

    def run(self, DropdownState="{}"):
        return (selected_value(DropdownState),)


NODE_CLASS_MAPPINGS = {"SFValueDropdown": ValueDropdown}
NODE_DISPLAY_NAME_MAPPINGS = {"SFValueDropdown": "SF Value Dropdown"}
