"""SFConditioningCombine —— 多路 Conditioning 合并。

等价于多个原生 Conditioning (Combine) 串联：原生 combine 只是列表拼接
(conditioning_1 + conditioning_2)，本节点按槽位序号依次拼接 N 路输入，
方便多条提示词流汇入同一 KSampler。

动态槽位由前端 web/sf_conditioning_combine.js 管理（复用
sf_dynamic_slots.installDynamicSlots）：初始 conditioning_1/2，全连自动
追加、断开回收尾部空槽，上限 MAX_CONDITIONING_INPUTS。后端用灵活
optional schema（复刻 nodes/logic.py _AnySwitchInputs 模式，类型换成
CONDITIONING）放行前端动态添加的槽名，未连接槽以 None 跳过。
"""

_CATEGORY = "sfnodes/model"

INITIAL_CONDITIONING_INPUTS = 2
MAX_CONDITIONING_INPUTS = 20
CONDITIONING_PREFIX = "conditioning_"


class _ConditioningCombineInputs(dict):
    """灵活 optional 输入：前端动态添加的 conditioning_N 槽名都能通过校验。

    复刻 nodes/logic.py _AnySwitchInputs 的模式（__contains__ 恒 True 让
    get_input_info / 前端 validatePrompt 放行，__getitem__ 回退返回类型），
    仅把类型参数换成 CONDITIONING——不可直接复用该类（它硬编码 any_type）。

    本模块的常量与本类由 SFConditioningConcat（conditioning_concat.py）
    同包复用：语义同为 conditioning_N 动态多槽，不再各写一份。
    """

    def __contains__(self, key):
        return True

    def __getitem__(self, key):
        return ("CONDITIONING",)


def conditioning_slot_key(key):
    """conditioning_N 槽名排序键：按数字后缀排，非数字后缀沉底。

    combine 与 concat 共用（kwargs 迭代序≠槽序，不可直接迭代）。
    """
    try:
        return (int(key.rsplit("_", 1)[1]), key)
    except (ValueError, IndexError):
        return (1 << 30, key)


class SFConditioningCombine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": _ConditioningCombineInputs(),
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "多路 Conditioning 合并：等价于多个原生 Conditioning (Combine) 串联，按槽位序号依次拼接。连上即用，输入槽会自动增删（初始 2，上限 20），未连接槽自动跳过"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # conditioning_N 槽由前端动态增删，超出静态 schema 时接管校验
        return True

    def execute(self, **kwargs):
        out = []
        for key in sorted(kwargs, key=conditioning_slot_key):
            if not key.startswith(CONDITIONING_PREFIX):
                continue
            value = kwargs[key]
            if value is None:
                continue
            out = out + list(value)
        return (out,)
