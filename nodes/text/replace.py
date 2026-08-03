import random
import re
import time
import uuid
from datetime import datetime

from comfy.comfy_types.node_typing import IO

_CATEGORY = "sfnodes/text"

_MAX_REPLACE_SLOTS = 20

_SPECIAL_TOKEN_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)(?::([^}]*))?\}")

_DEFAULT_DATE_FORMAT = "%Y-%m-%d"
_DEFAULT_TIME_FORMAT = "%H:%M:%S"
_DEFAULT_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

_DEFAULT_RANDOM_BOUND = 1000000

_MARKER_TOOLTIP = (
    "模板文本。支持占位符 {1} {2} ... 及特殊标记符（右键模板框快速插入）："
    "{date} 日期、{time} 时间、{datetime} 日期时间、{timestamp} Unix 时间戳、"
    "{random} 随机数、{random:N} 0~N-1 随机数、{uuid} 短 UUID；"
    "时间标记可带 strftime 参数，如 {date:%Y%m%d}、{time:%H:%M}；"
    "refresh 默认开启，每次执行刷新时间戳/随机数"
)


def _apply_special_tokens(text: str) -> str:
    now = datetime.now()

    def repl(match):
        name = match.group(1).lower()
        arg = match.group(2)
        if name == "date":
            return now.strftime(arg or _DEFAULT_DATE_FORMAT)
        if name == "time":
            return now.strftime(arg or _DEFAULT_TIME_FORMAT)
        if name == "datetime":
            return now.strftime(arg or _DEFAULT_DATETIME_FORMAT)
        if name == "timestamp":
            return str(int(time.time()))
        if name == "random":
            try:
                bound = int(arg) if arg is not None else _DEFAULT_RANDOM_BOUND
            except ValueError:
                bound = _DEFAULT_RANDOM_BOUND
            if bound <= 0:
                bound = 1
            return str(random.randint(0, bound - 1))
        if name == "uuid":
            return uuid.uuid4().hex[:8]
        return match.group(0)

    return _SPECIAL_TOKEN_RE.sub(repl, text)


class SFTextReplace:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "refresh": ("BOOLEAN", {
                "default": True,
                "label_on": "enabled",
                "label_off": "disabled",
                "tooltip": "每次执行强制刷新特殊标记符（时间戳/随机数）；关闭后遵循缓存，模板不变时保持生成时的值",
            }),
        }
        for i in range(1, _MAX_REPLACE_SLOTS + 1):
            optional[f"replace_{i}"] = (IO.STRING, {"multiline": False, "default": ""})
        return {
            "required": {
                "template": (IO.STRING, {"multiline": True, "default": "", "tooltip": _MARKER_TOOLTIP}),
            },
            "optional": optional,
        }

    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("string",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "将模板文本中的 {1} {2} 等占位符替换为指定文本；支持特殊标记符 {date} {time} {datetime} {timestamp} {random} {uuid}（可带格式参数，如 {date:%Y%m%d}）"

    @classmethod
    def IS_CHANGED(cls, template, refresh=False, **kwargs):
        if refresh:
            return float("NaN")
        return None

    def execute(self, template, refresh=False, **kwargs):
        result = _apply_special_tokens(template)
        for i in range(1, _MAX_REPLACE_SLOTS + 1):
            replacement = kwargs.get(f"replace_{i}", "") or ""
            if replacement:
                result = result.replace(f"{{{i}}}", replacement)
        return (result,)
