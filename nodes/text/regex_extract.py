import re

from comfy.comfy_types.node_typing import IO

_CATEGORY = "sfnodes/text"

_PRESET_CUSTOM = "自定义"

# 内置正则预设：(显示名, 正则, 默认捕获组)。与 web/regex_extract.js 中的 PRESETS 保持一致
_REGEX_PRESETS = (
    ("提取数字", r"-?\d+(?:\.\d+)?", 0),
    ("提取整数", r"\d+", 0),
    ("提取中文", r"[\u4e00-\u9fff]+", 0),
    ("提取英文单词", r"[A-Za-z]+", 0),
    ("提取邮箱", r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", 0),
    ("提取网址", r'''https?://[^\s"'<>]+''', 0),
    ("提取手机号", r"1[3-9]\d{9}", 0),
    ("提取日期", r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", 0),
    ("提取时间", r"\d{1,2}:\d{2}(?::\d{2})?", 0),
    ("提取圆括号内容", r"\(([^)]*)\)", 1),
    ("提取方括号内容", r"\[([^\]]*)\]", 1),
    ("提取文件扩展名", r"\.([A-Za-z0-9]+)", 1),
)

_PRESET_NAMES = [_PRESET_CUSTOM] + [name for name, _, _ in _REGEX_PRESETS]
_PRESET_BY_NAME = {name: (regex, group) for name, regex, group in _REGEX_PRESETS}


class SFTextRegexExtract:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (
                    IO.STRING,
                    {"multiline": True, "default": "", "tooltip": "要提取的文本，可手填或连接上游节点"},
                ),
                "preset": (
                    _PRESET_NAMES,
                    {
                        "default": _REGEX_PRESETS[0][0],
                        "tooltip": "常用正则预设，选中自动填入 pattern（选「自定义」时以 pattern 为准）",
                    },
                ),
                "pattern": (
                    IO.STRING,
                    {
                        "multiline": False,
                        "default": "",
                        "tooltip": "正则表达式，非空时优先于预设执行；支持捕获组，配合 group 参数提取指定分组",
                    },
                ),
                "match_mode": (
                    ["全部", "第一个", "最后一个", "第 N 个"],
                    {
                        "default": "全部",
                        "tooltip": "多个匹配时：输出全部（用分隔符连接）、只取第一个、最后一个，或按 index 取第 N 个",
                    },
                ),
                "index": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 999999,
                        "tooltip": "match_mode 选「第 N 个」时生效：取第 index 个匹配（从 1 开始计数）",
                    },
                ),
                "group": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 9,
                        "tooltip": "输出捕获组：0=完整匹配，1=第 1 个捕获组，2=第 2 个捕获组...",
                    },
                ),
                "ignore_case": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "on",
                        "label_off": "off",
                        "tooltip": "忽略大小写匹配",
                    },
                ),
                "separator": (
                    IO.STRING,
                    {
                        "multiline": False,
                        "default": "\\n",
                        "tooltip": "多个匹配结果的连接符，支持字面 \\n 与 \\t（分别转为换行与制表符）",
                    },
                ),
            },
        }

    RETURN_TYPES = (IO.STRING, "INT")
    RETURN_NAMES = ("text", "match_count")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "从文本中按正则提取内容：支持全部/第一个/最后一个/第 N 个匹配、捕获组与忽略大小写；内置 12 个常用预设（数字、中文、邮箱、网址、手机号、日期、括号内容等），选中自动填入正则，也可自定义"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # preset 组合选项为静态列表，但旧工作流可能残留已删除/改名的预设值，跳过默认校验由 execute 安全降级
        return True

    @staticmethod
    def _extract_group(match, group):
        try:
            return match.group(group)
        except IndexError:
            return None

    def execute(self, text, preset, pattern, match_mode, index, group, ignore_case, separator):
        group = group if group is not None else 0
        index = index if index is not None else 1
        regex = (pattern or "").strip()
        if not regex:
            entry = _PRESET_BY_NAME.get(preset)
            if entry is None:
                return ("", 0)
            regex = entry[0]

        try:
            compiled = re.compile(regex, re.IGNORECASE if ignore_case else 0)
        except re.error as e:
            print(f"[SFTextRegexExtract] 无效正则 {regex!r}: {e}")
            return ("", 0)

        if match_mode == "第一个":
            match = compiled.search(text or "")
            if match is None:
                return ("", 0)
            return (self._extract_group(match, group) or "", 1)

        if match_mode == "最后一个":
            last = None
            for m in compiled.finditer(text or ""):
                last = m
            if last is None:
                return ("", 0)
            return (self._extract_group(last, group) or "", 1)

        if match_mode == "第 N 个":
            count = 0
            for m in compiled.finditer(text or ""):
                count += 1
                if count == index:
                    return (self._extract_group(m, group) or "", 1)
            return ("", 0)

        results = [g for g in (self._extract_group(m, group) for m in compiled.finditer(text or "")) if g is not None]
        sep = (separator or "").replace("\\n", "\n").replace("\\t", "\t")
        return (sep.join(results), len(results))
