"""SFTextFindReplace - 拦截 STRING 并应用查找/替换规则。

复刻 Pixaroma Find & Replace Pixaroma：

后端契约：
- 1 个 required STRING 输入（text, forceInput）携带上游文本。
- 1 个 hidden STRING 输入（FindReplaceState）携带规则 + 全局开关
  （前端 app.graphToPrompt hook 注入，Pattern #9，与 SFPauseText 同架构）。
- 1 个 STRING 输出（text）携带编辑后的结果。

替换逻辑与 web/sf_find_replace_lib.js（applyRulesJS）镜像，使节点上的实时预览
与真实输出一致。本 Python 实现是权威版本；literal 模式完全一致，regex 反向引用
语法不同（这里 \\1，JS 里 $1）是文档化的预览偏差（见 lib 头注释）。
"""

import json
import re

_CATEGORY = "sfnodes/text"

# 存入并发送到前端预览的输入/输出字符数。实际 STRING 输出永不封顶——只有预览
# 样本被限制，避免膨胀工作流文件或 websocket 负载。
_PREVIEW_CAP = 4000


class SFTextFindReplace:
    DESCRIPTION = (
        "SF Text Find Replace - 拦截一段文本并应用查找/替换规则。把它放在文本源"
        "（LLM 节点、Show Text、任何 STRING 输出）与使用文本的节点之间的连线上："
        "它拦截文本、应用你的查找/替换规则、把编辑后的结果继续传下去，并在节点上"
        "直接预览前后对比。\n\n"
        "每条规则一次编辑：输入要查找的文本与替换文本。替换留空 = 删除查找到的"
        "文本。切换规则开关可跳过它而不删除；拖动 ⋮⋮ 把手调整顺序。规则从上到下"
        "依次应用，每条规则看到上一条的结果。\n\n"
        "全局开关：Case（区分大小写）、Whole word（只匹配整个单词，'art' 不会命中"
        "'artist'）、Regex（把查找内容当作正则表达式）、Tidy（编辑完成后，折叠多余"
        "空格并修复孤立的或重复的逗号）。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": (
                            "要编辑的文本。接入 STRING 输出（LLM 节点、Show Text、"
                            "任何文本源）。节点在透传时编辑它。"
                        ),
                    },
                ),
            },
            "hidden": {"FindReplaceState": ("STRING", {"default": ""})},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_TOOLTIPS = (
        "应用所有查找/替换规则（及 Tidy）之后的文本。",
    )
    FUNCTION = "apply"
    CATEGORY = _CATEGORY
    OUTPUT_NODE = True

    # 有意不设 IS_CHANGED（SFPauseText 踩坑经验）：规则与开关都在隐藏
    # FindReplaceState 输入里，属于 inputs、已在缓存键中，真正的变化仍会
    # 重跑本节点及其下游。设 NaN 会让缓存键折叠所有祖先，导致下游每次全量重跑。

    def apply(self, text, FindReplaceState=""):
        text = text if isinstance(text, str) else ("" if text is None else str(text))
        state = self._parse_state(FindReplaceState)
        result, warnings = _apply_rules(text, state)
        ui = {
            "sf_find_replace": [
                {
                    "input": text[:_PREVIEW_CAP],
                    "output": result[:_PREVIEW_CAP],
                    "truncated": len(text) > _PREVIEW_CAP or len(result) > _PREVIEW_CAP,
                    "warnings": warnings,
                }
            ]
        }
        return {"ui": ui, "result": (result,)}

    @staticmethod
    def _parse_state(raw):
        try:
            s = json.loads(raw) if isinstance(raw, str) else {}
            return s if isinstance(s, dict) else {}
        except (ValueError, TypeError):
            return {}


def _unbounded_quant_at(src, j):
    """True 如果无界量词（* + 或 {n,}）从索引 j 开始。"""
    if j >= len(src):
        return False
    c = src[j]
    if c == "*" or c == "+":
        return True
    return re.match(r"\{\d*,\}", src[j:]) is not None


# 交替型指数回溯：_alternation_overlap_risk 的专用哨兵。
_ANY_CHARS = object()   # 未知/任意字符集合（. [^..] \d 等）
_EMPTY = object()       # 空分支（如 (a|)+）：与任何非空分支都可重叠


def _split_top_level_alt(body):
    """按顶层 | 切分组体（跳过转义/字符类/嵌套组），返回分支片段列表。"""
    branches = []
    start = 0
    escaped = False
    in_class = False
    depth = 0
    i = 0
    n = len(body)
    while i < n:
        c = body[i]
        if escaped:
            escaped = False
            i += 1
            continue
        if c == "\\":
            escaped = True
            i += 1
            continue
        if in_class:
            if c == "]":
                in_class = False
            i += 1
            continue
        if c == "[":
            in_class = True
            i += 1
            continue
        if c == "(":
            depth += 1
            i += 1
            continue
        if c == ")":
            depth -= 1
            i += 1
            continue
        if c == "|" and depth == 0:
            branches.append(body[start:i])
            start = i + 1
            i += 1
            continue
        i += 1
    branches.append(body[start:])
    return branches


def _class_first_chars(seg):
    """解析字符类 [..] 的首字符集合；否定/含转义类/超大范围时保守返回 _ANY_CHARS。"""
    negate = False
    chars = set()
    j = 1
    n = len(seg)
    while j < n:
        c = seg[j]
        if c == "^" and j == 1:
            negate = True
            j += 1
            continue
        if c == "\\":
            if j + 1 < n:
                e = seg[j + 1]
                if e in "dDwWsS":
                    return _ANY_CHARS
                chars.add(e)
                j += 2
                continue
        if c == "]":
            break
        if j + 2 < n and seg[j + 1] == "-" and seg[j + 2] != "]":
            a, b = c, seg[j + 2]
            if ord(b) - ord(a) <= 64:
                chars.update(chr(x) for x in range(ord(a), ord(b) + 1))
            else:
                return _ANY_CHARS
            j += 3
            continue
        chars.add(c)
        j += 1
    if negate:
        return _ANY_CHARS
    return chars


def _branch_first_chars(branch):
    """分支的首字符集合。返回 set / _ANY_CHARS / _EMPTY / None（无法判定，跳过该组）。"""
    if not branch:
        return _EMPTY
    j = 0
    n = len(branch)
    # 跳过行/文本断言
    while j < n and branch[j] in "^$":
        j += 1
    if j >= n:
        return _EMPTY
    c = branch[j]
    if c == "\\":
        if j + 1 >= n:
            return None
        e = branch[j + 1]
        if e in "dDwWsS":
            return _ANY_CHARS
        if e in "bBAZ":  # 断言（\b \B \A \Z）：无字符，继续看下一个
            return _branch_first_chars(branch[j + 2:])
        return {e}
    if c == "[":
        return _class_first_chars(branch[j:])
    if c == ".":
        return _ANY_CHARS
    if c == "(":
        return None  # 嵌套组首字符难算；内层组会被独立检测
    if c in "*+?{":
        return None  # 量词开头（非法/罕见），保守跳过
    return {c}


def _alternation_overlap_risk(src):
    """交替型指数回溯启发式：(a|aa)+ / (a|a?)+ / (a|)+ 家族。

    组内顶层 | 分出至少两个分支、任意两分支的首字符集合重叠、且组后紧跟无界
    量词（* + {n,}）→ 该组匹配方式随输入长度指数增长（经典 ReDoS）。
    分支互斥（(a|b)+）不命中——两分支首字符 {a}/{b} 无交集。
    必须与 web/sf_find_replace_lib.js 的 alternationOverlapRisk 保持同步。
    """
    # 1) 找出所有 ( ... ) 组的范围（跳过转义与字符类）
    groups = []
    stack = []
    escaped = False
    in_class = False
    i = 0
    n = len(src)
    while i < n:
        c = src[i]
        if escaped:
            escaped = False
            i += 1
            continue
        if c == "\\":
            escaped = True
            i += 1
            continue
        if in_class:
            if c == "]":
                in_class = False
            i += 1
            continue
        if c == "[":
            in_class = True
            i += 1
            continue
        if c == "(":
            stack.append(i)
            i += 1
            continue
        if c == ")":
            if stack:
                groups.append((stack.pop(), i))
            i += 1
            continue
        i += 1

    for gs, ge in groups:
        body = src[gs + 1:ge]
        branches = _split_top_level_alt(body)
        if len(branches) < 2:
            continue
        # 组后必须紧跟无界量词（lazy 变体 +? *? 以 +/* 开头，同样命中）
        q = src[ge + 1] if ge + 1 < n else ""
        if q not in ("*", "+"):
            if q == "{" and re.match(r"\{\d*,\}", src[ge + 1:]):
                pass
            else:
                continue
        # 分支首字符两两重叠判定
        firsts = [_branch_first_chars(b) for b in branches]
        skip = False
        seen = set()
        for f in firsts:
            if f is None:
                skip = True
                break
            if f is _EMPTY or f is _ANY_CHARS:
                return True  # 空分支或任意匹配分支与其它分支必重叠
            if seen & f:
                return True
            seen |= f
        if skip:
            continue
    return False


def _is_catastrophic_regex(src):
    """启发式 ReDoS 防护 - 镜像 web/sf_find_replace_lib.js 的 isCatastrophicRegex。

    标记嵌套的无界量词（无界量词限定的组、其体内还含无界量词，如 (a+)+ (a*)*
    (.*)* (\\w+)+）与交替型指数回溯（(a|aa)+ (a|a?)+ (a|)+，两分支首字符重叠），
    这种模式可能指数级回溯。它在服务端每次 Run 时执行且无超时，所以此类模式会
    卡死 worker；我们跳过该规则并给出警告。启发式而非完备；误报率低（嵌套无界
    量词总是冗余的，交替型命中要求分支首字符重叠）。必须与 JS 版本保持同步，
    使节点上的预览与运行一致。
    """
    stack = []  # 每个打开的组一个 dict；"inner" = 组体内含无界量词
    escaped = False
    in_class = False
    i = 0
    n = len(src)
    while i < n:
        c = src[i]
        if escaped:
            escaped = False
            i += 1
            continue
        if c == "\\":
            escaped = True
            i += 1
            continue
        if in_class:
            if c == "]":
                in_class = False
            i += 1
            continue
        if c == "[":
            in_class = True
            i += 1
            continue
        if c == "(":
            stack.append({"inner": False})
            i += 1
            continue
        if c == ")":
            grp = stack.pop() if stack else {"inner": False}
            quant = _unbounded_quant_at(src, i + 1)
            if quant and grp["inner"]:
                return True
            if quant and stack:
                stack[-1]["inner"] = True
            i += 1
            continue
        if _unbounded_quant_at(src, i):
            if stack:
                stack[-1]["inner"] = True
            i += 1
            continue
        i += 1
    if _alternation_overlap_risk(src):
        return True
    return False


def _apply_rules(text, state):
    """按顺序应用启用的规则。返回 (result, warnings)。"""
    rules = state.get("rules", [])
    case_sensitive = bool(state.get("caseSensitive", False))
    whole_word = bool(state.get("wholeWord", False))
    use_regex = bool(state.get("regex", False))
    tidy = bool(state.get("tidy", True))
    warnings = []

    out = text
    if isinstance(rules, list):
        for idx, rule in enumerate(rules):
            if not isinstance(rule, dict):
                continue
            if not rule.get("enabled", True):
                continue
            # 非字符串 find/replace 强转为 ""（镜像 JS 的 readState 强转），使
            # 手工编辑过的畸形状态不会让 apply() 抛 TypeError/AttributeError——
            # 下面只捕获 re.error。
            find = rule.get("find", "")
            if not isinstance(find, str):
                find = ""
            if not find:
                continue
            repl = rule.get("replace", "")
            if not isinstance(repl, str):
                repl = ""
            flags = 0 if case_sensitive else re.IGNORECASE
            try:
                if use_regex:
                    if _is_catastrophic_regex(find):
                        warnings.append(
                            "Rule %d: pattern may be catastrophically slow "
                            "(nested quantifier) - simplify it" % (idx + 1)
                        )
                        continue
                    out = re.sub(find, repl, out, flags=flags)
                else:
                    pattern = re.escape(find)
                    if whole_word:
                        pattern = r"\b" + pattern + r"\b"
                    # 替换文本中的反斜杠转义，使字面含 "\1" 或 "\g<1>" 的字符串
                    # 不会被解释成反向引用。
                    safe_repl = repl.replace("\\", "\\\\")
                    out = re.sub(pattern, safe_repl, out, flags=flags)
            except re.error as exc:
                warnings.append("Rule %d: invalid regex (%s)" % (idx + 1, exc))
                continue

    if tidy:
        out = _tidy(out)
    return out, warnings


def _tidy(s):
    """保守清理。镜像 web/sf_find_replace_lib.js 的 tidy()。

    折叠连续空格/Tab 并修复逗号间距。内部换行保留（永不折叠）；最后的 strip()
    从整个字符串两端去掉首尾空白（含换行）。
    """
    # 连续空格/Tab 折叠为单个空格。
    s = re.sub(r"[ \t]+", " ", s)
    # 逗号前的空格/Tab -> 去掉。
    s = re.sub(r"[ \t]+,", ",", s)
    # 重复逗号（可含空格/Tab 分隔）折叠为一个。
    s = re.sub(r",(?:[ \t]*,)+", ",", s)
    # 每行行尾的尾随空格/Tab 去掉。
    s = re.sub(r"[ \t]+(\r?\n)", r"\1", s)
    # 去掉删除后留下的行首孤立逗号。
    s = re.sub(r"^[ \t]*,[ \t]*", "", s)
    # 去掉删除后留下的行尾孤立逗号。
    s = re.sub(r",[ \t]*$", "", s)
    return s.strip()
