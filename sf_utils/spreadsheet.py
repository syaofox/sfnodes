"""Spreadsheet OutputList 纯逻辑（复刻 geroldmeisinger/ComfyUI-outputlists-combiner）。

仅文本表格：CSV / TSV / Markdown。无 ComfyUI / pandas 依赖（stdlib csv），
宿主可直接单测。流程收敛为「文本 → 二维字符串矩阵 → 别名解析 → 7 路输出」：

- 矩阵单元格一律 str，空单元格为 ""（对齐上游 read_csv keep_default_na=False）
- 别名 = 表头/行头 + 列名（A、B…）或 1 基行号，二者都能作为 selector
- 输出顺序与上游 execute 一致：(count, values_dict, values_list, item_a..d)

上游用 pandas 解析；本实现按「仅文本 + 零依赖」收敛，解析差异：
多字符分隔符用 regex 逐行切分（不感知引号内的分隔符）、`#` 注释按行首判定。
"""

import csv
import io
import re

MAX_ROWS_COLS = 2 ** 16
_HEADER_SEPARATOR_RE = re.compile(r":?-{3,}:?")


def decode_separator(separator):
    """解码分隔符转义：`\\t`→制表符、`\\n`→换行、`\\\\`→反斜杠。

    用 latin-1 + backslashreplace 保留非 ASCII 字符（上游 UTF-8 unicode_escape
    会把中文分隔符解成乱码）；解码失败（如单个尾随反斜杠）回退原串。
    `None` 回退默认逗号；空串返回空串（parse_table 视为解析失败）。
    """
    if separator is None:
        return ","
    try:
        return separator.encode("latin-1", "backslashreplace").decode("unicode_escape")
    except Exception:
        return separator


def column_to_index(column):
    """列名 → 0 基下标（A→0、B→1…ZZZZ）；非法或超出 65536 列返回 None。"""
    if not re.fullmatch(r"[A-Z]{1,4}", column or ""):
        return None
    index = 0
    for char in column:
        index = index * 26 + ord(char) - ord("A") + 1
    index -= 1
    if index < 0 or index >= MAX_ROWS_COLS:
        return None
    return index


def column_to_name(index):
    """0 基下标 → 列名（0→A、25→Z、26→AA）。"""
    name = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(ord("A") + remainder) + name
    return name


def is_empty_value(value):
    if value is None:
        return True
    return str(value).strip() == ""


def normalise_value(value):
    """空值归一为 ""，其余保留原值（兼容带 .item() 的数值类型）。"""
    if is_empty_value(value):
        return ""
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    return value


def stringify_value(value):
    value = normalise_value(value)
    return "" if value == "" else str(value)


def _iter_rows(text, separator):
    if len(separator) == 1:
        yield from csv.reader(io.StringIO(text), delimiter=separator)
    else:
        pattern = re.compile(separator)
        for line in text.splitlines():
            yield pattern.split(line)


def parse_table(text, separator):
    """文本 → 二维字符串矩阵（list[list[str]]）；解析失败返回 None。

    - 空文本 / 空分隔符 → None（对齐上游 load 失败 → 全空输出）
    - 剥离 UTF-8 BOM；跳过空行与首列以 `#` 开头的注释行（对齐上游 comment="#"）
    - 各行列数必须一致（对齐 pandas 的 ParserError → 上游 load 失败），列数由首行决定
    """
    if not text or not text.strip() or not separator:
        return None
    text = text.lstrip("\ufeff")
    rows = []
    try:
        for row in _iter_rows(text, separator):
            if not row or (len(row) == 1 and not str(row[0]).strip()):
                continue
            if str(row[0]).lstrip().startswith("#"):
                continue
            rows.append(["" if cell is None else str(cell) for cell in row])
    except Exception:
        return None
    if not rows:
        return None
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        return None
    return rows


def parse_selectors(rows_and_cols, separator):
    """selector 串 → 列表（按解码后的分隔符切分，支持 `\\,` 转义，strip 去空）。"""
    if not rows_and_cols or not rows_and_cols.strip():
        return []
    pattern = re.compile(rf"(?<!\\){re.escape(separator)}")
    selectors = []
    for part in pattern.split(rows_and_cols):
        selector = part.replace("\\" + separator, separator).strip()
        if selector:
            selectors.append(selector)
    return selectors


def find_header(matrix, index, num_headers, is_topdown):
    """从最后一行（列）表头向前搜索首个有效表头；跳过 Markdown 的 `---` 分隔行。"""
    if is_topdown:
        for row_index in range(min(num_headers, len(matrix)) - 1, -1, -1):
            value = str(normalise_value(matrix[row_index][index])).strip()
            if value and not _HEADER_SEPARATOR_RE.fullmatch(value):
                return value
    else:
        for column_index in range(min(num_headers, len(matrix[0])) - 1, -1, -1):
            value = str(normalise_value(matrix[index][column_index])).strip()
            if value and not _HEADER_SEPARATOR_RE.fullmatch(value):
                return value
    return ""


def build_column_aliases(matrix, num_headers):
    """每列的别名列表：[表头, 列名]（无表头或与列名相同则仅 [列名]）。"""
    aliases = []
    for column_index in range(len(matrix[0]) if matrix else 0):
        header = find_header(matrix, column_index, num_headers, True)
        name = column_to_name(column_index)
        current = [name]
        if header and header != name:
            current.insert(0, header)
        aliases.append(current)
    return aliases


def build_row_aliases(matrix, num_headers):
    """每行的别名列表：[行头, 1 基行号]（无行头或与行号相同则仅 [行号]）。"""
    aliases = []
    for row_index in range(len(matrix)):
        header = find_header(matrix, row_index, num_headers, False)
        name = str(row_index + 1)
        current = [name]
        if header and header != name:
            current.insert(0, header)
        aliases.append(current)
    return aliases


def get_default_selectors(aliases):
    return [current[0] for current in aliases]


def resolve_alias(selector, aliases):
    for index, current in enumerate(aliases):
        if selector in current:
            return index
    return None


def build_outputs(matrix, rows_and_cols, separator, is_topdown, num_headers, select_nth):
    """矩阵 + 选择器 → 7 路输出（顺序对齐上游 execute）。

    返回 (count, values_dict, values_list, item_a, item_b, item_c, item_d)。
    矩阵为空、选择器未命中或 select_nth 越界时返回全空（count=0）。

    - top-down：表头行被跳过，记录=数据行，selector 解析到列
    - left-to-right：表头列被跳过，记录=数据列，selector 解析到行（含表头行）
    """
    empty = (0, [], [], [], [], [], [])
    if not matrix:
        return empty
    width = len(matrix[0])
    num_headers = min(int(num_headers or 0), len(matrix) if is_topdown else width)
    selectors = parse_selectors(rows_and_cols, separator)
    if is_topdown:
        aliases = build_column_aliases(matrix, num_headers)
    else:
        aliases = build_row_aliases(matrix, num_headers)
    if not selectors:
        selectors = get_default_selectors(aliases)
    resolved = []
    for selector in selectors:
        index = resolve_alias(selector, aliases)
        if index is None:
            print(f"[SFSpreadsheetOutputList] 选择器 {selector!r} 未命中表头/列名/行号，返回空输出")
            return empty
        resolved.append(index)
    if is_topdown:
        records = matrix[num_headers:]
    else:
        records = [
            [matrix[row_index][column_index] for row_index in range(len(matrix))]
            for column_index in range(num_headers, width)
        ]
    if int(select_nth) >= 0:
        nth = int(select_nth)
        if nth >= len(records):
            print(f"[SFSpreadsheetOutputList] select_nth={nth} 越界（共 {len(records)} 项），返回空输出")
            return empty
        records = [records[nth]]
    values_dict = []
    values_list = []
    for record in records:
        current_dict = {}
        current_values = []
        for index in resolved:
            value = normalise_value(record[index]) if index < len(record) else ""
            current_values.append(value)
            for alias in aliases[index]:
                current_dict[alias] = value
        values_dict.append(current_dict)
        values_list.append(current_values)
    lists = []
    for index in range(4):
        if index < len(resolved):
            lists.append([stringify_value(values[index]) for values in values_list])
        else:
            lists.append([])
    return (len(values_dict), values_dict, values_list, lists[0], lists[1], lists[2], lists[3])
