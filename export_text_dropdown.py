#!/usr/bin/env python3
"""一次性迁移工具：SF Text Dropdown 配置 -> SF Value Dropdown 导入文件。

读取 user/sfnodes/text-dropdown.json（TextDropdown 的全局设置），转换为与
SF Value Dropdown 面板 Export 完全一致的格式（categories + 行 category），
在画布加任意 SF Value Dropdown 节点 -> 齿轮 -> Import 即可导入。

用法：
    python3 export_text_dropdown.py [--input PATH] [--output PATH]

输入路径解析顺序（宿主与容器内布局一致，通常无需指定）：
    1. --input 显式指定
    2. 脚本所在目录的 ../../user/sfnodes/text-dropdown.json
    3. 当前目录的 user/sfnodes/text-dropdown.json

迁移确认成功后，本脚本与 text-dropdown.json 均可删除。
"""

import argparse
import json
import os
import sys


def find_input(explicit):
    if explicit:
        return explicit
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "..", "..", "user", "sfnodes", "text-dropdown.json"),
        os.path.join(os.getcwd(), "user", "sfnodes", "text-dropdown.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def convert(data):
    """TextDropdown config -> ValueDropdown export shape. Never raises."""
    if not isinstance(data, dict):
        data = {}

    raw_cats = data.get("categories")
    cats = []
    for c in raw_cats if isinstance(raw_cats, list) else []:
        if isinstance(c, str) and c.strip() and c.strip() not in cats:
            cats.append(c.strip())
    if "default" not in cats:
        cats.insert(0, "default")

    raw_opts = data.get("options")
    options = []
    seen = set()
    for o in raw_opts if isinstance(raw_opts, list) else []:
        if not isinstance(o, dict):
            continue
        alias = o.get("alias")
        content = o.get("content")
        alias = alias if isinstance(alias, str) else ""
        content = content if isinstance(content, str) else ("" if content is None else str(content))
        cat = o.get("category")
        cat = cat.strip() if isinstance(cat, str) and cat.strip() else "default"
        if cat not in cats:
            cats.append(cat)
        name = alias.strip()
        if not name:
            name = content[:20]  # 空别名防御：用内容开头可辨
        if name in seen:
            # 跨分类别名可能重名（TextDropdown 只在分类内唯一），加前缀可辨。
            name = "[%s] %s" % (cat, name)
        seen.add(name)
        options.append({"name": name, "value": content, "category": cat})

    # 当前分类选第一个有选项的分类，导入后立即看到内容而非空 default。
    current = "default"
    for c in cats:
        if any(o["category"] == c for o in options):
            current = c
            break

    return {
        "sfnodes": "value_dropdown",
        "version": 1,
        "type": "text",
        "categories": cats,
        "category": current,
        "options": options,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="text-dropdown.json 路径（默认自动探测）")
    parser.add_argument("--output", help="输出 json 路径（默认 value-dropdown-list.json）")
    args = parser.parse_args()

    src = find_input(args.input)
    if not src or not os.path.exists(src):
        print("找不到 text-dropdown.json。用 --input 显式指定路径。", file=sys.stderr)
        sys.exit(1)

    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)

    out = convert(data)
    dst = args.output or "value-dropdown-list.json"
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    counts = {}
    for o in out["options"]:
        counts[o["category"]] = counts.get(o["category"], 0) + 1
    print("已导出 %s" % dst)
    print("分类（%d 个，当前 '%s'）：" % (len(out["categories"]), out["category"]))
    for c in out["categories"]:
        print("  %-28s %d 条" % (c, counts.get(c, 0)))
    print("合计 %d 条。导入：画布加 SF Value Dropdown -> 齿轮 -> Import -> 选此文件。" % len(out["options"]))


if __name__ == "__main__":
    main()
