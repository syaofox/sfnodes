#!/usr/bin/env python3
"""sfnodes 文档/注册一致性静态检查（开发辅助，非测试）。

规则：
  A. 根 __init__.py 两注册字典：键集一致、键全部 SF 前缀、显示名全部 "SF " 前缀
  B. experience/：
     B1. 顶级章节 `## N.` 全局唯一、各文件内升序
     B2. README 顶部「当前最大 §N」== 实际最大号
     B3. README 索引：每章节恰好一条、条目标题是对应 `## N.` 标题的前缀、无悬空条目
  C. 引用解析（AGENTS.md + doc/**/*.md + 代码注释）：
     C1. 裸 `§N` / `§N.M`（含更深）的 N 与 N.M 必须存在
     C2. 限定引用「文件名.md §N」：该文件必须含 §N
  D. architecture.md 覆盖：nodes/ sf_utils/ web/ tools/ 下 .py/.js 以文件名、
     词干、目录名（如 rfmsr/）或 `<前缀>*` glob 形式出现
失败以非零退出；不修改任何文件。
"""
import os
import re
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIENCE = os.path.join(BASE, "doc", "experience")
THEME_FILES = [
    "platform.md", "patterns.md", "nodes-text.md", "nodes-image.md",
    "nodes-lora.md", "nodes-video.md", "apps.md",
]

errors = []


def fail(msg):
    errors.append(msg)


# ── A. 注册字典 ──────────────────────────────────────────────────────────
def _dict_body(src, name):
    m = re.search(rf"^{name}\s*=\s*\{{", src, re.M)
    if not m:
        fail(f"[A] 未找到 {name}")
        return None
    i, depth = m.end(), 1
    while depth and i < len(src):
        c = src[i]
        depth += (c == "{") - (c == "}")
        i += 1
    return src[m.end():i - 1]


def check_registrations():
    src = open(os.path.join(BASE, "__init__.py"), encoding="utf-8").read()
    dicts = {}
    for name in ("NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"):
        body = _dict_body(src, name)
        if body is None:
            return
        keys = re.findall(r'^\s*"([^"]+)"\s*:', body, re.M)
        dicts[name] = keys
        if len(keys) != len(set(keys)):
            fail(f"[A] {name} 存在重复键: {[k for k in set(keys) if keys.count(k) > 1]}")
        bad = [k for k in keys if not k.startswith("SF")]
        if bad:
            fail(f"[A] {name} 存在非 SF 前缀键: {bad[:5]}")
    class_keys = set(dicts.get("NODE_CLASS_MAPPINGS", []))
    display_keys = set(dicts.get("NODE_DISPLAY_NAME_MAPPINGS", []))
    if class_keys != display_keys:
        fail("[A] 两注册字典键集不一致: "
             f"仅 CLASS {sorted(class_keys - display_keys)[:5]}, "
             f"仅 DISPLAY {sorted(display_keys - class_keys)[:5]}")
    body = _dict_body(src, "NODE_DISPLAY_NAME_MAPPINGS")
    names = re.findall(r'^\s*"[^"]+"\s*:\s*"([^"]+)"', body, re.M)
    bad = [n for n in names if not n.startswith("SF ")]
    if bad:
        fail(f"[A] 显示名未以 'SF ' 开头: {bad[:5]}")
    print(f"[A] 注册字典：{len(class_keys)} 键，键集一致且全 SF 前缀")


# ── B. 章节唯一性与 README 索引 ──────────────────────────────────────────
def check_experience():
    heads = {}          # (file, N) -> title
    subs = {}           # (file, N) -> set(有效子节号)
    order = {}          # file -> [N]
    for name in THEME_FILES:
        path = os.path.join(EXPERIENCE, name)
        cur = None
        nums = []
        for line in open(path, encoding="utf-8"):
            m = re.match(r"^## (\d+)\.\s*(.+?)\s*$", line)
            if m:
                cur = int(m.group(1))
                if cur in nums:
                    fail(f"[B] {name} 章节号重复: §{cur}")
                nums.append(cur)
                heads[(name, cur)] = m.group(2)
                subs.setdefault((name, cur), set())
                continue
            if cur is None:
                continue
            m = re.match(r"^### §(\d+)\.(\d+)\s", line)
            if m and int(m.group(1)) == cur:
                subs[(name, cur)].add(int(m.group(2)))
                continue
            m = re.match(r"^### (\d+)\.(\d+)\s", line)
            if m:
                x, y = int(m.group(1)), int(m.group(2))
                subs[(name, cur)].add(y if x == cur else x)
                continue
            m = re.match(r"^### (\d+)\.\s", line)
            if m:
                subs[(name, cur)].add(int(m.group(1)))
        # 位置式子节：`### 1.` 自然序号即 §N.M
        cur, cnt = None, 0
        for line in open(path, encoding="utf-8"):
            m = re.match(r"^## (\d+)\.", line)
            if m:
                cur, cnt = int(m.group(1)), 0
                continue
            m = re.match(r"^### (\d+)\.\s", line)
            if m and cur:
                cnt += 1
                subs[(name, cur)] |= set(range(1, cnt + 1))
        if nums != sorted(nums):
            fail(f"[B] {name} 章节号非升序: {nums}")
        order[name] = nums

    all_nums = {n for _, n in heads}
    print(f"[B] experience：{len(heads)} 章节，最大 §{max(all_nums)}")

    readme = open(os.path.join(EXPERIENCE, "README.md"), encoding="utf-8").read()
    m = re.search(r"\*\*当前最大 §N：(\d+)\*\*", readme)
    if not m:
        fail("[B2] README 缺少「当前最大 §N：<数字>」声明")
    elif int(m.group(1)) != max(all_nums):
        fail(f"[B2] README 当前最大 §N={m.group(1)} 与实际 {max(all_nums)} 不一致")
    else:
        print(f"[B2] README 最大 §N 声明与实际一致：{m.group(1)}")

    # README 索引：解析三个表格列（§N 标题 · ...）
    for name in THEME_FILES:
        row = re.search(rf"^\| `{re.escape(name)}` \| [^|]* \| (.+) \|$", readme, re.M)
        if not row:
            fail(f"[B3] README 索引缺少 {name} 行")
            continue
        entries = [e.strip() for e in row.group(1).split(" · ") if e.strip()]
        seen = []
        for e in entries:
            m = re.match(r"^§(\d+)\s+(.+)$", e)
            if not m:
                fail(f"[B3] {name} 索引条目格式错误: {e[:60]}")
                continue
            n, title = int(m.group(1)), m.group(2)
            seen.append(n)
            if (name, n) not in heads:
                fail(f"[B3] {name} 索引引用了不存在的 §{n}")
            elif not heads[(name, n)].startswith(title):
                fail(f"[B3] {name} §{n} 索引标题与正文标题不符: 「{title}」!⊑「{heads[(name, n)][:60]}…」")
        if seen != sorted(seen) or len(seen) != len(set(seen)):
            fail(f"[B3] {name} 索引条目乱序或重复: {seen}")
        if set(seen) != set(order[name]):
            fail(f"[B3] {name} 索引与正文章节集合不一致: 漏 {sorted(set(order[name]) - set(seen))}, 多 {sorted(set(seen) - set(order[name]))}")
    print("[B3] README 索引：各文件条目与正文标题逐一对应")

    return heads, subs


# ── C. 引用解析 ──────────────────────────────────────────────────────────
def check_refs(heads, subs):
    files = [os.path.join(BASE, "AGENTS.md")]
    for root, dirs, names in os.walk(os.path.join(BASE, "doc")):
        for n in names:
            if n.endswith(".md"):
                files.append(os.path.join(root, n))
    for sub in ("nodes", "sf_utils", "web", "tests"):
        for root, dirs, names in os.walk(os.path.join(BASE, sub)):
            if "__pycache__" in root:
                continue
            for n in names:
                if n.endswith((".py", ".js", ".mjs")) and n != "check_docs.py":
                    files.append(os.path.join(root, n))

    owner = {}          # N -> file（唯一后）
    for (f, n) in heads:
        owner[n] = f
    theme_by_name = {name: name for name in THEME_FILES}

    bad_top, bad_sub, bad_qual = [], [], []
    for path in files:
        rel = os.path.relpath(path, BASE)
        for i, line in enumerate(open(path, encoding="utf-8", errors="replace"), 1):
            for m in re.finditer(r"§(\d+)((?:\.\d+)*)", line):
                n, rest = int(m.group(1)), m.group(2)
                if n not in owner:
                    bad_top.append(f"{rel}:{i} §{n}")
                    continue
                if rest:
                    first = int(rest.split(".")[1])
                    if first not in subs.get((owner[n], n), set()):
                        bad_sub.append(f"{rel}:{i} §{n}{rest}")
            for m in re.finditer(r"([A-Za-z0-9_-]+\.md)\s*§(\d+)", line):
                fn, n = m.group(1), int(m.group(2))
                if fn in theme_by_name and (fn, n) not in heads:
                    bad_qual.append(f"{rel}:{i} {fn} §{n}")
    for label, items in (("C1 顶级", bad_top), ("C1 子节", bad_sub), ("C2 限定", bad_qual)):
        if items:
            fail(f"[{label}] 无法解析的引用 {len(items)} 处：")
            for it in items[:12]:
                errors.append(f"      {it}")
    if not (bad_top or bad_sub or bad_qual):
        print(f"[C] 引用解析：{len(files)} 个文件全部可解析")


# ── D. architecture.md 文件覆盖 ─────────────────────────────────────────
def check_architecture():
    arch = open(os.path.join(BASE, "doc", "architecture.md"), encoding="utf-8").read()
    globs = re.findall(r"([A-Za-z0-9_]+)\*", arch)
    missing = []
    for sub in ("nodes", "sf_utils", "web", "tools"):
        for root, dirs, names in os.walk(os.path.join(BASE, sub)):
            if "__pycache__" in root:
                continue
            for n in names:
                if not n.endswith((".py", ".js")):
                    continue
                if n == "__init__.py":
                    continue
                stem = os.path.splitext(n)[0]
                parts = os.path.relpath(os.path.join(root, n), BASE).split(os.sep)
                if n in arch or stem in arch or any(stem.startswith(g) for g in globs):
                    continue
                # 子包内部实现（如 nodes/model/rfmsr/*.py）允许按目录名（rfmsr/）整体收录
                if len(parts) >= 3 and f"{parts[-2]}/" in arch:
                    continue
                missing.append("/".join(parts))
    if missing:
        fail(f"[D] architecture.md 未覆盖 {len(missing)} 个文件：")
        for m in missing[:15]:
            errors.append(f"      {m}")
    else:
        print("[D] architecture.md：nodes/ sf_utils/ web/ tools/ 文件全覆盖")


def main():
    check_registrations()
    heads, subs = check_experience()
    check_refs(heads, subs)
    check_architecture()
    if errors:
        print("\n失败：")
        for e in errors:
            print("  " + e if not e.startswith("  ") else e)
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
