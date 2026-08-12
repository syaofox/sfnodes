#!/usr/bin/env python3
"""sfnodes web/ 模块 import/export 交叉验证（开发辅助，非测试）。"""
import os
import re
import sys

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web")

MODS = [
    "sf_lora_stack", "sf_lora_stack_core", "sf_lora_stack_api",
    "sf_lora_stack_render", "sf_lora_stack_interaction",
    "sf_lora_stack_dropdown", "sf_lora_stack_info", "sf_lora_stack_settings",
    "sf_common", "sf_markdown", "sf_lora_info",
    "sf_workflows", "sf_workflows_ui", "sf_workflows_lib",
]

EXPORT_RE = re.compile(
    r"export\s+(?:async\s+)?function\s+(\w+)|export\s+const\s+(\w+)"
    r"|export\s*\{\s*([\w,\s]+?)\s*\}"
)
NAMED_IMPORT_RE = re.compile(r'import\s*\{([^}]*)\}\s*from\s*"\./([\w_]+)\.js"')

exps = {}
for name in MODS:
    src = open(os.path.join(WEB, name + ".js"), encoding="utf-8").read()
    found = set()
    for m in EXPORT_RE.finditer(src):
        if m.group(3):
            for item in m.group(3).split(","):
                item = item.strip()
                if item:
                    found.add(item.split(" as ")[0].strip())
        else:
            found.add(m.group(1) or m.group(2))
    exps[name] = found

bad = 0
for name in MODS:
    if name == "sf_common":
        continue
    src = open(os.path.join(WEB, name + ".js"), encoding="utf-8").read()
    for m in NAMED_IMPORT_RE.finditer(src):
        target = m.group(2)
        if target not in exps:
            print(f"MISSING MODULE: {name}.js -> {target}.js")
            bad += 1
            continue
        for item in m.group(1).split(","):
            item = item.strip()
            if not item:
                continue
            sym = item.split(" as ")[0].strip()
            if sym not in exps[target]:
                print(f"MISSING EXPORT: {name}.js imports '{sym}' from {target}.js")
                bad += 1

print("OK" if bad == 0 else f"{bad} PROBLEMS")
sys.exit(1 if bad else 0)
