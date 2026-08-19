#!/usr/bin/env python3
"""sfnodes web/ 模块 import/export 交叉验证（开发辅助，非测试）。

规则：
  A. MODS 内命名导入的符号必须存在于目标模块导出（跨模块契约）
  B. 文件级（全部 web/*.js）：
     B1. 相对导入（含副作用 import "./x.js"）目标文件必须存在
     B2. 含 app.registerExtension( 的文件必须直接 import /scripts/app.js
         （不允许依赖传递——ComfyUI 只加载 web/ 下每个文件，不保证顺序）
     B3. 扩展注册名必须 sfnodes.* 前缀
"""
import os
import re
import sys

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web")

# 参与符号级交叉验证的模块（多模块家族 + 共享库）。单文件节点只走 B 组文件级规则。
MODS = [
    # 共享库
    "sf_common", "sf_dynamic_slots", "sf_markdown", "sf_lora_info",
    # LoRA Stack 家族
    "sf_lora_stack", "sf_lora_stack_core", "sf_lora_stack_api",
    "sf_lora_stack_render", "sf_lora_stack_interaction",
    "sf_lora_stack_dropdown", "sf_lora_stack_info", "sf_lora_stack_settings",
    # Workflows 家族
    "sf_workflows", "sf_workflows_ui", "sf_workflows_lib",
    # LoRA 浏览器家族
    "sf_lora_browser", "sf_lora_browser_ui", "sf_lora_browser_lib",
    # Crop 家族
    "sf_crop", "sf_crop_core", "sf_crop_framework", "sf_crop_panel",
    "sf_crop_preview", "sf_crop_render", "sf_crop_interaction",
    "sf_crop_alignments", "sf_crop_undo_guard",
    # Inpaint 家族
    "sf_inpaint", "sf_inpaint_core", "sf_inpaint_geometry",
    "sf_inpaint_paint", "sf_inpaint_render",
    # 闸门家族（text/image/mask/latent）
    "sf_pause_text", "sf_pause_text_lib", "sf_pause_text_ui",
    "sf_pause_image", "sf_pause_image_lib", "sf_pause_image_ui",
    "sf_pause_mask", "sf_pause_mask_lib", "sf_pause_mask_ui",
    "sf_pause_latent", "sf_pause_latent_lib", "sf_pause_latent_ui",
    # 值下拉家族
    "sf_dropdown", "sf_dropdown_lib", "sf_dropdown_ui", "sf_dropdown_settings",
    # 查找替换家族
    "sf_find_replace", "sf_find_replace_lib", "sf_find_replace_ui",
    # wired 尺寸家族
    "sf_image_resize", "sf_image_resize_lib", "sf_image_resize_ui",
    # 加载图片家族
    "sf_load_image", "sf_load_image_api", "sf_load_image_ui", "sf_load_image_resize",
    # @tag 家族
    "sf_prompt_tags", "sf_prompt_tags_lib", "sf_prompt_tags_cursors",
    "sf_prompt_tags_guard", "sf_prompt_tags_editor", "sf_prompt_tags_store",
    "sf_prompt_tags_pinyin",
    # 区域 LoRA 家族
    "sf_regional_lora", "sf_regional_lora_lib",
    # 外绘家族
    "sf_outpaint", "sf_outpaint_core",
    # 动态 Prompt 列表家族
    "sf_prompt_stack", "sf_prompt_stack_core",
    # 公共弹层三件套
    "sf_popup",
    # Krea2 预设管理（Interrogator + SystemPrompt 共用）
    "sf_krea2_presets",
    # 单文件多依赖节点
    "load_images_path", "sf_prompt_reader", "sf_prompt_list",
]

EXPORT_RE = re.compile(
    r"export\s+(?:async\s+)?function\s+(\w+)|export\s+const\s+(\w+)"
    r"|export\s+class\s+(\w+)"
    r"|export\s*\{\s*([\w,\s]+?)\s*\}(?:\s+from\s+[\"'][^\"']+[\"'])?"
)
NAMED_IMPORT_RE = re.compile(r'import\s*\{([^}]*)\}\s*from\s*"\./([\w_]+)\.js"')
SIDE_EFFECT_IMPORT_RE = re.compile(r'import\s+"\./([\w_]+)\.js"')
REEXPORT_RE = re.compile(r'export\s*\{\s*([^}]*?)\s*\}\s*from\s*"\./([\w_]+)\.js"')
EXT_NAME_RE = re.compile(r'registerExtension\(\{\s*name:\s*"([^"]+)"')

bad = 0


def problem(msg):
    global bad
    print(msg)
    bad += 1


# 读全部 web 文件（key 为文件名，含 .js）
all_files = sorted(f for f in os.listdir(WEB) if f.endswith(".js"))
sources = {f: open(os.path.join(WEB, f), encoding="utf-8").read() for f in all_files}

# ── 规则 A：导出扫描（MODS 内）──
exps = {}
for name in MODS:
    src = sources[name + ".js"]
    found = set()
    for m in EXPORT_RE.finditer(src):
        if m.group(4):
            for item in m.group(4).split(","):
                item = item.strip()
                if item:
                    found.add(item.split(" as ")[0].strip())
        else:
            found.add(m.group(1) or m.group(2) or m.group(3))
    # re-export 透传符号也算本模块公共 API
    for m in REEXPORT_RE.finditer(src):
        for item in m.group(1).split(","):
            item = item.strip()
            if item:
                found.add(item.split(" as ")[0].strip())
    exps[name] = found

for name in MODS:
    if name == "sf_common":
        continue  # sf_common 只 import 绝对路径（/scripts/app.js、/scripts/api.js）
    src = sources[name + ".js"]
    for m in NAMED_IMPORT_RE.finditer(src):
        target = m.group(2)
        if target not in exps:
            problem(f"MISSING MODULE: {name}.js -> {target}.js")
            continue
        for item in m.group(1).split(","):
            item = item.strip()
            if not item:
                continue
            sym = item.split(" as ")[0].strip()
            if sym not in exps[target]:
                problem(f"MISSING EXPORT: {name}.js imports '{sym}' from {target}.js")

# ── 规则 B：文件级（全部 web/*.js）──
for f in all_files:
    src = sources[f]
    # B1: 相对导入目标必须存在
    for m in NAMED_IMPORT_RE.finditer(src):
        target = m.group(2)
        if target + ".js" not in sources:
            problem(f"MISSING TARGET: {f} -> {target}.js")
    for m in SIDE_EFFECT_IMPORT_RE.finditer(src):
        target = m.group(1)
        if target + ".js" not in sources:
            problem(f"MISSING TARGET: {f} -> {target}.js (side-effect import)")
    # B2: 注册扩展必须直接 import app.js
    if "app.registerExtension(" in src and 'from "/scripts/app.js"' not in src:
        problem(f"REGISTER WITHOUT APP: {f} calls app.registerExtension but does not import /scripts/app.js")
    # B3: 扩展注册名必须 sfnodes.* 前缀
    for m in EXT_NAME_RE.finditer(src):
        if not m.group(1).startswith("sfnodes."):
            problem(f"BAD EXT NAME: {f} registers '{m.group(1)}' (want sfnodes.* prefix)")

print("OK" if bad == 0 else f"{bad} PROBLEMS")
sys.exit(1 if bad else 0)
