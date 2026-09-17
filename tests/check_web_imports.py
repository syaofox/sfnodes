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
    "sf_common", "sf_dynamic_slots", "sf_markdown", "sf_lora_shared_info", "sf_lora_info",
    "image_browser",
    # LoRA Stack 家族
    "sf_lora_stack", "sf_lora_stack_core", "sf_lora_stack_api",
    "sf_lora_stack_render", "sf_lora_stack_interaction",
    "sf_lora_stack_dropdown", "sf_lora_stack_info", "sf_lora_stack_settings",
    "sf_lora_preset_filter", "sf_lora_preset_manager",
    # Workflows 家族
    "sf_workflows", "sf_workflows_ui", "sf_workflows_lib",
    # LoRA 浏览器家族
    "sf_lora_browser", "sf_lora_browser_ui", "sf_lora_browser_lib",
    # Crop 家族
    "sf_crop", "sf_crop_core", "sf_crop_framework", "sf_crop_panel",
    "sf_crop_preview", "sf_crop_render", "sf_crop_interaction",
    "sf_crop_alignments", "sf_crop_undo_guard",
    # Crop Expand（出界裁剪/外绘预处理，复用 sf_crop_core 的 CropAPI）
    "sf_crop_expand", "sf_crop_expand_lib",
    # Brush Mask（节点内画笔遮罩，复刻 YCNodes Load Image Brush Mask）
    "sf_brush_mask", "sf_brush_mask_lib",
    # Crop Expand Brush Mask（两节点合体：组合布局 + 源图链路/比例弹窗/画笔工具共享）
    "sf_crop_expand_brush_mask", "sf_crop_expand_brush_mask_lib",
    "sf_crop_source", "sf_crop_expand_ratios", "sf_brush_tools",
    # AI/工具右键菜单共享 UI（brush 两节点：SAM/人物/YOLO/导入/反选/卸载）
    "sf_brush_ai",
    # Inpaint 家族
    "sf_inpaint", "sf_inpaint_core", "sf_inpaint_geometry",
    "sf_inpaint_paint", "sf_inpaint_render",
    # 闸门家族（text/image/mask/latent；image/mask/latent 三闸门共享引擎收敛于 sf_pause_kit）
    "sf_pause_kit",
    "sf_pause_text", "sf_pause_text_lib", "sf_pause_text_ui",
    "sf_pause_image", "sf_pause_mask", "sf_pause_latent",
    "sf_pause_source",
    # 值下拉家族
    "sf_dropdown", "sf_dropdown_lib", "sf_dropdown_ui", "sf_dropdown_settings",
    # 查找替换家族
    "sf_find_replace", "sf_find_replace_lib", "sf_find_replace_ui",
    # wired 尺寸家族
    "sf_image_resize", "sf_image_resize_lib", "sf_image_resize_ui",
    # Image Resize Plus（size_mode 显隐切换）
    "sf_image_resize_plus",
    # 加载图片家族
    "sf_load_image", "sf_load_image_api", "sf_load_image_ui", "sf_load_image_resize",
    # @tag 家族
    "sf_prompt_tags", "sf_prompt_tags_lib", "sf_prompt_tags_cursors",
    "sf_prompt_tags_editor", "sf_prompt_tags_store",
    "sf_prompt_tags_pinyin",
    # 区域 LoRA 家族
    "sf_regional_lora", "sf_regional_lora_lib",
    # 外绘家族
    "sf_outpaint", "sf_outpaint_core",
    # 动态 Prompt 列表家族
    "sf_prompt_stack", "sf_prompt_stack_core",
    # 风格选择器家族（Easy-Use stylesSelector 复刻）
    "sf_styles_selector", "sf_styles_selector_lib",
    # 证件照服装单选器家族（复刻孤海画廊，复用 styles JSON 生态）
    "sf_id_clothing", "sf_id_clothing_lib",
    # 角色三分镜家族（单选画廊：脸部特写/半身像/全身像）
    "sf_character", "sf_character_lib",
    # 公共弹层三件套
    "sf_popup",
    # Canvas Size Preset（model 官方表联动 + 全局自定义分辨率库管理）
    "canvas_size", "sf_canvas_size_lib",
    # Krea2 预设管理（Interrogator + SystemPrompt 共用）
    "sf_krea2_presets",
    # Krea2 反推预设联动（本地版 + API 版 SFImageInterrogatorAPI 共用双 class）
    "krea2_interrogator",
    # 共享 LLM API 设置（翻译 / 图片反推共用 sfnodes.LLM.*）
    "sf_llm_settings",
    # Diffusion Model 信息面板家族（dmodel 域路由束 + 节点扩展）
    "sf_dmodel_api", "sf_load_diffusion_model",
    # 画布对齐（多选宽度对齐：lib 纯逻辑 + 主扩展画布菜单子菜单）
    "sf_canvas_align", "sf_canvas_align_lib",
    # 画布内存清理（SF Memory 子菜单：原生 /free + 自建 RAM 路由）
    "sf_memory_menu",
    # 任意节点颜色（SF Node Color 菜单项 + 取色面板；纯逻辑 + 主模块）
    "sf_node_color", "sf_node_color_lib",
    # 节点运行时间 badge（execution_start/executing 计时；纯逻辑 + 主模块）
    "sf_node_runtime", "sf_node_runtime_lib",
    # 画布聚合菜单（📦 SF Menu 唯一顶层入口，组装各特性 export 动作）
    "sf_canvas_menu",
    # 单文件多依赖节点
    "load_images_path", "sf_prompt_reader", "sf_prompt_list", "sf_mask_fill",
    # 磁盘缓存家族（lazy 跳过上游 + 缓存名下拉/新建；共用 sf_cache_name_lib）
    "sf_cache_name_lib", "sf_mask_cache", "sf_track_cache",
    # PointsEditor 底图刷新（上游解析纯逻辑 + 扩展）
    "sf_points_bg_lib", "sf_points_bg",
    # SeC 数值上限补丁（前端 INT 缺 max 默认 2048 的规避）
    "sf_sec_limits",
    # Convert Anything（combo→输出槽改型，复用 any_pack.setSlotType）
    "sf_convert_anything",
    # SimpleMath / SFNumber（number_type→输出槽改型，复用 any_pack.setSlotType）
    "simple_math",
    # Any Pack / Unpack 动态槽位（导出 setSlotType 供跨模块复用）
    "any_pack",
    # Any Switch（复刻 rgthree；复用 any_pack.setSlotType/slotLinkTypes）
    "sf_any_switch",
    # Conditioning Combine（多路拼接；复用 sf_dynamic_slots.installDynamicSlots）
    "sf_conditioning_combine",
    # Conditioning Concat（多路拼接；复用 sf_dynamic_slots.installDynamicSlots）
    "sf_conditioning_concat",
    # Track Data Subtract/Add（排除/叠加动态槽；复用 sf_dynamic_slots）
    "sf_track_data_slots",
    # Track Data Merge（逐槽 -/+ 模式单节点加减；复用 sf_dynamic_slots/sf_common）
    "sf_track_data_merge", "sf_track_data_merge_lib",
    # Painter Flux Image Edit（参考图动态槽位；复用 sf_dynamic_slots.installDynamicSlots）
    "sf_painter_flux_edit",
    # Wan Window LoRA（逐窗位置 preset 槽；复用 sf_dynamic_slots.installDynamicSlots）
    "sf_wan_window_lora",
    # Universal Slider（复刻孤海万能滑条 Canvas 大滑条；复用 sf_common/sf_popup/any_pack.setSlotType）
    "sf_universal_slider", "sf_universal_slider_lib",
    # Boolean Switch（复刻孤海布尔开关 Canvas 开关；复用 sf_common.el）
    "sf_boolean_switch", "sf_boolean_switch_lib",
    # Ignore Groups（复刻孤海忽略多组编组开关；复用 sf_common/sf_popup）
    "sf_ignore_groups", "sf_ignore_groups_lib",
    # Note（复刻孤海注释文本便签；复用 sf_common/sf_popup）
    "sf_note", "sf_note_lib",
    # SCAIL-2 四节点前端（复刻 ComfyUI-SCAIL2-Easy；动态槽/显隐）
    "sf_scail2",
    # SCAIL-2 预处理内存优化设置（后端 scail2_mem 补丁开关/帧数/f16）
    "sf_scail2_mem_settings",
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
