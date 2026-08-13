# AGENTS.md - sfnodes (ComfyUI Custom Node Pack)

## Project Overview

sfnodes 是一个 ComfyUI 自定义节点包，提供图像处理、人脸操作、遮罩编辑、文本处理、模型管理等增强功能。

ComfyUI 源码根目录即 `../..`（`<custom_nodes>/..` 的父目录，本机为 `/home/syaofox/Projects/ComfyUI/`，含 `comfy/`、`nodes.py` 等，**仅为源码副本**，实际运行实例为 docker 部署），可用于查阅 API 和参考实现。**不要尝试在本机启动 ComfyUI 或安装运行时依赖。**

## Architecture

```
sfnodes/
├── __init__.py          # 节点注册入口：NODE_CLASS_MAPPINGS + NODE_DISPLAY_NAME_MAPPINGS
├── requirements.txt     # Python 依赖（仅声明，不在本机安装）
├── nodes/               # 所有节点实现，按功能分子目录
│   ├── face/            # 人脸：分析、对齐、扭曲、裁剪粘贴、区域、遮挡
│   ├── image/           # 图片：加载、缩放（含工作流内缩放 resize_image.py：wired 尺寸）、拼接、处理、对比、三点色彩匹配（color_match_points.py：SFImageColorMatchByPoints 亮度分位自动提取暗/灰/亮三点 → 逐通道三点分段线性 LUT）、可视化裁剪+贴回（crop.py）、外绘填充+贴回（outpaint.py）、图片闸门（pause_image.py）、latent 闸门（pause_latent.py：SFPauseLatent 分段采样中间暂停）、预览保存路由（preview_routes.py）
│   ├── mask/            # 遮罩：参数、轮廓、模糊、缩放、填充、反转、遮罩闸门（pause_mask.py）
│   ├── model/           # 模型：LoRA加载（多行 LoRA 栈 lora_stack.py：SFLoraStack，含触发词/描述/封面/Civitai 查询；批量对比 lora_plot.py：SFLoraPlot 动态行模型输出列表 + SFLoraPlotImageSaver 文字标注，复用 stack 状态契约与 sf_utils/lora_plot.py、lora_cache.py）、CLIP编码、人像分割
│   ├── text/            # 文本：翻译、拼接、下拉选择、值下拉（dropdown_value.py：name→value 列表 + 四类型输出 + F/I/R 模式）、角色选择、提示词预设（prompt_preset.py）、工作流文本预设（text_preset.py）、@tag 标签库提示词（prompt_tags.py）、内联文本闸门（pause_text.py）、查找替换（find_replace.py）、PNG/视频元数据提示词恢复（prompt_reader.py：SFPromptReader，含 prompt_reader_routes.py 路由 /api/sfnodes/prompt_reader/{extract,list}）
│   ├── utils/           # 工具：数学、显示、内存清理、分辨率、图像编辑
│   ├── inpaint/         # 局部修复：裁剪、拼接、外扩
│   ├── workflow_routes.py # 工作流面板后端路由（/api/sfnodes/workflows/*）
│   └── logic.py         # 逻辑：索引切换、Any 打包/解包、遮罩判空、循环（For/While Loop）
├── sf_utils/            # 共享工具库
│   ├── common.py        # AnyType 通用类型
│   ├── image_convert.py # tensor/pil/numpy/mask 互转
│   ├── mask_utils.py    # 遮罩工具
│   ├── inpaint_helpers.py # 局部修复辅助（裁剪/拼接/缩放，无 ComfyUI 依赖）
│   ├── adv_encode.py    # 高级编码工具
│   ├── string.py        # 字符串工具
│   ├── translation.py   # 翻译封装
│   ├── downloader.py    # 下载工具
│   ├── model_manager.py # 模型管理
│   ├── cutpaste.py      # 剪切/拼接工具
│   ├── blend.py         # 混合工具
│   ├── insightface_utils.py # InsightFace 封装
│   ├── face_detector.py  # 人脸检测
│   ├── lora_notes.py     # LoRA 用户数据统一存储网关（Power 系对话框/loader 节点与 SFLoraStack 共用 lora_triggers.json 真源；旧 .sf.json 侧车惰性迁移，见经验摘要 §LoRA 信息数据统一）
│   ├── lora_presets.py   # LoRA 预设
│   ├── lora_samples.py   # LoRA 样例图处理
│   ├── lora_reader.py    # LoRA 元数据/触发词/内容指纹纯逻辑（SFLoraStack 用，无 ComfyUI 依赖）
│   ├── lora_plot.py      # LoRA 批量对比纯逻辑（文件名净化/元数据双向/字体选择含 CJK/文字覆盖，SFLoraPlot 用，无 ComfyUI 依赖）
│   ├── lora_cache.py     # LoRA 文件缓存 + 内存模式修剪（last/all/none，与 SFLoraStack 同语义，SFLoraPlot 用）
│   ├── lora_routes.py    # SFLoraStack 路由（/api/sfnodes/lora_*、civitai/account 等，见文件内注册清单）
│   ├── workflow_index_helpers.py # 工作流索引纯逻辑（Workflows 面板，无 ComfyUI 依赖）
│   ├── resize_engine.py  # 图片缩放引擎（8 模式 + wired 尺寸 _apply_wired_size，无 ComfyUI 依赖）
│   ├── color_match_points.py # 三点色彩匹配纯逻辑（亮度分位三点提取/逐通道分段线性 LUT/查表，SFImageColorMatchByPoints 用，无 ComfyUI 依赖）
│   ├── dropdown.py      # 值下拉纯逻辑（数字语法双端契约 readable/coerce，无 ComfyUI 依赖）
│   ├── disk_state.py    # 磁盘状态共享实现（safe_join/sanitize_id/decode_image，crop 与 inpaint 共用）
│   ├── skin.py          # 肤色估计纯逻辑（numpy RGB→LAB 肤色过滤取均值/回退，SFFaceWarp 未连接源图时填充近似肤色用，无 ComfyUI 依赖）
│   ├── prompt_reader.py # 提示词恢复纯逻辑（PNG tEXt + MP4 keys/ilst + WebM EBML Tags 解析、graph walker 反推 sampler 文本链，无 ComfyUI 依赖）
│   └── logger.py        # 日志
├── web/                 # 前端 JS Widget（含 sf_common.js 复刻节点公共小工具与全局强调色（getSfAccent/applySfAccentVar/sfAccent，document 根 --sf-acc CSS 变量体系）、sf_dynamic_slots.js 动态槽位公共库、prompt_preset.js 预设互斥联动/选中预设说明动态 tooltip、sf_prompt_tags*.js @tag 标签库六模块、sf_pause_text*.js 文本闸门三模块、sf_pause_image*.js 图片闸门三模块、sf_pause_mask*.js 遮罩闸门三模块、sf_pause_latent*.js latent 闸门三模块、sf_outpaint*.js 外绘预览两模块、sf_image_resize*.js 图片缩放三模块、sf_find_replace*.js 查找替换三模块、sf_dropdown*.js 值下拉四模块、sf_workflows*.js 工作流面板三模块、sf_prompt_reader.js 提示词恢复单模块、sf_load_image*.js 加载图片四模块（SFLoadImageResize）、load_images_path.js 渐进式目录浏览（SFLoadImagesPath 源切换 input/output/images + 面包屑/按需加载 + 直接输入路径）、sf_lora_stack*.js 多行 LoRA 栈模块系列（core/api/render/interaction/dropdown/info/settings + 主扩展）、sf_lora_plot.js 批量对比节点单模块（SFLoraPlot：行 UI 全复用 stack 的 core/api/dropdown/菜单/CSS））
├── data/                # 静态数据（anime_char CSV、face_distance 字体、prompt_presets.json 提示词预设等）
├── tests/               # 前端/后端模拟测试（Node/Python 直接运行，无测试框架）
└── doc/                 # 项目文档（vibecoding.md 开发流程、experience.md 历史经验归档等）
```

## Node Registration Convention

每个节点必须在根 `__init__.py` 的两个字典中注册：

- `NODE_CLASS_MAPPINGS`: 键为 `"SF<ClassName>"`，值为类本身
- `NODE_DISPLAY_NAME_MAPPINGS`: 键同上，值为显示名 `"SF <Display Name>"`

新增节点后，务必在两个字典中同步添加。

## Node Class Convention

所有节点类遵循 ComfyUI 标准 API：

```python
class SFMyNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { ... },
            "optional": { ... },  # 可选
        }

    RETURN_TYPES = ("TYPE",)
    RETURN_NAMES = ("name",)
    FUNCTION = "execute"          # 执行方法名
    CATEGORY = "sfnodes/<group>"  # 统一使用 sfnodes/ 前缀
    DESCRIPTION = "..."           # 必填

    def execute(self, ...):
        return (result,)
```

### CATEGORY 命名

统一格式：`sfnodes/<功能组>`，例如：
- `sfnodes/face` — 人脸相关
- `sfnodes/image` — 图片相关
- `sfnodes/mask` — 遮罩相关
- `sfnodes/model` — 模型相关
- `sfnodes/text` — 文本相关
- `sfnodes/utils` — 工具相关
- `sfnodes/logic` — 逻辑相关
- `sfnodes/inpaint` — 局部修复相关

## Key Dependencies (runtime only, do NOT install)

- `torch`, `torchvision` — 张量运算（由 ComfyUI 运行时提供，不在 `requirements.txt` 中声明）
- `opencv-contrib-python` — 图像处理
- `insightface`, `onnxruntime` — 人脸分析
- `mediapipe` — 人像分割
- `kornia` — 图像变换
- `color_matcher` — 色彩匹配
- `colour-science` — 色彩科学/LUT 处理
- `translators` — 文本翻译
- `scipy`, `aiohttp`, `safetensors`, `tqdm`
- `psutil` — 系统资源监控（内存清理节点使用）
- `sageattention` — 注意力优化
- `diffusers`, `einops`, `timm`, `huggingface_hub` — 图像模型/扩散相关（RFMSR 等）

## ComfyUI API Imports (for reference only)

以下模块在运行时由 ComfyUI 提供，可通过源码 `../..` 查阅实现：

- `comfy.utils` — 通用工具（缩放、文件加载等）
- `comfy.utils.common_upscale` — 图片缩放
- `comfy.utils.ProgressBar` — 进度条
- `comfy.model_management` — 显存/设备管理
- `comfy.comfy_types.node_typing.IO` — 类型注解
- `comfy.sd` — 模型加载（load_lora_for_models 等）
- `nodes.LoadImage`, `nodes.SaveImage`, `nodes.MAX_RESOLUTION` — 内置节点
- `nodes.LoraLoader` — LoRA 加载节点
- `nodes.NODE_CLASS_MAPPINGS` — 全部节点映射（含自定义节点；**运行时才包含全部，函数内 import 最安全**）
- `folder_paths` — 路径管理
- `comfy_extras.nodes_post_processing` — 后处理节点
- `comfy_execution.graph_utils` — `GraphBuilder`（图展开）、`is_link`、`ExecutionBlocker`（官方位置，graph.py 只是 re-export）
- `comfy_execution.graph` — `DynamicPrompt`（DYNPROMPT 隐藏输入对象：`get_node`/`get_display_node_id`/`get_original_prompt`，支持 ephemeral 前缀 id）

## Code Style

- Python 3.10+，无类型注解强制要求
- 使用 `_CATEGORY` 模块级常量定义分类前缀
- 工具函数放在 `sf_utils/` 下对应模块（无状态纯函数）
- 节点实现放在 `nodes/<功能组>/` 下对应文件
- JS Widget 放在 `web/` 目录，文件名与节点功能对应；**新增节点/功能前先查公共模块**：`web/sf_common.js`（sfApiUrl / isVueNodes / applyAdaptiveCanvasOnly / isGraphLoading / installCanvasZoomPassthrough / parseAnnotatedImageValue / buildSourceURL / getUpstreamImageURL / installPasteHandler）、`web/sf_dynamic_slots.js`（动态槽位）、`web/sf_crop_framework.js`（编辑器框架/预览系统）；后端查 `sf_utils/disk_state.py`（safe_join / sanitize_id / decode_image）与 `sf_utils/` 各纯逻辑模块。有公共实现必须复用，**禁止再写内联副本**
- `__init__.py` 文件在子目录中为空，仅根目录 `__init__.py` 负责注册（注意：`nodes/utils/` 目前无 `__init__.py`，依赖 namespace package 机制）

## Development Rules

1. **不要启动 ComfyUI 或运行 `pip install`** — 本机仅作为代码编辑环境。一次性生成工具（如拼音表用的 pinyin-pro）可装在 `/tmp` 使用，产物内联进 web/ 模块，**不得进入 requirements.txt**
2. 可以阅读 `../..` 源码以理解 API 和参考实现
3. 新增节点必须同步更新根 `__init__.py` 的两个注册字典
4. 新增依赖必须同步更新 `requirements.txt`
5. 保持节点类命名一致性：实现类 PascalCase，注册键 `"SF"` 前缀
6. 图像张量格式统一为 `[B, H, W, C]`（ComfyUI 标准）
7. 遮罩张量格式统一为 `[B, H, W]`
8. `sf_utils/` 中的工具函数应当是无状态的纯函数
9. JS Widget 使用 `app.registerExtension` 注册，遵循 ComfyUI LiteGraph API；纯工具模块（无扩展行为，如 `sf_dynamic_slots.js`）仅 export 函数即可，由使用者 import
10. 根 `__init__.py` 必须声明 `WEB_DIRECTORY = "web"` 以加载前端 JS Widget（新增 JS 文件后直接放入 `web/`，无需额外注册）
11. 动态槽位类 JS 优先复用 `web/sf_dynamic_slots.js` 公共库，勿重复实现
12. 部署：用户运行实例为 docker 部署（`/mnt/github/comfyui-docker/custom_nodes/sfnodes/`，与本地仓库内容一致）。后端改动需重启容器；`web/` JS 改动需同步该目录，且浏览器需**硬刷新**（Ctrl+Shift+R）才生效
13. **实际环境调试禁止自行浏览器访问 ComfyUI**（会 404 且可能干扰用户运行中的工作流）：一律用分段 console 诊断脚本（版本检查 → 节点状态 → 事件日志包装 → 数据层 → UI 层）交用户执行并反馈；节点请用户用 UI 添加（新版前端无 `graph.createNode`）
14. **新增节点/功能前先查复用**（见 Code Style）：前端 `web/sf_common.js` 等公共模块、后端 `sf_utils/` 纯逻辑模块。**禁止再次内联副本**——复制后语义分叉是 bug 温床（crop 的 `_safe_join` 双重拼接曾导致粘贴上传输出白图）。去重/重构注意：① 独立语句的包装块（如 `if (app && app.loadGraphData...)`）不在函数体内，按函数名删除会漏，需手动清理；② 文件已有某模块 import 时，脚本补 import 可能跳过导致缺符号（运行时 ReferenceError 被 try/catch 吞掉极难排查）；③ `node --check` 默认 CJS 解析查不出 ESM 结构错误，用 `node --input-type=module --check < file` 验证

## Code Discovery

优先使用 **codebase-memory 知识图谱**（`search_graph`、`trace_path`、`get_code_snippet`）查找函数、类及其调用关系，代替 grep/glob。该系统已索引整个项目，支持语义搜索和调用链追踪。仅在搜索字符串字面量、错误消息、配置文件等非代码内容时回退到 grep/glob。

## Testing

本项目无自动化测试框架。验证方式：
- 静态检查：确认 `NODE_CLASS_MAPPINGS` 和 `NODE_DISPLAY_NAME_MAPPINGS` 键一致
- 导入检查：确认所有节点类在根 `__init__.py` 中正确导入
- 依赖检查：确认 `requirements.txt` 包含所有第三方依赖
- 后端模拟测试（无需 ComfyUI）：mock `torch`/`comfy.utils` 后加载节点模块，用 FakeDynPrompt 断言图结构与返回值（循环节点有先例）
- 前端模拟测试：无 DOM 依赖的公共库复制为 `.mjs` 后用 Node 直接跑（FakeNode + 事件序列，`tests/` 有先例，如 `test_any_pack_js.js`）

## 经验摘要

> 具体机制的完整踩坑过程已归档至 `doc/experience.md`（2026-08 精简时从本文件迁出）；此处仅保留高频复用的结论，细节与案例见归档与代码。

**后端：循环/图展开（`nodes/logic.py`）**
- execute 可返回 `{"result": ..., "expand": {...}}` 展开动态子图；result 中的 link 值 `[id, slot]` 会被特殊解析为链接目标值。
- **ForLoopEnd 必须被下游消费才会被调度执行**（死端节点从不执行）——"循环不跑/只跑一轮"先查其输出有无下游。
- 隐藏输入（如 `initial_value0`）首轮不在 prompt 中 → kwargs 缺键而非 None，需默认值兜底。

**后端：动态 combo 校验（工作流绑定状态节点，`nodes/text/text_preset.py`）**
- 预设等状态数据存隐藏 STRING widget 值（JSON）→ **随 workflow 自动保存/加载/复制**，新工作流添加节点为全新默认值（数据载体模式，无需后端存储）。
- combo 选项由前端动态重建时，值会超出 INPUT_TYPES 静态列表 → 旧版 ComfyUI 执行前校验报 `Value not in list` → 必须 `VALIDATE_INPUTS` 返回 True 接管校验（动态选项节点标配）。
- combo 的 options 不随 workflow 保存（只存 value）；加载时 nodeCreated 早于 widget 值恢复 → 重建选项需挂 `onAfterGraphConfigured`（或 onConfigure）补同步。

**前后端：widget 值传后端必须先声明输入（`nodes/image/crop.py`、`sf_crop*.js`）**
- **前端提交 prompt 前 validatePrompt 会删除不在节点 schema 中的输入**——前端 addWidget/DOM widget 创建的任何"运行时状态"输入，若未在 Python `INPUT_TYPES`（hidden/required/optional）声明，后端 `kwargs` 里**根本没有该键**（表现为"值保存正常但执行时全丢"）。排查先打后端 `sorted(kwargs.keys())`。
- 正确通道：Python `hidden` 声明 `"SFCropJson": ("STRING", {"default": "{}"})` + 前端**同名隐藏 STRING widget** → 值走标准 widget 收集（graphToPrompt 读 widget.value，最基础机制不可破坏）。graphToPrompt/queuePrompt 注入只能作双保险，注入目标也必须是 schema 内输入名。
- **不要写 addDOMWidget 创建的 widget 的 `.value`**：Vue 的 DOMWidget value setter 会回调 `setValue` → 若 setValue 回调链里又写 `.value` → 无限递归（`Maximum call stack size exceeded`）。DOM widget 读取走 getValue 闭包，状态同步走独立通道（隐藏 STRING widget 无 setter 链）。

**前端：动态槽位（`web/sf_dynamic_slots.js`、`web/any_pack.js`）**
- 槽名渲染读 `label ?? localized_name ?? name`；初始槽自带 `localized_name`（动态槽没有）→ **改槽名必须同步 name + localized_name**，否则"只有第一个槽改名不生效"。
- `configure` 直赋 links 不触发 `onConnectionsChange` → 恢复场景需挂 `onAfterGraphConfigured` 补逻辑。
- 输入槽 `.link` / 输出槽 `.links` 判空结构不同（公共库 `isSlotConnected` 两者兼容）。

**前端：@tag 展开注入与 Picks 游标（`web/sf_prompt_tags*.js`，SFPromptTags）**
- 前端 `graphToPrompt` hook 在队列时展开 `@tag`/`*cat`/`#list` 并注入隐藏 PromptState（JSON），后端只做拼接；随机结果改变注入字符串 → 缓存键变化 → 自动重跑（无 nonce 即可）。
- `*wildcard`/`#list` 的 shuffle/order 游标**只在 queue 被真正接受后推进**：`beginPickBuild` 把 build id 挂 prompt 对象，`queuePrompt` patcher 成功后 `commitPicks`——Export/分享/校验失败的 build 不消耗选择；同 build 内重复 `#fruit` 用 per-use 计数器发新牌。
- 标签库存 ComfyUI **未注册设置**（机器私有、跨工作流共享、永不进 workflow）；全屏编辑器用工作副本，关闭时 `isSameAsStored` 判定才写回（防覆盖他标签页），`installGraphUndoGuard` 填 `maskeditor_is_opended` 官方槽屏蔽 Ctrl+Z。
- **内置默认库**（`web/prompt_tags_default.json`，prompt_presets 转换产物随插件分发）：设置缺失时（新环境/被清）首启异步 fetch 并落盘（`fetchDefaultLibrary` promise 缓存，失败会话内不重试；仅当仍为空库才应用，防覆盖用户已建标签）；编辑器页脚 ⋯ 菜单有 **Restore default library**（confirmDanger + 先导出备份 + 游标复位，已默认时不弹框）。
- **token 语法与中文**：token 名 `[\p{L}\p{N}_-]`（u flag，中文可作 tag/分类）；边界保护用 Latin/希腊/西里尔/数字/组合标记集合（email `user@name`、算式 `2*2` 不误判，`画@水彩` 识别）；拼音检索用内联 GB2312 一级字表（pinyin-pro 一次性生成，非运行时依赖），`pinyinMatch` 原名/全拼/首字母三路子串；中文 token 前后不插空格（`tagSep/tagTrail` 仅拉丁语境加空格）。

**前端：prompt 剪枝闸门（`web/sf_pause_text*.js`，SFPauseText）**
- **双钩子拆分（与 SFPromptTags 同款）**：`graphToPrompt` 只注入 {mode, text} 到隐藏 PauseState——它也会被 Export/分享/保存按钮触发，在那里删节点会把导出静默截断；`api.queuePrompt`（`args[1].output`）提交时才 PRUNE，且**continue 必须先于 pause/pass 处理**（continue 会删掉自己的下游分支，可能连带删掉上游的另一个闸门）。
- **prune 语义**：pause 删闸门下游（闸门是 OUTPUT_NODE = 分支终点）；continue 跳上游模型链——删 `text` 链接、**菱形重路由**（gate 之后直接读原文本源的链接改指闸门输出）、只删**会拉活被跳过上游的输出节点**（无关输出分支照跑）、非输出节点留作无害孤儿（不校验不运行）；pass 不剪。`isOutput` 从 `LiteGraph.registered_node_types[cls].nodeData.output_node` 读，注册表缺失回退删一切（安全）。
- 解析不到活节点时默认 **pass（不剪）** 而非破坏性的 pause；一次性提交模式（Continue/Regenerate 按钮）挂 `_sfPauseTextSubmitMode` 后 `app.queuePrompt(0,1)`，finally 清除；`executed` 事件接收 Python `ui` 键回填盒子（setModelText 换文本+基线）。
- **Python 侧禁设 `IS_CHANGED = float("nan")`**：NaN 恒不等自身 → 节点每次 Run 都"变化" → 缓存键折叠所有祖先 → 闸门下游每次全量重跑。模式与文本在隐藏输入里本就在缓存键中，无需额外失效。
- 状态存 `node.properties.pauseTextState`（gate/text/original）随工作流保存（保留编辑是设计）；无 widget 状态的 DOM-only restore 在加载路径安全。

**前端：快照闸门与预览保存（`web/sf_pause_image*.js`，SFPauseImage）**
- **图片无法随隐藏输入携带 → 快照机制**：pause 时后端把首帧存 `folder_paths.get_temp_directory()/sf_pause_<id>.png`，continue 时前端把上游剪出 prompt、后端读回同一文件（`UNIQUE_ID` 跨 run 稳定）。快照文件名前缀必须与源插件隔离（同 node_id 会撞文件互相覆盖）。重启后快照过期 → continue 抛清晰错误，需重新 Pause。
- **PNG 拖回重建（Save Output）**：`PngInfo.add_text("prompt"/"workflow")` 对齐 ComfyUI SaveImage 字节格式；嵌入前必须 `_json_safe`（NaN/Inf → 字符串，否则拖回时 workflow JSON.parse 炸）；尊重 `--disable-metadata`（`comfy.cli_args.args.disable_metadata` 实时读、fails open）。
- **`_safe_prefix` 段清洗（复查抓到的真 bug）**：leading `/` 与 `".."` 段检查必须在**任何清洗之前**——先删点/斜杠会让检查永远不命中（路径穿越失效）。段内 Windows 非法字符（`<>:"|?*` 与控制字符）替换 `_`、折叠、边沿剥离、保留设备名（CON/NUL…）加 `_` 后缀；非拉丁/空格原样通过。
- Save 链路：snapshotDataURL → `POST /api/sfnodes/preview/{save,prepare}`（前端 `api.apiURL()` 构建）→ `showSaveFilePicker` 优先 + `<a download>` 回退；executed 捕获的执行期工作流只存运行时（`node._sfPauseImageExecMeta`，绝不进 node.properties）。

**前端：SFPauseMask（`web/sf_pause_mask*.js`，SFPauseMask）**
- 与 SFPauseImage **同构**（MASK 类型闸门）：快照/剪枝/一次性模式/executed 回填机制全部复用，仅 CLASS/输入键（`mask`）/frame 键（`sf_pause_mask_frame`）/state 键（`pauseMaskState`）不同。
- **剪枝共用一份实现**：三闸门（text/image/mask）的 prune 全走 `sf_pause_text_lib.js::applyGateMode(out, id, entry, mode, isOutput, HIDDEN_INPUT, {inputKey})`——改 prune 语义只改一处。
- 快照为**灰度 PNG（L 模式，0-255 量化）**：遮罩通常二值/低精度，8bit 足够，与 ComfyUI 存遮罩惯例一致；tensor 转换防御非标准 `[1,H,W]`（部分节点输出带单例通道维，压平后 L 模式才接受 2D）。
- 快照前缀 `sf_pause_mask_` 与图片闸门的 `sf_pause_` 隔离命名空间；`executed` 回填遮罩 frame 到灰度预览。

**前后端：SFPauseLatent（`nodes/image/pause_latent.py`、`web/sf_pause_latent*.js`，SFPauseLatent）**
- **LATENT 闸门，专为"分段采样中间暂停"**：KSampler(A) [start=0,end=4] → latent 闸门 → KSampler(B) [start=4,end=8]，image 预览输入接 VAEDecode。Pause 停在第一段结束显示预览，Continue 跳过第一段整条链、从快照 latent 继续第二段（第一段零重跑），Regenerate 重跑第一段，Pass 一次跑完。
- **快照是 latent 张量（safetensors）而非 PNG**：`latent_tensor` 键 + `latent_format_version_0` 标记（对齐官方 SaveLatent 格式，官方 LoadLatent 读 multiplier=1）；**保存 latent dict 中全部张量键**（samples + noise_mask/batch_index）——继续采样需完整 batch 与重绘遮罩，不同于 image/mask 闸门仅首帧；读回时 `latent_tensor` 还原为 `samples` 键。`.latent` + 预览 `.png` 双快照同前缀 `sf_pause_latent_`。
- **预览输入（image）必须在 continue 时连同 latent 链接一并剪掉**：`applyGateMode` 新增 `opts.extraInputKeys`（continue 分支循环删除）——预览源（VAEDecode）在闸门上游，不删其输出仍被闸门消费，会把被跳过的第一段采样器拉活。extraInputKeys 仅 continue 生效（pause/pass 预览链接保留），不传时与 image/text/mask 旧调用行为完全一致（有回归测试锁定）。
- **无 image 预览输入也可用**：latent 快照照存照续，只是无 frame（前端不显示、Save/Copy/Open 不可用）。

**前后端：wired 尺寸输入节点（`nodes/image/resize_image.py`、`web/sf_image_resize*.js`，SFImageResize）**
- **三输入优先级**：`longest_side` > `width`/`height`；单轴 = 按该维等比缩放（scale_factor 路径）；双轴 = 精确盒（fit_inside 保持，其他强制 cover）；0/负 wired 值 = 直通。JS `effectiveWiredState` 逐分支镜像 Python `_apply_wired_size`（`sf_utils/resize_engine.py`），两侧测试同用例同期望值。
- **接线互斥自动断开**（longest_side ↔ width/height）必须三重守卫：onConfigure 窗口 + `app.loadGraphData` 包装 300ms 尾窗（**连接恢复发生在 onConfigure 之后**，无此守卫打开工作流会误断已保存的线）+ 自递归标志。显示模式不写 `state.mode`（双线时渲染强制 Crop to fill，断开后恢复用户原模式）。
- `readWiredInt` 只信任**恰好一个数值 widget** 的上游（多数值/字符串 → null → 显示"由接线输入决定"或上次运行 dims，绝不显示错误数字）；wired 字段锁定 = readOnly + opacity + makeNumericInput 的 readOnly 守卫（步进箭头天然失效）。
- **PIL NEAREST 缩小是 box 平均**（非点采样）：单像素点缩小会被稀释，放大/同尺寸才像素保真——mask 对齐断言用"同尺寸直通 + 放大角点"两场景。

**前后端：文本查找替换双端镜像（`nodes/text/find_replace.py`、`web/sf_find_replace*.js`，SFTextFindReplace）**
- 替换逻辑 Python 权威 + JS 预览镜像（`applyRulesJS` ≡ `_apply_rules`），测试同用例同期望值。**literal 模式**：替换文本反斜杠必须双写（`\1` 是字面量不是 backref），JS 端 `$` 转义；**regex 模式**：backref `\1`（Python）vs `$1`（JS）靠 pyTemplateToJs 翻译、`(?P<n>)`→`(?<n>)`、`/u` flag 匹配 Python Unicode 大小写折叠；`\w` 类在 JS 预览仅 ASCII——预览可能比实际窄，Python 是权威。
- **ReDoS 防护**：嵌套无界量词启发式（`(a+)+` `(a*)*`）双端 1:1 镜像——Python 服务端无超时执行，命中即跳过规则 + 警告；预览每次按键重算，同模式会冻结浏览器。
- **预览样本上限 4000 存 `node.properties.findReplacePreview`（不注入 prompt）**：预览 = 上次运行输入 × 当前规则实时重算；规则状态 `findReplaceState` 经 graphToPrompt 注入隐藏 FindReplaceState（Pattern #9，随 workflow 保存）。

**前端：值下拉与输出点对齐（`web/sf_dropdown*.js`、`nodes/text/dropdown_value.py`，SFValueDropdown）**
- **lean 注入形状作缓存键**：graphToPrompt 注入 `{"version", "type", "value"}`（只有选中行的值 + 类型）而非整个列表——注入字符串即缓存键，改行名/重排/改未选中行/切模式都不重跑；Python `selected_value` 接受 lean 与 full 双形状（full 兜底手写 API 文件）。
- **运行游标存节点内存而非未注册设置**（与 SFPromptTags 不同：列表是每节点的）：`_sfDropdownPending` 持有掷出的牌，`api.queuePrompt` 成功后才 `commitPick` 到 `_sfDropdownCursor`——Export/校验失败的 queue 不推进序列；写 node.properties 会把每次 Run 标 modified（Seed 陷阱）。刷新后从选中条目重新开始（可预测）。
- **双端数字语法契约（THE PARITY RULE）**：`sf_utils/dropdown.py` 与 lib 的 coerce 1:1 镜像——`_NUMBER_RE` 拒 `0x10`/`1_0`/Infinity（两侧原生解析器各自分歧，正则才是契约）、`_JS_WHITESPACE`（JS trim 集合含 BOM，Python strip 不含）、half-away-from-zero 取整（Python round 银行家舍入 vs Math.round 向 +∞）、1e12 钳制（可读性警告含钳制移动）、readable 坏行警告标记；两侧测试同用例同期望值。
- **输出点对齐双渲染器（本项目首个对齐节点）**：Classic 硬编码 `output.pos`（getConnectionPos 原样返回 + 自动堆叠跳过 positioned 输出；**注意 margin**：元素画在 node.pos+margin+widget.y 而 widget.y 不带 margin，点要 +margin）；Vue 无官方方式 → DOM nudge（槽位定一行高 → 块拉出文档流 `marginBottom:-offsetHeight` → `translateY` 点上行，**先定尺寸后测块**）；**350ms 自愈 poll**（Vue 重渲染替换节点元素，MutationObserver 被孤立；无变化早退，稳态成本一次 rect）；serialize 剥离 `output.pos`（Legacy 会写进文件，两渲染器文件不一致 → 打开即 modified）。
- 弹出列表（document.body，position:fixed 不继承画布 transform）：根 font-size 按 canvas scale 缩放（内尺寸全 em 联动）、锚点宽是最小值（内容可增长，先算 maxW 再钳 minW）、left 在宽度已知后钳、下方不足向上翻转；外部点击/Esc/滚轮三关闭（wheel 只豁免列表本身，因为坐标写一次、画布移动即搁浅）。
- **isGraphLoading**：包装 `app.loadGraphData` + 300ms 尾窗（连接恢复在 onConfigure 之后）——切换类型断线（dropIncompatibleLinks）与加载路径剪线防护；`slotAccepts` 兼容 `"FLOAT,INT,BOOLEAN"` 多类型槽（相等测试会剪掉用户刚画的线）。
- 写路径 vs 读路径对非对象行处理不同：`writeState` map 归一（null 行变空行），`readState` filter 丢弃——移植时别混。
- **分类（version 2，随 SFTextDropdown 移除加入，详见 experience.md §15.9）**：`categories`/`category`/行 `category` 随工作流保存，旧 v1 数据自动归 default；index/游标基于 `visibleOptions` 过滤列表——切分类必须 `writeState({category, index:0})` 清游标；lean 注入不变（分类是组织状态不进缓存键）；面板 `commit()` 必须重渲染分类区（Import 会漏）；节点面 cat 按钮文本包 span 才有 ellipsis（flex 容器直接文本只硬截断）+ `flex:0 1 auto` 可收缩防行宽溢出，行尾 padding 16px 给输出点让位（点 X：Classic `size[0]-10` 贴边，Vue 越界最小内移 2px）。

**前端：SF Workflows 工作流面板（`web/sf_workflows*.js`、`nodes/workflow_routes.py`、`sf_utils/workflow_index_helpers.py`）**
- **无节点设计**：面板是"应用"不是节点——节点会被存进工作流文件，分享工作流会把多余节点带给每个打开的人。打开方式：工具栏按钮 + 热键 + canvas 右键菜单。
- **热键撞车**：原版 Pixaroma Workflows 占用 `Alt+W`，并存时 ComfyUI 按键注册全局去重报 `Keybinding on Alt + w already exists` → 本项目用 `Alt+Shift+W`。
- 后端分层：`workflow_index_helpers.py` 纯逻辑（无 ComfyUI 依赖可独立测试，mtime+size 增量解析、24MB 文件上限、封面映射 60 框/文本 2KB cap）；`workflow_routes.py` 五资源路径 7 handler（/index 一次返回全部、/meta GET 自愈+POST 按键合并、/folder、/reveal、/cover POST+GET），**meta 读写用 asyncio.Lock 防两面板读-改-写互擦**。
- sidecar 三件套（user/default/ 下，bind mount 存活）：`sf_workflows_meta.json`（notes/covers/folderColors + folderOrder/folderExpanded）、`sf_workflows_cache.json`（索引缓存，条目形状变化递增 version 丢弃）、`sf_covers/`（手选封面以真实 jpg 文件保存，sidecar 只存文件名）。
- **收藏走 pinia（Vue 新版）**：ComfyUI 启动时不读收藏文件，书签 store 直到有人调 `loadBookmarks()` 才加载 → toggle 收藏前必须先 `await loadBookmarks()`，否则覆盖空列表。
- 设置键 `sfnodes.Workflows.{Rect,View,Sort,Density}`（comfy.settings.json 持久化）；**密度系统 `z(n)=calc(npx*var(--sfwb-k,1))`**：视觉尺寸全走 CSS 变量缩放（s/m/l 三档），窗口像素尺寸刻意不缩放保拖拽数学自洽；滚动容器 `overflow-y` 放**持久 main**（面板重建不重置滚动位置）；加载带票号 guard 防两次加载重叠。

**后端：自定义 API 路由（`sf_utils/lora_notes.py`、`nodes/image/preview_routes.py`、`nodes/workflow_routes.py`）**
- 注册先例：`from server import PromptServer` → `ins.routes` 装饰器注册，try/except 包裹（环境异常降级不注册），模块导入时副作用执行（`__init__.py` import）；前缀统一 `/api/sfnodes/...`。**改动路由后必须重启容器**，否则前端 404 静默降级。

**前端：全屏编辑器与冒烟测试**
- 全屏编辑器（DOM widget 之外的整页 overlay）：类名前缀与既有插件隔离（`sf-ptge-`），图标内联 data URI（无资产服务路由），Esc 用 window capture 分层处理（modal → 菜单 → 字段取消 `_sfCancel` → 关闭），危险操作统一 `confirmDanger`（无撤销）。
- 模块化：纯函数 lib（无 app/DOM，测试 copy .mjs 直跑）/ store / cursors / guard / editor / 主扩展，跨文件 import 契约即模块边界。
- 冒烟测试（mock DOM）：惰性元素（任何 querySelector 返回新元素）、`/scripts/app.js` → `globalThis.app`、相对 import 改 `.mjs` 同 tmp 目录；可抓纯语法检查漏掉的运行时错误（如缺 `getComputedStyle` mock）。

 **前端：Vue 新版（comfyui_frontend_package 1.x）**
- 先确认前端版本再选方案（容器内 `pip show comfyui-frontend-package`，Version 1.x = Vue）。
- 槽位数组为 shallowReactive：直接改元素属性不触发渲染，**替换数组元素**才触发。
- 动态 tooltip：写 `widget.tooltip`（nodeDef 兜底存在，清不掉）；canvas 事件/坐标方案在 Vue 下失效。

**前后端：提示词恢复（`sf_utils/prompt_reader.py`、`nodes/text/prompt_reader.py` + `prompt_reader_routes.py`、`web/sf_prompt_reader.js`，SFPromptReader）**
- **三种元数据容器，全纯标准库解析**：PNG tEXt/iTXt（PIL）；MP4/MOV/M4V 的 moov→udta→meta keys+ilst——**ffmpeg 系（VHS `-movflags use_metadata_tags`）ilst item 的 4 字节是 1-based INDEX 而非 iTunes 4cc**，按 `1<=idx<=len(keys)` 判定；WebM/MKV 的 EBML Tags/SimpleTag——**键名按 Matroska 规范大写**，读取归一小写。流式扫描只读入 moov/只进 Tags 容器、seek 跳过 mdat/Cluster（多 GB 视频廉价）。
- **graph walker 反推 sampler 正向文本链**（visited + 深度 24）：`_TEXT_KEYS`/`text_X` 正则/`_COND_LINK_KEYS` 启发式 + 特判分支——Pixaroma 生态 8 类（Switch/Stack/Multi/Pack/Dropdown/FromList/Prompt/SwitchSource/rgthree Any Switch，读他人 Pixaroma 图仍可恢复）与 sf 自家（SFPromptTags/SFValueDropdown 与 Pixaroma 同构共享分支；SFTextPreset/SFAnythingIndexSwitch/SFPauseText continue/SFPromptList/SFPromptPreset）。PromptReader 自追链最多 5 层（embedded workflow 只存 inputs.image）。
- **目录切换 IN/OUT（SFPromptReader + SFLoadImageResize 同款）**：output 项拼 `" [output]"` 注解全链贯通（`get_annotated_filepath` 原生解析、`/view` 缩略图按注解选 type、分组/显示剥离）；**output 模式下上传/拖拽/粘贴自动切回 input**（文件落 input/）。目录状态字段必须避开 `applyResult` 写入的提取来源 `source`——**撞名会被覆盖**，用 `folder`。
- **目录选择三源渐进式（SFLoadImagesPath）**：源切换（input/output/images）+ 面包屑/按需加载（`/api/sfnodes/images_path/subdirs?folder=`）+ popup 下拉（SFLoadImageResize 风格）。**模式状态必须显式存 properties**（值推导不可靠：路径模式下值可能仍是目录格式）；**渲染只读不加 isGraphLoading 门控**（尾窗内恢复渲染会丢）；同值 fetch 缓存防重复请求；空目录返回占位不抛错。完整设计见 `doc/experience.md` §18。
- **DOM widget 高度必须 ≥ 内容实际高度**：element 高度内容自适应、节点边框按 widget 声称高度（getMinHeight/getMaxHeight）绘制——声称 < 内容时底部行（如刷新按钮）溢出边框，节点未拖小时被边框遮住、初始/拖小即暴露。**别硬编码高度**：动态测量可见子行 offsetHeight + padding/gap，last-good 缓存防首帧/组折叠隐藏塌缩；Nodes 2.0 另配 `computeLayoutSize({minHeight, minWidth})`；宽度用 MIN_W（初始只抬升 + onResize 钳制）+ CSS（按钮 `min-width:0`+ellipsis、root `overflow:hidden`）。见 `doc/experience.md` §18.6。
- **上传路径 MIME 过滤必须与 accept 同步放宽**：`accept="image/*,video/*"` 但 drop handler 仍 `startsWith("image/")` → mp4 拖入静默无反应；type 为空（.mkv 未知扩展）放行交后端。
- IS_CHANGED 用 (mtime, size) 而非全文件哈希；VALIDATE_INPUTS 恒 True（缺文件/无元数据走输出文本不阻塞图）；`node.imgs` 抑制（defineProperty 前置探测 configurable）；extract 请求 reqId 单调防乱序。

**前后端：SFLoraStack 多行 LoRA 栈（`sf_utils/lora_reader.py` + `lora_routes.py`、`nodes/model/lora_stack.py`、`web/sf_lora_stack*.js` 模块系列）**
- **Civitai API 字段位置必须实测**：model-versions 响应里 `model` 对象只有 name/nsfw/poi/type——**说明文字在 version 顶层 `description`**（HTML，需剥标签/实体解码/空白折叠）；thumbnail 取 `images[]` 第一张非成人图。
- **用户数据以路径名为键 → 文件移动/改名失配**（自定义词/描述存 `user/sfnodes/lora_triggers.json`，预览图按键 hash 命名；侧车 `.civitai.info` 随文件走不受影响）。两级孤儿匹配：**内容指纹优先**（size + 采样哈希，改名不改内容 → 指纹不变）、**基名兜底**（仅覆盖文件夹改名；同名多目录歧义放弃）。匹配后前端提示条 + 用户确认迁移（不自动执行防误配）；迁移端点接收前端回传的 `old_key`（防御自迁移/不存在键）。封面可静默恢复：本地无预览且侧车有缩略图 → 自动重下载到新 hash 名。
- **面板风确认框必须豁免宿主面板的 document 捕获监听**（onKey/onPaste）：确认框挂在 body、不在面板 DOM 内，不豁免则其事件会穿透到面板监听（Esc 连关面板等）。信息面板本身**只经 ✕ 关闭**——画布点击不关闭（用户边看信息边操作工作流）；Esc 是主动关闭意图保留。
- **全局强调色统一走 `--sf-acc` CSS 变量**（`sf_common.js` getSfAccent/applySfAccentVar/sfAccent，注册在 SFLoraStack 扩展 init）：CSS 部分 `var(--sf-acc, #f66744)` 响应式自动生效；canvas 每帧 `sfAccent()`（inline 变量读取轻量）；**无节点级自定义**（旧 state 的 accent 字段被忽略），面板/下拉/菜单的局部 `--acc`/`--sf-acc` 只是把全局色带到局部作用域。**三个时序坑**：① ComfyUI 设置 onChange 在 store 更新前触发、参数是 (newValue, oldValue)——回调里读 getSettingValue 拿到旧值（"设了 red 显示 teal"），必须用传入参数；② 由此连带：**onChange 里的节点重绘也要 setTimeout(0) 推迟**——同步执行时 accentOf 读 store 仍是旧值（"SFLoraStack 设置后不立即生效"）；③ 初始 applySfAccentVar 必须在 addSetting 之后（未注册读不到用户值），且**设置值从服务器异步加载、晚于扩展 init**——需轮询重试几次（幂等），否则 --sf-acc 被钉死在默认色、CSS 变量类节点（Load Image Resize 等）硬刷新后不跟随（accentOf 直读 store 的节点不受影响，症状不一致易误判）。
- **保存成功后 `_infoSeq++` 作废在途旧响应**：面板打开时 loadInfo 在飞，用户保存描述后迟到响应落地会覆盖回旧值（"保存了仍显示来自 Civitai"）；设置面板同理用 `_accDirty` 挡 GET 迟到应答覆盖刚保存的 host。
- 存储形状升级必须兼容旧数据（`{key:[words]}` → `{key:{words,description,fp?}}` 读时归一）；`promptState` 只注入执行字段（cosmetic 剥掉避免改缓存签名），`cacheMode` 例外（Python 需要它决定内存策略）。完整踩坑见 `doc/experience.md` §19。
- **行名显示全局设置共享**：Power 系设置 `sfnodes.PowerLoraLoader.DisplayName`（full/filename/basename/folder）不只管 Power 行——Stack/Plot 行名同一真源 `sf_common.js::loraRowLabel`（模式 ≠ full 设置优先，basename 剥任意扩展名；full 默认回退每节点 hideExt 白名单，向后兼容）；onChange 经 `sfnodes.lora-display-mode-changed` 事件桥 → Stack/Plot 各自 `setTimeout(repaintAll/renderAllPlots, 0)`（DOM 行重绘；Power 不 import 节点模块防耦合）。细节见 `doc/experience.md` §19.7。

**前后端：SFLoraStack / SFPowerLoraLoader 正交堆叠 ortho_gs（2026-08，`sf_utils/lora_ortho.py` + `lora_ortho_load.py`）**
- **数学**：ΔW = Σ s·(α/r)·(A_i·B_i)，多个 LoRA 的 down 矩阵行空间重叠 = 干扰源（相似 LoRA 叠糊）。ortho_gs 把每个 down 的行投影到前序 down 行空间的正交补（`d' = d - (d@Qᵀ)@Q`，Q 用 SVD 右奇异向量扩基 + QR 去线性相关，float32 计算）——**第一个 LoRA 不动、后续让位**，up/alpha/strength 全不动；行空间被完全覆盖时投影归零（幅度损失是 tradeoff 非 bug）。
- **必须走独立加载路径**：链式 `load_lora_for_models` 的 patch 已展开进 patcher，拿不回 up/down——ortho 需自己 `model_lora_keys_unet`(+clip) 建 key map + `convert_lora`（官方路径有，DuoNodes 漏掉）+ `load_lora` + clone + add_patches + `set_attachments("lora_metadata")`，**按模型 key 分组**（同 key 多 LoRA 才 GS，单条直通），非 LoRA patch（conv/diff/set）该 key fallback 顺序；key map 构建失败整体 fallback 顺序，绝不报错。加载+应用路径收敛在 **`lora_ortho_load.ortho_apply(model, clip, entries, load_sd)`**——Stack 传 `self._get_lora`（复用缓存）与 Power 传 `_load_sd_direct`（直接读盘），**禁止两节点各写一份**（规则 14）；纯数学/格式探测在 `lora_ortho.py`（仅 torch，可单测）。
- **patch 结构**：当前 ComfyUI 是 `LoRAAdapter.weights = (up, down, alpha, mid, dora_scale, reshape)`（**up 是 [0]、down 是 [1]**）；`replace_down` 对 LoRAAdapter 浅拷贝换 weights[1]，字符串标签/tensor-first/float 前缀多格式回退；**replace_down 对 `("diff", (w,))` 之类 1 元素内部元组必须原样返回**（直接 `list(patch[1])` 会 IndexError）。
- **契约**：Stack 的 `mergeMethod` 与 cacheMode 同模式（前端 `DEFAULT_PREFS`/`normalize`/`promptState` 与 Python `parse_state` 双端 1:1，默认 `"sequential"`）；Power 是**节点面 combo**（`merge_method` ∈ `["linear","ortho_gs"]` 默认 `"linear"`，标准 widget 前端零改动）；**ortho 模式 run 内全栈 sd 驻留**（分组需要，与 "last" 逐行释放不同，峰值=栈大小），run 后仍按 cacheMode/无缓存统一修剪。**Power 的应用计划第一项必须存 `get_lora_by_filename` 的规范化结果**——官方 `LoraLoader.load_lora` 内部 `get_full_path_or_raise` 只做精确解析，短名/无扩展名 widget 值直接传会失败（重构搬移时曾回归，测试补短名用例）。
- **ok_paths 是 set——绝不能直接迭代它来组装 resolved/触发词/修剪游标**（顺序随机 → 触发词顺序/`last_this_run` 偶发不稳定，表现为测试偶发 FAIL）；必须按 plan 栈顺序扫描 `if zero or path in ok_paths`。
- **测试**：本机无 torch——GS 数学用 numpy 参考实现逐行对应验证（行两两正交/投影残差在基行空间/覆盖归零）；节点链路 monkeypatch GS + fake `load_lora` **必须按 key_map 值过滤**（unet 与 clip patch 键空间不同，不过滤会串侧）；Power 测试需 mock `nodes.LoraLoader`（顺序路径）与 `folder_paths.get_user_directory`（lora_presets 模块级调用）。

**前后端：LoRA 信息数据统一（2026-08，`sf_utils/lora_notes.py` 网关化）**
- **单一真源**：Power 系（SFPowerLoraLoader 对话框 / SFLoraLoader / ModelOnly 的 execute 输出）与 SFLoraStack 面板的用户自定义词/描述统一存 `user/sfnodes/lora_triggers.json`（路径只由 `lora_routes._custom_triggers_file()` 定义，网关 import 它而非复制）。lora_notes 只是形状转换网关：`trigger_words` 字符串 ↔ `words` 数组（`split_trigger_text` 按英文/中文逗号+换行拆，读写同源）。
- **读优先级三源合并**：统一存储 > `.civitai.info` 侧车（`read_sidecar_info`，去扩展名 `<base>.civitai.info`）> 文件内嵌元数据。Power 系对话框因此也能看到 Stack 面板查过 Civitai 的词。
- **旧 `.sf.json` 侧车彻底废弃**：约定是**保留扩展名**（`<路径>.safetensors.sf.json`，与 `.civitai.info` 的去扩展名约定不同！）；任一读取入口（lora_notes / lora_info）首次读到该 LoRA 时经 `migrate_legacy_sidecar` 惰性迁移并入新存储后删除（幂等：store 已有数据跳过）。`?type=` 类型泛化移除（key 空间无类型维度，混入 checkpoints 等会撞 key；消费节点本就用 loras）。
- **改名/移动后孤儿兜底（文件身份找回数据）**：store key 是路径名，文件改名/移动后 key 失配。两个读取入口（`lora_notes.get_merged_metadata` / `lora_routes.api_lora_info`）都做孤儿检测——**文件存在**时指纹优先（`find_orphan_by_fingerprint`，内容级证据）+ 基名兜底（`find_orphan_key`）；**文件不存在**（旧路径行）时仅基名兜底（无文件可算指纹），返回数据 + `orphan_key`/`_file_missing` 标记，前端提示"文件路径已变更，请重新选择"（Stack 面板提示条无迁移按钮——迁移端点需文件存在；Power 对话框顶部提示行）。同名多目录歧义降级 not found。
- **跨节点缓存失效用事件桥**：任一端保存成功 → `document.dispatchEvent("sfnodes.lora-data-changed", {detail:{name}})` → 另一端清自己的缓存（`loraMetadataCache.delete` / `invalidateInfo`）；对话框打开时 `getLoraMetadata(name, true)` force 重取双保险。
- **封面跨节点可见（只读）**：Power 系对话框 header 也显示封面（`/api/sfnodes/lora_thumb` 同路由：用户自定义预览 > 模型旁 .preview 图），URL 带 `&t=Date.now()` bust 越过一小时缓存；无图 404 → onerror 隐藏。封面编辑仍只在 SFLoraStack 面板（对话框无编辑入口）。
- **`_has_custom` 陷阱**：`desc` 变量会被 sidecar/embedded 兜底覆盖，自定义标志必须用独立变量（`entry_desc`）算，否则 embedded 有描述时 `_has_custom` 恒 True（i 图标误判蓝色）。**高亮语义（2026-08 修订）**：`_has_custom` = 统一存储有词/描述 **或** `.civitai.info` 侧车有词/描述（用户主动查过 Civitai）——侧车-only 的 LoRA（如只查过 Civitai 没写过自定义）也应高亮；刻意不含 embedded（文件自带词/描述几乎人人都有，无区分度）。

**前后端：Civitai 页面主体描述补充（2026-08，`lora_reader._html_to_markdown` / `extract_page_description` / `merge_descriptions` + `lora_routes._download_page`）**
- **API 描述常为空、页面有完整描述**：实测 model-versions 的 version 级 `description` 常是空串，而模型页 Description 卡显示**模型级**完整描述（4110 字符实测）。by-hash 找到模型后总是抓页面补充，拼接**API 在前、页面在后**（`"\n\n"` 分隔，不截断）。
- **页面是 Next.js SSR，别碰 DOM**：数据在 `<script id="__NEXT_DATA__">` JSON 里，描述在 `props.pageProps.trpcState.json.queries[]` 中 `queryKey[0]==["model","getById"]` 的 `state.data.description`。**mantine 随机 id 无关**——按 queryKey 结构定位（`[["model","getById"],...]`），绝不用 CSS 选择器。无 slug URL `/models/{id}?modelVersionId={vid}` 302 后数据完整。
- **抓页面必须模拟浏览器，不只是 UA——TLS 指纹（JA3）也会被 Cloudflare 拦截**：`ComfyUI-sfnodes` UA 直接 403；**连带 Chrome UA 的 aiohttp 请求也实测 403**（Python 默认 TLS 握手指纹被识别），curl / curl_cffi 的指纹才放行。教训：**用 curl 验证"页面可抓"不代表 aiohttp 能抓**——必须以实际代码路径验证。`_download_page` 因此主路径走 **curl_cffi**（`impersonate="chrome"`，自带 libcurl 轮子，executor 线程运行不阻塞事件循环），aiohttp 兜底，都失败返回 None——**降级语义**：页面抓取失败 = 维持仅有 API 描述，绝不影响查询成功路径。`_PAGE_MAX_BYTES=2MB`、15s 超时。
- **拼接结果写入侧车 `data["description"]`**（覆盖 API 空值）：读取端（lora_notes/Power 系走 parse_civitai_modelversion）零改动自然受益；删除侧车仍可清掉。
- **描述统一走 `_html_to_markdown`**（markdownify 转换，缺库/异常回退 `_clean_description` 纯文本，测试双环境全绿）：API/页面/文件内嵌/侧车描述同一入口。**幂等保护**：无 `<` 的输入（纯文本/已 markdown 化的侧车描述）只走轻清洗原样放行——markdownify 对非 HTML 输入不幂等（`*` 会转义成 `\*`），而侧车读取路径会二次处理，不保护则"首次查询正常、下次打开面板变转义文本"。**`_MAX_DESCRIPTION_LEN` 已删除**——不截断（来源有流量守卫：API 4MB/页面 2MB/文件本地；前端面板滚动展示）。

**实际环境调试**
- 禁止自行浏览器访问 ComfyUI；用分段 console 诊断脚本（版本检查 → 节点状态 → 事件日志包装 → 数据层 → UI 层）交用户执行并反馈（见 Development Rules 13）。

**静态检查脚本（AST）**
- `ast.unparse` 输出单引号、`ast.literal_eval` 遇变量引用抛错——检查脚本出错时先怀疑脚本，再怀疑被检查代码。

**复刻节点去重（sf_common.js / disk_state.py，2026-08）**
- 公共小工具单一实现收敛于 `web/sf_common.js`（9 函数，含 loadGraphData 全局单例守卫——**勿再各自包装**）与 `sf_utils/disk_state.py`（safe_join 解析根参数化）。新增节点先查复用（Development Rules 14），完整踩坑见 `doc/experience.md` §17。
- 磁盘源（粘贴/编辑器 Load Image）执行**必须输出源帧 ui_payload**（sf_crop_source/sf_inpaint_source），否则前端 executed 事件收不到、节点预览停留旧图；前端 jsonSync 检测 src_path 变化立即同步缓存（inpaint 无轮询需主动刷新）。
- 编辑器工具栏 **Reset ≠ Clear**：Reset 委托 `_resetCrop()` 保留源图，勿清 img。
