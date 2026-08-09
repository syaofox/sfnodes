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
│   ├── image/           # 图片：加载、缩放（含工作流内缩放 resize_image.py：wired 尺寸）、拼接、处理、对比、可视化裁剪+贴回（crop.py）、外绘填充+贴回（outpaint.py）、图片闸门（pause_image.py）、预览保存路由（preview_routes.py）
│   ├── mask/            # 遮罩：参数、轮廓、模糊、缩放、填充、反转、遮罩闸门（pause_mask.py）
│   ├── model/           # 模型：LoRA加载、CLIP编码、人像分割
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
│   ├── lora_notes.py     # LoRA 笔记/说明
│   ├── lora_presets.py   # LoRA 预设
│   ├── lora_samples.py   # LoRA 样例图处理
│   ├── workflow_index_helpers.py # 工作流索引纯逻辑（Workflows 面板，无 ComfyUI 依赖）
│   ├── resize_engine.py  # 图片缩放引擎（8 模式 + wired 尺寸 _apply_wired_size，无 ComfyUI 依赖）
│   ├── dropdown.py      # 值下拉纯逻辑（数字语法双端契约 readable/coerce，无 ComfyUI 依赖）
│   ├── prompt_reader.py # 提示词恢复纯逻辑（PNG tEXt + MP4 keys/ilst + WebM EBML Tags 解析、graph walker 反推 sampler 文本链，无 ComfyUI 依赖）
│   └── logger.py        # 日志
├── web/                 # 前端 JS Widget（含 sf_dynamic_slots.js 动态槽位公共库、prompt_preset.js 预设互斥联动/选中预设说明动态 tooltip、sf_prompt_tags*.js @tag 标签库六模块、sf_pause_text*.js 文本闸门三模块、sf_pause_image*.js 图片闸门三模块、sf_pause_mask*.js 遮罩闸门三模块、sf_outpaint*.js 外绘预览两模块、sf_image_resize*.js 图片缩放三模块、sf_find_replace*.js 查找替换三模块、sf_dropdown*.js 值下拉四模块、sf_workflows*.js 工作流面板三模块、sf_prompt_reader.js 提示词恢复单模块、sf_load_image*.js 加载图片四模块（SFLoadImageResize））
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
- JS Widget 放在 `web/` 目录，文件名与节点功能对应；动态槽位类功能复用 `web/sf_dynamic_slots.js` 公共库（`installDynamicSlots(node, config)` 配置化，案例见 `web/` 各节点文件与 `tests/` 模拟测试）
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
- **上传路径 MIME 过滤必须与 accept 同步放宽**：`accept="image/*,video/*"` 但 drop handler 仍 `startsWith("image/")` → mp4 拖入静默无反应；type 为空（.mkv 未知扩展）放行交后端。
- IS_CHANGED 用 (mtime, size) 而非全文件哈希；VALIDATE_INPUTS 恒 True（缺文件/无元数据走输出文本不阻塞图）；`node.imgs` 抑制（defineProperty 前置探测 configurable）；extract 请求 reqId 单调防乱序。

**实际环境调试**
- 禁止自行浏览器访问 ComfyUI；用分段 console 诊断脚本（版本检查 → 节点状态 → 事件日志包装 → 数据层 → UI 层）交用户执行并反馈（见 Development Rules 13）。

**静态检查脚本（AST）**
- `ast.unparse` 输出单引号、`ast.literal_eval` 遇变量引用抛错——检查脚本出错时先怀疑脚本，再怀疑被检查代码。
