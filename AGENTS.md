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
│   ├── image/           # 图片：加载、缩放、拼接、处理、对比
│   ├── mask/            # 遮罩：参数、轮廓、模糊、缩放、填充、反转
│   ├── model/           # 模型：LoRA加载、CLIP编码、人像分割
│   ├── text/            # 文本：翻译、拼接、下拉选择、角色选择、提示词预设（prompt_preset.py）、工作流文本预设（text_preset.py）
│   ├── utils/           # 工具：数学、显示、内存清理、分辨率、图像编辑
│   ├── inpaint/         # 局部修复：裁剪、拼接、外扩
│   └── logic.py         # 逻辑：索引切换、Any 打包/解包、遮罩判空、循环（For/While Loop）
├── sf_utils/            # 共享工具库
│   ├── common.py        # AnyType 通用类型
│   ├── image_convert.py # tensor/pil/numpy/mask 互转
│   ├── mask_utils.py    # 遮罩工具
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
│   └── logger.py        # 日志
├── web/                 # 前端 JS Widget（含 sf_dynamic_slots.js 动态槽位公共库、prompt_preset.js 预设互斥联动/选中预设说明动态 tooltip）
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

1. **不要启动 ComfyUI 或运行 `pip install`** — 本机仅作为代码编辑环境
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

**前端：动态槽位（`web/sf_dynamic_slots.js`、`web/any_pack.js`）**
- 槽名渲染读 `label ?? localized_name ?? name`；初始槽自带 `localized_name`（动态槽没有）→ **改槽名必须同步 name + localized_name**，否则"只有第一个槽改名不生效"。
- `configure` 直赋 links 不触发 `onConnectionsChange` → 恢复场景需挂 `onAfterGraphConfigured` 补逻辑。
- 输入槽 `.link` / 输出槽 `.links` 判空结构不同（公共库 `isSlotConnected` 两者兼容）。

**前端：Vue 新版（comfyui_frontend_package 1.x）**
- 先确认前端版本再选方案（容器内 `pip show comfyui-frontend-package`，Version 1.x = Vue）。
- 槽位数组为 shallowReactive：直接改元素属性不触发渲染，**替换数组元素**才触发。
- 动态 tooltip：写 `widget.tooltip`（nodeDef 兜底存在，清不掉）；canvas 事件/坐标方案在 Vue 下失效。

**实际环境调试**
- 禁止自行浏览器访问 ComfyUI；用分段 console 诊断脚本（版本检查 → 节点状态 → 事件日志包装 → 数据层 → UI 层）交用户执行并反馈（见 Development Rules 13）。

**静态检查脚本（AST）**
- `ast.unparse` 输出单引号、`ast.literal_eval` 遇变量引用抛错——检查脚本出错时先怀疑脚本，再怀疑被检查代码。
