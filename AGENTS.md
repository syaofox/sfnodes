# AGENTS.md - sfnodes (ComfyUI Custom Node Pack)

## Project Overview

sfnodes 是一个 ComfyUI 自定义节点包，提供图像处理、人脸操作、遮罩编辑、文本处理、模型管理等增强功能。

ComfyUI 源码位于 `../../ComfyUI/`（同级目录），可用于查阅 API 和参考实现。**不要尝试在本机启动 ComfyUI 或安装运行时依赖。**

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
│   ├── text/            # 文本：翻译、拼接、下拉选择、角色选择
│   ├── utils/           # 工具：数学、显示、内存清理、分辨率、图像编辑
│   ├── inpaint/         # 局部修复：裁剪、拼接、外扩
│   └── logic.py         # 逻辑：If-Else、索引切换
├── sf_utils/            # 共享工具库
│   ├── common.py        # AnyType 通用类型
│   ├── image_convert.py # tensor/pil/numpy/mask 互转
│   ├── mask_utils.py    # 遮罩工具
│   ├── adv_encode.py    # 高级编码工具
│   ├── string.py        # 字符串工具
│   ├── translation.py   # 翻译封装
│   ├── downloader.py    # 下载工具
│   ├── model_manager.py # 模型管理
│   ├── insightface_utils.py # InsightFace 封装
│   └── logger.py        # 日志
├── web/                 # 前端 JS Widget（ComfyUI LiteGraph 扩展）
└── data/                # 静态数据（anime_char CSV、face_distance 字体等）
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
    DESCRIPTION = "..."           # 可选

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

## Key Dependencies (runtime only, do NOT install)

- `torch`, `torchvision` — 张量运算（由 ComfyUI 运行时提供，不在 `requirements.txt` 中声明）
- `opencv-contrib-python` — 图像处理
- `insightface`, `onnxruntime` — 人脸分析
- `mediapipe` — 人像分割
- `kornia` — 图像变换
- `color_matcher` — 色彩匹配
- `translators` — 文本翻译
- `scipy`, `aiohttp`, `safetensors`, `tqdm`

## ComfyUI API Imports (for reference only)

以下模块在运行时由 ComfyUI 提供，可通过源码 `../../ComfyUI/` 查阅实现：

- `comfy.utils` — 通用工具（缩放、文件加载等）
- `comfy.utils.common_upscale` — 图片缩放
- `comfy.utils.ProgressBar` — 进度条
- `comfy.model_management` — 显存/设备管理
- `comfy.comfy_types.node_typing.IO` — 类型注解
- `comfy.sd` — 模型加载（load_lora_for_models 等）
- `nodes.LoadImage`, `nodes.SaveImage`, `nodes.MAX_RESOLUTION` — 内置节点
- `nodes.LoraLoader` — LoRA 加载节点
- `folder_paths` — 路径管理
- `comfy_extras.nodes_post_processing` — 后处理节点
- `comfy_execution.graph.ExecutionBlocker` — 执行阻断

## Code Style

- Python 3.10+，无类型注解强制要求
- 使用 `_CATEGORY` 模块级常量定义分类前缀
- 工具函数放在 `sf_utils/` 下对应模块
- 节点实现放在 `nodes/<功能组>/` 下对应文件
- JS Widget 放在 `web/` 目录，文件名与节点功能对应
- `__init__.py` 文件在子目录中为空，仅根目录 `__init__.py` 负责注册

## Development Rules

1. **不要启动 ComfyUI 或运行 `pip install`** — 本机仅作为代码编辑环境
2. 可以阅读 `../../ComfyUI/` 源码以理解 API 和参考实现
3. 新增节点必须同步更新根 `__init__.py` 的两个注册字典
4. 新增依赖必须同步更新 `requirements.txt`
5. 保持节点类命名一致性：实现类 PascalCase，注册键 `"SF"` 前缀
6. 图像张量格式统一为 `[B, H, W, C]`（ComfyUI 标准）
7. 遮罩张量格式统一为 `[B, H, W]`
8. `sf_utils/` 中的工具函数应当是无状态的纯函数
9. JS Widget 使用 `app.registerExtension` 注册，遵循 ComfyUI LiteGraph API

## Testing

本项目无自动化测试框架。验证方式：
- 静态检查：确认 `NODE_CLASS_MAPPINGS` 和 `NODE_DISPLAY_NAME_MAPPINGS` 键一致
- 导入检查：确认所有节点类在根 `__init__.py` 中正确导入
- 依赖检查：确认 `requirements.txt` 包含所有第三方依赖
