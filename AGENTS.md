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
│   ├── cutpaste.py      # 剪切/拼接工具
│   ├── insightface_utils.py # InsightFace 封装
│   ├── face_detector.py  # 人脸检测
│   ├── lora_notes.py     # LoRA 笔记/说明
│   └── logger.py        # 日志
├── web/                 # 前端 JS Widget（ComfyUI LiteGraph 扩展）
├── data/                # 静态数据（anime_char CSV、face_distance 字体等）
└── doc/                 # 项目文档（vibecoding.md 开发流程等）
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
- `folder_paths` — 路径管理
- `comfy_extras.nodes_post_processing` — 后处理节点
- `comfy_execution.graph_utils.ExecutionBlocker` — 执行阻断

## Code Style

- Python 3.10+，无类型注解强制要求
- 使用 `_CATEGORY` 模块级常量定义分类前缀
- 工具函数放在 `sf_utils/` 下对应模块
- 节点实现放在 `nodes/<功能组>/` 下对应文件
- JS Widget 放在 `web/` 目录，文件名与节点功能对应
- `__init__.py` 文件在子目录中为空，仅根目录 `__init__.py` 负责注册（注意：`nodes/utils/` 目前无 `__init__.py`，依赖 namespace package 机制）

## ComfyUI 前端机制（经验总结）

> 背景：源自 `SFLoadImageBrowser` 的两次排查（拖拽被第三方扩展劫持、拖拽后蓝框残留，2026-07）。下述为可迁移的通用机制，落地案例见 `web/image_browser.js` 的 `sfnodes.image_browser_drop` 扩展。

### 1. 事件接管（拦截）的通用规则

- 第三方扩展常在 **document 捕获阶段** 注册 `dragover`/`drop` 等监听，先于 ComfyUI 原生（冒泡阶段）执行；自定义扩展要"必然先执行"，就在 **`window` 捕获阶段** 注册（事件传播顺序 window → document → ...，与监听器注册先后无关）。
- `stopPropagation()` 不阻止**同一元素上**的其他监听器（需 `stopImmediatePropagation`）；`window` 捕获阶段调用 `stopPropagation` 可阻断 document 及更深层的所有监听器。
- **接管 = 替代了原生处理器的执行 → 必须自行补偿其状态维护职责**：被跳过的原生处理器中的清理副作用（如拖拽高亮 `app.dragOverNode = null`、hover 状态复位等）不会执行，需在自定义处理器的 `finally` 中复刻，并 `app.canvas?.setDirty?.(false, true)` 触发重绘。
- 只接管自己的目标场景，其余一律放行（不 preventDefault/stopPropagation），避免破坏其他扩展与原生行为。

### 2. 图片输入节点前端机制（做"加载图片"类节点必知）

- 后端 combo 输入带 `{"image_upload": True}` → 前端核心扩展 `Comfy.UploadImage` 自动追加隐藏 `IMAGEUPLOAD` 输入 → 节点自动获得：`node.pasteFiles`（剪贴板粘贴）、`node.onDragOver`/`node.onDragDrop`（文件拖拽）、上传按钮、`node.previewMediaType = 'image'`（预览加载后 `node.imgs` 非空）。自定义加载节点**继承 `LoadImage` 的 INPUT_TYPES 结构即可**获得全部能力。
- 粘贴链路：document `paste` 监听，目标 = 当前选中节点且 `isImageNode(node)` 为真（`previewMediaType === 'image'` 或 `imgs` 非空）→ `node.pasteFiles(files)` → 上传 `/upload/image`（子目录 `pasted`）→ 更新 widget 值 + 预览；否则**新建原生 `LoadImage` 节点**接收。
- 拖拽高亮（蓝框）：canvas 容器 `dragover` 命中 `graph.getNodeOnPos()` 且 `node.onDragOver(e)` 返回真 → `app.dragOverNode = node`。注意：**这是 `ComfyApp` 实例属性，不是 canvas 属性（源码中 `this` 常指 app，极易误判），且无节点类型限制**，自定义节点同样生效。原生在 canvas `dragleave` 或 document `drop` 开头无条件清除。

### 3. 第三方扩展"白名单式"节点判定劫持

- 部分扩展硬编码节点类型判定拖拽目标（如 `node.type === "LoadImage"`；案例：`ComfyUI_Fill-Nodes` 的 `load_image_drop_fix.js`），不识别自定义加载节点 → 拖拽到自定义节点被当空白画布处理，新建原生 `LoadImage` 节点。
- **任何自定义替代节点都可能踩中**；排查"拖拽没进我的节点/新建了 LoadImage"类问题时，优先检查第三方扩展在 document 捕获阶段的监听器，用 setter 拦截定位赋值者（见 §5）。

### 4. Chrome 拖拽隐私限制（易踩坑）

- `dragover` 阶段 `dataTransfer.items` 与 `dataTransfer.files` **为空**（受保护），只能通过 `Array.from(dataTransfer.types).includes("Files")` 判断是否拖文件；`drop` 阶段 `files` 才可用。

### 5. 运行时诊断方法与部署注意

```js
// 检查节点是否具备接收能力（三者齐全 = 粘贴/拖拽链路完整）
app.graph._nodes.forEach(n => console.log(n.comfyClass, {
  onDragOver: typeof n.onDragOver,
  onDragDrop: typeof n.onDragDrop,
  pasteFiles: typeof n.pasteFiles,
  previewMediaType: n.previewMediaType,
  imgs: n.imgs?.length,
}));
// 监控拖拽目标（先装监听，再拖拽，无需拖拽中操作 console）
window.addEventListener('dragover', e => {
  if (e.dataTransfer && Array.from(e.dataTransfer.types).includes('Files')) {
    setTimeout(() => console.log('[drag] dragOverNode:', app.dragOverNode?.comfyClass), 0);
  }
}, true);
// 拖拽后高亮残留时，确认残留状态归属
console.log('app.dragOverNode:', app.dragOverNode?.comfyClass, '| canvas:', app.canvas.dragOverNode?.comfyClass);
// 用 setter 拦截定位"谁在设置 app.dragOverNode"（打印调用栈，直接指向设置者）
let _dn = app.dragOverNode;
Object.defineProperty(app, 'dragOverNode', {
  get() { return _dn; },
  set(v) {
    if (v && v !== _dn) console.log('[SET dragOverNode]', v.comfyClass ?? v.type ?? v.constructor?.name, '<<', new Error().stack?.split('\n').slice(1, 3).join(' << '));
    _dn = v;
  },
  configurable: true,
});
```

**部署注意**：用户运行实例为 docker 部署（`/mnt/github/comfyui-docker/custom_nodes/sfnodes/`，与本地仓库内容一致），修改 `web/` 下 JS 后需**同步该目录**，且浏览器需**硬刷新**（Ctrl+Shift+R）才生效；后端改动需重启容器。

### 6. 文本 widget 自定义右键菜单（做"右键插入"类功能必知）

> 背景：`SFTextReplace` 模板框右键插入特殊标记符（2026-08）。踩坑过程：先依赖 `widget.options.contextMenu` → 弹出系统菜单；修好菜单后插入点又落文本末尾 → 最终方案见 `web/text_replace.js` 的 `showMarkerMenu`/`caretOffsetAt`。

- **`widget.options.contextMenu` 机制不可靠**：它仅在 canvas 绘制态（widget 未编辑、textarea 隐藏）走 LiteGraph 路径；文本 widget 一旦被点击过，其 DOM `inputEl`（textarea）显示并覆盖 widget，右键即触发**浏览器原生菜单**，与 `options.contextMenu` 无关。可靠做法：直接给 `widget.inputEl` 挂 `contextmenu` 监听（`preventDefault` + `stopPropagation`）弹**自绘 DOM 菜单**（fixed 定位、z-index 顶层、视口 clamp），零依赖 ComfyUI 内部 API；`options.contextMenu` 可保留作 canvas 态兜底（textarea 隐藏时其 DOM 监听不触发，两路径互斥）。
- **浏览器右键不更新文本光标**：右键点击 textarea 不移动 `selectionStart`（保留上次位置）也不聚焦。做"插入到鼠标处"必须显式计算 offset：用 `document.caretPositionFromPoint(x, y)` / `document.caretRangeFromPoint(x, y)` 把鼠标坐标换算为字符偏移，存入 `pendingInsertPos` 变量，插入时优先使用（**避免在 contextmenu 里调 `focus()`**，会干扰 ComfyUI widget 焦点状态）。
- **caret 两 API 返回结构不同（易踩坑）**：`caretRangeFromPoint` 返回 `Range`（`startContainer`/`startOffset`），`caretPositionFromPoint` 返回 `CaretPosition`（`offsetNode`/`offset`）。统一读 `startContainer` 会得到 undefined → 静默回退末尾。正确写法：两个 API 都尝试、两种属性名都兼容。
- **插入位置三级回退**：显式记录的鼠标 offset → `activeElement` 的 `selectionStart`（含选区替换）→ 追加末尾。
- **菜单关闭策略**：document 捕获阶段 `mousedown` 且 `!menuEl.contains(e.target)`、Escape、滚轮滚动时关闭；菜单项用 `click`（click 晚于 mousedown，捕获阶段判断不误关菜单项）。

## Code Discovery

优先使用 **codebase-memory 知识图谱**（`search_graph`、`trace_path`、`get_code_snippet`）查找函数、类及其调用关系，代替 grep/glob。该系统已索引整个项目，支持语义搜索和调用链追踪。仅在搜索字符串字面量、错误消息、配置文件等非代码内容时回退到 grep/glob。

## Development Rules

1. **不要启动 ComfyUI 或运行 `pip install`** — 本机仅作为代码编辑环境
2. 可以阅读 `../..` 源码以理解 API 和参考实现
3. 新增节点必须同步更新根 `__init__.py` 的两个注册字典
4. 新增依赖必须同步更新 `requirements.txt`
5. 保持节点类命名一致性：实现类 PascalCase，注册键 `"SF"` 前缀
6. 图像张量格式统一为 `[B, H, W, C]`（ComfyUI 标准）
7. 遮罩张量格式统一为 `[B, H, W]`
8. `sf_utils/` 中的工具函数应当是无状态的纯函数
9. JS Widget 使用 `app.registerExtension` 注册，遵循 ComfyUI LiteGraph API
10. 根 `__init__.py` 必须声明 `WEB_DIRECTORY = "web"` 以加载前端 JS Widget（新增 JS 文件后直接放入 `web/`，无需额外注册）

## Testing

本项目无自动化测试框架。验证方式：
- 静态检查：确认 `NODE_CLASS_MAPPINGS` 和 `NODE_DISPLAY_NAME_MAPPINGS` 键一致
- 导入检查：确认所有节点类在根 `__init__.py` 中正确导入
- 依赖检查：确认 `requirements.txt` 包含所有第三方依赖

## 静态检查脚本经验（AST 对比踩坑）

用 Python AST 做"前后端一致性/结构对比"验证时（如对比注册字典、检查节点 INPUT_TYPES），易踩两个坑：

1. **`ast.unparse` 输出的是单引号字面量**：`ast.unparse(v)` 生成的字符串（如 `'interrogate'`、`'CLIP'`）统一用单引号包裹，与手写断言中的双引号字面量（`"interrogate"`）直接比较会**误判不一致**。取值应优先用 `ast.literal_eval(node)`（常量），或按节点类型提取：`Constant.value` / `Name.id`（变量引用）/ `List.elts`。不要拿 unparse 文本与手写字面量做相等比较。
2. **`ast.literal_eval` 遇到变量引用会抛 `ValueError: malformed node or string`**：默认值引用模块常量的表达式（如 `"default": KREA2_INSTRUCT_SYSTEM`）无法直接求值。需分两步：先单独提取被引用的常量（`ast.literal_eval`），再遍历映射，遇 `Name` 节点取其 `id` 后查表替换。

真实案例：比对 `KREA2_PRESETS` 前后端一致性时，`ast.unparse` 残留的单引号让"文本一致"误判为 false；`ast.literal_eval` 直接解析含 `KREA2_INSTRUCT_SYSTEM` 引用的字典抛 ValueError。两处均为检查脚本问题，非代码问题——**先怀疑检查脚本，再怀疑被检查的代码**。
