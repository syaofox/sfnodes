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
│   ├── insightface_utils.py # InsightFace 封装
│   ├── face_detector.py  # 人脸检测
│   ├── lora_notes.py     # LoRA 笔记/说明
│   ├── lora_samples.py   # LoRA 样例图处理
│   └── logger.py        # 日志
├── web/                 # 前端 JS Widget（含 sf_dynamic_slots.js 动态槽位公共库）
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
- `nodes.NODE_CLASS_MAPPINGS` — 全部节点映射（含自定义节点；合并时机与使用注意见后端机制 §3）
- `folder_paths` — 路径管理
- `comfy_extras.nodes_post_processing` — 后处理节点
- `comfy_execution.graph_utils` — `GraphBuilder`（图展开）、`is_link`、`ExecutionBlocker`（官方位置，graph.py 只是 re-export）
- `comfy_execution.graph` — `DynamicPrompt`（DYNPROMPT 隐藏输入对象：`get_node`/`get_display_node_id`/`get_original_prompt`，支持 ephemeral 前缀 id）

## Code Style

- Python 3.10+，无类型注解强制要求
- 使用 `_CATEGORY` 模块级常量定义分类前缀
- 工具函数放在 `sf_utils/` 下对应模块
- 节点实现放在 `nodes/<功能组>/` 下对应文件
- JS Widget 放在 `web/` 目录，文件名与节点功能对应；动态槽位类功能复用 `web/sf_dynamic_slots.js` 公共库（见前端机制 §7）
- `__init__.py` 文件在子目录中为空，仅根目录 `__init__.py` 负责注册（注意：`nodes/utils/` 目前无 `__init__.py`，依赖 namespace package 机制）

## ComfyUI 后端机制（循环/图展开，经验总结）

> 背景：复刻 Easy-Use 的 `easy forLoopStart`/`easy batchAnything`/`easy forLoopEnd` 三个循环节点（2026-08），落地为 `nodes/logic.py` 的 SFForLoopStart/SFForLoopEnd/SFWhileLoopStart/SFWhileLoopEnd + SFMathInt/SFCompare/SFBatchAnything。该模式源于 ComfyUI 官方测试节点 `tests/execution/testing_nodes/testing-pack/flow_control.py`（TestWhileLoopOpen/Close）。

### 1. 节点"图展开"（expand）机制（做循环/动态图节点必知）

- 节点 execute 可返回 `{"result": tuple, "expand": {node_id: node_info}}`：`expand` 的节点会被加入动态 prompt 并执行（`add_ephemeral_node`，id 带前缀如 `0.0.0.5`，`override_display_id` 保持缓存一致性）。
- **result 里的 link 值 `[id, slot]` 会被 ComfyUI 特殊解析**（execution.py 对 `is_link` 的 result 做 `add_strong_link`）：下游消费者拿到的是链接目标节点的输出值，而非字面 `[id, slot]` 列表。这是循环"输出=内部节点输出"的实现基础。
- `GraphBuilder`（comfy_execution.graph_utils）只能按**已注册的类名**创建图内节点（`graph.node("SFWhileLoopStart", condition=total, ...)`）→ 支撑节点必须注册在 `NODE_CLASS_MAPPINGS`（会出现在节点菜单，Easy-Use 同样如此）。这是 forLoopStart/End 无法"独立存在"的原因：循环机制必须依赖注册的 while/math/compare 节点。

### 2. 循环实现模式（SFForLoopStart/End 如何工作）

- **SFForLoopStart**：执行时用 `GraphBuilder` 展开出 `SFWhileLoopStart`（condition=total，携带初始值），自身直接返回 `("stub", index, value1..19)`。循环状态经**隐藏输入 `initial_value0`** 传递。
- **隐藏输入首轮不发送的坑**：前端 `graphToPrompt` 只序列化 widget 值与连线输入，**无连线的 hidden 输入不会出现在 prompt 中** → 首轮 kwargs 无此键（而非 None）→ 代码需默认 `i = 0`；`whileLoopEnd` 重建 open 节点时用 `set_input` 写回 index，后续轮次才能读到。
- **SFForLoopEnd**：`flow` 输入带 `{"rawLink": True}` → 节点收到**原始链接 `[node_id, slot]`** 而非解析值 → `flow[0]` 定位起始节点 id，用 `dynprompt.get_node(id)` 读其 `total`（可能为 widget 值或 link，link 时由图内 compare 节点在运行时解析）。再展开出 `SFMathInt`（index+1）→ `SFCompare`（`index+1 < total`）→ `SFWhileLoopEnd`。
- **SFWhileLoopEnd 的递归（Recurse 机制）**：`condition` 为真时：
  1. `explore_dependencies`：沿 whileLoopEnd 输入链回溯依赖图（排除 `SFForLoopEnd`/`SFWhileLoopEnd` 自身防无限递归）；
  2. `explore_output_nodes`：把循环体内 `OUTPUT_NODE = True` 的节点（如 SaveImage）并入依赖图，保证每轮重跑；
  3. `collect_contained`：从 open 节点出发收集整个循环体（**循环体内可放任意类型节点**，重建时按 `class_type` 字符串创建）；
  4. 用 GraphBuilder 重建全部节点（自身克隆命名 `"Recurse"` 避免 id 指数膨胀），`new_open.set_input("initial_value0..19", 当前值)` 写回状态；
  5. result = Recurse 克隆的输出 links，expand = 重建图 → 下一轮迭代；`condition` 为假时直接返回当前 initial_value 值、不展开 → 循环终止。

### 3. 复刻/实现注意事项（踩坑）

- **`ByPassTypeTuple`/`TautologyStr` 是旧版遗留，可省略**：早期 ComfyUI 按索引校验链接类型时才需要它绕过；现代 ComfyUI 的类型校验仅用于 `VALIDATE_INPUTS`，链接类型不校验，RETURN_TYPES 用普通 tuple + `AnyType("*")` 即可。
- **`ExecutionBlocker` 官方位置是 `comfy_execution.graph_utils`**（graph.py 只是 re-export，避免过早 import torch）。
- **do-while 语义**：total 通过连线传 0 时循环体仍执行一次（widget 侧 min=1 已约束；Easy-Use 原版同行为，忠实保留）。
- **ForLoopEnd 必须被"消费"才会驱动循环（大坑）**：新版 `ExecutionList`（TopologicalSort）只调度被下游引用的节点——`add_node` 从输出节点回溯依赖链入队，**死端节点（输出无下游）从不执行**。`SFForLoopEnd` 的输出必须接一个 OUTPUT_NODE（如 PreviewImage/SaveImage）才被调度 → expand 出 `SFWhileLoopEnd` → 循环才启动。排查"循环不跑/只跑一轮"时先确认 ForLoopEnd 输出有下游消费者（2026-08 实测：删除循环外 PreviewImage 后循环完全不启动）。
- **`explore_output_nodes` 必须收集输出节点的全部链接输入**：原实现 `output_nodes[id] = v` 在遍历多个链接输入时被**最后一个**覆盖（如 SaveImage 的 `images←RMBG` 被 `filename_prefix←TextReplace` 覆盖）→ OUTPUT_NODE 无法并入循环体 → 每轮不重跑。正确写法：`output_nodes.setdefault(id, []).append(v)`，匹配时遍历任意一个 link（2026-08 修复）。
- **循环体内存线性累积（现状，无解）**：循环每轮重建的节点输出全部保留在 `HierarchicalCache` 嵌套 subcache 中直到 prompt 结束（`clean_unused` 只在 prompt 开始时对顶层缓存调用）。重节点（RMBG 3 输出 ~109MB/轮、LoadImagesPath 47MB/轮）× 67 轮 ≈ 13GB RAM。避免二次方增长：不要在循环内做 `SFBatchAnything` 每轮 cat 累积（Σk 张 ≈ 百 GB 级）。可用 `--cache-ram` 启动参数缓解（注意参数名是连字符 `--cache-ram`，`--cache ram` 不是合法参数会导致启动失败）。
- 每轮迭代 forLoopStart 重建时其 expand 会多产生一个无引用的 whileLoopStart 节点（原版同款，无害）。
- `nodes.NODE_CLASS_MAPPINGS` 在**运行时**才包含全部自定义节点（加载器逐个合并），函数内 import 最安全。
- **本地模拟验证**：mock `torch`/`comfy.utils` 后可直接加载 `nodes/logic.py`（构造 `sfnodes`/`sfnodes.nodes`/`sfnodes.sf_utils` 包上下文 + `spec_from_file_location`），用 FakeDynPrompt 断言 expand 图结构、result link 指向、终止分支返回值。

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

### 7. 动态槽位机制（做"多输入/输出节点"必知）

> 背景：`web/sf_dynamic_slots.js` 公共库（2026-08），将循环节点、Text/Image Concatenate、SimpleMath、LogicSwitch 等 6 个文件的动态槽位逻辑统一为配置化实现（`installDynamicSlots(node, config)`），本机用 FakeNode + 事件序列模拟测试（31 项断言）。

- **四种动态槽位模式**（按复杂度）：A. 连线自动增删（前缀匹配，公共库覆盖）；B. 全动态+自动命名/右键重命名/名称传播（`any_pack.js`，特例）；C. 成组配对+自愈（`krea2_dynamic_images.js` 的 imageN/maskN，onNodeCreated/onConnectionsChange/onConfigure 三钩子）；D. 按钮 + widget 显隐 + 状态持久化（`multi_lora.js`、`text_replace.js`，`visibleSlotCount` 随 workflow 序列化）。
- **新节点优先用公共库**：只需配置 `inputPrefix/inputStart/inputCount/inputType/initialInputs` + 输出侧同构；非连续命名用 `inputMatch`（正则，如 simple_math 的 `/^[a-z]$/`），非编号命名用 `nameFor`（如字母表回调）。
- **optional 无 widget 输入默认全显示**：新版前端 `addInputSocket` 对 optional 槽位直接 `addInput`（无隐藏机制）→ 必须 JS 在 `nodeCreated` 时 trim 到初始数量。动态槽位名字必须与后端 `INPUT_TYPES` 完全一致。
- **旧 workflow 恢复依赖前端合并机制**：`nodeCreated` 时 trim（此时无连线），随后 `configure` 时 litegraphService 把保存快照中多出的槽位（extraInputs/extraOutputs）合并回来（源码注释明确支持"custom nodes that dynamically add inputs/outputs via js logic"）。**configure 直赋 links 不触发 `onConnectionsChange`**，恢复时不会连锁加槽。
- **输入/输出判空结构不同**：输入槽位 `.link`（断开为 null，旧版可能 -1）；输出槽位 `.links`（数组，断开为 null/[]）。公共库 `isSlotConnected` 两者兼容（含 `!== -1` 防御）。
- **增删规则**：全部动态槽已连 → 追加下一个（注意空数组 `every()` 恒真，需 `length > 0` 防御）；断开时从尾部 reverse 遍历、遇已连槽即停（只回收尾部连续空槽），保底 initial 个。
- **模拟测试经验**：`cp web/sf_dynamic_slots.js /tmp/xxx.mjs` 后 Node 直接跑（公共库无 DOM 依赖）；FakeNode 需实现 `addInput/removeInput/addOutput/removeOutput/computeSize/setSize`；**事件序列用槽位名定位索引**（动态增删后绝对索引会错位，这是测试脚本最常见的错误来源）；断开事件触发前先把 `link` 置 null。

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
9. JS Widget 使用 `app.registerExtension` 注册，遵循 ComfyUI LiteGraph API；纯工具模块（无扩展行为，如 `sf_dynamic_slots.js`）仅 export 函数即可，由使用者 import
10. 根 `__init__.py` 必须声明 `WEB_DIRECTORY = "web"` 以加载前端 JS Widget（新增 JS 文件后直接放入 `web/`，无需额外注册）
11. 动态槽位类 JS 优先复用 `web/sf_dynamic_slots.js` 公共库，勿重复实现（见前端机制 §7）
12. 后端改动后需重启 docker 容器，`web/` JS 改动需同步 docker 目录并硬刷新（见前端机制 §5 部署注意）

## Testing

本项目无自动化测试框架。验证方式：
- 静态检查：确认 `NODE_CLASS_MAPPINGS` 和 `NODE_DISPLAY_NAME_MAPPINGS` 键一致
- 导入检查：确认所有节点类在根 `__init__.py` 中正确导入
- 依赖检查：确认 `requirements.txt` 包含所有第三方依赖
- 后端模拟测试（无需 ComfyUI）：mock `torch`/`comfy.utils` 后加载节点模块，用 FakeDynPrompt 断言图结构与返回值（循环节点有先例）
- 前端模拟测试：无 DOM 依赖的公共库复制为 `.mjs` 后用 Node 直接跑（FakeNode + 事件序列，动态槽位 JS 有先例）

## 静态检查脚本经验（AST 对比踩坑）

用 Python AST 做"前后端一致性/结构对比"验证时（如对比注册字典、检查节点 INPUT_TYPES），易踩两个坑：

1. **`ast.unparse` 输出的是单引号字面量**：`ast.unparse(v)` 生成的字符串（如 `'interrogate'`、`'CLIP'`）统一用单引号包裹，与手写断言中的双引号字面量（`"interrogate"`）直接比较会**误判不一致**。取值应优先用 `ast.literal_eval(node)`（常量），或按节点类型提取：`Constant.value` / `Name.id`（变量引用）/ `List.elts`。不要拿 unparse 文本与手写字面量做相等比较。
2. **`ast.literal_eval` 遇到变量引用会抛 `ValueError: malformed node or string`**：默认值引用模块常量的表达式（如 `"default": KREA2_INSTRUCT_SYSTEM`）无法直接求值。需分两步：先单独提取被引用的常量（`ast.literal_eval`），再遍历映射，遇 `Name` 节点取其 `id` 后查表替换。

真实案例：比对 `KREA2_PRESETS` 前后端一致性时，`ast.unparse` 残留的单引号让"文本一致"误判为 false；`ast.literal_eval` 直接解析含 `KREA2_INSTRUCT_SYSTEM` 引用的字典抛 ValueError。两处均为检查脚本问题，非代码问题——**先怀疑检查脚本，再怀疑被检查的代码**。
