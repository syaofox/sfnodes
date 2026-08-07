# 历史经验归档（experience）

> 本文件归档 AGENTS.md 精简（2026-08）时删除的具体机制与踩坑经验；主文档只保留通用约束与每类机制的结论摘要（见 AGENTS.md「经验摘要」）。
> 内容基于当时代码/前端版本（comfyui_frontend_package 1.47.x），可能随版本升级过时，使用时结合代码核实。

## 目录

- [1. ComfyUI 后端机制（循环/图展开）](#1-comfyui-后端机制循环图展开经验总结)
- [2. ComfyUI 前端机制（经验总结）](#2-comfyui-前端机制经验总结)
- [3. 静态检查脚本经验（AST 对比踩坑）](#3-静态检查脚本经验ast-对比踩坑)
- [4. 动态 combo 校验与工作流绑定状态（widget 数据载体）](#4-动态-combo-校验与工作流绑定状态widget-数据载体)
- [5. Qwen3 无审查微调版 + TextGenerate：thinking 参数与思考链（COT）](#5-qwen3-无审查微调版--textgeneratethinking-参数与思考链cot)
- [6. SFPromptTags：@tag 展开注入 / Picks 游标 / 全屏编辑器 / 中文与拼音（复刻 Pixaroma Prompt）](#6-sfprompttagstag-展开注入--picks-游标--全屏编辑器--中文与拼音复刻-pixaroma-prompt)
- [7. SFPauseText：prompt 剪枝闸门（复刻 Pixaroma Pause Text）](#7-sfpausetextprompt-剪枝闸门复刻-pixaroma-pause-text)
- [8. SFPauseImage：快照闸门与预览保存（复刻 Pixaroma Pause Image）](#8-sfpauseimage快照闸门与预览保存复刻-pixaroma-pause-image)
- [9. SFPauseMask：遮罩快照闸门（Pixaroma Pause Mask 同构扩展）](#9-sfpausemask遮罩快照闸门pixaroma-pause-mask-同构扩展)
- [10. SF Workflows：工作流面板（复刻 Pixaroma Workflows）](#10-sf-workflows工作流面板复刻-pixaroma-workflows)

---

## 1. ComfyUI 后端机制（循环/图展开，经验总结）

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

---

## 2. ComfyUI 前端机制（经验总结）

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

### 8. ComfyUI 新版 Vue 前端机制（做"悬停提示/DOM 交互"必知）

> 背景：SFPromptPreset 预设说明展示两次翻车（2026-08）。先做 canvas mousemove + 固定 DOM 卡片 → 完全不生效；清除 `widget.tooltip` 抑制原生提示 → 依然显示。最终发现用户跑的是 ComfyUI 新版 Vue 前端（comfyui_frontend_package 1.47.10），旧 LiteGraph canvas 机制已废弃。最终方案：动态写入 `widget.tooltip`，见 `web/prompt_preset.js`。

- **先确认前端版本再选方案**：ComfyUI 前端自 2025 年起从仓库 `web/` 目录改为独立 pip 包 `comfyui-frontend-package`（新版 Vue 重构）。判断方法：容器内 `pip show comfyui-frontend-package`（Version 1.x = Vue 前端）；**后端版本号 ≠ 前端版本号**；仓库源码副本（`../..`）无前端代码，需查 `Comfy-Org/ComfyUI_frontend` GitHub 仓库或容器内 pip 包 static 目录。
- **Vue 前端下 canvas 事件/坐标方案全部失效**：widget 是 Vue 渲染的 DOM 元素（覆盖在 canvas 上方），`app.canvas` 的 mousemove 监听收不到悬停 widget 的事件（DOM 遮挡）；`node.pos + widget.pos/size` 几何命中同样失去意义。做"悬停 widget 显示信息"类功能**不要走 canvas 事件路线**。
- **tooltip 是 PrimeVue v-tooltip 指令（DOM `.p-tooltip`），不是 canvas 绘制**：来源链 `createTooltipConfig(getWidgetTooltip(widget))`，`getWidgetTooltip` **优先读 `widget.tooltip`，其次 nodeDef 输入定义（后端 `/object_info` 的 tooltip）** → 仅清 `widget.tooltip`/`widget.options.tooltip` 无法抑制原生提示（nodeDef 兜底还在，JS 改不掉后端数据）。
- **动态 tooltip 的正确姿势（通用做法）**：把"当前选中值对应的说明"直接写入 `widget.tooltip`（callback 里随值更新；工作流恢复场景在数据就绪后遍历全图节点同步一次）。新旧前端均优先显示 widget.tooltip：旧版 canvas 绘制 tooltip 读 widget.tooltip，新版 Vue 前端 processedWidgets 为 computed（值变化 → v-tooltip 指令更新）。**后端 INPUT_TYPES 的 `"tooltip"` 键只适合静态提示；逐选项动态说明必须 JS 写 `widget.tooltip` + 前端拉数据**。
- **前端拿后端数据**：注册 `GET /api/sfnodes/...` 路由（server.PromptServer.instance.routes），前端 `api.fetchApi()` 拉取；路由在模块导入时注册，**改动后必须重启容器**，否则 404 且前端静默降级（表现为"功能不生效但无报错"）。
- **模拟测试**：`new Function("app", "api", code)`（去 import 行）注入 mock app/api，Node 直接跑；断言 callback 链（互斥/说明同步）与 `widget.tooltip` 赋值，无需真实 DOM。

### 9. 实际环境调试方式（console 诊断脚本，用户配合执行）

> 背景：SFAnyPack 首槽自动改名 bug（2026-08）。静态分析 + bundle 反查多轮仍未定位，最终靠用户粘贴 console 诊断脚本锁定根因：数据层改名成功但 UI 不刷新 → 槽名渲染读的是 `localized_name` 而非 `name`（见 §10）。

- **不要自行用浏览器访问 ComfyUI**：agent 浏览器访问 `localhost:8188` 会 404/不可用，且用户浏览器 tab 可能正在跑任务（打开即见用户真实工作流，勿动）。实际环境验证一律走"分段 console 诊断脚本 + 用户粘贴反馈"。
- **标准流程（每段只做一件事）**：
  1. **版本检查**：`fetch("/extensions/sfnodes/<file>.js")` 后检查是否含本次修复的特征字符串 —— 排除浏览器缓存/未同步（false = 加载旧 JS，先硬刷新）；
  2. **节点状态检查**：`app.graph._nodes.find(n => n.comfyClass === "SFAnyPack")`，打印 inputs 数量/名字、handler 是否安装（`onConnectionsChange.toString().includes("<内部函数名>")`）。**注意 handler 可能被其他扩展包装，toString 不含特征串 ≠ 未安装**，需结合行为判断；
  3. **事件日志包装**：把 `node.onConnectionsChange` 包一层打印参数（type/index/connected/origin_id/origin_slot），连接分支内再打印源节点/源输出名/目标槽名/sfManualName —— 一次拖线拿到"事件是否触发 + 参数是否正确 + 前置条件是否满足"三份证据；
  4. **数据层检查**：操作后打印 `node.inputs.map(i => i.name + "|" + i.type + "|link=" + i.link)` —— 判断逻辑是否执行；
  5. **UI 层检查**：`[...document.querySelectorAll("span")].map(s => s.textContent.trim())` 过滤目标文本 —— 判断渲染层是否更新。
- **关键判断表**：D0 false → 缓存/部署问题（硬刷新）；槽数未 trim / handler 特征缺失 → 扩展未生效（nodeCreated 没跑）；事件日志为空 → 交互根本没走该事件路径（换 hook）；**数据层已改但 UI 层未变 → 渲染字段问题**（读错字段，如 §10 的 localized_name）。
- **创建测试节点用 UI 添加**：新版前端 `graph.createNode` / `graph.constructor.createNode` 均不可用（LGraph 类静态 createNode 未暴露到实例），诊断脚本不要程序化创建节点。
- **渲染模式判断**：DOM 查不到槽名文本 → 用户跑的是 **litegraph canvas 渲染模式**（槽名画在 canvas 上）；能查到 `text-node-component-slot-text` span → Vue DOM 模式。两种模式对槽名渲染字段的优先级一致（见 §10），但响应式机制不同：Vue 模式直接改属性不触发渲染、替换数组元素触发；canvas 模式靠 setDirty 重绘。
- 可复用模板（本次实战精简版）：

```js
// 1) 版本检查
const t = await (await fetch("/extensions/sfnodes/any_pack.js")).text();
console.log("[D0] JS 含修复:", t.includes("修复特征串"));
// 2) 节点状态（节点请用户用 UI 添加）
const n = app.graph._nodes.find(n => n.comfyClass === "SFAnyPack");
console.log("[D1] inputs:", n.inputs.map(i => i.name + "|" + i.type), "| handler:", n.onConnectionsChange?.toString().includes("内部函数名"));
// 3) 事件日志包装（装好后让用户执行交互）
const orig = n.onConnectionsChange;
n.onConnectionsChange = function (type, index, connected, link_info, slot_info) {
  console.log("[D2] onConnectionsChange:", JSON.stringify({ type, index, connected, origin: link_info && link_info.origin_id + "." + link_info.origin_slot }));
  return orig.apply(this, arguments);
};
// 交互后：
// 4) 数据层
console.log("[D3] inputs:", n.inputs.map(i => i.name + "|" + i.type + "|link=" + i.link));
// 5) UI 层
console.log("[D4] 可见槽名:", [...document.querySelectorAll("span")].map(s => s.textContent.trim()).filter(t => /^(value\d*|out\d*)$/.test(t)));
```

### 10. 槽位显示名机制（localized_name 坑，做"动态改槽名"必知）

> 背景：SFAnyPack 首槽自动改名 bug 根因（2026-08）。症状：数据层 `slot.name` 已改，UI 仍显示旧名，且**只有初始槽受影响、动态加的槽正常**。

- **渲染读的字段优先级是 `label ?? localized_name ?? name`**：litegraph canvas 模式的 `SlotBase.renderingLabel`/`displayName`、Vue 模式的 `InputSlot`/`OutputSlot` 槽名文本、以及命中检测 `getNodeInputOnPos` 的宽度计算，全部优先 `label` → `localized_name` → `name`。
- **初始槽自带 `localized_name`，动态槽没有**：`addInputSocket` 创建槽时传 `localized_name: z(i18nKey, name)`（默认=原名）；`LGraphNode.addInput`（动态加槽）不设 `localized_name` → 渲染回退读 `name`。**"只有第一个/初始槽改名不生效、后加的槽正常"是 localized_name 未同步的典型症状**。
- **改槽名必须同步 `name` 和 `localized_name`** 两字段；若槽已有 `label`（优先级更高，`addInputSocket` 不设但第三方可能设）也需同步。Vue 模式下还需替换数组元素才触发渲染（见 §9 渲染模式差异）。
- 关联坑：动态槽位节点的**输入槽名必须与后端 `INPUT_TYPES` 键一致**（prompt 序列化依赖），改名后由前端 `graphToPrompt` 补丁映射回 `value{index}`（见 §7 与 any_pack.js 的 `installPromptMapping`）。

### 11. Vue 新版 LLink 字段差异与通用 combo 选择器（做"连接感知/选项同步"类功能必知）

> 背景：SFComboSelector 通用下拉选择器（2026-08，前端 1.48.6）：输出连到目标节点 combo 输入（Convert to input 后）→ 下拉选项自动同步为目标选项列表。踩坑链：连线后选项不动 → 事件没触发? → 数据层取不到列表 → 目标节点解析失败。

- **坑 1（根因）：Vue 新版 LLink 字段名变了**。旧版 `link.target_node`/`origin_node`，新版为 **`target_id`/`origin_id`**（`target_slot`/`origin_slot` 未变），且**节点 id 为字符串**。按旧字段取 → `undefined` → 目标解析失败 → 选项同步静默失效（无任何报错）。取目标必须 `link.target_node ?? link.target_id`，节点查找用 `String(n.id) === String(id)` 比较。
- **坑 2：combo 输入槽 `slot.type` 在新版是字符串 `"COMBO"`**（不是旧版的选项数组/逗号串）→ 从槽类型取不到列表。**Convert to input 后原 combo widget 仍保留在 `node.widgets`（含动态重建的 `options.values`）**——这是动态选项（如 SFPromptPreset 的 441 项）唯一可靠来源，nodeDef 兜底只有静态列表。三级兜底：槽类型（数组/JSON/逗号串归一化）→ 同名残留 widget 的 `options.values` → nodeDef。
- **坑 3：连接事件触发时 `outputs[0].links` 可能尚未更新** → `onConnectionsChange` 里 `setTimeout(syncOptions, 0)` 延迟执行；工作流加载恢复连接不触发该事件 → 挂 `onAfterGraphConfigured`/`onGraphConfigured` 补同步。
- **坑 4：combo widget 是 DOMWidget（ComboWidget，带 `element`）** → 更新选项需整体替换 `widget.options` 对象 + 重赋 `values` 数组引用（Vue 渲染监听引用变化）并 `setDirtyCanvas`；断线/无连接恢复占位 `[""]`。
- **通用输出类型**：目标不可预测时用 `RETURN_TYPES = (AnyType("*"), ...)`（项目 `sf_utils/common.py`）——后端 `validation.py` 与前端 `isValidConnection` 对 `*` 均直接放行，可连任意 combo 输入；动态选项节点标配 `VALIDATE_INPUTS → True`（见 §4.2）。ComfyUI 官方生态同类参考：`ControlNetPreprocessorSelector`（输出类型 = 具体 combo 列表，`isValidConnection` 对数组按元素逐项匹配，任一共有即可连）。
- **诊断**：node 上暴露 `_sfComboSync`/`_sfComboGetLinks`/`_sfComboFindTarget` 调试接口，console 分段脚本直接调用定位（见 §9）。

---

## 3. 静态检查脚本经验（AST 对比踩坑）

用 Python AST 做"前后端一致性/结构对比"验证时（如对比注册字典、检查节点 INPUT_TYPES），易踩两个坑：

1. **`ast.unparse` 输出的是单引号字面量**：`ast.unparse(v)` 生成的字符串（如 `'interrogate'`、`'CLIP'`）统一用单引号包裹，与手写断言中的双引号字面量（`"interrogate"`）直接比较会**误判不一致**。取值应优先用 `ast.literal_eval(node)`（常量），或按节点类型提取：`Constant.value` / `Name.id`（变量引用）/ `List.elts`。不要拿 unparse 文本与手写字面量做相等比较。
2. **`ast.literal_eval` 遇到变量引用会抛 `ValueError: malformed node or string`**：默认值引用模块常量的表达式（如 `"default": KREA2_INSTRUCT_SYSTEM`）无法直接求值。需分两步：先单独提取被引用的常量（`ast.literal_eval`），再遍历映射，遇 `Name` 节点取其 `id` 后查表替换。

真实案例：比对 `KREA2_PRESETS` 前后端一致性时，`ast.unparse` 残留的单引号让"文本一致"误判为 false；`ast.literal_eval` 直接解析含 `KREA2_INSTRUCT_SYSTEM` 引用的字典抛 ValueError。两处均为检查脚本问题，非代码问题——**先怀疑检查脚本，再怀疑被检查的代码**。

---

## 4. 动态 combo 校验与工作流绑定状态（widget 数据载体）

> 背景：SFTextPreset 工作流绑定文本预设节点（2026-08），落地为 `nodes/text/text_preset.py` + `web/sf_text_preset.js`。需求：预设绑定当前工作流，其他工作流添加此节点是全新空预设。

### 1. "状态绑定工作流"的标准模式：数据存 widget 值（数据载体）

- **所有 widget 值（含 `display: hidden`）都会随 workflow JSON 序列化**（前端保存/加载/复制/导出嵌入自动跟随，`serialize = false` 可排除）。把预设等状态数据以 JSON 字符串存进隐藏 STRING widget → 预设天然"绑定"当前工作流：保存即持久化、复制/导入跟随、**新工作流添加节点用 INPUT_TYPES 默认值 = 全新状态**，无需后端存储/路由（`TextDropdown` 的 `options_json` 隐藏 widget 是同类先例，但它叠加了全局 API 轮询，做全局共享才需要）。
- **combo 只保存 value，不保存 options 列表**：加载 workflow 时 combo 选项恢复为 INPUT_TYPES 静态列表 → 前端必须在数据就绪后重建 `widget.options.values` 并校验当前值（失效则回落第一个/空占位）。
- **恢复时序坑**：`onNodeCreated` 早于 widget 值恢复（`configure`）→ nodeCreated 里读隐藏 widget 拿到的是默认值。重建选项需挂 `node.onAfterGraphConfigured`（widget 值恢复完成后回调，项目先例 `any_pack.js`）或 prototype 的 `onConfigure`（先例 `krea2_dynamic_images.js`）补一次同步。

### 2. 动态 combo 的 "Value not in list" 校验坑（必踩）

- 症状：`[ERROR] * SFTextPreset 1: - Value not in list: preset: 'a' not in ['']` → `prompt_outputs_failed_validation`，输出被忽略。
- 机制：旧版 ComfyUI `execution.py` 的 `validate_inputs` 对 list 类型（combo）输入检查 **值 ∈ INPUT_TYPES 静态选项列表**；前端动态新增的选项（存于 workflow 数据）不在列表中 → 校验失败。新版（本机源码 `comfy_execution/validation.py` 已改为仅链接类型校验）不会报，**用户 docker 为旧版才会踩中**。
- 解法：节点定义 `@classmethod VALIDATE_INPUTS(cls, **kwargs): return True` 完全接管校验（项目先例 `load_images_path.py` 的目录校验）。动态选项节点标配；execute 内需自行容错任意值（找不到 → 空输出）。
- 注意：`VALIDATE_INPUTS` 生效于所有输入，只适合值本身无类型风险的情况（STRING 无碍）。

### 3. 实现与测试注意

- 按钮 widget 用 `node.addWidget("button", ...)`；预览/展示 widget 设 `serialize = false` 避免污染 workflow（按钮无 value 天然不保存）。
- **前端模拟测试能抓实现 bug**：`tests/test_text_preset_js.js`（FakeNode + DOM mock + 事件序列，38 项断言）抓出 `openMgr` 漏写 `mgrEl = overlay` 导致 Escape 无法关闭弹窗的 bug——弹窗类功能务必覆盖"打开/增删改/Escape 关闭"全链路断言。
- 测试断言别想当然：更新操作改的是被选中预设，新增预设的文本保持新增时写入值，断言文案需与操作序列一致（曾把 C 的文本误断为更新值导致误报）。


---

## 5. Qwen3 无审查微调版 + TextGenerate：thinking 参数与思考链（COT）

> 背景：SFPromptPreset 的 optimize_request 喂给官方 TextGenerate 节点（LLM）优化 Krea2 提示词（2026-08）。踩坑：Gemma 官方版安全拒绝 NSFW draft；换 qwen3-vl-4b-heretic（无审查微调版）后输出混入 "Wait, let me check..." 自发思考链。

### 1. Qwen3 的推理抑制机制（模板层，非参数层）

- ComfyUI 的 qwen3vl 模板（`comfy/text_encoders/qwen3vl.py`）在 `thinking=False` 时向输入末尾注入**空 think 块** `<think>\n\n</think>\n\n`——Qwen3 官方约定：模型见到空 think 块就不推理。此约定只对**遵守它的 instruct 版模型**有效。
- **无审查微调版（heretic 类）基于 thinking 变体训练，无视空 think 块约定**：`thinking=off` 时推理以**自发 COT 混入正常输出通道**（无 `<think>` 标签包裹，无法程序化剥离），污染 TextGenerate 输出。
- **实测反转**：对这类模型**打开 `thinking=True` 反而输出正常**——模型进入原生推理通道，推理走 `<think>` 标签规范结构，最终答案以训练分布内的规范形式给出。参数选择取决于模型微调来源，不能一概"保持默认"。

### 2. 相关参数结论（TextGenerate 节点）

- `thinking`：instruct 版保持 off；总是推理的无审查版实测 on 正常。
- `use_default_template`：**必须保持默认 True**。关闭 → `skip_template=True` → 模板与空 think 块注入逻辑整体跳过（qwen3vl.py 的 `if skip_template: llama_text = text` 分支），抑制机制失效且输入缺少 `<|im_start|>assistant` 引导。
- `CLIPLoader` 加载 qwen3-vl 系模型类型选 `qwen3vl_4b/8b`（模板正确才会走 think 抑制/原生通道逻辑）。

### 3. 指令层兜底（SFPromptPreset optimize_request）

- 自定义优化指令含防拒条款（"宁可直接回显 draft 也不输出拒绝文本"，解决对齐模型拒绝断管线）与防思考条款（"Do NOT think step by step, generate directly in a single pass"，对自发 COT 兜底）。
- 防拒/防思考条款对强对齐模型（Gemma 官方版）无效——安全拒绝发生在训练层，措辞无解，只能换无审查模型。

---

## 6. SFPromptTags：@tag 展开注入 / Picks 游标 / 全屏编辑器 / 中文与拼音（复刻 Pixaroma Prompt）

> 背景：复刻 Pixaroma 的 `PixaromaPrompt` 节点（2026-08），落地为 `nodes/text/prompt_tags.py` + `web/sf_prompt_tags*.js` 六模块（lib/store/cursors/guard/editor/主扩展）。后端极简（115 行拼接），主体是前端：DOM widget 输入框、@tag 自动补全、标签库、全屏编辑器、Picks 模式。本节点是项目内**最完整的前端注入 + 全屏 UI + 多模块**案例，经验可直接迁移。

### 1. @tag 展开与 PromptState 注入（Sliders/Seed 模式变体，做"队列时改写节点输入"必知）

- **分工**：前端 `app.graphToPrompt` hook 在队列时展开 token 并写入 `entry.inputs.PromptState`（隐藏 STRING 输入的 JSON），后端只解析拼接——浏览器才能读标签库，纯 API 运行拿到 `"{}"` 即空（文档写明需接 text_in）。
- **随机即缓存键**：`*wildcard`/`#list` 每次 run 重掷，展开结果串不同 → 节点缓存键不同 → 自动重跑，无需 nonce（Pixaroma 不变式 #3）。
- **遍历 prompt 的注意**：`Object.keys(prompt)` 里键是节点 id（子图含 `"5:3"` 复合 id）；用递归建立的节点索引（前缀 `"id:"`）精确匹配，复合 id 只接受"恰好一个节点携带该尾部且非顶层"的兜底，否则注入会把 A 节点的提示词换成 B 的。
- **防双包装**：模块被二次求值（带戳/不带戳的 import、热重载）会每次 run 掷两次 → 用 `app._sfPromptTagsPatched` 全局标志只包装一次。

### 2. Picks 游标与队列提交语义（做"随机/顺序选择跨 run 记忆"必知）

- **shuffle（默认，发牌不重复）/ random / in order（每 run 推进一次）**：位置存未注册设置 `sfnodes.PromptTags.Cursors`（键 `list:<name>`/`cat:<category>` 小写），**永不进 workflow、永不进库导出**。
- **大坑：`app.graphToPrompt` 不是队列**——Export、分享、各保存按钮都会触发。若在掷时直接写游标，"按顺序"列表会被这些场合白白推进、甚至跳过选项。解法（Pixaroma 语义）：
  - 掷出的选择先存内存 `_pending`（Map<key, {picks[], state, build}>），同 build 内重复 `#fruit` 按 occ 计数沿同一状态续发（in order 例外：`want=0`，同 build 全部同值）；
  - `beginPickBuild(prompt)` 把 build id 用 **WeakMap 挂到 prompt 对象**（不能挂模块全局：窗口期内别的 graphToPrompt 会把计数移走）；
  - `queuePrompt` patcher 成功后 `commitPicks(queued)`——**只消耗恰好被 POST 的那个 build** 的选择（从 args 里找带 `output` 的对象，不假设参数位置）。
- **落地细节**：池尺寸变化（st.n !== n）→ 重新开始；坏牌堆（重复/越界）整副丢弃重洗；新牌堆不开旧堆最后一张（随机交换而非轮转——轮转会把被挡牌堆映射到同一允许牌堆、概率翻倍，实测）；`resetCursor`（↺）删键；`renameCursor`（改名）搬位置——**改名不是内容变化，"next 4 of 12" 不能变回 1**，且改名瞬间就搬（拖到 blur 会让一次 run 查新名开新序列）。

### 3. 标签库存储与全屏编辑器（做"设置存储 + 整页 UI + 工作副本"必知）

- **未注册设置**（`app.ui.settings` 读写，键 `sfnodes.PromptTags.Library`）：不声明 settings[] 也能持久化（comfy.settings.json 是纯 JSON 合并），机器私有、跨工作流共享、随插件更新存活。库数据 `{version, categories, listCats, tags, catModes}` 单 JSON 字符串。
- **工作副本模式**（editor 正确性核心）：
  - 打开时 `reloadLibrary()` **强制重读**（不能读内存缓存：另一标签页可能已改）；
  - 编辑只动工作副本，`commitLibrary` 防抖 350ms 持久化 + 订阅者即时重高亮；
  - 关闭时 `isSameAsStored` 判定（两侧先 normalize 再 JSON 比较，键序/默认值差异不算不同）**有变化才写回**——无条件写回会把本标签页快照盖到他标签页的编辑上；
  - `flushLibrary` 只写"确实有待写的防抖"（无条件 flush 会静默取消 isSameAsStored 保护）。
- **Ctrl+Z 守卫**：ComfyUI 的 ChangeTracker 在 window+capture 注册、后注册监听永远无法抢先；唯一官方信号是构造器静态槽 `maskeditor_is_opended`（返回 true 整条撤销链跳过）。实现要点：引用计数（两个编辑器可同时开）、自愈（overlay 异常拆除自动放下）、归还前验证槽还是自己的。
- **Esc 分层（capture 阶段 window 监听，压过字段自身的冒泡处理）**：最上层 modal → 分类菜单 → 字段取消（**字段暴露 `_sfCancel` 句柄供 capture 直接调用——绝不能 fallback 到 `blur()`，blur 监听器会提交编辑**，Escape 变成"应用了要放弃的编辑"）→ 搜索框清空 → 编辑器关闭。
- **图标**：插件无资产服务路由 → delete/help 两个 SVG 内联 data URI（mask-image 用法不变）。
- **类名隔离**：全屏 overlay 类前缀 `sf-ptge-`（源插件是 `pix-prled`），防止与同装的其他插件 CSS 互踩。
- **无撤销设计**：一切可能丢失的操作先 `confirmDanger`（标题/说明/列出将被删内容/可选"先导出备份"，导出不关对话框），然后直接应用——历史证明给编辑器加 undo 栈是最大 bug 源。

### 4. 中文支持与拼音检索（做"中文 token/名称 + 拼音搜索"必知）

- **token 语法放宽为 `[\p{L}\p{N}_-]`（必须带 u flag）**，中文可作 tag/分类名；`NAME_RE` 清洗同理。名称清洗丢弃的导入项（dropped）文案同步（"letters, numbers, Chinese characters, - and _"）。
- **token 边界规则（关键决策）**：原版用 `\p{L}` 判边界（防 email），中文场景会误伤 `画@水彩`、`中文@tag`。改为**符号前是 `Latin/希腊/西里尔/数字/组合标记/_` 集合才不算 token**：`user@name`、`2*2` 保护保留，CJK 前照常识别。同一集合用于 AC 弹出条件与 `tagSep/tagTrail` 空格规则（中文前后不插空格，拉丁语境保持补空格防 `@a@b` 粘连）。链式规则（`@a@b`）不依赖字符类别，跨种不链式保留。
- **拼音表**：运行时不能引 npm 依赖 → **一次性生成后内联**。流程：`npm pack pinyin-pro`（网络可用时）→ Node `TextDecoder('gb2312')` 遍历 GB2312 一级区 `0xB0A1-0xD7F9`（3760 个编码位）→ pinyin-pro 取无声调常用音 → 生成 `PINYIN_MAP`（~47KB 模块）。`pinyinMatch(name, q)` 三路子串：原名小写 / 全拼（`youhua`）/ 首字母（`yh`）。缺表生僻字该字拼音搜不到，原名匹配始终可用。
- **多音字**：pinyin-pro 带上下文（`重庆` → chong qing），单字取常用音，个别不准可接受。

### 5. 模拟测试方法论（mock DOM 冒烟，抓语法检查漏掉的运行时错误）

- **纯函数层**（lib/cursors/pinyin）：copy 为 `.mjs` 直跑（无 app/DOM import）；有 app 依赖的文件把 `import { app } from "/scripts/app.js"` 替换为 `const app = globalThis.app` + mock `app.ui.settings`（内存对象模拟 comfy.settings.json），**相对 import 要一起 copy 到同 tmp 目录并改 `.mjs` 后缀**。
- **冒烟层**（editor/主扩展，`test_prompt_tags_editor_smoke.js`/`test_prompt_tags_main_smoke.js`）：mock DOM 用**惰性元素**（任何 `querySelector` 返回新元素、事件绑定/appendChild 不炸），模块级 `document.addEventListener`、`getComputedStyle`、`ResizeObserver`（Node 无此全局，代码需 `typeof` 判存在）都要 mock。editor 冒烟：空库/有库/prefill 打开、关闭、复位、onInsert 全链路；主扩展冒烟：**graphToPrompt 端到端**（构造 `{output: {"1": {class_type, inputs: {}}}}` 断言 PromptState 注入值）+ queuePrompt→commitPicks 游标落盘。运行时错误（缺 mock、绑定错）在此层必现。
- **测试环境坑**：直改 mock settings 后 store 内存缓存不会失效 → 需调 `reloadLibrary()`（真实场景改库都走 store API 自动同步，此坑纯属测试环境）。
- **断言反例（都是写错断言而非代码错）**：shuffle 发完牌 `cursorInfo` 显示 `0 || n` 即整副数量；首字母 `yh` 是 `y` 的超集（`pinyinMatch(x,"y")` 恒真）；旧库启发式（全 list 分类 → List 侧）是特性不是 bug；`import "x" * 30` 在 Python 字符串里不是合法 JSON。

### 6. 模块边界（复用/修改时的快速索引）

- `sf_prompt_tags_lib.js`：纯函数（normalize/scanTokens/expandAll/reorder/导入导出变换/模式常量）——无 app/DOM，测试 copy 直跑。
- `sf_prompt_tags_store.js`：库存储（settings 读写/防抖 commit/reload/isSameAsStored/applyImport 包装）。
- `sf_prompt_tags_cursors.js`：游标（nextIndex/commitPicks/reset/rename，含队列提交语义）。
- `sf_prompt_tags_guard.js`：Ctrl+Z 守卫（window.app 构造器槽）。
- `sf_prompt_tags_editor.js`：全屏编辑器（工作副本、侧栏、卡片、导入导出、confirmDanger）。
- `sf_prompt_tags_pinyin.js`：内联拼音表 + pinyinMatch（生成脚本一次性，勿手改数据）。
- `sf_prompt_tags.js`：DOM widget 节点本体 + graphToPrompt/queuePrompt patcher + 右键菜单。

---

## 7. SFPauseText：prompt 剪枝闸门（复刻 Pixaroma Pause Text）

> 背景：复刻 Pixaroma 的 `PixaromaPauseText`（2026-08），落地为 `nodes/text/pause_text.py` + `web/sf_pause_text*.js` 三模块（lib/ui/主扩展）。节点是内联文本闸门：停在节点处编辑 LLM 文本，Continue 只跑下游（模型被跳过）。本节点与 SFPromptTags 共享"前端改写 prompt"模式，但**引入了真正的 prompt 剪枝**（删除节点）与 Python→前端回填（executed 事件），是这两个机制的完整案例。

### 1. 双钩子拆分：注入 vs 剪枝（做"队列时删除/改写节点"必知）

- **`app.graphToPrompt` 不是队列**——Export（API）、工作流分享、多个保存按钮都会触发。在这里删除节点会把**导出的工作流静默截断**（用户拿到缺一半的导出文件）。所以：
  - `graphToPrompt` 只做 INJECT（写隐藏 PauseState 的 {mode, text}），绝不删节点；
  - **剪枝移到 `api.queuePrompt`**——所有浏览器 run 的唯一提交漏斗（普通 Run、局部"执行节点"、批量队列都经过），`args[1].output` 拿 prompt 对象，原样转发 ...args 保 partialExecutionTargets。
- **多闸门排序**：continue 闸门剪自己的下游分支时可能连带删掉位于其上游的另一个闸门 → 按 `MODE_RANK = {continue: 0, pause: 1, pass: 2}` 排序，**continue 先处理**；处理前检查 `if (!out[g.id]) continue`（已被更早的闸门剪掉则跳过）。
- **防双包装**：`api._sfPauseTextQueueWrapped`/`app._sfPauseTextPatched` 全局标志（模块二次求值/热重载时每个 run 会注入两次）。
- **FAIL OPEN**：注入/剪枝的异常只 console.error，绝不能让 `await _origGraphToPrompt` 或 `_origQueuePrompt` 抛错——否则整个工作流的 Run 都会坏。核心调用永远不包 try。

### 2. prune 语义（`applyGateMode`，纯函数可直接单测）

- **pause**：从闸门出发前向 BFS（buildConsumers → collectDownstream）删除其下游全部节点——闸门是 `OUTPUT_NODE = True`，成为该分支终点，并行分支不受影响。剪枝依赖"节点是 OUTPUT_NODE"这一事实：死端节点本就不执行，pause 的语义就是让闸门成为唯一终点。
- **continue**（最复杂，四个子步骤）：
  1. 删 `entry.inputs.text`（上游模型链断供），PauseState 注入 {mode: continue, text: 编辑文本}；
  2. **菱形重路由**：闸门之后还有节点直接读闸门原文本源（gateSrc 精确匹配 [origin, slot]）会把整个上游拉活——这些链接改写为 `[闸门id, 0]`（闸门现在发出同一份编辑后的文本），必须在重建 consumers 之前做；
  3. 保留下游（keep = downstream + 闸门 + addAncestors）；upstream = gateSrc 节点 + 祖先（被跳过的模型链）；
  4. **只删"会拉活被跳过上游"的输出节点**：pullsUpstream = 重路由后从 upstream 前向可达的一切；遍历全图，`keep` 之外且 `pullsUpstream` 之内且 `isOutput` 的才删。**无关输出分支（自有来源）必须保留照跑**——老 bug 是删掉 keep 之外的所有输出节点，静默杀死无关分支。非输出节点留作无害孤儿（永不校验/运行），下游 Save 节点因此保留完整生成元数据。
- **isOutput 判定**：从 `window.LiteGraph.registered_node_types[classType]?.nodeData?.output_node` 读（live，不缓存）；**注册表缺失 → null → 回退"删一切"**（安全方向：上游仍被跳过）。
- **解析不到活节点时默认 pass（不剪）**：找不到节点（子图 id 失配等）给破坏性的 pause 会截断工作流，给无害的 pass 只是不剪——fail-safe 方向选"少做"。
- **未接线闸门 continue**（gateSrc null）：upstream 为空 → pullsUpstream 为空 → 不删任何输出节点，只删自己断开的 text 链接——完全正确（没接模型就没什么可跳过的）。
- **无 IS_CHANGED（Python 侧大坑）**：曾用 `IS_CHANGED = float("nan")` 让闸门每次"变化" → 节点缓存键折叠**所有祖先**的 IS_CHANGED（caching.py::get_node_signature）→ 闸门下游每次 Run 全量重跑（固定种子下采样器照跑）。去掉它零损失：缓存节点仍会重发 ui payload（文本框照样刷新）；模式与文本在隐藏输入里本就在缓存键中。

### 3. 前端状态与回填（executed 事件）

- **状态存 `node.properties.pauseTextState`**（{gate: pause|pass|keep, text, original}）随工作流序列化——保留编辑是设计（重开工作流继续编辑）；text/original 只在真实动作（打字/Revert/Run 回填）时变化，纯加载路径不动 → 打开已保存工作流不误标 modified。
- **keep 是持久化的 continue**：普通 Run 时 `gate === "keep"` 映射为 continue 模式（跳模型、复用当前文本、下游照跑）——一次性 submit mode 与持久 gate 在 `collectGates` 里统一解析，注入与剪枝共用同一份 gates 列表，两者永不产生分歧。
- **一次性提交模式**：Continue/Regenerate 按钮 → 挂 `node._sfPauseTextSubmitMode = "continue"/"pause"` → `await app.queuePrompt(0, 1)` → **finally 清除**。剪枝钩子在 `app.queuePrompt` 的 await 内部运行，此时模式仍可读。同一时刻只允许一个闸门携带模式（其余清空）。
- **executed 回填**：Python 返回 `{"ui": {"sf_pause_text": [text]}}` → 前端 `api.addEventListener("executed")` 读 `d.output.<ui键>` → `setModelText`（替换盒子 + 重置 original 基线）。Python 只在"有线 Pause/Pass 的新鲜模型捕获"时 emit——收到即替换；无线时前端保留手打内容（不 emit 的设计）。节点 id 可能是字符串，`parseInt` 兜底。
- **Regenerate**：沿 `text` 输入上游递归（visited 集 + 深度 50），把所有名称含 "seed" 的数字 widget 滚成新随机值（跳过 `control_after_generate` combo——其值非数字）；没找到种子时 flash 提示"可能不会变化"。

### 4. 移植简化与测试

- **shared 依赖裁剪**（同 SFPromptTags 惯例）：`isVueNodes`（`window.LiteGraph?.vueNodesMode`）与 `applyAdaptiveCanvasOnly`（canvasOnly 实时 getter，Vue 下 false 否则不渲染）内联；省略 accent 设置、resize floor、canvas zoom；状态条由 canvas 绘制/Vue nudge 双路径简化为普通 DOM 行（两渲染器统一）。
- **测试**（3 个文件）：后端 19 断言（三模式 × 有线/无线 + 容错）；prune/state 纯函数 30 断言（pause 删下游/continue 菱形重路由/无关分支保留/未接线不删/keep 映射）；主扩展冒烟 17 断言（注入→剪枝→executed 端到端，mock `registered_node_types` 里的 `output_node` 标记）。
- **复查时抓到的断言反例（全是测试错、代码对）**：continue **只删下游链之外**拉活上游的输出节点（链上的 SaveImage 消费闸门输出 → 保留）；`addAncestors` 只并入**图中存在**的祖先；菱形重路由后 LLM 保留为无害孤儿（无输出节点读它则不删）；`editedTextOf` 优先读活 textarea——测试直改 properties.text 不会反映到注入值，需同步 `ta.value`；mock 注册表漏登记 class_type 会让 `isOutput` 返回 false → 该输出节点"没被删"。

### 5. 模块边界（复用/修改时的快速索引）

- `sf_pause_text_lib.js`：state（getState/setGate/setText/setModelText/revertText/isEdited）+ prune 纯函数（isLink/buildConsumers/collectDownstream/addAncestors/applyGateMode）——无 app/DOM，测试 copy 直跑。
- `sf_pause_text_ui.js`：DOM widget（状态条/文本框/三态切换/Copy-Revert/计数/按钮）+ renderPause/syncText/statusText。
- `sf_pause_text.js`：主扩展（setupNode/双钩子/executed/Regenerate/一次性提交模式/防双包装）。
- 后端 `nodes/text/pause_text.py`：无状态（文本随隐藏输入携带），OUTPUT_NODE = True，无 IS_CHANGED。

---

## 8. SFPauseImage：快照闸门与预览保存（复刻 Pixaroma Pause Image）

> 背景：复刻 Pixaroma 的 `PixaromaPauseImage`（2026-08），落地为 `nodes/image/pause_image.py` + `nodes/image/preview_routes.py`（新后端路由）+ `web/sf_pause_image*.js` 三模块（lib/ui/主扩展）。与 SFPauseText 是兄弟闸门（prune/双钩子/一次性模式/executed 机制完全同构），核心差异是**图片无法像文本一样随隐藏输入携带**——必须走快照文件，并引入 PNG 元数据嵌入与自定义保存路由。

### 1. 快照机制（做"跨 run 传递图片"必知）

- **图片不能塞进隐藏输入**（太大）→ pause 时后端把 `image[0]` 存 `folder_paths.get_temp_directory()/sf_pause_<id>.png`；continue 时前端把上游剪出 prompt、后端读回同一文件。**UNIQUE_ID 对节点跨 run 稳定**（ComfyUI 约定），所以 pause 写入的文件正是 continue 读回的文件——这是整个机制成立的前提。
- **快照文件名前缀必须与源插件隔离**：原版 `pixaroma_pause_<id>.png`，同 node_id 时与 pixaroma 插件撞文件互相覆盖 → sfnodes 改 `sf_pause_`。任何"按 id 落盘"的临时文件都该检查前缀隔离。
- **生命周期**：temp 目录随 ComfyUI 重启清空 → continue 时 `os.path.isfile` 检查 + 读失败（截断/损坏）都抛**清晰中文错误**（"快照已过期/无法读取，请重新 Pause"），而不是原始 PIL 回溯炸掉整个工作流。`with Image.open(...)` 释放句柄（Windows 文件锁，否则下次 pause 无法覆盖）。
- **保存失败降级**：`pil.save` 抛 OSError（temp 只读/磁盘满）不炸 run——图片照常透传，只是 continue 拿不到新快照（`ui` 键缺省为空 dict）。
- **batch 语义**：只快照 `image[0]`（首帧），continue 回放单帧 1xHxWxC——与原版一致（v1 限制）。

### 2. PNG 拖回重建与元数据（做"保存图片可拖回"必知）

- **PngInfo 字节格式对齐 ComfyUI SaveImage**：`PngInfo().add_text("prompt", json.dumps(prompt))` + `add_text("workflow", ...)` 写入 tEXt 块——拖回画布时 ComfyUI 读这两个块重建。`parameters` 块（Civitai/A1111）从源 PNG 的 `pil.info` 穿过重编码，不重建丢失。
- **嵌入前必须 `_json_safe`（NaN/Inf → 字符串）**：prompt 里任何节点的 `is_changed: [NaN]` 会让嵌入的 workflow 是非法 JSON——拖回时前端 JSON.parse 抛错、整个重建失败。这是复查抓到的真缺陷（初版直接 `json.dumps(prompt)`）。
- **尊重 `--disable-metadata`**：ComfyUI 启动参数全局关闭元数据。`comfy.cli_args.args.disable_metadata` **每次调用实时读**（import 顺序不保证，快照可能是解析前默认值），fails open（模块缺失时照常嵌入）。
- **执行期工作流只存运行时**：executed 事件把 `_sf_pause_meta`（pause/pass 新鲜捕获时 Python 嵌入的 {prompt, workflow}）存 `node._sfPauseImageExecMeta`——**绝不进 node.properties**（会撑爆已保存工作流）。Save 按钮优先用它（精确的生成种子），无则回退活图 `app.graphToPrompt()`。

### 3. `_safe_prefix` 段清洗（复查抓到的真 bug，做"文件名清洗"必知）

- **leading `/` 与 `".."` 段检查必须在任何清洗之前**。初版实现先 `re.sub` 删掉点/斜杠再检查——`".."` 被删成空串、`/abs` 被 `.strip("/")` 剥成 `abs`，两个检查永远不命中（路径穿越失效）。正确顺序：strip → 长度检查 → `startswith("/")` → `split("/")` 段级 `== ".."` 检查 → 才做段清洗。
- **段清洗对齐原生 SaveImage**：只替换 Windows 非法字符 `[<>:"|?*\x00-\x1f\x7f]` 为 `_`（**非拉丁文字/空格原样通过**，初版用 `[^A-Za-z0-9_-]` 全删是错的）；折叠重复 `_`；循环剥离边沿空白/下划线/尾点（`"test._"` 需多遍）；Windows 保留设备名（CON/NUL/COM1…）加 `_` 后缀；输入上限 256、输出 100。

### 4. 自定义保存路由（做"前端保存文件"必知）

- 两个 POST 路由（`nodes/image/preview_routes.py`，仿 `sf_utils/lora_notes.py::_register_routes` 先例：`from server import PromptServer` → `ins.routes` 装饰器、try/except 包裹、模块导入时副作用注册、`__init__.py` import）：
  - `/api/sfnodes/preview/save`：base64 PNG → `folder_paths.get_save_image_path` → `PIL.save(pnginfo=...)` 存 output/（Save Output 按钮）
  - `/api/sfnodes/preview/prepare`：嵌入元数据后返回 data URI + 自增建议文件名（Save Disk 按钮）
- **改动路由后必须重启容器**，否则前端 fetch 404 静默降级（表现为按钮报 Save failed）。
- 前端 URL 一律 `api.apiURL()` 构建（托管部署基址前缀），失败降级原样返回。

### 5. 复用与差异（相对 SFPauseText）

- **prune 完全复用 `sf_pause_text_lib.js::applyGateMode`**（单一实现）：PauseImage 传 `{inputKey: "image"}`；`editedText` 参数对图片无意义（传 ""，注入的 PauseState 带空 text 键，后端不读、无害）。PauseText 版就是由 PauseImage 版改的，逐字一致——两节点共用同一 prune 是本次的架构决策。
- 其余同构：双钩子（graphToPrompt 只注入 {mode} / queuePrompt 剪枝）、`MODE_RANK` continue 先排序、一次性提交模式 finally 清除、executed 回填、`findNode` 子图 id 兜底、解析不到节点默认 pass。
- 差异：gate 只有 pause/pass 两态（无 keep——图片没有"批量复用"语义）；collectGates 注入只带 mode；state 形状最小（{gate, frame}，`hasSnapshot` 运行时推导绝不住 properties）。

### 6. 测试方法论（延续冒烟 + 新增快照/路由层）

- **mock torch 的 MockTensor**：需支持 `image[0]`（返回 cpu().numpy() 帧壳）与 `[None, ...]`（加 batch 维）两种下标；numpy/PIL 本机真实可用（torch/folder_paths mock，`get_temp_directory` 指向临时目录）。
- **快照 round-trip 断言**：pause 写入 → 检查文件存在 → continue 读回 → `out.numpy()` 与输入逐元素相等；无快照/损坏快照（写入垃圾字节）→ RuntimeError 中文消息；只读目录（chmod 0o500）→ 保存降级仍透传。
- **PngInfo round-trip 验证**：`_build_pnginfo` 后 `Image.save(pnginfo=)` → 重新 `Image.open` 读 `img.text` 断言 tEXt 块内容（含 NaN 已清洗）。**PIL 12.x 的 PngInfo 无 `_text` 属性**（结构变化），探测内部字段会误导——用保存-读回断言，别摸内部。
- **测试文件结构坑**：追加用例块时别放到 `sys.exit` 之后（不可达）——先验证追加位置再跑。

### 7. 模块边界（复用/修改时的快速索引）

- `nodes/image/pause_image.py`：节点（快照/continue 读回/无 IS_CHANGED）+ `_json_safe`。
- `nodes/image/preview_routes.py`：save/prepare 路由 + `_safe_prefix`/`_sanitize_segment`/`_decode_image`/`_build_pnginfo`/`_metadata_disabled`。
- `web/sf_pause_image_lib.js`：state（{gate, frame}，30 行）。
- `web/sf_pause_image_ui.js`：DOM widget（预览/按钮行/尺寸行）+ `frameViewUrl`（/view + 缓存戳）。
- `web/sf_pause_image.js`：主扩展（双钩子/Save 链路/Copy/Open/executed）。
- prune 共享：`web/sf_pause_text_lib.js::applyGateMode`（两闸门共用，勿复制）。

---

## 9. SFPauseMask：遮罩快照闸门（Pixaroma Pause Mask 同构扩展）

> 背景：复刻 Pixaroma 的 Pause Mask 变体（2026-08），落地为 `nodes/mask/pause_mask.py` + `web/sf_pause_mask*.js` 三模块（lib/ui/主扩展）。与 SFPauseImage 完全同构（快照/剪枝/一次性模式/executed 回填机制全部复用），仅把输入类型换成 MASK 张量 `[B, H, W]`（ComfyUI 遮罩格式）。本节点是"类型化闸门复用"的最小改造成案例——架构决策：**只加一个类型参数，绝不复制三份**。

### 1. 与 SFPauseImage 的差异（都是类型相关的）

- **快照为单通道灰度 PNG（L 模式，0-255 量化）**：遮罩通常二值/低精度，8bit 足够；与 ComfyUI 自身把遮罩存灰度 PNG 的惯例一致。`_mask_to_pil` 先 `(arr * 255).clip(0, 255).astype(uint8)` 再 `Image.fromarray(arr, mode="L")`——**L 模式只接受 2D 数组**。
- **tensor 转换防御非标准 `[1,H,W]`**：部分节点输出的 MASK 带单例通道维（`arr.ndim == 3 and arr.shape[0] == 1` → `arr[0]` 压平）——标准帧是 `[H,W]`，不防御会因 3D 数组直接炸。
- **读回对齐**：`_pil_to_mask` 用 `torch.from_numpy(arr)[None, ...]` 补 batch 维回 `1xHxW`，与 ComfyUI 遮罩张量格式一致。
- **快照前缀 `sf_pause_mask_`**：与图片闸门的 `sf_pause_` 隔离命名空间（同 node_id 不撞文件；语义也清晰）。
- **frame 键/state 键**：executed 回填帧键 `sf_pause_mask_frame`；状态存 `node.properties.pauseMaskState`（{gate, frame}）。lib 与 sf_pause_image_lib.js 是 30 行平行实现，仅 STATE_PROP 键名不同。

### 2. 剪枝共享（三闸门同一份实现）

- **prune 全走 `sf_pause_text_lib.js::applyGateMode(out, id, entry, mode, isOutput, HIDDEN_INPUT, opts)`**：PauseMask 传 `{inputKey: "mask"}`（PauseText 省略、PauseImage 传 `"image"`）。`inputKey` 是唯一分叉点——删哪个输入键/注入什么。
- 改 prune 语义只改 `sf_pause_text_lib.js` 一处，三节点同步生效（2026-08 复查时确认 PauseText 版与 PauseImage 版逐字一致，本次把差异收敛成参数）。
- 其余同构机制全部复用：双钩子（graphToPrompt 只注入 {mode} / queuePrompt 剪枝）、gate 两态（pause/pass，无 keep）、一次性提交模式、executed 回填、Save 链路（`/api/sfnodes/preview/{save,prepare}`）、无 IS_CHANGED。

### 3. 测试与方法论

- `tests/test_pause_mask.py`（后端）+ `test_pause_mask_js.js`（state/prune 纯函数）+ `test_pause_mask_smoke.js`（主扩展冒烟）——镜像 PauseImage 三件套。
- 冒烟断言同款结构：注入 → 剪枝 → executed 回填端到端；快照 round-trip 逐元素相等；非标准 `[1,H,W]` 帧转换防御断言。

### 4. 模块边界（复用/修改时的快速索引）

- `nodes/mask/pause_mask.py`：节点（快照 L 模式/读回 [1,H,W] 防御/无 IS_CHANGED）+ `_json_safe`。
- `web/sf_pause_mask_lib.js`：state（{gate, frame}，平行实现仅键名不同）。
- `web/sf_pause_mask_ui.js`：DOM widget（遮罩灰度预览/按钮行）+ frameViewUrl。
- `web/sf_pause_mask.js`：主扩展（双钩子/Save/Copy/Open/executed）。
- prune 共享：`web/sf_pause_text_lib.js::applyGateMode`（**三闸门共用**，勿复制）。

---

## 10. SF Workflows：工作流面板（复刻 Pixaroma Workflows）

> 背景：复刻 Pixaroma Workflows 的浮动工作流管理面板（2026-08，三期落地），`web/sf_workflows*.js` 三模块（主扩展/纯函数/DOM）+ `nodes/workflow_routes.py`（后端路由）+ `sf_utils/workflow_index_helpers.py`（索引纯逻辑）。与前面所有节点不同：**这是项目第一个"无节点"功能**——前端应用，后端只提供 API，不注册任何节点类。

### 1. 无节点设计（做"面板类功能"必知）

- **刻意没有节点**：节点会被存进工作流文件，分享工作流会把一个多余节点带给每个打开的人。面板属于应用（像帮助对话框），打开方式是**工具栏按钮 + 热键 + canvas 右键菜单**，绝不走节点菜单。
- 分层（跨文件 import 契约即模块边界）：主扩展 = 唯一触碰服务端与 ComfyUI store 的代码（涉及让人丢工作的调用集中可通读）；`sf_workflows_lib.js` 纯函数（cleanName/文件夹顺序/搜索评分，无 app/DOM 可 copy .mjs 直测）；`sf_workflows_ui.js` DOM（窗口/菜单/封面/网格/文件夹/CSS）。

### 2. 热键撞车（做"注册全局快捷键"必知）

- 原版 Pixaroma Workflows 占用 `Alt+W`。ComfyUI 前端对按键注册**全局去重**——同时装的插件注册同 combo 直接抛 `Keybinding on Alt + w already exists`（前端报错、面板打不开）。
- 本项目改 `Alt+Shift+W`（ComfyUI 允许两插件同用单键，但同 combo 冲突是真实报错）。**复刻任何有热键的插件，先查原版 combo，避免同键**。

### 3. 后端分层（索引与路由分离，做"文件列表类 API"必知）

- `workflow_index_helpers.py` **纯逻辑无 ComfyUI 依赖**（可独立测试）：按 **mtime+size 增量解析**（只重解析变化的文件，二次打开零重读）、24MB 文件上限（超限跳过，防止大文件阻塞请求）、封面映射 60 框 cap（>60 个节点矩形后不可读，payload 只增无益）、搜索文本 2KB cap、条目形状变化递增 version 丢弃旧缓存。
- `workflow_routes.py` 五资源路径 7 handler（前缀 `/api/sfnodes/workflows/`）：
  - `/index` GET 一次返回浏览器绘制自身所需的全部（entries/folders/collections/issues）——浏览器从不自行 fetch 文件；
  - `/meta` GET 自愈（迁移旧内嵌封面/遗忘已消失图片的封面）+ POST 按键合并（notes/covers/folderColors/folderOrder/folderExpanded）；
  - `/folder` 创建/改名/删除（**工作流文件本身永不在此触碰**——走 ComfyUI 自己的 store）；
  - `/reveal` OS 文件管理器打开文件夹；`/cover` POST 存真实 jpg + GET 取图。
- **meta 读写必须 asyncio.Lock**：每次写入都是小文件读-改-写，无锁时一次文件夹重排与一次笔记自动保存同时落地会各自还原对方分区（"两个面板互擦"，合并本身跨请求解决不了）。
- sidecar 三件套（user/default/ 下，bind mount 容器重建存活）：`sf_workflows_meta.json`、`sf_workflows_cache.json`（索引缓存）、`sf_covers/`（手选封面以**真实 jpg 文件**保存，sidecar 只存文件名）。

### 4. 收藏走 pinia（做"Vue 新版书签交互"必知）

- ComfyUI 启动时**不读收藏文件**——书签 store 直到有人调 `loadBookmarks()` 才加载。toggle 收藏前必须先 `await bm.loadBookmarks()`，否则覆盖空列表（收藏全部丢失）。
- 收藏键用工作流相对路径（`toStorePath(rel)` 转换），与 ComfyUI 自己的收藏入口共用一套数据。
- 旧版前端（litegraph）无书签 store → `typeof bm?.loadBookmarks !== "function"` 判存在，收藏入口隐藏。

### 5. 状态与视觉（设置持久化/滚动/密度）

- 设置键 `sfnodes.Workflows.{Rect,View,Sort,Density}` 存 comfy.settings.json（未注册设置，读写走 `app.ui.settings`）；ui 模块经 `window.sfnodesGetSetting/SetSetting` 桥调用主扩展注入的实现。
- **滚动容器 `overflow-y: auto` 放持久 `main`**（不随面板重建）：面板重建不重置滚动位置（曾把滚动放面板内层容器，重建即滚回顶部）。
- **密度系统 `z(n) = calc(npx * var(--sfwb-k, 1))`**：所有能被感知为"大或小"的视觉尺寸经 z() 乘以 CSS 变量 `--sfwb-k`（s/m/l 三档按钮即时生效）；**窗口像素尺寸刻意不缩放**（拖拽数学保持自洽），且不用 CSS zoom（会破坏 fixed 定位子层）。运行时注入 CSS + `--sfwb-k` 数值。
- 加载带票号 guard：两次加载重叠（打开面板一次、任何动作经 guard 校验）防并发破坏。

### 6. 测试与方法论

- 三件套：`test_workflows.py`（后端 helpers 独立测试——无 ComfyUI 依赖直接跑）、`test_workflows_js.js`（lib 纯函数）、`test_workflows_smoke.js`（主扩展冒烟，mock DOM + app）。
- 冒烟含 **CSS 注入断言**：运行时注入的样式表存在、`--sfwb-k` 已设、无 `${z(` 字面量残留（z() 全部展开为 calc）——抓"尺寸漏走密度系统"的回归。
- 主扩展冒烟 mock：`app.ui.settings`、`api.fetchApi`、`app.graph`、菜单注册、pinia 书签 store（loadBookmarks/toggleBookmarked）。

### 7. 模块边界（复用/修改时的快速索引）

- `web/sf_workflows.js`：主扩展（api 层/收藏/热键/工具栏按钮/CMD_ID `sfnodes.OpenWorkflowBrowser`）。
- `web/sf_workflows_lib.js`：纯函数（cleanName/nameProblem/orderedFolders/siblingsOf/searchEntries）。
- `web/sf_workflows_ui.js`：DOM（窗口/网格/文件夹/详情/tidy/封面捕获/右键菜单/CSS `z()` 系统）。
- `nodes/workflow_routes.py`：五资源路径 7 handler + meta asyncio.Lock + sidecar 读写。
- `sf_utils/workflow_index_helpers.py`：索引/集合/问题检测纯逻辑（增量解析、cap 常量）。
- 注册：`__init__.py` 导入副作用注册路由（`# noqa: F401`）；无 NODE_CLASS_MAPPINGS 条目。
