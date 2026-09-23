# 经验归档：平台机制（ComfyUI 前后端通用）

> 全局章节号 §N 唯一、只增不复用；跨文件引用写「文件名 §N」（同文件内可简写 §N），映射与当前最大 §N 见 [README.md](README.md)。版本时效说明见 README。

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
- **LoopEnd 悬空输出的历史坑（2026-09 已修复）**：`ExecutionList`（TopologicalSort）只调度被下游引用的节点——`add_node` 从输出节点回溯依赖链入队，**死端节点（输出无下游）从不执行**。LoopEnd 原本非 OUTPUT_NODE，输出悬空时从不执行 → 不触发 expand → 整个循环静默不跑（2026-08 实测：删除循环外 PreviewImage 后循环完全不启动）。修复：`SFForLoopEnd`/`SFWhileLoopEnd` 声明 `OUTPUT_NODE = True` 成为调度根，输出未接下游也照常执行展开（官方先例 `TestParallelSleep`："OUTPUT_NODE=True + expand 返回" 组合合法；输出节点同样走缓存，不会强制重跑）。
- **LoopEnd 带 OUTPUT_NODE 后的衍生坑**：`SFWhileLoopEnd.while_loop_close` 收集 OUTPUT_NODE 节点并入循环体（保证体内 SaveImage 每轮重跑）时，**必须跳过 `SFForLoopEnd`**（`_collect_output_nodes` 内 class_type 过滤）——否则 ForLoopEnd 因带链接输入被误当"循环体内输出消费者"加入 `contained` 集，每轮迭代重建出其克隆并再次 expand → 嵌套错误展开。WhileLoopEnd 自身被收集则天然无害：`explore_dependencies` 已把它排除出 `parent_ids`（永不匹配），且 `upstream[parent_id]` 守卫挡住重复追加。
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

**部署注意**：用户运行实例为 docker 部署（以实际挂载路径为准，与本地仓库内容一致），修改 `web/` 下 JS 后需**同步该目录**，且浏览器需**硬刷新**（Ctrl+Shift+R）才生效；后端改动需重启容器。

### 6. 文本 widget 自定义右键菜单（做"右键插入"类功能必知）

> 背景：`SFTextReplace` 模板框右键插入特殊标记符（2026-08）。踩坑过程：先依赖 `widget.options.contextMenu` → 弹出系统菜单；修好菜单后插入点又落文本末尾 → 最终方案见 `web/text_replace.js` 的 `showMarkerMenu`/`caretOffsetAt`。

- **`widget.options.contextMenu` 机制不可靠**：它仅在 canvas 绘制态（widget 未编辑、textarea 隐藏）走 LiteGraph 路径；文本 widget 一旦被点击过，其 DOM `inputEl`（textarea）显示并覆盖 widget，右键即触发**浏览器原生菜单**，与 `options.contextMenu` 无关。可靠做法：直接给 `widget.inputEl` 挂 `contextmenu` 监听（`preventDefault` + `stopPropagation`）弹**自绘 DOM 菜单**（fixed 定位、z-index 顶层、视口 clamp），零依赖 ComfyUI 内部 API；`options.contextMenu` 可保留作 canvas 态兜底（textarea 隐藏时其 DOM 监听不触发，两路径互斥）。
- **浏览器右键不更新文本光标**：右键点击 textarea 不移动 `selectionStart`（保留上次位置）也不聚焦。做"插入到鼠标处"必须显式计算 offset：用 `document.caretPositionFromPoint(x, y)` / `document.caretRangeFromPoint(x, y)` 把鼠标坐标换算为字符偏移，存入 `pendingInsertPos` 变量，插入时优先使用（**避免在 contextmenu 里调 `focus()`**，会干扰 ComfyUI widget 焦点状态）。
- **caret 两 API 返回结构不同（易踩坑）**：`caretRangeFromPoint` 返回 `Range`（`startContainer`/`startOffset`），`caretPositionFromPoint` 返回 `CaretPosition`（`offsetNode`/`offset`）。统一读 `startContainer` 会得到 undefined → 静默回退末尾。正确写法：两个 API 都尝试、两种属性名都兼容。
- **插入位置三级回退**：显式记录的鼠标 offset → `activeElement` 的 `selectionStart`（含选区替换）→ 追加末尾。
- **菜单关闭策略**：document 捕获阶段 `mousedown` 且 `!menuEl.contains(e.target)`、Escape、滚轮滚动时关闭；菜单项用 `click`（click 晚于 mousedown，捕获阶段判断不误关菜单项）。

### 7. 动态槽位机制（做"多输入/输出节点"必知）

> 背景：`web/sf_dynamic_slots.js` 公共库（2026-08），将循环节点、Text/Image Concatenate、SimpleMath、LogicSwitch 等 6 个文件的动态槽位逻辑统一为配置化实现（`installDynamicSlots(node, config)`），本机用 FakeNode + 事件序列模拟测试（31 项断言）。

- **四种动态槽位模式**（按复杂度）：A. 连线自动增删（前缀匹配，公共库覆盖）；B. 全动态+自动命名/右键重命名/名称传播（`any_pack.js`，特例）；C. 成组配对+自愈（`krea2_dynamic_images.js` 的 imageN/maskN，onNodeCreated/onConnectionsChange/onConfigure 三钩子）；D. 按钮 + widget 显隐 + 状态持久化（`text_replace.js`，`visibleSlotCount` 随 workflow 序列化）。
- **新节点优先用公共库**：只需配置 `inputPrefix/inputStart/inputCount/inputType/initialInputs` + 输出侧同构；非连续命名用 `inputMatch`（正则，如 simple_math 的 `/^[a-z]$/`），非编号命名用 `nameFor`（如字母表回调）。
- **optional 无 widget 输入默认全显示**：新版前端 `addInputSocket` 对 optional 槽位直接 `addInput`（无隐藏机制）→ 必须 JS 在 `nodeCreated` 时 trim 到初始数量。动态槽位名字必须与后端 `INPUT_TYPES` 完全一致。
- **旧 workflow 恢复依赖前端合并机制**：`nodeCreated` 时 trim（此时无连线），随后 `configure` 时 litegraphService 把保存快照中多出的槽位（extraInputs/extraOutputs）合并回来（源码注释明确支持"custom nodes that dynamically add inputs/outputs via js logic"）。**configure 直赋 links 不触发 `onConnectionsChange`**，恢复时不会连锁加槽。
- **输入/输出判空结构不同**：输入槽位 `.link`（断开为 null，旧版可能 -1）；输出槽位 `.links`（数组，断开为 null/[]）。公共库 `isSlotConnected` 两者兼容（含 `!== -1` 防御）。
- **增删规则**：全部动态槽已连 → 追加下一个（注意空数组 `every()` 恒真，需 `length > 0` 防御）；断开时从尾部 reverse 遍历、遇已连槽即停（只回收尾部连续空槽），保底 initial 个。
- **模拟测试经验**：`cp web/sf_dynamic_slots.js /tmp/xxx.mjs` 后 Node 直接跑（公共库无 DOM 依赖）；FakeNode 需实现 `addInput/removeInput/addOutput/removeOutput/computeSize/setSize`；**事件序列用槽位名定位索引**（动态增删后绝对索引会错位，这是测试脚本最常见的错误来源）；断开事件触发前先把 `link` 置 null。

### 8. ComfyUI 新版 Vue 前端机制（做"悬停提示/DOM 交互"必知）

> 背景：SFPromptPreset 预设说明展示两次翻车（2026-08）。先做 canvas mousemove + 固定 DOM 卡片 → 完全不生效；清除 `widget.tooltip` 抑制原生提示 → 依然显示。最终发现用户跑的是 ComfyUI 新版 Vue 前端（comfyui_frontend_package 1.47.10，当时版本，当前以 `pip show comfyui-frontend-package` 为准），旧 LiteGraph canvas 机制已废弃。最终方案：动态写入 `widget.tooltip`，见 `web/prompt_preset.js`。

- **先确认前端版本再选方案**：ComfyUI 前端自 2025 年起从仓库 `web/` 目录改为独立 pip 包 `comfyui-frontend-package`（新版 Vue 重构）。判断方法：容器内 `pip show comfyui-frontend-package`（Version 1.x = Vue 前端）；**后端版本号 ≠ 前端版本号**；仓库源码副本（`../..`）无前端代码，需查 `Comfy-Org/ComfyUI_frontend` GitHub 仓库或容器内 pip 包 static 目录。
- **Vue 前端下 canvas 事件/坐标方案全部失效**：widget 是 Vue 渲染的 DOM 元素（覆盖在 canvas 上方），`app.canvas` 的 mousemove 监听收不到悬停 widget 的事件（DOM 遮挡）；`node.pos + widget.pos/size` 几何命中同样失去意义。做"悬停 widget 显示信息"类功能**不要走 canvas 事件路线**。
- **tooltip 是 PrimeVue v-tooltip 指令（DOM `.p-tooltip`），不是 canvas 绘制**：来源链 `createTooltipConfig(getWidgetTooltip(widget))`，`getWidgetTooltip` **优先读 `widget.tooltip`，其次 nodeDef 输入定义（后端 `/object_info` 的 tooltip）** → 仅清 `widget.tooltip`/`widget.options.tooltip` 无法抑制原生提示（nodeDef 兜底还在，JS 改不掉后端数据）。
- **动态 tooltip 的正确姿势（通用做法）**：把"当前选中值对应的说明"直接写入 `widget.tooltip`（callback 里随值更新；工作流恢复场景在数据就绪后遍历全图节点同步一次）。新旧前端均优先显示 widget.tooltip：旧版 canvas 绘制 tooltip 读 widget.tooltip，新版 Vue 前端 processedWidgets 为 computed（值变化 → v-tooltip 指令更新）。**后端 INPUT_TYPES 的 `"tooltip"` 键只适合静态提示；逐选项动态说明必须 JS 写 `widget.tooltip` + 前端拉数据**。
- **前端拿后端数据**：注册 `GET /api/sfnodes/...` 路由（server.PromptServer.instance.routes），前端 `api.fetchApi()` 拉取；路由在模块导入时注册，**改动后必须重启容器**，否则 404 且前端静默降级（表现为"功能不生效但无报错"）。
- **模拟测试**：`new Function("app", "api", code)`（去 import 行）注入 mock app/api，Node 直接跑；断言 callback 链（互斥/说明同步）与 `widget.tooltip` 赋值，无需真实 DOM。

### 9. 实际环境调试方式（console 诊断脚本，用户配合执行）

> ⚠️ 部分更新（2026-09）：后端/API 层实机调试已可直接进行，见 §106；本节"一律 console 脚本"仅前端 UI 层仍适用，浏览器访问禁令依旧有效。

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

### 11. 画布多选尺寸对齐（做"画布背景右键多选操作"必知）

> 背景：`SF Align` 画布多选尺寸对齐（2026-08，`web/sf_canvas_align*.js`）：系统右键已有对齐/分布，无统一尺寸；需求为选中 ≥2 节点后对齐宽度（Widest/Narrowest/Mouse Node）、高度（Tallest/Shortest/Mouse Node）与等大（Widest& Tallest / Narrowest&Shortest / Mouse Node 同时改两维）。原 `First Selected`（选中集首项）因点选顺序不可靠已删除，改为鼠标所在节点，见 §124。

- **入口是 `getCanvasMenuItems`，不是 `getExtraMenuOptions`**：后者是节点右键（单节点），前者是画布背景右键；多选操作必须走画布菜单。`sf_workflows` / `sf_lora_browser` 同款入口，无需 monkey-patch `LGraphCanvas.prototype.getCanvasMenuOptions`（随前端版本易碎）。
- **选中集四形态兼容**：`app.canvas.selected_nodes` 在不同 ComfyUI/前端版本为 `Object` / `Array` / `Map` / `Set` 四种之一；`Map values()` / `Set` 需 `[...sel.values()]` / `[...sel]`，`Object` 走 `Object.values`。为空时回退扫描 `graph._nodes` 上 `is_selected` / `flags.is_selected`（Vue 模式的框选标记）。**单选时不回退扫描**，避免把悬停误判为多选。`<2` 节点不注入菜单，保持右键纯净。
- **菜单子菜单键名**：LiteGraph `ContextMenu` 认 `has_submenu: true` + `submenu: { options: [...] }`；ComfyUI 前端透传该结构，Classic/Vue 双兼容。仅改对象键名即可修复"子菜单不展开"，零逻辑风险。
- **尺寸语义**：读 `size[0]/size[1]` 优先，回退 `computeSize()[0/1]`（节点新建时 size 可能未落盘）；写时 `Math.max(target, computeSize()[d])` 钳制到最小尺寸，避免缩到不可渲染；单维对齐保持另一维不变，等大 `alignNodesSize` 同时写两维（分别钳制）。`beforeChange/afterChange + setDirtyCanvas(true,true)` 保证撤销与重绘。
- **纯逻辑边界**：`sf_canvas_align_lib.js` 无 `app` 依赖，六函数 `getSelectedNodes / calcTargetWidth / calcTargetHeight / alignNodesWidth / alignNodesHeight / alignNodesSize` 可拷 `.mjs` 直测；`sf_canvas_align.js` 仅做接线（`import {app} from "/scripts/app.js"`，满足 `check_web_imports.py` B2/B3）。
- **菜单压平（2026-09）**：用户嫌 `SF Menu ▶ SF Align ▶ Width/Height/Size ▶ 动作` 四跳太深，`SF Align` 子菜单内改为单层平铺（3 个 `disabled` 分组头 + 动作，标签带组前缀 `Width:/Height:/Size:` 保唯一；分组头无 callback，点击无操作 fail-safe）。§15 的三级嵌套担忧随之消除。动作数随入口变化：画布入口 6 项，节点入口另加 3 项 Mouse Node（§124）。

### 12. Vue 新版 LLink 字段差异与通用 combo 选择器（做"连接感知/选项同步"类功能必知）

> 背景：SFComboSelector 通用下拉选择器（2026-08，前端 1.48.6，当时版本）：输出连到目标节点 combo 输入（Convert to input 后）→ 下拉选项自动同步为目标选项列表。踩坑链：连线后选项不动 → 事件没触发? → 数据层取不到列表 → 目标节点解析失败。

- **坑 1（根因）：Vue 新版 LLink 字段名变了**。旧版 `link.target_node`/`origin_node`，新版为 **`target_id`/`origin_id`**（`target_slot`/`origin_slot` 未变），且**节点 id 为字符串**。按旧字段取 → `undefined` → 目标解析失败 → 选项同步静默失效（无任何报错）。取目标必须 `link.target_node ?? link.target_id`，节点查找用 `String(n.id) === String(id)` 比较。
- **坑 2：combo 输入槽 `slot.type` 在新版是字符串 `"COMBO"`**（不是旧版的选项数组/逗号串）→ 从槽类型取不到列表。**Convert to input 后原 combo widget 仍保留在 `node.widgets`（含动态重建的 `options.values`）**——这是动态选项（如 SFPromptPreset 的 441 项）唯一可靠来源，nodeDef 兜底只有静态列表。三级兜底：槽类型（数组/JSON/逗号串归一化）→ 同名残留 widget 的 `options.values` → nodeDef。
- **坑 3：连接事件触发时 `outputs[0].links` 可能尚未更新** → `onConnectionsChange` 里 `setTimeout(syncOptions, 0)` 延迟执行；工作流加载恢复连接不触发该事件 → 挂 `onAfterGraphConfigured`/`onGraphConfigured` 补同步。
- **坑 4：combo widget 是 DOMWidget（ComboWidget，带 `element`）** → 更新选项需整体替换 `widget.options` 对象 + 重赋 `values` 数组引用（Vue 渲染监听引用变化）并 `setDirtyCanvas`；断线/无连接恢复占位 `[""]`。
- **通用输出类型**：目标不可预测时用 `RETURN_TYPES = (AnyType("*"), ...)`（项目 `sf_utils/common.py`）——后端 `validation.py` 与前端 `isValidConnection` 对 `*` 均直接放行，可连任意 combo 输入；动态选项节点标配 `VALIDATE_INPUTS → True`（见 §4.2）。ComfyUI 官方生态同类参考：`ControlNetPreprocessorSelector`（输出类型 = 具体 combo 列表，`isValidConnection` 对数组按元素逐项匹配，任一共有即可连）。
- **诊断**：node 上暴露 `_sfComboSync`/`_sfComboGetLinks`/`_sfComboFindTarget` 调试接口，console 分段脚本直接调用定位（见 §9）。

### 13. 槽点与槽名文字基线错位（前端包全局行为，勿在本节点修）

> 背景：SFImageCropExpand 用户反馈输出槽文字相对槽点下偏（2026-09）。前端包 1.51.10 实测（当时版本，升级后以实测为准）`NodeSlot.draw`（settingStore chunk）。

- 槽圆点画在 `boundingRect` 中心 `u[1]`，槽名文字**统一硬编码** `fillText(label, u[0]±10, u[1]+5)`（输出槽 textAlign=right，输入槽 left），且 draw 内未设 `textBaseline`（继承 alphabetic）→ 文字视觉中心低于圆点约 2-5px。
- **所有 canvas 渲染节点的输入/输出槽一致如此**（原生节点同款）——节点侧无钩子可修，hack 槽渲染会与其它节点不一致；归因时先对照其它节点确认全局性，勿误判为本节点绘制问题。

### 14. SF Memory 画布菜单清理 VRAM/RAM（做"免节点手动释放"必知）

> 背景：画布背景右键 `SF Memory ▶ Free VRAM / Free RAM`（2026-09，`web/sf_memory_menu.js` + `nodes/utils/memory_routes.py`，`tests/test_memory_menu.py`）：已有 `SFVRAMCleanup/SFRAMCleanup` 节点只能在队列中运行，用户要"不跑工作流点一下就释放"。

- **VRAM 走原生 `POST /free`**（`server.py:1192`，`{unload_models:true, free_memory:true}` → 队列 flag → `main.py:425` 有序执行 `unload_all_models + soft_empty_cache`），与官方释放语义一致——**不要自建 VRAM 路由**（直接调 `unload_all_models` 会与运行中任务竞态，原生走队列 flag 更安全，且随上游维护）。
- **RAM 必须自建路由**：浏览器 JS 够不着服务端进程内存。`POST /api/sfnodes/memory/ram` 实例化复用 `memory_cleanup.RAMCleanup`（`clean_ram(True,True,True,1)`，`retry_times=1`——菜单点击是交互操作，不按节点默认 3 次 `sleep` 等待），返回 `{ok, before_usage, after_usage, freed_mb}` 供 toast 显示。约 1s 阻塞事件循环（gc + sleep），可接受。
- **菜单形态**：`getCanvasMenuItems` 无选中门槛（§11 的 `≥2 节点`守卫是"多选操作"专用，内存清理是全局动作，`sf_lora_browser` 同款无条件返回）；子菜单 `has_submenu + submenu.options`（§11 已验证双兼容）；反馈走 `sf_common.sfToast`。

### 15. 画布菜单聚合（📦 SF Menu 唯一顶层入口，做"新增画布右键项"必知）

> 背景：包内 4 处画布背景菜单（对齐/浏览器/工作流/内存）曾各占顶层入口（2026-09），收敛为 `web/sf_canvas_menu.js` 唯一 `📦 SF Menu`（emoji 前缀视觉分组），`tests/test_canvas_menu_js.js` 锁定。现行子项（2026-09 扩充，§125）：`SF Align ▶`（<2 选中时退化为 disabled 提示行）/ `SF Node Color…` / `SF LoRA Browser` / `SF LoRA Presets` / `SF Workflows` / `SF Memory ▶` / `Add SF Note`——上下文相关项（Align/Node Color）在前，全局工具随后。

- **聚合器只组装、零逻辑**：各特性删自己的 `getCanvasMenuItems`，改 export 动作（对齐 `buildAlignMenuItems` / 内存 `buildMemoryMenuItem` / 工作流 `openWorkflowsPanel` / 浏览器 `openLoraBrowser` / 预设管理 `openLoraPresetManager` / 便签 `addNoteFromMenu`），`commands`/热键/工具栏不动。export 必须用 `export function/const` 前缀式——`export {}` 花括号式会让 Function-eval 系测试（`test_note_js.js` 同款 strip 手法）报 SyntaxError。
- **门槛语义保留在构建器侧**：对齐 `<2 节点返回 []`，聚合器改放 disabled 提示行 `SF Align (select ≥2 nodes)`（无 callback，fail-safe）；≥2 才包一层 `SF Align` 嵌套，内为单层平铺（动作 + disabled 分组头，见 §11 压平备注）。
- **测试**：`.mjs` 拷贝链真实加载（lora 冒烟同款）断言唯一入口 + 门槛 + 排序 + 逐项驱动（便签落图/Presets 面板挂载/对齐同宽/VRAM 调 `/free`/RAM toast）；旧分散断言改为"已移交聚合器"回归（`ext.getCanvasMenuItems === undefined`）。
- **节点右键入口（2026-09）**：聚合器同时注册 `getNodeMenuItems(node)` 返回同一 `📦 SF Menu`（组装抽为 `buildSfMenuOptions` 双钩子共用）；前端建节点菜单前已选中右键节点，选中集门槛语义不变。机制与实证见 §123。

### 16. 主题令牌层：sfnodes DOM UI 跟随 ComfyUI Color Palette（做"自定义弹层/面板适配明暗"必知）

> 背景：包内 DOM 弹层/node widget（LoRA Stack 家族、浏览器、加载/尺寸、提示词/@tag/查找替换、选择器/图库、闸门暂停、裁剪编辑器框架、工作流/预设面板等**全部 DOM UI**）长期硬编码深色（`#1a1a1a`/`#161616`/`rgba(255,255,255,.05)`），在 ComfyUI 亮色/自定义主题下始终黑底（2026-09）。`web/sf_common.js` 顶层注入 id `sf-theme-vars` 的 `SF_THEME_CSS`，各模块改用 `var(--sf-*)`；canvas 节点体/画布工具色不属此列（保留）。

- **ComfyUI 主题机制（前端包 1.52.7 实证，升级后以容器实测为准）**：选中调色板的 `comfy_base` 由 `loadComfyColorPalette` 逐键写成 `<html>` **内联 CSS 变量**（`--comfy-menu-bg / --comfy-menu-secondary-bg / --comfy-input-bg / --fg-color / --input-text / --descrip-text / --border-color / --content-* / --tr-*-bg-color`），含自定义主题、运行时切换即时生效；另有 `.dark-theme` 类驱动的新设计令牌（`--interface-*`）但只区分亮/暗。**跟随面板走旧 `--comfy-*` 全集最稳**。
- **令牌层单源**：`sf_common.js` 顶层 `injectCSSOnce("sf-theme-vars", SF_THEME_CSS)` 定义 `--sf-panel-bg / --sf-panel-bg-2 / --sf-input-bg / --sf-text / --sf-text-strong / --sf-text-dim / --sf-text-faint / --sf-border / --sf-border-soft / --sf-surface / --sf-surface-hover / --sf-positive(-soft) / --sf-negative(-soft)`，映射 `--comfy-*` 并带兜底。正向/负向提示词色（原写死 `#9ecfa8/#7bd88f/#e8928a` 等，亮色主题下浅色文字不可读）用 `color-mix(in srgb, 基色 62%, var(--fg-color))`——深色主题混亮、亮色主题混深，两端可读。只改 `sf_common.js` 一处即可全局调色。
- **半透明面用 `color-mix(in srgb, var(--fg-color) N%, transparent)`**：深色主题 `--fg-color` 浅 → 白蒙层，亮色主题 → 黑蒙层，一套定义两端自适；替代所有 `rgba(255,255,255,.05/.12/.14)`。`color-mix` 前端早已在 LoRA Stack 使用，无兼容风险。
- **保留不动的颜色**：强调色 `--sf-acc`、状态色（红/绿/蓝/警告橙）、黑色遮罩 `rgba(0,0,0,.55)`、缩略图占位渐变、canvas 节点体配色（布尔开关/万能滑条/便签是节点自身设计色，非弹层主题）。
- **canvas 取色**：DOM 的 CSS `var()` 自动响应主题；canvas 绘制用 `sf_common.sfThemeColors()`（读旧 `--comfy-*` **实色**返回 `{panel,panel2,surface,border,text,textStrong,textDim,light}`——`--sf-*` 令牌多为 var()/color-mix() 字符串，`ctx.fillStyle` 解析不了）。sf_crop_expand 的画布按钮/底条/扩展区改用它（明暗两端一致）；需要读单个 CSS 变量时 `getComputedStyle(document.documentElement).getPropertyValue("--sf-panel-bg")`（自定义属性 computed 阶段已替换 `var()`，如 sf_lora_stack_info 的 JPEG 透明底合成）。
- **Node 冒烟测试守卫**：`tests/test_*.js` 会把 `sf_common.js` 拷成 `.mjs` 裸跑，顶层注入无 `document` 会崩——调用必须包 `if (typeof document !== "undefined")`（纯逻辑模块不得 import sf_common 的边界依旧）。

### 17. 任意节点颜色（node.color/bgcolor 机制与前端调色板限制）

> 背景：ComfyUI 原生只提供固定调色板（`LGraphCanvas.node_colors`，9 项 red/brown/green/blue/pale_blue/cyan/purple/yellow/black，各 `{color,bgcolor,groupcolor}`），节点右键 Colors 子菜单、右侧属性面板 `SetNodeColor`、选中工具条 `ColorPickerButton` 都只列这些名字，**无任意色入口**（前端包 1.51/1.52 实证）。网上第三方（`comfyui-custom-node-color` / HouseKeeper / MurMur）自行弹取色器补这个缺口。本包按需实现为画布聚合菜单动作（`web/sf_node_color.js` + `sf_node_color_lib.js`，SF Node Color…）。

- **机制**：节点标题/节点体渲染直接读 `node.color` / `node.bgcolor`（`renderingColor`/`renderingBgColor` getter = `node.color || constructor.color || NODE_DEFAULT_COLOR`），且二者在序列化白名单内（`[...,color,bgcolor,...]`）→ **赋任意 hex 即生效并随工作流保存**，无需后端。`node.setColorOption({color,bgcolor})` 接受任意值（只是 `getColorOption()` 反查调色板会返回 null，原生选择器显示 No Color 属正常）；清除走 `setColorOption(null)` 等价原生「No Color」。
- **取色 UI**：原生 `<input type="color">`（浏览器级取色器，仅 6 位 hex）+ hex 文本框（接受 `#RGB`/`#RRGGBB`/无 `#`，`normalizeHexColor` 归一）+ `localStorage` 最近色（机器私有）。撤销用 `graph.beforeChange()/afterChange()`（`sf_canvas_align` 先例），完成后 `canvas.setDirty(true,true)`。
- **对比度限制（未处理）**：标题文字色取自 `constructor.title_text_color || canvas.node_title_color`（默认 `#999`，选中 `#FFF`，前端包全局行为），**不随 `node.color` 变** → 选浅色标题时文字可能偏淡。第三方 HouseKeeper 另按亮度自动切标题/正文黑白字；本包暂不做（改动最小、与原生调色板行为一致）。
- **入口形态**：`📦 SF Menu ▶ SF Node Color…`（仅 ≥1 选中节点时注入，`buildNodeColorMenuItem` 无选中返回 null）。旧限制「画布菜单只在空白处右键出现、须先选中再空白右键」已被 §123 取代（2026-09 起节点右键也有 📦 SF Menu，且前端建菜单前已选中该节点）；原路径（先选中、空白处右键）依旧可用。
- **测试**：`tests/test_node_color_lib.mjs`（纯逻辑：hex 归一/最近色 FIFO/apply/reset/read）+ `tests/test_node_color_js.js`（`.mjs` 拷贝链：菜单门槛/面板挂载/应用写色/撤销钩子/最近色持久化/清除/overlay 与 Esc 关闭）；`tests/test_canvas_menu_js.js` 增补菜单项门槛断言。

### 18. 节点运行时间显示（execution 事件计时 + Classic onDrawForeground / Vue node.badges 双路）

> 背景：需求「节点显示运行时间 + 设置开关默认关」，参考 `ComfyUI-Easy-Use` 的 `Comfy.EasyUse.TimeTaken`（`web_version/v2/assets/extensions-*.js`）。实现 `web/sf_node_runtime.js` + `sf_node_runtime_lib.js`。

- **计时来源**：`/scripts/api.js` 的 `execution_start` / `executing` 事件。Easy-Use 做法：记录「上一个 executing 的节点 id + 开始时间」，收到下一个 executing 时把 `Date.now() - 开始时间` 结算给上一个节点（墙钟近似，含调度开销）；`execution_start` 清空全部（每轮重新计时），节点多次执行则累加。缓存命中节点只发 `execution_cached`、无 executing → 无耗时（与 Easy-Use 一致）。
- **⚠ 事件载荷陷阱（首次实测不显示的根因）**：新版前端 `ComfyApi extends EventTarget`（`api-DclbNWWy.js`），socket 分发是 `case "executing": this.dispatchCustomEvent("executing", data.display_node || data.node)` —— **`event.detail` 直接就是节点 id（数字/字符串），整轮结束为 `null`，不是 `{node, display_node}` 对象**。若按对象解析（`detail.display_node ?? detail.node`）会全部得到 undefined → 永不结算、无显示。`resolveNodeId` 必须同时兼容 primitive（新）与对象（旧/第三方）。
- **显示双路**：Classic 渲染器用 `onDrawForeground` 逐节点类型补丁绘制（Easy-Use 原做法；项目内 sf_dropdown/sf_find_replace/sf_image_resize 已验证）；Vue Nodes 2.0 的 per-node `onDrawForeground` 不触发，改用原生 `node.badges`（`window.LGraphBadge` 实例 = 官方 `Comfy.NodeBadge` 同款），并以 `graph.trigger("node:property:changed",{property:"badges",...})` 触发刷新。颜色取 `LiteGraph.NODE_TITLE_COLOR/NODE_DEFAULT_BGCOLOR`（随调色板主题变）。
- **自管理边界**：Vue badge 实例打 `_sfRuntimeBadge` 标记，`findIndex` 只替换/删除自己的，不影响官方 NodeId/生命周期/API 价格等 badge。Classic 的绘制同样只读 `node.executionDuration`。
- **开关**：`sfnodes.NodeRuntime.Enabled`（boolean，**默认 false**，Easy-Use 默认 true）；`onChange(false)` 立即 `clearAll()` 清除耗时与 badge，开启后需下一轮执行才有数据。`execution_start` 无条件清空（即使关闭，避免残留）。
- **测试**：`tests/test_node_runtime_lib.mjs`（formatDuration/accumulateSeconds/resolveNodeId 双形态/badge 标记）+ `tests/test_node_runtime_js.js`（受控 `Date.now`；executing 以 primitive id 触发模拟真实前端，断言开关门槛、Classic executionDuration 结算 + onDrawForeground 绘制文案/折叠与关闭跳过、Vue badge 生成/重建、execution_start 清空、onChange(false) 清除）。

---

## 106. 容器内 API 实机调试（免浏览器，2026-09）

> 背景：运行实例为 docker `comfyui-docker`（compose 在宿主 `/mnt/github/comfyui-docker/docker-compose.yml`，端口 `8188:8188`），此前实机验证只允许"用户执行 console 脚本"（§9）。实测 agent 可经 HTTP API 与 `docker exec` 直接对后端实机诊断：宿主 `curl http://localhost:8188/...` 与容器内 `docker exec comfyui-docker curl -s http://127.0.0.1:8188/...` 均可达（8188 已发布到宿主）；`/object_info`、`/queue`、`/api/sfnodes/workflows/index` 实测均 200。UI/DOM 层仍走 §9。

- **可自行操作（后端/API 层）**：
  - `GET /object_info/{节点类名}`：验节点注册与 INPUT_TYPES / RETURN_TYPES / RETURN_NAMES / tooltip（`server.py::node_info()` 每次请求惰性读取类属性，见 patterns.md §79.2——改动重启后即见，无需前端 JS）；
  - `GET /queue`、`GET /history`（含 `/history/{prompt_id}`）、`GET /system_stats`：队列/历史/运行状态；
  - 自定义路由：`GET /api/sfnodes/...`（workflows/index、prompt_reader/list 等）直接验证；POST 写路由（workflows/meta、preview/flip、translate 等）仅用测试数据且可回滚，勿污染用户真实数据；
  - `docker logs comfyui-docker`：启动注册日志与 traceback（各路由注册成功行），排查"路由 404 / 节点未注册"的第一现场；
  - `docker exec comfyui-docker python3 -c "..."`：真实运行时环境一次性检查（容器 Python 3.12.x vs 宿主 3.14.x 行为差异见 patterns.md §27.1；`pip show comfyui-frontend-package` 查前端版本）。
- **执行级验证（POST /prompt）**：构造轻量测试工作流（少量纯计算/文本节点，**不加载大模型**）POST `/prompt`，再轮询 `GET /history/{prompt_id}` 取结果——等价于"请用户 UI 添加节点跑一遍"的后端替代；队列与用户任务共用，注意不抢占 GPU/长时间占用。
- **部署同步（重启须用户同意）**：宿主工作副本与容器挂载副本（当前 `/mnt/github/comfyui-docker/custom_nodes/sfnodes`，以实际挂载为准）是独立 git 副本，改动须显式同步后重启容器才在实机生效；**重启打断用户任务，先确认**。entrypoint 的 `.update` 机制会 `git reset --hard origin/HEAD` 清除部署副本全部未提交改动——手工同步不持久。
- **浏览器禁令仍有效**：agent 浏览器/浏览器自动化访问用户 ComfyUI 页面会干扰用户 tab 与工作流；UI/DOM/Vue 渲染、画布交互、widget 行为无法经 API 验证，仍按 §9 分段 console 脚本交用户执行（节点由用户 UI 添加，新版前端无 graph.createNode）。

---

## 121. legacy（Classic）模式 widget 宽度冻结（前端 1.53.6 bug 内置规避，2026-09）

> 背景：第三方包 [ComfyUI-LegacyWidgetWidthFix](https://github.com/pekkAi-dev/ComfyUI-LegacyWidgetWidthFix)（规避 Comfy-Org/ComfyUI_frontend#12443，上游 #11574 引入的回归；修复 PR #12444 截至 2026-09 仍未合并）。需求为不引入第三方包、在 sfnodes 内做等价处理 → `web/sf_widget_width_lib.js` + `web/sf_widget_width_fix.js`。

### 1. 根因链（前端 1.53.6 打包产物实测）

- Vue 的 `WidgetLegacy` 组件在 **Classic（LiteGraph）模式仍挂载**，其 `draw()` 每帧无条件执行 `widgetInstance.width = DOM容器getBoundingClientRect().width`。打包代码原文（`settingStore-*.js`，变量名压缩）：`s.y=0,s.width=e,n.value.height=(t+2)*RP,n.value.width=e*RP;`。
- LiteGraph 渲染器消费宽度是**回退式**：`drawWidgets` 里 `let l=s.width||r`（`r=this.renderingSize[0]`）；`DomWidgets.vue` 是 `posWidget.width ?? posNode.width`。widget 出厂时 `width` 本是 **undefined**，所以回退生效；一旦被 Vue 写成数字，`node.size[0]` 就被永久忽略 → 拖节点/改宽后 widget 冻结在原宽度（溢出/悬空），移动节点也会触发。
- 结论：**上游不修就必须阻止这个写入落到 widget 上**——清理一次没用（下一次 draw 又写回），必须换成受控访问器。

### 2. 修复形态（`sf_widget_width_lib.js` 纯逻辑）

- `guardWidgetWidth(widget, isVueNodes)`：幂等（`_sfWidgetWidthGuarded` 标记），把实例 `width` 换成 `Object.defineProperty` 访问器（`configurable/enumerable: true`，保留同构）——**Vue 模式读写透传；Classic 模式写入丢弃、读取 undefined**（回到出厂未定义态，让 `width || nodeWidth` 回退）。模式判断是 getter/setter 内**动态**执行，切渲染器无需重装。
- `guardNodeWidgets` / `sweepGraphWidgets`：扫 `node.widgets`（DOM/custom widget 都在内）；图遍历兼容 `_nodes` 与 `nodes`、`graph.subgraphs`（Map）与节点 `.subgraph` 两形态，`Set` 去重防子图互引死循环。
- `patchWidgetFactories(LGraphNode, isVueNodes)`：包装 `prototype.addWidget` + `prototype.addCustomWidget`（`addWidget` 内部委托 `addCustomWidget`，DOM widget 助手 `addWidget(node,widget)` 也走 `addCustomWidget`），之后创建的 widget 即时受控；`_sfWidgetWidthPatched` 防重复包装。
- 探测回调异常时**保守透传**（不误丢合法写入）。

### 3. 安装三路（`sf_widget_width_fix.js`，扩展名 `sfnodes.WidgetWidthFix`）

1. 模块加载即 `install()` + `sweepAll()`：全局 `LiteGraph.LGraphNode` / `globalThis.LGraphNode` 包装 + 现有图扫描；
2. `init()/setup()/afterConfigureGraph()` 再 install+sweep（图对象在 setup 后才就绪；加载工作流后补扫）；
3. `nodeCreated()` / `loadedGraphNode()` 逐节点补扫，兜底不经过 addWidget 自建 widget 的扩展。

无设置开关、始终生效：上游修好后写入只发生在 Vue 模式（守卫透传）→ 自动 no-op，可长期保留。跨扩展全局生效是刻意选择（bug 在前端渲染器层，范围限 sfnodes 会让混用工作流的其它包继续冻结）。

### 4. 风险与边界

- Classic 下若有扩展**故意**依赖数字 `widget.width`（自绘/命中计算）会读到 undefined；LiteGraph 核心与 vueNodes 核心消费点全是 `width || nodeWidth` / `?? nodeWidth`，正是目标回退。仓库内已 grep 无 `widget.width =` 用法。
- 若同时装了第三方 LegacyWidgetWidthFix 包：双方各自幂等守卫，先装者生效，语义一致，无冲突。
- 上游一旦合并 #12444/#15331，本模块无需改动（透传语义与上游修复方向一致），也可删除。

### 5. 测试

`tests/test_sf_widget_width_lib.mjs`（Node 直跑）：守卫读写语义（Classic 丢弃读 undefined / Vue 透传 / 动态切档保留值 / 属性可配置）/ 幂等 / 探测异常透传 / 节点扫描计数 / 全图遍历（`_nodes`·`nodes`·subgraph Map·节点 `.subgraph`·防环）/ 工厂包装（两条路径即时守卫、二次安装不叠包、空原型安全）。
`tests/test_widget_width_fix_js.js`（.mjs 拷贝链真实加载主模块）：扩展注册名 / 模块加载即工厂包装 + 全图扫描（已有节点·subgraph·节点 `.subgraph`）/ 新 widget 即时受控 / Classic 写丢弃·Vue 透传 / init·setup 幂等不叠包 / nodeCreated·loadedGraphNode·afterConfigureGraph 三路补扫。

---

## 123. SF Menu 节点右键入口（getNodeMenuItems 扩展钩子，2026-09）

> 背景：📦 SF Menu（§15）原仅挂画布背景 `getCanvasMenuItems`，用户要求节点上右键也能弹出。实现：`web/sf_canvas_menu.js` 把组装抽为 `buildSfMenuOptions`，`getCanvasMenuItems` / `getNodeMenuItems(node)` 两钩子共用同一顶层项（零逻辑复制），`tests/test_canvas_menu_js.js` 锁定双入口一致。

- **前端收编链（前端 1.53.6 打包产物实证）**：`GraphView-*.js` 包装 `LGraphCanvas.prototype.getNodeMenuOptions`——先取原生项，再 `collectNodeMenuItems(node)`（= `invokeExtensions("getNodeMenuItems", node).flat()`）append，最后 append legacy `getNodeMenuOptions` 项。Classic 由 `processContextMenu(node)` 走该函数，Vue 由 `canvas.getNodeMenuOptions(node)`（`useMoreOptionsMenu-*.js`）消费——双模式同一钩子，无需 monkey-patch。
- **钩子返回值是菜单项数组**（同 `getCanvasMenuItems`，不是 `{title,options}` 对象）；`null` 项渲染为分隔线（`sf_outpaint.js` `[null, ...]` 先例）。本包早用者：`sf_lora_stack.js` / `sf_outpaint.js` 挂节点专属项，与聚合器全局项互不影响（都被 append 进同一数组）。
- **右键即选中（选中集门槛语义不变的关键）**：Classic mousedown 的 `e.button===2` 分支先 `processSelect(node, e, true)` 再 `pointer.onClick ??= () => processContextMenu(node, e)`；Vue `handleNodeRightClick` 未选中时 `deselectAll + select(右键节点)`。故 SF Align（≥2）/ SF Node Color（≥1）在节点入口按同一门槛注入。
- **形态与门槛**：节点入口与画布入口返回同一 `{content:"📦 SF Menu", has_submenu:true, submenu:{options}}`；不按节点类型过滤（任意节点可用），门槛留在构建器侧（`buildAlignMenuItems` <2 返回 []、`buildNodeColorMenuItem` 无选中返回 null）。
- **已知边界**：节点菜单路径不跑画布路径的 `contextMenu.*` 翻译（原样显示，无影响）；扩展项 append 在原生项之后、legacy 项之前。
- **测试**：`tests/test_canvas_menu_js.js` 断言 `getNodeMenuItems` 存在、顶层同项、0/2 选中门槛与画布一致；不影响 `test_lora_browser_smoke.js` / `test_note_js.js` 的"已移交聚合器"回归（它们只查 `getCanvasMenuItems`）。
- **Vue nodes 模式限制（1.53.6 实证，仅记录）**：Vue 的 `NodeContextMenu`（`useMoreOptionsMenu`）只在 `selectedNodes.length === 1` 时才合并 `canvas.getNodeMenuOptions(node)` 的结果——**≥2 选中时扩展项完全不出现**（走 Vue 原生选择菜单）。即 Vue nodes 下多选右键节点看不到 📦 SF Menu（Mouse Node 对齐不可达）；单选时可见（LiteGraph 块前置，菜单顶部）但 Align 因 <2 不注入。默认 `Comfy.VueNodes.Enabled=false`（Classic 不受影响），本包按 Classic 为主不另做兼容。

---

## 124. SF Align 基准改为鼠标所在节点（First Selected 退场，2026-09）

> 背景：`SF Align` 的 `First Selected` 三项（§11）依赖选中集首项，实测点选顺序不确定（`selected_nodes` 四形态顺序语义不同：Array/Set/Map 为加入顺序，旧 Object 形态 `Object.values` 按键升序＝节点 id 序），用户改用「鼠标所在的节点」作基准。

- **基准捕获时机是菜单构建时**，不是动作执行时：点击菜单项时鼠标已移到菜单上。节点入口 `getNodeMenuItems(node)` 的 node 参数即右键节点，聚合器把它透传给 `buildAlignMenuItems(refNode)`；动作闭包捕获 refNode。
- **画布空白右键无基准**：三项 Mouse Node 不注入（画布入口 6 动作；节点入口 9 动作），避免死行。`calcTargetWidth/Height` 的 `"mouse"` 档在无 refNode 时返回 0（动作守卫 `if (!tw) return` 兜底）。
- **多选保持的前端依据**：右键**已选中**节点不会清空选择——Classic `processSelect(node, e, true)` 命中 `else return`（force 且已选中，选中集不变）；Vue `handleNodeRightClick` 已选中时跳过重选。故流程为「Ctrl/Shift 多选 → 右键其中一个（基准）→ SF Align ▶ …: Mouse Node」；右键**未选中**节点会塌缩为单选（`<2` 不注入 SF Align，属预期）。
- **纯逻辑**：`sf_canvas_align_lib.js` 的 `calcTargetWidth(nodes, mode, refNode)` / `calcTargetHeight(...)` 新增 `refNode` 形参（`"first"` 档删除）；`buildAlignMenuItems(refNode)` 签名扩展，聚合器 `buildSfMenuOptions(refNode)` / `buildSfMenuItem(refNode)` 贯通。
- **测试**：`tests/test_canvas_align.mjs`（mouse 档取 refNode 尺寸 / 缺基准 0 / 集成锚点）、`tests/test_canvas_menu_js.js`（画布入口 6 动作无 Mouse Node、节点入口 9 动作、驱动 Width: Mouse Node 以右键节点为准）。

---

## 125. SF Menu 内容扩充与画布分隔线（2026-09）

> 背景：📦 SF Menu 子项扩充与打磨（`web/sf_canvas_menu.js`）：新增 `SF LoRA Presets`、`Add SF Note`，画布入口加前导分隔线，Align 不可用时出 disabled 提示行，排序改为上下文相关项在前。`tests/test_canvas_menu_js.js` 锁定。

- **画布入口前导 `null` 分隔线、节点入口不加**：LiteGraph `ContextMenu` 与 Vue 转换器都把 `null` 当分隔线（`convertContextMenuToOptions`：`n===null → {type:'divider'}`）；画布菜单原生项之后紧跟本包项，加 null 视觉分组。节点入口不能加——Vue 单选时 LiteGraph 块整体前置，前导 null 会变成**菜单顶部横线**。
- **`SF LoRA Presets` 独立模式**：`openLoraPresetManager()` 无 ctx 直接可用——`canSave = ctx.canSave !== false && !!ctx.node` 门控隐藏"保存当前为预设"整块（含 readState 动态 import，全在 canSave 分支内），其余 `ctx.node` 访问均 `if (ctx.node)` / `?.` 守卫；列表/搜索/编辑/删除/重命名不依赖节点。回调包一层 `() => openLoraPresetManager()`，避免 LiteGraph 把 `option.value` 当 ctx 传入。
- **`Add SF Note` 回归 + 落点=鼠标位置**：`sf_note.js` 的 `addNoteFromMenu(pos)` 此前只 export 未接线（§15 曾"便签不占菜单入口"），现接回 SF Menu 尾部（`Comfy.AddNode` 优先、`LiteGraph.createNode` 兜底）；落点改造：**菜单项点击时鼠标已移到菜单上**，位置必须在菜单构建时捕获——聚合器模块顶层装 `window` capture `pointerdown`（`button===2`）记录画布坐标，换算与前端 `adjustMouseEvent` 同式（`(clientX-rect.left)/ds.scale - ds.offset[0]`），Classic/Vue 通用且早于前端建菜单（Classic 在 pointerup 的 `onClick` 里 `processContextMenu`）；缺省回退 `canvas.graph_mouse` → 视口中心+抖动（键盘开菜单等无 pointerdown 场景）。pos 有效时节点左上角落点即鼠标位置（无抖动）。
- **Align 提示行**：`buildAlignMenuItems` 仍 `<2 返回 []`（门槛留在构建器侧），聚合器改放 `{content:"SF Align (select ≥2 nodes)", disabled:true}`（无 callback，fail-safe）提升可发现性；≥2 时正常嵌套子菜单。
- **排序**：Align/Node Color（随选中集出现，上下文相关）在前，`SF LoRA Browser → SF LoRA Presets → SF Workflows → SF Memory → Add SF Note` 随后（LoRA 两工具相邻）。
- **测试**：`tests/test_canvas_menu_js.js` 增补——画布前导 null / 节点无 null、0 选中提示行 disabled 无 callback 且居首、2 选中提示行消失、排序链、`Add SF Note` 落图（先 `registerCustomNodes()`，test_note_js 先例）、`SF LoRA Presets` 面板挂载（`document.body.appendChild` spy + overlay id）；DOM 桩 `style` 补 `setProperty/removeProperty/getPropertyValue`。

---

## 134. IS_CHANGED 拿不到连线输入的值（链接输入传 None，2026-09）

> 背景：用户的循环工作流出现"单轮耗时随 index 线性增长"（`SFForLoopStart.index` → `SFLoadImagesPath`），日志里每轮一条 `WARNING: '>' not supported between instances of 'NoneType' and 'int'`，间隔 0.05s → 0.96s。按该 WARNING 反查：只有 `IS_CHANGED` 的 `image_load_cap > 0` 能产生它 → 那一轮 `image_load_cap` 是**连线输入**（运行时值 = index，第 k 轮加载 k 张图）；正确接法是 index 接 `skip_first_images`、`image_load_cap` 留 widget=1。

- **机制**：`IsChangedCache.get` 用 `get_input_data(node["inputs"], class_def, node_id, execution_list=None)` 取"常量"——链接输入没有执行列表可查 → `mark_missing()` 写 `(None,)`，`_async_map_node_over_list` 的 `slice_dict` 再取 `v[0]` → **IS_CHANGED 形参拿到 None**（既不是链接对象也不是 widget 默认值）。于是 IS_CHANGED 里对可连线输入的常规写法都会踩空：
  - `x > 0` / 算术比较 → TypeError（被 `except Exception` 吞成 `logging.warning("WARNING: {}")` + `is_changed = NaN`）；
  - `lst[x:]` → **不报错**：`lst[None:]` 等价于 `lst[:]`，静默按"全量"走（本次 skip 连线就是这种，哈希只覆盖第一张，与 skip/nth 无关）。
- **本次误诊现场**：`image_load_cap` 被连线（运行时值 = index）→ 第 k 轮解码+预览 k 张图（780×1200 单张：JPEG 解码 ≈7ms + 预览 PNG ≈17ms）→ 总耗时 O(N²)、单轮 +30ms ✓ 与日志间隔增量吻合。**每轮一条的 WARNING 是定位这类接线错误的关键线索：其时间戳间隔就是单轮耗时**（`docker exec comfyui-docker` 看 `user/comfyui.log` / `comfyui.prev*.log`，重启会清 `/history`、清不了 rotated 日志）。
- **修法（`nodes/image/load_images_path.py` 已改）**：任一形参为 None 即视为"未知切片"，退化为对**目录全部图片**哈希（cap=0/skip=0/nth=1）：目录一变即失效、绝不误用缓存，且**不返回 NaN**——NaN 会沿祖先签名折叠下游全部缓存（patterns.md §89），本节点靠稳定签名支持重复 Run 命中缓存（容器日志里的 `Prompt executed in 0.00 seconds`）。
- **测试**：`tests/test_load_images_path.py`——哈希稳定/切片参与（cap、skip）/三输入各自 None 与全 None 均等于全量哈希/目录不存在 False/mtime 变化与复原（用 `st_mtime_ns` 精确回写，防 `os.utime(float)` 纳秒漂移导致假失败）。
