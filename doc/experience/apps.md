# 经验归档：无节点面板应用（§10、§30）

> 全局章节号 §N 与拆分前的 experience.md 一致；跨节/跨文件引用一律写 §N，映射见 [README.md](README.md)。版本时效说明见 README。

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

---

## 30. SF LoRA 浏览器：工具栏应用 + 信息面板宿主 ctx 适配（浏览全部 LoRA 并编辑信息）

> 2026-08。需求：工作流界面顶部 sf workflows 按钮旁加按钮，打开界面浏览全部 LoRA、并编辑 LoRA 信息（触发词/描述/封面/Civitai），编辑体验对齐 SFLoraStack 信息面板。

### 1. 设计：无节点应用 + 后端零新增

- **无节点设计**（同 §10 Workflows）：浏览器是"应用"不是节点——节点会被存进工作流文件，分享污染他人。打开方式：工具栏按钮（紧贴 Workflows 按钮）+ `Alt+Shift+L` + canvas 右键菜单 + 命令面板。
- **后端零新增**：列表 `/api/sfnodes/lora_list`、信息 `/api/sfnodes/lora_info`、封面 `/api/sfnodes/lora_thumb`、自定义词/描述/封面/Civitai 全部复用 SFLoraStack 既有路由与 `sf_lora_stack_api.js` 封装（`listLoras/loraInfo/thumbUrl/saveCustomTriggers/saveCustomDescription/saveLoraPreview/...`）。改动纯前端，无需重启容器，硬刷新即生效。
- 分层（对齐项目模块惯例）：`sf_lora_browser.js`（状态+数据+信息面板宿主 ctx+扩展注册）/ `sf_lora_browser_ui.js`（窗口/网格/CSS）/ `sf_lora_browser_lib.js`（纯函数：splitName/filterLoras/groupLoras/sortWithinGroup，可 .mjs 直测）。

### 2. 关键机制：信息面板宿主 ctx 适配（sf_lora_stack_info.js）

SFLoraStack 信息面板原本只依赖节点做四件事：① `readState(node)` 读行（triggers/custom）、② `patchLora(node,id,patch)` 写行、③ `accentOf(node)` 强调色、④ `place()/startFollowing()` 锚定节点。其余全部按 LoRA 名走服务器 API，与节点无关——这是可解耦复用的边界。

- **`openInfoPanel(node, id, refresh)` 兼容入口保留**（Stack 行 UI 与冒烟测试零改动）：内部构造 node ctx（`getRow=readState(node).loras.find(id)`、`patchRow=patchLora(node,id,…)`、`prefs` 由 readState 读 thumbs/civitai、key=node 对象）后委托 `openInfoPanelFor(ctx, id)`。
- **新增 `openInfoPanelFor(ctx, id)`**：ctx = `{ key, node?, anchorRect?, getRow, patchRow, accent, prefs?, refresh? }`。浏览器宿主：key=字符串 `"sfnodes.lora-browser"`（`closeInfoPanelFor(node)` 只关自己 key 的面板，互不干扰）、getRow/patchRow 走会话内内存行、`anchorRect` 返回被点击卡片的 `getBoundingClientRect()`、**无 `ctx.node` 时跳过 `startFollowing`**（不跟随画布）、`place()` 按 `ctx.node ? getNodeRect(ctx.node) : ctx.anchorRect()` 选锚。
- 替换点清单（约 15 处）：`readState(node).loras.find(id)` → `ctx.getRow()`；`patchLora(node,id,…)` → `ctx.patchRow(…)`；`readState(node).thumbs/civitai` → `ctx.prefs().thumbs/civitai`；`place(panel, node)` → `place(panel, ctx)`；模块级 `_ownerNode` → `_ownerKey`。
- **注意 startFollowing 内部自己的 `place(panel, node)` 不能跟着 replace_all 改**——它的参数是 node 不是 ctx，误改会 ReferenceError。

### 3. 浏览器行宿主：会话内存副本，真源在服务器

- 浏览器行只是面板的可读写对象 `{id,name,triggers,custom}`：勾选/自定义词显示在面板内，随会话存活。**真源始终在统一存储**（`saveCustomTriggers/saveCustomDescription` 按 LoRA 名写回 `user/sfnodes/lora_triggers.json`，Loader 系对话框/Stack/浏览器三端互通）；行副本不持久化、不进任何工作流文件。
- `hydrateCustom` 把服务器 `info.custom_triggers` 合入行副本，`persistCustom` 在改动时写回——两端行为与 Stack 面板 1:1。

### 4. 列表与封面

- `listLoras(true)` 打开/刷新按钮时强制重取（no-store 路由 + 模块内会话缓存共享自 Stack——force 绕过缓存防改名/增删后过期）。
- **文件夹层级浏览（2026-08 改进，对齐 SF Load Image Browser 的浏览器）**：`sf_lora_browser_lib.js::folderContents(list, folder)` 返回「立即子目录 + 当前层文件」（镜像 image_browser 的 getFolderContents：文件夹只取第一段去重、文件只收当前层直接文件，均已排序）——**不再平铺分组**。面包屑（根 All LoRAs + 逐级、非当前级可点击跳转、当前级 .cur 样式）**用 DOM API 构建**（`textContent`/`dataset` 赋值——目录名来自用户文件系统，不经过 innerHTML 注入面，`<`/`"` 目录名天然安全；image_browser 的面包屑是 innerHTML 字符串 + escapeHtml，此处在 mock 测试中暴露后改 DOM 构建更稳）。
- **搜索语义**：搜索激活时忽略目录层级、跨全部分层扁平匹配（与 image_browser 同语义）；计数显示「命中 / 总数」。
- **浏览位置记忆**：设置键 `sfnodes.LoraBrowser.Folder` 记住所在目录——打开窗口时恢复，列表到达后按目录存在性校正（`validFolder`：目录被删/改名回根）；搜索时面包屑仍显示当前层 context。
- **网格/列表双视图（2026-08）**：bar 第二个 seg（九宫格/三横线 SVG 图标，无文字），记忆 `sfnodes.LoraBrowser.View`；列表行 `.sf-lb-row` = 40px 缩略图 + 文件名 + 目录/扩展名（文件夹行 = 📁 图标 + 名称，进入下钻）；**单击/双击防抖提取为 `attachPickAdd(el, name, onPick, onAdd)` 与网格卡片共用**（列表行同样支持单击开信息面板、双击加载到工作流），缩略图 error 占位提取为 `wireThumb`（卡片 108px 与行 40px 共用）；`renderFolder`/`renderFlat` 增加 `view` 参数按容器分支（`.sf-lb-grid` / `.sf-lb-list`）；平面模式滚动分批对列表视图同样生效（行高小、FLAT_STEP=60 也够）。**flat 模式图标从三横线改为层叠（layers）**——三横线语义被列表视图占用，两图标同形会混淆。
- **性能**：卡片 `content-visibility:auto` + `img.loading="lazy"`——数百上千 LoRA 不一次性拉图，浏览器按视口渲染与解码；缩略图 404 → onerror **替换为内联 SVG 占位图**（深色圆角底 + 层叠图标，data URI 无网络请求必成功渲染；`removeAttribute("src")` 在实测中某些渲染路径仍残留浏览器破损图，有 src 的占位才彻底），守卫防二次 error 循环。
- **封面跨端刷新**：缩略图路由发 `max-age=3600` 且 URL 不变——任一端（浏览器/Stack/信息对话框）改了数据经 `sfnodes.lora-data-changed`（detail.name）事件刷新可见卡片封面，URL 带 `&t=Date.now()` bust。
- **双击加载到工作流（2026-08）**：文件卡片双击 → 用 SF LoRA Stack 加载该 LoRA 并添加节点，**三分支**：
  - **无 Stack 节点** → 新建：**优先官方命令 `app.extensionManager.command.execute("Comfy.AddNode", { type: "SFLoraStack" })`**，执行后从 `app.graph._nodes` 按「新增集合差」找回节点（不依赖命令返回值形状）；命令缺失/失败兜底 `LiteGraph.createNode` + `graph.add`。位置 = 画布视口中心换算（`(p - ds.offset)/ds.scale`，ds 缺失回退左上）+ 随机 ±30px 防连续双击重叠；尝试 `app.canvas.selectNode` 选中（Vue 无此 API，try/catch 忽略）。
  - **恰一个 Stack 节点** → 直接向它插入（不新建）。
  - **多个 Stack 节点** → 弹出选择器（`sf-lb-pick-mask` 面板风模态：每行 `#id · title（用户改过的节点标题，缺省回退类名 SFLoraStack）· N LoRA(s) · 首行名`（readState 读行数）；点行选择，遮罩/Esc/Cancel 取消；打开前 closeInfoPanel 收掉可能开着的面板）。
  - 插入统一走 `addLoraRow(node, name)`：`addLora`（core 状态机写 `properties.loraStackState`）→ `node._sfLsRefresh(true)`（renderNode + fitToContent，setupNode 在 nodeCreated 时挂）→ selectNode。
  - **单击/双击防抖**：浏览器双击先派发两次 click 再 dblclick——单击延迟 250ms、dblclick 时 clearTimeout 取消在途单击（第二次 click 覆盖第一次 timer，dblclick 再清一次），双击不误开信息面板。
  - **Vue 新版实测大坑（2026-08 实测）**：裸 `LiteGraph.createNode` + `app.graph.add` **只弹成功 toast 却不渲染节点**——Vue 前端的节点创建/类型注册/widget store 同步必须走官方 AddNode 命令；Classic 前端（及命令缺失兜底）下 createNode + graph.add 仍可用。测试 mock `extensionManager.command.execute` 覆盖命令路径 + 断言命令被调用。
- **文件夹/平面双模式（2026-08）**：bar 上 seg 切换（纯 SVG 图标按钮：文件夹/层叠——mask-image data URI，无文字，title 承载说明，与工具栏按钮图标同风格），模式记忆设置键 `sfnodes.LoraBrowser.Mode`（与 Folder 位置记忆同机制，打开窗口恢复）。**平面模式 = 全量列表分批渲染 + 滚动动态加载**（防 LoRA 上千时一次性建 DOM/拉图卡死）：
  - `renderFlat(main, {names, shown})` 一次只建 `shown` 项卡片（`FLAT_STEP=60`，主扩展 `S.flat.page` 批次游标），未载完时 main 底部挂 `sf-lb-loadmore` 哨兵（显示「已载 / 总数」）；
  - `attachFlatScroll(main, onNeedMore)` 幂等绑定 scroll 监听，距底 300px 回调续批（主扩展判断还有更多才推进 page）；
  - **视口未满自动续批**：render 后若 `scrollHeight <= clientHeight + 8` 且还有更多，立即 page++ 再 render（有限步，防高窗口/小步长空转）——注意必须带「还有更多」守卫，否则空列表死循环；
  - 面包屑行平面模式隐藏（`.sf-lb-path` display:none）；计数未载完显示「已载 / 总数」、载完显示总数；切换模式/搜索均重置批次游标；两种模式搜索都跨层级扁平匹配；
  - 卡片函数（`folderCard`/`fileCard`）提取为模块级供两个渲染器复用，避免内联副本。

### 5. 工具栏按钮定位

- 复刻 Workflows 的 `app.menu.settingsGroup.element` 前插模式：**已挂载 Workflows 按钮时插其 group 之后（实现"sf workflows 按钮边"）**，否则兜底插 settingsGroup 前 + 25 次×250ms 重试。两种顺序下两按钮都相邻（谁后挂载谁贴近 settings 组）。
- 热键 `Alt+Shift+L`：包内唯一冲突面是 workflows 的 Alt+Shift+W；ComfyUI 按 combo 全局去重，若第三方占用注册会抛错——诊断脚本会暴露，换修饰键即可。

### 6. 测试与验证

- `tests/test_lora_browser_lib.mjs`：lib 纯函数（路径拆分/过滤/分组/排序）。
- `tests/test_lora_browser_smoke.js`：mock DOM 真实加载全依赖链——扩展注册（name/command/keybinding/canvas 菜单）→ 按钮挂载 → 点击开窗 → lora_list 数据层 → 网格渲染/计数 → 搜索过滤 → 点击卡片真实打开信息面板（浏览器 ctx 路径）→ 关闭。**同时锁定重构回归**：`test_lora_stack_info_desc_smoke.js` 25 断言全绿证明 Stack 路径逐字节不变。
- 诊断脚本（交付用户）：版本检查 → 扩展注册状态 → 按钮挂载 DOM 检查 → 打开窗口 → 数据层（lora_list 计数）→ 网格渲染 → 信息面板编辑往返（勾词/存描述）→ 热键。部署：`web/` 同步 docker 目录 + 浏览器硬刷新。
