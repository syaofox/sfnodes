# 经验归档：文本与提示词节点（§6、§7、§14、§15、§16、§18、§23、§24、§29）

> 全局章节号 §N 与拆分前的 experience.md 一致；跨节/跨文件引用一律写 §N，映射见 [README.md](README.md)。版本时效说明见 README。

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

## 14. SFTextFindReplace：查找替换双端镜像（复刻 Pixaroma Find & Replace）

> 背景：复刻 Pixaroma 的 `PixaromaFindReplace`（2026-08），落地为 `nodes/text/find_replace.py` + `web/sf_find_replace*.js` 三模块（lib/ui/主扩展）。节点坐在文本连线中间，按规则编辑流过它的文本：每条规则 find→replace，四个全局开关（Case / Whole word / Regex / Tidy），节点上实时预览"上次运行输入 × 当前规则"的前后对比（LCS 词级 diff，红删绿增）。核心机制是**替换逻辑 Python/JS 双端镜像 + ReDoS 启发式防护**——"预览必须与真实运行一致"的完整案例。

### 1. 双端镜像的差异点（做"节点上实时预览"必知）

- **literal 模式**：Python 侧 `safe_repl = repl.replace("\\", "\\\\")` 双写替换文本的反斜杠，使字面含 `\1`/`\g<1>` 的字符串不被当反向引用；JS 侧只转义 `$`（`repl.replace(/\$/g, "$$$$")`，反斜杠天然字面）——两边各转各的特殊字符，语义对齐。
- **regex 模式**：Python 用 `re.sub`，JS 用 `String.replace`。三处翻译（`pyTemplateToJs`/`pyPatternToJs`/`makeRegexU`）：
  - 反向引用 `\1`（Python）→ `$1`（JS）；`\g<0>`（整个匹配）→ `$&`；命名组 `(?P<n>)` → `(?<n>)`，`(?P=n)` → `\k<n>`；替换文本中的 `\n`/`\t`/`\r`/`\f`/`\v`/`\\` 字符转义 Python 会处理而 JS 不处理，需手工展开。
  - 忽略大小写用 `/u` flag 使 Unicode 折叠对齐 Python `re.IGNORECASE`（Kelvin 符号 K→k、重音 A-ring 等）；`/u` 编译失败回退非 `/u`（正则模式，转义字面量总是 `/u` 安全）。
  - **已知偏差（文档化，Python 是权威）**：`\w \d \s \b` 类在 JS 预览仅 ASCII、Python 中 Unicode 感知（`\w+` 作用于重音/希腊/CJK 预览比实际窄）；内联 flags `(?s)/(?m)`、`\10` 两位引用、替换模板未知转义 `\q` 等 Python 报错跳规则而 JS 静默成字面文本——这些情况预览稍有出入但运行正确。whole-word 的 JS 侧不用 `\b`（ASCII 边界），改用显式 `(?<![\\p{L}\\p{N}_])`/`(?![\\p{L}\\p{N}_])` 断言镜像 Python 的 Unicode `\b`。
- **测试同用例同期望值**：literal/whole-word/大小写/中文/`$` 转义/backref 翻译/命名组翻译两侧各跑一遍（Python 直接跑 `_apply_rules`，JS 复制 lib 为 .mjs 直跑 `applyRulesJS`），两侧独立通过即视为镜像一致。**复查抓到的断言反例全是测试错、代码对**：`(a+)+b` 本身就是嵌套量词（组内 `+` 又被 `+` 限定）；`tidy("a  x , ,  b,")` = "a x, b"（tidy 只修空格/逗号，不删 x——先应用规则再 tidy）；删除规则后的空格计数要先数清（`"a  x b"` 删 x 后是三空格）。

### 2. ReDoS 防护：嵌套无界量词启发式（双端 1:1 镜像）

- **问题**：Python 侧 `re.sub` 无超时执行，`(a+)+` 对不匹配输入可指数级回溯卡死 worker；前端预览每次按键重算，同一模式冻结浏览器。原生正则无法限时，只能拒绝明显形状。
- **启发式**：栈扫描——每个打开组记 `inner`（组体内出现过无界量词 `*`/`+`/`{n,}`）；组关闭时若组后紧跟无界量词且 `inner` 为真 → 判定灾难性，跳过该规则 + 警告。`{n}`/`{n,m}` 有界不触发；跳过转义字符与字符类 `[...]`（`[()]+` 不误报）。误报率低：嵌套无界量词总是冗余的（`(a+)+ == a+`），合法模式不用。Python `_is_catastrophic_regex` 与 JS `isCatastrophicRegex` 逐分支镜像，两侧测试同用例（嵌套四例触发、有界/转义/字符类不触发）。

### 3. 状态与预览持久化（数据载体模式）

- 规则状态存 `node.properties.findReplaceState`（{version, caseSensitive, wholeWord, regex, tidy, rules:[{id, enabled, find, replace}]}），`graphToPrompt` hook 经 `readState` 规范化后注入隐藏 `FindReplaceState` 输入（Pattern #9，随 workflow 保存、在缓存键中，规则变化自动失效下游，Python 侧不设 IS_CHANGED——SFPauseText 的 NaN 踩坑同款）。id 用 Date+counter+random 保证跨刷新唯一（删除/排序的键）。
- **预览样本与规则状态分离**：输入样本存 `node.properties.findReplacePreview`（{input, truncated}，4000 字符自我保护上限）**绝不注入 prompt**——否则每次 Run 的文本会膨胀工作流文件与 websocket 负载。预览 = 上次运行输入 × 当前规则实时重算：Run 一次后编辑任意规则，前后对比即时刷新（diff 高亮是预览的卖点）。`readState` 只在畸形时重写（加载不脏），`onNodeCreated` 时深拷贝 state 防粘贴/克隆节点共享 rules 数组引用。
- executed 回填：Python 返回 `{"ui": {"sf_find_replace": [{input, output, truncated, warnings}]}}` → 前端 `onExecuted` 读 `message.sf_find_replace[0]` → `setPreviewInput` + 重绘预览。**onExecuted 不调整节点尺寸**（Run 不改变规则数，每次 setSize 会把普通 Run 误标 modified）。

### 4. 移植简化与测试

- **shared 依赖裁剪**（第三个同款案例，惯例确认）：`isVueNodes`/`applyAdaptiveCanvasOnly` 内联（sf_pause_text.js 同款）；省略 accent 颜色（CSS 固定 `#f66744`）、注册帮助面板、resize floor、canvas zoom 穿透。CSS 类名全 `sf-fr-` 前缀 + 全局钩子 `app._sfFindReplacePatched` guard——**与 Pixaroma 共存时类名不互相污染、graphToPrompt 各包装一次链式组合**。
- **高度自适应简化**：`measureMinHeight(root)` 实时测量固定部分（开关+规则行+操作，预览用 PREVIEW_MIN 100 代替），4px 网格取整防累积；`refitNode` 只在用户操作时调用（加载路径不动 node.size 防误标 modified）——增行增高、删行且用户未手动拉高则缩回、拖高保留；Nodes 2.0 用 `computeLayoutSize`（忽略 legacy getMinHeight）喂同款测量。
- **测试**（2 个文件，49+70 断言）：后端 `test_find_replace.py` 直测 `_apply_rules`（literal/whole-word/大小写/regex backref/`$`/tidy/ReDoS 警告/非法正则/畸形状态容错/中文/Unicode 折叠/预览截断 4000+结果全长）；前端 `test_find_replace_js.js` 复制 lib 为 .mjs 测 state 全 mutator/readState 容错/applyRulesJS 全分支/diffTokens（含 1M 上限退化）/isCatastrophicRegex/escapeHtml。**lib 额外导出 `isCatastrophicRegex`**（原件不导出）供直接与 Python 版对照。

### 5. 模块边界（复用/修改时的快速索引）

- `nodes/text/find_replace.py`：`SFTextFindReplace`（apply/_parse_state）+ 模块级纯函数（`_apply_rules`/`_tidy`/`_is_catastrophic_regex`/`_unbounded_quant_at`），无 torch 依赖可直接单测。
- `web/sf_find_replace_lib.js`：纯函数（state 全套 mutators/readState 容错/`applyRulesJS`/`tidy`/`isCatastrophicRegex`/`diffTokens`/`escapeHtml` + 内部 `pyTemplateToJs`/`pyPatternToJs`/`makeRegexU`）——无 app/DOM，测试 copy 直跑。
- `web/sf_find_replace_ui.js`：DOM widget（injectCSS/buildRoot/renderAll/buildRuleRow/refreshResetState/renderPreview + 交互 attachFieldEditor/attachDragHandlers/autoGrowAllFields/sfConfirm 主题确认框）。
- `web/sf_find_replace.js`：主扩展（onNodeCreated 微任务 setup/onConfigure/onExecuted/onResize/onDrawForeground legacy 钳制/onRemoved + graphToPrompt 注入，subgraph 复合 id 递归索引）。
- 数据契约：`FindReplaceState`（hidden STRING，graphToPrompt 从 node.properties.findReplaceState 注入）；`sf_find_replace` ui 键；`node.properties.findReplacePreview`（预览样本，不注入 prompt）；`node.properties._sfFrAutoH`（自动高度记忆，区分自动适配与手动拖拽）。

---

## 15. SFValueDropdown：值下拉与输出点对齐（复刻 Pixaroma Dropdown）

> 背景：复刻 PixaromaDropdown（2026-08），落地为 `nodes/text/dropdown_value.py` + `sf_utils/dropdown.py` + `web/sf_dropdown*.js` 四模块（lib/ui/settings/主扩展）。节点是"自己写的 name→value 列表"：每行短名 + 实际值，四种输出类型（text/int/float/bool），三种运行模式（F 固定 / I 递增 / R 随机）。**本项目第一个做输出点对齐的节点**（把输出点移上节点行、随类型改名），也是第三个"纯逻辑双端契约 + 注入"案例。以下为可迁移结论，细节见代码注释。

### 1. lean 注入形状作缓存键（改行名不重跑）

- 前端 `graphToPrompt` 注入 `{"version": 1, "type": ..., "value": ...}`（**只有选中行的值 + 类型**），Python `selected_value` 接受两种形状：**LEAN**（`{"type","value"}`，浏览器注入，键判断 `"value" in state` 而非真值——空串/0/False 都是合法值）与 **FULL**（`{"type","index","options"}`，工作流存储形状，兜底手写 API 文件）。
- **注入字符串即缓存键**：只含影响结果的部分。行名、列表其余行、模式、任何 UI 标志都是显示用——改行名/重排/改未选中行/切模式都不触发重跑。这是 Pattern #9 的"缓存键最小化"原则，与 SFPromptTags 的注入同款。
- 隐藏输入必须声明为 Python `INPUT_TYPES` 的 `hidden`（required STRING 会在 Vue 前端同时显示为 widget 和可转换输入点）；键名 PascalCase（`DropdownState`），node.properties 键 camelCase（`dropdownState`），两者刻意不同——第二个打错 Python 永远看到默认值、节点"无视一切修改"。

### 2. 运行游标：pending 持有 + commitPick（存节点内存而非设置）

- **与 SFPromptTags 的关键差异**：游标存 `node._sfDropdownPending`/`node._sfDropdownCursor`（**节点内存**），不存未注册设置。因为列表是每节点的（prompt_tags 的 #list 跨节点共享才需要设置），而写 node.properties 会把每次 Run 标 modified（Seed 陷阱）。
- 语义照搬 prompt_tags：`pendingIndex` 掷出牌存 `_pending`（同一次 queue 的多次 graphToPrompt——Export/保存/校验失败——都发同一张）；`api.queuePrompt` **成功后才** `commitPick`（`_pending` → `_cursor`，清空 `_pending`）。
- **首轮语义**：Increment 首轮发"节点正显示的条目"（`shownIndex` = pending ?? cursor ?? index），之后 +1 回绕；Random 首轮也真随机（不套用 increment 的首轮分支，否则面板承诺"每次随机"而首轮发面上那条）；手工选择（箭头/点选）清空 pending+cursor，从选中处重新开始。
- Fixed 模式不产生游标（`commitPick` 把 cursor 置 null）；空列表 `pendingIndex` 恒 0、注入 value null → Python 发类型 fallback。
- commitPick 后只 `renderRow`（DOM），绝不写序列化状态；`_pending` 只在仍指向真实行时有效（删行后失效）。

### 3. 输出点对齐双渲染器（本项目首个对齐节点，两个独立机制）

- **CLASSIC：硬编码 `output.pos`**。`getConnectionPos` 原样返回 `node.pos + slot.pos`，且自动堆叠跳过已 positioned 输出。**MIND THE MARGIN**：Legacy 按 `widget.margin`（默认 10）内缩 DOM widget 的 ELEMENT——元素画在 `node.pos + margin + widget.y` 而 `widget.y` 不带 margin，点在 `widget.y + ROW_H/2` 会整体高 10px（26px 行上几乎是顶边）。原版真出过这个 bug，用户一眼抓到。`arrange()` 第二遍重测槽位（widget.y 就位后）。
- **NODES 2.0：DOM nudge**（无官方方式——NodeSlots.vue 把所有输出渲染在右上角列）。三步且**顺序要紧**：① 槽位定一行高（改变块高度）→ ② `block.marginBottom = -offsetHeight`（把尺寸正确的块拉出文档流）→ ③ `translateY` 点上行。**先测块后定尺寸 = 少拉一行、点神秘偏高**（Control Panel 追了几个回合的 bug）。样式写 LAYOUT px 而 getBoundingClientRect 返回 SCREEN px（节点被图缩放 CSS 缩放）→ 从已知 layout 高度的元素量比例换算，任何缩放下正确。全程 try/catch：失败点回角落、节点照常工作。
- **350ms 自愈 poll**：MutationObserver 不够——Vue 重渲染**替换**节点元素，静默孤立旧 observer。`alignOutput` 无变化早退，稳态成本一次 rect 读取。**不门控 isVueNodes**：渲染器可在节点已存在时切换（切换不重跑 onNodeCreated），两种渲染器下都跑 poll、Classic 早退。
- **serialize 剥离 `output.pos`**：Legacy 把它写进工作流文件，Nodes 2.0 不认 → 两渲染器保存的文件不同 → 干净工作流打开即 modified。每次 arrange 重建，剥离无损。
- 行上用 `widgets_start_y = 2` 钉顶（否则 widget 从量得的槽界之下开始 → 输出点依赖 widget.y 而 widget.y 依赖槽界 → 节点每帧长高）；`computeSize` 返回 `[MIN_W, bodyHeight()]`（**绝不 this.size[0]**——computeSize[0] 也是拖拽下限，返回活宽度会让下限随加宽垫高、节点只能长）。

### 4. 弹出列表：zoom 跟随 + 三关闭

- 列表在 `document.body` 上 `position:fixed`，**不继承画布 transform**：canvas 缩放到 1.5x 时节点 DOM 行跟着长、固定 12px 的列表在旁边读着像芝麻。**根 font-size 按 `app.canvas.ds.scale` 缩放**（钳 1..2.5），内部尺寸全 em 联动。缩放字体与自适应宽度必须一起做（缩放文字 + 锁锚点宽 = 重新切掉刚放开的行名）；**字体先设再测量**（向上翻转分支读 offsetHeight，依赖已应用的字体）。
- 锚点宽是**最小值**不是宽度（内容可增长，长名显示全）；**CSS min-width 压过 max-width** → 先算 maxW 上限、minW 在它下面钳（宽节点高缩放时锚点 rect 能超上限，否则 1409px 列表挂 1350px 窗口）；left 在宽度已知后钳；下方不足向上翻转。
- 三关闭：外部 pointerdown（CAPTURE 阶段，只豁免 field 本身而非整行——整行豁免会让类型标签/间隙/输出点预留内边距点击关不掉列表）、Esc、**wheel**（坐标写一次、画布移动即搁浅，所以行上滚轮也必须关；只豁免列表本身的滚动）。

### 5. 双端数字语法契约（THE PARITY RULE，第三个同款案例）

- **`_NUMBER_RE` 是契约，两侧原生解析器都不是**：`Number("0x10")` = 16 而 Python `float("0x10")` 拒绝；Python 接受 `"1_0"` 而 `Number("1_0")` 是 NaN——parity 实测抓到，正则才一致。`[0-9]` 而非 `\d`（Python `\d` 还匹配全角/阿拉伯印度数字，JS `\d` 严格 ASCII）。
- **`_JS_WHITESPACE` 对齐 JS trim**（Python strip 集合两边都不对）：JS trim 含 U+FEFF（Excel CSV / BOM 文件粘贴带的）而 Python strip 不含；Python strip 掉 U+001C..U+001F/U+0085 而 JS 保留。
- **half-away-from-zero 取整**：Python round 银行家舍入（2.5→2）、Math.round 向 +∞（-3.5→-3），每个精确半值都分歧。
- **1e12 钳制**（对齐 Control Panel `_value_of`）：超钳制的值 `readable` 判 False（面板打 ⚠）——只解析不算可读，否则 15 位种子无警告而运行发 1000000000000。
- 测试同用例同期望值：Python 直测 `_as_number`/`readable`/`coerce_value`/`selected_value`，JS 复制 lib 为 .mjs 直测同批断言（数字语法 19 例 + readable 10 例 + coerce 17 例）。
- **防御增强**：text 分支 `str(raw)` 兜底包 RecursionError（深嵌套容器）→ 空串；`parse_state`/`_loads` 的 `json.loads` 捕获 RecursionError（C 实现约 10 万层才抛，纯 Python 更早）。实测 2000 层嵌套 str() 不炸（CPython 3.11+ 迭代式 repr），10 万层由 json.loads 防御兜住。

### 6. 切换类型断线（slotAccepts + isGraphLoading）

- Python 声明 ANY，前端改 `node.outputs[0].type`（STRING/INT/FLOAT/BOOLEAN）让画布拒绝不兼容拖拽——背后没有第二次服务端检查（与 Switch/Control Panel 相同）。
- 切换类型/Import 改类型时 `dropIncompatibleLinks`：**`slotAccepts` 而非 `===`**——ComfyUI V3 起多类型输入以字面 `"FLOAT,INT,BOOLEAN"` 到达（核心 Math Expression），相等测试读成一个未知名字、剪掉用户刚画的线；通配 `*`（Reroute/Set/Get）任一侧都接受。**只在真实用户动作时剪**（加载期间 `isGraphLoading()` 返回即 0——已保存图定义上自洽，剪 = 打开文件就损坏）。
- **isGraphLoading**：包装 `app.loadGraphData`（打开/切页/undo 唯一漏斗）一次 + 300ms 尾窗——LiteGraph 在节点 onConfigure 返回**之后**才图级别恢复已保存的线（sf_image_resize 的三重守卫同款结论）。
- 剪线/坏行都要 toast 明说（"N wires were unplugged; M entries send the fallback"）——静默剪线 = 工作流悄悄停转。

### 7. 移植简化与测试

- **shared 依赖裁剪**（第四个同款案例，惯例确认）：`isVueNodes`/`applyAdaptiveCanvasOnly` 内联（sf_pause_text.js 同款）；省略 accent 颜色（CSS 固定 `#f66744`）、XY Plot sweep provider（sfnodes 无 XY Plot）、帮助系统、registerNodeSettings 中央注册（右键菜单用 LiteGraph 原生 `getExtraMenuOptions`，any_pack.js 先例）；`popupZoom`/`placeZoomedPopup`/`installCanvasZoomPassthrough`（滚轮设置项固定默认：可滚动区域滚动、否则转发 canvas）内联；gear 图标 data URI（无资产服务路由）。
- **面板保留全量**：类型/模式/列表增删拖拽排序（grip 是 draggable 元素而非行——行 draggable 让 e.target 是行、守卫不匹配、拖拽静默无效且劫持文本选择）/autoGrow 值框（空框钉单行防 placeholder 撑高）/⚠ 警告列（readable）/Export/Import（含类型变更断线提示）/Clear（**面板内**确认框——放 document.body 会被外部点击关闭器第一击带走；Esc 听 window capture 先于面板的 document capture 应答提问）/面板拖拽（pointer capture 双防线）+ rAF 跟随 canvas（用户拖动后停止跟随）。
- **测试**（2 个文件，62+86 断言）：后端 `test_dropdown.py` 直测纯逻辑（数字语法 23 例/readable 14/coerce 16/parse_state 8/selected_value 12，含 10 万层深嵌套与 300 位大整数钳制）；前端 `test_dropdown_js.js` 复制 lib 为 .mjs 测同批 coerce 用例 + 状态归一（**writeState map 归一 null 行、readState filter 丢弃——写读路径不同，测试各验一个**）+ 游标全套（fixed 不动/increment 首轮+推进+wrap+pending 持有+commit 花牌+手工选择压过/random 20 轮无连续重复）+ syncOutput 四类型 + slotAccepts 多类型/通配。首次测试失败全是断言错（引用比较 vs 深度比较、map 归一误解为丢弃），代码零改动。

### 8. 模块边界（复用/修改时的快速索引）

- `sf_utils/dropdown.py`：纯函数（`normalize_type`/`_as_number`/`_round_half_away`/`_number_to_text`/`readable`/`coerce_value`/`parse_state`/`selected_value`），无 ComfyUI 依赖可直接单测；`TYPES`/`FALLBACKS` 是线格式（JS 与 hidden 输入共用，勿改名）。
- `nodes/text/dropdown_value.py`：`ValueDropdown` 薄封装（hidden `DropdownState` + ANY 输出 + `run`），`_CATEGORY = "sfnodes/text"`。
- `web/sf_dropdown_lib.js`：纯函数（coerce 镜像 `normalizeType`/`readable`/`coerceValue`/`previewText` + 状态 `readState`/`writeState`/`defaultState` + 游标 `pendingIndex`/`commitPick`/`shownIndex` + `injectedState`/`syncOutput`/`slotAccepts`）——无 app/DOM，测试 copy 直跑。
- `web/sf_dropdown_ui.js`：节点面 DOM（`buildRow`/`renderRow`/`step`/`cycleMode`）+ 弹出列表（`openPopup`/`closePopup`）+ 对齐（`alignOutputLegacy`/`alignOutput`/`scheduleAlign`/`watchAlign`/`unwatchAlign`）+ 内联 shared（isVueNodes/applyAdaptiveCanvasOnly/placeZoomedPopup/installCanvasZoomPassthrough）。
- `web/sf_dropdown_settings.js`：浮动面板（`openDropdownPanel`/`closeDropdownPanelFor`）+ 内联 isGraphLoading/dropIncompatibleLinks。
- `web/sf_dropdown.js`：主扩展（beforeRegisterNodeDef 全钩子 + serialize 剥 pos + getExtraMenuOptions + graphToPrompt 注入含子图复合 id 递归索引 + api.queuePrompt commitPick，防重标志 `app._sfDropdownQueuePatched`）。
- 数据契约：hidden `DropdownState`（lean 注入 / full 兜底）；`node.properties.dropdownState`（随工作流保存）；游标 `_sfDropdownPending`/`_sfDropdownCursor`（节点内存，不序列化）。

### 9. 分类支持（version 2，随 SFTextDropdown 移除加入）

> 背景：SFTextDropdown（分类 + 别名 + 全局 API 轮询配置）被移除前，把分类能力并入 SFValueDropdown，一次迁移 85 条 6 分类数据。状态模型升 version 2，旧工作流无缝归一化。

- **数据模型**：`{version:2, categories:[...], category:"default", options:[{name,value,category?}]}`。行 category 缺省 "default"；`categories` 权威有序（去空去重、default 恒在首位），**行里出现列表外的分类（手改文件）补进去而非丢数据**；`category` 不在列表 → 回退 categories[0]。旧 v1 状态 readState 自动归一（全归 default，行为不变）。
- **index 语义收紧**：index 与游标永远是"当前分类过滤后"（`visibleOptions`）列表内的索引——**切分类必须 `writeState({category, index:0})` + 清 pending/cursor**（旧索引属于另一分类的行）。Python `parse_state` 镜像过滤（full 兜底按 category 过滤后取 index）。
- **lean 注入不变**：`{version,type,value}` 仍是缓存键——切分类/改分类名/重排分类不重跑图（选中值变了才重跑）。分类是"组织/显示"状态，不是结果。
- **行归属 = 添加时的面板当前分类**，行级不改分类；新建/重命名/删除只在设置面板（节点面按钮只管切换）；删除分类选项并入 default，default 不可删（TextDropdown 同规则）。
- **Import 兼容三类**：新格式（categories + 行 category）/ 旧格式 / 任意 `{options}`——categories 缺省从行 category 收集，default 保证存在；Export 带全部分类。**面板 `commit()` 必须重渲染分类区（renderCatBtn）**——Import/Clear 只走 commit，漏了分类按钮显示旧分类。
- **节点面分类按钮（两个 UI 坑）**：① 按钮是 flex 容器时 `text-overflow: ellipsis` 对直接文本**不生效**（匿名 flex item 只硬截断）→ 文本必须包 span；② 行内固定宽度（flex:none 项合计）超过窄节点宽会把行尾类型词/输出点挤出节点右缘（Vue 默认节点宽约 200px，行固定宽曾 222px 越界）→ cat 按钮 `flex:0 1 auto; min-width:24px; max-width:84px` 可收缩 + 行尾 `padding-right:16px` 给输出点让位。**输出点 X 的三次修正轨迹**：size[0]（右半越界）→ size[0]-12（Classic 压上类型词，行有 margin+padding 内缩）→ size[0]-10 贴边；Vue 点列越界时 translateX 最小内移 2px——点列紧贴类型词，留 10px 大边距 = 直接重叠。
- prompt_reader 的 `_pix_dropdown_extract` 只读 lean 形状（注入值），不受分类影响；其 full 兜底分支不感知分类（手写 API 文件极端场景，记录为已知限制）。

---

## 16. SFPromptReader：PNG/视频元数据提示词恢复（复刻 Pixaroma Prompt Reader）

> 背景：复刻 PixaromaPromptReader（2026-08）。核心能力：读图片/视频元数据里的正向提示词（ComfyUI workflow JSON 或 A1111 parameters），graph walker 从 sampler 反推文本链。踩坑集中在**视频元数据的真实二进制格式**（文档与实测不符）与**目录切换状态字段撞名**。

### 1. 三种元数据容器，全纯标准库解析

- **PNG**：PIL `img.info` 读 tEXt/iTXt chunks——`prompt`（ComfyUI workflow JSON）/ `workflow` / `parameters`（A1111）。
- **MP4/MOV/M4V**：ISO BMFF box 链 `moov→(udta→)meta→keys+ilst`。**ffmpeg 系布局（VideoHelperSuite 用 `-movflags use_metadata_tags` 写入）与 iTunes 不同：ilst 每个 item 的 4 字节是 1-based INDEX 指向 keys 列表，不是 4cc**（实测 ffmpeg 9.0：item 头 `\x00\x00\x00\x01` 对应 keys[0]=prompt）。判定：`1 <= idx <= len(keys)` 视为 index 风格，否则按 4cc（iTunes ©too 等）兼容。**注意 ffmpeg 默认不写非 4cc metadata 键**——不带 `use_metadata_tags` 时只有 `©too`，prompt 直接丢失。
- **WebM/MKV**：EBML `Segment→Tags→Tag→SimpleTag`（TagName 0x45A3 + TagString 0x4487）。**键名按 Matroska 规范大写**（PROMPT/WORKFLOW）→ 读取归一小写。EBML ID 变长（首字节前导 1 位的位置决定 1-4 字节，0x80→1 / 0x40→2 / 0x20→3 / 0x10→4）；size vint 全 1 位 = unknown size（Segment 常见，按"延伸到容器尾"处理）。
- **流式扫描**：MP4 逐 box 读 8 字节头、非 moov 用 seek 跳过（size==0 时 size = fsize - 当前指针 + 8）；EBML 只进入 Tags 容器链、其余（Cluster/Tracks 等）seek 跳过——多 GB 视频廉价。**调试教训：EBML walker 的进入集合漏了 Segment（0x18538067），walker 从不进 Segment 导致 webm 永远解析为空**——用 ffmpeg 生成真实文件（`-f ffmetadata -i meta.txt -map 0:v -map_metadata 1`）再 dump 二进制验证。

### 2. graph walker 反推 sampler 正向文本链

- 启发式：`_TEXT_KEYS`（text/text_g/text_l/string/...）+ `_TEXT_KEY_RE`（`text_X`/`string_X`/`prompt_X`）+ `_COND_LINK_KEYS`（conditioning/positive/from/...）；sampler 用 `/sampler/i` 正则匹配类名；DFS 从 `inputs.positive` 往回走，visited 集合 + 深度 24 防环。
- 特判分支（mux/switch/隐藏状态节点，启发式覆盖不到）：**Pixaroma 生态 8 类全保留**（SwitchState 选行 / PromptStackState 拼接 / PromptMulti activePrompt / Pack / Dropdown lean+full 双形状 / FromList 索引 / PromptState+text_in 拼接 / SwitchSource 按 origin_slot 选行 / rgthree any_NN 数字序）——读别人用 Pixaroma 生成的图仍可恢复；**sf 自家节点共享同构分支**（SFPromptTags≡PixaromaPrompt、SFValueDropdown≡PixaromaDropdown），其余直读（SFTextPreset presets_json / SFAnythingIndexSwitch index / SFPauseText continue 盒子文本 / SFPromptList 拆行 / SFPromptPreset 基础文本）。
- **自追链**：PromptReader 节点输出是运行时值，embedded workflow 只存 `inputs.image` → 解析源文件元数据递归（最多 5 层）；源图缺失时给专属提示（"source image is no longer in the input folder"）而非通用文案。**无 sampler 或读不到文本返回 None，让调用方走 A1111 fallback**。

### 3. 目录切换（IN/OUT）：状态字段撞名坑

- output 文件值拼 `" [output]"` 注解全链贯通：`folder_paths.get_annotated_filepath` 原生解析（无需后端改动）、extract 路由 allowed_roots 已含 output、`/view` 缩略图按注解选 `type`、下拉分组/显示剥离注解。
- **坑（真 bug）：目录状态最初存 `promptReaderState.source`，与 `applyResult` 写入的"提取来源"（comfyui/a1111）撞名被覆盖** → `currentSource` 读到 "comfyui" 误判为 input → 切换后上传不切回。**状态字段名必须避开同一 state 对象里其它写入者的键**，改用 `folder`。
- **output 模式下上传/拖拽/粘贴自动切回 input**（上传的文件落在 input/）：`ensureSourceIsInput` 返回 true 时调用方不再重复刷新；原生 drop（无注解值）走 widget callback 防御分支。
- 列表路由按媒体类型分流：SFPromptReader 自己 `/api/sfnodes/prompt_reader/list`（image+video），SFLoadImageResize 复用 Load Image Browser 的 `/api/sfnodes/images/list`（image-only）——别把 video 塞进图片节点。
- 加载恢复：值带 `[output]` 注解或 `state.folder==="output"` → 拉 output 列表（initReadout 统一 setupNode/onConfigure 两入口；SFLoadImageResize 在 onConfigure microtask + setup queueMicrotask 各挂一次）。

### 4. 上传路径 MIME 过滤必须与 accept 同步放宽

- `accept="image/*,video/*"` 但 drop handler 仍 `file.type.startsWith("image/")` → mp4 拖入**静默无反应**（上传按钮正常，造成"按钮能传、拖动不行"的割裂体验）。
- 过滤与 accept 一致 + **type 为空放行**（浏览器对 .mkv 等未知扩展不给 MIME，交给后端上传决定）。

### 5. 其余要点

- `IS_CHANGED` 用 (mtime, size) 而非全文件哈希（50MB PNG 每 run 哈希浪费；mtime+size 覆盖所有现实编辑）；`VALIDATE_INPUTS` 恒 True（缺文件/无元数据走输出文本不阻塞图）。
- `node.imgs` 抑制（image_upload 会拉预览，本节点只要文本）：`defineProperty` 前**探测 configurable**（不可配置时 console.warn 一次而非静默吞错）。
- extract 请求 reqId 单调递增防乱序（快速连点文件下拉时旧响应不覆盖新读出版）；wired filename 跟随用 350ms poll（Vue 无 onDraw）；`isGraphLoading`（loadGraphData + 300ms 尾窗）防加载路径误触发手动接管。
- **前端 upload 不复用 sf_load_image_api.js 的 uploadImageToInput**：其 `setSelectedImage` 依赖 SFLoadImage 专属预览链（updateNativePreview 写 node.imgs），与本节点 imgs 抑制冲突——自写最小 upload（fetch /upload/image + 更新 options + w.value）。

### 6. 模块边界（复用/修改时的快速索引）

- `sf_utils/prompt_reader.py`：纯函数（`read_png_text_chunks`/`read_video_text_chunks`（MP4+EBML 解析）/`resolve_input_image_name`（裸名/注解/子目录，50000 扫描上限）/`extract_positive_from_comfy_prompt`/`extract_positive_from_a1111`/`read_prompt_from_image` + walker 全套），无 ComfyUI 依赖（folder_paths try/except 降级），PIL 仅用于 PNG chunk。
- `nodes/text/prompt_reader.py`：`SFPromptReader` 薄封装（INPUT_TYPES image combo + optional filename 接线 / `_effective_name` / `read` / `IS_CHANGED` / `VALIDATE_INPUTS`），`_CATEGORY = "sfnodes/text"`。
- `nodes/text/prompt_reader_routes.py`：两路由（`extract` 实时读出 + `list` 目录列表）+ `_is_path_under` realpath 穿越防护（input/output/temp）；`_list_media_recursive` 模块级纯函数。
- `web/sf_prompt_reader.js`：单文件主扩展（buildRoot 含 IN/OUT toggle / upload / 下拉分组 / readout + Copy / 拖拽 / wired 跟随 / PageUp-PageDown / 状态 `node.properties.promptReaderState`：filename+found+text+message+source+folder / reqId 竞态防护 / node.imgs 抑制）。
- 数据契约：hidden 无（pick 值直接走 image combo）；`/api/sfnodes/prompt_reader/extract?filename=`（恒 200，`{found,text|message}`）；`/list?type=input|output`（纯相对路径数组，注解由前端拼）。

---

## 18. SFLoadImagesPath 目录切换：三源 + 渐进式浏览 + popup 下拉

> 背景：SFLoadImagesPath（批量加载图片）原为单 combo（folder 列表含 input/output/images 前缀 + 一级子目录）。2026-08 改造为 Pixaroma 风格：源切换三档（input/output/images）+ 渐进式目录浏览（面包屑 + 按需加载）+ 直接输入路径模式 + SFLoadImageResize 风格 popup 下拉。

### 1. 设计决策

- **folder combo 值 = 唯一事实来源**：隐藏原生 combo（值随 workflow 保存 + graphToPrompt 自动收集），前端 DOM UI 读写其 value；组合校验由 `VALIDATE_INPUTS` 接管（动态值）。
- **显式模式状态**：`node.properties.sfLoadImagesPathMode`（"dir"/"path"）。**不能用值推导**——路径模式下值可能仍是目录格式（如 "input/faces"），切到路径模式后值不变会被误判回目录模式。properties 随 workflow 保存、不注入 prompt（无缓存影响）。
- **选中 = 当前位置**：folder 值恒等于面包屑路径（`source/path/...`）；下拉/步进/面包屑回退都是改值，无"停在父层选中"概念。
- **同级切换不改变层级深度**：◀▶ 定位父层 → fetch 父层子目录 → 替换面包屑末段；根层（无父层）按钮禁用。与"下拉选择 = 进入子层"（追加段）语义分离（`switchSibling` vs `enterSubdir`）。

### 2. 渐进式按需加载

- 后端 `GET /api/sfnodes/images_path/subdirs?folder=`：**复用 `_resolve_folder` 解析**（前缀/绝对路径/包含性安全校验一套逻辑），只列当前层一级子目录（隐藏目录过滤在 `_list_one_level_subdirs` 统一）。
- 前端同值缓存（`_lastFetched`）：重复渲染/恢复不重复请求；刷新按钮/打开 popup 时 `force=true`。竞态用 reqId 单调（快速切换丢弃旧响应）。
- 渲染只读**不加 isGraphLoading 门控**：门控会在 300ms 尾窗内跳过恢复渲染，尾窗后无触发 → DOM 停在初始状态与保存值不同步（渲染不写序列化状态，门控多余且有害）。

### 3. popup 下拉骨架（SFLoadImageResize 风格，选目录版）

- 触发按钮 `[◀] [ 📁 当前目录名 2目录 ▼ ] [▶]`：name 显示当前目录（末段/源根）、counter 显示子目录数——**仿 SFLoadImageResize 的 name+counter 结构，两个渲染函数写不同元素**（曾把 name 同时写"目录名"与"X 个子目录"导致互相覆盖）。
- popup：锚点 getBoundingClientRect 下方 fixed 定位、宽度 max(锚点宽, 240)、头部显示完整路径、列表项点击即进入并关闭；空目录显示"（无子目录）"。
- 关闭机制（同 SFLoadImageResize）：外部 mousedown/pointerdown/wheel（capture）+ Escape；`_openPopup` 引用 + `removeEventListener` 全摘（`_sfClose` 式清理，防泄漏）。
- **不用 innerHTML 建子元素**：mock DOM（tests）不解析 innerHTML 字符串，显式 `createElement`/`append` 真实 DOM 与 mock 一致（曾因 `trigger.innerHTML = '<span class="name">…'` 在 mock 下拿不到子元素）。

### 4. 空目录语义

- 空目录/目录不存在不再抛 `FileNotFoundError`：返回 `torch.ones((1,64,64,3))` 占位图 + 全 0 mask + `frame_count=0` + 空文件名列表。占位图保证下游（反推等）拿到可处理张量；count=0 + 空列表明确"没有内容"。`VALIDATE_INPUTS` 保留面板提示（配置期提示，运行宽容）。

### 5. 测试要点

- smoke 测试的 **async 步进需 await**：`() => stepSubdir(prev)` 箭头函数返回 promise，`await _handlers.click()` 才能拿到完成后的值。
- mock 增强：`document.addEventListener` 记录（Esc 关闭断言）、`body.appendChild` 记录（拿 popup 元素）、`innerHTML` setter 清空 children（模拟真实 DOM 重建）。

### 6. DOM widget 高度/宽度溢出修复（2026-08）

- **现象**：初始添加节点"刷新当前目录"按钮就落在节点边框外（下方），拖窄后按钮行横向溢出。根因：widget 高度硬编码 `getMinHeight/getMaxHeight = 138`，而 DOM 内容实际高度 ~140-190px（6 行 + padding 16 + gap 36，随 dir/path 模式切换变化）。**DOM widget element 高度是内容自适应的（CSS 无显式 height），节点边框却按 widget 声称的高度绘制**——声称 < 内容时底部内容必然溢出边框；节点未被拖小时被更大的边框遮住（假正常），初始添加/拖小即暴露。
- **主修复 = 动态测量内容高度**：`measureHeight()` 求和各可见子行 `offsetHeight` + padding 16 + gap 6×(行数-1)，`getMinHeight/getMaxHeight` 双锁改用它（保持原"锁定高度"语义但锁在正确高度）。**last-good 缓存防塌缩**（首帧未布局/组折叠隐藏时 offsetHeight 全 0 → 返回上次良好值，初始 138）——sf_load_image `_lastGoodH` 同款，硬编码常量做兜底值而非目标值。
- **Nodes 2.0 双保险**：`widget.computeLayoutSize = () => ({ minHeight: measureHeight(), minWidth: MIN_W })`——Vue 前端忽略 legacy getMinHeight/getMaxHeight，改走 computeLayoutSize；顺带借 `minWidth` 兜住拖拽宽度（Vue 下 onResize 不可靠）。
- **宽度钳制**：`MIN_W=320`（源三档按钮行 + 面包屑行容纳所需）。初始只抬升过小的尺寸（已保存宽度永不变更 → 不脏加载）+ 实例包装 `node.onResize` 钳制（legacy 拖拽路径）。
- **CSS 兜底**：`.sf-lip-btn` 加 `min-width:0; overflow:hidden; white-space:nowrap; text-overflow:ellipsis`（按钮收缩省略而非撑破边界）；`.sf-lip-root` 加 `overflow:hidden`（最后防线；popup 挂 document.body 不受影响）。
- **顺带清理**：移除冗余的"📁 当前路径"显示行——它曾躺在 138px 裁剪区外被遮挡，修复后露出才发现与面包屑导航重复（面包屑 + 下拉按钮 title 已承载路径）；按钮文案统一英文化（Folder Mode / Path Mode / Apply / Refresh）。

---

## 23. SFPromptList：行号编辑器与 wrap 镜像测量

> 背景：`nodes/text/prompt_list.py` + `web/sf_prompt_list.js`（2026-08）。行拆分/切片/空白行过滤（skip_empty）的文本节点，前端提供带行号的 DOM 编辑器。

- **值真源 = 隐藏原生 multiline_text widget**（值随工作流保存、graphToPrompt 自动收集），DOM widget 行号栏只是视图：行号从 0 起、跳过空白行对齐输出 index、超 500 行虚拟化、值恢复三通道。
- **wrap 开启走镜像测量**：mirror 与 textarea 同几何的块级 div，行高按行缓存、宽度变化清空；渲染后强制重同步 scrollTop（浏览器对超长文本的 scrollTop 钳制会错位）。
- **start_index/max_rows 切片范围高亮跟随**：仅裁剪时文本背景块 + 行号联动；wrap 开时高亮随测量行高展开（与行号同源）。
- 测试：`tests/test_prompt_list_lines_js.js`（行号/高亮对齐）+ `test_prompt_list_smoke.js`。

---

## 24. SFPromptStack：动态 Prompt 列表与行高拖拽

> 背景：`nodes/text/prompt_stack.py` + `web/sf_prompt_stack_core.js`（纯逻辑）+ `sf_prompt_stack.js`（行 UI）（2026-08）。行动态添加/每条开关/右下角角标拖拽调行高。

- **状态对齐 Pixaroma PromptStack 形状** `rows/enabled/text`：prompt_reader 的 graph walker 恢复共享该形状，改形状会破坏跨插件恢复。
- **行高 `state.rows[i].h` 随工作流保存**：右下角角标拖拽调行高；核心 UI 逻辑在 `sf_prompt_stack_core.js`（无 app/DOM 可直测），`sf_prompt_stack.js` 只做行渲染。
- 测试：`tests/test_prompt_stack_core.js`（纯逻辑）+ `test_prompt_stack_smoke.js` + `test_prompt_stack.py`（后端行拆分）。

---

## 29. SFPromptPreset：十一分类组合预设（正交原则 / 随机机制 / LLM 优化链路）

> 背景：`nodes/text/prompt_preset.py` + `web/prompt_preset.js` + `data/prompt_presets.json`（2026-08）。11 分类 949 预设（分类内分组 58 组次、跨分类去重 56 组）组合提示词的预设选择器，针对 Krea2 Turbo 优化（自然语言、无 SD 质量标签、分类正交），兼容 SD/Flux。原 `doc/prompt_preset.md` 使用指南已随本归档移除（输入/输出、分组明细以节点 DESCRIPTION、输入 tooltip、弹窗 UI 与数据文件为准），经验沉淀如下。

### 1. 正交原则（数据设计核心，防组合污染）

- 各分类 prompt **不得内嵌其他分类职责**，否则组合互相污染（动作内嵌场景/灯光、服装内嵌灯光、风格内嵌镜头参数、名人内嵌风格、动作内嵌裸体都是反例）；允许例外：动作要素（靠墙必须有墙、坐床边必须有床）、场景固有照明（停车场荧光灯）、风格光效特征（巴洛克戏剧光）、服装场合属性（通勤装/沙滩装）。
- **NSFW 动作正交**：Pose/Couple Pose 的 NSFW 组只描述动作（姿态/情绪），不内嵌裸体词——裸体由服装分类"全裸"（或 NSFW 服装）控制，配合任意动作。曾有的"全裸站立"预设因去裸体词后与"站立肖像"重复已删除；旧工作流该值经 `VALIDATE_INPUTS` 降级为空串。

### 2. 随机机制与可复现性

- 三类随机：全分类随机（`随机`）、组内随机（`随机·组名`）、`[选项A, 选项B]` 括号随机（input_text 内也支持，由种子决定选取）。
- **`IS_CHANGED = seed`**（固定 seed 可复现，seed 变化自动重跑）；**seed 偏移 +1..+11**（名人 +1 … 镜头 +11）——同 seed 下各分类取不同伪随机序列，避免所有分类同值同步。
- **pose/couple 互斥**：前端联动（启用一个禁用另一个）+ 后端兜底（同时启用保留 pose，防旧工作流/手写 API）。

### 3. Krea2 grounded phrasing：空间介词引导

- 环境片段自动加 `in the` / `on the` 前缀：`_ENV_ON_PREFIX` 正则匹配表面类环境（rooftop/beach/street/ground/bridge 等）用 `on the`，其余用 `in the`——让主体明确"身处"场景之中，避免人物与环境贴图感、比例互动失调（对 Krea2 这类自然语言模型尤其重要）。

### 4. LLM 优化链路（optimize_request → 官方 TextGenerate）

- optimize_request = 创作声明（虚构艺术任务、中立事实化描述）+ 11 条指令（第 11 条为 few-shot 示例）+ `{}` 占位拼接原文 + 末尾 `Optimized prompt:` 续写锚定（LLM 从锚定处直接生成单段提示词，杜绝解释/前言/后缀）。指令含：短语流顺序、主体-环境 grounded、简洁去重、禁质量标签、忠实原文、**保持衣着水平（不添加/不纠正衣着，即不采用强制穿衣策略）**、单段纯净输出、轻量润色、**防拒条款**（宁可直接回显 draft 也不输出拒绝文本）、**防思考条款**。
- 强对齐模型（Gemma 官方版）即使有防拒条款也可能仍拒绝（安全拒绝发生在训练层，措辞无解）→ 换无审查本地 LLM；**Qwen3 系 thinking 参数选择见 §5**（instruct 版 off、总是自发推理的无审查微调版实测 on 反而正常；`use_default_template` 保持 True；CLIPLoader 类型选 `qwen3vl_4b/8b`）。

### 5. 数据与兼容性

- `data/prompt_presets.json` **热加载**：`_load_presets` 按 mtime 检测 + 线程锁，编辑后无需重启容器。自定义预设字段：`name_zh`（下拉显示名，同分类唯一）/ `prompt`（英文自然语言、遵守正交原则）/ `description`（悬浮说明）/ `tags` / `weight`（加权随机权重）/ `group`（弹窗筛选分组，任意字符串按出现顺序展示）。
- **破坏性变更**：旧版输出 12 条 STRING → 现 3 条（combined_prompt / prompt_pack / optimize_request）；还原 11 条分类文本需接 **SFUnpackPromptPreset**（顺序与旧版一致）。prompt_pack 是运行时对象（SF_PROMPT_PACK），**不可**接入 Primitive/保存类节点。
- 预设被删除/改名的旧工作流 combo 值超出静态选项列表 → `VALIDATE_INPUTS` 恒 True + `_resolve_preset` 安全降级为空串（动态 combo 校验通用模式，见 §4）。
- **伦理**：数据含 NSFW 预设（仓库分发注意许可与政策）；名人 + NSFW 组合存在肖像权/伦理风险；亚洲名人（22 个）无社区实测依据。
- 测试：`tests/test_prompt_preset.py`（后端 200+ 断言）+ `test_prompt_preset_js.js`（前端 40+ 断言）。
