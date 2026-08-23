# 经验归档：LoRA / Civitai / Krea2 预设生态（§5、§19、§20、§21、§25、§28、§31）

> 全局章节号 §N 与拆分前的 experience.md 一致；跨节/跨文件引用一律写 §N，映射见 [README.md](README.md)。版本时效说明见 README。

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

### 4. SFImageInterrogator：thinking 透传 + 输出剥离思考块（2026-08）

> 背景：`nodes/model/krea2.py` 的 SFImageInterrogator 用 Krea2 的 CLIP 做图像反推（`clip.tokenize` + `clip.generate` + `clip.decode`）。用户反馈 Think 模型会把思考过程内容输出到结果。

- **根因**：Krea2 tokenizer（`comfy/text_encoders/krea2.py::Krea2Tokenizer`）默认 `thinking=True`（为 conditioning 设计，不注入空 think 块）——生成路径若不显式传 `thinking`，Think/无审查变体自由推理，`clip.decode` 原样返回 ` thinking...\n response\n\n最终答案`，思考内容混入结果。
- **修复（对齐原生 Generate Text）**：① 新增 `thinking` BOOLEAN 输入（默认 False）显式传入 `clip.tokenize`——False 时 qwen3vl 注入空 think 块抑制推理（仅对遵守约定的 instruct 模型有效）；② 输出剥离思考块 `re.sub(r"^\s*<think>.*?(</think>|\Z)", "", out, flags=re.DOTALL)`。
- **锚定加固（优于原生）**：原生 `TextGenerateLTX2Prompt` 的 `r" thinking.*?(?: response|$)"` 未锚定，会把正文里的 "thinking" 一词当成思考块起点截断（如 "A person thinking about the sunset" → 剩 "A person"）；本节点用 `^\s*<think>` 行首锚定（实测 Think/无审查变体运行时标记为 `<think>`...`</think>` 格式，行首锚定避免误伤正文）。**闭合标签 `</think>` 直接匹配（独特、无歧义，行内/行首均可）**；`|\Z` 覆盖 max_length 截断、未及 `</think>` 就中断的思考块。
- **空输出兜底（修正）**：`(?: response|$)` 覆盖 max_length 截断、未及 ` response` 就中断的思考块。**剥离后为空（整段都是被截断的推理，无最终回答）时直接返回空串**——曾误做"回退保留原始文本"，结果把整段推理又还了回去（正是"输出仍带思考过程"的根因，实测复现：中文长推理占满 max_length=256 即触发）。空串即"无最终回答"信号，用户应增大 max_length。
- **widget 位置**：`thinking` 追加到 optional 末尾（`user_prompt` 之后），遵守"新增 widget 一律追加到末尾"约定，旧工作流 widgets_values 不错位；纯 BOOLEAN widget 无需前端 JS 改动。
- **不触及其他路径**：TextEncodeKrea2 的 conditioning 编码不传 thinking，保持 Krea2 默认（无空 think 块），条件编码不受影响。

---

## 19. SFLoraStack：多行 LoRA 栈复刻（触发词/描述/封面/Civitai 查询/孤儿数据迁移）

> 背景：复刻 PixaromaLoraLoader 为 SFLoraStack。核心 = 多行 LoRA 栈（行级 on/off + sm/sc 强度 + 触发词勾选输出 triggers 字符串）；信息面板 = 离线元数据/触发词读取 + 可选 Civitai 查询（文本+封面本地保存）；用户数据（自定义词/描述/预览图）按 LoRA 路径名键控，文件移动/改名后的孤儿迁移是主要难点。双端模式与 SFPauseText 同构：状态存 `node.properties.loraStackState` → graphToPrompt 注入隐藏 LoraLoaderState 输入。

### 1. Civitai API 字段位置必须实测（description/thumbnail）

- **`model-versions/by-hash` 响应的 `model` 对象只有 name/nsfw/poi/type**——说明文字在 **version 顶层 `description`**（HTML 字符串）。曾按 `model.description` 提取永远为空，实测 API 后修正（顶层优先 + `model.description` 兼容兜底旧侧车）。
- description 清洗：`<br>/</p>/</div>/</li>` 换行标签转 `\n` → 其余标签剥掉 → 实体解码（`&amp;`/`&#x...;`/`&nbsp;`）→ 空白折叠 → 2000 字符截断。纯函数 `_clean_description`。
- **thumbnail 取 `images[]` 第一张非成人图**（`nsfwLevel` 位掩码 `>=4` 即成人；`allow_adult` 设置才用全显式画廊）。`nsfwLevel: 16` 的成人模型默认无封面——这是预期，不是查询失败。
- 双主机（com/red）：404 只在最后一个主机定论（成人模型主站用 404 隐藏）；401/403 绝不在循环内返回（备胎主机存在就是为了按域名屏蔽）；200 非 JSON = 屏蔽页/登录页而非"没有"。

### 2. 封面：查询时自动保存 + 确认覆盖 + 移动后静默恢复

- 查询成功 → 服务端把缩略图**下载到本地**（`user/sfnodes/lora_previews/<sha1(键)>.jpg`，与手动自定义预览同目录同名规则；https-only + 4MB cap + magic bytes 校验；失败不致命，文本照常返回）。`_download_thumb` 流式 iter_chunked 同 civitai body 模式。
- **已有用户自定义预览 → 查询不覆盖**（返回 thumb_skipped），found 后面板风确认框询问 → 确认后走独立端点 `POST /lora/civitai_thumb_save`（读侧车 `sidecar_thumbnail` 拿同一张图重下载覆盖，**无需重新查询**）。
- **面板风确认框必须豁免宿主面板的 document 捕获监听**：确认框挂 `document.body`、不在面板 DOM 内——不豁免则其事件会穿透到面板监听（Esc 连关面板、Ctrl+V 误设图）。onKey/onPaste 都要 `closest('.sf-ls-confirm-mask')` 豁免。**信息面板只经 ✕ 关闭**（画布点击不关闭，用户边看信息边操作工作流；面板随节点跟随移动）——曾有过 onDown（外部点击关闭），按用户需求移除。（2026-08 修订：按用户需求重新引入外部点击关闭——**查看态点击面板外关闭；编辑态 `_descDirty` 不关**（防误关丢草稿，与 Esc/✕ 确认保护同对象）；拖动位移 > 6px 不算点击（LiteGraph 拖动后 mouseup 也在同一 canvas 上触发 click，浏览器不查位移，必须 pointerdown 记坐标判定）；确认框/面板内点击豁免。）
- **文件移动后封面自动恢复**：预览图按路径 hash 命名，移动后 hash 失配本地找不到；`/lora_info` 检测"本地无预览 && 侧车有缩略图" → `restorable_thumb` → 前端打开面板时静默 `saveCivitaiThumb` 重下载到新 hash 名（一次会话一次，失败静默下次再试）。

### 3. 用户数据键失配与两级孤儿匹配（核心难点）

- 自定义词/描述存 `user/sfnodes/lora_triggers.json`（键 = 归一化 LoRA 相对路径），预览图按同一键 sha1 命名。**侧车（`<base>.civitai.info`）随文件走天然跟上**；user 目录数据移动/改名后失配。
- 匹配两级：
  - **内容指纹优先**：`file_fingerprint` = `(size, sha256(头64KB), sha256(中64KB), sha256(尾64KB))`，~192KB 读取；改名/移动不改内容 → 指纹不变，可作"同一文件"的强证据（顺带修正基名匹配在"同名文件被替换"场景的误配）。存储条目可选 `fp` 字段，**写入端点（词/描述）保存时由路由层计算**（存储层无文件路径）；`_norm_fp` 形状清洗、旧数据无 fp 兼容。
  - **基名兜底**：`base_key` = 去目录去扩展名，仅覆盖文件夹改名；同名多目录歧义放弃。
- **只提示不自动执行**：`/lora_info` 附 `orphan_*` 字段 → 前端迁移条（Migrate/Dismiss，Dismiss 仅本会话）→ `POST /lora/migrate` 接收前端回传 `old_key`（**防御自迁移/不存在键**：`old_key == key` 或不在 store 拒绝）→ 词/描述键转移 + 删旧键、预览图同目录 rename（目标已存在不覆盖）→ 迁移后新键补记指纹（此后改名也能找回）。
- 指纹匹配成本：仅 `has_custom=False` 且基名未命中时计算一次（~192KB/面板打开）。

### 4. 状态契约与双端镜像

- `promptState` 只注入执行字段（name/on/sm/sc/triggers + sep + cacheMode）：cosmetic（accent/thumbs/step/defStrength/linkStrength/id/custom）剥掉避免改缓存签名；**cacheMode 例外**——Python 需要它决定 run 间内存策略（切换会重跑一次，可接受）。
- `parse_state` 容错契约：sc 缺省 = sm；强度钳 [-100,100]；nan/inf → 0；空名/非 dict 行丢弃；cacheMode 未知钳 last。前端 normalize 强制 `linkStrength` 时 sc=sm（写/读双端都强制，切回单强度永不留陈旧 clip 值）。
- 存储形状升级兼容：`{key:[words]}` → `{key:{words,description,fp?}}` 读时归一（`_norm_store_entry`）；词空但描述在 → 条目保留（原语义"空词删条目"需改）。
- **cacheMode 内存管理（last/all/none）**：`last_this_run`（本次 run 最近加载）与 `self._last_path`（跨 run 保留条目）**必须分离**——本 run 第一行应用时就逐出保留条目会让 last 对任何 2+ 行栈表现得像 none（暖文件在被复用前一刻被丢掉）；last 模式保留条件 = 本次加载过（否则跨 run 条目仅在仍属栈时存活，清空栈真的释放）。

### 5. 竞态：迟到旧响应覆盖用户刚保存的值

- **信息面板 `_infoSeq++` 作废在途响应**：面板打开时 loadInfo 在飞，用户保存描述成功后迟到响应落地会把 `info.custom_description` 覆盖回旧值（"保存了仍显示来自 Civitai"——实际是用户忘了点 Save 的误报，但竞态真实存在，防御性修复）。`attemptInfo` 票号机制天然丢弃 superseded。
- **设置面板 `_accDirty` 挡 GET 迟到应答**：面板打开时 GET 账户在飞，用户先点了 host（red）保存成功，迟到 GET 旧快照（com）落地覆盖面板显示（"设了 red 它显示 com"）。保存成功置 dirty，迟到 GET 不再覆盖。
- 孤儿迁移提示条按面板会话 dismiss（文件没动则每次打开都值得再看）。

### 6. 其余要点

- **全局强调色统一（2026-08）**：ComfyUI 系统设置 `sfnodes.Accent`（combo 8 预设，注册在 SFLoraStack 扩展 init）→ `applySfAccentVar` 写 document 根 inline CSS 变量 `--sf-acc` → 全 sf 节点主题色统一 `var(--sf-acc, #f66744)`（CSS 响应式自动生效，无需逐节点重绘）→ canvas 绘制每帧 `sfAccent()` 读 inline 变量（轻量）。**无节点级自定义**（2026-08 应需求移除：设置面板颜色行/Every SF node 按钮删除、accentOf 直读全局、prompt_tags 编辑器硬编码传参移除）——旧 state/defaults 里的 accent 字段被忽略，面板/下拉/菜单的局部 `--acc`/`--sf-acc` 只是把全局色带到局部作用域。
  - **时序坑 1（onChange 参数）**：ComfyUI settingStore 的 `applySettingLocally` 先调 `onChange(t.value[n], a, o)`（参数 = newValue, oldValue）**再**更新 store（`e.value[n] = s`）——onChange 回调里读 `getSettingValue` 拿到的是**旧值**（"设了 red 显示 teal"）。必须用回调传入的 newValue。
  - **时序坑 2（注册顺序）**：初始 `applySfAccentVar()` 必须在 `addSetting` **之后**——设置项未注册时 `getSettingValue` 拿不到用户保存值（返回 defaultValue），会把 `--sf-acc` 钉死在品牌橙且注册后不再刷新（"Crop 品牌文字还是橙色"）。
  - **时序坑 3（重绘推迟）**：坑 1 的连带——onChange 里**同步** `repaintAll()` 时 store 尚未更新，`accentOf`（直读 store）渲染出旧色（"SFLoraStack 设置后不立即生效"）。必须 `setTimeout(repaintAll, 0)` 推迟到 `e.value[n]=s` 之后。
  - **时序坑 4（异步加载轮询）**：设置值从服务器加载（`V.getSettings()`）是异步的、可能晚于扩展 init——此时 `getSettingValue` 返回 defaultValue，初始 apply 会把 `--sf-acc` 钉死在默认色，且**加载完成后不再刷新**（"硬刷新后 SFLoraStack 生效、Load Image Resize 恢复橙色"——accentOf 直读 store 的节点在后续渲染时自然拿到新值，而 CSS 变量类节点不生效，**症状不一致极易误判**）。修复：初始 apply 后轮询重试数次（幂等廉价；用户改设置走 onChange）。
  - **复刻节点硬编码橙色的统一改造**：原版 accent 体系被丢弃的节点（Load Image Resize/crop 品牌文字等）全是硬编码 `#f66744`——CSS 插值改 `var(--sf-acc, #f66744)`、canvas 常量改 `sfAccent()`；sed 批量替换后必须清理孤儿常量并检查 SVG data URI/JS 逻辑误入。crop/inpaint 编辑器的 canvas 工具色（涂抹笔/裁剪框）保留品牌橙（工具色非主题标识，避免回归）。
- 查询路由打 hosts/key 日志（`civitai lookup for <name>: hosts=<顺序> key=yes/no`）——"设了 red 走 com"类问题一眼定位。
- View on Civitai 链接按账户 host 偏好生成域（red → civitai.red，成人模型在 com 网页可能受限）；`/lora_info` 附 `civitai_host`。
- `_is_path_under` 用 realpath 双端严格检查 + 跨盘（junction）lexical 回退（原版 `_path_guard` 语义）；纯 abspath 会让同盘 symlink 逃逸误判通过。
- `hideJsonWidget` 四件套（hidden + computeSize=[0,-4] + canvasOnly + element display:none）：Vue 下隐藏 STRING widget 会渲染成显示原始 JSON 的 textarea。
- 测试：纯逻辑模块（lora_reader）无 ComfyUI 依赖直跑（tests/test_lora_reader.py 百余断言，含 symlink 逃逸拒绝）；web import/export 交叉验证 tests/check_web_imports.py。

### 7. 行名显示全局设置共享（2026-08）

> 演变注：设置原为 Power 系的 `sfnodes.PowerLoraLoader.DisplayName`（注册在已删除的 power_lora_loader.js），Power 节点移除后键更名为 `sfnodes.Lora.DisplayName`、注册迁至 SFLoraStack 扩展 init，旧键直接废弃不读取。

- **单一真源**：设置 `sfnodes.Lora.DisplayName`（full/filename/basename/folder/parent_basename，注册在 sf_lora_stack.js 扩展 init）——SFLoraStack/SFLoraPlot 行名共用。逻辑收敛于 `sf_common.js`：`LORA_DISPLAY_MODES`/`LORA_DISPLAY_SETTING`/`loraDisplayName(path, mode)`（纯函数）/`getLoraDisplayMode()`/`loraRowLabel(name, hideExt)`（行名统一入口），**禁止各节点内联副本**。
- **语义与边界**：模式 ≠ full 时设置优先——basename 用 `lastIndexOf(".")` 剥**任意**扩展名（"xyz.v1.0"→"xyz.v1"，与 Stack 白名单语义不同）；full（默认）回退每节点 hideExt（白名单 `LORA_EXT_RE` 只剥模型扩展名，"xyz.v1.0" 保留）——默认行为与旧版逐字节一致，向后兼容。hideExt 仅在 full 模式参与（全局非默认时让位）。parent_basename = 上级目录名 + basename（"sdxl/style/beauty.safetensors"→"style/beauty"），根目录文件降级仅 basename（与 folder 同降级策略）。
- **事件桥重绘**：设置 onChange（sf_lora_stack.js 注册处）追加 `document.dispatchEvent(new CustomEvent("sfnodes.lora-display-mode-changed"))`——Stack 不被 Plot import（避免节点间耦合，同 lora-data-changed 先例）；sf_lora_stack.js 与 sf_lora_plot.js 各自监听 → `setTimeout(repaintAll / renderAllPlots, 0)`。DOM 行重绘必须走 repaintAll/renderAllPlots（setDirtyCanvas 只重绘画布层，管不到 widget DOM）；setTimeout(0) 推迟到设置 store 更新后（同 Accent 时序坑 3）。
- **下拉弹窗定位：方向打开时定一次 + maxHeight 钳制（2026-08）**：`sf_lora_stack_dropdown.js` 的 `place()` 演进两版：① 原本只在打开时与首次 `renderList()` 后调用，目录导航后不重算——进大目录高度增长、top 是"向下"旧值 → 底边越出视口被 `overflow:hidden` 裁掉；把 `place()` 移入 `renderList()` 末尾修复后出现新问题：内容变化反复翻转方向（一会上、一会下）打断视觉锚定。② 终版语义（Floating UI 空间感知同款）：**方向在打开时比较上下可用空间选大者定一次**（`goUp = upSpace > downSpace`，相等向下），展开期间永不翻转；`maxHeight` 动态钳到所选方向空间（`min(60vh, 方向空间)`，下限 40px）——内容超高时 list 内部滚动（`overflow-y:auto`），弹窗实际高度 ≤ 方向空间 → 恒完整可见；top：向下恒定 `r.bottom+4`，向上 `max(8, r.top-4-h)`（底边贴锚点、顶边延伸）。对可增长内容的目录导航弹窗，"选大侧"能容纳更多增长、减少翻转概率。测试 `tests/test_lora_stack_dropdown_smoke.js`：注入可写 `offsetHeight` 模拟高度增长，断言方向恒定（向上 536→36→536）、向下 top 恒定 124、方向空间 < 60vh 时 maxHeight 钳制（356px）；fileRow 文本断言查 `createTextNode` 节点而非元素 `_text`。

### 8. Description 未保存修改：Save 高亮 + 关闭确认（2026-08）

- **LoRA 信息对话框（sf_lora_info.js，SFLoraLoader/SFLoraLoaderModelOnly 共用）本已具备**（`row._dirty` 追踪 + `closeDialog` confirm 覆盖 ✕/Esc/背景点击）——只补了视觉高亮：dirty 时 Save 按钮实底主色（`#4f7cff` 底 + 白字粗体），替代纯文本 `Save*`。
- **SFLoraStack 信息面板（sf_lora_stack_info.js）完整补齐**：`_descBase`（进入编辑时 `shownDesc()` 快照）+ `_descDirty` 模块级状态；textarea `input` 事件比较草稿 → Save 按钮 `.qa.dirty`（accent 实底）高亮，`renderBody` 重建按持久 `_descDirty` 重画；改回基准自动不高亮。
- **关闭确认**：`closeInfoPanel()` 改造为返回 `Promise<boolean>`——dirty 时经同主题 `confirmDialog`（"Discard description changes?"）确认才关；✕/Esc 忽略返回值（内部异步自关）；**`openInfoPanel` 切换行 `await closeInfoPanel()`，取消则不打开新行面板**；**节点删除路径 `closeInfoPanelFor` 走 `doCloseInfoPanel` 不弹框**（删除不能被阻塞）。textarea 内 Esc（原直接丢草稿）dirty 时也弹确认（误按保护）。
- **顺带修复草稿泄漏 bug**：`_descEditing/_descDraft` 原是 openInfoPanel 闭包内变量，`closeInfoPanel` 是模块级函数读不到——提升为模块级（闭包共享）后，`doCloseInfoPanel` 关闭时统一重置；旧代码关闭面板不清状态，重开另一行会带着上一行的旧草稿直接进编辑态。
- 测试 `tests/test_lora_stack_info_desc_smoke.js`：**mock 的 className 必须与 classList 双向同步**（单向时 `el.className="qa"` 后 `_s` 为空，第一次 `toggle("dirty",true)` 的 sync 用 `_s` 覆盖 className 把 "qa" 冲掉，第二次 `querySelector(".qa")` 误匹配 Cancel——真实 DOM 没有此问题，是 mock 失真）；断言链：改动高亮 → 改回基准不高亮 → Esc 确认框 → 取消保留草稿 → ✕ 确认 Discard → 重开无残留编辑态。

### 9. 行 i 按钮 _has_custom 高亮（2026-08）

- **需求**：SFLoraStack 行的 info 按钮（`.sf-ls-info` "i"）按"该 LoRA 是否有用户编辑过的信息"高亮（用户以 `.civitai.info` 侧车为标志物）。
- **判定源与信息对话框 i 图标统一**：用 `lora_notes` 网关的 `_has_custom`（统一存储有词/描述 **或** 侧车有词/描述，lora_notes.py 计算）——比"仅 source==='sidecar'"准（lora_reader 只在侧车有 **triggers** 时置 source=sidecar，侧车仅 description 会漏判）且覆盖统一存储（面板保存的词/描述也亮）。**两节点同一数据源同一语义，跨节点高亮一致**。
- **实现**：`sf_lora_stack_render.js` 行渲染时 `getLoraMetadata(e.name).then(meta => info.classList.toggle("net", !!meta?._has_custom))`（`isConnected` 守卫：行被 renderNode 重建则丢弃，新行自己会查；缓存命中零请求，未命中 lora_notes 端点轻量）。**`classList.toggle` 传单个类名 `"net"` 而非 `"sf-ls-info.net"`**（后者是含点字符串，会作为单个非法类名添加——CSS 选择器 `.sf-ls-info.net` = 两个类）。
- **即时刷新**：`sf_lora_stack.js` init 监听 `sfnodes.lora-data-changed` → `setTimeout(repaintAll, 0)`——保存触发词/描述/封面后行高亮更新（loraMetadataCache 已被 sf_lora_info.js 的同事件监听清掉，重渲染时重新查询）。
- 测试：presets smoke 扩展（makeEl 的 classList 升级为真实 Set + className 双向同步，行 i 高亮的 toggle 依赖）；断言注意 **linkStrength=false 时行结构为 [grip, name, wm, wm(c), info, sw]**，info 是 children[4] 不是 children[3]。

---

## 20. SFLoraStack：正交堆叠 ortho_gs（2026-08）

> 背景：`sf_utils/lora_ortho.py`（纯数学）+ `lora_ortho_load.py`（加载应用，2026-08）。相似 LoRA 叠加糊脸——多个 LoRA 的 down 矩阵行空间重叠 = 干扰源。ortho_gs 把后续 LoRA 的 down 行投影到前序 down 行空间的正交补。落地于 SFLoraStack（`mergeMethod`，与 cacheMode 同模式切换）。

### 1. 数学

- ΔW = Σ s·(α/r)·(A_i·B_i)，多个 LoRA 的 down 矩阵行空间重叠 = 干扰源（相似 LoRA 叠糊）。ortho_gs 把每个 down 的行投影到前序 down 行空间的正交补（`d' = d - (d@Qᵀ)@Q`，Q 用 SVD 右奇异向量扩基 + QR 去线性相关，float32 计算）——**第一个 LoRA 不动、后续让位**，up/alpha/strength 全不动；行空间被完全覆盖时投影归零（幅度损失是 tradeoff 非 bug）。

### 2. 必须走独立加载路径

- 链式 `load_lora_for_models` 的 patch 已展开进 patcher，拿不回 up/down——ortho 需自己 `model_lora_keys_unet`(+clip) 建 key map + `convert_lora`（官方路径有，DuoNodes 漏掉）+ `load_lora` + clone + add_patches + `set_attachments("lora_metadata")`，**按模型 key 分组**（同 key 多 LoRA 才 GS，单条直通），非 LoRA patch（conv/diff/set）该 key fallback 顺序；key map 构建失败整体 fallback 顺序，绝不报错。
- 加载+应用路径收敛在 **`lora_ortho_load.ortho_apply(model, clip, entries, load_sd)`**——调用方传自己的 sd 加载函数（Stack 传 `self._get_lora` 复用缓存），**禁止另写一份**（AGENTS.md 规则 11）；纯数学/格式探测在 `lora_ortho.py`（仅 torch，可单测）。

### 3. patch 结构

- 当前 ComfyUI 是 `LoRAAdapter.weights = (up, down, alpha, mid, dora_scale, reshape)`（**up 是 [0]、down 是 [1]**）；`replace_down` 对 LoRAAdapter 浅拷贝换 weights[1]，字符串标签/tensor-first/float 前缀多格式回退；**replace_down 对 `("diff", (w,))` 之类 1 元素内部元组必须原样返回**（直接 `list(patch[1])` 会 IndexError）。

### 4. 契约与内存

- Stack 的 `mergeMethod` 与 cacheMode 同模式（前端 `DEFAULT_PREFS`/`normalize`/`promptState` 与 Python `parse_state` 双端 1:1，默认 `"sequential"`）。
- **ortho 模式 run 内全栈 sd 驻留**（分组需要，与 "last" 逐行释放不同，峰值=栈大小），run 后仍按 cacheMode/无缓存统一修剪。
- **ok_paths 是 set——绝不能直接迭代它来组装 resolved/触发词/修剪游标**（顺序随机 → 触发词顺序/`last_this_run` 偶发不稳定，表现为测试偶发 FAIL）；必须按 plan 栈顺序扫描 `if zero or path in ok_paths`。

### 5. 测试

- 本机无 torch——GS 数学用 numpy 参考实现逐行对应验证（行两两正交/投影残差在基行空间/覆盖归零）；节点链路 monkeypatch GS + fake `load_lora` **必须按 key_map 值过滤**（unet 与 clip patch 键空间不同，不过滤会串侧）。

---

## 21. Civitai 页面主体描述补充（curl_cffi / __NEXT_DATA__ 与 Cloudflare 拦截）

> 背景：`lora_reader._html_to_markdown` / `extract_page_description` / `merge_descriptions` + `lora_routes._download_page`（2026-08）。model-versions API 的 version 级 `description` 常是空串，而模型页 Description 卡显示**模型级**完整描述（4110 字符实测）。by-hash 找到模型后总是抓页面补充。

### 1. 页面结构（Next.js SSR，别碰 DOM）

- 数据在 `<script id="__NEXT_DATA__">` JSON 里，描述在 `props.pageProps.trpcState.json.queries[]` 中 `queryKey[0]==["model","getById"]` 的 `state.data.description`。**mantine 随机 id 无关**——按 queryKey 结构定位（`[["model","getById"],...]`），绝不用 CSS 选择器。无 slug URL `/models/{id}?modelVersionId={vid}` 302 后数据完整。
- 拼接：**API 在前、页面在后**（`"\n\n"` 分隔，不截断）。拼接结果写入侧车 `data["description"]`（覆盖 API 空值）：读取端（lora_routes api_lora_info / 侧车缩略图提取走 parse_civitai_modelversion）零改动自然受益；删除侧车仍可清掉。

### 2. Cloudflare 拦截（TLS 指纹，JA3）

- 抓页面必须模拟浏览器，不只是 UA——`ComfyUI-sfnodes` UA 直接 403；**连带 Chrome UA 的 aiohttp 请求也实测 403**（Python 默认 TLS 握手指纹被识别），curl / curl_cffi 的指纹才放行。教训：**用 curl 验证"页面可抓"不代表 aiohttp 能抓**——必须以实际代码路径验证。
- `_download_page` 主路径走 **curl_cffi**（`impersonate="chrome"`，自带 libcurl 轮子，executor 线程运行不阻塞事件循环），aiohttp 兜底，都失败返回 None——**降级语义**：页面抓取失败 = 维持仅有 API 描述，绝不影响查询成功路径。`_PAGE_MAX_BYTES=2MB`、15s 超时。

### 3. 描述统一清洗（_html_to_markdown）

- markdownify 转换，缺库/异常回退 `_clean_description` 纯文本，测试双环境全绿；API/页面/文件内嵌/侧车描述同一入口。
- **幂等保护**：无 `<` 的输入（纯文本/已 markdown 化的侧车描述）只走轻清洗原样放行——markdownify 对非 HTML 输入不幂等（`*` 会转义成 `\*`），而侧车读取路径会二次处理，不保护则"首次查询正常、下次打开面板变转义文本"。
- **`_MAX_DESCRIPTION_LEN` 已删除**——不截断（来源有流量守卫：API 4MB/页面 2MB/文件本地；前端面板滚动展示）。

---

## 25. SFRegionalLoRA：多区域角色 LoRA（token 网格注入与匹配诊断）

> 背景：`nodes/model/regional_lora.py` + `sf_utils/regional_engine.py`（纯逻辑）+ `web/sf_regional_lora*.js` 两模块（2026-08）。Krea2 专用多区域角色 LoRA：每 box 一个 LoRA，激活 delta 只注入 box 内 image token。

- **纯逻辑在 `sf_utils/regional_engine.py`**（键归一化/矩阵解析/regions JSON/层规划 + 每区域匹配诊断/token 网格 mask 数学/彩虹预览，无 ComfyUI 依赖可独立测试）；节点层 forward hook 稀疏注入。
- **前端 DOM canvas 多 box 编辑**：拖拽/8 向 resize/画新框/背景图对齐；隐藏 `SFRegionsJson` widget 为真源（值随工作流保存），行控件 enable/lora/strength/remove。
- 测试：`tests/test_regional_engine.py`（纯逻辑）+ `test_regional_lora_node.py`（节点 mock）+ `test_regional_lora_js.js`（前端）。

---

## 28. SFStylesSelector：风格选择器复刻（Easy-Use stylesSelector）

> 背景：复刻 Easy-Use 的 `easy stylesSelector`（2026-08），落地为 `nodes/text/styles_selector.py`（节点 + 路由同文件注册）+ `web/sf_styles_selector.js`（主扩展）+ `web/sf_styles_selector_lib.js`（纯逻辑）+ `data/styles/fooocus_styles.json`（275 条内置数据）。复刻范围 1:1 拼接语义，UI 交互对齐原版（搜索/清空/选中置顶/hover 缩略图），但工程上按 sfnodes 惯例重做。

### 1. 数据三通道与优先级

- **内置只读**：`data/styles/*.json`（随包分发，如 fooocus_styles.json 119KB/275 样式，全含 name/prompt/negative_prompt/name_cn/thumbnail）。
- **用户自定义**：`<user>/sfnodes/styles/*.json`（复用 lora_routes `_sf_user_dir` 惯例，docker bind mount 存活），**同名文件覆盖内置**；`styles/samples/` 放本地缩略图 `<name>.jpg`。
- **远程兜底**：`fooocus_styles` 库无本地 samples 图时，image 路由返回 Fooocus GitHub raw URL 文本，前端按 http 前缀直用（原版 `FOOOCUS_STYLES_SAMPLES` 语义）。
- 库名枚举 = 内置 + 用户 json 去扩展名去重（先用户后内置稳定顺序）；文件加载 mtime+size 缓存（prompt_preset `_load_presets` 同款）。

### 2. 值通道：隐藏真源 + DOM widget 纯交互（同 regional_lora 模式）

- Python `hidden` 声明 `SFStylesState`（STRING default "[]"）→ 标准 widget 收集进 prompt、随 workflow 保存；前端 `find` 该 widget（缺则 addWidget 补建），标记 `hidden + computeSize 归零 + options.canvasOnly`。
- DOM widget `sf_styles_panel` `getValue: () => null` 不承担值传输（规避 Vue DOMWidget value setter 链，见 §11）；点击标签 → `writeState(node, names)` 写隐藏真源 `.value`（普通 STRING widget 可安全写）→ 重渲染。
- **加载/尾窗点击门控**：`isGraphLoading()` 为真时忽略标签点击（连接恢复/值恢复晚于 onNodeCreated，误点会覆盖刚恢复的选择）。
- 工作流加载后 `onConfigure` 触发 `ensureLoaded` 重渲染（读恢复后的真源值）；库切换（styles combo callback 包装）强制重拉列表。

### 3. 拼接语义 1:1 复刻（含原版怪癖，测试锁定）

- `{prompt}` 占位消费：**第一个**含占位的样式用用户输入替换（即使输入为空也替换）；后续含占位的样式剥离 `", {prompt}"` 片段尾接；无占位样式直接尾接。
- 用户输入未被任何样式消费 → `positive + positive_prompt + ', '`——**原版怪癖 1：无分隔逗号**（"a girl" + "masterpiece" = "a girlmasterpiece"）；**怪癖 2：末尾尾逗号**。行为一致复刻，注释标明。
- negative：样式负面提示词尾接在用户 negative 之后（空时直接取样式负面）。
- 原版 bug 修复（记录在案）：`select_styles.split(',')` 不去空格 → 接线带空格（`" Fooocus Sharp"`）匹配不到样式名；本实现 strip 后匹配。
- 空 values（无选择/全未知）→ execute 提前透传 `(positive, negative)`；未知样式名跳过不影响其余。

### 4. 路由（/api/sfnodes/styles）

- **列表** `?name=`：返回 `[{name, name_cn?, thumbnail, prompt?, negative_prompt?}]`——prompt/negative_prompt 供 hover 信息浮窗展示（对齐 v2 previewer；空串条目不携带该键）。thumbnail 归一化：http(s) 原样（远程直链）/ 本地路径转 image 路由 URL / 缺省兜底 `?name=&styles_name=` 查询。
- **图片** `?path=`：用户 + 内置双目录 ×（根 + samples/）四路查找；**穿越防护**：`os.path.normpath(join(base, rel))` 后 `os.path.commonpath == base` 才放行（未 normpath 前 join 的 `../` 不会折叠，commonpath 检查会漏）。
- **图片** `?name=&styles_name=`：本地 samples 优先 → `fooocus_styles` 库返回远程 URL 文本（`web.Response(text=...)`，前端 `resp.text()` 后按 http 前缀直用）→ 404。

### 5. 前端要点

- **hover 预览图修复原版全局 id 冲突**：Easy-Use 用 `document.getElementById('show_image_id')` 全局 img，多节点并存时 id 重复、互相覆盖。本实现预览 img 挂**每个 DOM widget 内部**（absolute + pointer-events:none），坐标相对 widget 容器换算（÷画布 scale 对齐 DOM widget 内容坐标系），clamp 在容器内防溢出裁剪。
- 搜索/清空/选中置顶（稳定排序，选中项永不隐藏）逻辑在 `sf_styles_selector_lib.js`（纯逻辑，拷 .mjs 可测）；搜索匹配原始 name 与语言化 label。
- 中文环境（`navigator.language` zh 前缀）显示 `name_cn`（值键恒为原始 name）。
- 样式列表 fetch 走 promise 级缓存（加载期重复调用复用同一请求；失败缓存空列表会话内不重试，对齐 prompt_tags `fetchDefaultLibrary` 语义）。
- 搜索框挂 `installWheelZoomPassthrough`（DOM widget 不在 Vue 滚轮转发路径，见 §27）；不拦截任何 keydown（ctrl 组合天然放行）。
- 选中状态变化走隐藏真源 widget 值 → 缓存键自然包含选择（无需 IS_CHANGED 抖动）；`IS_CHANGED` 只返回样式库文件 (mtime, size)。

### 6. 测试

- `tests/test_styles_selector.py`：mock aiohttp/server（canvas_size 先例）——拼接全分支（占位消费/剥离/未消费前置怪癖/negative/未知跳过）、目录 monkeypatch 覆盖优先级（**务必 finally 恢复模块函数**，否则泄漏污染后续用例）、归一化、IS_CHANGED、路由 handler 形状与穿越防护。
- `tests/test_styles_selector_lib.mjs`：lib 纯逻辑（parse/serialize/label/thumbnail/filterAndSort 全分支）。

### 7. v2 差异：Grid/List 显示模式与 Reset（对齐 v2 stylesSelectorDisplay）

- v2 的视图切换是**全局设置** `EasyUse.StylesSelector.DisplayType`（combo Grid/List，默认 Grid）+ 节点内下拉选择器，切换时 `ke()` 写回设置；本实现改为 `node.properties.sfStylesView` 随 workflow 保存（显示偏好按工作流区分，不做全局设置、不注入 prompt——与执行无关的 UI 状态）。切换按钮为工具条右侧两键分段控件（▦/☰）。
- v2 的 **Reset** = 清空选择（trash 图标 + "Reset" 文本）；v2 在 styles 库切换时**自动清空选择**（callback 里调 g()），本实现保留选择（旧库选中项在新库不存在时被 filterAndSort 忽略，不自动清空更友好）。
- **Grid 卡片**：缩略图 + 名字（ellipsis，title 全名），选中边框高亮（`--sf-acc`）；`img.loading="lazy"` 防 275 张远程图一次性拉取，onerror 占位；搜索过滤/选中置顶两种视图共用 `lib.filterAndSort`（hidden 项 display:none）。
- **grid 行高 min-content 塌缩（真 bug，headless Chromium 复现）**：grid 容器高度确定（calc/内联高度）时，`auto` 行在内容总高超出容器时**收缩到 min-content**——flex 卡片内 img（可替换元素 min-content=0）+ ellipsis span（min-content=0）→ 卡片 min-content ≈ 10px（仅 padding+border）→ 每行塌缩、img 溢出被卡片 `overflow:hidden` 裁成细条（诊断特征：cardH≈10 而 imgH=72、scrollHeight 极小）。修复：**`grid-auto-rows:max-content`** 强制按内容撑开（行高 10px→97px）+ img `min-height` 双保险。此前多次"高度声称"修复无效的原因：root/list 高度都正常，坏的是 grid track sizing。
- **List 行 label 点击双重触发（v2 样式选择器第二轮）**：List 行是 `<label>` 包裹 checkbox——点击 label 的**默认激活行为**（合成 input.click()）产生的新 click 事件**冒泡回 label** 再次触发 onclick → toggleSelect 两次、状态复原（"点不动"）。修复 `e.preventDefault()`（checkbox 显示由 renderList 重建控制，不需要默认激活）。Grid 卡片是 div 无默认激活行为不受影响——两种行控件事件模型不同，症状不对称时先查元素类型。
- **视图切换按钮双高亮（第三轮）**：viewBtn 工厂的 `sync` 闭包只 toggle **自身按钮**——点击 A 只同步 A，B 之前加的 `sf-ss-viewon` 残留 → 两个按钮恒高亮（"切换后不恢复"）。修复：统一 `syncViewBtns()` 遍历容器全部按钮按 `dataset.mode === viewMode(node)` 逐一 toggle（创建时 + 每次点击各调一次）。教训：**多按钮互斥高亮必须遍历全组同步，禁止 per-button 闭包自同步**。
- **网格列数设置（sfnodes.StylesSelector.GridColumns，第四轮）**：全局 combo 设置（Auto/4/5/6/8/10/12，默认 Auto 保持 CSS 自适应）——固定列数时 renderList 内联 `gridTemplateColumns: repeat(N, 1fr)`、Auto 清空内联回落 CSS。注册写法对齐 SFLoraStack accent 设置：扩展 `init()` 钩子 `addSetting`（id `sfnodes.*` 前缀）+ `onChange` 里 `setTimeout(refreshAll, 0)`（store 更新时序）+ **异步加载轮询**（设置值晚于 init 到达时补全局刷新，否则保存的列数不生效）；全局刷新遍历 `app.graph._nodes` 调 `ctx.reRender()`。
- **hover 信息浮窗（对齐 v2 previewer）**：图 + 名称 + Positive/Negative（3 行 clamp）挂 widget 内部，Grid/List 两视图共用 showPop；列表 API 因此补发 prompt/negative_prompt（空串条目不携带）。强调色全部走 `color-mix(in srgb, var(--sf-acc, #f66744) N%, transparent)`（项目标准，sf_load_image_ui 等同款）——**禁止硬编码 rgba(246,103,68)**（不跟随 sfnodes.Accent 设置）。
- List 模式保留 checkbox 行；Grid 模式卡片自带缩略图，hover 浮窗两视图一致。

---

## 31. Krea2 预设管理：SFImageInterrogator / SFKrea2SystemPrompt（内置 + 用户覆盖 + 墓碑复位）

> 背景：两类预设（反推指令 `INTERROGATOR_PRESETS` + 系统指令 `KREA2_PRESETS`）原为 krea2.py 硬编码 dict、仅 GET 返回，无法管理。用户要求"添加/删除/修改/复位"。落地为 `sf_utils/krea2_presets.py`（纯逻辑）+ `web/sf_krea2_presets.js`（共享管理 popup）。

### 1. 数据模型：内置 + 用户覆盖 + 墓碑

- **内置默认**：krea2.py 硬编码 dict 作默认源（不迁 data JSON，改动最小）。
- **用户存储**：`<user>/sfnodes/{interrogator,krea2}_presets.json`，结构 `{"overrides": {"名": "文本"}, "deleted": ["内置名"]}`：
  - `overrides` 兼两职：修改内置（按名覆盖、保持内置位置）+ 新增（内置没有的名字追加到末尾）；
  - `deleted` 是墓碑，标记被删的内置（复位=清除墓碑还原）。
- **merge(builtin, store) 纯函数**（确定性）：以内置顺序为基准 → 跳过墓碑 → overrides 覆盖文本 → 追加新增。**墓碑胜出**：同名既覆盖又删除时以删为准（API 路径不会产生该状态——POST 存会清墓碑、DELETE 删内置会清 override，仅直接编辑文件可能并存，取删优先安全）。
- **受保护名**：`register(kind, builtin, protected=("none",))` —— Krea2SystemPrompt 的 `"none"` 虚拟项不可删除/复位（DELETE/reset 返回 400）。
- **语义映射**：修改/新增=写 overrides；删除内置=记墓碑、删用户新增=移除 override；复位单个=清该名 override+墓碑；复位全部=清空整个 store 文件。

### 2. 后端：sf_utils/krea2_presets.py

- `_sf_user_dir()`：本地镜像（styles_selector 同款，避免拉入 lora_routes 重依赖）；`folder_paths.get_user_directory()` 兜底 `<pkg>/user`。
- `load_store/save_store`：mtime+size 缓存热加载（prompt_preset 范式）+ tmp 带线程 id + `os.replace` 原子替换（lora_presets 同款）。
- `asyncio.Lock` 每 kind 一把：并发读-改-写防互擦。
- 路由（`register(kind, builtin, protected)` 注册，由 krea2.py 模块末尾 `_register_krea2_routes()` 调用——**必须在内置 dict 定义之后**，注册捕获 builtin 引用）：
  - `GET /api/sfnodes/{kind}_presets` → `{presets(合并), builtin, user, deleted}`（前端需区分内置/用户以显示复位与徽标）；
  - `POST` `{name,text}` 新增/修改（存时清墓碑=复活）；`DELETE ?name=` 删除（内置墓碑/用户移除）；`POST /reset` `{name}` 或 `{all:true}`。
- **旧 GET 路由迁移**：原 krea2.py `_register_krea2_routes()` 里直接返回 dict 的两个 GET 迁到本模块（返回结构变更需同步前端）。

### 3. 节点改动（动态 combo，见 §4）

- `preset` combo 静态只列内置（INPUT_TYPES 在 import 时求值，无法预知运行时新增用户预设）→ 两节点加 `VALIDATE_INPUTS=True`（值超出静态列表不拦截），**前端加载后重建 combo options**。
- 执行回退改用合并预设：`prompt or _merged_presets(kind, builtin).get(preset, ...)`——`_merged_presets` 调用 `krea2_presets.merged(kind)`（`merge(_builtin[kind], load_store(kind))`），失败降级内置。
- 注意：krea2_presets.py 顶层 import **无副作用**（不自动注册路由），路由只在 `register()` 触发；krea2.py 顶层 `try: from ...sf_utils import krea2_presets` 失败降级 `None` → `_merged_presets` 回退内置，功能不崩。

### 4. 前端：web/sf_krea2_presets.js（共享）

- **combo 动态重建**：`setPresetOptions(node, presets)` 把 `preset` widget 的 `options` 设为合并名（保留当前值，VALIDATE_INPUTS 兜底）；`refreshAllNodes(kind, comfyClass)` 改动后重拉 → 重建所有同 class 节点 → 派发 `sfnodes.<kind>-presets-changed` 事件（跨节点/跨窗口兜底同步）。
- **管理按钮**：`addManageButton(node, kind)` 用 `addDOMWidget` 加纯按钮（`serialize:false`、`getValue→null`，**不写 .value → 无 §2.7 值写入递归风险**）；DOM widget 追加在 INPUT_TYPES 各 widget 之后，不移动它们的索引（旧工作流 widgets_values 不错位）。
- **管理 popup**（`openPresetManager`）：复用 `sf_popup.js` 三件套（attachPopupDismiss + clampToViewport + exempt 豁免）；列表每行 名称+内置/用户徽标+文本预览+编辑/复位/删除；顶部 新增/复位全部；编辑=名称 input + 多行 textarea。破坏性操作 `confirm()` 确认；复位仅对"内置被改或用户新增"显示。
- 两节点 JS（krea2_interrogator.js / krea2_system_prompt.js）各自 import 共享模块，仅传 `kind`/`comfyClass`；原预设→文本填充逻辑（含 Krea2SystemPrompt 的 `krea2PresetName` 派生标记、configure 同步）保留不变，`presets` 缓存改为合并视图。

### 5. 测试与部署

- `tests/test_krea2_presets.py`：merge 纯函数（覆盖/墓碑/新增/并存/非法兜底）、校验、store 读写缓存、路由 CRUD（含 404/400/保护名/复活/复位全部）。**坑**：路由测试前必须清空用户存储（上面 store 读写测试写入了数据，污染 GET 断言）；改/删并存语义定为墓碑胜出。
- `tests/test_krea2_presets_smoke.mjs`：mock fetch/app/document 真实加载模块（拷 .mjs + 替换 `/scripts/app.js` 绝对导入为本地 stub）验证 API 封装、setPresetOptions、refreshAllNodes 重建+广播。**坑**：stub app 的 `_nodes` 在 import 时按值捕获 global → 必须在 import 前设数组、之后只 push 不重赋值。
- 部署：后端需重启容器（新增模块/路由），`web/` 同步 docker 目录 + 浏览器硬刷新；`tests/check_web_imports.py` MODS 已加 `sf_krea2_presets`。

### 6. 快照陈旧导致新预设切换不填充（2026-08 修复）

- **现象**：`SFImageInterrogator` 管理预设新增「通用测试」「通用(可多人nsfw)」后，下拉可见新项但切换不填文本框。
- **根因**：预设→文本联动的唯一数据源是 `node.properties._krea2PresetData` 快照（`nodeCreated` 时 `init(data)` 写入，随 workflow 保存）。管理 popup 的 `refreshAllNodes` / 跨窗口 `presetsChanged` 监听的 `reloadNodes` 原只调 `setPresetOptions` 重建下拉选项，**未同步快照** → 快照仍是节点创建时的旧合并视图，新键 `data[value]===undefined` 分支不执行。
- **修复**：① `sf_krea2_presets.js:reloadNodes` 遍历 `nodesOfClass(comfyClass)` 时同步 `n.properties._krea2PresetData = data.presets` 再 `setPresetOptions`；② 两节点 `loadPresets` 首载分支同布快照；③ 回调 `presetWidget.callback` 双源容错 `const cur = presets || node.properties._krea2PresetData`（全局最新优先，快照兜底），后续新增无需再等快照同步也能填充。`SFKrea2SystemPrompt` 同构修复（其 `syncFromPreset` 亦受益）。
- **教训**：动态 combo 的“选项列表”与“选项→值映射”是两份状态，必须同更新；只更选项不更映射是“下拉可见但联动失效”的典型症状。

## 32. SF Load Diffusion Model：信息面板跨域复用（数据域分派 + ctx.api 整束注入，2026-08）

官方 UNETLoader 强化版（`SFLoadDiffusionModel`）复用 SFLoraStack 信息面板的完整做法——**面板本体零分支、两域零内联副本**。

### 1. 后端：同一 handler 别名路由 + 请求路径分派存储域

- Civitai 查询/自定义描述/预览图/孤儿迁移等 9 个路由与 LoRA **完全同构**，差异只在三个存储锚点（模型目录类型、用户数据文件、预览图目录）。在 `lora_routes.py` 加 `_is_dmodel_req(request)`（按 `request.path` 前缀判定域）+ `_dom_resolve/_dom_dirs/_dom_notes_file/_dom_previews_dir` 四个分派函数，各 handler 首行换用；再把**同一协程注册到第二路径**：`routes.get("/api/sfnodes/dmodel_thumb")(api_lora_thumb)` 式别名（aiohttp RouteTableDef 支持同 handler 多路径）。
- 存储分域硬约束：`dmodels.json` / `previews_model/` 与 lora 的文件**物理分离**——lora_reader 的存储函数全部按 `(file/folder, name)` 参数化是前提，传不同 file/folder 即得全套指纹/孤儿/迁移语义，一行不用抄。预览图 sha1 槽位目录必须分开，否则同名相对路径撞槽。
- 真正新写的只有 info 组装（`diffusion_routes.py::dmodel_info`）：safetensors 头部 `__metadata__`（config JSON 架构串）替换 LoRA 训练字段，**触发词三组恒空数组**（响应形状对齐 `build_lora_info` 让前端零分支），侧车 `<base>.civitai.info` 跟随模型文件天然隔离。
- 触发词是 LoRA 专属概念：`custom_triggers`/`lora_info`/`lora_list` **不设别名**。

### 2. 前端：openInfoPanelFor 的 ctx 三开关 + api 整束

- `ctx.hideTriggers`（缺省 false）整块跳过触发词区块；`ctx.samplesKind="diffusion_models"` 让 samples URL 带 kind 参数（后端 lora_samples.py 四个解析函数加 `dir_type` 缺省参数，旧调用逐字节不变）；`ctx.autoCivitai` 打开面板即自动匹配（侧车已有 model_id 则不打扰）。既有 LoRA 宿主全部走缺省值，行为不变。
- **api 束注入**是关键机制：面板内部 12 个路由调用收敛为顶部一处解构 `const A = {...默认LoRA函数, ...ctx.api}`。⚠️ **键名错配会静默回退 LoRA 路由**（Object.assign 只覆盖同名键）——必须用冒烟测试锁定束形状（`tests/test_load_dmodel_panel_smoke.js` 逐键 typeof 断言 + fetch 记录断言"绝无 /lora_info 回退"）。
- i 图标绘制与 configure 时序抽成参数化工厂：`sf_lora_info.js` 导出 `createInfoWidget(comboName,{hasCustomOf,onOpen})` 与 `setupLoaderInfoWidget(node,comboName,{prefetch,...})`，LoRA 版签名不变委托内部。

### 3. 测试

- 后端 `tests/test_diffusion_routes_smoke.py`：mock folder_paths 需补 `supported_pt_extensions` 与 `get_filename_list`；侧车要用**原始 Civitai API 形状**（trainedWords/model.name/baseModel/modelId/id），不是解析后的形状。
- 大文件 SHA256 慢：by-hash 查询本就走 executor 且结果持久化在侧车（第二次离线秒开）；hash 本身暂无独立缓存，多 GB 文件首次匹配数十秒属预期（面板 civStrip searching 态可见进度）。
