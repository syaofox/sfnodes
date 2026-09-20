# 经验归档：图片 / 遮罩 / latent 节点（§8、§9、§11、§12、§13、§22、§34、§35、§36、§37、§44、§45、§51、§60、§64、§65、§66、§67、§75、§76、§80、§81、§83、§85、§88）

> 全局章节号 §N 与拆分前的 experience.md 一致；跨节/跨文件引用一律写 §N，映射见 [README.md](README.md)。版本时效说明见 README。

## 8. SFPauseImage：快照闸门与预览保存（复刻 Pixaroma Pause Image）

> 背景：复刻 Pixaroma 的 `PixaromaPauseImage`（2026-08），落地为 `nodes/image/pause_image.py` + `nodes/image/preview_routes.py`（新后端路由）+ `web/sf_pause_kit.js` 共享引擎（image/mask/latent 三闸门共用）+ `web/sf_pause_image.js` 薄配置。与 SFPauseText 是兄弟闸门（prune/双钩子/一次性模式/executed 机制完全同构），核心差异是**图片无法像文本一样随隐藏输入携带**——必须走快照文件，并引入 PNG 元数据嵌入与自定义保存路由。

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
- 其余同构：双钩子（graphToPrompt 只注入 {mode} / queuePrompt 剪枝）、`MODE_RANK` continue 先排序、一次性提交模式 finally 清除、executed 回填、子图 id 解析（现收敛 kit `findNodeByPromptId`：复合 id 精确匹配 + 冒号尾段兜底）、解析不到节点默认 pass。
- 差异：gate 只有 pause/pass 两态（无 keep——图片没有"批量复用"语义）；collectGates 注入只带 mode；state 形状最小（{gate, frame}，`hasSnapshot` 运行时推导绝不住 properties）。

### 6. 测试方法论（延续冒烟 + 新增快照/路由层）

- **mock torch 的 MockTensor**：需支持 `image[0]`（返回 cpu().numpy() 帧壳）与 `[None, ...]`（加 batch 维）两种下标；numpy/PIL 本机真实可用（torch/folder_paths mock，`get_temp_directory` 指向临时目录）。
- **快照 round-trip 断言**：pause 写入 → 检查文件存在 → continue 读回 → `out.numpy()` 与输入逐元素相等；无快照/损坏快照（写入垃圾字节）→ RuntimeError 中文消息；只读目录（chmod 0o500）→ 保存降级仍透传。
- **PngInfo round-trip 验证**：`_build_pnginfo` 后 `Image.save(pnginfo=)` → 重新 `Image.open` 读 `img.text` 断言 tEXt 块内容（含 NaN 已清洗）。**PIL 12.x 的 PngInfo 无 `_text` 属性**（结构变化），探测内部字段会误导——用保存-读回断言，别摸内部。
- **测试文件结构坑**：追加用例块时别放到 `sys.exit` 之后（不可达）——先验证追加位置再跑。

### 7. 模块边界（复用/修改时的快速索引）

- `nodes/image/pause_image.py`：节点（快照/continue 读回/无 IS_CHANGED）+ `_json_safe`（已收敛入 `sf_utils/common.py`，见 patterns.md §57）。
- `nodes/image/preview_routes.py`：save/prepare 路由 + `_build_pnginfo`/`_metadata_disabled`（`_safe_prefix`/`_sanitize_segment`/`_decode_image` 已收敛入 `sf_utils/disk_state.py`，见 patterns.md §57–58）。
- `web/sf_pause_kit.js::makeGateState`：state 工厂（{gate, frame}，image/mask/latent 共用，仅 stateProp 配置不同）。
- `web/sf_pause_kit.js::buildPauseBody`：DOM widget 工厂（预览/按钮行/尺寸行 + frameViewUrl /view+缓存戳）。
- `web/sf_pause_image.js`：薄配置（调 definePauseGate；双钩子/Save 链路/Copy/Open/executed 全在 kit）。
- prune 共享：`web/sf_pause_text_lib.js::applyGateMode`（四闸门共用，勿复制）。

---

## 9. SFPauseMask：遮罩快照闸门（Pixaroma Pause Mask 同构扩展）

> 背景：复刻 Pixaroma 的 Pause Mask 变体（2026-08），落地为 `nodes/mask/pause_mask.py` + `web/sf_pause_kit.js` 共享引擎（image/mask/latent 三闸门共用）+ `web/sf_pause_mask.js` 薄配置。与 SFPauseImage 完全同构（快照/剪枝/一次性模式/executed 回填机制全部复用），仅把输入类型换成 MASK 张量 `[B, H, W]`（ComfyUI 遮罩格式）。本节点是"类型化闸门复用"的最小改造成案例——架构决策：**只加一个类型参数，绝不复制三份**。

### 1. 与 SFPauseImage 的差异（都是类型相关的）

- **快照为单通道灰度 PNG（L 模式，0-255 量化）**：遮罩通常二值/低精度，8bit 足够；与 ComfyUI 自身把遮罩存灰度 PNG 的惯例一致。`_mask_to_pil` 先 `(arr * 255).clip(0, 255).astype(uint8)` 再 `Image.fromarray(arr, mode="L")`——**L 模式只接受 2D 数组**。
- **tensor 转换防御非标准 `[1,H,W]`**：部分节点输出的 MASK 带单例通道维（`arr.ndim == 3 and arr.shape[0] == 1` → `arr[0]` 压平）——标准帧是 `[H,W]`，不防御会因 3D 数组直接炸。
- **读回对齐**：`_pil_to_mask` 用 `torch.from_numpy(arr)[None, ...]` 补 batch 维回 `1xHxW`，与 ComfyUI 遮罩张量格式一致。
- **快照前缀 `sf_pause_mask_`**：与图片闸门的 `sf_pause_` 隔离命名空间（同 node_id 不撞文件；语义也清晰）。
- **frame 键/state 键**：executed 回填帧键 `sf_pause_mask_frame`；状态存 `node.properties.pauseMaskState`（{gate, frame}）。三闸门 state 同构，共用 sf_pause_kit.js::makeGateState（仅 stateProp 配置不同）。

### 2. 剪枝共享（三闸门同一份实现）

- **prune 全走 `sf_pause_text_lib.js::applyGateMode(out, id, entry, mode, isOutput, HIDDEN_INPUT, opts)`**：PauseMask 传 `{inputKey: "mask"}`（PauseText 省略、PauseImage 传 `"image"`）。`inputKey` 是唯一分叉点——删哪个输入键/注入什么。
- 改 prune 语义只改 `sf_pause_text_lib.js` 一处，三节点同步生效（2026-08 复查时确认 PauseText 版与 PauseImage 版逐字一致，本次把差异收敛成参数）。
- 其余同构机制全部复用：双钩子（graphToPrompt 只注入 {mode} / queuePrompt 剪枝）、gate 两态（pause/pass，无 keep）、一次性提交模式、executed 回填、Save 链路（`/api/sfnodes/preview/{save,prepare}`）、无 IS_CHANGED。

### 3. 测试与方法论

- `tests/test_pause_mask.py`（后端）+ `test_pause_mask_js.js`（state/prune 纯函数）+ `test_pause_mask_smoke.js`（主扩展冒烟）——镜像 PauseImage 三件套。
- 冒烟断言同款结构：注入 → 剪枝 → executed 回填端到端；快照 round-trip 逐元素相等；非标准 `[1,H,W]` 帧转换防御断言。

### 4. 模块边界（复用/修改时的快速索引）

- `nodes/mask/pause_mask.py`：节点（快照 L 模式/读回 [1,H,W] 防御/无 IS_CHANGED）+ `_json_safe`（已收敛入 `sf_utils/common.py`，见 patterns.md §57）。
- `web/sf_pause_kit.js`：state/UI/主扩展引擎（image/mask/latent 共用，仅 stateProp="pauseMaskState" 等配置差异）。
- `web/sf_pause_mask.js`：薄配置（调 definePauseGate）。
- prune 共享：`web/sf_pause_text_lib.js::applyGateMode`（**三闸门共用**，勿复制）。

---

## 11. SFImageCrop/SFImageUncrop：可视化裁剪与贴回（复刻 Pixaroma Crop/Uncrop）

> 背景：复刻 PixaromaCrop/PixaromaUncrop（2026-08），`nodes/image/crop.py`（两节点 + 2 条路由）+ `web/sf_crop*.js` 九模块（编辑器 framework/核心/面板/交互/渲染 + 预览/撤销守卫/对齐）。本轮排查了本项目迄今最深的"值传递"坑链，前后四轮修复才打通。以下按"可迁移结论"记录，细节见代码注释。

### 1. 前端 widget 值要传给后端 → 必须在 Python INPUT_TYPES 声明（本项目最大坑）

- **症状链**：裁剪数据保存正常（编辑器可编辑、SFCropJson widget 值正确），但后端每次收到空数据（`kwargs` 只有 `image`/`mask`）→ 节点透传原图。注入到 `graphToPrompt`、`api.queuePrompt` 的值全部"神秘消失"。
- **根因**：ComfyUI 前端提交 prompt 前有 validatePrompt——**删除不在节点 schema 中的输入**。schema 来自 Python `INPUT_TYPES`；`SFCropJson`（前端 addWidget 创建）与 `CropWidget`（DOM widget）都未在 Python 侧声明 → 前端直接剥离 → 后端 `kwargs` 里根本没有该键。
- **正确做法**（sf_pause 的 PauseState 同款）：Python `INPUT_TYPES` 的 `"hidden"` 里声明 `"SFCropJson": ("STRING", {"default": "{}"})` → 输入进入 schema → 前端不剥离；前端创建**同名隐藏 STRING widget**，值经标准 widget 通道收集（graphToPrompt 读 `widget.value`，最基础机制，任何插件/渲染器不可破坏）。
- **判据**：任何"前端把运行时状态交给后端"的输入，先问"Python 侧声明了吗？"——没声明就是会被剥离。排查"值丢了"先用后端打印 `sorted(kwargs.keys())` 一锤定音（比猜前端快得多）。
- **注**：`graphToPrompt`/`api.queuePrompt` 注入只是双保险（覆盖加载/保存时序差），不是可靠通道——注入目标也必须是 schema 内的输入名。

### 2. Vue DOMWidget 的 value setter 会回调 setValue → 写 widget.value 会无限递归

- **症状**：保存裁剪时 `Maximum call stack size exceeded`。根因：`_sfCropJsonSync` 里 `widget.value = {...}` → Vue DOMWidget setter（`domWidget.ts`：`set value(v) { this.options.setValue?.(v) }`）→ 我们传入的 setValue → 又调 `_sfCropJsonSync` → 循环。
- **规则**：对 addDOMWidget 创建的 widget，**不要写 `.value` 去"同步状态"**——写它等于触发 setValue 回调链。DOM widget 的值读取应走 `getValue` 闭包（graphToPrompt 收集 `widget.value` → getter → getValue）。状态同步走独立通道（隐藏 STRING widget 是普通 widget，无 setter 链，随便写）。

### 3. 拼接/裁剪移植时的两类机械性 bug（复刻大模块必查）

- **漏模块级常量**：从 `canvas.mjs` 提取 `createCanvasSettings` 时漏了文件顶部的 `CANVAS_RATIOS` → 编辑器打开即 `ReferenceError`（界面"没反应"，console 有报错）。提取函数片段后必须 **grep 函数体内引用的大写常量，确认其定义也被提取**。
- **漏依赖函数**：替换 core.mjs 的 import 块时漏了 `pixApiUrl`（原版从 shared import）→ `CropAPI.saveComposite` 运行时 `ReferenceError`，**被编辑器 catch 吞掉** → composite_path 永远为空、无任何报错。catch 吞错的路径要格外留意——"保存成功但没保存"是最难发现的失败。

### 4. 磁盘状态链路（编辑器保存 → 后端读取）

- 路由：`/api/sfnodes/crop/save`（composite）+ `/upload_src`（dataURL → PNG 存 `input/sfnodes_crop/`）。**改动路由必须重启容器**，否则前端 fetch 404 被 catch 静默降级。
- 路径守卫 `_safe_join`：词法拒绝绝对路径/UNC/`..`（**任何 resolve 之前**——UNC 仅 realpath 就会触发 SMB 认证泄露），再 realpath + startswith 包含检查。
- `_sanitize_id`：project_id 只留 `[A-Za-z0-9_-]`，防路径穿越。
- 后端解析 `_crop_meta_from_widget` 兼容 4 种形状（`{crop_json}` dict / 直接 meta dict / JSON 字符串 / 字符串套层）——前端不同版本/渲染器发来的形状不定，防御性解析 + 坏数据回退透传。

### 5. 编辑器/预览相关的渲染器差异（用户环境实测）

- 用户环境 `LiteGraph.vueNodesMode = false`（legacy 渲染器）——DOM widget 值收集在 legacy 下**是工作的**（serializeValue→getValue），此前"DOM widget 值不可靠"的假设不成立；真正断点始终是 schema 剥离。排查时先确认渲染器模式，别在错误的环节上打转。
- **加载工作流后预览 404**：`node.properties` 缓存上次执行的 temp source URL，重启容器 temp 清空 → fetch 404 → 预览空白（"接了输入图却没加载"）。修复：image 输入**已接线时优先解析上游**（LoadImage widget / 上游 imgs），缓存只在未接线或上游解析失败时使用。

### 6. 测试方法论

- 后端：`test_crop.py` mock torch/aiohttp/folder_paths + sfnodes 包上下文，覆盖结构断言、`_safe_join`/`_sanitize_id`/`_decode_image`/`_rect_from_meta`/`_crop_meta_from_widget` 形状兼容。
- 前端：`test_crop_js.js` 冒烟——**mock 里模拟 Vue DOMWidget 的 value getter/setter 链**（getter→getValue、setter→setValue），断言 `_sfCropJsonSync` 写 DOM widget 不递归（回归第 2 节 bug）；mock `addWidget` 返回**对象语义**（`{name, type, value, options}`），数组模拟会踩"`.value` 属性与索引元素分离"的假象。
- 诊断先例：后端 `kwargs_keys=` 打印 + 前端分段 console（版本 → 节点状态 → graphToPrompt 数据层）——四轮排查中唯一一次一锤定音的就是后端 `kwargs_keys=['image','mask']`。

### 7. 模块边界（复用/修改时的快速索引）

- `nodes/image/crop.py`：SFImageCrop（可视化裁剪）+ SFImageUncrop（贴回/feather）+ `_crop_meta_from_widget`/`_safe_join`/`_sanitize_id` + 2 条路由（`/api/sfnodes/crop/*`）。
- `web/sf_crop.js`：主扩展（nodeCreated/面板/编辑器接线/拖放粘贴/queuePrompt+graphToPrompt 注入 SFCropJson）。
- `web/sf_crop_framework.js`：精简编辑器框架（theme CSS/组件/canvas settings/布局/焦点陷阱/下载）。
- `web/sf_crop_core.js`（编辑器核心+API）、`sf_crop_panel.js`（节点面板）、`sf_crop_interaction.js`（鼠标键盘）、`sf_crop_render.js`（绘制/保存）、`sf_crop_preview.js`（节点预览）、`sf_crop_undo_guard.js`（Ctrl+Z 守卫）、`sf_crop_alignments.js`（对齐常量）。
- 数据契约：`SFCropJson`（Python hidden + 前端同名 STRING widget，crop_json 文本）；`SF_CROP_INFO` 线类型（原图+rect+可选 mask）；`sf_crop_source` ui 键（temp 预览）；`sfnodes_crop` subfolder。

---

## 12. SFImageOutpaint/Stitch：外绘填充与原始图贴回（复刻 Pixaroma Outpaint）

> 背景：复刻 PixaromaOutpaint/PixaromaOutpaintStitch（2026-08），落地为 `nodes/image/outpaint.py`（两节点同文件，crop.py 先例）+ `web/sf_outpaint*.js` 两模块（core 纯数学 + 主扩展精简版）。后端完整复刻（含全部防御与设计注释），前端按精简策略移植。以下为可迁移结论，细节见代码注释。

### 1. 复刻先查已有基建，别重复移植引擎

- `sf_utils/resize_engine.py` 已是 Pixaroma `_resize_helpers.py` 的移植（SFLoadImageResize 时做的），outpaint 节点需要的一切（`_apply_pad`/`_apply_max_mp`/`_round_half_up`）都在——**零新增引擎代码**，节点只做组合（pad → max_mp 两段式，snap 只触发一次的安排照搬）。
- 复用前提是契约一致：源节点构造 `pad_state`/`mp_state` 字典时用的键名（`pad_top`/`max_mp`/`allow_upscale`/`resample`）与 resize_engine 完全同源，直接可用。

### 2. Python 端三件防御（移植时不可精简）

- **`_parse_state` 的 OverflowError**：`json.loads` 按文档扩展接受字面量 `Infinity`，`int(inf)` 抛 **OverflowError 而非 ValueError**——只捕 ValueError 的话，手改 API 文件带 `Infinity` 会让整个节点倒下。
- **`_fit_pad` 防 OOM**：`_apply_pad` 以未夹紧尺寸分配 `Image.new`、clamp 只缩结果——极端比例（1:1000）或四边 8192 会在 clamp 前分配数 GB。先按比例收缩 pad 到 16384 上限内再构建画布。
- **`_round_half_up` 而非内建 `round()`**：银行家舍入（`round(1498.5)=1498`）会让 999 高源在 3:2 下与 JS 预览差 1 像素。所有 factor*dim 数学与 JS `Math.floor(x+0.5)` 对齐。
- **anchor 语义是反约定的**（易被"纠正"）："right" = 新空间在右边（绿色去哪边），与 resize_engine `_anchor_offsets`（图片贴哪边）刻意相反——因为 sides 模式已是每边绿色，两种模式同一个词必须同义。注释里显式警告勿改。

### 3. 自定义线型 + temp 预览存档（sf_crop 同款链路的复用）

- `SF_OUTPAINT_INFO` 纯字符串常量，两节点同文件定义天然解耦（crop.py 的 `SF_CROP_INFO` 同款；Pixaroma 用跨文件复制字符串避免 import 链，同文件时不需要）。
- info dict 携带**原始张量**（Python 侧私有，从不进 ui）+ 四边 pad + orig/canvas 尺寸；stitch 据此 resize 回画布、贴回、生成"生成区"遮罩（`1 - alpha`）。
- 预览第二层：`folder_paths.get_temp_directory()` + uuid PNG + ui payload `sf_outpaint_base` + 前端 `executed` 监听解码（sf_crop 的 `_save_source_temp` 同款）。文件名前缀 `sf_` 隔离命名空间，uuid 兼作缓存失效。stash 失败 print 降级，绝不让预览弄死真实运行。

### 4. 前端精简移植：跨文件复用共享组件时的机械性坑

- 精简策略：去 Pixaroma 品牌功能（accent 主题色、齿轮设置面板、比例/MP 列表管理——用固定默认列表），保留核心交互（预览、模式/比例/anchor/MP 芯片、L/T/R/B 输入、绿色边拖拽（ratio 首拖自动切 sides 带数值）、折叠、右键菜单折叠/重置、graphToPrompt 注入、executed 收帧）。
- **复用 `sf_load_image_resize.js` 的 `makeNumericInput` 而非重写**（同款 opts 契约 `{value,min,max,step,format,onCommit}`，返回 `{wrap,input}`）——但 CSS 覆盖选择器必须用**本项目类名** `sf-li-numinput`/`sf-li-spin`，机械复制 Pixaroma 的 `pix-li-*` 会让样式静默失效（数值输入渲染成未剥离样式的样子）。跨文件复用前先 grep 目标模块的实际类名/导出名。
- 纯数学抽到 `sf_outpaint_core.js`（无 app/DOM，.mjs 直测），UI 文件只做渲染与事件——与 Python 的镜像关系写在文件头，两侧公式改动后必须重跑交叉测试。

### 5. 测试方法论（FakeTensor 数值路径扩展）

- 本机无 torch：`test_outpaint.py` 用 numpy 代理 FakeTensor（test_inpaint_helpers.py 先例）驱动**全流程数值断言**——outpaint execute（ratio/sides/limit/snap 尺寸、填充色、info dict、ui 存档落盘）+ stitch（贴回像素级、mask 语义、批次配对、resize 恢复、`_color_match` 均匀色调平移）。为 stitch 扩展了 `narrow`/`index_select`/`permute`/`clamp(min=,max=)` 与 `__getattr__` 里 `dim→axis`、`keepdim→keepdims` 重映射（torch API 到 numpy 的关键适配）；`F.interpolate` 用 PIL BILINEAR mock。
- 两个坑：**numpy 2.x 的 `np.clip` 只传 `a_max=` 会报缺 `a_min`**（mock torch.clamp 必须给双边界默认 ±inf）；**PIL 通道 128/255 量化**让 `0.5 → 0.50196`，经 PIL 路径的断言容差要到 1e-2，纯张量路径才是精确的。
- JS 侧 `test_outpaint_js.js`：core 复制 .mjs 直跑，与 Python 测试**同用例同期望值**（含 round-half-up 边界、limit 缩放、snap-once），两侧独立跑通即视为镜像一致。

### 6. 模块边界（复用/修改时的快速索引）

- `nodes/image/outpaint.py`：`SFImageOutpaint`（pad→max_mp→ui 存档，复用 resize_engine）+ `SFImageOutpaintStitch`（resize 恢复→`_color_match` 连续色域匹配→`_feather_sides` 边选择性羽化→贴回+遮罩）+ 模块级纯函数（`_parse_state`/`_parse_ratio`/`_pads_for_ratio`/`_fit_pad`/`_tensor_to_pils`）。
- `web/sf_outpaint_core.js`：纯数学（parseRatio/padsForRatio/finalSize/readState 等，镜像 Python）。
- `web/sf_outpaint.js`：主扩展（预览/芯片/拖拽/折叠/graphToPrompt 注入 SFOutpaintState/executed 收 `sf_outpaint_base`）。
- 数据契约：`SFOutpaintState`（hidden STRING，graphToPrompt 从 node.properties.outpaintState 注入）；`SF_OUTPAINT_INFO` 线类型（original 张量 + 四边 pad + orig/canvas 尺寸）；`sf_outpaint_base` ui 键（temp 预览）；temp 前缀 `sf_outpaint_base_`。Stitch 无 JS（feather/color_match 原生 INT widget 直通后端）。

---

## 13. SFImageResize：wired 尺寸缩放（复刻 Pixaroma Image Resize）

> 背景：复刻 PixaromaImageResize（2026-08），落地为 `nodes/image/resize_image.py` + `web/sf_image_resize*.js` 三模块（lib 纯函数 + ui DOM + 主扩展）。**区别于此前复刻：引擎与面板全部复用既有生态**（resize_engine.py + buildModePanel + renderGlobalControls），新增代码集中在 wired 输入交互与 readout 卡片。以下为可迁移结论，细节见代码注释。

### 1. 复刻前先盘点既有移植生态（本任务收益最大的一条）

- `sf_utils/resize_engine.py`（8 模式引擎）+ `web/sf_load_image_resize.js`（buildModePanel 面板全家桶 + previewResize）+ `web/sf_load_image_ui.js`（模式芯片/全局控件行/面板后处理）三件套都在——新节点只补"wired 尺寸 + 中间态交互"，引擎与面板零新增。
- 复用要付出最小契约成本：`renderGlobalControls` 内部 `readStateLocal` 硬编码 prop 名 → 加 `statePropName` 参数（默认值保持旧行为，改动 1 处 6 调用）；`buildModePanel` 本就有 `stateKey` 参数，直接传新 prop；`applyInlineLabel`/`applyWHLayout`/`applyCoverControls` 无 prop 耦合直接复用。
- Pixaroma 原版跨文件复制两套 UI（pix-ir 与 pix-li），本项目复用同一个 sf-li 类族——**面板类名共享时无需再注入 CSS**，新 CSS 只有 chrome（chips / wire panel / readout canvas）。

### 2. wired 三输入优先级（核心设计，JS/Python 镜像）

- `longest_side` > `width`/`height`；单轴 = 按该维等比缩放（scale_factor 路径，尊重 allow_upscale）；双轴 = 精确盒（fit_inside 保持，其他强制 cover）；0/负 wired 值 = "无目标" → off 直通（避免极小输出，JS 预览一致）。
- 互斥：连接 longest_side 自动断开 width/height（反之亦然），width/height 可共存。**count 只统计 width/height**（longest_side 独立标志）：1 线禁用全部模式芯片，2 线只留 Fit/Crop。
- **显示模式不写 `state.mode`**：双线时渲染强制 Crop to fill，但 state.mode 不动 → 断开后用户原模式恢复，也不弄脏工作流（连接/断开操作零序列化副作用）。
- JS `effectiveWiredState` 逐分支镜像 Python `_apply_wired_size`，两侧测试**同用例同期望值**（含 0 值、val 不可读、fit_inside 保持）。

### 3. 交互细节（照搬原版，含反直觉点）

- `readWiredInt` 只信任"恰好一个数值 widget"的上游（多数值/字符串 → null → readout 显示"由接线输入决定"或回退上次运行 dims，**绝不显示错误数字**）；上游分辨率类节点特判属 Pixaroma 插件耦合，本项目不做（FluxResolution 无 widget 可读 → 走兜底）。
- wired 字段锁定 = readOnly + opacity 0.55 + makeNumericInput 的 readOnly 守卫（步进箭头天然失效）；锁值与单线/最长边汇总单元格的值由绘制轮询刷新（DOM 无上游值变化事件，onDrawForeground / Vue setInterval 是唯一信号）。
- 接线互斥断开三重守卫：onConfigure 窗口 + `app.loadGraphData` 包装的 300ms 尾窗 + 自递归标志。**连接恢复发生在 onConfigure 之后**，无 loadGraphData 尾窗守卫，打开工作流会误断已保存的线（与 SFImageResize 先例同款，Load Image Resize 的 wired 版本可复用）。
- 读卡器回退链：live 上游预览（upstream.imgs[0].naturalWidth）→ wired 镜像计算 → 上次运行 dims（executed 回填 node.properties.sfIrDims）→ 消息（"连接图片"/"运行一次"/"由接线输入决定"）。live 不可读而缓存存在时显示缓存——与 Pixaroma 原版一致（原版注释自认这是缺陷，见其 index.js 大段说明）。

### 4. 后端防御（可精简但不能丢）

- `_tensor_to_pils` 通道防御：1ch 复制 RGB、≥4ch 裁 3ch 且 alpha 走 MASK 输出（VAEEncode 不做通道切片，4ch 进采样器必炸）；alpha → MASK 必须反转（**1 = 透明**，LoadImage 惯例）。
- 显式 mask 优先于图片自带 alpha（显式接线是用户选择，第二猜测更糟）。
- `_apply_wired_size` 放 `resize_engine.py`（纯函数不依赖 torch/numpy）→ 节点层只管 tensor/PIL 转换；返回新 dict 不修改入参（sf_utils 纯函数风格，注意与 Pixaroma 原版原地修改的差异）。

### 5. 测试方法论

- Python：FakeTensor 驱动 execute 全流程（wired 尺寸、RGBA→mask 反转、pad 边框语义、显式 mask 覆盖）+ `_apply_wired_size` 纯函数分支全覆盖（含"原 state 不被修改"断言）。
- **PIL NEAREST 缩小是 box 平均**（非点采样）：单像素点缩小会被稀释（64² 角点 → 8x5 全 0），放大/同尺寸才像素保真——mask 对齐断言用"同尺寸直通 + 放大角点"两场景，缩小场景只断言尺寸对齐。
- JS：lib 复制 .mjs 直跑（stageJs 链自动带上 sf_load_image_resize.js）；FakeNode 提供 inputs/graph.links/getNodeById/widgets 模拟 wired 读取；主扩展 smoke 断言原型钩子安装 + graphToPrompt 注入 + executed 回填。**wireInfo 断言注意 count 只含 width/height**（不含 longest_side）。

### 6. 模块边界（复用/修改时的快速索引）

- `nodes/image/resize_image.py`：`SFImageResize` + `_tensor_to_pils`/`_alpha_to_mask_pils`/`_mask_to_pils`（张量→PIL 与 alpha 提取，依赖 torch/numpy）。
- `sf_utils/resize_engine.py`：`_apply_wired_size`（纯函数，无 ComfyUI 依赖，SFImageResize 与未来 wired 节点共享）。
- `web/sf_image_resize_lib.js`：纯函数（readState/writeState 泛化、wireInfo/readWiredInt/effectiveWiredState/getReadoutInfo、gcd/ratioLabel/aspectRectDims/roundRectPath，镜像 Python）。
- `web/sf_image_resize_ui.js`：DOM（injectCSS/buildChips/wired 面板/applyWiredLocks/refreshReadout/paintReadout/renderUI），复用 `sf_load_image_resize.js` + `sf_load_image_ui.js`（后者 renderGlobalControls 已参数化 statePropName）。
- `web/sf_image_resize.js`：扩展注册（onNodeCreated/onConfigure/onConnectionsChange 互斥+守卫/onRemoved/onResize/onDrawForeground + Vue cards canvas）+ graphToPrompt 注入 + executed 收 `sf_image_resize`。
- 数据契约：`SFImageResizeState`（hidden STRING，graphToPrompt 从 node.properties.sfImageResizeState 注入，随 workflow 保存）；`sf_image_resize` ui 键（in/out dims 回填）；**未移植 temp PNG 预览**——原版 JS 的 executed 处理器从不读 filename（只读 dims），省掉垃圾文件。

---

## 22. SFPauseLatent：latent 快照闸门（分段采样中间暂停）

> 背景：`nodes/image/pause_latent.py` + `web/sf_pause_kit.js` 共享引擎（image/mask/latent 三闸门共用）+ `web/sf_pause_latent.js` 薄配置（extraInputKeys:["image"] 由配置传入）。LATENT 闸门，专为"分段采样中间暂停"：KSampler(A) [start=0,end=4] → latent 闸门 → KSampler(B) [start=4,end=8]，image 预览输入接 VAEDecode。

### 1. 与 image/mask 闸门的核心差异

- Pause 停在第一段结束显示预览，Continue 跳过第一段整条链、从快照 latent 继续第二段（第一段零重跑），Regenerate 重跑第一段，Pass 一次跑完。
- **快照是 latent 张量（safetensors）而非 PNG**：`latent_tensor` 键 + `latent_format_version_0` 标记（对齐官方 SaveLatent 格式，官方 LoadLatent 读 multiplier=1）；**保存 latent dict 中全部张量键**（samples + noise_mask/batch_index）——继续采样需完整 batch 与重绘遮罩，不同于 image/mask 闸门仅首帧；读回时 `latent_tensor` 还原为 `samples` 键。`.latent` + 预览 `.png` 双快照同前缀 `sf_pause_latent_`。

### 2. 预览输入剪枝（applyGateMode 扩展）

- **预览输入（image）必须在 continue 时连同 latent 链接一并剪掉**：`applyGateMode` 新增 `opts.extraInputKeys`（continue 分支循环删除）——预览源（VAEDecode）在闸门上游，不删其输出仍被闸门消费，会把被跳过的第一段采样器拉活。extraInputKeys 仅 continue 生效（pause/pass 预览链接保留），不传时与 image/text/mask 旧调用行为完全一致（有回归测试锁定）。
- **无 image 预览输入也可用**：latent 快照照存照续，只是无 frame（前端不显示、Save/Copy/Open 不可用）。

### 3. 模块边界

- `nodes/image/pause_latent.py`：节点（快照/continue 读回/无 IS_CHANGED）。
- `web/sf_pause_kit.js`：state/prune/UI/主扩展引擎（prune 仍复用 `sf_pause_text_lib.js::applyGateMode`，extraInputKeys:["image"] 由薄配置传入）。
- `web/sf_pause_latent.js`：薄配置（调 definePauseGate）。
- 测试：`tests/test_pause_latent.py` + `test_pause_latent_js.js`（快照 round-trip、extraInputKeys 仅 continue 生效）。

---

## 34. SFLoadImageBrowser 右键菜单：提示词复制与工作流载入（全链路复用零后端改动）

> 背景：`web/image_browser.js` 弹窗浏览器图片项新增右键菜单——①复制正向提示词 ②载入内嵌工作流（新标签）。两能力全部复用既有实现，后端零改动（无需重启容器）。

### 1. 复用路由图（关键：先查复用再动手）

- **提示词提取**：`GET /api/sfnodes/prompt_reader/extract?filename=<path>[output]`（`nodes/text/prompt_reader_routes.py`，启动时副作用注册）。返回 `{found, text|message}` 恒 200；后端权威解析 ComfyUI prompt JSON（追 KSampler 正向）→ A1111 `parameters` 兜底。output 目录文件拼 `" [output]"` 注解即被 `folder_paths.get_annotated_filepath` 正确解析，input/output/temp 均在 allowed_roots 内。
- **PNG 内嵌工作流**：`sf_lora_shared_info.js::loadWorkflowFromImageUrl(url, onError)`（本次从 `loadImageAsWorkflow` 参数化导出；原函数变 lora_samples URL 薄包装，两个既有调用方签名不变、冒烟测试桩兼容）。内部 readPngWorkflowData 前端 chunk 解析 → prompt chunk 走 `app.loadApiJson`、workflow chunk 走 `loadGraphData`；新标签经 `app.extensionManager.command.execute("Comfy.NewBlankWorkflow")`，旧前端降级 confirm 后替换画布。
- **取原始字节**：ComfyUI 内置 `/view?filename=<basename>&subfolder=<dir>&type=input|output`——不带 preview/channel 参数时 FileResponse 返回原文件字节（PNG 元数据完整）；`sf_common.parseAnnotatedImageValue` + `buildSourceURL` 现成拼 URL。

### 2. DOM 右键菜单要点

- 菜单单例挂 `document.body`，z-index 100000 > 浏览弹窗 overlay 的 99999；三关闭（外点/Esc/滚轮）直接 `sf_popup.attachPopupDismiss` + `clampToViewport` 钳位，勿手写监听。
- `close()` 必须联动 `closeContextMenu()`——菜单不在 overlay DOM 子树内，overlay.remove() 不会带走它。
- contextmenu 处理器需 `preventDefault()` + `stopPropagation()`；右键另一图片时 pointerdown 先触发外点关闭旧菜单，再开新单，顺序天然安全。

### 3. 行为约定

- 非 PNG（jpg/webp 等）两菜单项恒显：readPngWorkflowData 按 PNG magic 校验返回 null → toast「未内嵌工作流数据」，fail-safe 与 LoRA 面板一致。
- 载入工作流先关浏览弹窗再异步载入（用户意图明确离开浏览），失败仅 toast 可见。

---

## 35. SFMaskFill：统一填充节点（合并 SFMaskedFill / SFMaskFillColor）

> 背景：`nodes/mask/masks.py:MaskFill` 合并原 `MaskedFill`（neutral/telea/navier-stokes）与 `MaskFillColor`（纯色+opacity+skip）为单节点 `SFMaskFill`，前端 `web/sf_mask_fill.js` 按 `fill_mode` 条件显隐，`web/sf_color_picker.js` 适配新类名。决策：**直接删除旧键**（破裂式合并，已获确认）、`falloff` 与 `skip_if_all_white` 提升为全局（对所有模式生效，color 也羽化）、`fill_color/opacity` 仅 color 模式显示。

### 1. 合并策略与兼容性

- **破裂删除**：`__init__.py` 仅留 `SFMaskFill`，历史工作流含 `SFMaskedFill`/`SFMaskFillColor` 将加载失败（Missing node type），需用户手动替换。保留别名可无破裂，但本次按任务目标直接删除。
- **参数统一**：`fill_mode=[color,neutral,telea,navier-stokes]`；`fill_color:COLOR` + `opacity:FLOAT` 仅 color 分支读取；`falloff:INT` 与 `skip_if_all_white:BOOLEAN` 全局；`DESCRIPTION` 中文并注明合并。
- **纯函数抽取**：`_parse_fill_color`（hex 字符串与 RGB 列表双形态）与 `_apply_falloff(alpha,falloff)`（`make_odd+binary_erosion*gaussian_blur`）供单测与复用，避免分支内联副本。

### 2. falloff 全局化与尺寸/批次对齐

- **尺寸重采样**：沿用 `MaskFillColor` 的 `mask2tensor→rescale_image→tensor2mask` 宽松策略（`image H/W != mask H/W` 时缩放），对 telea/neutral 同样生效，替代旧 `MaskedFill` 的严格 `assert`。
- **批次广播**：`alpha` 单例（`[1,1,H,W]`）自动 `repeat` 到 `image` batch；多 batch 仍逐 slice 处理。
- **falloff 顺序**：`mask_floor→mask_unsqueeze→[resize 重算]→_apply_falloff→分支混合`；color 分支的 `alpha_with_opacity=alpha*opacity` 使用已羽化的 `alpha`，与 telea 的 `alpha_bc` 羽化同源。

### 3. 前端条件显隐（`web/sf_mask_fill.js`）

- 仅对 `SFMaskFill` 生效：`fill_mode` 非 `color` 时 `fill_color.hidden=opacity.hidden=true`，`setDirtyCanvas` 重绘；`fill_mode.callback` 包装原回调 + `configure`/`onAfterGraphConfigured` 双重 `setTimeout(toggle,0)` 保工作流恢复后状态一致。
- `sf_color_picker.js` 的 COLOR 序列化 hack（`SFImageResizePlus` 同款）同步改 `SFMaskFill` 类名判定，不新增逻辑；`check_web_imports.py` MODS 加 `sf_mask_fill`。

### 4. 模块边界

- `nodes/mask/masks.py:MaskFill`（统一实现，FUNCTION=execute，5 输入 + skip 全局）
- `web/sf_mask_fill.js`（单文件扩展，`sfnodes.SFMaskFill`，无导出）
- `web/sf_color_picker.js`（COLOR widget，适配新类名）
- `__init__.py` 唯一真源 `SFMaskFill`

---

## 36. SFImageSceneSplit：镜头切分（硬切/黑白场/溶解 + 负索引/max_frames 首 N 帧 + LIST 全段）

> 背景：`nodes/image/scene_split.py:SFImageSceneSplit` + `sf_utils/scene_detect.py` 纯逻辑（无 torch/ComfyUI 依赖）。输入为视频连续帧 `IMAGE [B,H,W,C]`，按阈值检测硬切/黑场/白闪/溶解四类切点，去抖后按 `segment_index` 输出指定段与全段 LIST（首 N 帧截断、负索引、越界抛错）。

### 1. 检测策略（纯逻辑，无重依赖）

- **缩略**：每帧最长边 `160` 下采样（`cv2.INTER_AREA`→PIL→最近邻三级回退），转灰度后 `32-bin` 归一化直方图 + 均值亮度 `[0,1]`。千帧 256px 内 <1s。
- **硬切**：`hist` 用 Bhattacharyya `1-BC`，`diff` 用缩略图均差 `/255`，`d>threshold` 即切 `i+1`。阈值默认 `0.30` 对应 `BC=0.7` 中等相似度。
- **黑/白场**：灰度均值 `<black_threshold`（默认 0.08）/ `>white_threshold`（默认 0.92）的连续段边界各切一刀（`s` 与 `e+1`），全黑/全白不切。
- **溶解**：滑窗 `W=dissolve_window`（默认 8），累积距离 `D=dist(h[i],h[i+W])`，`D>threshold && max_step<threshold && avg_step>dissolve_threshold(0.18)` 时在 `i+W//2+1` 记切点；已硬切的窗被 `max_step` 过滤不重复。
- **去抖**：候选切点排序后 `c-last<min_scene_len(12)` 删后者；尾段不足也合并到上一段，最终补 `[0,B]` 哨兵。

### 2. 节点契约（与 ComfyUI 张量/列表约定）

- **逐帧转 `uint8` 生成器**：`for i in range(B): arr=(images[i].cpu().numpy()*255).astype(uint8)` 避免一次性 `B*H*W` 批拷贝 OOM；`C==1` 复制为 3 通道，`>3` 截断 RGB。
- **输出**：`images [N,H,W,C]`（选中段首 N 帧截断后）、`count INT`（截断后）、`cuts STRING(JSON [0,..,B])`、`scene_count INT`、`all_segments IMAGE+OUTPUT_IS_LIST`（每段一批，`all_segments[segment_index]` 即选中段原长）。
- **索引**：`segment_index` 负数走 `scene_count+idx`，越界 `ValueError("越界 ... cuts=...")`（对齐 `batch_index.py` 风格）。

### 3. 测试与复用

- 纯逻辑 `sf_utils/scene_detect.py` 无 comfy：`detect_scenes` 支持 `np.ndarray [B,H,W,3]` 与 `iterable` 帧，`_to_uint8_rgb/_downscale_and_gray/_hist` 内部三级回退保证 CI 无 `cv2` 仍可跑。
- `tests/test_scene_detect.py` 覆盖硬切/黑白场/溶解/min_len 去抖/单帧/空/float + 节点 mock `torch`（`FakeTensor` 模拟 `detach/cpu/numpy/__getitem__` 切片）断言选段/负索引/max_frames/越界。

### 4. 模块边界

- `sf_utils/scene_detect.py`：`detect_scenes/split_scenes` + `_process_frame/_hist_distance/_downscale_and_gray`（纯函数）
- `nodes/image/scene_split.py`：`SFImageSceneSplit`（9 输入 + 5 输出，`OUTPUT_IS_LIST[4]=True`）
- `__init__.py` 唯一真源 `SFImageSceneSplit`

## 37. SFImageCropExpand：出界裁剪/外绘预处理（复刻 YCNodes Load Image Crop Expand）

> 背景：`nodes/image/crop_expand.py:SFImageCropExpand` + `web/sf_crop_expand_lib.js`（纯几何）+ `web/sf_crop_expand.js`（主扩展）。复刻 ComfyUI-YCNodes_Toolkit `ycImageCrop`：节点上加载图片，拖拽一个**可出界**的裁剪框（负坐标/越界=外扩），输出填充色画布（交集=原像素，出界=纯色）+ 白=扩展区遮罩 + 宽高——外绘工作流的预处理节点。

### 1. 与原版的三处确认差异（用户拍板）

- **图片持久化**：原版序列化时清空 base64（widget+properties），工作流重载即丢图（仅会话内 id 缓存）。本实现复用 `crop.py` 的 `/api/sfnodes/crop/upload_src` 路由落盘 `input/sfnodes_crop/`，状态只存 `src_path`，重载经 `/view`（`buildSourceURL` + cacheBust）恢复预览。
- **不接上游 IMAGE**：与原版一致，仅手动加载（Load Image 按钮 / 拖放文件到节点显示区）。
- **隐藏状态收敛**：原版 7 个隐藏 widget 逐个同步；本实现单个 `properties.sfCropExpandState`（JSON 字符串随工作流保存）+ graphToPrompt 注入隐藏输入 `SFCropExpandJson`（outpaint 先例），Python `_parse_state` dict/str 双容错。`aspect_ratio`/`custom_w/h` 只进状态给前端拖拽用，后端忽略。

### 2. 交互层复用判定（为何不复用 sf_crop_framework）

- `sf_crop_core.js:CropEditor` 把裁剪矩形**钳制在图界内**（restore/snap 全链路 `Math.max(0,...)`），支持出界需深改 core/render/save 全链路并波及既有 SFImageCrop——违背最小改动。故交互层按原版形态新写（节点上直接画布拖拽，约 500 行），**但交互数学全部抽入纯库** `sf_crop_expand_lib.js`（无 app 依赖可 .mjs 直测）：`computeDisplayMetrics`（含**拖拽冻结快照**——onMouseDown 时冻结 scale/offset/displayMin，全程共用防"框扩张→scale 变小→鼠标反算漂移"自反馈）、`getHandleAtPoint`（命中半径=10px/scale）、`updateCropByDrag`（八向+比例约束+最小 10px）、`applyRatioToRect`。
- 复用清单：`CropAPI`（crop_core 新增 `uploadSrc` 方法——crop.js 内联上传片段收敛进共享 API 束）、`sf_common.js`（sfToast/buildSourceURL/getSfAccent——比例按钮选中色跟随全局强调色）、`sf_popup.js`（Custom 比例弹窗三关闭；输入框 keydown 放行 ctrl/meta/alt）、`sf_utils/common.py:_parse_fill_color`（自 `nodes/mask/masks.py` 提升为公共实现，masks.py 改同源导入——提升而非副本）。

### 3. 后端合成与契约

- `_compose_expand(src, x, y, w, h, fill_rgb)` 纯 numpy：fill 画布 → 源图矩形与裁剪矩形求交贴回 → mask 交集处置 0（黑=原图），其余 1（白=扩展区）。输出 `[1,H,W,3]` / `[1,H,W]` + `(w,h)` INT + `filename` STRING（src_path 原样——input 相对路径可直连 LoadImage，无源空串）。
- `_clamp_crop` 对齐原版输入域（x/y ±4096、w/h 1..8192）；`_safe_join` 复用 crop.py（同包 `from .crop import _safe_join`）防路径穿越；`IS_CHANGED` 键 = `(mtime_ns, size) + rect + fill_color`（§3 禁 NaN）。
- 缺源/载入失败退化：纯填充画布 + 全白遮罩（不崩，语义正确——整幅都是扩展区）。

### 4. 测试

- `tests/test_crop_expand.py`：mock torch/aiohttp/folder_paths（同 test_crop.py 机制，先加载 crop.py 再加载 crop_expand.py 满足同包导入），覆盖结构/注册键（根 `__init__.py` 文本断言）/纯函数/execute 磁盘源与缺源退化/IS_CHANGED。
- `tests/test_crop_expand_js.mjs`：lib 拷 .mjs 直跑——冻结快照透传、坐标往返、八向拖拽+比例约束（⚠ 自写期望值时要按 `startRect.w + dx` 算，两处期望值算错被测试反抓）、`RATIO_PRESETS_ROW2`/`LAYOUT` 与原版逐字一致。

### 5. Browse 按钮：复用 Image Browser 弹窗（2026-09 补充）

- `image_browser.js:showImageBrowser(node)` 原与 LoadImage 系的 `image` widget 强耦合（选中高亮/定位/写入）。参数化加可选 `opts.onPick(value, item, type)` **选择器模式**：传入后 `imageWidget` 置 null，widget 写入路径整体跳过，选中项交宿主回调，「定位当前」按钮一并隐藏（无 widget 值可定位）——SF Load Image Browser 原行为零改动（不传 opts 时走原 widget 路径）。
- SFImageCropExpand 的 Browse 按钮走选择器模式：onPick 里 `parseAnnotatedImageValue` + `buildSourceURL`（output 项自带 `[output]` 注解）→ fetch `/view` 原始字节 → FileReader dataURL → 既有 `loadAndStoreImage`（落盘 + 满框 + 状态同步）。零后端改动。
- **Ctrl+V 粘贴剪贴板图片（2026-09）**：复用 `sf_common.js:installPasteHandler`（选中节点判定/防抢输入框 paste/清扫 ComfyUI 自动创建的 `pasted/` LoadImage 均在公共实现内，按 `comfyClass:hook` 幂等），宿主只赋 `node._sfExpandPaste = dataURL → loadAndStoreImage`（与 Load Image/Browse/拖放同链路）；本节点无 image 输入，公共实现的断 upstream wire 逻辑自动跳过。未做按钮式主动粘贴（`navigator.clipboard.read` 需安全上下文，LAN http 不可用）。
- ⚠ 跨模块 import 的 `image_browser.js` 必须入 `check_web_imports.py` MODS（否则 MISSING MODULE 报错）。
- **布局改版（2026-09，用户草图）**：比例按钮改**画布区左侧竖列**（`RATIO_PRESETS_COL`，Free 置顶 + 7 预设 + Custom/Reset/Color 收尾，节点顶直通画布区底，`LAYOUT.ratioColW/ratioColGap` 由图片区让宽）；Load Image/Browse 移**底行**与信息文本同排（右对齐到输出槽区前；按钮 y 标记 `BOTTOM_Y="bottom"` 经 `buttonRect` 运行时解析——节点高度可变，坐标不能在 buildButtons 固定）。**MIN 收敛**（460→480→360/300→320/300）：底行宽度瓶颈靠信息文本 `measureText` 溢出截断 "…" 解除（按钮右缘 135 + 最小文本窗 120）；2026-09 底行再压缩（SFImageBrushMask §45.13 同款：`Load Image`→`Load` 短文案，右缘 135→106，文本窗同步收窄），高度由竖列 11 项驱动不变。

## 44. 画布节点拖拽缩小外溢：computeSize 包装钳最小尺寸（2026-09）

> 背景：SFImageCropExpand 用户手动把节点拖到 `MIN_NODE_WIDTH/HEIGHT` 以下，固定像素布局的按钮行外溢出节点。`clampNodeSize` 只在 onNodeCreated/onConfigure 钩一次，管不住后续拖拽。

- **双端拖拽 resize 的最小值都取自 `node.computeSize()`**（前端包 1.51.10 实测，当时版本：onDrag 处理 `let l=node.computeSize(); c.width<l[0]&&(c.width=l[0]); c.height<l[1]&&(c.height=l[1]); node.setSize(c.size)`；legacy litegraph 同款）——**包装 `nodeType.prototype.computeSize` 返回 `max(原值, MIN)` 一处改动同时钳住两条路径**，且只抬不降（`expandToFitContent`/初始 sizing 均安全）。
- ⚠ **`node.onResize` 是 legacy-only**：Vue 前端（1.x）拖拽 resize 与布局同步路径均不触发（容器内实测仅 widget finalize 与 DOM 尺寸回写各一处调用）；sf_find_replace/sf_lora_plot 等先例的 onResize 钳制都显式 `isVueNodes()` 守卫。新钳制需求优先选 computeSize 包装。
- 纯函数 `ensureMinSize(w,h)` 收敛 lib（computeSize 包装与 clampNodeSize 共用，避免双份 Math.max）。⚠ 底部信息文本超界与节点大小无关：`computeDisplayMetrics` 的 areaH 原本未预留文本行，显示区填满画布时文本基线 y = nodeH+5 **恒超界 5px**（抬 MIN 无效，曾误判为"368 够用"）——修法是画布区高度让出 `TEXT_RESERVE=20`（基线 +15 + 字形余量），此后基线最坏 y = nodeH-15 恒在界内。
- 已保存工作流恢复（configure 直接赋 size 不走 computeSize）仍由 onConfigure 的 clampNodeSize 兜底；工作流里偏小的存量 size 会被抬到 MIN（可接受的显示修正）。
- **右下角 resize cursor 视觉修正**：拖动命中区维持原生 15×15（`resizeHandleSize`，用户确认无需扩大）；问题是 cursor 几乎不显示——`pointer.resizeDirection` 被**两处反复清空**（诊断栈：①`updateMouseOverNodes` hover 判定切换时无条件清，右下角边缘 over 判定与命中区不重合时闪清；②**第三方扩展重放 `processMouseMove`**——如 `Comfyui_LG_Tools/queue_shortcut.js` 包装后每次物理移动跑 ≥2 遍，第二遍清掉刚设的 dir；帧尾 `updateCursorStyle` 读到空即写回 default）。修法：扩展注册一个后执行的 `window mousemove` listener，原生 15×15 区内**直接写 `canvas.style.cursor="nwse-resize"`**（dir→帧尾 updateCursorStyle→css 的间接链路被清空/时序问题拖垮，直接写绕开；同时补写 `pointer.resizeDirection="SE"` 双保险），区外仅在 cursor 是自己写入时恢复 `""`（`_sfCursorOwned` 标志，不覆盖槽 grab 等原生 cursor），`pointer.eDown` 时跳过——诊断手段：包装属性 setter（`Object.defineProperty` + 打印 set 调用栈）定位"状态被谁清"；曾试过包装 `findResizeDirection` 扩大到 30×30，被清空问题依旧（清空与命中区大小无关），按用户要求回退。

---

## 45. SFImageBrushMask：节点内画笔遮罩（复刻 YCNodes Load Image Brush Mask）

> 背景：`nodes/image/brush_mask.py:SFImageBrushMask` + `sf_utils/brush_mask.py` 纯逻辑 + `web/sf_brush_mask_lib.js`（纯几何/解析）+ `web/sf_brush_mask.js`（主扩展）。复刻 ComfyUI-YCNodes_Toolkit `ycimagebrushmask`：节点上加载图片后画笔直接涂抹二值遮罩（brush 涂白 / eraser 擦除），输出源图 + 遮罩 + 宽高 + filename。

### 1. 与原版的确认差异（开工前拍板：5 输出 / 纯手动 / 预览语义保留 / 复用 crop 路由 / 隐藏 JSON）

- **图片持久化**：原版 base64 进 widget（工作流巨大，作者 README 自述"导出前先清图片重建节点"）+ 会话内 `Map` 缓存（刷新即丢图）。本实现复用 `crop.py` 的 `/api/sfnodes/crop/upload_src` 落盘 `input/sfnodes_crop/`（`CropAPI.uploadSrc` 前端束 + `_safe_join` 后端守卫），状态只存 `src_path`，重载经 `/view` 恢复预览——**零新增后端路由**（§37 同款）。
- **状态收敛**：原版 5 个 widget（brush_data/brush_size/image_base64/image_width/image_height，`widgets_start_y=-4.8e8` hack 藏件 + `onSerialize` 清空 base64）→ 单个 `properties.sfBrushMaskState` JSON + graphToPrompt 注入隐藏输入 `SFBrushMaskJson`（§37 outpaint 先例；**无隐藏 widget 创建**——注入即通道，properties 随工作流保存即恢复通道）。
- **输出 +1**：原版 4 输出（IMAGE/MASK/INT/INT）→ 加 `filename` STRING（src_path 原样，可直连 LoadImage，未加载空串；§37 同款）。
- **mask 恒二值**：Opacity/取色仅前端预览叠加透明度/颜色，后端忽略（与原版语义一致——原版解析 opacity/color 后丢弃；DESCRIPTION 注明，避免"调了没效果"的误报）。

### 2. 颜色误判 bug（原版真 bug，复刻时修复）

- **症状**：旧格式多点笔触 `brush:20:1.0:10,10;20,20` 中 `parts[3]="10,10;20,20"`，`split(",").length===3` 被当成 `r,g,b` 颜色——JS `parseInt("10;20")→10` 静默截断不抛错，`(10,10,20)` 通过 RGB 校验 → 整笔点列被吃掉（`pointsStr=""`），该笔凭空消失。Python 侧 `int(float("10;20"))` 抛 ValueError 反而幸免——**双端行为分叉**。
- **修法**：颜色判定加严格形态检查——三段必须全匹配 `/^\s*\d+\s*$/`（`sf_brush_mask_lib.js::parseStroke`），`"255,0,0"` 走颜色分支，`"10,10;20,20"` 走点列分支。Python 侧 `int(float())` 天然严格，无需改动。回归用例：新无色多点格式双端同期望锁定。

### 3. lean 注入（改预览不重跑）

- graphToPrompt 注入载荷经 `leanState()` 过滤：只含 `src_path/src_w/src_h/brush_size/strokes`；`brush_opacity/brush_color/eraser_color/brush_mode` 不进 prompt——改颜色/透明度不改变 prompt 输入哈希。`IS_CHANGED` 的 `_lean_key` 同口径（`|` 分隔 strokes JSON + brush_size + src），预览字段双重排除（测试锁定：改三预览字段键不变）。
- 笔触落定时坐标 `Math.round` 取整存 strokes（后端 `int(float())` 同语义，双端镜像）。

### 4. 前端移植要点（§37/§44 同款修复全套）

- 面板几何：顶面板（Load/Browse/Clear/Undo/Eraser + 右对齐双色块 + Size/Opacity 双滑块）与绘制/命中共用 `buildControls()` 几何；`MIN_NODE_WIDTH/HEIGHT=420×320` 按按钮行宽（≈300 + shiftRight 80 + 边距）推导，注释留推导过程；底信息行文本 `measureText` 溢出截断 `…`（§37 同款）。
- 绘制：`computeSize` 包装 + 创建/恢复 `clampNodeSize`（§44）；右下角 cursor 后注册 mousemove 补写（§44，独立 `_sfBrushMaskCursorPatch` 标志——与 CropExpand 的 patch 各扫自家 `comfyClass`，共存无干扰）；`installPasteHandler`（`comfyClass:hook` 键已防多类互斥）+ 拖放 + Browse 选择器模式共用 `loadAndStoreImage` 单链路；换图清空 strokes（原版语义）。
- 落笔：`onMouseDown` 面板命中（滑块 > 色块 > 按钮）→ 画布区左键起笔（`clampToImage` 钳制，与原版 `valueUpdate` 一致）→ `onMouseMove` 距离 >1px 追加（降噪）→ `onMouseUp` + 全局 mouseup 兜底落定（移出节点松开也落笔）。进行中笔触只进内存 `_sfBrushCur`（不写 properties），落定才追加 strokes 写回——拖拽中途不污染可序列化状态。

### 5. 测试

- `tests/test_brush_mask.py`：mock torch（numpy 代理，`[None,]`/`.shape` 原生支持）/aiohttp/folder_paths（先加载 crop.py 满足同包 `_safe_join` 导入）；覆盖结构 + 注册键文本断言 + `_parse_state` + 纯逻辑三格式/越界/往返 + 栅格化（单点/线段连续/erase 清零/空笔全黑）+ execute 磁盘源与缺源退化 + IS_CHANGED（含预览字段不进键）。
- `tests/test_brush_mask_lib.mjs`：lib 拷 `.mjs` 直跑——MIN/cursor 判定/显示坐标往返/钳制/三格式解析/build 往返（含 §2 颜色误判回归）。

### 6. 模块边界

- `sf_utils/brush_mask.py`：`parse_strokes`（旧串→裁剪笔触）/`parse_state_strokes`（state 结构化→裁剪笔触）/`build_brush_data`（往返/兼容）/`rasterize_strokes` + `_draw/_erase_circle/_stamp_line`（仅 numpy）。
- `nodes/image/brush_mask.py`：`SFImageBrushMask` + `_parse_state`/`_lean_key`（5 输出，`filename`=src_path 原样）。
- `web/sf_brush_mask_lib.js`：`LAYOUT/MIN/ensureMinSize/hitResizeCornerSE/computeDisplayMetrics/localToImage/imageToLocal/clampToImage/parseStroke/parseBrushData/buildBrushData`（禁 import sf_common）。
- `web/sf_brush_mask.js`：主扩展（`sfnodes.BrushMask`，复用 `CropAPI.uploadSrc`/`installPasteHandler`/`showImageBrowser` 选择器/`buildSourceURL`/`sfToast`/`getSfAccent`）。
- 数据契约：`SFBrushMaskJson`（hidden STRING，graphToPrompt 注入 lean JSON）；`properties.sfBrushMaskState`（全量，含预览字段）；`sfnodes_crop` subfolder（与 Crop 共用目录，前缀 `brushmask_` + 时间戳防撞名）。

### 7. 控件 CropExpand 化（2026-09，用户确认改版）

- **布局**：初版顶面板（按钮行 + Size/Opacity 横向滑块 + 色块）改为 CropExpand 同形——左工具竖列 `TOOL_COL` 10 项（Brush/Erase 模式 → Clear/Undo → Size±/Opa± → BCol/ECol）+ 底行 Load Image/Browse 与信息文本同排。`LAYOUT` 与 `computeDisplayMetrics` 公式与 `sf_crop_expand_lib.js` 逐字同形（竖列 `toolColW/toolColGap` 让宽、底行 `bottomH` 让高），`buttonRect` 的 `BOTTOM_Y` 运行时解析亦同款。
- **滑块→步进器**：30px 竖列放不下横向拖拽滑块，改为纯函数步进（`stepBrushSize` 步长 2 钳制 1..200 / `stepOpacity` 步长 5% 钳制 0.1..1.0，lib 可测）；实时数值进底行信息文本（`Brush 80 · Op 50% · Strokes 3 · 512×512`，超宽截断 `…`）。取色按钮背景即当前色、文字按亮度取黑/白（CropExpand Color 按钮同款）。
- **步进器滚轮快调**：悬停 S±/O± 时滚轮直调（上滚增大/下滚减小，每 tick 一步）。引擎无节点级 `onMouseWheel` 钩子（1.51.9 实测零命中），故用 window capture + `{passive:false}` 先手拦截（`installPasteHandler` 同款）；仅命中四步进器（纯函数 `hitStepper`/`wheelDir`/`wheelAction`，lib 可测）且无 Ctrl/Meta（捏合缩放）时 `preventDefault+stopPropagation`，其余放行画布缩放；折叠节点与界外坐标跳过。
- **影响面**：纯前端（lib + 主扩展 + mjs 测试 + 本节/架构一行）；后端、隐藏输入契约、`tests/test_brush_mask.py` 零改动。

### 8. 底栏残影：半透明背景盖住按钮（2026-09，用户实测）

- **症状**：落笔时底栏出现多重半透明边框，Load/Browse 按钮发虚（竖列按钮个个清晰）。
- **排查**：先怀疑帧间因素——分段 console 诊断（205 次 `onDrawForeground` 调用：size 恒 420×320、ds 恒定、零重入、零同帧多画、`clear_background=true`+`dirty_area=null`）一锤排除（诊断脚本见 §45 配套：包装计数 `size/ds` 分布 + 重入 + 同毫秒多画，四个假说一次证伪）。帧间静态 + 逐帧全清 ⇒ 只能是**单帧内画序错**。
- **根因**：底栏背景（`rgba(40,40,40,0.9)`）画在底行按钮**之后**——后画盖先画，按钮以一成亮度透出：发虚的按钮 + 按钮边框与底栏边框错层 = "多重边框"。CropExpand 是先底栏后 `drawButtons`，合并循环时把顺序搞反了。
- **修法**：底栏背景块移到按钮循环之前（`tests/test_brush_mask_smoke.js` 锁定：FakeCtx 记录 op 流，断言底行按钮 fill 在底栏 fill 之后；已验证旧错序 FAIL/新顺序 PASS）。**教训：同节点多层半透明 chrome 必须按"背景→控件→文本"分层绘制，写循环合并时保持层序**——冒烟断言时注意同色复用陷阱（底栏与竖列底条同 `rgba(40,40,40,0.9)`，必须用 `roundRect(高29)` 定位底栏，不能按色取第一个 fill）。

### 9. SAM 右键图层：核心 SAM3_Detect 委托（2026-09）

> 背景：刷子节点右键菜单用文本 prompt 跑 SAM 分割，结果转 fill 矢量笔触并入统一管理。`nodes/image/brush_mask_sam.py`（委托链 + 三路由）+ `brush_mask.py`（fill 栅格化接入）+ `web/sf_brush_mask.js`（菜单/对话框/fill 绘制）。

### 1. 只调核心，不碰第三方（此前走弯路的教训）

- 初版调研误判 core 无 SAM（只搜了 `comfy/`，漏了 `comfy_extras/`），一度计划复用 GPL-3.0 的 ComfyUI-RMBG 包（拷贝传染 / 自研 safetensors 权重桥 / iopath-ftfy 依赖链全是坑）。用户指正后确认：**core 0.35 `comfy_extras/nodes_sam3.py:SAM3_Detect` + `comfy/ldm/sam3/` + 判型 `SAM31` 全套原生支持**，`models/checkpoints/sam3.1_multiplex_fp16.safetensors` 三 marker 全中。结论：**新模型先查 `comfy/model_detection.py` + `supported_models.py` 再谈复用第三方**。
- 委托链（与手写工作流等价，全 core API）：`get_full_path_or_raise("checkpoints")` → `load_checkpoint_guess_config(output_vae=False)` 同文件拆 MODEL+CLIP → `CLIPTextEncode.encode(clip, prompt or "object")` → `SAM3_Detect.execute(model, image, conditioning, threshold, refine_iterations（对话框 0-5，默认 2）, individual_masks=False)` → `out[0]`（`NodeOutput` 支持下标）。MODEL/CLIP 按 checkpoint 常驻缓存（core 自带 `cached_patcher_init` 之外再包一层防重复拆文件）；全部 import 函数内 lazy，旧版 core 缺席时节点照常加载、点击才报中文错。

### 2. 位图→矢量 fill 笔触，不设覆盖层（现状；决策见 §8）

- SAM 输出位图，经 `mask_to_fill_strokes` 转 fill 笔触直接并入列表——添加/擦除/撤销/Clear 全通用。`sam_prompt`/`sam_threshold`/`sam_refine` 仅对话框记忆，不进注入；`sam_mask_path` 已退出状态（旧工作流残留被忽略）；lean/IS_CHANGED 回归纯笔触键。
- 前端 `drawStrokePath` fill 分支整体填充；后端 `rasterize_strokes` fill 分支 PIL 多边形填充。

### 3. 右键菜单与对话框

- `getExtraMenuOptions` 原型包装（any_pack 先例）：「SAM 蒙版：文本选择…」（无源图先 toast 拦）/「卸载 SAM 模型」（清缓存 + `empty_cache`）。对话框仿 CropExpand Custom 比例弹窗（prompt + 阈值 + refine 0-5，Enter 确认/Esc 关闭/放行组合键，记住上次）。
- 门禁诊断路由 `GET …/sam_status`（core/模型/已加载三态）；`POST …/sam(_unload)` 走 `api.fetchApi + sfApiUrl`（托管基址/鉴权，crop_core 同款）。

### 4. 测试

- `tests/test_brush_mask_sam.py`：mock torch（numpy 代理 + 链式 FakeTensor）/comfy.sd/nodes/CLIPTextEncode/comfy_extras SAM3_Detect——委托链（空 prompt→object/阈值钳制/refine 上下限与非法兜底/union 参数）/缓存单拆/缺模型报错/unload/`_handle_sam`（400/回 strokes+coverage/空结果/500）/轮廓转换（点序/面积过滤/数量上限）/fill 栅格化真实像素/execute 忽略旧残留。
- smoke 扩展：菜单三项存在性 + 无层/有层清除项显隐；`/scripts/api.js` 绝对导入同步进改写表（漏改会 `ERR_MODULE_NOT_FOUND`，已踩）。

### 5. 路由外推理必关进度钩子（2026-09，真机报错）

- **症状**：`[SFImageBrushMask:sam] inference failed: 'PromptServer' object has no attribute 'last_prompt_id'`——模型/CLIP 加载日志全正常，死在 `SAM3_Detect.execute` 内。
- **根因**：新版 core 的 `PROGRESS_BAR_HOOK` 在广播进度时读 `PromptServer.last_prompt_id`（该属性只在队列执行上下文中存在）；SAM 路由跑在执行之外，`ProgressBar(B)` 首次 `update` 即炸。**凡在路由/后台线程里调 core 节点 execute，都要先处理进度上下文**。
- **修法**：推理期 `comfy.utils.PROGRESS_BAR_HOOK = None` + `finally` 还原（`ProgressBar` 为 None 时跳过广播分支，core 源码确认）。不写 server 状态（并发执行的已创建 bar 持有旧引用不受影响）；mock 测试锁“期内置空 + 事后还原”。

### 6. SAM 叠加槽串扰源图槽（2026-09，真机白屏）

- **症状**：SAM 蒙版显示为全屏白色遮罩。
- **根因**：`loadSamOverlay`/`clearSamLayer` 错用源图槽 `_sfSamImg`（加载时把蒙版图写进源图槽并清空源预览，清除时再清空一次）——照片被蒙版顶替 + 白色叠加 = 满屏白。**教训：同节点多图层预览必须一层一槽（源图/笔触/SAM 各自独立属性），命名即注释槽位归属**。
- **修法**：SAM 专用 `_sfSamMaskImg` 槽，smoke 锁“恢复/清除不碰源图槽”。另加覆盖率回传（`coverage` 进 toast）：若覆盖 ~100% 则是 prompt/阈值问题（检测框罩全图），非显示 bug——显示与内容二分法的第一手证据。

### 7. 叠加预览：亮度蒙版 ≠ 透明蒙版（2026-09，满屏白）

- **症状**：后端输出 mask 正常，节点上 SAM 叠加层满屏白。
- **根因**：染白用 `destination-in` 想按蒙版裁形状——但该算子按 **alpha** 合成，灰度 PNG 无 alpha 通道（全像素 alpha=255），整张白底一像素都没被裁。错把“亮度蒙版”当“透明蒙版”（inpaint 的 `destination-in` 操作的是自带 alpha 的 RGBA 遮罩画布，两码事）。
- **修法**：逐像素把亮度搬到 alpha（白 RGB + alpha=亮度，inpaint `_loadMaskFromURL` 同款转换，灰度取三通道均值保渐变）。smoke 锁：4×2 半白蒙版桩 → putImageData 的 alpha 通道必须白区 255/黑区 0。**教训：canvas 合成前先问“形状载体是 alpha 还是亮度”**——排查时顺带抓到 harness 旧桩残留（`globalThis.Image = class {}` 把新桩覆盖，`new Image()` 静默走空类，属“测试桩自覆盖”一类，改桩后 grep 全文件确认无残留）。
- **附带**：`sam_status` 门禁 + 成功 toast 带覆盖率，分流“内容问题（prompt/阈值）vs 显示问题”。

### 8. SAM 转矢量 fill 笔触，不再区分图层（2026-09，用户拍板）

- **动因**：覆盖层链路连环出显示 bug（槽串扰源图槽 §6、亮度/alpha 误用 §7），且位图层与笔触是两套心智（清除/撤销/擦除语义分叉）。改为：SAM 结果经轮廓追踪转为 fill 笔触直接并入列表——添加/擦除/撤销/Clear 全通用，覆盖层/并集/文件键整类删除。
- **管线**：`mask_to_fill_strokes`（`cv2.findContours` RETR_EXTERNAL + `approxPolyDP(eps=1.5)` + 面积 <16px 丢弃 + 只留最大 64 个，一轮廓一笔，Undo 以物体为粒度）→ 后端 `rasterize_strokes` fill 分支（PIL `ImageDraw.polygon` 整体填充，无需 cv2，本地可真测）→ 前端 `drawStrokePath` fill 分支（`closePath+fill`，擦除笔触按画序覆盖）。`parse_state_strokes` 放行 `fill` 模式；旧串格式无 fill（只走 state）。
- **契约**：路由回 `{strokes, count, coverage}`（不落盘）；`sam_mask_path` 退出状态（旧工作流残留被忽略）；lean/IS_CHANGED 回归纯笔触键。空结果回 `strokes: []` + warn toast（笔触不变）。
- **测试**：cv2 用行为桩（方框轮廓 + 鞋带面积 + approx 恒等——测我方管线：点序/面积过滤/数量上限/格式，而非 cv2 本体）+ PIL 真实像素断言（fill 块内外/erase 可擦 fill）+ smoke（菜单两项 + fill 触发 `fill` op）。

### 9. 橡皮擦真擦除预览（2026-09，用户诉求）

- **动因**：后端早就是真擦除（`_erase_circle` 置 0），唯独前端预览是红色叠加——红块≠输出，擦空区凭空出红块，fill/擦除交错时预览与输出分叉。
- **做法**：离屏遮罩画布按画序合成（brush/fill 盖不透明白 → erase `destination-out` 打洞）再 `globalAlpha` 一次贴回；离屏按源图像素绘制（lineW 取源图直径，与后端印章同语义），画布按源图尺寸缓存。主画布禁用 `destination-out`（会连照片一起擦）。inpaint 编辑器同款架构（其遮罩 canvas + 烘焙），此处无羽化需求故单画布逐帧重建。
- **ECol 删除**：真擦除后无红色可染，竖列 10→9 项（MIN 保持 320，与存量 size 兼容）；`eraser_color` 状态惰性遗留，旧工作流无感。后端零改动。
- **测试**：smoke 离屏 op 流断言（`destination-out` 出现且事后恢复 `source-over`、纯 brush 无打洞）+ 主画布无红色 style + 单次 blit；画布缓存命中断言前先清缓存（与实现同因）。

### 10. 悬停笔刷光环（2026-09）

- **做法**：`onMouseMove` 常驻记录图片区内光标（画与不画都记，区外置空）；`onDrawForeground` 经 `canvas.node_over === node` 门控画环——离开节点后无额外监听，靠 hover 切换自带重绘消环。半径 `size/2×scale` 随缩放/步进/滚轮实时生效；brush 强调色实线 + 圆点、erase 白虚线 + 圆点（inpaint `_drawCursor` 同款）。
- **测试**：smoke 断言位置记录/区外清空/hover 时 arc 半径（100px 图在 420×320 节点下 scale=2.74，80 笔刷环半径 109.6）/非 hover 无 arc；app 桩 canvas 改为测试可注入的 `__bmCanvas` 对象。

### 11. 笔刷 shortcut [ ]（2026-09）

- **做法**：选中本类节点时 `[` 缩小/`]` 放大，步长同 S± 步进器；OS 自动连发即按住连调，故不做 inpaint 式按住加速。选中集复用 `sf_canvas_align_lib.js::getSelectedNodes`（四形态兼容），`e.repeat` 放行。（初版误判无官方钩子只写 window 通道，见下“实测无效复修”。）
- **互斥**：输入框内 / 修饰键 / 全屏编辑器打开（`.sf-px-overlay` 存活，crop/inpaint 自家 handler 接管）时跳过；命中才 `preventDefault+stopPropagation`，不干扰画布其他按键。
- **测试**：smoke 升级 window 桩为捕获型 + `__bmGraph` 可注入伪图/伪选中——放大/缩小/无关键/输入框/修饰键/编辑器互斥/未选中七用例；新 import 的 `sf_canvas_align_lib.js`（零依赖纯模块）同步进改写表拷贝。

### 12. 步进器步长进设置页（2026-09）

- **做法**：ComfyUI 设置页新增两个 slider 项（`sf_load_image_ui.js` 先例）——`sfnodes.BrushMask.SizeStep`（1–20，默认 2）与 `sfnodes.BrushMask.OpacityStep`（1–25 整数百分比，默认 5）；`init()` 幂等注册，读取失败回默认值。lib 步进函数步长参数化（默认保持原值，纯函数可测）；`buttonAction` 内读取——步进器/滚轮/`[ ]` 三路同一入口，自动统一，设置页改完即时生效（每次动作现读，无需 onChange 监听）。
- **测试**：mjs 加显式步长/非法步长回退用例；smoke 伪 settings 断言 init 注册默认值 + S+ 点击走自定义步长 + 非法值回退（用例间重置 size 前置，避免互相污染——已踩）。
- **复修（2026-09，真机）**：设置页改完按钮/滚轮生效、唯独 `[ ]` 恒 ±2——快捷键分支内联了 `stepBrushSize` 却漏传设置值（三路统一入口的设计被绕过）。改回经 `buttonAction` 收敛 + smoke 加“快捷键走自定义步长”回归锁。**教训：声称统一入口必须有测试锁，否则旁路悄悄分叉**。
- **实测无效复修（2026-09）**：初版 window 通道在用户环境无响应。排查确认两点：① 快捷键代码当时未进 HEAD（工作区未提交，部署侧无此代码）；② 发现官方通道——画布 `processKey` 把 keydown 分发给选中节点的 `onKeyDown`（core `addNodeKeyHandler` 再包一层，`=== false` 表已处理，链式兼容）。改为双通道：`nodeType.prototype.onKeyDown` 主办 + window 冒泡兜底（焦点在 body 时画布收不到），同物理按键经 `e.timeStamp` 去重（多选同戳亦然，首个节点全量调整后其余跳过）。smoke 加官方通道/同戳/跨通道去重/放行四用例。**教训：先查引擎有无官方钩子（`processKey`/`addNode*Handler`），再写全局监听；未提交代码先查部署差再查 bug**。

### 13. 底行压缩：MIN 420→320（2026-09）

- **做法**：底行按钮 `Load Image`(72)→`Load`(44)，Browse 移 x58（右缘 135→106）；信息文本窗同步收窄（溢出截断逻辑不变）；`MIN_NODE_WIDTH` 420→320（推导见 lib 注释），高度不动。显示坐标系公式自适应（areaW 随节点宽），cursor/MIN 高度/滚轮/快捷键均不受影响。
- **测试**：既有 mjs/smoke 用例与具体 MIN 值解耦（`ensureMinSize` 断言用 `MIN_*` 常量本身），零用例改动，全套通过即回归。

---

## 51. SFImageBatchRange：批次区间切片（复刻 KJ GetImageRangeFromBatch，2026-09）

- **动因**：`SFImageBatchIndex` 只能取单张（`num_frames=1` 特例），视频帧/遮罩序列需要 `start_index+num_frames` 任意区间；KJ 原版在 `ComfyUI-KJNodes/nodes/image_nodes.py:GetImageRangeFromBatch`，语义是 `images[start:end]` / `masks[start:end]` 双路可选、`start_index=-1` 取尾部 N 帧、尾部超出截断。
- **做法**：新建 `nodes/image/batch_range.py::SFImageBatchRange`（不动 `batch_index.py`，最小改动）——required `start_index(INT,min -1)` + `num_frames(INT,min 1)`，optional 双路 `images(IMAGE)/masks(MASK)`，`RETURN_TYPES=(IMAGE,MASK)`；区间解析抽模块级纯函数 `_resolve_range(count,start,num)`（`-1→max(0,n-num)`、`end=min(start+num,n)`、start 越界抛中文错），双路独立调用；IMAGE 要求 ndim==4、MASK 要求 ndim∈(2,3)，双空抛错。无前端 JS（无动态槽位，`check_web_imports.py` MODS 不加行）。后续 `SFImageBatchIndex` 亦补 `-1` 取尾帧（`index` min 0→-1，`-1→B-1`，其余负值抛错，与本节点一致）。
- **测试**：`tests/test_batch_range.py`（FakeTensor numpy 代理切片语义，零 torch 依赖）：结构 + 双字典一致 + `_resolve_range` 六用例（正常/截断/-1/-1 不足/越界/负值）+ execute 双路切片/单路透传/-1/截断/双空/ndim 非法。

---

## 60. SFImageCropExpand 自定义比例预设持久化（2026-09）

> 背景：§37 的 Custom 弹窗只能手输宽高比、不改不存。本次把它改造为**管理窗**：左侧列出全局保存的命名比例定义（单击选中回填字段/双击直接套用），右侧名称+宽/高+Save/Delete，定义持久化到 `user/sfnodes/` 跨工作流共享。

### 1. 后端（零新增依赖）

- 新建 `sf_utils/crop_expand_presets.py`，照 `text_presets.py` 范式：存储 `user/sfnodes/crop_expand_presets.json` 结构 `{"presets": [{"name", "w", "h"}]}`（数组保序）；路由 `GET/POST/DELETE /api/sfnodes/crop_expand_presets`（POST 按名 upsert 不追加），`import` 时注册；复用 `disk_state.atomic_write_json/mtime_size_sig/sf_user_dir` + `common.valid_name(max_len=200)`；`asyncio.Lock` 互斥读改写 + mtime+size 缓存。
- 校验：`w/h` 必须为正有限数（**bool 显式排除**——`isinstance(True, int)` 为真会混入）且 ≤ 10000；`_normalize_presets` 接受 dict/裸列表、重名保留首个。
- 路由触发注册：`nodes/image/crop_expand.py` 顶部 `from ...sf_utils import crop_expand_presets`（text_preset.py 同款 import 副作用注册）。
- 为何不复用 text_presets：项目按数据域各持一模块（lora/krea2/text_presets），仅下沉低层；跨域复用会污染命名空间且 `{name,text}` 与 `{name,w,h}` 语义不同。

### 2. 前端

- `sf_crop_expand_lib.js` 新增纯函数 `normalizeRatioPresets(raw)`（接受裸数组/`{presets}`；name 非空、w/h 正有限数≤10000、重名保序保首个）——与后端同口径前端兜底，可 `.mjs` 直测。
- `sf_crop_expand.js`：新增 API 束 `fetchRatioPresets/apiSaveRatioPreset/apiDeleteRatioPreset`（走 `sfApiUrl`，后端不可用返回 null → 列表显示 "Preset store unavailable"，**手输照常可用**）；`openCustomRatioDialog` 改造为左右两栏管理窗，保留原 id/Esc·滚轮三关闭（`sf_popup.attachPopupDismiss`）/Enter 应用·放行 ctrl·meta·alt 组合键；单击列表项=选中回填、双击=套用；Save 后 `reload(name)` 重新拉取（后端为真源，不在前端维护本地镜像）；错误反馈统一走 `sfToast`。
- 状态契约不变：套用仍写 `properties.sfCropExpandState`（`custom_w/custom_h/aspect_ratio:"custom"` + `applyRatioToRect`），随工作流保存；全局库只读用于选择，不随工作流走。

### 3. 测试

- `tests/test_crop_expand_presets.py`（mock aiohttp/server/folder_paths，仿 test_text_presets.py）：路由注册、空库、POST 全量校验（空名/路径分隔符/控制字符/超长名/非数/0/负/超上限/bool/缺字段）、落盘结构、同名覆盖不追加、DELETE 校验/删除、mtime 自动重载、非法 JSON 容错、归一化、`crop_expand.py` 导入行文本断言。
- `tests/test_crop_expand_js.mjs` 追加 `normalizeRatioPresets` 解包/过滤/去重/非法用例。

## 62. SFPauseImage 水平翻转开关（复用闸门引擎 opt-in 扩展，2026-09）

> 背景：Pause Image 复查阶段常需要把图左右镜像后再喂给昂贵下游。翻转必须满足"预览所见 = 下游所得 = 保存所得"，且不能为了翻转把上游重跑一遍。

### 1. 语义与机制

- state 加 `flip` 布尔（随 `node.properties` 持久化）；它随既有 `PauseState` JSON 注入后端，是唯一需要后端落地的状态。
- **已有快照**：点 Flip → 前端 `POST /api/sfnodes/preview/flip`，后端 `ImageOps.mirror` 就地镜像 temp PNG → 之后预览（`/view` 带缓存戳重载）、Continue 读回、Copy/Save/Open 全部天然一致，前端零特判。路由失败则回滚开关（文件与状态保持同步）。
- **无快照**：只改 state；下次 Pause/Pass 运行时后端按 `flip` 先 `torch.flip(image, dims=[2])`（`[B,H,W,C]` 的 W 维）再存快照与输出。
- **continue 分支读的文件已是镜像产物**——不能再翻转，否则翻回去。这是"就地镜像文件"方案最容易被忽略的一环。
- 按钮放第一行 `[⟳ Regenerate] [⇋ Flip] [▶ Continue]`（第二行 4 个工具按钮已挤，5 个会省略号截断）；`.on` 高亮表示开启，任何模式（含 Pass）都可切。

### 2. 复用与边界

- 引擎 opt-in：`definePauseGate`/`buildPauseBody` 新增 `flip`/`flipTitle` 配置，只有 `sf_pause_image.js` 传 `flip:true`，mask/latent 完全不变（`makeGateState` 仅做 `flip` 布尔规范化，对三闸门都安全）。
- 注入链：`sf_pause_text_lib.applyGateMode` 新增可选 `opts.extraState`，三个模式分支都 `{...extraState}` 合并。**关键坑**：`graphToPrompt` hook 的注入会在 `api.queuePrompt` 提交时被 `applyGateMode` 覆盖，所以 flip 必须同时经 `collectGates().extraState` 传给 applyGateMode，两处共用同一份，保持 INJECT/PRUNE 不分歧。
- `_mirror_png` 抽成 `preview_routes.py` 模块级纯副作用函数（可测），路由只做文件名安全校验（temp 目录 + basename + `sf_pause` 前缀）。
- latent 未纳入：快照是 safetensors（非图像），且 latent 空间翻转 ≠ 图像空间翻转。

### 3. 测试

- `tests/test_pause_image.py`：mock `torch.flip`；flip pause/pass 输出镜像、快照字节镜像（红像素跑到最右）、continue 读回镜像、flip=false 不翻转；`_mirror_png` 原地镜像 + 两次还原。
- `tests/test_pause_image_js.js` / `_smoke.js`：flip 默认值与布尔规范化、`extraState` 在 pause/pass/continue 三模式注入、Flip 按钮构建、剪枝后仍保留 flip 注入。

## 63. SFPauseImage 外载图（Load/Browse/拖放/粘贴，Design D，2026-09）

> 背景：用户正常用接线图，但希望"不断开接线的前提下，加载一张图后点 Continue 让这张显示的图作为输出"。据此定为 Design D：外载图只替换 Continue 提交的 temp 快照，Pause/Pass 仍走接线。

### 1. 语义（Design D）

- **Pause/Pass（Run）**：优先接线图；未接线时才回退读 temp 快照（`pause_image.run` 里 `image is None` 分支，读回后 `flip=False`——快照已是最终产物，重复翻转会翻回去）。
- **Load / Browse / 拖放到预览 / Ctrl+V**：`CropAPI.uploadSrc("pause_"+ts, dataURL)` 落盘 `input/sfnodes_crop/` → `POST /api/sfnodes/pause/load {unique_id, src_path, flip}` 物化写入该节点 temp 快照 → 预览显示。**不动接线**。
- **Continue**：读 temp 快照 → 输出显示的外载图；接线保持连接。
- **Clear**：清标记 + `state.frame=null` + 预览复位 + Continue 禁用。
- 标记 `node._sfPauseSrcPath` **仅内存、不自动恢复**（temp 本就随重启清空，避免与"最后一次 Run 结果"语义冲突——用户拍板）。

### 2. 复用与边界

- 独立模块 `web/sf_pause_source.js`：把 `image_browser`/`CropAPI` 重依赖隔离，**不进 `sf_pause_kit.js`**（kit 是 image/mask/latent 共享引擎，且其 `_js`/smoke 测试只桩 `sf_common`，引重依赖会连带改一堆测试）。复用 `installPasteHandler`/`parseAnnotatedImageValue`/`buildSourceURL`/`showImageBrowser`/`CropAPI.uploadSrc`，零新增落盘路由。
- kit 只加两个无依赖接入点：`cfg.extraHeight`（把外插行高度算进 `NODE_MIN_H`）+ `els.preview`（供 `preview.before(row)` 插行与挂原生 `dragover/drop`）。
- `sf_pause_source` 注册**第二个** extension，在 kit 之后链式包装 `onNodeCreated/onConfigure`（source 的 wrapper 在 kit 之后跑，故 `els` 已就绪；`origCreated?.apply` 先跑 kit 的 setup）。
- 粘贴传 `installPauseHandler({disconnectInput:false})`：落实"加载不断线"。为此给 `sf_common.installPasteHandler` 加可选 `disconnectInput=true`（默认值保持既有四类调用行为不变，仅断开块加条件）。
- 后端 `preview_routes.materialize_pause_snapshot` 抽出可测（读 input → 按 flip 镜像 → 写 `pause_image._snapshot_path`），`_resolve_input_path` 拒绝绝对路径/`..`/symlink 逃逸（realpath+commonpath）。
- 拖放挂 DOM 预览区（`sf_inpaint.js` 先例）——本节点预览是 DOM `<img>`，无需 canvas 的 `onDragOver/onDragDrop`。
- mask/latent 未纳入：kit 改动向后兼容，两闸门不传 `extraHeight`/不接 source 模块，行为零变化。

### 3. 测试

- `tests/test_pause_image.py`：mock `folder_paths.get_input_directory`；`materialize_pause_snapshot` 未翻转/翻转/越界拒绝；pause 无接线回退外载图、已镜像快照不重复翻转。
- `tests/test_pause_source_smoke.js`（新增，stub image_browser/crop_core/sf_common）：按钮行构建并插预览前、拖放/粘贴接线（`disconnectInput:false`）、`loadSource` 调 uploadSrc + load 路由并回填 frame/预览、Clear 复位。
- `tests/test_common_paste_js.js`：`disconnectInput:false` 不断线、默认 true 仍断线。
- `tests/test_pause_image_smoke.js`：把新增的 `attachSourceControls` import 桩掉（保持引擎冒烟，外载图由专门 smoke 覆盖）。
- `tests/check_web_imports.py` MODS 增 `sf_pause_source`。

## 64. SFInvertTrackData：SAM3 追踪数据反转（跨核心封闭类型，2026-09）

> 背景：SCAIL-2 工作流里 `SAM3_VideoTrack` prompt 写 `person` 得到人物追踪；需求是"追踪 person 以外区域"，且结果要接 `SCAIL2ColoredMask.driving_track_data`。

### 1. 为什么必须在 track_data 层反转

- `SAM3_TRACK_DATA` 是核心 `comfy_extras/nodes_sam3.py` 定义的封闭类型（`io.Custom("SAM3_TRACK_DATA")`，实为 `{packed_masks, n_frames, scores, orig_size}`）。
- `SCAIL2ColoredMask.driving_track_data` **只接受 track_data**（只有 `ref_track_data` 的 MultiType 兼容普通 MASK）。所以"先 `SAM3_TrackToMask` 再 `InvertMask`"的常规做法喂不进 driving，**必须在 track_data（位打包）层取反**。
- 位打包：`comfy.ldm.sam3.tracker.pack_masks/unpack_masks`（`[T,N,H,W//8] uint8` ↔ bool），`packed[:, indices]` 选中对象 → `unpack_masks().any(dim=1)` 取并集 → `~` 取反 → `pack_masks().unsqueeze(1)` 还原单身份。

### 2. 语义与边界（节点 `nodes/image/invert_track.py`）

- 反转粒度 = **选中对象并集的反集**（多对象各自反集会重叠，语义混乱）。
- **逐帧空遮罩 → 全选**：`any` 全 False 的帧取反即整帧身份；为一致性，`packed_masks=None`（整条无对象）也按 `orig_size` 补全帧（W 补到 8 的倍数再打包），而非返回空——否则同一节点在不同帧上语义分叉。
- 显式给了 `object_indices` 但全部非法（越界/非数字）→ 返回空身份（尊重用户的显式选择，不偷偷变全屏）。
- `dict(track_data)` 浅拷贝输出，保留 `orig_size/n_frames`，`scores` 重置为 `[1.0]`，不改写输入。
- lazy import `comfy.ldm.sam3.tracker`，核心缺席时节点仍可加载。

### 3. 测试

- `tests/test_invert_track.py`：numpy 严格复刻位序的 pack/unpack 桩 + `FakeArr`（支持 `any(dim=)`/`~`/索引/`unsqueeze`）注入 `invert_track_data`；覆盖单对象/多对象并集/子集/空帧全选/非法索引空/packed None 全帧/键保留 + execute 集成（stub `torch` 与 `comfy.ldm.sam3.tracker`）。
- 用 numpy 而非 mock 位运算，保证位序与核心一致，能抓打包错误。

## 65. SFMaskToTrackData：MASK→SAM3_TRACK_DATA 桥接（第三方追踪器接入 SCAIL2ColoredMask，2026-09）

> 背景：SCAIL-2 工作流原用核心 `SAM3_VideoTrack`（**文本 prompt**）追踪，需求改为 `Comfyui-SecNodes` 的 `SeCVideoSegmentation`（**视觉 prompt**：points/bbox/mask）提取遮罩。两者输出类型不兼容。

### 1. 类型断点与桥接决策

- `SCAIL2ColoredMask.ref_track_data` 声明为 `MultiType[SAM3_TRACK_DATA, MASK]` → **参考路可直接接 SeC 的 `MASK`，零桥接**。
- `SCAIL2ColoredMask.driving_track_data` **只收 `SAM3_TRACK_DATA`** → 驱动 pose 遮罩路必须把逐帧 MASK 打包成 track_data，否则连不上。故新增 `SFMaskToTrackData`。
- 不采用"改核心 `nodes_scail.py` 放宽 driving 类型"（跨仓库、运行实例在 docker 镜像内改源码副本无效）；节点级桥接可复用、可测、可版本化。

### 2. 实现（节点 `nodes/image/mask_to_track_data.py`）

- 纯函数 `mask_to_track_data(masks, pack_masks, torch)`：`[H,W]`/`[T,H,W]` → `[T,1,H,W]`；**单对象约束**（对象维必须为 1，多对象抛错）；`W` 非 8 倍数时 `torch.nn.functional.pad` 补零；`pack_masks` 位打包 → `{packed_masks:[T,1,H,W//8], orig_size:(H,W), n_frames:T, scores:[1.0]}`。空帧返回 `packed_masks=None`。
- **位打包直接复用核心 `comfy.ldm.sam3.tracker.pack_masks`**（lazy import，与 §64 同源），不内联副本；`pad` 后宽度=packed 宽度×8，`_render_colored_masks` 再插值回 `orig_size`，所以非 8 倍数宽度不影响渲染。
- 与 §64 同构：纯函数签名注入 `pack_masks/torch` 便于离线测试。

### 3. 工作流接线（SeC 替代 SAM3 追踪）

- 驱动：`VHS_LoadVideo.IMAGE → SeCVideoSegmentation.frames` + `PointsEditor.positive_coords → positive_points`；`SeCVideoSegmentation.masks → SFMaskToTrackData → SCAIL2ColoredMask.driving_track_data`。
- 参考：`ImageScale.IMAGE → SeCVideoSegmentation.frames`；`masks → SCAIL2ColoredMask.ref_track_data`（走 MASK 分支）+ `ImageCompositeMasked.mask` / `MaskToImage.mask`（替代原 `SAM3_TrackToMask`）。
- 两路共用 1 个 `SeCModelLoader`；SeC 默认 `auto_unload_model=True`，第二次分割会卸载后自动重载（省显存、多一次加载耗时）。
- 移除字幕去除整条（含其哑节点）后，驱动的 `Any Switch`/`ComfySwitchNode` 成为纯透传，一并删除，`VHS_LoadVideo.IMAGE` 直连 `SeC.frames` 与 `WanSCAILToVideo.pose_video`。

### 4. 测试

- `tests/test_mask_to_track_data.py`：numpy 严格位序 pack 桩 + `FakeArr`（`dim/unsqueeze`）+ `torch.nn.functional.pad` 桩；覆盖 3D/2D、非 8 倍数补零、多对象报错、空帧/None、execute 集成、双字典键一致。

## 66. SFMaskCache：遮罩磁盘缓存 + lazy 跳过上游（2026-09）
> ⚠ 已被 §111 取代：`source`/`signature` 已删除，缓存键收口为「名字 + required 非空 `source_key`」。本节保留为历史设计记录。


> 背景：SeC 视频分割要加载 4B 模型，同样的视频/图片重跑一次很慢。需求是把驱动/参考的提取遮罩按源持久化，下次直接复用。

### 1. 用 lazy 输入做"真跳过"而非只缓存结果

- ComfyUI 的 `comfy_execution/graph.py::add_node` 对标记 `{"lazy": True}` 的输入**不建依赖**（`include_lazy=False`），节点可在 lazy 上游未算时先调度；节点实现 `check_lazy_status` 返回需要求值的输入键，返回 `[]` 即放行且 lazy 输入为 `None`。
- 因此 `SFMaskCache` 在缓存命中时 `check_lazy_status` 返回 `[]` → SeC 节点整条分支**根本不执行**（省掉模型加载+推理），而不是算了再丢弃。这是比"保存/加载两节点 + 手动 mute"优越之处。仓库已有先例 `nodes/logic.py::AnythingIndexSwitch`（`lazy_options = {"lazy": True}`）。
- 未命中/`force=True` 返回 `["masks"]`，节点重入时拿到上游遮罩并落盘。

### 2. 缓存键与失效

- 键 = `name` + `signature`（可选，接 `PointsEditor.positive_coords`）+ `source` 首末帧 16×16 下采样哈希（可选，接源视频/参考图 IMAGE）。**任一不匹配即失效**；`force`（非 lazy，`check_lazy_status` 能读到）强制重算。
- `source_signature` 只哈希首末帧 + 形状，避免整批几十帧哈希开销；非 4D/None 返回 `""`。
- 语义是"完全相同才命中"：签名/源未接线时按空串参与比较（保存时接了、后来断了也会失配重算），不做"缺失即忽略"的宽松匹配。

### 3. 存储与预览（`nodes/mask/mask_cache.py`）

- 目录 `user/sfnodes/mask_cache/`（`disk_state.sf_user_dir()`），三件同名前缀：`<name>.safetensors`（uint8 `[T,H,W]`，0-255 量化保留软遮罩）+ `<name>.json`（元数据，`atomic_write_json` 原子写）+ `<name>.png`（最多 4 帧横排预览拼图，便于文件管理器查看）。
- 名称清洗复用 `disk_state.sanitize_filename`；`cache_paths` 非法名返回 None 并抛错。
- 路由 `GET /api/sfnodes/mask_cache/list`（导入时注册、try/except 包裹），前端 `web/sf_mask_cache.js` 在 `onNodeCreated`/`onAfterGraphConfigured` 重建 `name` 下拉并加「↻ 刷新缓存列表」按钮；下拉末位「＋ 新建缓存…」用 `window.prompt` 输入新名（combo 不能自由输入）；`VALIDATE_INPUTS` 返回 True 接管动态选项校验。

### 4. 接线

- 驱动：`SeC.masks → SFMaskCache.lazy masks → SFMaskToTrackData → SCAIL driving`；参考：`SeC.masks → SFMaskCache → SCAIL ref / 合成 / 转图 / 预览桥接`。`source` 接源图、`signature` 接点选坐标。
- `execute` 的 `masks is None` 即命中/纯读取路径（从盘读回）；`masks` 非空即保存路径——上游若因其他消费者仍被计算，会用新结果覆盖缓存（符合预期）。

### 5. 测试

- `tests/test_mask_cache.py`：内存 `safetensors/torch` 桩 + tempdir（monkeypatch `mod.cache_dir`）；覆盖结构/注册、clean_name/quantize/source_signature、save→load 往返、cache_hit 三态、list/read_meta、`check_lazy_status` 各分支、execute 读/写/缺失报错。相对导入经 `sys.modules` 注册 `sfnodes` 包占位（test_save_image_exact.py 同款）。

## 67. SFTrackDataCache：SAM3_TRACK_DATA 层缓存 + 缓存逻辑单源抽取（2026-09）
> ⚠ 已被 §111 取代：`source`/`signature` 已删除，缓存键收口为「名字 + required 非空 `source_key`」。本节保留为历史设计记录。


> 背景：§66 的 SFMaskCache 只在 MASK 层缓存，若上游是 SAM3.1 追踪（`SAM3_TRACK_DATA`），走 `SAM3_TrackToMask → 缓存 → SFMaskToTrackData` 会把**多对象并集塌缩成单身份、丢失 scores**。需要 track_data 层缓存。

### 1. 与 §66 同构 + track_data 专属存取

- lazy `track_data`(SAM3_TRACK_DATA) 输入，命中 `check_lazy_status` 返回 `[]` → `SAM3_VideoTrack` 整条不执行；`execute` 的 `track_data is None` 即读盘路径。
- 落盘保留核心 dict 原貌：`packed_masks`（uint8 `[T,N,H//... W//8]` 位打包**原样存**，不 unpack）+ `scores`（**float64**，避免 float32 往返精度误差）+ json 里的 `n_frames/orig_size/num_objects`；`packed_masks is None`（无对象）时写 `_empty` 占位满足 `save_file`，读回 `packed_masks=None`。
- 重建 dict 必须含 `{packed_masks, n_frames, scores, orig_size}`（`SAM3_TrackToMask`/`SAM3_TrackPreview`/`SCAIL2ColoredMask`/`SFInvertTrackData` 消费的键全在内）；`packed_masks` 必须是 torch 张量（下游 `.to(device)`）。
- 预览：`user/sfnodes/track_cache/<name>.png`，lazy import 核心 `unpack_masks` 解码后**按对象上色**（本地小调色板常量，非核心 `COLORS` 的 import 副本），最多 4 帧横排；失败忽略。

### 2. 缓存通用逻辑单源抽取（sf_utils/cache_store.py）

- §66 把"名字清洗/三件路径/源图指纹/元数据读写/命中判定/列表/列表路由"都放在 mask_cache 内；新增 track 缓存会逐字复制 → 抽到 `sf_utils/cache_store.py`，**按显式 `base_dir` 参数化**（`sf_cache_dir(subdir)` 建目录，其余函数收 `base_dir`）。
- mask_cache 保留同名薄包装（`cache_dir()` 等）委托 cache_store：调用点与 `test_mask_cache.py` 的 `monkeypatch(mod.cache_dir)` 零改动——包装内仍调模块级 `cache_dir()`，注入的 tempdir 生效（若直接调用 cache_store 的绝对路径则会绕过 patch，测试隔离失效）。
- `register_list_route(route_path, subdir)` 收拢两处同构路由（惰性 import server/aiohttp，无 ComfyUI 静默）。

### 3. 前端 lib 抽取（web/sf_cache_name_lib.js）

- 两节点共享 `name` 下拉重建/「＋ 新建」prompt/「↻ 刷新」按钮逻辑；抽成**无 `app` 依赖**的纯模块，`sf_mask_cache.js`（重构）与新增 `sf_track_cache.js` 各自 `registerExtension` 注入 `{api}`。
- 纯逻辑 `buildNameOptions(names,current,newLabel)`（保序去重 + 当前值插入队首 + 新建入口）可 `.mjs` 直测（`tests/test_cache_name_lib.mjs`，复刻 `sf_boolean_switch_lib` 的拷 .mjs import 法）；combo 不能自由输入故用「＋ 新建…」魔法项 + `window.prompt`。
- MODS 增 `sf_cache_name_lib`/`sf_track_cache`（registerExtension 文件必须直接 import `/scripts/app.js`；lib 无 import 依赖）。

### 4. 测试

- `tests/test_track_data_cache.py`：内存 safetensors/torch 桩 + tempdir（monkeypatch `mod.cache_dir`）；覆盖结构/注册、多对象 packed+scores 往返、packed=None、cache_hit 三态、check_lazy 各分支、execute 读/写/缺失/非法名；相对导入经 `sys.modules` 注册 `sfnodes` 包占位。
- `tests/test_mask_cache.py` 回归通过（验证抽取不破坏既有行为）。

## 69. 画布节点拖拽释放兜底 + 按键守卫：SFImageCropExpand/SFImageBrushMask 截选框/笔触粘鼠标（2026-09）

> 背景：SFImageCropExpand 用户反馈"裁剪超出图片大小后，选取框会随鼠标移动而改变，而不是按住拖动才改"。SFImageBrushMask 是同一复刻范式的内联副本，同病。

### 1. 根因（前端包源码实锤，非猜测）

前端 1.x `LGraphCanvas.processMouseUp` 三处导致节点自持拖拽状态收不到释放：
- 纯点击（未超拖拽阈值）走 `pointer.up(e)` 提前 `return`，**不调 `node.onMouseUp`**；
- 非 click 路径只对 `this.node_over?.onMouseUp?.(...)` 回调——**松开时鼠标不在节点上**（出界拖拽的典型：拉框扩图把手柄拉出节点外再松开）不回调；
- 结尾**无条件 `e.stopPropagation(); e.preventDefault();`**——本包 bubble 阶段的 `document.addEventListener("mouseup")` 兜底**永不触发**。

三者叠加 → `node._sfExpandDrag`（/`_sfBrushDrawing`）残留；`onMouseMove` 只要状态为真就继续 `updateCropByDrag`（/续笔），鼠标再移回节点上方即"无人按键也跟着动"。这正是上游 YCNodes 原版 README 记录的"双击停止"怪癖（本包继承了 `onDblClick=finalize`）。`e.buttons` 在 `onMouseMove` 里毫无校验是直接漏洞。

### 2. 修复（两道防线，公共实现进 sf_common）

- `web/sf_common.js` 新增 `primaryButtonReleased(e)`：原生 `e.buttons` 主键（bit0）已松返回 true，缺 `buttons` 返回 false（兼容旧调用/测试桩，保持"不主动结束"旧行为）。
- 新增 `installNodeReleaseGuard(node, onRelease, {hook})` / `removeNodeReleaseGuard(node, {hook})`：`window.addEventListener(["mouseup","pointerup","pointercancel","blur"], h, true)`，幂等、ref 存 `node[hook]`，`onRemoved` 解绑。**capture 阶段先于目标 `stopPropagation` 执行，释放必达**（家规同 `sf_dropdown_settings.js` / `sf_lora_stack_settings.js` / canvas `CanvasPointer.move` 的 `if(!e.buttons) reset()`）。
- 两节点：`onMouseMove` 开头 `if (状态 && primaryButtonReleased(e)) { finalize; return true; }`（兜底彻底丢失的释放：窗口外/第二显示器，下次移回节点即落定）；bubble `document` 兜底 → `installNodeReleaseGuard`；`onMouseDown` 加 `e.button !== 0` 早退（右键/中键不起拖；BrushMask 原已有）。
- `onDblClick` 保留（原版习惯、无害）。

### 3. 测试与复用

- `tests/test_common_release_guard.js`：剥 `sf_common` import 直跑（同 `test_common_paste_js.js`）——`primaryButtonReleased` 语义、四事件 capture 注册、幂等安装、`remove` 解绑清 hook、回调抛错吞掉。
- `tests/test_crop_expand_smoke.js`：桩 app/CropAPI/sf_popup/image_browser，sf_common 用真实剥 import 版，纯库拷真实——断言左键命中手柄起拖、按住改框、`buttons:0` 清状态且不再改框、落定后移动不改、右键不起拖、`onMouseUp` 与 window capture 释放、`onRemoved` 解绑。
- `tests/test_brush_mask_smoke.js` 的 `stub_common` 同步补三导出（桩模块必须与真实导出面一致，否则 ESM 命名导入直接 SyntaxError——本轮踩到）。
- 不新增公共纯库函数（判定逻辑放 sf_common，DOM 事件面本就不属纯库）。

## 71. 官方原生 LoadImage/LoadImageMask 挂 Browse 按钮（复用 showImageBrowser 选择器模式，2026-09）

> 背景：希望官方原生 Load Image 也拥有 SF Load Image Browser 的"浏览图片"能力，纯前端、不改 Python。

### 1. 复用与实现

- 直接复用 `web/image_browser.js::showImageBrowser(node, opts)` 的 **onPick 选择器模式**（§34 参数化的产物）：宿主传 `onPick` + `selectedValue`，弹层不碰 widget、选中值交回调。
- 新增 `applyNativeLoadImagePick(node, value)`（导出，纯函数好测）：定位 `widgets` 里 `name==="image"` 的 widget → 赋 `value` → 调其 `callback(value)`（核心 `image_upload` 借此刷新预览）→ `setDirtyCanvas`。无 widget/null 值返回 false。
- 独立扩展 `sfnodes.native_load_image_browse`，`nodeCreated` 精确匹配 `comfyClass ∈ {LoadImage, LoadImageMask}`（**不误伤 SFLoadImageBrowser**——它已有自挂按钮），`addWidget("button", "Browse Images", ...)`。
- 显隐受设置 `sfnodes.LoadImage.BrowseButton.Enabled`（boolean，默认开）门控：`nodeCreated` 时读设置决定是否挂载；`onChange` 遍历 `app.graph._nodes` 对现存节点即时增删（`widgets.splice`）——与官方 info 开关同款，**onChange 时 store 尚未更新，须 `setTimeout(...,0)` 延后一 tick 再读值**。`findNativeBrowseButton(widgets)` 纯函数（返回下标/-1）供增删去重与测试复用。

### 2. 关键发现：combo 列表校验被 VALIDATE_INPUTS 跳过

- 弹层可切到 output 目录，选中值形如 `"a.png [output]"`，而原生 `image` combo 的静态 options 只有 input 文件。原以为会踩 `Value not in list`（patterns §2），实测**不会**：
- `execution.py::validate_prompt` 里 combo 成员检查被 `if x not in validate_function_inputs and not validate_has_kwargs:` 守卫包裹——只要节点 `VALIDATE_INPUTS` 的形参含该输入名即整段跳过。原生 `LoadImage.VALIDATE_INPUTS(s, image)` 正好含 `image`（LoadImageMask 继承同款），`get_annotated_filepath` 原生解析 output 路径。
- 结论：**无需**像 SFLoadImageResize 那样把 output 值塞进 `imageWidget.options.values`（那会污染原生可见下拉）。判断依据是"节点是否声明了 VALIDATE_INPUTS 且形参覆盖该 combo"。

### 3. 测试

- `tests/test_image_browser_js.js` 文本提取 `applyNativeLoadImagePick`/`findNativeBrowseButton` 进 `.mjs` 直跑（依赖的模块级常量按源文件正则取值注入，守单真源）：写入值/触发 callback/setDirtyCanvas、无 callback 不抛错、无 widget/空值/node 空返回 false；按钮下标命中/无返回 -1/同名非 button 不算。
- 实机验证走分段 console 诊断（platform §2.9），重点看：按钮存在、pick input/output 后 widget 值与预览刷新、队列不报 `Value not in list`。

## 75. SFImageResizePlus size_mode 复刻原生 ResizeImageMaskNode（2026-09）

> 背景：原生 `ResizeImageMaskNode.resize_type` 有 9 种模式，而 `SFImageResizePlus` 原 `size_mode` 只有 `width & height` / `total pixels`。目标是把"缺少的、有意义的"模式补进 `size_mode`。

### 1. 覆盖判定（哪些该复刻）

- 已覆盖：`scale dimensions`（= width & height + method 的 stretch/keep/fill crop/pad）、`scale total pixels`。
- **冗余不复刻**：`scale width` / `scale height` —— `width & height` + `method="keep proportion"` 把另一维填 0 即等价（实测 2000×1000 缩宽 1024 → 1024×512，与原生一致）。单边 auto 逻辑见 `execute` 里 `method in (keep proportion, pad)` 分支。
- 新增 4 个：`scale by multiplier` / `longer dimension` / `shorter dimension` / `scale to multiple`。
- `match size`（按参考图尺寸）需额外 optional 参考输入，本次不做（可后续单独加）。

### 2. 数学单源收敛到 resize_engine.py

- 新增纯函数 `multiplier_to_wh` / `longer_dimension_to_wh` / `shorter_dimension_to_wh` / `multiple_to_wh`（与既有 `total_pixels_to_wh` 同文件同风格，非法输入返回 None，1px 下限）。
- **不要在 scale.py 内联**：`SFImageResize` 的 `resize_engine` 已有 longest_side/scale_factor 的 PIL 版实现，目标尺寸数学应收敛为纯函数单源，可直测。
- `scale to multiple` 语义 = 先 `floor` 到倍数得目标，再 **cover 缩放 + 居中裁剪**（原生 `scale_to_multiple_cover`）；与既有 `divisible_by`（只裁不 cover）不同。

### 3. execute 分支与 method 特例

- 新分支只改"目标尺寸计算"段（在 `divisible_by` 取整之前），算出的 width/height 仍走既有 `divisible_by` → `method` → `condition` 通路，mask/pad/interpolation 全复用。
- 倍率/长边/短边：默认 `method="keep proportion"` 即得原生精确结果；用户改 method 则是合理超集。
- **`scale to multiple` 内部强制 `method="fill / crop"`**（忽略 method widget，前端同步隐藏）——因为原生该模式恒为 cover+裁剪，与 method 无关。退化（multiple<=1 / 取整为 0）时 `width=height=0` 走自动档直通，**不能**回落到隐藏 widget 里残留的旧 width/height。
- **`scale to multiple` 必须让 `divisible_by` 退场**：`divisible_by` 是全局后置取整，会在 multiple 目标上再 `floor`，若 `multiple % divisible_by != 0` 会破坏倍数网格（如 multiple=6、divisible_by=8 会把 996 压成 992）。原生 `ResizeImageMaskNode` 无 `divisible_by`，故本模式**后端 `divisible_by=1` 跳过、前端隐藏该 widget**；其余模式 `divisible_by` 照常生效。

### 4. widgets_values 位置敏感与新控件兼容（核心坑）

- 新增 4 个 widget（multiplier/longer_size/shorter_size/multiple）紧跟 total_pixels，canonical 长度 10 → **14**。旧工作流按位恢复会错位，`web/sf_image_resize_plus.js::configure` 必须**逐级 remap**：先 8→10（size_mode 置顶重排前），再 10→14（在 `total_pixels` 后插入 4 个默认值 `1.0/512/512/8`）。判据用 `widgets_values.length`，新版 14 项原样放行。
- 前端显隐按 `size_mode` 分支；`scale to multiple` 额外 `methodW.hidden=true`，切回其它模式由统一 `toggle()` 恢复。

### 5. 测试

- `tests/test_image_resize_plus.py`：4 个纯函数（含倍数已整除/取整为 0/竖图）与 `execute` 各新模式出图尺寸、multiple 退化直通、六选项/顺序断言。
- `tests/test_image_resize_plus_js.js`：各模式参数显隐、scale to multiple 隐藏 method、8→14 与 10→14 remap、14 项不改写。

---

## 76. SFReferenceRegionNeutralize：Krea2 参考区域中和（保留姿势/背景，只换脸，2026-09）

> 背景：Krea2 洗图时用深度图作参考复刻姿势，配角色 LoRA 换脸。用户反馈只能靠 `SFKrea2EditApply.ref_strength` 调节——调高姿势稳，但脸也贴近原人物、角色 LoRA 被压制。

### 1. 根因：ref_strength 是全局时间旋钮，无法空间解耦

- `ref_strength` 控制的是**整块参考 token 参与采样的进度比例**（进度 ≥ 它即丢弃缓存的 ref K/V）。参考 token 同时编码结构与外观/身份，二者在同一 attention 里、没有空间区分 → 锁姿势必然连带锁脸。
- 提高角色 LoRA 权重是**全局对抗**，会同时改身体/风格，无法只在脸上生效。
- 深度参考下身份泄漏来自**深度保留了原人物头部/面部几何**（骨相/脸型），强 ref 时模型忠实还原该几何，LoRA 只能在纹理/五官上争。

### 2. 为什么不做 attention 级"参考区域遮罩"

- Krea2 是 single-stream 全注意力（heads=48，序列 3~6k）。query 级稠密偏置 `(b·h, Lq, Lk)` 到 **GB 级显存**，不可行。
- 退化成"仅 key 的布尔 mask"（屏蔽参考图脸区 token）内存可控，但：运行时启用 flash-attn 时 `attention_flash` **遇 mask 直接抛错并回退 SDPA**（整模型变慢）；sage 是否支持取决编译（`SAGE_ATTENTION_SUPPORTS_MASK`）；还要与文本 padding 的 `attention_mask` 合并。风险/收益不划算。
- 对"参考=深度图"的场景，**在图像/latent 域中和脸区**更安全且等价：`SFKrea2EditTextEncode` 的**初始 latent 就是主图 ref latent**，图像域改完后 ref latent 与初始 latent 同步失去面部几何。

### 3. 节点与纯逻辑

- `SFReferenceRegionNeutralize`（`nodes/image/region_neutralize.py`，CATEGORY `sfnodes/image`）：`image` + `mask` + `mode`(blur/mean/fill) + `strength` + `blur_radius` + `feather` + `fill_value` → 中和后的 `image`。
- 纯逻辑 `sf_utils/image_region.py::neutralize_region`（numpy + PIL，顶层无 torch 依赖，可直测）：`blur` 用高斯模糊保留大致深度/头姿（推荐）、`mean` 区域均值、`fill` 常量；`strength` 按羽化 mask 混合，区域外严格不变；支持 3D/4D 与 mask 批量广播（批量小于图像复用末帧），mask 尺寸不符在节点层 bilinear 缩放。模糊惰性复用 `sf_utils/inpaint_helpers.gaussian_blur_np`（单源）。

### 4. 推荐连线配方（不改工作流文件，仅说明）

```
SFImageCropExpand.image → DepthAnythingPreprocessor → SFReferenceRegionNeutralize.image → SFKrea2ConfigPreparer.image
原图(RGB) → GeneratePreciseFaceMask → SFReferenceRegionNeutralize.mask
SFKrea2EditApply.ref_strength 降到 ~0.3–0.5
（可选）SFRegionalLoRA：脸框 + 角色 LoRA，strength 1.2–1.5、羽化 0.08，串在 LoraLoader 之后
```
协同后：深度参考负责身体/背景/服装，区域 LoRA 负责脸，`ref_strength` 只再微调整体结构强度。`blur + strength 0.6~0.8` 起步，不够再试 `mean`。

### 5. 测试

- `tests/test_image_region.py`：strength=0 原样、blur/mean/fill 区域内外行为、羽化过渡、批量与 mask 广播、mask 尺寸不符抛错、3D 输入 3D 输出；节点 CATEGORY/RETURN_NAMES/INPUT_TYPES 与 execute（尺寸一致路径 + 尺寸不符走缩放分支）。mock：fake torch（模糊走 PIL）。

## 80. 缓存节点名接入文本：name_text 覆盖输入 + 下拉置灰（2026-09）

> 需求：SF Track Data Cache / SF Mask Cache 的缓存名 `name` 是磁盘列表重建的 combo（「＋ 新建…」走 window.prompt），希望把缓存名接到文本来源（文件名 / SFParsePath / 工作流名等），不再手选。

### 1. 方案：可选 STRING 覆盖输入，而非改造 combo

- `name` 是 required combo（数组型输入）：新前端虽让 widget 与 socket 共存，但数组型 widget 槽的 socket 类型是 `COMBO`，STRING 直连不可靠，且本节点前端会动态重建 options——不要把赌注押在"连到下拉"。
- 正确做法：可选 `name_text`（`"STRING"` + `forceInput: True` + `default: ""`），后端解析单源 `_resolve_name(name, name_text)`：**文本 strip 非空优先，空回退下拉**；`check_lazy_status` 与 `execute` **共用同一函数**（懒命中判断与写盘名字必须同源，否则命中/落盘分叉）。
- `forceInput` 无 widget → 不进 `widgets_values`，旧工作流按位兼容；文本照旧走 `clean_name`，清洗后为空且下拉为空 → 报错（与旧行为一致）。
- 列表上游（如 `SFParsePath.filename` 是 `OUTPUT_IS_LIST`）：`_resolve_name` 对 list/tuple 取首个非空项（check_lazy_status 用）；execute 按 ComfyUI 语义逐项执行，每个名字各存一份、输出列表。

### 2. 前端置灰：`widget.disabled` + 必须调 `updateComputedDisabled()`

- Vue 前端 `DomWidget.vue` 的透明度/pointer-events 读的是 **`computedDisabled`**，而它由 `LGraphNode.updateComputedDisabled()` 计算（`widget.disabled || connected`）——所以只写 `widget.disabled = true` 不会立即重绘，**必须再调 `node.updateComputedDisabled?.()`**（先例：该函数同时兜住"widget 槽被直接连线"的自动压制）。
- 连接态检测复用 `sf_dynamic_slots.isSlotConnected`（`link` 数字 / `links` 数组两形态；纯模块可复用，禁内联副本）。
- 安装点：`installCacheNameList` 内挂实例级 `onConnectionsChange`（幂等标志防重复包裹）+ 既有 `onNodeCreated`/`onAfterGraphConfigured` 安装点兜底（configure 恢复不触发 `onConnectionsChange`）；断开时还原 `disabled`（首次记录原值，避免覆盖他人设置）；初始未连接时直接记录状态、不触碰 widget。
- `sf_cache_name_lib.js` 纯模块新增依赖 `sf_dynamic_slots.js`（同为无 app 依赖纯模块）→ 其 .mjs 测试改为**拷贝链**（两文件 + 相对导入改 `.mjs`，test_lora_browser_smoke.js 同款）。

### 3. 测试

- 后端（`tests/test_track_data_cache.py` / `test_mask_cache.py`）：schema（forceInput/default）、`_resolve_name`（文本优先/空白回退/列表取首项/非字符串回退/全空）、lazy 命中与未命中（name_text 路径）、execute 覆盖写入与读取、文本空白且下拉为空报错。
- 前端（`tests/test_cache_name_lib.mjs`）：`isNameTextLinked` 两形态、`applyNameOverrideState` 置灰/恢复/保留原有 disabled/状态未变提前返回、`installCacheNameList` 按钮幂等与 hook 驱动置灰（stub fetch/window）。

## 81. SF Track Data Add：track_data 并集叠加 + 动态槽位（2026-09）

> 背景：`SFTrackDataSubtract`（见 `nodes-video.md` §77）只能"减"。需要其逆操作——把多路 MASK / `SAM3_TRACK_DATA` 逐帧并集叠加到基础追踪数据并**合成单一身份**（例如把分别追踪的目标合成一个驱动遮罩）。

### 1. 纯逻辑单源

- `sf_utils/track_data_ops.py::add_to_track_data`：复用同模块 `_exclusion_frames`（MASK/TRACK_DATA 归一为逐帧 bool `[T,H,W]`）与 `_resize_frames`（尺寸对齐），零重复实现。
- 基础有对象：`unpack(packed).any(dim=1)` 跨对象塌平 → 各路叠加归一/对齐后逐帧 OR → `pack_masks(union).unsqueeze(1)`，输出对象维恒为 1；`n_frames`/`orig_size` 保留、`scores=[1.0]`（对齐 `invert_track_data` 塌单身份）。
- 基础为空（packed None/0 对象）：无叠加浅拷贝直通；有叠加按 `orig_size`/`n_frames` 补零基础（缺省回退首个叠加的尺寸/帧数）。
- 叠加帧数与基础不一致 → ValueError（对齐 Subtract）；宽度补 8 倍数抽为 `pad_width_to_8`，`mask_to_track_data` 同源复用（原先内联副本）。

### 2. 节点与动态槽位（4 → 动态）

- `SFTrackDataAdd`（`nodes/image/track_data_add.py`，CATEGORY `sfnodes/image`）：required `track_data` + optional `add_1..20`（`"MASK,SAM3_TRACK_DATA"`）。
- **Subtract 同步改造**：`exclude_1..4` → `exclude_1..20` 静态 schema，`execute` 收集 `kwargs` 前缀值。静态 20 槽 + 前端裁剪是既有范式（`SFImageBatch`），无需灵活 schema / `VALIDATE_INPUTS`；并集/相减满足交换律 → 无需数字序排序。
- 前端 `web/sf_track_data_slots.js` 单文件管两节点（`exclude_` / `add_`），初始 4、上限 20（`installDynamicSlots`）。

### 3. 抽出 `installConfiguredSlotRecovery`（去重）

- `sf_conditioning_combine.js` 与 `sf_conditioning_concat.js` 曾各内联一份逐字相同的 `onAfterGraphConfigured` 补齐/回收逻辑；新增 `web/sf_dynamic_slots.js::installConfiguredSlotRecovery(node, {inputPrefix, inputStart, inputType, initialInputs, inputCount})` 为单源实现，三节点共用（命名统一 `prefix + (start + count)`，等价原 `condName`）。
- 加载/粘贴 configure 直赋 links 不触发 `onConnectionsChange`，按链接数补齐到 `linked+1`（上限内）并回收尾部空槽。

### 4. 测试

- 后端 `tests/test_track_data_ops.py`：Subtract/Add 结构（20 槽）+ `add_to_track_data`（单/多对象塌单身份、TRACK_DATA/MASK 两种叠加、多路并集、resize、帧数不一致报错、空基础叠加/直通、补宽路径）+ 两节点 execute。
- 前端 `tests/test_track_data_slots_js.js`：初始 4 槽、全连追加、断开回收、`installConfiguredSlotRecovery` 补齐/回收、非目标 class 不处理。
- `tests/test_mask_to_track_data.py` 补 sfnodes 包结构桩（节点相对导入 `sf_utils.track_data_ops` 后必需）。

### 5. 追踪器工作分辨率 ≠ `orig_size`：add 必须保留 `orig_size`（2026-09 修复）

- **症状**：`SF Track Data Add`（及 `SFTrackDataMerge` 带 `+`）的输出接 `SAM3 Track to Mask` 后 mask 宽高比变成 **1:1**。
- **根因**：SAM3 追踪器在固定**方形**工作分辨率 `image_size`（默认 1008）上打包（`comfy/ldm/sam3/tracker.py` 里 `F.interpolate(size=(image_size, image_size))`），所以 `packed_masks` 的 H/W 是方形、与真实宽高无关；真实宽高由 `SAM3_VideoTrack.execute` 事后写入 `result["orig_size"] = (H, W)`。`SAM3_TrackToMask` 正是用 `orig_size` 把 mask 插值回真实 `(H, W)`。
- **bug**：`add_to_track_data` 曾用 `unpack_masks(packed).shape[-2:]`（方形工作网格）覆盖输出 `orig_size` → 下游拿到方形尺寸 → 1:1。`subtract_from_track_data` 用 `dict(track_data)` 保留了 `orig_size`，故只有含 `+` 的路径出问题。
- **修复**：`orig_size`/`n_frames` 继承输入；`unpack` 的 H/W 仅作**对齐各叠加路的工作网格**（`_resize_frames` 目标）。测试需用"方形 packed + 非方形 orig_size"才能暴露（现有用例 orig_size 与 packed 同形，测不出）。

## 83. SFTrackDataMerge：单节点逐槽加减（动态槽 + 每槽 -/+ 模式）

> 背景：`SFTrackDataSubtract` 串联 `SFTrackDataAdd` 完成"基础 − 排除 + 叠加"需要两个节点、两段接线；一个工作流里加减路一多，接线与节点数都很繁琐（见 `[scail2]SCAIL-sam3.1-原生(排除).json`：452 相减 → 474 叠加）。目标=一个节点、一列动态槽，每路可独立定义为"减"或"加"。

### 1. 语义：等价串联，先减后加、顺序无关

- 纯逻辑 `sf_utils/track_data_ops.py::merge_track_data(base, subtract_masks, add_masks, ...)` **只做组合**：先 `subtract_from_track_data`（多路并集后逐对象相减、保留对象数），**仅当存在至少一个有效叠加时**再 `add_to_track_data`（并集塌单身份、`scores=[1.0]`）。
- **坑**：`add_to_track_data` 对非空基础即使 `add_masks=[]` 也会塌成单身份（base 的 `any(dim=1)` 无条件打包）。因此组合前必须过滤出至少一个有效叠加才调用它，否则"纯相减"会意外丢对象数。
- 相减各路并集、叠加各路并集，相减恒先于叠加 → 结果与槽序无关；"先加后减"需求请另接一个相减节点。
- **`orig_size` 必须来自输入而非 `unpack` 尺寸**（SAM3 追踪器工作网格是方形，`orig_size` 才是真实宽高）——详见 §81.5，否则下游 `SAM3 Track to Mask` 宽高比变 1:1。

### 2. 每槽模式的数据通道（值通道模式 + 注入）

- 后端 `SFTrackDataMerge`（`nodes/image/track_data_merge.py`）：required `track_data` + optional `track_1..20`（`"MASK,SAM3_TRACK_DATA"`）+ hidden STRING `SlotModes`。`execute` 按 **槽序号**（非 kwargs 顺序）收集，`_parse_modes` 容错解析 JSON，非法值归 `sub`。
- 默认模式 `sub`（保守）：叠加会塌单身份，默认不加，避免误塌。
- 前端模式状态存 `node.properties.sfTrackMergeModes`（随工作流保存，`SFPromptStack` 同款），提交时经 `graphToPrompt` 钩子注入 `SlotModes`（只注入不剪枝）。**properties 不会自动进 prompt，注入不可省**。
- 节点体 DOM 行列表逐槽 `-`/`+` 切换（Classic/Vue 均渲染；不用 canvas 手绘——Vue 下 `onDrawForeground` 不保证调用）。复用 `sf_common.el/injectCSSOnce/isVueNodes` 与 `sf_prompt_stack` 的 DOM widget 形态。

### 3. 前缀撞名：`track_data` vs `track_N`

- 基础槽名 `track_data` 与动态前缀 `track_` 撞名，`installDynamicSlots`/`installConfiguredSlotRecovery` 的裸 `startsWith(prefix)` 会把基础槽当动态槽：初始会被裁成 3 个而非 4 个、下一槽号跳号、恢复时基础链接被计入 `linked`。
- 修复：`sf_dynamic_slots.js::installConfiguredSlotRecovery` 新增可选 `inputMatch`（与 `installDynamicSlots` 的 `inputMatch` 同款，默认 `startsWith` 保旧行为），本节点两处均传 `(name) => /^track_\d+$/.test(name)`；纯 lib `collectSlotNames` 同样要求**数字后缀**。**任何"前缀 + 编号"动态槽若固定槽同前缀，都必须用 inputMatch/数字后缀排除。**

### 4. 复用与测试

- 复用：`track_data_ops`（组合既有两函数）、`sf_dynamic_slots`（动态槽 + 恢复）、`sf_common`（DOM/主题）、`sf_prompt_stack` 的注入范式；既有 Subtract/Add 节点**保持不动**（向后兼容）。
- 测试：后端 `tests/test_track_data_ops.py`（merge 纯逻辑：仅减保留对象数/scores、先减后加塌单身份且减掉区域不复活、仅叠加等价 Add、空基础、全空直通 + Merge 结构与 execute 默认 sub/混合模式/非法输入）；前端 `tests/test_track_data_merge_lib.mjs`（parse/get/set/toggle/serialize/collectSlotNames）与 `tests/test_track_data_merge_js.js`（含基础槽的裁剪/追加/回收、DOM 行渲染与点击写 properties、graphToPrompt 注入）。

## 85. SFCanvasSizePreset 全局自定义分辨率库（2026-09）

> 背景：节点原为纯官方预设表（按模型/档位、16 整除），无法保存常用非官方尺寸。本次加"全局命名自定义分辨率库"：节点挂「⚙ 自定义分辨率」DOM 按钮，弹窗内增删命名条目 `{name,w,h}`，持久化到 `user/sfnodes/canvas_size_presets.json` 跨工作流共享。
>
> **关键设计（2026-09 二次修订）**：自定义库是 model 下拉里的独立分类 `-- Custom --`（伪模型 `"Custom Resolution"`），选中它时 `resolution` **只列自定义库**。**不可**把自定义项混进真实模型的 resolution 列表——ComfyUI 前端并不渲染 `--x--` 分组头（源码无对应逻辑），自定义项放末/置顶都只会变成该模型列表里的普通条目，用户会认为"被归类到某模型"。

### 1. combo 值编码：`"WxH (name)"`

- `resolution` 是 combo，值即显示串。自定义条目编码为 `"1600x900 (My Wide)"`：既有 `_parse_resolution` 取 `split(" ",1)[0]` 得宽高、首个 `(...)` 得 ratio 文本，**天然解析，后端节点执行逻辑零改动**。
- 代价：name 不得含 `(` / `)`（否则解析截断），后端 `_valid_name` 与前端 `validCustomName` 双端拒绝；名称若恰为 `16:9` 则 `resolution` 输出即 `16:9`（合理）。
- 删除预设后旧工作流里的值仍可解析执行（不查表），健壮。

### 2. 伪模型 + 路由命名避让

- 后端 `MODEL_GROUPS` 增第三组 `("-- Custom --", [CUSTOM_MODEL])`，`PRESETS[CUSTOM_MODEL] = {}`（无官方档位），路由 payload 增 `custom_model` 字段（前端不硬编码模型名）。
- 既有 `GET /api/sfnodes/canvas_size_presets` 返回官方模型表；自定义库占用 `/api/sfnodes/canvas_size_custom`，避免与旧接口冲突（零改动旧链路）。

### 3. 前端重建与同步

- 纯逻辑 `sf_canvas_size_lib.js`：`normalizeCustomPresets` / `customOptionValue` / `mergeResolutionValues`（`--Custom--` 分组头 + 自定义项 + 官方表；与官方重复的值跳过）。
- `syncNode` 分两路：真实模型 `values = merge(official, custom)`；伪模型 `official=[]`，`values = merge([], custom)`，空库给占位 `[--Custom--]` 保证 combo 非空。回退目标：真实模型取首个**官方**选项（不落到自定义项），伪模型取首个自定义项。
- 两接口并行加载并缓存：model 切换用缓存**同步**重建；自定义库增删后 refetch + syncAll 重建**所有**同类节点。
- 管理弹窗双击/套用条目时**顺带把节点 model 切到伪模型**（否则值不在真实模型的 options 里，再次 sync 会被回退）。
- 降级：官方表失败保持 INPUT_TYPES 静态选项（绝不用默认模型表冒充其他 model）；自定义库失败仅隐藏自定义项。
- **恢复时 preserveValue**：`syncNode(node, true)`（数据加载 / `onAfterGraphConfigured`）当前值不在列表时**保留**（删库/离线旧值仍可执行），仅用户切换 model 才回退。回退目标取该 model 的**首个官方选项**（`firstSelectable(official)`）而非合并表首项，避免落到置顶的自定义项。工作流恢复须等 `Promise.all([loadOfficial, loadCustom])` 都就绪再同步，否则自定义库未到会把恢复的自定义值误回退。

### 4. 复用与测试

- 复用后端 `disk_state.atomic_write_json/mtime_size_sig/sf_user_dir`、`common.valid_name`、`logger`；前端 `sf_common.el/injectCSSOnce/sfApiUrl`、`sf_popup.attachPopupDismiss/clampToViewport`。存储骨架对齐 `crop_expand_presets.py`（w/h 语义：像素整数 vs 比例浮点，故独立成模块而非参数化）。
- 测试：后端 `tests/test_canvas_size_presets.py`（CRUD/校验/归一化 mtime 重载/路由）+ `tests/test_canvas_size.py`（自定义值解析、路由注册、import 触发；加载改包上下文以支持相对 import）；前端 `tests/test_canvas_size_lib.mjs` + `tests/test_canvas_size_js.js`。

## 88. SFTrackDataSlice：SAM3_TRACK_DATA 时间维切片（外部分段配套，2026-09）

### 88.1 背景：为什么需要切片

外部分段跑 SCAIL-2（见 nodes-video.md §87）时，`SFSCAIL2SimpleVideo` 每次只看到本段 `pose_video`，但 `driving_track_data` 若传整段，节点内 `SCAIL2ColoredMask` 会按**整段 T** 渲染彩色蒙版（§82 的分块补丁只压中间峰值，输出仍 O(T)：4492 帧 ≈ 31 GB f16），且段内 pose 与 track 帧序错位（track 从 0 起 vs pose 从 skip 起）。

正解：整段追踪只做一次（或读 `SFTrackDataCache` 缓存），循环内用 `SFTrackDataSlice` 按 `start=skip, length=段长` 切片后喂给节点——每段只渲染段内蒙版，内存回到 O(段长)。

### 88.2 纯逻辑 slice_track_data（sf_utils/track_data_ops.py）

```python
slice_track_data(track_data, start=0, length=0)
```

- `packed_masks[start:end].contiguous()`，`n_frames = end - start`（**以 span 为准**，与 packed 首维一致；`packed_masks is None` 的整条空追踪只调 `n_frames`，时间语义不变）。
- `start` 支持负值（相对末尾，-1=最后一帧），越界夹到 `[0, T]`；`length <= 0` 表示切到结尾，超出尾部自动截断；空切片合法（首维 0、`n_frames=0`）。
- **保留 `scores`（per-object，不随帧变）与 `orig_size`**——追踪器方形工作网格 ≠ 真实宽高（§81.5），切片绝不能碰 `orig_size`，否则下游 `SAM3_TrackToMask` 输出 1:1。
- 输出浅拷贝 dict，不改原输入；`T` 来源优先 `packed.shape[0]`，无 packed 时用 `n_frames`（与 `pad_track_data_front` 同约定）。

### 88.3 节点与测试

- `nodes/image/track_data_slice.py`：`SFTrackDataSlice`（required `track_data`/`start`/`length`，`start` min=-1000000 支持负值取尾），非法输入抛错；无前端 JS（静态输入）。
- 测试走 §83 同款 numpy 位序桩（`FakeTensor` 增补 `contiguous()`），覆盖区间/负 start/长度 0/截断/越界空/多对象保留/`packed None`/execute 集成——`tests/test_track_data_ops.py`。

## 90. SFImageCropExpandBrushMask：出界裁剪 + 画笔遮罩合体（2026-09）

> 背景：把 SFImageCropExpand（出界裁剪/外绘预处理）与 SFImageBrushMask（节点内画笔遮罩）合成为一个节点。`nodes/image/crop_expand_brush_mask.py:SFImageCropExpandBrushMask` + `web/sf_crop_expand_brush_mask{,_lib}.js` + 三个新共享模块（源图链路/比例弹窗/画笔工具）。开工前已确认：单 mask = 扩展区 ∪ 笔触 / Crop-Brush-Erase 三模式 / 全量提取共享 / 本版不做 SAM。

### 1. 设计要点（用户拍板）

- **mask 单一输出**：扩展区（结构性，Erase 不作用于它）∪ 笔触（brush 置 1 / erase 置 0）——后端即 `_compose_expand(..., overlay=)` 内 `np.maximum` 并入，overlay 为源图坐标系的笔触栅格化结果；不传 overlay 时原行为逐字节不变（既有测试锁定）。
- **笔触只在源图区域内可画**：坐标钳制源图（`parse_state_strokes` 的裁剪面直接复用），随图移动、只有落在裁剪框内的部分进入输出；扩展区本就遮罩白，且避免前端离屏画布与后端栅格化随出界框膨胀到 12k² 的内存坑。
- **三模式单选 Crop/Brush/Erase**：Crop 模式整框可拖可移（与 CropExpand 完全一致），Brush/Erase 模式手柄失效——贴边涂抹不会误拖框。
- **双列布局**：列1 = 比例预设（CropExpand 同款，含 Custom 全局库弹窗），列2 = BrushMask 工具列前置 Crop；显示区多让一列（base lib `computeDisplayMetrics` 加 `extraLeft` 可选参数，默认 0 = 原行为）。MIN 360×300（宽 320+40）。

### 2. 共享提取（全量，避免第 3/4 份副本）

- 后端：`crop.py:load_src_rgb`（三节点源图读取）、`crop_expand.py:_compose_expand(overlay=)`、`sf_utils/brush_mask.py:lean_key`（原 `brush_mask._lean_key` 提升）+ 复用 `_clamp_crop/_parse_fill_color/parse_json_dict/parse_state_strokes/rasterize_strokes`；零新增路由、零新依赖。
- 前端新增：`sf_crop_source.js`（Load/Browse/拖放/Ctrl+V → `CropAPI.uploadSrc` → 宿主 `cfg.onStored` 回写 → /view 恢复；cfg 参数化上传前缀/文案/imgProp/getArea）、`sf_crop_expand_ratios.js`（预设库 API + `ratioLabel` + `openCustomRatioDialog(cfg)`）、`sf_brush_tools.js`（步长设置读取/注册 + `[`/`]` 快捷键双通道与时间戳去重 + S±/O± 滚轮；模块级注册表 + 单监听，action 经宿主 `buttonAction` 保证步长三路同源）。
- 前端提升：`sf_common.js` 新增 `pickColorInput` + `rgbStringToHex/hexToRgbString`（三节点取色弹窗与颜色换算收敛）；`sf_crop_expand_lib.js` 收编 `drawCropBox/drawPlaceholder` + 补 `imageToLocal`（与 localToImage 互逆）+ `extraLeft/minW/minH` 参数；`sf_brush_mask_lib.js` 收编 `drawStrokePath/paintStrokeMask/colorTextStyle`；`sf_common.js` 新增 `installResizeCornerCursor`（多类注册 + 单 mousemove 监听）。既有两节点改为 import（等价搬运，冒烟测试同步）。
- 图索引直接复用 `sf_pause_kit.buildClassNodeIndex/findNodeByPromptId`（四闸门单源，不引第 4 份 buildNodeIndex 副本）；随后把既有两节点的本地 `buildNodeIndex/findNodeById` 副本一并收敛到同一对（顺带获得复合 id 精确匹配与子图环守卫）。

### 3. 测试

- `tests/test_crop_expand_brush_mask.py`（mock torch/aiohttp/folder_paths）：结构/注册键/`_compose_expand(overlay=)`（点并入、不穿扩展区、无源忽略、坏 shape 防御）/execute（brush、按画序 erase、fill 多边形、框外笔触裁掉、缺源缺口退化）+ IS_CHANGED（预览字段不进键）。
- `tests/test_crop_expand_brush_mask_lib.mjs`：组合布局/MIN/`extraLeft` 与基库等价/冻结快照透传/`imageToLocal` 互逆/共享绘制 FakeCtx op 流（`paintStrokeMask` 真擦除 `destination-out` 画序、`drawCropBox` 框外压暗 + 8 手柄）。
- `tests/test_crop_expand_brush_mask_smoke.js`：23 控件、三模式切换、Crop 拖框 + `buttons:0` 释放兜底、Brush 源图内落笔/扩展区不起笔、离屏真擦除预览、lean 注入（不含预览字段）、`[`/`]`、computeSize MIN、onRemoved 解绑。
- 旧测试桩同步：两个既有 smoke 现在拷贝真实共享模块（`sf_crop_source`/`sf_brush_tools`/`sf_crop_expand_ratios`）并改写其 import 指向桩（桩必须提供与真实导出面一致的符号）。

## 91. SAM 菜单二期：合体节点接入 + 忙时熔断（comfy-aimdo 并发读崩修复，2026-09）

> 背景：SFImageCropExpandBrushMask（§90）接入 SFImageBrushMask 的 SAM 右键图层；用户真机报错——工作流有 SAM 任务进行中时点菜单：`aimdo: hostbuf_file_reader_retire_active: active slot 1 already has a completion event` + `[SFImageBrushMask:sam] inference failed: hostbuf_file_reader_read failed`。

### 1. 根因：路由线程与执行线程并发模型加载，撞 comfy-aimdo 全局 file reader

- 环境：容器 `comfy-aimdo 0.5.5`（AI Model Dynamic Offloader，`comfy/memory_management.py` / `model_patcher` 动态加载链）。native `.so` 内 file reader 为**进程级全局状态**（`_hostbuf_file_reader_slots/_active`；`hostbuf_file_reader_read` 无跨线程锁，`memory_management.py:59` 的 to-device 分支连 `info.lock` 都没有）。
- 触发：工作流 SAM 任务（执行线程）正在读权重，同时右键菜单路由在 **aiohttp 事件循环线程**同步跑 `_load_stack()/SAM3_Detect.execute` → 两线程并发进同一 reader → slot 冲突。
- 判定：不是本包代码写错——核心模型管理全局态（`current_loaded_models` / patcher 缓存 / 显存管理）**本就只支持执行线程单线程**，路由侧推理无法靠"给自己加锁"变安全。

### 2. 修法：忙时熔断（409）+ 同步路由封死竞态

- 后端 `brush_mask_sam.py`：`queue_busy()`（`PromptServer.instance.prompt_queue.get_tasks_remaining() > 0`，取不到视为不忙）；`_handle_sam(data, busy=None)` 在**一切校验/加载/推理之前**忙时返回 `409 {"busy": True, "error": ...}`；`sam_status()` 增 `busy` 字段；500 分支对含 `hostbuf` 的异常追加"工作流空闲时重试"提示（覆盖任务刚结束、aimdo 预取线程未退的极小残留窗口）。
- 路由**保持同步**（阻塞事件循环）是刻意的：检查通过后事件循环被占住，`/prompt` 无法入队、执行线程无新任务可领 → 竞态窗口实际关闭（改线程池/异步等待反而会重新打开）。
- 前端 `sf_brush_sam.js`：Run 前先 `GET sam_status` 预检 `busy`（warn toast，不发推理请求）；服务端 409 仍兜底（`samPost` 抛错携带 `status/payload`，409 走 warn）。
- 语义代价（用户拍板）：**工作流执行期间菜单 SAM 不可用**（两个节的 DESCRIPTION 已注明）；不用"服务端排队等空闲"（HTTP 长挂起/多等待者/关页后仍执行），也不做"全局 aimdo 读锁补丁"（改全 App 行为，且救不了核心全局态并发）。

### 3. 前端共享提取（二期顺带收敛）

- `web/sf_brush_sam.js`：菜单安装 `installSamMenu(cfg, nodeType)` / 对话框 `openSamDialog` / 调用 `runSamMask` / 卸载 `unloadSamModel`；cfg = `{ toastTag, logTag, getState, addStrokes(node, incoming, meta) }`——宿主决定笔触并入与记忆字段写入（`sam_prompt/sam_threshold/sam_refine` 仅对话框记忆，不进 lean 注入）。SFImageBrushMask 与 SFImageCropExpandBrushMask 单源（brush 节点原内联实现删除）。
- 路由名 `/api/sfnodes/brush_mask/sam*` 保留（按 `src_path` 通用；改名会破坏既有节点，无必要）。

### 4. 测试

- `tests/test_brush_mask_sam.py`：假 `server.PromptServer` 注入 `queue_busy` 三态（空闲/有任务/异常）与 `sam_status.busy`；忙时 409 且**不加载模型/不触发推理**（`comfy_sd.calls` 与 `SAM3_Detect.last` 均不变）；hostbuf 异常附带提示。
- 合体节点 smoke：菜单两项 + `sam_*` 默认记忆字段；brush smoke harness 增拷真实 `sf_brush_sam.js`（改写 import 指向桩）。
- `tests/check_web_imports.py` MODS 增 `sf_brush_sam`。

## 92. 画笔节点模式快捷键（B/E，合体节点 +C）与前端默认键位核查（2026-09）

> 背景：两个画笔节点（SFImageBrushMask / SFImageCropExpandBrushMask）此前只有 `[`/`]` 尺寸快捷键，模式切换只能点竖列按钮。按用户确认加 `B`=Brush、`E`=Erase（合体节点另加 `C`=Crop），不做 X 切换键；只在选中该类节点时生效。

### 1. 注册器泛化（web/sf_brush_tools.js）

- `registerBrushSizeKeys` → **`registerBrushKeys({ classNames, controlsProp, keyMap, applyAction })`**：`keyMap` 由宿主配置（键 → 宿主动作 id），模块内仍是单 window keydown 监听 + 模块级注册表 + `e.timeStamp` 双通道去重；返回 `keyStep(e)` 供节点 `onKeyDown`（官方通道）使用。
- 键归一化：单字母忽略大小写（`B`/`b` 同键），`[`/`]` 等符号精确匹配。守卫全部保留：修饰键 / 输入框 / `.sf-px-overlay` 跳过；按住连发与多选同戳去重行为不变（动作仅执行一次）。
- 动作仍走各节点 `buttonAction` 单一入口：brush 节点 keyMap `{ "]": "sizePlus", "[": "sizeMinus", "b": "brush", "e": "modeErase" }`（新增 `modeErase` 确定性分支——鼠标 Erase 按钮保持原 toggle 语义）；合体节点 keyMap `{ "c": "crop", "b": "brush", "e": "erase" }`（其 buttonAction 对这三位本就是确定性写入）。

### 2. 前端默认键位核查（防单字母冲突，方法可复用）

容器 `comfyui-frontend-package`（1.52.7）的默认绑定数组在 `static/assets/keybindingService-*.js` 顶部（`{combo:{ctrl/alt/shift/meta,key},commandId,targetElementId}`）。提取法：`docker cp` 该文件到 /tmp 后本机 python 正则解析（一次性脚本，不入库）。结论——**纯单键默认仅占用 `r/w/n/m/a/./p/v/h/Escape/Delete/Backspace`**，`b`/`e`/`c` 空闲；`Ctrl+B/M`（bypass/mute）、`Alt+C`（collapse）等带修饰键绑定不冲突（我们的 handler 仅在无 Ctrl/Meta/Alt 时响应）。

### 3. 测试与文档

- `tests/test_brush_mask_smoke.js`：E 切 Eraser、B 切 Brush（大写）、输入框 / 修饰键 / 编辑器打开 / 未选中不改模式（原有 `[ ]` / 滚轮 / 去重断言不动）。
- `tests/test_crop_expand_brush_mask_smoke.js`：E/B/C 三模式切换。
- 两节点 DESCRIPTION 增补快捷键说明（brush：B/E + `[ ]`；合体：C/B/E + `[ ]`）。

## 93. 画笔节点 AI 菜单三期：人物部位 / YOLO / 导入遮罩 / 反选（sf_brush_sam → sf_brush_ai，2026-09）

> 背景：在两个画笔节点（SFImageBrushMask / SFImageCropExpandBrushMask）的右键菜单里除 SAM 外再挂 5 类能力。用户拍板：人物部位、YOLO、导入遮罩、SAM 点/框选、反选（不做形态学/复制粘贴/导出）；合体节点反选语义为「只反笔触层」；点选为多点收集后 Enter 执行；前端模块改名 sf_brush_ai.js（§91 中的 sf_brush_sam.js 即其前身，`installSamMenu` → `installBrushMenu`）。

### 1. 菜单 8 项与后端路由

| 菜单 | 路由 | 实现要点 |
|---|---|---|
| SAM 文本 | `POST …/sam` | 原有（prompt/threshold/refine） |
| SAM 点选/框选 | 同上（payload 加 `positive_coords`/`negative_coords`/`bbox`） | core `SAM3_Detect` 已支持点/框提示，**仅点框时 `conditioning=None`**（已核实 core 走 SAM decoder 路径）；`_normalize_points/_normalize_bbox` 纯函数归一（越界丢弃、框钳制 ≥2×2、显式提供但全无效 → 400，避免静默退化成文本 "object"）；点选左键=正点、**Shift+左键=负点**（不用 Alt，见 §93.5） |
| 人物部位 | `POST …/person_parts` | MediaPipe selfie multiclass（脸/发/身体/衣服/背景）；分割核心抽到 `sf_utils/person_mask.py`（SFPersonMask 节点改委托，**懒 import mediapipe**——旧节点顶层 import 会让全包依赖 mediapipe 才能加载） |
| YOLO | `GET …/yolo_models` + `POST …/yolo` | 扫 `models/ultralytics/{bbox,segm}`；`ultralytics` 为**运行时可选依赖**（AGPL，不进 requirements，缺失给中文报错）；模型名白名单（单段文件名 + 在扫描清单内，防路径穿越）；bbox→矩形/椭圆用**纯 numpy**（不依赖 cv2）、segm→`masks.xy` 多边形经 PIL 填充；`.predict` 源按 BGR 约定故通道反转 |
| 导入遮罩 | `POST …/import_mask` | 纯本地无模型无熔断：灰度 → NEAREST 缩放到**当前源图尺寸** → `mask_to_fill_strokes` |
| 反选 | 无路由（状态位） | `invert` 进 state/lean/IS_CHANGED；后端笔刷节点 `1-笔触`、合体节点 `扩展区 ∪ (1-笔触)`（扩展区是结构性重绘区不被反选取消）；前端预览用 `paintInvertMask(srcCvs, outCvs)` 在离屏画布「白底 + destination-out 打洞」（主画布禁 destination-out 的纪律不破） |
| 卸载 AI | `POST …/unload_all` | 清 SAM/人物/YOLO 三层缓存 + empty_cache |

- 模型路由全部沿用 §91 的**忙时熔断 + 同步处理**（queue_busy() 409；同步阻塞事件循环封死入队竞态）。
- 共享 `open_src_image`（crop.py，PIL RGB）提升：SAM/人物/YOLO/导入四处源图读取收敛；`load_src_rgb` 改为其薄包装。

### 2. 前端 `web/sf_brush_ai.js`（两节点单源）

- `installBrushMenu(cfg, nodeType)` 装 8 项；cfg = `{ toastTag, logTag, getState, patchState, addStrokes(node, incoming, extra), toImage, fromImage, inDisplay, displayOrigin }`——`addStrokes` 的 extra 直接展开进 state（SAM/人物/YOLO 参数记忆统一，不进 lean 注入）；坐标换算由宿主注入（笔刷节点与合体节点的 metrics/displayMin 语义不同，模块保持节点无关）。
- 通用弹窗 `createModal({id,title,bodyHtml})` + `wireDialogKeys(inputs, apply)`（Enter 执行 / Esc 关闭 / 放行 ctrl·meta·alt；回车吞掉防上浮触发全局键位），SAM/人物/YOLO 三个弹窗共用。
- **点/框模式状态机**：`beginSamMode(cfg,node,kind)`（模块级单活动节点，切换即取消旧的；注册 window **capture** keydown：Enter 执行 / Esc 取消 / Backspace 撤回上一点——仅激活期消费，不干扰前端 Delete 删节点与 Escape 退子图）→ 节点 `onMouseDown/Move/Up` 首部 `handleSamPointer`（点选：左键正点、Shift+左键负点；框选：拖橡皮筋松开即跑）→ `onDrawForeground` 调 `drawSamOverlay`（绿/红编号点、虚线框、顶部提示条）。控件命中优先于模式（模式激活时按钮仍可用）；模式激活时光环置空、cursor 置 crosshair。
- 忙时：`aiBusy()`（GET sam_status.busy）预检 + 409 兜底 warn（§91 同源）。

### 3. 测试

- `tests/test_brush_mask_tools.py`（新，mock torch/aiohttp/folder_paths/cv2/mediapipe/ultralytics）：person_mask 纯核心（parts 归一/阈值并集/refine 二次分割/缓冲注入）；YOLO 清单扫描与白名单防穿越、`_boxes_to_mask` rect+ellipse、`_polygons_to_mask`、run_yolo bbox/segm（conf 钳制、labels、BGR）；三条模型路由忙时 409（不加载模型）与成功回笔触；导入遮罩缩放追踪；`unload_all` 清三层。
- `tests/test_brush_mask_sam.py` 扩展：点/框委托（conditioning None）、文本+点并存、`_handle_sam` 归一/400 路径。
- 两节点后端测试：invert 输出（笔刷全反 / 合体扩展区保留）与 IS_CHANGED（lean_key 追 `inv=`，默认键串变化已同步断言）。
- 两个 smoke：8 项菜单、反选 toggle + 反相离屏打洞、点选（正/Shift 负点 + Enter POST 载荷 + 覆盖层绘制断言）/框选（橡皮筋 + bbox POST）；stub_api 记录调用。
- `check_web_imports.py` MODS：`sf_brush_sam` → `sf_brush_ai`。

### 4. 真机修复：mediapipe 属性名与「mock 镜像 bug」陷阱（2026-09）

- 症状：人物部位菜单报 `module 'mediapipe.tasks.python.vision' has no attribute 'VisionRunningMode'`。
- 根因：原 `nodes/face/person_mask.py` 顶层有**局部别名** `VisionRunningMode = mp.tasks.vision.RunningMode`；抽取到 `sf_utils/person_mask.py` 时把别名误当成模块属性名写成 `mp.tasks.vision.VisionRunningMode`（容器实测该属性不存在，只有 `RunningMode`）。
- 教训：**mock 不要照抄被抽取代码的属性名**——本轮测试的 fake mediapipe 也写了 `VisionRunningMode`，于是单测全绿而真机炸（mock 与实现同错）。正确做法：mock 只暴露真实 API 的符号面（本次已把 fake 改为只提供 `RunningMode`），依赖真实名不符即失败。抽取"顶层别名"时先 grep 别名的定义来源，别按别名赋属性。

### 5. 真机修复二：覆盖层不显示 + Alt 点击克隆（2026-09）

- **点不显示**：`sf_crop_expand_brush_mask.js` 的覆盖层闭包用了 `imageToLocal` 但该文件**只 import 了 localToImage**——`drawSamOverlay` 抛 `ReferenceError` 中断整帧（同 §93.4 的教训：绘制期异常=节点内容整帧不画）。修法：补 import；两个 smoke 增加「模式激活时 onDrawForeground 不抛错 + arc/strokeRect/提示文本」断言（已用"去掉 import"反例验证会 FAIL——此前的 smoke 只测交互不测模式绘制，故漏网）。
- **Alt+左键克隆节点**：前端在**派发给 `node.onMouseDown` 之前**就处理 Alt（canvas 模式 `_processPrimaryButton` 的 `alt_drag_do_clone_nodes` 分支；Vue 节点 `nodeOnPointerdown` 先 `cloneNodes` 再 `ve(e)`），宿主 handler 返回 true 也拦不到。修法：负点改用 **Shift+左键**（节点体上 Shift 无早期分支；handler 返回 true 后前端清掉该次 onClick，不影响选择）。教训：**节点画布交互的修饰键先查前端默认行为表再定**（Alt=克隆、Ctrl/Meta=加选/panning、Shift=插槽连线/加选；后两者在节点体路径可消费）。

### 6. 真机修复三：合体节点画笔拖动时圆环不跟随（2026-09）

- 症状：Brush/Erase 拖动时笔触轨迹正常，但笔刷光环（圆环）停在起笔前的位置不动（用户反馈"拖动时显示轨迹，但鼠标图标没有改变位置"）。
- 根因：合体节点 `onMouseMove` 把 `_sfCEBCursor`（光环位置）记录写在**悬停分支**里，而落笔分支与裁剪拖框分支都提前 `return true` → 拖动期间悬停分支不可达，光环位置停留在拖动前最后一次悬停值。笔刷节点在函数顶部先记录再分支，故无此问题（同语义两实现、顺序不同 → 行为分叉）。
- 修法：`_sfCEBCursor` 记录上移到所有分支之前（每次移动都记；绘制侧仍有 `node_over` 与模式门控，SAM 模式/折叠等不受影响）；两个 smoke 的拖动用例补「光环跟随鼠标」断言（已用"移回悬停分支"反例验证会 FAIL）。
- 教训：同一处理函数里"状态记录"与"提前 return 的分支"并存时，记录必须在分支之前；跨节点镜像实现（两个 brush 节点的 mousemove）要逐段比对语句顺序，别只比函数名。

## 94. YOLO 菜单增强：多目录扫描 / 任务自检 / 类别过滤 / imgsz（2026-09）

> 背景：三期 YOLO 菜单只用 `models/ultralytics/{bbox,segm}` 的目录名推断任务类型，且无类别过滤/分辨率选项。核查用户全部 YOLO 权重后发现两类问题：`femaleBodyDetection_typea/yolo26.pt` 是 **seg 权重却放在 bbox 目录**（选 bbox 只出框、反过来选 segm 会 `masks=None` → 静默"未检出"）；多类模型（ntd11 7 类 / breast 2 类 / watermark 同名 2 类）无法只取某几类。按用户确认加四项增强（并授权整理模型目录）。

### 1. 四项增强

- **多目录扫描**：`_YOLO_DIRS = {"bbox": ("ultralytics/bbox", "yolo"), "segm": ("ultralytics/segm",)}`（相对 `folder_paths.models_dir`）。`models/yolo` 是 ComfyUI-RMBG 的 legacy 目录（person/COCO 检测器），**只读并入 bbox 清单**（不移动文件、不影响 RMBG）；`list_yolo_models` 多目录合并去重（前序优先），`resolve_yolo_model` 按序首命中。
- **任务自检** `_check_yolo_task(model, kind)`（纯函数）：`bbox` 选到 `task=segment` 权重 → 放行 + 响应带 `warning`（前端 warn toast）；`segm` 选到非 segment 权重 → **400** 明确报错（而不是空掩码）；`task` 取不到时不拦截。响应恒带 `task` 字段。
- **类别过滤**：`GET /yolo_classes?kind&model` 返回 `{names: {id: label}, task}`（懒加载并复用 `_yolo_cache`；**只构造模型读元数据、不推理 → 不加忙时熔断**，工作流运行中也能浏览）；`POST /yolo` 接受 `classes`（id 或类名混合，`_normalize_classes` 去重/丢弃未知，空=不过滤）→ `predict(classes=ids)`。
- **imgsz**：白名单 `640/960/1280`（小目标 nipples/eyes 用 960/1280；白名单外回退 640）→ `predict(imgsz=...)`。

### 2. 前端（sf_brush_ai.js YOLO 弹窗）

- imgsz 下拉 + Confidence 同行；**类别勾选区**（按 kind+model 拉取；`selectedIds = null` 表示"默认全选"，Set 表示显式选择；同名类显示 `label (#id)`、值恒为 id（避免同名歧义）；全选/全不选链接；**全不选在 apply 拦下**）；kind 与 task 不匹配时提前 warn 并中止（后端 400 兜底）。
- 记忆字段 `yolo_imgsz/yolo_classes` 随 extra 进 state（不进 lean 注入）。

### 3. 模型目录整理（宿主机 `/mnt/github/comfyui-docker/models`，绑定挂载=容器 models/）

- 确认所有下载/放置的模型**都持久化在宿主机**（models、custom_nodes、input/output/user 均 bind mount）；`models-ext`（宿主 `/home/syaofox/comfymodels`，只读）无 ultralytics/yolo 目录。
- 整理（只移不删）：bbox 重复的 `femaleBodyDetection_typea.pt`（与 segm 版同 md5）移入 `ultralytics/_archive/`；`femaleBodyDetection_yolo26.pt`（seg 权重）移入 `ultralytics/segm/`；segm 里的示例工作流 JSON 移入 `segm/_workflows/`；`models/sfnodes/` 内**陈旧仓库克隆**（自带 .git，非 ComfyUI 加载路径）移入 `models/_archive_sfnodes_repo_20260917/`，**保留 `person_mask/`**（MediaPipe tflite 落盘处，ModelManager 一律写 `models/sfnodes/<sub_dir>`，person_mask/occluder/region 三维度共用此持久化根）。
- 排查要点：`folder_paths.models_dir` 实测 `/home/comfy/app/models`；Node 侧 `models/sfnodes/person_mask` 已存在（16MB）；`models/yolo` 未被 `folder_paths` 注册（无 'yolo' 键）→ 我们的扫描自行解析路径，勿动该目录。

### 4. 测试

- `tests/test_brush_mask_tools.py`：多目录合并/首命中（同名 a.pt 双目录）、`_check_yolo_task` 四态、`_normalize_classes` 混合/去重、predict 的 `classes/imgsz` 透传与白名单回退、`_handle_yolo` 任务不匹配 400 / bbox+segment warning、`_handle_yolo_classes` 成功与无效模型。⚠ `_load_yolo` 按路径缓存 → 涉及 task/names 的用例必须先 `_yolo_cache.clear()` 再改 fake（本轮踩坑）。
- brush smoke：直接 import 共享模块 `runYolo`，断言 POST body 含 `imgsz/classes`。

### 5. 推荐模型补齐下载（2026-09，来源与验证）

按用户确认从两个 HF 镜像合集补齐缺失类别（宿主 `models/ultralytics/{bbox,segm}`，持久化；下载后逐个用 ultralytics 8.4.154 加载验证 task/names）：

| 用途 | 文件 | 来源 | 验证结果 |
|---|---|---|---|
| 写实 pussy（分割） | `pussy_yolo11s_seg_best.pt` | `Nudimmud/adetailers` | segment / `{0: Pussy}` |
| 写实 pussy（分割，旧命名 PubesAnus） | `pussyPubesAnusDetector_v10.pt` | `iahhnim/adetailer_collection` | segment / `{0: pussy}`（实际单类，name 里的 pubes/anus 未出现在 names） |
| 写实 anus（分割） | `anus_v4.pt` | `Nudimmud/adetailers` | segment / `{0: anus}` |
| 臀/肛周（分割） | `assdetailer-seg.pt` | `Nudimmud/adetailers` | segment / `{0: ass}` |
| 全身含男性（分割） | `person_yolov8s-seg.pt` | `iahhnim/adetailer_collection` | segment / `{0: person}` |
| 动漫多类（分割，yolo26） | `nsfw-anime-medium-x1280.pt` / `nsfw-anime-xl-x1280.pt` | `01miku/anime-nsfw-segm-yolo26` | segment / `{0: anus,1: nipple,2: penis,3: vagina,4: female face,5: male face,6: pubic hair}` |
| 去码（马赛克分割） | `mosaic_detection_seg.pt` | `iahhnim/adetailer_collection` | segment / `{0: mosaic_penetration,1: mosaic_penis,2: mosaic_pussy,3: mosaic_anus}` |
| 足部（检测） | `adetailerFootYolov8x_v20.pt` | `iahhnim/adetailer_collection` | **detect** / `{0: foot}`（唯一 detect，归 bbox） |
| 高端乳首（分割） | `Nipple-yoro11x_seg.pt` | `Nudimmud/adetailers` | segment / `{0: nipple}` |
| 周边小件（均分割） | `breasts_seg.pt` / `belly_seg_v2.42_less_groin.pt` / `armpit_seg_v1.1.pt` / `womensUnderwear_pantiesSegV3b.pt` / `Anzhc Face -seg.pt`（`{0: male,1: female}`）/ `Anzhc Eyes -seg-hd.pt` | `iahhnim/adetailer_collection` / `Nudimmud/adetailers` | segment（各类名单一） |

- 下载要点：HF `resolve/main` 直链（302→CDN，无需 token）；大文件走 curl `--retry 5 --retry-all-errors`——首次批量下载遇到 `SSL unexpected eof`（文件截断而 curl 未判失败），**必须下载后按仓库 tree 的 size 逐一校验**（本次靠校验才发现 2 个不完整文件）；含空格文件名用 `${f// /%20}`。
- ⚠ 脚本坑：`local a="$1" b="...$a..."` 单条 `local` 内引用尚未赋值的变量会展开为空 → URL 变 `https://huggingface.co//resolve/main/` 全部 404；须拆成多条 `local`/赋值。
- 仍未覆盖：**男性专属躯干**（用 `person_yolov8s-seg` 兜）、**精液/体液**（未找到可靠 YOLO 权重）。
- 归属：全部 segment 权重放 `models/ultralytics/segm/`、足部 detect 放 `bbox/`（与目录 kind 语义一致；选错类型会被 §94.1 的 task 自检拦下）。

### 6. 真机修复：YOLO 全模型漏检（ultralytics numpy 输入必须 uint8 0..255，2026-09）

- **症状**：YOLO 菜单"很多模型检测不到，返回空笔触"。
- **根因**：ultralytics `engine/predictor.py::preprocess` 对 **list-of-numpy** 输入**无条件** `im.float().div_(255)`（注释明说 "uint8 to fp16/32, 0-255 to 0.0-1.0"），只有直接传 `torch.Tensor` 才跳过除法（"already 0.0-1.0"）。我们的路由把源图建成 `float32 0..1` 再传 numpy → 被二次 /255 → 模型看到近全黑图 → 漏检（同图同模型实测：float 路径 boxes=0/masks=0，uint8 路径 boxes=1/masks=1）。
- **修法**：`brush_mask_tools.run_yolo` 统一输入 dtype——非 uint8 的按 0..1 语义映射到 0..255（`np.clip(x,0,1)*255+0.5`），再 `ascontiguousarray(img[..., ::-1])` 转 uint8 BGR；`_handle_yolo` 直接传 `np.array(pil)`（uint8 RGB，勿预除 255）。⚠ 仅影响 YOLO 路由：SAM 路由是 comfy 节点链，其张量本就要求 float 0..1，不要"顺手"改。
- **回归防护**：`tests/test_brush_mask_tools.py` 增断言——float 0..1 → uint8（R=1.0→255、B=0.5→128 且 RGB→BGR）、uint8 原样透传。
- **真机复核（同一张图全模型扫描）**：修复后 30 个权重多数正常检出（female_body 0.96 / nipples 0.86 / anime 多类 5 个 / person / 各类 seg 等）；仅内容不匹配的（无马赛克/无脚/无 watermark）为 0。进一步换图复核：`armpit_seg`(0.13@0.08) / `cockAndBallDetection2D`(0.30) / `pussy_yolo11s_seg_best`(0.30, 低至 0.13@0.08) / `panties` / `adetailerFootYolov8x`(0.29) 均可检出。
- **使用建议**：小目标/低分模型把 Confidence 降到 0.08–0.2；x1280 训练模型（nsfw-anime-*）用 imgsz 960/1280；内容专属模型（马赛克/足/水印）只在含对应内容的图上出结果。

### 7. 模型提示文案动态化：mediapipe 落盘位置与"首次"语义（2026-09）

- **落盘位置**：`ModelManager.get_model_path(name, sub_dir="person_mask")` = `folder_paths.models_dir/sfnodes/person_mask/<file>` → 容器 `/home/comfy/app/models/sfnodes/person_mask/selfie_multiclass_256x256.tflite` = **宿主 `/mnt/github/comfyui-docker/models/sfnodes/person_mask/`**（bind mount，持久化；同根另有 occluder/region 两个潜在 sub_dir）。
- **不会重复下载**：`downloader.download_model` 首行 `if (save_loc / model_name).is_file(): return True` 短路；进程内另有 `_person_cache["buffer"]` 字节缓存。只有文件缺失（新容器无该 mount / 手动删除）才会走网络。用户反馈"每次提示首次下载"实为**前端 toast 静态文案**（`runPersonParts` 写死"首次需下载 tflite 模型"），与真实行为不符。
- **修法（动态文案）**：新只读路由 `GET /api/sfnodes/brush_mask/ai_status` → `{busy, sam, person:{model_found,loaded}, yolo:{loaded:[...]}, yolo_models}`（不加载模型、不触发下载；`person_status()` 只做 `os.path.isfile` 探测）。前端 `aiStatus()` 取代原 `aiBusy()`（旧后端回退 sam_status 取 busy），`runAiRequest` 把 `opts.running` 支持为 `(status)=>string` 函数：
  - 人物部位：未落盘 → "首次需下载 tflite 模型，约 16MB"；已落盘未驻留 → "首次加载 tflite 模型"；已驻留 → 无后缀；
  - SAM 文本：仅在 `sam.loaded=false` 时提示"首次需加载 1.7GB 模型"；
  - YOLO：模型不在 `yolo.loaded` 时提示"首次加载权重"。
- 测试：`test_brush_mask_tools.py` 增 `person_status`（未落盘/已落盘/已驻留三态，含不触发下载语义）与 `_handle_ai_status` 汇总键断言。

## 95. 反选遮罩面板按钮（状态色可视化，2026-09）

> 背景：反选（Invert）此前只有右键菜单（✓ 前缀）与信息文本 "Inv" 后缀，节点面板上看不出状态。用户要求改为节点界面按钮并按状态变色。

- `sf_brush_mask_lib.js`：`TOOL_COL` 在 Clear/Undo 之后插入 `invert`（画笔列 9→10 项、合体节点列2 10→11 项；两节点最小高度不变：列底 232/254 均小于既有下限 320/300）；新增 `INVERT_ON_COLOR = "rgba(196,124,34,0.95)"`（琥珀，区别于强调色 accent 的模式高亮，明暗主题都醒目）。
- 两节点：`buildControls` 标 `isInvert`；`buttonAction` 的 `invert` 分支直接调共享 `toggleInvert(AI_CFG, node)`——与右键菜单同一实现（状态写入 + toast 单源，勿另写 toggle）；`drawButtons` ON 时琥珀底 + 白字（该钮字号 9px：30px 列宽容纳 "Invert"），OFF 常规底色与文字色。信息文本 "Inv" 后缀保留。
- 测试：lib 测试更新 TOOL_COL 项数/顺序并断言状态色常量；两个 smoke 增「点面板按钮 toggle + ON 状态色 fillRect」断言。
- ⚠ 教训：`TOOL_COL` 是共享列定义，增删项会整体移动后续按钮行——**测试与文档不要写死行号/项数注释**；本轮画笔 smoke 里写死的 S+ 坐标（`[25, 126+11]`）在插入 invert 后点到别行导致 3 个用例失败，已改为按控件几何解析（`_sfBrushCtrls.find(b=>b.id==="sizePlus")`）。同步检查了合体 smoke 的列2 坐标（Crop/Brush/Erase 在 invert 之前，未受影响）。

## 96. 组合节点非 Crop 模式隐藏裁剪框（2026-09）

> 背景：用户反馈"crop 没点选时，图片四周还是出现裁剪框，应在非 crop 状态不显示"。

- 症状：合体节点在 Brush/Erase 模式下仍绘制裁剪框全套（框外压暗 + 白框线 + 九宫格 + 8 手柄），涂抹时视觉干扰（压暗压住照片、手柄/网格无意义且手柄本就不可交互）。
- 修法：`setupDrawing` 里 `drawCropBox(ctx, rect, ...)` 加 `if (st.brush_mode === "crop")` 门控——非 Crop 模式完全隐藏框组件；**扩展区白色提示保留**（它表达的是 mask 恒白区，属于笔触语义而非框组件）。
- 交互不受影响：非 Crop 模式手柄在 `onMouseDown`/hover cursor 路径本就被忽略（只走 SAM/涂抹分支），仅视觉变化。
- 测试：合体 smoke 增「Crop 模式有 8 手柄 + 九宫格；Brush/Erase 均无」断言。⚠ 断言陷阱：手柄与 BCol 按钮同为白色 `rgba(255,255,255,0.9)`，须按尺寸区分（手柄 10×10 / 按钮 30×18）；九宫格用 `set:strokeStyle = rgba(255,255,255,0.4)` 唯一识别（源图边界虚线/resize 等是别的色值）；不要用 `strokeRect` 宽度判断——源图边界虚线也有大尺寸 strokeRect。

## 97. 画布线条粗细全局设置（sfnodes.Canvas.FrameWidth / CursorWidth，2026-09）

> 背景：用户反馈裁剪框/边界线/光环"都太粗了"，希望默认改细且可在设置页调。

- 两个设置（用户拍板拆开，不要合并成一个）：`sfnodes.Canvas.FrameWidth`（裁剪框边框 / 源图边界虚线 / SAM 框选橡皮筋）、`sfnodes.Canvas.CursorWidth`（画笔光环）；slider 0.5–3 step 0.25，**默认都 1.0**（原 2 / 1.5）。
- 层级靠派生值而非第二设置：辅助细线（九宫格 / 手柄描边 / 显示区网格 / SAM 点描边）统一 `sfFrameThin() = max(0.5, frame×0.5)`（默认 0.5，原 1）——一条"框 > 辅助线"规则，改框线时辅助线自动跟随。
- 实现落点：`sf_common.js` 定义 + 幂等注册（`registerSfLineWidthSettings`，三节点 `init()` 调用；SFImageCropExpand 原本无 init 需新增）+ 每帧直读 `getSettingValue`（轻量 map 查找，异常/非法回退默认）；纯库 `sf_crop_expand_lib.drawCropBox(ctx, rect, srcW, srcH, m, lineW = 1)` 尾参收宽度（裸调用/测试仍细），主扩展调用点传 `sfFrameWidth()`。
- 范围边界：只调"画布内容线条"；面板/底栏/按钮等控件 chrome 恒 1 不动；全屏编辑器（SF Image Crop / Inpaint）不在内。
- 测试陷阱：源图边界虚线也用 FrameWidth 且**全模式保留**，不能用"线宽值"判断裁剪框是否绘制（旧断言 `!lws.includes(2.5)` 会假失败）——按框线唯一色 `rgba(255,255,255,0.9)` 判 strokeRect；smoke 桩 ctx 需记录 `set:lineWidth`（brush smoke 的 makeCtx 原来不记录，补 `lw` 字段）与 stub_common 需补新导出。
- 设置注册断言用真实 sf_common 的 smoke（合体节点）做：`ext.init()` 后 `__settingDefs` 校验 defaultValue/type/attrs；brush smoke 的 stub_common 自己实现同名读写以驱动绘制断言。

## 99. 画笔节点 AI 识别结果随模式加减（Brush=添加 / Eraser=减去，fill_erase，2026-09）

> 背景：两个画笔节点（SFImageBrushMask / SFImageCropExpandBrushMask）右键菜单识别出的遮罩（SAM 文本/点/框、人物部位、YOLO、导入遮罩）此前一律并入（并集）。用户先提"加一个添加/减去开关按钮"，随后拍板更简方案：**复用现有 Brush/Erase 模式**——Brush（合体节点 Crop 模式同）结果添加；Eraser 模式从现有遮罩中减去识别区域；导入遮罩同样跟随。

- 语义与数据：新增笔触模式 `fill_erase`（与 `fill` 成对：fill 多边形整体置 1，fill_erase 多边形整体清 0/打洞）。改写只发生在共享并入点 `sf_brush_ai.js::mergeStrokes`（读 `cfg.getState(node).brush_mode === "erase"`，把 incoming 的 `mode==="fill"` 映射为 `fill_erase`）——五条结果来源全走此函数，后端五条路由零改动（模式是客户端编辑决策而非推理参数）。结果仍是普通笔触项：Undo/Clear/Invert/预览/lean_key 全沿用，**无新状态字段、无 UI 变化**。
- 后端 `sf_utils/brush_mask.py`：`parse_state_strokes` mode 白名单加 `fill_erase`；`rasterize_strokes` 的 fill 分支同吃两种，`_fill_polygon(..., erase=True)` 用 PIL 直接以 `fill=0` 在现遮罩上描画（多边形内清 0、外部不动）——画序语义与 brush/erase 一致，后画的 fill/brush 可补回。
- 前端 `sf_brush_mask_lib.js::paintStrokeMask`：`erase` 与 `fill_erase` 都走 `destination-out`，区别只在 `drawStrokePath(..., fill)` 的 fill 参数（erase 按线宽、fill_erase 按多边形整体）；主画布禁 destination-out 的纪律不破。
- 文案：减模式 toast 用「减去 N 个填充笔触」（**不套 mergedPrefix**——YOLO 的 prefix 是 "YOLO 并入"，会读成"并入减去"）；SAM 点/框进入提示追加"（Eraser 模式：识别结果从遮罩中减去）"。
- 测试：纯逻辑 `test_brush_mask.py`（打洞 + 先减后加画序 + 白名单保留）、`test_brush_mask_sam.py`、`test_crop_expand_brush_mask.py`（execute 层 fill→fill_erase 交集打洞）；前端 lib test 补 destination-out 多边形断言；两个 smoke 的 stub_api 支持 `globalThis.__aiResponse` 按用例返回 strokes，断言 Brush/Crop=fill、Eraser=fill_erase（合体节点直接调共享 `runYolo` 验证映射与文案）。
- 陷阱：① smoke 里 `findIndex(closePath)` 会命中前一笔 fill 的 closePath，须限定在 destination-out 之后查找（`(o,i)=> i>dstIdx`）；② 合体节点 Erase 仍只擦笔触层（扩展区结构性保留）——fill_erase 从 overlay 减去，`_compose_expand` 的扩展区不受影响。

## 101. 画笔节点多边形套索（多次点选闭合填充，Brush=添加 / Eraser=减去，2026-09）

> 背景：用户要求 Brush/Erase 增加"多次点选闭合涂抹"（PS 多边形套索）。复用既有 fill/fill_erase（§99）与预览/笔触管理 → **零后端改动**。已确认：工具列 Poly 状态开关（紧接 Erase）、Brush=添加 / Erase=减去、合体节点 Crop 下点击自动切 Brush、PS 风格闭合键位。

- 数据/复用：闭合 = 普通笔触 `{mode: "fill"|"fill_erase", size: 0, points}`（Eraser 模式 fill_erase 打洞）→ Undo/Clear/Invert/lean 注入/栅格化/预览全沿用；`sf_utils/brush_mask.py` 与 `paintStrokeMask` 零改动。顶点纯逻辑进 `sf_brush_mask_lib.js`（`polyStrokeForMode`/`polyCanClose`/`polyShouldAppend`/`hitPolyFirst`，可 .mjs 直测）。
- 共享 UI `web/sf_brush_poly.js`（两节点单源，仿 sf_brush_ai）：`togglePoly`（宿主 buttonAction 统一入口；关闭丢弃会话 + 卸键盘）、`handlePolyPointer`（down/move）、`handlePolyDblClick`、`drawPolyOverlay`、`cancelPoly`/`disposePoly`。会话 `node._sfBrushPoly = {points, cursor}` 为内存态（闭合前不进 lean/工作流）；键盘 capture 仅 Poly 开启期安装，Enter/Esc/Backspace 消费、输入框放行。
- 交互：左键落点（相邻 <1 图像像素忽略 → 双击第二击不写重复顶点）、点首点（显示像素 10px 容差）/双击/Enter 闭合、<3 点取消并 toast、Backspace/Delete 删末点、Esc「有会话取消会话 / 无会话关工具」、右键取消；关闭/换图/删节点清理会话。
- 宿主差异：落点接口 `AI_CFG.toSource`——画笔节点钳制到源图（同自由笔）；合体节点源图外（扩展区）返回 null 拒绝（扩展区恒遮罩白，同画笔落笔限制）。合体节点 Crop 模式点击 Poly 自动切 Brush；两节点信息文本加 Poly 状态。
- 按钮状态色：`POLY_ON_COLOR = "rgba(46,160,67,0.95)"`（绿）——Poly 按钮随开关变色，与模式强调色、INVERT 琥珀均区分；两宿主共用同一常量（sf_brush_mask_lib），ON 时文字转白。⚠ 本轮实装踩坑：画笔节点只在 `buildControls` 加了 `isPoly` 标记，漏了绘制分支（合体节点有）→ 真机按钮开关不变色；**共享标记 + 各宿主各自绘制分支**的模式下，新增状态位必须两边都断言 ON 色 fillRect（smoke 无法跨端发现"标记齐但绘制漏"）。
- 顶点标记尺寸可调：`sfnodes.Canvas.PolyVertexSize`（sf_common 注册 + `sfPolyVertexSize()` 每帧直读，slider 1–8 step 0.5，**默认 2**——用户反馈原 4×4 方块过大；普通顶点画 `size×size` 方块、首点圆半径 = `size/2+1`（靠近闭合再 +2 高亮））。smoke 桩 `__bmSettingVals` 可驱动尺寸断言（2×2 默认 / 5×5 自定义），合体 smoke 用真实 sf_common 断言设置项注册。
- 模式互斥：开 Poly 时宿主 `cancelSamMode`；`beginSamMode` 调 `cfg.cancelPoly?.()`（sf_brush_ai 单行），双方向互为兜底（`polyReady` 检查 `_sfAiSam`，SAM 指针处理优先）。
- 布局影响（⚠）：TOOL_COL 插入第 3 项（brush, erase, poly）→ 画笔列 10→11（MIN 320 仍够，列底 254 < 284）；合体列2 11→12 → MIN 高 300→**320**（列底 276 + 底行 26 + 边距；存量节点载入时 `clampNodeSize` 自动抬升）。合体 smoke 里所有 offsetY=42 的写死坐标随之 +10（42→52）。
- 测试：lib .mjs 补 TOOL_COL 11 与四个纯函数；brush smoke 补按钮开关/三次落点/Backspace/Enter 不足 3 点/点首点闭合 fill/Erase 双击 fill_erase/覆盖层/Esc 两段语义/关闭丢弃；合体 smoke 补 Crop 自动切 Brush、双击 fill、扩展区拒绝、关闭丢弃；合体 lib test 更新 TOOL_COL 12 与 MIN 360×320。
- 陷阱：① `polyShouldAppend` 必须挡双击第二击，否则重复顶点破坏首点吸附（首点取数组第 0 项）；② 首点闭合判定用局部坐标 + `cfg.fromImage`，别在共享模块里猜 scale（两节点 metrics 公式不同，合体还有 displayMin 偏移）；③ Poly 键盘监听模块级单活动节点（同 SAM），换节点/删节点须 `disposePoly`，否则 Esc/Backspace 会被陈旧节点吞掉（Backspace 还会拦掉前端删节点）；④ 关 Poly 必须丢弃未闭合会话，否则再次开启会续上旧点；⑤ 合体 `toSource` 拒绝时直接 `return false`，把点击交还宿主后续分支（Crop 拖框等）而非消费掉。

## 104. 移除 insightface 依赖：人脸推理自研 ONNX 化（SCRFD / 2d106 / ArcFace，2026-09）

> 背景：insightface 是全包最难安装的依赖（编译/版本敏感），实际只用到三个 ONNX 模型：检测 `det_10g`/`scrfd_10g_bnkps`、关键点 `2d106det`、识别 `w600k_r50`/`glintr100`。方案：保留全部人脸相关节点与现有模型文件，`sf_utils/face_onnx.py` 自实现预处理/解码，`insightface_utils.py` 更名 `face_analysis.py` 保留 `InsightFace` 兼容层（六方法签名/返回结构不变），`requirements.txt` 删 `insightface`（`onnxruntime` 保留）。旧代码 `nodes/face/analysis.py` 顶层 import insightface + 根 `__init__.py` 直接导入该模块 → 缺依赖时**全包 195 节点不可用**，本改造同时解除该硬耦合。已确认：模型下载清单不变（仍走 downloader/HF）。

- **文件布局**：`face_onnx.py` 纯推理（numpy/cv2/onnxruntime，无 torch/folder_paths/skimage/onnx）；`face_analysis.py` 兼容层（`INSIGHTFACE_DIR`/`THRESHOLDS`/`InsightFace`）；`face_detector.py` 用 `FaceEngine(allowed_modules=["detection"])` 仅加载检测；`nodes/face/analysis.py` 的 `load_insight_face` 改 `FaceEngine(models/<name>)`，节点签名/输出/下载不变。
- ⚠ **模型 mean/std 不能按“人脸模型”一刀切**：SCRFD 检测 `127.5/128`（swapRB）；`2d106det` 是 MXNet 转换模型，`_minusscalar0`/`_mulscalar0` 已把归一化烘焙进图 → 预处理必须 `mean/std=0/1`；ArcFace 识别 `127.5/127.5`。判定依据是“图头 8 节点是否含 Sub/Mul”（insightface 原逻辑），别凭文件名猜——parity 对不上时先查这里。
- ⚠ **SCRFD 9 输出是 group-major**：`[score@8/16/32, bbox@8/16/32, kps@8/16/32]`，取 `idx+fmc`/`idx+fmc*2`；不是 stride-major。anchor 生成用 `np.mgrid` 序（x,y）+ `num_anchors=2` 交错，NMS 为面积 +1 版、阈值 0.4，letterbox 后坐标统一 `/det_scale`。
- 对齐语义差异：`2d106det` 用 **bbox 中心**相似变换（`scale=input/(max(w,h)*1.5)`，输出 `(pred+1)*(input//2)` 后逆仿射，rotate=0），不是 5 点 `norm_crop`（那是 ArcFace 的 `arcface_dst` 模板 + Umeyama）。5 点对齐用 numpy Umeyama 替代 skimage（矩阵差 ~1e-5，106 点最大差 ~0.005px，embedding 余弦 1.0，可忽略）。
- 动态 shape：`det_10g` 输出形状元数据烘焙为 640（12800），换尺寸推理时 ORT 打 `VerifyOutputSizes` W 告警但结果正确，`set_default_logger_severity(3)` 压掉（原 insightface 同样处理）。多尺寸回退（640→576..320）改为显式逐尺寸 `detect`——旧封装改 `det_model.input_size` 在 insightface 2.0 实际不生效（`detect` 走 `input_sizes` 而非 `input_size`），新实现恢复原意图。
- 懒推理：`FaceEngine.get(..., need_landmark/need_embedding)` 按需跑模块（旧 `FaceAnalysis.get` 每次都跑全部已加载模型）；`get_face` 只检测 → `get_bbox`/`get_single_bbox` 不再白算关键点/识别，顺带提速。`Face` 为载体 dict（键+属性双访问 + `normed_embedding` 属性）；`SFFaceAnalysisModels` 的 `FACEANALYSIS` 输出对象改为 `FaceEngine`（无消费者）。
- 测试：`tests/test_face_onnx.py`（mock cv2/onnxruntime/torch/torchvision/folder_paths）覆盖纯函数（anchor/distance2bbox/kps/NMS/Umeyama/bbox 变换）+ fake session 解码（含空检出）+ engine（allowed_modules/懒推理）+ 兼容层（多尺寸回退/面积排序/get_embeds/get_keypoints）。
- 容器 parity 方法（可复用为回归手段）：`docker cp sf_utils/face_onnx.py` 到容器 /tmp，与 `insightface.app.FaceAnalysis` 同图对拍 bbox/kps/106 点/embedding——`buffalo_l` 8 图 + `antelopev2` 4 图均 PASS（bbox/kps ≤0.0001px，106 点 ≤0.005px，余弦 ≥0.999999）。

## 109. 组合节点双列视觉分层（列头 / 分组 / 重命名 / 悬停说明，2026-09）

> 背景：SFImageCropExpandBrushMask（§90）两列 34px 面板同底色、间隔仅 4px，23 个同款按钮实际读作一整块按钮墙；且语义交叉（列1 Color=扩展区填充色 vs 列2 BCol=画笔色；Reset/Clear/Undo 分属两列但上下相邻；Crop 模式在列2，而"裁剪"比例预设整列在列1）。用户反馈"既不美观也容易混淆"，选定**方案 A：按钮位置全不变**，只做视觉分层 + 悬停中文说明（否决了"删列1 改比例弹窗"的方案 B 与"Crop 模式换列"的 A+）。

- 布局纯逻辑（`sf_crop_expand_brush_mask_lib.js`）：`HEADER_H=10`（首行 y 16→26）、`GROUP_EXTRA=3`（组间在常规 4px 间距上额外让 3px）、`FIRST_ROW_Y=COL_TOP+HEADER_H`、`COL1_GROUPS=[8,3]`（比例 | Custom/Reset/Fill）、`COL2_GROUPS=[4,3,4,1]`（模式 | Clear/Undo/Invert | S±/O± | Pen）、`columnYs(groups, topY)` 生成逐项 y（列2 末项 Pen=277、底 295）。MIN 高 320→**340**。⚠ 存量 320 节点载入时 `clampNodeSize` 自动抬升 20px（工作流无需手工改）。
- 主扩展绘制：面板顶画列头 chip（RATIO 用 `getSfAccent()` 0.22 透明底 + accent 字，TOOLS 用 `th.surface` + `th.textDim`），两列一眼可分；组间分隔线用**同一个 `columnYs` 数组**推算（画在空隙中央）——绘制与命中共用 `buttonRect`，无第二套公式；`Color→Fill`、`BCol→Pen` 消除"两个色块按钮"歧义。
- 悬停说明：`HINTS`（id→中文文案）+ `controlHint`（比例按钮单独拼"裁剪框比例：…"）；`onMouseMove` 末尾用 `buttonRect` 扫按钮记录 `node._sfCEBHover`（**只在变化时** `setDirtyCanvas`，避免每帧重绘），`onDrawForeground` 在 `canvas.node_over === node` 且非拖拽/落笔时用提示替换底行右侧信息文本（截断逻辑复用）。离开节点靠 `node_over` 门控不显示，无需 onMouseLeave。
- 测试：lib test 更新 MIN 340 + 列头/分组常量 + `columnYs` 行位快照；smoke 更新 `computeSize [360,340]`、显示区 `offsetY 52→62`（nodeH 变大 → 全体显示区写死坐标 +10，同 §101 的教训）、新增列头/重命名/悬停断言；按钮点击坐标一律改为按 `_sfCEBCtrls.find(id)` 解析，不再写死行号。
- 被否决方案存档：B = 删列1、比例收进 `sf_popup` 弹窗（画布可再宽 40px、外观最简，但切比例多一步且测试坐标大改）；A+ = Crop 模式移到列1（列1 纯裁剪/列2 纯画笔，但三模式单选按钮被拆到两列）。

## 110. 缓存节点 source 惰性化 + source_key 轻量键：长视频命中不再整段求值、命中不重写（2026-09）

> ⚠ 已被 §111 取代：`source`/`signature` 已删除，缓存键收口为「名字 + required 非空 `source_key`」。本节保留为历史设计记录。

> 背景：用户 5090 上测试 2 分钟视频（1920 帧 @1280×896），只要把 `SFTrackDataCache`/`SFMaskCache` 接进链路，"读取视频"阶段就极慢——即使缓存命中、tracker 已被 lazy 跳过。根因：`source`(IMAGE) 不是 lazy，ComfyUI 执行节点前必先递归求值所有非 lazy 输入 → **整段视频被解码/常驻**（≈26 GB），只为在 `check_lazy_status` 里算首末帧 16×16 哈希；长视频每次 miss 还会整段重追踪 + 整文件重写 packed（1008² 工作网格 ≈ 127 KB/帧 → 1920 帧 ≈ 244 MB/对象），同名覆盖（`cache_store.cache_paths` 只由 name 决定）→ 观感即"频繁写盘 + 读取极慢"。

### 1. 方案：source 转 lazy + 新增 source_key（两节点同构）

- `source` 加 `lazy: True`；optional 末尾追加 `source_key`(STRING, forceInput, default "")。**非空时优先于 source 作为缓存键的源分量**（直接存 meta 的 `source` 字段，零 schema 变更）；旧工作流不接则行为与现状一致（仍求 source 算首末帧哈希）。
- `check_lazy_status` 分轮（`source` 有意**不作为显式形参**，否则被 Python 绑定后无法从 `**kwargs` 判别"已接线未求值"）：
  1. `track_data`/`masks` 未接线（不在 kwargs）→ `[]`（纯读取）；
  2. 无轻量键且 `"source" in kwargs and kwargs["source"] is None`（已接线未求值）→ 返回 `["source"]` 先补算指纹；
  3. `force` → 返回上游输入；
  4. 键 = `source_key or source_signature(source)`；命中 → `[]`，否则拉上游。
  - 有 `source_key` 时全程不请求 `source` → **命中判定不触发视频解码**；force 且无键时也先补 source，保证重算后 meta 仍写真实帧哈希。
- `execute`：`src = _resolve_source_key(source_key) or source_signature(source)`（key 非空时短路，不碰 source 张量）；新增守卫 `if not force and cache_hit(nm, sig, src): return (上游数据)`——上游因其他消费者仍被算出/force 误开时不再整文件重写。
- 共享：`cache_store.first_nonempty_text`（list/tuple 取首个非空 str）单源供 name_text/source_key；`cache_store.atomic_save_tensors`（pid+tid 临时名 + `os.replace`，不整块进内存）替换裸 `sf.save_file`；落盘后打印一行（名字/帧数/对象数/MB/耗时）便于诊断写盘频率。
- ⚠ **source_key 必须覆盖所有影响追踪输入的变量**（视频路径 + `frame_load_cap`/`skip_first_frames` + 追踪参数等）。只接文件名时改帧窗口会命中旧数据；长视频推荐"整段追踪一次 + 缓存放循环外 + 循环内 `SFTrackDataSlice`"（§87/§88），或把窗口文本一起串进 key。

### 2. 兼容与测试

- `source` 转 lazy 对旧工作流透明（未接 source_key 时语义/缓存文件不变）；接 source_key 后旧条目首次 miss 重写一次即稳定。两节点（`nodes/image/track_data_cache.py`、`nodes/mask/mask_cache.py`）同步，`cache_store` 共享。
- `tests/test_track_data_cache.py`·`test_mask_cache.py`：FakeSF 增加 `.tmp` 名归一化 + `save_paths` 记录；覆盖 source lazy/分轮请求、source_key 命中不请求 source、force+无键先拉 source、key 优先写入 meta、**命中带上游数据不重写守卫**、原子写（先写 .tmp、无残留、最终文件存在）。
- 本次踩坑：把 `source` 从 `check_lazy_status` 形参移除后，旧测试里第 4 个位置实参（原 source）会落到 `name_text`，与 `name_text=` 关键字调用冲突（TypeError）；测试统一改 keyword 传参。ComfyUI 侧始终 keyword 调用，无影响。

### 3. 长视频工作流接线示例（`[scail2]SCAIL2_手绘遮罩_分段追踪_原生` 实改）

- 顶层用 `VHS_LoadVideo.filename`（§79 补丁输出）+ `frame_count`（已加载帧数）+ 画幅宽/高，经 3 个 `SFAnyToString`（prefix `n=`/`w=`/`h=`，pad_digits=0）与 `SFTextConcatenate`（delimiter `|`）拼成 `rimp2.mp4|n=64|w=512|h=896`，接两个追踪子图新增的 promoted `source_key`（子图内 cache.source_key）；`source` 连线保留作回退（键非空时不求值）。
- 覆盖：换视频、换 `frame_load_cap`/`select_every_nth`（frame_count 变）、换分辨率（w/h 变）。**不覆盖** `skip_first_frames` 同 cap 变化（skip 是未连线 widget）——改 skip 用缓存实例的「刷新缓存」(force)，或把 skip 也接进键。
- 该工作流首次运行因键从帧哈希变文本而 miss 一次（重追踪 + 写新 meta），之后命中不再整段求值。

## 111. 缓存节点接口收口：source/signature 移除，source_key 成为唯一失效键（破坏性，2026-09）

> 背景：§110 引入 source_key 后仍保留 source（帧哈希，lazy）与 signature（文本）双键，"键源二选一/优先级/懒求值分轮"复杂度高，长视频下 source 帧哈希仍会逼迫整段求值。用户拍板收口：**缓存键 = 名字 + `source_key`**，`source`/`signature` 一并删除，接受破坏性（旧工作流缺 required 直接校验报错）。

### 1. 接口与语义

- `SFTrackDataCache` / `SFMaskCache` 的 `source_key`(STRING, forceInput) 移到 **required**（必接）；`source`(IMAGE, lazy)、`signature`(STRING) 删除。optional 只剩 lazy 上游（`track_data`/`masks`）与 `name_text`；widget 型输入（name/force）顺序不变，旧工作流 `widgets_values` 不受影响。
- `cache_store.required_source_key(raw)`：list/tuple 取首个非空并 strip；**空串（含接了线但值为 ""）抛 `ValueError`**，不静默退化。两节点共用同一错误文案。
- meta 字段：`signature`/`source` 合并为 `source_key`（新写只写该字段）；`cache_hit(base_dir, name, key)` 回退读旧 `source` 字段 → §111 前的缓存首次必然 miss、重写一次后稳定。
- 代码删减：`cache_store.source_signature` 与 `import hashlib` 删除；`check_lazy_status` 去掉 source 分轮（不再返回 `["source"]`），只剩「上游未接线→[] / force→拉上游 / 键命中→[] / 否则拉上游」；`execute` 去掉签名与 source 分支；两节点的 `_resolve_source_key` 包装删除，改用共享 `required_source_key`。

### 2. 迁移指引（旧工作流会报 "Required input is missing"）

- **视频源**：`VHS_LoadVideo.filename`（§79 补丁输出）+ `frame_count`（已加载帧数）+ 画幅宽/高 → `SFAnyToString`（prefix `n=`/`w=`/`h=`，pad_digits=0）→ `SFTextConcatenate`（delimiter `|`），示例 `rimp2.mp4|n=64|w=512|h=896`（§110.3 的实改范式）。
- **图片源**：没有自动帧哈希可用，需自备文本身份（图片路径/文件名、点选坐标文本、手填 `PrimitiveString` 等）。`skip_first_frames` 同 cap 变化等未进键的改动，用实例「刷新缓存」(force) 重算。
- **提示词仍留在缓存名**（`name_text` 链）而非塞进 key：名字 = 存储槽（同视频多对象各自一份），key = 有效性指纹；只把提示词放 key 会让多对象共用同一文件、切换时互相覆盖。
- `source_key` 是唯一自动失效信号：不接或空串都会报错，避免"以为有键其实只按名字命中"的静默旧缓存。

### 3. 测试

- 两测试文件：schema（`source_key` 在 required 且 forceInput、无 `source`/`signature`）、`required_source_key` 空串报错、`check_lazy_status`/`execute` 空 key 报错、meta 写 `source_key`、旧 meta `source` 字段兼容读、命中守卫（带上游数据且键命中不重写）、原子写回归；§110 的 source 分轮用例移除。

## 112. 组合节点源图翻转/旋转工具（第三列 ORIENT，纯前端落盘改写，2026-09）

> 背景：SFImageCropExpandBrushMask 需要"翻转图片和旋转图片"的编辑工具。用户拍板：作用对象=**源图整体联动**（裁剪框与笔触随图变换、笔触仍粘在画面内容上）；操作集=水平/垂直翻转 + 逆/顺时针 90°（不做任意角度——裁剪框将不再轴对齐）；位置=**新增第三列 ORIENT**（只加宽不加高）；实现=**纯前端落盘改写**（零后端改动、无需重启容器）。

- 布局（`sf_crop_expand_brush_mask_lib.js`）：`ORIENT_COL=["flipH","flipV","rotL","rotR"]`、`COL3_GROUPS=[4]`（单组无分隔线）、`ORIENT_COL_X=TOOL_COL_X+toolColW+toolColGap=90`、`EXTRA_LEFT=40→80`、`MIN_NODE_WIDTH=320+80=400`（高仍 340，列3 4 项底 110 非瓶颈）。存量 360 宽节点载入 `clampNodeSize` 自动抬宽 40px。
- 变换纯函数 `orientState(st, op)`（返回状态补丁，不改入参）：裁剪框连续坐标 `flipH: x'=W-x-w`／`flipV: y'=H-y-h`／`rotL: (x,y,w,h)'=(y, W-x-w, h, w)`／`rotR: (H-y-h, x, h, w)`；笔触按**整型像素索引**（与后端 `rasterize_strokes` 口径一致）：镜像 `W-1-x`/`H-1-y`，旋转 `rotL: (y, W-1-x)`／`rotR: (H-1-y, x)`。旋转后宽高互换 → `aspect_ratio` 复位 `"free"`（翻转不动）。`flipH²`/`rotL∘rotR` 恒等还原（lib test 断言）。
- 源图重绘与上传（`sf_crop_source.js` 新增，三节点共享源图链路内）：`renderOrientedImage(img, op)` 1:1 离屏 canvas（旋转交换画布宽高；canvas 变换的屏幕映射与像素索引公式等价——如 rotR 的 `translate+rotate(π/2)` 把源像素中心映射到 `(H-y, x)`，floor 后即 `H-1-y`）；`orientSource(node, op, cfg)` 校验源图已就绪（无源图/未解码/上传失败 → `sfToast` 返回 null），PNG dataURL 走既有 `CropAPI.uploadSrc` 落盘为**新文件**（不删旧文件：与换图行为一致，且复制节点共享的 `src_path` 不被就地改写；代价是每次操作一份新源图）。
- 主扩展 `applyOrientation`：`_sfCEBOrientBusy` 串行化连续点击；patch 在**上传成功后**按当时状态计算（等待期内的框/笔触编辑不被旧快照覆盖），并以 `src_path` 是否变化判定等待期换图 → 丢弃本次变换；落定前 `cancelPoly`/`cancelSamMode` + 清 `_sfCEBDrag`/`_sfCEBDrawing`/`_sfCEBCur`；`setState(remap + 新 src_path)` 后 `new Image()` 载入 dataURL 替换预览。`filename` 输出与节点输出始终一致（文件本身就是变换后的图）。
- 被否决方案：**状态位 `src_rot/flip` + 后端 `np.rot90/flip`**（原文件不动、无 8-bit 重编码，但要改后端/重启容器，且 `filename` 指向未变换的原文件，与节点输出不一致）。当前方案的代价：浏览器 canvas 重编码（8-bit）与孤儿文件累积，与既有上传链路同性质。
- 未加键盘快捷键：前端默认单键 `r/w/n/m/a/./p/v/h` 已占用（§92 核查表），字母键大小写不敏感也无法用 Shift 区分。
- 测试：lib test 更新三列布局/MIN/`columnYs`/显示坐标系（`extraLeft=80`）+ `orientState` 四变换期望值/逆变换恒等/纯函数/非法 op；smoke 更新控件 29 项、`computeSize [400,340]`、ORIENT 列头/列3 几何，新增 rotR（画布宽高交换 + rotate π/2 + drawImage 全尺寸 + 上传 + 状态重映射）、flipH、无源图不上传断言。⚠ 旧测试里按旧布局写死的显示区坐标全部 +40 偏移（§109 同款教训），其中 `[100,70]` 这类点还落进新列按钮命中区（会误触发旋转）——显示区交互坐标一律按 metrics/控件几何解析。
