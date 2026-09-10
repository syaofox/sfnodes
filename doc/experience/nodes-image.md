# 经验归档：图片 / 遮罩 / latent 节点（§8、§9、§11、§12、§13、§22、§34、§35、§36、§37、§44、§45）

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

- `nodes/image/pause_image.py`：节点（快照/continue 读回/无 IS_CHANGED）+ `_json_safe`。
- `nodes/image/preview_routes.py`：save/prepare 路由 + `_safe_prefix`/`_sanitize_segment`/`_decode_image`/`_build_pnginfo`/`_metadata_disabled`。
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

- `nodes/mask/pause_mask.py`：节点（快照 L 模式/读回 [1,H,W] 防御/无 IS_CHANGED）+ `_json_safe`。
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
- **布局改版（2026-09，用户草图）**：比例按钮改**画布区左侧竖列**（`RATIO_PRESETS_COL`，Free 置顶 + 7 预设 + Custom/Reset/Color 收尾，节点顶直通画布区底，`LAYOUT.ratioColW/ratioColGap` 由图片区让宽）；Load Image/Browse 移**底行**与信息文本同排（右对齐到输出槽区前；按钮 y 标记 `BOTTOM_Y="bottom"` 经 `buttonRect` 运行时解析——节点高度可变，坐标不能在 buildButtons 固定）。**MIN 收敛**（460→480→360/300）：底行宽度瓶颈靠信息文本 `measureText` 溢出截断 "…" 解除（按钮右缘 135 + 最小文本窗 120），高度由竖列 11 项驱动。

## 44. 画布节点拖拽缩小外溢：computeSize 包装钳最小尺寸（2026-09）

> 背景：SFImageCropExpand 用户手动把节点拖到 `MIN_NODE_WIDTH/HEIGHT` 以下，固定像素布局的按钮行外溢出节点。`clampNodeSize` 只在 onNodeCreated/onConfigure 钩一次，管不住后续拖拽。

- **双端拖拽 resize 的最小值都取自 `node.computeSize()`**（前端包 1.51.10 实测 onDrag 处理：`let l=node.computeSize(); c.width<l[0]&&(c.width=l[0]); c.height<l[1]&&(c.height=l[1]); node.setSize(c.size)`；legacy litegraph 同款）——**包装 `nodeType.prototype.computeSize` 返回 `max(原值, MIN)` 一处改动同时钳住两条路径**，且只抬不降（`expandToFitContent`/初始 sizing 均安全）。
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
- **实测无效复修（2026-09）**：初版 window 通道在用户环境无响应。排查确认两点：① 快捷键代码当时未进 HEAD（工作区未提交，部署侧无此代码）；② 发现官方通道——画布 `processKey` 把 keydown 分发给选中节点的 `onKeyDown`（core `addNodeKeyHandler` 再包一层，`=== false` 表已处理，链式兼容）。改为双通道：`nodeType.prototype.onKeyDown` 主办 + window 冒泡兜底（焦点在 body 时画布收不到），同物理按键经 `e.timeStamp` 去重（多选同戳亦然，首个节点全量调整后其余跳过）。smoke 加官方通道/同戳/跨通道去重/放行四用例。**教训：先查引擎有无官方钩子（`processKey`/`addNode*Handler`），再写全局监听；未提交代码先查部署差再查 bug**。
