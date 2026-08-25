# 经验归档：图片 / 遮罩 / latent 节点（§8、§9、§11、§12、§13、§22、§34）

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
