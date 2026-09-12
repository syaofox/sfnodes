# 经验归档：横切模式与修复批次（§3、§4、§17、§26、§27、§39、§40、§41、§43、§49、§50、§52、§53、§54、§55）

> 全局章节号 §N 与拆分前的 experience.md 一致；跨节/跨文件引用一律写 §N，映射见 [README.md](README.md)。版本时效说明见 README。

## 3. 静态检查脚本经验（AST 对比踩坑）

用 Python AST 做"前后端一致性/结构对比"验证时（如对比注册字典、检查节点 INPUT_TYPES），易踩两个坑：

1. **`ast.unparse` 输出的是单引号字面量**：`ast.unparse(v)` 生成的字符串（如 `'interrogate'`、`'CLIP'`）统一用单引号包裹，与手写断言中的双引号字面量（`"interrogate"`）直接比较会**误判不一致**。取值应优先用 `ast.literal_eval(node)`（常量），或按节点类型提取：`Constant.value` / `Name.id`（变量引用）/ `List.elts`。不要拿 unparse 文本与手写字面量做相等比较。
2. **`ast.literal_eval` 遇到变量引用会抛 `ValueError: malformed node or string`**：默认值引用模块常量的表达式（如 `"default": KREA2_INSTRUCT_SYSTEM`）无法直接求值。需分两步：先单独提取被引用的常量（`ast.literal_eval`），再遍历映射，遇 `Name` 节点取其 `id` 后查表替换。

真实案例：比对 `KREA2_PRESETS` 前后端一致性时，`ast.unparse` 残留的单引号让"文本一致"误判为 false；`ast.literal_eval` 直接解析含 `KREA2_INSTRUCT_SYSTEM` 引用的字典抛 ValueError。两处均为检查脚本问题，非代码问题——**先怀疑检查脚本，再怀疑被检查的代码**。

---

## 4. 动态 combo 校验与工作流绑定状态（widget 数据载体）

> 背景：SFTextPreset 工作流绑定文本预设节点（2026-08），落地为 `nodes/text/text_preset.py` + `web/sf_text_preset.js`。需求：预设绑定当前工作流，其他工作流添加此节点是全新空预设。**2026-09 已改为全局持久化（user/sfnodes/text_presets.json 真源 + presets_json 回退兼容），见 nodes-text.md §42；本节"数据载体"模式与 VALIDATE_INPUTS 经验仍适用。**

### 1. "状态绑定工作流"的标准模式：数据存 widget 值（数据载体）

- **所有 widget 值（含 `display: hidden`）都会随 workflow JSON 序列化**（前端保存/加载/复制/导出嵌入自动跟随，`serialize = false` 可排除）。把预设等状态数据以 JSON 字符串存进隐藏 STRING widget → 预设天然"绑定"当前工作流：保存即持久化、复制/导入跟随、**新工作流添加节点用 INPUT_TYPES 默认值 = 全新状态**，无需后端存储/路由（早期 `TextDropdown` 的 `options_json` 隐藏 widget 是同类先例，但它叠加了全局 API 轮询，做全局共享才需要；该节点已由 SFValueDropdown 取代）。
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

## 17. 复刻节点去重：sf_common.js / disk_state.py 公共模块收敛与踩坑

> 背景：多个复刻 Pixaroma 的节点（crop/inpaint/load_image/outpaint/dropdown/pause 三件套/find_replace/prompt_reader）各自内联了一份 pixaroma js/shared/ 的小工具（isGraphLoading / isVueNodes / applyAdaptiveCanvasOnly / installCanvasZoomPassthrough / sfApiUrl / buildSourceURL / getUpstreamImageURL / installPasteHandler），后端 crop.py 与 inpaint_editor.py 也各持一份 `_safe_join`/`_sanitize_id`/`_decode_image`（2026-08 收敛）。教训：**复制后语义分叉是 bug 温床**——crop 的 `_safe_join` join 到子目录本身、inpaint 的 join 到 input 根，crop 版在路由返回 `sfnodes_crop/` 前缀路径时双重拼接（`input/sfnodes_crop/sfnodes_crop/...`）导致粘贴上传执行输出白图。

### 1. 收敛产物（新节点先查这里）

- **`web/sf_common.js`**（纯工具模块，使用者 import）：`sfApiUrl`（api.apiURL 包装）/ `isVueNodes` / `applyAdaptiveCanvasOnly` / `isGraphLoading`（**全局单例守卫**：模块顶层自动包装 `app.loadGraphData` + 300ms 尾窗，幂等，勿再各自包装）/ `installCanvasZoomPassthrough`（**统一为增强版**：滚动容器穿透检测，无滚动容器时行为等价简单版）/ `parseAnnotatedImageValue` / `buildSourceURL`（cache-buster 进 ROUTE 不进 RESULT——托管部署 token 顺序）/ `getUpstreamImageURL(node, cachedUrl)`（cachedUrl 参数化：crop 传 `node._sfCropSourceURL`，inpaint 传 `node._sfInpaintSourceURL`）/ `installPasteHandler({comfyClass, hook, onPasteImage, allowPaste})`（findActiveNode 4 源查找统一，hook 检查保留原版防御语义，allowPaste 承接 inpaint 的"编辑器开着"守卫）。
- **`sf_utils/disk_state.py`**：`safe_join(root_dir, rel, strip_prefix)`（解析根参数化：crop 传子目录 + 剥 `sfnodes_crop/` 前缀；inpaint 传 input 根不剥）/ `sanitize_id` / `decode_image`。节点文件留薄包装保持调用点不变。
- **CSS 类名前缀**：编辑器框架原 `pxf-`（Pixaroma Framework 缩写）→ `sf-px-`（321 处 + `_pxfSliderFillInit`/`_pxfUpdateFill` 全局变量）。**与源插件共存时全局 CSS/变量名冲突是真实风险**（后加载覆盖先加载，两边样式都乱）。

### 2. 去重/重构踩坑（自动化批量改动的三大陷阱）

1. **独立语句的包装块按函数名删除会漏**：`isGraphLoading` 是函数，但其配套的 `let _sfXxxGraphLoading = false; if (app && app.loadGraphData && !app._sfXxxGraphLoadWrapped) {...}` 是**顶层独立语句**——按函数名删除只删了函数，包装块残留形成双包装（行为无害但冗余，且死变量误导）。清理时要连注释块一起扫。
2. **文件已有某模块 import 时，脚本补 import 可能跳过**：去重脚本检测"已 import sf_common"就跳过补符号 → 新引用的函数（如 `isGraphLoading`）未导入 → 运行时 ReferenceError。**若该引用在 try/catch 包裹的路径内（如 onConnectionsChange 的 `try { refreshWiredState(this) } catch {}`），错误被静默吞掉**，表现为"某交互功能失效"而非报错——极难排查。改完必须逐个文件核对 import 符号清单。
3. **`node --check` 默认按 CJS 解析**：`import {` 多行块中间被插进 `export {...}` 这种 ESM 结构错误，CJS parse 不报（`import` 在 CJS 是普通标识符），测试却能在 stageJs 加载时炸或更隐蔽地错乱。**统一用 `node --input-type=module --check < file` 验证 web/ 全部 JS**。
4. edit 删除大块函数时 oldString/newString 边界易丢行（如 `app.registerExtension({` 被吞）——CJS 模式下顶层 `name: "..."` 是合法 label、`async beforeRegisterNodeDef(...)` 却非法，CJS check 有时能兜住；但 ESM check 才是权威。

### 3. 磁盘源预览缓存：后端必须向 executed 事件输出源帧

- SFImageCrop/SFInpaintCrop 的磁盘源路径（粘贴/拖放/编辑器 Load Image，无上游接线）执行时，后端原本**只在上游 tensor 存在时**输出 `sf_crop_source`/`sf_inpaint_source` ui_payload → 磁盘路径执行后前端 executed 事件收不到帧 → `_sfCropSourceURL` 不更新 → **节点预览停留在旧图（运行结果正确但预览错）**。
- 修复：磁盘路径执行也输出源帧（`{filename, subfolder: "sfnodes_crop"/"sfnodes_inpaint", type: "input"}`），前端 onExec 既有逻辑自动刷新缓存。
- 前端双保险：jsonSync 检测 `src_path` 变化立即同步缓存（**inpaint 无 crop 的 500ms pollInterval 轮询兜底，jsonSync 内要主动 refreshSourcePreview，加载路径用 isGraphLoading() 门控**）。编辑器 Load Image 只更新内存 `_pendingSrcDataURL`，保存时才上传 + 写 src_path。

### 4. 编辑器工具栏语义：Reset ≠ Clear

- `createCanvasToolbar` 有独立 **Clear**（清空画布）与 **Reset to Default**（重置为默认）按钮；本项目隐藏 Clear（`showClear: false`），但 onReset 误把 `this.img = null`（把 Reset 当 Clear 用）→ 点击 Reset 清空已加载图片。
- 修复：onReset 委托 `_resetCrop()`（保留源图、重置为全图裁切、free 对齐、输出尺寸跟随图片）。复刻时注意按钮语义与源实现一一对应，隐藏了 Clear 不等于 Clear 的行为并进 Reset。

---

## 26. 前端架构治理（2026-08）：工具收敛 / 弹层三件套 / 纯模块边界

> 背景：对 `web/` 全量架构评审（102 文件/4.3 万行）后的治理改动。评审结论：架构总体合理（枢纽-辐射依赖、全局钩子组合式补丁、复用纪律、测试覆盖均为优），本次只治理发现的低/中危问题，不重构双渲染器与超大文件（规范已并入 AGENTS.md「Code Style」）。

### 1. 通用工具收敛到 sf_common.js（消除跨家族依赖与副本分叉）

- **`escapeHtml` / `downloadDataURL` / `copyText` 单一实现入 `sf_common.js`**：
  - `downloadDataURL` 从 `sf_crop_framework.js` 迁入（showSaveFilePicker 优先 + `<a download>` 回退，AbortError 豁免）；`copyText` 从 `sf_workflows_ui.js` 迁入（clipboard + execCommand 双回退）；`escapeHtml` 取 `sf_lora_stack_info.js` 的五字符全集（`& < > " '`，innerHTML 注入最安全）。
  - **迁移模式用 re-export 保持调用方零改动**：`sf_crop_framework.js` / `sf_workflows_ui.js` 改为 `export { x } from "./sf_common.js";`（项目已有先例：`sf_dropdown_ui.js` re-export `isVueNodes`/`applyAdaptiveCanvasOnly`）。调用方（sf_crop.js / sf_inpaint.js / sf_workflows.js / sf_lora_stack_info.js）一行不改。
- **刻意保留的两处本地 `escapeHtml`（不是重复，是边界）**：
  - `sf_find_replace_lib.js`：纯模块公共 API，**`tests/test_find_replace_js.js:176` 断言锁定转义集合**（引号不转义）——删了测试就红。
  - `sf_markdown.js`：无 app 依赖的纯渲染模块，markdown 前置转义语义独立。
- **纯模块边界（新规范）**：`*_lib.js` / `*_core.js` / `sf_markdown.js` 等纯逻辑模块**不得 import `sf_common.js`**（它依赖 `/scripts/app.js`，会破坏 Node 测试拷贝能力）；这类模块的公共函数共享应放无依赖模块或独立小模块。DOM 层模块（直接依赖 app 的）可自由用 sf_common。

### 2. 公共弹层三件套 `web/sf_popup.js`（新弹层优先使用）

- 13+ 个浮动弹层各自重复"外部点击/Esc/滚轮三关闭 + 定位钳位"，且踩坑记录在 §15.4（popup 缩放跟随定位）与 §19（确认框豁免宿主面板捕获监听）。收敛为：
  - `attachPopupDismiss(overlay, { onClose, exempt })`：外部 pointerdown / Esc / wheel 三关闭，capture 阶段 document 监听，`exempt(e)` 豁免（面板风确认框场景），返回幂等 detach。
  - `clampToViewport(el, { margin, scale })`：viewport 四向钳位，`scale` 折算边距（position:fixed 弹层在 canvas 缩放下 root font-size 已缩放）。
- **验证迁移**：`text_replace.js` 的 marker 菜单（原手写三关闭 + Math.min/max 钳位）改为调用 sf_popup，行为等价。**存量 12 个弹层不强制迁移**（dropdown 弹层与分类弹层耦合深、lora 面板含 dirty 语义），新弹层优先用 sf_popup。
- 测试：`tests/test_popup_smoke.js`（mock document：三关闭 / 内部点击豁免 / exempt / detach 幂等 / 四向钳位 / scale 折算）。

### 3. 注册规范固化（check_web_imports.py 扩展）

- `tests/check_web_imports.py` 从 17 模块扩到全部 ~46 个多模块/共享文件，并新增三条文件级规则：
  1. 相对导入（含副作用 `import "./x.js"`）目标文件必须存在；
  2. 含 `app.registerExtension(` 的文件必须**直接** import `/scripts/app.js`（不允许依赖传递；`sf_regional_lora.js` 曾用 `../../scripts/app.js` 相对路径——碰巧在 `/extensions/<name>/` 挂载下能解析，是脆弱依赖，已统一为绝对路径）；
  3. 扩展注册名必须 `sfnodes.*` 前缀（顺带修复三处既有不一致：`Sfnodes.PromptReader` 首字母笔误、`SFRegionalLoRA.editor` → `sfnodes.RegionalLoRA.editor`、`inpaint-cropandstitch.showcontrol` → `sfnodes.showcontrol` 历史命名空间）。
- 扩展名仅作调试标识，不进工作流文件，改名无行为影响。

### 4. 明确不做的治理项（规范替代重构）

- **双渲染器（Classic/Vue）不抽象**：34 处 `isVueNodes()` 分支语义各异（槽位替换 / shallowReactive / DOM nudge 各不同），强行抽象收益不确定；新增分支受控、常见适配优先入 sf_common。
- **超大文件不强制拆分**（13 个 >1000 行）：纯搬移改 import 图，收益低于风险；新代码优先进已有拆分模块或新建模块。
- **旧文件不重命名**：27 个无 `sf_` 前缀文件 + `DisplayText.js`/`SFLogicSwitch.js` PascalCase 为历史遗留（git 历史/用户记忆成本），保持现状。

---

## 27. 2026-08 健壮性修复批次：表达式防御 / ReDoS 交替型 / 路径净化 / 双端镜像补缺

> 背景：全量代码审查（4 子代理 + 主代理逐条复核）后的一轮修复批次。涉及 `nodes/`、`sf_utils/`、`web/` 共 20 余个文件，全部为 bug 修复、无新节点、无注册字典改动。配套测试：`tests/test_simple_math.py` / `test_logic.py` / `test_downloader.py` / `test_seed.py` / `test_disk_state.py`（新建），`test_find_replace.py` / `test_outpaint_js.js` / `test_image_resize_js.js`（追加用例）。

### 1. `ast.Constant.n` 版本陷阱（Python 3.13 deprecated / 3.14 removed）——simple_math.py

- `ast.Constant.n` 是 `value` 的旧别名：**3.13 起访问抛 DeprecationWarning，3.14 起属性移除**（`_fields = ('value', 'kind')`，实测 3.14.x `node.n` AttributeError，当时版本）。修复一律写 `node.value`（3.8+ 全版本存在）。
- **开发环境与容器的版本差**：本机 python3 是 3.14.x（`__pycache__` 的 cpython-314 是**本机**产物，不代表容器，版本号会随升级变化）；comfyui-docker 容器实测 Python 3.12.x（当时版本，以容器内 `python3 --version` 为准）。同一段代码在两个版本可能行为不同（3.14 上 `ast.parse('1+2*3')` 直接 AttributeError，3.12 正常）——涉及 ast/语法 API 的改动要以**容器版本**为行为基准写测试断言。
- SimpleMath 表达式求值的完整崩溃面（修复前）：语法错误 SyntaxError、`1/0` ZeroDivisionError、未注册运算符（`^`/`@`）KeyError、**字符串常量 `"abc"` 与字符串变量 → `math.isnan(str)` TypeError**、`0**-1` ZeroDivisionError。修复：整段 eval_ 包 try/except（SyntaxError/ZeroDivisionError/KeyError/TypeError/AttributeError）回退 `(0, 0.0)` + warning；`isnan` 前 `isinstance(result, (int, float))` 校验。
- 教训：`__pycache__` 的 `cpython-3xx` 目录是**本机解释器**版本而非运行环境；判断运行版本必须问用户/查容器。

### 2. ReDoS 启发式补交替型（(a|aa)+ 家族）——find_replace.py + sf_find_replace_lib.js + regex_extract.py

- 原 `_is_catastrophic_regex` 只覆盖嵌套无界量词 `(a+)+ (a*)* (.*)*`——**漏掉交替型指数回溯**：`(a|aa)+`、`(a|a?)+`、`(a|)+`（无嵌套量词，同样指数级）。
- 新增 `_alternation_overlap_risk`：组内顶层 `|` 分出 ≥2 分支、任意两分支**首字符集合重叠**、且组后紧跟无界量词（`*`/`+`/`{n,}`，lazy 变体 `+?`/`*?` 以 `+`/`*` 开头天然命中）→ 危险。首字符集合解析：字面字符 / 字符类（含否定与转义类 → ANY）/ `.` → ANY / 空分支 → EMPTY（与任何分支重叠）/ 断言（`^` `$` `\b`）跳过 / 嵌套组 → 保守跳过该组。
- **判别精度**：`(a|b)+` 与 `(x|aa|b)+`（分支首字符互斥）**不命中**——线性安全；`(a|a|a)+`、`(a[0-9]|aa)+` 命中。测试 13 例双端（Python + JS）同用例。
- **JS 镜像必须 1:1 同步**（`sf_find_replace_lib.js` 的 `alternationOverlapRisk`）：预览每次按键重算，与 Python 服务端行为不一致时预览对运行说谎。
- **内置预设跳过检查**：regex_extract 的 12 个内置预设是项目自维护正则，其中"提取邮箱" `[\w.+-]+@[\w-]+(?:\.[\w-]+)+` 会被**嵌套量词检测**保守误报（`(?:\.\w+)+` 组体以固定前缀开头其实线性安全）——接入 ReDoS 检查时预设原样跳过（`is_preset_untouched`），只检查用户改动/自定义的模式。

### 3. 文件名净化收敛：`disk_state.sanitize_filename`（H3/H5 共用）

- 新公共函数 `sanitize_filename(raw, fallback)`：保留 Unicode/空格，拒绝绝对路径/`..`/`.`/空段（**在任何清洗之前检查**——清洗会把 `..` 吃掉）、路径分隔符拍平为 `_`、Windows 非法字符替换、边沿剥离、隐藏文件拒绝、保留设备名加 `_` 后缀、截断 128。hyperlora 的 `char_name`（自由 STRING → 路径穿越写 `models/hyper_lora/chars/`）与 SFExtractLUT 的 `filename`（→ `user/sfnodes/lut/`）共用。
- 教训：**节点里"自由 STRING → 文件路径"是路径穿越高危点**（hyperlora/lut 两处原实现都直接 `os.path.join`）；新写这类节点必须净化。

### 4. cropstitch 多帧必崩 + 设备不匹配（nodes/inpaint/cropstitch.py）

- 顶部/底部镜像填充用了**整批 `image`** 写进单帧 `new_image`（batch>1 形状失配 RuntimeError）——必须用 `one_image`。
- `torch.zeros`/`torch.ones` 画布**未指定 device**（默认 CPU），CUDA 输入在赋值点设备不匹配崩——一律 `device=one_image.device`。
- 教训：局部新建张量必须继承输入张量的 device；镜像填充/拼接只允许单帧语义时明确用单帧。

### 5. outpaint 双端镜像补缺：`fitPad`（sf_outpaint_core.js ≡ outpaint.py::_fit_pad）

- Python `_fit_pad` 在分配画布**之前**把相对两边 pad 收缩到 `extent + pad <= 16384`（防极端比例 1:1000 / sides 四边全开 8192 先分配数 GB 再 clamp）；JS `finalSize` 此前只对最终像素 clamp——极端 pad 下预览/上报尺寸对真实输出说谎。
- JS 新增 `fitPad(padA, padB, extent)`（`room = max(0, 16384-extent)`、按比例拆分、`Math.floor(padA*room/total)` 镜像 Python `//`），`finalSize` 在 pad 应用前对 (left,right)/(top,bottom) 各调一次。`MAX_DIM = 16384` 常量导出、`clampDims` 复用。
- **误报确认（防未来误修）**：曾有报告称 cover 模式预览绕过 8× 上限（JS `nw=tw` vs Python `factor=min(factor,8)`）——复核 Python `_apply_cover`：8× cap 只限制**内容放大倍数**（`scaled = orig×min(factor,8)` 后仍 **crop 到目标 `(tw,th)`**，输出尺寸恒为 tw/th），JS 报 tw/th 与真实输出一致，**不是 bug**。修复反而会引入不一致。修"镜像不一致"必须先确认 Python 的**最终输出尺寸**而非中间量。

### 6. innerHTML 注入面：image_browser.js / multi_lora_tree.js

- 目录/文件夹名来自**用户可写文件系统**，此前直接拼 `innerHTML`（面包屑 `data-folder="${accumulated}"` 双引号未转义 + 文本未转义；multi_lora_tree 文件夹项同款）——含 `<` 或 `"` 的名字可注入 HTML/破坏属性。统一改 `escapeHtml`（`sf_common.js`，转五字符含引号）。
- 教训：文件系统名（目录/文件名/路径）是不可信输入，任何拼进 HTML 的位置都要转义，属性值上下文必须转引号。

### 7. `mask_process` 的 `squeeze(0).unsqueeze(-1)`（mask_utils.py）

- 原实现 `squeeze(0)` 在 **2D 且 H==1** 时把 `[1,W]` 错 squeeze 成 1D；B>1 时 squeeze 不动、输出 `[B,H,W,1]`（与单帧 `[H,W,1]` 维度数不一致）。改为 `_mask_to_wh1` 按 `dim()` 显式分派（2D→`[H,W,1]`、`[1,H,W]`→`[H,W,1]`、`[B,H,W]`B>1→`[B,H,W,1]`），调用方形状契约不变。

### 8. 其它一批修复（摘要）

- **downloader.py**：`requests.get` 移入 try + `timeout=(10,120)` + `raise_for_status`；写 `.part` 临时文件 + `os.replace` 原子替换；失败 `finally` 删除半成品（否则下次 `is_file()` 误判"已下载"用坏文件加载）。**模型下载统一到 huggingface_hub（2026-08 方案 A）**：HF resolve URL（`https://huggingface.co/<repo>/resolve/<rev>/<path>`，含子目录如 `antelopev2/1k3d68.onnx`）→ `parse_hf_url` 解析 → `hf_hub_download`（官方缓存/etag 校验/断点续传）→ `shutil.copy2` 到约定路径 `save_loc/model_name`（落盘契约不变，调用方零改动）；`requests` 仅兜底非 HF URL（当前无使用方）。**不用 local_dir**：保留子目录结构会破坏 `save_loc/filename` 拼接（`antelopev2/xxx.onnx` → `save_loc/antelopev2/`），且 `local_dir_use_symlinks` 新旧 huggingface_hub 签名不同（rfmsr 踩过）——缓存+复制多占一份磁盘（HF 缓存 `~/.cache/huggingface/hub/`，可安全清理，不影响已落盘的项目文件）。**HF 失败不回退 requests**（同一网络下 requests 也大概率失败，静默回退难排查）。rfmsr 保持自身 `hf_hub_download`/`snapshot_download`（repo 子目录快照语义 + local_dir，测试锁定不动）。
- **logic.py**：SFMathInt divide/modulo 除零回退 0 + 告警（b 默认 0）；power 负指数/`0**-1` 兜底。SFBatchAnything 张量分支改 `and` 双端判断（None 直通由末尾兜底），末尾 `try: any_1+any_2 except TypeError: return ([any_1,any_2],)`。
- **lut.py**：SFLoadLUT.IS_CHANGED 文件缺失 `float("NaN")` → `f"missing:{file_name}"`；SFExtractLUT 文件名净化 + 强制 `.cube`。
- **replace.py / prompt_batcher.py**：`refresh`/`load_always` 的 `float("NaN")` → `str(time.time_ns())`（NaN 折叠祖先缓存反模式）；prompt_batcher 的 IS_CHANGED 聚合目录 txt `(name, mtime)`（修"新增文件不感知"陈旧）；空目录/无匹配 `raise` → 空列表降级；`_resolve_folder` 加 realpath 二次校验（防 symlink 逃逸）。
- **nodes/face/analysis.py**：两处 `torch.where(mask)` 判空兜底（mask_process 腐蚀/裁剪清空遮罩时 `x.min()` 崩）——照抄 `landmarks is None` 的全零占位模式保持 batch 对齐。
- **seed.py**：-2/-3 继承语义实现（实例属性 `_sf_last_seed` 跨 run 保留，首次随机起点；IS_CHANGED 每次随机保证重跑）。
- **image_convert.py**：CAS 补 `_min_tensors`/`_max_tensors`（原 `min_`/`max_` 未定义，开锐化必 NameError）。
- **lora_routes.py / lora_presets.py / workflow_routes.py**：`asyncio.get_event_loop()` → `get_running_loop()`（3.12 弃用告警、3.14 移除），闭包内冗余 `import asyncio` 删除；`.tmp` 临时名带 `threading.get_ident()`（并发写同文件互覆盖）；预设 POST/DELETE 加 `asyncio.Lock`。
- **requirements.txt**：补 `requests`、`typing_extensions`（代码已在用但未声明）。
- **自定义输入框键盘/滚轮（2026-08 快捷键拦截修复批次）**：① 输入框 keydown 必须放行 `ctrl/meta/alt` 组合键（否则焦点在输入框时 Ctrl+S 漏成浏览器"保存网页"——sf_prompt_list/prompt_stack/pause_text/prompt_tags/find_replace/crop_panel/lora_stack_*/load_image_ui/workflows_ui/prompt_tags_editor 等 11+ 处统一修复）；② **sf 的 DOM widget 输入框挂载在 canvas DOM 层，不在 Vue 新版 TransformPane 的 @wheel.capture 转发路径内——ComfyUI 画布缩放/滚动在编辑框上完全失效（连 Ctrl+滚轮都不缩放）**。修复：`sf_common.installWheelZoomPassthrough(el)` 挂输入框——Ctrl/⌘+滚轮总转发 canvas 缩放；普通滚轮在输入框可滚动（scrollHeight>clientHeight）时滚动文本、否则转发缩放（对齐 ComfyUI 原生输入框行为）。

---

## 39. COLOR 输入类型被 Vue 前端内置 widget 收编（2026-09）

> 背景：SF Image Resize Plus 的 `pad_color` 默认显示 "0,0,0"、点一次取色器后才变 "#000000"+色块。同病灶：SFMaskFill 的 `fill_color`。涉及 `web/sf_color_picker.js`、`nodes/image/scale.py`、`nodes/mask/masks.py`。

### 1. 机制：内置 COLOR widget 覆盖自定义注册

- 新版 Vue 前端 `widgetRegistry` 已注册 `'color'`（别名 `COLOR`）→ 内置 `ColorWidget`（左侧画 hex 文本、右侧画色块，点击弹原生取色器）。`widgetStore` 合并时 `new Map([...customWidgets, ...coreWidgets])` —— **重复键后写胜出，core 覆盖同名 `getCustomWidgets` 自定义注册**，`sf_color_picker.js` 的自定义 COLOR widget（色块 + RGB 文本）在新前端实际已死代码。
- 内置 ColorWidget 的 value 必须是 **hex 字符串**：数组 `[0,0,0]` 作默认值时 `fillStyle` 无效（无色块）、`fillText` 直出 "0,0,0"；用户点一次取色器后 value 变 hex 才正常显示。**教训：COLOR 输入的 default 一律写 hex 字符串，不写数组。**
- 自定义 widget 的 type 是大写 `"COLOR"`，内置是小写 `"color"`——按 type 查找 widget 的逻辑（如旧 serialize hack）必须大小写不敏感匹配。

### 2. 修复形态

- 后端 default 改 hex（`pad_color="#000000"` / `fill_color="#ffffff"`）；execute 对 hex 字符串与 `[r,g,b]` 数组双兼容（`_parse_fill_color` / scale.py 内联解析），旧工作流数组值后端照常工作。
- 前端 `sf_color_picker.js`：nodeCreated + loadedGraphNode + configure 包装三时序把 widget 值归一为 hex（`toHexColor`，数组四舍五入取整 / hex 字符串规范化）；旧工作流已存的数组值经归一后显示恢复正常。旧 serialize hack（写 `widgets_data`）确认是死代码（LiteGraph 序列化字段是 `widgets_values`，`widgets_data` 无人消费）删除。配套 `tests/test_color_picker_js.js`。

---

## 40. number widget 整数/小数输入切换（SFNumber，2026-09）

> 背景：SFNumber 的 `value` 固定声明为 FLOAT（step 0.01），切到 INT 后输入框仍是 0.01 步进的小数框。需求：INT 输入整数、FLOAT/PERCENT 输入小数；另 PERCENT 后端原钳制 0-1 不合理，应允许突破 1。后续演进（2026-09 同日）：解除钳制后 PERCENT 与 FLOAT 完全同质成冗余 → PERCENT 改为 **÷100 百分数书写语义**（输入 150 → 输出 1.5，突破 1 即输入 >100），切档自动换算。

### 1. 方案：改 options，不换 widget 类型

- 后端输入类型保持 FLOAT 不变（换 widget type 有提交类型校验风险，双输入显隐改 schema 破旧工作流）。前端只改 `value.options`：`step`（步进）+ `round`（旧 LiteGraph 取整档）+ `precision`（Vue 新前端小数位）+ `step2`（**前端创建 widget 时按 step 派生的精调步进（修饰键拖拽用），派生一次性、不随 step 联动——只改 step 会残留旧 step2**，实测诊断 D3 锁定）——四键同设兼容两代前端。**切 INT 档时同步 `Math.round` 当前值**，避免显示与步进不一致；切回 FLOAT 值保留。
- 联动三时序沿用 sf_mask_fill 同款（§27 复用）：combo 包装 `callback`（不吞原回调）+ 包装 `node.configure`（工作流恢复不触发 callback，值还原后按保存的 number_type 重应用）+ nodeCreated 初始应用。实现在既有 `web/simple_math.js`（与 SFSimpleMath 同一 Python 文件的节点族共用一个 JS 模块）。
- 后端 `execute` PERCENT 最终语义：`float(value) / 100`（百分数书写，150→1.5，可负可 >100）；INT 取整、FLOAT 直通。

### 1.5 切档换算与恢复时序（PERCENT ÷100 语义追加）

- **切档换算仅限用户显式切换**（callback 包装路径）：以 FLOAT 语义量为规范值 q——q = FLOAT/INT 档显示值；PERCENT 档 q = 显示值 ÷100。切档回填：FLOAT→PERCENT ×100（0.5→50）、PERCENT→FLOAT ÷100（150→1.5）、任意→INT `Math.round(q)`（PERCENT 150→2）、INT→PERCENT ×100（2→200）。节点级 `_sfNumberMode` 记录当前档位供换算判断。
- **configure 恢复路径绝不换算**（存量值原样，只重应用 options + INT 取整）——否则旧工作流 PERCENT 存量值 150 会被错误 ÷100。这与 §27"恢复挂 configure 包装"的模式叠加时须区分两条路径的职责：callback 路径 = 换算 + 应用，configure 路径 = 仅应用。
- PERCENT 档 widget 整数书写 `{step:1, step2:1, precision:0}`（输入 150 而非 150.00）。**旧工作流破坏性**：PERCENT 存量直通值语义突变（存 0.5 → 现输出 0.005），已确认接受。

### 2. 踩坑

- 测试模拟 configure 时，wrapper 捕获的是 **nodeCreated 时刻的 `node.configure`**——先赋还原逻辑再 nodeCreated 才能链上；顺序反了 wrapper 被覆盖、恢复路径测不到（这正是"恢复挂 configure 包装"模式能否生效的关键时序）。
- `SF_NUMBER_OPTS` 档位表集中定义（INT {1,1,1,0} / FLOAT {0.01,0.01,0.01,2} / PERCENT 整数书写 {1,1,1,0}，含 step2），新类型只需加一行；换算矩阵集中在 `SF_NUMBER_TO_Q`/`SF_NUMBER_FROM_Q`。配套 `tests/test_number_js.js`（stage simple_math.js + sf_dynamic_slots.js 同目录暂存）与 `tests/test_simple_math.py` PERCENT ÷100 用例。

### 3. 单输出 any 化（2026-09 追加）

- 原 `RETURN_TYPES = ("INT","FLOAT")` 双输出改单输出 `RETURN_TYPES = (any,)` / `RETURN_NAMES = ("value",)`（复用 `sf_utils/common.py` AnyType）——输出槽类型是静态类属性不能随 number_type 运行时切换，固定 FLOAT 会接不上 INT 输入，固定 INT 丢精度，any 是唯一兼顾方案；execute 按档位返回真类型值（INT→`int`、FLOAT/PERCENT→`float`），下游类型检测仍可 `isinstance` 判断。
- **旧工作流槽位兼容（按槽索引恢复）**：接槽 0（原 int）的链接理论可恢复（any 与 INT 兼容）；接槽 1（原 float）的链接加载时丢弃需手动重接——破裂式改动，已获确认接受。

## 41. combo→输出槽类型前端改型（SFConvert Anything，2026-09）

> 背景：复刻 easy convertAnything（`nodes/logic.py::SFConvertAnything`）——任意输入按 `output_type` combo（string/int/float/boolean）转换输出。后端 `RETURN_TYPES = (any_type,)` 静态声明无法随 combo 变化，输出槽类型同步是纯前端职责（渲染槽点颜色 + 连线校验）。

### 1. 模式

- **改型实现复用 `any_pack.js::setSlotType`**（本次起 export，并扩展可选 `patch` 参数附带 name 等字段，diff 门控含 name）：Vue 前端 `node.outputs` 是 reactive 数组，原地改 `slot.type` 不重渲染槽点，必须**元素替换**（`slots[i] = Object.assign({}, slot, { type }, patch)`，同官方 dynamic-type 模式）。跨模块复用即 `export` 后 import，不内联副本。
- **挂点两条路径**：callback 包装（callback 参数即新值，§27 同款）+ `onAfterGraphConfigured` 恢复（nodeCreated 早于 widgets_values 恢复，lora §31 同款）。easy 原件只有 callback（还靠 setTimeout 300），**无恢复逻辑**——重载工作流槽型回退 `*`，复刻时补上。
- 热重载防双包装 `_sfConvertAnythingPatched`（sf_dropdown.js 先例）；未知 combo 值回退 `"*"` 通配。

### 2. 决策记录

- 输入名照抄原件字面量 `"*"`（kwargs取 `kwargs["*"]`）；None 输入直通返回 None（原件对 None 转 int/float 直接崩，防御性改进）；OUTPUT_NODE=True 照抄（输出悬空也强制执行）；输出槽名跟随类型值改名（string/int/…，同原件；初期曾定保留 "output"，实测后确认跟随才符合预期——改槽名必须 name+localized_name 一并同步，渲染读 label ?? localized_name ?? name）。
- 前端测试 `tests/test_convert_anything_js.js`：Function-eval 双模块注入（any_pack 尾部追加一行把 setSlotType 挂 globalThis 再喂给 convert 模块作用域）。**注意**：`new Function` eval 源码时 strip 正则必须同时去 `import` 语句与 `export ` 关键字（`export function` 直接语法错误）——any_pack 本次新增 export 后 `test_any_pack_js.js` 的旧 strip 正则同步补了 export 剥离。
- `tests/check_web_imports.py` MODS 新增 `sf_convert_anything` 与 `any_pack`（规则 A 只扫 MODS 成员的导出；any_pack 有了导出符号被跨模块 import 后必须入列）。

## 43. 灵活 optional schema 与 Any Switch 复刻（2026-09）

> 背景：复刻 rgthree Any Switch（`nodes/logic.py::SFAnySwitch` + `web/sf_any_switch.js`）——多路任意类型输入取第一个非 None 输出，前端动态增删输入槽。

### 1. 灵活 optional schema（后端）

- 前端动态声明的输入（any_01…）不在后端 schema 里，validatePrompt 会按「schema 外输入」剪掉（§4）——放行靠 dict 子类 `_AnySwitchInputs`：`__contains__` 恒 True + `__getitem__` 回退 `(any_type,)`（照抄 rgthree `FlexibleOptionalInputType`）。后端 `get_input_info` 与前端校验都走这两个协议，schema 即「任意输入名都合法」。
- 不可复用 `_CropOptionalInputs`（crop.py/inpaint_editor.py 同类 dict 子类）：它缺 `__contains__` 覆盖（concrete 键之外一律被剪），且硬编码 image/mask 条目，语义不同。
- 选择语义：按 kwargs 迭代序（即 prompt JSON 里 inputs 顺序 = 前端槽位顺序）取第一个非 None 的 `any_*`；rgthree 的 context 空判断是 RGTHREE_CONTEXT 专属，不复刻。上限 20 由前端约束（对齐项目 MAX_FLOW_NUM 先例），后端不设限。

### 2. 前端槽位与着色

- 槽位增删直接复用 `sf_dynamic_slots.js::installDynamicSlots`（全连→加 1、尾部空→回收、保底 initial=4 ⇒ 稳态恒有 1 个空闲槽），与 rgthree `removeUnusedInputsFromEnd(node,4)+addAnyInput` 的稳态行为等价。**灵活 schema 下前端节点创建时 0 个输入**（object_info optional 为空 dict），初始 4 槽由 nodeCreated 手动 addInput 补齐——与固定 20 输入先例（SFLogicSwitch 靠 schema 提供槽位）的关键差异。
- 着色复用 `any_pack.js::setSlotType/slotLinkTypes/unionType`（本次起 export）：每输入按自身连线类型独立着色（unionType 并集），输出跟随**第一个非 "\*" 的输入类型**（= 选择优先序），label 同步类型名、全断开复位；`onAfterGraphConfigured` 恢复重算（§41 同款挂点）。**有意简化**：不做 rgthree 的 reroute 穿透推断（followConnectionUntilType），经 reroute 时 link.type 为 "\*" 保持不着色——any_pack 同样接受此限制。
- `setSlotType` 在类型未变时提前返回（diff 门控），**patch 里的 label 也不会应用**——初始态/未变态的 label 由槽位自身 localized_name（RETURN_NAMES）提供，不要依赖 patch。
- 包装顺序：nodeCreated 先挂类型重算 wrapper，再 `installDynamicSlots`（它会再包装一层，槽位增删先发生、着色后执行，读到的是最终槽集）。

---

## 49. 固定类型动态多槽：SFConditioningCombine（§43 的定型变体，2026-09）

> 背景：原生 Conditioning (Combine) 只支持 2 路（`nodes.py::ConditioningCombine.combine` 实为 `conditioning_1 + conditioning_2` 列表拼接），新增 `nodes/model/conditioning_combine.py::SFConditioningCombine` + `web/sf_conditioning_combine.js` 实现 N 路动态合并（初始 2、上限 20）。

### 1. 后端：灵活 schema 类型参数化

- 复用 §43 的 `_AnySwitchInputs` 模式（`__contains__` 恒 True + `__getitem__` 回退），但**不可直接复用该类**——它硬编码返回 `(any_type,)`，CONDITIONING 需独立小类 `_ConditioningCombineInputs` 返回 `("CONDITIONING",)`。模式复用、类型参数不同。
- 叠加 `VALIDATE_INPUTS -> True` 接管校验（§4 ComboSelector 先例）：动态重建的槽名超出静态 schema 时默认 "Value not in list" 校验会拦截。
- 执行语义：按槽名数字后缀排序后依次 `out + list(v)` 拼接（kwargs 迭代序≠槽序，不可直接迭代），`conditioning_` 前缀外键忽略、None 槽跳过、全空返回 `([],)` 不抛错（抛错会中断整图）。无 lazy（combine 本就需全输入求值，与 Switch 的 `check_lazy_status` 差异点）。
- 文件位置：`nodes/model/conditioning_combine.py`（CATEGORY=`sfnodes/model`，与动态槽位家族 `nodes/logic.py` 解耦——CATEGORY 与文件目录解耦在项目中有先例）。

### 2. 前端：固定类型免着色 + 恢复补齐

- 槽位增删复用 `sf_dynamic_slots.js::installDynamicSlots`（`inputType:"CONDITIONING"`，`initialInputs:2` 对齐原生 2 路）；**不复用** `any_pack` 着色三件套——CONDITIONING 是固定类型，无改型需求；不做自动重命名——槽序即拼接语义，改名破坏可读性。
- `onAfterGraphConfigured` 恢复须做**槽数补齐/回收**（读实际链接数 `linked`，补到 `min(max(linked+1, INITIAL), MAX)`），而不只是 §43 式的重算类型——configure 直赋 links 不触发 `onConnectionsChange`，光靠公共库的连接时增删，工作流重载后多出的已连槽无空闲后继、新节点无法续接。

---

## 50. SFConditioningConcat 多路拼接：to/from 槽角色与原生对齐（2026-09）

> 背景：原生 Conditioning (Concat) 只支持 2 路（`nodes.py::ConditioningConcat.concat`：from 仅取首条沿 dim=1 拼到 to 每条之后，多条即 warning），新增 `nodes/model/conditioning_concat.py::SFConditioningConcat` + `web/sf_conditioning_concat.js` 实现 N 路动态拼接（初始 2、上限 20）。

### 1. N 路语义：slot1=to 被拼接方，2..N=from 拼接源

- `conditioning_1` 为 to：逐条保留、`dict` 上下文逐条 `.copy()`（原生逐行对齐）；`conditioning_2` 起为 from：各取首条 `[0][0]` 沿 dim=1 依次拼接。与链式串联多个原生 Concat 等价（dim1 拼接满足结合律，一次拼完还少中间临时张量）。
- from 多条目：仅取首条 + `logging.warning` 点名该槽（原生同款警告，不发明展平语义）。
- `conditioning_1` 未连接：抛 `ValueError` 明示（原生此时是晦涩 TypeError，fail-fast 改进）；from 侧 None/空列表跳过，无 from 时 to 原样透传。
- embedding 维度不一致：由 `torch.cat` 原生抛错，不发明 padding；pooled_output 沿用 to 方原样（原生不做加权混合）。

### 2. 同包复用（§49 的延续）

- 槽名前缀/上下限常量、灵活 schema 小类、槽序排序键全部 `from .conditioning_combine import`（语义同为 conditioning_N 动态多槽，禁止内联副本）；为此把 combine 内嵌的排序闭包提升为模块级 `conditioning_slot_key`（等价重构，`test_conditioning_combine.py` 13 断言 unchanged 验证）。
- 前端为 combine JS 的同构薄配置（§49 三条结论延续：固定类型免着色、不自动重命名、恢复按链接数补齐/回收）。
- 本机无 torch：测试注入 stub `torch.cat`（嵌套 list 行内拼接）+ 相对导入改写绝对后 exec（`test_any_to_string.py` 先例），16 断言；warning 路径会打印一行预期日志，非失败。

---

## 52. SFUniversalSlider 万能滑条：孤海 Canvas 复刻的规范化收敛（2026-09）

> 背景：复刻孤海 `GoohaiUniversalSlider`（后端 `万能滑条.py` + 前端 `goohai_universal_slider.js`），新节点 `nodes/utils/universal_slider.py::SFUniversalSlider`（`sfnodes/utils`，与 SFNumber/SimpleMath 同组）+ `web/sf_universal_slider_lib.js` 纯逻辑 + `web/sf_universal_slider.js` 主扩展。后端除 widget 名中文 `值`→英文 `value`（用户确认的破兼容点，孤海旧工作流不能直接替换）外与原版 1:1（FLOAT default 0.75/min -999999/max 999999 + hidden `output_type` float/int + `round(x,10)` + int 取整 + 同逻辑 `IS_CHANGED`）。

### 1. 复用清单与新增边界

- 复用 `sf_utils/common.py::AnyType("*")`（替代原文件内联 `AnyType`，禁内联副本）、`sf_common.js::injectCSSOnce/el`（替代原版直接 `<style>` 注入）、`sf_popup.js::attachPopupDismiss/clampToViewport`（替代原版 overlay 点击/Esc 直写）、`any_pack.js::setSlotType`（输出槽改型元素替换，§41 同款）。
- 新增仅 Canvas 滑条数学（`pct/clamp/snap/fmtVal/castVal/calcValue`）+ 设置归一化（`normalizeSliderSettings`：min/max 对调、step 非法回退、int 档取整——原版分散在设置弹窗 `bOk.onclick`，集中后前后端同测）+ 绘制/拖拽/弹窗装配。不复用 `sf_dynamic_slots`（固定输入）、`sf_crop_framework`（钳制图界无关）、`sf_pause_kit`（闸门无关）。

### 2. 去掉的两个全局副作用

- 原版 `LGraphCanvas.prototype.drawNode` 圆角补丁污染所有节点——删除，仅保留本节点级 `onDrawForeground` 标题重绘（用户确认保留）。
- 原版 `import ... from "../../../scripts/app.js"` 相对路径违反 `check_web_imports.py B2`——改绝对路径 `/scripts/app.js`；扩展名改 `sfnodes.UniversalSlider`（B3）；CSS 前缀 `ghs-`→`sf-us-`、自定义 widget type `goohai_slider`→`sf_universal_slider`（与原插件共存不冲突）；挂点用 `nodeCreated` + 实例级 `configure`/`onAfterGraphConfigured` 包装（`simple_math.js`/`sf_image_resize_plus.js` 先例），不用原版 `beforeRegisterNodeDef` 原型补丁。

### 3. RETURN_NAMES 静态约束的折中

- `RETURN_NAMES` 是类静态属性，不能随 `output_type` 动态切换——后端固定 `("value",)`（`SFNumber` 先例），前端随 `sliderType` 把输出槽类型+槽名 patch 为 `INT/int` 或 `FLOAT/float`（`setSlotType` patch 含 `name/localized_name`，§41 同款；创建/callback 经 `syncOutputType`、恢复经 `configure` + `onAfterGraphConfigured` 双钩子）。
- 测试坑：Function-eval 剥离 import 的正则必须行首锚定（`/^import[^;]+;/gm`）——纯模块边界注释里含 `import xxx.js` 字样，非锚定正则会从注释一路吃到下一个分号、吞掉真实代码（`DEFAULTS is not defined` 误报）；拖拽断言前须先跑一次 `draw` 确立轨道几何（`ds.trackW` 按 `W-ml-mr` 计算，不跑 draw 还是初始值 192）。
- 弹窗按钮必须调包装后的 `ov.remove()`（解绑监听 + 移出 DOM）——裸 `detach()` 只解绑 document 监听，弹窗留在页面上关不掉（确定/取消点击无反应坑，已补开/关断言锁定）。

---

## 53. SFBooleanSwitch 布尔开关：孤海 Canvas 开关复刻（2026-09）

> 背景：复刻孤海 `布尔孤海`（后端 `布尔开关.py` + 前端 `BOOLEAN.js`），新节点 `nodes/utils/boolean_switch.py::SFBooleanSwitch`（`sfnodes/utils`，与 SimpleMathBoolean 同组）+ `web/sf_boolean_switch_lib.js` 纯逻辑 + `web/sf_boolean_switch.js` 主扩展。后端除 widget 名中文 `开关`→英文 `value` 外与原版 1:1（BOOLEAN default True + 单 BOOLEAN 输出直通）；与 SimpleMathBoolean 的差异是单口直通（无 INT 副口）+ Canvas 大开关 UI + 双击改标签。

### 1. 复用与新增边界

- 后端无复用点（BOOLEAN 原生类型，SimpleMathBoolean 同文件亦零导入）；前端复用 `sf_common.js::el`（标签编辑浮层输入框创建）。
- 新增：`TOGGLE` 常量（tw 72/th 28/m 10/xOff 6/clickPad 14）统一绘制与命中——原版两处魔法数字各写一份（draw 侧 `_w-72-10-6`、mouse 侧 `_w-72-10-20`），收敛后 `toggleHit(posX,W) ≡ posX > W-102` 与原版逐字等价（lib 测试锁定）；`normalizeLabel`（去空回落）/`ellipsisText`（measure 注入可测）；命中宽度直接读 `node.size[0]`（不依赖 draw 先跑，原版靠闭包 `_w` 初值 200 同理回退）。

### 2. 规范化改动（相对原版）

- 原型补丁改 `nodeCreated` 实例装配；import 改绝对路径；扩展名 `sfnodes.BooleanSwitch`；自定义 widget `toggle_custom/guhai_toggle`→`sf_boolean_switch/sf_bool_ui`；标签 properties 键 `guhai_label`→`sfBoolLabel`、默认标签 `开关`→`value`（widget 改名口径一致）；配色 `#4F4047/#493C42` 保留（仅创建时设色，configure 不覆盖用户改色——原版同语义）。
- 补原版缺失的 `setDirtyCanvas`（切换/改名/回填三处，原版靠画布偶然重绘刷新）；`configure` 重抓 widgets 引用 + 开关态同步 + 编辑中输入框落盘；`onWidgetChanged` 回填同步。

---

## 54. SFIgnoreGroups 忽略多组：编组开关面板复刻（2026-09）

> 背景：复刻孤海 `忽略多组`（后端空壳 OUTPUT_NODE + 前端 "nodes pass.js" 约 700 行 DOM 面板），新节点 `nodes/utils/ignore_groups.py::SFIgnoreGroups`（`sfnodes/utils`，与前两个孤海复刻同组）+ `web/sf_ignore_groups_lib.js` 纯逻辑 + `web/sf_ignore_groups.js` 主扩展。后端与原版同形（空 required/空 RETURN_TYPES/OUTPUT_NODE/空执行）；properties 键 `guhai_ig_*`→`sf_ig_*`（8 键，lib 读写）。

### 1. 复用与新增边界

- 复用 `sf_common.js::el/injectCSSOnce/installWheelZoomPassthrough`（滚轮转发是原版手写版的超集：可滚动的颜色下拉走原生滚动，其余转发画布缩放）与 `sf_popup.js::attachPopupDismiss/clampToViewport`（设置弹窗）；`addDOMWidget` + `setInterval` 轮询是项目既有模式（dropdown/prompt_tags/load_image 先例），无组旁路逻辑可复用。
- 新增：组几何（`groupBounds/nodeBounds/hit/inside`，折叠节点估宽与原版同式）、颜色归一、嵌套组递归、`groupState` 三态（空组按 true，原版同款）、`filterSortGroups`（空组过滤 + 关键词（`|` 分隔多词 OR，`splitKeywords` 去空段、无有效段等同留空） + 颜色 + 位置/字母排序）、`toggleTransition` 三模式纯函数（含嵌套连带开关）。

### 2. 规范化改动（相对原版）

- 原型补丁改 `nodeCreated` 实例装配；import 绝对路径；扩展名 `sfnodes.IgnoreGroups`；CSS/组件名前缀 `guhai-ig-`→`sf-ig-`；旁路/禁用魔法数字具名为 `MODE_BYPASS=4/MODE_NEVER=2/MODE_ALWAYS=0`。
- 全局 `app.graph.change` 逐节点重复包装收敛为守卫单例（`_sfIgPatched` + `_live` 实例集广播 dirty，自切换实例豁免）；定时器/document 监听/style 片段按节点清理（`onRemoved` + 脱图兜底双保险，测试锁定 timer 置空与监听成对移除）。
- 补原版缺失的切换/恢复后 `setDirtyCanvas`（面板重建只刷自身，画布节点灰化需显式 dirty）。
- 弹窗坑：overlay 与 pop 同级挂载时 `attachPopupDismiss` 只认 `overlay.contains`，pop 内点击会被误判外部点击关窗——必须传 `exempt: (e) => pop.contains(e.target)`（universal slider 的 panel 是 overlay 子节点，无此问题；测试用 sf_popup 真源锁定：去 exempt 必现面板内点击关窗）。

---

## 55. SF 注释便签：原版纯前端架构 1:1（2026-09）

> 背景：复刻孤海 `孤海注释`。第一版曾按 sfnodes 形态重做（后端 SFNote 节点 + DOM 面板），但 Vue 前端节点覆盖层吞双击导致编辑器打不开；遂改回原版纯前端架构：`web/sf_note.js`（节点类 + 编辑器 + 全局补丁 + 画布菜单入口，无后端、无 hidden 真源）+ `web/sf_note_lib.js` 纯逻辑。状态存节点 properties（随 workflow 保存，原版同款）。

### 1. 移植边界（相对原版 GoohaiNote.js）

- 1:1：节点类（默认样式/尺寸/序列化/链接点击/双击编辑/工具条全套/固定模式/右键菜单）、drawNode/processMouseDown/getNodeOnPos 全局补丁行为、document 鼠标监听、Vue dblclick 中继、CJK 换行引擎语义。
- 适配集：import 绝对路径；扩展名 `sfnodes.Note`；样式 id 改 `sf-note-*` 前缀；节点类型 `孤海注释`→`SF Note`（与原插件共存不撞槽）；全局补丁加 once 守卫（重复加载不叠包）；换行引擎三函数复用 lib 真源（`wrapCharList` 以 `measure` 注入替代 ctx，其余逐字一致；非 ASCII 正则保持 `\u` 转义原样，勿写字面量）。
- 新增唯一功能：画布背景菜单 `Add SF Note`（`getCanvasMenuItems`，sf_canvas_align 同款入口）——官方 `Comfy.AddNode` 优先、`LiteGraph.createNode` + `graph.add` 兜底（sf_lora_browser 同款顺序），落点视口中心 + 随机抖动。

### 2. 测试注意

- 主扩展测试需 LiteGraph 全套 mock（`LGraphNode` 基类/`registerNodeType`/`NO_TITLE`/`active_canvas`/原型方法 + `window.open`），模块顶层求值时即需存在；`registerCustomNodes` 显式调用后断言类型注册。
- 等宽测量（1/字符）锁定引擎语义：英文断词、CJK 整字、禁则回拉形态（`ab，cd` 宽 2 → `a/b，/cd`）均可字面断言；手算期望时注意禁则 `…` 等占宽（§53 同类坑）。
- 编辑器测试开闭成对（`createTextEditor` 起 16ms 坐标跟随定时器，`removeTextEditor`/finish 统一清，否则挂起进程）。
- 测试 stub 坑（测试 bug、产品代码无辜）：dblclick 假事件必须包成 `{target,...}` 真结构（中继读 `e.target`）。
- 残留风险（本机不可验，需真机确认）：Vue 下前端独占类型能否经 palette/菜单正常添加、workflow 重载是否保留（见 §1 入口与序列化说明）；验证步骤：画布右键 Add SF Note → 双击出工具条 → 改字号保存 → 存工作流重载对文本。

---

## 57. 纯函数单源收敛与测试桩同步（2026-09）

> 背景：全仓扫描出的逐字重复纯函数（`tensor2images`/`_json_safe`/`_decode_image`/`_sf_user_dir`/`_valid_name`/隐藏状态解析/LoRA stem/hex 颜色/`canvasBackingScale`/`escapeHTML`/POST 样板）收敛到 `sf_utils/common.py`、`sf_utils/disk_state.py`、`sf_common.js` 与文件内 `_post`。零行为差是硬要求：每个替换点先逐行对照再删副本。

### 1. 收敛手法（调用点零改动优先）

- import 别名保持本地名：`from ...sf_utils.common import json_safe as _json_safe`（pause_image/pause_mask/preview_routes/pause_latent）、`parse_json_dict as _parse_state`（brush_mask/crop_expand）、`decode_image as _decode_image`（preview_routes）、`sf_user_dir as _sf_user_dir`（5 处）、`valid_name as _valid_name`（krea2）、`lora_stem`（三 LoRA 加载器，去掉变无用的 `import os`）。已有 `_parse_fill_color`/`_safe_join` 同款先例。
- 语义差必须参数化而非强合：`_valid_name` 两处差长度上限 → `valid_name(name, max_len=None)`（text_presets 传 200）；`load_image_resize._parse_state` 多一行 malformed 日志 → 保留校验包装，合并逻辑委托 `resize_engine.parse_resize_state`（精确等价：空/坏 JSON/非 dict/未知键四路行为不变，`tests/test_load_image_resize.py` 原三项断言全过）。
- 前端：`sf_outpaint.js` 删本地 `canvasBackingScale` 改 import（`sf_common.js` 注释早已点名"×2 复制是 bug 温床"）；`sf_prompt_tags.js` 删本地 `escapeHTML` 改 `escapeHtml as escapeHTML`（新版转义引号，对属性上下文更安全）；`sf_lora_stack_api.js` 抽文件内 `_post`（照抄 `sf_dmodel_api.js`，成功侧 `invalidateInfo+broadcast`，`deleteCivitai`/`setCivitaiAccount` 用 `{ invalidate: false }` 保原语义）。

### 2. 测试桩必须同步（本次真坑）

- 被测模块新增相对导入时，三种测试桩反应不同：真实路径桩（`test_load_image_resize.py` 先例）自动解析，无需改；`__path__ = []` 桩（character/id_clothing 先例）必须预加载新依赖子模块（`_load("sfnodes.sf_utils.disk_state", ...)`）；裸 spec 无桩（pause_* / styles_selector）直接炸（`attempted relative import`/`No module named`），补真实路径桩。
- `sf_utils/__init__.py` 为空，真实路径桩执行它无重依赖，安全。
- 新增 `tests/test_common_pure.py` 锁定新纯函数（`json_safe`/`parse_json_dict`/`lora_stem`/`valid_name`/`_parse_fill_color`/`sf_user_dir`），纳入后端回归循环。

### 3. 否决项（查到但不动）

- `sf_load_image_resize.js` 的 `openSimpleColorPicker`：该文件零 import、刻意不依赖 app.js 可单测，移入 `sf_common` 会破坏设计。
- 路径包含性 7 实现：`commonpath` vs `startswith`、跨盘回退、`exists` 语义各异且各有测试锁定，不盲合。
- 纯模块边界（`*_lib.js` 不得 import `sf_common`）、数值取整三契约、ReDoS 双端镜像：故意不共享（见 §26 与 AGENTS）。

---

## 58. 第二批收敛：守卫单源/前缀清洗/原子写盘/样例 kind 参数化（2026-09）

> 背景：§57 审计的 Tier 2 四项（A1 守卫合并、save_image_exact 前缀副本、原子写盘 9 处、B7 样例漏迁移）。共同点：收敛前先证明"同构或参数化后同构"，调用点行为（含日志/返回值/异常类型）逐项对齐。

### 1. A1 graph-undo 守卫合并（真缺陷修复）

- `sf_prompt_tags_guard.js` 与 `sf_crop_undo_guard.js` 逐行等价却各持独立 `_tokens`——引用计数跨模块拆分，关一个编辑器时另一模块可能误判 `_anyAlive()==false` 提前 stand down。删前者，`sf_prompt_tags_editor.js` 改 import（其余两调用点不动）。
- 删 JS 文件的联动清单：`check_web_imports.py` MODS 去行（规则 A 会扫导入符号存在性）+ 冒烟测试 staging 列表换名（copy→改写 import 后缀机制）+ `architecture.md` + `nodes-text.md` 模块索引。文件名保留 `sf_crop_undo_guard`（改名扩大 churn），头注改为"全编辑器单源"并顺手修正过时用法路径。
- 合并后引用计数 reunite：crop/inpaint/tags 三编辑器同开同关走同一 `_tokens`，正是注释要求的"单一系统"。

### 2. safe_prefix 收敛（扩展名钩子保留）

- `disk_state.safe_prefix/sanitize_segment` 提升 preview 版；save 的 `_safe_filename` 只剩扩展名剥离 + 二次截断（`_PREFIX_OUTPUT_MAX` 本地留值，`_WIN_RESERVED_NAMES` 本地保留供剥离后二次守卫）。
- 等价性证明：save 版截断后 `if not result: return ""` 与 preview 版直接返回 rstripped 结果——rstrip 为空时两者同为 `""`，行为一致；`%date:FMT%` 省略说明是 preview 特有知识，随函数迁入共享 docstring，不丢。
- `import re` 两处随正则定义一并删除（grep 确认无他用）。

### 3. 原子写盘：诚实限界（9→8+1 有意例外）

- 9 处并非完全同构（tmp 命名/dump 参数/makedirs/清理/异常类型/返回值各异），强合需参数爆炸。收敛形状取"抛错上浮 + helper 内清理"：`atomic_write_bytes`（pid+tid 临时名，比 tid 版严格更安全）+ `atomic_write_json(path, obj, *, ensure_ascii=False, indent=2)`；makedirs/日志/返回值/异常处理全留调用点。
- 字节等价：标准库默认形参的调用点显式传 `ensure_ascii=True, indent=None`（与原 `json.dump(data, f)` 逐字节一致，实测锁定）；bytes 点 `bytes(raw)` 转换保留。
- `lora_routes` 采样下载的 content-hash 临时名 rationale 不同（同文件并发下载），不迁、记为有意例外；`mtime_size_sig` 只收 text/krea2（character/styles 是三元组键，不硬合）。
- 迁移后 `threading` 在 lora_presets/krea2/text_presets/workflow_index_helpers/workflow_routes 变无用导入，一并删除（grep 逐文件确认）。

### 4. B7 样例 kind 参数化（零外部调用方是安全前提）

- 动手前先 grep：共享版 5 函数除 stack_info 本地版外**零调用方**，加可选 `kind` 参数不可能回归他人；`fetchSamplesCached` 的 kind 支持已先行存在，只需贯穿 preview/hover/resolve。
- 面板耦合点：本地 overlay/hover 类（`sf-ls-sample-preview`/`sf-ls-desc-hover`）绑定面板 CSS + 两处 dismiss 豁免引用。共享版行内样式与面板 CSS 值逐字一致，外观不变；dismiss 两处（onKey + onDocClick）同步换类名，死 CSS 删除。共享版多出的 `sfPreviewEsc` 标记只被对话框 cancel 读取，面板无读者，无害。
- dmodel 冒烟测试（`samplesKind: "diffusion_models"` ctx）直接覆盖 kind 路径。

### 5. 编辑工具卫生（本次连踩两次）

- 用 edit 删除整函数时，`oldString` 不要把被保留的 def 行也包进去——一次误配会整段删掉邻居函数体。删后立即 `py_compile` + `git diff --stat` 确认零误删（本批一次误删 `decode_image`、一次误删 `sf_user_dir`，均当场恢复）。
