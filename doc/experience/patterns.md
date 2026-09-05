# 经验归档：横切模式与修复批次（§3、§4、§17、§26、§27、§39）

> 全局章节号 §N 与拆分前的 experience.md 一致；跨节/跨文件引用一律写 §N，映射见 [README.md](README.md)。版本时效说明见 README。

## 3. 静态检查脚本经验（AST 对比踩坑）

用 Python AST 做"前后端一致性/结构对比"验证时（如对比注册字典、检查节点 INPUT_TYPES），易踩两个坑：

1. **`ast.unparse` 输出的是单引号字面量**：`ast.unparse(v)` 生成的字符串（如 `'interrogate'`、`'CLIP'`）统一用单引号包裹，与手写断言中的双引号字面量（`"interrogate"`）直接比较会**误判不一致**。取值应优先用 `ast.literal_eval(node)`（常量），或按节点类型提取：`Constant.value` / `Name.id`（变量引用）/ `List.elts`。不要拿 unparse 文本与手写字面量做相等比较。
2. **`ast.literal_eval` 遇到变量引用会抛 `ValueError: malformed node or string`**：默认值引用模块常量的表达式（如 `"default": KREA2_INSTRUCT_SYSTEM`）无法直接求值。需分两步：先单独提取被引用的常量（`ast.literal_eval`），再遍历映射，遇 `Name` 节点取其 `id` 后查表替换。

真实案例：比对 `KREA2_PRESETS` 前后端一致性时，`ast.unparse` 残留的单引号让"文本一致"误判为 false；`ast.literal_eval` 直接解析含 `KREA2_INSTRUCT_SYSTEM` 引用的字典抛 ValueError。两处均为检查脚本问题，非代码问题——**先怀疑检查脚本，再怀疑被检查的代码**。

---

## 4. 动态 combo 校验与工作流绑定状态（widget 数据载体）

> 背景：SFTextPreset 工作流绑定文本预设节点（2026-08），落地为 `nodes/text/text_preset.py` + `web/sf_text_preset.js`。需求：预设绑定当前工作流，其他工作流添加此节点是全新空预设。

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

- 13+ 个浮动弹层各自重复"外部点击/Esc/滚轮三关闭 + 定位钳位"，且踩坑记录在 §15.6（canvas 缩放定位）与 §19（确认框豁免宿主面板捕获监听）。收敛为：
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

### 4. cropstitch 多帧必崩 + 设备不匹配（cropstitch.py）

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
- **analysis.py**：两处 `torch.where(mask)` 判空兜底（mask_process 腐蚀/裁剪清空遮罩时 `x.min()` 崩）——照抄 `landmarks is None` 的全零占位模式保持 batch 对齐。
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
