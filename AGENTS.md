# AGENTS.md - sfnodes (ComfyUI Custom Node Pack)

## Project Overview

sfnodes 是一个 ComfyUI 自定义节点包，提供图像处理、人脸操作、遮罩编辑、文本处理、模型管理等增强功能。

ComfyUI 源码根目录即 `../..`（`custom_nodes/` 的父目录，含 `comfy/`、`nodes.py` 等，**仅为源码副本**，实际运行实例为 docker 部署——以实际挂载路径为准），可用于查阅 API 和参考实现。**不要尝试在本机启动 ComfyUI 或安装运行时依赖。**

## Architecture

```
sfnodes/
├── __init__.py      # 注册入口：NODE_CLASS_MAPPINGS + NODE_DISPLAY_NAME_MAPPINGS + WEB_DIRECTORY="web"
├── requirements.txt # Python 依赖（仅声明，不在本机安装）
├── nodes/           # 节点实现：face/ image/ mask/ model/ text/ utils/ inpaint/ latent/ 子目录 + logic.py（循环/Any 打包）、workflow_routes.py
├── sf_utils/        # 共享工具库（无状态纯函数为主）：image/mask 转换、lora_* 系列、resize_engine / dropdown / regional_engine / krea2_presets / disk_state / prompt_reader 等纯逻辑模块
├── web/             # 前端 JS Widget：sf_common.js（公共小工具/微工具 injectCSSOnce·sfToast·el·hideJsonWidget/强调色/LoRA 行名）+ sf_popup.js（弹层三件套）+ 各节点模块（单文件或 *_lib/*_core/_ui 多模块系列）
├── data/            # 静态数据（prompt_presets.json、styles/ 内置风格库+samples、CSV/字体等）
├── tests/           # 前端/后端模拟测试（Node/Python 直接运行，无测试框架）
└── doc/             # 文档：architecture.md 逐文件细目 / experience/ 经验归档（README 索引 + 六主题文件）/ vibecoding.md 任务模板
```

**逐文件职责与机制说明见 `doc/architecture.md`**——新增/删除文件必须同步其条目。

## Node Registration & Class Convention

根 `__init__.py` 两字典同步注册：

- `NODE_CLASS_MAPPINGS`: 键 `"SF<ClassName>"`，值为类本身（历史键可能无 SF 前缀如 `LoadImages`，新增一律带前缀）
- `NODE_DISPLAY_NAME_MAPPINGS`: 键同上，显示名 `"SF <Display Name>"`

```python
class SFMyNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {...}, "optional": {...}}

    RETURN_TYPES = ("TYPE",)
    RETURN_NAMES = ("name",)
    FUNCTION = "execute"          # 执行方法名
    CATEGORY = "sfnodes/<group>"  # 统一 sfnodes/<功能组>：face/image/mask/model/text/utils/logic/inpaint/latent
    DESCRIPTION = "..."           # 必填

    def execute(self, ...):
        return (result,)
```

## Dependencies & ComfyUI APIs

- 运行时第三方依赖见 `requirements.txt`（带用途注释的完整清单在 `doc/architecture.md`）；`torch/torchvision` 由 ComfyUI 运行时提供，不入 requirements。**新增依赖必须同步 requirements.txt。**
- ComfyUI 运行时提供的常用 import 清单见 `doc/architecture.md`；易错点：
  - `nodes.NODE_CLASS_MAPPINGS` **运行时才包含全部自定义节点**——函数内 import 最安全
  - `ExecutionBlocker` 官方位置 `comfy_execution.graph_utils`（graph.py 只是 re-export）；DYNPROMPT 隐藏输入对象是 `comfy_execution.graph.DynamicPrompt`

## Code Style

- Python 3.10+，无类型注解强制；用 `_CATEGORY` 模块级常量定义分类前缀；工具函数放 `sf_utils/`（无状态纯函数），节点放 `nodes/<组>/`
- JS Widget 放 `web/`；**动手前先查公共模块复用**：前端 `sf_common.js`（小工具/微工具 injectCSSOnce·sfToast·el·hideJsonWidget/强调色/LoRA 行名）、`sf_dynamic_slots.js`（动态槽位）、`sf_popup.js`（新弹层优先，见 experience/patterns.md §26）、`sf_crop_framework.js`（编辑器框架）；后端 `disk_state.py` 与 `sf_utils/` 各纯逻辑模块。有公共实现必须复用，**禁止内联副本**
- **纯模块边界**：`*_lib.js` / `*_core.js` / `sf_markdown.js` 纯逻辑模块（无 app 依赖、可拷 .mjs 单测）**不得 import sf_common.js**（它依赖 /scripts/app.js）；通用纯函数共享放无依赖模块
- **注册规范**（由 `tests/check_web_imports.py` 固化校验）：registerExtension 文件必须直接 `import { app } from "/scripts/app.js"`（禁相对路径）、扩展注册名 `sfnodes.*` 前缀、相对导入目标必须存在；新增 web 模块需加入该脚本 MODS 列表
- 子目录 `__init__.py` 为空（`nodes/utils/` 无此文件走 namespace package），仅根 `__init__.py` 负责注册

## Development Rules

1. **不要启动 ComfyUI 或运行 pip install**——本机仅代码编辑环境；一次性生成工具装 `/tmp` 用，产物内联进 web/ 模块，不得进 requirements.txt
2. 可阅读 `../..` 源码理解 API 与参考实现
3. 新增节点同步更新根 `__init__.py` 两个注册字典
4. 新增依赖同步 `requirements.txt`
5. 实现类 PascalCase，注册键 `"SF"` 前缀
6. 图像张量 `[B,H,W,C]`；遮罩张量 `[B,H,W]`
7. JS Widget 用 `app.registerExtension` 注册；纯工具模块仅 export 函数由使用者 import
8. 动态槽位类 JS 复用 `web/sf_dynamic_slots.js` 公共库，勿重复实现
9. 部署为 docker：后端改动需重启容器；`web/` JS 改动需同步该目录且浏览器硬刷新（Ctrl+Shift+R）才生效
10. **实际环境调试禁止自行浏览器访问 ComfyUI**（404 且干扰用户工作流）：一律分段 console 诊断脚本（版本检查→节点状态→事件日志→数据层→UI 层）交用户执行反馈 → experience/platform.md §2.9；节点请用户 UI 添加（新版前端无 graph.createNode）
11. **新增节点/功能前先查复用**（见 Code Style），禁止内联副本——语义分叉是 bug 温床。去重/重构注意：① 独立语句的包装块不在函数体内按名删除会漏；② 文件已有某模块 import 时脚本补 import 可能跳过致缺符号（被 try/catch 吞掉极难排查）；③ ESM 结构错误用 `node --input-type=module --check < file` 验证
12. 新增/删除 py/js 文件同步 `doc/architecture.md` 条目；沉淀新经验按主题写入 `doc/experience/` 对应主题文件（下一个全局 §N）并同步 README.md 索引表；**确属新类别且现有主题均不适配时可新建主题文件**（英文短名对齐节点族，标题注明所含章节）

## Testing

本项目无自动化测试框架。验证方式：

- 静态检查：两注册字典键一致、所有节点类正确导入、requirements.txt 含全部第三方依赖
- 后端模拟测试：mock torch/comfy.utils 加载节点模块，FakeDynPrompt 断言图结构与返回值（循环节点有先例）
- 前端模拟测试：无 DOM 依赖的公共库复制 `.mjs` 用 Node 直跑（FakeNode + 事件序列，tests/ 有先例）
- 快速回归命令（文件自含断言，任一失败非零退出即停）：
  - 后端：`for f in tests/test_*.py; do python3 "$f" || break; done`
  - 前端：`for f in tests/test_*.js tests/test_*.mjs; do node "$f" || break; done`
  - 静态一致性：`python3 tests/check_web_imports.py`

## 经验摘要（不变式索引）

> 完整机制与踩坑案例在 `doc/experience/` 六主题文件：platform / patterns / nodes-text(简写 text) / nodes-image(image) / nodes-lora(lora) / apps，全局 § 号映射见目录内 README.md；摘要括号内 `<简称> §N` 指向对应文件章节，改动对应功能前先读该文件。

- **循环/图展开**（`nodes/logic.py`，platform §1）：execute 可返回 `{"result","expand"}` 展开动态子图，result 中 link 值 `[id,slot]` 被解析为链接目标值；**ForLoopEnd 必须被下游消费才会被调度执行**（死端节点从不执行——"循环不跑"先查其输出有无下游）；隐藏输入首轮不在 prompt 中→kwargs 缺键而非 None，需默认值兜底。
- **widget 值传后端必须先声明输入**（patterns §4 / image §11）：前端提交 prompt 前 validatePrompt 删除 schema 外输入——任何"运行时状态"输入须 Python hidden 声明 + 同名隐藏 STRING widget 走标准收集；注入只能作双保险。**勿写 addDOMWidget 的 .value**（Vue setter 回调链无限递归），读取走 getValue。
- **graphToPrompt ≠ 队列**（text §6/§7）：Export/分享/保存也触发——注入可在此做，剪枝/游标 commit 只能在 `api.queuePrompt` 成功后；闸门 continue 先于 pause/pass 处理，解析不到节点默认 pass（fail-safe 不剪）。**Python 禁 IS_CHANGED=float("nan")**（NaN 折叠祖先缓存键→下游每次全量重跑），要重跑用 time_ns 或 (mtime,size)。
- **数据载体与动态 combo**（patterns §4 / lora §31）：状态存隐藏 STRING widget 值随 workflow 自动保存/复制，新节点=全新默认值；前端动态重建 combo options 必须 `VALIDATE_INPUTS` 返回 True 接管校验，恢复挂 onAfterGraphConfigured（nodeCreated 早于值恢复）。
- **动态槽位**（platform §2.7）：改槽名必须同步 name+localized_name（渲染读 label ?? localized_name ?? name）；configure 直赋 links 不触发 onConnectionsChange。
- **isGraphLoading / 接线互斥守卫**（image §13 / text §15）：包装 loadGraphData +300ms 尾窗（连接恢复发生在 onConfigure 之后）；互斥断线三重守卫 = onConfigure 窗口 + 尾窗 + 自递归标志。
- **Vue 新版前端**（platform §2.8 / apps §30）：先容器内 `pip show comfyui-frontend-package` 确认版本（1.x=Vue）；槽位数组 shallowReactive 替换元素才触发渲染；动态 tooltip 写 widget.tooltip（nodeDef 兜底清不掉）；程序化建节点走官方 `Comfy.AddNode` 命令（裸 createNode+graph.add 只弹 toast 不渲染）。
- **画布多选尺寸对齐**（platform §11）：`sf_canvas_align` 画布背景 `getCanvasMenuItems` 三入口（SF Align Width/Height/Size：Widest/Narrowest/Tallest/Shortest/First Selected + Size 同时改两维）单维保持另一维不变，等大分别钳制 `computeSize()[0/1]`，选中集兼容 Object/Array/Map/Set 四形态 + `is_selected` 回退，撤销走 `beforeChange/afterChange`。
- **四闸门 prune 共用一份实现**（text §7 / image §8·§9·§22）：text/image/mask/latent 全走 `sf_pause_text_lib.js::applyGateMode`（latent 加 extraInputKeys）；image/mask/latent 三闸门共用 `sf_pause_kit.js` 引擎（definePauseGate/buildPauseBody/makeGateState 工厂，text 结构独立不入 kit）——改闸门行为只动 kit，⚠ frameEventKey 与 `_sfPauseXxx*` 属性前缀逐字保留；快照文件前缀隔离命名空间（图片 PNG / 遮罩灰度 PNG / latent safetensors 全张量键）；PNG 拖回嵌入前 _json_safe（NaN/Inf→字符串）；_safe_prefix 先查 ".."/绝对路径再清洗。
- **图片浏览器右键菜单**（image §34）：SFLoadImageBrowser 图片项右键=复制正向提示词 + 载入内嵌工作流，全链路复用零后端改动——提示词走 `/api/sfnodes/prompt_reader/extract`（output 拼 `[output]` 注解），工作流走 `loadWorkflowFromImageUrl(url)`（sf_lora_shared_info 参数化导出）+ 内置 `/view` 原始字节（无 preview 参数即原文件）；DOM 菜单挂 body、z-index 高于弹窗 overlay、close() 必须联动关菜单。
- **统一填充 SFMaskFill**（image §35）：合并 SFMaskedFill/SFMaskFillColor 单节点 `SFMaskFill`，`fill_mode` 四态 + `fill_color/opacity` 仅 color 显隐（`sf_mask_fill.js` hidden 切换+双钩子保恢复）、`falloff/skip_if_all_white` 全局；`_parse_fill_color/_apply_falloff` 纯函数复用 `mask_utils`；宽松 batch/尺寸策略对齐 color 旧实现。
- **双端镜像**（text §14/§15 / patterns §27）：替换/数字语法逻辑 Python 权威 + JS 预览镜像，两侧测试同用例同期望值锁定；ReDoS 启发式（嵌套量词+交替型）双端 1:1（regex_extract 复用、内置预设跳过）；数字契约 _NUMBER_RE/_JS_WHITESPACE/half-away-from-zero 取整。
- **lean 注入作缓存键**（text §6/§15）：graphToPrompt 注入只含影响结果的字段（选中值+类型）——改行名/重排/切模式不重跑；游标 pending 在 queue 成功后 commitPick，位置存节点内存或未注册设置按共享范围选，写 properties 会误标 modified。
- **SF Workflows 面板**（apps §10）：面板是"应用"非节点（分享工作流不携带）；热键避开原版 combo（全局去重报错）；sidecar meta 读写 asyncio.Lock 防读改写互擦；收藏前先 await loadBookmarks()。
- **SF LoRA 浏览器**（apps §30）：后端零新增全复用 lora_* 路由；信息编辑经 openInfoPanelFor(ctx,id) 宿主适配复用 Stack 面板；平面模式分批渲染防千级列表卡死。
- **信息面板跨域复用**（lora §32）：新数据域（如 SF Load Diffusion Model）接同一面板 = 后端同 handler 别名路由（`routes.get(别名)(handler)`，handler 内 `_dom_*(request)` 按路径分派存储域：dmodels.json/previews_model/diffusion_models，物理分离防撞槽）+ 前端 ctx 三开关（hideTriggers/samplesKind/autoCivitai）+ **ctx.api 整束注入（键名错会静默回退 lora 路由——冒烟测试锁形状与"绝无 /lora_* 回退"）**。
- **自定义 API 路由**（image §8.4 等）：`from server import PromptServer` → ins.routes 装饰器 try/except 包裹、导入时副作用注册；前缀统一 /api/sfnodes/；**改动路由必须重启容器**否则 404 静默降级。
- **全屏编辑器与冒烟测试**（text §6.3/§6.5）：类名前缀与既有插件隔离；Esc 用 window capture 分层处理；危险操作 confirmDanger 无撤销设计；mock DOM 冒烟能抓语法检查漏掉的运行时错误。
- **中文 token/标签库**（text §6）：token 名 `[\p{L}\p{N}_-]` 带 u flag（中文可作 tag）；标签库存未注册设置（机器私有跨工作流）+ 工作副本 isSameAsStored 判定才写回；拼音表一次性生成内联，非运行时 npm 依赖。
- **提示词恢复 SFPromptReader**（text §16）：三种元数据容器纯标准库解析（MP4 ilst 是 1-based INDEX 非 4cc；WebM 键大写归一小写）；目录状态字段避开 applyResult 写入键的撞名（用 folder）；**DOM widget 高度 ≥ 内容实际高度**（动态测量 + computeLayoutSize，别硬编码）。
- **SFPromptPreset / Krea2 预设**（text §29 / lora §31·§5）：分类正交原则防组合污染；IS_CHANGED=seed 可复现随机 + seed 偏移防各分类同值；预设管理内置+用户覆盖+墓碑删除 merge() 墓碑胜出；Qwen3 thinking 参数按模型微调来源选（instruct 版 off、无审查微调版 on 反而正常）。
- **SFLongTextToList**（text §36）：复刻 ComfyUI_Lam LongTextToList——任意分隔符分割→索引取值/列表/长度，分隔符 `\n`/`\t` 转义、空分隔符退化为单元素、越界返回空串不崩、`filter_empty` 空行过滤（默认 True）；类型收敛为 `STRING+OUTPUT_IS_LIST` 对齐 `SFPromptList` 生态，无前端。
- **SFTextListAffix**（text §37）：输入 `STRING` 列表逐项加前后缀，`INPUT_IS_LIST+OUTPUT_IS_LIST` 透传；前后缀 `\n`/`\t` 转义、`filter_empty` 去空白空项（对齐 `SFPromptList`），`sf_utils/string.py:affix_list` 纯函数复用。
- **TextEncodeKrea2 视觉通路**（lora §33）：Qwen3-VL tokenizer 每视觉占位符只绑 images 列表单元素、batch 参考图只取 [0]（官方编辑节点同款）；min_pixels=3136 官方兜底极小裁剪图，勿自建最小尺寸保护；RGBA 先黑底预乘再缩放（反序会把透明区杂色扩散进边缘）。
- **SFLoraStack**（lora §19/§19.7/§20·§34）：Civitai API 字段位置必须实测（description 在 version 顶层）；用户数据以路径为键→改名失配两级孤儿匹配（内容指纹优先基名兜底）；强调色 --sf-acc 三时序坑（onChange 参数即新值 / 重绘 setTimeout(0) / 异步加载轮询）；行名设置 sfnodes.Lora.DisplayName 单真源 sf_common.loraRowLabel；ortho_gs 独立加载路径收敛 ortho_apply，ok_paths 是 set 勿直接迭代组装顺序敏感结果；复合预设 positive 与 triggers 分离、机器级存储、栈内 Presets 菜单同表单保存、SFLoraPreset 的 STRING 输出流通。
- **LoRA 数据统一网关**（lora §19）：lora_triggers.json 单一真源（lora_notes 只做形状转换）；跨节点缓存失效经 sfnodes.lora-data-changed 事件桥；信息对话框与 Stack 面板同一数据语义。
- **Civitai 页面抓取**（lora §21 / patterns §27）：页面是 Next.js SSR，数据在 `__NEXT_DATA__` 按 queryKey 定位勿碰 DOM；**TLS 指纹被 Cloudflare 拦截——curl_cffi impersonate="chrome"，Chrome UA 的 aiohttp 也 403**；描述统一 _html_to_markdown 幂等保护（无 `<` 输入只轻清洗原样放行）。
- **值通道模式**（lora §25/§28）：hidden STRING 真源随 workflow 保存 + DOM widget 纯交互不承担值传输（regional_lora/styles_selector 同款）；加载期 isGraphLoading 门控点击防覆盖刚恢复的选择。
- **复刻去重与磁盘链路**（patterns §17）：磁盘源执行必须输出源帧 ui_payload 否则前端预览停留旧图；编辑器 Reset≠Clear 语义一一对应。
- **模型下载统一**（patterns §27）：HF resolve URL → hf_hub_download 缓存+copy2 到约定路径；不用 local_dir（子目录破坏平铺拼接）；HF 失败不回退 requests。
- **输入框键盘/滚轮**（patterns §27）：keydown 必须放行 ctrl/meta/alt 组合键（否则 Ctrl+S 漏成浏览器保存）；DOM widget 输入框不在 Vue wheel 转发路径 → installWheelZoomPassthrough。
- **静态检查与版本陷阱**（patterns §3/§27）：ast.unparse 输出单引号、literal_eval 遇变量引用抛错——先怀疑检查脚本再怀疑代码；ast.Constant.n 已移除一律写 node.value；__pycache__ 的 cpython-3xx 是本机解释器版本，不代表运行容器。

## Code Discovery

优先使用 **codebase-memory 知识图谱**（`search_graph`、`trace_path`、`get_code_snippet`）查找函数、类及其调用关系，代替 grep/glob。仅在搜索字符串字面量、错误消息、配置文件等非代码内容时回退 grep/glob。
