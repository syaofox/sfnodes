# AGENTS.md - sfnodes (ComfyUI Custom Node Pack)

## Project Overview

sfnodes 是一个 ComfyUI 自定义节点包，提供图像处理、人脸操作、遮罩编辑、文本处理、模型管理等增强功能。

ComfyUI 源码根目录即 `../..`（`custom_nodes/` 的父目录，含 `comfy/`、`nodes.py` 等，**仅为源码副本**，实际运行实例为 docker 部署——以实际挂载路径为准），可用于查阅 API 和参考实现。**不要尝试在本机启动 ComfyUI 或安装运行时依赖。**

## Architecture

```
sfnodes/
├── __init__.py      # 注册入口：NODE_CLASS_MAPPINGS + NODE_DISPLAY_NAME_MAPPINGS + WEB_DIRECTORY="web"
├── requirements.txt # Python 依赖（仅声明，不在本机安装）
├── nodes/           # 节点实现：face/ image/ mask/ model/ text/ utils/ inpaint/ latent/ video/ 子目录 + logic.py（循环/Any 打包）、workflow_routes.py
├── tools/           # 一次性脚本（extract_lora_diff.py 模型差异提 LoRA，自带 README，不进 requirements.txt）
├── sf_utils/        # 共享工具库（无状态纯函数为主）：image/mask 转换、lora_* 系列、resize_engine / dropdown / regional_engine / krea2_presets / disk_state / prompt_reader 等纯逻辑模块
├── web/             # 前端 JS Widget：sf_common.js（公共小工具/微工具 injectCSSOnce·sfToast·el·hideJsonWidget/强调色/LoRA 行名）+ sf_popup.js（弹层三件套）+ 各节点模块（单文件或 *_lib/*_core/_ui 多模块系列）
├── data/            # 静态数据（prompt_presets.json、styles/ 内置风格库+samples、CSV/字体等）
├── tests/           # 前端/后端模拟测试（Node/Python 直接运行，无测试框架）
└── doc/             # 文档：architecture.md 逐文件细目 / experience/ 经验归档（README 索引 + 六主题文件）/ vibecoding.md 任务模板
```

**逐文件职责与机制说明见 `doc/architecture.md`**——新增/删除文件必须同步其条目。

## Node Registration & Class Convention

根 `__init__.py` 两字典同步注册：

- `NODE_CLASS_MAPPINGS`: 键 `"SF<ClassName>"`，值为类本身（现 171 键全部带 SF 前缀；新增一律带前缀）
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

- **循环/图展开**（`nodes/logic.py`，platform §1）：execute 可返回 `{"result","expand"}` 展开动态子图，result 中 link 值 `[id,slot]` 被解析为链接目标值；**LoopEnd 已声明 OUTPUT_NODE=True（2026-09），输出悬空也会被调度执行循环**——但 WhileLoopEnd 重建收集（`_collect_output_nodes`）必须跳过 SFForLoopEnd，否则其被误当循环体内输出消费者纳入 contained → 每轮克隆再 expand 嵌套错误展开（WhileLoopEnd 自身被收集无害：explore_dependencies 已排除 + upstream 守卫去重）；隐藏输入首轮不在 prompt 中→kwargs 缺键而非 None，需默认值兜底。
- **widget 值传后端必须先声明输入**（patterns §4 / image §11）：前端提交 prompt 前 validatePrompt 删除 schema 外输入——任何"运行时状态"输入须 Python hidden 声明 + 同名隐藏 STRING widget 走标准收集；注入只能作双保险。**勿写 addDOMWidget 的 .value**（Vue setter 回调链无限递归），读取走 getValue。
- **graphToPrompt ≠ 队列**（text §6/§7）：Export/分享/保存也触发——注入可在此做，剪枝/游标 commit 只能在 `api.queuePrompt` 成功后；闸门 continue 先于 pause/pass 处理，解析不到节点默认 pass（fail-safe 不剪）。**Python 禁 IS_CHANGED=float("nan")**（NaN 折叠祖先缓存键→下游每次全量重跑），要重跑用 time_ns 或 (mtime,size)。
- **数据载体与动态 combo**（patterns §4 / lora §31）：状态存隐藏 STRING widget 值随 workflow 自动保存/复制，新节点=全新默认值；前端动态重建 combo options 必须 `VALIDATE_INPUTS` 返回 True 接管校验，恢复挂 onAfterGraphConfigured（nodeCreated 早于值恢复）。
- **动态槽位**（platform §2.7）：改槽名必须同步 name+localized_name（渲染读 label ?? localized_name ?? name）；configure 直赋 links 不触发 onConnectionsChange。
- **isGraphLoading / 接线互斥守卫**（image §13 / text §15）：包装 loadGraphData +300ms 尾窗（连接恢复发生在 onConfigure 之后）；互斥断线三重守卫 = onConfigure 窗口 + 尾窗 + 自递归标志。
- **Vue 新版前端**（platform §2.8 / apps §30）：先容器内 `pip show comfyui-frontend-package` 确认版本（1.x=Vue）；槽位数组 shallowReactive 替换元素才触发渲染；动态 tooltip 写 widget.tooltip（nodeDef 兜底清不掉）；程序化建节点走官方 `Comfy.AddNode` 命令（裸 createNode+graph.add 只弹 toast 不渲染）；**槽名文字相对槽点固定下偏 2-5px 是前端包全局行为**（`NodeSlot.draw` 硬编码 `u[1]+5`+alphabetic，所有 canvas 节点一致，节点侧勿修，platform §2.13）。
- **画布多选尺寸对齐**（platform §11）：`sf_canvas_align` 画布背景 `getCanvasMenuItems` 三入口（SF Align Width/Height/Size：Widest/Narrowest/Tallest/Shortest/First Selected + Size 同时改两维）单维保持另一维不变，等大分别钳制 `computeSize()[0/1]`，选中集兼容 Object/Array/Map/Set 四形态 + `is_selected` 回退，撤销走 `beforeChange/afterChange`。
- **SF Memory 画布菜单清理**（platform §2.14）：画布背景 `getCanvasMenuItems` 无门槛注入 `SF Memory ▶ Free VRAM / Free RAM` 子菜单；VRAM 走原生 `POST /free`（队列 flag 有序执行，勿自建路由与运行任务竞态），RAM 走自建 `POST /api/sfnodes/memory/ram` 复用 `memory_cleanup.RAMCleanup`（`retry_times=1`）；反馈走 `sf_common.sfToast`。
- **四闸门 prune 共用一份实现**（text §7 / image §8·§9·§22）：text/image/mask/latent 全走 `sf_pause_text_lib.js::applyGateMode`（latent 加 extraInputKeys）；image/mask/latent 三闸门共用 `sf_pause_kit.js` 引擎（definePauseGate/buildPauseBody/makeGateState 工厂，text 结构独立不入 kit）——改闸门行为只动 kit，⚠ frameEventKey 与 `_sfPauseXxx*` 属性前缀逐字保留；快照文件前缀隔离命名空间（图片 PNG / 遮罩灰度 PNG / latent safetensors 全张量键）；PNG 拖回嵌入前 _json_safe（NaN/Inf→字符串）；_safe_prefix 先查 ".."/绝对路径再清洗。
- **图片浏览器右键菜单**（image §34）：SFLoadImageBrowser 图片项右键=复制正向提示词 + 载入内嵌工作流，全链路复用零后端改动——提示词走 `/api/sfnodes/prompt_reader/extract`（output 拼 `[output]` 注解），工作流走 `loadWorkflowFromImageUrl(url)`（sf_lora_shared_info 参数化导出）+ 内置 `/view` 原始字节（无 preview 参数即原文件）；DOM 菜单挂 body、z-index 高于弹窗 overlay、close() 必须联动关菜单。
- **统一填充 SFMaskFill**（image §35）：合并 SFMaskedFill/SFMaskFillColor 单节点 `SFMaskFill`，`fill_mode` 四态 + `fill_color/opacity` 仅 color 显隐（`sf_mask_fill.js` hidden 切换+双钩子保恢复）、`falloff/skip_if_all_white` 全局；`_parse_fill_color/_apply_falloff` 纯函数复用 `mask_utils`；宽松 batch/尺寸策略对齐 color 旧实现。
- **镜头切分 SFImageSceneSplit**（image §36）：缩略灰度直方图/像素差 + 黑白场连续段 + 溶解滑窗累积（纯逻辑 `sf_utils/scene_detect.py`，无 ComfyUI 依赖）；逐帧生成器防 OOM；`segment_index` 负索引 + `max_frames` 首 N 帧截断 + `all_segments` LIST 全段 + `cuts/scene_count` 辅助输出，越界抛错。
- **出界裁剪 SFImageCropExpand**（image §37）：复刻 YCNodes Load Image Crop Expand——可出界裁剪框（负坐标/越界=外扩），输出填充色画布 + 白=扩展区遮罩 + filename（src_path 原样可直连 LoadImage）；源图持久化复用 crop.py upload_src 路由（原版清 base64 重载丢图，已确认差异）；状态单 JSON 存 properties + graphToPrompt 注入（outpaint 先例）；交互数学抽纯库 sf_crop_expand_lib.js（拖拽冻结快照防自反馈飘移），sf_crop_framework 因钳制图界不复用；Browse 按钮复用 image_browser 弹窗 onPick 选择器模式（/view 取字节→落盘链路）；`_parse_fill_color` 自 masks.py 提升到 sf_utils/common.py 同源导入。
- **画布节点拖拽缩小外溢**（image §44）：双端拖拽 resize 的最小值都取自 `node.computeSize()`（onDrag 里 clamp 到 computeSize 再 setSize）——包装 computeSize 返回 max(原值, MIN) 一处钳住两条路径；**`node.onResize` 是 legacy-only**（Vue 前端拖拽不触发，先例钳制均 isVueNodes() 守卫），新钳制需求优先选 computeSize 包装；已保存 size 恢复不走 computeSize，由 onConfigure 钳制兜底；底部信息文本超界与节点大小无关（画布区填满时基线恒 nodeH+5），修法是显示区高度让出 `TEXT_RESERVE` 预留行；右下角 resize cursor 常不显示：命中区原生 15×15 本身可用，**`pointer.resizeDirection` 会被 hover 判定切换与第三方扩展重放 processMouseMove 反复清空**，dir→帧尾 updateCursorStyle→css 间接链路不可靠——后执行的 window mousemove listener 在原生命中区内**直接写 style.cursor**（区外经 owned 标志恢复 ""）（§44）。
- **画笔遮罩 SFImageBrushMask**（image §45）：复刻 YCNodes Load Image Brush Mask——节点上加载图片直接涂抹二值遮罩（brush 涂白/erase 擦除，Clear/Undo，Size 滑块；Opacity/取色仅预览叠加后端忽略）；源图持久化复用 crop.py upload_src 路由（零新增路由，原版 base64+Map 缓存刷新丢图，已确认差异）；状态单 JSON 存 properties + graphToPrompt lean 注入（仅 src/strokes，改预览不重跑）；栅格化纯逻辑 `sf_utils/brush_mask.py`（向量化印章 brush 置1/erase 置0）；旧串颜色判定须严格形态（三段纯整数，否则 `10,10;20,20` 被 parseInt 截断误判吃掉整笔，原版真 bug）；输出 +filename（src_path 原样可直连 LoadImage）；MIN/cursor/Browse/粘贴链路同 §37/§44；右键 SAM 蒙版走核心 SAM3_Detect（checkpoints multiplex 同文件拆 MODEL+CLIP，转 fill 矢量笔触并入统一管理/常驻可卸载，§45.9）。
- **批次区间 SFImageBatchRange**（image §51）：复刻 KJ GetImageRangeFromBatch——`start_index+num_frames` 双路 IMAGE/MASK 独立切片（`_resolve_range` 纯函数），`-1` 取尾部、尾部超出截断、start 越界抛错；新建文件不动 `SFImageBatchIndex` 取单逻辑（后者同步补 `-1` 取尾帧）。
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
- **SFCharacterSelect 角色多选图库**（text §47/§48）：单节点多角色切换，角色内勾选不限数量图片——输出拼接提示词 STRING + 角色级提示词 STRING（草稿优先，双路同覆盖）+ images batch IMAGE（单选即单张，首图尺寸归一，空选 1×1 占位）；v2 破兼容 4→3 路（老 face/half/full 连线失效）；角色库 `character_` 前缀 JSON（`{name,prompt,images[]}`，旧 face/half/full 形状与旧数组状态双读兼容），新双路由 `/api/sfnodes/characters*`；隐藏双真源 SFCharacterState（{role,shots}，空/失效回落首角首图）+ SFCharacterPrompt（手改草稿整体覆盖拼接路，切换/改选即清）。
- **SFIDClothingSelector 证件照服装单选**（text §46）：复刻孤海 IDPhotoClothingSelector 画廊——单选模板输出提示词 STRING + 模板图 IMAGE（无选择/远图/失败时 1×1 黑占位）；模板库复用 styles JSON 生态 `id_` 前缀子集（user/sfnodes/styles/id_*.json + samples_id_clothing/ 独立目录，孤海 `标题-提示词` 文件名一次性转写，包内不放二进制），列表/缩略图零新增路由全复用 `/api/sfnodes/styles*`；隐藏双真源 SFIDClothingState（单选取首）+ SFIDClothingPrompt（手改草稿，切换即清）；前端搜索含 prompt（英文描述在 prompt 里）；lib 自包含可拷测不跨 import 纯模块。
- **SFTextPreset 全局持久化**（text §42）：预设真源 `user/sfnodes/text_presets.json`（数组保序），execute 草稿 text_override（工作流级）优先 → 全局库 → 旧工作流 presets_json 回退兼容；前端 combo=全局+工作流残留、编辑框草稿语义（切预设即清、↧ 保存到预设显式入库、残留项可晋升）、API 失败降级；范式复用 lora_presets（Lock/原子写/校验）+ krea2_presets（mtime+size 缓存/_sf_user_dir）；归一化入口须同时收 dict 与裸列表（save 传列表否则缓存清空丢数据）；诊断脚本 D0 fetch 版本检查会假阴性（ES module 磁盘缓存），实锤要用调用栈行号比对各版本源码。
- **TextEncodeKrea2 视觉通路**（lora §33）：Qwen3-VL tokenizer 每视觉占位符只绑 images 列表单元素、batch 参考图只取 [0]（官方编辑节点同款）；min_pixels=3136 官方兜底极小裁剪图，勿自建最小尺寸保护；RGBA 先黑底预乘再缩放（反序会把透明区杂色扩散进边缘）。
- **SFLoraStack**（lora §19/§19.7/§20·§34·§36）：Civitai API 字段位置必须实测（description 在 version 顶层）；用户数据以路径为键→改名失配两级孤儿匹配（内容指纹优先基名兜底）；强调色 --sf-acc 三时序坑（onChange 参数即新值 / 重绘 setTimeout(0) / 异步加载轮询）；行名设置 sfnodes.Lora.DisplayName 单真源 sf_common.loraRowLabel；ortho_gs 独立加载路径收敛 ortho_apply，ok_paths 是 set 勿直接迭代组装顺序敏感结果；复合预设 positive 与 triggers 分离、机器级存储、栈内 Presets 菜单同表单保存、SFLoraPreset 的 STRING 输出流通；触发词勾选全局默认 `selected`（新建行默认值，工作流覆盖，空行回填，见 §36）。
- **SFImageInterrogator seed**（lora §35）：`seed` 必须显式声明 `control_after_generate` 或显式移除，禁止依赖前端隐式追加；隐式追加导致 `widgets_values` 位置敏感错位 `control=1`；自愈需全覆盖 `seed/control/vision/thinking` 四槽。
- **SFImageInterrogator 三模态可选**（lora §38）：`image/video/audio` 均 `optional` 对齐原生 `Generate Text`，无图时纯文本生成；`video` 为 `IMAGE` batch 24→1FPS 抽帧逐帧缩放、多帧按 `Picture N:` 前缀；`user_prompt` 紧邻 `prompt` 重排（破兼容，文本→视觉→采样→模板分区），`min_p/presence_penalty/use_default_template` 对齐原生 `sampling_mode`，`max_length` 放宽至 `8192`；`_scale_image None→[]` 防御。
- **LoRA 数据统一网关**（lora §19/§36）：lora_triggers.json 单一真源 `{words,description,selected,fp}`（lora_notes 只做形状转换）；跨节点缓存失效经 sfnodes.lora-data-changed 事件桥；信息对话框与 Stack 面板同一数据语义；触发词勾选全局默认 `selected` 仅空行回填（工作流触发词覆盖全局）。
- **SFWanWindowLoRA 逐窗 LoRA**（lora §39/§39.11）：EVALUATE 回调按 `window_idx % N` 换槽（空槽=上游直通）；独立槽表永不写 `patcher.patches`，普通层后装 LowVramPatch 进 `weight_function`，GGUF 量化层改写张量 `patches`（GGMLOps 绕过 `weight_function`，duck-typing 探测）；上游补丁保留叠加；回调双形状注册（上游 reader 只认 `["callbacks"]` 分支）。
- **SFWanWindowPlanner + preset 直连**（lora §39.6/§39.9）：Planner 只算不跑，直调原生 schedule/region 出窗 latent·实帧区间 × cond 段 × slot 对照表；SFLoraStack 第 5 输出 `preset_export` 转 SFLoraPreset 形状直连窗槽（model 悬空即零 IO 纯配置源）。
- **Civitai 页面抓取**（lora §21 / patterns §27）：页面是 Next.js SSR，数据在 `__NEXT_DATA__` 按 queryKey 定位勿碰 DOM；**TLS 指纹被 Cloudflare 拦截——curl_cffi impersonate="chrome"，Chrome UA 的 aiohttp 也 403**；描述统一 _html_to_markdown 幂等保护（无 `<` 输入只轻清洗原样放行）。
- **值通道模式**（lora §25/§28）：hidden STRING 真源随 workflow 保存 + DOM widget 纯交互不承担值传输（regional_lora/styles_selector 同款）；加载期 isGraphLoading 门控点击防覆盖刚恢复的选择。
- **复刻去重与磁盘链路**（patterns §17）：磁盘源执行必须输出源帧 ui_payload 否则前端预览停留旧图；编辑器 Reset≠Clear 语义一一对应。
- **模型下载统一**（patterns §27）：HF resolve URL → hf_hub_download 缓存+copy2 到约定路径；不用 local_dir（子目录破坏平铺拼接）；HF 失败不回退 requests。
- **输入框键盘/滚轮**（patterns §27）：keydown 必须放行 ctrl/meta/alt 组合键（否则 Ctrl+S 漏成浏览器保存）；DOM widget 输入框不在 Vue wheel 转发路径 → installWheelZoomPassthrough。
- **静态检查与版本陷阱**（patterns §3/§27）：ast.unparse 输出单引号、literal_eval 遇变量引用抛错——先怀疑检查脚本再怀疑代码；ast.Constant.n 已移除一律写 node.value；__pycache__ 的 cpython-3xx 是本机解释器版本，不代表运行容器。
- **COLOR 输入被内置 widget 收编**（patterns §39）：新版 Vue 前端 `widgetStore` 合并时 core 覆盖同名 getCustomWidgets 注册，COLOR 实际渲染内置 ColorWidget（hex 文本+色块）；**COLOR default 一律写 hex 字符串不写数组**（数组显示 "0,0,0" 且无色块），按 type 查 widget 需大小写不敏感（自定义 "COLOR" vs 内置 "color"），后端 hex/数组双兼容，旧工作流数组值前端三时序归一（nodeCreated/loadedGraphNode/configure 包装）。
- **number widget 整数/小数切换**（patterns §40）：SFNumber 切 INT/FLOAT/PERCENT 只改 `value.options` 四键（step+step2+round+precision 双代前端兼容；**step2 是前端创建 widget 时按 step 派生的精调步进，派生一次性不随 step 联动须一并覆盖**），**不换 widget type**；切档换算以 FLOAT 语义量为规范值 q（PERCENT 显示值 = q×100），**仅 callback 路径换算、configure 恢复路径只应用档位不换算存量值**；输出单槽 any 化（随档位输出真类型值，旧工作流 float 槽 1 链接丢弃已确认接受）；PERCENT 后端 ÷100 百分数语义（150→1.5），PERCENT 存量直通值语义突变已确认接受。
- **combo→输出槽类型前端改型**（patterns §41）：SFConvert Anything（复刻 easy convertAnything）后端 `RETURN_TYPES=(any_type,)` 静态声明，输出槽类型+槽名同步是纯前端职责——复用 `any_pack.js::setSlotType`（已 export，patch 参数附带 name/localized_name；Vue reactive 数组必须**元素替换**非原地改）；挂点 = callback 包装（参数即新值）+ `onAfterGraphConfigured` 恢复（nodeCreated 早于值恢复）；None 输入直通（原件对 None 转 int 直接崩）；Function-eval 测试 strip 正则须同剥 `import` 语句与 `export ` 关键字；跨模块 import 的模块必须入 check_web_imports MODS。
- **灵活 optional schema 与 Any Switch**（patterns §43）：SFAnySwitch（复刻 rgthree）前端动态输入靠 dict 子类 `_AnySwitchInputs`（`__contains__` 恒 True + `__getitem__` 回退 any_type）放行 schema 外输入——`_CropOptionalInputs` 缺 `__contains__` 不可复用；灵活 schema 下前端 0 初始槽由 nodeCreated 补齐（固定 20 先例靠 schema 提供槽位）；着色复用 `any_pack.js` export 的 `setSlotType/slotLinkTypes/unionType`（输出跟随第一个非 `*` 输入，label 同步类型名）；不做 reroute 穿透着色（经 reroute 保持 `*`）；setSlotType 类型未变时提前返回，patch 的 label 不应用。
- **固定类型动态多槽 SFConditioningCombine**（patterns §49）：N 路 CONDITIONING 按槽序拼接（原生 Combine 串联语义）；灵活 schema 类型参数化独立小类（`_AnySwitchInputs` 硬编码 any_type 不可直复）；固定类型免着色+免重命名；`onAfterGraphConfigured` 按链接数补齐/回收槽数（configure 不触发 onConnectionsChange，重载后须补空闲后继）。
- **多路拼接 SFConditioningConcat**（patterns §50）：slot1=to 被拼接方逐条保留（dict 逐条 copy），2..N=from 各取首条沿 dim1 拼接（多条 warning，原生对齐）；缺 base 抛 ValueError 明示；槽名前缀/上下限/schema/排序键同包复用 combine 模块；本机无 torch 时测试注入 stub `torch.cat`。
- **万能滑条 SFUniversalSlider**（patterns §52）：复刻孤海万能滑条 Canvas 大滑条——后端除 widget 名 `值`→`value` 外 1:1（单 any 输出 + hidden output_type + `round(x,10)` + IS_CHANGED 回显）；RETURN_NAMES 静态不可随档变，后端固定 `("value",)`、前端随档改输出槽类型+槽名（`setSlotType` patch，§41 同款）；去全局 drawNode 补丁/CSS 前缀改 `sf-us-`/相对导入改绝对路径；值数学收敛纯 lib 可 `.mjs` 直测。
- **布尔开关 SFBooleanSwitch**（patterns §53）：复刻孤海布尔开关 Canvas 开关——后端除 widget 名 `开关`→`value` 外 1:1（BOOLEAN default True + 单口直通，无 INT 副口）；绘制/命中魔法数字收敛 `TOGGLE` 常量（`toggleHit ≡ >W-102` 与原版等价）；单击切换 + 双击改标签（`sfBoolLabel`）+ 配色保留；补原版缺失的 `setDirtyCanvas` 三处。
- **忽略多组 SFIgnoreGroups**（patterns §54）：复刻孤海忽略多组编组开关——后端空壳 OUTPUT_NODE，前端 DOM 行列表一键旁路/禁用整组（三模式 + 筛选 + 设置弹窗 + 500ms 轮询同步外部改动）；`app.graph.change` 重复包装收敛守卫单例 + 定时器/监听按节点清理；状态键改 `sf_ig_*`。
- **SF 注释便签**（patterns §55）：复刻孤海注释（原版纯前端架构 1:1：节点类 + 编辑器 + 三全局补丁 + Vue 中继，无后端）——类型改 `SF Note` 防撞、补丁加 once 守卫、引擎复用纯 lib（`measure` 注入替 ctx，正则保 `\u` 转义）；入口为画布菜单 `Add SF Note`（`Comfy.AddNode` 优先 + `LiteGraph` 兜底）。
