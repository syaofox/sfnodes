# 架构细目（architecture）

> AGENTS.md「Architecture」的逐文件展开。新增/删除文件后**必须同步本文档对应条目**（一行职责描述即可，机制细节归 `experience/` 目录对应章节（索引见其 README.md））。

## 目录结构

```
sfnodes/
├── __init__.py          # 节点注册入口：NODE_CLASS_MAPPINGS + NODE_DISPLAY_NAME_MAPPINGS
├── requirements.txt     # Python 依赖（仅声明，不在本机安装）
├── nodes/               # 所有节点实现，按功能分子目录
│   ├── face/            # 人脸：分析、对齐、扭曲、区域、遮挡、人像分割（person_mask.py：SFPersonMask）
│   ├── image/           # 图片：加载（files.py：SFLoadImages / browser.py：SFLoadImageBrowser / load_images_path.py：SFLoadImagesPath 三源渐进式 / load_image_resize.py：SFLoadImageResize）、缩放（resize_image.py：SFImageResize wired 尺寸）、拼接（concatenate.py）、混合（blend.py）、切块（tile.py）、变换（transform.py/scale.py）、处理（processing.py）、对比（compare.py）、三点色彩匹配（color_match_points.py：SFImageColorMatchByPoints 亮度分位自动提取暗/灰/亮三点 → 逐通道三点分段线性 LUT）、LUT（lut.py）、仿色（imitation_hue.py）、批次索引（batch_index.py）、可视化裁剪+贴回（crop.py）、外绘填充+贴回（outpaint.py）、RFMSR 超分（rfmsr_upscale.py：SFRFMSRUpscale）、图片闸门（pause_image.py）、latent 闸门（pause_latent.py：SFPauseLatent 分段采样中间暂停）、预览保存路由（preview_routes.py）
│   ├── mask/            # 遮罩：参数、轮廓、模糊、缩放、填充、反转、遮罩闸门（pause_mask.py）
│   ├── model/           # 模型：LoRA加载（多行 LoRA 栈 lora_stack.py：SFLoraStack，含触发词/描述/封面/Civitai 查询；预设 lora_preset.py：SFLoraPreset（原 Power 预设改名，供 Stack 的 preset 输入复用）；批量对比 lora_plot.py：SFLoraPlot 动态行模型输出列表 + SFLoraPlotImageSaver 文字标注，复用 stack 状态契约与 sf_utils/lora_plot.py、lora_cache.py；区域注入 regional_lora.py：SFRegionalLoRA 多区域角色 LoRA（每 box 一个 LoRA，激活 delta 只注入 box 内 image token，Krea2 专用，forward hook 稀疏注入 + 每区域匹配诊断，纯逻辑在 sf_utils/regional_engine.py）；扩散模型强化加载 load_diffusion_model.py：SFLoadDiffusionModel 官方 UNETLoader 行为超集 + i 信息面板（执行委托原生 load_unet 零漂移，前端 web/sf_load_diffusion_model.js）；其余 lora_loader.py/lora_loader_model_only.py/lora_selector.py、hyperlora.py、sage_attention.py、krea2.py（TextEncodeKrea2 视觉条件编码 + SFImageInterrogator 图像反推，thinking 显式透传 + 输出剥离 Qwen3 思考块，见 experience/nodes-lora.md §5.4；参考图 RGBA 黑底预乘合成（_flatten_to_rgb）+ batch 取首帧 + 遮罩包围盒裁剪，见 experience/nodes-lora.md §33；Interrogator/SystemPrompt 预设可管理——内置+用户覆盖，见 experience/nodes-lora.md §31）、adv_clip.py、rfmsr/（RFMSR 网络实现，节点在 image/rfmsr_upscale.py））、CLIP编码
│   ├── text/            # 文本：翻译/拼接/角色选择（text.py：TextTranslation/TextCombine/AnimeCharSelect；另 concatenate.py：SFTextConcatenate）、值下拉（dropdown_value.py：name→value 列表 + 四类型输出 + F/I/R 模式）、提示词列表（prompt_list.py：SFPromptList 行拆分/切片/空白行过滤 skip_empty，行号编辑器 sf_prompt_list.js）、动态 Prompt 列表（prompt_stack.py：SFPromptStack 行动态添加/每条开关/右下角角标拖拽调行高（state.rows[i].h，随工作流保存），状态对齐 Pixaroma PromptStack 形状 rows/enabled/text，prompt_reader 恢复共享，sf_prompt_stack_core.js 纯逻辑 + sf_prompt_stack.js 行 UI）、提示词预设（prompt_preset.py：SFPromptPreset 十一分类组合/加权随机/[A,B] 括号随机/LLM 优化链路 optimize_request + SFUnpackPromptPreset 解包，数据 data/prompt_presets.json 热加载）、工作流文本预设（text_preset.py）、@tag 标签库提示词（prompt_tags.py）、风格选择器（styles_selector.py：SFStylesSelector 复刻 Easy-Use stylesSelector——Fooocus 275 风格多选/搜索/悬停缩略图，内置 data/styles/*.json + 用户 user/sfnodes/styles/*.json 同名覆盖，{prompt} 占位拼接 1:1，隐藏 SFStylesState 真源，同文件注册路由 /api/sfnodes/styles*）、内联文本闸门（pause_text.py）、查找替换（find_replace.py）、替换模板（replace.py）、正则提取（regex_extract.py）、任意转字符串（any_to_string.py）、提示词批处理（prompt_batcher.py）、随机编辑（random_edit_prompt.py）、多机位相机（multiangle_camera.py）、PNG/视频元数据提示词恢复（prompt_reader.py：SFPromptReader，含 prompt_reader_routes.py 路由 /api/sfnodes/prompt_reader/{extract,list}）
│   ├── utils/           # 工具：数学、显示、内存清理、分辨率、图像编辑
│   ├── inpaint/         # 局部修复：裁剪、拼接、外扩
│   ├── latent/          # latent 分块采样（klein_tiled_ksampler.py：SFKleinTiledKSampler 复刻 Comfy-SZ-KleinKSampler，FLUX.2 Klein 放大修复/细节增强——latent_blend 全局引导、整图连续全局噪声、同尺寸 tile 两两 batch=2 并行采样约 40-50% 加速、overlap 羽化写回、色彩统计量对齐）
│   ├── workflow_routes.py # 工作流面板后端路由（/api/sfnodes/workflows/*）
│   └── logic.py         # 逻辑：索引切换、Any 打包/解包、遮罩判空、循环（For/While Loop）
├── sf_utils/            # 共享工具库
│   ├── common.py        # AnyType 通用类型
│   ├── image_convert.py # tensor/pil/numpy/mask 互转
│   ├── mask_utils.py    # 遮罩工具
│   ├── inpaint_helpers.py # 局部修复辅助（裁剪/拼接/缩放，无 ComfyUI 依赖）
│   ├── adv_encode.py    # 高级编码工具
│   ├── string.py        # 字符串工具
│   ├── translation.py   # 翻译封装
│   ├── downloader.py    # 模型下载工具（HF resolve URL → huggingface_hub.hf_hub_download 缓存+复制到约定路径；requests 兜底非 HF URL；timeout/.part 原子替换）
│   ├── model_manager.py # 模型管理
│   ├── cutpaste.py      # 剪切/拼接工具
│   ├── blend.py         # 混合工具
│   ├── insightface_utils.py # InsightFace 封装
│   ├── face_detector.py  # 人脸检测
│   ├── lora_constants.py # LoRA 扩展名单单点真源（LORA_EXTS/LORA_EXT_RE，lora_reader/lora_samples/sf_common 共用，禁内联副本）
│   ├── lora_notes.py     # LoRA 用户数据统一存储网关（SFLoraStack 与 SFLoraLoader 系共用 lora_triggers.json 真源；旧 .sf.json 侧车惰性迁移，见 experience/nodes-lora.md §19）
│   ├── lora_presets.py   # LoRA 预设
│   ├── lora_samples.py   # 模型样例图处理（sample/ 旁目录约定；kind 参数分派 loras/diffusion_models 两域，SF Load Diffusion Model 复用）
│   ├── lora_reader.py    # LoRA 元数据/触发词/内容指纹纯逻辑（SFLoraStack 用，无 ComfyUI 依赖；read_safetensors_metadata/file_sha256/侧车与用户数据存储函数全部按 (file/folder, name) 参数化，diffusion 域直接复用）
│   ├── lora_plot.py      # LoRA 批量对比纯逻辑（文件名净化/元数据双向/字体选择含 CJK/文字覆盖，SFLoraPlot 用，无 ComfyUI 依赖）
│   ├── lora_cache.py     # LoRA 文件缓存 + 内存模式修剪（last/all/none，与 SFLoraStack 同语义，SFLoraPlot 用）
│   ├── lora_routes.py    # SFLoraStack 路由（/api/sfnodes/lora_*、civitai/account 等，见文件内注册清单）+ 数据域分派（_is_dmodel_req/_dom_*：/api/sfnodes/dmodel_* 别名路由同 handler 服务 diffusion 域，存储换 dmodels.json + previews_model/ + diffusion_models 目录）
│   ├── diffusion_routes.py # SF Load Diffusion Model 路由（GET /dmodel_info：safetensors __metadata__ 架构/config + 大小/mtime + 用户数据 + 孤儿兜底，形状对齐 build_lora_info、触发词恒空；查询/描述/预览等由 lora_routes 别名提供，模块尾副作用注册经节点 import 触发）
│   ├── lora_ortho.py     # 正交堆叠纯数学（GS 投影，仅 torch；ortho_gs 用）
│   ├── lora_ortho_load.py # ortho_gs 独立加载+应用路径（ortho_apply(model, clip, entries, load_sd)，SFLoraStack 专用，禁止各写一份）
│   ├── workflow_index_helpers.py # 工作流索引纯逻辑（Workflows 面板，无 ComfyUI 依赖）
│   ├── resize_engine.py  # 图片缩放引擎（8 模式 + wired 尺寸 _apply_wired_size，无 ComfyUI 依赖）
│   ├── tiling.py         # 图片切块纯逻辑（行/列/重叠 → 块矩形，SFImageTile/Untile 共用，无 ComfyUI 依赖）
│   ├── color_match_points.py # 三点色彩匹配纯逻辑（亮度分位三点提取/逐通道分段线性 LUT/查表，SFImageColorMatchByPoints 用，无 ComfyUI 依赖）
│   ├── regional_engine.py # 区域 LoRA 纯逻辑（键归一化/矩阵解析/regions JSON/层规划+每区域匹配诊断/token 网格 mask 数学/彩虹预览，SFRegionalLoRA 用，无 ComfyUI 依赖）
│   ├── dropdown.py      # 值下拉纯逻辑（数字语法双端契约 readable/coerce，无 ComfyUI 依赖）
│   ├── krea2_presets.py # Krea2 预设管理纯逻辑（内置+用户覆盖+墓碑删除+复位，merge/校验/读写，register(kind,builtin,protected) 注册路由，SFImageInterrogator 反推预设 + SFKrea2SystemPrompt 系统指令预设共用，见 experience/nodes-lora.md §31）
│   ├── video_thumb.py   # 视频首帧提取纯逻辑（cv2 首帧→jpeg，Civitai 视频缩略与 Sample 视频缩略共用，无 ComfyUI 依赖）
│   ├── disk_state.py    # 磁盘状态共享实现（safe_join/sanitize_id/sanitize_filename/decode_image，crop 与 inpaint 共用；sanitize_filename 供 hyperlora/lut 等"自由 STRING → 文件路径"净化）
│   ├── skin.py          # 肤色估计纯逻辑（numpy RGB→LAB 肤色过滤取均值/回退，SFFaceWarp 未连接源图时填充近似肤色用，无 ComfyUI 依赖）
│   ├── prompt_reader.py # 提示词恢复纯逻辑（PNG tEXt + MP4 keys/ilst + WebM EBML Tags 解析、graph walker 反推 sampler 文本链，无 ComfyUI 依赖）
│   └── logger.py        # 日志
├── web/                 # 前端 JS Widget
│   ├── sf_common.js     # 复刻节点公共小工具（sfApiUrl / isVueNodes / applyAdaptiveCanvasOnly / isGraphLoading / installGraphLoadingGuard / installCanvasZoomPassthrough / installWheelZoomPassthrough / parseAnnotatedImageValue / buildSourceURL / getUpstreamImageURL / installPasteHandler / escapeHtml / downloadDataURL / copyText）+ 全局强调色（getSfAccent/applySfAccentVar/sfAccent，document 根 --sf-acc CSS 变量体系）+ LoRA 行名真源（loraDisplayName/getLoraDisplayMode/loraRowLabel，Stack/Plot 共享，设置键 sfnodes.Lora.DisplayName——旧 sfnodes.PowerLoraLoader.DisplayName 键已废弃不读取）+ 微工具（injectCSSOnce 守卫式样式注入统一入口 / sfToast extensionManager 封装 / el DOM 快捷创建 / hideJsonWidget 隐藏序列化 widget / canvasBackingScale CSS→物理像素换算）；**依赖 /scripts/app.js——纯逻辑模块（*_lib.js/*_core.js/sf_markdown.js）不得 import 本文件**
│   ├── sf_dynamic_slots.js # 动态槽位公共库
│   ├── sf_popup.js      # 浮动弹层公共三件套（attachPopupDismiss 外部点击/Esc/滚轮三关闭 + exempt 豁免；clampToViewport 四向钳位 + scale 边距折算；无 app 依赖可 .mjs 冒烟测试）——新弹层优先使用（见 experience/patterns.md §26）
│   ├── sf_crop*.js      # 可视化裁剪九模块（SFImageCrop/Uncrop：framework 编辑器框架 + core/panel/interaction/render/preview/undo_guard/alignments + 主扩展）
│   ├── sf_inpaint*.js   # 局部修复编辑器五模块（SFInpaintCrop/Stitch：core/paint/geometry/render + 主扩展）
│   ├── sf_pause_text*.js  # 文本闸门三模块（lib：state + applyGateMode prune 纯函数，四闸门共用；text 结构独立——keep 三态 + editedText 注入 + textarea 编辑器）
│   ├── sf_pause_kit.js    # 闸门共享引擎（image/mask/latent 三闸门共用：makeGateState state 工厂 + buildPauseBody 节点体 UI 工厂（CSS 类前缀参数化）+ definePauseGate 主扩展工厂（Copy/Open/Save 链路、双钩子 INJECT/PRUNE、executed 回填、buildClassNodeIndex/findNodeByPromptId 复合 id 图索引单源）；⚠ frameEventKey 等配置逐字对应 Python ui 键（image 是遗留键 "sf_pause_frame"），运行时属性名 _sfPauseXxx* 前缀派生不可改）
│   ├── sf_pause_image.js  # 图片闸门薄配置（快照/预览保存；调 definePauseGate）
│   ├── sf_pause_mask.js   # 遮罩闸门薄配置（灰度快照；调 definePauseGate）
│   ├── sf_pause_latent.js # latent 闸门薄配置（分段采样中间暂停，safetensors 快照，extraInputKeys:["image"]；调 definePauseGate）
│   ├── sf_outpaint*.js  # 外绘预览两模块（core 纯数学 + 主扩展）
│   ├── sf_image_resize*.js # wired 尺寸缩放三模块（复用 sf_load_image_resize.js 面板 + sf_load_image_ui.js）
│   ├── sf_find_replace*.js # 查找替换三模块（双端镜像 applyRulesJS ≡ Python _apply_rules）
│   ├── sf_dropdown*.js  # 值下拉四模块（lib/ui/settings/主扩展；输出点对齐双渲染器）
│   ├── sf_workflows*.js # 工作流面板三模块（主扩展/lib/UI，无节点设计）
│   ├── sf_lora_browser*.js # LoRA 浏览器三模块（主扩展/UI/lib，无节点设计：工具栏按钮紧贴 Workflows 按钮 + Alt+Shift+L + canvas 菜单；**文件夹/平面双模式**（seg 切换、模式记忆 sfnodes.LoraBrowser.Mode）——文件夹模式：面包屑 + 下钻 + 当前层文件（对齐 SF Load Image Browser）；平面模式：全部 LoRA 分批渲染 + 滚动动态加载（FLAT_STEP=60，attachFlatScroll 距底 300px 续批）防千级列表卡死；搜索两模式均跨层级扁平匹配；浏览位置记忆（sfnodes.LoraBrowser.Folder）；**网格/列表双视图**（seg 切换、记忆 sfnodes.LoraBrowser.View；列表行 = 40px 缩略图 + 文件名 + 目录/扩展名，单击/双击与卡片共用 attachPickAdd）；单击卡片打开 LoRA Stack 同款信息编辑（250ms 防抖区分双击）——复用 sf_lora_stack_info 宿主 ctx 入口 openInfoPanelFor；**双击用 SF LoRA Stack 加载到当前工作流**（三分支：无节点新建 / 单节点插入 / 多节点选择器显示 title+数量）；后端零新增全复用 /api/sfnodes/lora_*）
│   ├── sf_prompt_reader.js # 提示词恢复单模块（IN/OUT 目录切换）
│   ├── sf_prompt_list.js  # 行号编辑器单模块（SFPromptList：隐藏原生 multiline_text widget 作值真源 + DOM widget 行号栏从 0 起/跳过空白行对齐输出 index/超 500 行虚拟化，值恢复三通道；wrap 开启走镜像测量（mirror 与 textarea 同几何块级 div，行高按行缓存/宽度变化清空，渲染后强制重同步 scrollTop 防浏览器钳制错位）；start_index/max_rows 切片范围高亮跟随——仅裁剪时文本背景块+行号联动，wrap 开时高亮随测量行高展开（与行号同源））
│   ├── sf_prompt_stack*.js # 动态 Prompt 列表两模块（core 纯逻辑 + 行 UI，SFPromptStack 行动态添加/每条开关/右下角角标拖拽调行高 state.rows[i].h 随工作流保存）
│   ├── sf_text_preset.js  # 工作流绑定文本预设单模块
│   ├── sf_prompt_tags*.js # @tag 标签库七模块（lib/store/cursors/guard/editor/pinyin + 主扩展）+ prompt_tags_default.json 内置默认库
│   ├── prompt_preset.js   # 预设互斥联动/选中预设说明动态 tooltip
│   ├── sf_load_image*.js  # 加载图片四模块（SFLoadImageResize）+ load_images_path.js 渐进式目录浏览（SFLoadImagesPath 源切换 input/output/images + 面包屑/按需加载 + 直接输入路径）
│   ├── sf_lora_stack*.js  # 多行 LoRA 栈模块系列（core/api/render/interaction/dropdown/info/settings + 主扩展；info 面板经宿主 ctx 适配——openInfoPanel(node,id,refresh) 兼容入口保留，新增 openInfoPanelFor(ctx,id) 供 LoRA 浏览器等非节点宿主复用同一编辑面板；ctx 另支持 api 整束注入（路由域替换）/hideTriggers/samplesKind/autoCivitai，SF Load Diffusion Model 复用同一面板）
│   ├── sf_dmodel_api.js   # dmodel 域路由薄封装（与 sf_lora_stack_api.js 同形函数束 info/thumbUrl/civitai/saveDescription/savePreview/migrate/merge 等，URL 指向 /api/sfnodes/dmodel_*；事件 sfnodes.model-data-changed 与 lora 域隔离；导出 dmodelApi 整束供面板 ctx.api 注入——键名错会静默回退 LoRA 路由，tests/test_load_dmodel_panel_smoke.js 锁定契约）
│   ├── sf_load_diffusion_model.js # SF Load Diffusion Model 单模块（i 信息图标复用 sf_lora_info.js 的 setupLoaderInfoWidget 工厂（prefetch:null/hasCustomOf/onOpen 注入）+ dmodelPanelCtx 宿主适配（api/hideTriggers/samplesKind/autoCivitai 四件套），isGraphLoading 门控点击）
│   ├── sf_lora_plot.js    # 批量对比节点单模块（SFLoraPlot：行 UI 全复用 stack 的 core/api/dropdown/菜单/CSS）
│   ├── sf_lora_info.js    # LoRA 信息对话框（SFLoraLoader/SFLoraLoaderModelOnly 共用，sf_markdown.js 渲染描述；createInfoWidget/setupLoaderInfoWidget 参数化工厂导出供非 LoRA 加载器（SF Load Diffusion Model）复用图标绘制与 configure 时序）
│   ├── sf_lora_shared_info.js # 样例图网格/预览/hover/markdown 复用内核（Stack 面板与 info 对话框共享；loadWorkflowFromImageUrl(url) PNG 内嵌工作流通用载入——readPngWorkflowData 前端 chunk 解析 + Comfy.NewBlankWorkflow 新标签，loadImageAsWorkflow 是 lora_samples 路径薄包装，image_browser 经 /view 复用；attachSamplePromptCopyButtons(container,notify) 描述内 civitai 样例 prompt 代码块右上角常驻复制按钮——h3 紧邻 pre 判定 + copyText/injectCSSOnce 复用 sf_common）
│   ├── sf_markdown.js     # Markdown 渲染纯模块（无 app 依赖，纯模块边界成员——不得 import sf_common）
│   ├── sf_lora_preset.js # 预设选择节点前端（原 power_lora_preset.js 改名，SFLoraPreset）
│   ├── sf_krea2_presets.js # Krea2 预设管理共享模块（Interrogator/SystemPrompt 共用：API 封装 + combo 动态重建 + 节点"管理预设"按钮 + 管理 popup，复用 sf_popup.js；改动派发 sfnodes.<kind>-presets-changed 事件）
│   ├── sf_regional_lora*.js # 多区域 LoRA 两模块（SFRegionalLoRA：lib 纯函数 + 主扩展，DOM canvas 多 box 拖拽/8 向 resize/画新框/背景图对齐，隐藏 SFRegionsJson widget 真源，行控件 enable/lora/strength/remove）
│   ├── sf_styles_selector*.js # 风格选择器两模块（SFStylesSelector：lib 纯函数 + 主扩展，标签多选列表搜索/清空/选中置顶/hover 缩略图，隐藏 SFStylesState widget 真源，DOM widget 纯交互不承担值传输）
│   └── 其余单节点 JS（text_replace/text_concatenate/simple_math/loop_flow/any_pack/image_browser（SFLoadImageBrowser 弹窗浏览器：缩略图网格/面包屑/排序/删除，图片右键菜单——复制正向提示词走 /api/sfnodes/prompt_reader/extract、载入工作流经 loadWorkflowFromImageUrl+/view 原始字节）/lora_loader*/lora_loader_model_only/multi_lora_tree/image_compare/image_concatenate/regex_extract/prompt_batcher/empty_latent_ratio/krea2_*/seed/canvas_size/workflow_name/sf_combo_selector/sf_color_picker/showcontrol/DisplayText/SFLogicSwitch/...）
├── data/                # 静态数据（anime_char CSV、face_distance 字体、prompt_presets.json 提示词预设、styles/fooocus_styles.json 内置风格库 + samples/ 缩略图等）
├── tests/               # 前端/后端模拟测试（Node/Python 直接运行，无测试框架）
└── doc/                 # 项目文档（vibecoding.md 开发流程、experience/ 历史经验归档目录等）
```

## Key Dependencies (runtime only, do NOT install)

- `torch`, `torchvision` — 张量运算（由 ComfyUI 运行时提供，不在 `requirements.txt` 中声明）
- `opencv-contrib-python` — 图像处理
- `insightface`, `onnxruntime` — 人脸分析
- `mediapipe` — 人像分割
- `kornia` — 图像变换
- `color_matcher` — 色彩匹配
- `colour-science` — 色彩科学/LUT 处理
- `translators` — 文本翻译
- `scipy`, `aiohttp`, `safetensors`, `tqdm`, `requests`, `typing_extensions`
- `curl_cffi`, `markdownify` — Civitai 页面抓取（TLS 指纹过 Cloudflare）与 HTML 描述清洗（lora_routes/lora_reader）
- `psutil` — 系统资源监控（内存清理节点使用）
- `sageattention` — 注意力优化
- `diffusers`, `einops`, `timm`, `huggingface_hub` — 图像模型/扩散相关（RFMSR 等）

## ComfyUI API Imports (for reference only)

以下模块在运行时由 ComfyUI 提供，可通过源码 `../..` 查阅实现：

- `comfy.utils` — 通用工具（缩放、文件加载等）
- `comfy.utils.common_upscale` — 图片缩放
- `comfy.utils.ProgressBar` — 进度条
- `comfy.model_management` — 显存/设备管理
- `comfy.comfy_types.node_typing.IO` — 类型注解
- `comfy.sd` — 模型加载（load_lora_for_models 等）
- `nodes.LoadImage`, `nodes.SaveImage`, `nodes.MAX_RESOLUTION` — 内置节点
- `nodes.LoraLoader` — LoRA 加载节点
- `nodes.NODE_CLASS_MAPPINGS` — 全部节点映射（含自定义节点；**运行时才包含全部，函数内 import 最安全**）
- `folder_paths` — 路径管理
- `comfy_extras.nodes_post_processing` — 后处理节点
- `comfy_execution.graph_utils` — `GraphBuilder`（图展开）、`is_link`、`ExecutionBlocker`（官方位置，graph.py 只是 re-export）
- `comfy_execution.graph` — `DynamicPrompt`（DYNPROMPT 隐藏输入对象：`get_node`/`get_display_node_id`/`get_original_prompt`，支持 ephemeral 前缀 id）
