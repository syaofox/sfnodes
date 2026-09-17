# nodes-video.md — 视频与视频生成节点

> 所含章节：§72 SCAIL-2 四节点复刻（ComfyUI-SCAIL2-Easy）· §77 SAM3 视觉点选追踪与 track_data 排除（驱动遮罩排男/阴茎/精液）· §78 SCAIL-2 上下文窗口参数对齐原生 WanContextWindowsManual + 单图负向条件修正 · §82 SCAIL-2 预处理 O(T) 内存分块（运行时补丁 + 设置）· §87 SCAIL-2 外部分段处理（VHS 分段循环 + 外锚接续 + SFVideoConcat 合并）。本文件为 `doc/experience/` 第七个主题（2026-09）：视频生成节点族（SCAIL-2 / Wan）不与 platform / patterns / nodes-text / nodes-image / nodes-lora / apps 适配，故新建。

## 72. SCAIL-2 四节点复刻（ComfyUI-SCAIL2-Easy，2026-09）

### 72.1 来源与范围

复刻 `ComfyUI-SCAIL2-Easy`（Apache-2.0）的 4 个节点，**不重写 SCAIL-2 推理**，只复用 ComfyUI 原生 SCAIL-2 支持做编排封装：

| 源节点 | 本包节点 | 作用 |
|---|---|---|
| `SCAIL2FitVideo` | `SFSCAIL2FitVideo` | 输入视频尺寸统一（512p/704p 短边缩放，custom 直取，均对齐 32） |
| `SCAIL2ReferencePack` | `SFSCAIL2ReferencePack` | 多主体主图/补充参考图/场景图 → `SCAIL2_REFERENCE_PACK` dict |
| `SCAIL2ReferenceSAMBuilder` | `SFSCAIL2ReferenceSAMBuilder` | 对参考包每张图跑 SAM3_VideoTrack，生成彩色参考蒙版回填 |
| `SCAIL2SimpleVideo` | `SFSCAIL2SimpleVideo` | 主生成：animation/replacement + 长视频 chunk 接续 / context_sampling |

完整 1:1 移植（含 color_correction、旧版 `subject_N_image_M` 输入迁移、多主体拼图布局、上下文采样补丁）。

### 72.2 复用与去重（禁内联副本）

- 原生核心全部函数内 import：`comfy_extras.nodes_scail` 的 `WanSCAILToVideo` / `SCAIL2ColoredMask` / `_extract_mask_to_28ch`、`comfy_extras.nodes_sam3.SAM3_VideoTrack`、`comfy.context_windows`、`node_helpers.conditioning_set_values*`、`comfy.utils.common_upscale`、`nodes.VAEDecode` / `nodes.CLIPVisionEncode`、`comfy_extras.nodes_custom_sampler.SamplerCustom`。
- 源 `_extract_mask_to_28ch` 自带一份本地 fallback 实现（native 缺失时用）——本环境的 `nodes_scail` 已确认提供该函数，**删掉内联 fallback**，只保留薄包装 + 缺失即抛错。
- 源 `_sample_and_decode` / `_reference_pack_clip_image` / `_reference_pack_primary_image` 是定义后从未调用的死代码，未移植。
- 纯逻辑（常量、帧数 `4n+1` 数学、32 对齐、Fit Video 尺寸、多主体拼图布局）抽到 `sf_utils/scail2_easy.py`（**无 torch / ComfyUI 依赖**），节点层只留张量运算与原生编排。

### 72.3 类型串与兼容

- 自定义类型串**保持原名** `SCAIL2_REFERENCE_PACK`（源同名字符串不注册不冲突），与核心 `SCAIL2ColoredMask` 生态、原插件产物互通。
- 节点 ID / 显示名按项目规范加 `SF` 前缀（`SFSCAIL2*`），CATEGORY `sfnodes/video`；因此**源插件工作流的节点映射不自动迁移**（节点类型不同），但 ReferencePack 数据形状一致。
- 源 `requirements.txt` 的 `decord/easydict/imageio-ffmpeg/pillow/safetensors` 经 grep 确认代码完全未使用，未引入，无新增依赖。

### 72.4 context_sampling 上下文窗口补丁

长视频 `context_sampling` 模式在 `comfy.context_windows` 的窗口调度下采样。核心 `IndexListContextHandler` 只按 `index_list` 切普通条件，不会切 SCAIL-2 的两类 28ch 蒙版：

- `_patch_scail_forward()`：给 `SCAILWanModel._forward` 打**一次性幂等** patch（带 `_SCAIL_PATCHED` 全局守卫 + `_sf_scail2_context_patch` 标记），在每窗口把 `driving_mask_28ch` / `ref_mask_28ch`（及 `_latents` 变体）按同一 index 列表切片；`ref_mask_*` 额外保留前 `reference_frames` 帧（参考段不参与时间采样）。
- `SCAIL2EasyContextHandler.get_resized_cond`：覆写以同步处理 `model_conds` 里的蒙版。
- 时间维判定 `_mask_time_dim`：`shape[1]==28` → dim 2；`shape[2]==28` → dim 1；ref 偏好更大的维 1。

⚠ 仅 `context_sampling` 触发，chunk 模式不打 patch；补丁只在首次调用时安装。

### 72.5 前端动态槽与显隐（web/sf_scail2.js）

- 源文件的双语探测（`isChineseLocale` / `EASY_TRANSLATIONS.zh|en`）按项目约定收敛：**仅 widget 标签中文化**（`SCAIL2_LABELS`），combo 选项值保持英文原文（曾用 `SIMPLE_COMBO_LABELS` + `getOptionLabel` 汉化选项值，已按需求移除）；`applyLabel` 同时写 `label/localized_name/display_name` 与 `options`/`_state`（Vue 前端渲染读 `label ?? localized_name ?? name`）。
- Reference Pack 是**数量驱动**动态槽（`subject_count`/`reference_count` 计数框），与项目 `sf_dynamic_slots.js` 的**连线驱动**语义不同，未复用。流程：`beforeRegisterNodeDef` 里 `trimReferencePackNodeDataInputs` 把 schema 可选输入裁到 `subject_1_image`+`scene_image` → 计数框 callback 触发 `updateReferencePackWidgets`（旧版迁移 → 期望列表签名 → 重建输入槽 → widget 顺序同步 → fitNode）。
- 显隐工具（`storeWidgetDefaults` / `setWidgetVisible` / `hideComputeSize`）项目内无共享实现，本模块内实现（不擅自改 `sf_common.js`）：隐藏时 `type="hidden"` + `options.hidden/canvasOnly=true` + computeSize 归零，恢复时按存储的 defaults 还原并同步 `_state`。
- `SFSCAIL2SimpleVideo` 的 `widgets_values` 是**位置敏感**的，源提供的 `repairSimpleVideoWidgetOrder` 在 `onConfigure` 时按 mode 值定位并纠正 `advanced`/`long_video_mode` 顺序颠倒，保留。
- 4 节点的**全部输入/选项在 Python `INPUT_TYPES` 内联中文 `tooltip`**（含 combo 的 512p/704p/custom、模式、长视频方式说明与旧版迁移输入），`tests/test_scail2.py` 断言每个输入都有非空 tooltip，防止后续新增输入漏写。

### 72.6 解码显存开关（tiled_decode）

- 编码/解码默认均为**非 tiled**（`vae.encode` + `nodes.VAEDecode`），分块只分采样：长视频/高分辨率的显存峰值主要在整段解码与参考图全图编码。
- 仅提供**低风险可选项** `SFSCAIL2SimpleVideo.tiled_decode`（BOOLEAN，默认关，随 `advanced` 显示）：开启后 `_decode_latent_to_frames` 改走 `nodes.VAEDecodeTiled`（tile 512 / overlap 64 / temporal 64 / temporal_overlap 8，即原生默认），降低解码显存峰值、速度略慢。
- 编码侧 tiled（参考图 / pose）**不做**：Wan VAE 的时空压缩在 tiled 边界可能产生伪影，且 `reference_latents` 要求逐帧单 latent，风险高。
- `tiled_decode` 已贯通四个 decode 调用点（native chunk / multi-ref chunk / context）并写入各层 summary；`tests/test_scail2.py` 用 fake `nodes` 模块断言 plain/tiled 分支与参数。

### 72.7 测试与回归

- `tests/test_scail2.py`：纯逻辑断言（`4n+1` cover/floor、32 对齐、Fit Video 尺寸、色板、布局候选）+ 节点结构元数据 + 根 `__init__.py` 4 键各出现两次。mock `torch` 顶层符号，stub `sfnodes` 包结构使相对导入可解析。
- `tests/test_scail2_js.js`：mock node + `new Function` 剥离 import/export，覆盖扩展注册、数量驱动槽重建、旧版迁移、Fit Video 显隐、Simple Video 排序/错位修复、`setWidgetVisible` 往返、nodeData 裁剪。
- 已纳入 AGENTS.md 三条快速回归循环。

## 77. SAM3 视觉点选追踪与 track_data 排除（驱动遮罩排男/阴茎/精液，2026-09）

> 背景：SCAIL-2 "简单版"用 SAM3 文本 prompt `woman` 追驱动视频女性，但 SAM3 常把阴茎（进入她体内/与之相连的区域）与精液并入女性轮廓。`driving_track_data` 会被渲染成 28ch "身份区"条件，模型于是把这些男体/体液形状当成角色本体重绘。需求：**纯 SAM3 视觉点选**（不引入 SeC）把男/阴茎/精液从驱动遮罩中排除，参考路不动。

### 77.1 两个核心限制决定节点形态

- 核心 `comfy.ldm.sam3.tracker.track_video` 的 `initial_masks` **只对应第 0 帧**（`detector.forward_video` 文档 `[N_obj,1,H,W] binary masks for first frame`）。**不能中途锚定**：精液等中途出现的目标必须从出现帧起锚定，否则无从 seed。
- `MaskComposite(subtract)`（§69）要求两路帧数一致。若直接从出现帧切片追踪，输出会短于基础序列 → 帧错位。
- 故新增 `SFSAM3PointTrack`：在节点内完成"锚帧检测 + 切片传播 + **前补空帧回全长**"，使输出帧数与输入驱动序列一致；排除做成 `SFTrackDataSubtract` 在 track_data 层一步完成。

### 77.2 SFSAM3PointTrack（`nodes/video/sam3_point_track.py`，CATEGORY `sfnodes/video`）

- 输入：`images` / `model`（SAM3.1）/ `anchor_frame` + 可选 `positive_coords`、`negative_coords`（PointsEditor 输出，`forceInput`）、`initial_mask`、`refine_iterations`。
- 流程：`images[anchor:anchor+1]` → 核心 `SAM3_Detect.execute`（点提示 union；接了 `initial_mask` 则跳过）→ `SAM3_VideoTrack.execute(images[anchor:], initial_mask=…)` → `pad_track_data_front` 前补 `anchor_frame` 帧空 mask。
- **复用核心节点**：不重写检测/传播，只用 `_node_result`（同 `scail2.py`）解包 `io.NodeOutput`；`conditioning=None`（纯视觉，不做文本检测）。
- 2D `initial_mask` 自动升维（核心要求 `[N,H,W]`，`unsqueeze(1)` 后成 `[N,1,H,W]`）。
- 空点提示守卫：PointsEditor 未点选时输出 `"[]"`（非空字符串，truthy），故 `_has_points` 解析 JSON 判空——无有效点且无 `initial_mask` 时抛明确错误，避免空 seed 传播出无意义结果。

### 77.3 SFTrackDataSubtract（`nodes/image/track_data_subtract.py`，CATEGORY `sfnodes/image`）

- track_data 层逐帧**逐对象**相减并**保留对象数**（不塌成单身份）；`exclude_1..4` 为 `"MASK,SAM3_TRACK_DATA"` 多类型，未连接跳过。
- 排除输入可为 `MASK`（`[T,H,W]`/`[H,W]`/`[T,1,H,W]`）或 `SAM3_TRACK_DATA`；各路先并集再 `base & ~exclude`。
- **帧数必须一致**（不广播，冲突即报错）；尺寸不一致走 `nearest` resize；基础为空/无有效排除浅拷贝直通。

### 77.4 纯逻辑单源（`sf_utils/track_data_ops.py`）

- `pad_track_data_front(track_data, total_frames, torch)`：`packed_masks is None` 仅补 `n_frames`，否则 `torch.cat` 前置零帧。
- `subtract_from_track_data(track_data, exclusion_masks, pack_masks, unpack_masks, torch, interpolate)`：与 `invert_track_data` 同构（依赖注入），位打包复用核心 `pack_masks/unpack_masks`，**不内联副本**。节点只做核心符号 import 与参数注入。

### 77.5 为什么不用现成节点拼

`SAM3_TrackToMask → MaskComposite(subtract) → SFMaskToTrackData` 也能相减，但：① 多对象基础 track 会塌成单一身份；② 需手工保证两路尺寸/帧数对齐；③ 中途锚定仍需切片+补帧的胶水。节点把这三件事收敛为可测纯函数 + 两个薄节点。

### 77.6 工作流接线（SCAIL-2 简单版）

- 基础女性：`SAM3_VideoTrack(node32) → SFTrackDataCache → SFTrackDataSubtract.track_data`。
- 男/阴茎：`PointsEditor(驱动首帧) → SFSAM3PointTrack(anchor_frame=0) → exclude_1`。
- 精液：`PointsEditor(精液出现帧) → SFSAM3PointTrack(anchor_frame=出现帧) → exclude_2`。
- 输出 → `SFSCAIL2SimpleVideo.driving_track_data`；参考路（`node 33/405`）不动。参考路同样可配一组（`参考追踪 → SFTrackDataCache → SFTrackDataSubtract → reference_track_data`）。
- **排除追踪也缓存**：`SFTrackDataCache` 包在 `SFSAM3PointTrack` 与相减之间（`tracker → cache(lazy) → subtract.exclude_*`），命中即跳过「检测+传播」。缓存键 = 名称 + `signature`(PointsEditor 正点|负点，经 `SFTextConcatenate` 拼接) + `source` 首末帧哈希；**中途锚定路（精液）的 `source` 接 `ImageFromBatch` 输出**，于是改 `batch_index` 会自动失效——但仅改 `anchor_frame` 不动 `batch_index` 时需把 cache 的 `force` 拨 recompute（`anchor_frame` 是 widget，不在键里）。**静音对象是 cache 而不是 tracker**：muted cache 的输入（tracker）同样不被调度，避免「cache 未命中→向 muted tracker 取 None」的报错；unmute 后 miss 才触发追踪。
- **两个 `SFSAM3PointTrack` 默认静音（mode 2）**：KJNodes `PointsEditor` 在画布无点时 `pointdata` 直接抛 `No points on the canvas...`（它自身行为，节点侧无法兜底），且它无条件执行会中断整个 prompt。静音后下游 optional 输入为 `None` → `SFTrackDataSubtract` 直通；用户点选后再 `Ctrl+M` 启用即可（静音节点的上游 PointsEditor 不被调度，故不会执行报错）。

### 77.7 边界与未验证项

- SAM3 视觉点选对 NSFW 解剖/体液（阴茎/精液）的分割精度取决于模型本身，**需实跑验证**；本框架只保证"给定点/遮罩后的传播与相减对齐"稳定。
- `anchor_frame` 之前的帧恒为空（不排除）；精液消失后追踪可能残留，必要时把 `anchor_frame` 后段再切分处理。

### 77.8 测试

- `tests/test_track_data_ops.py`：numpy 严格位序 pack/unpack 桩；覆盖前补帧/已足够长/packed None、逐对象相减、多路并集、MASK 与 TRACK_DATA 两种排除、resize、4D 多通道报错、帧数不一致报错、空排除直通、execute 集成 + 双字典注册一致。
- `tests/test_sam3_point_track.py`：stub `torch` + `comfy_extras.nodes_sam3`；覆盖锚帧切片、点提示透传、anchor>0 前补空帧回全长、`initial_mask` 跳过检测、2D 升维、缺提示/越界/非 IMAGE 报错。

## 78. SCAIL-2 上下文窗口参数对齐原生 WanContextWindowsManual + 单图负向条件修正（2026-09）

> 需求：参考原生 `WanContextWindowsManual`（`comfy_extras/nodes_context_windows.py`）审视 `SFSCAIL2SimpleVideo` 的 `context_sampling`。此前窗口参数全部硬编码在 `sf_utils/scail2_context.py::apply_scail2_easy_context`，逐项对照后只补真正有差距的项。

### 78.1 对照与取舍

| 原生 Wan 参数 | 原生默认 | SF 原状 | 结论 |
|---|---|---|---|
| `context_schedule` | `standard_uniform` | 硬编码 `standard_static` | 新增 widget（4 档 `ContextSchedules`，默认仍 `standard_static` 保存量行为） |
| `freenoise` | True（并挂 sampler wrapper） | 硬编码 False、无 wrapper | 新增 widget（默认关；开启补挂 wrapper） |
| `context_stride` | 1（仅 uniform 系生效） | 硬编码 1 | 新增 widget（默认 1；仅 `standard_uniform`/`looped_uniform` 显示） |
| `closed_loop` | False（仅 looped 生效） | 硬编码 False | 新增 widget（默认关；仅 `looped_uniform` 显示） |
| `fuse_method` | pyramid | pyramid | 与原生默认一致，未暴露 |
| `retain_first_frame`（`cond_retain_index_list="0"`） | False | 不适用 | 明确不做，见 78.2 |
| `split_conds_to_windows` | False | 不适用（无多区域条件） | 明确不做 |

- `context_schedule` 全部枚举走核心 `comfy.context_windows.get_matching_context_schedule`（常量表 `CONTEXT_SCHEDULES` 放 `sf_utils/scail2_easy.py` 无依赖单源）；`updateSimpleVideoWidgets` 仅在 `advanced && long_video_mode=context_sampling` 显示，combo 选项值（`standard_static`/`standard_uniform`/`looped_uniform`/`batched`）直接显示英文原文（曾汉化，已移除）。
- `context_stride`/`closed_loop` 由前端 `simpleVisibleAdvancedWidgets()` **按调度条件显隐**（与原生 tooltip 语义一致，避免设置无效项）：stride 仅 uniform 系（`standard_uniform`/`looped_uniform`）显示、closed_loop 仅 `looped_uniform` 显示；`context_schedule` 的 callback + `onWidgetChanged` 触发刷新。后端 `apply_scail2_easy_context` 对 stride 做 `max(1, int(...))` 夹取并透传（`closed_loop` 仅 loop 调度有意义，非 loop 传入由核心窗口生成器自然忽略）。
- 新增 widget **追加在 Python `required` 末尾（`tiled_decode` 之后）与 JS `SIMPLE_WIDGET_ORDER` 末尾**：SimpleVideo 的 `widgets_values` 位置敏感，追加式新增使旧工作流按位对齐不受影响（旧数组短于新 widget 数 → 新 widget 取默认值）。
- `freenoise=True` 必须补调 `comfy.context_windows.create_sampler_sample_wrapper(model)`（核心注释：该 wrapper 目前仅 freenoise 使用；缺失即抛明确错误）。FreeNoise 只扰动噪声本身，与 SCAIL 蒙版按 `index_list` 切片正交。

### 78.2 retain_first_frame 对 SCAIL-2 无效的原因

- 核心 `WAN21_SCAIL2.resize_cond_for_context_window`（`comfy/model_base.py`）对 `sam_latents`（驱动蒙版，来自 `driving_mask_28ch`）/`pose_latents` 走 `slice_cond`（显式忽略 retain），`ref_mask_latents` 走"取前 N 帧 + 零填充"分支；`cond_retain_index_list`（`retain_first_frame`）只对与 x 同时间长的通用 cond 生效。
- SCAIL-2 的"首帧/起始内容"在 `reference_latents` 与两类 28ch 蒙版里，且参考帧本就整段保留给每个窗口（不在窗口时间采样范围内），故无需也不应再 retain。

### 78.3 核心已原生切 SCAIL 蒙版（兼容垫片勿删）

- `WAN21_SCAIL2.resize_cond_for_context_window`（2026-06 SCAIL-2 支持）已原生处理 `sam_latents`/`ref_mask_latents` 的逐窗口切片；`extra_conds` 把 `driving_mask_28ch`/`ref_mask_28ch` 映射为 `sam_latents`/`ref_mask_latents`。
- `sf_utils/scail2_context.py` 的 `_patch_scail_forward`（按 kwargs 旧键名 `driving_mask_28ch` 等查找）与 `SCAIL2EasyContextHandler.get_resized_cond` 里的 `_resize_scail_model_conds`（按 model_conds 旧键名查找）在当前核心上均为**幂等空转**。
- **保留原因**：运行实例（docker）核心版本可能早于该支持，垫片是兼容回退且无副作用；**勿因"看似无用"删除**。删除前先用 §77 式诊断核实运行实例的 `comfy/model_base.py` 是否含 `WAN21_SCAIL2.resize_cond_for_context_window`。

### 78.4 单图负向条件修正

- 单图（非 Reference Pack）路径 `_set_scail_single_reference_conditioning` 的 negative 原为 `torch.zeros_like(ref_latent)`：与核心 `WanSCAILToVideo`（`nodes_scail.py` 正负同 `reference_latents`）及 SF 自身多参考 Pack 路径（`_set_scail_reference_pack_conditioning`）均不一致；`cfg>1` 时负向会拿黑参考帧做引导（`cfg=1.0` 无影响）。
- 现改为正负共享同一 `ref_latent`；`tests/test_scail2.py` 以 stub `node_helpers.conditioning_set_values` 记录调用并断言正负 `reference_latents` 同值，锁死回归。

### 78.5 测试与回归

- `tests/test_scail2.py` 新增：`CONTEXT_SCHEDULES` 常量表；新 widget 的存在/默认值（schedule/freenoise/context_stride/closed_loop）；stub `comfy.context_windows` 断言 handler kwargs（schedule/fuse/freenoise/stride/closed_loop/dim）、stride 夹取、`create_sampler_sample_wrapper` 仅 freenoise 时调用、overlap 越界抛错、正负同 ref。
- `tests/test_scail2_js.js` 新增：chunk 模式隐藏 / context 模式显示上下文 widget；`context_stride` 随 uniform 系显示、`closed_loop` 仅 looped 显示（static 下两者隐藏）；中文标签；排序末尾追加且与 Python `required` 同序（`context_schedule` → `freenoise` → `context_stride` → `closed_loop`）——顺序必须一致，否则旧工作流 `widgets_values` 末位会错位（`freenoise` 的值会落到 `context_stride`）。

## 82. SCAIL-2 预处理 O(T) 内存分块（运行时补丁 + 设置，2026-09）

### 82.1 问题：整段预处理的 O(T) 峰值

原生工作流（`WanSCAILToVideo` + `WanContextWindowsManual`，如 `[scail2]SCAIL-sam3.1-原生(排除).json`）把 `VHS_LoadVideo.frame_count` 直连 `length`，**一次前向生成整段视频**。采样显存本身对总帧数不敏感（上下文窗口把每次前向限制在 `context_length` 帧，`estimate_memory` 对 `sam_latents/pose_latents` 的时间维按 `memory_usage_shape_process` 的 1.5 帧估算），但核心 `comfy_extras/nodes_scail.py` 的两个纯函数按整段 T 构造中间张量：

- `_render_colored_masks`：`unpack_masks` → `[T,N,H,W]` f32 + `color_overlay` `[T,H,W,3]` f32，2500 帧 @704×1280、N=1 时约 30–55 GB。
- `_extract_mask_to_28ch`：7 通道阈值图 `[T,7,H,W]` f32，同尺寸约 16 GB。

`intermediate_device()` 默认 CPU（`--gpu-only` 时 GPU），故这些是系统内存峰值；`--gpu-only` 下直接爆显存。此外 `WanSCAILToVideo.execute` 对整段 `reference_image_mask` 做 `common_upscale`（`[T,3,H,W]` f32，可数十 GB），而实际只用前 `n_ref` 帧。

### 82.2 机制：分块包装 + 逐元素等价

`sf_utils/scail2_mem.py` 给核心函数打一次性幂等 patch（`_MARK` 守卫，对齐 `scail2_context.py`），不修改核心文件、不改工作流接线：

- **渲染分块**：按 `packed_masks[start:end]` 逐块执行原生表达式，写入预分配 `out[T,H,W,3]`。等价依据：`any(dim=1)`/`argmax(dim=1)`/`interpolate(mode="nearest")`/`where` 与背景色广播**逐帧独立**，不存在时间维耦合。
- **28ch 分块**：按输入帧逐块算 7 通道阈值图并做**仅空间**下采样（`mode="area"` 逐 `(t,c)` 独立），再按原生时间打包顺序写 `padded`：`padded[0:4]=帧0`、`padded[3+j]=帧j (j≥1)`，最后 `.view(T_latent,28,...)`。要求 `T ≡ 1 (mod 4)`（核心 `view` 前置条件，调用方先截断到 4n+1），非该形态走原生。
- **参考蒙版裁剪**：包装 `WanSCAILToVideo.execute`（执行引擎走 `getattr(type_obj,"execute").__func__(cls, **inputs)`，`classmethod` 包装可安全收 kwargs），仅当 `reference_image_mask` 帧数 > `reference_image` 批数时取前 `n_ref`。等价依据：原生 `ref_mask_hw` 只用 `[:1]` 与 `min(i, n_masks-1) for i in range(n_ref)`，当 `n_masks ≥ n_ref` 时恒为前 `n_ref` 帧（`n_masks < n_ref` 时广播最后一帧，故只在此情形才裁剪）。
- **短输入/异常回退**：`T ≤ chunk`、空追踪或分块路径抛异常时一律回退原生函数并告警，短工作流零变化；`Enabled=False` 同。

### 82.3 设置与 dtype

`sfnodes.SCAIL2Mem.{Enabled(默认开), ChunkFrames(默认 32), HalfPrecision(默认开)}`，前端 `web/sf_scail2_mem_settings.js` 自注册（`app.registerExtension` + `addSetting`，独立于 SF 节点），后端每节点执行时经 `read_options()` 读 `comfy.settings.json`（复用 `sf_utils/llm_client.read_comfy_settings`，唯一读取实现），改动即时生效无需重启。`HalfPrecision=True` 时彩色蒙版输出 f16（值仅 0/1，与 f32 逐元素等价，仅省一半常驻）；下游原生 `common_upscale→F.interpolate`（`area`/`nearest-exact`）需构建支持 CPU half，报错时把开关关掉即可。`ChunkFrames` 与后端设置同名同默认值（前端 slider / 后端每调用读盘），避免双默认漂移。

### 82.4 残留与边界

- 分块只压中间峰值，**输出本身仍 O(T)**：`pose_video_mask`（f16 下 ~13.5 GB @704）、`reference_image_mask`（渲染在追踪分辨率如 512²，~7.9 GB）；VHS 载入与 VAE 解码各 ~13.5–27 GB。
- 彻底支持 2500+ 帧需配合分块生成（`SFSCAIL2SimpleVideo` 的 `long_video_mode=chunk`，见 §72）；或在工作流层面把参考视频经 `SFImageBatchIndex`/`ImageFromBatch` 取单帧再送 `reference_image_mask`。
- `_extract_mask_to_28ch` 的输入 `[T,3,H/2,W/2]` 上采样仍在 `execute` 内整段分配（~6.8 GB），本轮未触及（须重写 `execute`）。

## 87. SCAIL-2 外部分段处理：VHS 分段循环 + 外锚接续 + SFVideoConcat 合并（2026-09）

### 87.1 背景：§82 只压显存，RAM 仍是 O(T)

`SFSCAIL2SimpleVideo` 的 `long_video_mode=chunk` 把**采样显存**压到 O(chunk_frames)，但整条链路的系统内存仍随总帧数线性增长，且都在进节点之前就发生：

| 项 | 量级（4492 帧 @1280×896 f32 实测工作流） |
|---|---|
| `VHS_LoadVideo`（`frame_load_cap=0`）整段加载 | ≈ 62 GB 常驻 |
| replacement 模式整段 `pose_video_mask`（f16，§82 补丁输出） | ≈ 31 GB |
| 节点 `stitched` 累积输出 + 最终 `torch.cat`/`clamp` 瞬时峰值 | ≈ 62 / 最高 ~185 GB |
| `VHS_VideoCombine` / 输出缓存 | 再一份 |

结论：**要支持任意长度，必须"分段加载 → 分段生成 → 分段落盘 → 最后文件级合并"**，把每轮工作集压到 O(段长)。本节方案已落地（`previous_frames` 外锚 + `SFTrackDataSlice` + `SFVideoConcat`）。

### 87.2 分段循环接线（用户工作流层）

```
循环外: 画幅子图 ──width/height──> VHS_LoadVideo(custom_w/h)     # 分辨率仍由现有子图控
        整段追踪子图 → SFTrackDataCache → track_data ─┐
        VHS_LoadVideo(audio) ──────────────────────┐ │
SFForLoopStart(total)                              │ │
  index → SFMathInt(multiply, step) → skip          │ │
  VHS_LoadVideo(video, force_rate, skip, cap=L) → 段帧
  SFTrackDataSlice(整段track, start=skip, len=L) ←─┘ │   # 循环外输入，每轮复用
  previous_frames ← 循环状态（首轮 None）
  SFSCAIL2SimpleVideo(pose_video=段帧, driving_track_data=段track,
                      previous_frames, chunk_frames=81, overlap=5) → 段输出
  ImageFromBatch(-5, length=5) → 尾帧 → 循环状态
  VHS_VideoCombine(段输出, fps, prefix="…/seg_") → VHS_FILENAMES
      → SFBatchAnything(累加) → 循环状态
SFForLoopEnd
循环外: SFVideoConcat(segments=文件列表, audio=VHS.audio, prefix="…/final")
```

- **skip/cap 转输入**：VHS 这些 INT widget 可直接 Convert to Input，由 `SFMathInt` 驱动（VHS 无额外校验，只要求非负）。
- **段长参数**：取 `L = overlap + 76k` 且 `L ≡ 1 (mod 4)`（默认 **461 = 5 + 76×6**，与内部 chunk 81/步长 76 完全对齐，段内无残段）；`step = L - overlap`（461→456）；**首段** `skip=0`，其后 `skip = index × step`；`total = (N - overlap - 1) // step + 1`（N=源总帧数；也可手填）。每段输入 cap=L 且第 i≥1 段的段首 overlap 帧正是上一段尾部输出（重叠输入 + 外锚丢弃 = 无缝）。
- **循环状态**只放「上一段尾 overlap 帧」与「文件列表」，绝不能把整段帧放进状态槽（会跨轮持有，RAM 又变 O(T)）。
- `SFBatchAnything` 对 `VHS_FILENAMES` 是 `tuple(bool, list)`，两个 tuple 相加会拼成 `(bool, list, bool, list…)` 混合结构——`SFVideoConcat` 内部递归展平，无需前置转换。

### 87.3 外锚 previous_frames 语义（跨段无缝的关键）

`SFSCAIL2SimpleVideo` 新增 optional `previous_frames`（IMAGE）。实现要点（`nodes/video/scail2.py` + `sf_utils/scail2_easy.py`）：

- **归一**：`normalize_external_anchor(输入帧数, previous_frame_count)` → 0（不锚，overlap 关或空输入）或 `1..previous_frame_count`，实际取输入尾部这些帧。
- **偏移起算**：`video_frame_offset` 初值 = 实际锚帧数。原生 `WanSCAILToVideo` 内部会把传入偏移**再减 `previous_frames` 帧数**（`max(0, ·)`）后切片并编码为 latent 前几帧（`noise_mask=0`），所以外锚与内部 chunk 衔接走的是**同一条核心路径**：段首 overlap 帧由锚 latent 占据、不重新生成。
- **首段也丢弃**：`chunk_discard_head(chunk_index, previous_frame_count, external_anchor_frames)`：`chunk_index>0` 恒丢 `previous_frame_count`；首段丢**实际锚帧数**；无锚首段丢 0。丢弃后输出与输入段的 pose 帧一一对齐（段输出 = 输入段去掉重叠前缀）。
- **尾帧自然截断**：循环内 `length <= previous_frame_count` 时 break，故每段末尾至多丢 overlap 帧（末段不足时丢尾），总输出 = `N - 丢弃`（对齐 4n+1 的既有行为）。
- **context_sampling 不支持外锚**（窗口调度自带时序管理），传入时忽略并在 summary 标 `external_anchor_ignored`。
- 边界：`overlap_frames=0` 时外锚不生效（归一为 0）；锚帧数不等于段首重叠帧数属用户参数错误（会重复/跳帧），按"锚帧数=段首重叠帧数"使用。

### 87.4 SFVideoConcat：文件级合并（不装回内存）

`nodes/video/video_concat.py` + 纯逻辑 `sf_utils/video_concat.py`：

- **输入解析** `collect_segment_paths`：递归（深度≤8）提取字符串路径，保序去重；跳过 `.png`（VHS 首帧元数据）与被 `X-audio.mp4` 终版覆盖的 `X.mp4` 中间文件（VHS `output_files` 结构：`[png, mid.mp4, final-audio.mp4]`，`VHS_KeepIntermediate=False` 会删中间文件，故只保留终版）。
- **合并**：`InputImpl.VideoFromFile(路径)` 逐段引用（`get_stream_source` 直接返回路径，零拷贝）→ `InputImpl.VideoFromList(videos, complete_audio=audio, codec=Types.VideoCodec("auto"))` → `save_to(output_path, format=Types.VideoContainer(format))`。各段与目标容器/编码签名一致时**纯 remux 不重编码**（`save_to` 内 `reuse_streams` 判定）；命名复用 `folder_paths.get_save_image_path`（`_00001_.mp4` 计数不覆盖）。
- **音频**：分段时各段不接 VHS 的 audio（否则每段音轨与段内容错位）；`SFVideoConcat.audio` 接整段 `VHS_LoadVideo.audio`（lazy AUDIO dict）统一注入，`VideoFromList` 的 `complete_audio` 自动截到视频长度（`len(images)/frame_rate × sample_rate`）。
- **cleanup**：可选删除段文件（默认关，便于失败排查）。
- 若合并文件在 output 目录，可通过 `LoadVideo` + `ConcatenateVideo` 继续做后处理（Video 对象为文件引用，仍不装帧）。

### 87.5 已知开销与边界

- **VHS 普通 LoadVideo 无 seek**：`skip_first_frames` 是从 0 帧起 `grab()` 逐帧丢弃（不解码输出但要走解码器），12 段的全长任务约多解 1.5× 前缀；长片建议 `VHS_LoadVideoFFmpeg*`（输入侧 `-ss` 快速 seek），代价是换节点重连。
- **每轮重新执行**：循环里 VHS/VAE encode/采样都重跑（本来就该跑）；模型权重不卸载，`_empty_cache(force=True)` 仍在每段末清理。
- **单段显存不变**：外层分段只解决 RAM/mask；单段 81 帧 @生成分辨率的采样峰值依旧（能跑通单段即可跑全长）。
- **SAM3 追踪**：整段追踪一次（可 `SFTrackDataCache` 磁盘缓存）后循环内只切片；不要分段重追踪（ID/颜色跨段不一致）。

## 98. SFSAM3ReanchorTrack：指定帧重锚追踪（2026-09）

> 背景：视频中途人物从全身变半身/换拍摄角度时，SAM3 记忆传播失效、遮罩从切换帧起只抓住头（或丢失目标）。原生 `SAM3_VideoTrack` 只在第 0 帧条件化（`comfy_extras/nodes_sam3.py` 的 `if frame_idx == 0 and initial_masks is not None`），无法中途重锚。需求：指定若干帧，在这些帧用提示词重新检测，并以该帧为起点重新传播。

### 98.1 为什么是 sfnodes 包装节点

- **不改核心**：中途条件帧需要改 `comfy/ldm/sam3/tracker.py`（`_condition_with_masks` 本可接受任意 `frame_idx`，但 `track_video_with_detection` 只在 0 帧调用）+ `detector.forward_video` + 原生节点签名；docker `patches/` 无落地机制、升级即丢。
- **图内拼接不可行**：多个 `SFSAM3PointTrack` + `SFTrackDataAdd` 要求各输入帧数一致且无补帧节点；且每段会追到片尾，锚点越多重复算力越大。
- **硬重置 vs 软纠正**：原生逐帧检测的 recondition（`tracker.py` 高置信 ≥0.8 且重叠 ≥0.5 时替换遮罩）是"软纠正"，旧记忆仍在；分段调用原生节点 = 每段全新 tracker 状态（无历史记忆污染），锚帧重新检测，是真正的"从该帧重新扩散追踪"。

### 98.2 节点设计（`nodes/video/sam3_reanchor_track.py`，CATEGORY `sfnodes/video`）

- required：`images` / `model`（SAM3.1）/ `anchor_frames`（STRING，逗号/空格/分号分隔，空串回退 `[0]`；排序去重、越界/非整数报错）。
- optional：`clip`（SAM3 文本编码器，CheckpointLoader 的 CLIP）、`prompts`（multiline，**每行对应一个锚帧**，可各不相同，如切镜后 `person`→`woman`；空行/行数不足回退 `conditioning`，多余行忽略）、`conditioning`（共用现成条件）、`initial_mask`（**仅首锚**种子，2D 自动升维）、`detection_threshold` / `max_objects` / `detect_interval`（段内透传原生节点）。
- 流程：锚帧 `a_i` 切段 `images[a_i:a_{i+1}]`（末段到片尾）→ 每段 `clip.encode_from_tokens_scheduled(clip.tokenize(line))`（该行非空时）→ 原生 `SAM3_VideoTrack.execute(images=段, initial_mask=首锚种子, conditioning=cond_i, …)` → `concat_track_data_segments` 拼回全长。
- 输出恒与输入等长、`orig_size` 继承、各段跨对象位或**塌单身份**（`scores=[1.0]`），可直接接 `SAM3_TrackToMask` / `SCAIL-2 driving_track_data`。
- 某段起始检测为空（`packed_masks is None`）→ 该段帧区间补零，不影响其他段。
- 算力：各段帧不重叠，总计 ≈ 一遍全片 + 每段起始一次检测；每段检测进度条独立（原生 execute 自建 pbar）。
- 与"加 conditioning 让原生节点自己 recondition"（§上文对话结论）的区别：后者依赖检测分 ≥0.8 的硬编码阈值且不清理旧记忆；本节点是用户可控的强制重锚，且允许每段换提示词。

### 98.3 纯逻辑与复用

- `sf_utils/track_data_ops.py::concat_track_data_segments(segments, total_frames, torch)`：`(起始帧, track_data)` 列表 → 每段跨对象**位或**（packed uint8 直接按位或，无需 unpack）塌单身份 → 按偏移放入零张量；首/尾/段间空隙补零；空段跳过；重叠/越界/工作网格不一致报错；`orig_size` 取首个带值段；全空返回 `packed_masks=None`。补零帧对象维恒为 1（回归：曾用 `ref.shape[1:]`，多对象首段会让零帧带 2 通道导致 cat 失败）。
- `sf_utils/common.py::node_result`：核心 V3 节点 `io.NodeOutput` 解包归一，从 `sam3_point_track.py` 提升为公共实现（两节点共用，禁止内联副本）。

### 98.4 边界

- 提示词行需要 `clip`；缺 `clip` 且该行非空时报错。某锚帧既无提示词也无 `conditioning`/首锚种子时报错。
- 非 multiplex 老 SAM3 不支持检测路径，仅首锚 `initial_mask` 可用（其余段会由核心抛错）。
- 首锚 >0 时其前帧恒为空（与 `SFSAM3PointTrack` 一致）。
- `max_objects` 是**每段**上限；段间对象身份不保证对应，输出已塌单身份，故跨段多主体场景不适用（多主体请用单段 `SAM3_VideoTrack`）。

### 98.5 测试

- `tests/test_reanchor_track.py`：stub `torch` + `comfy_extras.nodes_sam3` + FakeClip；覆盖锚帧解析（排序去重/空串回退/越界与非法报错）、逐行编码与切片调用、空行回退 conditioning、`initial_mask` 仅首锚与 2D 升维、空段补零、输出全长单身份、结构元数据与双字典注册。
- `tests/test_track_data_ops.py`：补 `concat_track_data_segments` 用例（段间空隙、多对象并集、多对象段在前补零、单段直通、全空、重叠/越界/网格不一致报错）。
