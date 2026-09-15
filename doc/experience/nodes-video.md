# nodes-video.md — 视频与视频生成节点

> 所含章节：§72 SCAIL-2 四节点复刻（ComfyUI-SCAIL2-Easy）· §77 SAM3 视觉点选追踪与 track_data 排除（驱动遮罩排男/阴茎/精液）。本文件为 `doc/experience/` 第七个主题（2026-09）：视频生成节点族（SCAIL-2 / Wan）不与 platform / patterns / nodes-text / nodes-image / nodes-lora / apps 适配，故新建。

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

- 源文件的双语探测（`isChineseLocale` / `EASY_TRANSLATIONS.zh|en`）按项目约定收敛为**中文单语** `SCAIL2_LABELS` + `SIMPLE_COMBO_LABELS`；`applyLabel` 同时写 `label/localized_name/display_name` 与 `options`/`_state`（Vue 前端渲染读 `label ?? localized_name ?? name`）。
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
