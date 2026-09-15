# nodes-video.md — 视频与视频生成节点

> 所含章节：§72 SCAIL-2 四节点复刻（ComfyUI-SCAIL2-Easy）。本文件为 `doc/experience/` 第七个主题（2026-09）：视频生成节点族（SCAIL-2 / Wan）不与 platform / patterns / nodes-text / nodes-image / nodes-lora / apps 适配，故新建。

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

### 72.6 测试与回归

- `tests/test_scail2.py`：纯逻辑断言（`4n+1` cover/floor、32 对齐、Fit Video 尺寸、色板、布局候选）+ 节点结构元数据 + 根 `__init__.py` 4 键各出现两次。mock `torch` 顶层符号，stub `sfnodes` 包结构使相对导入可解析。
- `tests/test_scail2_js.js`：mock node + `new Function` 剥离 import/export，覆盖扩展注册、数量驱动槽重建、旧版迁移、Fit Video 显隐、Simple Video 排序/错位修复、`setWidgetVisible` 往返、nodeData 裁剪。
- 已纳入 AGENTS.md 三条快速回归循环。
