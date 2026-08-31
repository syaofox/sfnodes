# 经验归档索引（experience）

> 本目录由原单文件 `doc/experience.md` 拆分（2026-08）：**全局章节号 §N 保持不变**，AGENTS.md 经验摘要与文内互引的 `§N` 由下表定位到具体文件，按需读取避免整档加载。
> 本目录归档 AGENTS.md 精简（2026-08）时删除的具体机制与踩坑经验；主文档只保留通用约束与每类机制的结论索引（见 AGENTS.md「经验摘要」）。
> 内容基于当时代码/前端版本（comfyui_frontend_package 1.x，Vue 重构后，版本号会随升级变化，以容器内 `pip show comfyui-frontend-package` 为准，各节另有标注），可能随版本升级过时，使用时结合代码核实。

## 文件与章节映射

| 文件 | 主题 | 章节 |
|---|---|---|
| `platform.md` | 平台机制（ComfyUI 前后端通用） | §1 ComfyUI 后端机制（循环/图展开） · §2 ComfyUI 前端机制（经验总结，含 §11 画布多选宽度对齐） |
| `patterns.md` | 横切模式与修复批次 | §3 静态检查脚本经验（AST 对比踩坑） · §4 动态 combo 校验与工作流绑定状态（widget 数据载体） · §17 复刻节点去重：sf_common.js / disk_state.py 公共模块收敛与踩坑 · §26 前端架构治理（2026-08）：工具收敛 / 弹层三件套 / 纯模块边界 · §27 2026-08 健壮性修复批次：表达式防御 / ReDoS 交替型 / 路径净化 / 双端镜像补缺 |
| `nodes-text.md` | 文本与提示词节点 | §6 SFPromptTags：@tag 展开注入 / Picks 游标 / 全屏编辑器 / 中文与拼音（复刻 Pixaroma Prompt） · §7 SFPauseText：prompt 剪枝闸门（复刻 Pixaroma Pause Text） · §14 SFTextFindReplace：查找替换双端镜像（复刻 Pixaroma Find & Replace） · §15 SFValueDropdown：值下拉与输出点对齐（复刻 Pixaroma Dropdown） · §16 SFPromptReader：PNG/视频元数据提示词恢复（复刻 Pixaroma Prompt Reader） · §18 SFLoadImagesPath 目录切换：三源 + 渐进式浏览 + popup 下拉 · §23 SFPromptList：行号编辑器与 wrap 镜像测量 · §24 SFPromptStack：动态 Prompt 列表与行高拖拽 · §29 SFPromptPreset：十一分类组合预设（正交原则 / 随机机制 / LLM 优化链路） |
| `nodes-image.md` | 图片 / 遮罩 / latent 节点 | §8 SFPauseImage：快照闸门与预览保存（复刻 Pixaroma Pause Image） · §9 SFPauseMask：遮罩快照闸门（Pixaroma Pause Mask 同构扩展） · §11 SFImageCrop/SFImageUncrop：可视化裁剪与贴回（复刻 Pixaroma Crop/Uncrop） · §12 SFImageOutpaint/Stitch：外绘填充与原始图贴回（复刻 Pixaroma Outpaint） · §13 SFImageResize：wired 尺寸缩放（复刻 Pixaroma Image Resize） · §22 SFPauseLatent：latent 快照闸门（分段采样中间暂停） · §34 SFLoadImageBrowser 右键菜单：提示词复制与工作流载入（全链路复用零后端改动，2026-08） · §35 SFMaskFill：统一填充节点（合并 SFMaskedFill/SFMaskFillColor，falloff/skip 全局 + 条件显隐，2026-08） |
| `nodes-lora.md` | LoRA / Civitai / Krea2 预设生态 | §5 Qwen3 无审查微调版 + TextGenerate：thinking 参数与思考链（COT） · §19 SFLoraStack：多行 LoRA 栈复刻（触发词/描述/封面/Civitai 查询/孤儿数据迁移） · §20 SFLoraStack：正交堆叠 ortho_gs（2026-08） · §21 Civitai 页面主体描述补充（curl_cffi / __NEXT_DATA__ 与 Cloudflare 拦截） · §25 SFRegionalLoRA：多区域角色 LoRA（token 网格注入与匹配诊断） · §28 SFStylesSelector：风格选择器复刻（Easy-Use stylesSelector） · §31 Krea2 预设管理：SFImageInterrogator / SFKrea2SystemPrompt（内置 + 用户覆盖 + 墓碑复位） · §32 SF Load Diffusion Model：信息面板跨域复用（别名路由域分派 + ctx.api 整束注入，2026-08） · §33 TextEncodeKrea2 视觉通路机制（Qwen3-VL 单帧绑定 / min_pixels 兜底 / RGBA 预乘时序，2026-08） · §34 SFLoraStack 复合预设：LoRA 顺序/强度 + 正向提示词（与 triggers 分离，2026-08） |
| `apps.md` | 无节点面板应用 | §10 SF Workflows：工作流面板（复刻 Pixaroma Workflows） · §30 SF LoRA 浏览器：工具栏应用 + 信息面板宿主 ctx 适配（浏览全部 LoRA 并编辑信息） |

## 维护规则

- 新增经验：按主题追加到对应文件的**下一个全局 §N**（只增不改不重排），并在上表加行；同时在 `AGENTS.md` 经验摘要增补/更新索引行。**新类别且现有主题均不适配时可新建主题文件**：英文短名对齐节点族/领域（如未来的 nodes-video.md）、标题行注明所含 § 章节、本表加行——勿把无关章节塞进既有文件（检索会整文件加载）。
- 跨文件引用一律写 §N（不写相对路径），由本表定位；各文件开头注明所含章节。
- 章节内小节号（如 §19.7）指「§19 文件内的 ### 7.」，随所属章节走。
