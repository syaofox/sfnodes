# 经验归档：音频节点

> 全局章节号 §N 唯一、只增不复用；跨文件引用写「文件名 §N」（同文件内可简写 §N），映射与当前最大 §N 见 [README.md](README.md)。版本时效说明见 README。

## 146. AuK 四节点复刻（ComfyUI-AuK_Doc，2026-09）

### 146.1 来源与范围

复刻 `/mnt/github/comfyui-docker/custom_nodes/ComfyUI-AuK_Doc`（DocWorkBox 对 Tencent-Hunyuan/AuK 的 ComfyUI 改版，MIT，上游快照 1.0.0 + 工作副本未提交改动：int8 repack 支持与内置 `configs/`）的全部 4 个节点，功能 1:1：

| 源节点（V3 io.ComfyNode） | 本包节点（V1） | 作用 |
|---|---|---|
| `AuKModelsLoader_Doc` | `SFAuKModelsLoader` | 选 Base/Flash/int8 repack + Qwen2.5-Omni + 显存档位 → 引擎 |
| `AuKGenerateEdit_Doc` | `SFAuKGenerateEdit` | 指令 TTS / 零样本音色克隆 / 语音编辑 → AUDIO |
| `AuKOpenAISettings_Doc` | `SFAuKOpenAISettings` | PE 的 OpenAI 兼容配置（不联网） |
| `AuKLlamaCppSettings_Doc` | `SFAuKLlamaCppSettings` | llama-cpp_vlm 本地模型接入 PE |

新槽类型 `SF_AUK_ENGINE` / `SF_AUK_LLM_CONFIG`（不与原包 `AUK_ENGINE_Doc` 互连，避免两包混用时错接）。类别 `sfnodes/audio`（本包首个音频组）。

### 146.2 目录布局与移植策略

```
nodes/audio/
├── auk_paths.py     # 模型发现（逐字移植上游 model_paths.py，仅 logger 与 config_model_name 公开化）
├── auk_loader.py    # SFAuKModelsLoader + AuKEngine（weakref 缓存 + 锁）
├── auk_generate.py  # SFAuKGenerateEdit + 音频规范化/重采样/30s 预算校验
├── auk_config.py    # 两个 Settings 节点 + LLMSettings/LlamaSettings/_LocalClient
├── configs/         # 内置 AuK.yaml / AuK-Flash.yaml 兜底（上游工作副本新增）
└── auk/             # 推理引擎（LICENSE 随包；相对 import 层级保持不变）
    ├── infer/{infer_auk,pe,audio_io}.py + pe.config.yaml
    └── model/**（CFMEdit / Flux2Edit / BigVGANFlowVAE + alias_free_torch/commons/vits + mel_filters.npz）
```

- **不移植** `infer_cli.py` / `infer_gradio.py`；`pe.py` / `infer_auk.py` 中 CLI/gradio 专用函数（`argparse`/`shlex` 链路、`main`、`prepare_generation`、`get_gen_duration`/`save_audio`、`PromptEnhancerOutput` 的 gradio 参数转换与 `debug_dict`）一并剥除，避免死代码。
- `AukInfer` 在 loader 的 `execute` 内延迟导入：节点注册/ComfyUI 启动阶段不加载 transformers/torchdiffeq（启动速度不受影响）。
- 模型目录注册沿用上游的 `folder_paths.add_model_folder_path`（`auk`/`auk_qwen`/`auk_vae` 共 7 条）；该 API 幂等，与原包共存不产生重复项。
- 引擎与节点包同相对层级（`nodes/audio/auk/` ↔ 上游 `auk/`），内部 `from ..model import`、`from ...model.vae` 等无需改写，便于日后对照上游同步。
- 上游 UI 文案（tooltip/DESCRIPTION）中文化；引擎内部报错与 `pe.py` 中文提示保持原文。

### 146.3 V1 适配要点

- V3 `io.Schema` → V1 `INPUT_TYPES`；`AUK_ENGINE`/`AUK_LLM_CONFIG` 自定义类型直接作为 (`"SF_AUK_ENGINE"`,) 字符串槽；`io.NodeOutput` → tuple。
- `seed` 的 `io.ControlAfterGenerate.fixed` → V1 选项 `"control_after_generate": "fixed"`（`ControlAfterGenerate` 是 str Enum，object_info 序列化后前端同样识别；不要只在后端置 fixed——V1 的种子控件模式由该选项值驱动）。
- `advanced=True` 在 V1 选项字典中同样支持（核心节点已在使用）。
- `display_name`/`label_on`/`label_off`：仅保留 `label_on/off`（V1 已支持），不用 per-input `display_name`。
- 动态 combo 在 `INPUT_TYPES()` 每次调用时扫描磁盘（上游 V3 只在扩展加载时求值一次）；无模型时给占位项 `No AuK models found`，选中后由 `resolve_choice` 抛 `FileNotFoundError`。
- 节点设备列表 `[f'cuda:{i}' ...] or ['CUDA unavailable']` 在 `INPUT_TYPES()` 内计算，避免导入期固化。
- 槽类型常量定义在拥有者模块：`AUK_ENGINE`（auk_loader）与 `AUK_LLM_CONFIG`（auk_config），generate 只 import 使用。

### 146.4 Prompt Enhancer 链路（pe.py）

`SFAuKGenerateEdit.use_prompt_enhancer=True` 时走 `PromptEnhancer.prepare`：

1. 有输入音频先做 ASR（腾讯云录音文件识别，凭据经 `TENCENTCLOUD_SECRET_ID/KEY` 或参数；不可用时回退 funasr SenseVoiceSmall 本地推理）；
2. LLM 分类（单任务 + 槽位抽取，`pe.config.yaml` 的 `classify` 提示词，输出严格 JSON）→ 不支持的请求抛 `UnsupportedRequestError`；
3. 非语言声编辑再匹配 canonical event、带 `rewrite` 的任务再扩写描述；
4. 时长解析：用户显式 `generation_seconds>0` 优先，否则按任务规则（F5 启发式 / 逐字节语速 / LLM 预测 / 输入时长）估计并量化到模型帧；
5. 按任务模板渲染标准模型指令 + 输入音频预处理（VAD 裁剪、响度归一化）。

模型输入侧：`prepare_model_audio` 同时产出目标采样率波形（VAE 编码用）与 16 kHz 单声道 numpy（Qwen 消息的 audio 部分）；无参考音频时向文本追加 `|<no_prompt_audio>|` 标记（上游行为，勿删）。`_call_llm` 对 deepseek/tokenhub 端点自动附 `thinking={"type":"disabled"}`；llama.cpp 适配节点用 `_LocalClient` 伪装成 `openai` 客户端对象，因此 pe.py 无需感知本地/远程差异。

### 146.5 显存档位与 30 秒预算

- `memory_mode`：`low_vram`（Qwen+VAE 在 CPU、半精度 DiT 在 GPU）/ `balanced`（Qwen 与 DiT 轮流上 GPU）/ `max_vram`（DiT+VAE 常驻；int8 Qwen 用 forward pre/post hook 逐层反量化、常驻显存）。
- int8 权重依赖 `comfy_kitchen.tensor.TensorWiseINT8Layout`（`int8_tensorwise`，可带 convrot）；`w4a8` 明确报错不支持。
- 源/参考 + 生成目标共享 30s 序列预算：`validate_sequence_duration` 在采样前按 `downsample_rate` 折算模型帧校验，超限报出实际秒数。
- Flash（`config.model.name == "AuK-Flash"`）在引擎层强制 4 步/CFG 0/sway None/t_grid 固定；节点层检测到 widget 值与配方不一致时**自动锁定**（改为 4/0/-1 并打印一行说明），不因参数拒绝工作流——引擎本就忽略这三个调用参数，拦截只会制造无意义的报错。

### 146.6 复用与依赖决策

- **llama.cpp 插件发现抽公共模块**：`sf_utils/llama_cpp.py::find_llama_plugin()` 原为 `SFQwenImage21PromptEnhancer` 私有函数（按属性指纹扫 `sys.modules`，防二次实例化导致 storage 分裂）；本次移植 AuK 适配器时抽到 `sf_utils/` 两处共用，删除原私有副本。上游 `_resolve_storage` 的 `__globals__['LLAMA_CPP_STORAGE']` 探测法与本实现等价，统一后少一份副本。
- **不复用 `sf_utils/llm_client.py`**：其契约是读 GUI 全局设置（`sfnodes.LLM.*`）+ 只返回文本；AuK PE 需要随工作流保存的 per-node 配置、OpenAI SDK 响应对象（`model_dump`/usage/reasoning_content）与 `llm_client` 注入点（llama.cpp 本地 client），契约不同，强行合并会同时破坏两侧行为。
- **依赖下限对齐实机容器实测版本**：`requirements.txt` 的 AuK 段全部按下限 = 容器已装版本声明（transformers 5.16.1 / openai 2.54.0 / accelerate 1.15.0 / funasr 1.4.16 / modelscope 1.40.1 等），容器启动期 `uv pip install --system -r requirements.txt` 审计为零变更；上游的 `transformers>=4.52,<5` 不采纳（实机 5.16.1 验证可用，且 ComfyUI 自身 `transformers>=4.50.3` 无上界，照抄 `<5` 会触发降级）；`funasr<2`/`modelscope<2` 语义大版本上界保留（容器版本满足）。`torch`/`torchvision`/`torchaudio`/NumPy 由 ComfyUI 运行时提供，不入 requirements。
- `funasr`/`modelscope`/`tencentcloud-sdk-python-asr`/`WeTextProcessing` 仅 PE 使用，但按上游模块级 import（requirements 全量声明）；`WeTextProcessing` 缺失时上游已有回退原文的路径。

### 146.7 测试与实机边界

- 新增 `tests/test_auk_paths.py`（临时目录构造 safetensors 头与 config.yaml：变体推断/checkpoint 列表/外部与内置 config 解析/VAE 相邻→metadata→config 相对路径/Qwen 目录与单文件/路径包含校验）与 `tests/test_auk_nodes.py`（桩 torch/torchaudio/soundfile/folder_paths/omegaconf/openai + FakeEngine：schema、Settings 校验与归一化、llama 本地 client 与 cleanup、Loader 档位/设备校验、Generate 输入校验与输出形状、消息与采样参数透传、30s 预算）。上游 `tests/test_package.py` 是包元数据检查，不适用，未移植。
- 容器冒烟（syntax + 真实依赖导入）通过：`infer_auk`/`pe`/`audio_io` 与 4 节点 schema 均可在容器 Python 3.12 下导入。
- **未覆盖**：真实模型推理（DiT/VAE/Qwen 权重加载与采样）、PE 网络链路（LLM ASR/改写）、llama.cpp 真机增强。需在实机工作流验证生成质量与显存档位；引擎内部为逐字移植，出现行为差异优先对照上游同文件排查。

### 146.8 模型目录与使用

沿用上游目录约定（无需改动既有下载）：`models/auk/{AuK,AuK-Flash}/`（权重 + `config.yaml`，VAE 放 `models/auk/vae/`，也可与权重同目录）、`models/text_encoders/Qwen2.5-Omni-3B/`；ComfyUI 格式 repack 放 `models/diffusion_models/auk/` 亦可。Qwen 需完整目录（config/tokenizer），int8 单文件需同目录有 `config.json`。

### 146.9 节点进度条与采样中断（sfnodes 扩展，2026-09）

上游无进度反馈：AuK 采样（Base 32 步）与 PE 阶段在节点上完全是黑盒。本包在引擎/PE 中加了**可选回调**（默认 `None`，不改变上游行为）：

- **回调协议**：`progress_cb(phase, done, total)`；phase ∈ `pe`（5 个检查点：ASR / 分类 / 改写或非语言声匹配 / 时长 / 音频预处理）、`vae_encode`、`encode`（条件编码）、`sample`（每步）、`decode`。
- **引擎侧落点**：`pe.py::PromptEnhancer.prepare(progress_cb=)`、`infer_auk.py::AukInfer.generate/_run`、`cfm_edit.py::CFMEdit.sample`、`memory_utils.py::euler_final`（每步后回调）。`sample` 的标准模式（odeint 路径）只在前后各报一次；我们的加载器恒为 `low_memory=True` 走 euler 路径，逐步进度始终可用。
- **节点侧**：`SFAuKGenerateEdit.execute` 建 `comfy.utils.ProgressBar(grand_total)`，总格数 = PE 5（关闭时不占格）+ VAE 编码 1 + 条件编码 1 + 采样 nfe（Flash 4）+ 解码 1，按 `update_absolute` 映射总进度；前端 1.53+ 按 `nodeProgressStates` 在节点上渲染（`ProgressBar` 自带 0.1s / 0.5% 节流）。
- **中断**：回调内调用 `comfy.model_management.throw_exception_if_processing_interrupted()`——排队取消/中断可打断长采样（上游原本只能等整段结束）。异常沿 `euler_final → sample → _run → generate` 抛出，引擎 `finally`（offload hook 归还）与节点 `finally`（PE 临时文件清理）照常执行。
- 已知边界：ASR 首次模型下载（funasr 内部）与单次 LLM 请求不可中断，只在阶段边界生效；`ProgressBar` 的节点 id 绑定依赖「execute 内创建」，勿提前到模块级。

### 146.10 切换模型释放旧引擎（sfnodes 扩展，2026-09）

问题：AuK 引擎不在 `comfy.model_management` 管理范围内，`unload_all_models()` 不会释放它；ComfyUI 执行缓存会长期持有 loader 输出（引擎对象），换模型再排队时旧引擎权重仍占显存——实测 Flash→Base 在 max_vram 下报「needs 4.4 GiB but only 1.9 GiB available」。

- `AukInfer.release()`：主动丢 `model`/`vae_model`/offload hooks + `gc.collect()` + `torch.cuda.empty_cache()`，并置 `released=True`（此后 `generate()` 直接报错提示重跑 loader）。
- `SFAuKModelsLoader`：模块内 `_CACHE` 改**强引用**；`_release_others(key)` 释放非当前 key 的全部引擎；`IS_CHANGED` 恒 `NaN`（ComfyUI 执行缓存不复用引擎对象，避免命中已释放的旧引擎；引擎复用由 `_CACHE` 负责）。
- 释放时机：仅在**显存不足**时触发——max_vram 放置失败走引擎 `vram_retry` 回调（已建好的 CPU 模型无需重载），low_vram/balanced 放置抛 `torch.cuda.OutOfMemoryError` 走 loader catch 后重试一次（重新构建）。同图多个 AuK 引擎只要显存放得下就不会被提前释放；放不下时后者会释放前者（该组合本身跑不动，日志有 `Released N previous AuK engine(s)`）。
- 引擎侧新增 `InsufficientVRAMError(ValueError)`（max_vram 显存不足专用），报错文案不变。

### 146.11 官方提示词预设下拉（sfnodes 扩展，2026-09）

`SFAuKGenerateEdit` 顶部（instruction 之前）挂「预设分类 → 提示词模板」两个下拉 + 「填入 instruction」按钮：选模板即把官方 instruction 模板（保留 `{占位符}`）整段写入 instruction 文本控件，分类切换只重建模板选项不写文本。

- 数据：`web/sf_auk_presets_lib.js`（纯模块，17 组 = 16 类 + 附音质改善，含各语言/变体与「官方演示原句」条目；来源 AuK 官方 COOKBOOK.md / infer_gradio.py，MIT），纯函数 `groupNames/itemsOf/templateText`。
- 挂载：`web/sf_auk_generate.js`（registerExtension `sfnodes.auk_generate`）在 `nodeCreated` 用 `addWidget("combo"/"button")` 追加后**原地重排**（`node.widgets.length = 0` + push，保持数组引用）到 instruction 之前。
- **不占 widgets_values**：三个控件都带 `serialize:false`（前端在序列化与位置恢复时均跳过 `serialize === false` 的 widget，`addCustomWidget` 也因此不做即时恢复）——旧工作流的 8 项位置值原样归位，零迁移；选择状态改存 `node.properties.sfAukPresetGroup/sfAukPresetTemplate`，加载时在 `configure` 包装与 `onAfterGraphConfigured` 双路恢复（base configure 先恢复 properties 再恢复 widget 值，故包装内读取可靠）。
- 「填入」按钮用于重复套用：下拉值未变时 callback 不触发，按钮可再填一次；加载恢复只重建选项、不覆盖 instruction。
- 纯前端改动：同步部署目录后浏览器硬刷新即可，无需重启后端。

### 146.12 SFAuKAudioTranscribe：本地语音识别节点（2026-09）

`SFAuKAudioTranscribe`（显示名 SF AuK Audio Transcribe）：输入 AUDIO、输出识别文字，目标语言下拉（自动/中文/英文/粤语/日语/韩语，默认自动）。

- **引擎侧复用**：`pe.py::SenseVoiceSmallASR` 新增可选 `language="auto"` 参数（默认值不变，PE 链路零影响）；节点另复用 `auk_generate.normalize_audio`（AUDIO 校验/批=1/多声道均值/NaN 检查）与 `auk.infer.audio_io.write_wav`（临时 WAV 桥接，`finally` 清理）。
- **仅本地**：不接腾讯云 ASR（避免独立工具节点静默上传音频）；首次使用经 funasr/ModelScope 下载 SenseVoiceSmall（~900MB，缓存于 `.cache`），之后离线可用。语言码映射：自动→auto、中文→zh、英文→en、粤语→yue、日语→ja、韩语→ko。
- **报错**：未知语言、识别失败（`ASRCall.error`）、空文本（无人声/过短）均抛 `ValueError` 带原因，不静默返回空串。
- 测试在 `tests/test_auk_nodes.py`（schema/语言映射/临时文件写入与清理/失败与空文本路径）。

### 146.13 SFAuKLongSpeech：长文本语音合成（2026-09）

30 秒是**单次模型序列预算**（参考 + 目标），拼接总时长不受限。`SFAuKLongSpeech`（显示名 SF AuK Long Speech）把长文本做成：分句 → 按时长估计打包 → 逐段生成 → 拼接。

- **分句与打包**（纯逻辑 `sf_utils/text_chunk.py`）：中英句末（`。！？…` 与后接空白/行尾的 `.!?`）与换行切句并保留后置引号；超长单元先按 `，,、；;：:` 次级切分、再按字符二分硬切（切出的纯标点片并入前片，避免孤标点成段）；按 `estimate(text)` 贪心打包并留 8% 余量。
- **时长估计单源 + 语速直控**：`pe.py` 新增公共包装 `estimate_speech_seconds`（instruct_tts 的 F5/utf8 权重口径，zh 0.0803、en 0.0656 s/字节）与 `estimate_speech_units`（加权字数：中文字=1、英文/数字/标点按字节权重≈0.27 折算，`DEFAULT_SPEECH_RATE = 1/(3×0.0803) ≈ 4.15` 字/秒）。节点用 `speech_rate`（3.0–6.0，默认 4.15）直控语速：`estimated = 加权字数 ÷ speech_rate`、`target = estimated + 0.15s`（不再叠加百分比余量，自然语速约 4.2–4.8）；报告带 `speech_rate` 与每段 `raw_estimated_seconds`（默认口径）便于对照。
- **逐段生成**：首段参考 = `input_audio`（按 `reference_seconds` 截前 N 秒，0=全长）；后续段参考 = 上一段生成音频尾部 `ref_tail_seconds`（滚动参考，模型把参考当前缀续说；`同一参考`档则复用首段参考）。每段 `gen_seconds = 估计 × 1.08 + 0.15` 并夹到 `30s − 参考` 预算；`engine.lock` 全程持有、`unload_all_models` 只在循环前一次；逐段 `seed + 段序号` 可复现。
- **指令模板**：参考音色模式全段 `Say the following with the same voice: "{段文本}"`；声音描述模式仅首段用 `Generate speech based on ... description ...`（后续段音色已由滚动参考建立，改用 same-voice 模板避免描述与参考冲突）。
- **拼接**：每段可选按 20ms 窗口能量裁尾静音（防段间静音累积），段间 `pause_seconds` 静音 + 5ms 淡入淡出（仅拼接边），`torch.cat`。
- **进度/中断**：总格 = 段数 ×（nfe+3：vae_encode/encode/sample/decode），沿用 `progress_cb` + `throw_exception_if_processing_interrupted`，长任务可取消。
- **失败**：任一段失败抛 `ValueError` 带段号（不输出半截音频）。
- 测试：`tests/test_text_chunk.py`（分句/硬切/打包纯逻辑）+ `tests/test_auk_nodes.py` LongSpeech 段（schema/校验/逐段参数与种子/滚动与同一参考/描述模式模板/拼接长度/进度/Flash 锁定/裁尾接线）。

### 146.14 SFAuKLongSpeech mode 联动显隐（2026-09）

`mode` 切换时按模式显隐参数（纯前端，`web/sf_auk_long_speech.js`，机制与 §126.4 同款）：

- **映射**：参考音色 TTS 显示 `reference_seconds`、隐藏 `voice_description`；声音描述 TTS 反之；其余参数两模式通用常显。
- **widget 显隐**：复用 `sf_widget_visibility_lib.setWidgetVisible/refreshWidgetSnapshot`（隐藏只影响渲染，值仍随工作流保存与提交）；显隐后调该库新增的通用 `fitNodeToContent(node, graph)` 按内容自适应节点高度。
- **插槽增删**：`input_audio` 在声音描述模式**未连线时移除、已连线保留**（不静默断线，后端按模式忽略该输入）；复用 `sf_dynamic_slots.removeInputAt/syncInputLinkTargets` 修正后续 `link.target_slot`。
- **重放**：mode callback + `onAfterGraphConfigured` 双路（加载/粘贴恢复时 configure 直赋 widget 值不触发 callback）。
- 测试 `tests/test_auk_long_speech_js.js`（17 断言）；`check_web_imports.py` MODS 登记 `sf_auk_long_speech`。
