# extract_lora_diff.py — 从模型差异提取 LoRA

从两个**同架构**扩散模型的权重差 ΔW = W_微调 − W_基座 中提取低秩 LoRA。
经典"checkpoint 差异提取 LoRA"思路（kohya extract_lora_from_models 同款），
专为 ComfyUI 原生 int8+convrot 量化模型（comfy_kitchen）适配：
自动识别 convrot 反旋转，保证 ΔW 落在 LoRA patch 的作用空间。

本项目首个实例：`krea2_raw_int8_convrot`（基座）与 `museByStableYogi_v30TurboInt8`（微调）
提取出 `krea2_muse_v30turbo_r64.safetensors`（r=64，409.5 MB，224 层），
经 ComfyUI 真实加载路径验证 patch 全部生效。

## 适用范围（重要）

**仅支持 ComfyUI 原生 int8 量化模型对**（`*.weight` + `*.weight_scale` 配对，
含 convrot；加载方不限于 Krea2——SD3.5/Flux/SD1.5/SDXL 等同架构量化模型对
理论上均可，key 命名从文件推导不写死架构）。以下情况**不支持**：

| 场景 | 表现 | 原因 |
|---|---|---|
| fp16/bf16/fp32 普通 checkpoint | 直接报 "no common layers found" | 层发现依赖 `weight_scale` 存在 |
| fp8 量化模型（float8_e4m3fn 等） | **静默产出错误结果**（不乘 scale） | 反量化只处理 int8 dtype |
| conv 层（4D 权重） | 脚本崩溃或 SVD 结果错误 | `svd_lowrank` 只接受 2D |
| CLIP / text encoder | 不提取 | 仅做 UNet 模型差值 |

普通 checkpoint 场景请用 kohya `extract_lora_from_models.py` 等通用工具。

## 用法

```bash
python extract_lora_diff.py \
    --base      /path/to/base.safetensors \
    --finetuned /path/to/finetuned.safetensors \
    --output    out_lora.safetensors \
    --rank 64
```

### 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--base` | 必填 | 基座模型（差值基准） |
| `--finetuned` | 必填 | 微调模型 |
| `--output` | 必填 | 输出 LoRA 路径 |
| `--rank` | 64 | SVD 截断阶数；64 ≈ 400MB，128 ≈ 800MB |
| `--device` | 自动 | `cuda`（默认，有 GPU 时）/ `cpu` |
| `--dtype` | fp16 | LoRA 存储精度 fp16 / fp32 |
| `--min-rel-delta` | 0.005 | 平均相对 delta 低于此值仅警告不中断 |

### 在本项目 docker 环境运行

```bash
docker cp extract_lora_diff.py comfyui-docker:/tmp/  # 或已同步到 custom_nodes/sfnodes/tools/
docker exec comfyui-docker python -u /tmp/extract_lora_diff.py \
    --base /home/comfy/app/models/diffusion_models/krea2/krea2_raw_int8_convrot.safetensors \
    --finetuned "/home/comfy/app/models/diffusion_models/krea2/museByStableYogi/museByStableYogi_v30TurboInt8.safetensors" \
    --output /home/comfy/app/models/loras/krea2/my_extract_r64.safetensors --rank 64
```

模型路径在容器内为 `/home/comfy/app/models/...`（与宿主机 `/mnt/github/comfyui-docker/models/...` 同一挂载）。
**输出后记得修正属主**：容器内以 root 写入的文件宿主侧 chmod 不了，
必须 `docker exec comfyui-docker chown comfy:comfy <输出> && chmod 644 <输出>`，
否则 ComfyUI（comfy 用户 uid 1000）读不到。

## 工作原理

1. **找公共层**：扫描两个文件的 `*.weight` + `*.weight_scale` 对（量化线性层），按 key 精确匹配。
   key 不完全一致时会打印差异清单（仅出现在一侧的层跳过）。
2. **双端 dequantize**：
   - 普通 int8：`W = w_i8.to(f32) × weight_scale`
   - **convrot int8（关键）**：调用 `comfy_kitchen` 的 `TensorWiseINT8Layout.dequantize` 做
     反 Hadamard 旋转。ComfyUI 对带 LoRA patch 的层回退到 dequantize 后的 fp 权重
     （`comfy/ops.py cast_bias_weight`），所以必须取**反旋转后**的权重做差，
     否则 ΔW 活在旋转空间，patch 上去驴唇不对马嘴。
   - 量化配置来源：`*.comfy_quant` marker（raw 风格）或 `__metadata__._quantization_metadata`
     （muse 风格），两种都解析。
3. **逐层 SVD**：`ΔW = U·S·Vᵀ`（`torch.svd_lowrank` 随机化 SVD），截断到 rank r：
   - `up = U·√S [out, r]`、`down = √S·Vᵀ [r, in]`
   - `alpha = r` → alpha/rank = 1，LoRA strength=1.0 时精确还原 ΔW
4. **写文件**：ComfyUI SD3.5/Flux kohya 约定 `lora_unet_<key点转下划线>.lora_up.weight` /
   `.lora_down.weight` / `.alpha`（`comfy/lora.py model_lora_keys_unet` 的
   `diffusion_model.` 前缀分支直接匹配，generic 分支也可用）。

## 验证方法

提取后务必验证，两步：

**1. 数值重建检查**（脚本日志已含信号强度）：
- 平均相对 delta `||ΔW||/||W_base||` 应 > 1%（0.5% 以下是纯 int8 量化噪声，没信号）。
  本实例 6.15%。
- SVD 能谱平缓属正常：int8 量化噪声是高秩的，截断天然滤掉；低秩部分才是系统性微调信号。

**2. ComfyUI 真实加载**（本实例实测步骤）：

```python
# 容器内 /home/comfy/app 下执行
import comfy.sd, comfy.lora, comfy.utils, comfy.lora_convert
from comfy.model_patcher import get_key_weight

model = comfy.sd.load_diffusion_model('<base>')
key_map = comfy.lora.model_lora_keys_unet(model.model, {})
lora = comfy.lora_convert.convert_lora(comfy.utils.load_torch_file('<lora>'))
loaded = comfy.lora.load_lora(lora, key_map)          # 期望 224 条全匹配
patched = model.clone()
print(len(patched.add_patches(loaded, 1.0)))           # 期望 224

# 抽样层：patch 后权重 vs 微调模型 dequant 权重，相对误差应 ~5%（r=64）
w = get_key_weight(patched.model, 'diffusion_model.blocks.0.attn.wq.weight')[0].float()
```

**3. UI 实测**：UNETLoader(基座) + LoraLoaderModelOnly(strength 1.0) vs 直接加载微调模型，
同 seed 同采样参数对比。风格偏淡可把 strength 提到 1.2–1.5。

## 已知局限

- **仅 int8 量化模型**（详见上方"适用范围"表）：fp16/bf16 普通 checkpoint、fp8 量化、
  conv 层均不支持；误用会报错或静默产出错误结果。
- **int8 量化噪声进入 ΔW**：两个模型各自独立量化，噪声互相独立、无法消除；
  SVD 截断只能滤高秩部分，rank 越高保真度越高、噪声进得越多（tradeoff）。
- **turbo 蒸馏差异**：对 turbo 微调模型（如 muse）与 raw 基座做差，ΔW 包含蒸馏语义
  （8 步 CFG 1 行为），是"还原该模型行为"的 LoRA，不是纯风格 LoRA。
  想要纯风格：对 `muse − krea2_turbo`（同目录有 turbo 基座）做差，但两个 turbo
  系文件的量化参数一致性未验证过，需先跑一遍看信号强度。
- **仅覆盖量化线性层**：norm scales（prenorm/postnorm/qknorm）与嵌入表不提取
  （差异极小，LoRA 惯例跳过）。
- **无 ComfyUI 依赖**：脚本可独立运行（torch + safetensors）；没有 comfy_kitchen 时
  convrot 层退化为普通 `i8×scale`（不反旋转），结果会错，仅作降级兜底。
- **full-rank diff 替代**：若 rank 截断损失不可接受，可改用 ComfyUI `{key}.diff`
  全秩差 patch（`comfy/lora.py load_lora` 原生支持），但文件 ≈ 模型本身大小，非 LoRA。

## 环境要求

- Python 3.10+，`torch`（SVD 建议 GPU，6144²×224 层 CPU 也能跑但慢）、`safetensors`
- 可选 `comfy_kitchen`（convrot 反旋转必需；在 ComfyUI 容器内天然可用）
- 本机仅作编辑环境，实际运行在 docker 容器内（AGENTS.md 规则 1、12）
