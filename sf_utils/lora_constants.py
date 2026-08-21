"""LoRA 扩展名单源（单点真源，禁止各处内联副本）

三处历史白名单已收敛至此：
- sf_utils/lora_reader._LORA_KEY_EXTS
- sf_utils/lora_samples._LORA_EXTS
- web/sf_common.js LORA_EXT_RE

保持大小写不敏感，新增扩展名仅改此处。
"""
import re

LORA_EXTS = (".safetensors", ".safetensor", ".ckpt", ".pt", ".pth", ".bin", ".sft", ".gguf")
LORA_EXT_SET = {e.lower() for e in LORA_EXTS}
# 仅用于显示剥离（白名单，非“最后一个点后一切”），版本化名如 MoXin_v1.0 保留 .0
LORA_EXT_RE = re.compile(r"\.(safetensors|safetensor|ckpt|pt|pth|bin|sft|gguf)$", re.IGNORECASE)
