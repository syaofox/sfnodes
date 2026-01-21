import folder_paths
import comfy.utils
import comfy.sd


class MultiLoraLoader:
    """
    支持加载多个LoRA的节点，最多20个LoRA槽位
    每个LoRA可以单独开关，设置强度（两位小数精度）
    最终权重通过normalize_weight进行归一化转换
    """
    def __init__(self):
        self.loaded_loras = {}

    @classmethod
    def INPUT_TYPES(cls):
        # 获取所有可用的LoRA文件列表，添加None选项
        lora_list = folder_paths.get_filename_list("loras")
        if lora_list:
            lora_list_with_none = ["[None]"] + lora_list
        else:
            lora_list_with_none = ["[None]"]
        
        # 构建20个LoRA槽位的输入定义
        required = {
            "model": ("MODEL", {"tooltip": "The diffusion model the LoRAs will be applied to."}),
            "clip": ("CLIP", {"tooltip": "The CLIP model the LoRAs will be applied to."}),
            "normalize_weight": ("FLOAT", {
                "default": 1.0, 
                "min": 0.0, 
                "max": 10.0, 
                "step": 0.01,
                "tooltip": "归一化权重系数。实际权重 = (目标权重 / 所有权重之和) × normalize_weight"
            }),
        }
        
        # 为每个LoRA槽位添加输入
        for i in range(1, 21):
            required[f"lora_{i}_enabled"] = ("BOOLEAN", {
                "default": False,
                "tooltip": f"是否启用LoRA {i}"
            })
            required[f"lora_{i}_name"] = (lora_list_with_none, {
                "default": "[None]",
                "tooltip": f"选择LoRA {i}的文件"
            })
            required[f"lora_{i}_strength"] = ("FLOAT", {
                "default": 1.0,
                "min": -100.0,
                "max": 100.0,
                "step": 0.01,
                "round": 0.01,  # 两位小数精度
                "tooltip": f"LoRA {i}的强度（两位小数精度）"
            })
        
        return {"required": required}

    RETURN_TYPES = ("MODEL", "CLIP")
    OUTPUT_TOOLTIPS = ("应用了多个LoRA的扩散模型", "应用了多个LoRA的CLIP模型")
    FUNCTION = "load_multiple_loras"
    CATEGORY = "loaders"
    DESCRIPTION = "加载多个LoRA（最多20个），每个LoRA可单独开关和设置强度，权重通过normalize_weight归一化"

    def load_multiple_loras(self, model, clip, normalize_weight, **kwargs):
        """
        加载多个LoRA并应用归一化权重
        
        Args:
            model: 扩散模型
            clip: CLIP模型
            normalize_weight: 归一化权重系数
            **kwargs: 包含所有lora_i_enabled, lora_i_name, lora_i_strength参数
        """
        # 收集所有启用的LoRA及其目标权重
        enabled_loras = []
        target_weights = []
        
        for i in range(1, 21):
            enabled = kwargs.get(f"lora_{i}_enabled", False)
            if enabled:
                lora_name = kwargs.get(f"lora_{i}_name", "[None]")
                if lora_name and lora_name != "[None]":
                    strength = kwargs.get(f"lora_{i}_strength", 1.0)
                    enabled_loras.append((i, lora_name, strength))
                    target_weights.append(abs(strength))  # 使用绝对值计算归一化
        
        # 如果没有启用的LoRA，直接返回原始模型
        if not enabled_loras:
            return (model, clip)
        
        # 计算权重总和
        total_weight = sum(target_weights)
        
        # 如果总和为0，直接返回原始模型
        if total_weight == 0:
            return (model, clip)
        
        # 应用归一化权重转换
        current_model = model
        current_clip = clip
        
        for i, lora_name, original_strength in enabled_loras:
            # 计算归一化后的实际权重
            normalized_strength = (abs(original_strength) / total_weight) * normalize_weight
            
            # 保持原始符号
            if original_strength < 0:
                normalized_strength = -normalized_strength
            
            # 如果归一化后的权重为0，跳过
            if normalized_strength == 0:
                continue
            
            # 加载LoRA文件
            try:
                lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
                
                # 检查缓存
                lora = None
                if lora_path in self.loaded_loras:
                    lora = self.loaded_loras[lora_path]
                else:
                    lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                    self.loaded_loras[lora_path] = lora
                
                # 应用LoRA到模型和CLIP
                # 使用相同的强度值应用到model和clip
                current_model, current_clip = comfy.sd.load_lora_for_models(
                    current_model, 
                    current_clip, 
                    lora, 
                    normalized_strength, 
                    normalized_strength
                )
            except Exception as e:
                print(f"加载LoRA {i} ({lora_name}) 时出错: {str(e)}")
                continue
        
        return (current_model, current_clip)
