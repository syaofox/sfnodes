import folder_paths
import comfy.utils
import comfy.sd


class MultiLoraLoader:
    """
    支持加载多个LoRA的节点，最多50个LoRA槽位
    每个LoRA可以单独开关，设置强度（两位小数精度）
    最终权重通过normalize_weight进行归一化转换
    槽位通过JavaScript前端动态显示/隐藏
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
        
        # 构建必需的输入定义
        required = {
            "model": ("MODEL", {"tooltip": "The diffusion model the LoRAs will be applied to."}),
            "clip": ("CLIP", {"tooltip": "The CLIP model the LoRAs will be applied to."}),
            "normalize_weight": ("FLOAT", {
                "default": 0.0, 
                "min": 0.0, 
                "max": 10.0, 
                "step": 0.01,
                "tooltip": "归一化权重系数。当为0时不归一化，直接使用原始强度；当>0时，实际权重 = (目标权重 / 所有权重之和) × normalize_weight"
            }),
        }
        
        # 为每个LoRA槽位添加可选输入（最多50个槽位，由JavaScript前端动态控制显示）
        optional = {}
        for i in range(1, 51):
            optional[f"lora_{i}_enabled"] = ("BOOLEAN", {
                "default": False,
                "tooltip": f"是否启用LoRA {i}"
            })
            optional[f"lora_{i}_name"] = (lora_list_with_none, {
                "default": "[None]",
                "tooltip": f"选择LoRA {i}的文件"
            })
            optional[f"lora_{i}_strength"] = ("FLOAT", {
                "default": 1.0,
                "min": -100.0,
                "max": 100.0,
                "step": 0.01,
                "round": 0.01,  # 两位小数精度
                "tooltip": f"LoRA {i}的强度（两位小数精度）"
            })
        
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("MODEL", "CLIP")
    OUTPUT_TOOLTIPS = ("应用了多个LoRA的扩散模型", "应用了多个LoRA的CLIP模型")
    FUNCTION = "load_multiple_loras"
    CATEGORY = "loaders"
    DESCRIPTION = "加载多个LoRA（最多50个），每个LoRA可单独开关和设置强度，权重通过normalize_weight归一化。槽位可通过前端动态添加。"

    def load_multiple_loras(self, model, clip, normalize_weight, **kwargs):
        """
        加载多个LoRA并应用归一化权重
        
        Args:
            model: 扩散模型
            clip: CLIP模型
            normalize_weight: 归一化权重系数
            **kwargs: 包含所有lora_i_enabled, lora_i_name, lora_i_strength参数（可选）
        """
        # 收集所有启用的LoRA及其目标权重
        enabled_loras = []
        target_weights = []
        
        # 遍历所有可能的槽位（1-50），检查是否存在且启用
        for i in range(1, 51):
            # 检查输入是否存在（因为是optional，可能不存在）
            if f"lora_{i}_enabled" not in kwargs:
                continue
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
            if normalize_weight == 0:
                # 当normalize_weight为0时，不归一化，直接使用原始强度
                normalized_strength = original_strength
            else:
                normalized_strength = (abs(original_strength) / total_weight) * normalize_weight
                
                # 保持原始符号
                if original_strength < 0:
                    normalized_strength = -normalized_strength
            
            # 打印实际LoRA权重
            print(f"LoRA {i} ({lora_name}): 原始强度={original_strength:.2f}, 归一化后实际权重={normalized_strength:.4f}")
            
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


class MultiLoraLoaderModelOnly(MultiLoraLoader):
    """
    支持加载多个LoRA的节点（仅模型），最多50个LoRA槽位
    每个LoRA可以单独开关，设置强度（两位小数精度）
    最终权重通过normalize_weight进行归一化转换
    只应用到MODEL，不应用到CLIP
    槽位通过JavaScript前端动态显示/隐藏
    """
    @classmethod
    def INPUT_TYPES(cls):
        # 获取所有可用的LoRA文件列表，添加None选项
        lora_list = folder_paths.get_filename_list("loras")
        if lora_list:
            lora_list_with_none = ["[None]"] + lora_list
        else:
            lora_list_with_none = ["[None]"]
        
        # 构建必需的输入定义（不包含clip）
        required = {
            "model": ("MODEL", {"tooltip": "The diffusion model the LoRAs will be applied to."}),
            "normalize_weight": ("FLOAT", {
                "default": 0.0, 
                "min": 0.0, 
                "max": 10.0, 
                "step": 0.01,
                "tooltip": "归一化权重系数。当为0时不归一化，直接使用原始强度；当>0时，实际权重 = (目标权重 / 所有权重之和) × normalize_weight"
            }),
        }
        
        # 为每个LoRA槽位添加可选输入（最多50个槽位，由JavaScript前端动态控制显示）
        optional = {}
        for i in range(1, 51):
            optional[f"lora_{i}_enabled"] = ("BOOLEAN", {
                "default": False,
                "tooltip": f"是否启用LoRA {i}"
            })
            optional[f"lora_{i}_name"] = (lora_list_with_none, {
                "default": "[None]",
                "tooltip": f"选择LoRA {i}的文件"
            })
            optional[f"lora_{i}_strength"] = ("FLOAT", {
                "default": 1.0,
                "min": -100.0,
                "max": 100.0,
                "step": 0.01,
                "round": 0.01,  # 两位小数精度
                "tooltip": f"LoRA {i}的强度（两位小数精度）"
            })
        
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("MODEL",)
    OUTPUT_TOOLTIPS = ("应用了多个LoRA的扩散模型",)
    FUNCTION = "load_multiple_loras_model_only"
    CATEGORY = "loaders"
    DESCRIPTION = "加载多个LoRA（最多50个）仅应用到MODEL，每个LoRA可单独开关和设置强度，权重通过normalize_weight归一化。槽位可通过前端动态添加。"

    def load_multiple_loras_model_only(self, model, normalize_weight, **kwargs):
        """
        加载多个LoRA并应用归一化权重（仅应用到MODEL）
        
        Args:
            model: 扩散模型
            normalize_weight: 归一化权重系数
            **kwargs: 包含所有lora_i_enabled, lora_i_name, lora_i_strength参数（可选）
        """
        # 收集所有启用的LoRA及其目标权重
        enabled_loras = []
        target_weights = []
        
        # 遍历所有可能的槽位（1-50），检查是否存在且启用
        for i in range(1, 51):
            # 检查输入是否存在（因为是optional，可能不存在）
            if f"lora_{i}_enabled" not in kwargs:
                continue
            enabled = kwargs.get(f"lora_{i}_enabled", False)
            if enabled:
                lora_name = kwargs.get(f"lora_{i}_name", "[None]")
                if lora_name and lora_name != "[None]":
                    strength = kwargs.get(f"lora_{i}_strength", 1.0)
                    enabled_loras.append((i, lora_name, strength))
                    target_weights.append(abs(strength))  # 使用绝对值计算归一化
        
        # 如果没有启用的LoRA，直接返回原始模型
        if not enabled_loras:
            return (model,)
        
        # 计算权重总和
        total_weight = sum(target_weights)
        
        # 如果总和为0，直接返回原始模型
        if total_weight == 0:
            return (model,)
        
        # 应用归一化权重转换
        current_model = model
        
        for i, lora_name, original_strength in enabled_loras:
            # 计算归一化后的实际权重
            if normalize_weight == 0:
                # 当normalize_weight为0时，不归一化，直接使用原始强度
                normalized_strength = original_strength
            else:
                normalized_strength = (abs(original_strength) / total_weight) * normalize_weight
                
                # 保持原始符号
                if original_strength < 0:
                    normalized_strength = -normalized_strength
            
            # 打印实际LoRA权重
            print(f"LoRA {i} ({lora_name}): 原始强度={original_strength:.2f}, 归一化后实际权重={normalized_strength:.4f}")
            
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
                
                # 应用LoRA到模型（仅MODEL，CLIP强度设为0）
                current_model, _ = comfy.sd.load_lora_for_models(
                    current_model, 
                    None, 
                    lora, 
                    normalized_strength, 
                    0.0  # CLIP强度设为0，只应用到MODEL
                )
            except Exception as e:
                print(f"加载LoRA {i} ({lora_name}) 时出错: {str(e)}")
                continue
        
        return (current_model,)
