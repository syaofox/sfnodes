"""
面部区域分割模型管理模块
此模块管理BiSeNet模型的来源和配置，用于精确的面部区域分割
"""

from pathlib import Path
import folder_paths
from .downloader import download_model

# 面部区域集合
FACE_MASK_REGION_SET = {
    'skin': 1,
    'left-eyebrow': 2,
    'right-eyebrow': 3,
    'left-eye': 4,
    'right-eye': 5,
    'glasses': 6,
    'nose': 10,
    'mouth': 11,
    'upper-lip': 12,
    'lower-lip': 13
}

# 区域组合
REGION_GROUPS = {
    'eyes': ['left-eye', 'right-eye'], 
    'eyebrows': ['left-eyebrow', 'right-eyebrow'],
    'fullmouth': ['mouth', 'upper-lip', 'lower-lip'],
    'lips': ['upper-lip', 'lower-lip'],
    'face': ['skin', 'nose', 'left-eyebrow', 'right-eyebrow', 'left-eye', 'right-eye']
}

# 模型配置
REGION_MODELS = {
    'bisenet_resnet_18': {
        'url': 'https://huggingface.co/Syaofox/sfnodes/resolve/main/bisenet_resnet_18.onnx',
        'filename': 'bisenet_resnet_18.onnx',
        'description': '轻量级BiSeNet模型，使用ResNet-18作为骨干网络'
    },
    'bisenet_resnet_34': {
        'url': 'https://huggingface.co/Syaofox/sfnodes/resolve/main/bisenet_resnet_34.onnx',
        'filename': 'bisenet_resnet_34.onnx',
        'description': '高精度BiSeNet模型，使用ResNet-34作为骨干网络'
    }
}

def get_model_info(model_name):
    """获取指定模型的信息"""
    if model_name not in REGION_MODELS:
        raise ValueError(f"未知的模型名称: {model_name}，可用模型有: {', '.join(REGION_MODELS.keys())}")
    return REGION_MODELS[model_name]

def get_model_path(model_name):
    """获取模型路径，如果不存在会自动下载"""
    model_info = get_model_info(model_name)
    save_loc = Path(folder_paths.models_dir) / 'sfnodes' / 'region'
    save_loc.mkdir(parents=True, exist_ok=True)
    
    model_url = model_info['url']
    filename = model_info['filename']
    
    # 下载模型（如果尚未下载）
    download_model(model_url, save_loc, filename)
    
    return str(save_loc / filename)

def list_available_models():
    """列出所有可用的模型名称"""
    return list(REGION_MODELS.keys())

def list_available_regions():
    """列出所有可用的面部区域"""
    return list(FACE_MASK_REGION_SET.keys())

def list_region_groups():
    """列出预定义的区域组合"""
    return list(REGION_GROUPS.keys())

def get_regions_in_group(group_name):
    """获取指定组合中的区域"""
    if group_name not in REGION_GROUPS:
        raise ValueError(f"未知的区域组合: {group_name}，可用组合有: {', '.join(REGION_GROUPS.keys())}")
    return REGION_GROUPS[group_name]

def get_region_indices(regions):
    """获取区域对应的索引值"""
    return [FACE_MASK_REGION_SET.get(region) for region in regions if region in FACE_MASK_REGION_SET]

def get_model_description(model_name):
    """获取模型描述"""
    model_info = get_model_info(model_name)
    return model_info.get('description', '无可用描述') 