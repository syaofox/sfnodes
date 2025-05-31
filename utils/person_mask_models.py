"""
XSEG模型管理模块
此模块管理多种XSEG模型的来源和配置
"""

from pathlib import Path
import folder_paths
from .downloader import download_model

# 模型配置
PERSON_MASK_MODELS = {
    'selfie_multiclass_256x256': {
        'url': 'https://huggingface.co/Syaofox/sfnodes/resolve/main/selfie_multiclass_256x256.tflite',
        'filename': 'selfie_multiclass_256x256.tflite',
        'description': '原始DFL-XSEG模型，针对人脸分割进行优化'
    }

}

def get_model_info(model_name):
    """获取指定模型的信息"""
    if model_name not in PERSON_MASK_MODELS:
        raise ValueError(f"未知的模型名称: {model_name}，可用模型有: {', '.join(PERSON_MASK_MODELS.keys())}")
    return PERSON_MASK_MODELS[model_name]

def get_model_path(model_name):
    """获取模型路径，如果不存在会自动下载"""
    model_info = get_model_info(model_name)
    save_loc = Path(folder_paths.models_dir) / 'sfnodes' / 'person_mask'
    save_loc.mkdir(parents=True, exist_ok=True)
    
    model_url = model_info['url']
    filename = model_info['filename']
    
    # 下载模型（如果尚未下载）
    download_model(model_url, save_loc, filename)
    
    return str(save_loc / filename)

def list_available_models():
    """列出所有可用的模型名称"""
    return list(PERSON_MASK_MODELS.keys())

def get_model_description(model_name):
    """获取模型描述"""
    model_info = get_model_info(model_name)
    return model_info.get('description', '无可用描述') 