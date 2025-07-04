import glob    
import os
import folder_paths
from PIL import Image
from typing import Dict, Iterable, Tuple

from .utils.image_convert import images2tensor
from safetensors.torch import load_file, save_file

_CATEGORY = "sfnodes/hyperlora"


def inputs_def(required: Iterable = [], optional: Iterable = []) -> Dict[str, Dict]:
    return {
        'required': {
            k: v for k, v in required
        },
        'optional': {
            k: v for k, v in optional
        }
    }


def enum_field(name: str = 'enum_field', options: Iterable = []) -> Tuple[str, Tuple]:
    return name, (options, )


def list_chars(sub_dir):
    full_folder = os.path.join(folder_paths.models_dir, sub_dir)
    models = []
    if os.path.exists(full_folder):
        for fn in os.listdir(full_folder):
            if fn.endswith('.safetensors'):
                models.append(os.path.splitext(fn)[0])
    if len(models) == 0:
        models.append('Not found!')
    return models

class HyperLoRALoadCharLoRANode:

    @classmethod
    def INPUT_TYPES(cls):
        return inputs_def(required=[
            enum_field('charname', options=list_chars('hyper_lora/chars'))
        ])

    RETURN_TYPES = ('LORA', 'IMAGE', 'STRING')
    RETURN_NAMES = ('lora', 'images', 'charname')
    FUNCTION = 'execute'
    CATEGORY = _CATEGORY

    def execute(self, charname):

        filename = os.path.join(folder_paths.models_dir, 'hyper_lora/chars', f"{charname}.safetensors")
        assert os.path.isfile(filename), f'LoRA文件未找到: {filename}'
        lora = load_file(filename)
        # 查找同名图片
        img_pattern = os.path.join(folder_paths.models_dir, 'hyper_lora/chars', f"{charname}_??.png")
        img_files = sorted(glob.glob(img_pattern))
        img_list = []

        for img_path in img_files:
            try:
                img = Image.open(img_path).convert('RGB')
                img_list.append(img)
            except Exception as e:
                print(f"图片读取失败: {img_path}, {e}")
        images_tensor = images2tensor(img_list) if img_list else None
        return (lora, images_tensor, charname)
