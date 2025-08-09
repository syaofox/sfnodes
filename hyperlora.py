import glob    
import os
import folder_paths
from PIL import Image
from typing import Dict, Iterable, Tuple
import torch
import numpy as np
from typing import List

from .utils.image_convert import images2tensor
from safetensors.torch import load_file, save_file

_CATEGORY = "sfnodes/hyperlora"

def str_field(name: str = 'str_field', default: str = '', multiline: bool = False) -> Tuple[str, Tuple]:
    return name, ('STRING', {
        'default': default,
        'multiline': multiline
    })

def image_field(name: str = 'image') -> Tuple[str, Tuple]:
    return name, ('IMAGE', )

def custom_field(name: str = 'custom', type_name: str = 'CUSTOM') -> Tuple[str, Tuple]:
    return name, (type_name, )


def tensor2images(tensor: torch.Tensor) -> List[Image.Image]:
    images = []
    for i in range(tensor.shape[0]):
        image = tensor[i].cpu().numpy()
        image = (image.clip(0.0, 1.0) * 255.0).astype(np.uint8)
        images.append(Image.fromarray(image))
    return images

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



class HyperLoRASaveCharLoRANode:

    @classmethod
    def INPUT_TYPES(cls):
        return inputs_def(required=[
            str_field('char_name', default='char1'),
            custom_field('lora', type_name='LORA'),
        ], optional=[
            image_field('images')
        ])

    RETURN_TYPES = ()
    FUNCTION = 'execute'
    CATEGORY = 'HyperLoRA'
    OUTPUT_NODE = True

    def execute(self, char_name, lora, images=None):
        filename = os.path.join(folder_paths.models_dir, 'hyper_lora/chars', f"{char_name}.safetensors")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        save_file(lora, filename)
        # 保存图片batch
        if images is not None and hasattr(images, 'shape') and images.shape[0] > 0:
            img_list = tensor2images(images)
            for idx, img in enumerate(img_list, 1):
                img_path = os.path.join(folder_paths.models_dir, 'hyper_lora/chars', f"{char_name}_{idx:02d}.png")
                img.save(img_path)
        return (True, )