import math
import cv2
import numpy as np
import onnxruntime
import torch
from PIL import Image

import folder_paths
from comfy.utils import ProgressBar

from .utils.image_convert import np2tensor, tensor2np
from .utils.mask_utils import blur_mask, expand_mask, fill_holes, invert_mask
from .utils.region_models import (
    get_model_path, 
    list_available_models, 
    list_available_regions,
    list_region_groups,
    get_regions_in_group,
    get_region_indices,
    get_model_description
)

_CATEGORY = 'sfnodes/face_region'


class RegionExtractor:
    def __init__(self, region_model_path):
        self.region_model_path = region_model_path
        self.region_model = self.load_region_model()

    def load_region_model(self):
        available_providers = onnxruntime.get_available_providers()
        preferred_providers = []
        
        # 尝试使用GPU如果可用
        if 'CUDAExecutionProvider' in available_providers:
            preferred_providers.append('CUDAExecutionProvider')
        
        # 总是添加CPU作为备选
        preferred_providers.append('CPUExecutionProvider')
        
        return onnxruntime.InferenceSession(
            self.region_model_path,
            providers=preferred_providers,
        )

    def create_region_mask(self, image, region_indices, threshold=0.5):
        """创建面部区域遮罩"""
        if len(region_indices) == 0:
            print('\033[96m没有选择有效的面部区域\033[0m')
            return np.zeros(image.shape[:2], dtype=np.float32)
            
        # 准备输入数据
        model_size = (512, 512)  # BiSeNet模型的标准输入大小
        prepare_image = cv2.resize(image, model_size)
        prepare_image = prepare_image[:, :, ::-1].astype(np.float32) / 255  # BGR转RGB并归一化
        prepare_image = np.subtract(prepare_image, np.array([0.485, 0.456, 0.406]).astype(np.float32))  # 减去均值
        prepare_image = np.divide(prepare_image, np.array([0.229, 0.224, 0.225]).astype(np.float32))    # 除以标准差
        prepare_image = np.expand_dims(prepare_image, axis=0)
        prepare_image = prepare_image.transpose(0, 3, 1, 2)  # NHWC -> NCHW
        
        # 运行推理
        try:
            region_mask = self.region_model.run(None, {'input': prepare_image})[0][0]
        except Exception as e:
            print(f'\033[91m运行模型推理时出错: {e}\033[0m')
            return np.zeros(image.shape[:2], dtype=np.float32)
        
        # 处理输出 - 为选定的区域创建二值遮罩
        region_mask = np.isin(region_mask.argmax(0), region_indices)
        region_mask = cv2.resize(region_mask.astype(np.float32), image.shape[:2][::-1])
        
        return region_mask


class BiSeNetLoader:
    def __init__(self):
        # 初始化，不加载模型，等待get_region_extractor被调用
        self.selected_model = 'bisenet_resnet_34'  # 默认使用高精度模型
        self.region_extractor = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'model_choice': (list_available_models(), {'default': 'bisenet_resnet_34', 'tooltip': '选择要加载的模型'})
            }
        }

    RETURN_TYPES = ('REGION_EXTRACTOR',)
    RETURN_NAMES = ('region_extractor',)
    FUNCTION = 'get_region_extractor'
    CATEGORY = _CATEGORY
    DESCRIPTION = '加载面部区域分割模型'

    def get_region_extractor(self, model_choice='bisenet_resnet_34'):
        self.selected_model = model_choice
        model_path = get_model_path(self.selected_model)
        self.region_extractor = RegionExtractor(model_path)
        print(f"已加载面部区域分割模型: {self.selected_model} - {get_model_description(self.selected_model)}")
        return (self.region_extractor,)


class RegionSelector:
    @classmethod
    def INPUT_TYPES(cls):
        all_regions = list_available_regions()
        
        # 为每个区域创建一个开关选项
        region_checkboxes = {}
        for region in all_regions:
            region_checkboxes[f"use_{region}"] = ("BOOLEAN", {"default": False})
        
        return {
            'required': {},
            'optional': region_checkboxes
        }

    RETURN_TYPES = ('REGIONS',)
    RETURN_NAMES = ('selected_regions',)
    FUNCTION = 'select_regions'
    CATEGORY = _CATEGORY
    DESCRIPTION = '选择要生成遮罩的面部区域（通过复选框）'

    def select_regions(self, **kwargs):
        # 通过复选框选择区域
        selected_regions = []
        all_regions = list_available_regions()
        for region in all_regions:
            if kwargs.get(f"use_{region}", False):
                selected_regions.append(region)
        
        # 如果没有选择任何区域，默认选择皮肤
        if len(selected_regions) == 0:
            selected_regions = ['skin']
            
        print(f"已选择的面部区域: {', '.join(selected_regions)}")
        return (selected_regions,)


class GenerateRegionFaceMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'region_extractor': ('REGION_EXTRACTOR',{"tooltip": "面部区域分割模型"}),
                'input_image': ('IMAGE',{"tooltip": "输入图像"}),
                'regions': ('REGIONS',{"tooltip": "要生成的面部区域"}),
                'post_process': (['none', 'gaussian_blur'], {'default': 'gaussian_blur', 'tooltip': '选择后处理方法'}),
            },
            'optional': {
                'grow': ('INT', {'default': 0, 'min': -4096, 'max': 4096, 'step': 1, 'tooltip': '设置遮罩增长'}),
                'grow_percent': (
                    'FLOAT',
                    {'default': 0.00, 'min': 0.00, 'max': 2.0, 'step': 0.01},
                ),
                'grow_tapered': ('BOOLEAN', {'default': False, 'tooltip': '是否使用锥形增长'    }),
                'blur': ('INT', {'default': 4, 'min': 0, 'max': 4096, 'step': 1, 'tooltip': '设置模糊值'}),
                'fill': ('BOOLEAN', {'default': True, 'tooltip': '是否填充孔洞'}),
            },
        }

    RETURN_TYPES = (
        'MASK',
        'MASK',
        'IMAGE',
    )
    RETURN_NAMES = (
        'mask',
        'inverted_mask',
        'image',
    )
    FUNCTION = 'generate_mask'
    CATEGORY = _CATEGORY
    DESCRIPTION = '生成精确面部区域遮罩'

    def generate_mask(self, region_extractor, input_image, regions, post_process, grow=0, grow_percent=0.0, grow_tapered=False, blur=4, fill=True):
        region_indices = get_region_indices(regions)
        
        out_mask, out_inverted_mask, out_image = [], [], []

        steps = input_image.shape[0]
        if steps > 1:
            pbar = ProgressBar(steps)

        for i in range(steps):
            mask, processed_img = self._process_single_image(
                input_image[i], 
                region_extractor, 
                region_indices, 
                post_process, 
                grow, 
                grow_percent, 
                grow_tapered, 
                blur, 
                fill
            )
            out_mask.append(mask)
            out_inverted_mask.append(invert_mask(mask))
            out_image.append(processed_img)
            if steps > 1:
                pbar.update(1)

        return torch.stack(out_mask).squeeze(-1), torch.stack(out_inverted_mask).squeeze(-1), torch.stack(out_image)

    def _process_single_image(self, img, region_extractor, region_indices, post_process, grow, grow_percent, grow_tapered, blur, fill):
        """处理单张图像"""
        face = tensor2np(img)
        if face is None:
            print('\033[96m无效的输入图像\033[0m')
            return torch.zeros_like(img)[:, :, :1], torch.zeros_like(img)

        cv2_image = cv2.cvtColor(np.array(face), cv2.COLOR_RGB2BGR)
        region_mask = region_extractor.create_region_mask(cv2_image, region_indices)

        if region_mask is None or np.max(region_mask) == 0:
            print('\033[96m未能创建有效的区域遮罩\033[0m')
            return torch.zeros_like(img)[:, :, :1], torch.zeros_like(img)

        # 根据后处理设置进行额外的处理
        if post_process == 'gaussian_blur':
            region_mask = cv2.GaussianBlur(region_mask.clip(0, 1), (0, 0), 5).clip(0, 1)

        mask = self._process_mask(region_mask, img, grow, grow_percent, grow_tapered, blur, fill)
        processed_img = img * mask.repeat(1, 1, 3)
        return mask, processed_img

    def _process_mask(self, region_mask, img, grow, grow_percent, grow_tapered, blur, fill):
        """处理遮罩"""
        mask = np2tensor(region_mask).unsqueeze(0).squeeze(-1).clamp(0, 1).to(device=img.device)

        grow_count = int(grow_percent * max(mask.shape)) + grow
        if grow_count > 0:
            mask = expand_mask(mask, grow_count, grow_tapered)

        if fill:
            mask = fill_holes(mask)

        if blur > 0:
            mask = blur_mask(mask, blur)

        return mask.squeeze(0).unsqueeze(-1)


