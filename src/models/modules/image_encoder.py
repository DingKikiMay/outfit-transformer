# -*- coding:utf-8 -*-
"""
Author:
    Wonjun Oh, owj0421@naver.com
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import resnet18, ResNet18_Weights
from transformers import (
    AutoModel, 
    AutoTokenizer, 
    ChineseCLIPModel, ChineseCLIPProcessor
)
from typing import Literal
from torchvision import datasets, transforms
from abc import ABC, abstractmethod
from typing import List
from PIL import Image
from typing import Dict, Any, Optional

from ...utils.model_utils import freeze_model, mean_pooling

import numpy as np

# import ChineseCLIPImageEncoder相关
# Load model directly


class BaseImageEncoder(nn.Module, ABC):
    
    def __init__(self):
        super().__init__()
        
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
    
    @property
    @abstractmethod
    def image_size(self) -> int:
        raise NotImplementedError('The image_size property must be implemented by subclasses.')
    
    @property
    @abstractmethod
    def d_embed(self) -> int:
        raise NotImplementedError('The d_embed property must be implemented by subclasses.')

    @abstractmethod
    def _forward(
        self, 
        images: List[List[np.ndarray]]
    ) -> torch.Tensor:
        raise NotImplementedError('The embed method must be implemented by subclasses.')

    def forward(
        self, 
        images: List[List[np.ndarray]], 
        normalize: bool = True,
        *args, **kwargs
    ) -> torch.Tensor:
        if len(set(len(image_seq) for image_seq in images)) > 1:
            raise ValueError('All sequences in images should have the same length.')
        
        image_embeddings = self._forward(images, *args, **kwargs)
        
        if normalize:
            image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
            
        return image_embeddings
    

class Resnet18ImageEncoder(BaseImageEncoder):
    
    def __init__(
        self, d_embed: int = 64,
        size: int = 224, crop_size: int = 224, freeze: bool = False
    ):
        super().__init__()

        # Load pre-trained ResNet-18 and adjust the final layer to match d_embed
        self.d_embed = d_embed
        self.size = size
        self.crop_size = crop_size
        self.freeze = freeze
        
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(
            in_features=self.model.fc.in_features, 
            out_features=d_embed
        )
        if freeze:
            freeze_model(self.model)
        
        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    @property
    def image_size(self) -> int:
        return self.crop_size
    
    @property
    def d_embed(self) -> int:
        return self.d_embed
    
    def _forward(
        self, 
        images: List[List[np.ndarray]]
    ):  
        batch_size = len(images)
        images = sum(images, [])
        
        transformed_images = torch.stack(
            [self.transform(image) for image in images]
        ).to(self.device)
        image_embeddings = self.model(
            transformed_images
        )
        image_embeddings = image_embeddings.view(
            batch_size, -1, self.d_embed
        )
        
        return image_embeddings
    
# 实现基于 ChineseCLIP 的图像编码器
class ChineseCLIPImageEncoder(BaseImageEncoder):

    def __init__(
        self, 
        model_name_or_path: str = 'OFA-Sys/chinese-clip-vit-base-patch16',
        freeze: bool = True
    ):
        super().__init__()
        self.model = ChineseCLIPModel.from_pretrained(model_name_or_path)
        self.model.eval()
        if freeze:
            freeze_model(self.model)
        # 预处理器（processor），用于图片的 resize、归一化、转 tensor 等操作，保证输入格式和模型训练时一致。
        self.processor = ChineseCLIPProcessor.from_pretrained(model_name_or_path)
    
    # 返回模型要求的输入图片最短边尺寸（通常为 224），用于外部自动适配图片大小。
    @property
    def image_size(self) -> int:
        return self.processor.image_processor.size['shortest_edge']  # 返回224
    
    # 返回模型输出的embedding维度（通常为512），用于外部获取模型输出维度。
    @property
    def d_embed(self) -> int:
        return self.model.visual_projection.out_features  # 通常为512
    
    @torch.no_grad()
    def _forward(
        self, 
        images: List[List[np.ndarray]],
        # 传给 processor 的额外参数（如自定义 resize、padding 等）
        processor_kargs: Dict[str, Any] = None
    ):
        batch_size = len(images)
        # 把二维图片列表展平成一维（所有 outfit 的图片拼成一个大 list），方便批量处理
        images = sum(images, [])
        processor_kargs = processor_kargs if processor_kargs is not None else {}
        processor_kargs['return_tensors'] = 'pt'
        inputs = self.processor(images=images, **processor_kargs)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        image_embeds = self.model.get_image_features(**inputs)
        image_embeds = image_embeds.view(batch_size, -1, self.d_embed)
        # 最终的图片 embedding，shape 为 [batch, seq, d_embed]
        return image_embeds
