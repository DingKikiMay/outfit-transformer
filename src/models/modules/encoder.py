from torch import nn
from dataclasses import dataclass, field

from typing import List, Tuple, Dict, Any, Union, Literal, Optional
from torch import Tensor
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import os

# Import custom modules
from ...data import datatypes
from .image_encoder import Resnet18ImageEncoder, ChineseCLIPImageEncoder
from .text_encoder import HuggingFaceTextEncoder, ChineseCLIPTextEncoder
from ...utils.model_utils import freeze_model, mean_pooling, aggregate_embeddings
from transformers import AutoModel, AutoTokenizer, AutoProcessor

# 多模态（图文）编码器
class ItemEncoder(nn.Module):
    def __init__(
        self, 
        model_name,
        enc_dim_per_modality,
        enc_norm_out,
        aggregation_method
    ):
        super().__init__()
        self.enc_dim_per_modality = enc_dim_per_modality
        self.aggregation_method = aggregation_method
        self.enc_norm_out = enc_norm_out
        self._build_encoders(model_name)
    
    def _build_encoders(self, model_name):
        self.image_enc = ChineseCLIPImageEncoder(
            model_name_or_path=model_name
        )
        self.text_enc = ChineseCLIPTextEncoder(
            model_name_or_path=model_name
        )
        
    @property
    def d_embed(self):  # 返回最终embedding的维度
        if self.aggregation_method == 'concat':
            d_model = self.enc_dim_per_modality * 2 
        else:
            d_model = self.enc_dim_per_modality
            
        return d_model
    
    @property
    def image_size(self):
        return self.image_enc.image_size

    def forward(self, images, texts, *args, **kwargs):        
        # Encode images and texts
        image_embeddings = self.image_enc(
            images, normalize=self.enc_norm_out, *args, **kwargs
        )
        text_embeddings = self.text_enc(
            texts, normalize=self.enc_norm_out, *args, **kwargs
        )
        # Aggregate embeddings 聚合嵌入
        encoder_outputs = aggregate_embeddings(
            image_embeddings=image_embeddings,
            text_embeddings=text_embeddings,
            aggregation_method=self.aggregation_method
        )
        
        return encoder_outputs

# 图像和文本特征在同一个语义空间中，更适合多模态任务
class ChineseCLIPItemEncoder(ItemEncoder):
    def __init__(
        self, 
        model_name,
        enc_norm_out,
        aggregation_method
    ):
        super().__init__(
            model_name=model_name,
            # 默认输出维度为512
            enc_dim_per_modality=512,
            enc_norm_out=enc_norm_out,
            aggregation_method=aggregation_method
        )
    
    def _build_encoders(self, model_name):
        self.image_enc = ChineseCLIPImageEncoder(
            model_name_or_path=model_name
        )
        self.text_enc = ChineseCLIPTextEncoder(
            model_name_or_path=model_name
        )