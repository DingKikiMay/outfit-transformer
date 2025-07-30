import torch
from torch import nn
from typing import List
from ..data.datatypes import FashionItem
from dataclasses import dataclass
from .modules.encoder import ChineseCLIPItemEncoder 
from .outfit_transformer import OutfitTransformer, OutfitTransformerConfig
import numpy as np

@dataclass
class OutfitCLIPTransformerConfig(OutfitTransformerConfig):
    """Chinese-CLIP Transformer 专用配置"""
    # Chinese-CLIP 模型名称
    item_enc_model_name: str = "/root/outfit-transformer-api/outfit-transformer/OFA-Sys/chinese-clip-vit-base-patch16"
    # Chinese-CLIP 每个模态的输出维度（图像和文本各512维）
    item_enc_dim_per_modality: int = 512
    # 强制使用 concat 聚合方法，确保维度一致性
    aggregation_method: str = "concat"
    # 确保输出维度与Chinese-CLIP兼容
    d_embed: int = 512

class OutfitCLIPTransformer(OutfitTransformer):
    def __init__(
        self, 
        cfg: OutfitCLIPTransformerConfig = OutfitCLIPTransformerConfig()
    ):
        super().__init__(cfg)
        # 验证配置是否正确适配Chinese-CLIP
        self.validate_config()

    def validate_config(self):
       # 可选：对 cfg 的某些字段做检查
       assert self.cfg.aggregation_method == "concat", \
           "OutfitCLIPTransformer 目前只支持 concat 聚合方式"

    def _init_item_enc(self):
        """使用 Chinese-CLIP 作为 outfit encoder，支持中文和图片多模态输入。"""
        self.item_enc = ChineseCLIPItemEncoder(
            model_name=self.cfg.item_enc_model_name,
            enc_norm_out=self.cfg.item_enc_norm_out,
            aggregation_method=self.cfg.aggregation_method
        )
    
    def precompute_clip_embedding(self, item: List[FashionItem]) -> np.ndarray:
        """Precomputes the encoder(backbone) embeddings for a list of fashion items."""
        outfits = [[item_] for item_ in item]
        images, texts, mask = self._pad_and_mask_for_outfits(outfits)
        enc_outs = self.item_enc(images, texts) # [B, 1, D]
        embeddings = enc_outs[:, 0, :] # [B, D]
        return embeddings.detach().cpu().numpy()