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
    # 只保留 Chinese-CLIP 相关参数
    item_enc_model_name: str = "OFA-Sys/chinese-clip-vit-base-patch16"
    item_enc_dim_per_modality: int = 512

class OutfitCLIPTransformer(OutfitTransformer):
    def __init__(
        self, 
        cfg: OutfitCLIPTransformerConfig = OutfitCLIPTransformerConfig()
    ):
        super().__init__(cfg)

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