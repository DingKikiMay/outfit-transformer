from torch import nn
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Union, Literal, Optional, Sequence
from torch import Tensor
from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import os
import pathlib
from ..data.datatypes import (
    FashionCompatibilityQuery, FashionComplementaryQuery, FashionItem
)
from .modules.encoder import ChineseCLIPItemEncoder
from ..utils.model_utils import get_device

@dataclass
class OutfitTransformerConfig:
    padding: Literal['longest', 'max_length'] = 'longest'
    max_length: int = 16
    # 是否截断超长序列，保证输入不会超出 max_length
    truncation: bool = True
    
    # 统一使用 Chinese-CLIP 作为图文编码器
    item_enc_model_name: str = "OFA-Sys/chinese-clip-vit-base-patch16"  # Chinese-CLIP 模型名
    item_enc_dim_per_modality: int = 512  # Chinese-CLIP 输出维度
    # 是否对编码器输出做归一化（L2 norm），常用于多模态对齐和检索
    item_enc_norm_out: bool = True
    # 多模态融合方法，可选 concat、sum、mean
    aggregation_method: Literal['concat', 'sum', 'mean'] = 'concat'  # 若用concat，d_model=1024
    
    transformer_n_head: int = 16 # 当使用concat时，1024/16=64，当使用其他方法时，512/16=32
    transformer_d_ffn: int = 4096 # 当使用concat时，1024*4=4096，当使用其他方法时，512*4=2048
    transformer_n_layers: int = 6 # Original: 6
    transformer_dropout: float = 0.3 # Original: Unknown
    transformer_norm_out: bool = False
    # 原本128太小了，现在512
    d_embed: int = 512  # transformer输出embedding维度，可自定义


class OutfitTransformer(nn.Module):
    
    def __init__(self, cfg: Optional[OutfitTransformerConfig] = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else OutfitTransformerConfig()
        # 初始化多模态编码器
        self._init_item_enc()
        # 初始化 transformer 编码器
        self._init_style_enc()
        # 初始化可学习参数
        self._init_variables()
        
    def _init_item_enc(self):
        """使用 Chinese-CLIP 作为 outfit encoder，支持中文和图片多模态输入。"""
        self.item_enc = ChineseCLIPItemEncoder(
            model_name=self.cfg.item_enc_model_name,
            enc_norm_out=self.cfg.item_enc_norm_out,
            aggregation_method=self.cfg.aggregation_method
        )
    
    def _init_style_enc(self):
        """构建 transformer encoder，输入维度与 Chinese-CLIP 输出一致。"""
        # 动态调整前馈网络维度
        if self.cfg.aggregation_method == 'concat':
            # 当使用concat时，输入维度是1024，前馈网络应该是4096
            d_ffn = max(self.cfg.transformer_d_ffn, self.item_enc.d_embed * 4)
        else:
            # 当使用sum/mean时，输入维度是512，前馈网络应该是2048
            d_ffn = max(self.cfg.transformer_d_ffn, self.item_enc.d_embed * 4)
        
        print(f"Transformer配置: d_model={self.item_enc.d_embed}, d_ffn={d_ffn}, nhead={self.cfg.transformer_n_head}")
        
        style_enc_layer = nn.TransformerEncoderLayer(
            d_model=self.item_enc.d_embed,
            nhead=self.cfg.transformer_n_head,
            dim_feedforward=d_ffn,
            dropout=self.cfg.transformer_dropout,
            batch_first=True,
            norm_first=True,
            activation=F.mish,
        )
        self.style_enc = nn.TransformerEncoder(
            encoder_layer=style_enc_layer,
            num_layers=self.cfg.transformer_n_layers,
            enable_nested_tensor=False
        )
        self.predict_ffn = nn.Sequential(
            nn.Dropout(self.cfg.transformer_dropout),
            nn.Linear(self.item_enc.d_embed, 1),
            nn.Sigmoid()
        )
        self.embed_ffn = nn.Sequential(
            nn.Linear(self.item_enc.d_embed, self.cfg.d_embed, bias=False)
        )
    
    def _init_variables(self):
        image_size = (self.item_enc.image_size, self.item_enc.image_size)
        # self.image_query = Image.open(self.cfg.query_img_path).resize(image_size)
        self.image_pad = Image.new("RGB", image_size)
        
        # 获取Chinese-CLIP的pad token
        try:
            from transformers import ChineseCLIPProcessor
            processor = ChineseCLIPProcessor.from_pretrained(self.cfg.item_enc_model_name)
            self.text_pad = processor.tokenizer.pad_token
            print(f"使用Chinese-CLIP的pad token: {self.text_pad}")
        except:
            # 如果获取失败，使用默认的pad token
            self.text_pad = '[PAD]'
            print("使用默认pad token: [PAD]")
        
        # 任务嵌入，用于区分任务类型（兼容性预测、查询补全、单品嵌入）
        self.task_emb = nn.Parameter(
            torch.randn(self.item_enc.d_embed // 2) * 0.02, requires_grad=True
        )
        # 兼容性预测嵌入，用于区分兼容性预测和查询补全
        self.predict_emb = nn.Parameter(
            torch.randn(self.item_enc.d_embed // 2) * 0.02, requires_grad=True
        )
        # 查询补全嵌入，用于区分查询补全和单品嵌入
        self.embed_emb = nn.Parameter(
            torch.randn(self.item_enc.d_embed // 2) * 0.02, requires_grad=True
        )
        # 填充嵌入，用于填充超长序列
        self.pad_emb = nn.Parameter(
            torch.randn(self.item_enc.d_embed) * 0.02, requires_grad=True
        )
    
    def _get_max_length(self, sequences):
        # 如果padding为max_length，则返回max_length
        if self.cfg.padding == 'max_length':
            return self.cfg.max_length
        # 否则返回最长的序列长度
        max_length = max(len(seq) for seq in sequences)
        # 如果truncation为True，则返回最长的序列长度和max_length中的较小值
        return min(self.cfg.max_length, max_length) if self.cfg.truncation else max_length

    def _pad_sequences(self, sequences, pad_value, max_length):
        return [seq[:max_length] + [pad_value] * (max_length - len(seq)) for seq in sequences]

    def _pad_and_mask_for_outfits(self, outfits):
        max_length = self._get_max_length(outfits)
        images = self._pad_sequences(
            [[item.image for item in outfit] for outfit in outfits], 
            self.image_pad, max_length
        )
        texts = self._pad_sequences(
            [[f"{item.description}" for item in outfit] for outfit in outfits], 
            self.text_pad, max_length
        )
        mask = [[0] * len(seq) + [1] * (max_length - len(seq)) for seq in outfits]
        
        return images, texts, torch.BoolTensor(mask).to(self.device)
    
    def _pad_and_mask_for_embs(self, embs_of_outfits):
        # 过滤掉空的outfit
        valid_embs_of_outfits = []
        for embs_of_outfit in embs_of_outfits:
            if len(embs_of_outfit) > 0:
                valid_embs_of_outfits.append(embs_of_outfit)
        
        if len(valid_embs_of_outfits) == 0:
            # 如果没有有效的outfit，返回一个空的tensor
            return torch.empty((0, 1, self.item_enc.d_embed), dtype=torch.float, device=self.device), torch.BoolTensor([]).to(self.device)
        
        max_length = self._get_max_length(valid_embs_of_outfits)
        batch_size = len(valid_embs_of_outfits)

        embeddings = torch.empty((batch_size, max_length, self.item_enc.d_embed), 
                                 dtype=torch.float, device=self.device)
        mask = []

        for i, embs_of_outfit in enumerate(valid_embs_of_outfits):
            embs_of_outfit = torch.tensor(
                np.array(embs_of_outfit[:max_length]), dtype=torch.float
            ).to(self.device)
            length = len(embs_of_outfit)

            embeddings[i, :length] = embs_of_outfit
            embeddings[i, length:] = self.pad_emb  # 패딩 부분을 학습 가능한 벡터로 채움
            mask.append([0] * length + [1] * (max_length - length))
        
        return embeddings, torch.BoolTensor(mask).to(self.device)
    
    def _style_enc_forward(self, embs_of_inputs, src_key_padding_mask):
        if self.cfg.aggregation_method == 'concat':
            half_d_embed = self.item_enc.d_embed // 2
            normalized_embs = torch.cat([
                F.normalize(embs_of_inputs[:, :, :half_d_embed], p=2, dim=-1),
                F.normalize(embs_of_inputs[:, :, half_d_embed:], p=2, dim=-1)
            ], dim=-1)
        else:
            normalized_embs = F.normalize(embs_of_inputs, p=2, dim=-1)
        
        return self.style_enc(normalized_embs, src_key_padding_mask=src_key_padding_mask)
    
    def predict_score(self, query: List[FashionCompatibilityQuery], use_precomputed_embedding: bool = False) -> Tensor:
        outfits = [query_.outfit for query_ in query]
        if use_precomputed_embedding:
            embs_of_inputs = []
            for outfit in outfits:
                embs_of_inputs.append([
                    item_.embedding for item_ in outfit if hasattr(item_, 'embedding') and item_.embedding is not None
                ])
            embs_of_inputs, mask = self._pad_and_mask_for_embs(embs_of_inputs)
        else:
            images, texts, mask = self._pad_and_mask_for_outfits(outfits)
            embs_of_inputs = self.item_enc(images, texts)
            
        task_emb = torch.cat([self.task_emb, self.predict_emb], dim=-1)
        
        embs_of_inputs = torch.cat([
            task_emb.view(1, 1, -1).expand(len(query), -1, -1), # [B, 1, D]
            embs_of_inputs # [B, L, D]
        ], dim=1) # [B, L+1, D]
        mask = torch.cat([
            torch.zeros(len(query), 1, dtype=torch.bool, device=self.device), # [B, 1]
            mask # [B, L]
        ], dim=1) # [B, L+1]
        
        last_hidden_states = self._style_enc_forward(embs_of_inputs, src_key_padding_mask=mask)
        scores = self.predict_ffn(last_hidden_states[:, 0, :])
        
        return scores
    
    def embed_query(self, query: List[FashionComplementaryQuery], use_precomputed_embedding: bool=False) -> Tensor:
        outfits = [query_.outfit for query_ in query]
        if use_precomputed_embedding:
            embs_of_inputs = []
            for outfit in outfits:
                embs_of_inputs.append([
                    item_.embedding for item_ in outfit if hasattr(item_, 'embedding') and item_.embedding is not None
                ])
            embs_of_inputs, mask = self._pad_and_mask_for_embs(embs_of_inputs)
        else:
            images, texts, mask = self._pad_and_mask_for_outfits(outfits)
            embs_of_inputs = self.item_enc(images, texts)
            
        task_emb = torch.cat([self.task_emb, self.embed_emb], dim=-1)
        
        embs_of_inputs = torch.cat([
            task_emb.view(1, 1, -1).expand(len(query), -1, -1), embs_of_inputs
        ], dim=1)
        mask = torch.cat([
            torch.zeros(len(query), 1, dtype=torch.bool, device=self.device), mask
        ], dim=1)

        last_hidden_states = self._style_enc_forward(embs_of_inputs, src_key_padding_mask=mask)
        embeddings = self.embed_ffn(last_hidden_states[:, 0, :])
        
        return F.normalize(embeddings, p=2, dim=-1) if self.cfg.transformer_norm_out else embeddings

    def embed_item(self, item: List[FashionItem], use_precomputed_embedding: bool=False) -> Tensor:
        if use_precomputed_embedding:
            embs_of_inputs = []
            for item_ in item:
                if hasattr(item_, 'embedding') and item_.embedding is not None:
                    embs_of_inputs.append([item_.embedding])
            embs_of_inputs, mask = self._pad_and_mask_for_embs(embs_of_inputs)
        else:
            outfits = [[item_] for item_ in item]
            images, texts, mask = self._pad_and_mask_for_outfits(outfits)
            embs_of_inputs = self.item_enc(images, texts)
        
        last_hidden_states = self._style_enc_forward(embs_of_inputs, src_key_padding_mask=mask)
        embeddings = self.embed_ffn(last_hidden_states[:, 0, :]) # [B, D]
            
        return F.normalize(embeddings, p=2, dim=-1) if self.cfg.transformer_norm_out else embeddings

    def forward(
        self, 
        inputs: List[Union[FashionCompatibilityQuery, FashionComplementaryQuery, FashionItem]],
        *args, **kwargs
    ) -> Tensor:
        # 更健壮的类型判断，确保传递类型正确
        if all(isinstance(x, FashionCompatibilityQuery) for x in inputs):
            return self.predict_score(list(inputs), *args, **kwargs)
        elif all(isinstance(x, FashionComplementaryQuery) for x in inputs):
            return self.embed_query(list(inputs), *args, **kwargs)
        elif all(isinstance(x, FashionItem) for x in inputs):
            return self.embed_item(list(inputs), *args, **kwargs)
        else:
            raise ValueError("Invalid input type.")
        
    @property
    def device(self) -> torch.device:
        """Returns the device on which the model's parameters are stored."""
        return get_device(self)