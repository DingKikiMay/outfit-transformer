import json
import logging
import os
import pathlib
import pickle
import sys
import tempfile
from argparse import ArgumentParser
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

import wandb

from ..data import collate_fn
from ..data.datasets import polyvore
from ..models.load import load_model
from ..utils.distributed_utils import cleanup, setup
from ..utils.logger import get_logger
from ..utils.utils import seed_everything

SRC_DIR = pathlib.Path(__file__).parent.parent.parent.absolute()
LOGS_DIR = SRC_DIR / 'logs'
# 关闭 tokenizer 并行化
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.makedirs(LOGS_DIR, exist_ok=True)
# 定义嵌入保存路径模板
POLYVORE_PRECOMPUTED_REC_EMBEDDING_DIR = "{polyvore_dir}/precomputed_rec_embeddings"

'''
- 该脚本用于分布式地为服饰 item 生成推荐嵌入。
- 支持多 GPU 并行，自动分配数据。
- 每个进程处理一部分数据，最后分别保存嵌入结果。 
- 主要流程：参数解析 → 数据加载 → 分布式初始化 → 模型加载 → 嵌入生成 → 结果保存。
- 参数解析：
    --model_type：模型类型，可选 'original' 或 'clip'，默认 'clip'
    --polyvore_dir：Polyvore 数据集目录，默认 './datasets/polyvore'
    --polyvore_type：Polyvore 数据集类型，可选 'nondisjoint' 或 'disjoint'，默认 'nondisjoint'
    --batch_sz_per_gpu：每个 GPU 处理的 batch 大小，默认 128
    --n_workers_per_gpu：每个 GPU 使用的数据加载器线程数，默认 4
'''
def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model_type', type=str, choices=['original', 'clip'],
                        default='clip')
    parser.add_argument('--polyvore_dir', type=str, 
                        default='./datasets/polyvore')
    parser.add_argument('--polyvore_type', type=str, choices=['nondisjoint', 'disjoint'],
                        default='nondisjoint')
    parser.add_argument('--batch_sz_per_gpu', type=int,
                        default=128)
    parser.add_argument('--n_workers_per_gpu', type=int,
                        default=4)
    parser.add_argument('--checkpoint', type=str, 
                        default=None)
    parser.add_argument('--world_size', type=int, 
                        default=-1)
    parser.add_argument('--demo', action='store_true')
    
    return parser.parse_args()

# 设置数据加载器
def setup_dataloaders(rank, world_size, args):
    # 返回一个字典，键是item_id，值是item的元数据`​{item_id: item_metadata}`
    metadata = polyvore.load_metadata(args.polyvore_dir)

    # 加载预计算的嵌入字典
    embedding_dict = polyvore.load_embedding_dict(args.polyvore_dir)

    # 创建一个PolyvoreItemDataset对象，用于加载和处理Polyvore数据集中的物品数据
    item_dataset = polyvore.PolyvoreItemDataset(
        dataset_dir=args.polyvore_dir, metadata=metadata,
        load_image=False, embedding_dict=embedding_dict
    )
    # rank（进程编号）
    # world_size（总进程数）
    n_items = len(item_dataset)
    # 每个GPU处理的数据量
    n_items_per_gpu = n_items // world_size
    # 每个GPU处理的数据范围
    start_idx = n_items_per_gpu * rank
    end_idx = (start_idx + n_items_per_gpu) if rank < world_size - 1 else n_items
    item_dataset = torch.utils.data.Subset(item_dataset, range(start_idx, end_idx))
    
    item_dataloader = DataLoader(
        dataset=item_dataset, batch_size=args.batch_sz_per_gpu, shuffle=False,
        num_workers=args.n_workers_per_gpu, collate_fn=collate_fn.item_collate_fn
    )

    return item_dataloader


def compute(rank: int, world_size: int, args: Any):  
    # Setup
    # 设置分布式训练环境
    setup(rank, world_size)
    
    # Logging Setup
    # 设置日志记录器
    logger = get_logger('generate_rec_embeddings', LOGS_DIR, rank)
    logger.info(f'Logger Setup Completed')
    
    # Dataloaders 设置数据加载器
    item_dataloader = setup_dataloaders(rank, world_size, args)
    logger.info(f'Dataloaders Setup Completed')
    
    # Model setting 设置模型    
    model = load_model(model_type=args.model_type, checkpoint=args.checkpoint)
    model.eval()
    logger.info(f'Model Loaded')
    
    # 遍历数据集，前向推理得到嵌入，收集所有 item 的 id 和嵌入向量
    all_ids, all_embeddings = [], []
    with torch.no_grad():
        for batch in tqdm(item_dataloader):
            if args.demo and len(all_embeddings) > 10:
                break
            
            embeddings = model(batch, use_precomputed_embedding=True)  # (batch_size, d_embed)
            
            all_ids.extend([item.item_id for item in batch])
            all_embeddings.append(embeddings.detach().cpu().numpy())
    # 合并所有 batch 的嵌入，保存为 pickle 文件（每个进程一个文件）        
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    logger.info(f"Computed {len(all_embeddings)} embeddings")

    # numpy 数组保存
    save_dir = POLYVORE_PRECOMPUTED_REC_EMBEDDING_DIR.format(polyvore_dir=args.polyvore_dir)
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/polyvore_{rank}.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump({'ids': all_ids, 'embeddings': all_embeddings}, f)
    
    # DDP 清理
    cleanup()
    
    
if __name__ == '__main__':
    args = parse_args()
    
    if args.world_size == -1:
        args.world_size = torch.cuda.device_count()
        
    mp.spawn(
        compute, args=(args.world_size, args), 
        nprocs=args.world_size, join=True
    )