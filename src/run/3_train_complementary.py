'''import json
import logging
import os
import pathlib
import sys
import tempfile
from argparse import ArgumentParser
from typing import Any, Optional

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
from ..data.datasets import polyvore_utils as polyvore
from ..evaluation.metrics import compute_cir_scores, compute_cp_scores
from ..models.load import load_model
from ..utils.distributed_utils import cleanup, gather_results, setup
from ..utils.logger import get_logger
from ..utils.loss import InBatchTripletMarginLoss
from ..utils.utils import seed_everything

SRC_DIR = pathlib.Path(__file__).parent.parent.parent.absolute()
CHECKPOINT_DIR = SRC_DIR / 'checkpoints'
RESULT_DIR = SRC_DIR / 'results'
LOGS_DIR = SRC_DIR / 'logs'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["USE_LIBUV"] = "0"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

metadata = None
all_embeddings_dict = None

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model_type', type=str, choices=['original', 'clip'],
                        default='clip')
    parser.add_argument('--polyvore_dir', type=str, 
                        default='./src/data/datasets/polyvore')
    parser.add_argument('--polyvore_type', type=str, choices=['nondisjoint', 'disjoint'],
                        default='nondisjoint')
    parser.add_argument('--batch_sz_per_gpu', type=int,
                        default=64)
    parser.add_argument('--n_workers_per_gpu', type=int,
                        default=4)
    parser.add_argument('--n_epochs', type=int,
                        default=200)
    parser.add_argument('--lr', type=float,
                        default=2e-5)
    parser.add_argument('--accumulation_steps', type=int,
                        default=4)
    parser.add_argument('--wandb_key', type=str, 
                        default=None)
    parser.add_argument('--seed', type=int, 
                        default=42)
    parser.add_argument('--checkpoint', type=str, 
                        default=None)
    parser.add_argument('--world_size', type=int, 
                        default=-1)
    parser.add_argument('--project_name', type=str, 
                        default=None)
    parser.add_argument('--demo', action='store_true')
    
    return parser.parse_args()


def setup_dataloaders(rank, world_size, args):    
    metadata = polyvore.load_metadata(args.polyvore_dir)
    embedding_dict = polyvore.load_embedding_dict(args.polyvore_dir)
    
    train = polyvore.PolyvoreTripletDataset(
        dataset_dir=args.polyvore_dir, dataset_type=args.polyvore_type,
        dataset_split='train', metadata=metadata, embedding_dict=embedding_dict
    )
    valid = polyvore.PolyvoreFillInTheBlankDataset(
        dataset_dir=args.polyvore_dir, dataset_type=args.polyvore_type,
        dataset_split='valid', metadata=metadata, embedding_dict=embedding_dict
    )
    
    if world_size == 1:
        train_dataloader = DataLoader(
            dataset=train, batch_size=args.batch_sz_per_gpu, shuffle=True,
            num_workers=args.n_workers_per_gpu, collate_fn=collate_fn.triplet_collate_fn
        )
        valid_dataloader = DataLoader(
            dataset=valid, batch_size=args.batch_sz_per_gpu, shuffle=False,
            num_workers=args.n_workers_per_gpu, collate_fn=collate_fn.fitb_collate_fn
        )
        
    else:
        train_sampler = DistributedSampler(
            train, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
        valid_sampler = DistributedSampler(
            valid, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True
        )
        train_dataloader = DataLoader(
            dataset=train, batch_size=args.batch_sz_per_gpu, shuffle=False,
            num_workers=args.n_workers_per_gpu, collate_fn=collate_fn.triplet_collate_fn, sampler=train_sampler
        )
        valid_dataloader = DataLoader(
            dataset=valid, batch_size=args.batch_sz_per_gpu, shuffle=False,
            num_workers=args.n_workers_per_gpu, collate_fn=collate_fn.fitb_collate_fn, sampler=valid_sampler
        )

    return train_dataloader, valid_dataloader


def train_step(
    rank, world_size, 
    args, epoch, logger, wandb_run,
    model, optimizer, scheduler, loss_fn, dataloader
):
    model.train()
    pbar = tqdm(dataloader, desc=f'Train Epoch {epoch+1}/{args.n_epochs}')
    
    all_loss, all_preds, all_labels = torch.zeros(1, device=rank), [], []
    for i, data in enumerate(pbar):
        if args.demo and i > 2:
            break
        
        # 过滤掉无效的数据
        valid_indices = []
        for j, (query, answer) in enumerate(zip(data['query'], data['answer'])):
            # 检查query中的outfit是否有有效的embedding
            valid_query = all(hasattr(item, 'embedding') and item.embedding is not None for item in query.outfit)
            # 检查answer是否有有效的embedding
            valid_answer = hasattr(answer, 'embedding') and answer.embedding is not None
            # 检查outfit是否为空
            valid_outfit = len(query.outfit) > 0
            
            if valid_query and valid_answer and valid_outfit:
                valid_indices.append(j)
        
        if len(valid_indices) == 0:
            continue  # 跳过这个batch
            
        # 只保留有效的数据
        filtered_query = [data['query'][j] for j in valid_indices]
        filtered_answer = [data['answer'][j] for j in valid_indices]
        
        if len(filtered_query) == 0:
            continue
            
        batched_q_emb = model(filtered_query, use_precomputed_embedding=True) # (batch_sz, embedding_dim)
        batched_a_emb = model(filtered_answer, use_precomputed_embedding=True) # (batch_sz, embedding_dim)
        
        loss = loss_fn(batched_q_emb, batched_a_emb)
        loss = loss / args.accumulation_steps
        
        loss.backward()
        if (i + 1) % args.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        
        dists = torch.cdist(batched_q_emb, batched_a_emb, p=2)  # (batch_sz, batch_sz)
        preds = torch.argmin(dists, dim=1) # (batch_sz,)
        labels = torch.arange(len(preds), device=rank)

        # Accumulate Results
        all_loss += loss.item() * args.accumulation_steps / len(dataloader)
        all_preds.append(preds.detach())
        all_labels.append(labels.detach())

        # Logging
        score = compute_cir_scores(all_preds[-1], all_labels[-1])
        logs = {
            'loss': loss.item() * args.accumulation_steps,
            'steps': len(pbar) * epoch + i,
            'lr': scheduler.get_last_lr()[0] if scheduler else args.lr,
            **score
        }
        pbar.set_postfix(**logs)
        if args.wandb_key and rank == 0:
            logs = {f'train_{k}': v for k, v in logs.items()}
            wandb_run.log(logs)
    
    all_preds = torch.cat(all_preds).to(rank)
    all_labels = torch.cat(all_labels).to(rank)

    gathered_loss, gathered_preds, gathered_labels = gather_results(all_loss, all_preds, all_labels)
    output = {'loss': gathered_loss.item(), **compute_cir_scores(gathered_preds, gathered_labels)} if rank == 0 else {}
    output = {f'train_{key}': value for key, value in output.items()}
    logger.info(f'Epoch {epoch+1}/{args.n_epochs} --> End {output}')

    return output


@torch.no_grad()
def valid_step(
    rank, world_size, 
    args, epoch, logger, wandb_run,
    model, loss_fn, dataloader
):
    model.eval()
    pbar = tqdm(dataloader, desc=f'Valid Epoch {epoch+1}/{args.n_epochs}')
    
    all_loss, all_preds, all_labels = torch.zeros(1, device=rank), [], []
    for i, data in enumerate(pbar):
        if args.demo and i > 2:
            break
        
        # 过滤掉无效的数据
        valid_indices = []
        for j, (query, candidates) in enumerate(zip(data['query'], data['candidates'])):
            # 检查query中的outfit是否有有效的embedding
            valid_query = all(hasattr(item, 'embedding') and item.embedding is not None for item in query.outfit)
            # 检查candidates是否有有效的embedding
            valid_candidates = all(hasattr(item, 'embedding') and item.embedding is not None for item in candidates)
            # 检查outfit是否为空
            valid_outfit = len(query.outfit) > 0
            
            if valid_query and valid_candidates and valid_outfit:
                valid_indices.append(j)
        
        if len(valid_indices) == 0:
            continue  # 跳过这个batch
            
        # 只保留有效的数据
        filtered_query = [data['query'][j] for j in valid_indices]
        filtered_candidates = [data['candidates'][j] for j in valid_indices]
        filtered_labels = [data['label'][j] for j in valid_indices]
        
        if len(filtered_query) == 0:
            continue
            
        batched_q_emb = model(filtered_query, use_precomputed_embedding=True).unsqueeze(1) # (batch_sz, 1, embedding_dim)
        batched_c_embs = model(sum(filtered_candidates, []), use_precomputed_embedding=True) # (batch_sz * 4, embedding_dim)
        batched_c_embs = batched_c_embs.view(-1, 4, batched_c_embs.shape[1]) # (batch_sz, 4, embedding_dim)
        
        dists = torch.norm(batched_q_emb - batched_c_embs, dim=-1) # (batch_sz, 4)
        preds = torch.argmin(dists, dim=-1) # (batch_sz,)
        labels = torch.tensor(filtered_labels, device=rank)

        # Accumulate Results
        all_preds.append(preds.detach())
        all_labels.append(labels.detach())

        # Logging
        score = compute_cir_scores(all_preds[-1], all_labels[-1])
        logs = {
            'steps': len(pbar) * epoch + i,
            **score
        }
        pbar.set_postfix(**logs)
        if args.wandb_key and rank == 0:
            logs = {f'valid_{k}': v for k, v in logs.items()}
            wandb_run.log(logs)
    
    all_preds = torch.cat(all_preds).to(rank)
    all_labels = torch.cat(all_labels).to(rank)

    _, gathered_preds, gathered_labels = gather_results(all_loss, all_preds, all_labels)
    output = {**compute_cir_scores(gathered_preds, gathered_labels)} if rank == 0 else {}
    output = {f'valid_{key}': value for key, value in output.items()}
    logger.info(f'Epoch {epoch+1}/{args.n_epochs} --> End {output}')

    return output

    
def train(
    rank: int, world_size: int, args: Any,
    wandb_run: Optional[wandb.sdk.wandb_run.Run] = None
):  
    # Setup
    setup(rank, world_size)
    
    # Logging Setup
    project_name = f'complementary_{args.model_type}_' + (
        args.project_name if args.project_name 
        else (wandb_run.name if wandb_run else 'test')
    )
    logger = get_logger(project_name, LOGS_DIR, rank)
    logger.info(f'Logger Setup Completed')
    
    # Dataloaders
    train_dataloader, valid_dataloader = setup_dataloaders(rank, world_size, args)
    logger.info(f'Dataloaders Setup Completed')
    
    # Model setting
    model = load_model(model_type=args.model_type, checkpoint=args.checkpoint)
    logger.info(f'Model Loaded and Wrapped with DDP')
    
    # Optimizer, Scheduler, Loss Function
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr, epochs=args.n_epochs, steps_per_epoch=int(len(train_dataloader) / args.accumulation_steps),
        pct_start=0.3, anneal_strategy='cos', div_factor=25, final_div_factor=1e4
    )
    loss_fn = InBatchTripletMarginLoss(margin=2.0, reduction='mean')
    logger.info(f'Optimizer and Scheduler Setup Completed')

    # 添加最佳模型跟踪
    best_valid_score = -float('inf')
    best_epoch = 0
    best_model_state = None

    # Training Loop
    for epoch in range(args.n_epochs):
        if world_size > 1:
            train_dataloader.sampler.set_epoch(epoch)
        train_logs = train_step(
            rank, world_size, 
            args, epoch, logger, wandb_run,
            model, optimizer, scheduler, loss_fn, train_dataloader
        )

        valid_logs = valid_step(
            rank, world_size, 
            args, epoch, logger, wandb_run,
            model, loss_fn, valid_dataloader
        )
        
        # 检查是否为最佳模型
        valid_score = valid_logs.get('valid_accuracy', -float('inf'))
        if valid_score > best_valid_score:
            best_valid_score = valid_score
            best_epoch = epoch + 1
            # 保存最佳模型状态
            if world_size > 1:
                best_model_state = model.module.state_dict()
            else:
                best_model_state = model.state_dict()
            logger.info(f'新的最佳模型！Epoch {epoch+1}, 验证准确率: {valid_score:.4f}')
        
        checkpoint_dir = CHECKPOINT_DIR / project_name
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}.pth')
            
        if rank == 0:
            torch.save({
                'config': model.module.cfg.__dict__ if world_size > 1 else model.cfg.__dict__,
                'model': model.state_dict()
            }, checkpoint_path)
            
            score_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}_score.json')
            with open(score_path, 'w') as f:
                score = {**train_logs, **valid_logs}
                json.dump(score, f, indent=4)
            logger.info(f'Checkpoint saved at {checkpoint_path}')
            
        dist.barrier()
        map_location = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
        state_dict = torch.load(checkpoint_path, map_location=map_location)
        model.load_state_dict(state_dict['model'])
        logger.info(f'Checkpoint loaded from {checkpoint_path}')

    # 保存最佳模型
    if best_model_state is not None and rank == 0:
        best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save({
            'config': model.module.cfg.__dict__ if world_size > 1 else model.cfg.__dict__,
            'model': best_model_state,
            'best_epoch': best_epoch,
            'best_valid_score': best_valid_score
        }, best_checkpoint_path)
        
        # 保存最佳模型信息
        best_info_path = os.path.join(checkpoint_dir, 'best_model_info.json')
        with open(best_info_path, 'w') as f:
            json.dump({
                'best_epoch': best_epoch,
                'best_valid_score': best_valid_score,
                'total_epochs': args.n_epochs
            }, f, indent=4)
        
        logger.info(f'最佳模型已保存！Epoch {best_epoch}, 验证准确率: {best_valid_score:.4f}')
        logger.info(f'最佳模型路径: {best_checkpoint_path}')
    
    # 分布式同步，等待主进程保存最佳模型
    dist.barrier()

    cleanup()


if __name__ == '__main__':
    args = parse_args()
    
    if args.world_size == -1:
        args.world_size = torch.cuda.device_count()
        
    if args.wandb_key:
        wandb.login(key=args.wandb_key)
        wandb_run = wandb.init(project='outfit-transformer-cir', config=args.__dict__)
    else:
        wandb_run = None
    
    mp.spawn(
        train, args=(args.world_size, args, wandb_run), 
        nprocs=args.world_size, join=True
    )
    '''
import json
import logging
import os
import pathlib
import sys
import tempfile
from argparse import ArgumentParser
from typing import Any, Optional

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

import swanlab

from ..data import collate_fn
from ..data.datasets import polyvore_utils as polyvore
from ..evaluation.metrics import compute_cir_scores, compute_cp_scores
from ..models.load import load_model
from ..utils.distributed_utils import cleanup, gather_results, setup
from ..utils.logger import get_logger
from ..utils.loss import InBatchTripletMarginLoss
from ..utils.utils import seed_everything

SRC_DIR = pathlib.Path(__file__).parent.parent.parent.absolute()
# 使用数据盘和文件存储，避免系统盘空间不足
CHECKPOINT_DIR = pathlib.Path('/root/autodl-tmp/checkpoints')  # 数据盘：模型文件
RESULT_DIR = pathlib.Path('/root/autodl-fs/results')  # 文件存储：结果文件
LOGS_DIR = pathlib.Path('/root/autodl-fs/logs')  # 文件存储：日志文件
os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

metadata = None
all_embeddings_dict = None

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model_type', type=str, choices=['original', 'clip'],
                        default='clip')
    parser.add_argument('--polyvore_dir', type=str, 
                        default='./src/data/datasets/polyvore')
    parser.add_argument('--polyvore_type', type=str, choices=['nondisjoint', 'disjoint'],
                        default='nondisjoint')
    parser.add_argument('--batch_sz_per_gpu', type=int,
                        default=64)
    parser.add_argument('--n_workers_per_gpu', type=int,
                        default=4)
    parser.add_argument('--n_epochs', type=int,
                        default=200)
    parser.add_argument('--lr', type=float,
                        default=2e-5)
    parser.add_argument('--accumulation_steps', type=int,
                        default=4)
    parser.add_argument('--swanlab_project', type=str, 
                        default='outfit-transformer-cir')
    parser.add_argument('--swanlab_experiment_name', type=str, 
                        default=None)
    parser.add_argument('--seed', type=int, 
                        default=42)
    parser.add_argument('--checkpoint', type=str, 
                        default=None)
    parser.add_argument('--world_size', type=int, 
                        default=-1)
    parser.add_argument('--project_name', type=str, 
                        default=None)
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--save_interval', type=int, default=20,
                        help='Save checkpoint every N epochs (default: 20)')
    parser.add_argument('--save_last_n', type=int, default=3,
                        help='Keep only the last N checkpoints to save disk space (default: 3)')
    
    return parser.parse_args()


def setup_dataloaders(rank, world_size, args):    
    metadata = polyvore.load_metadata(args.polyvore_dir)
    embedding_dict = polyvore.load_embedding_dict(args.polyvore_dir)
    
    train = polyvore.PolyvoreTripletDataset(
        dataset_dir=args.polyvore_dir, dataset_type=args.polyvore_type,
        dataset_split='train', metadata=metadata, embedding_dict=embedding_dict
    )
    valid = polyvore.PolyvoreFillInTheBlankDataset(
        dataset_dir=args.polyvore_dir, dataset_type=args.polyvore_type,
        dataset_split='valid', metadata=metadata, embedding_dict=embedding_dict
    )
    
    if world_size == 1:
        train_dataloader = DataLoader(
            dataset=train, batch_size=args.batch_sz_per_gpu, shuffle=True,
            num_workers=args.n_workers_per_gpu, collate_fn=collate_fn.triplet_collate_fn
        )
        valid_dataloader = DataLoader(
            dataset=valid, batch_size=args.batch_sz_per_gpu, shuffle=False,
            num_workers=args.n_workers_per_gpu, collate_fn=collate_fn.fitb_collate_fn
        )
        
    else:
        train_sampler = DistributedSampler(
            train, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
        valid_sampler = DistributedSampler(
            valid, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True
        )
        train_dataloader = DataLoader(
            dataset=train, batch_size=args.batch_sz_per_gpu, shuffle=False,
            num_workers=args.n_workers_per_gpu, collate_fn=collate_fn.triplet_collate_fn, sampler=train_sampler
        )
        valid_dataloader = DataLoader(
            dataset=valid, batch_size=args.batch_sz_per_gpu, shuffle=False,
            num_workers=args.n_workers_per_gpu, collate_fn=collate_fn.fitb_collate_fn, sampler=valid_sampler
        )

    return train_dataloader, valid_dataloader


def train_step(
    rank, world_size, 
    args, epoch, logger, swanlab_run,
    model, optimizer, scheduler, loss_fn, dataloader
):
    model.train()
    pbar = tqdm(dataloader, desc=f'Train Epoch {epoch+1}/{args.n_epochs}')
    
    all_loss, all_preds, all_labels = torch.zeros(1, device=rank), [], []
    for i, data in enumerate(pbar):
        if args.demo and i > 2:
            break
        
        # 过滤掉无效的数据
        valid_indices = []
        for j, (query, answer) in enumerate(zip(data['query'], data['answer'])):
            # 检查query中的outfit是否有有效的embedding
            valid_query = all(hasattr(item, 'embedding') and item.embedding is not None for item in query.outfit)
            # 检查answer是否有有效的embedding
            valid_answer = hasattr(answer, 'embedding') and answer.embedding is not None
            # 检查outfit是否为空
            valid_outfit = len(query.outfit) > 0
            
            if valid_query and valid_answer and valid_outfit:
                valid_indices.append(j)
        
        if len(valid_indices) == 0:
            continue  # 跳过这个batch
            
        # 只保留有效的数据
        filtered_query = [data['query'][j] for j in valid_indices]
        filtered_answer = [data['answer'][j] for j in valid_indices]
        
        if len(filtered_query) == 0:
            continue
            
        batched_q_emb = model(filtered_query, use_precomputed_embedding=True) # (batch_sz, embedding_dim)
        batched_a_emb = model(filtered_answer, use_precomputed_embedding=True) # (batch_sz, embedding_dim)
        
        loss = loss_fn(batched_q_emb, batched_a_emb)
        loss = loss / args.accumulation_steps
        
        loss.backward()
        if (i + 1) % args.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        
        dists = torch.cdist(batched_q_emb, batched_a_emb, p=2)  # (batch_sz, batch_sz)
        preds = torch.argmin(dists, dim=1) # (batch_sz,)
        labels = torch.arange(len(preds), device=rank)

        # Accumulate Results
        all_loss += loss.item() * args.accumulation_steps / len(dataloader)
        all_preds.append(preds.detach())
        all_labels.append(labels.detach())

        # Logging - 显示累积的准确率而不是当前batch的准确率
        if len(all_preds) > 0:
            # 计算累积的准确率
            accumulated_preds = torch.cat(all_preds)
            accumulated_labels = torch.cat(all_labels)
            accumulated_score = compute_cir_scores(accumulated_preds, accumulated_labels)
        else:
            accumulated_score = compute_cir_scores(all_preds[-1], all_labels[-1])
        
        logs = {
            'loss': loss.item() * args.accumulation_steps,
            'steps': len(pbar) * epoch + i,
            'lr': scheduler.get_last_lr()[0] if scheduler else args.lr,
            **accumulated_score
        }
        pbar.set_postfix(**logs)
        if swanlab_run and rank == 0:
            logs = {f'train_{k}': v for k, v in logs.items()}
            swanlab_run.log(logs)
    
    all_preds = torch.cat(all_preds).to(rank)
    all_labels = torch.cat(all_labels).to(rank)

    gathered_loss, gathered_preds, gathered_labels = gather_results(all_loss, all_preds, all_labels)
    output = {'loss': gathered_loss.item(), **compute_cir_scores(gathered_preds, gathered_labels)} if rank == 0 else {}
    output = {f'train_{key}': value for key, value in output.items()}
    logger.info(f'Epoch {epoch+1}/{args.n_epochs} --> End {output}')

    return output


@torch.no_grad()
def valid_step(
    rank, world_size, 
    args, epoch, logger, swanlab_run,
    model, loss_fn, dataloader
):
    model.eval()
    pbar = tqdm(dataloader, desc=f'Valid Epoch {epoch+1}/{args.n_epochs}')
    
    all_loss, all_preds, all_labels = torch.zeros(1, device=rank), [], []
    for i, data in enumerate(pbar):
        if args.demo and i > 2:
            break
        
        # 过滤掉无效的数据
        valid_indices = []
        for j, (query, candidates) in enumerate(zip(data['query'], data['candidates'])):
            # 检查query中的outfit是否有有效的embedding
            valid_query = all(hasattr(item, 'embedding') and item.embedding is not None for item in query.outfit)
            # 检查candidates是否有有效的embedding
            valid_candidates = all(hasattr(item, 'embedding') and item.embedding is not None for item in candidates)
            # 检查outfit是否为空
            valid_outfit = len(query.outfit) > 0
            
            if valid_query and valid_candidates and valid_outfit:
                valid_indices.append(j)
        
        if len(valid_indices) == 0:
            continue  # 跳过这个batch
            
        # 只保留有效的数据
        filtered_query = [data['query'][j] for j in valid_indices]
        filtered_candidates = [data['candidates'][j] for j in valid_indices]
        filtered_labels = [data['label'][j] for j in valid_indices]
        
        if len(filtered_query) == 0:
            continue
            
        batched_q_emb = model(filtered_query, use_precomputed_embedding=True).unsqueeze(1) # (batch_sz, 1, embedding_dim)
        batched_c_embs = model(sum(filtered_candidates, []), use_precomputed_embedding=True) # (batch_sz * 4, embedding_dim)
        batched_c_embs = batched_c_embs.view(-1, 4, batched_c_embs.shape[1]) # (batch_sz, 4, embedding_dim)
        
        dists = torch.norm(batched_q_emb - batched_c_embs, dim=-1) # (batch_sz, 4)
        preds = torch.argmin(dists, dim=-1) # (batch_sz,)
        labels = torch.tensor(filtered_labels, device=rank)

        # Accumulate Results
        all_preds.append(preds.detach())
        all_labels.append(labels.detach())

        # Logging - 显示累积的准确率而不是当前batch的准确率
        if len(all_preds) > 0:
            # 计算累积的准确率
            accumulated_preds = torch.cat(all_preds)
            accumulated_labels = torch.cat(all_labels)
            accumulated_score = compute_cir_scores(accumulated_preds, accumulated_labels)
        else:
            accumulated_score = compute_cir_scores(all_preds[-1], all_labels[-1])
        
        logs = {
            'steps': len(pbar) * epoch + i,
            **accumulated_score
        }
        pbar.set_postfix(**logs)
        if swanlab_run and rank == 0:
            logs = {f'valid_{k}': v for k, v in logs.items()}
            swanlab_run.log(logs)
    
    all_preds = torch.cat(all_preds).to(rank)
    all_labels = torch.cat(all_labels).to(rank)

    _, gathered_preds, gathered_labels = gather_results(all_loss, all_preds, all_labels)
    output = {**compute_cir_scores(gathered_preds, gathered_labels)} if rank == 0 else {}
    output = {f'valid_{key}': value for key, value in output.items()}
    logger.info(f'Epoch {epoch+1}/{args.n_epochs} --> End {output}')

    return output

    
def train(
    rank: int, world_size: int, args: Any,
    swanlab_run: Optional[swanlab.Run] = None
):  
    # Setup
    setup(rank, world_size)
    
    # 初始化SwanLab（只在主进程中）
    if args.swanlab_project and rank == 0:
        try:
            swanlab_run = swanlab.init(project=args.swanlab_project, name=args.swanlab_experiment_name)
        except Exception as e:
            print(f"Warning: Failed to initialize SwanLab: {e}")
            swanlab_run = None
    
    # Logging Setup
    project_name = f'complementary_{args.model_type}_' + (
        args.project_name if args.project_name 
        else (args.swanlab_experiment_name if args.swanlab_experiment_name else 'test')
    )
    logger = get_logger(project_name, LOGS_DIR, rank)
    logger.info(f'Logger Setup Completed')
    
    # Dataloaders
    train_dataloader, valid_dataloader = setup_dataloaders(rank, world_size, args)
    logger.info(f'Dataloaders Setup Completed')
    
    # Model setting
    model = load_model(model_type=args.model_type, checkpoint=args.checkpoint)
    logger.info(f'Model Loaded and Wrapped with DDP')
    
    # Optimizer, Scheduler, Loss Function
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr, epochs=args.n_epochs, steps_per_epoch=int(len(train_dataloader) / args.accumulation_steps),
        pct_start=0.3, anneal_strategy='cos', div_factor=25, final_div_factor=1e4
    )
    loss_fn = InBatchTripletMarginLoss(margin=2.0, reduction='mean')
    logger.info(f'Optimizer and Scheduler Setup Completed')

    # 添加最佳模型跟踪
    best_valid_score = -float('inf')
    best_epoch = 0
    best_model_state = None
    # 创建checkpoint保存目录
    checkpoint_dir = os.path.join('/root/autodl-tmp', f'{project_name}_checkpoints')
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 记录已保存的checkpoint文件，用于清理旧文件
    saved_checkpoints = []

    # Training Loop
    for epoch in range(args.n_epochs):
        if world_size > 1:
            train_dataloader.sampler.set_epoch(epoch)
        train_logs = train_step(
            rank, world_size, 
            args, epoch, logger, swanlab_run,
            model, optimizer, scheduler, loss_fn, train_dataloader
        )

        valid_logs = valid_step(
            rank, world_size, 
            args, epoch, logger, swanlab_run,
            model, loss_fn, valid_dataloader
        )
        
        # 检查是否为最佳模型
        valid_score = valid_logs.get('valid_acc', -float('inf'))
        if valid_score > best_valid_score:
            best_valid_score = valid_score
            best_epoch = epoch + 1
            # 保存最佳模型状态
            if world_size > 1:
                best_model_state = model.module.state_dict()
            else:
                best_model_state = model.state_dict()
            logger.info(f'新的最佳模型！Epoch {epoch+1}, 验证准确率: {valid_score:.4f}')
        
        # 定期保存checkpoint（每save_interval个epoch保存一次）
        if (epoch + 1) % args.save_interval == 0 and rank == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}.pth')
            torch.save({
                'config': model.module.cfg.__dict__ if world_size > 1 else model.cfg.__dict__,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
                'epoch': epoch + 1,
                'best_valid_score': best_valid_score,
                'best_epoch': best_epoch,
                'train_logs': train_logs,
                'valid_logs': valid_logs
            }, checkpoint_path)
            
            # 保存训练信息
            info_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}_info.json')
            with open(info_path, 'w') as f:
                json.dump({
                    'epoch': epoch + 1,
                    'best_valid_score': best_valid_score,
                    'best_epoch': best_epoch,
                    'train_logs': train_logs,
                    'valid_logs': valid_logs,
                    'total_epochs': args.n_epochs
                }, f, indent=4)
            
            saved_checkpoints.append(checkpoint_path)
            logger.info(f'Checkpoint saved at epoch {epoch+1}: {checkpoint_path}')
            
            # 清理旧的checkpoint文件，只保留最近的save_last_n个
            if len(saved_checkpoints) > args.save_last_n:
                old_checkpoint = saved_checkpoints.pop(0)
                old_info = old_checkpoint.replace('.pth', '_info.json')
                try:
                    os.remove(old_checkpoint)
                    if os.path.exists(old_info):
                        os.remove(old_info)
                    logger.info(f'Removed old checkpoint: {old_checkpoint}')
                except Exception as e:
                    logger.warning(f'Failed to remove old checkpoint {old_checkpoint}: {e}')
        
        # 分布式同步，确保所有进程等待checkpoint保存完成
        dist.barrier()

    # 保存最佳模型
    if best_model_state is not None and rank == 0:
        # 使用数据盘路径
        best_checkpoint_path = os.path.join('/root/autodl-tmp', f'{project_name}_best_model.pth')
        torch.save({
            'config': model.module.cfg.__dict__ if world_size > 1 else model.cfg.__dict__,
            'model': best_model_state,
            'best_epoch': best_epoch,
            'best_valid_score': best_valid_score
        }, best_checkpoint_path)
        
        # 保存最佳模型信息到数据盘
        best_info_path = os.path.join('/root/autodl-tmp', f'{project_name}_best_model_info.json')
        with open(best_info_path, 'w') as f:
            json.dump({
                'best_epoch': best_epoch,
                'best_valid_score': best_valid_score,
                'total_epochs': args.n_epochs
            }, f, indent=4)
        
        logger.info(f'最佳模型已保存！Epoch {best_epoch}, 验证准确率: {best_valid_score:.4f}')
        logger.info(f'最佳模型路径: {best_checkpoint_path}')
    
    # 分布式同步，等待主进程保存最佳模型
    dist.barrier()

    cleanup()


if __name__ == '__main__':
    args = parse_args()
    
    if args.world_size == -1:
        args.world_size = torch.cuda.device_count()
        
    # 不在这里初始化SwanLab，避免pickle问题
    mp.spawn(
        train, args=(args.world_size, args, None), 
        nprocs=args.world_size, join=True
    )