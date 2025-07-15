'''import json
import logging
import os
import pathlib
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
from ..data.datasets import polyvore_utils as polyvore
from ..evaluation.metrics import compute_cp_scores
from ..models.load import load_model
from ..utils.distributed_utils import cleanup, gather_results, setup
from ..utils.logger import get_logger
from ..utils.loss import FocalLoss
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


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model_type', type=str, choices=['original', 'clip'],
                        default='clip')
    parser.add_argument('--polyvore_dir', type=str, 
                        default='./src/data/datasets/polyvore')
    parser.add_argument('--polyvore_type', type=str, choices=['nondisjoint', 'disjoint'],
                        default='nondisjoint')
    parser.add_argument('--batch_sz_per_gpu', type=int,
                        default=512)
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
    # 加载预计算的嵌入字典
    embedding_dict = polyvore.load_embedding_dict(args.polyvore_dir)
    # 创建训练和验证数据集
    train = polyvore.PolyvoreCompatibilityDataset(
        dataset_dir=args.polyvore_dir, dataset_type=args.polyvore_type, 
        dataset_split='train', metadata=metadata, load_image=False, embedding_dict=embedding_dict
    )
    valid = polyvore.PolyvoreCompatibilityDataset(
        dataset_dir=args.polyvore_dir, dataset_type=args.polyvore_type, 
        dataset_split='valid', metadata=metadata, load_image=False, embedding_dict=embedding_dict
    )
    # 单GPU训练
    if world_size == 1:
        train_dataloader = DataLoader(
            dataset=train, batch_size=args.batch_sz_per_gpu, shuffle=True,
            num_workers=args.n_workers_per_gpu, collate_fn=collate_fn.cp_collate_fn
        )
        valid_dataloader = DataLoader(
            dataset=valid, batch_size=args.batch_sz_per_gpu, shuffle=False,
            num_workers=args.n_workers_per_gpu, collate_fn=collate_fn.cp_collate_fn
        )
    # 分布式训练
    else:
        train_sampler = DistributedSampler(
            train, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
        valid_sampler = DistributedSampler(
            valid, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True
        )
        train_dataloader = DataLoader(
            dataset=train, batch_size=args.batch_sz_per_gpu, shuffle=False,
            num_workers=args.n_workers_per_gpu, collate_fn=collate_fn.cp_collate_fn, sampler=train_sampler
        )
        valid_dataloader = DataLoader(
            dataset=valid, batch_size=args.batch_sz_per_gpu, shuffle=False,
            num_workers=args.n_workers_per_gpu, collate_fn=collate_fn.cp_collate_fn, sampler=valid_sampler
        )
    # 返回训练和验证数据加载器      
    return train_dataloader, valid_dataloader


def train_step(
    rank, world_size, 
    args, epoch, logger, wandb_run,
    model, optimizer, scheduler, loss_fn, dataloader
):
    # 1. 设置模型为训练模式（启用dropout/bn等）
    model.train()  
    pbar = tqdm(dataloader, desc=f'Train Epoch {epoch+1}/{args.n_epochs}', disable=(rank != 0))
    # 2. tqdm进度条，只有主进程(rank==0)显示
    # 3. 初始化损失、预测和标签
    all_loss, all_preds, all_labels = torch.zeros(1, device=rank), [], []
    # 4. 遍历每个batch
    for i, data in enumerate(pbar):
        # 5. 如果demo模式，只训练前3个batch
        if args.demo and i > 2:
            break
        # 6. 获取query和label
        queries = data['query']
        labels = torch.tensor(data['label'], dtype=torch.float32).to(rank)
        # 7. 前向推理，得到预测分数（兼容性分数），并去掉多余维度
        preds = model(queries, use_precomputed_embedding=True).squeeze(1)
        # 8. 计算损失
        loss = loss_fn(y_true=labels, y_prob=preds) / args.accumulation_steps
        # 9. 反向传播
        loss.backward()
        # 10. 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # 11. 如果累积步数达到args.accumulation_steps，则更新模型参数
        if (i + 1) % args.accumulation_steps == 0:
            # 12. 更新模型参数
            optimizer.step()
            # 13. 清空梯度
            optimizer.zero_grad()
            # 14. 更新学习率
            scheduler.step()
        # 15. 累积结果
        # Accumulate Results
        all_loss += loss.item() * args.accumulation_steps / len(dataloader)
        # 16. 保存本批次预测和标签
        all_preds.append(preds.detach())
        all_labels.append(labels.detach())
        # 17. 计算本批次得分    
        # Logging 
        score = compute_cp_scores(all_preds[-1], all_labels[-1])
        logs = {
            'loss': loss.item() * args.accumulation_steps,
            'steps': len(pbar) * epoch + i,
            'lr': scheduler.get_last_lr()[0] if scheduler else args.lr,
            **score
        }
        # 18. 更新进度条
        pbar.set_postfix(**logs)
        # 19. 如果wandb_key存在，则记录日志
        if args.wandb_key and rank == 0:
            logs = {f'train_{k}': v for k, v in logs.items()}
            wandb_run.log(logs)
    
    # 20. 将所有预测和标签拼接起来
    all_preds = torch.cat(all_preds).to(rank)
    all_labels = torch.cat(all_labels).to(rank)
    # 21. 收集所有进程的结果
    gathered_loss, gathered_preds, gathered_labels = gather_results(all_loss, all_preds, all_labels)
    # 22. 计算所有进程的平均损失和得分
    output = {'loss': gathered_loss.item(), **compute_cp_scores(gathered_preds, gathered_labels)} if rank == 0 else {}
    logger.info(f'Epoch {epoch+1}/{args.n_epochs} --> End {output}')

    return {f'train_{key}': value for key, value in output.items()}
   
        
@torch.no_grad()
def valid_step(
    rank, world_size, 
    args, epoch, logger, wandb_run,
    model, loss_fn, dataloader
):
    model.eval()
    pbar = tqdm(dataloader, desc=f'Valid Epoch {epoch+1}/{args.n_epochs}', disable=(rank != 0))
    
    all_loss, all_preds, all_labels = torch.zeros(1, device=rank), [], []
    for i, data in enumerate(pbar):
        if args.demo and i > 2:
            break
        queries = data['query']
        labels = torch.tensor(data['label'], dtype=torch.float32).to(rank)
    
        preds = model(queries, use_precomputed_embedding=True).squeeze(1)
        
        loss = loss_fn(y_true=labels, y_prob=preds) / args.accumulation_steps
        
        # Accumulate Results
        all_loss += loss.item() * args.accumulation_steps / len(dataloader)
        all_preds.append(preds.detach())
        all_labels.append(labels.detach())

        # Logging
        score = compute_cp_scores(all_preds[-1], all_labels[-1])
        logs = {
            'loss': loss.item() * args.accumulation_steps,
            'steps': len(pbar) * epoch + i,
            **score
        }
        pbar.set_postfix(**logs)
        if args.wandb_key and rank == 0:
            logs = {f'valid_{k}': v for k, v in logs.items()}
            wandb_run.log(logs)
        
    
    all_preds = torch.cat(all_preds).to(rank)
    all_labels = torch.cat(all_labels).to(rank)

    gathered_loss, gathered_preds, gathered_labels = gather_results(all_loss, all_preds, all_labels)
    output = {}
    if rank == 0:
        all_score = compute_cp_scores(gathered_preds, gathered_labels)
        output = {'loss': gathered_loss.item(), **all_score}
        
    logger.info(f'Epoch {epoch+1}/{args.n_epochs} --> End {output}')

    return {f'valid_{key}': value for key, value in output.items()}


def train(
    rank: int, world_size: int, args: Any,
    wandb_run: Optional[wandb.sdk.wandb_run.Run] = None
):  
    # Setup
    setup(rank, world_size)
    
    # Logging Setup
    project_name = f'compatibility_{args.model_type}_' + (
        args.project_name if args.project_name 
        else (wandb_run.name if wandb_run else 'test')
    )
    logger = get_logger(project_name, LOGS_DIR, rank)
    logger.info(f'Logger Setup Completed')
    
    # Dataloaders
    # 调用setup_dataloaders，构建训练和验证集的DataLoader，支持分布式采样。日志记录数据加载完成。
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
    loss_fn = FocalLoss(alpha=0.5, gamma=2) # focal_loss(alpha=0.5, gamma=2)
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
        
        # 构建模型保存目录和文件名
        checkpoint_dir = CHECKPOINT_DIR / project_name
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}.pth')
        # 只在主进程(rank==0)保存模型权重和配置到epoch_x.pth 
        if rank == 0:
            torch.save({
                'config': model.module.cfg.__dict__ if world_size > 1 else model.cfg.__dict__,
                'model': model.state_dict()
            }, checkpoint_path)
            # 保存得分到epoch_x_score.json
            score_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}_score.json')
            with open(score_path, 'w') as f:
                score = {**train_logs, **valid_logs}
                json.dump(score, f, indent=4)
            logger.info(f'Checkpoint saved at {checkpoint_path}')
        # 分布式同步，所有进程等待主进程保存好模型     
        dist.barrier()
        # 加载模型权重
        map_location = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
        state_dict = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        # 加载模型权重
        model.load_state_dict(state_dict['model'])
        # 日志记录模型加载完成
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
        wandb_run = wandb.init(project='outfit-transformer-cp', config=args.__dict__)
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

import swanlab

from ..data import collate_fn
from ..data.datasets import polyvore_utils as polyvore
from ..evaluation.metrics import compute_cp_scores
from ..models.load import load_model
from ..utils.distributed_utils import cleanup, gather_results, setup
from ..utils.logger import get_logger
from ..utils.loss import FocalLoss
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


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model_type', type=str, choices=['original', 'clip'],
                        default='clip')
    parser.add_argument('--polyvore_dir', type=str, 
                        default='./src/data/datasets/polyvore')
    parser.add_argument('--polyvore_type', type=str, choices=['nondisjoint', 'disjoint'],
                        default='nondisjoint')
    parser.add_argument('--batch_sz_per_gpu', type=int,
                        default=512)
    parser.add_argument('--n_workers_per_gpu', type=int,
                        default=4)
    parser.add_argument('--n_epochs', type=int,
                        default=200)
    parser.add_argument('--lr', type=float,
                        default=2e-5)
    parser.add_argument('--accumulation_steps', type=int,
                        default=4)
    parser.add_argument('--swanlab_project', type=str, 
                        default='outfit-transformer-cp')
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
    # 定期保存checkpoints
    parser.add_argument('--save_interval', type=int, default=20,
                        help='Save checkpoint every N epochs (default: 20)')
    parser.add_argument('--save_last_n', type=int, default=3,
                        help='Keep only the last N checkpoints to save disk space (default: 3)')
    
    return parser.parse_args()


def setup_dataloaders(rank, world_size, args):
    metadata = polyvore.load_metadata(args.polyvore_dir)
    # 加载预计算的嵌入字典
    embedding_dict = polyvore.load_embedding_dict(args.polyvore_dir)
    # 创建训练和验证数据集
    train = polyvore.PolyvoreCompatibilityDataset(
        dataset_dir=args.polyvore_dir, dataset_type=args.polyvore_type, 
        dataset_split='train', metadata=metadata, load_image=False, embedding_dict=embedding_dict
    )
    valid = polyvore.PolyvoreCompatibilityDataset(
        dataset_dir=args.polyvore_dir, dataset_type=args.polyvore_type, 
        dataset_split='valid', metadata=metadata, load_image=False, embedding_dict=embedding_dict
    )
    # 单GPU训练
    if world_size == 1:
        train_dataloader = DataLoader(
            dataset=train, batch_size=args.batch_sz_per_gpu, shuffle=True,
            num_workers=args.n_workers_per_gpu, collate_fn=collate_fn.cp_collate_fn
        )
        valid_dataloader = DataLoader(
            dataset=valid, batch_size=args.batch_sz_per_gpu, shuffle=False,
            num_workers=args.n_workers_per_gpu, collate_fn=collate_fn.cp_collate_fn
        )
    # 分布式训练
    else:
        train_sampler = DistributedSampler(
            train, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
        valid_sampler = DistributedSampler(
            valid, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True
        )
        train_dataloader = DataLoader(
            dataset=train, batch_size=args.batch_sz_per_gpu, shuffle=False,
            num_workers=args.n_workers_per_gpu, collate_fn=collate_fn.cp_collate_fn, sampler=train_sampler
        )
        valid_dataloader = DataLoader(
            dataset=valid, batch_size=args.batch_sz_per_gpu, shuffle=False,
            num_workers=args.n_workers_per_gpu, collate_fn=collate_fn.cp_collate_fn, sampler=valid_sampler
        )
    # 返回训练和验证数据加载器      
    return train_dataloader, valid_dataloader


def train_step(
    rank, world_size, 
    args, epoch, logger, swanlab_run,
    model, optimizer, scheduler, loss_fn, dataloader
):
    # 1. 设置模型为训练模式（启用dropout/bn等）
    model.train()  
    pbar = tqdm(dataloader, desc=f'Train Epoch {epoch+1}/{args.n_epochs}', disable=(rank != 0))
    # 2. tqdm进度条，只有主进程(rank==0)显示
    # 3. 初始化损失、预测和标签
    all_loss, all_preds, all_labels = torch.zeros(1, device=rank), [], []
    # 4. 遍历每个batch
    for i, data in enumerate(pbar):
        # 5. 如果demo模式，只训练前3个batch
        if args.demo and i > 2:
            break
        # 6. 获取query和label
        queries = data['query']
        labels = torch.tensor(data['label'], dtype=torch.float32).to(rank)
        # 7. 前向推理，得到预测分数（兼容性分数），并去掉多余维度
        preds = model(queries, use_precomputed_embedding=True).squeeze(1)
        # 8. 计算损失
        loss = loss_fn(y_true=labels, y_prob=preds) / args.accumulation_steps
        # 9. 反向传播
        loss.backward()
        # 10. 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # 11. 如果累积步数达到args.accumulation_steps，则更新模型参数
        if (i + 1) % args.accumulation_steps == 0:
            # 12. 更新模型参数
            optimizer.step()
            # 13. 清空梯度
            optimizer.zero_grad()
            # 14. 更新学习率
            scheduler.step()
        # 15. 累积结果
        # Accumulate Results
        all_loss += loss.item() * args.accumulation_steps / len(dataloader)
        # 16. 保存本批次预测和标签
        all_preds.append(preds.detach())
        all_labels.append(labels.detach())
        # 17. 计算累积得分    
        # Logging - 显示累积的准确率而不是当前batch的准确率
        if len(all_preds) > 0:
            # 计算累积的准确率
            accumulated_preds = torch.cat(all_preds)
            accumulated_labels = torch.cat(all_labels)
            accumulated_score = compute_cp_scores(accumulated_preds, accumulated_labels)
        else:
            accumulated_score = compute_cp_scores(all_preds[-1], all_labels[-1])
        
        logs = {
            'loss': loss.item() * args.accumulation_steps,
            'steps': len(pbar) * epoch + i,
            'lr': scheduler.get_last_lr()[0] if scheduler else args.lr,
            **accumulated_score
        }
        # 18. 更新进度条
        pbar.set_postfix(**logs)
        # 19. 如果swanlab_project存在，则记录日志
        if swanlab_run:
            logs = {f'train_{k}': v for k, v in logs.items()}
            swanlab_run.log(logs)
    
    # 20. 将所有预测和标签拼接起来
    all_preds = torch.cat(all_preds).to(rank)
    all_labels = torch.cat(all_labels).to(rank)
    # 21. 收集所有进程的结果
    gathered_loss, gathered_preds, gathered_labels = gather_results(all_loss, all_preds, all_labels)
    # 22. 计算所有进程的平均损失和得分
    output = {'loss': gathered_loss.item(), **compute_cp_scores(gathered_preds, gathered_labels)} if rank == 0 else {}
    logger.info(f'Epoch {epoch+1}/{args.n_epochs} --> End {output}')

    return {f'train_{key}': value for key, value in output.items()}
   
        
@torch.no_grad()
def valid_step(
    rank, world_size, 
    args, epoch, logger, swanlab_run,
    model, loss_fn, dataloader
):
    model.eval()
    pbar = tqdm(dataloader, desc=f'Valid Epoch {epoch+1}/{args.n_epochs}', disable=(rank != 0))
    
    all_loss, all_preds, all_labels = torch.zeros(1, device=rank), [], []
    for i, data in enumerate(pbar):
        if args.demo and i > 2:
            break
        queries = data['query']
        labels = torch.tensor(data['label'], dtype=torch.float32).to(rank)
    
        preds = model(queries, use_precomputed_embedding=True).squeeze(1)
        
        loss = loss_fn(y_true=labels, y_prob=preds) / args.accumulation_steps
        
        # Accumulate Results
        all_loss += loss.item() * args.accumulation_steps / len(dataloader)
        all_preds.append(preds.detach())
        all_labels.append(labels.detach())

        # Logging - 显示累积的准确率而不是当前batch的准确率
        if len(all_preds) > 0:
            # 计算累积的准确率
            accumulated_preds = torch.cat(all_preds)
            accumulated_labels = torch.cat(all_labels)
            accumulated_score = compute_cp_scores(accumulated_preds, accumulated_labels)
        else:
            accumulated_score = compute_cp_scores(all_preds[-1], all_labels[-1])
        
        logs = {
            'loss': loss.item() * args.accumulation_steps,
            'steps': len(pbar) * epoch + i,
            **accumulated_score
        }
        
    
    all_preds = torch.cat(all_preds).to(rank)
    all_labels = torch.cat(all_labels).to(rank)

    gathered_loss, gathered_preds, gathered_labels = gather_results(all_loss, all_preds, all_labels)
    output = {}
    if rank == 0:
        all_score = compute_cp_scores(gathered_preds, gathered_labels)
        output = {'loss': gathered_loss.item(), **all_score}
        
    logger.info(f'Epoch {epoch+1}/{args.n_epochs} --> End {output}')

    return {f'valid_{key}': value for key, value in output.items()}


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
    project_name = f'compatibility_{args.model_type}_' + (
        args.project_name if args.project_name 
        else (args.swanlab_experiment_name if args.swanlab_experiment_name else 'test')
    )
    logger = get_logger(project_name, LOGS_DIR, rank)
    logger.info(f'Logger Setup Completed')
    
    # Dataloaders
    # 调用setup_dataloaders，构建训练和验证集的DataLoader，支持分布式采样。日志记录数据加载完成。
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
    # 训练召回率太低，修改参数
    loss_fn = FocalLoss(alpha=0.7, gamma=1.5) # focal_loss(alpha=0.5, gamma=2)
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
        
        # 检查是否为最佳模型 修改！！！！！！！！！！！！！！！！！！！！！！
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
        
        # 极简存储策略：只在训练结束时保存最佳模型，不保存中间checkpoint
        # 这样可以大幅减少磁盘占用，避免系统盘满
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
        best_info_path = os.path.join('/root/autodl-tmp', f'{project_name}.json')
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
