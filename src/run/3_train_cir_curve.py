import json
import logging
import os
import pathlib
import random
from argparse import ArgumentParser
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

import wandb

from ..data import collate_fn
from ..data.datasets import polyvore_utils as polyvore
from ..evaluation.metrics import compute_cir_scores
from ..models.load import load_model
from ..utils.distributed_utils import cleanup, gather_results, setup
from ..utils.logger import get_logger
from ..utils.loss import InBatchTripletMarginLoss

SRC_DIR = pathlib.Path(__file__).parent.parent.parent.absolute()
CHECKPOINT_DIR = SRC_DIR / 'checkpoints'
RESULT_DIR = SRC_DIR / 'results'
LOGS_DIR = SRC_DIR / 'logs'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model_type', type=str, choices=['original', 'clip'], default='clip')
    parser.add_argument('--polyvore_dir', type=str, default='./src/data/datasets/polyvore')
    parser.add_argument('--polyvore_type', type=str, choices=['nondisjoint', 'disjoint'], default='nondisjoint')
    parser.add_argument('--batch_sz_per_gpu', type=int, default=64)
    parser.add_argument('--n_workers_per_gpu', type=int, default=4)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--accumulation_steps', type=int, default=4)
    parser.add_argument('--wandb_key', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--world_size', type=int, default=-1)
    parser.add_argument('--project_name', type=str, default=None)
    parser.add_argument('--demo', action='store_true')
    return parser.parse_args()

def setup_dataloaders(rank, world_size, args, data_limit=None):
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
    if data_limit is not None:
        train.data = train.data[:data_limit]
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
    pbar = tqdm(dataloader, desc=f'Train Epoch {epoch+1}/{args.n_epochs}', disable=(rank != 0))
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
            
        batched_q_emb = model(filtered_query, use_precomputed_embedding=True)
        batched_a_emb = model(filtered_answer, use_precomputed_embedding=True)
        loss = loss_fn(batched_q_emb, batched_a_emb)
        loss = loss / args.accumulation_steps
        loss.backward()
        if (i + 1) % args.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        dists = torch.cdist(batched_q_emb, batched_a_emb, p=2)
        preds = torch.argmin(dists, dim=1)
        labels = torch.arange(len(preds), device=rank)
        all_loss += loss.item() * args.accumulation_steps / len(dataloader)
        all_preds.append(preds.detach())
        all_labels.append(labels.detach())
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
    all_preds, all_labels = [], []
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
            
        batched_q_emb = model(filtered_query, use_precomputed_embedding=True).unsqueeze(1)
        batched_c_embs = model(sum(filtered_candidates, []), use_precomputed_embedding=True)
        batched_c_embs = batched_c_embs.view(-1, 4, batched_c_embs.shape[1])
        dists = torch.norm(batched_q_emb - batched_c_embs, dim=-1)
        preds = torch.argmin(dists, dim=-1)
        labels = torch.tensor(filtered_labels, device=rank)
        all_preds.append(preds.detach())
        all_labels.append(labels.detach())
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
    _, gathered_preds, gathered_labels = gather_results(torch.zeros(1, device=rank), all_preds, all_labels)
    output = {**compute_cir_scores(gathered_preds, gathered_labels)} if rank == 0 else {}
    logger.info(f'Epoch {epoch+1}/{args.n_epochs} --> End {output}')
    return {f'valid_{key}': value for key, value in output.items()}

def train(
    rank: int, world_size: int, args: Any,
    wandb_run: Optional[wandb.sdk.wandb_run.Run] = None
):  
    setup(rank, world_size)
    project_name = f'cir_curve_{args.model_type}_' + (
        args.project_name if args.project_name 
        else (wandb_run.name if wandb_run else 'test')
    )
    logger = get_logger(project_name, LOGS_DIR, rank)
    logger.info(f'Logger Setup Completed')
    # 加载元数据和嵌入字典
    metadata = polyvore.load_metadata(args.polyvore_dir)
    embedding_dict = polyvore.load_embedding_dict(args.polyvore_dir)
    full_train = polyvore.PolyvoreTripletDataset(
        dataset_dir=args.polyvore_dir, dataset_type=args.polyvore_type,
        dataset_split='train', metadata=metadata, embedding_dict=embedding_dict
    )
    # 计算总数据量
    total_len = len(full_train)
    data_points = [500, 1000, 2000, 3000, 4000, 5000]
    cur = 6000
    # 计算数据点
    while cur <= total_len:
        data_points.append(cur)
        cur += 1000
    n_epochs = args.n_epochs
    # 每个数据量下训练n_epochs个epoch
    for used_data in data_points:
        logger.info(f'当前训练数据量: {used_data}')
        train_dataloader, valid_dataloader = setup_dataloaders(rank, world_size, args, data_limit=used_data)
        logger.info(f'Dataloaders Setup Completed')
        model = load_model(model_type=args.model_type, checkpoint=None)
        logger.info(f'Model Loaded and Wrapped with DDP')
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr, epochs=n_epochs, steps_per_epoch=max(1, used_data // args.batch_sz_per_gpu),
            pct_start=0.3, anneal_strategy='cos', div_factor=25, final_div_factor=1e4
        )
        loss_fn = InBatchTripletMarginLoss(margin=2.0, reduction='mean')
        best_valid_score = -float('inf')
        best_valid_logs = None
        for epoch in range(n_epochs):
            # 分布式训练
            if world_size > 1:
                from torch.utils.data import DistributedSampler
                if hasattr(train_dataloader, 'sampler') and isinstance(train_dataloader.sampler, DistributedSampler):
                    train_dataloader.sampler.set_epoch(epoch)
            # 训练一个epoch
            train_logs = train_step(
                rank, world_size, 
                args, epoch, logger, wandb_run,
                model, optimizer, scheduler, loss_fn, train_dataloader
            )
            # 验证一个epoch
            valid_logs = valid_step(
                rank, world_size, 
                args, epoch, logger, wandb_run,
                model, loss_fn, valid_dataloader
            )
            valid_score = valid_logs.get('valid_cir_accuracy', None)
            if valid_score is not None and valid_score > best_valid_score:
                best_valid_score = valid_score
                best_valid_logs = valid_logs
        # 保存最佳验证得分和日志
        if rank == 0 and wandb_run is not None:
            wandb_run.log({'used_data': used_data, 'best_valid_score': best_valid_score, **(best_valid_logs or {})})
        # 构建模型保存目录和文件名
        checkpoint_dir = CHECKPOINT_DIR / project_name
        os.makedirs(checkpoint_dir, exist_ok=True)
        # 保存模型权重和配置
        checkpoint_path = os.path.join(checkpoint_dir, f'data{used_data}_best.pth')
        if rank == 0:
            torch.save({
                'config': model.module.cfg.__dict__ if world_size > 1 else model.cfg.__dict__,
                'model': model.state_dict()
            }, checkpoint_path)
            score_path = os.path.join(checkpoint_dir, f'data{used_data}_best_score.json')
            with open(score_path, 'w') as f:
                json.dump({'used_data': used_data, 'best_valid_score': best_valid_score, **(best_valid_logs or {})}, f, indent=4)
            logger.info(f'Checkpoint saved at {checkpoint_path}')
        dist.barrier()
        map_location = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
        state_dict = torch.load(checkpoint_path, map_location=map_location)
        model.load_state_dict(state_dict['model'])
        logger.info(f'Checkpoint loaded from {checkpoint_path}')
    cleanup()

if __name__ == '__main__':
    args = parse_args()
    if args.world_size == -1:
        args.world_size = torch.cuda.device_count()
    if args.wandb_key:
        wandb.login(key=args.wandb_key)
        wandb_run = wandb.init(project='outfit-transformer-cir-curve', config=args.__dict__)
    else:
        wandb_run = None
    mp.spawn(
        train, args=(args.world_size, args, wandb_run), 
        nprocs=args.world_size, join=True
    ) 