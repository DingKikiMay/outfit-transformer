#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为test_data生成推荐embedding
适用于CPU环境，不使用分布式训练
"""

import json
import logging
import os
# 设置环境变量解决OpenMP冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pathlib
import pickle
import sys
from argparse import ArgumentParser
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# 添加项目根目录到Python路径
project_root = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ..data import datatypes
from ..models.load import load_model

SRC_DIR = pathlib.Path(__file__).parent.parent.parent.absolute()
LOGS_DIR = SRC_DIR / 'logs'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.makedirs(LOGS_DIR, exist_ok=True)

# 定义嵌入保存路径
TEST_DATA_PRECOMPUTED_REC_EMBEDDING_DIR = "{test_data_dir}/precomputed_rec_embeddings"

def parse_args():
    parser = ArgumentParser(description="为test_data生成推荐embedding")
    parser.add_argument('--model_type', type=str, choices=['original', 'clip'],
                        default='clip', help="模型类型")
    parser.add_argument('--test_data_dir', type=str, 
                        default='./src/test/test_data', help="测试数据目录")
    parser.add_argument('--checkpoint', type=str, 
                        default=None, help="模型检查点路径")
    parser.add_argument('--batch_size', type=int,
                        default=32, help="批处理大小")
    parser.add_argument('--demo', action='store_true', 
                        help="演示模式，只处理少量数据")
    
    return parser.parse_args()


def load_test_data(test_data_dir):
    """加载test_data中的商品数据"""
    result_file = os.path.join(test_data_dir, 'test.json')
    images_dir = os.path.join(test_data_dir, 'images')
    
    if not os.path.exists(result_file):
        raise FileNotFoundError(f"找不到文件: {result_file}")
    
    with open(result_file, 'r', encoding='utf-8') as f:
        products = json.load(f)
    
    # 转换为FashionItem格式
    fashion_items = []
    for i, product in enumerate(products):
        try:
            # 获取item_id，如果JSON中有则使用，否则使用索引+2（从2开始）
            item_id = str(product.get('item_id', i + 2))
            
            # 构建图片路径，使用item_id
            image_path = os.path.join(images_dir, f"{item_id}.jpg")
            if os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
                
                # 处理场景字段，确保是列表格式
                scene_value = product.get('场景', '通用')
                if isinstance(scene_value, str):
                    scene_list = [scene_value] if scene_value else ['通用']
                else:
                    scene_list = scene_value if scene_value else ['通用']
                
                # 创建FashionItem
                fashion_item = datatypes.FashionItem(
                    item_id=int(item_id) if item_id.isdigit() else None,
                    category=product.get('主类别', '未知'),
                    scene=scene_list,
                    image=image,
                    description=product.get('名称', ''),
                    metadata={
                        'item_id': item_id,
                        'semantic_category': product.get('主类别', '未知'),
                        'scene': product.get('场景', '通用'),
                        'url_name': product.get('名称', ''),
                        'title': product.get('名称', ''),
                        'price': product.get('发售价格'),
                        'brand': product.get('品牌', ''),
                        'sub_category': product.get('子类别', ''),
                        'style': product.get('风格', ''),
                        'pattern': product.get('图案', ''),
                        'season': product.get('适用季节', ''),
                        'fit': product.get('版型', '')
                    },
                    embedding=None
                )
                fashion_items.append(fashion_item)
        except Exception as e:
            print(f"加载商品 {i} 失败: {e}")
            continue
    
    print(f"成功加载 {len(fashion_items)} 件商品")
    return fashion_items


def create_batches(items, batch_size):
    """将商品列表分成批次"""
    batches = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batches.append(batch)
    return batches


def main(args):
    """主函数"""
    print("=" * 60)
    print("为test_data生成推荐embedding")
    print("=" * 60)
    print(f"模型类型: {args.model_type}")
    print(f"测试数据目录: {args.test_data_dir}")
    print(f"模型检查点: {args.checkpoint}")
    print(f"批处理大小: {args.batch_size}")
    print(f"演示模式: {args.demo}")
    print("=" * 60)
    
    # 检查test_data目录
    if not os.path.exists(args.test_data_dir):
        print(f"错误：测试数据目录不存在: {args.test_data_dir}")
        return
    
    # 加载模型
    print("正在加载模型...")
    model = load_model(model_type=args.model_type, checkpoint=args.checkpoint)
    model.eval()
    print("模型加载完成")
    
    # 加载test_data
    print("正在加载测试数据...")
    test_items = load_test_data(args.test_data_dir)
    
    if args.demo:
        # 演示模式，只处理前100个商品
        test_items = test_items[:100]
        print(f"演示模式：只处理 {len(test_items)} 件商品")
    
    # 创建批次
    batches = create_batches(test_items, args.batch_size)
    print(f"创建了 {len(batches)} 个批次")
    
    # 生成embedding
    print("正在生成embedding...")
    all_ids, all_embeddings = [], []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(batches, desc="生成embedding")):
            try:
                # 计算embedding
                embeddings = model.embed_item(batch, use_precomputed_embedding=False)
                
                # 收集结果
                batch_ids = [item.item_id for item in batch]
                batch_embeddings = embeddings.detach().cpu().numpy()
                
                all_ids.extend(batch_ids)
                all_embeddings.append(batch_embeddings)
                
            except Exception as e:
                print(f"处理批次 {batch_idx} 失败: {e}")
                continue
    
    if not all_embeddings:
        print("错误：没有成功生成任何embedding")
        return
    
    # 合并所有embedding
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"成功生成 {len(all_embeddings)} 个embedding")
    print(f"Embedding维度: {all_embeddings.shape}")
    
    # 保存embedding
    save_dir = TEST_DATA_PRECOMPUTED_REC_EMBEDDING_DIR.format(test_data_dir=args.test_data_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, "test_data_embeddings.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump({
            'ids': all_ids, 
            'embeddings': all_embeddings
        }, f)
    
    print(f"Embedding已保存到: {save_path}")
    
    # 创建embedding字典
    embedding_dict = {item_id: embedding for item_id, embedding in zip(all_ids, all_embeddings)}
    
    # 保存embedding字典
    dict_path = os.path.join(save_dir, "test_data_embedding_dict.pkl")
    with open(dict_path, 'wb') as f:
        pickle.dump(embedding_dict, f)
    
    print(f"Embedding字典已保存到: {dict_path}")
    print("=" * 60)
    print("Embedding生成完成！")
    print("=" * 60)


if __name__ == "__main__":
    args = parse_args()
    
    # 如果没有指定checkpoint，使用默认路径
    if args.checkpoint is None:
        args.checkpoint = "./beat_path/complementary_clip_cir_experiment_001_best_model.pth"
        print(f"使用默认模型路径: {args.checkpoint}")
        print("请确保模型文件存在，或使用 --checkpoint 参数指定正确的路径")
    
    try:
        main(args)
    except Exception as e:
        print(f"程序执行失败: {e}")
        print("\n可能的解决方案：")
        print("1. 检查模型文件路径是否正确")
        print("2. 检查测试数据目录结构")
        print("3. 确保已安装所有依赖包")
        print("4. 检查是否有足够的内存") 