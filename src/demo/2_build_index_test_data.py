#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为test_data构建FAISS索引
适用于CPU环境
"""

import os
import pathlib
import pickle
import sys
from argparse import ArgumentParser

# 设置环境变量解决OpenMP冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np

# 添加项目根目录到Python路径
project_root = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 使用绝对导入
from src.demo.vectorstore import FAISSVectorStore

SRC_DIR = pathlib.Path(__file__).parent.parent.parent.absolute()
LOGS_DIR = SRC_DIR / 'logs'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.makedirs(LOGS_DIR, exist_ok=True)

# 定义嵌入保存路径
TEST_DATA_PRECOMPUTED_REC_EMBEDDING_DIR = "{test_data_dir}/precomputed_rec_embeddings"


def parse_args():
    parser = ArgumentParser(description="为test_data构建FAISS索引")
    parser.add_argument('--test_data_dir', type=str, 
                        default='./src/test/test_data', help="测试数据目录")
    parser.add_argument('--d_embed', type=int, default=512, 
                        help="embedding维度")
    parser.add_argument('--faiss_type', type=str, default='IndexFlatIP',
                        help="FAISS索引类型")
    
    return parser.parse_args()


def load_rec_embedding_dict(test_data_dir):
    """加载test_data的推荐embedding字典"""
    e_dir = TEST_DATA_PRECOMPUTED_REC_EMBEDDING_DIR.format(test_data_dir=test_data_dir)
    
    # 检查embedding文件是否存在
    embedding_file = os.path.join(e_dir, "test_data_embeddings.pkl")
    dict_file = os.path.join(e_dir, "test_data_embedding_dict.pkl")
    
    if os.path.exists(dict_file):
        # 优先使用字典文件
        print(f"加载embedding字典: {dict_file}")
        with open(dict_file, 'rb') as f:
            embedding_dict = pickle.load(f)
    elif os.path.exists(embedding_file):
        # 使用embeddings文件
        print(f"加载embedding文件: {embedding_file}")
        with open(embedding_file, 'rb') as f:
            data = pickle.load(f)
            all_ids = data['ids']
            all_embeddings = data['embeddings']
            embedding_dict = {item_id: embedding for item_id, embedding in zip(all_ids, all_embeddings)}
    else:
        raise FileNotFoundError(f"找不到embedding文件: {e_dir}")
    
    print(f"加载了 {len(embedding_dict)} 个embedding")
    return embedding_dict


def main(args):
    """主函数"""
    print("=" * 60)
    print("为test_data构建FAISS索引")
    print("=" * 60)
    print(f"测试数据目录: {args.test_data_dir}")
    print(f"Embedding维度: {args.d_embed}")
    print(f"FAISS索引类型: {args.faiss_type}")
    print("=" * 60)
    
    # 检查test_data目录
    if not os.path.exists(args.test_data_dir):
        print(f"错误：测试数据目录不存在: {args.test_data_dir}")
        return
    
    # 加载embedding
    print("正在加载embedding...")
    try:
        embedding_dict = load_rec_embedding_dict(args.test_data_dir)
    except FileNotFoundError as e:
        print(f"错误：{e}")
        print("请先运行 1_generate_rec_embeddings_test_data.py 生成embedding")
        return
    
    if not embedding_dict:
        print("错误：没有找到任何embedding")
        return
    
    # 创建FAISS索引
    print("正在创建FAISS索引...")
    indexer = FAISSVectorStore(
        index_name='test_data_rec_index',
        d_embed=args.d_embed,
        faiss_type=args.faiss_type,
        base_dir=TEST_DATA_PRECOMPUTED_REC_EMBEDDING_DIR.format(test_data_dir=args.test_data_dir),
    )
    
    # 准备数据
    embeddings = list(embedding_dict.values())
    ids = list(embedding_dict.keys())
    
    print(f"添加 {len(embeddings)} 个embedding到索引...")
    
    # 添加到索引
    try:
        indexer.add(embeddings=embeddings, ids=ids)
        print("成功添加到索引")
    except Exception as e:
        print(f"添加到索引失败: {e}")
        return
    
    # 保存索引
    print("正在保存索引...")
    try:
        indexer.save()
        print("索引保存成功")
    except Exception as e:
        print(f"保存索引失败: {e}")
        return
    
    print("=" * 60)
    print("FAISS索引构建完成！")
    print("=" * 60)
    
    # 测试索引
    print("正在测试索引...")
    try:
        # 随机选择一个embedding进行测试
        test_id = list(embedding_dict.keys())[0]
        test_embedding = embedding_dict[test_id]
        
        # 搜索最相似的5个商品
        results = indexer.search(
            embeddings=[test_embedding], 
            k=min(5, len(embedding_dict))
        )
        
        print(f"测试搜索成功，找到 {len(results[0])} 个结果")
        print(f"测试商品ID: {test_id}")
        print(f"搜索结果: {[r[1] for r in results[0]]}")
        
    except Exception as e:
        print(f"索引测试失败: {e}")


if __name__ == "__main__":
    args = parse_args()
    
    try:
        main(args)
    except Exception as e:
        print(f"程序执行失败: {e}")
        print("\n可能的解决方案：")
        print("1. 确保已运行 1_generate_rec_embeddings_test_data.py")
        print("2. 检查embedding文件是否存在")
        print("3. 确保embedding维度与模型输出一致")
        print("4. 检查是否有足够的内存") 