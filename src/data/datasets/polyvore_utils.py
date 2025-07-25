# -*- coding:utf-8 -*-
"""
Author:
    Wonjun Oh, owj0421@naver.com
"""
from typing import Literal
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from multiprocessing import Pool, cpu_count
import os
import cv2
import json
import random
import pickle
from tqdm import tqdm
from ..datatypes import (
    FashionItem, 
    FashionCompatibilityQuery, 
    FashionComplementaryQuery, 
    FashionCompatibilityData, 
    FashionFillInTheBlankData, 
    FashionTripletData
)
from functools import lru_cache
import numpy as np

POLYVORE_PRECOMPUTED_CLIP_EMBEDDING_DIR = (
    "{dataset_dir}/precomputed_clip_embeddings"
)
POLYVORE_METADATA_PATH = (
    "{dataset_dir}/filtered_item_metadata.json"
)
POLYVORE_SET_DATA_PATH = (
    "{dataset_dir}/{dataset_type}/{dataset_split}_filtered.json"
)
POLYVORE_TASK_DATA_PATH = (
    "{dataset_dir}/{dataset_type}/{dataset_task}/{dataset_split}_filtered.json"
)
POLYVORE_IMAGE_DATA_PATH = (
    "{dataset_dir}/filtered_images/{item_id}.jpg"
)
# 过滤，保留“上装”“下装”
def load_metadata(dataset_dir):
    metadata = {}
    with open(
        POLYVORE_METADATA_PATH.format(dataset_dir=dataset_dir), 'r', encoding='utf-8'
    ) as f:
        metadata_ = json.load(f)
        for item in metadata_:
            metadata[item['item_id']] = item
    print(f"Loaded {len(metadata)} metadata")
    return metadata # 返回一个字典，键是item_id，值是item的元数据`​{item_id: item_metadata}`

# 加载预计算的embedding字典
def load_embedding_dict(dataset_dir):
    e_dir = POLYVORE_PRECOMPUTED_CLIP_EMBEDDING_DIR.format(dataset_dir=dataset_dir)
    filenames = [filename for filename in os.listdir(e_dir) if filename.endswith(".pkl")]
    filenames = sorted(filenames, key=lambda x: int(x.split('.')[0].split('_')[-1]))
    
    all_ids, all_embeddings = [], []
    for filename in filenames:
        filepath = os.path.join(e_dir, filename)
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            all_ids += data['ids']
            all_embeddings.append(data['embeddings'])
            
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"Loaded {len(all_embeddings)} embeddings")
    
    all_embeddings_dict = {item_id: embedding for item_id, embedding in zip(all_ids, all_embeddings)}
    print(f"Created embeddings dictionary")
    
    return all_embeddings_dict

# 根据指定的id读取图片
def _load_image(dataset_dir, item_id, size=(224, 224)):
    image_path = POLYVORE_IMAGE_DATA_PATH.format(
        dataset_dir=dataset_dir,
        item_id=item_id
    )
    try:
        image = Image.open(image_path)
        return image
    
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def load_image_wrapper(args):
    dataset_dir, item_id, size = args
    return item_id, _load_image(dataset_dir, item_id, size)
    
def load_images_parallel(dataset_dir, item_ids, size=(224, 224), num_workers=None):
    if num_workers is None:
        num_workers = min(cpu_count(), 4)  # CPU 核心数的一半
    # 创建一个进程池，用于并行加载图像
    with Pool(num_workers) as pool:
        images = pool.map(load_image_wrapper, [(dataset_dir, item_id, size) for item_id in item_ids])
    # 返回一个列表，列表中的每个元素是一个元组，元组包含item_id和对应的图像
    return images

# def load_image_dict(dataset_dir, metadata, size=(224, 224)):
#     all_image_dict = {}
#     num_workers = min(cpu_count(), 8)  # 최대 8개 프로세스 사용

#     with Pool(num_workers) as p:
#         results = list(tqdm(
#             p.imap(load_image_wrapper, [(dataset_dir, item_id, size) for item_id in metadata.keys()]),
#             total=len(metadata),
#             desc="Loading Images"
#         ))

#     all_image_dict = {item_id: img for item_id, img in results if img is not None}
#     print(f"Loaded {len(all_image_dict)} images")
#     return all_image_dict


def load_item(dataset_dir, metadata, item_id, load_image: bool = False, embedding_dict: dict = None) -> FashionItem:
    metadata_ = metadata[item_id]

    return FashionItem(
        item_id=metadata_['item_id'],
        category=metadata_['semantic_category'],
        # 添加场景
        scene=metadata_['scene'],
        image=_load_image(dataset_dir, item_id) if load_image else None,
        # description: 商品标题或URL名称
        description=metadata_['url_name'] if metadata_['url_name'] else metadata_['title'],
        metadata=metadata_,
        embedding=embedding_dict[item_id] if embedding_dict else None
    )
    
# 加载不同任务的json文件
def load_task_data(dataset_dir, dataset_type, task, dataset_split):
    with open(
        POLYVORE_TASK_DATA_PATH.format(
            dataset_dir=dataset_dir,
            dataset_type=dataset_type,
            dataset_task=task,
            dataset_split=dataset_split
        ), 'r', encoding='utf-8'
    ) as f:
        data = json.load(f)
        
    return data

# 加载不同数据集的json文件
def load_set_data(dataset_dir, dataset_type, dataset_split):
    with open(
        POLYVORE_SET_DATA_PATH.format(
            dataset_dir=dataset_dir,
            dataset_type=dataset_type,
            dataset_split=dataset_split
        ), 'r', encoding='utf-8'
    ) as f:
        data = json.load(f)
        
    return data


class PolyvoreCompatibilityDataset(Dataset):

    def __init__(
        self,
        dataset_dir: str,
        dataset_type: Literal[
            'nondisjoint', 'disjoint'
        ] = 'nondisjoint',
        dataset_split: Literal[
            'train', 'valid', 'test'
        ] = 'train',
        metadata: dict = None,
        embedding_dict: dict = None,
        load_image: bool = False
    ):
        self.dataset_dir = dataset_dir
        self.metadata = metadata if metadata else load_metadata(dataset_dir)
        self.data = load_task_data(
            dataset_dir, dataset_type, 'compatibility', dataset_split
        )
        self.load_image = load_image
        self.embedding_dict = embedding_dict
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> FashionCompatibilityData:
        label = self.data[idx]['label']
        # 存储CP任务要查询的第idx个outfit的单品组成的列表
        outfit = [
            load_item(self.dataset_dir, self.metadata, item_id, 
                      self.load_image, self.embedding_dict) 
            for item_id in self.data[idx]['question']
        ]
        
        return FashionCompatibilityData(
            label=label,
            # 创建一个FashionCompatibilityQuery对象，用于存储兼容性查询数据
            query=FashionCompatibilityQuery(outfit=outfit)
        )
        
class PolyvoreFillInTheBlankDataset(Dataset):

    def __init__(
        self,
        dataset_dir: str,
        dataset_type: Literal[
            'nondisjoint', 'disjoint'
        ] = 'nondisjoint',
        dataset_split: Literal[
            'train', 'valid', 'test'
        ] = 'train',
        metadata: dict = None,
        embedding_dict: dict = None,
        load_image: bool = False
    ):
        self.dataset_dir = dataset_dir
        self.metadata = metadata if metadata else load_metadata(dataset_dir)
        self.data = load_task_data(
            dataset_dir, dataset_type, 'fill_in_the_blank', dataset_split
        )
        self.load_image = load_image
        self.embedding_dict = embedding_dict
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> FashionFillInTheBlankData:
        label = self.data[idx]['label']
        candidates = [
            load_item(self.dataset_dir, self.metadata, item_id, 
                      self.load_image, self.embedding_dict) 
            for item_id in self.data[idx]['answers']
        ]
        outfit = [
            load_item(self.dataset_dir, self.metadata, item_id, 
                      self.load_image, self.embedding_dict)
            for item_id in self.data[idx]['question']
        ]
        
        # 过滤掉无效的items
        valid_candidates = [item for item in candidates if hasattr(item, 'embedding') and item.embedding is not None]
        valid_outfit = [item for item in outfit if hasattr(item, 'embedding') and item.embedding is not None]
        
        # 如果candidates或outfit无效，使用原始数据
        if len(valid_candidates) == 0:
            valid_candidates = candidates
        if len(valid_outfit) == 0:
            valid_outfit = outfit
        
        return FashionFillInTheBlankData(
            query=FashionComplementaryQuery(outfit=valid_outfit, category=valid_candidates[label].category),
            label=label,
            candidates=valid_candidates
        )
    
# 创建一个PolyvoreTripletDataset对象，用于加载和处理Polyvore数据集中的三元组数据
class PolyvoreTripletDataset(Dataset):

    def __init__(
        self,
        dataset_dir: str,
        dataset_type: Literal[
            'nondisjoint', 'disjoint'
        ] = 'nondisjoint',
        dataset_split: Literal[
            'train', 'valid', 'test'
        ] = 'train',
        metadata: dict = None,
        embedding_dict: dict = None,
        load_image: bool = False
    ):
        self.dataset_dir = dataset_dir
        self.metadata = metadata if metadata else load_metadata(dataset_dir)
        self.data = load_set_data(
            dataset_dir, dataset_type, dataset_split
        )
        self.load_image = load_image
        self.embedding_dict = embedding_dict
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> FashionTripletData:
        items = [
            load_item(self.dataset_dir, self.metadata, item_id, 
                      self.load_image, self.embedding_dict)
            for item_id in self.data[idx]['item_ids']
        ]
        
        # 过滤掉无效的items（没有embedding的）
        valid_items = [item for item in items if hasattr(item, 'embedding') and item.embedding is not None]
        
        # 如果有效items少于2个，跳过这个样本
        if len(valid_items) < 2:
            # 返回一个默认的有效样本，或者抛出异常
            # 这里我们选择返回第一个样本，让训练步骤来处理
            if len(valid_items) == 1:
                answer = valid_items[0]
                outfit = [answer]  # 使用自己作为outfit，训练时会过滤掉
            else:
                # 如果没有有效items，使用原始items
                answer = items[0] if items else None
                outfit = items[1:] if len(items) > 1 else [answer]
        else:
            answer = valid_items[random.randint(0, len(valid_items) - 1)]
            outfit = [item for item in valid_items if item != answer]
        
        return FashionTripletData(
            query=FashionComplementaryQuery(outfit=outfit, category=answer.category),
            answer=answer
        )
        
# 创建一个PolyvoreItemDataset对象，用于加载和处理Polyvore数据集中的物品数据 
class PolyvoreItemDataset(Dataset):

    def __init__(
        self,
        dataset_dir: str,
        metadata: dict = None,
        embedding_dict: dict = None,
        load_image: bool = False
    ):
        self.dataset_dir = dataset_dir
        self.metadata = metadata if metadata else load_metadata(dataset_dir)
        self.load_image = load_image
        self.embedding_dict = embedding_dict
        
        self.all_item_ids = list(self.metadata.keys())
        # self.item_id_to_idx = {item_id: idx for idx, item_id in enumerate(self.all_item_ids)}
        
    def __len__(self):
        return len(self.all_item_ids)
    
    def __getitem__(self, idx) -> FashionItem:
        item = load_item(self.dataset_dir, self.metadata, self.all_item_ids[idx], 
                         load_image=self.load_image, embedding_dict=self.embedding_dict)

        return item
    
    def get_item_by_id(self, item_id):
        return load_item(self.dataset_dir, self.metadata, item_id, 
                         load_image=self.load_image, embedding_dict=self.embedding_dict)
        
        
if __name__ == '__main__':
    # Test the dataset
    # 修改了数据路径，从/home/owj0421/datasets/polyvore 到 ./src/data/datasets/polyvore
    dataset_dir = "./src/data/datasets/polyvore"
    
    dataset = PolyvoreCompatibilityDataset(
        dataset_dir,
        dataset_type='nondisjoint',
        dataset_split='train'
    )
    print(len(dataset))
    print(dataset[0])
    
    dataset = PolyvoreFillInTheBlankDataset(
        dataset_dir,
        dataset_type='nondisjoint',
        dataset_split='train'
    )
    print(len(dataset))
    print(dataset[0])
    
    dataset = PolyvoreTripletDataset(
        dataset_dir,
        dataset_type='nondisjoint',
        dataset_split='train'
    )
    print(len(dataset))
    print(dataset[0])