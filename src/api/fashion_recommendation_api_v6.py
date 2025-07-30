#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时尚单品推荐API接口 V6
优化版本：通过HTTP API间接访问数据库，简化请求体，只需要ID即可
"""

import os
import sys
import json
import base64
import io
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
import torch
from PIL import Image
import logging
import faiss
import requests
from urllib.parse import urljoin

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.load import load_model
from src.demo.vectorstore import FAISSVectorStore

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProductInfo:
    """商品信息数据结构（简化版）"""
    product_id: int
    description: str
    image_url: str
    scene: str
    category_id: int  # 新增：商品所属子类ID


@dataclass
class UserInput:
    """用户输入数据结构（简化版）"""
    product_id: Optional[int] = None  # 用户选择的商品ID
    scene: Optional[str] = None       # 目标场景：casual/sports


@dataclass
class RecommendationResult:
    """推荐结果数据结构"""
    product_id: int                   # 推荐商品ID



class FashionRecommendationAPIV6:
    """时尚单品推荐API类 V6 - 通过API间接访问数据库"""
    
    def __init__(self, model_path: str, model_type: str = 'clip', faiss_index_path: str = None, api_base_url: str = None):
        """
        初始化API
        
        Args:
            model_path: 训练好的模型权重路径
            model_type: 模型类型 ('original' 或 'clip')
            faiss_index_path: FAISS索引文件路径
            api_base_url: 后端API基础URL
        """
        self.model_path = model_path
        self.model_type = model_type
        self.faiss_index_path = faiss_index_path
        # 确保API基础URL包含完整的路径
        default_api_url = "https://cloudawn3d.com/"
        self.api_base_url = api_base_url or default_api_url
        
        self.ssl_verify = True  # SSL证书验证开关
        self.model = None
        self.faiss_index = None
        self.metadata = {}  # product_id -> product_info
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 商品分类索引（启动时构建，运行时查询）
        self.product_category_index = {}  # product_id -> category ('tops' or 'bottoms')
        
        # 上下装分类映射
        self.category_mapping = {
            23: "tops",      # 上装
            24: "bottoms"    # 下装
        }
        
        # 初始化模型
        self._load_model()
        
        # 构建商品分类索引
        self._build_product_category_index()
        
        # 加载FAISS索引（如果存在）
        if faiss_index_path and os.path.exists(faiss_index_path):
            self._load_faiss_index()
        
        logger.info(f"API V6初始化完成，使用设备: {self.device}")
        logger.info(f"后端API地址: {self.api_base_url}")
    
    def _load_model(self):
        """加载模型"""
        try:
            logger.info(f"正在加载模型: {self.model_path}")
            self.model = load_model(model_type=self.model_type, checkpoint=self.model_path)
            self.model.eval()
            self.model.to(self.device)
            logger.info("模型加载成功")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def _load_faiss_index(self):
        """加载FAISS索引"""
        try:
            logger.info(f"正在加载FAISS索引: {self.faiss_index_path}")
            # 使用构造函数加载FAISS索引
            self.faiss_index = FAISSVectorStore(
                index_name=os.path.splitext(os.path.basename(self.faiss_index_path))[0],
                faiss_type='IndexFlatIP',
                base_dir=os.path.dirname(self.faiss_index_path),
                d_embed=512
            )
            
            # 加载元数据
            metadata_path = self.faiss_index_path.replace('.faiss', '_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                logger.info(f"加载元数据成功，共{len(self.metadata)}个商品")
            
            logger.info("FAISS索引加载成功")
        except Exception as e:
            logger.error(f"FAISS索引加载失败: {e}")
            raise
    
    def _build_product_category_index(self):
        """构建商品ID到分类的索引"""
        try:
            logger.info("开始构建商品分类索引...")
            
            # 遍历上装和下装分类
            for parent_id, category in self.category_mapping.items():
                logger.info(f"构建{category}分类索引...")
                
                # 获取子类ID
                subcategory_ids = self._get_subcategory_ids(parent_id)
                logger.info(f"{category}分类有{len(subcategory_ids)}个子类")
                
                # 获取所有子类的商品
                for subcategory_id in subcategory_ids:
                    try:
                        products = self._get_products_by_type_id(subcategory_id, page_size=1000)
                        logger.info(f"子类{subcategory_id}获取到{len(products)}个商品")
                        
                        # 将商品ID映射到分类
                        for product in products:
                            product_id = product.get('product_id')
                            if product_id:
                                self.product_category_index[product_id] = category
                        
                    except Exception as e:
                        logger.warning(f"获取子类{subcategory_id}商品失败: {e}")
                        continue
            
            logger.info(f"商品分类索引构建完成，共{len(self.product_category_index)}个商品")
            
        except Exception as e:
            logger.error(f"构建商品分类索引失败: {e}")
            # 不抛出异常，允许程序继续运行
    
    def _call_backend_api(self, endpoint: str, params: dict = None) -> Optional[Dict]:
        """调用后端API"""
        try:
            logger.info(f"API基础URL: {self.api_base_url}")
            logger.info(f"Endpoint: {endpoint}")
            url = urljoin(self.api_base_url, endpoint)
            logger.info(f"调用后端API: {url}")
            
            # 对于HTTPS请求，添加SSL验证配置
            response = requests.get(
                url, 
                params=params, 
                timeout=10,
                verify=self.ssl_verify  # 使用配置的SSL验证设置
            )
            response.raise_for_status()
            
            result = response.json()
            if result.get('code') == 200:  # 200表示成功
                return result.get('data')
            else:
                logger.error(f"后端API返回错误: {result.get('message')}")
                return None
                
        except requests.exceptions.SSLError as e:
            logger.error(f"SSL证书验证失败: {e}")
            # 如果SSL证书有问题，可以尝试不验证证书（仅用于测试）
            try:
                logger.warning("尝试跳过SSL证书验证...")
                response = requests.get(url, params=params, timeout=10, verify=False)
                response.raise_for_status()
                
                result = response.json()
                if result.get('code') == 200:
                    return result.get('data')
                else:
                    logger.error(f"后端API返回错误: {result.get('message')}")
                    return None
            except Exception as e2:
                logger.error(f"跳过SSL验证后仍然失败: {e2}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"调用后端API失败: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"解析API响应失败: {e}")
            return None
    
    def _get_product_by_id(self, product_id: int) -> Optional[Dict]:
        """根据商品ID从后端API获取商品信息"""
        try:
            endpoint = f"mall/getProductDetail/{product_id}"
            product_data = self._call_backend_api(endpoint)
            
            if product_data:
                # 从预构建的分类索引中获取分类信息
                category = self.product_category_index.get(product_id, "unknown")
                
                # 转换API响应格式为内部格式
                return {
                    'product_id': product_data.get('id'),
                    'description': product_data.get('description', ''),
                    'image_url': product_data.get('imageUrl', [None])[0] if product_data.get('imageUrl') else None,
                    'scene': product_data.get('scene', ''),
                    'category': category  # 使用预构建的分类索引
                }
            
            return None
        except Exception as e:
            logger.error(f"获取商品信息失败: {e}")
            return None
    
    def _get_subcategory_ids(self, parent_id: int) -> List[int]:
        """获取指定父类下的所有子类ID"""
        try:
            endpoint = "mall/getSubCategory"
            params = {'parentId': parent_id}
            
            subcategories = self._call_backend_api(endpoint, params)
            
            if subcategories:
                subcategory_ids = [cat['id'] for cat in subcategories]
                logger.info(f"获取到父类{parent_id}的子类ID: {subcategory_ids}")
                return subcategory_ids
            else:
                logger.warning(f"未获取到父类{parent_id}的子类信息")
                return []
                
        except Exception as e:
            logger.error(f"获取子类ID失败: {e}")
            return []
    
    def _get_products_by_type_id(self, type_id: int, page: int = 1, page_size: int = 100) -> List[Dict]:
        """根据子类ID获取商品列表"""
        try:
            endpoint = f"mall/getProductByTypeId/{type_id}"
            params = {
                'page': page,
                'pageSize': page_size
            }
            
            products_data = self._call_backend_api(endpoint, params)
            
            if products_data:
                # 转换API响应格式
                products = []
                for product in products_data:
                    products.append({
                        'product_id': product.get('id'),
                        'description': product.get('description', ''),
                        'image_url': product.get('imageGif', ''),  # 使用imageGif字段作为图片URL
                        'scene': product.get('scene', ''),
                        'category_id': type_id  # 使用传入的type_id作为category_id
                    })
                return products
            
            return []
        except Exception as e:
            logger.error(f"获取子类{type_id}商品列表失败: {e}")
            return []
    
    def _get_complementary_products(self, user_category_id: int) -> List[Dict]:
        """
        获取互补类别的商品（不进行场景筛选）
        
        Args:
            user_category_id: 用户商品的类别ID
            
        Returns:
            互补类别的所有商品列表（包含所有场景）
        """
        try:
            # 1. 确定互补类别的父类ID
            if user_category_id in self.category_mapping:
                user_category = self.category_mapping[user_category_id]
                if user_category == "tops":
                    complementary_parent_id = 24  # 下装
                else:
                    complementary_parent_id = 23  # 上装
            else:
                # 如果无法确定，默认获取下装
                complementary_parent_id = 24
                logger.warning(f"无法确定用户商品类别{user_category_id}，默认获取下装")
            
            logger.info(f"用户商品类别: {user_category_id}, 互补父类ID: {complementary_parent_id}")
            
            # 2. 获取互补类别的所有子类ID
            subcategory_ids = self._get_subcategory_ids(complementary_parent_id)
            if not subcategory_ids:
                logger.error("未获取到互补类别的子类ID")
                return []
            
            # 3. 获取所有互补子类的商品（不进行场景筛选）
            all_complementary_products = []
            for subcategory_id in subcategory_ids:
                logger.info(f"获取子类{subcategory_id}的商品...")
                products = self._get_products_by_type_id(subcategory_id, page_size=1000)
                all_complementary_products.extend(products)
            
            logger.info(f"获取到{len(all_complementary_products)}个互补类别商品（包含所有场景）")
            return all_complementary_products
                
        except Exception as e:
            logger.error(f"获取互补商品失败: {e}")
            return []
    
    def _get_complementary_product_ids(self, user_category_id: int) -> List[int]:
        """
        获取互补类别的商品ID列表（用于FAISS索引筛选）
        
        Args:
            user_category_id: 用户商品的类别ID
            
        Returns:
            互补类别的商品ID列表
        """
        try:
            # 1. 确定互补类别的父类ID
            if user_category_id in self.category_mapping:
                user_category = self.category_mapping[user_category_id]
                if user_category == "tops":
                    complementary_parent_id = 24  # 下装
                else:
                    complementary_parent_id = 23  # 上装
            else:
                # 如果无法确定，默认获取下装
                complementary_parent_id = 24
                logger.warning(f"无法确定用户商品类别{user_category_id}，默认获取下装")
            
            logger.info(f"用户商品类别: {user_category_id}, 互补父类ID: {complementary_parent_id}")
            
            # 2. 获取互补类别的所有子类ID
            subcategory_ids = self._get_subcategory_ids(complementary_parent_id)
            if not subcategory_ids:
                logger.error("未获取到互补类别的子类ID")
                return []
            
            # 3. 获取所有互补子类的商品ID
            all_complementary_product_ids = []
            for subcategory_id in subcategory_ids:
                logger.info(f"获取子类{subcategory_id}的商品ID...")
                products = self._get_products_by_type_id(subcategory_id, page_size=1000)
                product_ids = [p['product_id'] for p in products]
                all_complementary_product_ids.extend(product_ids)
            
            logger.info(f"获取到{len(all_complementary_product_ids)}个互补类别商品ID")
            return all_complementary_product_ids
                
        except Exception as e:
            logger.error(f"获取互补商品ID失败: {e}")
            return []
    
    def _get_products_by_scene(self, scene: str, page: int = 1, page_size: int = 100) -> List[Dict]:
        """根据场景从后端API获取商品列表"""
        try:
            endpoint = f"mall/getProductByScene/{scene}"
            params = {
                'page': page,
                'pageSize': page_size
            }
            
            products_data = self._call_backend_api(endpoint, params)
            
            if products_data:
                # 转换API响应格式
                products = []
                for product in products_data:
                    products.append({
                        'product_id': product.get('id'),
                        'description': product.get('description', ''),
                        'image_url': product.get('imageGif', ''),  # 使用gif图作为主图
                        'scene': product.get('scene', '')
                    })
                return products
            
            return []
        except Exception as e:
            logger.error(f"获取场景商品列表失败: {e}")
            return []
    
    def _download_image(self, image_url: str) -> Image.Image:
        """下载图片"""
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))
            return image
        except Exception as e:
            logger.error(f"图片下载失败: {e}")
            raise
    
    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """预处理图片，返回numpy数组格式"""
        # 调整图片大小到模型要求的尺寸
        image = image.resize((224, 224))
        # 转换为numpy数组，保持RGB格式 (H, W, C)
        image_array = np.array(image)
        return image_array
    
    def _generate_embedding(self, image: Image.Image, description: str) -> np.ndarray:
        """生成embedding（只需要图片和描述）"""
        try:
            with torch.no_grad():
                # 创建FashionItem对象，只需要图片和描述
                from src.data.datatypes import FashionItem
                
                fashion_item = FashionItem(
                    image=image,
                    description=description
                )
                
                # 使用模型的embed_item方法
                embedding = self.model.embed_item([fashion_item], use_precomputed_embedding=False)
                
                # 确保返回的是numpy数组
                if isinstance(embedding, torch.Tensor):
                    embedding = embedding.detach().cpu().numpy()
                
                # 如果是batch结果，取第一个
                if embedding.ndim > 1 and embedding.shape[0] == 1:
                    embedding = embedding[0]
                
                return embedding
        except Exception as e:
            logger.error(f"生成embedding失败: {e}")
            raise
    
    def build_faiss_index_from_api(self, scene: str = None, save_path: str = None):
        """
        从后端API构建FAISS索引（预处理阶段）
        按上装/下装分类分别构建索引
        
        Args:
            scene: 场景参数（已废弃，索引将包含所有场景的商品）
            save_path: 保存路径（基础路径，会自动添加分类后缀）
        
        Note:
            - 索引构建时会包含所有场景的商品，不进行场景过滤
            - 场景筛选在推荐阶段进行，确保推荐的灵活性
        """
        if save_path is None:
            save_path = self.faiss_index_path or "faiss_index.faiss"
        
        try:
            logger.info(f"开始从API构建分类FAISS索引")
            
            # 分别构建上装和下装的索引（不传入scene参数）
            tops_result = self._build_category_index('tops', save_path)
            bottoms_result = self._build_category_index('bottoms', save_path)
            
            total_valid = tops_result + bottoms_result
            logger.info(f"分类FAISS索引构建完成，上装: {tops_result}个，下装: {bottoms_result}个，总计: {total_valid}个")
            
            return total_valid
            
        except Exception as e:
            logger.error(f"构建分类FAISS索引失败: {e}")
            raise
    
    def _build_category_index(self, category: str, base_save_path: str = None):
        """
        构建特定分类的FAISS索引
        
        Args:
            category: 分类 ('tops' 或 'bottoms')
            base_save_path: 基础保存路径
        """
        # 确定分类ID
        if category == 'tops':
            parent_id = 23  # 上装父类ID
        elif category == 'bottoms':
            parent_id = 24  # 下装父类ID
        else:
            raise ValueError(f"不支持的分类: {category}")
        
        # 构建保存路径
        category_save_path = base_save_path.replace('.faiss', f'_{category}.faiss')
        
        try:
            logger.info(f"开始构建{category}分类的FAISS索引")
            
            # 初始化FAISS索引
            d_embed = 512  # 根据你的模型输出维度调整
            category_index = FAISSVectorStore(d_embed=d_embed, faiss_type='IndexFlatIP')
            
            # 1. 获取该分类的所有子类ID
            logger.info(f"获取{category}分类的子类ID...")
            subcategory_ids = self._get_subcategory_ids(parent_id)
            
            if not subcategory_ids:
                logger.warning(f"没有获取到{category}分类的子类ID")
                return 0
            
            logger.info(f"获取到{category}分类的子类ID: {subcategory_ids}")
            
            # 2. 获取所有子类的商品数据
            all_products = []
            for subcategory_id in subcategory_ids:
                logger.info(f"获取子类{subcategory_id}的商品...")
                products = self._get_products_by_type_id(subcategory_id, page_size=1000)
                all_products.extend(products)
                logger.info(f"子类{subcategory_id}获取到{len(products)}个商品")
            
            if not all_products:
                logger.warning(f"没有获取到{category}分类的商品数据")
                return 0
            
            logger.info(f"从API获取到 {len(all_products)} 个{category}商品（包含所有子类）")
            
            # 生成该分类商品的embedding
            embeddings = []
            valid_products = []
            category_metadata = {}
            
            for i, product in enumerate(all_products):
                try:
                    logger.info(f"处理{category}商品 {i+1}/{len(all_products)}: {product['product_id']}")
                    
                    if not product['image_url']:
                        logger.warning(f"商品{product['product_id']}没有图片URL，跳过")
                        continue
                    
                    # 下载图片
                    image = self._download_image(product['image_url'])
                    
                    # 生成embedding（只需要图片和描述）
                    embedding = self._generate_embedding(image, product['description'])
                    embeddings.append(embedding)
                    valid_products.append(product)
                    
                    # 保存元数据
                    category_metadata[str(product['product_id'])] = {
                        'product_id': product['product_id'],
                        'description': product['description'],
                        'image_url': product['image_url'],
                        'scene': product['scene'],
                        'category_id': product.get('category_id'),
                        'category': category
                    }
                    
                    # 填充商品分类索引
                    self.product_category_index[product['product_id']] = category
                    
                except Exception as e:
                    logger.warning(f"处理{category}商品{product['product_id']}失败: {e}")
                    continue
            
            if not embeddings:
                logger.warning(f"没有成功生成任何{category}商品的embedding")
                return 0
            
            # 添加到FAISS索引
            embeddings_array = np.array(embeddings)
            # 准备商品ID列表
            product_ids = [product['product_id'] for product in valid_products]
            # 转换为列表格式
            embeddings_list = embeddings_array.tolist()
            category_index.add(embeddings=embeddings_list, ids=product_ids)
            
            # 保存商品ID到索引位置的映射
            id_to_index = {product_id: idx for idx, product_id in enumerate(product_ids)}
            
            # 保存FAISS索引
            # 临时修改index_path，然后保存
            original_path = category_index.index_path
            category_index.index_path = category_save_path
            category_index.save()
            category_index.index_path = original_path  # 恢复原始路径
            
            # 保存元数据（包含ID到索引的映射）
            metadata_data = {
                'metadata': category_metadata,
                'id_to_index': id_to_index,
                'product_ids': product_ids
            }
            metadata_path = category_save_path.replace('.faiss', '_metadata.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"{category}分类FAISS索引构建完成，保存到: {category_save_path}")
            logger.info(f"{category}有效商品数量: {len(valid_products)}")
            
            return len(valid_products)
            
        except Exception as e:
            logger.error(f"构建{category}分类FAISS索引失败: {e}")
            raise
    
    def recommend_best_item(self, user_input: UserInput) -> Optional[RecommendationResult]:
        """
        推荐最佳单品（在线检索阶段）
        使用分类FAISS索引进行推荐：用户商品是上装就推荐下装，反之亦然
        
        Args:
            user_input: 用户输入
            
        Returns:
            推荐结果（top-1最佳单品）
        """
        try:
            # 1. 获取用户单品信息
            if user_input.product_id is not None:
                # 从后端API获取商品信息
                product_info = self._get_product_by_id(user_input.product_id)
                
                if not product_info:
                    raise ValueError(f"无法找到商品ID {user_input.product_id}")
                
                if not product_info['image_url']:
                    raise ValueError(f"商品ID {user_input.product_id} 没有图片URL")
                
                image = self._download_image(product_info['image_url'])
                description = product_info['description']
            else:
                raise ValueError("必须提供product_id")
            
            # 2. 确定用户商品的分类和互补分类
            user_category = None
            complementary_category = None
            
            # 从分类索引中查找商品分类
            if user_input.product_id in self.product_category_index:
                user_category = self.product_category_index[user_input.product_id]
                logger.info(f"商品{user_input.product_id}分类: {user_category}")
            else:
                # 索引中没有，说明商品不存在或未收录
                raise ValueError(f"商品ID {user_input.product_id} 不在任何分类中，请检查商品是否存在")
            
            # 确定互补分类
            if user_category == 'tops':
                complementary_category = 'bottoms'
            else:
                complementary_category = 'tops'
            
            logger.info(f"用户商品分类: {user_category}, 需要推荐的互补分类: {complementary_category}")
            
            # 3. 加载互补分类的FAISS索引
            complementary_index_path = self.faiss_index_path.replace('.faiss', f'_{complementary_category}.faiss')
            complementary_metadata_path = complementary_index_path.replace('.faiss', '_metadata.json')
            
            if not os.path.exists(complementary_index_path):
                raise ValueError(f"互补分类 {complementary_category} 的FAISS索引不存在: {complementary_index_path}")
            
            # 加载互补分类的FAISS索引
            # 检查索引文件是否存在
            if not os.path.exists(complementary_index_path):
                raise ValueError(f"互补分类 {complementary_category} 的FAISS索引文件不存在: {complementary_index_path}")
            
            # 直接使用faiss读取索引
            import faiss
            faiss_index = faiss.read_index(complementary_index_path)
            
            # 创建FAISSVectorStore包装器
            complementary_index = FAISSVectorStore(
                index_name=os.path.splitext(os.path.basename(complementary_index_path))[0],
                faiss_type='IndexFlatIP',
                base_dir=os.path.dirname(complementary_index_path),
                d_embed=512
            )
            # 替换内部的index
            complementary_index.index = faiss_index
            
            # 加载互补分类的元数据
            with open(complementary_metadata_path, 'r', encoding='utf-8') as f:
                metadata_data = json.load(f)
            
            # 兼容旧格式和新格式
            if isinstance(metadata_data, dict) and 'metadata' in metadata_data:
                # 新格式
                complementary_metadata = metadata_data['metadata']
                id_to_index = metadata_data.get('id_to_index', {})
                product_ids = metadata_data.get('product_ids', [])
            else:
                # 旧格式
                complementary_metadata = metadata_data
                id_to_index = {}
                product_ids = []
            
            logger.info(f"加载了 {len(complementary_metadata)} 个{complementary_category}商品的索引")
            logger.info(f"元数据键: {list(complementary_metadata.keys())}")
            logger.info(f"商品ID列表: {product_ids}")
            
            # 4. 生成用户单品的embedding
            logger.info("正在生成用户单品的embedding...")
            user_embedding = self._generate_embedding(image, description)
            
            # 5. 在互补分类的FAISS索引中进行检索
            logger.info(f"正在{complementary_category}分类中进行FAISS检索...")
            k = min(100, len(complementary_metadata))  # 检索前100个候选
            logger.info(f"检索参数: k={k}, 元数据数量={len(complementary_metadata)}")
            
            search_results = complementary_index.search([user_embedding.tolist()], k)
            logger.info(f"检索结果: {len(search_results)} 个查询结果")
            
            # 检查检索结果是否为空
            if not search_results or len(search_results) == 0:
                logger.error("FAISS检索返回空结果")
                raise ValueError("FAISS检索失败，没有返回任何结果")
            
            # 6. 在检索结果中筛选最佳单品
            best_product = None
            best_score = 0.0
            
            # search_results[0] 是第一个查询的结果列表
            logger.info(f"第一个查询结果数量: {len(search_results[0])}")
            
            # 处理检索结果
            logger.info(f"FAISS返回的索引: {[idx for _, idx in search_results[0]]}")
            
            for score, idx in search_results[0]:
                if idx == -1:  # FAISS返回-1表示无效索引
                    continue
                
                # FAISS返回的索引就是商品ID本身，直接使用
                candidate_product_id = idx
                logger.info(f"FAISS索引 {idx} 对应商品ID: {candidate_product_id}")
                
                # 获取候选商品信息
                if str(candidate_product_id) not in complementary_metadata:
                    logger.warning(f"商品ID {candidate_product_id} 不在元数据中，跳过")
                    continue
                    
                candidate_product = complementary_metadata[str(candidate_product_id)]
                
                # 场景筛选：如果指定了目标场景，只选择符合场景的商品
                if user_input.scene and candidate_product['scene'] != user_input.scene:
                    continue
                
                # 计算相似度分数
                score = float(score)  # 对于IndexFlatIP，距离就是相似度
                
                # 更新最佳单品（取最相似的）
                if score > best_score:
                    best_score = score
                    best_product = candidate_product
            
            if best_product is None:
                logger.warning(f"没有找到符合目标场景的{complementary_category}推荐单品")
                return None
            
            result = RecommendationResult(
                product_id=best_product['product_id'],
            )
            
            logger.info(f"推荐完成，最佳{complementary_category}单品ID: {best_product['product_id']}, 分数: {best_score}")
            return result
            
        except Exception as e:
            logger.error(f"推荐过程中出现错误: {e}")
            raise
    
    def get_api_info(self) -> Dict[str, Any]:
        """获取API信息"""
        info = {
            'version': '6.1',  # 更新版本号
            'model_type': self.model_type,
            'device': str(self.device),
            'backend_api_url': self.api_base_url,
            'supported_scenes': ['casual', 'sports'],
            'category_mapping': self.category_mapping,  # 新增类别映射
            'endpoints': {
                'recommend': '/api/recommend_best_item',
                'build_index': '/api/build_faiss_index_from_api',
                'health': '/health',
                'info': '/api_info'
            },
            'features': {
                'top_k': 1,  # 固定返回top-1
                'embedding_input': ['image', 'description'],  # 只需要图片和描述
                'subcategory_filtering': True,  # 新增：支持子类筛选
                'complementary_matching': True,  # 新增：支持互补匹配
                'backend_integration': 'HTTP API'  # 通过HTTP API访问后端
            }
        }
        
        if self.faiss_index is not None:
            info['faiss_index'] = {
                'loaded': True,
                'total_items': len(self.metadata),
                'index_path': self.faiss_index_path
            }
        else:
            info['faiss_index'] = {
                'loaded': False,
                'message': '需要先构建FAISS索引'
            }
        
        # 添加分类索引信息
        info['category_index'] = {
            'total_products': len(self.product_category_index),
            'tops_count': sum(1 for cat in self.product_category_index.values() if cat == 'tops'),
            'bottoms_count': sum(1 for cat in self.product_category_index.values() if cat == 'bottoms')
        }
        
        return info
