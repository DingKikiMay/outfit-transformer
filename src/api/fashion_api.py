# -*- coding:utf-8 -*-
"""
时尚推荐API接口
基于FastAPI构建，支持图片+描述+场景的融合搜索
"""
import os
import base64
import io
from typing import List, Optional, Dict, Any
from pathlib import Path
from PIL import Image
import torch
import numpy as np
import pickle
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# 导入项目模块
from ..models.load import load_model
from ..data.datatypes import FashionItem, FashionComplementaryQuery, FashionCompatibilityQuery, SCENE_TYPES
from ..demo.vectorstore import FAISSVectorStore
from ..data.datasets import polyvore

# 配置
POLYVORE_DIR = "./datasets/polyvore"
MODEL_CHECKPOINT = "./checkpoints/best_model.pth"
POLYVORE_PRECOMPUTED_REC_EMBEDDING_DIR = f"{POLYVORE_DIR}/precomputed_rec_embeddings"

# 服装类别列表
FASHION_CATEGORIES = ['tops', 'bottoms']
SCENE_TAGS = SCENE_TYPES  # ['casual', 'sport']

# 全局变量
model = None
items = None
indexer = None
metadata = None

# Pydantic模型定义
class FusionSearchRequest(BaseModel):
    """融合搜索请求模型"""
    description: Optional[str] = None
    scene_filter: Optional[str] = None
    top_k: int = 4

class FusionSearchResponse(BaseModel):
    """融合搜索响应模型"""
    success: bool
    message: str
    results: List[Dict[str, Any]]
    total_count: int

class ComplementarySearchRequest(BaseModel):
    """互补搜索请求模型"""
    user_items: List[Dict[str, Any]]
    scene_filter: Optional[str] = None
    top_k: int = 4

class CompatibilityScoreRequest(BaseModel):
    """兼容性评分请求模型"""
    outfit_items: List[Dict[str, Any]]

class CompatibilityScoreResponse(BaseModel):
    """兼容性评分响应模型"""
    success: bool
    score: float
    message: str

class ItemInfo(BaseModel):
    """商品信息模型"""
    id: str
    description: str
    category: str
    scene: List[str]
    image_base64: Optional[str] = None

class APIResponse(BaseModel):
    """通用API响应模型"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

# 初始化函数
def initialize_model():
    """初始化模型和数据"""
    global model, items, indexer, metadata
    
    print("[API] 正在初始化模型...")
    
    # 加载模型
    model = load_model(
        model_type="clip",
        checkpoint=MODEL_CHECKPOINT
    )
    model.eval()
    print("[API] 模型加载完成")
    
    # 加载数据
    metadata = polyvore.load_metadata(POLYVORE_DIR)
    items = polyvore.PolyvoreItemDataset(
        POLYVORE_DIR, metadata=metadata, load_image=True
    )
    print(f"[API] 数据加载完成，商品总数: {len(items)}")
    
    # 加载FAISS索引
    indexer = FAISSVectorStore(
        index_name='rec_index',
        d_embed=128,
        faiss_type='IndexFlatIP',
        base_dir=POLYVORE_PRECOMPUTED_REC_EMBEDDING_DIR,
    )
    print("[API] FAISS索引加载完成")

def base64_to_image(image_base64: str) -> Optional[Image.Image]:
    """将base64编码的图片转换为PIL Image"""
    try:
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        print(f"图片解码失败: {e}")
        return None

def image_to_base64(image: Image.Image) -> str:
    """将PIL Image转换为base64编码"""
    try:
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str
    except Exception as e:
        print(f"图片编码失败: {e}")
        return ""

def filter_items_by_scene(scene_filter: Optional[str] = None) -> List[int]:
    """根据场景筛选商品ID列表"""
    if not scene_filter:
        return list(range(len(items)))
    
    filtered_indices = []
    for i in range(len(items)):
        item = items.get_item_by_id(i)
        if item and hasattr(item, 'scene') and scene_filter in item.scene:
            filtered_indices.append(i)
    return filtered_indices

def create_filtered_faiss_index(filtered_indices: List[int]) -> Optional[FAISSVectorStore]:
    """为筛选后的商品创建临时FAISS索引"""
    if not filtered_indices:
        return None
    
    try:
        # 从原始索引中提取筛选后的embedding
        all_embeddings = []
        all_ids = []
        
        # 读取所有embedding文件
        embedding_files = [f for f in os.listdir(POLYVORE_PRECOMPUTED_REC_EMBEDDING_DIR) if f.endswith('.pkl')]
        for filename in sorted(embedding_files):
            filepath = os.path.join(POLYVORE_PRECOMPUTED_REC_EMBEDDING_DIR, filename)
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                all_ids.extend(data['ids'])
                all_embeddings.append(data['embeddings'])
        
        # 合并所有embedding
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        
        # 创建ID到索引的映射
        id_to_idx = {item_id: idx for idx, item_id in enumerate(all_ids)}
        
        # 提取筛选后的embedding
        filtered_embeddings = []
        filtered_ids = []
        
        for item_id in filtered_indices:
            if item_id in id_to_idx:
                idx = id_to_idx[item_id]
                filtered_embeddings.append(all_embeddings[idx])
                filtered_ids.append(item_id)
        
        if not filtered_embeddings:
            return None
        
        # 创建临时FAISS索引
        filtered_embeddings = np.array(filtered_embeddings)
        temp_indexer = FAISSVectorStore(
            index_name='temp_filtered_index',
            d_embed=128,
            faiss_type='IndexFlatIP',
            base_dir="",
        )
        
        # 添加筛选后的embedding到临时索引
        temp_indexer.add(embeddings=filtered_embeddings.tolist(), ids=filtered_ids)
        
        return temp_indexer
        
    except Exception as e:
        print(f"[WARNING] 创建筛选索引失败: {e}")
        return None

def create_fashion_item(item_data: Dict[str, Any]) -> FashionItem:
    """从请求数据创建FashionItem对象"""
    image = None
    if item_data.get('image_base64'):
        image = base64_to_image(item_data['image_base64'])
    
    return FashionItem(
        item_id=item_data.get('item_id'),
        image=image,
        description=item_data['description'],
        category=item_data['category'],
        scene=item_data.get('scene', ['casual'])
    )

# 创建FastAPI应用
app = FastAPI(
    title="时尚推荐API",
    description="基于CLIP和Transformer的时尚推荐系统API",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应该指定具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    initialize_model()

@app.get("/")
async def root():
    """根路径"""
    return {"message": "时尚推荐API服务正在运行"}

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "data_loaded": items is not None,
        "index_loaded": indexer is not None
    }

@app.get("/categories")
async def get_categories():
    """获取所有服装类别"""
    return {
        "success": True,
        "categories": FASHION_CATEGORIES
    }

@app.get("/scenes")
async def get_scenes():
    """获取所有场景标签"""
    return {
        "success": True,
        "scenes": SCENE_TAGS
    }

@app.post("/search/fusion", response_model=FusionSearchResponse)
async def fusion_search(
    image: UploadFile = File(...),
    description: Optional[str] = Form(None),
    scene_filter: Optional[str] = Form(None),
    top_k: int = Form(4)
):
    """
    融合搜索接口
    支持图片+描述+场景的融合搜索
    """
    try:
        # 验证输入
        if not image:
            raise HTTPException(status_code=400, detail="请上传图片")
        
        if scene_filter and scene_filter not in SCENE_TAGS:
            raise HTTPException(status_code=400, detail=f"无效的场景标签，可选值: {SCENE_TAGS}")
        
        print(f"[API] 融合搜索请求 - 描述: {description}, 场景: {scene_filter}, top_k: {top_k}")
        
        # 读取图片
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data))
        
        # 第一步：根据场景筛选商品
        filtered_indices = filter_items_by_scene(scene_filter)
        print(f"[API] 筛选后商品数量: {len(filtered_indices)}")
        
        if not filtered_indices:
            return FusionSearchResponse(
                success=True,
                message="筛选后没有符合条件的商品",
                results=[],
                total_count=0
            )
        
        # 第二步：创建筛选后的FAISS索引
        filtered_indexer = create_filtered_faiss_index(filtered_indices)
        
        if filtered_indexer is None:
            print("[API] 创建筛选索引失败，使用原始索引")
            filtered_indexer = indexer
        
        # 第三步：创建融合的FashionItem
        desc = description if description else "fashion item"
        
        fusion_item = FashionItem(
            item_id=None,
            image=pil_image,
            description=desc,
            category='tops',
            scene=['casual']
        )
        
        # 第四步：构建查询
        query = FashionComplementaryQuery(
            outfit=[fusion_item],
            category='tops'
        )
        
        # 第五步：生成融合查询向量
        with torch.no_grad():
            query_embedding = model.embed_query(
                query=[query],
                use_precomputed_embedding=False
            ).detach().cpu().numpy().tolist()
        
        # 第六步：在筛选后的索引中进行KNN检索
        search_results = filtered_indexer.search(
            embeddings=query_embedding,
            k=min(top_k, len(filtered_indices))
        )[0]
        
        # 第七步：构建搜索结果
        results = []
        for result in search_results:
            if isinstance(result, (list, tuple)) and len(result) >= 2:
                score, item_id = result
                item = items.get_item_by_id(item_id)
                if item:
                    results.append({
                        'id': str(item_id),
                        'description': item.description,
                        'category': item.category,
                        'scene': item.scene,
                        'score': float(score),
                        'image_base64': image_to_base64(item.image) if item.image else None
                    })
                    if len(results) >= top_k:
                        break
        
        print(f"[API] 融合搜索完成，返回 {len(results)} 个结果")
        
        return FusionSearchResponse(
            success=True,
            message="搜索成功",
            results=results,
            total_count=len(results)
        )
        
    except Exception as e:
        print(f"[API] 融合搜索失败: {e}")
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")

@app.post("/search/complementary", response_model=FusionSearchResponse)
async def complementary_search(request: ComplementarySearchRequest):
    """
    互补商品搜索接口
    根据用户已有的商品推荐互补商品
    """
    try:
        if not request.user_items:
            raise HTTPException(status_code=400, detail="请提供用户商品列表")
        
        print(f"[API] 互补搜索请求 - 商品数量: {len(request.user_items)}, 场景: {request.scene_filter}")
        
        # 转换为FashionItem对象
        fashion_items = [create_fashion_item(item) for item in request.user_items]
        
        # 自动判断目标推荐类别
        user_category = fashion_items[0].category if fashion_items else 'tops'
        target_category = 'bottoms' if user_category == 'tops' else 'tops'
        
        print(f"[API] 用户类别: {user_category} -> 目标类别: {target_category}")
        
        # 第一步：根据类别和场景筛选商品
        filtered_indices = filter_items_by_scene(request.scene_filter)
        print(f"[API] 筛选后商品数量: {len(filtered_indices)}")
        
        if not filtered_indices:
            return FusionSearchResponse(
                success=True,
                message="筛选后没有符合条件的商品",
                results=[],
                total_count=0
            )
        
        # 第二步：创建筛选后的FAISS索引
        filtered_indexer = create_filtered_faiss_index(filtered_indices)
        
        if filtered_indexer is None:
            print("[API] 创建筛选索引失败，使用原始索引")
            filtered_indexer = indexer
        
        # 第三步：构建查询
        query = FashionComplementaryQuery(
            outfit=fashion_items,
            category=target_category
        )
        
        # 第四步：生成查询向量
        with torch.no_grad():
            query_embedding = model.embed_query(
                query=[query],
                use_precomputed_embedding=False
            ).detach().cpu().numpy().tolist()
        
        # 第五步：在筛选后的索引中进行KNN检索
        search_results = filtered_indexer.search(
            embeddings=query_embedding,
            k=min(request.top_k, len(filtered_indices))
        )[0]
        
        # 第六步：构建推荐结果
        results = []
        for result in search_results:
            if isinstance(result, (list, tuple)) and len(result) >= 2:
                score, item_id = result
                item = items.get_item_by_id(item_id)
                if item:
                    results.append({
                        'id': str(item_id),
                        'description': item.description,
                        'category': item.category,
                        'scene': item.scene,
                        'score': float(score),
                        'image_base64': image_to_base64(item.image) if item.image else None
                    })
                    if len(results) >= request.top_k:
                        break
        
        print(f"[API] 互补搜索完成，返回 {len(results)} 个结果")
        
        return FusionSearchResponse(
            success=True,
            message="搜索成功",
            results=results,
            total_count=len(results)
        )
        
    except Exception as e:
        print(f"[API] 互补搜索失败: {e}")
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")

@app.post("/compatibility/score", response_model=CompatibilityScoreResponse)
async def compute_compatibility_score(request: CompatibilityScoreRequest):
    """
    计算搭配兼容性分数
    """
    try:
        if not request.outfit_items:
            raise HTTPException(status_code=400, detail="请提供搭配商品列表")
        
        print(f"[API] 兼容性评分请求 - 商品数量: {len(request.outfit_items)}")
        
        # 转换为FashionItem对象
        fashion_items = [create_fashion_item(item) for item in request.outfit_items]
        
        # 构建查询
        query = FashionCompatibilityQuery(outfit=fashion_items)
        
        # 计算分数
        with torch.no_grad():
            score = model.predict_score(
                query=[query],
                use_precomputed_embedding=False
            )[0].detach().cpu()
            
        score_value = float(score)
        print(f"[API] 兼容性评分完成: {score_value}")
        
        return CompatibilityScoreResponse(
            success=True,
            score=score_value,
            message="评分成功"
        )
        
    except Exception as e:
        print(f"[API] 兼容性评分失败: {e}")
        raise HTTPException(status_code=500, detail=f"评分失败: {str(e)}")

@app.get("/items/{item_id}")
async def get_item_info(item_id: str):
    """根据ID获取商品信息"""
    try:
        item = items.get_item_by_id(int(item_id))
        if not item:
            raise HTTPException(status_code=404, detail="商品不存在")
        
        return {
            "success": True,
            "item": {
                'id': item_id,
                'description': item.description,
                'category': item.category,
                'scene': item.scene,
                'image_base64': image_to_base64(item.image) if item.image else None
            }
        }
        
    except Exception as e:
        print(f"[API] 获取商品信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取商品信息失败: {str(e)}")

@app.get("/stats")
async def get_stats():
    """获取系统统计信息"""
    try:
        total_items = len(items) if items else 0
        
        # 按类别统计
        category_stats = {}
        for category in FASHION_CATEGORIES:
            count = 0
            for i in range(total_items):
                item = items.get_item_by_id(i)
                if item and item.category == category:
                    count += 1
            category_stats[category] = count
        
        # 按场景统计
        scene_stats = {}
        for scene in SCENE_TAGS:
            count = 0
            for i in range(total_items):
                item = items.get_item_by_id(i)
                if item and hasattr(item, 'scene') and scene in item.scene:
                    count += 1
            scene_stats[scene] = count
        
        return {
            "success": True,
            "stats": {
                "total_items": total_items,
                "categories": category_stats,
                "scenes": scene_stats
            }
        }
        
    except Exception as e:
        print(f"[API] 获取统计信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "fashion_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    ) 