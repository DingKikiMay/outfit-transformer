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
POLYVORE_DIR = "./src/data/datasets/polyvore"
MODEL_CHECKPOINT = "./src/data/checkpoints/best_model.pth"
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

@app.post("/search/complementary", response_model=FusionSearchResponse)
async def complementary_search(request: ComplementarySearchRequest):
    """
    互补推荐接口：
    - 用户传入已有单品（如上装/下装），可选描述和场景
    - 自动推断互补类别
    - 支持description和scene筛选候选集
    - 用模型编码图片和文本embedding，融合后送入transformer生成检索embedding
    - 用FAISS索引检索top_k互补单品，返回图片、描述、类别等信息
    """
    try:
        global items, indexer, model
        user_items = request.user_items
        scene_filter = request.scene_filter
        top_k = request.top_k
        # 1. 检查用户单品
        if not user_items or len(user_items) == 0 or items is None:
            return FusionSearchResponse(success=False, message="请至少选择一个单品", results=[], total_count=0)
        user_item = user_items[0]  # 只支持单选
        user_category = user_item.get('category', None)
        if user_category not in ['tops', 'bottoms']:
            return FusionSearchResponse(success=False, message="请选择上装或下装！", results=[], total_count=0)
        # 2. 自动推断互补类别
        target_category = 'bottoms' if user_category == 'tops' else 'tops'
        # 3. 按互补类别筛选候选集
        candidate_indices = [i for i in range(len(items)) if getattr(items.get_item_by_id(i), 'category', None) == target_category]
        # 4. 如有场合再筛选
        if scene_filter:
            candidate_indices = [i for i in candidate_indices if scene_filter in getattr(items.get_item_by_id(i), 'scene', [])]
        if not candidate_indices:
            return FusionSearchResponse(success=False, message="没有符合条件的互补单品！", results=[], total_count=0)
        # 5. 获取用户单品图像embedding
        user_img = user_item.get('image_base64', None)
        if user_img:
            from PIL import Image
            import base64, io
            image = Image.open(io.BytesIO(base64.b64decode(user_img)))
            user_img_emb = model.image_encoder(image)
        else:
            user_img_emb = None
        # 6. 如有description，编码文本embedding
        comp_description = user_item.get('description', None)
        if comp_description:
            text_emb = model.text_encoder([comp_description])
        else:
            text_emb = None
        # 7. 融合embedding
        if user_img_emb is not None and text_emb is not None:
            fusion_emb = torch.cat([user_img_emb, text_emb], dim=-1)
        elif user_img_emb is not None:
            fusion_emb = user_img_emb
        elif text_emb is not None:
            fusion_emb = text_emb
        else:
            return FusionSearchResponse(success=False, message="缺少有效的图片或描述embedding", results=[], total_count=0)
        # 8. 送入transformer生成检索embedding
        query_emb = model.transformer(fusion_emb).detach().cpu().numpy().tolist()
        # 9. 创建筛选后的FAISS索引
        filtered_indexer = create_filtered_faiss_index(candidate_indices) if candidate_indices else None
        if filtered_indexer is None:
            filtered_indexer = indexer
        # 10. 检索
        res = filtered_indexer.search(
            embeddings=query_emb,
            k=min(top_k, len(candidate_indices))
        )[0] if candidate_indices else []
        # 11. 返回图片、描述、类别等信息
        results = []
        for score, item_id in res:
            item = items.get_item_by_id(item_id)
            if item is not None:
                results.append({
                    'id': item_id,
                    'description': item.description,
                    'category': item.category,
                    'scene': item.scene,
                    'image_base64': image_to_base64(item.image) if item.image else None
                })
        return FusionSearchResponse(success=True, message="success", results=results, total_count=len(results))
    except Exception as e:
        return FusionSearchResponse(success=False, message=f"Error: {e}", results=[], total_count=0)

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