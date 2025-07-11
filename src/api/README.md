# 时尚推荐API接口文档

## 概述

基于CLIP和Transformer的时尚推荐系统API，支持图片+描述+场景的融合搜索、互补商品推荐和搭配兼容性评分。

## 功能特性

- 🖼️ **融合搜索**: 支持图片+文本描述+场景筛选的智能搜索
- 👕 **互补推荐**: 根据用户已有商品推荐互补单品
- 📊 **兼容性评分**: 计算搭配的兼容性分数
- 🏷️ **场景筛选**: 支持按场景（日常/运动）筛选商品
- 🔍 **商品信息**: 获取商品详细信息和系统统计

## 快速开始

### 1. 环境准备

确保已安装必要的依赖：

```bash
pip install fastapi uvicorn python-multipart requests pillow torch numpy faiss-cpu
```

### 2. 数据准备

确保以下文件存在：
- `./datasets/polyvore/item_metadata.json` - 商品元数据
- `./datasets/polyvore/precomputed_rec_embeddings/` - 预计算的embedding
- `./checkpoints/best_model.pth` - 训练好的模型

### 3. 启动API服务

```bash
# 方式1: 使用启动脚本（推荐）
python src/api/start_api.py

# 方式2: 直接启动
python src/api/fashion_api.py
```

### 4. 访问API

- API文档: http://localhost:8000/docs
- 健康检查: http://localhost:8000/health
- 服务器地址: http://localhost:8000

## API接口详情

### 基础接口

#### 健康检查
```http
GET /health
```

**响应示例:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "data_loaded": true,
  "index_loaded": true
}
```

#### 获取类别
```http
GET /categories
```

**响应示例:**
```json
{
  "success": true,
  "categories": ["tops", "bottoms"]
}
```

#### 获取场景
```http
GET /scenes
```

**响应示例:**
```json
{
  "success": true,
  "scenes": ["casual", "sport"]
}
```

### 核心功能接口

#### 融合搜索
```http
POST /search/fusion
```

**请求参数:**
- `image` (file): 上传的图片文件
- `description` (string, optional): 文本描述
- `scene_filter` (string, optional): 场景筛选 (casual/sport)
- `top_k` (int, optional): 返回结果数量，默认4

**响应示例:**
```json
{
  "success": true,
  "message": "搜索成功",
  "results": [
    {
      "id": "123",
      "description": "白色T恤",
      "category": "tops",
      "scene": ["casual"],
      "score": 0.85,
      "image_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
    }
  ],
  "total_count": 4
}
```

#### 互补商品搜索
```http
POST /search/complementary
```

**请求体:**
```json
{
  "user_items": [
    {
      "item_id": "user_item_1",
      "description": "白色T恤",
      "category": "tops",
      "scene": ["casual"],
      "image_base64": null
    }
  ],
  "scene_filter": "casual",
  "top_k": 4
}
```

**响应示例:**
```json
{
  "success": true,
  "message": "搜索成功",
  "results": [
    {
      "id": "456",
      "description": "蓝色牛仔裤",
      "category": "bottoms",
      "scene": ["casual"],
      "score": 0.92,
      "image_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
    }
  ],
  "total_count": 4
}
```

#### 兼容性评分
```http
POST /compatibility/score
```

**请求体:**
```json
{
  "outfit_items": [
    {
      "item_id": "item_1",
      "description": "白色T恤",
      "category": "tops",
      "scene": ["casual"],
      "image_base64": null
    },
    {
      "item_id": "item_2",
      "description": "蓝色牛仔裤",
      "category": "bottoms",
      "scene": ["casual"],
      "image_base64": null
    }
  ]
}
```

**响应示例:**
```json
{
  "success": true,
  "score": 0.87,
  "message": "评分成功"
}
```

### 辅助接口

#### 获取商品信息
```http
GET /items/{item_id}
```

**响应示例:**
```json
{
  "success": true,
  "item": {
    "id": "123",
    "description": "白色T恤",
    "category": "tops",
    "scene": ["casual"],
    "image_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
  }
}
```

#### 获取统计信息
```http
GET /stats
```

**响应示例:**
```json
{
  "success": true,
  "stats": {
    "total_items": 1000,
    "categories": {
      "tops": 500,
      "bottoms": 500
    },
    "scenes": {
      "casual": 800,
      "sport": 200
    }
  }
}
```

## 使用示例

### Python客户端示例

```python
import requests
from PIL import Image
import io

# 融合搜索示例
def fusion_search_example():
    # 准备图片
    img = Image.new('RGB', (224, 224), color='white')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    # 发送请求
    files = {'image': ('test.jpg', img_bytes, 'image/jpeg')}
    data = {
        'description': '白色T恤，运动风格',
        'scene_filter': 'casual',
        'top_k': 4
    }
    
    response = requests.post('http://localhost:8000/search/fusion', 
                           files=files, data=data)
    
    if response.status_code == 200:
        results = response.json()
        print(f"找到 {results['total_count']} 个匹配商品")
        for item in results['results']:
            print(f"- {item['description']} (分数: {item['score']:.3f})")

# 互补搜索示例
def complementary_search_example():
    user_items = [
        {
            "description": "白色T恤",
            "category": "tops",
            "scene": ["casual"]
        }
    ]
    
    response = requests.post('http://localhost:8000/search/complementary',
                           json={"user_items": user_items, "top_k": 4})
    
    if response.status_code == 200:
        results = response.json()
        print(f"推荐 {results['total_count']} 个互补商品")

# 兼容性评分示例
def compatibility_score_example():
    outfit = [
        {"description": "白色T恤", "category": "tops", "scene": ["casual"]},
        {"description": "蓝色牛仔裤", "category": "bottoms", "scene": ["casual"]}
    ]
    
    response = requests.post('http://localhost:8000/compatibility/score',
                           json={"outfit_items": outfit})
    
    if response.status_code == 200:
        result = response.json()
        print(f"搭配兼容性分数: {result['score']:.3f}")
```

### JavaScript客户端示例

```javascript
// 融合搜索
async function fusionSearch(imageFile, description, sceneFilter) {
    const formData = new FormData();
    formData.append('image', imageFile);
    formData.append('description', description);
    formData.append('scene_filter', sceneFilter);
    formData.append('top_k', 4);
    
    const response = await fetch('http://localhost:8000/search/fusion', {
        method: 'POST',
        body: formData
    });
    
    const result = await response.json();
    return result;
}

// 互补搜索
async function complementarySearch(userItems, sceneFilter) {
    const response = await fetch('http://localhost:8000/search/complementary', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            user_items: userItems,
            scene_filter: sceneFilter,
            top_k: 4
        })
    });
    
    const result = await response.json();
    return result;
}

// 兼容性评分
async function compatibilityScore(outfitItems) {
    const response = await fetch('http://localhost:8000/compatibility/score', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            outfit_items: outfitItems
        })
    });
    
    const result = await response.json();
    return result;
}
```

## 测试API

运行测试脚本验证API功能：

```bash
python src/api/example_usage.py
```

## 配置说明

### 环境变量

- `TOKENIZERS_PARALLELISM=false`: 禁用tokenizer并行化

### 文件路径配置

在 `fashion_api.py` 中修改以下路径：

```python
POLYVORE_DIR = "./datasets/polyvore"  # 数据集目录
MODEL_CHECKPOINT = "./checkpoints/best_model.pth"  # 模型检查点
```

### 服务器配置

```python
# 修改服务器配置
uvicorn.run(
    app,
    host="0.0.0.0",  # 监听地址
    port=8000,       # 端口号
    reload=False,    # 是否自动重载
    log_level="info" # 日志级别
)
```

## 错误处理

### 常见错误码

- `400 Bad Request`: 请求参数错误
- `404 Not Found`: 资源不存在
- `500 Internal Server Error`: 服务器内部错误

### 错误响应格式

```json
{
  "detail": "错误描述信息"
}
```

## 性能优化

### 生产环境建议

1. **使用GPU**: 安装 `faiss-gpu` 替代 `faiss-cpu`
2. **模型缓存**: 模型加载后常驻内存
3. **索引优化**: 使用更高效的FAISS索引类型
4. **并发处理**: 配置适当的worker数量

### 监控指标

- 请求响应时间
- 内存使用情况
- GPU利用率（如果使用）
- 错误率统计

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型文件路径
   - 确认模型文件完整性

2. **数据加载失败**
   - 检查数据集路径
   - 确认数据文件格式

3. **FAISS索引错误**
   - 检查embedding文件
   - 确认索引类型匹配

4. **内存不足**
   - 减少batch size
   - 使用更小的模型

### 日志查看

API服务会输出详细的日志信息，包括：
- 模型加载状态
- 请求处理过程
- 错误堆栈信息

## 更新日志

### v1.0.0
- 初始版本发布
- 支持融合搜索、互补推荐、兼容性评分
- 提供完整的REST API接口
- 包含Python和JavaScript客户端示例 