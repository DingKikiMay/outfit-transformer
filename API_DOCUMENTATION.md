# 时尚AI算法端API文档

## 概述

本API提供时尚单品互补推荐服务，基于训练好的CIR（Complementary Item Retrieval）模型，实现用户单品的互补搭配推荐。

## 部署信息

### 服务器配置
- **推荐配置**: 8核16G内存，NVIDIA GPU
- **API地址**: `http://your-server-ip:5000`
- **健康检查**: `http://your-server-ip:5000/health`

### 环境变量
```bash
MODEL_PATH=/app/models/cir_best_model.pth
MODEL_TYPE=clip
POLYVORE_DIR=/app/src/data/datasets/polyvore
FAISS_INDEX_PATH=/app/src/data/datasets/polyvore/precomputed_rec_embeddings/rec_index.faiss
HOST=0.0.0.0
PORT=5000
DEBUG=False
```

## API接口

### 1. 健康检查

**接口地址**: `GET /health`

**请求示例**:
```bash
curl http://your-server-ip:5000/health
```

**响应示例**:
```json
{
  "status": "ok",
  "message": "API运行正常",
  "api_info": {
    "version": "2.0",
    "model_type": "clip",
    "device": "cuda",
    "total_items": 10000,
    "supported_categories": ["tops", "bottoms"],
    "supported_scenes": ["casual", "sport", "work", "party"]
  }
}
```

### 2. 互补单品检索

**接口地址**: `POST /api/retrieve_complementary_items`

**Content-Type**: `application/json`

#### 请求体（Request Body）

| 字段名      | 类型   | 是否必填 | 说明                | 示例值                     |
| ----------- | ------ | -------- | ------------------- | -------------------------- |
| product_id  | int    | 否       | 商品ID（优先使用）  | 123                        |
| image_url   | string | 否       | 商品图片URL         | "https://xxx.com/xxx.jpg"  |
| description | string | 否       | 商品描述            | "黑色牛仔裤，适合休闲场合" |
| category    | string | 是       | 商品类别            | "bottoms"                  |
| scene       | string | 否       | 目标场景            | "casual"                   |
| top_k       | int    | 否       | 返回结果数量，默认8 | 8                          |

**字段说明**:
- `product_id`、`image_url`、`description` 至少要有一项能获取到商品内容
- `category` 必须是 "tops" 或 "bottoms"
- `scene` 可选值：["casual", "sport", "work", "party"]
- `top_k` 范围：1-20，默认8

#### 响应体（Response Body）

| 字段名  | 类型    | 说明         |
| ------- | ------- | ------------ |
| success | boolean | 请求是否成功 |
| message | string  | 响应消息     |
| data    | object  | 响应数据     |

**data字段结构**:
| 字段名      | 类型  | 说明         |
| ----------- | ----- | ------------ |
| results     | array | 检索结果列表 |
| total_count | int   | 总结果数量   |

**results数组每个元素**:
| 字段名      | 类型   | 说明              |
| ----------- | ------ | ----------------- |
| product_id  | int    | 商品ID            |
| score       | float  | 相似度分数（0-1） |
| category    | string | 商品类别          |
| scene       | array  | 场景标签列表      |

#### 请求示例

**方式一：通过商品ID**
```bash
curl -X POST http://your-server-ip:5000/api/retrieve_complementary_items \
  -H "Content-Type: application/json" \
  -d '{
    "product_id": 123,
    "category": "bottoms",
    "scene": "casual",
    "top_k": 8
  }'
```

**方式二：通过图片URL和描述**
```bash
curl -X POST http://your-server-ip:5000/api/retrieve_complementary_items \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/image.jpg",
    "description": "黑色牛仔裤，适合休闲场合",
    "category": "bottoms",
    "scene": "casual",
    "top_k": 8
  }'
```

#### 响应示例

**成功响应**:
```json
{
  "success": true,
  "message": "检索成功",
  "data": {
    "results": [
      {
        "product_id": 456,
        "score": 0.92,
        "category": "tops",
        "scene": ["casual", "sport"]
      },
      {
        "product_id": 789,
        "score": 0.89,
        "category": "tops",
        "scene": ["work"]
      },
      {
        "product_id": 101,
        "score": 0.85,
        "category": "tops",
        "scene": ["casual"]
      }
    ],
    "total_count": 3
  }
}
```

**错误响应**:
```json
{
  "success": false,
  "message": "category必须是tops或bottoms",
  "data": null
}
```

### 3. API信息

**接口地址**: `GET /api_info`

**请求示例**:
```bash
curl http://your-server-ip:5000/api_info
```

**响应示例**:
```json
{
  "success": true,
  "message": "获取成功",
  "data": {
    "version": "2.0",
    "model_type": "clip",
    "device": "cuda",
    "total_items": 10000,
    "supported_categories": ["tops", "bottoms"],
    "supported_scenes": ["casual", "sport", "work", "party"],
    "endpoints": {
      "recommend": "/api/retrieve_complementary_items",
      "health": "/health",
      "info": "/api_info"
    }
  }
}
```

### 4. 测试接口

**接口地址**: `POST /test`

**请求示例**:
```bash
curl -X POST http://your-server-ip:5000/test \
  -H "Content-Type: application/json"
```

## 业务流程

### 完整流程

1. **用户输入**：用户选择单品（图片+描述+类别）
2. **embedding生成**：用模型把用户单品编码成embedding
3. **场景筛选**：根据场景标签筛选候选商品
4. **FAISS检索**：用embedding在FAISS中检索最相似的商品ID
5. **返回结果**：返回商品ID，后端用ID查数据库获取详细信息

### 算法端职责

- ✅ 接收用户单品信息
- ✅ 生成embedding
- ✅ 场景和类别筛选
- ✅ FAISS向量检索
- ✅ 返回推荐商品ID列表

### 后端职责

- ✅ 接收前端请求
- ✅ 调用算法端API
- ✅ 根据商品ID查询数据库
- ✅ 组装完整商品信息
- ✅ 返回给前端

## 错误码说明

| 错误码 | 说明 |
|--------|------|
| 400 | 请求参数错误 |
| 500 | 服务器内部错误 |

### 常见错误信息

- `"category必须是tops或bottoms"` - 商品类别参数错误
- `"必须提供product_id、image_url或description中的至少一个"` - 缺少商品信息
- `"无法获取商品ID xxx的信息"` - 商品ID不存在
- `"没有符合条件的候选商品"` - 筛选后无候选商品

## 性能指标

- **响应时间**: 平均 < 2秒
- **并发支持**: 支持多用户同时请求
- **准确率**: 基于训练好的CIR模型
- **召回率**: 支持场景和类别筛选

## 部署检查清单

### 前置条件
- [ ] 训练好的模型文件 (`cir_best_model.pth`)
- [ ] 预计算的embedding文件 (`.pkl`文件)
- [ ] FAISS索引文件 (`rec_index.faiss`)
- [ ] 商品元数据文件 (`metadata.json`)

### 部署步骤
1. [ ] 准备服务器环境
2. [ ] 上传模型和数据文件
3. [ ] 构建Docker镜像
4. [ ] 启动服务
5. [ ] 测试API接口
6. [ ] 配置监控和日志

### 测试验证
- [ ] 健康检查接口正常
- [ ] 推荐接口返回正确结果
- [ ] 错误处理正常
- [ ] 性能满足要求

## 联系支持

如有问题，请联系算法端开发团队。 