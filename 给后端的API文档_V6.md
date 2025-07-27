# 算法端API文档 V6

## 重要说明

**算法端V6.1版本：通过调用getSubCategory和getProductByTypeId API实现上下装筛选和FAISS检索**

## 核心特点

- ✅ **子类筛选**：通过子类API获取互补类别的商品
- ✅ **场景过滤**：支持按场景筛选商品
- ✅ **FAISS检索**：高效相似度检索
- ✅ **Top-1返回**：固定返回最佳匹配单品
- ✅ **实时API调用**：动态获取商品数据

## 服务器信息

- **算法端服务器IP**: `你的服务器IP`
- **算法端API地址**: `http://你的服务器IP:5000`
- **后端API地址**: `http://8.134.151.199:8800`
- **端口**: 5000

## 核心接口

### 1. 从后端API构建FAISS索引（预处理阶段）

**接口**: `POST /api/build_faiss_index_from_api`

**说明**: 从后端API获取商品数据，算法端生成FAISS索引

**请求体**:

```json
{
  "scene": "casual",  // 可选，场景筛选。如果为null则获取所有场景
  "save_path": "/root/fashion-ai-project/faiss_index.faiss"  // 可选，保存路径
}
```

**响应体**:

```json
{
  "success": true,
  "message": "FAISS索引构建成功",
  "data": {
    "scene": "casual",
    "valid_products": 998,
    "save_path": "/root/fashion-ai-project/faiss_index.faiss",
    "metadata_path": "/root/fashion-ai-project/faiss_index_metadata.json"
  }
}
```

### 2. 推荐最佳单品（在线检索阶段）

**接口**: `POST /api/recommend_best_item`

**说明**: 通过FAISS全库检索返回top-1最佳单品

**请求体**:

```json
{
  "product_id": 123,        // 必需：用户选择的商品ID
  "scene": "casual"         // 可选：目标场景（casual/sport/work/party）
}
```

**响应体**:

```json
{
  "success": true,
  "message": "推荐成功",
  "data": {
    "product_id": 456,           // 推荐商品ID
    "score": 0.92,               // 相似度分数（0-1）
    "scene": "casual"            // 场景标签
  }
}
```

## 完整业务流程

### 阶段一：预处理阶段（构建FAISS索引）

```
1. 算法端调用后端API获取商品数据：
   - GET /mall/getProductByScene/{scene} - 获取特定场景商品
   - GET /mall/getProductDetail/{productId} - 获取商品详情

2. 算法端处理：
   - 下载所有商品图片
   - 生成每个商品的embedding（图片+description）
   - 构建FAISS索引
   - 保存索引文件和元数据

3. 返回构建结果
```

### 阶段二：在线检索阶段

```
1. 用户选择商品（提供product_id）
2. 算法端调用后端API获取用户商品信息：
   - GET /mall/getProductDetail/{productId}
   - 从商品详情中提取categoryId

3. 算法端确定互补类别：
   - 用户商品是上装(categoryId=23) → 获取下装(categoryId=24)
   - 用户商品是下装(categoryId=24) → 获取上装(categoryId=23)

4. 算法端获取互补类别的子类ID：
   - GET /mall/getSubCategory?parentId={互补父类ID}

5. 算法端获取互补类别的商品ID列表：
   - GET /mall/getProductByTypeId/{typeId} - 获取每个子类的商品ID

6. 算法端处理：
   - 生成用户商品embedding
   - 先筛选互补类型：筛选出所有上装or下装的商品
   - 为互补商品生成embedding并创建临时FAISS索引
   - 在互补商品中进行FAISS检索
   - 在检索到的互补商品中取最相似且场景符合的单品
   - 返回最佳匹配单品（top-1）

7. 后端根据返回的商品ID查询数据库
8. 返回完整的商品信息给前端
```

## 测试命令

### 1. 健康检查

```bash
curl http://你的服务器IP:5000/health
```

### 2. 从API构建FAISS索引

```bash
# 构建特定场景的索引
curl -X POST http://你的服务器IP:5000/api/build_faiss_index_from_api \
  -H "Content-Type: application/json" \
  -d '{
    "scene": "casual",
    "save_path": "/root/fashion-ai-project/faiss_index.faiss"
  }'

# 构建所有场景的索引
curl -X POST http://你的服务器IP:5000/api/build_faiss_index_from_api \
  -H "Content-Type: application/json" \
  -d '{
    "scene": null,
    "save_path": "/root/fashion-ai-project/faiss_index.faiss"
  }'
```

### 3. 测试推荐

```bash
curl -X POST http://你的服务器IP:5000/test \
  -H "Content-Type: application/json"
```

### 4. 实际推荐

```bash
curl -X POST http://你的服务器IP:5000/api/recommend_best_item \
  -H "Content-Type: application/json" \
  -d '{
    "product_id": 123,
    "scene": "casual"
  }'
```

**响应示例**:

```json
{
  "success": true,
  "message": "推荐成功",
  "data": {
    "product_id": 456
  }
}
```

## 数据格式要求

### 后端API响应格式

#### 商品详情接口 `/mall/getProductDetail/{productId}`

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "id": 123,
    "productName": "黑色牛仔裤",
    "description": "经典黑色牛仔裤，修身版型，适合休闲和通勤场合",
    "brand": "Nike",
    "price": 199.00,
    "scene": "casual",
    "createTime": "2024-01-01T00:00:00Z",
    "imageUrl": ["https://example.com/images/jeans.jpg"],
    "categoryId": 101,  // 新增：商品所属子类ID
    "variations": [...]
  }
}
```

#### 子类信息接口 `/mall/getSubCategory`

```json
{
  "code": 0,
  "message": "success",
  "data": [
    {
      "id": 101,
      "categoryName": "T恤",
      "parentId": 23,
      "sortOrder": 1
    },
    {
      "id": 102,
      "categoryName": "衬衫",
      "parentId": 23,
      "sortOrder": 2
    }
  ]
}
```

#### 子类商品接口 `/mall/getProductByTypeId/{typeId}`

```json
{
  "code": 0,
  "message": "success",
  "data": [
    {
      "id": 123,
      "productName": "黑色T恤",
      "description": "经典黑色T恤，修身版型，适合休闲场合",
      "imageGif": "https://example.com/images/tshirt.gif",
      "brand": "Nike",
      "price": 99.00,
      "scene": "casual",
      "createTime": "2024-01-01T00:00:00Z"
    }
  ]
}
```

#### 场景商品列表接口 `/mall/getProductByScene/{scene}`

```json
{
  "code": 0,
  "message": "success",
  "data": [
    {
      "id": 123,
      "productName": "黑色牛仔裤",
      "description": "经典黑色牛仔裤，修身版型，适合休闲和通勤场合",
      "imageGif": "https://example.com/images/jeans.gif",
      "brand": "Nike",
      "price": 199.00,
      "scene": "casual",
      "createTime": "2024-01-01T00:00:00Z"
    }
  ]
}
```

## 环境变量配置

```bash
# 模型配置
export MODEL_PATH="/root/fashion-ai-project/models/cir_best_model.pth"
export MODEL_TYPE="clip"

# FAISS索引配置
export FAISS_INDEX_PATH="/root/fashion-ai-project/faiss_index.faiss"

# 后端API配置（HTTPS）
export BACKEND_API_URL="https://m1.apifoxmock.com/m1/6328147-0-default"

# SSL证书验证配置（可选）
export SSL_VERIFY="true"  # true: 验证SSL证书, false: 跳过验证
```

**注意**：

- 代码已支持HTTPS请求，包含SSL证书验证处理
- 如果遇到SSL证书问题，可以设置 `SSL_VERIFY=false` 跳过验证
- 生产环境建议启用SSL证书验证以确保安全性

# 服务器配置

export HOST="0.0.0.0"
export PORT="5000"
export DEBUG="False"

```

## 优化特点

### 1. 预计算FAISS索引 + 互补筛选

- 预处理阶段：预计算所有商品的FAISS索引
- 在线检索：先筛选互补类型，再创建临时FAISS索引
- 精确检索：只在互补类别的商品中进行相似度计算
- 场景匹配：在互补商品候选中选择最相似且场景符合的单品
- 简化返回：只返回目标互补商品的ID

### 2. HTTP API集成

- 通过标准HTTP API访问后端数据
- 支持实时数据获取
- 避免直接数据库连接

### 3. 数据一致性

- 算法端实时获取最新商品信息
- 确保推荐结果基于最新数据
- 支持商品信息的动态更新

### 4. 高效检索

- 先筛选互补类型，再创建临时FAISS索引
- 只在互补商品中进行相似度计算
- 在互补商品候选中选择最相似且场景符合的单品
- 毫秒级响应速度
- 优化的内存使用

### 5. 容错机制

- API调用失败时回退到本地元数据
- 支持网络异常处理
- 提供详细的错误信息

## 性能优势

1. **预计算索引**: 预处理阶段预计算所有商品的FAISS索引
2. **互补筛选**: 先筛选出所有上装or下装的商品
3. **精确检索**: 只在互补商品中进行FAISS相似度计算
4. **场景匹配**: 在互补商品候选中选择最相似且场景符合的单品
5. **简化返回**: 只返回目标互补商品的ID，减少数据传输

## 部署建议

### 1. 索引构建频率

- **首次部署**: 必须构建一次索引
- **数据更新**: 建议每周或每月重建一次
- **紧急更新**: 可随时重建索引

### 2. 性能优化

- 使用GPU加速embedding生成
- 选择合适的FAISS索引类型
- 监控内存和磁盘使用情况

### 3. 监控维护

- 定期检查索引文件状态
- 监控API响应时间
- 备份索引文件

## 错误处理

### 常见错误

- `400`: 请求参数错误
- `404`: 没有找到合适的推荐单品
- `500`: 服务器内部错误

### 错误响应格式

```json
{
  "success": false,
  "message": "错误信息",
  "data": null
}
```

## 联系信息

如有问题，请联系算法端开发团队。
