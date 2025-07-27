# 时尚单品推荐API接口文档

## 概述

本API提供时尚单品推荐服务，根据用户提供的单品图片和需求，从商品库中推荐最合适的互补单品。

## 基础信息

- **API版本**: 1.0.0
- **协议**: HTTP/HTTPS
- **数据格式**: JSON
- **字符编码**: UTF-8

## 接口列表

### 1. 健康检查接口

**接口地址**: `GET /health`

**功能描述**: 检查API服务状态

**请求参数**: 无

**响应示例**:
```json
{
    "status": "ok",
    "message": "API运行正常",
    "api_info": {
        "api_name": "Fashion Recommendation API",
        "version": "1.0.0",
        "model_type": "clip",
        "model_path": "/path/to/model.pth",
        "device": "cuda",
        "supported_features": [
            "多图片输入",
            "场景指定",
            "描述指定",
            "类别指定",
            "全库检索",
            "相似度排序"
        ]
    }
}
```

### 2. 推荐接口

**接口地址**: `POST /recommend`

**功能描述**: 根据用户输入推荐商品

**请求头**:
```
Content-Type: application/json
```

**请求参数**:

| 参数名 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| user_images | Array[string] | 是 | 用户图片base64编码列表 |
| product_database | Array[Object] | 是 | 商品数据库 |
| target_scene | string | 否 | 目标场景 |
| target_description | string | 否 | 目标描述 |
| target_category | string | 否 | 目标类别 |
| top_k | int | 否 | 返回推荐数量，默认5 |

**user_images格式**:
```json
[
    "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD...",
    "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD..."
]
```

**product_database格式**:
```json
[
    {
        "product_id": "商品ID",
        "title": "商品标题",
        "category": "商品类别",
        "scene": "适用场景",
        "description": "商品描述",
        "image_base64": "商品图片base64",
        "price": 199.0,
        "brand": "品牌",
        "metadata": {}
    }
]
```

**响应格式**:
```json
{
    "status": "success",
    "message": "推荐成功",
    "data": {
        "recommendations": [
            {
                "product_id": "推荐商品ID",
                "score": 0.85,
                "reason": "最匹配的推荐，适合casual场景，风格高度一致",
                "product_info": {
                    "product_id": "商品ID",
                    "title": "商品标题",
                    "category": "商品类别",
                    "scene": "适用场景",
                    "description": "商品描述",
                    "price": 199.0,
                    "brand": "品牌",
                    "metadata": {}
                }
            }
        ],
        "total_count": 5,
        "request_info": {
            "user_images_count": 2,
            "product_database_count": 1000,
            "target_scene": "casual",
            "target_description": "牛仔裤",
            "target_category": "下装",
            "top_k": 5
        }
    }
}
```

### 3. API信息接口

**接口地址**: `GET /api_info`

**功能描述**: 获取API详细信息

**请求参数**: 无

**响应示例**:
```json
{
    "status": "success",
    "data": {
        "api_name": "Fashion Recommendation API",
        "version": "1.0.0",
        "model_type": "clip",
        "model_path": "/path/to/model.pth",
        "device": "cuda",
        "supported_features": [
            "多图片输入",
            "场景指定",
            "描述指定",
            "类别指定",
            "全库检索",
            "相似度排序"
        ]
    }
}
```

### 4. 测试接口

**接口地址**: `POST /test`

**功能描述**: 使用示例数据测试推荐功能

**请求参数**: 无

**响应格式**: 同推荐接口

## 数据格式说明

### 图片格式要求

- **格式**: JPEG, PNG
- **编码**: Base64
- **前缀**: 可选 `data:image/jpeg;base64,` 或 `data:image/png;base64,`
- **大小**: 建议不超过5MB

### 商品类别

支持的商品类别：
- 上衣
- 下装
- 外套
- 鞋子
- 配饰

### 场景类型

支持的场景类型：
- casual (休闲)
- sports (运动)
- party (派对)
- business (商务)
- formal (正式)

## 错误码说明

| 状态码 | 说明 |
|--------|------|
| 200 | 请求成功 |
| 400 | 请求参数错误 |
| 500 | 服务器内部错误 |

**错误响应格式**:
```json
{
    "status": "error",
    "message": "错误描述"
}
```

## 使用示例

### Python示例

```python
import requests
import base64

# 读取图片并转换为base64
def image_to_base64(image_path):
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

# 准备请求数据
user_images = [
    image_to_base64('user_image1.jpg'),
    image_to_base64('user_image2.jpg')
]

product_database = [
    {
        "product_id": "product_001",
        "title": "休闲牛仔裤",
        "category": "下装",
        "scene": "casual",
        "description": "舒适休闲牛仔裤",
        "image_base64": image_to_base64('product1.jpg'),
        "price": 199.0,
        "brand": "测试品牌"
    }
]

request_data = {
    "user_images": user_images,
    "product_database": product_database,
    "target_scene": "casual",
    "target_description": "牛仔裤",
    "target_category": "下装",
    "top_k": 5
}

# 发送请求
response = requests.post(
    'http://localhost:5000/recommend',
    json=request_data,
    headers={'Content-Type': 'application/json'}
)

# 处理响应
if response.status_code == 200:
    result = response.json()
    print("推荐成功:", result['data']['recommendations'])
else:
    print("请求失败:", response.json())
```

### JavaScript示例

```javascript
// 将图片文件转换为base64
function imageToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

// 准备请求数据
async function recommendProducts(userImages, productDatabase) {
    const requestData = {
        user_images: userImages,
        product_database: productDatabase,
        target_scene: "casual",
        target_description: "牛仔裤",
        target_category: "下装",
        top_k: 5
    };

    try {
        const response = await fetch('http://localhost:5000/recommend', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });

        const result = await response.json();
        
        if (result.status === 'success') {
            console.log('推荐结果:', result.data.recommendations);
            return result.data.recommendations;
        } else {
            console.error('推荐失败:', result.message);
            throw new Error(result.message);
        }
    } catch (error) {
        console.error('请求失败:', error);
        throw error;
    }
}
```

## 性能说明

- **响应时间**: 通常在1-5秒内返回结果
- **并发支持**: 支持多并发请求
- **内存使用**: 根据商品库大小动态调整
- **GPU加速**: 自动使用GPU加速（如果可用）

## 部署说明

### 环境要求

- Python 3.8+
- PyTorch 1.8+
- CUDA 11.0+ (可选，用于GPU加速)
- 至少4GB内存

### 启动服务

```bash
# 设置环境变量
export MODEL_PATH=/path/to/your/cir_best_model.pth
export MODEL_TYPE=clip
export HOST=0.0.0.0
export PORT=5000

# 启动服务
python src/api/app.py
```

### Docker部署

```dockerfile
FROM python:3.8-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "src/api/app.py"]
```

## 注意事项

1. **模型路径**: 确保模型文件存在且路径正确
2. **图片格式**: 支持JPEG和PNG格式，建议压缩到合理大小
3. **并发限制**: 建议控制并发请求数量，避免内存溢出
4. **错误处理**: 客户端需要处理网络错误和API错误
5. **数据验证**: 服务端会验证输入数据格式，客户端应确保数据正确

## 联系信息

如有问题，请联系算法团队。 