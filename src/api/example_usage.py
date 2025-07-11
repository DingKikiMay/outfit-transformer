# -*- coding:utf-8 -*-
"""
API使用示例
展示如何调用时尚推荐API的各个接口
"""
import requests
import base64
import json
from PIL import Image
import io

# API基础URL
BASE_URL = "http://localhost:8000"

def image_to_base64(image_path: str) -> str:
    """将图片文件转换为base64编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_health_check():
    """测试健康检查接口"""
    print("=" * 50)
    print("测试健康检查接口")
    print("=" * 50)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")

def test_get_categories():
    """测试获取类别接口"""
    print("\n" + "=" * 50)
    print("测试获取类别接口")
    print("=" * 50)
    
    response = requests.get(f"{BASE_URL}/categories")
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")

def test_get_scenes():
    """测试获取场景接口"""
    print("\n" + "=" * 50)
    print("测试获取场景接口")
    print("=" * 50)
    
    response = requests.get(f"{BASE_URL}/scenes")
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")

def test_fusion_search():
    """测试融合搜索接口"""
    print("\n" + "=" * 50)
    print("测试融合搜索接口")
    print("=" * 50)
    
    # 创建一个示例图片（1x1像素的白色图片）
    img = Image.new('RGB', (224, 224), color='white')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    # 准备请求数据
    files = {
        'image': ('test_image.jpg', img_bytes, 'image/jpeg')
    }
    data = {
        'description': '白色T恤，运动风格',
        'scene_filter': 'casual',
        'top_k': 4
    }
    
    response = requests.post(f"{BASE_URL}/search/fusion", files=files, data=data)
    print(f"状态码: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"搜索成功，返回 {result['total_count']} 个结果")
        for i, item in enumerate(result['results']):
            print(f"  {i+1}. {item['description']} (场景: {item['scene']}, 分数: {item['score']:.3f})")
    else:
        print(f"搜索失败: {response.text}")

def test_complementary_search():
    """测试互补搜索接口"""
    print("\n" + "=" * 50)
    print("测试互补搜索接口")
    print("=" * 50)
    
    # 创建示例用户商品
    user_items = [
        {
            "item_id": "user_item_1",
            "description": "白色T恤",
            "category": "tops",
            "scene": ["casual"],
            "image_base64": None  # 实际使用时可以提供图片
        }
    ]
    
    request_data = {
        "user_items": user_items,
        "scene_filter": "casual",
        "top_k": 4
    }
    
    response = requests.post(
        f"{BASE_URL}/search/complementary",
        json=request_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"状态码: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"搜索成功，返回 {result['total_count']} 个结果")
        for i, item in enumerate(result['results']):
            print(f"  {i+1}. {item['description']} (场景: {item['scene']}, 分数: {item['score']:.3f})")
    else:
        print(f"搜索失败: {response.text}")

def test_compatibility_score():
    """测试兼容性评分接口"""
    print("\n" + "=" * 50)
    print("测试兼容性评分接口")
    print("=" * 50)
    
    # 创建示例搭配
    outfit_items = [
        {
            "item_id": "item_1",
            "description": "白色T恤",
            "category": "tops",
            "scene": ["casual"],
            "image_base64": None
        },
        {
            "item_id": "item_2", 
            "description": "蓝色牛仔裤",
            "category": "bottoms",
            "scene": ["casual"],
            "image_base64": None
        }
    ]
    
    request_data = {
        "outfit_items": outfit_items
    }
    
    response = requests.post(
        f"{BASE_URL}/compatibility/score",
        json=request_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"状态码: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"评分成功: {result['score']:.3f}")
    else:
        print(f"评分失败: {response.text}")

def test_get_item_info():
    """测试获取商品信息接口"""
    print("\n" + "=" * 50)
    print("测试获取商品信息接口")
    print("=" * 50)
    
    # 假设商品ID为0
    item_id = 0
    
    response = requests.get(f"{BASE_URL}/items/{item_id}")
    print(f"状态码: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        item = result['item']
        print(f"商品信息: {item['description']} (类别: {item['category']}, 场景: {item['scene']})")
    else:
        print(f"获取商品信息失败: {response.text}")

def test_get_stats():
    """测试获取统计信息接口"""
    print("\n" + "=" * 50)
    print("测试获取统计信息接口")
    print("=" * 50)
    
    response = requests.get(f"{BASE_URL}/stats")
    print(f"状态码: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        stats = result['stats']
        print(f"总商品数: {stats['total_items']}")
        print(f"类别统计: {stats['categories']}")
        print(f"场景统计: {stats['scenes']}")
    else:
        print(f"获取统计信息失败: {response.text}")

def main():
    """运行所有测试"""
    print("开始测试时尚推荐API...")
    
    try:
        # 测试基础接口
        test_health_check()
        test_get_categories()
        test_get_scenes()
        
        # 测试核心功能接口
        test_fusion_search()
        test_complementary_search()
        test_compatibility_score()
        
        # 测试辅助接口
        test_get_item_info()
        test_get_stats()
        
        print("\n" + "=" * 50)
        print("所有测试完成！")
        print("=" * 50)
        
    except requests.exceptions.ConnectionError:
        print("错误: 无法连接到API服务器，请确保服务器正在运行")
        print("启动命令: python src/api/fashion_api.py")
    except Exception as e:
        print(f"测试过程中出现错误: {e}")

if __name__ == "__main__":
    main() 