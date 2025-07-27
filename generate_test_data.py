#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成测试数据脚本
从训练数据中提取商品信息，用于API测试
"""

import json
import os
import random
from pathlib import Path

def generate_test_data():
    """生成测试数据"""
    
    # 测试数据
    test_products = [
        {
            "product_id": 1,
            "title": "白色基础T恤",
            "category": "tops",
            "scene": ["casual", "sport"],
            "description": "纯白色基础款T恤，面料舒适，适合日常休闲和运动场合",
            "image_url": "https://example.com/images/white_tshirt.jpg",
            "price": 99.0,
            "brand": "基础品牌"
        },
        {
            "product_id": 2,
            "title": "黑色牛仔裤",
            "category": "bottoms",
            "scene": ["casual", "work"],
            "description": "经典黑色牛仔裤，修身版型，适合休闲和通勤场合",
            "image_url": "https://example.com/images/black_jeans.jpg",
            "price": 299.0,
            "brand": "牛仔品牌"
        },
        {
            "product_id": 3,
            "title": "蓝色衬衫",
            "category": "tops",
            "scene": ["work", "casual"],
            "description": "商务蓝色衬衫，正式场合必备，也可搭配休闲裤",
            "image_url": "https://example.com/images/blue_shirt.jpg",
            "price": 199.0,
            "brand": "商务品牌"
        },
        {
            "product_id": 4,
            "title": "灰色休闲裤",
            "category": "bottoms",
            "scene": ["casual", "work"],
            "description": "舒适灰色休闲裤，面料柔软，适合日常穿着",
            "image_url": "https://example.com/images/gray_pants.jpg",
            "price": 159.0,
            "brand": "休闲品牌"
        },
        {
            "product_id": 5,
            "title": "红色运动衫",
            "category": "tops",
            "scene": ["sport", "casual"],
            "description": "活力红色运动衫，透气面料，适合运动和日常休闲",
            "image_url": "https://example.com/images/red_sportswear.jpg",
            "price": 129.0,
            "brand": "运动品牌"
        },
        {
            "product_id": 6,
            "title": "黑色运动裤",
            "category": "bottoms",
            "scene": ["sport", "casual"],
            "description": "舒适黑色运动裤，弹性面料，适合运动和日常穿着",
            "image_url": "https://example.com/images/black_sport_pants.jpg",
            "price": 89.0,
            "brand": "运动品牌"
        },
        {
            "product_id": 7,
            "title": "白色衬衫",
            "category": "tops",
            "scene": ["work", "party"],
            "description": "经典白色衬衫，正式场合必备，搭配西装或休闲裤都很合适",
            "image_url": "https://example.com/images/white_shirt.jpg",
            "price": 179.0,
            "brand": "商务品牌"
        },
        {
            "product_id": 8,
            "title": "深蓝色西装裤",
            "category": "bottoms",
            "scene": ["work", "party"],
            "description": "正式深蓝色西装裤，商务场合必备，搭配衬衫或西装外套",
            "image_url": "https://example.com/images/dark_blue_suit_pants.jpg",
            "price": 399.0,
            "brand": "商务品牌"
        }
    ]
    
    # 保存测试数据
    with open('test_products.json', 'w', encoding='utf-8') as f:
        json.dump(test_products, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 生成测试数据完成，共{len(test_products)}个商品")
    print("📁 文件保存为: test_products.json")
    
    return test_products

def generate_test_requests():
    """生成测试请求示例"""
    
    test_requests = [
        {
            "name": "上衣推荐下装",
            "request": {
                "product_id": 1,
                "category": "tops",
                "scene": "casual",
                "top_k": 3,
                "candidate_products": [
                    {
                        "product_id": 2,
                        "title": "黑色牛仔裤",
                        "category": "bottoms",
                        "scene": ["casual", "work"],
                        "description": "经典黑色牛仔裤，修身版型，适合休闲和通勤场合",
                        "image_url": "https://example.com/images/black_jeans.jpg",
                        "price": 299.0,
                        "brand": "牛仔品牌"
                    },
                    {
                        "product_id": 4,
                        "title": "灰色休闲裤",
                        "category": "bottoms",
                        "scene": ["casual", "work"],
                        "description": "舒适灰色休闲裤，面料柔软，适合日常穿着",
                        "image_url": "https://example.com/images/gray_pants.jpg",
                        "price": 159.0,
                        "brand": "休闲品牌"
                    },
                    {
                        "product_id": 6,
                        "title": "黑色运动裤",
                        "category": "bottoms",
                        "scene": ["sport", "casual"],
                        "description": "舒适黑色运动裤，弹性面料，适合运动和日常穿着",
                        "image_url": "https://example.com/images/black_sport_pants.jpg",
                        "price": 89.0,
                        "brand": "运动品牌"
                    }
                ]
            }
        },
        {
            "name": "下装推荐上衣",
            "request": {
                "product_id": 2,
                "category": "bottoms",
                "scene": "casual",
                "top_k": 3,
                "candidate_products": [
                    {
                        "product_id": 1,
                        "title": "白色基础T恤",
                        "category": "tops",
                        "scene": ["casual", "sport"],
                        "description": "纯白色基础款T恤，面料舒适，适合日常休闲和运动场合",
                        "image_url": "https://example.com/images/white_tshirt.jpg",
                        "price": 99.0,
                        "brand": "基础品牌"
                    },
                    {
                        "product_id": 3,
                        "title": "蓝色衬衫",
                        "category": "tops",
                        "scene": ["work", "casual"],
                        "description": "商务蓝色衬衫，正式场合必备，也可搭配休闲裤",
                        "image_url": "https://example.com/images/blue_shirt.jpg",
                        "price": 199.0,
                        "brand": "商务品牌"
                    },
                    {
                        "product_id": 5,
                        "title": "红色运动衫",
                        "category": "tops",
                        "scene": ["sport", "casual"],
                        "description": "活力红色运动衫，透气面料，适合运动和日常休闲",
                        "image_url": "https://example.com/images/red_sportswear.jpg",
                        "price": 129.0,
                        "brand": "运动品牌"
                    }
                ]
            }
        }
    ]
    
    # 保存测试请求
    with open('test_requests.json', 'w', encoding='utf-8') as f:
        json.dump(test_requests, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 生成测试请求完成，共{len(test_requests)}个示例")
    print("📁 文件保存为: test_requests.json")
    
    return test_requests

def generate_curl_commands():
    """生成curl测试命令"""
    
    curl_commands = [
        {
            "name": "健康检查",
            "command": "curl http://localhost:5000/health"
        },
        {
            "name": "API信息",
            "command": "curl http://localhost:5000/api_info"
        },
        {
            "name": "测试推荐",
            "command": """curl -X POST http://localhost:5000/test \\
  -H "Content-Type: application/json" """
        },
        {
            "name": "实际推荐测试",
            "command": """curl -X POST http://localhost:5000/api/retrieve_complementary_items \\
  -H "Content-Type: application/json" \\
  -d '{
    "product_id": 1,
    "category": "tops",
    "scene": "casual",
    "top_k": 3,
    "candidate_products": [
      {
        "product_id": 2,
        "title": "黑色牛仔裤",
        "category": "bottoms",
        "scene": ["casual", "work"],
        "description": "经典黑色牛仔裤，修身版型，适合休闲和通勤场合",
        "image_url": "https://example.com/images/black_jeans.jpg",
        "price": 299.0,
        "brand": "牛仔品牌"
      },
      {
        "product_id": 4,
        "title": "灰色休闲裤",
        "category": "bottoms",
        "scene": ["casual", "work"],
        "description": "舒适灰色休闲裤，面料柔软，适合日常穿着",
        "image_url": "https://example.com/images/gray_pants.jpg",
        "price": 159.0,
        "brand": "休闲品牌"
      }
    ]
  }' """
        }
    ]
    
    # 保存curl命令
    with open('curl_commands.txt', 'w', encoding='utf-8') as f:
        for cmd in curl_commands:
            f.write(f"# {cmd['name']}\n")
            f.write(f"{cmd['command']}\n\n")
    
    print(f"✅ 生成curl命令完成，共{len(curl_commands)}个命令")
    print("📁 文件保存为: curl_commands.txt")
    
    return curl_commands

if __name__ == "__main__":
    print("🚀 开始生成测试数据...")
    
    # 生成测试数据
    test_products = generate_test_data()
    
    # 生成测试请求
    test_requests = generate_test_requests()
    
    # 生成curl命令
    curl_commands = generate_curl_commands()
    
    print("\n🎉 所有测试数据生成完成！")
    print("\n📋 生成的文件：")
    print("  - test_products.json: 测试商品数据")
    print("  - test_requests.json: 测试请求示例")
    print("  - curl_commands.txt: curl测试命令")
    
    print("\n💡 使用说明：")
    print("  1. 启动API服务: python src/api/app_v3.py")
    print("  2. 使用curl_commands.txt中的命令测试API")
    print("  3. 将test_requests.json中的示例发给后端参考") 