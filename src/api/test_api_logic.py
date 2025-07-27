#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修改后的API逻辑
验证按照正确的API流程：获取子类 -> 获取商品 -> 构建索引
"""

import requests
import json
import time
from typing import Dict, Any

class APILogicTester:
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_subcategory_api(self):
        """测试获取子类别API"""
        print("=== 测试获取子类别API ===")
        
        # 测试获取上装子类
        print("\n1. 测试获取上装子类 (parentId=23)")
        try:
            response = self.session.get(
                f"{self.base_url}/api/test_subcategory",
                params={'parentId': 23}
            )
            print(f"状态码: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"响应: {json.dumps(data, indent=2, ensure_ascii=False)}")
            else:
                print(f"错误: {response.text}")
        except Exception as e:
            print(f"异常: {e}")
        
        # 测试获取下装子类
        print("\n2. 测试获取下装子类 (parentId=24)")
        try:
            response = self.session.get(
                f"{self.base_url}/api/test_subcategory",
                params={'parentId': 24}
            )
            print(f"状态码: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"响应: {json.dumps(data, indent=2, ensure_ascii=False)}")
            else:
                print(f"错误: {response.text}")
        except Exception as e:
            print(f"异常: {e}")
    
    def test_product_api(self):
        """测试获取商品API"""
        print("\n=== 测试获取商品API ===")
        
        # 假设有一个子类ID为91（从上一步测试中获取）
        subcategory_id = 91
        print(f"\n测试获取子类{subcategory_id}的商品")
        
        try:
            response = self.session.get(
                f"{self.base_url}/api/test_products",
                params={'typeId': subcategory_id}
            )
            print(f"状态码: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"响应: {json.dumps(data, indent=2, ensure_ascii=False)}")
            else:
                print(f"错误: {response.text}")
        except Exception as e:
            print(f"异常: {e}")
    
    def test_build_index_with_new_logic(self):
        """测试使用新逻辑构建索引"""
        print("\n=== 测试构建索引（新逻辑） ===")
        
        try:
            payload = {}
            response = self.session.post(
                f"{self.base_url}/api/build_faiss_index_from_api",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            print(f"状态码: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"响应: {json.dumps(data, indent=2, ensure_ascii=False)}")
                return True
            else:
                print(f"错误: {response.text}")
                return False
        except Exception as e:
            print(f"异常: {e}")
            return False
    
    def test_recommendation_with_new_logic(self):
        """测试使用新逻辑进行推荐"""
        print("\n=== 测试推荐（新逻辑） ===")
        
        # 测试上装推荐下装
        print("\n1. 测试上装推荐下装")
        try:
            payload = {
                "product_id": 74,  # 假设这是一个上装商品ID
                "scene": "casual"
            }
            response = self.session.post(
                f"{self.base_url}/api/recommend_best_item",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            print(f"状态码: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"响应: {json.dumps(data, indent=2, ensure_ascii=False)}")
            else:
                print(f"错误: {response.text}")
        except Exception as e:
            print(f"异常: {e}")
        
        # 测试下装推荐上装
        print("\n2. 测试下装推荐上装")
        try:
            payload = {
                "product_id": 20,  # 假设这是一个下装商品ID
                "scene": "casual"
            }
            response = self.session.post(
                f"{self.base_url}/api/recommend_best_item",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            print(f"状态码: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"响应: {json.dumps(data, indent=2, ensure_ascii=False)}")
            else:
                print(f"错误: {response.text}")
        except Exception as e:
            print(f"异常: {e}")
    
    def run_all_tests(self):
        """运行所有测试"""
        print("=== 开始API逻辑测试 ===")
        print(f"测试地址: {self.base_url}")
        
        # 1. 测试子类别API
        self.test_subcategory_api()
        
        # 2. 测试商品API
        self.test_product_api()
        
        # 3. 测试构建索引
        print("\n" + "="*50)
        if self.test_build_index_with_new_logic():
            time.sleep(2)
            
            # 4. 测试推荐
            self.test_recommendation_with_new_logic()
        
        print("\n=== 测试完成 ===")

if __name__ == '__main__':
    # 可以通过命令行参数指定服务器地址
    import sys
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000"
    
    tester = APILogicTester(base_url)
    tester.run_all_tests() 