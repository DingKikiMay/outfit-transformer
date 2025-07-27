#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试商品详情API和分类逻辑
验证直接获取商品详情的功能
"""

import requests
import json
import time
from typing import Dict, Any

class ProductDetailTester:
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_product_detail_api(self, product_id: int = 57):
        """测试商品详情API"""
        print(f"=== 测试商品详情API (商品ID: {product_id}) ===")
        
        try:
            # 直接调用后端API
            response = self.session.get(
                f"https://m1.apifoxmock.com/m1/6328147-0-default/mall/getProductDetail/{product_id}"
            )
            print(f"状态码: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"API响应: {json.dumps(data, indent=2, ensure_ascii=False)}")
                return data
            else:
                print(f"错误: {response.text}")
                return None
        except Exception as e:
            print(f"异常: {e}")
            return None
    
    def test_recommendation_with_product_detail(self, product_id: int = 57):
        """测试使用商品详情API进行推荐"""
        print(f"\n=== 测试推荐功能 (商品ID: {product_id}) ===")
        
        try:
            payload = {
                "product_id": product_id,
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
                print(f"推荐结果: {json.dumps(data, indent=2, ensure_ascii=False)}")
                return data
            else:
                print(f"错误: {response.text}")
                return None
        except Exception as e:
            print(f"异常: {e}")
            return None
    
    def test_multiple_products(self):
        """测试多个商品的推荐"""
        print("\n=== 测试多个商品的推荐 ===")
        
        # 测试不同的商品ID
        test_products = [57, 74, 20]  # 从API响应中获取的商品ID
        
        for product_id in test_products:
            print(f"\n--- 测试商品ID: {product_id} ---")
            
            # 测试推荐
            result = self.test_recommendation_with_product_detail(product_id)
            
            if result and result.get('success'):
                print(f"✅ 商品{product_id}推荐成功")
            else:
                print(f"❌ 商品{product_id}推荐失败")
            
            time.sleep(1)  # 避免请求过快
    
    def test_cache_performance(self):
        """测试缓存性能"""
        print("\n=== 测试缓存性能 ===")
        
        product_id = 57
        
        # 第一次请求（需要搜索分类）
        print("第一次请求（需要搜索分类）:")
        start_time = time.time()
        result1 = self.test_recommendation_with_product_detail(product_id)
        time1 = time.time() - start_time
        print(f"耗时: {time1:.3f}秒")
        
        # 第二次请求（使用缓存）
        print("\n第二次请求（使用缓存）:")
        start_time = time.time()
        result2 = self.test_recommendation_with_product_detail(product_id)
        time2 = time.time() - start_time
        print(f"耗时: {time2:.3f}秒")
        
        # 性能对比
        if time1 > 0 and time2 > 0:
            speedup = time1 / time2
            print(f"\n性能提升: {speedup:.2f}x")
            if speedup > 1.5:
                print("✅ 缓存机制工作正常")
            else:
                print("⚠️ 缓存效果不明显")
    
    def run_all_tests(self):
        """运行所有测试"""
        print("=== 开始商品详情API测试 ===")
        print(f"测试地址: {self.base_url}")
        
        # 1. 测试商品详情API
        product_detail = self.test_product_detail_api()
        
        if product_detail:
            # 2. 测试推荐功能
            self.test_recommendation_with_product_detail()
            
            # 3. 测试多个商品
            self.test_multiple_products()
            
            # 4. 测试缓存性能
            self.test_cache_performance()
        
        print("\n=== 测试完成 ===")

if __name__ == '__main__':
    # 可以通过命令行参数指定服务器地址
    import sys
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000"
    
    tester = ProductDetailTester(base_url)
    tester.run_all_tests() 