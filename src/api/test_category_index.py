#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试分类索引功能
验证商品ID到分类的映射是否正确
"""

import requests
import json
import time
from typing import Dict, Any

class CategoryIndexTester:
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_category_index_stats(self):
        """测试分类索引统计信息"""
        print("=== 测试分类索引统计信息 ===")
        
        try:
            response = self.session.get(f"{self.base_url}/api/test_category_index")
            print(f"状态码: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"分类索引统计: {json.dumps(data, indent=2, ensure_ascii=False)}")
                return data
            else:
                print(f"错误: {response.text}")
                return None
        except Exception as e:
            print(f"异常: {e}")
            return None
    
    def test_product_category(self, product_id: int):
        """测试特定商品的分类"""
        print(f"\n=== 测试商品分类 (商品ID: {product_id}) ===")
        
        try:
            response = self.session.get(f"{self.base_url}/api/test_category_index?productId={product_id}")
            print(f"状态码: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"商品分类信息: {json.dumps(data, indent=2, ensure_ascii=False)}")
                return data
            else:
                print(f"错误: {response.text}")
                return None
        except Exception as e:
            print(f"异常: {e}")
            return None
    
    def test_recommendation_with_category_index(self, product_id: int):
        """测试使用分类索引进行推荐"""
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
    
    def test_multiple_products_category(self):
        """测试多个商品的分类"""
        print("\n=== 测试多个商品的分类 ===")
        
        # 测试不同的商品ID（这些ID应该存在于分类索引中）
        test_products = [57, 74, 20, 100, 200]  # 示例商品ID
        
        for product_id in test_products:
            print(f"\n--- 测试商品ID: {product_id} ---")
            
            # 测试分类
            category_result = self.test_product_category(product_id)
            
            if category_result and category_result.get('success'):
                category = category_result['data']['category']
                print(f"✅ 商品{product_id}分类: {category}")
                
                # 测试推荐
                recommend_result = self.test_recommendation_with_category_index(product_id)
                
                if recommend_result and recommend_result.get('success'):
                    print(f"✅ 商品{product_id}推荐成功")
                else:
                    print(f"❌ 商品{product_id}推荐失败")
            else:
                print(f"❌ 商品{product_id}不在分类索引中")
            
            time.sleep(0.5)  # 避免请求过快
    
    def test_category_index_performance(self):
        """测试分类索引性能"""
        print("\n=== 测试分类索引性能 ===")
        
        # 测试API信息获取性能
        print("测试API信息获取性能:")
        start_time = time.time()
        response = self.session.get(f"{self.base_url}/api_info")
        time1 = time.time() - start_time
        print(f"API信息获取耗时: {time1:.3f}秒")
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                category_info = data['data'].get('category_index', {})
                print(f"分类索引信息: {json.dumps(category_info, indent=2, ensure_ascii=False)}")
        
        # 测试分类查询性能
        print("\n测试分类查询性能:")
        test_product_id = 57
        
        start_time = time.time()
        category_result = self.test_product_category(test_product_id)
        time2 = time.time() - start_time
        print(f"分类查询耗时: {time2:.3f}秒")
        
        if category_result and category_result.get('success'):
            print("✅ 分类查询性能正常")
        else:
            print("❌ 分类查询失败")
    
    def test_category_index_accuracy(self):
        """测试分类索引准确性"""
        print("\n=== 测试分类索引准确性 ===")
        
        # 测试上装和下装的分类是否正确
        test_cases = [
            # (product_id, expected_category)
            (57, None),  # 未知预期分类
            (74, None),  # 未知预期分类
        ]
        
        for product_id, expected_category in test_cases:
            print(f"\n--- 测试商品ID: {product_id} ---")
            
            category_result = self.test_product_category(product_id)
            
            if category_result and category_result.get('success'):
                actual_category = category_result['data']['category']
                print(f"实际分类: {actual_category}")
                
                if expected_category:
                    if actual_category == expected_category:
                        print(f"✅ 分类正确")
                    else:
                        print(f"❌ 分类错误，期望: {expected_category}, 实际: {actual_category}")
                else:
                    print(f"ℹ️ 分类: {actual_category}")
            else:
                print(f"❌ 无法获取分类信息")
    
    def run_all_tests(self):
        """运行所有测试"""
        print("=== 开始分类索引测试 ===")
        print(f"测试地址: {self.base_url}")
        
        # 1. 测试分类索引统计
        stats = self.test_category_index_stats()
        
        if stats and stats.get('success'):
            print("✅ 分类索引统计正常")
            
            # 2. 测试多个商品分类
            self.test_multiple_products_category()
            
            # 3. 测试性能
            self.test_category_index_performance()
            
            # 4. 测试准确性
            self.test_category_index_accuracy()
        else:
            print("❌ 分类索引统计失败")
        
        print("\n=== 测试完成 ===")

if __name__ == '__main__':
    # 可以通过命令行参数指定服务器地址
    import sys
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000"
    
    tester = CategoryIndexTester(base_url)
    tester.run_all_tests() 