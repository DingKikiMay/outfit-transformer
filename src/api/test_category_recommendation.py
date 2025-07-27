#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分类推荐功能测试脚本
测试上装推荐下装、下装推荐上装的逻辑
"""

import requests
import json
import time
from typing import Dict, Any

class CategoryRecommendationTester:
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_build_category_index(self, scene: str = None) -> bool:
        """测试构建分类索引"""
        try:
            payload = {
                "scene": scene
            }
            response = self.session.post(
                f"{self.base_url}/api/build_faiss_index_from_api",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            print(f"构建分类索引: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"索引构建结果: {json.dumps(data, indent=2, ensure_ascii=False)}")
                return True
            else:
                print(f"索引构建失败: {response.text}")
                return False
        except Exception as e:
            print(f"索引构建异常: {e}")
            return False
    
    def test_tops_recommendation(self, tops_product_id: int = 1, scene: str = "casual") -> bool:
        """测试上装推荐下装"""
        try:
            payload = {
                "product_id": tops_product_id,
                "scene": scene
            }
            response = self.session.post(
                f"{self.base_url}/api/recommend_best_item",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            print(f"上装推荐下装测试: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"推荐结果: {json.dumps(data, indent=2, ensure_ascii=False)}")
                return True
            else:
                print(f"推荐测试失败: {response.text}")
                return False
        except Exception as e:
            print(f"推荐测试异常: {e}")
            return False
    
    def test_bottoms_recommendation(self, bottoms_product_id: int = 2, scene: str = "casual") -> bool:
        """测试下装推荐上装"""
        try:
            payload = {
                "product_id": bottoms_product_id,
                "scene": scene
            }
            response = self.session.post(
                f"{self.base_url}/api/recommend_best_item",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            print(f"下装推荐上装测试: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"推荐结果: {json.dumps(data, indent=2, ensure_ascii=False)}")
                return True
            else:
                print(f"推荐测试失败: {response.text}")
                return False
        except Exception as e:
            print(f"推荐测试异常: {e}")
            return False
    
    def test_different_scenes(self) -> bool:
        """测试不同场景的推荐"""
        scenes = ['casual', 'sports']
        results = []
        
        for scene in scenes:
            print(f"\n--- 测试场景: {scene} ---")
            
            # 测试上装推荐下装
            tops_result = self.test_tops_recommendation(1, scene)
            results.append((f"上装推荐下装({scene})", tops_result))
            
            time.sleep(1)
            
            # 测试下装推荐上装
            bottoms_result = self.test_bottoms_recommendation(2, scene)
            results.append((f"下装推荐上装({scene})", bottoms_result))
            
            time.sleep(1)
        
        return results
    
    def run_all_tests(self):
        """运行所有测试"""
        print("=== 开始分类推荐功能测试 ===")
        print(f"测试地址: {self.base_url}")
        
        # 1. 构建分类索引
        print("\n--- 步骤1: 构建分类索引 ---")
        if not self.test_build_category_index():
            print("❌ 索引构建失败，无法继续测试")
            return
        
        time.sleep(2)
        
        # 2. 测试不同场景的推荐
        print("\n--- 步骤2: 测试分类推荐 ---")
        scene_results = self.test_different_scenes()
        
        # 3. 输出测试结果
        print("\n=== 测试结果汇总 ===")
        for test_name, result in scene_results:
            status = "✓ 通过" if result else "✗ 失败"
            print(f"{test_name}: {status}")
        
        passed = sum(1 for _, result in scene_results if result)
        total = len(scene_results)
        print(f"\n总体结果: {passed}/{total} 测试通过")

if __name__ == '__main__':
    # 可以通过命令行参数指定服务器地址
    import sys
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000"
    
    tester = CategoryRecommendationTester(base_url)
    tester.run_all_tests() 