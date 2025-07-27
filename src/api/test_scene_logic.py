#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试场景逻辑的正确性
验证FAISS索引包含所有场景，推荐时进行场景筛选
"""

import requests
import json
import time
from typing import Dict, Any

class SceneLogicTester:
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_build_index_with_all_scenes(self):
        """测试构建包含所有场景的索引"""
        try:
            # 构建索引时不指定场景，应该包含所有场景
            payload = {}
            response = self.session.post(
                f"{self.base_url}/api/build_faiss_index_from_api",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            print(f"构建索引: {response.status_code}")
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
    
    def test_recommendation_with_different_scenes(self, product_id: int = 1):
        """测试同一商品在不同场景下的推荐结果"""
        scenes = ['casual', 'sports']
        results = {}
        
        for scene in scenes:
            try:
                payload = {
                    "product_id": product_id,
                    "scene": scene
                }
                response = self.session.post(
                    f"{self.base_url}/api/recommend_best_item",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                print(f"场景 {scene} 推荐: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    results[scene] = data
                    print(f"场景 {scene} 推荐结果: {json.dumps(data, indent=2, ensure_ascii=False)}")
                else:
                    print(f"场景 {scene} 推荐失败: {response.text}")
                    results[scene] = None
                
                time.sleep(1)
                
            except Exception as e:
                print(f"场景 {scene} 推荐异常: {e}")
                results[scene] = None
        
        return results
    
    def test_recommendation_without_scene(self, product_id: int = 1):
        """测试不指定场景的推荐结果"""
        try:
            payload = {
                "product_id": product_id
                # 不指定scene
            }
            response = self.session.post(
                f"{self.base_url}/api/recommend_best_item",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            print(f"无场景推荐: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"无场景推荐结果: {json.dumps(data, indent=2, ensure_ascii=False)}")
                return data
            else:
                print(f"无场景推荐失败: {response.text}")
                return None
                
        except Exception as e:
            print(f"无场景推荐异常: {e}")
            return None
    
    def analyze_results(self, scene_results: Dict, no_scene_result: Dict):
        """分析推荐结果"""
        print("\n=== 结果分析 ===")
        
        # 检查是否有推荐结果
        if not scene_results or not any(scene_results.values()):
            print("❌ 没有获得有效的推荐结果")
            return
        
        # 检查不同场景的推荐结果是否不同
        scene_products = {}
        for scene, result in scene_results.items():
            if result and result.get('success'):
                product_id = result.get('data', {}).get('product_id')
                scene_products[scene] = product_id
        
        if len(set(scene_products.values())) > 1:
            print("✅ 不同场景推荐了不同的商品，场景筛选功能正常")
            for scene, product_id in scene_products.items():
                print(f"  场景 {scene}: 商品ID {product_id}")
        else:
            print("⚠️  不同场景推荐了相同的商品")
            print("   可能原因：")
            print("   1. 该场景下没有足够的商品")
            print("   2. 相似度计算优先于场景筛选")
        
        # 检查无场景推荐
        if no_scene_result and no_scene_result.get('success'):
            no_scene_product = no_scene_result.get('data', {}).get('product_id')
            print(f"✅ 无场景推荐: 商品ID {no_scene_product}")
            print("   说明：索引包含所有场景的商品，推荐时没有场景限制")
    
    def run_all_tests(self):
        """运行所有测试"""
        print("=== 开始场景逻辑测试 ===")
        print(f"测试地址: {self.base_url}")
        
        # 1. 构建包含所有场景的索引
        print("\n--- 步骤1: 构建索引 ---")
        if not self.test_build_index_with_all_scenes():
            print("❌ 索引构建失败，无法继续测试")
            return
        
        time.sleep(2)
        
        # 2. 测试不同场景的推荐
        print("\n--- 步骤2: 测试场景推荐 ---")
        scene_results = self.test_recommendation_with_different_scenes()
        
        # 3. 测试无场景推荐
        print("\n--- 步骤3: 测试无场景推荐 ---")
        no_scene_result = self.test_recommendation_without_scene()
        
        # 4. 分析结果
        print("\n--- 步骤4: 分析结果 ---")
        self.analyze_results(scene_results, no_scene_result)
        
        print("\n=== 测试完成 ===")

if __name__ == '__main__':
    # 可以通过命令行参数指定服务器地址
    import sys
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000"
    
    tester = SceneLogicTester(base_url)
    tester.run_all_tests() 