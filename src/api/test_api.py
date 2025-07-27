#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API测试脚本 - 用于测试部署后的API功能
"""

import requests
import json
import time
from typing import Dict, Any

class APITester:
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_health(self) -> bool:
        """测试健康检查接口"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            print(f"健康检查: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"API状态: {data.get('status')}")
                print(f"消息: {data.get('message')}")
                return True
            else:
                print(f"健康检查失败: {response.text}")
                return False
        except Exception as e:
            print(f"健康检查异常: {e}")
            return False
    
    def test_api_info(self) -> bool:
        """测试API信息接口"""
        try:
            response = self.session.get(f"{self.base_url}/api_info")
            print(f"API信息: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"API信息: {json.dumps(data, indent=2, ensure_ascii=False)}")
                return True
            else:
                print(f"API信息获取失败: {response.text}")
                return False
        except Exception as e:
            print(f"API信息异常: {e}")
            return False
    
    def test_recommendation(self, product_id: int = 1, scene: str = "casual") -> bool:
        """测试推荐接口"""
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
            print(f"推荐测试: {response.status_code}")
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
    
    def test_build_index(self, scene: str = None) -> bool:
        """测试构建索引接口"""
        try:
            payload = {
                "scene": scene
            }
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
    
    def run_all_tests(self):
        """运行所有测试"""
        print("=== 开始API测试 ===")
        print(f"测试地址: {self.base_url}")
        
        tests = [
            ("健康检查", self.test_health),
            ("API信息", self.test_api_info),
            ("推荐测试", lambda: self.test_recommendation(1, "casual")),
            ("构建索引", lambda: self.test_build_index("casual")),
        ]
        
        results = []
        for test_name, test_func in tests:
            print(f"\n--- {test_name} ---")
            try:
                result = test_func()
                results.append((test_name, result))
                time.sleep(1)  # 避免请求过快
            except Exception as e:
                print(f"{test_name} 测试异常: {e}")
                results.append((test_name, False))
        
        print("\n=== 测试结果汇总 ===")
        for test_name, result in results:
            status = "✓ 通过" if result else "✗ 失败"
            print(f"{test_name}: {status}")
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        print(f"\n总体结果: {passed}/{total} 测试通过")

if __name__ == '__main__':
    # 可以通过命令行参数指定服务器地址
    import sys
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000"
    
    tester = APITester(base_url)
    tester.run_all_tests() 