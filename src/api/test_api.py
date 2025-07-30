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
    def __init__(self, base_url: str = "http://localhost:6006"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_build_faiss_index(self, scene: str = None) -> bool:
        """测试构建FAISS索引接口"""
        try:
            print("=" * 60)
            print("测试构建FAISS索引接口")
            print("=" * 60)
            
            data = {}
            if scene:
                data['scene'] = scene
            
            response = self.session.post(
                f"{self.base_url}/api/build_faiss_index_from_api",
                json=data,
                headers={'Content-Type': 'application/json'}
            )
            
            print(f"状态码: {response.status_code}")
            print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
            
            if response.status_code == 200:
                result = response.json()
                if result.get('code') == 200 and result.get('success'):
                    print("构建FAISS索引成功")
                    return True
                else:
                    print("构建FAISS索引失败")
                    return False
            else:
                print("构建FAISS索引失败")
                return False
                
        except Exception as e:
            print(f"构建FAISS索引异常: {e}")
            return False
    
    def test_recommendation(self, product_id: int = 1, scene: str = "casual") -> bool:
        """测试推荐接口"""
        try:
            print("=" * 60)
            print("测试推荐接口")
            print("=" * 60)
            
            data = {
                'product_id': product_id,
                'scene': scene
            }
            
            response = self.session.post(
                f"{self.base_url}/api/recommend_best_item",
                json=data,
                headers={'Content-Type': 'application/json'}
            )
            
            print(f"状态码: {response.status_code}")
            print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
            
            if response.status_code == 200:
                result = response.json()
                if result.get('code') == 200 and result.get('success'):
                    print("推荐成功")
                    return True
                else:
                    print("推荐失败")
                    return False
            else:
                print("推荐失败")
                return False
                
        except Exception as e:
            print(f"推荐异常: {e}")
            return False
    
    def run_full_test(self):
        """运行完整测试"""
        print("开始API测试")
        print(f"测试地址: {self.base_url}")
        print("=" * 60)
        
        # 测试构建FAISS索引
        print("\n1. 测试构建FAISS索引（所有场景）")
        build_success = self.test_build_faiss_index()
        
        if not build_success:
            print("FAISS索引构建失败，跳过推荐测试")
            return False
        
        # 等待一下，确保索引构建完成
        print("等待索引构建完成...")
        time.sleep(2)
        
        # 测试推荐接口
        print("\n2. 测试推荐接口")
        recommend_success = self.test_recommendation(product_id=1, scene="casual")
        
        # 测试不同场景
        print("\n3. 测试不同场景推荐")
        recommend_success2 = self.test_recommendation(product_id=2, scene="sports")
        
        # 总结
        print("\n" + "=" * 60)
        print("测试结果总结")
        print("=" * 60)
        print(f"构建FAISS索引: {'成功' if build_success else '失败'}")
        print(f"推荐接口(casual): {'成功' if recommend_success else '失败'}")
        print(f"推荐接口(sports): {'成功' if recommend_success2 else '失败'}")
        
        overall_success = build_success and recommend_success and recommend_success2
        print(f"\n总体结果: {'全部通过' if overall_success else '部分失败'}")
        
        return overall_success


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='API测试脚本')
    parser.add_argument('--url', type=str, default='http://localhost:6006', 
                       help='API服务器地址')
    parser.add_argument('--product-id', type=int, default=1,
                       help='测试用的商品ID')
    parser.add_argument('--scene', type=str, default='casual',
                       help='测试用的场景')
    
    args = parser.parse_args()
    
    # 创建测试器
    tester = APITester(args.url)
    
    # 运行测试
    success = tester.run_full_test()
    
    # 退出码
    exit(0 if success else 1)


if __name__ == '__main__':
    main() 