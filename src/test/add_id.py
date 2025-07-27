#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为test.json的每一项添加从1开始的id
"""

import json
import os
from pathlib import Path

def add_ids_to_test_json():
    """为test.json的每一项添加id"""
    
    # 文件路径
    test_file = Path(__file__).parent / "test_data" / "test.json"
    output_file = Path(__file__).parent / "test_data" / "result.json"
    
    print(f"正在读取文件: {test_file}")
    
    # 检查文件是否存在
    if not test_file.exists():
        print(f"错误：文件不存在 {test_file}")
        return False
    
    try:
        # 读取原始数据
        with open(test_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"成功读取 {len(data)} 条商品数据")
        
        # 为每个商品添加item_id，从2开始
        for i, item in enumerate(data, start=2):
            item['item_id'] = i
        
        print(f"已为 {len(data)} 条商品添加item_id（从2开始）")
        
        # 保存到新文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        print(f"结果已保存到: {output_file}")
        
        # 显示前几条数据作为示例
        print("\n前5条数据示例:")
        for i in range(min(5, len(data))):
            item = data[i]
            print(f"item_id: {item['item_id']}, 名称: {item['名称'][:30]}...")
        
        # 显示最后几条数据作为示例
        print("\n最后5条数据示例:")
        for i in range(max(0, len(data)-5), len(data)):
            item = data[i]
            print(f"item_id: {item['item_id']}, 名称: {item['名称'][:30]}...")
        
        return True
        
    except Exception as e:
        print(f"处理文件时出错: {e}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("🆔 为test.json添加item_id")
    print("=" * 60)
    
    success = add_ids_to_test_json()
    
    if success:
        print("=" * 60)
        print("✅ item_id添加完成！")
        print("=" * 60)
    else:
        print("=" * 60)
        print("❌ item_id添加失败！")
        print("=" * 60)

if __name__ == "__main__":
    main() 