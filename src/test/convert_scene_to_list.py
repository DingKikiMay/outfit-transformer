#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将test.json中的场景标签从字符串转换为列表形式
"""

import json
import os

def convert_scene_to_list(input_file, output_file):
    """将场景标签从字符串转换为列表"""
    
    # 读取原始文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"处理 {len(data)} 个商品...")
    
    # 转换场景标签
    converted_count = 0
    for item in data:
        if '场景' in item:
            scene_value = item['场景']
            
            # 如果场景是字符串，转换为列表
            if isinstance(scene_value, str):
                if scene_value.strip():  # 非空字符串
                    item['场景'] = [scene_value]
                else:  # 空字符串
                    item['场景'] = ['通用']
                converted_count += 1
            # 如果已经是列表，保持不变
            elif isinstance(scene_value, list):
                if not scene_value:  # 空列表
                    item['场景'] = ['通用']
                    converted_count += 1
                # 非空列表保持不变
            else:
                # 其他类型（如None），设置为默认值
                item['场景'] = ['通用']
                converted_count += 1
    
    print(f"转换了 {converted_count} 个场景标签")
    
    # 保存转换后的文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    print(f"转换完成！结果保存到: {output_file}")
    
    # 显示一些示例
    print("\n转换示例:")
    for i, item in enumerate(data[:5]):
        print(f"商品 {i+1}: {item.get('名称', 'N/A')[:30]}... -> 场景: {item.get('场景', 'N/A')}")

def main():
    input_file = "src/test/test_data/test.json"
    output_file = "src/test/test_data/test_converted.json"
    
    if not os.path.exists(input_file):
        print(f"错误：找不到输入文件 {input_file}")
        return
    
    try:
        convert_scene_to_list(input_file, output_file)
        
        # 询问是否要替换原文件
        response = input("\n是否要替换原文件？(y/n): ").lower().strip()
        if response == 'y':
            # 备份原文件
            backup_file = input_file + ".backup"
            os.rename(input_file, backup_file)
            print(f"原文件已备份为: {backup_file}")
            
            # 重命名转换后的文件
            os.rename(output_file, input_file)
            print(f"转换后的文件已替换原文件: {input_file}")
        else:
            print(f"转换后的文件保存在: {output_file}")
            
    except Exception as e:
        print(f"转换失败: {e}")

if __name__ == "__main__":
    main() 