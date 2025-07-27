#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行基于test_data的Gradio演示
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from demo.gradio_demo_test_data import run, parse_args

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="运行基于test_data的Gradio演示")
    parser.add_argument('--model_type', type=str, choices=['original', 'clip'],
                        default='clip', help="模型类型")
    parser.add_argument('--test_data_dir', type=str, 
                        default='./src/test/test_data', help="测试数据目录")
    parser.add_argument('--checkpoint', type=str, 
                        default=None, help="模型检查点路径")
    parser.add_argument('--port', type=int, default=7860, help="服务器端口")
    parser.add_argument('--host', type=str, default="0.0.0.0", help="服务器主机")
    parser.add_argument('--share', action='store_true', help="是否分享链接")
    
    args = parser.parse_args()
    
    # 检查test_data目录是否存在
    if not os.path.exists(args.test_data_dir):
        print(f"错误：测试数据目录不存在: {args.test_data_dir}")
        print("请确保以下文件存在：")
        print(f"  - {args.test_data_dir}/result.json")
        print(f"  - {args.test_data_dir}/images/")
        return
    
    # 检查模型文件
    if args.checkpoint and not os.path.exists(args.checkpoint):
        print(f"警告：模型文件不存在: {args.checkpoint}")
        print("请使用 --checkpoint 参数指定正确的模型路径")
        print("或者将模型文件放在默认位置")
    
    print("=" * 60)
    print("🎨 互补单品推荐演示系统")
    print("=" * 60)
    print(f"模型类型: {args.model_type}")
    print(f"测试数据目录: {args.test_data_dir}")
    print(f"模型检查点: {args.checkpoint}")
    print(f"服务器地址: http://{args.host}:{args.port}")
    print("=" * 60)
    
    try:
        # 创建演示
        demo = run(args)
        
        # 启动服务器
        demo.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            debug=True,
            show_error=True
        )
        
    except Exception as e:
        print(f"启动演示失败: {e}")
        print("\n可能的解决方案：")
        print("1. 确保已安装所有依赖: pip install gradio torch pillow")
        print("2. 检查模型文件路径是否正确")
        print("3. 检查测试数据目录结构")
        print("4. 确保有足够的GPU内存（如果使用GPU）")

if __name__ == "__main__":
    main() 