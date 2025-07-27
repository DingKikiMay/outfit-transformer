#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的互补单品推荐演示运行脚本
按顺序执行：embedding生成 → 索引构建 → Gradio演示
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def run_command(cmd, description):
    """运行命令并显示结果"""
    print(f"\n{'='*60}")
    print(f"步骤: {description}")
    print(f"{'='*60}")
    print(f"执行命令: {cmd}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✅ 执行成功")
        if result.stdout:
            print("输出:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 执行失败: {e}")
        if e.stdout:
            print("标准输出:")
            print(e.stdout)
        if e.stderr:
            print("错误输出:")
            print(e.stderr)
        return False

def check_file_exists(file_path, description):
    """检查文件是否存在"""
    if os.path.exists(file_path):
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description}不存在: {file_path}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="完整的互补单品推荐演示")
    parser.add_argument('--model_type', type=str, choices=['original', 'clip'],
                        default='clip', help="模型类型")
    parser.add_argument('--test_data_dir', type=str, 
                        default='./src/test/test_data', help="测试数据目录")
    parser.add_argument('--checkpoint', type=str, 
                        default='./best_path/complementary_clip_cir_experiment_001_best_model.pth', help="模型检查点路径")
    parser.add_argument('--batch_size', type=int, default=32, help="批处理大小")
    parser.add_argument('--d_embed', type=int, default=512, help="embedding维度")
    parser.add_argument('--port', type=int, default=8080, help="Gradio服务器端口")
    parser.add_argument('--host', type=str, default="0.0.0.0", help="Gradio服务器主机")
    parser.add_argument('--share', action='store_true', help="是否分享链接")
    parser.add_argument('--skip_embedding', action='store_true', help="跳过embedding生成")
    parser.add_argument('--skip_index', action='store_true', help="跳过索引构建")
    parser.add_argument('--demo', action='store_true', help="演示模式，只处理少量数据")
    
    args = parser.parse_args()
    
    print("互补单品推荐演示 - 完整流程")
    print("=" * 60)
    print(f"模型类型: {args.model_type}")
    print(f"测试数据目录: {args.test_data_dir}")
    print(f"模型检查点: {args.checkpoint}")
    print(f"批处理大小: {args.batch_size}")
    print(f"Embedding维度: {args.d_embed}")
    print(f"服务器地址: http://{args.host}:{args.port}")
    print(f"演示模式: {args.demo}")
    print("=" * 60)
    
    # 检查test_data目录
    if not os.path.exists(args.test_data_dir):
        print(f"❌ 错误：测试数据目录不存在: {args.test_data_dir}")
        print("请确保以下文件存在：")
        print(f"  - {args.test_data_dir}/result.json")
        print(f"  - {args.test_data_dir}/images/")
        return False
    
    # 检查模型文件
    if args.checkpoint and not os.path.exists(args.checkpoint):
        print(f"❌ 错误：模型文件不存在: {args.checkpoint}")
        print("请使用 --checkpoint 参数指定正确的模型路径")
        return False
    
    # 步骤1: 生成embedding
    if not args.skip_embedding:
        embedding_dir = os.path.join(args.test_data_dir, 'precomputed_rec_embeddings')
        embedding_file = os.path.join(embedding_dir, 'test_data_embeddings.pkl')
        
        if os.path.exists(embedding_file):
            print(f"✅ 发现已存在的embedding文件: {embedding_file}")
            print("如需重新生成，请删除该文件或使用 --skip_embedding 跳过此步骤")
        else:
            cmd = f"python -m src.demo.1_generate_rec_embeddings_test_data"
            cmd += f" --model_type {args.model_type}"
            cmd += f" --test_data_dir {args.test_data_dir}"
            cmd += f" --batch_size {args.batch_size}"
            if args.checkpoint:
                cmd += f" --checkpoint {args.checkpoint}"
            if args.demo:
                cmd += " --demo"
            
            if not run_command(cmd, "步骤1: 生成推荐embedding"):
                return False
    
    # 步骤2: 构建索引
    if not args.skip_index:
        embedding_dir = os.path.join(args.test_data_dir, 'precomputed_rec_embeddings')
        index_file = os.path.join(embedding_dir, 'test_data_rec_index.faiss')
        
        if os.path.exists(index_file):
            print(f"✅ 发现已存在的索引文件: {index_file}")
            print("如需重新构建，请删除该文件或使用 --skip_index 跳过此步骤")
        else:
            cmd = f"python -m src.demo.2_build_index_test_data"
            cmd += f" --test_data_dir {args.test_data_dir}"
            cmd += f" --d_embed {args.d_embed}"
            
            if not run_command(cmd, "步骤2: 构建FAISS索引"):
                return False
    
    # 步骤3: 启动Gradio演示
    print(f"\n{'='*60}")
    print("步骤3: 启动Gradio演示")
    print(f"{'='*60}")
    
    cmd = f"python -m src.demo.gradio_demo_test_data"
    cmd += f" --model_type {args.model_type}"
    cmd += f" --test_data_dir {args.test_data_dir}"
    cmd += f" --port {args.port}"
    cmd += f" --host {args.host}"
    if args.checkpoint:
        cmd += f" --checkpoint {args.checkpoint}"
    if args.share:
        cmd += " --share"
    
    print(f"启动命令: {cmd}")
    print(f"演示将在以下地址启动: http://{args.host}:{args.port}")
    print("按 Ctrl+C 停止演示")
    print("=" * 60)
    
    try:
        subprocess.run(cmd, shell=True, check=True)
    except KeyboardInterrupt:
        print("\n👋 演示已停止")
    except subprocess.CalledProcessError as e:
        print(f"❌ 启动演示失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ 演示启动失败")
        print("\n可能的解决方案：")
        print("1. 检查模型文件路径是否正确")
        print("2. 检查测试数据目录结构")
        print("3. 确保已安装所有依赖包")
        print("4. 检查端口是否被占用")
        print("5. 使用 --demo 参数进行快速测试")
        sys.exit(1)
    else:
        print("\n✅ 演示流程完成") 