# -*- coding:utf-8 -*-
"""
API启动脚本
检查环境并启动时尚推荐API服务
"""
import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """检查必要的依赖包"""
    required_packages = [
        'fastapi',
        'uvicorn',
        'python-multipart',
        'requests',
        'pillow',
        'torch',
        'numpy',
        'faiss-cpu'  # 或者 faiss-gpu
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"缺少以下依赖包: {missing_packages}")
        print("正在安装...")
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"✓ {package} 安装成功")
            except subprocess.CalledProcessError:
                print(f"✗ {package} 安装失败")
                return False
    else:
        print("✓ 所有依赖包已安装")
    
    return True

def check_data_files():
    """检查必要的数据文件"""
    required_files = [
        './datasets/polyvore/item_metadata.json',
        './datasets/polyvore/precomputed_rec_embeddings/',
        './checkpoints/best_model.pth'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("缺少以下数据文件:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\n请确保:")
        print("1. 数据集已下载到 ./datasets/polyvore/")
        print("2. 模型已训练并保存到 ./checkpoints/best_model.pth")
        print("3. 预计算的embedding已生成")
        return False
    else:
        print("✓ 所有数据文件存在")
    
    return True

def check_environment():
    """检查运行环境"""
    print("=" * 50)
    print("检查运行环境...")
    print("=" * 50)
    
    # 检查Python版本
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print(f"✗ Python版本过低: {python_version.major}.{python_version.minor}")
        print("需要Python 3.8或更高版本")
        return False
    else:
        print(f"✓ Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 检查依赖包
    if not check_dependencies():
        return False
    
    # 检查数据文件
    if not check_data_files():
        return False
    
    print("=" * 50)
    print("环境检查完成！")
    print("=" * 50)
    return True

def start_api_server():
    """启动API服务器"""
    print("\n" + "=" * 50)
    print("启动时尚推荐API服务器...")
    print("=" * 50)
    
    # 设置环境变量
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # 启动服务器
    try:
        import uvicorn
        from fashion_api import app
        
        print("服务器启动中...")
        print("API文档地址: http://localhost:8000/docs")
        print("健康检查地址: http://localhost:8000/health")
        print("按 Ctrl+C 停止服务器")
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            reload=False,  # 生产环境建议设为False
            log_level="info"
        )
        
    except KeyboardInterrupt:
        print("\n服务器已停止")
    except Exception as e:
        print(f"启动服务器失败: {e}")
        return False
    
    return True

def main():
    """主函数"""
    print("时尚推荐API启动脚本")
    print("=" * 50)
    
    # 检查环境
    if not check_environment():
        print("\n环境检查失败，请解决上述问题后重试")
        return
    
    # 启动服务器
    start_api_server()

if __name__ == "__main__":
    main() 