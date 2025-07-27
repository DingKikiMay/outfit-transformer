#!/bin/bash

# AutoDL服务器启动脚本

echo "=== 启动时尚推荐API服务器 ==="

# 设置环境变量
export ENVIRONMENT=production
export HOST=0.0.0.0
export PORT=5000
export DEBUG=False
export LOG_LEVEL=INFO

# 可选：设置模型路径（如果模型文件不在默认位置）
# export MODEL_PATH=/path/to/your/model.pth
# export FAISS_INDEX_PATH=/path/to/your/index.faiss

# 可选：设置后端API地址
# export BACKEND_API_URL=https://your-backend-api.com

# 可选：SSL验证设置
# export SSL_VERIFY=false

echo "环境变量设置完成"
echo "HOST: $HOST"
echo "PORT: $PORT"
echo "DEBUG: $DEBUG"
echo "ENVIRONMENT: $ENVIRONMENT"

# 检查Python环境
echo "检查Python环境..."
python --version

# 检查依赖
echo "检查依赖..."
pip list | grep -E "(flask|torch|faiss|requests)"

# 启动服务器
echo "启动服务器..."
cd "$(dirname "$0")"
python app_v6.py 