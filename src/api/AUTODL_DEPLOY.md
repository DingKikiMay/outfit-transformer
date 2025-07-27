# AutoDL 4090 服务器部署指南

## 概述

本指南详细说明如何在AutoDL 4090服务器上部署时尚推荐API V6。

## 修改说明

### 1. 路径配置优化

- **原因**: 原代码使用硬编码的绝对路径 `/root/fashion-ai-project/`，不适用于所有AutoDL环境
- **修改**: 使用相对路径和动态路径检测，提高兼容性

### 2. 分类推荐功能

- **原因**: 实现更合理的推荐策略：上装推荐下装，下装推荐上装
- **修改**: 
  - 按分类分别构建FAISS索引（`faiss_index_tops.faiss` 和 `faiss_index_bottoms.faiss`）
  - 推荐时根据用户商品分类加载对应的互补分类索引
  - 提高推荐准确性和效率

### 3. 服务器配置优化

- **原因**: 生产环境需要更稳定的配置
- **修改**: 添加环境变量控制、线程支持、生产环境配置

### 4. 错误处理增强

- **原因**: 服务器环境需要更好的错误处理和日志记录
- **修改**: 增加详细的日志输出和异常处理

## 部署步骤

### 1. 环境准备

```bash
# 确保在项目根目录
cd /path/to/outfit-transformer-api

# 检查Python环境
python --version
pip --version

# 安装依赖（如果还没安装）
pip install -r src/api/requirements.txt
```

### 2. 配置环境变量

```bash
# 设置环境变量
export ENVIRONMENT=production
export HOST=0.0.0.0
export PORT=5000
export DEBUG=False
export LOG_LEVEL=INFO

# 可选：自定义路径
export MODEL_PATH=/path/to/your/model.pth
export FAISS_INDEX_PATH=/path/to/your/index.faiss
export BACKEND_API_URL=https://your-backend-api.com
```

### 3. 启动服务器

```bash
# 方法1：使用启动脚本
chmod +x src/api/start_server.sh
./src/api/start_server.sh

# 方法2：直接启动
cd src/api
python app_v6.py

# 方法3：后台运行
nohup python app_v6.py > api.log 2>&1 &
```

### 4. 测试API

```bash
# 本地测试
python test_api.py

# 指定服务器地址测试
python test_api.py http://your-server-ip:5000
```

## 常见问题

### 1. 端口被占用

```bash
# 查看端口占用
netstat -tlnp | grep 5000

# 杀死进程
kill -9 <PID>

# 或使用其他端口
export PORT=5001
```

### 2. 模型文件不存在

```bash
# 检查模型文件
ls -la models/

# 设置正确的模型路径
export MODEL_PATH=/correct/path/to/model.pth
```

### 3. 权限问题

```bash
# 确保脚本有执行权限
chmod +x start_server.sh

# 确保目录有写权限
chmod 755 models/
chmod 755 logs/
```

## 监控和日志

### 1. 查看日志

```bash
# 实时查看日志
tail -f api.log

# 查看错误日志
grep ERROR api.log
```

### 2. 监控服务状态

```bash
# 检查进程
ps aux | grep app_v6.py

# 检查端口
netstat -tlnp | grep 5000
```

## 性能优化建议

### 1. GPU使用

- 确保CUDA环境正确配置
- 检查GPU内存使用情况

### 2. 内存优化

- 监控内存使用
- 适当调整批处理大小

### 3. 网络优化

- 使用内网地址访问
- 配置防火墙规则

## 安全注意事项

### 1. 生产环境

- 关闭DEBUG模式
- 配置防火墙
- 使用HTTPS（如需要）

### 2. 访问控制

- 限制访问IP
- 添加认证机制
- 监控异常访问

## 故障排除

### 1. 启动失败

```bash
# 检查依赖
pip list | grep -E "(flask|torch|faiss)"

# 检查Python路径
python -c "import sys; print(sys.path)"
```

### 2. API调用失败

```bash
# 检查服务状态
curl http://localhost:5000/health

# 查看详细错误
tail -f api.log
```

### 3. 模型加载失败

```bash
# 检查模型文件
file models/cir_best_model.pth

# 检查CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

## 联系支持

如遇到问题，请提供以下信息：

- 错误日志
- 环境配置
- 复现步骤
