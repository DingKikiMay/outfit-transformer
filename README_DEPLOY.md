# 时尚AI算法端部署指南

## 概述

本指南将帮助你部署时尚AI算法端API服务，支持Docker和云服务器两种部署方式。

## 部署方案

### 方案一：Docker部署（推荐）

#### 前置要求

1. **安装Docker**
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install docker.io docker-compose
   
   # CentOS/RHEL
   sudo yum install docker docker-compose
   
   # macOS
   brew install docker docker-compose
   
   # Windows
   # 下载并安装 Docker Desktop
   ```

2. **启动Docker服务**
   ```bash
   sudo systemctl start docker
   sudo systemctl enable docker
   ```

#### 快速部署

1. **克隆项目**
   ```bash
   git clone <your-repo-url>
   cd outfit-transformer-api
   ```

2. **准备模型文件**
   ```bash
   # 创建模型目录
   mkdir -p models
   
   # 将训练好的模型文件放到models目录
   # 例如：models/cir_best_model.pth
   ```

3. **一键部署**
   ```bash
   # 给部署脚本执行权限
   chmod +x deploy.sh
   
   # 执行部署
   ./deploy.sh
   ```

#### 手动部署

1. **构建镜像**
   ```bash
   docker-compose build
   ```

2. **启动服务**
   ```bash
   docker-compose up -d
   ```

3. **查看日志**
   ```bash
   docker-compose logs -f fashion-ai-api
   ```

4. **停止服务**
   ```bash
   docker-compose down
   ```

### 方案二：云服务器部署

#### 服务器要求

- **CPU**: 4核心以上
- **内存**: 8GB以上
- **GPU**: 可选（推荐NVIDIA GPU）
- **存储**: 50GB以上
- **系统**: Ubuntu 18.04+ / CentOS 7+

#### 部署步骤

1. **安装Python环境**
   ```bash
   # 安装Python 3.9
   sudo apt-get update
   sudo apt-get install python3.9 python3.9-pip python3.9-venv
   
   # 创建虚拟环境
   python3.9 -m venv venv
   source venv/bin/activate
   ```

2. **安装依赖**
   ```bash
   pip install -r src/api/requirements.txt
   ```

3. **配置环境变量**
   ```bash
   export MODEL_PATH=/path/to/your/model.pth
   export MODEL_TYPE=clip
   export HOST=0.0.0.0
   export PORT=5000
   ```

4. **启动服务**
   ```bash
   python src/api/app.py
   ```

5. **使用systemd管理服务**
   ```bash
   # 创建服务文件
   sudo nano /etc/systemd/system/fashion-ai-api.service
   ```

   ```ini
   [Unit]
   Description=Fashion AI API
   After=network.target
   
   [Service]
   Type=simple
   User=your-user
   WorkingDirectory=/path/to/outfit-transformer-api
   Environment=PATH=/path/to/outfit-transformer-api/venv/bin
   Environment=MODEL_PATH=/path/to/your/model.pth
   Environment=MODEL_TYPE=clip
   Environment=HOST=0.0.0.0
   Environment=PORT=5000
   ExecStart=/path/to/outfit-transformer-api/venv/bin/python src/api/app.py
   Restart=always
   
   [Install]
   WantedBy=multi-user.target
   ```

   ```bash
   # 启动服务
   sudo systemctl daemon-reload
   sudo systemctl enable fashion-ai-api
   sudo systemctl start fashion-ai-api
   ```

## 配置说明

### 环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| MODEL_PATH | /app/models/cir_best_model.pth | 模型文件路径 |
| MODEL_TYPE | clip | 模型类型 |
| HOST | 0.0.0.0 | 监听地址 |
| PORT | 5000 | 监听端口 |
| DEBUG | False | 调试模式 |

### 端口配置

- **API端口**: 5000
- **健康检查**: http://localhost:5000/health
- **API信息**: http://localhost:5000/api_info

## 测试部署

### 健康检查
```bash
curl http://localhost:5000/health
```

### API信息
```bash
curl http://localhost:5000/api_info
```

### 测试推荐
```bash
curl -X POST http://localhost:5000/test \
  -H "Content-Type: application/json"
```

## 监控和维护

### 查看日志
```bash
# Docker方式
docker-compose logs -f fashion-ai-api

# 系统服务方式
sudo journalctl -u fashion-ai-api -f
```

### 性能监控
```bash
# 查看容器资源使用
docker stats fashion-ai-api

# 查看系统资源
htop
```

### 备份和恢复
```bash
# 备份模型文件
tar -czf model_backup_$(date +%Y%m%d).tar.gz models/

# 备份配置文件
cp docker-compose.yml docker-compose.yml.backup
```

## 故障排除

### 常见问题

1. **端口被占用**
   ```bash
   # 查看端口占用
   netstat -tulpn | grep 5000
   
   # 修改端口
   export PORT=5001
   ```

2. **模型文件不存在**
   ```bash
   # 检查模型文件
   ls -la models/
   
   # 设置正确的模型路径
   export MODEL_PATH=/correct/path/to/model.pth
   ```

3. **内存不足**
   ```bash
   # 增加swap空间
   sudo fallocate -l 4G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

4. **GPU不可用**
   ```bash
   # 检查GPU
   nvidia-smi
   
   # 安装GPU驱动
   sudo apt-get install nvidia-driver-470
   ```

## 与后端对接

### API接口

- **推荐接口**: POST /recommend
- **健康检查**: GET /health
- **API信息**: GET /api_info
- **测试接口**: POST /test

### 数据格式

参考 `src/api/726后端对接.md` 中的接口文档。

## 扩展部署

### 负载均衡

使用Nginx进行负载均衡：

```nginx
upstream fashion_ai {
    server 127.0.0.1:5000;
    server 127.0.0.1:5001;
}

server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://fashion_ai;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 容器编排

使用Kubernetes进行容器编排：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fashion-ai-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fashion-ai-api
  template:
    metadata:
      labels:
        app: fashion-ai-api
    spec:
      containers:
      - name: fashion-ai-api
        image: fashion-ai-api:latest
        ports:
        - containerPort: 5000
        env:
        - name: MODEL_PATH
          value: "/app/models/cir_best_model.pth"
```

## 联系支持

如果遇到部署问题，请：

1. 查看日志文件
2. 检查环境配置
3. 联系技术支持团队 