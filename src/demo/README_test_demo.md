# 基于test_data的互补单品推荐演示

这是一个基于你的test_data的互补单品推荐演示系统，使用Gradio构建用户界面。

## 功能特性

### 🎨 界面功能
- **我的搭配**: 用户可以添加、删除和管理自己的单品
- **商品库浏览**: 从test_data中浏览和选择商品
- **筛选功能**: 按类别和场景筛选商品
- **分页显示**: 支持分页浏览大量商品

### 🎯 推荐功能
- **互补单品推荐**: 基于用户选择的单品推荐互补商品
- **场景匹配**: 支持按场景筛选推荐结果

## 文件结构

```
src/demo/
├── gradio_demo_test_data.py              # 主要的Gradio演示代码
├── run_test_demo.py                      # 运行脚本
├── run_full_demo.py                      # 完整流程运行脚本
├── 1_generate_rec_embeddings_test_data.py # 生成推荐embedding
├── 2_build_index_test_data.py            # 构建FAISS索引
├── README_test_demo.md                   # 本文档
└── ...                                   # 其他demo文件
```

## 安装依赖

确保已安装以下依赖：

```bash
pip install gradio torch pillow numpy
```

## 使用方法

### 方法1: 完整流程（推荐）

一键运行完整流程，包括embedding生成、索引构建和Gradio演示：

```bash
cd src/demo
python run_full_demo.py --checkpoint /path/to/your/best_model.pth
```

### 方法2: 分步运行

#### 步骤1: 生成推荐embedding

```bash
cd src/demo
python 1_generate_rec_embeddings_test_data.py \
    --model_type clip \
    --test_data_dir ./src/test/test_data \
    --checkpoint /path/to/your/best_model.pth \
    --batch_size 32
```

#### 步骤2: 构建FAISS索引

```bash
python 2_build_index_test_data.py \
    --test_data_dir ./src/test/test_data \
    --d_embed 512
```

#### 步骤3: 启动Gradio演示

```bash
python run_test_demo.py \
    --model_type clip \
    --test_data_dir ./src/test/test_data \
    --checkpoint /path/to/your/best_model.pth \
    --port 7860 \
    --host 0.0.0.0 \
    --share
```

### 方法3: 快速演示

使用演示模式快速测试：

```bash
python run_full_demo.py \
    --checkpoint /path/to/your/best_model.pth \
    --demo \
    --batch_size 16
```

### 参数说明

#### 通用参数
- `--model_type`: 模型类型，可选 'original' 或 'clip'，默认 'clip'
- `--test_data_dir`: 测试数据目录路径，默认 './src/test/test_data'
- `--checkpoint`: 模型检查点文件路径

#### Embedding生成参数
- `--batch_size`: 批处理大小，默认 32
- `--demo`: 演示模式，只处理少量数据

#### 索引构建参数
- `--d_embed`: embedding维度，默认 512
- `--faiss_type`: FAISS索引类型，默认 'IndexFlatIP'

#### Gradio演示参数
- `--port`: 服务器端口，默认 7860
- `--host`: 服务器主机地址，默认 "0.0.0.0"
- `--share`: 是否生成可分享的公共链接

#### 完整流程参数
- `--skip_embedding`: 跳过embedding生成步骤
- `--skip_index`: 跳过索引构建步骤

## 数据要求

确保你的test_data目录包含以下文件：

```
src/test/test_data/
├── result.json          # 商品元数据JSON文件
└── images/              # 商品图片目录
    ├── 1.jpg
    ├── 2.jpg
    ├── ...
    └── 199.jpg
```

运行后会自动生成以下文件：

```
src/test/test_data/
├── result.json          # 商品元数据JSON文件
├── images/              # 商品图片目录
└── precomputed_rec_embeddings/  # 自动生成的embedding目录
    ├── test_data_embeddings.pkl      # 所有商品的embedding
    ├── test_data_embedding_dict.pkl  # embedding字典
    └── test_data_rec_index.faiss     # FAISS索引文件
```

### result.json格式

```json
[
    {
        "名称": "商品名称",
        "主类别": "上衣",
        "子类别": "T恤",
        "品牌": "品牌名",
        "发售价格": 299,
        "场景": "休闲",
        "风格": "简约",
        "图案": "纯色",
        "适用季节": "春秋",
        "版型": "修身"
    },
    ...
]
```

## 使用流程

### 1. 启动演示
运行脚本后，浏览器会自动打开演示页面，或者手动访问 `http://localhost:7860`

### 2. 添加单品
- 从商品库中选择商品，或上传自己的图片
- 填写商品描述和类别
- 点击"添加"按钮

### 3. 获取推荐
- 确保已添加至少一个单品
- 可选择填写目标描述和场景
- 点击"搜索推荐"按钮
- 查看推荐的互补单品

## 界面说明

### 主要区域

1. **我的搭配**: 显示用户添加的单品，支持选择和删除
2. **商品库选择**: 浏览test_data中的商品，支持筛选和分页
3. **推荐功能**: 互补单品推荐

### 筛选功能

- **按类别筛选**: 上衣、下装等
- **按场景筛选**: 休闲、运动等
- **分页浏览**: 每页显示12个商品

### 推荐算法

- **互补推荐**: 基于embedding相似度搜索互补商品

## 故障排除

### 常见问题

1. **模型文件不存在**
   ```
   错误：模型文件不存在
   解决：使用 --checkpoint 参数指定正确的模型路径
   ```

2. **测试数据目录不存在**
   ```
   错误：测试数据目录不存在
   解决：确保 test_data 目录和文件结构正确
   ```

3. **依赖包缺失**
   ```
   错误：ModuleNotFoundError
   解决：pip install gradio torch pillow numpy
   ```

4. **GPU内存不足**
   ```
   错误：CUDA out of memory
   解决：使用CPU模式或减少batch size
   ```

### 调试模式

启动时添加 `--debug` 参数可以查看详细错误信息：

```bash
python run_test_demo.py --debug
```

## 自定义修改

### 修改界面布局

编辑 `gradio_demo_test_data.py` 中的界面构建部分：

```python
with gr.Blocks(title="自定义标题", theme=gr.themes.Soft()) as demo:
    # 修改界面布局
    pass
```

### 修改推荐算法

在 `search_item` 函数中修改推荐逻辑：

```python
def search_item(comp_description, comp_scene):
    # 自定义推荐算法
    pass
```

### 添加新功能

在界面中添加新的组件和事件处理：

```python
# 添加新组件
new_component = gr.Button("新功能")

# 添加事件处理
new_component.click(
    new_function,
    inputs=[...],
    outputs=[...]
)
```

## 性能优化

### 内存优化

- 使用 `@torch.no_grad()` 装饰器减少内存使用
- 及时清理不需要的变量
- 使用CPU模式如果GPU内存不足

### 速度优化

- 预计算商品embedding
- 使用向量索引加速搜索
- 批量处理多个查询

## 扩展功能

可以考虑添加的功能：

1. **历史记录**: 保存用户的搭配历史
2. **收藏功能**: 允许用户收藏喜欢的搭配
3. **社交分享**: 分享搭配到社交媒体
4. **个性化推荐**: 基于用户历史推荐
5. **多语言支持**: 支持多种语言界面

## 联系支持

如果遇到问题，请检查：

1. 模型文件路径是否正确
2. 测试数据格式是否符合要求
3. 依赖包是否完整安装
4. 系统资源是否充足 