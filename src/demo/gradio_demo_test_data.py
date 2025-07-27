import os
# 设置环境变量解决OpenMP冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import gradio as gr
from dataclasses import dataclass
from typing import List, Optional, Literal
from PIL import Image
import torch
import random
import json
import base64
import io
import pickle
from argparse import ArgumentParser
import pathlib
import numpy as np

# 添加项目根目录到Python路径
project_root = pathlib.Path(__file__).parent.parent.parent
import sys
sys.path.insert(0, str(project_root))

# 使用绝对导入
from src.demo.vectorstore import FAISSVectorStore
from src.models.load import load_model
from src.data import datatypes
from src.data.datasets import polyvore_utils as polyvore

SRC_DIR = pathlib.Path(__file__).parent.parent.parent.absolute()
LOGS_DIR = SRC_DIR / 'logs'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.makedirs(LOGS_DIR, exist_ok=True)

# 设置每页显示12个商品，每次搜索返回8个结果
ITEM_PER_PAGE = 12
ITEM_PER_SEARCH = 8

# 全局状态变量：用户选择的商品和候选商品
state_my_items = []
state_candidate_items = []


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model_type', type=str, choices=['original', 'clip'],
                        default='clip')
    parser.add_argument('--test_data_dir', type=str, 
                        default='./src/test/test_data')
    parser.add_argument('--checkpoint', type=str, 
                        default='./best_path/complementary_clip_cir_experiment_001_best_model.pth')
    parser.add_argument('--port', type=int, default=8080, help="服务器端口")
    parser.add_argument('--host', type=str, default="0.0.0.0", help="服务器主机")
    parser.add_argument('--share', action='store_true', help="是否分享链接")
    
    return parser.parse_args()


def load_test_data(test_data_dir):
    """加载test_data中的商品数据"""
    result_file = os.path.join(test_data_dir, 'test.json')
    images_dir = os.path.join(test_data_dir, 'images')
    
    with open(result_file, 'r', encoding='utf-8') as f:
        products = json.load(f)
    
    # 转换为FashionItem格式
    fashion_items = []
    for i, product in enumerate(products):
        try:
            # 获取item_id，如果JSON中有则使用，否则使用索引+2（从2开始）
            item_id = str(product.get('item_id', i + 2))
            
            # 构建图片路径，使用item_id
            image_path = os.path.join(images_dir, f"{item_id}.jpg")
            if os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
                
                # 创建FashionItem
                scene_value = product.get('场景', '通用')
                # 确保scene是列表类型
                if isinstance(scene_value, str):
                    scene_list = [scene_value] if scene_value else ['通用']
                else:
                    scene_list = scene_value if scene_value else ['通用']
                
                fashion_item = datatypes.FashionItem(
                    item_id=int(item_id) if item_id.isdigit() else None,
                    category=product.get('主类别', '未知'),
                    scene=scene_list,
                    image=image,
                    description=product.get('名称', ''),
                    metadata={
                        'item_id': item_id,
                        'semantic_category': product.get('主类别', '未知'),
                        'scene': product.get('场景', '通用'),
                        'url_name': product.get('名称', ''),
                        'title': product.get('名称', ''),
                        'price': product.get('发售价格'),
                        'brand': product.get('品牌', ''),
                        'sub_category': product.get('子类别', ''),
                        'style': product.get('风格', ''),
                        'pattern': product.get('图案', ''),
                        'season': product.get('适用季节', ''),
                        'fit': product.get('版型', '')
                    },
                    embedding=None
                )
                fashion_items.append(fashion_item)
        except Exception as e:
            print(f"加载商品 {i} 失败: {e}")
            continue
    
    print(f"成功加载 {len(fashion_items)} 件商品")
    return fashion_items


def run(args):
    """运行Gradio演示"""
    
    # 加载test_data
    test_items = load_test_data(args.test_data_dir)
    
    # 按类别分组
    categories = list(set([item.category for item in test_items]))
    
    # 处理场景列表，提取所有唯一的场景值
    all_scenes = []
    for item in test_items:
        if item.scene:
            if isinstance(item.scene, list):
                all_scenes.extend(item.scene)
            else:
                all_scenes.append(item.scene)
    scenes = list(set(all_scenes))
    
    num_pages = len(test_items) // ITEM_PER_PAGE
    
    # 根据页码返回对应页的商品列表
    def get_items(page):
        start_idx = page * ITEM_PER_PAGE
        end_idx = min((page + 1) * ITEM_PER_PAGE, len(test_items))
        return test_items[start_idx:end_idx]

    # 加载模型
    model = load_model(
        model_type=args.model_type, checkpoint=args.checkpoint
    )
    model.eval()
    
    # 加载预计算的embedding字典
    embedding_dict = None
    embedding_dir = os.path.join(args.test_data_dir, 'precomputed_rec_embeddings')
    embedding_dict_path = os.path.join(embedding_dir, 'test_data_embedding_dict.pkl')
    
    if os.path.exists(embedding_dict_path):
        print("加载预计算的embedding字典...")
        with open(embedding_dict_path, 'rb') as f:
            embedding_dict = pickle.load(f)
        print(f"成功加载 {len(embedding_dict)} 个预计算embedding")
    else:
        print("警告：未找到预计算的embedding，将实时计算")
    
    # 加载FAISS索引
    indexer = None
    try:
        indexer = FAISSVectorStore(
            index_name='test_data_rec_index',
            d_embed=512,  # 根据模型输出维度调整
            faiss_type='IndexFlatIP',
            base_dir=embedding_dir,
        )
        print("成功加载FAISS索引")
    except Exception as e:
        print(f"警告：加载FAISS索引失败: {e}")
        print("将使用余弦相似度进行搜索")
    
    '''-----------------------------Web界面构建-----------------------------'''
    # 创建Gradio Web界面
    with gr.Blocks(title="互补单品推荐系统", theme=gr.themes.Soft()) as demo:
        
        with gr.Row(equal_height=True):
            gr.Markdown(
                "# 🎨 互补单品推荐系统"
            )
        
        with gr.Row(equal_height=True):
            gr.Markdown(
                "## 📋 使用说明"
            )
        
        with gr.Row(equal_height=True):
            gr.Markdown(
                """
                1. **选择商品**：从下方素材库中选择一件上装或下装
                2. **设置目标**：输入目标单品的场景和描述（可选）
                3. **获取推荐**：点击搜索按钮，系统会推荐最佳的互补单品
                """
            )
        
        # 商品库选择区域
        with gr.Row(equal_height=True):
            gr.Markdown(
                "## 🛍️ 素材库"
            )
        
        with gr.Row(equal_height=True):
            with gr.Column(scale=2, variant='compact'):
                # 筛选选项
                with gr.Row(equal_height=True):
                    category_filter = gr.Dropdown(
                        label="按类别筛选",
                        choices=["全部"] + categories, value="全部",
                    )
                with gr.Row(equal_height=True):
                    scene_filter = gr.Dropdown(
                        label="按场景筛选",
                        choices=["全部"] + scenes, value="全部",
                    )
                with gr.Row(equal_height=True):
                    polyvore_page = gr.Dropdown(
                        label="页码",
                        choices=list(range(1, num_pages + 1)),
                        value=1
                    )
            
            with gr.Column(scale=10, variant='compact'):
                with gr.Row(equal_height=True):
                    with gr.Column(variant='compact'):
                        polyvore_gallery = gr.Gallery(
                            allow_preview=True,
                            show_label=True,
                            columns=6, 
                            rows=3,
                            type="pil",
                            object_fit='contain',
                            height=300,
                            label="素材库 - 点击选择商品"
                        )
        
        # 推荐区域
        with gr.Row(equal_height=True):
            gr.Markdown(
                "## 🎯 互补单品推荐"
            )
        
        with gr.Row(equal_height=True):
            with gr.Column(scale=1, variant='compact'):
                # 当前选择的商品显示
                with gr.Row(equal_height=True):
                    gr.Markdown(
                        "### 当前选择的商品"
                    )
                with gr.Row(equal_height=True):
                    selected_item_display = gr.Gallery(
                        allow_preview=True,
                        show_label=True,
                        columns=1,
                        rows=1,
                        type="pil",
                        object_fit='contain',
                        height=150,
                        label="已选择的商品"
                    )
                
                # 推荐参数设置
                with gr.Row(equal_height=True):
                    gr.Markdown(
                        "### 推荐设置"
                    )
                with gr.Row(equal_height=True):
                    comp_description = gr.Textbox(
                        label="目标单品描述（可选）",
                        placeholder="如：黑色牛仔裤",
                        lines=2
                    )
                with gr.Row(equal_height=True):
                    comp_scene = gr.Dropdown(
                        label="目标场景（可选）",
                        choices=[''] + scenes, value='',
                    )
                with gr.Row(equal_height=True):
                    btn_search_item = gr.Button(
                        "搜索推荐", variant="primary", size="lg"
                    )
            
            with gr.Column(scale=2, variant='compact'):
                # 推荐结果
                with gr.Row(equal_height=True):
                    gr.Markdown(
                        "### 推荐结果"
                    )
                with gr.Row(equal_height=True):
                    searched_item_gallery = gr.Gallery(
                        allow_preview=True,
                        show_label=True,
                        columns=4, 
                        rows=2,
                        type="pil",
                        object_fit='contain',
                        height=300,
                        label="推荐的单品"
                    )
              
        # 全局变量：当前选择的商品
        global selected_item
        selected_item = None
        
        # Functions
        # 从商品库选择商品
        def select_item_from_library(selected: gr.SelectData):
            global selected_item, state_candidate_items
            
            if selected.index < len(state_candidate_items):
                selected_item = state_candidate_items[selected.index]
                return {
                    selected_item_display: [selected_item.image],
                }
            return {
                selected_item_display: [],
            }
        
        # 筛选商品
        def filter_and_get_items(category_filter, scene_filter, page):
            global state_candidate_items
            
            # 获取所有商品索引
            all_indices = list(range(len(test_items)))
            
            # 按类别筛选
            if category_filter and category_filter != "全部":
                all_indices = [i for i in all_indices if test_items[i].category == category_filter]
            
            # 按场景筛选
            if scene_filter and scene_filter != "全部":
                all_indices = [i for i in all_indices if test_items[i].scene and scene_filter in test_items[i].scene]
            
            # 分页
            page = page - 1
            start_idx = page * ITEM_PER_PAGE
            end_idx = min(start_idx + ITEM_PER_PAGE, len(all_indices))
            
            if start_idx >= len(all_indices):
                state_candidate_items = []
                return {
                    polyvore_gallery: []
                }
            
            page_indices = all_indices[start_idx:end_idx]
            state_candidate_items = [test_items[i] for i in page_indices]
            
            return {
                polyvore_gallery: [item.image for item in state_candidate_items],
            }
        

        

        
        # 搜索互补商品函数：根据当前选择的商品和用户描述，搜索与之互补的商品
        @torch.no_grad()
        def search_item(comp_description, comp_scene):
            global selected_item
            
            if selected_item is None:
                gr.Warning("错误：请先从商品库选择一件商品！")
                return {
                    searched_item_gallery: []
                }
            
            try:
                # 创建目标商品描述
                target_category = "下装" if selected_item.category == "上衣" else "上衣"
                
                # 构建目标商品描述
                target_description = ""
                if comp_description and comp_description.strip():
                    target_description = comp_description.strip()
                else:
                    # 如果没有用户输入描述，使用默认描述
                    target_description = f"一件{target_category}"
                
                # 如果有场景要求，添加到描述中
                if comp_scene and comp_scene.strip():
                    target_description += f"，适合{comp_scene}场合"
                
                print(f"目标商品描述: {target_description}")
                
                # 创建目标商品（虚拟商品，用于查询）
                target_item = datatypes.FashionItem(
                    item_id=None,  # 使用None而不是字符串
                    image=selected_item.image,  # 使用占位图片
                    description=target_description,
                    category=target_category,
                    scene=[comp_scene] if comp_scene else ["通用"],
                    metadata={
                        'item_id': 'target_item',
                        'semantic_category': target_category,
                        'scene': comp_scene if comp_scene else "通用",
                        'url_name': target_description,
                    }
                )
                
                # 计算目标商品的embedding（融合图片和文本）
                target_embedding = model.embed_item([target_item], use_precomputed_embedding=False)
                target_embedding = target_embedding[0].cpu().numpy()
                
                # 计算当前选择商品的embedding
                item_embedding = model.embed_item([selected_item], use_precomputed_embedding=False)
                selected_item.embedding = item_embedding[0].cpu().numpy()
                
                # 创建outfit查询
                outfit_query = datatypes.FashionComplementaryQuery(
                    outfit=[selected_item],
                    category=target_category
                )
                
                # 使用目标商品的embedding进行搜索（融合了图片和文本描述）
                search_embedding = target_embedding
                
                # 使用FAISS索引或余弦相似度搜索
                if indexer is not None:
                    try:
                        # 使用FAISS索引搜索
                        results = indexer.search(
                            embeddings=[search_embedding],
                            k=min(ITEM_PER_SEARCH * 2, len(test_items))  # 搜索更多结果用于过滤
                        )
                        
                        # 检查结果是否为空
                        if not results or len(results) == 0 or len(results[0]) == 0:
                            gr.Warning("FAISS搜索未返回结果，使用余弦相似度搜索")
                            raise Exception("FAISS search returned empty results")
                        
                        # 过滤结果
                        candidate_items = []
                        for score, item_id in results[0]:
                            if int(item_id) >= len(test_items):
                                continue  # 跳过无效的item_id
                            item = test_items[int(item_id)]
                            # 过滤类别
                            if item.category != target_category:
                                continue
                            # 过滤场景（如果指定）
                            if comp_scene and (not item.scene or comp_scene not in item.scene):
                                continue
                            candidate_items.append(item)
                            if len(candidate_items) >= ITEM_PER_SEARCH:
                                break
                        
                        top_items = candidate_items
                        
                    except Exception as e:
                        print(f"FAISS搜索失败: {e}，切换到余弦相似度搜索")
                        # 如果FAISS失败，回退到余弦相似度搜索
                        raise e
                    
                else:
                    # 使用余弦相似度搜索
                    similarities = []
                    candidate_items = []
                    
                    for item in test_items:
                        # 过滤类别
                        if item.category != target_category:
                            continue
                        
                        # 过滤场景（如果指定）
                        if comp_scene and (not item.scene or comp_scene not in item.scene):
                            continue
                        
                        try:
                            # 使用预计算的embedding或实时计算
                            if embedding_dict and item.item_id in embedding_dict:
                                item_embedding = embedding_dict[item.item_id]
                            else:
                                item_embedding = model.embed_item([item], use_precomputed_embedding=False)
                                item_embedding = item_embedding[0].cpu().numpy()
                            
                            # 计算余弦相似度
                            similarity = np.dot(search_embedding, item_embedding) / (
                                np.linalg.norm(search_embedding) * np.linalg.norm(item_embedding)
                            )
                            
                            similarities.append(similarity)
                            candidate_items.append(item)
                        except Exception as e:
                            print(f"计算商品 {item.item_id} 的相似度时出错: {e}")
                            continue
                    
                    if not candidate_items:
                        gr.Warning("没有找到符合条件的互补商品！")
                        return {
                            searched_item_gallery: []
                        }
                    
                    # 按相似度排序
                    sorted_indices = np.argsort(similarities)[::-1]
                    top_items = [candidate_items[i] for i in sorted_indices[:ITEM_PER_SEARCH]]
                
                return {
                    searched_item_gallery: [item.image for item in top_items]
                }
                
            except Exception as e:
                gr.Warning(f"搜索互补商品失败: {e}")
                return {
                    searched_item_gallery: []
                }
        
        # Event Handlers
        # 筛选和分页
        category_filter.change(
            filter_and_get_items,
            inputs=[category_filter, scene_filter, polyvore_page],
            outputs=[polyvore_gallery]
        )
        
        scene_filter.change(
            filter_and_get_items,
            inputs=[category_filter, scene_filter, polyvore_page],
            outputs=[polyvore_gallery]
        )
        
        polyvore_page.change(
            filter_and_get_items,
            inputs=[category_filter, scene_filter, polyvore_page],
            outputs=[polyvore_gallery]
        )
        
        # 从商品库选择商品
        polyvore_gallery.select(
            select_item_from_library,
            inputs=None,
            outputs=[selected_item_display]
        )
        
        # 搜索推荐
        btn_search_item.click(
            search_item,
            inputs=[comp_description, comp_scene],
            outputs=[searched_item_gallery]
        )
        
        # 初始化显示第一页商品
        demo.load(
            lambda: filter_and_get_items("全部", "全部", 1),
            inputs=None,
            outputs=[polyvore_gallery]
        )
    
    return demo


if __name__ == "__main__":
    args = parse_args()
    
    # 如果没有指定checkpoint，使用默认路径
    if args.checkpoint is None:
        args.checkpoint = "./best_path/complementary_clip_cir_experiment_001_best_model.pth"
        print(f"使用默认模型路径: {args.checkpoint}")
        print("请确保模型文件存在，或使用 --checkpoint 参数指定正确的路径")
    
    demo = run(args)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=True,  # 强制启用分享功能
        debug=True
    ) 